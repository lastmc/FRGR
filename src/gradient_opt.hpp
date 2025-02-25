#include <luisa/luisa-compute.h>
#include "common.hpp"

[[nodiscard]] inline auto gradient_optimization_iter(
    luisa::compute::Device &device, luisa::compute::Stream &stream,
    const cv::Mat &image,
    const cv::Mat &dx, const cv::Mat &dy, const cv::Mat &albedo, const cv::Mat &normal,
    float lambda, const std::filesystem::path &output_dir) {
    auto rows = dx.rows;
    auto cols = dx.cols;

    auto dx_buffer = device.create_buffer<luisa::float3>(rows * cols);
    auto dy_buffer = device.create_buffer<luisa::float3>(rows * cols);
    auto albedo_dx_buffer = device.create_buffer<luisa::float3>(rows * cols);
    auto albedo_dy_buffer = device.create_buffer<luisa::float3>(rows * cols);
    auto normal_dx_buffer = device.create_buffer<luisa::float3>(rows * cols);
    auto normal_dy_buffer = device.create_buffer<luisa::float3>(rows * cols);
    auto image_buffer = device.create_buffer<luisa::float3>(rows * cols);

    cv::Mat albedo_dx = calc_dx(albedo);
    cv::Mat albedo_dy = calc_dy(albedo);
    cv::Mat normal_dx = calc_dx(normal);
    cv::Mat normal_dy = calc_dy(normal);
    stream << dx_buffer.copy_from(dx.data) << dy_buffer.copy_from(dy.data)
           << albedo_dx_buffer.copy_from(albedo_dx.data) << albedo_dy_buffer.copy_from(albedo_dy.data)
           << normal_dx_buffer.copy_from(normal_dx.data) << normal_dy_buffer.copy_from(normal_dy.data)
           << image_buffer.copy_from(image.data);

    auto coeff_dx_buffer = device.create_buffer<float>(rows * cols);
    auto coeff_dy_buffer = device.create_buffer<float>(rows * cols);
    auto m_dx_buffer = device.create_buffer<float>(rows * cols);
    auto m_dy_buffer = device.create_buffer<float>(rows * cols);
    auto coeff_kernel = device.compile<2>([&] {
        auto row = luisa::compute::dispatch_x();
        auto col = luisa::compute::dispatch_y();
        auto index = row * cols + col;
        auto dx_v = dx_buffer->read(index);
        auto dy_v = dy_buffer->read(index);
        auto a_dx_v = albedo_dx_buffer->read(index);
        auto a_dy_v = albedo_dy_buffer->read(index);
        auto n_dx_v = normal_dx_buffer->read(index);
        auto n_dy_v = normal_dy_buffer->read(index);

        auto mid = 0.09f;
        auto eps_d = .01f;
        auto mix_channel = [](auto item) {
            return abs(item.x) + abs(item.y) + abs(item.z);
        };
        auto m_dx = mix_channel(dx_v) + eps_d;
        m_dx = (m_dx * m_dx + mid) / (m_dx * mid);
        auto m_dy = mix_channel(dy_v) + eps_d;
        m_dy = (m_dy * m_dy + mid) / (m_dy * mid);
        m_dx *= m_dx;
        m_dy *= m_dy;
        m_dx_buffer->write(index, m_dx);
        m_dy_buffer->write(index, m_dy);

        auto a_dx = mix_channel(a_dx_v);
        auto a_dy = mix_channel(a_dy_v);
        auto n_dx = mix_channel(n_dx_v);
        auto n_dy = mix_channel(n_dy_v);

        auto aux_dx = 1.f / (a_dx + n_dx + eps_d);
        auto aux_dy = 1.f / (a_dy + n_dy + eps_d);
        aux_dx *= aux_dx;
        aux_dy *= aux_dy;
        coeff_dx_buffer->write(index, m_dx + lambda * aux_dx);
        coeff_dy_buffer->write(index, m_dy + lambda * aux_dy);
    });

    stream << coeff_kernel().dispatch(rows, cols);

    // Output
    {
        cv::Mat coeff_dx(dx.size(), CV_32FC1, 0.f);
        cv::Mat coeff_dy(dx.size(), CV_32FC1, 0.f);
        cv::Mat m_dx(dx.size(), CV_32FC1, 0.f);
        cv::Mat m_dy(dx.size(), CV_32FC1, 0.f);
        stream << coeff_dx_buffer.copy_to(coeff_dx.data) << coeff_dy_buffer.copy_to(coeff_dy.data)
               << m_dx_buffer.copy_to(m_dx.data) << m_dy_buffer.copy_to(m_dy.data) << luisa::compute::synchronize();
        imsave(output_dir, "coeff_dx.exr", coeff_dx);
        imsave(output_dir, "coeff_dy.exr", coeff_dy);
        imsave(output_dir, "m_dx.exr", m_dx);
        imsave(output_dir, "m_dy.exr", m_dy);
    }

    auto diff_x_buffer = device.create_buffer<luisa::float3>(rows * cols);
    auto diff_y_buffer = device.create_buffer<luisa::float3>(rows * cols);

    auto diff_kernel = device.compile<2>([&]() {
        auto row = luisa::compute::dispatch_x();
        auto col = luisa::compute::dispatch_y();
        auto index = [&](auto r, auto c) {
            return r * cols + c;
        };
        auto index_cur = index(row, col);
        auto dx0 = dx_buffer->read(index(row, col));
        auto dy0 = dy_buffer->read(index(row, col));
        auto dx_v = -image_buffer->read(index_cur);
        $if(col == cols - 1) {
            dx_v += image_buffer->read(index(row, col - 1));
        }
        $else {
            dx_v += image_buffer->read(index(row, col + 1));
        };
        auto dy_v = -image_buffer->read(index_cur);
        $if(row == rows - 1) {
            dy_v += image_buffer->read(index(row - 1, col));
        }
        $else {
            dy_v += image_buffer->read(index(row + 1, col));
        };
        auto m_dx = m_dx_buffer->read(index_cur);
        auto coeff_dx = coeff_dx_buffer->read(index_cur);
        auto diff_x = 2.f * (coeff_dx * dx_v - m_dx * dx0);
        auto m_dy = m_dy_buffer->read(index_cur);
        auto coeff_dy = coeff_dy_buffer->read(index_cur);
        auto diff_y = 2.f * (coeff_dy * dy_v - m_dy * dy0);
        diff_x_buffer->write(index_cur, diff_x);
        diff_y_buffer->write(index_cur, diff_y);
    });

    auto delta_buffer = device.create_buffer<luisa::float3>(rows * cols);

    auto iter_kernel = device.compile<2>([&](luisa::compute::Float step) {
        auto row = luisa::compute::dispatch_x();
        auto col = luisa::compute::dispatch_y();
        auto index = [&](auto r, auto c) {
            return r * cols + c;
        };
        auto index_cur = index(row, col);
        auto df_di = -diff_x_buffer->read(index_cur) - diff_y_buffer->read(index_cur);
        $if(col > 0) {
            df_di += diff_x_buffer->read(index(row, col - 1));
        };
        $if(row > 0) {
            df_di += diff_y_buffer->read(index(row - 1, col));
        };
        auto delta = -step * df_di;
        delta_buffer->write(index_cur, delta_buffer->read(index_cur) + delta);
        image_buffer->write(index_cur, image_buffer->read(index_cur) + delta);
    });

    auto revert_kernel = device.compile<2>([&]{
        auto row = luisa::compute::dispatch_x();
        auto col = luisa::compute::dispatch_y();
        auto index = row * cols + col;
        image_buffer->write(index, image_buffer->read(index) - delta_buffer->read(index));
    });

    auto clear_delta_kernel = device.compile<2>([&] {
        auto row = luisa::compute::dispatch_x();
        auto col = luisa::compute::dispatch_y();
        auto index = row * cols + col;
        delta_buffer->write(index, luisa::make_float3(0.f));
    });

    auto loss_buffer = device.create_buffer<float>(rows * cols);
    auto loss_kernel = device.compile<2>([&] {
        auto row = luisa::compute::dispatch_x();
        auto col = luisa::compute::dispatch_y();
        auto index = [&](auto r, auto c) {
            return r * cols + c;
        };
        auto index_cur = index(row, col);
        auto dx0 = dx_buffer->read(index(row, col));
        auto dy0 = dy_buffer->read(index(row, col));
        auto dx_v = -image_buffer->read(index_cur);
        $if(col == cols - 1) {
            dx_v += image_buffer->read(index(row, col - 1));
        }
        $else {
            dx_v += image_buffer->read(index(row, col + 1));
        };
        auto dy_v = -image_buffer->read(index_cur);
        $if(row == rows - 1) {
            dy_v += image_buffer->read(index(row - 1, col));
        }
        $else {
            dy_v += image_buffer->read(index(row + 1, col));
        };
        auto m_dx = m_dx_buffer->read(index_cur);
        auto l_aux_dx_2 = coeff_dx_buffer->read(index_cur) - m_dx;
        auto m_dy = m_dy_buffer->read(index_cur);
        auto l_aux_dy_2 = coeff_dy_buffer->read(index_cur) - m_dy;
        auto loss =
            m_dx * length_squared(dx_v - dx0) + m_dy * length_squared(dy_v - dy0) +
            l_aux_dx_2 * length_squared(dx_v) + l_aux_dy_2 * length_squared(dy_v);
        loss_buffer->write(index_cur, loss);
    });
    cv::Mat loss_img(dx.size(), CV_32FC1, 0.f);

    float step = .0001f;
    double loss = 1e30;
    stream << clear_delta_kernel().dispatch(rows, cols);
    for (int i = 0; i < 1000; i++) {
        if(i < 20 || i % 50 == 0) {
            stream << loss_kernel().dispatch(rows, cols) << loss_buffer.copy_to(loss_img.data) << luisa::compute::synchronize();
            auto cur_loss = cv::sum(loss_img)[0];
            if (cur_loss > loss) {
                stream << revert_kernel().dispatch(rows, cols) << clear_delta_kernel().dispatch(rows, cols);
                step /= 2.f;
            } else {
                loss = cur_loss;
            }
        }

        stream << diff_kernel().dispatch(rows, cols);

        //        // Output
        //        {
        //            cv::Mat diff_x(dx.size(), CV_32FC4, 0.f);
        //            cv::Mat diff_y(dx.size(), CV_32FC4, 0.f);
        //            stream << diff_x_buffer.copy_to(diff_x.data) << diff_y_buffer.copy_to(diff_y.data) << luisa::compute::synchronize();
        //            imsave(output_dir, "diff_x.exr", cvt_to_bgr(diff_x));
        //            imsave(output_dir, "diff_y.exr", cvt_to_bgr(diff_y));
        //        }

        stream << iter_kernel(step).dispatch(rows, cols);

        // Output
//        if ((i+1) % 1000 == 0) {
//            cv::Mat output(dx.size(), CV_32FC4, 0.f);
//            stream << image_buffer.copy_to(output.data) << luisa::compute::synchronize();
//            imsave(output_dir, "output" + std::to_string(i) + ".exr", cvt_to_bgr(output));
//        }
    }

    cv::Mat output(dx.size(), CV_32FC4, 0.f);
    stream << image_buffer.copy_to(output.data) << luisa::compute::synchronize();
    return std::make_pair(cvt_to_bgr(calc_dx(output)), cvt_to_bgr(calc_dy(output)));
}