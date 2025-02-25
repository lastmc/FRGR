#pragma once

#include <luisa/luisa-compute.h>
#include "common.hpp"

[[nodiscard]] inline auto bilateral_filter_with_var(
    luisa::compute::Device &device, luisa::compute::Stream &stream,
    const cv::Mat &image, const cv::Mat &var,
    float sigma_dis, float sigma_color,
    int K) noexcept {
    auto rows = image.rows;
    auto cols = image.cols;
    auto N = (2 * K + 1) * (2 * K + 1);

    cv::Mat result{rows, cols, CV_32FC4, cv::Scalar::all(0)};
    cv::Mat result_var{rows, cols, CV_32FC4, cv::Scalar::all(0)};

    using namespace luisa::compute;

    auto image_4 = cvt_to_rgba(image);
    auto var_4 = cvt_to_rgba(var);
    auto image_buffer = device.create_buffer<float3>(rows * cols);
    auto var_buffer = device.create_buffer<float3>(rows * cols);
    auto result_buffer = device.create_buffer<float3>(rows * cols);
    auto result_var_buffer = device.create_buffer<float3>(rows * cols);

    stream << image_buffer.copy_from(image_4.data) << var_buffer.copy_from(var_4.data);

    uint block_size = 8;
    auto padded_block_size = block_size + 2 * K;
    auto img_pixel_size = padded_block_size * padded_block_size;
    uint shared_mem_size = img_pixel_size * 3 * 2;// image, var
    if (shared_mem_size > 16384) {
        // Bigger than 64K
        fmt::print("Shared mem size exceeded limit. ({}:{})\n", __FILE__, __LINE__);
        std::abort();
    }

    auto filter_kernel = device.compile<2>([&]() {
        // x : row, y : col
        set_block_size(block_size, block_size);
        Shared<float> shared_mem{shared_mem_size};
        auto pid = dispatch_id().xy();
        auto tid = thread_id().xy();
        // pid of first pixel in block
        auto base_id = make_uint2(pid.x - tid.x, pid.y - tid.y);
        auto index_image = [&](auto c, auto i) {
            return c * img_pixel_size + i;
        };
        auto index_var = [&](auto c, auto i) {
            return index_image(c + 3, i);
        };

        // Write shared memory

        auto pixels_per_thread = img_pixel_size / (block_size * block_size);
        auto more_pixels_thread = img_pixel_size % (block_size * block_size);
        auto pixels = ite(tid.x * block_size + tid.y < more_pixels_thread, pixels_per_thread + 1, pixels_per_thread);
        $for(p, pixels) {
            auto img_idx_relative = p * block_size * block_size + tid.x * block_size + tid.y;
            // xy in padded block
            auto img_idx_offset = make_uint2(img_idx_relative / padded_block_size, img_idx_relative % padded_block_size);
            // xy in image
            auto img_idx = make_int2(img_idx_offset + base_id) - make_int2(K, K);
            // BORDER_REFLECT_101
            img_idx.x = ite(img_idx.x < 0, -img_idx.x, img_idx.x);
            img_idx.x = ite(img_idx.x >= rows, 2 * rows - 2 - img_idx.x, img_idx.x);
            img_idx.y = ite(img_idx.y < 0, -img_idx.y, img_idx.y);
            img_idx.y = ite(img_idx.y >= cols, 2 * cols - 2 - img_idx.y, img_idx.y);

            auto img_v = image_buffer->read(img_idx.x * cols + img_idx.y);
            auto var_v = var_buffer->read(img_idx.x * cols + img_idx.y);
            for (auto c = 0u; c < 3u; c++) {
                shared_mem.write(index_image(c, img_idx_relative), img_v[c]);
                shared_mem.write(index_var(c, img_idx_relative), var_v[c]);
            }
        };

        sync_block();

        $if(any(pid >= make_uint2(rows, cols))) {
            $return();
        };

        auto calc_q_id = [&](auto i) {
            // calc indices of aux_i & aux_0 in padded block
            auto di = i / (2 * K + 1);
            auto dj = i % (2 * K + 1);
            auto q_x = tid.x + di;
            auto q_y = tid.y + dj;
            return q_x * padded_block_size + q_y;
        };
        auto center_id = (tid.x + K) * padded_block_size + (tid.y + K);

        auto filtered_color = def(make_float3());
        auto filtered_var = def(make_float3());
        auto center_color = make_float3(
            shared_mem.read(index_image(0, center_id)),
            shared_mem.read(index_image(1, center_id)),
            shared_mem.read(index_image(2, center_id)));
        auto center_v =
            shared_mem.read(index_var(0, center_id)) +
            shared_mem.read(index_var(1, center_id)) +
            shared_mem.read(index_var(2, center_id));
        auto sum_w = def(0.f);
        $for(i, N) {
            auto q_id = calc_q_id(i);
            auto q_color = make_float3(
                shared_mem.read(index_image(0, q_id)),
                shared_mem.read(index_image(1, q_id)),
                shared_mem.read(index_image(2, q_id)));
            auto q_v =
                shared_mem.read(index_var(0, q_id)) +
                shared_mem.read(index_var(1, q_id)) +
                shared_mem.read(index_var(2, q_id));
            auto dis_x = i / (2 * K + 1) - K;
            auto dis_y = i % (2 * K + 1) - K;
            auto dis = cast<float>(dis_x * dis_x + dis_y * dis_y);

            auto sqr = [](auto x) { return x * x; };
            auto wi = exp(-dis / sqr(sigma_dis) - max(length_squared(q_color - center_color) - (center_v + q_v), 0.f) /
                                                      max(sqr(sigma_color) * center_v, 1e-4f));
            sum_w += wi;
            filtered_color += wi * q_color;
            filtered_var += wi * wi * q_v;
        };
        filtered_color /= sum_w;
        filtered_var /= sum_w * sum_w;
        result_buffer->write(pid.x * cols + pid.y, filtered_color);
        result_var_buffer->write(pid.x * cols + pid.y, filtered_var);
    });

    stream << synchronize();
    auto padded_size = (make_uint2(rows, cols) + block_size - 1u) / block_size * block_size;
    stream << filter_kernel().dispatch(padded_size)
           << result_buffer.copy_to(result.data)
           << result_var_buffer.copy_to(result_var.data)
           << synchronize();

    return std::make_pair(cvt_to_bgr(result), cvt_to_bgr(result_var));
}

[[nodiscard]] inline auto feature_filter_with_var(
    luisa::compute::Device &device, luisa::compute::Stream &stream,
    const cv::Mat &input,
    const cv::Mat &image, const cv::Mat &var,
    const std::vector<cv::Mat> &auxs, const std::vector<cv::Mat> &aux_vars,
    float sigma_color, const std::vector<float> &aux_sigmas,
    int K, int nlm_K,
    bool calc_sure) noexcept {
    using namespace luisa::compute;

    auto rows = image.rows;
    auto cols = image.cols;
    auto N = (2 * K + 1) * (2 * K + 1);
    auto border_K = K + max(1, nlm_K);// K + nlm (1 for aux grad)

    cv::Mat result{rows, cols, CV_32FC4, cv::Scalar::all(0)};
    cv::Mat result_var{rows, cols, CV_32FC4, cv::Scalar::all(0)};
    cv::Mat result_sure{rows, cols, CV_32FC4, cv::Scalar::all(0)};
    cv::Mat result_d{rows, cols, CV_32FC4, cv::Scalar::all(0)};

    auto input_buffer = device.create_buffer<float3>(rows * cols);
    auto image_buffer = device.create_buffer<float3>(rows * cols);
    auto var_buffer = device.create_buffer<float3>(rows * cols);
    auto result_buffer = device.create_buffer<float3>(rows * cols);
    auto result_var_buffer = device.create_buffer<float3>(rows * cols);
    auto result_sure_buffer = device.create_buffer<float3>(rows * cols);
    auto result_d_buffer = device.create_buffer<float3>(rows * cols);

    std::vector<Buffer<float3>> aux_buffers;
    std::vector<Buffer<float3>> aux_var_buffers;

    auto var_4 = var.empty() ? cv::Mat{rows, cols, CV_32FC4, cv::Scalar::all(0)} : var;
    stream << input_buffer.copy_from(input.data) << image_buffer.copy_from(image.data) << var_buffer.copy_from(var_4.data);

    for (auto i = 0; i < auxs.size(); i++) {
        auto a_buffer = device.create_buffer<float3>(rows * cols);
        auto a_v_buffer = device.create_buffer<float3>(rows * cols);
        stream << a_buffer.copy_from(auxs[i].data) << a_v_buffer.copy_from(aux_vars[i].data);
        aux_buffers.emplace_back(std::move(a_buffer));
        aux_var_buffers.emplace_back(std::move(a_v_buffer));
    }

    auto block_size = 16u;
    auto padded_block_size = static_cast<int>(block_size) + 2 * border_K;
    auto padded_pixel_size = padded_block_size * padded_block_size;
    uint shared_mem_size = padded_pixel_size * 3 * 2;                              // image, var
    uint shared_buffer_size = padded_pixel_size * 3 * (1 + 2 * aux_buffers.size());// input, auxs, aux_vars
    auto padded_size = (make_uint2(rows, cols) + block_size - 1u) / block_size * block_size;
    if (shared_mem_size > 16384) {
        // Bigger than 64K
        fmt::print("Shared mem size exceeded limit. ({}:{})\n", __FILE__, __LINE__);
        //        std::abort();
    }

    // Hack shared memory
    auto blocks_per_row = padded_size.x / block_size;
    auto blocks_per_col = padded_size.y / block_size;
    auto shared_buffer = device.create_buffer<float>(
        shared_buffer_size * blocks_per_row * blocks_per_col);
    auto filter_kernel = device.compile<2>([&]() {
        // x : row, y : col
        set_block_size(block_size, block_size);
        Shared<float> shared_mem{shared_mem_size};
        auto block_shared_buffer_offset = (block_x() * blocks_per_col + block_y()) * shared_buffer_size;
        auto write_shared_buffer = [&](auto idx, auto value) {
            shared_buffer->write(block_shared_buffer_offset + idx, value);
        };
        auto read_shared_buffer = [&](auto idx) {
            return shared_buffer->read(block_shared_buffer_offset + idx);
        };

        auto pid = dispatch_id().xy();
        auto tid = thread_id().xy();
        // pid of first pixel in block
        auto base_id = make_uint2(pid.x - tid.x, pid.y - tid.y);
        auto index_a = [&](auto a, auto c, auto i) {
            return (3 * a + c) * padded_pixel_size + i;
        };
        auto index_input = [&](auto c, auto i) {
            return index_a(0, c, i);
        };
        auto index_image = [&](auto c, auto i) {
            // in shared mem
            return index_a(0, c, i);
        };
        auto index_var = [&](auto c, auto i) {
            // in shared mem
            return index_a(1, c, i);
        };
        auto index_aux = [&](auto a, auto c, auto i) {
            return index_a(1 + 2 * a, c, i);
        };
        auto index_aux_var = [&](auto a, auto c, auto i) {
            return index_a(2 + 2 * a, c, i);
        };

        // Write shared memory

        auto pixels_per_thread = padded_pixel_size / (block_size * block_size);
        auto more_pixels_thread = padded_pixel_size % (block_size * block_size);
        auto pixels = ite(tid.x * block_size + tid.y < more_pixels_thread, pixels_per_thread + 1, pixels_per_thread);
        $for(p, pixels) {
            auto img_idx_relative = p * block_size * block_size + tid.x * block_size + tid.y;
            // xy in padded block
            auto img_idx_offset = make_uint2(img_idx_relative / padded_block_size, img_idx_relative % padded_block_size);
            // xy in image
            auto img_idx = make_int2(img_idx_offset + base_id) - make_int2(border_K, border_K);
            // BORDER_REFLECT_101
            img_idx.x = ite(img_idx.x < 0, -img_idx.x, img_idx.x);
            img_idx.x = ite(img_idx.x >= rows, 2 * rows - 2 - img_idx.x, img_idx.x);
            img_idx.y = ite(img_idx.y < 0, -img_idx.y, img_idx.y);
            img_idx.y = ite(img_idx.y >= cols, 2 * cols - 2 - img_idx.y, img_idx.y);

            auto in_v = input_buffer->read(img_idx.x * cols + img_idx.y);
            auto img_v = image_buffer->read(img_idx.x * cols + img_idx.y);
            auto var_v = var_buffer->read(img_idx.x * cols + img_idx.y);
            for (auto c = 0u; c < 3u; c++) {
                //                shared_mem.write(index_input(c, img_idx_relative), in_v[c]);
                shared_mem.write(index_image(c, img_idx_relative), img_v[c]);
                shared_mem.write(index_var(c, img_idx_relative), var_v[c]);
                write_shared_buffer(index_input(c, img_idx_relative), in_v[c]);
                //                write_shared_mem(index_image(c, img_idx_relative), img_v[c]);
                //                write_shared_mem(index_var(c, img_idx_relative), var_v[c]);
            }

            for (auto a = 0u; a < aux_buffers.size(); a++) {
                auto aux_v = aux_buffers[a]->read(img_idx.x * cols + img_idx.y);
                auto aux_var_v = aux_var_buffers[a]->read(img_idx.x * cols + img_idx.y);
                for (auto c = 0u; c < 3u; c++) {
                    write_shared_buffer(index_aux(a, c, img_idx_relative), aux_v[c]);
                    write_shared_buffer(index_aux_var(a, c, img_idx_relative), aux_var_v[c]);
                }
            }
        };

        sync_block();

        $if(any(pid >= make_uint2(rows, cols))) {
            $return();
        };

        auto sqr = [](auto x) { return x * x; };
        auto calc_q_id = [&](auto i) {
            // calc indices of aux_i & aux_0 in padded block
            auto di = i / (2 * K + 1);
            auto dj = i % (2 * K + 1);
            auto q_x = tid.x + di + border_K - K;
            auto q_y = tid.y + dj + border_K - K;
            return q_x * padded_block_size + q_y;
        };
        auto center_id = (tid.x + border_K) * padded_block_size + (tid.y + border_K);

        auto buffer_color = [&](auto a, auto i) {
            return make_float3(
                read_shared_buffer(index_a(a, 0, i)),
                read_shared_buffer(index_a(a, 1, i)),
                read_shared_buffer(index_a(a, 2, i)));
        };
        auto mem_color = [&](auto a, auto i) {
            return make_float3(
                shared_mem.read(index_a(a, 0, i)),
                shared_mem.read(index_a(a, 1, i)),
                shared_mem.read(index_a(a, 2, i)));
        };
        auto input_color = [&](auto i) {
            return buffer_color(0, i);
        };
        auto image_color = [&](auto i) {
            return mem_color(0, i);
        };
        auto var_color = [&](auto i) {
            return mem_color(1, i);
        };
        auto aux_color = [&](auto a, auto i) {
            return buffer_color(1 + 2 * a, i);
        };
        auto aux_var_color = [&](auto a, auto i) {
            return buffer_color(2 + 2 * a, i);
        };
        auto reduce_mean = [&](auto x) {
            return reduce_sum(x) / 3.f;
        };

        auto filtered_color = def(make_float3());
        auto filtered_var = def(make_float3());
        auto filtered_color_d = def(make_float3());
        auto center_color = input_color(center_id);
        auto center_color_d = center_color * 1.01f;
        auto center_var = var_color(center_id);
        auto sum_w = def(0.f);
        auto sum_w_d = def(0.f);
        // a minimum value of max weight
        auto max_w = def(1e-3f);
        auto max_w_d = def(1e-3f);

        auto wi_save = def(make_float3());

        $for(i, N / 2) {
            auto q_id = calc_q_id(i);
            auto q_id_2 = calc_q_id(N - 1 - i);
            auto q_var = var_color(q_id);
            auto q_var_2 = var_color(q_id_2);

            auto wi = def(1.f);
            auto wi_2 = def(1.f);
            auto wi_d = def(1.f);
            auto wi_d_2 = def(1.f);

            if (sigma_color < 1e3) {
                // NL means
                auto color_dis = def(0.f);
                auto color_dis_2 = def(0.f);
                auto color_dis_s = def(0.f);
                auto color_dis_d = def(0.f);
                auto color_dis_d_2 = def(0.f);
                auto color_dis_d_s = def(0.f);
                for (int di = -nlm_K; di <= nlm_K; di++) {
                    for (int dj = -nlm_K; dj <= nlm_K; dj++) {
                        auto offset_q_id = cast<uint>(cast<int>(q_id) + di * padded_block_size + dj);
                        auto offset_q_id_2 = cast<uint>(cast<int>(q_id_2) + di * padded_block_size + dj);
                        auto offset_p_id = cast<uint>(cast<int>(center_id) + di * padded_block_size + dj);
                        auto q_c = image_color(offset_q_id);
                        auto q_c_2 = image_color(offset_q_id_2);
                        auto p_c = image_color(offset_p_id);
                        auto q_v = var_color(offset_q_id);
                        auto q_v_2 = var_color(offset_q_id_2);
                        auto p_v = var_color(offset_p_id);
                        // delta
                        auto p_c_d = (di == 0 && dj == 0) ? p_c * 1.01f : p_c;

                        auto color_weight = [&](auto q_c, auto p_c, auto q_v, auto p_v) {
                            return reduce_mean((sqr(q_c - p_c) - (p_v + min(p_v, q_v))) /
                                               (1e-10f + sqr(sigma_color) * (p_v + q_v)));
                        };
                        color_dis += color_weight(q_c, p_c, q_v, p_v);
                        color_dis_2 += color_weight(q_c_2, p_c, q_v_2, p_v);
                        color_dis_s += color_weight((q_c + q_c_2) / 2.f, p_c, (q_v + q_v_2) / 4.f, p_v);
                        if (calc_sure) {
                            color_dis_d += color_weight(q_c, p_c_d, q_v, p_v);
                            color_dis_d_2 += color_weight(q_c_2, p_c_d, q_v_2, p_v);
                            color_dis_d_s += color_weight((q_c + q_c_2) / 2.f, p_c_d, (q_v + q_v_2) / 4.f, p_v);
                        }
                    }
                }
                color_dis /= sqr(2.f * cast<float>(nlm_K) + 1.f);
                color_dis_2 /= sqr(2.f * cast<float>(nlm_K) + 1.f);
                color_dis_s /= sqr(2.f * cast<float>(nlm_K) + 1.f);
                if (calc_sure) {
                    color_dis_d /= sqr(2.f * cast<float>(nlm_K) + 1.f);
                    color_dis_d_2 /= sqr(2.f * cast<float>(nlm_K) + 1.f);
                    color_dis_d_s /= sqr(2.f * cast<float>(nlm_K) + 1.f);
                }

                auto symmetric_weight = [](auto d1, auto d2, auto ds) {
                    auto w1 = exp(-max(0.f, d1));
                    auto w2 = exp(-max(0.f, d2));
                    auto ws = exp(-max(0.f, ds));
                    // symmetric weight, magic const
                    auto lambda = ite(
                        d1 < 25.f & d2 < 25.f,
                        min(1.f, max(0.f, ws / (w1 + w2) - 1.f)),
                        0.f);
                    return std::make_pair(
                        lerp(w1, ws, lambda),
                        lerp(w2, ws, lambda));
                };

                auto [wi_c, wi_c_2] = symmetric_weight(color_dis, color_dis_2, color_dis_s);
                wi = min(wi, wi_c);
                wi_2 = min(wi_2, wi_c_2);
                if (calc_sure) {
                    auto [wi_c_d, wi_c_d_2] = symmetric_weight(color_dis_d, color_dis_d_2, color_dis_d_s);
                    wi_d = min(wi_d, wi_c_d);
                    wi_d_2 = min(wi_d_2, wi_c_d_2);
                }
            }

            for (auto a = 0u; a < aux_buffers.size(); a++) {
                auto c_c = aux_color(a, center_id);
                auto q_c = aux_color(a, q_id);
                auto q_c_2 = aux_color(a, q_id_2);
                auto c_v = aux_var_color(a, center_id);
                auto q_v = aux_var_color(a, q_id);
                auto q_v_2 = aux_var_color(a, q_id_2);

                auto calc_grad = [&](auto id, auto color) {
                    auto grad_h = min(
                        sqr((aux_color(a, id - 1) - color) / 2.f),
                        sqr((aux_color(a, id + 1) - color) / 2.f));
                    auto grad_v = min(
                        sqr((aux_color(a, id - padded_block_size) - color) / 2.f),
                        sqr((aux_color(a, id + padded_block_size) - color) / 2.f));
                    auto grad = (grad_h + grad_v);
                    return grad;
                };
                auto c_grad = calc_grad(center_id, c_c);
                auto q_grad = calc_grad(q_id, q_c);
                auto q_grad_2 = calc_grad(q_id_2, q_c_2);

                auto c_domi = max(max(c_v, c_grad), 1e-3f);
                auto q_domi = max(max(q_v, q_grad), 1e-3f);
                auto q_domi_2 = max(max(q_v, q_grad), 1e-3f);

                auto feature_weight = [&](auto q_c, auto c_c, auto q_v, auto c_v, auto q_domi, auto c_domi) {
                    return reduce_mean((sqr(q_c - c_c) - (c_v + min(q_v, c_v))) /
                                       (sqr(aux_sigmas[a]) * (c_domi + q_domi)));
                };
                auto feature_dis = feature_weight(q_c, c_c, q_v, c_v, q_domi, c_domi);
                auto feature_dis_2 = feature_weight(q_c_2, c_c, q_v_2, c_v, q_domi_2, c_domi);
                auto feature_dis_s = feature_weight((q_c + q_c_2) / 2.f, c_c, (q_v + q_v_2) / 4.f, c_v, (q_domi + q_domi_2) / 4.f, c_domi);

                auto wi_f = exp(-max(0.f, feature_dis));
                auto wi_f_2 = exp(-max(0.f, feature_dis_2));
                auto wi_f_s = exp(-max(0.f, feature_dis_s));

                // symmetric weight, magic const
                auto lambda = ite(
                    feature_dis < 25.f & feature_dis_2 < 25.f,
                    min(1.f, max(0.f, wi_f_s / (wi_f + wi_f_2) - 1.f)),
                    0.f);
                wi_f = lerp(wi_f, wi_f_s, lambda);
                wi_f_2 = lerp(wi_f_2, wi_f_s, lambda);

                wi = min(wi, wi_f);
                wi_2 = min(wi_2, wi_f_2);
                if (calc_sure) {
                    wi_d = min(wi_d, wi_f);
                    wi_d_2 = min(wi_d_2, wi_f_2);
                }
            }
            sum_w += wi + wi_2;
            max_w = max(max_w, max(wi, wi_2));
            if (calc_sure) {
                sum_w_d += wi_d + wi_d_2;
                max_w_d = max(max_w_d, max(wi_d, wi_d_2));
            }

            filtered_color += wi * input_color(q_id) + wi_2 * input_color(q_id_2);
            filtered_var += wi * wi * q_var + wi_2 * wi_2 * q_var_2;
            if (calc_sure) {
                filtered_color_d += wi_d * input_color(q_id) + wi_d_2 * input_color(q_id_2);
            }
        };
        filtered_color = (filtered_color + max_w * center_color) / (sum_w + max_w);
        filtered_var /= sqr(sum_w + max_w);
        result_buffer->write(pid.x * cols + pid.y, filtered_color);
        result_var_buffer->write(pid.x * cols + pid.y, filtered_var);

        if (calc_sure) {
            filtered_color_d = (filtered_color_d + max_w_d * center_color_d) / (sum_w_d + max_w_d);
            // df/du is correct only when input = image
            auto du = 0.01f * center_color + make_float3(1e-10f);
            auto df_du = filtered_color_d / du - filtered_color / du;
            $if(any(df_du < make_float3(0.f)) | any(df_du > make_float3(1.f))) {
                df_du = make_float3(max_w / (max_w + sum_w));
            };
            result_d_buffer->write(pid.x * cols + pid.y, df_du);
            auto sure = sqr(filtered_color - center_color) - center_var +
                        2.f * center_var * df_du;
            result_sure_buffer->write(pid.x * cols + pid.y, sure);
        }
    });

    stream << synchronize();
    stream << filter_kernel().dispatch(padded_size)
           << result_buffer.copy_to(result.data)
           << result_var_buffer.copy_to(result_var.data);
    if (calc_sure) {
        stream << result_sure_buffer.copy_to(result_sure.data)
               << result_d_buffer.copy_to(result_d.data);
    }
    stream << synchronize();

    return std::make_tuple(result, result_var, result_sure, result_d);
}

[[nodiscard]] inline auto calc_selection_map(
    luisa::compute::Device &device, luisa::compute::Stream &stream,
    const std::vector<cv::Mat> &sures, const std::vector<cv::Mat> &ds) {
    if (sures.size() < 3 || ds.size() < 2) {
        fmt::print("Wrong sure or d.\n");
        std::abort();
    }
    using namespace luisa::compute;
    auto rows = sures[0].rows;
    auto cols = sures[0].cols;
    auto r_sure_buffer = device.create_buffer<float3>(rows * cols);
    auto g_sure_buffer = device.create_buffer<float3>(rows * cols);
    auto b_sure_buffer = device.create_buffer<float3>(rows * cols);
    auto r_d_buffer = device.create_buffer<float3>(rows * cols);
    auto g_d_buffer = device.create_buffer<float3>(rows * cols);
    auto sel_buffer = device.create_buffer<float3>(rows * cols);

    stream << r_sure_buffer.copy_from(sures[0].data)
           << g_sure_buffer.copy_from(sures[1].data)
           << b_sure_buffer.copy_from(sures[2].data)
           << r_d_buffer.copy_from(ds[0].data)
           << g_d_buffer.copy_from(ds[1].data);

    auto selection_kernel = device.compile<2>([&]() {
        auto id = dispatch_x() * cols + dispatch_y();
        auto reduce_mean = [](auto x) {
            return reduce_sum(x) / 3.f;
        };
        auto r_sure = reduce_mean(r_sure_buffer->read(id));
        auto g_sure = reduce_mean(g_sure_buffer->read(id));
        auto b_sure = reduce_mean(b_sure_buffer->read(id));
        auto r_d = reduce_mean(r_d_buffer->read(id));
        auto g_d = reduce_mean(g_d_buffer->read(id));

        auto sel = def(make_float3());
        $if(r_d < g_d) {
            sel = make_float3(
                ite(r_sure <= g_sure & r_sure <= b_sure, 1.f, 0.f),
                ite(g_sure < r_sure & g_sure <= b_sure, 1.f, 0.f),
                ite(b_sure < r_sure & b_sure < g_sure, 1.f, 0.f));
        }
        $else {
            sel = make_float3(
                0.f,
                ite(g_sure <= b_sure, 1.f, 0.f),
                ite(b_sure < g_sure, 1.f, 0.f));
        };
        sel_buffer->write(id, sel);
    });

    cv::Mat sel{rows, cols, CV_32FC4, cv::Scalar::all(0)};
    stream << selection_kernel().dispatch(rows, cols)
           << sel_buffer.copy_to(sel.data)
           << synchronize();

    return sel;
}

[[nodiscard]] inline auto fuse_selection_map(
    luisa::compute::Device &device, luisa::compute::Stream &stream,
    const std::vector<cv::Mat> &rgb, const cv::Mat &sel) {
    if (rgb.size() < 3) {
        fmt::print("Wrong rgb.\n");
        std::abort();
    }
    using namespace luisa::compute;
    auto rows = sel.rows;
    auto cols = sel.cols;
    auto r_buffer = device.create_buffer<float3>(rows * cols);
    auto g_buffer = device.create_buffer<float3>(rows * cols);
    auto b_buffer = device.create_buffer<float3>(rows * cols);
    auto sel_buffer = device.create_buffer<float3>(rows * cols);
    auto result_buffer = device.create_buffer<float3>(rows * cols);

    stream << r_buffer.copy_from(rgb[0].data)
           << g_buffer.copy_from(rgb[1].data)
           << b_buffer.copy_from(rgb[2].data)
           << sel_buffer.copy_from(sel.data);

    auto fuse_kernel = device.compile<2>([&]() {
        auto id = dispatch_x() * cols + dispatch_y();
        auto r = r_buffer->read(id);
        auto g = g_buffer->read(id);
        auto b = b_buffer->read(id);
        auto sel = sel_buffer->read(id);

        result_buffer->write(id, sel.x * r + sel.y * g + sel.z * b);
    });

    cv::Mat result{rows, cols, CV_32FC4, cv::Scalar::all(0)};
    stream << fuse_kernel().dispatch(rows, cols)
           << result_buffer.copy_to(result.data)
           << synchronize();

    return result;
}

[[nodiscard]] inline auto coarsely_filter(
    luisa::compute::Device &device, luisa::compute::Stream &stream,
    const cv::Mat& image, const cv::Mat& var,
    const std::vector<cv::Mat> &auxs, const std::vector<cv::Mat> &aux_vars,
    const std::filesystem::path &output_dir) {
    auto aux_size = auxs.size();
    std::vector<float> aux_weights;
    for(auto k = 0; k < aux_size; k++) {
        aux_weights.emplace_back(0.6);
    }
    auto [r_img, r_var, r_sure, r_d] = feature_filter_with_var(
        device, stream, image, image, var, auxs, aux_vars, 0.45, aux_weights, 10, 1, true);

    auto [g_img, g_var, g_sure, g_d] = feature_filter_with_var(
        device, stream, image, image, var, auxs, aux_vars, 0.45, aux_weights, 10, 3, true);

    auto [b_img, b_var, b_sure, b_d] = feature_filter_with_var(
        device, stream, image, image, var, auxs, aux_vars, 1e10, aux_weights, 10, 3, true);

    auto filter_without_feature = [&](auto input, auto sigma, auto K, auto nlm_K) {
        auto result = feature_filter_with_var(
            device, stream, input, image, var, {}, {}, sigma, {}, K, nlm_K, false);
        return std::get<0>(result);
    };

    auto filter_with_feature = [&](auto input, auto sigma, auto aux_sigmas, auto K, auto nlm_K) {
        auto result = feature_filter_with_var(
            device, stream, input, image, var, auxs, aux_vars, sigma, aux_sigmas, K, nlm_K, false);
        return std::get<0>(result);
    };

    auto r_sure_filtered = filter_without_feature(r_sure, 1, 1, 3);
    auto g_sure_filtered = filter_without_feature(g_sure, 1, 1, 3);
    auto b_sure_filtered = filter_without_feature(b_sure, 1, 1, 3);

    auto sel = calc_selection_map(device, stream, {r_sure_filtered, g_sure_filtered, b_sure_filtered}, {r_d, g_d});

    auto sel_filtered = filter_with_feature(sel, 0.5, std::vector{0.6f, 0.6f}, 3, 1);

    auto fused_filtered = fuse_selection_map(device, stream, {r_img, g_img, b_img}, sel_filtered);

    return cvt_to_bgr(fused_filtered);
}