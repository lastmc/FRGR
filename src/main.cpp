#include <thread>
#include <atomic>
#include <future>
#include <algorithm>
#include <numeric>

#include "common.hpp"
#include "gradient_opt.hpp"
#include "filter.hpp"

#include <luisa/luisa-compute.h>

auto get_output_dir(const std::string &base, const std::string &id) noexcept {
    static auto timestamp = [] {
        std::time_t t = std::time(nullptr);
        std::tm *now = std::localtime(&t);
        return fmt::format("{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}",
                           now->tm_year + 1900,
                           now->tm_mon + 1,
                           now->tm_mday,
                           now->tm_hour,
                           now->tm_min,
                           now->tm_sec);
    }();
    // base: <scene>/<spp>/file
    auto dir = std::filesystem::canonical(base).parent_path().parent_path();
    auto output_dir = dir / "outputs" / (id + "-" + timestamp);
    if (!std::filesystem::exists(output_dir))
        std::filesystem::create_directories(output_dir);
    return output_dir;
}

[[nodiscard]] auto remove_outliers(const cv::Mat &image, float k_sigma, float delta_k_sigma, int r) noexcept {
    cv::Mat m;
    cv::medianBlur(image, m, r * 2 + 1);

    auto L = [](auto v) noexcept {
        auto [b, g, r] = v.val;
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };

    auto n = (r * 2 + 1) * (r * 2 + 1) - 1;
    for (auto i = 0; i < image.rows; i++) {
        for (auto j = 0; j < image.cols; j++) {
            auto c = image.at<cv::Vec3f>(i, j);
            auto lum = L(c);
            auto sum_v = 0.0;
            auto sum_vv = 0.0;
            auto max_surround_lum = -1e10;
            auto min_surround_lum = 1e10;
            for (auto ii = i - r; ii <= i + r; ii++) {
                for (auto jj = j - r; jj <= j + r; jj++) {
                    if (ii == i && jj == j) { continue; }
                    auto pi = cv::borderInterpolate(ii, image.rows, cv::BORDER_REFLECT_101);
                    auto pj = cv::borderInterpolate(jj, image.cols, cv::BORDER_REFLECT_101);
                    auto v = image.at<cv::Vec3f>(pi, pj);
                    auto l = L(v);
                    max_surround_lum = std::max(l, max_surround_lum);
                    min_surround_lum = std::min(l, min_surround_lum);
                    sum_v += l;
                    sum_vv += l * l;
                }
            }
            auto mean = sum_v / n;
            //   sum((x - x_bar)^2) / (n - 1)
            // = sum(x^2 - 2*x_bar*x + x_bar^2) / (n - 1)
            // = (sum(x^2) - 2 * x_bar * sum(x) + n * x_bar^2) / (n - 1)
            auto var = (sum_vv - 2 * mean * sum_v + n * mean * mean) / (n - 1);
            auto std = std::sqrt(var);
            if (std::abs(lum - mean) < k_sigma * std) {
                m.at<cv::Vec3f>(i, j) = c;
            }
        }
    }
    return m;
}

[[nodiscard]] auto load_images(const std::string &dx_file,
                               const std::string &dy_file, const std::filesystem::path &output_dir) noexcept {

    static constexpr auto load = [](auto file) noexcept {
        auto image = cv::imread(file, cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
        return image;
    };
    auto dx = load(dx_file);
    auto dy = load(dy_file);

    assert(dx.cols == dy.cols && dx.rows == dy.rows && dx.cols && dx.rows &&
           dx.type() == CV_32FC3 && dy.type() == CV_32FC3 && "Invalid format");

    auto clean_dx = remove_outliers(dx, 3, 0.5, 2);
    auto clean_dy = remove_outliers(dy, 3, 0.5, 2);

    return std::make_tuple(std::move(dx), std::move(dy),
                           std::move(clean_dx), std::move(clean_dy));
}

template<int K>
[[nodiscard]] auto optimal_filter_iter_gpu(
    luisa::compute::Device &device,
    luisa::compute::Stream &stream,
    const cv::Mat &noisy,
    const cv::Mat &noisy_guidance,
    const cv::Mat &var,
    const cv::Mat &guidance_var,
    const std::vector<cv::Mat> &aux,
    const std::vector<float> &aux_weights,
    int spp) noexcept {

    // for each pixel p:
    // Q = b * b^T + diag(v)
    // min x^T Q x
    // s.t. 1^T x = 1 && -I x <= 0
    auto rows = noisy.rows;
    auto cols = noisy.cols;
    static constexpr auto N = (2 * K + 1) * (2 * K + 1);

    // solve and apply optimal weights for each pixel
    cv::Mat result{noisy.rows, noisy.cols, CV_32FC4, cv::Scalar::all(0)};
    cv::Mat result_var{var.rows, var.cols, CV_32FC4, cv::Scalar::all(0)};

    using namespace luisa::compute;

    // Load all buffers
    auto noisy_4 = cvt_to_rgba(noisy);
    auto noisy_guidance_4 = cvt_to_rgba(noisy_guidance);
    auto noisy_buffer = device.create_buffer<luisa::float3>(rows * cols);
    auto noisy_guidance_buffer = device.create_buffer<luisa::float3>(rows * cols);
    stream << noisy_buffer.copy_from(noisy_4.data) << noisy_guidance_buffer.copy_from(noisy_guidance_4.data);

    std::vector<Buffer<luisa::float3>> aux_buffers;
    for (auto &a : aux) {
        auto buf = device.create_buffer<luisa::float3>(rows * cols);
        auto a_4 = cvt_to_rgba(a);
        stream << buf.copy_from(a_4.data) << synchronize();
        aux_buffers.emplace_back(std::move(buf));
    }
    // v = var + g_var
    auto v_4 = cvt_to_rgba(guidance_var.empty() ? var : var + guidance_var);
    auto var_4 = cvt_to_rgba(var);
    auto v_buffer = device.create_buffer<luisa::float3>(rows * cols);
    auto var_buffer = device.create_buffer<luisa::float3>(rows * cols);
    auto g_var_buffer = device.create_buffer<luisa::float3>(rows * cols);
    stream << v_buffer.copy_from(v_4.data) << var_buffer.copy_from(var_4.data);

    auto result_buffer = device.create_buffer<luisa::float3>(rows * cols);
    auto result_var_buffer = device.create_buffer<luisa::float3>(rows * cols);

    uint block_size = 8;
    auto padded_block_size = block_size + 2 * K;
    auto aux_pixel_size = padded_block_size * padded_block_size;
    // N is for w and padded block size is for aux and var
    uint shared_mem_size = N * block_size * block_size + aux_pixel_size * 3 * (aux_buffers.size() + 4 /* var, g_var, noisy, noisy_guidance */);
    if (shared_mem_size > 16384) {
        // Bigger than 64K
        fmt::print("Shared mem size exceeded limit.\n");
        std::abort();
    }

    auto optimal_filter_kernel = device.compile<2>([&] {
        // x : row, y : col
        set_block_size(block_size, block_size);
        Shared<float> shared_mem{shared_mem_size};
        auto pid = dispatch_id().xy();
        auto tid = thread_x() * block_size + thread_y();
        // pid of first pixel in block
        auto base_id = make_uint2(pid.x - thread_x(), pid.y - thread_y());
        auto index_w = [&](auto i) {
            return tid * N + i;
        };
        auto index_aux = [&](auto a, auto c, auto i) {
            return block_size * block_size * N + (3 * a + c) * aux_pixel_size + i;
        };
        auto index_v = [&](auto c, auto i) {
            return index_aux(static_cast<uint>(aux_buffers.size()), c, i);
        };
        auto index_var = [&](auto c, auto i) {
            return index_aux(static_cast<uint>(aux_buffers.size()) + 1u, c, i);
        };
        auto index_noisy = [&](auto c, auto i) {
            return index_aux(static_cast<uint>(aux_buffers.size()) + 2u, c, i);
        };
        auto index_noisy_guidance = [&](auto c, auto i) {
            return index_aux(static_cast<uint>(aux_buffers.size()) + 3u, c, i);
        };

        // Write shared memory
        auto pixels_per_thread = aux_pixel_size / (block_size * block_size);
        auto more_pixels_thread = aux_pixel_size % (block_size * block_size);
        auto pixels = ite(tid < more_pixels_thread, pixels_per_thread + 1, pixels_per_thread);
        $for(p, pixels) {
            auto aux_idx_re = p * block_size * block_size + tid;
            // xy in padded block
            auto aux_idx_offset = make_uint2(aux_idx_re / padded_block_size, aux_idx_re % padded_block_size);
            // xy in image
            auto aux_idx = make_int2(aux_idx_offset + base_id) - make_int2(K, K);
            // BORDER_REFLECT_101
            aux_idx.x = ite(aux_idx.x < 0, -aux_idx.x, aux_idx.x);
            aux_idx.x = ite(aux_idx.x >= rows, 2 * rows - 2 - aux_idx.x, aux_idx.x);
            aux_idx.y = ite(aux_idx.y < 0, -aux_idx.y, aux_idx.y);
            aux_idx.y = ite(aux_idx.y >= cols, 2 * cols - 2 - aux_idx.y, aux_idx.y);
            for (auto a = 0u; a < aux_buffers.size(); a++) {
                auto aux_v = aux_buffers[a]->read(aux_idx.x * cols + aux_idx.y);
                for (auto c = 0u; c < 3u; c++) {
                    auto aid = index_aux(a, c, aux_idx_re);
                    shared_mem.write(aid, aux_v[c]);
                }
            }
            auto v_v = v_buffer->read(aux_idx.x * cols + aux_idx.y);
            auto var_v = var_buffer->read(aux_idx.x * cols + aux_idx.y);
            auto noisy_v = noisy_buffer->read(aux_idx.x * cols + aux_idx.y);
            auto noisy_guidance_v = noisy_guidance_buffer->read(aux_idx.x * cols + aux_idx.y);
            for (auto c = 0u; c < 3u; c++) {
                shared_mem.write(index_v(c, aux_idx_re), v_v[c]);
                shared_mem.write(index_var(c, aux_idx_re), var_v[c]);
                shared_mem.write(index_noisy(c, aux_idx_re), noisy_v[c]);
                shared_mem.write(index_noisy_guidance(c, aux_idx_re), noisy_guidance_v[c]);
            }
        };

        sync_block();

        $if(any(pid >= make_uint2(rows, cols))) {
            $return();
        };
        auto calc_indices = [&](auto i) {
            // calc indices of aux_i & aux_0 in padded block
            auto di = i / (2 * K + 1);
            auto dj = i % (2 * K + 1);
            auto center_x = thread_x() + K;
            auto center_y = thread_y() + K;
            auto q_x = thread_x() + di;
            auto q_y = thread_y() + dj;
            return make_uint2(q_x * padded_block_size + q_y, center_x * padded_block_size + center_y);
        };

        auto calc_aux = [&](auto a, auto c, auto indices) {
            auto aux_x = shared_mem.read(index_aux(a, c, indices.x));
            auto aux_y = shared_mem.read(index_aux(a, c, indices.y));
            auto aux = aux_x - aux_y;
            if (a == 1 || a == 2) {
                auto lambda = 1.f;
                aux += (shared_mem.read(index_noisy(c, indices.x)) - shared_mem.read(index_noisy_guidance(c, indices.x))) * lambda;
            } else if (a == 3) {
                aux += shared_mem.read(index_noisy(c, indices.x)) - shared_mem.read(index_aux(a, c, indices.x));
            }
            return aux;
        };

        // Init w
        auto inv_sum_w = def(0.f);
        $for(i, N) {
            auto indices = calc_indices(i);
            auto v = shared_mem.read(index_v(0, indices.x)) + shared_mem.read(index_v(1, indices.x)) + shared_mem.read(index_v(2, indices.x));
            auto b = def(0.f);
            for (int a = 0; a < aux_buffers.size(); a++) {
                for (int c = 0; c < 3; c++) {
                    auto aux = calc_aux(a, c, indices);
                    b += aux * aux;
                }
            }
            auto init_w = 1.f / sqrt(v + b + 0.1f);
            inv_sum_w += init_w;
            shared_mem.write(index_w(i), init_w);
        };
        inv_sum_w = 1.f / inv_sum_w;
        $for(i, N) {
            shared_mem.write(index_w(i), shared_mem.read(index_w(i)) * inv_sum_w);
        };

        // Iteration
        auto counter = def(1u);
        auto not_changed = def(0u);
        $while(true) {
            // d = e_i - w, i = counter % N
            auto dTQd = def(0.f);
            auto dTQw = def(0.f);
            for (int a = 0; a < aux_buffers.size(); a++) {
                for (int c = 0; c < 3; c++) {
                    auto bTd = def(0.f);
                    auto bTw = def(0.f);
                    $for(i, N) {
                        auto indices = calc_indices(i);
                        auto w_i = shared_mem.read(index_w(i));
                        auto b_i = calc_aux(a, c, indices);
                        bTd += (ite(i == counter % N, 1.f, 0.f) - w_i) * b_i;
                        bTw += w_i * b_i;
                    };
                    dTQw += aux_weights[a] * bTw * bTd;
                    dTQd += aux_weights[a] * bTd * bTd;
                }
            }
            $for(i, N) {
                auto indices = calc_indices(i);
                auto w_i = shared_mem.read(index_w(i));
                auto d_i = ite(i == counter % N, 1.f, 0.f) - w_i;
                for (int c = 0; c < 3; c++) {
                    auto vd_i = d_i * shared_mem.read(index_v(c, indices.x));
                    dTQw += w_i * vd_i;
                    dTQd += d_i * vd_i;
                }
            };

            auto lambda = clamp(-dTQw / dTQd, 0.f, 1.f);
            lambda = ite(dsl::isnan(lambda) | dsl::isinf(lambda), 0.f, lambda);

            inv_sum_w = 0.f;
            $for(i, N) {
                auto w_i = shared_mem.read(index_w(i));
                auto new_w_i = (1.f - lambda) * w_i + lambda * ite(i == counter % N, 1.f, 0.f);
                shared_mem.write(index_w(i), new_w_i);
                inv_sum_w += new_w_i;
            };
            inv_sum_w = 1.f / inv_sum_w;
            $for(i, N) {
                shared_mem.write(index_w(i), shared_mem.read(index_w(i)) * inv_sum_w);
            };

            auto delta_f = 2.f * lambda * dTQw + lambda * lambda * dTQd;

            $if(abs(delta_f) < 1e-8f) {
                not_changed += 1;
                $if(not_changed >= N) {
                    $break;
                };
            }
            $else {
                not_changed = 0;
            };
            $if(counter > 3 * N) {
                $break;
            };
            counter += 1u;
        };

        auto filtered_color = def(make_float3(0.f));
        auto filtered_var = def(make_float3(0.f));
        $for(i, N) {
            auto w_i = shared_mem.read(index_w(i));
            auto indices = calc_indices(i);
            for (int c = 0; c < 3; c++) {
                filtered_color[c] += w_i * shared_mem.read(index_noisy(c, indices.x));
                filtered_var[c] += w_i * w_i * shared_mem.read(index_var(c, indices.x));
            }
        };
        noisy_buffer->write(pid.x * cols + pid.y, filtered_color);
        var_buffer->write(pid.x * cols + pid.y, filtered_var);
    });

    auto padded_size = (make_uint2(rows, cols) + block_size - 1u) / block_size * block_size;
    stream << optimal_filter_kernel().dispatch(padded_size)
           << noisy_buffer.copy_to(result.data)
           << var_buffer.copy_to(result_var.data)
           << synchronize();

    return std::make_pair(cvt_to_bgr(result), cvt_to_bgr(result_var));
}

constexpr auto optimal_filter_max_radius = 15;

template<int R, typename... Args>
auto optimal_filter_with_radius_impl(int N, Args &&...args) noexcept -> std::pair<cv::Mat, cv::Mat> {
    if constexpr (R < optimal_filter_max_radius) {
        if (N == R) {
            return optimal_filter_iter_gpu<R>(std::forward<Args>(args)...);
        } else {
            return optimal_filter_with_radius_impl<R + 1>(N, std::forward<Args>(args)...);
        }
    } else {
        std::abort();
    }
}

template<typename... Args>
decltype(auto) optimal_filter_with_radius(int N, Args &&...args) noexcept {
    return optimal_filter_with_radius_impl<1>(N, std::forward<Args>(args)...);
}

auto load_all_inputs(const cxxopts::ParseResult &args) {
    // Returns : output_dir & all inputs
    auto args_to_full_path = [&args](const std::string &name) {
        auto base_path = std::filesystem::current_path();
        auto relative = args[name].as<std::string>();
        auto spp_folder = fmt::format("{}spp", args["spp"].as<int>());
        if (args.count("dir")) {
            base_path = std::filesystem::canonical(args["dir"].as<std::string>());
        }
        return (base_path / spp_folder / relative).string();
    };
    auto scene_id = fmt::format("{}spp-{}level-{}ker", args["spp"].as<int>(), args["level"].as<int>(), args["r"].as<int>());
    auto output_dir = get_output_dir(args_to_full_path("dx"), scene_id);
    auto [input_dx, input_dy, clean_dx, clean_dy] = load_images(args_to_full_path("dx"), args_to_full_path("dy"), output_dir);

    std::unordered_map<std::string, cv::Mat> inputs;
    std::vector<std::string> input_buffer_names{
        "albedo", "normal", "image", "var", "albedo_var", "normal_var", "image_pt", "var_pt"};

    inputs.emplace("dx", clean_dx);
    inputs.emplace("dy", clean_dy);
    for (auto &&key : input_buffer_names) {
        if (args.count(key)) {
            inputs.emplace(key, cv::imread(args_to_full_path(key), cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH));
        } else {
            inputs.emplace(key, cv::Mat{});
        }
    }

    return std::make_tuple(output_dir, std::move(inputs));
}

auto reconstruct_from_gradient(cv::Mat dx, cv::Mat dy, cv::Mat image) {
    auto mean = cv::mean(image);
    cv::Mat recon{image.rows, image.cols, CV_32FC3, cv::Scalar::all(0)};
    auto v = cv::Vec3d{0., 0., 0.};
    for (auto row = 0; row < image.rows; row++) {
        auto gy = row == 0 ? cv::Vec3f{0., 0., 0.} : dy.at<cv::Vec3f>(row - 1, 0);
        auto c = v += gy;
        for (auto col = 0; col < image.cols; col++) {
            auto gx = col == 0 ? cv::Vec3f{0., 0., 0.} : dx.at<cv::Vec3f>(row, col - 1);
            recon.at<cv::Vec3f>(row, col) = c += gx;
        }
    }
    auto recon_mean = cv::mean(recon);
    auto shift = mean - recon_mean;
    recon += shift;
    return recon;
}

auto calculate_var(const cv::Mat &image, const cv::Mat &EX2, const cv::Mat &EX, int spp, const cv::Mat &effective) {
    // EX2 is E(X^2)
    cv::Mat var;
    if (EX.empty()) {
        cv::Mat EX_2;
        if (effective.empty()) {
            cv::multiply(image, image, EX_2);
            var = EX2 - EX_2;
            var = var / (spp - 1);
        } else {
            cv::multiply(image / effective * spp, image / effective * spp, EX_2);
            var = EX2 / effective * spp / 5.f - EX_2;
            var = var / (effective - 1) * 5.f;
        }
    } else {
        fmt::print("Using guidance.\n");
        // use provided guidance to calculate variance
        // assume that g = E(X), so that
        // Var(X) = sigma^2 = sum(x_i - g)^2 / spp = (sum(x_i^2) - 2 * spp * g * avg(x_i) + spp * g^2) / spp = avg(x_i^2) - 2 * g * avg(x_i) + g^2
        // Var(X_bar) = Var(X) / spp
        var = EX2 - 2 * EX.mul(image) + EX.mul(EX);
        var = var / spp;
    }
    return var;
}

auto joint_bilateral_upsample(cv::Mat input, cv::Mat guidance, int radius, double sigma_color, double sigma_space) {
    cv::Mat output{input.rows * 2, input.cols * 2, input.type(), cv::Scalar{0.0}};
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            double sum_w = 0;
            for (int di = -radius; di <= radius; di++) {
                for (int dj = -radius; dj <= radius; dj++) {
                    auto space_dis = std::abs(di) + std::abs(dj);
                    auto row_up = cv::borderInterpolate(i + di, output.rows, cv::BORDER_REFLECT_101);
                    auto col_up = cv::borderInterpolate(j + dj, output.cols, cv::BORDER_REFLECT_101);
                    auto row_down = cv::borderInterpolate((i + di) / 2, input.rows, cv::BORDER_REFLECT_101);
                    auto col_down = cv::borderInterpolate((j + dj) / 2, input.cols, cv::BORDER_REFLECT_101);
                    auto w_space = std::exp(-space_dis / sigma_space);
                    auto w_color = std::exp(-cv::norm(guidance.at<cv::Vec3f>(row_up, col_up) - guidance.at<cv::Vec3f>(i, j), cv::NORM_L1) / sigma_color);
                    output.at<cv::Vec3f>(i, j) += w_space * w_color * input.at<cv::Vec3f>(row_down, col_down);
                    sum_w += w_space * w_color;
                }
            }
            output.at<cv::Vec3f>(i, j) *= 1.0 / sum_w;
        }
    }
    return output;
}

int main(int argc, char *argv[]) {
    cxxopts::Options options{"FRGR", ""};
    auto adder = options.add_options();
    adder("dx", "dx", cxxopts::value<std::string>());
    adder("dy", "dy", cxxopts::value<std::string>());
    adder("s,spp", "spp", cxxopts::value<int>());
    adder("d,dir", "base dir of all inputs", cxxopts::value<std::string>());
    adder("i,image", "noisy image", cxxopts::value<std::string>());
    adder("image_pt", "noisy image pt", cxxopts::value<std::string>());
    adder("v,var", "image of E(X^2)", cxxopts::value<std::string>());
    adder("var_pt", "noisy var pt", cxxopts::value<std::string>());
    adder("a,albedo", "albedo image", cxxopts::value<std::string>());
    adder("albedo_var", "albedo image", cxxopts::value<std::string>());
    adder("n,normal", "normal image", cxxopts::value<std::string>());
    adder("normal_var", "albedo image", cxxopts::value<std::string>());
    adder("G,gradient-weight", "gradient weight", cxxopts::value<double>()->default_value("1.0"));
    adder("A,albedo-weight", "albedo weight", cxxopts::value<double>()->default_value("1.0"));
    adder("N,normal-weight", "normal weight", cxxopts::value<double>()->default_value("1.0"));
    adder("lambda", "lambda of bilateral target", cxxopts::value<double>()->default_value("1.0"));
    adder("r,radius", "filter radius", cxxopts::value<int>()->default_value("4"));
    adder("l,level", "level of pyramid", cxxopts::value<int>()->default_value("1"));
    adder("o,output", "path to output file", cxxopts::value<std::string>());
    adder("h,help", "help");

    options.parse_positional({"dx", "dy"});
    options.positional_help("dx dy");
    auto args = options.parse(argc, argv);

    auto check_required_params = [&args]() {
        std::vector<std::string> required_list{"dx", "dy", "spp", "albedo", "normal", "albedo_var", "normal_var", "image", "var"};
        for (auto &param : required_list)
            if (!args.count(param)) return false;
        return true;
    }();

    if (args.count("help") || !check_required_params) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    auto [output_dir, inputs] = load_all_inputs(args);

#define LOAD_INPUT_BUFFER(name) \
    auto input_##name = inputs.at(#name)
    LOAD_INPUT_BUFFER(dx);
    LOAD_INPUT_BUFFER(dy);
    LOAD_INPUT_BUFFER(albedo);
    LOAD_INPUT_BUFFER(normal);
    LOAD_INPUT_BUFFER(albedo_var);
    LOAD_INPUT_BUFFER(normal_var);
    LOAD_INPUT_BUFFER(image);
    LOAD_INPUT_BUFFER(image_pt);
    LOAD_INPUT_BUFFER(var);
    LOAD_INPUT_BUFFER(var_pt);
#undef LOAD_INPUT_BUFFER
    input_image = input_image_pt;
    input_var = input_var_pt;

    namespace lc = luisa::compute;
    lc::Context ctx{argv[0]};
    auto device = ctx.create_device("cuda");
    auto stream = device.create_stream(lc::StreamTag::COMPUTE);

    auto albedo_filtered = bilateral_filter_with_var(device, stream, input_albedo, input_albedo_var, 1, 10, 10);
    input_albedo = albedo_filtered.first;
    input_albedo_var = albedo_filtered.second;
    auto normal_filtered = bilateral_filter_with_var(device, stream, input_normal, input_normal_var, 1, 10, 10);
    input_normal = normal_filtered.first;
    input_normal_var = normal_filtered.second;

    std::vector<cv::Mat> auxs{cvt_to_rgba(input_albedo), cvt_to_rgba(input_normal)};
    std::vector<cv::Mat> aux_vars{cvt_to_rgba(input_albedo_var), cvt_to_rgba(input_normal_var)};

    auto fused_filtered = coarsely_filter(device, stream, cvt_to_rgba(input_image), cvt_to_rgba(input_var),
                                          auxs, aux_vars, output_dir);

    auto input_guidance = fused_filtered;

    int level = args["level"].as<int>();

    // Generate dx, dy pyramid
    std::vector<cv::Mat> dx_pyramid, dy_pyramid;
    dx_pyramid.emplace_back(input_dx);
    dy_pyramid.emplace_back(input_dy);
    for (int i = 1; i < level; i++) {
        auto dx_down = downsample_dx(dx_pyramid.back());
        dx_pyramid.emplace_back(dx_down);
    }
    for (int i = 1; i < level; i++) {
        auto dy_down = downsample_dy(dy_pyramid.back());
        dy_pyramid.emplace_back(dy_down);
    }

    // Generate auxiliary pyramid
    auto [albedo_G, albedo_L] = pyramid(input_albedo, level);
    auto [normal_G, normal_L] = pyramid(input_normal, level);
    auto [image_G, image_L] = pyramid(input_image, level);
    auto [guidance_G, guidance_L] = pyramid(input_guidance, level);
    auto [albedo_var_G, albedo_var_L] = pyramid(input_albedo_var, level);
    auto [normal_var_G, normal_var_L] = pyramid(input_normal_var, level);
    auto [var_G, var_L] = pyramid(input_var, level);

    std::vector<cv::Mat> image_opt, albedo_opt, normal_opt, dx_opt, dy_opt, guidance_opt, weights,
        var_opt, albedo_var_opt, normal_var_opt;
    image_opt.emplace_back(input_image);
    albedo_opt.emplace_back(input_albedo);
    normal_opt.emplace_back(input_normal);
    guidance_opt.emplace_back(input_guidance);
    dx_opt.emplace_back(input_dx);
    dy_opt.emplace_back(input_dy);
    var_opt.emplace_back(input_var);
    albedo_var_opt.emplace_back(input_albedo_var);
    normal_var_opt.emplace_back(input_normal_var);
    for (int cur_level = 1; cur_level < level; cur_level++) {
        cv::Mat input{image_G[cur_level - 1].rows, image_G[cur_level - 1].cols, CV_32FC(9), cv::Scalar{0}};
        cv::mixChannels(std::vector<cv::Mat>{guidance_opt.back(), albedo_opt.back() * 0.1, normal_opt.back() * 0.1}, input, {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8});
        auto [down, weight, down_flag] = glu_optimized<cv::Vec<float, 9>>(input, 0.15);
        weights.emplace_back(weight);
        cv::Mat image_down{image_G[cur_level].rows, image_G[cur_level].cols, CV_32FC3, cv::Scalar{0}};
        cv::Mat albedo_down{image_G[cur_level].rows, image_G[cur_level].cols, CV_32FC3, cv::Scalar{0}};
        cv::Mat normal_down{image_G[cur_level].rows, image_G[cur_level].cols, CV_32FC3, cv::Scalar{0}};
        cv::mixChannels(down, image_down, {0, 0, 1, 1, 2, 2});
        cv::mixChannels(down, albedo_down, {3, 0, 4, 1, 5, 2});
        cv::mixChannels(down, normal_down, {6, 0, 7, 1, 8, 2});
        image_down = image_down / 0.1;
        image_down = downsample_by_flag<cv::Vec3f>(image_opt.back(), down_flag);
        image_opt.emplace_back(image_down);
        albedo_opt.emplace_back(albedo_down / 0.1);
        normal_opt.emplace_back(normal_down / 0.1);
        if (!input_guidance.empty()) {
            guidance_opt.emplace_back(downsample_by_flag<cv::Vec3f>(cv::Mat{guidance_opt.back()}, down_flag));
        }
        cv::Mat cur_dx = dx_opt.back();
        cv::Mat cur_dy = dy_opt.back();
        dx_opt.emplace_back(downsample_dx_by_flag(cur_dx, cur_dy, down_flag));
        dy_opt.emplace_back(downsample_dy_by_flag(cur_dx, cur_dy, down_flag));
        var_opt.emplace_back(downsample_by_flag<cv::Vec3f>(var_opt.back(), down_flag, 2.f));
        albedo_var_opt.emplace_back(downsample_by_flag<cv::Vec3f>(albedo_var_opt.back(), down_flag, 2.f));
        normal_var_opt.emplace_back(downsample_by_flag<cv::Vec3f>(normal_var_opt.back(), down_flag, 2.f));
    }

    cv::Mat upper_guidance;    // Guidance from upper level of pyramid
    cv::Mat upper_guidance_var;// Variance of upper guidance
    for (int cur_level = level - 1; cur_level >= 0; cur_level--) {
        auto dx = dx_opt[cur_level];
        auto dy = dy_opt[cur_level];
        auto albedo = albedo_opt[cur_level];
        auto normal = normal_opt[cur_level];
        auto image = image_opt[cur_level];
        auto var = var_opt[cur_level];
        auto albedo_var = albedo_var_opt[cur_level];
        auto normal_var = normal_var_opt[cur_level];
        auto guidance = guidance_opt[cur_level];
        imsave(output_dir, "guidance_" + std::to_string(cur_level) + ".exr", guidance);
        imsave(output_dir, "albedo_" + std::to_string(cur_level) + ".exr", albedo);

        float lambda = static_cast<float>(args["lambda"].as<double>() * std::pow(256. / args["spp"].as<int>(), 0.5));
        auto [output_dx, output_dy] = gradient_optimization_iter(
            device, stream,
            cvt_to_rgba(guidance), cvt_to_rgba(dx), cvt_to_rgba(dy), cvt_to_rgba(albedo), cvt_to_rgba(normal),
            lambda, output_dir);
        imsave(output_dir, "output_dx_" + std::to_string(cur_level) + ".exr", output_dx, cur_level == 0);
        imsave(output_dir, "output_dy_" + std::to_string(cur_level) + ".exr", output_dy, cur_level == 0);

        auto ref = reconstruct_from_gradient(output_dx, output_dy, image);

        // filter
        var = cv::max(var, 0.);
        imsave(output_dir, "variance_" + std::to_string(cur_level) + ".exr", var);
        cv::Mat filtered, filtered_var;
        if (!upper_guidance.empty()) {
            auto weight = weights[cur_level];
            upper_guidance = glu_using_weight<cv::Vec3f>(upper_guidance, weight);
            upper_guidance_var = glu_using_weight<cv::Vec3f>(upper_guidance_var, weight, 2.);
        } else {
            upper_guidance = image;
        }

        std::vector<cv::Mat> aux;
        std::vector<float> aux_weights;

        // First aux !!!!!!!!!!!!!!!!!!!!!
        aux.emplace_back(ref);
        aux_weights.emplace_back(args["G"].as<double>());

        aux.emplace_back(albedo);
        aux_weights.emplace_back(args["A"].as<double>() * std::pow(256. / args["spp"].as<int>(), 0.5));

        aux.emplace_back(normal);
        aux_weights.emplace_back(args["N"].as<double>() * std::pow(256. / args["spp"].as<int>(), 0.5));

        aux.emplace_back(guidance);
        aux_weights.emplace_back(std::pow(64. / args["spp"].as<int>(), 0.5));

        if(cur_level < level - 1) {
            aux.emplace_back(upper_guidance);
            aux_weights.emplace_back(std::clamp(std::pow(64. / args["spp"].as<int>(), 0.5), 0.0, 1.0));
        }

        std::tie(filtered, filtered_var) = optimal_filter_with_radius(std::clamp(args["r"].as<int>(), 1, optimal_filter_max_radius),
                                                                      device, stream, image, upper_guidance, var, upper_guidance_var,
                                                                      aux, aux_weights, args["spp"].as<int>());
        imsave(output_dir, "filtered_" + std::to_string(cur_level) + ".exr", filtered, cur_level == 0);
        if (cur_level == 0 && args.count("output")) {
            cv::imwrite(args["output"].as<std::string>(), filtered);
        }

        upper_guidance = filtered;
        upper_guidance_var = filtered_var;
    }
}