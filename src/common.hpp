#pragma once

#include <filesystem>
#include <concepts>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <cxxopts.hpp>

template<typename T>
constexpr auto is_cv_vec = false;

template<int N>
constexpr auto is_cv_vec<cv::Vec<float, N>> = true;

template<typename T>
concept cv_vec = is_cv_vec<T>;

[[nodiscard]] inline auto index_of(int m, int row, int col, int rows, int cols) noexcept {
    // m = 0, dx; m = 1, dy
    return row * cols + col + m * rows * cols;
}

[[nodiscard]] inline auto cvt_to_rgba(const cv::Mat &mat) {
    if (mat.type() == CV_32FC4) return mat;
    cv::Mat output(mat.size(), CV_32FC4, 0.f);
    cv::cvtColor(mat, output, cv::COLOR_BGR2RGBA);
    return output;
}

[[nodiscard]] inline auto cvt_to_bgr(const cv::Mat &mat) {
    if (mat.type() == CV_32FC3) return mat;
    cv::Mat output(mat.size(), CV_32FC3, 0.f);
    cv::cvtColor(mat, output, cv::COLOR_RGBA2BGR);
    return output;
}

inline void imsave(const std::filesystem::path &base_dir, const std::string &name, const cv::Mat &image, bool force_save=false) noexcept {
    if (force_save) { // For debug
        if (image.type() == CV_32FC4) {
            cv::imwrite((base_dir / name).string(), cvt_to_bgr(image));
        } else {
            cv::imwrite((base_dir / name).string(), image);
        }
    }
}

inline auto joint_bilateral_filter(const cv::Mat &input, const cv::Mat &guidance, int diameter, double sigma_color, double sigma_space) {
    cv::Mat output;
    cv::ximgproc::jointBilateralFilter(guidance, input, output, diameter, sigma_color, sigma_space, cv::BORDER_REFLECT_101);
    return output;
}

inline auto downsample(const cv::Mat &input) {
    cv::Mat output;
    cv::resize(input, output, cv::Size(0, 0), 0.5, 0.5);
    return output;
}

inline auto upsample(const cv::Mat &input) {
    cv::Mat output;
    cv::resize(input, output, cv::Size(0, 0), 2, 2);
    return output;
}

inline auto pyramid(const cv::Mat &image, int level) {
    std::vector<cv::Mat> G;
    std::vector<cv::Mat> L;

    G.emplace_back(image);
    for (int i = 1; i < level; i++) {
        if (image.empty()) {
            G.emplace_back(image);
            L.emplace_back(image);
            continue;
        }
        auto I = G.back();
        auto I_down = downsample(I);
        G.emplace_back(I_down);
        L.emplace_back(I - upsample(I_down));
    }

    L.emplace_back(G.back());

    return std::make_pair(G, L);
}

inline auto downsample_dx(cv::Mat image) {
    auto cols = image.cols / 2;
    auto rows = image.rows / 2;
    auto to_x = [&](auto i) {
        return cv::borderInterpolate(i, image.rows, cv::BORDER_REFLECT_101);
    };
    auto to_y = [&](auto j) {
        return cv::borderInterpolate(j, image.cols, cv::BORDER_REFLECT_101);
    };
    auto img_at = [&](auto i, auto j) {
        return image.at<cv::Vec3f>(to_x(i), to_y(j));
    };

    cv::Mat output{cv::Size(cols, rows), image.type()};
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output.at<cv::Vec3f>(i, j) = img_at(i * 2, j * 2) + img_at(i * 2, j * 2 + 1) * 2 + img_at(i * 2, j * 2 + 2);
            output.at<cv::Vec3f>(i, j) += img_at(i * 2 + 1, j * 2) + img_at(i * 2 + 1, j * 2 + 1) * 2 + img_at(i * 2 + 1, j * 2 + 2);
            output.at<cv::Vec3f>(i, j) /= 4.0;
        }
    }

    return output;
}

inline auto downsample_dy(cv::Mat image) {
    auto cols = image.cols / 2;
    auto rows = image.rows / 2;
    auto to_x = [&](auto i) {
        return cv::borderInterpolate(i, image.rows, cv::BORDER_REFLECT_101);
    };
    auto to_y = [&](auto j) {
        return cv::borderInterpolate(j, image.cols, cv::BORDER_REFLECT_101);
    };
    auto img_at = [&](auto i, auto j) {
        return image.at<cv::Vec3f>(to_x(i), to_y(j));
    };

    cv::Mat output{cv::Size(cols, rows), image.type()};
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output.at<cv::Vec3f>(i, j) = img_at(i * 2, j * 2) + img_at(i * 2 + 1, j * 2) * 2 + img_at(i * 2 + 2, j * 2);
            output.at<cv::Vec3f>(i, j) += img_at(i * 2, j * 2 + 1) + img_at(i * 2 + 1, j * 2 + 1) * 2 + img_at(i * 2 + 2, j * 2 + 1);
            output.at<cv::Vec3f>(i, j) /= 4.0;
        }
    }

    return output;
}

inline auto calc_dx(const cv::Mat &image) {
    auto image_x1 = image(cv::Rect(1, 0, image.cols - 1, image.rows));
    cv::copyMakeBorder(image_x1, image_x1, 0, 0, 0, 1, cv::BORDER_REFLECT_101);
    auto dx = image_x1 - image;
    return dx;
}

inline auto calc_dy(const cv::Mat &image) {
    auto image_y1 = image(cv::Rect(0, 1, image.cols, image.rows - 1));
    cv::copyMakeBorder(image_y1, image_y1, 0, 1, 0, 0, cv::BORDER_REFLECT_101);
    auto dy = image_y1 - image;
    return dy;
}

auto concat(cv::Vec3f a, cv::Vec3f b, cv::Vec3f c) {
    return cv::Vec<float, 9>{
        a[0], a[1], a[2],
        b[0], b[1], b[2],
        c[0], c[1], c[2]};
}

template<typename T>
    requires cv_vec<T>
inline auto update_glu_at(cv::Mat &image, cv::Mat &down, int i, int j) {
    auto i_d = i / 2;
    auto j_d = j / 2;
    auto I_p = image.at<T>(i, j);
    T I_hat;
    double w_ab;
    int delta_a, delta_b;

    auto img_at = [&](auto i, auto j) {
        auto x = cv::borderInterpolate(i, down.rows, cv::BORDER_REFLECT_101);
        auto y = cv::borderInterpolate(j, down.cols, cv::BORDER_REFLECT_101);
        return down.at<T>(x, y);
    };

    auto error = 1e10;
    for (int ii = 0; ii < 9; ii++) {
        for (int jj = ii + 1; jj < 9; jj++) {
            auto I_a = img_at(ii % 3 + i_d - 1, ii / 3 + j_d - 1);
            auto I_b = img_at(jj % 3 + i_d - 1, jj / 3 + j_d - 1);
            auto w = (I_p - I_b).dot(I_a - I_b) / ((I_a - I_b).dot(I_a - I_b) + 1e-5);
            w = std::clamp(w, 0.0, 1.0);
            auto hat = w * I_a + (1 - w) * I_b;
            if (auto e = cv::norm(hat - I_p); e < error) {
                I_hat = hat;
                w_ab = w;
                delta_a = ii;
                delta_b = jj;
                error = e;
            }
        }
    }

    return std::make_tuple(I_hat, delta_a, delta_b, w_ab);
}

template<typename T>
    requires cv_vec<T>
inline auto guided_linear_upsample(cv::Mat image, cv::Mat down) {
    cv::Mat output{cv::Size{image.cols, image.rows}, image.type(), cv::Scalar(0)};
    cv::Mat w{cv::Size{image.cols, image.rows}, CV_32FC3};
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            auto [I_hat, delta_a, delta_b, w_ab] = update_glu_at<T>(image, down, i, j);
            output.at<T>(i, j) = I_hat;
            w.at<cv::Vec3f>(i, j) = {static_cast<float>(delta_a), static_cast<float>(delta_b), static_cast<float>(w_ab)};
        }
    }
    return std::make_pair(output, w);
}

template<typename T>
    requires cv_vec<T>
inline auto glu_using_weight(cv::Mat down, cv::Mat weight, double exponent = 1.) {
    cv::Mat output{cv::Size(down.cols * 2, down.rows * 2), down.type(), cv::Scalar(0)};
    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            auto i_d = i / 2;
            auto j_d = j / 2;
            auto a = static_cast<int>(weight.at<cv::Vec3f>(i, j)[0]);
            auto b = static_cast<int>(weight.at<cv::Vec3f>(i, j)[1]);
            auto w = static_cast<double>(weight.at<cv::Vec3f>(i, j)[2]);

            auto img_at = [&](auto i, auto j) {
                auto x = cv::borderInterpolate(i, down.rows, cv::BORDER_REFLECT_101);
                auto y = cv::borderInterpolate(j, down.cols, cv::BORDER_REFLECT_101);
                return down.at<T>(x, y);
            };

            output.at<T>(i, j) = std::pow(w, exponent) * img_at(i_d - 1 + a % 3, j_d - 1 + a / 3) +
                                 std::pow(1 - w, exponent) * img_at(i_d - 1 + b % 3, j_d - 1 + b / 3);
        }
    }
    return output;
}

template<typename T>
    requires cv_vec<T>
inline auto glu_optimized(cv::Mat input, double threshold) {
    auto down = downsample(input);
    cv::Mat down_flag{down.rows, down.cols, CV_32SC1, cv::Scalar(0)};// A mat records how downsampling is achieved
    auto [recon, weight] = guided_linear_upsample<T>(input, down);

    for (auto iter = 0; iter < 1; iter++) {
        cv::Mat error = recon - input;
        cv::Mat flag{error.rows, error.cols, CV_32SC1, cv::Scalar{0}};
        auto pixel_count = error.rows * error.cols;
        std::vector<std::vector<std::pair<int, int>>> point_lists;
        for (int idx = 0; idx < pixel_count; idx++) {
            auto i = idx / error.cols;
            auto j = idx % error.cols;
            if (cv::norm(error.at<T>(i, j)) > threshold && flag.at<int>(i, j) == 0) {
                point_lists.emplace_back();
                auto &point_list = point_lists.back();
                // Search cluster
                int dx[8] = {1, 0, -1, 0, 1, 1, -1, -1};
                int dy[8] = {0, 1, 0, -1, 1, -1, 1, -1};
                std::vector<std::pair<int, int>> search_list;
                search_list.emplace_back(i, j);
                flag.at<int>(i, j) = 1;
                while (!search_list.empty()) {
                    auto [cur_i, cur_j] = search_list.back();
                    search_list.pop_back();
                    point_list.emplace_back(cur_i, cur_j);
                    for (int k = 0; k < 8; k++) {
                        auto new_i = cur_i + dx[k], new_j = cur_j + dy[k];
                        if (new_i < 0 || new_i >= error.rows || new_j < 0 || new_j >= error.cols) {
                            continue;
                        }
                        if (flag.at<int>(new_i, new_j) == 0 && cv::norm(error.at<T>(new_i, new_j)) > threshold) {
                            search_list.emplace_back(new_i, new_j);
                            flag.at<int>(new_i, new_j) = 1;
                        }
                    }
                }
            }
        }

        cv::Mat new_recon = recon;
        cv::Mat new_weight = weight;
        cv::Mat new_error = error;
        cv::Mat new_down = down;
        cv::Mat new_down_flag = down_flag;

        for (auto &point_list : point_lists) {
            cv::Mat max_ep{down.rows, down.cols, CV_32FC1, cv::Scalar(0)};
            float sum_error = 0, new_sum_error = 0;
            for (auto [pi, pj] : point_list) {
                auto Ep = static_cast<float>(cv::norm(error.at<T>(pi, pj)));
                sum_error += Ep;
                if (Ep > max_ep.at<float>(pi / 2, pj / 2)) {
                    max_ep.at<float>(pi / 2, pj / 2) = Ep;
                    new_down.at<T>(pi / 2, pj / 2) = input.at<T>(pi, pj);
                    // 1 + index in super block
                    // 1 2
                    // 3 4
                    new_down_flag.at<int>(pi / 2, pj / 2) = 1 + pi % 2 * 2 + pj % 2;
                }
            }
            for (auto [pi, pj] : point_list) {
                auto [I_hat, delta_a, delta_b, w_ab] = update_glu_at<T>(input, new_down, pi, pj);
                new_recon.at<T>(pi, pj) = I_hat;
                new_weight.at<cv::Vec3f>(pi, pj) = {static_cast<float>(delta_a), static_cast<float>(delta_b), static_cast<float>(w_ab)};
                new_error.at<T>(pi, pj) = new_recon.at<T>(pi, pj) - input.at<T>(pi, pj);
            }

            for (auto [pi, pj] : point_list) {
                auto Ep = static_cast<float>(cv::norm(new_error.at<T>(pi, pj)));
                new_sum_error += Ep;
            }
            if (new_sum_error > sum_error) {
                // roll back
                for (auto [pi, pj] : point_list) {
                    new_down.at<T>(pi / 2, pj / 2) = down.at<T>(pi / 2, pj / 2);
                    new_down_flag.at<int>(pi / 2, pj / 2) = down_flag.at<int>(pi / 2, pj / 2);
                    new_recon.at<T>(pi, pj) = recon.template at<T>(pi, pj);
                    new_error.at<T>(pi, pj) = error.at<T>(pi, pj);
                    new_weight.at<cv::Vec3f>(pi, pj) = weight.template at<cv::Vec3f>(pi, pj);
                }
            }
        }

        down = new_down;
        down_flag = new_down_flag;
        weight = guided_linear_upsample<T>(input, new_down).second;
        recon = glu_using_weight<T>(down, weight);
    }

    return std::make_tuple(down, weight, down_flag);
}

template<typename T>
    requires cv_vec<T>
inline auto downsample_by_flag(const cv::Mat &input, const cv::Mat &down_flag, float exponent = 1.f) {
    cv::Mat output{down_flag.rows, down_flag.cols, input.type(), cv::Scalar(0)};
    for (int i = 0; i < down_flag.rows; i++) {
        for (int j = 0; j < down_flag.cols; j++) {
            auto flag = down_flag.at<int>(i, j);
            if (flag == 0) {
                // avg
                output.at<T>(i, j) = std::pow(0.25, exponent) * (input.at<T>(i * 2, j * 2) +
                                                                 input.at<T>(i * 2 + 1, j * 2) +
                                                                 input.at<T>(i * 2, j * 2 + 1) +
                                                                 input.at<T>(i * 2 + 1, j * 2 + 1));
            } else {
                output.at<T>(i, j) = input.at<T>(i * 2 + (flag - 1) / 2, j * 2 + (flag - 1) % 2);
            }
        }
    }
    return output;
}

inline auto dx_between_super_block(const cv::Mat &dx, const cv::Mat &dy, int i1, int j1, int i2, int j2) {
    // I(i2, j2) - I(i1, j1)

    auto to_x = [&](auto image, auto i) {
        return cv::borderInterpolate(i, dx.rows, cv::BORDER_REFLECT_101);
    };
    auto to_y = [&](auto image, auto j) {
        return cv::borderInterpolate(j, dx.cols, cv::BORDER_REFLECT_101);
    };
    auto img_at = [&](auto image, auto i, auto j) {
        return image.template at<cv::Vec3f>(to_x(image, i), to_y(image, j));
    };

    if (j1 == j2 || i1 - i2 < -1 || i1 - i2 > 1) {
        // fmt::print("Error ({}, {}) - ({}, {})", i2, j2, i1, j1);
        std::abort();
    }
    auto step = j1 < j2 ? 1 : -1;
    auto left = j1 < j2 ? j1 : j2;
    auto right = j1 < j2 ? j2 : j1;
    cv::Vec3f result{0, 0, 0};
    if (i1 == i2) {
        for (int k = left; k < right; k++) {
            result += step * img_at(dx, i1, k);
        }
    } else if (i1 == i2 - 1) {
        //o-2-o-1-o
        //|   |   |
        //o-1-o-2-o
        for (int k = 0; k < right - left; k++) {
            result += (right - left - k) * step * img_at(dx, i1, k + left);
        }
        for (int k = 0; k < right - left; k++) {
            result += (k + 1) * step * img_at(dx, i2, k + left);
        }
        for (int k = left; k <= right; k++) {
            result += img_at(dy, i1, k);
        }
        result /= right - left + 1;
    } else if (i1 == i2 + 1) {
        //o-1-o-2-o
        //|   |   |
        //o-2-o-1-o
        for (int k = 0; k < right - left; k += step) {
            result += (right - left - k) * step * img_at(dx, i1, k + left);
        }
        for (int k = 0; k < right - left; k += step) {
            result += (k + 1) * step * img_at(dx, i2, k + left);
        }
        for (int k = left; k <= right; k++) {
            result += -img_at(dy, i2, k);
        }
        result /= right - left + 1;
    }
    return result;
}

inline auto dy_between_super_block(const cv::Mat &dx, const cv::Mat &dy, int i1, int j1, int i2, int j2) {
    return dx_between_super_block(dy.t(), dx.t(), j1, i1, j2, i2);
}

inline auto downsample_dx_by_flag(const cv::Mat &dx, const cv::Mat &dy, const cv::Mat &flag) {
    cv::Mat output{flag.rows, flag.cols, CV_32FC3, cv::Scalar(0)};
    // TODO: clean bunch of anonymous functions
    auto to_x = [&](auto image, auto i) {
        return cv::borderInterpolate(i, image.rows, cv::BORDER_REFLECT_101);
    };
    auto to_y = [&](auto image, auto j) {
        return cv::borderInterpolate(j, image.cols, cv::BORDER_REFLECT_101);
    };
    auto img_at = [&](auto image, auto i, auto j) {
        return image.template at<cv::Vec3f>(to_x(image, i), to_y(image, j));
    };
    auto img_at_int = [&](auto image, auto i, auto j) {
        return image.template at<int>(to_x(image, i), to_y(image, j));
    };

    for (int i = 0; i < flag.rows; i++) {
        for (int j = 0; j < flag.cols; j++) {
            auto f_p = img_at_int(flag, i, j);
            auto f_q = img_at_int(flag, i, j + 1);
            if (f_p == 0 && f_q == 0) {
                // base
                output.at<cv::Vec3f>(i, j) = img_at(dx, i * 2, j * 2) + img_at(dx, i * 2, j * 2 + 1) * 2 + img_at(dx, i * 2, j * 2 + 2);
                output.at<cv::Vec3f>(i, j) += img_at(dx, i * 2 + 1, j * 2) + img_at(dx, i * 2 + 1, j * 2 + 1) * 2 + img_at(dx, i * 2 + 1, j * 2 + 2);
                output.at<cv::Vec3f>(i, j) /= 4.0;
            } else if (f_p != 0 && f_q != 0) {
                // grad between pixel
                auto i1 = i * 2 + (f_p - 1) / 2;
                auto j1 = j * 2 + (f_p - 1) % 2;
                auto i2 = i * 2 + (f_q - 1) / 2;
                auto j2 = (j + 1) * 2 + (f_q - 1) % 2;
                output.at<cv::Vec3f>(i, j) = dx_between_super_block(dx, dy, i1, j1, i2, j2);
            } else if (f_p == 0) {
                // f_q != 0, avg between super block p
                auto i1 = i * 2;
                auto j1 = j * 2;
                auto i2 = i * 2 + (f_q - 1) / 2;
                auto j2 = (j + 1) * 2 + (f_q - 1) % 2;
                output.at<cv::Vec3f>(i, j) = dx_between_super_block(dx, dy, i1, j1, i2, j2) +
                                             dx_between_super_block(dx, dy, i1 + 1, j1, i2, j2) +
                                             dx_between_super_block(dx, dy, i1, j1 + 1, i2, j2) +
                                             dx_between_super_block(dx, dy, i1 + 1, j1 + 1, i2, j2);
                output.at<cv::Vec3f>(i, j) /= 4.0;
            } else {
                // f_p != 0, avg between super block q
                auto i1 = i * 2 + (f_p - 1) / 2;
                auto j1 = j * 2 + (f_p - 1) % 2;
                auto i2 = i * 2;
                auto j2 = (j + 1) * 2;
                output.at<cv::Vec3f>(i, j) = dx_between_super_block(dx, dy, i1, j1, i2, j2) +
                                             dx_between_super_block(dx, dy, i1, j1, i2 + 1, j2) +
                                             dx_between_super_block(dx, dy, i1, j1, i2, j2 + 1) +
                                             dx_between_super_block(dx, dy, i1, j1, i2 + 1, j2 + 1);
                output.at<cv::Vec3f>(i, j) /= 4.0;
            }
        }
    }
    return output;
}

inline auto downsample_dy_by_flag(const cv::Mat &dx, const cv::Mat &dy, const cv::Mat &flag) {
    return downsample_dx_by_flag(dy.t(), dx.t(), flag.t()).t();
}

inline auto downsample_max_filter(const cv::Mat &var) {
    auto var_down = cv::Mat{var.rows / 2, var.cols / 2, var.type(), cv::Scalar::all(0)};
    for (auto row = 0; row < var_down.rows; row++) {
        for (auto col = 0; col < var_down.cols; col++) {
            std::array ps{
                std::make_pair(row * 2, col * 2),
                std::make_pair(row * 2, col * 2 + 1),
                std::make_pair(row * 2 + 1, col * 2),
                std::make_pair(row * 2 + 1, col * 2 + 1),
            };
            cv::Vec3f v{0, 0, 0};
            for (auto [r, c] : ps) {
                auto p = var.at<cv::Vec3f>(r, c);
                for (auto i = 0; i < 3; i++) {
                    v[i] = std::max(v[i], p[i]);
                }
            }
            var_down.at<cv::Vec3f>(row, col) = v;
        }
    }
    return var_down;
}