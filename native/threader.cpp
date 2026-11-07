#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>
#include <queue>
#include <array>
#include <iomanip>
#include <filesystem>
#include <cstring>
#include <cstdint>
#include <set>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

constexpr float QMIN = 0.08f;
constexpr double PI = 3.14159265358979323846;

// ----------------------------------------------------------------------------
// IEEE 754 half-precision float (16-bit)
// ----------------------------------------------------------------------------
struct float16 {
    uint16_t u = 0;

    constexpr float16() = default;
    constexpr float16(const float16&) = default;
    constexpr float16& operator=(const float16&) = default;

    float16(float val) {
        union { uint32_t u; float f; } f32infty = { 255u << 23 };
        union { uint32_t u; float f; } f16max   = { (127u + 16u) << 23 };
        union { uint32_t u; float f; } denorm_magic = { ((127u - 15u) + (23u - 10u) + 1u) << 23 };
        uint32_t sign_mask = 0x80000000u;
        union { uint32_t u; float f; } f;
        f.f = val;
        uint32_t sign = f.u & sign_mask;
        f.u ^= sign;

        if (f.u >= f16max.u) {
            u = (f.u > f32infty.u) ? 0x7e00 : 0x7c00;
        } else {
            if (f.u < (113u << 23)) {
                f.f += denorm_magic.f;
                u = static_cast<uint16_t>(f.u - denorm_magic.u);
            } else {
                uint32_t mant_odd = (f.u >> 13) & 1;
                f.u = f.u + ((15u - 127u) << 23) + 0xfff;
                f.u += mant_odd;
                u = static_cast<uint16_t>(f.u >> 13);
            }
        }
        u |= static_cast<uint16_t>(sign >> 16);
    }

    float16(double val) : float16(static_cast<float>(val)) {}
    float16(int val) : float16(static_cast<float>(val)) {}
    float16(int16_t val) : float16(static_cast<float>(val)) {}

    float to_float() const {
        union { uint32_t u; float f; } magic = { (254u - 15u) << 23 };
        union { uint32_t u; float f; } was_infnan = { (127u + 16u) << 23 };
        union { uint32_t u; float f; } o;

        o.u = (u & 0x7fff) << 13;
        o.f *= magic.f;
        if (o.f >= was_infnan.f)
            o.u |= 255u << 23;
        o.u |= (u & 0x8000) << 16;
        return o.f;
    }

    operator float() const {
        return to_float();
    }

    explicit operator int() const {
        return static_cast<int>(std::round(to_float()));
    }
};

inline std::ostream& operator<<(std::ostream& os, const float16& h) {
    return os << h.to_float();
}

inline bool operator==(const float16& a, const float16& b) {
    return a.u == b.u;
}

inline bool operator!=(const float16& a, const float16& b) {
    return a.u != b.u;
}

struct Thread {
    uint8_t r = 0, g = 0, b = 0, a = 255;
    float16 x1, y1, x2, y2;

    Thread() = default;
    Thread(uint8_t r, uint8_t g, uint8_t b, uint8_t a, float16 x1, float16 y1, float16 x2, float16 y2)
        : r(r), g(g), b(b), a(a), x1(x1), y1(y1), x2(x2), y2(y2) {}
};

struct PixelCoord {
    int y, x;
};

struct ImageBuffer {
    int width = 0;
    int height = 0;
    std::vector<float> data; // width * height * 3, RGB in [0, 1]
};

// ----------------------------------------------------------------------------
// basic helpers
// ----------------------------------------------------------------------------
static std::array<float, 3> parse_color(std::string text) {
    if (!text.empty() && text[0] == '#') {
        text = text.substr(1);
    }
    if (text.size() == 3) {
        std::string expanded;
        for (char ch : text) {
            expanded += ch;
            expanded += ch;
        }
        text = expanded;
    }
    if (text.size() != 6) {
        return {1.0f, 1.0f, 1.0f};
    }
    try {
        int r = std::stoi(text.substr(0, 2), nullptr, 16);
        int g = std::stoi(text.substr(2, 2), nullptr, 16);
        int b = std::stoi(text.substr(4, 2), nullptr, 16);
        return {r / 255.0f, g / 255.0f, b / 255.0f};
    } catch (...) {
        return {1.0f, 1.0f, 1.0f};
    }
}

static ImageBuffer load_image(const std::string& path, const std::array<float, 3>& bg) {
    int raw_w = 0, raw_h = 0, raw_channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &raw_w, &raw_h, &raw_channels, 4);
    if (!data) {
        std::cerr << "Error: failed to load image " << path << ": " << stbi_failure_reason() << "\n";
        std::exit(1);
    }

    ImageBuffer img;
    img.width = raw_w;
    img.height = raw_h;
    img.data.resize(raw_w * raw_h * 3);

    for (int y = 0; y < raw_h; ++y) {
        for (int x = 0; x < raw_w; ++x) {
            int src_idx = (y * raw_w + x) * 4;
            float r = data[src_idx + 0] / 255.0f;
            float g = data[src_idx + 1] / 255.0f;
            float b = data[src_idx + 2] / 255.0f;
            float a = data[src_idx + 3] / 255.0f;

            int dst_idx = (y * raw_w + x) * 3;
            img.data[dst_idx + 0] = std::clamp(r * a + bg[0] * (1.0f - a), 0.0f, 1.0f);
            img.data[dst_idx + 1] = std::clamp(g * a + bg[1] * (1.0f - a), 0.0f, 1.0f);
            img.data[dst_idx + 2] = std::clamp(b * a + bg[2] * (1.0f - a), 0.0f, 1.0f);
        }
    }
    stbi_image_free(data);
    return img;
}

template<int CHANNELS>
static std::vector<float> gauss_blur(const std::vector<float>& in, int W, int H, float sigma) {
    if (sigma <= 0.0f) return in;
    int r = static_cast<int>(std::ceil(3.0f * sigma));
    int k_size = 2 * r + 1;
    std::vector<float> k(k_size);
    float sum_k = 0.0f;
    for (int i = -r; i <= r; ++i) {
        float val = std::exp(-0.5f * (static_cast<float>(i * i)) / (sigma * sigma));
        k[i + r] = val;
        sum_k += val;
    }
    for (float& val : k) val /= sum_k;

    // Horizontal pass (axis 1)
    std::vector<float> tmp(W * H * CHANNELS, 0.0f);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int i = 0; i < k_size; ++i) {
                int px = std::clamp(x + i - r, 0, W - 1);
                float w = k[i];
                for (int c = 0; c < CHANNELS; ++c) {
                    tmp[(y * W + x) * CHANNELS + c] += w * in[(y * W + px) * CHANNELS + c];
                }
            }
        }
    }

    // Vertical pass (axis 0)
    std::vector<float> out(W * H * CHANNELS, 0.0f);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int i = 0; i < k_size; ++i) {
                int py = std::clamp(y + i - r, 0, H - 1);
                float w = k[i];
                for (int c = 0; c < CHANNELS; ++c) {
                    out[(y * W + x) * CHANNELS + c] += w * tmp[(py * W + x) * CHANNELS + c];
                }
            }
        }
    }
    return out;
}

static void sobel(const std::vector<float>& g, int W, int H, std::vector<float>& gx, std::vector<float>& gy) {
    gx.assign(W * H, 0.0f);
    gy.assign(W * H, 0.0f);
    auto sample = [&](int y, int x) -> float {
        int py = std::clamp(y, 0, H - 1);
        int px = std::clamp(x, 0, W - 1);
        return g[py * W + px];
    };
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float p_tl = sample(y - 1, x - 1);
            float p_tc = sample(y - 1, x);
            float p_tr = sample(y - 1, x + 1);
            float p_bl = sample(y + 1, x - 1);
            float p_bc = sample(y + 1, x);
            float p_br = sample(y + 1, x + 1);

            gx[y * W + x] = ((p_tr + 2.0f * sample(y, x + 1) + p_br) - (p_tl + 2.0f * sample(y, x - 1) + p_bl)) / 8.0f;
            gy[y * W + x] = ((p_bl + 2.0f * p_bc + p_br) - (p_tl + 2.0f * p_tc + p_tr)) / 8.0f;
        }
    }
}

static void line_pixels(int x1, int y1, int x2, int y2, std::vector<PixelCoord>& out) {
    out.clear();
    int n = std::max(std::abs(x2 - x1), std::abs(y2 - y1)) + 1;
    if (n <= 1) {
        out.push_back({y1, x1});
        return;
    }
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n - 1);
        int px = static_cast<int>(std::floor(x1 + (x2 - x1) * t + 0.5));
        int py = static_cast<int>(std::floor(y1 + (y2 - y1) * t + 0.5));
        out.push_back({py, px});
    }
}

static inline float ink_need_pixel(float r, float g, float b, const std::array<float, 3>& bg) {
    float up_r = (r - bg[0]) / std::max(1.0f - bg[0], 1e-6f);
    float dn_r = (bg[0] - r) / std::max(bg[0], 1e-6f);
    float q_r = std::clamp(r > bg[0] ? up_r : dn_r, 0.0f, 1.0f);

    float up_g = (g - bg[1]) / std::max(1.0f - bg[1], 1e-6f);
    float dn_g = (bg[1] - g) / std::max(bg[1], 1e-6f);
    float q_g = std::clamp(g > bg[1] ? up_g : dn_g, 0.0f, 1.0f);

    float up_b = (b - bg[2]) / std::max(1.0f - bg[2], 1e-6f);
    float dn_b = (bg[2] - b) / std::max(bg[2], 1e-6f);
    float q_b = std::clamp(b > bg[2] ? up_b : dn_b, 0.0f, 1.0f);

    return std::max({q_r, q_g, q_b});
}

static inline void ink_colour(const float m[3], const std::array<float, 3>& bg, float out[3]) {
    float q = std::max(ink_need_pixel(m[0], m[1], m[2], bg), 0.05f);
    for (int c = 0; c < 3; ++c) {
        out[c] = std::clamp(bg[c] + (m[c] - bg[c]) / q, 0.0f, 1.0f);
    }
}

static inline std::array<uint8_t, 3> rgb255(const float c[3]) {
    return {
        static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(c[0] * 255.0f)), 0, 255)),
        static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(c[1] * 255.0f)), 0, 255)),
        static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(c[2] * 255.0f)), 0, 255))
    };
}

static Thread ink_thread(const std::vector<float>& T, int W, int H, const std::array<float, 3>& bg, int x1, int y1, int x2, int y2) {
    std::vector<PixelCoord> pixels;
    line_pixels(x1, y1, x2, y2, pixels);
    float m[3] = {0.0f, 0.0f, 0.0f};
    for (const auto& p : pixels) {
        int idx = (p.y * W + p.x) * 3;
        m[0] += T[idx + 0];
        m[1] += T[idx + 1];
        m[2] += T[idx + 2];
    }
    float n_inv = 1.0f / static_cast<float>(pixels.size());
    m[0] *= n_inv;
    m[1] *= n_inv;
    m[2] *= n_inv;

    float col[3];
    ink_colour(m, bg, col);
    auto rgb = rgb255(col);
    return {rgb[0], rgb[1], rgb[2], 255, x1, y1, x2, y2};
}

struct TangentField {
    std::vector<float> vx;
    std::vector<float> vy;
};

static TangentField tangent_field(const std::vector<float>& T, int W, int H, float hatch_deg = 45.0f, float sigma_pre = 1.0f, float sigma_tensor = 4.0f) {
    std::vector<float> Tb = gauss_blur<3>(T, W, H, sigma_pre);
    std::vector<float> jxx(W * H, 0.0f);
    std::vector<float> jxy(W * H, 0.0f);
    std::vector<float> jyy(W * H, 0.0f);

    std::vector<float> ch_buf(W * H);
    std::vector<float> gx, gy;
    for (int ch = 0; ch < 3; ++ch) {
        for (int i = 0; i < W * H; ++i) {
            ch_buf[i] = Tb[i * 3 + ch];
        }
        sobel(ch_buf, W, H, gx, gy);
        for (int i = 0; i < W * H; ++i) {
            jxx[i] += gx[i] * gx[i];
            jxy[i] += gx[i] * gy[i];
            jyy[i] += gy[i] * gy[i];
        }
    }

    jxx = gauss_blur<1>(jxx, W, H, sigma_tensor);
    jxy = gauss_blur<1>(jxy, W, H, sigma_tensor);
    jyy = gauss_blur<1>(jyy, W, H, sigma_tensor);

    double sum_j = 0.0;
    for (int i = 0; i < W * H; ++i) {
        sum_j += jxx[i] + jyy[i];
    }
    float bias = static_cast<float>(0.5 * (sum_j / (W * H)) + 1e-9);

    float phi_n = static_cast<float>((hatch_deg + 90.0f) * PI / 180.0f);
    float nx = std::cos(phi_n);
    float ny = std::sin(phi_n);

    TangentField tf;
    tf.vx.resize(W * H);
    tf.vy.resize(W * H);

    for (int i = 0; i < W * H; ++i) {
        float xx = jxx[i] + bias * nx * nx;
        float xy = jxy[i] + bias * nx * ny;
        float yy = jyy[i] + bias * ny * ny;
        float phi = 0.5f * std::atan2(2.0f * xy, xx - yy);
        tf.vx[i] = -std::sin(phi);
        tf.vy[i] = std::cos(phi);
    }
    return tf;
}

static std::vector<std::pair<int, int>> split_colour(const std::vector<std::array<float, 3>>& cols, float tol = 0.16f, int minlen = 6) {
    int n = static_cast<int>(cols.size());
    std::vector<std::pair<int, int>> out;
    int i = 0;
    while (i < n) {
        float ref[3] = {0.0f, 0.0f, 0.0f};
        int ref_count = std::min(4, n - i);
        for (int k = 0; k < ref_count; ++k) {
            ref[0] += cols[i + k][0];
            ref[1] += cols[i + k][1];
            ref[2] += cols[i + k][2];
        }
        ref[0] /= ref_count;
        ref[1] /= ref_count;
        ref[2] /= ref_count;

        int j = n;
        for (int k = i; k < n; ++k) {
            float diff = std::max({
                std::abs(cols[k][0] - ref[0]),
                std::abs(cols[k][1] - ref[1]),
                std::abs(cols[k][2] - ref[2])
            });
            if (diff > tol) {
                j = k;
                break;
            }
        }
        if (j <= i) j = i + 1;
        if (j - i >= minlen) {
            out.push_back({i, j - 1});
        }
        i = j;
    }
    return out;
}

struct ProgressLogger {
    std::string name;
    std::chrono::steady_clock::time_point last_time;

    ProgressLogger(const std::string& n) : name(n), last_time(std::chrono::steady_clock::now()) {}

    void log(int done, int total) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_time).count();
        if (elapsed > 0.4 || done >= total) {
            last_time = now;
            double pct = std::min(100.0, 100.0 * done / std::max(total, 1));
            std::fprintf(stderr, "\r    %-17s %5.1f%%", name.c_str(), pct);
            std::fflush(stderr);
        }
    }
};

// ----------------------------------------------------------------------------
// ALGORITHM 1: greedy residual search (analysis by synthesis)
// ----------------------------------------------------------------------------
static std::vector<Thread> algo_greedy_residual(
    const std::vector<float>& T,
    int W, int H,
    const std::array<float, 3>& bg,
    const TangentField& field,
    std::mt19937_64& rng,
    float budget,
    ProgressLogger& logger,
    int target_threads = -1
) {
    int n_target = (target_threads > 0) ? target_threads : static_cast<int>(5500 * budget);
    std::vector<double> canvas(W * H * 3);
    for (int i = 0; i < W * H; ++i) {
        canvas[i * 3 + 0] = bg[0];
        canvas[i * 3 + 1] = bg[1];
        canvas[i * 3 + 2] = bg[2];
    }

    std::vector<double> err(W * H, 0.0);
    for (int i = 0; i < W * H; ++i) {
        double dr = T[i * 3 + 0] - canvas[i * 3 + 0];
        double dg = T[i * 3 + 1] - canvas[i * 3 + 1];
        double db = T[i * 3 + 2] - canvas[i * 3 + 2];
        err[i] = dr * dr + dg * dg + db * db;
    }

    std::vector<float> ang(W * H);
    for (int i = 0; i < W * H; ++i) {
        ang[i] = std::atan2(field.vy[i], field.vx[i]);
    }

    const double ALPHAS[3] = {0.3, 0.6, 1.0};
    std::vector<Thread> threads;
    threads.reserve(n_target);

    auto clip_x = [W](double v) -> int {
        return std::clamp(static_cast<int>(std::round(v)), 0, W - 1);
    };
    auto clip_y = [H](double v) -> int {
        return std::clamp(static_cast<int>(std::round(v)), 0, H - 1);
    };

    struct CandidateResult {
        double gain;
        double alpha;
        double col[3];
        bool valid;
    };

    std::vector<PixelCoord> pix_buf;
    auto evaluate = [&](int x1, int y1, int x2, int y2) -> CandidateResult {
        line_pixels(x1, y1, x2, y2, pix_buf);
        int n = static_cast<int>(pix_buf.size());
        if (n < 4) return {0, 0, {0,0,0}, false};

        double e_old = 0.0;
        double sD[3] = {0, 0, 0};
        double sDT[3] = {0, 0, 0};
        double sT[3] = {0, 0, 0};
        double sTT[3] = {0, 0, 0};

        for (const auto& p : pix_buf) {
            int idx = (p.y * W + p.x) * 3;
            for (int c = 0; c < 3; ++c) {
                double V = canvas[idx + c];
                double Tp = T[idx + c];
                double D = V - Tp;
                e_old += D * D;
                sD[c] += D;
                sDT[c] += D * Tp;
                sT[c] += Tp;
                sTT[c] += Tp * Tp;
            }
        }

        double mD[3] = {sD[0] / n, sD[1] / n, sD[2] / n};
        double mV[3] = {(sD[0] + sT[0]) / n, (sD[1] + sT[1]) / n, (sD[2] + sT[2]) / n};

        CandidateResult best{ -1e9, 0, {0,0,0}, false };
        for (double a : ALPHAS) {
            double c[3];
            for (int ch = 0; ch < 3; ++ch) {
                c[ch] = std::clamp(mV[ch] - mD[ch] / a, 0.0, 1.0);
            }
            double term2 = 0.0;
            double term3 = 0.0;
            for (int ch = 0; ch < 3; ++ch) {
                term2 += c[ch] * sD[ch] - sDT[ch];
                term3 += n * c[ch] * c[ch] - 2.0 * c[ch] * sT[ch] + sTT[ch];
            }
            double e_new = (1.0 - a) * (1.0 - a) * e_old
                         + 2.0 * a * (1.0 - a) * term2
                         + a * a * term3;
            double g = e_old - e_new;
            if (!best.valid || g > best.gain) {
                best = { g, a, {c[0], c[1], c[2]}, true };
            }
        }
        return best;
    };

    std::vector<double> cdf(W * H);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    std::normal_distribution<double> norm0_3(0.0, 0.3);

    for (int it = 0; it < n_target * 2; ++it) {
        if (static_cast<int>(threads.size()) >= n_target) break;

        if (it % 25 == 0) {
            double acc = 0.0;
            for (int i = 0; i < W * H; ++i) {
                acc += err[i];
                cdf[i] = acc;
            }
            if (cdf.back() < 1e-6) break;
        }

        double progress = static_cast<double>(threads.size()) / n_target;
        double min_dim = std::min(W, H);
        double Lmax = (min_dim * 0.5625) * std::pow(0.12, progress);
        double Lmin = std::max(8.0, 0.2 * Lmax);
        int K = 20;

        CandidateResult best_r{ -1e9, 0, {0,0,0}, false };
        int best_cand[4] = {0, 0, 0, 0};

        for (int k = 0; k < K; ++k) {
            double r_val = dist01(rng) * cdf.back();
            auto it_pos = std::lower_bound(cdf.begin(), cdf.end(), r_val);
            int flat = std::min(static_cast<int>(std::distance(cdf.begin(), it_pos)), W * H - 1);
            int cy = flat / W;
            int cx = flat % W;

            double theta;
            if (dist01(rng) < 0.6) {
                theta = ang[flat] + norm0_3(rng);
            } else {
                theta = dist01(rng) * PI;
            }
            double L = Lmin + dist01(rng) * (Lmax - Lmin);
            double dx = std::cos(theta) * L * 0.5;
            double dy = std::sin(theta) * L * 0.5;

            int cand[4] = {
                clip_x(cx - dx), clip_y(cy - dy),
                clip_x(cx + dx), clip_y(cy + dy)
            };
            CandidateResult r = evaluate(cand[0], cand[1], cand[2], cand[3]);
            if (r.valid && (!best_r.valid || r.gain > best_r.gain)) {
                best_r = r;
                std::memcpy(best_cand, cand, sizeof(cand));
            }
        }

        if (!best_r.valid || best_r.gain <= 0.0) continue;

        int x1 = best_cand[0], y1 = best_cand[1], x2 = best_cand[2], y2 = best_cand[3];
        double sig = std::max(2.0, 0.05 * std::hypot(x2 - x1, y2 - y1));
        std::normal_distribution<double> norm_sig(0.0, sig);
        std::uniform_int_distribution<int> mode_dist(0, 2);

        for (int nudge = 0; nudge < 8; ++nudge) {
            int mode = mode_dist(rng);
            double j0 = norm_sig(rng);
            double j1 = norm_sig(rng);
            double j2 = norm_sig(rng);
            double j3 = norm_sig(rng);

            if (mode == 0) {
                j2 = 0; j3 = 0;
            } else if (mode == 1) {
                j0 = 0; j1 = 0;
            } else {
                j2 = j0; j3 = j1;
            }

            int cand[4] = {
                clip_x(x1 + j0), clip_y(y1 + j1),
                clip_x(x2 + j2), clip_y(y2 + j3)
            };
            CandidateResult r = evaluate(cand[0], cand[1], cand[2], cand[3]);
            if (r.valid && r.gain > best_r.gain) {
                best_r = r;
                x1 = cand[0]; y1 = cand[1]; x2 = cand[2]; y2 = cand[3];
            }
        }

        if (best_r.gain <= 0.0) continue;

        line_pixels(x1, y1, x2, y2, pix_buf);
        double a = best_r.alpha;
        for (const auto& p : pix_buf) {
            int idx = (p.y * W + p.x) * 3;
            for (int ch = 0; ch < 3; ++ch) {
                double V = canvas[idx + ch];
                double newV = V + a * (best_r.col[ch] - V);
                canvas[idx + ch] = newV;
            }
            double dr = T[idx + 0] - canvas[idx + 0];
            double dg = T[idx + 1] - canvas[idx + 1];
            double db = T[idx + 2] - canvas[idx + 2];
            err[p.y * W + p.x] = dr * dr + dg * dg + db * db;
        }

        uint8_t cr = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(best_r.col[0] * 255.0)), 0, 255));
        uint8_t cg = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(best_r.col[1] * 255.0)), 0, 255));
        uint8_t cb = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(best_r.col[2] * 255.0)), 0, 255));
        uint8_t ca = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(a * 255.0)), 0, 255));

        threads.push_back({cr, cg, cb, ca, x1, y1, x2, y2});
        if (threads.size() % 50 == 0) {
            logger.log(static_cast<int>(threads.size()), n_target);
        }
    }
    logger.log(n_target, n_target);
    return threads;
}

// ----------------------------------------------------------------------------
// ALGORITHM 2: edge-tangent streamlines with a local ink budget
// ----------------------------------------------------------------------------
struct Point2D {
    double x, y;
};

static std::vector<std::pair<int, int>> greedy_chords(
    const std::vector<Point2D>& P,
    const std::vector<std::array<float, 3>>& cols,
    double tol = 0.8,
    float ctol = 0.18f
) {
    int n = static_cast<int>(P.size());
    std::vector<std::pair<int, int>> out;
    int i = 0;
    while (i < n - 1) {
        int j = i + 1;
        while (j + 1 < n) {
            Point2D a = P[i];
            Point2D b = P[j + 1];
            double dx = b.x - a.x;
            double dy = b.y - a.y;
            double L = std::hypot(dx, dy);
            if (L < 1e-9) break;

            double max_dev = 0.0;
            for (int k = i + 1; k <= j; ++k) {
                double dev = std::abs((P[k].x - a.x) * dy - (P[k].y - a.y) * dx) / L;
                if (dev > max_dev) max_dev = dev;
            }
            if (max_dev > tol) break;

            float min_c[3] = {1.0f, 1.0f, 1.0f};
            float max_c[3] = {0.0f, 0.0f, 0.0f};
            for (int k = i; k <= j + 1; ++k) {
                for (int ch = 0; ch < 3; ++ch) {
                    min_c[ch] = std::min(min_c[ch], cols[k][ch]);
                    max_c[ch] = std::max(max_c[ch], cols[k][ch]);
                }
            }
            float max_diff = std::max({max_c[0] - min_c[0], max_c[1] - min_c[1], max_c[2] - min_c[2]});
            if (max_diff > ctol) break;

            j++;
        }
        out.push_back({i, j});
        i = j;
    }
    return out;
}

static std::vector<Thread> algo_flow_streamlines(
    const std::vector<float>& T,
    int W, int H,
    const std::array<float, 3>& bg,
    const TangentField& field,
    std::mt19937_64& rng,
    float budget,
    ProgressLogger& logger,
    int target_threads = -1
) {
    std::vector<float> q(W * H);
    for (int i = 0; i < W * H; ++i) {
        float val = ink_need_pixel(T[i * 3 + 0], T[i * 3 + 1], T[i * 3 + 2], bg);
        q[i] = (val > QMIN) ? val : 0.0f;
    }

    constexpr int C = 4;
    int nc_x = (W + C - 1) / C;
    int nc_y = (H + C - 1) / C;
    std::vector<double> deficit(nc_y * nc_x, 0.0);
    for (int y = 0; y < H; ++y) {
        int cy = y / C;
        for (int x = 0; x < W; ++x) {
            int cx = x / C;
            deficit[cy * nc_x + cx] += q[y * W + x];
        }
    }

    int max_threads = (target_threads > 0) ? target_threads : static_cast<int>(14000 * budget);
    constexpr double STEP = 3.0;
    constexpr int MAXSTEPS = 70;
    constexpr double THR_SEED = 2.0;
    constexpr double THR_CONT = 1.0;

    std::vector<uint8_t> inked(W * H, 0);
    std::vector<Thread> threads;
    threads.reserve(max_threads);

    auto trace = [&](double x0, double y0) -> std::vector<Point2D> {
        std::vector<Point2D> halves[2];
        double signs[2] = {1.0, -1.0};
        for (int s_idx = 0; s_idx < 2; ++s_idx) {
            double sign = signs[s_idx];
            double x = x0, y = y0;
            double pdx = 0, pdy = 0;
            bool has_p = false;
            for (int step = 0; step < MAXSTEPS; ++step) {
                int iy = static_cast<int>(y + 0.5);
                int ix = static_cast<int>(x + 0.5);
                if (iy < 0 || iy >= H || ix < 0 || ix >= W) break;
                double dx = field.vx[iy * W + ix];
                double dy = field.vy[iy * W + ix];
                if (!has_p) {
                    dx *= sign;
                    dy *= sign;
                    has_p = true;
                } else {
                    double dot = dx * pdx + dy * pdy;
                    if (dot < 0) {
                        dx = -dx;
                        dy = -dy;
                        dot = -dot;
                    }
                    if (dot < 0.5) break;
                }
                double nx_ = x + STEP * dx;
                double ny_ = y + STEP * dy;
                if (nx_ < 0 || ny_ < 0 || nx_ > W - 1 || ny_ > H - 1) break;
                int c_ny = static_cast<int>(ny_) / C;
                int c_nx = static_cast<int>(nx_) / C;
                if (deficit[c_ny * nc_x + c_nx] < THR_CONT) break;
                halves[s_idx].push_back({nx_, ny_});
                x = nx_; y = ny_;
                pdx = dx; pdy = dy;
            }
        }
        std::vector<Point2D> full;
        for (int i = static_cast<int>(halves[1].size()) - 1; i >= 0; --i) {
            full.push_back(halves[1][i]);
        }
        full.push_back({x0, y0});
        for (const auto& pt : halves[0]) {
            full.push_back(pt);
        }
        return full;
    };

    std::vector<double> cdf(nc_y * nc_x);
    int since = 0;
    bool cdf_valid = false;
    int max_attempts = std::max(80000, max_threads * 6);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    std::vector<PixelCoord> pix_buf;

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        if (static_cast<int>(threads.size()) >= max_threads) break;

        if (!cdf_valid || since >= 12) {
            double acc = 0.0;
            for (size_t i = 0; i < deficit.size(); ++i) {
                acc += std::max(deficit[i] - THR_SEED, 0.0);
                cdf[i] = acc;
            }
            since = 0;
            cdf_valid = true;
            if (cdf.back() <= 0.0) break;
        }
        since++;

        double r_val = dist01(rng) * cdf.back();
        auto it_pos = std::lower_bound(cdf.begin(), cdf.end(), r_val);
        int flat = std::min(static_cast<int>(std::distance(cdf.begin(), it_pos)), nc_y * nc_x - 1);
        int cy = flat / nc_x;
        int cx = flat % nc_x;

        if (deficit[cy * nc_x + cx] <= THR_SEED) continue;

        double x0 = std::min((cx + dist01(rng)) * C, static_cast<double>(W - 1));
        double y0 = std::min((cy + dist01(rng)) * C, static_cast<double>(H - 1));

        std::vector<Point2D> pts = trace(x0, y0);
        if (pts.size() < 2) {
            deficit[cy * nc_x + cx] -= THR_SEED;
            continue;
        }

        std::vector<std::array<float, 3>> cols(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            int py = std::clamp(static_cast<int>(std::round(pts[i].y)), 0, H - 1);
            int px = std::clamp(static_cast<int>(std::round(pts[i].x)), 0, W - 1);
            int idx = (py * W + px) * 3;
            cols[i] = {T[idx + 0], T[idx + 1], T[idx + 2]};
        }

        int added = 0;
        auto chords = greedy_chords(pts, cols);
        for (const auto& ch : chords) {
            if (target_threads > 0 && static_cast<int>(threads.size()) >= max_threads) break;
            int x1 = static_cast<int>(std::round(pts[ch.first].x));
            int y1 = static_cast<int>(std::round(pts[ch.first].y));
            int x2 = static_cast<int>(std::round(pts[ch.second].x));
            int y2 = static_cast<int>(std::round(pts[ch.second].y));
            if (x1 == x2 && y1 == y2) continue;

            line_pixels(x1, y1, x2, y2, pix_buf);
            int fresh_cnt = 0;
            for (const auto& p : pix_buf) {
                if (!inked[p.y * W + p.x]) fresh_cnt++;
            }
            if (static_cast<double>(fresh_cnt) / pix_buf.size() < 0.5) continue;

            for (const auto& p : pix_buf) {
                if (!inked[p.y * W + p.x]) {
                    inked[p.y * W + p.x] = 1;
                    deficit[(p.y / C) * nc_x + (p.x / C)] -= 1.0;
                }
            }
            threads.push_back(ink_thread(T, W, H, bg, x1, y1, x2, y2));
            added++;
        }

        if (added == 0) {
            deficit[cy * nc_x + cx] -= THR_SEED;
        }
        if (attempt % 40 == 0) {
            logger.log(static_cast<int>(threads.size()), max_threads);
        }
    }

    if (target_threads > 0 && static_cast<int>(threads.size()) > max_threads) {
        threads.resize(max_threads);
    }
    logger.log(1, 1);
    return threads;
}

// ----------------------------------------------------------------------------
// ALGORITHM 3: scanline dashes with equal ink mass
// ----------------------------------------------------------------------------
static std::vector<Thread> scanline_dashes_pass(
    const std::vector<float>& T,
    int W, int H,
    const std::array<float, 3>& bg,
    const std::vector<float>& q,
    int M,
    std::mt19937_64& rng,
    ProgressLogger* logger
) {
    constexpr int MAXLEN = 200;
    constexpr int CTOL = 24;

    std::vector<float> phases(H);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    for (int y = 0; y < H; ++y) phases[y] = dist01(rng);

    std::vector<Thread> threads;
    std::vector<double> cum0(W);
    std::vector<double> cum(W);
    std::vector<double> cw(W * 3);

    struct ActiveDash {
        int start, end;
        double col[3];
        bool active = false;
    };

    for (int y = 0; y < H; ++y) {
        double acc0 = 0.0;
        double acc_cw[3] = {0, 0, 0};
        for (int x = 0; x < W; ++x) {
            float q_val = q[y * W + x];
            acc0 += q_val;
            cum0[x] = acc0;
            cum[x] = acc0 + phases[y] * M;

            int idx = (y * W + x) * 3;
            acc_cw[0] += q_val * T[idx + 0];
            acc_cw[1] += q_val * T[idx + 1];
            acc_cw[2] += q_val * T[idx + 2];
            cw[x * 3 + 0] = acc_cw[0];
            cw[x * 3 + 1] = acc_cw[1];
            cw[x * 3 + 2] = acc_cw[2];
        }

        int nd = static_cast<int>(cum.back() / M);
        if (nd < 2) continue;

        ActiveDash cur;
        for (int k = 1; k < nd; ++k) {
            double target_ctr = (k + 0.5) * M;
            auto it_ctr = std::lower_bound(cum.begin(), cum.end(), target_ctr);
            int ctr = static_cast<int>(std::distance(cum.begin(), it_ctr));

            double target_lo = k * M;
            auto it_lo = std::upper_bound(cum.begin(), cum.end(), target_lo);
            int lo = static_cast<int>(std::distance(cum.begin(), it_lo));

            double target_hi = (k + 1.0) * M;
            auto it_hi = std::lower_bound(cum.begin(), cum.end(), target_hi);
            int hi = std::min(static_cast<int>(std::distance(cum.begin(), it_hi)), W - 1);

            double cq_hi = (hi >= 0) ? cum0[hi] : 0.0;
            double cq_lo = (lo > 0) ? cum0[lo - 1] : 0.0;
            double mass = std::max(cq_hi - cq_lo, 1e-6);

            float m[3];
            for (int ch = 0; ch < 3; ++ch) {
                double cw_hi = (hi >= 0) ? cw[hi * 3 + ch] : 0.0;
                double cw_lo = (lo > 0) ? cw[(lo - 1) * 3 + ch] : 0.0;
                m[ch] = static_cast<float>((cw_hi - cw_lo) / mass);
            }

            float ink_c[3];
            ink_colour(m, bg, ink_c);
            auto rgb = rgb255(ink_c);

            int x1 = std::clamp(static_cast<int>(std::floor(ctr - (M - 1) / 2.0 + 0.5)), 0, W - 1);
            int x2 = std::clamp(x1 + M - 1, 0, W - 1);

            if (cur.active && x1 <= cur.end + 1 && (x2 - cur.start) < MAXLEN &&
                std::max({std::abs(rgb[0] - cur.col[0]), std::abs(rgb[1] - cur.col[1]), std::abs(rgb[2] - cur.col[2])}) <= CTOL) {
                int n0 = cur.end - cur.start + 1;
                int n1 = x2 - cur.end;
                cur.col[0] = (cur.col[0] * n0 + rgb[0] * n1) / (n0 + n1);
                cur.col[1] = (cur.col[1] * n0 + rgb[1] * n1) / (n0 + n1);
                cur.col[2] = (cur.col[2] * n0 + rgb[2] * n1) / (n0 + n1);
                cur.end = std::max(cur.end, x2);
            } else {
                if (cur.active && cur.end > cur.start) {
                    threads.push_back({
                        static_cast<uint8_t>(std::round(cur.col[0])),
                        static_cast<uint8_t>(std::round(cur.col[1])),
                        static_cast<uint8_t>(std::round(cur.col[2])),
                        255,
                        cur.start, y,
                        cur.end, y
                    });
                }
                cur.start = x1;
                cur.end = x2;
                cur.col[0] = rgb[0];
                cur.col[1] = rgb[1];
                cur.col[2] = rgb[2];
                cur.active = true;
            }
        }
        if (cur.active && cur.end > cur.start) {
            threads.push_back({
                static_cast<uint8_t>(std::round(cur.col[0])),
                static_cast<uint8_t>(std::round(cur.col[1])),
                static_cast<uint8_t>(std::round(cur.col[2])),
                255,
                cur.start, y,
                cur.end, y
            });
        }

        if (logger && y % 16 == 0) {
            logger->log(y, H);
        }
    }
    if (logger) logger->log(1, 1);
    return threads;
}

static std::vector<Thread> algo_scanline_dashes(
    const std::vector<float>& T,
    int W, int H,
    const std::array<float, 3>& bg,
    const TangentField& field,
    std::mt19937_64& rng,
    float budget,
    ProgressLogger& logger,
    int target_threads = -1
) {
    std::vector<float> q(W * H);
    double q_sum = 0.0;
    for (int i = 0; i < W * H; ++i) {
        float val = ink_need_pixel(T[i * 3 + 0], T[i * 3 + 1], T[i * 3 + 2], bg);
        q[i] = (val > QMIN) ? val : 0.0f;
        q_sum += q[i];
    }

    if (target_threads <= 0) {
        int M = std::max(3, static_cast<int>(std::round(10.0 / budget)));
        return scanline_dashes_pass(T, W, H, bg, q, M, rng, &logger);
    }

    int M = std::max(3, static_cast<int>(std::round(q_sum / (target_threads * 1.08))));
    uint64_t seed_val = rng();
    std::mt19937_64 rng_pass1(seed_val);
    auto th = scanline_dashes_pass(T, W, H, bg, q, M, rng_pass1, nullptr);

    auto best_th = th;
    if (static_cast<int>(th.size()) > target_threads && M < W) {
        int M2 = M + 1;
        std::mt19937_64 rng_pass2(seed_val);
        auto th2 = scanline_dashes_pass(T, W, H, bg, q, M2, rng_pass2, nullptr);
        if (std::abs(static_cast<int>(th2.size()) - target_threads) < std::abs(static_cast<int>(best_th.size()) - target_threads)) {
            best_th = th2;
        }
    } else if (static_cast<int>(th.size()) < target_threads && M > 3) {
        int M2 = M - 1;
        std::mt19937_64 rng_pass2(seed_val);
        auto th2 = scanline_dashes_pass(T, W, H, bg, q, M2, rng_pass2, nullptr);
        if (std::abs(static_cast<int>(th2.size()) - target_threads) < std::abs(static_cast<int>(best_th.size()) - target_threads)) {
            best_th = th2;
        }
    }

    logger.log(1, 1);
    return best_th;
}

// ----------------------------------------------------------------------------
// ALGORITHM 4: Radon-style chord peeling
// ----------------------------------------------------------------------------
static std::vector<Thread> algo_radon_peel(
    const std::vector<float>& T,
    int W, int H,
    const std::array<float, 3>& bg,
    const TangentField& field,
    std::mt19937_64& rng,
    float budget,
    ProgressLogger& logger,
    int target_threads = -1
) {
    std::vector<float> q(W * H);
    for (int i = 0; i < W * H; ++i) {
        q[i] = ink_need_pixel(T[i * 3 + 0], T[i * 3 + 1], T[i * 3 + 2], bg);
    }

    constexpr float TAU = QMIN;
    constexpr int HBAND = 2;
    constexpr int MINLEN = 8;
    constexpr int GAP = 4;
    int n_chords = (target_threads > 0) ? std::max(target_threads, 4000) : static_cast<int>(8000 * budget);
    constexpr double S_MIN = 30.0;

    constexpr int A = 120;
    double cs[A], sn[A];
    for (int a = 0; a < A; ++a) {
        double theta = a * PI / A;
        cs[a] = std::cos(theta);
        sn[a] = std::sin(theta);
    }

    double c0_x = (W - 1) / 2.0;
    double c0_y = (H - 1) / 2.0;
    int R = static_cast<int>(std::ceil(std::sqrt(c0_x * c0_x + c0_y * c0_y))) + 2;
    int NB = 2 * R + 1;

    std::vector<uint8_t> inked(W * H, 0);
    std::vector<float> Dm = q;

    auto eff = [&](float v) -> double {
        return (v > TAU) ? static_cast<double>(v) : 0.0;
    };

    std::vector<double> S(A * NB, 0.0);
    for (int y = 0; y < H; ++y) {
        double yc = y - c0_y;
        for (int x = 0; x < W; ++x) {
            double xc = x - c0_x;
            double E = eff(Dm[y * W + x]);
            if (E <= 0.0) continue;
            for (int a = 0; a < A; ++a) {
                int idx = static_cast<int>(std::floor(xc * cs[a] + yc * sn[a] + 0.5)) + R;
                S[a * NB + idx] += E;
            }
        }
    }

    auto chord = [&](int a, int b, std::vector<int>& out_x, std::vector<int>& out_y) -> bool {
        out_x.clear();
        out_y.clear();
        int rho = b - R;
        double c_ = cs[a], s_ = sn[a];
        double lo = -1e9, hi = 1e9;

        // Intersection with x in [0, W - 1]
        double p0_x = c0_x + rho * c_;
        double dd_x = -s_;
        if (std::abs(dd_x) < 1e-9) {
            if (p0_x < -0.5 || p0_x > W - 0.5) return false;
        } else {
            double t1 = (0.0 - p0_x) / dd_x;
            double t2 = (W - 1.0 - p0_x) / dd_x;
            lo = std::max(lo, std::min(t1, t2));
            hi = std::min(hi, std::max(t1, t2));
        }

        // Intersection with y in [0, H - 1]
        double p0_y = c0_y + rho * s_;
        double dd_y = c_;
        if (std::abs(dd_y) < 1e-9) {
            if (p0_y < -0.5 || p0_y > H - 0.5) return false;
        } else {
            double t1 = (0.0 - p0_y) / dd_y;
            double t2 = (H - 1.0 - p0_y) / dd_y;
            lo = std::max(lo, std::min(t1, t2));
            hi = std::min(hi, std::max(t1, t2));
        }

        if (hi - lo < MINLEN) return false;
        double ds = 1.0 / std::max(std::abs(s_), std::abs(c_));
        for (double t = lo; t <= hi + 1e-9; t += ds) {
            int x = static_cast<int>(std::floor(c0_x + rho * c_ - t * s_ + 0.5));
            int y = static_cast<int>(std::floor(c0_y + rho * s_ + t * c_ + 0.5));
            if (x >= 0 && x < W && y >= 0 && y < H) {
                out_x.push_back(x);
                out_y.push_back(y);
            }
        }
        return !out_x.empty();
    };

    std::vector<PixelCoord> pix_buf;
    auto deposit = [&](int x1, int y1, int x2, int y2) {
        line_pixels(x1, y1, x2, y2, pix_buf);
        bool horiz = std::abs(x2 - x1) >= std::abs(y2 - y1);

        std::vector<PixelCoord> fresh_pixels;
        fresh_pixels.reserve(pix_buf.size());
        for (const auto& p : pix_buf) {
            if (!inked[p.y * W + p.x]) {
                fresh_pixels.push_back(p);
            }
            inked[p.y * W + p.x] = 1;
        }

        if (fresh_pixels.empty()) return;

        std::vector<int> counts(W * H, 0);
        std::vector<int> touched;
        touched.reserve(fresh_pixels.size() * (2 * HBAND + 1));

        for (const auto& p : fresh_pixels) {
            for (int k = -HBAND; k <= HBAND; ++k) {
                int fy = horiz ? (p.y + k) : p.y;
                int fx = horiz ? p.x : (p.x + k);
                if (fy >= 0 && fy < H && fx >= 0 && fx < W) {
                    int idx = fy * W + fx;
                    if (counts[idx] == 0) touched.push_back(idx);
                    counts[idx]++;
                }
            }
        }

        std::vector<int> thread_pix_set;
        thread_pix_set.reserve(pix_buf.size());
        for (const auto& p : pix_buf) {
            thread_pix_set.push_back(p.y * W + p.x);
        }
        std::sort(thread_pix_set.begin(), thread_pix_set.end());

        for (int idx : touched) {
            float old_val = Dm[idx];
            float new_val = std::max(old_val - static_cast<float>(counts[idx]) / (2 * HBAND + 1), 0.0f);
            if (std::binary_search(thread_pix_set.begin(), thread_pix_set.end(), idx)) {
                new_val = 0.0f;
            }
            Dm[idx] = new_val;
            double delta = eff(new_val) - eff(old_val);
            if (std::abs(delta) > 1e-9) {
                double px = (idx % W) - c0_x;
                double py = (idx / W) - c0_y;
                for (int a = 0; a < A; ++a) {
                    int b = static_cast<int>(std::floor(px * cs[a] + py * sn[a] + 0.5)) + R;
                    S[a * NB + b] += delta;
                }
            }
        }
    };

    std::vector<Thread> threads;
    int chords_used = 0;
    int total_iters = n_chords * 4;
    std::vector<int> cx_, cy_;

    for (int it = 0; it < total_iters; ++it) {
        if (target_threads > 0) {
            if (static_cast<int>(threads.size()) >= target_threads) break;
        } else {
            if (chords_used >= n_chords) break;
        }

        auto max_it = std::max_element(S.begin(), S.end());
        int flat = static_cast<int>(std::distance(S.begin(), max_it));
        if (*max_it < S_MIN) break;

        int a = flat / NB;
        int b = flat % NB;

        bool ok = chord(a, b, cx_, cy_);
        bool made = false;

        if (ok) {
            std::vector<int> valid_indices;
            for (size_t i = 0; i < cx_.size(); ++i) {
                if (Dm[cy_[i] * W + cx_[i]] > TAU) {
                    valid_indices.push_back(static_cast<int>(i));
                }
            }

            if (valid_indices.size() >= MINLEN) {
                std::vector<int> brk;
                for (size_t i = 0; i + 1 < valid_indices.size(); ++i) {
                    if (valid_indices[i + 1] - valid_indices[i] > GAP + 1) {
                        brk.push_back(static_cast<int>(i));
                    }
                }

                std::vector<int> starts, ends;
                starts.push_back(valid_indices[0]);
                for (int b_idx : brk) {
                    ends.push_back(valid_indices[b_idx]);
                    starts.push_back(valid_indices[b_idx + 1]);
                }
                ends.push_back(valid_indices.back());

                for (size_t seg = 0; seg < starts.size(); ++seg) {
                    if (target_threads > 0 && static_cast<int>(threads.size()) >= target_threads) break;
                    int s = starts[seg], e = ends[seg];
                    if (e - s + 1 < MINLEN) continue;

                    std::vector<std::array<float, 3>> cols;
                    cols.reserve(e - s + 1);
                    for (int k = s; k <= e; ++k) {
                        int idx = (cy_[k] * W + cx_[k]) * 3;
                        cols.push_back({T[idx + 0], T[idx + 1], T[idx + 2]});
                    }

                    auto runs = split_colour(cols, 0.16f, MINLEN);
                    for (const auto& r : runs) {
                        if (target_threads > 0 && static_cast<int>(threads.size()) >= target_threads) break;
                        int x1 = cx_[s + r.first];
                        int y1 = cy_[s + r.first];
                        int x2 = cx_[s + r.second];
                        int y2 = cy_[s + r.second];
                        if (x1 == x2 && y1 == y2) continue;

                        threads.push_back(ink_thread(T, W, H, bg, x1, y1, x2, y2));
                        deposit(x1, y1, x2, y2);
                        made = true;
                    }
                }
            }
        }

        if (made) {
            chords_used++;
        } else {
            S[flat] = 0.0;
        }

        if (it % 20 == 0) {
            if (target_threads > 0) {
                logger.log(static_cast<int>(threads.size()), target_threads);
            } else {
                logger.log(chords_used, n_chords);
            }
        }
    }

    if (target_threads > 0 && static_cast<int>(threads.size()) > target_threads) {
        threads.resize(target_threads);
    }
    logger.log(1, 1);
    return threads;
}

// ----------------------------------------------------------------------------
// ALGORITHM 5: stipple tour (one walking thread)
// ----------------------------------------------------------------------------
static std::vector<Thread> algo_stipple_tour(
    const std::vector<float>& T,
    int W, int H,
    const std::array<float, 3>& bg,
    const TangentField& field,
    std::mt19937_64& rng,
    float budget,
    ProgressLogger& logger,
    int target_threads = -1
) {
    std::vector<float> q(W * H);
    for (int i = 0; i < W * H; ++i) {
        q[i] = ink_need_pixel(T[i * 3 + 0], T[i * 3 + 1], T[i * 3 + 2], bg);
    }

    std::vector<float> q_sorted = q;
    size_t p99_idx = static_cast<size_t>(q.size() * 0.99);
    std::nth_element(q_sorted.begin(), q_sorted.begin() + p99_idx, q_sorted.end());
    float p99 = std::max(q_sorted[p99_idx], 1e-3f);

    std::vector<float> qn(W * H);
    for (int i = 0; i < W * H; ++i) {
        qn[i] = std::clamp(q[i] / p99, 0.0f, 1.0f);
    }

    std::vector<float> qb = gauss_blur<1>(qn, W, H, 1.5f);

    double g;
    if (target_threads > 0) {
        double keep_sum = 0.0;
        for (int i = 0; i < W * H; ++i) {
            if (qb[i] > QMIN) {
                keep_sum += std::pow(qb[i], 0.8f);
            }
        }
        double p_keep = keep_sum / (W * H);
        double diag = std::sqrt(static_cast<double>(W * H));
        g = std::max(1.5, diag / std::sqrt((target_threads / 0.97) / std::max(p_keep, 1e-6)));
    } else {
        g = 5.0 / std::sqrt(budget);
    }

    std::vector<Point2D> P;
    std::uniform_real_distribution<double> jit_dist(-g / 2.0, g / 2.0);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);

    for (double cy = g / 2.0; cy < H; cy += g) {
        for (double cx = g / 2.0; cx < W; cx += g) {
            double px = std::clamp(cx + jit_dist(rng), 0.0, static_cast<double>(W - 1));
            double py = std::clamp(cy + jit_dist(rng), 0.0, static_cast<double>(H - 1));
            int ix = static_cast<int>(std::round(px));
            int iy = static_cast<int>(std::round(py));
            float dens = qb[iy * W + ix];
            if (dens > QMIN && dist01(rng) < std::pow(dens, 0.8f)) {
                P.push_back({px, py});
            }
        }
    }

    int N = static_cast<int>(P.size());
    std::vector<Thread> threads;
    if (N < 2) return threads;

    std::vector<uint8_t> visited(N, 0);
    int cur = 0;
    double min_coord = 1e9;
    for (int i = 0; i < N; ++i) {
        double c = P[i].x + P[i].y;
        if (c < min_coord) {
            min_coord = c;
            cur = i;
        }
    }

    visited[cur] = 1;
    Point2D pos = P[cur];
    double hop_max = 3.2 * g;
    double hop_max_sq = hop_max * hop_max;

    for (int step = 0; step < N - 1; ++step) {
        double min_d_sq = 1e18;
        int nxt = -1;
        for (int i = 0; i < N; ++i) {
            if (visited[i]) continue;
            double dx = P[i].x - pos.x;
            double dy = P[i].y - pos.y;
            double d_sq = dx * dx + dy * dy;
            if (d_sq < min_d_sq) {
                min_d_sq = d_sq;
                nxt = i;
            }
        }

        if (nxt < 0) break;
        Point2D new_pt = P[nxt];
        if (min_d_sq <= hop_max_sq) {
            int x1 = static_cast<int>(std::round(pos.x));
            int y1 = static_cast<int>(std::round(pos.y));
            int x2 = static_cast<int>(std::round(new_pt.x));
            int y2 = static_cast<int>(std::round(new_pt.y));
            if (x1 != x2 || y1 != y2) {
                threads.push_back(ink_thread(T, W, H, bg, x1, y1, x2, y2));
            }
        }
        pos = new_pt;
        visited[nxt] = 1;

        if (step % 200 == 0) {
            logger.log(step, N);
        }
    }
    logger.log(1, 1);
    return threads;
}

// ----------------------------------------------------------------------------
// Core Utilities for Algorithms 6-9
// ----------------------------------------------------------------------------
static void compute_gradients_central(const std::vector<float>& gray, int W, int H, std::vector<float>& gx, std::vector<float>& gy, std::vector<float>& mag) {
    gx.assign(W * H, 0.0f);
    gy.assign(W * H, 0.0f);
    mag.assign(W * H, 0.0f);

    for (int y = 0; y < H; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            gx[y * W + x] = (gray[y * W + (x + 1)] - gray[y * W + (x - 1)]) * 0.5f;
        }
    }
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 0; x < W; ++x) {
            gy[y * W + x] = (gray[(y + 1) * W + x] - gray[(y - 1) * W + x]) * 0.5f;
        }
    }
    for (int i = 0; i < W * H; ++i) {
        mag[i] = std::hypot(gx[i], gy[i]);
    }
}

static void sample_line_color(
    const std::vector<uint8_t>& img_rgba,
    int W, int H,
    int x1, int y1, int x2, int y2,
    int n_samples,
    uint8_t& out_r, uint8_t& out_g, uint8_t& out_b, uint8_t& out_a
) {
    double sum_r = 0, sum_g = 0, sum_b = 0, sum_a = 0;
    std::vector<std::array<float, 3>> samples(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        double t = (n_samples <= 1) ? 0.0 : static_cast<double>(i) / (n_samples - 1);
        int px = std::clamp(static_cast<int>(std::round(x1 + (x2 - x1) * t)), 0, W - 1);
        int py = std::clamp(static_cast<int>(std::round(y1 + (y2 - y1) * t)), 0, H - 1);
        int idx = (py * W + px) * 4;

        float r = img_rgba[idx + 0];
        float g = img_rgba[idx + 1];
        float b = img_rgba[idx + 2];
        float a = img_rgba[idx + 3];

        sum_r += r; sum_g += g; sum_b += b; sum_a += a;
        samples[i] = {r, g, b};
    }

    double mean_r = sum_r / n_samples;
    double mean_g = sum_g / n_samples;
    double mean_b = sum_b / n_samples;
    double mean_a = sum_a / n_samples;

    double var_r = 0, var_g = 0, var_b = 0;
    for (const auto& s : samples) {
        var_r += (s[0] - mean_r) * (s[0] - mean_r);
        var_g += (s[1] - mean_g) * (s[1] - mean_g);
        var_b += (s[2] - mean_b) * (s[2] - mean_b);
    }
    double rgb_std = (std::sqrt(var_r / n_samples) + std::sqrt(var_g / n_samples) + std::sqrt(var_b / n_samples)) / 3.0;

    double coherence_alpha = std::clamp(255.0 - rgb_std * 2.2, 70.0, 255.0);
    out_a = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(coherence_alpha * (mean_a / 255.0))), 0, 255));
    out_r = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(mean_r)), 0, 255));
    out_g = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(mean_g)), 0, 255));
    out_b = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(mean_b)), 0, 255));
}

static std::vector<uint8_t> to_img_rgba(const std::vector<float>& T, int W, int H) {
    std::vector<uint8_t> rgba(W * H * 4);
    for (int i = 0; i < W * H; ++i) {
        rgba[i * 4 + 0] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(T[i * 3 + 0] * 255.0f)), 0, 255));
        rgba[i * 4 + 1] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(T[i * 3 + 1] * 255.0f)), 0, 255));
        rgba[i * 4 + 2] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(T[i * 3 + 2] * 255.0f)), 0, 255));
        rgba[i * 4 + 3] = 255;
    }
    return rgba;
}

// ----------------------------------------------------------------------------
// ALGORITHM 6: edge-guided tangent tracing
// ----------------------------------------------------------------------------
static std::vector<Thread> algo_edge_tangent(
    const std::vector<float>& T,
    int W, int H,
    const std::array<float, 3>& bg,
    const TangentField& field,
    std::mt19937_64& rng,
    float budget,
    ProgressLogger& logger,
    int target_threads = -1
) {
    int n_threads = (target_threads > 0) ? target_threads : static_cast<int>(std::round(8000 * budget));
    n_threads = std::max(1, n_threads);
    std::vector<uint8_t> img_rgba = to_img_rgba(T, W, H);

    std::vector<float> gray(W * H);
    for (int i = 0; i < W * H; ++i) {
        gray[i] = (img_rgba[i * 4 + 0] + img_rgba[i * 4 + 1] + img_rgba[i * 4 + 2]) / (3.0f * 255.0f);
    }

    std::vector<float> gx, gy, mag;
    compute_gradients_central(gray, W, H, gx, gy, mag);

    std::vector<double> weights(W * H);
    float max_mag = 0.0f;
    for (int i = 0; i < W * H; ++i) {
        weights[i] = mag[i] + 6.0f;
        if (mag[i] > max_mag) max_mag = mag[i];
    }

    std::discrete_distribution<int> weight_dist(weights.begin(), weights.end());
    std::normal_distribution<double> jitter_dist(0.0, 0.12);
    std::uniform_real_distribution<double> len_rand(0.6, 1.4);

    double base_scale = std::max(4.0, std::min(W, H) * 0.045);
    std::vector<Thread> threads;
    threads.reserve(n_threads);

    for (int i = 0; i < n_threads; ++i) {
        int flat = weight_dist(rng);
        int yc = flat / W;
        int xc = flat % W;

        double angle = std::atan2(gy[flat], gx[flat]) + (PI * 0.5) + jitter_dist(rng);
        double norm_mag = mag[flat] / (max_mag + 1e-5);
        double half_len = std::clamp(base_scale * (1.25 - 0.85 * norm_mag) * len_rand(rng), 2.0, base_scale * 2.0);

        double dx = std::cos(angle) * half_len;
        double dy = std::sin(angle) * half_len;

        int x1 = std::clamp(static_cast<int>(std::round(xc - dx)), 0, W - 1);
        int y1 = std::clamp(static_cast<int>(std::round(yc - dy)), 0, H - 1);
        int x2 = std::clamp(static_cast<int>(std::round(xc + dx)), 0, W - 1);
        int y2 = std::clamp(static_cast<int>(std::round(yc + dy)), 0, H - 1);

        uint8_t r, g, b, a;
        sample_line_color(img_rgba, W, H, x1, y1, x2, y2, 7, r, g, b, a);
        threads.push_back({r, g, b, a, x1, y1, x2, y2});
    }

    logger.log(1, 1);
    return threads;
}

// ----------------------------------------------------------------------------
// ALGORITHM 7: greedy error-reduction loom
// ----------------------------------------------------------------------------
static std::vector<Thread> algo_greedy_error(
    const std::vector<float>& T,
    int W, int H,
    const std::array<float, 3>& bg,
    const TangentField& field,
    std::mt19937_64& rng,
    float budget,
    ProgressLogger& logger,
    int target_threads = -1
) {
    int n_threads = (target_threads > 0) ? target_threads : static_cast<int>(std::round(8000 * budget));
    n_threads = std::max(1, n_threads);
    std::vector<uint8_t> img_rgba = to_img_rgba(T, W, H);

    std::vector<float> gray(W * H);
    for (int i = 0; i < W * H; ++i) {
        gray[i] = (img_rgba[i * 4 + 0] + img_rgba[i * 4 + 1] + img_rgba[i * 4 + 2]) / (3.0f * 255.0f);
    }

    std::vector<float> gx, gy, mag;
    compute_gradients_central(gray, W, H, gx, gy, mag);

    std::vector<float> importance(W * H);
    for (int i = 0; i < W * H; ++i) {
        importance[i] = mag[i] * 2.5f + 15.0f;
    }

    int max_len = std::max(5, static_cast<int>(std::min(W, H) * 0.06));
    double ux[8], uy[8];
    for (int d = 0; d < 8; ++d) {
        double theta = d * (2.0 * PI / 8.0);
        ux[d] = std::cos(theta);
        uy[d] = std::sin(theta);
    }

    std::uniform_int_distribution<int> coord_dist_y(0, H - 1);
    std::uniform_int_distribution<int> coord_dist_x(0, W - 1);
    std::uniform_int_distribution<int> len_dist(3, max_len);

    std::vector<Thread> threads;
    threads.reserve(n_threads);

    for (int i = 0; i < n_threads; ++i) {
        int best_y1 = 0, best_x1 = 0;
        float max_imp = -1.0f;
        for (int t = 0; t < 24; ++t) {
            int ty = coord_dist_y(rng);
            int tx = coord_dist_x(rng);
            float imp = importance[ty * W + tx];
            if (imp > max_imp) {
                max_imp = imp;
                best_y1 = ty;
                best_x1 = tx;
            }
        }

        int y1 = best_y1;
        int x1 = best_x1;
        int idx_start = (y1 * W + x1) * 4;
        float c_start[3] = {
            static_cast<float>(img_rgba[idx_start + 0]),
            static_cast<float>(img_rgba[idx_start + 1]),
            static_cast<float>(img_rgba[idx_start + 2])
        };

        float min_err = 1e9f;
        int best_x2 = x1, best_y2 = y1, best_mx = x1, best_my = y1;
        float best_c_mid[3] = {0,0,0}, best_c_end[3] = {0,0,0};

        for (int d = 0; d < 8; ++d) {
            int length = len_dist(rng);
            int cx2 = std::clamp(static_cast<int>(std::round(x1 + ux[d] * length)), 0, W - 1);
            int cy2 = std::clamp(static_cast<int>(std::round(y1 + uy[d] * length)), 0, H - 1);
            int cmx = (x1 + cx2) / 2;
            int cmy = (y1 + cy2) / 2;

            int idx_end = (cy2 * W + cx2) * 4;
            int idx_mid = (cmy * W + cmx) * 4;

            float c_end[3] = {
                static_cast<float>(img_rgba[idx_end + 0]),
                static_cast<float>(img_rgba[idx_end + 1]),
                static_cast<float>(img_rgba[idx_end + 2])
            };
            float c_mid[3] = {
                static_cast<float>(img_rgba[idx_mid + 0]),
                static_cast<float>(img_rgba[idx_mid + 1]),
                static_cast<float>(img_rgba[idx_mid + 2])
            };

            float err = (std::abs(c_end[0] - c_start[0]) + std::abs(c_end[1] - c_start[1]) + std::abs(c_end[2] - c_start[2]))
                      + (std::abs(c_mid[0] - c_start[0]) + std::abs(c_mid[1] - c_start[1]) + std::abs(c_mid[2] - c_start[2]));

            if (err < min_err) {
                min_err = err;
                best_x2 = cx2; best_y2 = cy2;
                best_mx = cmx; best_my = cmy;
                std::memcpy(best_c_mid, c_mid, sizeof(c_mid));
                std::memcpy(best_c_end, c_end, sizeof(c_end));
            }
        }

        float mean_col[3] = {
            (c_start[0] + best_c_mid[0] + best_c_end[0]) / 3.0f,
            (c_start[1] + best_c_mid[1] + best_c_end[1]) / 3.0f,
            (c_start[2] + best_c_mid[2] + best_c_end[2]) / 3.0f
        };
        float line_err = min_err / 6.0f;
        int alpha = static_cast<int>(std::clamp(255.0f - line_err * 1.5f, 85.0f, 255.0f));

        threads.push_back({
            static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(mean_col[0])), 0, 255)),
            static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(mean_col[1])), 0, 255)),
            static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(mean_col[2])), 0, 255)),
            static_cast<uint8_t>(alpha),
            x1, y1,
            best_x2, best_y2
        });

        importance[y1 * W + x1] *= 0.25f;
        importance[best_my * W + best_mx] *= 0.45f;
        importance[best_y2 * W + best_x2] *= 0.45f;
    }

    logger.log(1, 1);
    return threads;
}

// ----------------------------------------------------------------------------
// ALGORITHM 8: quadtree structural cross-hatching
// ----------------------------------------------------------------------------
struct QuadBox {
    double priority;
    int uid;
    int x0, y0, x1, y1;
    bool operator>(const QuadBox& o) const {
        if (priority != o.priority) return priority > o.priority;
        return uid > o.uid;
    }
};

static std::vector<Thread> algo_quadtree_hatch(
    const std::vector<float>& T,
    int W, int H,
    const std::array<float, 3>& bg,
    const TangentField& field,
    std::mt19937_64& rng,
    float budget,
    ProgressLogger& logger,
    int target_threads = -1
) {
    int n_threads = (target_threads > 0) ? target_threads : static_cast<int>(std::round(8000 * budget));
    n_threads = std::max(1, n_threads);
    std::vector<uint8_t> img_rgba = to_img_rgba(T, W, H);

    std::vector<float> gray(W * H);
    for (int i = 0; i < W * H; ++i) {
        gray[i] = (img_rgba[i * 4 + 0] + img_rgba[i * 4 + 1] + img_rgba[i * 4 + 2]) / (3.0f * 255.0f);
    }

    std::vector<float> gx, gy, mag;
    compute_gradients_central(gray, W, H, gx, gy, mag);

    int target_leaves = std::max(64, std::min(n_threads / 3, 2500));
    std::priority_queue<QuadBox, std::vector<QuadBox>, std::greater<QuadBox>> pq;
    int uid = 0;

    auto calc_box_priority = [&](int x0, int y0, int x1, int y1) -> double {
        int area = (x1 - x0) * (y1 - y0);
        if (area <= 16 || (x1 - x0) < 4 || (y1 - y0) < 4) return 0.0;

        double sum_r = 0, sum_g = 0, sum_b = 0;
        int count = 0;
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                int idx = (y * W + x) * 4;
                sum_r += img_rgba[idx + 0];
                sum_g += img_rgba[idx + 1];
                sum_b += img_rgba[idx + 2];
                count++;
            }
        }
        double mr = sum_r / count, mg = sum_g / count, mb = sum_b / count;
        double vr = 0, vg = 0, vb = 0;
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                int idx = (y * W + x) * 4;
                vr += (img_rgba[idx + 0] - mr) * (img_rgba[idx + 0] - mr);
                vg += (img_rgba[idx + 1] - mg) * (img_rgba[idx + 1] - mg);
                vb += (img_rgba[idx + 2] - mb) * (img_rgba[idx + 2] - mb);
            }
        }
        double var = (std::sqrt(vr / count) + std::sqrt(vg / count) + std::sqrt(vb / count)) / 3.0;
        return -(var * std::pow(static_cast<double>(area), 0.65));
    };

    pq.push({calc_box_priority(0, 0, W, H), uid++, 0, 0, W, H});

    while (static_cast<int>(pq.size()) < target_leaves) {
        QuadBox top = pq.top();
        pq.pop();
        if (top.priority == 0.0 || (top.x1 - top.x0) < 6 || (top.y1 - top.y0) < 6) {
            top.priority = 0.0;
            top.uid = uid++;
            pq.push(top);
            break;
        }
        int mx = (top.x0 + top.x1) / 2;
        int my = (top.y0 + top.y1) / 2;
        pq.push({calc_box_priority(top.x0, top.y0, mx, my), uid++, top.x0, top.y0, mx, my});
        pq.push({calc_box_priority(mx, top.y0, top.x1, my), uid++, mx, top.y0, top.x1, my});
        pq.push({calc_box_priority(top.x0, my, mx, top.y1), uid++, top.x0, my, mx, top.y1});
        pq.push({calc_box_priority(mx, my, top.x1, top.y1), uid++, mx, my, top.x1, top.y1});
    }

    std::vector<QuadBox> leaves;
    while (!pq.empty()) {
        leaves.push_back(pq.top());
        pq.pop();
    }

    int n_leaves = static_cast<int>(leaves.size());
    std::vector<double> weights(n_leaves);
    double weight_sum = 0.0;

    for (int i = 0; i < n_leaves; ++i) {
        const auto& box = leaves[i];
        int count = (box.x1 - box.x0) * (box.y1 - box.y0);
        double sum_r = 0, sum_g = 0, sum_b = 0;
        for (int y = box.y0; y < box.y1; ++y) {
            for (int x = box.x0; x < box.x1; ++x) {
                int idx = (y * W + x) * 4;
                sum_r += img_rgba[idx + 0];
                sum_g += img_rgba[idx + 1];
                sum_b += img_rgba[idx + 2];
            }
        }
        double mr = sum_r / count, mg = sum_g / count, mb = sum_b / count;
        double vr = 0, vg = 0, vb = 0;
        for (int y = box.y0; y < box.y1; ++y) {
            for (int x = box.x0; x < box.x1; ++x) {
                int idx = (y * W + x) * 4;
                vr += (img_rgba[idx + 0] - mr) * (img_rgba[idx + 0] - mr);
                vg += (img_rgba[idx + 1] - mg) * (img_rgba[idx + 1] - mg);
                vb += (img_rgba[idx + 2] - mb) * (img_rgba[idx + 2] - mb);
            }
        }
        double std_val = std::sqrt((vr + vg + vb) / (3.0 * count));
        weights[i] = (std_val + 5.0) * std::sqrt(count);
        weight_sum += weights[i];
    }

    std::vector<int> counts(n_leaves, 0);
    int sum_counts = 0;
    std::vector<std::pair<double, int>> remainders;
    for (int i = 0; i < n_leaves; ++i) {
        weights[i] /= weight_sum;
        double float_cnt = weights[i] * n_threads;
        counts[i] = static_cast<int>(std::floor(float_cnt));
        sum_counts += counts[i];
        remainders.push_back({float_cnt - counts[i], i});
    }

    int rem = n_threads - sum_counts;
    if (rem > 0) {
        std::sort(remainders.rbegin(), remainders.rend());
        for (int i = 0; i < rem && i < n_leaves; ++i) {
            counts[remainders[i].second]++;
        }
    }

    std::vector<Thread> threads;
    threads.reserve(n_threads);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);

    for (int i = 0; i < n_leaves; ++i) {
        int count = counts[i];
        if (count <= 0) continue;
        const auto& box = leaves[i];
        int bw = std::max(1, box.x1 - box.x0 - 1);
        int bh = std::max(1, box.y1 - box.y0 - 1);

        double sum_gx = 0, sum_gy = 0;
        int p_cnt = 0;
        for (int y = box.y0; y < box.y1; ++y) {
            for (int x = box.x0; x < box.x1; ++x) {
                sum_gx += gx[y * W + x];
                sum_gy += gy[y * W + x];
                p_cnt++;
            }
        }
        double mgx = sum_gx / p_cnt;
        double mgy = sum_gy / p_cnt;
        double angle = std::atan2(mgy, mgx) + PI * 0.5;

        for (int k = 0; k < count; ++k) {
            double ang = (k % 3 != 2) ? angle : (angle + PI * 0.5);
            double cx = box.x0 + dist01(rng) * (box.x1 - 1 - box.x0);
            double cy = box.y0 + dist01(rng) * (box.y1 - 1 - box.y0);
            double span = std::max(2.0, std::hypot(bw, bh) * 0.48);
            double dx = std::cos(ang) * span;
            double dy = std::sin(ang) * span;

            int lx1 = std::clamp(static_cast<int>(std::round(cx - dx)), box.x0, box.x1 - 1);
            int ly1 = std::clamp(static_cast<int>(std::round(cy - dy)), box.y0, box.y1 - 1);
            int lx2 = std::clamp(static_cast<int>(std::round(cx + dx)), box.x0, box.x1 - 1);
            int ly2 = std::clamp(static_cast<int>(std::round(cy + dy)), box.y0, box.y1 - 1);

            uint8_t r, g, b, a;
            sample_line_color(img_rgba, W, H, lx1, ly1, lx2, ly2, 5, r, g, b, a);
            threads.push_back({r, g, b, a, lx1, ly1, lx2, ly2});
        }
    }

    logger.log(1, 1);
    return threads;
}

// ----------------------------------------------------------------------------
// ALGORITHM 9: anchor-graph proximity web
// ----------------------------------------------------------------------------
static std::vector<Thread> algo_anchor_web(
    const std::vector<float>& T,
    int W, int H,
    const std::array<float, 3>& bg,
    const TangentField& field,
    std::mt19937_64& rng,
    float budget,
    ProgressLogger& logger,
    int target_threads = -1
) {
    int n_threads = (target_threads > 0) ? target_threads : static_cast<int>(std::round(8000 * budget));
    n_threads = std::max(1, n_threads);
    std::vector<uint8_t> img_rgba = to_img_rgba(T, W, H);

    std::vector<float> gray(W * H);
    for (int i = 0; i < W * H; ++i) {
        gray[i] = (img_rgba[i * 4 + 0] + img_rgba[i * 4 + 1] + img_rgba[i * 4 + 2]) / (3.0f * 255.0f);
    }

    std::vector<float> gx, gy, mag;
    compute_gradients_central(gray, W, H, gx, gy, mag);

    int n_anchors = std::max(300, static_cast<int>(n_threads * 0.45));
    std::vector<double> weights(W * H);
    for (int i = 0; i < W * H; ++i) {
        weights[i] = mag[i] + 4.0;
    }

    std::discrete_distribution<int> weight_dist(weights.begin(), weights.end());
    std::vector<int> anchor_indices;
    anchor_indices.reserve(n_anchors);
    std::vector<uint8_t> picked(W * H, 0);

    for (int i = 0; i < n_anchors; ++i) {
        int idx = weight_dist(rng);
        if (n_anchors < W * H) {
            int tries = 0;
            while (picked[idx] && tries < 20) {
                idx = weight_dist(rng);
                tries++;
            }
            picked[idx] = 1;
        }
        anchor_indices.push_back(idx);
    }

    int strip_w = std::max(8, W / 25);
    struct AnchorNode {
        int x, y, key;
        bool operator<(const AnchorNode& o) const { return key < o.key; }
    };

    std::vector<AnchorNode> nodes(n_anchors);
    for (int i = 0; i < n_anchors; ++i) {
        int idx = anchor_indices[i];
        int y = idx / W;
        int x = idx % W;
        nodes[i] = {x, y, (x / strip_w) * H + y};
    }
    std::sort(nodes.begin(), nodes.end());

    double max_dist_sq = std::pow(std::min(W, H) * 0.09, 2.0);
    struct EdgeCand {
        int x1, y1, x2, y2;
    };
    std::vector<EdgeCand> candidates;

    for (int offset = 1; offset <= 8; ++offset) {
        for (int i = 0; i + offset < n_anchors; ++i) {
            int dx = nodes[i + offset].x - nodes[i].x;
            int dy = nodes[i + offset].y - nodes[i].y;
            int d_sq = dx * dx + dy * dy;
            if (d_sq >= 4 && d_sq <= max_dist_sq) {
                candidates.push_back({nodes[i].x, nodes[i].y, nodes[i + offset].x, nodes[i + offset].y});
            }
        }
    }

    if (static_cast<int>(candidates.size()) < n_threads) {
        int extra = n_threads - static_cast<int>(candidates.size());
        std::uniform_int_distribution<int> a_dist(0, n_anchors - 1);
        std::uniform_int_distribution<int> delta_x(-std::max(4, W / 50), std::max(4, W / 50));
        std::uniform_int_distribution<int> delta_y(-std::max(4, H / 50), std::max(4, H / 50));
        for (int i = 0; i < extra; ++i) {
            int a_idx = a_dist(rng);
            int x1 = nodes[a_idx].x;
            int y1 = nodes[a_idx].y;
            int x2 = std::clamp(x1 + delta_x(rng), 0, W - 1);
            int y2 = std::clamp(y1 + delta_y(rng), 0, H - 1);
            candidates.push_back({x1, y1, x2, y2});
        }
    }

    struct ScoredThread {
        Thread th;
        double score;
        bool operator<(const ScoredThread& o) const { return score > o.score; }
    };

    std::vector<ScoredThread> scored;
    scored.reserve(candidates.size());
    std::uniform_real_distribution<double> jit_score(0.0, 15.0);

    for (const auto& c : candidates) {
        uint8_t r, g, b, a;
        sample_line_color(img_rgba, W, H, c.x1, c.y1, c.x2, c.y2, 5, r, g, b, a);
        double sc = a + jit_score(rng);
        scored.push_back({{r, g, b, a, c.x1, c.y1, c.x2, c.y2}, sc});
    }

    std::vector<Thread> threads;
    if (static_cast<int>(scored.size()) > n_threads) {
        std::nth_element(scored.begin(), scored.begin() + n_threads, scored.end());
        threads.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i) {
            threads.push_back(scored[i].th);
        }
    } else {
        threads.reserve(scored.size());
        for (const auto& s : scored) {
            threads.push_back(s.th);
        }
    }

    logger.log(1, 1);
    return threads;
}

// ============================================================================
// ALGORITHM 10: ThreadPainter (Continuous Greedy Optimization)
// ============================================================================
struct LineCandidate {
    float x0, y0, x1, y1;
};

static std::vector<Thread> algo_thread_painter(
    const std::vector<float>& T,
    int W, int H,
    const std::array<float, 3>& bg,
    const TangentField& field,
    std::mt19937_64& rng,
    float budget,
    ProgressLogger& logger,
    int target_threads = -1
) {
    int total_threads = (target_threads > 0) ? target_threads : static_cast<int>(8000 * budget);
    if (total_threads <= 0) total_threads = 1;

    constexpr int SAMPLES = 48;
    constexpr int INITIAL = 160;
    constexpr int PARENTS = 4;
    constexpr int CHILDREN = 12;
    const float ROUNDS[3] = {1.0f, 0.5f, 0.25f};
    constexpr int REFRESH = 8;
    constexpr int MAX_REJECTS = 40;

    float alpha = 1.0f;
    float half_width = 1.0f; // line width 2.0 -> half_width 1.0
    float reach = half_width + 1.0f; // 2.0f

    std::vector<float> canvas(W * H * 3);
    for (int i = 0; i < W * H; ++i) {
        canvas[i * 3 + 0] = bg[0];
        canvas[i * 3 + 1] = bg[1];
        canvas[i * 3 + 2] = bg[2];
    }

    std::vector<float> theta(W * H);
    for (int i = 0; i < W * H; ++i) {
        theta[i] = std::atan2(field.vy[i], field.vx[i]);
    }

    float diag = std::hypot(static_cast<float>(W), static_cast<float>(H));
    float min_len = 3.0f;
    float max_len = diag;
    float log_len = std::log(0.3f * diag);

    std::vector<double> cdf(W * H, 0.0);
    auto refresh_error_map = [&]() {
        double cum = 0.0;
        for (int i = 0; i < W * H; ++i) {
            float dr = T[i * 3 + 0] - canvas[i * 3 + 0];
            float dg = T[i * 3 + 1] - canvas[i * 3 + 1];
            float db = T[i * 3 + 2] - canvas[i * 3 + 2];
            cum += static_cast<double>(dr * dr + dg * dg + db * db);
            cdf[i] = cum;
        }
    };
    refresh_error_map();

    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::normal_distribution<float> norm01(0.0f, 1.0f);

    auto clip_line = [W, H](LineCandidate& l) {
        l.x0 = std::clamp(l.x0, 0.0f, static_cast<float>(W));
        l.y0 = std::clamp(l.y0, 0.0f, static_cast<float>(H));
        l.x1 = std::clamp(l.x1, 0.0f, static_cast<float>(W));
        l.y1 = std::clamp(l.y1, 0.0f, static_cast<float>(H));
    };

    auto propose_line = [&](float cur_log_len) -> LineCandidate {
        float mx = u01(rng) * W;
        float my = u01(rng) * H;
        double total_err = cdf.back();
        if (total_err > 0.0) {
            std::uniform_real_distribution<double> u_err(0.0, total_err);
            double pick_val = u_err(rng);
            auto it = std::lower_bound(cdf.begin(), cdf.end(), pick_val);
            int idx = static_cast<int>(it - cdf.begin());
            idx = std::clamp(idx, 0, W * H - 1);
            if (u01(rng) >= 0.15f) {
                mx = static_cast<float>(idx % W) + u01(rng);
                my = static_cast<float>(idx / W) + u01(rng);
            }
        }

        int ix = std::clamp(static_cast<int>(mx), 0, W - 1);
        int iy = std::clamp(static_cast<int>(my), 0, H - 1);

        float along_edge = theta[iy * W + ix] + norm01(rng) * 0.12f;
        float angle = (u01(rng) < 0.6f) ? along_edge : (u01(rng) * static_cast<float>(PI));

        float l_len;
        if (u01(rng) < 0.7f) {
            l_len = cur_log_len + norm01(rng) * 0.6f;
        } else {
            l_len = std::log(min_len) + u01(rng) * (std::log(max_len) - std::log(min_len));
        }
        float half = 0.5f * std::clamp(std::exp(l_len), min_len, max_len);
        float dx = std::cos(angle) * half;
        float dy = std::sin(angle) * half;

        LineCandidate l{mx - dx, my - dy, mx + dx, my + dy};
        clip_line(l);
        return l;
    };

    auto evaluate_line = [&](const LineCandidate& l, float phase) -> float {
        float dx = l.x1 - l.x0;
        float dy = l.y1 - l.y0;
        float su[3] = {0.0f, 0.0f, 0.0f};
        float qu = 0.0f;
        float r2 = 0.0f;

        for (int s = 0; s < SAMPLES; ++s) {
            float t = (static_cast<float>(s) + phase) / static_cast<float>(SAMPLES);
            int ix = std::clamp(static_cast<int>(l.x0 + t * dx), 0, W - 1);
            int iy = std::clamp(static_cast<int>(l.y0 + t * dy), 0, H - 1);
            int idx = (iy * W + ix) * 3;

            for (int c = 0; c < 3; ++c) {
                float tgt = T[idx + c];
                float cur = canvas[idx + c];
                float u = tgt - (1.0f - alpha) * cur;
                float r = tgt - cur;
                su[c] += u;
                qu += u * u;
                r2 += r * r;
            }
        }

        float ak = alpha * static_cast<float>(SAMPLES);
        float col[3];
        float col_dot_su = 0.0f;
        float col_dot_col = 0.0f;
        for (int c = 0; c < 3; ++c) {
            col[c] = std::clamp(su[c] / ak, 0.0f, 1.0f);
            col_dot_su += col[c] * su[c];
            col_dot_col += col[c] * col[c];
        }

        float err_after = qu - 2.0f * alpha * col_dot_su + static_cast<float>(SAMPLES) * alpha * alpha * col_dot_col;
        float length = std::hypot(dx, dy);
        return (r2 - err_after) * (std::max(length, 1.0f) / static_cast<float>(SAMPLES));
    };

    auto commit_line = [&](const LineCandidate& l, bool force, std::array<float, 3>& out_col) -> bool {
        int xa = std::max(static_cast<int>(std::floor(std::min(l.x0, l.x1) - reach)), 0);
        int xb = std::min(static_cast<int>(std::ceil(std::max(l.x0, l.x1) + reach)), W);
        int ya = std::max(static_cast<int>(std::floor(std::min(l.y0, l.y1) - reach)), 0);
        int yb = std::min(static_cast<int>(std::ceil(std::max(l.y0, l.y1) + reach)), H);

        if (xa >= xb || ya >= yb) return false;

        float dx = l.x1 - l.x0;
        float dy = l.y1 - l.y0;
        float len2 = dx * dx + dy * dy;

        double den = 0.0;
        double mu[3] = {0.0, 0.0, 0.0};

        int box_w = xb - xa;
        int box_h = yb - ya;
        std::vector<float> m_buf(box_w * box_h);

        for (int y = ya; y < yb; ++y) {
            float cy = static_cast<float>(y) + 0.5f;
            for (int x = xa; x < xb; ++x) {
                float cx = static_cast<float>(x) + 0.5f;
                float ddx, ddy;
                if (len2 > 1e-9f) {
                    float t = std::clamp(((cx - l.x0) * dx + (cy - l.y0) * dy) / len2, 0.0f, 1.0f);
                    ddx = cx - (l.x0 + t * dx);
                    ddy = cy - (l.y0 + t * dy);
                } else {
                    ddx = cx - l.x0;
                    ddy = cy - l.y0;
                }
                float dist = std::hypot(ddx, ddy);
                float coverage = std::clamp(half_width + 0.5f - dist, 0.0f, 1.0f);
                float m = alpha * coverage;
                m_buf[(y - ya) * box_w + (x - xa)] = m;

                den += static_cast<double>(m * m);
                int idx = (y * W + x) * 3;
                for (int c = 0; c < 3; ++c) {
                    float tp = T[idx + c];
                    float cp = canvas[idx + c];
                    float u = tp - (1.0f - m) * cp;
                    mu[c] += static_cast<double>(m * u);
                }
            }
        }

        if (den < 1e-6) return false;

        float colour[3];
        for (int c = 0; c < 3; ++c) {
            float c_val = static_cast<float>(mu[c] / den);
            colour[c] = std::clamp(c_val, 0.0f, 1.0f);
            colour[c] = std::round(colour[c] * 255.0f) / 255.0f;
        }

        double before = 0.0;
        double after = 0.0;
        for (int y = ya; y < yb; ++y) {
            for (int x = xa; x < xb; ++x) {
                float m = m_buf[(y - ya) * box_w + (x - xa)];
                int idx = (y * W + x) * 3;
                for (int c = 0; c < 3; ++c) {
                    float tp = T[idx + c];
                    float cp = canvas[idx + c];
                    float d_before = tp - cp;
                    before += static_cast<double>(d_before * d_before);

                    float new_c = cp * (1.0f - m) + m * colour[c];
                    float d_after = tp - new_c;
                    after += static_cast<double>(d_after * d_after);
                }
            }
        }

        if (after >= before && !force) return false;

        for (int y = ya; y < yb; ++y) {
            for (int x = xa; x < xb; ++x) {
                float m = m_buf[(y - ya) * box_w + (x - xa)];
                int idx = (y * W + x) * 3;
                for (int c = 0; c < 3; ++c) {
                    canvas[idx + c] = canvas[idx + c] * (1.0f - m) + m * colour[c];
                }
            }
        }

        out_col = {colour[0], colour[1], colour[2]};
        return true;
    };

    std::vector<Thread> threads;
    threads.reserve(total_threads);

    int n = 0;
    int steps = 0;
    int rejects = 0;

    struct ScoredCand {
        LineCandidate line;
        float gain;
        bool operator<(const ScoredCand& o) const { return gain > o.gain; }
    };

    while (n < total_threads) {
        if (steps % REFRESH == 0) {
            refresh_error_map();
        }
        steps++;

        float phase = u01(rng);
        std::vector<ScoredCand> pool;
        pool.reserve(INITIAL + PARENTS * CHILDREN * 3);

        for (int i = 0; i < INITIAL; ++i) {
            LineCandidate l = propose_line(log_len);
            float g = evaluate_line(l, phase);
            pool.push_back({l, g});
        }

        for (float scale : ROUNDS) {
            std::partial_sort(pool.begin(), pool.begin() + PARENTS, pool.end());
            for (int p = 0; p < PARENTS; ++p) {
                const auto& parent = pool[p].line;
                float len = std::hypot(parent.x1 - parent.x0, parent.y1 - parent.y0);
                float sigma = std::max(1.5f, 0.1f * len) * scale;

                for (int c = 0; c < CHILDREN; ++c) {
                    LineCandidate kid{
                        parent.x0 + norm01(rng) * sigma,
                        parent.y0 + norm01(rng) * sigma,
                        parent.x1 + norm01(rng) * sigma,
                        parent.y1 + norm01(rng) * sigma
                    };
                    clip_line(kid);
                    float g = evaluate_line(kid, phase);
                    pool.push_back({kid, g});
                }
            }
        }

        auto best_it = std::max_element(pool.begin(), pool.end(), [](const ScoredCand& a, const ScoredCand& b) {
            return a.gain < b.gain;
        });
        LineCandidate best_line = best_it->line;

        std::array<float, 3> committed_col;
        bool committed = commit_line(best_line, rejects >= MAX_REJECTS, committed_col);
        if (!committed) {
            rejects++;
            continue;
        }

        rejects = 0;
        uint8_t cr = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(committed_col[0] * 255.0f)), 0, 255));
        uint8_t cg = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(committed_col[1] * 255.0f)), 0, 255));
        uint8_t cb = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(committed_col[2] * 255.0f)), 0, 255));
        uint8_t ca = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(alpha * 255.0f)), 0, 255));

        threads.push_back({cr, cg, cb, ca, best_line.x0, best_line.y0, best_line.x1, best_line.y1});

        float len = std::hypot(best_line.x1 - best_line.x0, best_line.y1 - best_line.y0);
        log_len = 0.97f * log_len + 0.03f * std::log(std::max(len, min_len));
        n++;

        if (n % 200 == 0 || n == total_threads) {
            logger.log(n, total_threads);
        }
    }

    return threads;
}

// ----------------------------------------------------------------------------
// output: thread array, PNG, SVG and NPY
// ----------------------------------------------------------------------------
static std::vector<Thread> finalize_threads(const std::vector<Thread>& threads, int W, int H) {
    std::vector<Thread> out;
    out.reserve(threads.size());
    for (const auto& th : threads) {
        int x1 = std::clamp(static_cast<int>(std::round(th.x1.to_float())), 0, W - 1);
        int y1 = std::clamp(static_cast<int>(std::round(th.y1.to_float())), 0, H - 1);
        int x2 = std::clamp(static_cast<int>(std::round(th.x2.to_float())), 0, W - 1);
        int y2 = std::clamp(static_cast<int>(std::round(th.y2.to_float())), 0, H - 1);
        if (x1 == x2 && y1 == y2) continue;
        out.push_back({th.r, th.g, th.b, th.a, x1, y1, x2, y2});
    }
    return out;
}

static void render_png(const std::string& path, const std::vector<Thread>& threads, int W, int H, const std::array<float, 3>& bg = {1.0f, 1.0f, 1.0f}) {
    std::vector<float> prem(W * H * 3, 0.0f);
    std::vector<float> alpha(W * H, 0.0f);
    std::vector<PixelCoord> pix_buf;

    for (const auto& th : threads) {
        if (th.a <= 0) continue;
        line_pixels(static_cast<int>(std::round(th.x1.to_float())),
                    static_cast<int>(std::round(th.y1.to_float())),
                    static_cast<int>(std::round(th.x2.to_float())),
                    static_cast<int>(std::round(th.y2.to_float())), pix_buf);
        float af = th.a / 255.0f;
        float col[3] = {th.r / 255.0f, th.g / 255.0f, th.b / 255.0f};

        for (const auto& p : pix_buf) {
            int idx = p.y * W + p.x;
            for (int c = 0; c < 3; ++c) {
                prem[idx * 3 + c] = prem[idx * 3 + c] * (1.0f - af) + col[c] * af;
            }
            alpha[idx] = alpha[idx] * (1.0f - af) + af;
        }
    }

    std::vector<uint8_t> out(W * H * 4);
    for (int i = 0; i < W * H; ++i) {
        float a = alpha[i];
        float r = prem[i * 3 + 0] + bg[0] * (1.0f - a);
        float g = prem[i * 3 + 1] + bg[1] * (1.0f - a);
        float b = prem[i * 3 + 2] + bg[2] * (1.0f - a);

        out[i * 4 + 0] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(r * 255.0f)), 0, 255));
        out[i * 4 + 1] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(g * 255.0f)), 0, 255));
        out[i * 4 + 2] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::round(b * 255.0f)), 0, 255));
        out[i * 4 + 3] = 255;
    }

    if (!stbi_write_png(path.c_str(), W, H, 4, out.data(), W * 4)) {
        std::cerr << "Error: failed to write " << path << "\n";
        std::exit(1);
    }
}

static void write_svg(const std::string& path, const std::vector<Thread>& threads, int W, int H, const std::array<float, 3>& bg = {1.0f, 1.0f, 1.0f}) {
    std::ofstream f(path);
    if (!f) return;
    f << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W << "\" height=\"" << H
      << "\" viewBox=\"-0.5 -0.5 " << W << " " << H << "\" shape-rendering=\"crispEdges\">\n";
    int bgr = static_cast<int>(std::clamp(static_cast<int>(std::round(bg[0] * 255.0f)), 0, 255));
    int bgg = static_cast<int>(std::clamp(static_cast<int>(std::round(bg[1] * 255.0f)), 0, 255));
    int bgb = static_cast<int>(std::clamp(static_cast<int>(std::round(bg[2] * 255.0f)), 0, 255));
    f << "<rect width=\"100%\" height=\"100%\" fill=\"rgb(" << bgr << "," << bgg << "," << bgb << ")\"/>\n";
    f << "<g fill=\"none\" stroke-width=\"1\" stroke-linecap=\"square\">\n";
    for (const auto& th : threads) {
        char hex[16];
        std::snprintf(hex, sizeof(hex), "#%02x%02x%02x", th.r, th.g, th.b);
        f << "<line x1=\"" << th.x1 << "\" y1=\"" << th.y1
          << "\" x2=\"" << th.x2 << "\" y2=\"" << th.y2
          << "\" stroke=\"" << hex << "\"";
        if (th.a < 255) {
            char op[32];
            std::snprintf(op, sizeof(op), " stroke-opacity=\"%.3f\"", th.a / 255.0f);
            f << op;
        }
        f << "/>\n";
    }
    f << "</g></svg>\n";
}

static void write_npy(const std::string& path, const std::vector<Thread>& threads) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;
    const char magic[] = "\x93NUMPY\x01\x00";
    f.write(magic, 8);
    std::string header = "{'descr': '<i4', 'fortran_order': False, 'shape': (" +
                         std::to_string(threads.size()) + ", 8), }";
    size_t total = 10 + header.size() + 1;
    size_t pad = (64 - (total % 64)) % 64;
    header.append(pad, ' ');
    header += '\n';
    uint16_t hlen = static_cast<uint16_t>(header.size());
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(header.data(), header.size());
    for (const auto& th : threads) {
        int32_t row[8] = {
            th.r, th.g, th.b, th.a,
            static_cast<int32_t>(std::round(th.x1.to_float())),
            static_cast<int32_t>(std::round(th.y1.to_float())),
            static_cast<int32_t>(std::round(th.x2.to_float())),
            static_cast<int32_t>(std::round(th.y2.to_float()))
        };
        f.write(reinterpret_cast<const char*>(row), sizeof(row));
    }
}

// ----------------------------------------------------------------------------
// Algorithm registry and CLI
// ----------------------------------------------------------------------------
using AlgoFunc = std::vector<Thread>(*)(
    const std::vector<float>&,
    int, int,
    const std::array<float, 3>&,
    const TangentField&,
    std::mt19937_64&,
    float,
    ProgressLogger&,
    int
);

struct AlgoEntry {
    int id;
    std::string name;
    AlgoFunc func;
};

static const std::vector<AlgoEntry> ALGORITHMS = {
    {1, "greedy_residual", algo_greedy_residual},
    {2, "flow_streamlines", algo_flow_streamlines},
    {3, "scanline_dashes", algo_scanline_dashes},
    {4, "radon_peel", algo_radon_peel},
    {5, "stipple_tour", algo_stipple_tour},
    {6, "edge_tangent", algo_edge_tangent},
    {7, "greedy_error", algo_greedy_error},
    {8, "quadtree_hatch", algo_quadtree_hatch},
    {9, "anchor_web", algo_anchor_web},
    {10, "thread_painter", algo_thread_painter},
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <path_to_image> [options]\n\n"
              << "Options:\n"
              << "  --out <dir>       output folder (default: <image name>_threads)\n"
              << "  --bg <color>      background color hex (default: #ffffff)\n"
              << "  --budget <float>  more threads (>1) or fewer threads (<1) (default: 1.0)\n"
              << "  --threads <int>   target number of threads for all algorithms, e.g. 8000\n"
              << "  --seed <int>      random seed (default: 7)\n"
              << "  --only <list>     run only some algorithms, e.g. 1,3 or radon_peel\n"
              << "  --npy             also save the (N, 8) thread arrays as .npy\n"
              << "  -h, --help        display this help message\n";
}

static int run(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string image_path;
    std::string out_dir;
    std::string bg_str = "#ffffff";
    float budget = 1.0f;
    int target_threads = -1;
    int seed = 7;
    std::string only_str;
    bool save_npy = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (arg == "--bg" && i + 1 < argc) {
            bg_str = argv[++i];
        } else if (arg == "--budget" && i + 1 < argc) {
            budget = std::stof(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            target_threads = std::stoi(argv[++i]);
            if (target_threads <= 0) {
                std::cerr << "Error: --threads must be a positive integer\n";
                return 1;
            }
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else if (arg == "--only" && i + 1 < argc) {
            only_str = argv[++i];
        } else if (arg == "--npy") {
            save_npy = true;
        } else if (!arg.empty() && arg[0] != '-') {
            image_path = arg;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (image_path.empty()) {
        std::cerr << "Error: no input image path provided\n";
        return 1;
    }

    auto bg = parse_color(bg_str);
    ImageBuffer img = load_image(image_path, bg);
    int W = img.width;
    int H = img.height;

    fs::path p(image_path);
    std::string stem = p.stem().string();
    if (out_dir.empty()) {
        out_dir = stem + "_threads";
    }
    fs::create_directories(out_dir);

    std::set<std::string> wanted;
    if (!only_str.empty()) {
        std::stringstream ss(only_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            while (!item.empty() && (item.front() == ' ' || item.front() == '\t')) item.erase(item.begin());
            while (!item.empty() && (item.back() == ' ' || item.back() == '\t')) item.pop_back();
            if (!item.empty()) wanted.insert(item);
        }
    }

    TangentField field = tangent_field(img.data, W, H);
    std::string target_info = (target_threads > 0) ? (", target threads: ~" + std::to_string(target_threads)) : "";
    std::cout << "Input " << image_path << " -> " << W << "x" << H
              << ", background " << bg_str << target_info
              << ", output folder: " << out_dir << "\n";
    std::cout.flush();

    int completed = 0;
    for (const auto& algo : ALGORITHMS) {
        std::set<std::string> aliases = {
            algo.name,
            std::to_string(algo.id),
            "algo_" + algo.name,
            "algo" + std::to_string(algo.id) + "_" + algo.name
        };
        if (algo.name == "edge_tangent") aliases.insert("algo1_edge_tangent");
        else if (algo.name == "greedy_error") aliases.insert("algo2_greedy_error");
        else if (algo.name == "quadtree_hatch") aliases.insert("algo3_quadtree_hatch");
        else if (algo.name == "anchor_web") aliases.insert("algo4_anchor_web");
        else if (algo.name == "thread_painter") {
            aliases.insert("painter");
            aliases.insert("threadpainter");
        }

        if (!wanted.empty()) {
            bool found = false;
            for (const auto& alias : aliases) {
                if (wanted.count(alias)) {
                    found = true;
                    break;
                }
            }
            if (!found) continue;
        }

        std::mt19937_64 rng(static_cast<uint64_t>(seed + algo.id));
        ProgressLogger logger(algo.name);
        auto t0 = std::chrono::steady_clock::now();

        auto raw_threads = algo.func(img.data, W, H, bg, field, rng, budget, logger, target_threads);
        auto threads = finalize_threads(raw_threads, W, H);

        std::fprintf(stderr, "\r%40s\r", "");
        std::fflush(stderr);

        fs::path base_path = fs::path(out_dir) / (stem + "_" + algo.name);
        std::string base = base_path.string();

        render_png(base + ".png", threads, W, H, bg);
        write_svg(base + ".svg", threads, W, H, bg);
        if (save_npy) {
            write_npy(base + ".npy", threads);
        }

        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "[" << algo.id << "] "
                  << std::left << std::setw(17) << algo.name << " "
                  << std::right << std::setw(6) << threads.size() << " threads  "
                  << std::fixed << std::setprecision(1) << std::setw(5) << elapsed << " s  -> "
                  << base << ".png / .svg\n";
        std::cout.flush();
        ++completed;
    }

    if (completed == 0) {
        std::cerr << "Error: --only matched no algorithm: " << only_str << "\n";
        return 1;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {  // bad numeric option, allocation failure
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
