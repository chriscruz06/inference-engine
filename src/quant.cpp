#include "quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ie {

float quantize_group_q8(const float* w, int8_t* q, int n) {
    float absmax = 0.0f;
    for (int i = 0; i < n; ++i) {
        absmax = std::max(absmax, std::fabs(w[i]));
    }
    const float scale = absmax > 0.0f ? absmax / 127.0f : 1.0f;
    const float inv_scale = 1.0f / scale;
    for (int i = 0; i < n; ++i) {
        int v = static_cast<int>(std::lround(w[i] * inv_scale));
        v = std::max(-127, std::min(127, v));
        q[i] = static_cast<int8_t>(v);
    }
    return scale;
}

namespace {

// Symmetric absmax scale for a group of `n` weights mapped to [-qmax, qmax].
float group_scale(const float* w, int n, int qmax) {
    float absmax = 0.0f;
    for (int i = 0; i < n; ++i) absmax = std::max(absmax, std::fabs(w[i]));
    return absmax > 0.0f ? absmax / static_cast<float>(qmax) : 1.0f;
}

// Round to nearest, clamp to [-qmax, qmax].
inline int quant_one(float w, float inv_scale, int qmax) {
    int v = static_cast<int>(std::lround(w * inv_scale));
    return std::max(-qmax, std::min(qmax, v));
}

}  // namespace

void quantize_tensor(const float* w, int out_dim, int in_dim, QuantType type, int group,
                     QuantTensor& out) {
    out.type = type;
    out.out_dim = out_dim;
    out.in_dim = in_dim;
    out.group = group;

    const int gpr = (in_dim + group - 1) / group;  // groups per row
    out.scales.assign(static_cast<std::size_t>(out_dim) * gpr, 1.0f);

    if (type == QuantType::Q8) {
        out.q.assign(static_cast<std::size_t>(out_dim) * in_dim, 0);
        int8_t* q = reinterpret_cast<int8_t*>(out.q.data());
        for (int o = 0; o < out_dim; ++o) {
            const float* wr = w + static_cast<std::size_t>(o) * in_dim;
            int8_t* qr = q + static_cast<std::size_t>(o) * in_dim;
            for (int g = 0; g < gpr; ++g) {
                const int gs = g * group;
                const int n = std::min(group, in_dim - gs);
                const float scale = group_scale(wr + gs, n, 127);
                const float inv = 1.0f / scale;
                out.scales[static_cast<std::size_t>(o) * gpr + g] = scale;
                for (int i = 0; i < n; ++i)
                    qr[gs + i] = static_cast<int8_t>(quant_one(wr[gs + i], inv, 127));
            }
        }
    } else {  // Q4: two nibbles per byte, value v in [-8,7] stored as (v + 8).
        const std::size_t row_bytes = static_cast<std::size_t>((in_dim + 1) / 2);
        out.q.assign(static_cast<std::size_t>(out_dim) * row_bytes, 0);
        for (int o = 0; o < out_dim; ++o) {
            const float* wr = w + static_cast<std::size_t>(o) * in_dim;
            std::uint8_t* qr = out.q.data() + static_cast<std::size_t>(o) * row_bytes;
            for (int g = 0; g < gpr; ++g) {
                const int gs = g * group;
                const int n = std::min(group, in_dim - gs);
                const float scale = group_scale(wr + gs, n, 7);
                const float inv = 1.0f / scale;
                out.scales[static_cast<std::size_t>(o) * gpr + g] = scale;
                for (int i = 0; i < n; ++i) {
                    const int col = gs + i;
                    const std::uint8_t nib =
                        static_cast<std::uint8_t>(quant_one(wr[col], inv, 7) + 8);  // [0,15]
                    std::uint8_t& byte = qr[col / 2];
                    if ((col & 1) == 0)
                        byte = static_cast<std::uint8_t>((byte & 0xF0) | nib);  // low nibble
                    else
                        byte = static_cast<std::uint8_t>((byte & 0x0F) | (nib << 4));  // high nibble
                }
            }
        }
    }
}

}  // namespace ie
