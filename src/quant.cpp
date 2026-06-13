#include "quant.hpp"

#include <algorithm>
#include <cmath>

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

}  // namespace ie
