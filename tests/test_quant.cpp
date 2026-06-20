// tests/test_quant.cpp
//
// Quantized linear kernels (Q8 / Q4) vs their references. Hermetic: random
// inputs, no weights and no Python. Three checks per shape:
//   1. kernel equivalence -- the dispatched linear() (AVX2 q8/q4 path) matches
//      the scalar q8/q4 kernel on the SAME quantized weights, to FMA tolerance.
//      Catches SIMD bugs (bad nibble unpack, wrong tail, NaN).
//   2. dequant consistency -- the scalar quantized kernel matches a plain
//      linear_scalar() over the explicitly dequantized weight matrix. Catches
//      scale-indexing / int4 packing bugs independent of the kernel's own
//      dequant (the classic "int4 is garbage" failure CLAUDE.md warns about).
//   3. accuracy -- L2 relative error of the quantized result vs the true fp32
//      result is bounded (q8 tight, q4 looser). Catches gross quantization
//      damage while tolerating normal rounding noise.
// The real end-to-end gate is still token identity vs HuggingFace; this isolates
// the kernel. Wired into CTest as quant_equivalence.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "gemm.hpp"
#include "quant.hpp"

namespace {

std::uint32_t g_state = 0x243F6A88u;
float rnd() {  // deterministic xorshift in [-1, 1)
    g_state ^= g_state << 13;
    g_state ^= g_state >> 17;
    g_state ^= g_state << 5;
    return (static_cast<float>(g_state & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f;
}

int g_fail = 0;

// Explicitly dequantize a QuantTensor back to a [out, in] fp32 matrix, mirroring
// the storage layout in quant.cpp. Independent of the gemm kernels, so it pins
// down the scale/packing convention on its own.
std::vector<float> dequant(const ie::QuantTensor& w) {
    std::vector<float> out(static_cast<std::size_t>(w.out_dim) * w.in_dim);
    const int gpr = w.groups_per_row();
    if (w.type == ie::QuantType::Q8) {
        const std::int8_t* q = reinterpret_cast<const std::int8_t*>(w.q.data());
        for (int o = 0; o < w.out_dim; ++o)
            for (int i = 0; i < w.in_dim; ++i) {
                const float s = w.scales[static_cast<std::size_t>(o) * gpr + i / w.group];
                out[static_cast<std::size_t>(o) * w.in_dim + i] =
                    static_cast<float>(q[static_cast<std::size_t>(o) * w.in_dim + i]) * s;
            }
    } else {
        const std::size_t row_bytes = static_cast<std::size_t>((w.in_dim + 1) / 2);
        for (int o = 0; o < w.out_dim; ++o)
            for (int i = 0; i < w.in_dim; ++i) {
                const std::uint8_t byte = w.q[static_cast<std::size_t>(o) * row_bytes + i / 2];
                const int code = ((i & 1) == 0) ? (byte & 0x0F) : (byte >> 4);
                const float s = w.scales[static_cast<std::size_t>(o) * gpr + i / w.group];
                out[static_cast<std::size_t>(o) * w.in_dim + i] = static_cast<float>(code - 8) * s;
            }
    }
    return out;
}

const char* type_name(ie::QuantType t) {
    return t == ie::QuantType::Q8 ? "q8" : "q4";
}

void check_shape(ie::QuantType type, int rows, int in_dim, int out_dim, bool with_bias) {
    std::vector<float> x(static_cast<std::size_t>(rows) * in_dim);
    std::vector<float> w(static_cast<std::size_t>(out_dim) * in_dim);
    std::vector<float> bias(static_cast<std::size_t>(out_dim));
    for (float& v : x) v = rnd();
    for (float& v : w) v = rnd();
    for (float& v : bias) v = rnd();
    const float* bptr = with_bias ? bias.data() : nullptr;

    ie::QuantTensor qw;
    ie::quantize_tensor(w.data(), out_dim, in_dim, type, ie::kQuantGroup, qw);

    const std::size_t yn = static_cast<std::size_t>(rows) * out_dim;
    std::vector<float> y_scalar(yn), y_fast(yn), y_deq(yn), y_true(yn);

    // (1) scalar quantized kernel and (2) dispatched fast (AVX2) kernel.
    if (type == ie::QuantType::Q8)
        ie::linear_q8_scalar(x.data(), qw, bptr, y_scalar.data(), rows);
    else
        ie::linear_q4_scalar(x.data(), qw, bptr, y_scalar.data(), rows);
    ie::linear(x.data(), ie::LinearWeight{nullptr, &qw, in_dim, out_dim}, bptr, y_fast.data(),
               rows);

    // (2 ref) plain fp32 dot over explicitly dequantized weights.
    const std::vector<float> wdq = dequant(qw);
    ie::linear_scalar(x.data(), wdq.data(), bptr, y_deq.data(), rows, in_dim, out_dim);

    // (3 ref) true fp32 result over the original (unquantized) weights.
    ie::linear_scalar(x.data(), w.data(), bptr, y_true.data(), rows, in_dim, out_dim);

    float d_fast = 0.0f, d_deq = 0.0f, mag = 0.0f;
    double err2 = 0.0, ref2 = 0.0;
    for (std::size_t i = 0; i < yn; ++i) {
        d_fast = std::max(d_fast, std::fabs(y_fast[i] - y_scalar[i]));
        d_deq = std::max(d_deq, std::fabs(y_scalar[i] - y_deq[i]));
        mag = std::max(mag, std::fabs(y_scalar[i]));
        const double e = static_cast<double>(y_scalar[i]) - static_cast<double>(y_true[i]);
        err2 += e * e;
        ref2 += static_cast<double>(y_true[i]) * static_cast<double>(y_true[i]);
    }
    const float tol = 1e-4f * mag + 1e-3f;                         // FMA-reorder tolerance
    const double rel = ref2 > 0.0 ? std::sqrt(err2 / ref2) : 0.0;  // L2 relative quant error
    const double rel_max = (type == ie::QuantType::Q8) ? 0.03 : 0.20;

    const bool ok = d_fast <= tol && d_deq <= tol && rel <= rel_max;
    std::printf(
        "  %s rows=%-3d in=%-5d out=%-5d bias=%d  fast|scalar=%.2e deq=%.2e (tol=%.2e)  "
        "relerr=%.3f%% (<%.0f%%)  %s\n",
        type_name(type), rows, in_dim, out_dim, with_bias ? 1 : 0, static_cast<double>(d_fast),
        static_cast<double>(d_deq), static_cast<double>(tol), rel * 100.0, rel_max * 100.0,
        ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

}  // namespace

int main() {
#if defined(__AVX2__) && defined(__FMA__)
    std::printf("[test_quant] linear() -> AVX2 q8/q4 path; comparing against scalar references\n");
#else
    std::printf("[test_quant] no AVX2; linear() uses the scalar q8/q4 kernels\n");
#endif
    for (ie::QuantType t : {ie::QuantType::Q8, ie::QuantType::Q4}) {
        // GPT-2 124M weight shapes (in_dim a multiple of the group), prefill + decode rows.
        check_shape(t, 8, 768, 2304, true);   // QKV projection
        check_shape(t, 8, 768, 768, true);    // attention output projection
        check_shape(t, 8, 3072, 768, true);   // MLP down
        check_shape(t, 1, 768, 1024, false);  // single decode row, null bias
        // Odd dims: partial final group + SIMD remainder/tail.
        check_shape(t, 3, 100, 7, true);   // groups 64 + 36 (36 = 32 + 4 tail)
        check_shape(t, 5, 33, 17, false);  // one group of 33 (32 + 1 tail)
    }

    if (g_fail == 0) {
        std::printf("test_quant: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_quant: %d shape(s) failed\n", g_fail);
    return 1;
}
