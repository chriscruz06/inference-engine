// tests/test_linear.cpp
//
// AVX2 (and the dispatched linear()) vs the scalar reference linear_scalar.
// Hermetic: random inputs, no weights and no Python. The two kernels must agree
// to floating-point tolerance -- not bit-for-bit, because AVX2's FMA reorders
// the summation, so the tolerance is relative to the output magnitude. The real
// end-to-end gate is still token identity vs the HuggingFace reference
// (tests/check_argmax.py); this catches kernel bugs (wrong layout, bad tail,
// NaN) without needing the model. Wired into CTest as linear_equivalence.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "gemm.hpp"

namespace {

std::uint32_t g_state = 0x12345678u;
float rnd() {  // deterministic xorshift in [-1, 1)
    g_state ^= g_state << 13;
    g_state ^= g_state >> 17;
    g_state ^= g_state << 5;
    return (static_cast<float>(g_state & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f;
}

int g_fail = 0;

void check_shape(int rows, int in_dim, int out_dim, bool with_bias) {
    std::vector<float> x(static_cast<std::size_t>(rows) * in_dim);
    std::vector<float> w(static_cast<std::size_t>(out_dim) * in_dim);
    std::vector<float> bias(static_cast<std::size_t>(out_dim));
    for (float& v : x) v = rnd();
    for (float& v : w) v = rnd();
    for (float& v : bias) v = rnd();
    const float* bptr = with_bias ? bias.data() : nullptr;

    std::vector<float> y_ref(static_cast<std::size_t>(rows) * out_dim);
    std::vector<float> y_fast(static_cast<std::size_t>(rows) * out_dim);
    ie::linear_scalar(x.data(), w.data(), bptr, y_ref.data(), rows, in_dim, out_dim);
    ie::linear(x.data(), w.data(), bptr, y_fast.data(), rows, in_dim, out_dim);

    float diff = 0.0f, mag = 0.0f;
    for (std::size_t i = 0; i < y_ref.size(); ++i) {
        diff = std::max(diff, std::fabs(y_ref[i] - y_fast[i]));
        mag = std::max(mag, std::fabs(y_ref[i]));
    }
    const float tol = 1e-5f * mag + 1e-4f;  // relative to output scale, with an absolute floor
    const bool ok = diff <= tol;
    std::printf("  rows=%-4d in=%-5d out=%-6d bias=%d  max|diff|=%.3e  tol=%.3e  %s\n", rows,
                in_dim, out_dim, with_bias ? 1 : 0, static_cast<double>(diff),
                static_cast<double>(tol), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

}  // namespace

int main() {
#if defined(__AVX2__) && defined(__FMA__)
    std::printf("[test_linear] linear() -> AVX2 path; comparing against linear_scalar\n");
#else
    std::printf("[test_linear] build has no AVX2; linear() == linear_scalar (trivially equal)\n");
#endif
    // GPT-2 124M shapes (prefill row count, then a single decode row), bias on.
    check_shape(256, 768, 2304, true);  // QKV projection
    check_shape(256, 768, 768, true);   // attention output projection
    check_shape(256, 768, 3072, true);  // MLP up
    check_shape(256, 3072, 768, true);  // MLP down
    check_shape(1, 768, 50257, false);  // tied LM head, decode step (null bias)
    check_shape(1, 3072, 768, true);    // a single-row MLP-down (decode)
    // Odd dims to exercise the 8-wide remainder and the scalar tail.
    check_shape(3, 100, 7, true);
    check_shape(5, 33, 17, false);
    check_shape(2, 1, 4, true);  // in_dim < 8: scalar tail only

    if (g_fail == 0) {
        std::printf("test_linear: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_linear: %d shape(s) failed\n", g_fail);
    return 1;
}