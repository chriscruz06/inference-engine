// src/gemm.hpp
#pragma once

#include "quant.hpp"

namespace ie {

// Matrix-multiply kernels. The entire performance story of this project lives
// in this file: naive -> AVX2 SIMD -> multithreaded -> cache-tiled.
//
// Convention: row-major storage. C[M*N] = A[M*K] * B[K*N].

// Correct, simple, slow. The reference matmul_naive is checked bit-for-bit by
// test_gemm. Kept for that test and as documentation of the plain algorithm.
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K);

// Linear layer: y = x * W^T + bias, with W stored [out, in] row-major -- the
// layout of every Linear weight in the model. x is [rows, in_dim], y is
// [rows, out_dim]. `bias` may be null (Llama's bias-free linears; the GPT-2 LM
// head). linear() dispatches to the fastest kernel the build targets; the named
// kernels below are exposed so a test can check them against each other.
void linear(const float* x, const float* w, const float* bias, float* y, int rows, int in_dim,
            int out_dim);

// Correctness reference: plain scalar dot products. Always available. Every
// faster kernel is checked against this -- token-identical end to end, ~1e-4 at
// the kernel level (AVX2's FMA reorders the summation, so it is not bit-equal).
void linear_scalar(const float* x, const float* w, const float* bias, float* y, int rows,
                   int in_dim, int out_dim);

#if defined(__AVX2__) && defined(__FMA__)
// 8-wide FMA dot products with 4 independent accumulators to hide FMA latency.
// Only declared when the build targets AVX2+FMA (e.g. -march=native on x86); on
// other builds linear() is linear_scalar, so the engine stays portable.
void linear_avx2(const float* x, const float* w, const float* bias, float* y, int rows, int in_dim,
                 int out_dim);
#endif

// ---- Phase 5: weight-only quantized linear (y = x * dequant(W)^T + bias) ----
// Same math as linear(), but W is a group-wise quantized QuantTensor [out, in].
// Activations x stay fp32; the kernel dequantizes each weight group on the fly
// (int8/int4 -> fp32) so it streams 1/4 or 1/8 the weight bytes per token -- the
// decode (GEMV, memory-bandwidth-bound) win. Dims come from the tensor. `bias`
// may be null. The *_scalar kernels are the always-available correctness
// reference for tests/test_quant.cpp; the *_avx2 kernels are the fast path.
void linear_q8_scalar(const float* x, const QuantTensor& w, const float* bias, float* y, int rows);
void linear_q4_scalar(const float* x, const QuantTensor& w, const float* bias, float* y, int rows);

#if defined(__AVX2__) && defined(__FMA__)
void linear_q8_avx2(const float* x, const QuantTensor& w, const float* bias, float* y, int rows);
void linear_q4_avx2(const float* x, const QuantTensor& w, const float* bias, float* y, int rows);
#endif

// A linear weight, either fp32 (a raw row-major [out, in] buffer) or a quantized
// QuantTensor. Lets the forward pass select the kernel per weight without knowing
// which it is: exactly one of `f32` / `q` is non-null. in_dim/out_dim are always
// set (they mirror the QuantTensor's dims in the quantized case).
struct LinearWeight {
    const float* f32 = nullptr;
    const QuantTensor* q = nullptr;
    int in_dim = 0;
    int out_dim = 0;
};

// Dispatch over a LinearWeight: fp32 -> linear_avx2/scalar, quantized -> the
// matching q8/q4 kernel. y is [rows, out_dim]. One profiler scope per call.
void linear(const float* x, const LinearWeight& w, const float* bias, float* y, int rows);

// Phase 4 GEMM optimization ends here. linear_avx2 above is the 8-wide FMA
// kernel parallelized over output features (OpenMP, gated by work size so
// decode's per-token GEMVs stay serial). Cache-blocking the prefill GEMM over
// the token dimension was implemented and benchmarked, but it regressed prefill
// ~6% on this hardware and did nothing for decode: mlp_proj's activation already
// fits in shared L3, so its re-reads were cheap L3 hits rather than RAM traffic,
// and blocking only added per-call fork/join overhead. The kernel is left at
// AVX2 + gated OpenMP, its measured local optimum here; the remaining prefill
// scaling gap is shared-L3 bandwidth and all-core throttling, which blocking
// cannot touch. Decode's wall is memory bandwidth, addressed in Phase 5.
//
// Phase 5 attacks that wall with the quantized linear kernels above (int8/int4
// weights, fp32 activations): fewer weight bytes streamed per decode token.

}  // namespace ie