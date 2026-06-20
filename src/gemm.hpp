// src/gemm.hpp
#pragma once

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

// TODO (next in Phase 4), each verified token-identical against linear_scalar:
//   linear with OpenMP over output rows  (prefill should scale; decode less so)
//   cache-tiled GEMM for the prefill path

}  // namespace ie