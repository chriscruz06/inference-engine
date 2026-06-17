// src/gemm.hpp
#pragma once

namespace ie {

// Matrix-multiply kernels. The entire performance story of this project lives
// in this file: naive -> AVX2 SIMD -> multithreaded -> cache-tiled.
//
// Convention: row-major storage. C[M*N] = A[M*K] * B[K*N].
//
// Two regimes share these kernels (see README "How it works"):
//   * Prefill  -> matrix x matrix (GEMM), compute-bound -> tiling pays off.
//   * Decode   -> matrix x vector (GEMV), memory-bandwidth-bound.

// Correct, simple, slow. The reference that every faster kernel is checked
// against bit-for-bit. Implemented today so CI has something real to test.
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K);

// Linear layer: y = x * W^T + bias, with W stored [out, in] row-major -- the
// layout of every Linear weight in the model (QKV, c_proj, both MLP matrices,
// and the tied LM head). x is [rows, in_dim], y is [rows, out_dim]. `bias` may
// be null (Llama's bias-free linears); GPT-2 always supplies one.
//
// Note this is NOT the C=A*B convention above: the weight is transposed, so the
// inner loop walks x and one weight row, both contiguously -- which is exactly
// the access pattern the AVX2 kernel will vectorize. The optimization phase
// swaps this body and every call site (all the projections) is unchanged.
void linear(const float* x, const float* w, const float* bias, float* y,
            int rows, int in_dim, int out_dim);

// TODO (weeks 5-6), each verified token-identical against matmul_naive:
//   void matmul_avx2(...);      // 8-wide FMA inner loop
//   void matmul_threaded(...);  // split output rows across cores
//   void matmul_tiled(...);     // L1/L2-blocked GEMM for prefill

}  // namespace ie