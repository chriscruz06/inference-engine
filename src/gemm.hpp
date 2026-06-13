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

// TODO (weeks 5-6), each verified token-identical against matmul_naive:
//   void matmul_avx2(...);      // 8-wide FMA inner loop
//   void matmul_threaded(...);  // split output rows across cores
//   void matmul_tiled(...);     // L1/L2-blocked GEMM for prefill

}  // namespace ie
