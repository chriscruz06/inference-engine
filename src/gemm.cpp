// src/gemm.cpp
#include "gemm.hpp"

#include <cstddef>

#include "profile.hpp"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace ie {

void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

void linear_scalar(const float* x, const float* w, const float* bias, float* y, int rows,
                   int in_dim, int out_dim) {
    for (int m = 0; m < rows; ++m) {
        const float* xr = x + static_cast<std::size_t>(m) * in_dim;
        float* yr = y + static_cast<std::size_t>(m) * out_dim;
        for (int o = 0; o < out_dim; ++o) {
            const float* wr = w + static_cast<std::size_t>(o) * in_dim;
            float acc = 0.0f;
            for (int i = 0; i < in_dim; ++i) acc += xr[i] * wr[i];
            yr[o] = bias ? acc + bias[o] : acc;
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)
namespace {
// Horizontal sum of the 8 lanes of a __m256.
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);            // 4 partial sums
    __m128 shuf = _mm_movehdup_ps(lo);  // duplicate odd lanes
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Matmuls with fewer than this many multiplies (rows*out_dim*in_dim) run
// serially: below it the OpenMP fork/join cost (and, for a bandwidth-bound GEMV,
// cross-core memory contention) outweighs the benefit. This is what keeps every
// decode step's tiny per-token GEMVs off the thread pool while prefill's big
// GEMMs still parallelize. Empirical -- tune for your cores / memory bandwidth.
[[maybe_unused]] constexpr long long kLinearParallelMinWork = 1LL << 26;  // ~67M
}  // namespace

void linear_avx2(const float* x, const float* w, const float* bias, float* y, int rows, int in_dim,
                 int out_dim) {
    // Parallelize over the output features (the rows of W, each producing one
    // output element via a dot product with x). Each thread owns a contiguous
    // block of features, so the work is independent and the result is
    // deterministic for ANY thread count -- every y[m][o] is computed by exactly
    // one thread with the same reduction order, so this stays bit-identical to
    // the serial version. Making o the outer loop also keeps each weight row
    // resident across the rows loop (helps the prefill GEMM). The if() clause
    // runs small matmuls serially (see kLinearParallelMinWork); the whole pragma
    // is a no-op when the build has no OpenMP, so the engine stays portable.
    [[maybe_unused]] const long long work = static_cast<long long>(rows) *
                                            static_cast<long long>(out_dim) *
                                            static_cast<long long>(in_dim);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (work >= kLinearParallelMinWork)
#endif
    for (int o = 0; o < out_dim; ++o) {
        const float* wr = w + static_cast<std::size_t>(o) * in_dim;
        const float bo = bias ? bias[o] : 0.0f;
        for (int m = 0; m < rows; ++m) {
            const float* xr = x + static_cast<std::size_t>(m) * in_dim;
            // Four independent accumulators: FMA has ~4-5 cycle latency but high
            // throughput, so a single accumulator would serialize on its own
            // dependency chain. Four lets the pipeline stay full.
            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();
            int i = 0;
            for (; i + 32 <= in_dim; i += 32) {  // 32 floats/iter; GPT-2 dims (768, 3072) are exact
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + i), _mm256_loadu_ps(wr + i), a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + i + 8), _mm256_loadu_ps(wr + i + 8), a1);
                a2 =
                    _mm256_fmadd_ps(_mm256_loadu_ps(xr + i + 16), _mm256_loadu_ps(wr + i + 16), a2);
                a3 =
                    _mm256_fmadd_ps(_mm256_loadu_ps(xr + i + 24), _mm256_loadu_ps(wr + i + 24), a3);
            }
            for (; i + 8 <= in_dim; i += 8)  // 8-wide remainder
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + i), _mm256_loadu_ps(wr + i), a0);
            __m256 acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            float s = hsum256(acc);
            for (; i < in_dim; ++i) s += xr[i] * wr[i];  // scalar tail (< 8 elems)
            y[static_cast<std::size_t>(m) * out_dim + o] = s + bo;
        }
    }
}
#endif

void linear(const float* x, const float* w, const float* bias, float* y, int rows, int in_dim,
            int out_dim) {
    const prof::Scope prof_scope(in_dim, out_dim);  // no-op unless IE_PROFILE is set
#if defined(__AVX2__) && defined(__FMA__)
    linear_avx2(x, w, bias, y, rows, in_dim, out_dim);
#else
    linear_scalar(x, w, bias, y, rows, in_dim, out_dim);
#endif
}

}  // namespace ie