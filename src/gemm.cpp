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

// Load 8 signed int8 weights and widen them to 8 fp32 lanes. The dequant scale
// is applied once per group by the caller, not here -- so this is just the
// int8 -> fp32 conversion the quantized FMA loop needs.
inline __m256 cvt8_i8_ps(const std::int8_t* p) {
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));  // 8 bytes
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));                      // sign-extend, to ps
}
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

// int8 quantized linear. Same structure as linear_avx2 (parallelize over output
// features o so the result is deterministic for any thread count, gated by work
// size so decode's per-token GEMV stays serial), but each output row's weights
// are int8 in `w.q` with one fp32 scale per group along `in`. Per group: FMA the
// dequantized-to-fp32 int8 weights against fp32 x into 4 accumulators, horizontal
// sum, multiply by the group scale, add to the row total. Streaming 1 byte per
// weight instead of 4 is the decode win.
void linear_q8_avx2(const float* x, const QuantTensor& w, const float* bias, float* y, int rows) {
    const int in_dim = w.in_dim, out_dim = w.out_dim, group = w.group;
    const int gpr = w.groups_per_row();
    const std::int8_t* q = reinterpret_cast<const std::int8_t*>(w.q.data());
    [[maybe_unused]] const long long work =
        static_cast<long long>(rows) * out_dim * in_dim;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (work >= kLinearParallelMinWork)
#endif
    for (int o = 0; o < out_dim; ++o) {
        const std::int8_t* qrow = q + static_cast<std::size_t>(o) * in_dim;
        const float* sc = w.scales.data() + static_cast<std::size_t>(o) * gpr;
        const float bo = bias ? bias[o] : 0.0f;
        for (int m = 0; m < rows; ++m) {
            const float* xr = x + static_cast<std::size_t>(m) * in_dim;
            float total = 0.0f;
            for (int g = 0; g < gpr; ++g) {
                const int gs = g * group;
                const int n = (group < in_dim - gs) ? group : (in_dim - gs);
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                int i = 0;
                for (; i + 32 <= n; i += 32) {
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + gs + i), cvt8_i8_ps(qrow + gs + i), a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + gs + i + 8),
                                         cvt8_i8_ps(qrow + gs + i + 8), a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + gs + i + 16),
                                         cvt8_i8_ps(qrow + gs + i + 16), a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + gs + i + 24),
                                         cvt8_i8_ps(qrow + gs + i + 24), a3);
                }
                for (; i + 8 <= n; i += 8)
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + gs + i), cvt8_i8_ps(qrow + gs + i), a0);
                __m256 acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
                float part = hsum256(acc);
                for (; i < n; ++i)
                    part += xr[gs + i] * static_cast<float>(qrow[gs + i]);  // scalar tail
                total += sc[g] * part;
            }
            y[static_cast<std::size_t>(m) * out_dim + o] = total + bo;
        }
    }
}

// int4 quantized linear. Two weights per byte (low nibble = even column). The
// hot path unpacks 16 nibbles (8 packed bytes) per iteration with a branch-free
// AND/shift/interleave, centers them to [-8,7], widens to fp32, and FMAs two
// 8-wide chunks against x. Group scaling and the OpenMP/work gate match the int8
// kernel. Streams 0.5 byte per weight: the int4 decode win.
void linear_q4_avx2(const float* x, const QuantTensor& w, const float* bias, float* y, int rows) {
    const int in_dim = w.in_dim, out_dim = w.out_dim, group = w.group;
    const int gpr = w.groups_per_row();
    const std::size_t row_bytes = static_cast<std::size_t>((in_dim + 1) / 2);
    const __m128i lomask = _mm_set1_epi8(0x0F);
    const __m128i eight = _mm_set1_epi8(8);
    [[maybe_unused]] const long long work =
        static_cast<long long>(rows) * out_dim * in_dim;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (work >= kLinearParallelMinWork)
#endif
    for (int o = 0; o < out_dim; ++o) {
        const std::uint8_t* qrow = w.q.data() + static_cast<std::size_t>(o) * row_bytes;
        const float* sc = w.scales.data() + static_cast<std::size_t>(o) * gpr;
        const float bo = bias ? bias[o] : 0.0f;
        for (int m = 0; m < rows; ++m) {
            const float* xr = x + static_cast<std::size_t>(m) * in_dim;
            float total = 0.0f;
            for (int g = 0; g < gpr; ++g) {
                const int gs = g * group;  // even (group is even), so byte-aligned
                const int n = (group < in_dim - gs) ? group : (in_dim - gs);
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                int i = 0;
                for (; i + 16 <= n; i += 16) {
                    const __m128i packed =
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qrow + (gs + i) / 2));
                    const __m128i lo = _mm_and_si128(packed, lomask);             // even columns
                    const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lomask);  // odd
                    __m128i nib = _mm_unpacklo_epi8(lo, hi);  // 16 nibbles in column order
                    nib = _mm_sub_epi8(nib, eight);           // center to [-8,7]
                    const __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(nib));
                    const __m256 wf1 =
                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(nib, 8)));
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + gs + i), wf0, a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + gs + i + 8), wf1, a1);
                }
                float part = hsum256(_mm256_add_ps(a0, a1));
                for (; i < n; ++i) {  // scalar tail (< 16 columns left in group)
                    const int col = gs + i;
                    const std::uint8_t byte = qrow[col / 2];
                    const int code = ((col & 1) == 0) ? (byte & 0x0F) : (byte >> 4);
                    part += xr[col] * static_cast<float>(code - 8);
                }
                total += sc[g] * part;
            }
            y[static_cast<std::size_t>(m) * out_dim + o] = total + bo;
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

// Always-available scalar references for the quantized kernels: plain per-group
// dequant-and-dot. Every fast quantized kernel is checked against these
// (tests/test_quant.cpp), the same way linear_scalar anchors linear_avx2. The
// per-group scale is applied to the group's partial sum, matching the AVX2 path
// so the two agree to FMA reorder tolerance.
void linear_q8_scalar(const float* x, const QuantTensor& w, const float* bias, float* y, int rows) {
    const int in_dim = w.in_dim, out_dim = w.out_dim, group = w.group;
    const int gpr = w.groups_per_row();
    const std::int8_t* q = reinterpret_cast<const std::int8_t*>(w.q.data());
    for (int m = 0; m < rows; ++m) {
        const float* xr = x + static_cast<std::size_t>(m) * in_dim;
        float* yr = y + static_cast<std::size_t>(m) * out_dim;
        for (int o = 0; o < out_dim; ++o) {
            const std::int8_t* qr = q + static_cast<std::size_t>(o) * in_dim;
            const float* sc = w.scales.data() + static_cast<std::size_t>(o) * gpr;
            float acc = 0.0f;
            for (int g = 0; g < gpr; ++g) {
                const int gs = g * group;
                const int n = (group < in_dim - gs) ? group : (in_dim - gs);
                float part = 0.0f;
                for (int i = 0; i < n; ++i) part += xr[gs + i] * static_cast<float>(qr[gs + i]);
                acc += sc[g] * part;
            }
            yr[o] = bias ? acc + bias[o] : acc;
        }
    }
}

void linear_q4_scalar(const float* x, const QuantTensor& w, const float* bias, float* y, int rows) {
    const int in_dim = w.in_dim, out_dim = w.out_dim, group = w.group;
    const int gpr = w.groups_per_row();
    const std::size_t row_bytes = static_cast<std::size_t>((in_dim + 1) / 2);
    for (int m = 0; m < rows; ++m) {
        const float* xr = x + static_cast<std::size_t>(m) * in_dim;
        float* yr = y + static_cast<std::size_t>(m) * out_dim;
        for (int o = 0; o < out_dim; ++o) {
            const std::uint8_t* qr = w.q.data() + static_cast<std::size_t>(o) * row_bytes;
            const float* sc = w.scales.data() + static_cast<std::size_t>(o) * gpr;
            float acc = 0.0f;
            for (int g = 0; g < gpr; ++g) {
                const int gs = g * group;
                const int n = (group < in_dim - gs) ? group : (in_dim - gs);
                float part = 0.0f;
                for (int i = 0; i < n; ++i) {
                    const int col = gs + i;
                    const std::uint8_t byte = qr[col / 2];
                    const int code = ((col & 1) == 0) ? (byte & 0x0F) : (byte >> 4);
                    part += xr[col] * static_cast<float>(code - 8);
                }
                acc += sc[g] * part;
            }
            yr[o] = bias ? acc + bias[o] : acc;
        }
    }
}

void linear(const float* x, const LinearWeight& w, const float* bias, float* y, int rows) {
    const prof::Scope prof_scope(w.in_dim, w.out_dim);  // no-op unless IE_PROFILE is set
    if (w.q) {
#if defined(__AVX2__) && defined(__FMA__)
        if (w.q->type == QuantType::Q8)
            linear_q8_avx2(x, *w.q, bias, y, rows);
        else
            linear_q4_avx2(x, *w.q, bias, y, rows);
#else
        if (w.q->type == QuantType::Q8)
            linear_q8_scalar(x, *w.q, bias, y, rows);
        else
            linear_q4_scalar(x, *w.q, bias, y, rows);
#endif
        return;
    }
#if defined(__AVX2__) && defined(__FMA__)
    linear_avx2(x, w.f32, bias, y, rows, w.in_dim, w.out_dim);
#else
    linear_scalar(x, w.f32, bias, y, rows, w.in_dim, w.out_dim);
#endif
}

}  // namespace ie