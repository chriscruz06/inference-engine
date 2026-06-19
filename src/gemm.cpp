// src/gemm.cpp
#include "gemm.hpp"

#include <cstddef>

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

void linear(const float* x, const float* w, const float* bias, float* y, int rows, int in_dim,
            int out_dim) {
    for (int m = 0; m < rows; ++m) {
        const float* xr = x + static_cast<std::size_t>(m) * in_dim;
        float* yr = y + static_cast<std::size_t>(m) * out_dim;
        for (int o = 0; o < out_dim; ++o) {
            const float* wr = w + static_cast<std::size_t>(o) * in_dim;  // row o of [out, in]
            float acc = 0.0f;
            for (int i = 0; i < in_dim; ++i) acc += xr[i] * wr[i];
            yr[o] = bias ? acc + bias[o] : acc;
        }
    }
}

}  // namespace ie