// Minimal, dependency-free correctness test (no gtest, so CI stays simple).
// Returns 0 on success, 1 on failure; wired into CTest in CMakeLists.txt.

#include <cmath>
#include <cstdio>

#include "gemm.hpp"

static bool approx_equal(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

int main() {
    // A (2x3) * B (3x2) = C (2x2)
    //   A = [[1, 2, 3],      B = [[ 7,  8],      C = [[ 58,  64],
    //        [4, 5, 6]]           [ 9, 10],           [139, 154]]
    //                             [11, 12]]
    const float A[6] = {1, 2, 3, 4, 5, 6};
    const float B[6] = {7, 8, 9, 10, 11, 12};
    const float expected[4] = {58, 64, 139, 154};
    float C[4] = {0, 0, 0, 0};

    ie::matmul_naive(A, B, C, /*M=*/2, /*N=*/2, /*K=*/3);

    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        if (!approx_equal(C[i], expected[i])) {
            std::fprintf(stderr, "mismatch at index %d: got %.3f, expected %.3f\n", i, C[i],
                         expected[i]);
            ++failures;
        }
    }

    if (failures == 0) {
        std::printf("matmul_naive: OK\n");
        return 0;
    }
    std::fprintf(stderr, "matmul_naive: %d failure(s)\n", failures);
    return 1;
}
