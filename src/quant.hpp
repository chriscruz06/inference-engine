#pragma once

#include <cstdint>

namespace ie {

// Weight-only quantization. int8 first (simpler), then packed int4. Group-wise
// scales (a shared scale per group of ~32-64 weights, like llama.cpp's Q8_0 /
// Q4_0) give better accuracy and match the format philosophy we benchmark
// against.

// Minimal symmetric int8 quantizer for a single group of `n` weights: finds the
// absmax, derives a scale, and writes rounded int8 values to `q`. Returns the
// scale. This is a correct starting point, NOT the final grouped scheme.
float quantize_group_q8(const float* w, int8_t* q, int n);

// TODO (weeks 7-8):
//   struct Q8Block { int8_t q[GROUP]; float scale; };
//   struct Q4Block { uint8_t q[GROUP / 2]; float scale; };  // 2 vals / byte
//   void quantize_q4(...);  // packed int4, group-wise scales (Q4_0-style)

}  // namespace ie
