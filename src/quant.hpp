#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ie {

// Weight-only quantization (Phase 5). int8 first (simpler), then packed int4.
// Group-wise symmetric scales (a shared scale per group of contiguous weights,
// like llama.cpp's Q8_0 / Q4_0) give better accuracy than one scale per tensor
// and match the format philosophy we benchmark against. Activations stay fp32;
// the kernel dequantizes weights on the fly (src/gemm.cpp), so the only thing
// that shrinks is the weight bytes streamed per token -- which is exactly the
// decode bottleneck (memory bandwidth), making this a speed win, not just a
// footprint win.

// Minimal symmetric int8 quantizer for a single group of `n` weights: finds the
// absmax, derives a scale, and writes rounded int8 values to `q`. Returns the
// scale. The building block the grouped scheme below is built from.
float quantize_group_q8(const float* w, int8_t* q, int n);

// Group size: a shared fp32 scale covers this many contiguous weights along the
// `in` dimension. 64 divides every in_dim that flows through linear() in the
// model (768 and 3072), so groups are byte-aligned for the int4 packing too.
constexpr int kQuantGroup = 64;

enum class QuantType { Q8, Q4 };

// A quantized [out_dim, in_dim] weight matrix (row-major, the layout linear()
// wants). Each output row's `in_dim` values are split into ceil(in_dim/group)
// contiguous groups; each group has one fp32 scale.
//
//   Q8: `q` holds out_dim*in_dim signed int8 values, stored as their uint8 bit
//       pattern (reinterpret_cast<const int8_t*> in the kernel).
//   Q4: `q` packs two signed-4-bit values per byte, out_dim*ceil(in_dim/2) bytes.
//       Value v in [-8,7] is stored as the nibble (v + 8) in [0,15]; the even
//       column is the low nibble, the odd column the high nibble. Dequant is
//       (nibble - 8) * scale.
struct QuantTensor {
    QuantType type = QuantType::Q8;
    int out_dim = 0;
    int in_dim = 0;
    int group = kQuantGroup;
    std::vector<std::uint8_t> q;     // packed weights (see per-type note above)
    std::vector<float> scales;       // out_dim * groups_per_row, one per group

    int groups_per_row() const { return (in_dim + group - 1) / group; }
    std::size_t bytes() const { return q.size() + scales.size() * sizeof(float); }
};

// Quantize a [out_dim, in_dim] row-major fp32 weight matrix into `out`. Symmetric
// absmax per group: Q8 maps the group's absmax to 127, Q4 to 7. A zero group
// (all-zero weights) gets scale 1 and all-zero codes. `group` defaults to the
// model-wide kQuantGroup; pass another value to experiment with int4 accuracy.
void quantize_tensor(const float* w, int out_dim, int in_dim, QuantType type, int group,
                     QuantTensor& out);

}  // namespace ie
