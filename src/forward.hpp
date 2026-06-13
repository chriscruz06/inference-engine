#pragma once

#include <vector>

#include "model.hpp"

namespace ie {

// Numerical building blocks of the transformer forward pass. Each is filled in
// during weeks 2-3 and diffed against the Python reference ONE AT A TIME before
// moving on (embedding -> LN -> attention -> MLP -> logits).

// LayerNorm over the last dimension: normalize, then scale + shift.
void layernorm(const float* x, const float* weight, const float* bias, float* out, int n,
               float eps);

// GELU, tanh approximation. GPT-2 was trained with this, NOT the exact erf
// version; matching it keeps the layer-diff harness quiet.
void gelu_tanh(float* x, int n);

// Numerically stable softmax over n elements (subtract max, exp, normalize).
void softmax(float* x, int n);

// Run the full forward pass for a sequence of token ids, returning logits over
// the vocabulary for the final position. (Skeleton.)
std::vector<float> forward(const Model& model, const std::vector<int>& tokens);

}  // namespace ie
