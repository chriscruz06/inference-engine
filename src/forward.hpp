#pragma once

#include <vector>

#include "kv_cache.hpp"
#include "model.hpp"

namespace ie {

// Numerical building blocks of the transformer forward pass. Each was filled in
// during weeks 2-3 and diffed against the Python reference ONE AT A TIME
// (embedding -> LN -> attention -> MLP -> logits).

// LayerNorm over the last dimension: normalize, then scale + shift.
void layernorm(const float* x, const float* weight, const float* bias, float* out, int n,
               float eps);

// GELU, tanh approximation. GPT-2 was trained with this, NOT the exact erf
// version; matching it keeps the layer-diff harness quiet.
void gelu_tanh(float* x, int n);

// Numerically stable softmax over n elements (subtract max, exp, normalize).
void softmax(float* x, int n);

// Run the full forward pass for a sequence of token ids and return logits over
// the vocabulary for the final position. Recomputes the whole sequence every
// call (no cache) and dumps every activation when IE_DUMP_DIR is set, so this is
// the verification oracle backing tests/check_layers.py. forward_cached() is the
// fast path used for generation.
std::vector<float> forward(const Model& model, const std::vector<int>& tokens);

// Cache-aware forward for incremental decoding. Runs `tokens` as a contiguous
// chunk beginning at absolute position `n_past`, reading and extending `cache`
// (which must be init()'d for this model and have len == n_past on entry).
//
//   prefill:  forward_cached(model, cache, prompt, 0);          // many tokens
//   decode:   forward_cached(model, cache, {next}, cache.len);  // one at a time
//
// On return cache.len == n_past + tokens.size(). Returns logits for the LAST
// token only -- the only ones greedy decoding needs -- so, unlike forward(), it
// does not run the LM head over earlier positions and does not dump activations.
std::vector<float> forward_cached(const Model& model, KVCache& cache,
                                  const std::vector<int>& tokens, int n_past);

}  // namespace ie
