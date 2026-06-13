#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

// Configuration for a GPT-2-style decoder-only transformer.
// Defaults below are GPT-2 124M ("small"). Getting these exactly right is
// what makes the reference-diff harness pass.
struct ModelConfig {
    int n_layers = 12;
    int n_heads = 12;
    int d_model = 768;
    int head_dim = 64;  // d_model / n_heads
    int n_ctx = 1024;
    int vocab_size = 50257;
    int d_mlp = 3072;  // 4 * d_model
    float ln_eps = 1e-5f;
    bool tied_embeddings = true;  // GPT-2 ties wte with the LM head
    bool linear_bias = true;      // GPT-2 linears have biases (Llama: false)
};

// Holds model weights loaded from disk. Skeleton for now: the forward-pass
// phase wires typed views/offsets into the flat buffer.
struct Model {
    ModelConfig config;
    std::vector<float> weights;  // flat fp32 buffer (placeholder layout)
    // TODO: per-tensor offsets into `weights` (wte, wpe, per-layer LN/attn/mlp).
};

// Load a model from a flat binary produced by tools/export_weights.py.
// Format: a small header (magic + config) followed by raw fp32 tensors in a
// fixed order. Returns false on failure. (Skeleton: not yet implemented.)
bool load_model(const std::string& path, Model& out);

}  // namespace ie
