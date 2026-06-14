#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ie {

// Configuration for a GPT-2-style decoder-only transformer.
// Defaults below are GPT-2 124M ("small"). load_model() overwrites these from
// the file header, so the on-disk model is always the source of truth.
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

// Offsets (in floats) of one transformer block's tensors into Model::weights.
// Linear weights are [out, in] row-major; biases are [out].
struct LayerWeights {
    std::size_t ln_1_w = 0, ln_1_b = 0;
    std::size_t c_attn_w = 0, c_attn_b = 0;  // [3*d_model, d_model], [3*d_model]
    std::size_t c_proj_w = 0, c_proj_b = 0;  // [d_model, d_model], [d_model]
    std::size_t ln_2_w = 0, ln_2_b = 0;
    std::size_t mlp_fc_w = 0, mlp_fc_b = 0;      // [d_mlp, d_model], [d_mlp]
    std::size_t mlp_proj_w = 0, mlp_proj_b = 0;  // [d_model, d_mlp], [d_model]
};

// Per-tensor offsets into the flat weight buffer, recorded at load time so the
// forward pass can index tensors without re-deriving any layout.
struct Weights {
    std::size_t wte = 0;  // [vocab_size, d_model], tied with the LM head
    std::size_t wpe = 0;  // [n_ctx, d_model]
    std::vector<LayerWeights> layers;
    std::size_t ln_f_w = 0, ln_f_b = 0;
};

// Holds model weights loaded from disk: one flat fp32 buffer plus the offsets
// that carve it into named tensors.
struct Model {
    ModelConfig config;
    std::vector<float> weights;  // flat fp32 buffer, canonical tensor order
    Weights w;

    // Pointer to a tensor given its float offset into `weights`.
    const float* at(std::size_t offset) const { return weights.data() + offset; }
};

// Load a model from a flat binary produced by tools/export_weights.py.
// Reads the header (magic "GPT2" + config), populates `out.config`, loads the
// tensors in canonical order, and records per-tensor offsets in `out.w`.
// Returns false (with a message on stderr) on any mismatch or I/O error.
bool load_model(const std::string& path, Model& out);

}  // namespace ie
