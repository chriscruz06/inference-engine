#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "quant.hpp"

namespace ie {

// Which architecture a loaded model is. Selected by the file magic ("GPT2" vs
// "LLAM") and used to dispatch the forward pass and weight layout.
enum class Arch { GPT2, Llama };

// Configuration for a decoder-only transformer (GPT-2 or Llama).
// Defaults below are GPT-2 124M ("small"). load_model() overwrites these from
// the file header, so the on-disk model is always the source of truth.
struct ModelConfig {
    Arch arch = Arch::GPT2;
    int n_layers = 12;
    int n_heads = 12;
    int n_kv_heads = 0;  // KV heads for GQA; 0 means MHA (n_kv_heads == n_heads), as in GPT-2
    int d_model = 768;
    int head_dim = 64;  // d_model / n_heads
    int n_ctx = 1024;
    int vocab_size = 50257;
    int d_mlp = 3072;             // GPT-2: 4*d_model; Llama: SwiGLU intermediate_size
    float ln_eps = 1e-5f;         // LayerNorm eps (GPT-2) or RMSNorm eps (Llama)
    float rope_theta = 10000.0f;  // RoPE base frequency (Llama only)
    bool tied_embeddings = true;  // GPT-2 ties wte with the LM head; TinyLlama does not
    bool linear_bias = true;      // GPT-2 linears have biases (Llama: false)

    // KV-cache row width: n_kv_heads*head_dim under GQA, d_model for MHA/GPT-2.
    int kv_heads() const { return n_kv_heads > 0 ? n_kv_heads : n_heads; }
    int kv_dim() const { return kv_heads() * head_dim; }
};

// Offsets (in floats) of one GPT-2 block's tensors into Model::weights.
// Linear weights are [out, in] row-major; biases are [out].
struct LayerWeights {
    std::size_t ln_1_w = 0, ln_1_b = 0;
    std::size_t c_attn_w = 0, c_attn_b = 0;  // [3*d_model, d_model], [3*d_model]
    std::size_t c_proj_w = 0, c_proj_b = 0;  // [d_model, d_model], [d_model]
    std::size_t ln_2_w = 0, ln_2_b = 0;
    std::size_t mlp_fc_w = 0, mlp_fc_b = 0;      // [d_mlp, d_model], [d_mlp]
    std::size_t mlp_proj_w = 0, mlp_proj_b = 0;  // [d_model, d_mlp], [d_model]
};

// Offsets of one Llama block's tensors. Bias-free; RMSNorm has weight only.
// Attention is GQA: q is [n_heads*head_dim, d], k/v are [n_kv_heads*head_dim, d]
// (narrower), o is [d, n_heads*head_dim]. MLP is SwiGLU (gate/up/down).
struct LlamaLayerWeights {
    std::size_t input_ln_w = 0;                  // RMSNorm before attention
    std::size_t q_w = 0, k_w = 0, v_w = 0;       // [q_dim,d], [kv_dim,d], [kv_dim,d]
    std::size_t o_w = 0;                         // [d, q_dim]
    std::size_t post_attn_ln_w = 0;              // RMSNorm before MLP
    std::size_t gate_w = 0, up_w = 0, down_w = 0;  // [d_mlp,d], [d_mlp,d], [d,d_mlp]
};

// Per-tensor offsets into the flat weight buffer, recorded at load time so the
// forward pass can index tensors without re-deriving any layout. Only the set
// matching config.arch is populated.
struct Weights {
    std::size_t wte = 0;  // [vocab, d]; GPT-2: tied LM head. Llama: token embedding.
    std::size_t wpe = 0;  // [n_ctx, d] (GPT-2 only; Llama uses RoPE)
    std::vector<LayerWeights> layers;             // GPT-2 blocks
    std::vector<LlamaLayerWeights> llama_layers;  // Llama blocks
    std::size_t ln_f_w = 0, ln_f_b = 0;  // final norm (Llama uses ln_f_w only, RMSNorm)
    std::size_t lm_head = 0;             // output projection [vocab, d]; == wte when tied
};

// Quantized copies of one block's four streamed matmul weights (Phase 5). Empty
// until quantize_model() runs. The fp32 originals stay in Model::weights (biases,
// LayerNorm, and the embedding lookup still read them); only the matmul reads
// switch to these.
struct QuantLayerWeights {
    QuantTensor c_attn_w;    // [3*d_model, d_model]
    QuantTensor c_proj_w;    // [d_model, d_model]
    QuantTensor mlp_fc_w;    // [d_mlp, d_model]
    QuantTensor mlp_proj_w;  // [d_model, d_mlp]
};

// Quantized copies of one Llama block's seven streamed matmul weights. Empty
// until quantize_model() runs. The fp32 originals stay in Model::weights (RMSNorm
// and the embedding lookup still read them).
struct LlamaQuantLayerWeights {
    QuantTensor q_w, k_w, v_w, o_w;    // attention projections (k/v narrower under GQA)
    QuantTensor gate_w, up_w, down_w;  // SwiGLU MLP
};

// Holds model weights loaded from disk: one flat fp32 buffer plus the offsets
// that carve it into named tensors.
struct Model {
    ModelConfig config;
    std::vector<float> weights;  // flat fp32 buffer, canonical tensor order
    Weights w;

    // Phase 5: optional in-memory weight quantization. `quantized` flips the
    // forward pass onto the quantized matmul kernels; the stores below hold the
    // int8/int4 weights. wte stays fp32 for the embedding lookup, so the tied LM
    // head gets its own quantized copy in `q_head`.
    bool quantized = false;
    QuantType quant_type = QuantType::Q8;
    std::vector<QuantLayerWeights> qlayers;             // GPT-2: parallel to w.layers
    std::vector<LlamaQuantLayerWeights> llama_qlayers;  // Llama: parallel to w.llama_layers
    QuantTensor q_head;  // quantized output projection (GPT-2 tied wte, or Llama lm_head)

    // Pointer to a tensor given its float offset into `weights`.
    const float* at(std::size_t offset) const { return weights.data() + offset; }
};

// Load a model from a flat binary produced by tools/export_weights.py.
// Reads the magic ("GPT2" or "LLAM") to pick the architecture, then the matching
// header + config, loads the tensors in canonical order, and records per-tensor
// offsets in `out.w` (the GPT-2 set or the Llama set, per config.arch).
// Returns false (with a message on stderr) on any mismatch or I/O error.
bool load_model(const std::string& path, Model& out);

// Quantize the streamed matmul weights (per-layer c_attn / c_proj / mlp_fc /
// mlp_proj, plus the tied LM head) in place into the model's quantized stores and
// set `quantized = true`. The fp32 buffer is left intact. `group` defaults to the
// model-wide kQuantGroup. Returns the total bytes of the quantized weights (for
// footprint reporting); `fp32_bytes_out`, if non-null, receives the fp32 size of
// the same matrices so the caller can quote the ratio.
std::size_t quantize_model(Model& m, QuantType type, int group = kQuantGroup,
                           std::size_t* fp32_bytes_out = nullptr);

}  // namespace ie
