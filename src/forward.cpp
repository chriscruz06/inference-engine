// src/forward.cpp
#include "forward.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include "gemm.hpp"
#include "npy.hpp"

namespace ie {

void layernorm(const float* x, const float* weight, const float* bias, float* out, int n,
               float eps) {
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= static_cast<float>(n);

    float var = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float c = x[i] - mean;
        var += c * c;
    }
    var /= static_cast<float>(n);  // population variance, matches torch (unbiased=False)

    const float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < n; ++i) out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
}

void gelu_tanh(float* x, int n) {
    const float k = 0.7978845608028654f;  // sqrt(2/pi)
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        const float inner = k * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

void softmax(float* x, int n) {
    float m = x[0];
    for (int i = 1; i < n; ++i) m = (x[i] > m) ? x[i] : m;

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - m);
        sum += x[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

// One multi-head self-attention block with a causal mask.
//   x        : [seq, d] -- the ln_1 output (input to attention)
//   out      : [seq, d] -- result AFTER c_proj, BEFORE the residual add
//              (this is exactly what HF's block.attn hook captures)
// Q/K/V are the three contiguous d-blocks of the fused c_attn output; head h
// occupies columns [h*head_dim, (h+1)*head_dim) within each block.
static void attention(const float* x, const float* c_attn_w, const float* c_attn_b,
                      const float* c_proj_w, const float* c_proj_b, float* out, int seq, int d,
                      int n_heads) {
    const int hd = d / n_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // 1) Fused QKV projection -> [seq, 3d].
    std::vector<float> qkv(static_cast<std::size_t>(seq) * 3 * static_cast<std::size_t>(d));
    linear(x, c_attn_w, c_attn_b, qkv.data(), seq, d, 3 * d);

    // 2) Per-head scaled dot-product attention. Query i attends to keys 0..i.
    std::vector<float> context(static_cast<std::size_t>(seq) * d, 0.0f);
    std::vector<float> scores(static_cast<std::size_t>(seq));  // only [0..i] used per query
    const std::size_t row = static_cast<std::size_t>(3) * d;   // stride of one qkv row

    for (int h = 0; h < n_heads; ++h) {
        const int qoff = h * hd;          // Q block for this head
        const int koff = d + h * hd;      // K block
        const int voff = 2 * d + h * hd;  // V block
        for (int i = 0; i < seq; ++i) {
            const float* qi = qkv.data() + static_cast<std::size_t>(i) * row + qoff;
            for (int j = 0; j <= i; ++j) {  // causal: keys up to and including i
                const float* kj = qkv.data() + static_cast<std::size_t>(j) * row + koff;
                float dot = 0.0f;
                for (int e = 0; e < hd; ++e) dot += qi[e] * kj[e];
                scores[static_cast<std::size_t>(j)] = dot * scale;
            }
            softmax(scores.data(), i + 1);  // normalize over the valid prefix only

            float* ctx = context.data() + static_cast<std::size_t>(i) * d + qoff;
            for (int j = 0; j <= i; ++j) {
                const float* vj = qkv.data() + static_cast<std::size_t>(j) * row + voff;
                const float a = scores[static_cast<std::size_t>(j)];
                for (int e = 0; e < hd; ++e) ctx[e] += a * vj[e];
            }
        }
    }

    // 3) Output projection -> [seq, d].
    linear(context.data(), c_proj_w, c_proj_b, out, seq, d, d);
}

std::vector<float> forward(const Model& model, const std::vector<int>& tokens) {
    const ModelConfig& c = model.config;
    const int seq = static_cast<int>(tokens.size());
    const int d = c.d_model;
    const char* dump_dir = std::getenv("IE_DUMP_DIR");

    // Residual stream, [seq, d]. Embedding: wte[token] + wpe[pos] (no scale in GPT-2).
    std::vector<float> x(static_cast<std::size_t>(seq) * static_cast<std::size_t>(d));
    const float* wte = model.at(model.w.wte);
    const float* wpe = model.at(model.w.wpe);
    for (int t = 0; t < seq; ++t) {
        const float* tok_row = wte + static_cast<std::size_t>(tokens[t]) * d;
        const float* pos_row = wpe + static_cast<std::size_t>(t) * d;  // position == t (prefill)
        float* dst = x.data() + static_cast<std::size_t>(t) * d;
        for (int j = 0; j < d; ++j) dst[j] = tok_row[j] + pos_row[j];
    }
    if (dump_dir) npy::save_2d(std::string(dump_dir) + "/00_embed.npy", x, seq, d);

    // --- TEMPORARY bringup scaffold ---------------------------------------
    // Dump layer 0's ln_1 and attn so each can be diffed in isolation before
    // the MLP and the full block loop exist. Once the per-layer loop below is
    // written, these dumps move inside it and this whole block is deleted.
    if (dump_dir) {
        const std::string dir(dump_dir);
        const LayerWeights& L0 = model.w.layers[0];

        std::vector<float> ln1(x.size());
        for (int t = 0; t < seq; ++t)
            layernorm(x.data() + static_cast<std::size_t>(t) * d, model.at(L0.ln_1_w),
                      model.at(L0.ln_1_b), ln1.data() + static_cast<std::size_t>(t) * d, d,
                      c.ln_eps);
        npy::save_2d(dir + "/01_L00.ln_1.npy", ln1, seq, d);

        std::vector<float> attn(x.size());
        attention(ln1.data(), model.at(L0.c_attn_w), model.at(L0.c_attn_b), model.at(L0.c_proj_w),
                  model.at(L0.c_proj_b), attn.data(), seq, d, c.n_heads);
        npy::save_2d(dir + "/01_L00.attn.npy", attn, seq, d);
    }
    // ----------------------------------------------------------------------

    // TODO (next): residual add, then ln_2 -> mlp -> residual for the full
    // block; loop x n_layers; final ln_f -> tied LM head -> logits.
    // Returns last-position logits once that lands; zero placeholder for now.
    return std::vector<float>(static_cast<std::size_t>(c.vocab_size), 0.0f);
}

}  // namespace ie