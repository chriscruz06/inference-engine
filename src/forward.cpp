// src/forward.cpp
#include "forward.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
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
    const int dm = c.d_mlp;
    const int V = c.vocab_size;
    const char* dump_dir = std::getenv("IE_DUMP_DIR");
    std::string dir;
    if (dump_dir) dir = dump_dir;

    // ---- Embedding: wte[token] + wpe[pos] (no scale in GPT-2). ----
    std::vector<float> x(static_cast<std::size_t>(seq) * static_cast<std::size_t>(d));
    const float* wte = model.at(model.w.wte);  // [vocab, d]; reused as the tied LM head below
    const float* wpe = model.at(model.w.wpe);
    for (int t = 0; t < seq; ++t) {
        const float* tok_row = wte + static_cast<std::size_t>(tokens[t]) * d;
        const float* pos_row = wpe + static_cast<std::size_t>(t) * d;  // position == t (prefill)
        float* dst = x.data() + static_cast<std::size_t>(t) * d;
        for (int j = 0; j < d; ++j) dst[j] = tok_row[j] + pos_row[j];
    }
    if (dump_dir) npy::save_2d(dir + "/00_embed.npy", x, seq, d);

    // ---- Per-layer temporaries: allocated once, overwritten each layer. ----
    std::vector<float> ln(x.size());                              // ln_1 / ln_2 output, [seq, d]
    std::vector<float> attn(x.size());                            // attention output, [seq, d]
    std::vector<float> mlp_h(static_cast<std::size_t>(seq) * dm); // MLP hidden, [seq, d_mlp]
    std::vector<float> mlp(x.size());                             // MLP output, [seq, d]

    // ---- Transformer blocks. x is the residual stream, carried across layers.
    // Pre-LN GPT-2 block:  x += attn(ln_1(x));  x += mlp(ln_2(x)).
    // Dumps are gated on IE_DUMP_DIR; the computation always runs. ----
    for (int l = 0; l < c.n_layers; ++l) {
        const LayerWeights& L = model.w.layers[static_cast<std::size_t>(l)];
        char tag[16];
        std::snprintf(tag, sizeof(tag), "%02d_L%02d", l + 1, l);  // "01_L00" .. "12_L11"

        // ln_1 -> attention -> residual 1.
        for (int t = 0; t < seq; ++t)
            layernorm(x.data() + static_cast<std::size_t>(t) * d, model.at(L.ln_1_w),
                      model.at(L.ln_1_b), ln.data() + static_cast<std::size_t>(t) * d, d,
                      c.ln_eps);
        if (dump_dir) npy::save_2d(dir + "/" + tag + ".ln_1.npy", ln, seq, d);

        attention(ln.data(), model.at(L.c_attn_w), model.at(L.c_attn_b), model.at(L.c_proj_w),
                  model.at(L.c_proj_b), attn.data(), seq, d, c.n_heads);
        if (dump_dir) npy::save_2d(dir + "/" + tag + ".attn.npy", attn, seq, d);

        for (std::size_t i = 0; i < x.size(); ++i) x[i] += attn[i];

        // ln_2 -> MLP -> residual 2.
        for (int t = 0; t < seq; ++t)
            layernorm(x.data() + static_cast<std::size_t>(t) * d, model.at(L.ln_2_w),
                      model.at(L.ln_2_b), ln.data() + static_cast<std::size_t>(t) * d, d,
                      c.ln_eps);
        if (dump_dir) npy::save_2d(dir + "/" + tag + ".ln_2.npy", ln, seq, d);

        linear(ln.data(), model.at(L.mlp_fc_w), model.at(L.mlp_fc_b), mlp_h.data(), seq, d, dm);
        gelu_tanh(mlp_h.data(), static_cast<int>(mlp_h.size()));
        linear(mlp_h.data(), model.at(L.mlp_proj_w), model.at(L.mlp_proj_b), mlp.data(), seq, dm,
               d);
        if (dump_dir) npy::save_2d(dir + "/" + tag + ".mlp.npy", mlp, seq, d);

        for (std::size_t i = 0; i < x.size(); ++i) x[i] += mlp[i];
        if (dump_dir) npy::save_2d(dir + "/" + tag + ".block.npy", x, seq, d);
    }

    // ---- Final LayerNorm. ----
    std::vector<float> lnf(x.size());
    for (int t = 0; t < seq; ++t)
        layernorm(x.data() + static_cast<std::size_t>(t) * d, model.at(model.w.ln_f_w),
                  model.at(model.w.ln_f_b), lnf.data() + static_cast<std::size_t>(t) * d, d,
                  c.ln_eps);
    if (dump_dir) npy::save_2d(dir + "/zz_ln_f.npy", lnf, seq, d);

    // ---- Tied LM head: logits = lnf @ wte^T. wte is [vocab, d] = [out, in], so
    // it drops straight into linear() as the weight. The head has no bias of its
    // own; feed a zero vector so this is correct whether or not linear() guards a
    // null bias pointer. (If yours does, you can pass nullptr and drop no_bias.)
    std::vector<float> logits(static_cast<std::size_t>(seq) * static_cast<std::size_t>(V));
    std::vector<float> no_bias(static_cast<std::size_t>(V), 0.0f);
    linear(lnf.data(), wte, no_bias.data(), logits.data(), seq, d, V);
    if (dump_dir) npy::save_2d(dir + "/zz_logits.npy", logits, seq, V);

    // Last-position logits drive greedy decoding.
    return std::vector<float>(logits.end() - V, logits.end());
}

}  // namespace ie