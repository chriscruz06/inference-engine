// src/forward.cpp
#include "forward.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>

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
    // Dump layer 0's ln_1 so it can be diffed in isolation before attention
    // exists. Once the real per-layer loop below is written, ln_1 gets dumped
    // from inside it and this whole block is deleted.
    if (dump_dir) {
        const LayerWeights& L0 = model.w.layers[0];
        std::vector<float> ln(x.size());
        for (int t = 0; t < seq; ++t)
            layernorm(x.data() + static_cast<std::size_t>(t) * d, model.at(L0.ln_1_w),
                      model.at(L0.ln_1_b), ln.data() + static_cast<std::size_t>(t) * d, d,
                      c.ln_eps);
        npy::save_2d(std::string(dump_dir) + "/01_L00.ln_1.npy", ln, seq, d);
    }
    // ----------------------------------------------------------------------

    // TODO (next): per layer, ln_1 -> attn -> +residual -> ln_2 -> mlp ->
    // +residual, dumping each tensor; then final ln_f -> tied LM head -> logits.
    // Returns last-position logits once that lands; zero placeholder for now.
    return std::vector<float>(static_cast<std::size_t>(c.vocab_size), 0.0f);
}

}  // namespace ie