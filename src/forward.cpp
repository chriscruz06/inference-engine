// src/forward.cpp
#include "forward.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gemm.hpp"
#include "kv_cache.hpp"
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

// One multi-head self-attention block with a causal mask (no cache).
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

// One multi-head self-attention block backed by the KV cache.
//   xn   : [n_new, d] -- the ln_1 output for the new tokens only
//   out  : [n_new, d] -- result AFTER c_proj, BEFORE the residual add
// The new tokens' K/V are appended to the cache at positions
// [n_past, n_past + n_new); the query at new-token i (absolute position
// n_past + i) attends causally over cached keys 0..(n_past + i). With n_past == 0
// and n_new == seq this computes exactly what attention() above does; with
// n_new == 1 it is the single-token decode step (Q is one vector -> GEMV).
static void attention_kv(const float* xn, const LinearWeight& c_attn_w, const float* c_attn_b,
                         const LinearWeight& c_proj_w, const float* c_proj_b, float* out,
                         KVCache& cache, int layer, int n_new, int n_past, int d, int n_heads) {
    const int hd = d / n_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // 1) Fused QKV projection for the new tokens only -> [n_new, 3d]. The weight
    // is fp32 or quantized depending on the LinearWeight (decode's bandwidth win).
    std::vector<float> qkv(static_cast<std::size_t>(n_new) * 3 * static_cast<std::size_t>(d));
    linear(xn, c_attn_w, c_attn_b, qkv.data(), n_new);

    // 2) Append this chunk's K and V into the cache (full d-row per position).
    const std::size_t row = static_cast<std::size_t>(3) * d;  // stride of one qkv row
    for (int i = 0; i < n_new; ++i) {
        const float* qkv_i = qkv.data() + static_cast<std::size_t>(i) * row;
        std::copy(qkv_i + d, qkv_i + 2 * d, cache.k_row(layer, n_past + i));
        std::copy(qkv_i + 2 * d, qkv_i + 3 * d, cache.v_row(layer, n_past + i));
    }

    // 3) Per-head scaled dot-product attention over the cached K/V.
    std::vector<float> context(static_cast<std::size_t>(n_new) * d, 0.0f);
    std::vector<float> scores(static_cast<std::size_t>(n_past) + n_new);  // [0..p] used per query

    for (int h = 0; h < n_heads; ++h) {
        const int hoff = h * hd;  // this head's slice within a d-row (Q in qkv; K/V in cache)
        for (int i = 0; i < n_new; ++i) {
            const int p = n_past + i;  // absolute query position
            const float* qi = qkv.data() + static_cast<std::size_t>(i) * row + hoff;
            for (int j = 0; j <= p; ++j) {  // causal: cached keys 0..p inclusive
                const float* kj = cache.k_row(layer, j) + hoff;
                float dot = 0.0f;
                for (int e = 0; e < hd; ++e) dot += qi[e] * kj[e];
                scores[static_cast<std::size_t>(j)] = dot * scale;
            }
            softmax(scores.data(), p + 1);

            float* ctx = context.data() + static_cast<std::size_t>(i) * d + hoff;
            for (int j = 0; j <= p; ++j) {
                const float* vj = cache.v_row(layer, j) + hoff;
                const float a = scores[static_cast<std::size_t>(j)];
                for (int e = 0; e < hd; ++e) ctx[e] += a * vj[e];
            }
        }
    }

    // 4) Output projection -> [n_new, d].
    linear(context.data(), c_proj_w, c_proj_b, out, n_new);
}

// ============================================================================
// Llama path: RMSNorm, RoPE (rotate-half), GQA attention, SwiGLU MLP. Selected
// by config.arch; the GPT-2 functions above are untouched. forward_llama is the
// dump oracle (mirrors forward()); forward_llama_cached is the KV-cache fast path.
// ============================================================================

// RMSNorm over the last dim: x / sqrt(mean(x^2) + eps) * weight. No mean
// subtraction and no bias (the subtle difference from layernorm()).
static void rmsnorm(const float* x, const float* weight, float* out, int n, float eps) {
    float ms = 0.0f;
    for (int i = 0; i < n; ++i) ms += x[i] * x[i];
    ms /= static_cast<float>(n);
    const float inv = 1.0f / std::sqrt(ms + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * weight[i];
}

// SiLU (swish) in place: x * sigmoid(x). The SwiGLU gate activation.
static void silu(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        x[i] = v / (1.0f + std::exp(-v));
    }
}

// RoPE inverse frequencies: inv_freq[i] = theta^(-2i/head_dim), i in [0, hd/2).
static std::vector<float> rope_inv_freq(int head_dim, float theta) {
    std::vector<float> f(static_cast<std::size_t>(head_dim / 2));
    for (int i = 0; i < head_dim / 2; ++i)
        f[static_cast<std::size_t>(i)] = static_cast<float>(
            1.0 / std::pow(static_cast<double>(theta), (2.0 * i) / head_dim));
    return f;
}

// Apply rotary position embedding (HF "rotate-half") in place to one head vector
// `v` of length hd at absolute position `pos`. The two HALVES are paired (index i
// with i+hd/2), NOT adjacent elements -- matching HF Llama, not the interleaved
// convention. This pairing is the classic RoPE divergence source.
static void rope_inplace(float* v, int hd, int pos, const float* inv_freq) {
    const int half = hd / 2;
    for (int i = 0; i < half; ++i) {
        const float angle = static_cast<float>(pos) * inv_freq[static_cast<std::size_t>(i)];
        const float cs = std::cos(angle);
        const float sn = std::sin(angle);
        const float a = v[i];
        const float b = v[i + half];
        v[i] = a * cs - b * sn;
        v[i + half] = b * cs + a * sn;
    }
}

// Llama multi-head attention, no cache (the oracle path). GQA: query head h reads
// KV head h/(n_heads/n_kv_heads). RoPE is applied to Q and K (not V) at each
// token's absolute position (== t with no cache). out is [seq, d] after o_proj,
// before the residual add (what HF's self_attn hook captures).
static void llama_attention(const float* x, const LlamaLayerWeights& L, const Model& model,
                            float* out, int seq, const ModelConfig& c, const float* inv_freq) {
    const int d = c.d_model;
    const int hd = c.head_dim;
    const int nh = c.n_heads;
    const int nkv = c.kv_heads();
    const int q_dim = nh * hd;
    const int kv_dim = nkv * hd;
    const int group = nh / nkv;  // query heads sharing one KV head
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    std::vector<float> q(static_cast<std::size_t>(seq) * q_dim);
    std::vector<float> k(static_cast<std::size_t>(seq) * kv_dim);
    std::vector<float> v(static_cast<std::size_t>(seq) * kv_dim);
    linear(x, model.at(L.q_w), nullptr, q.data(), seq, d, q_dim);
    linear(x, model.at(L.k_w), nullptr, k.data(), seq, d, kv_dim);
    linear(x, model.at(L.v_w), nullptr, v.data(), seq, d, kv_dim);

    for (int t = 0; t < seq; ++t) {
        float* qt = q.data() + static_cast<std::size_t>(t) * q_dim;
        float* kt = k.data() + static_cast<std::size_t>(t) * kv_dim;
        for (int h = 0; h < nh; ++h) rope_inplace(qt + h * hd, hd, t, inv_freq);
        for (int g = 0; g < nkv; ++g) rope_inplace(kt + g * hd, hd, t, inv_freq);
    }

    std::vector<float> context(static_cast<std::size_t>(seq) * q_dim, 0.0f);
    std::vector<float> scores(static_cast<std::size_t>(seq));
    for (int h = 0; h < nh; ++h) {
        const int g = h / group;  // KV head this query head reads
        const int qoff = h * hd;  // Q slice within a q_dim row
        const int koff = g * hd;  // K/V slice within a kv_dim row
        for (int i = 0; i < seq; ++i) {
            const float* qi = q.data() + static_cast<std::size_t>(i) * q_dim + qoff;
            for (int j = 0; j <= i; ++j) {
                const float* kj = k.data() + static_cast<std::size_t>(j) * kv_dim + koff;
                float dot = 0.0f;
                for (int e = 0; e < hd; ++e) dot += qi[e] * kj[e];
                scores[static_cast<std::size_t>(j)] = dot * scale;
            }
            softmax(scores.data(), i + 1);
            float* ctx = context.data() + static_cast<std::size_t>(i) * q_dim + qoff;
            for (int j = 0; j <= i; ++j) {
                const float* vj = v.data() + static_cast<std::size_t>(j) * kv_dim + koff;
                const float a = scores[static_cast<std::size_t>(j)];
                for (int e = 0; e < hd; ++e) ctx[e] += a * vj[e];
            }
        }
    }
    linear(context.data(), model.at(L.o_w), nullptr, out, seq, q_dim, d);
}

// Llama attention backed by the KV cache (fast path). The new tokens' K/V are
// RoPE'd then appended at [n_past, n_past+n_new); each query attends causally over
// cached keys 0..(n_past+i). Cache rows are kv_dim wide (GQA), KV head g at
// columns [g*head_dim, (g+1)*head_dim). Post-RoPE keys are stored (HF caches the
// rotated K); V is stored unrotated.
static void llama_attention_kv(const float* xn, const LinearWeight& w_q, const LinearWeight& w_k,
                               const LinearWeight& w_v, const LinearWeight& w_o, float* out,
                               KVCache& cache, int layer, int n_new, int n_past,
                               const ModelConfig& c, const float* inv_freq) {
    const int hd = c.head_dim;
    const int nh = c.n_heads;
    const int nkv = c.kv_heads();
    const int q_dim = nh * hd;
    const int kv_dim = nkv * hd;
    const int group = nh / nkv;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    std::vector<float> q(static_cast<std::size_t>(n_new) * q_dim);
    std::vector<float> k(static_cast<std::size_t>(n_new) * kv_dim);
    std::vector<float> v(static_cast<std::size_t>(n_new) * kv_dim);
    linear(xn, w_q, nullptr, q.data(), n_new);  // fp32 or quantized per LinearWeight
    linear(xn, w_k, nullptr, k.data(), n_new);
    linear(xn, w_v, nullptr, v.data(), n_new);

    // RoPE Q and K at the absolute position, then append K (post-RoPE) and V.
    for (int i = 0; i < n_new; ++i) {
        const int pos = n_past + i;
        float* qi = q.data() + static_cast<std::size_t>(i) * q_dim;
        float* ki = k.data() + static_cast<std::size_t>(i) * kv_dim;
        const float* vi = v.data() + static_cast<std::size_t>(i) * kv_dim;
        for (int h = 0; h < nh; ++h) rope_inplace(qi + h * hd, hd, pos, inv_freq);
        for (int g = 0; g < nkv; ++g) rope_inplace(ki + g * hd, hd, pos, inv_freq);
        std::copy(ki, ki + kv_dim, cache.k_row(layer, pos));
        std::copy(vi, vi + kv_dim, cache.v_row(layer, pos));
    }

    std::vector<float> context(static_cast<std::size_t>(n_new) * q_dim, 0.0f);
    std::vector<float> scores(static_cast<std::size_t>(n_past) + n_new);
    for (int h = 0; h < nh; ++h) {
        const int g = h / group;
        const int qoff = h * hd;
        const int koff = g * hd;  // KV head slice within a kv_dim cache row
        for (int i = 0; i < n_new; ++i) {
            const int p = n_past + i;  // absolute query position
            const float* qi = q.data() + static_cast<std::size_t>(i) * q_dim + qoff;
            for (int j = 0; j <= p; ++j) {
                const float* kj = cache.k_row(layer, j) + koff;
                float dot = 0.0f;
                for (int e = 0; e < hd; ++e) dot += qi[e] * kj[e];
                scores[static_cast<std::size_t>(j)] = dot * scale;
            }
            softmax(scores.data(), p + 1);
            float* ctx = context.data() + static_cast<std::size_t>(i) * q_dim + qoff;
            for (int j = 0; j <= p; ++j) {
                const float* vj = cache.v_row(layer, j) + koff;
                const float a = scores[static_cast<std::size_t>(j)];
                for (int e = 0; e < hd; ++e) ctx[e] += a * vj[e];
            }
        }
    }
    linear(context.data(), w_o, nullptr, out, n_new);
}

// Llama forward (dump oracle). Mirrors forward(): recompute the whole sequence,
// run the LM head over every position, dump activations when IE_DUMP_DIR is set.
static std::vector<float> forward_llama(const Model& model, const std::vector<int>& tokens) {
    const ModelConfig& c = model.config;
    const int seq = static_cast<int>(tokens.size());
    const int d = c.d_model;
    const int dm = c.d_mlp;
    const int V = c.vocab_size;
    const char* dump_dir = std::getenv("IE_DUMP_DIR");
    std::string dir;
    if (dump_dir) dir = dump_dir;

    const std::vector<float> inv_freq = rope_inv_freq(c.head_dim, c.rope_theta);

    // ---- Embedding: tok_embed[token] only; RoPE supplies position later. ----
    std::vector<float> x(static_cast<std::size_t>(seq) * d);
    const float* wte = model.at(model.w.wte);
    for (int t = 0; t < seq; ++t) {
        const float* tok = wte + static_cast<std::size_t>(tokens[t]) * d;
        std::copy(tok, tok + d, x.data() + static_cast<std::size_t>(t) * d);
    }
    if (dump_dir) npy::save_2d(dir + "/00_embed.npy", x, seq, d);

    std::vector<float> ln(x.size());
    std::vector<float> attn(x.size());
    std::vector<float> gate(static_cast<std::size_t>(seq) * dm);
    std::vector<float> up(static_cast<std::size_t>(seq) * dm);
    std::vector<float> mlp(x.size());

    for (int l = 0; l < c.n_layers; ++l) {
        const LlamaLayerWeights& L = model.w.llama_layers[static_cast<std::size_t>(l)];
        char tag[32];
        std::snprintf(tag, sizeof(tag), "%02d_L%02d", l + 1, l);

        // input RMSNorm -> attention -> residual 1.
        for (int t = 0; t < seq; ++t)
            rmsnorm(x.data() + static_cast<std::size_t>(t) * d, model.at(L.input_ln_w),
                    ln.data() + static_cast<std::size_t>(t) * d, d, c.ln_eps);
        if (dump_dir) npy::save_2d(dir + "/" + tag + ".ln_1.npy", ln, seq, d);

        llama_attention(ln.data(), L, model, attn.data(), seq, c, inv_freq.data());
        if (dump_dir) npy::save_2d(dir + "/" + tag + ".attn.npy", attn, seq, d);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += attn[i];

        // post-attention RMSNorm -> SwiGLU MLP -> residual 2.
        for (int t = 0; t < seq; ++t)
            rmsnorm(x.data() + static_cast<std::size_t>(t) * d, model.at(L.post_attn_ln_w),
                    ln.data() + static_cast<std::size_t>(t) * d, d, c.ln_eps);
        if (dump_dir) npy::save_2d(dir + "/" + tag + ".ln_2.npy", ln, seq, d);

        linear(ln.data(), model.at(L.gate_w), nullptr, gate.data(), seq, d, dm);
        linear(ln.data(), model.at(L.up_w), nullptr, up.data(), seq, d, dm);
        silu(gate.data(), static_cast<int>(gate.size()));
        for (std::size_t i = 0; i < gate.size(); ++i) gate[i] *= up[i];
        linear(gate.data(), model.at(L.down_w), nullptr, mlp.data(), seq, dm, d);
        if (dump_dir) npy::save_2d(dir + "/" + tag + ".mlp.npy", mlp, seq, d);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += mlp[i];

        if (dump_dir) npy::save_2d(dir + "/" + tag + ".block.npy", x, seq, d);
    }

    // ---- Final RMSNorm + untied LM head over every position. ----
    std::vector<float> lnf(x.size());
    for (int t = 0; t < seq; ++t)
        rmsnorm(x.data() + static_cast<std::size_t>(t) * d, model.at(model.w.ln_f_w),
                lnf.data() + static_cast<std::size_t>(t) * d, d, c.ln_eps);
    if (dump_dir) npy::save_2d(dir + "/zz_ln_f.npy", lnf, seq, d);

    std::vector<float> logits(static_cast<std::size_t>(seq) * V);
    linear(lnf.data(), model.at(model.w.lm_head), nullptr, logits.data(), seq, d, V);
    if (dump_dir) npy::save_2d(dir + "/zz_logits.npy", logits, seq, V);

    return std::vector<float>(logits.end() - V, logits.end());
}

// Llama cache-aware forward (fast path). Same block as forward_llama, but
// attention reads/extends the KV cache and only the last position's logits are
// returned. fp32 weights for now (quantization wired in a later step).
static std::vector<float> forward_llama_cached(const Model& model, KVCache& cache,
                                               const std::vector<int>& tokens, int n_past) {
    const ModelConfig& c = model.config;
    const int n_new = static_cast<int>(tokens.size());
    const int d = c.d_model;
    const int dm = c.d_mlp;
    const int V = c.vocab_size;
    assert(cache.len == n_past);
    assert(n_past + n_new <= c.n_ctx);

    const std::vector<float> inv_freq = rope_inv_freq(c.head_dim, c.rope_theta);

    std::vector<float> x(static_cast<std::size_t>(n_new) * d);
    const float* wte = model.at(model.w.wte);
    for (int i = 0; i < n_new; ++i) {
        const float* tok = wte + static_cast<std::size_t>(tokens[i]) * d;
        std::copy(tok, tok + d, x.data() + static_cast<std::size_t>(i) * d);
    }

    std::vector<float> ln(x.size());
    std::vector<float> attn(x.size());
    std::vector<float> gate(static_cast<std::size_t>(n_new) * dm);
    std::vector<float> up(static_cast<std::size_t>(n_new) * dm);
    std::vector<float> mlp(x.size());

    // Resolve each streamed matmul weight to fp32 or its quantized copy, like the
    // GPT-2 cached path. When the model is not quantized this is byte-identical to
    // using the fp32 weights directly (the QuantTensor pointer is ignored).
    const bool Q = model.quantized;
    const int QD = c.n_heads * c.head_dim;
    const int KV = c.kv_heads() * c.head_dim;
    auto LW = [&](std::size_t off, const QuantTensor* qt, int in, int out) -> LinearWeight {
        if (Q) return LinearWeight{nullptr, qt, in, out};
        return LinearWeight{model.at(off), nullptr, in, out};
    };

    for (int l = 0; l < c.n_layers; ++l) {
        const LlamaLayerWeights& L = model.w.llama_layers[static_cast<std::size_t>(l)];
        const LlamaQuantLayerWeights* ql =
            Q ? &model.llama_qlayers[static_cast<std::size_t>(l)] : nullptr;

        for (int i = 0; i < n_new; ++i)
            rmsnorm(x.data() + static_cast<std::size_t>(i) * d, model.at(L.input_ln_w),
                    ln.data() + static_cast<std::size_t>(i) * d, d, c.ln_eps);

        const LinearWeight w_q = LW(L.q_w, ql ? &ql->q_w : nullptr, d, QD);
        const LinearWeight w_k = LW(L.k_w, ql ? &ql->k_w : nullptr, d, KV);
        const LinearWeight w_v = LW(L.v_w, ql ? &ql->v_w : nullptr, d, KV);
        const LinearWeight w_o = LW(L.o_w, ql ? &ql->o_w : nullptr, QD, d);
        llama_attention_kv(ln.data(), w_q, w_k, w_v, w_o, attn.data(), cache, l, n_new, n_past, c,
                           inv_freq.data());
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += attn[i];

        for (int i = 0; i < n_new; ++i)
            rmsnorm(x.data() + static_cast<std::size_t>(i) * d, model.at(L.post_attn_ln_w),
                    ln.data() + static_cast<std::size_t>(i) * d, d, c.ln_eps);

        const LinearWeight w_gate = LW(L.gate_w, ql ? &ql->gate_w : nullptr, d, dm);
        const LinearWeight w_up = LW(L.up_w, ql ? &ql->up_w : nullptr, d, dm);
        const LinearWeight w_down = LW(L.down_w, ql ? &ql->down_w : nullptr, dm, d);
        linear(ln.data(), w_gate, nullptr, gate.data(), n_new);
        linear(ln.data(), w_up, nullptr, up.data(), n_new);
        silu(gate.data(), static_cast<int>(gate.size()));
        for (std::size_t i = 0; i < gate.size(); ++i) gate[i] *= up[i];
        linear(gate.data(), w_down, nullptr, mlp.data(), n_new);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += mlp[i];
    }

    cache.len = n_past + n_new;

    const float* x_last = x.data() + static_cast<std::size_t>(n_new - 1) * d;
    std::vector<float> lnf(static_cast<std::size_t>(d));
    rmsnorm(x_last, model.at(model.w.ln_f_w), lnf.data(), d, c.ln_eps);

    std::vector<float> logits(static_cast<std::size_t>(V));
    const LinearWeight head = Q ? LinearWeight{nullptr, &model.q_head, d, V}
                                : LinearWeight{model.at(model.w.lm_head), nullptr, d, V};
    linear(lnf.data(), head, nullptr, logits.data(), 1);
    return logits;
}

std::vector<float> forward(const Model& model, const std::vector<int>& tokens) {
    if (model.config.arch == Arch::Llama) return forward_llama(model, tokens);
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
    std::vector<float> ln(x.size());                               // ln_1 / ln_2 output, [seq, d]
    std::vector<float> attn(x.size());                             // attention output, [seq, d]
    std::vector<float> mlp_h(static_cast<std::size_t>(seq) * dm);  // MLP hidden, [seq, d_mlp]
    std::vector<float> mlp(x.size());                              // MLP output, [seq, d]

    // ---- Transformer blocks. x is the residual stream, carried across layers.
    // Pre-LN GPT-2 block:  x += attn(ln_1(x));  x += mlp(ln_2(x)).
    // Dumps are gated on IE_DUMP_DIR; the computation always runs. ----
    for (int l = 0; l < c.n_layers; ++l) {
        const LayerWeights& L = model.w.layers[static_cast<std::size_t>(l)];
        char tag[32];  // wide enough for any int (silences -Wformat-truncation at -O2)
        std::snprintf(tag, sizeof(tag), "%02d_L%02d", l + 1, l);  // "01_L00" .. "12_L11"

        // ln_1 -> attention -> residual 1.
        for (int t = 0; t < seq; ++t)
            layernorm(x.data() + static_cast<std::size_t>(t) * d, model.at(L.ln_1_w),
                      model.at(L.ln_1_b), ln.data() + static_cast<std::size_t>(t) * d, d, c.ln_eps);
        if (dump_dir) npy::save_2d(dir + "/" + tag + ".ln_1.npy", ln, seq, d);

        attention(ln.data(), model.at(L.c_attn_w), model.at(L.c_attn_b), model.at(L.c_proj_w),
                  model.at(L.c_proj_b), attn.data(), seq, d, c.n_heads);
        if (dump_dir) npy::save_2d(dir + "/" + tag + ".attn.npy", attn, seq, d);

        for (std::size_t i = 0; i < x.size(); ++i) x[i] += attn[i];

        // ln_2 -> MLP -> residual 2.
        for (int t = 0; t < seq; ++t)
            layernorm(x.data() + static_cast<std::size_t>(t) * d, model.at(L.ln_2_w),
                      model.at(L.ln_2_b), ln.data() + static_cast<std::size_t>(t) * d, d, c.ln_eps);
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

std::vector<float> forward_cached(const Model& model, KVCache& cache,
                                  const std::vector<int>& tokens, int n_past) {
    if (model.config.arch == Arch::Llama)
        return forward_llama_cached(model, cache, tokens, n_past);
    const ModelConfig& c = model.config;
    const int n_new = static_cast<int>(tokens.size());
    const int d = c.d_model;
    const int dm = c.d_mlp;
    const int V = c.vocab_size;

    // The chunk must append contiguously and fit inside the context window.
    // (Callers in this project always satisfy these; the asserts document it.)
    assert(cache.len == n_past);
    assert(n_past + n_new <= c.n_ctx);

    // ---- Embedding: wte[token] + wpe[n_past + i]. The position is the ABSOLUTE
    // offset in the full sequence, not the index within this chunk -- the classic
    // KV-cache off-by-one if you forget it. ----
    std::vector<float> x(static_cast<std::size_t>(n_new) * static_cast<std::size_t>(d));
    const float* wte = model.at(model.w.wte);  // [vocab, d]; tied LM head below
    const float* wpe = model.at(model.w.wpe);
    for (int i = 0; i < n_new; ++i) {
        const float* tok_row = wte + static_cast<std::size_t>(tokens[i]) * d;
        const float* pos_row = wpe + static_cast<std::size_t>(n_past + i) * d;
        float* dst = x.data() + static_cast<std::size_t>(i) * d;
        for (int j = 0; j < d; ++j) dst[j] = tok_row[j] + pos_row[j];
    }

    // ---- Per-chunk temporaries, allocated once. ----
    std::vector<float> ln(x.size());
    std::vector<float> attn(x.size());
    std::vector<float> mlp_h(static_cast<std::size_t>(n_new) * dm);
    std::vector<float> mlp(x.size());

    // ---- Transformer blocks. Identical pre-LN GPT-2 block to forward(), except
    // attention reads/writes the KV cache instead of recomputing the past. ----
    // Resolve a streamed matmul weight to fp32 or its quantized copy. When the
    // model is not quantized the QuantTensor pointer is ignored and the fp32
    // offset is used, so this path is byte-identical to before. (qlayers is empty
    // unless quantize_model() ran, so it is only indexed when `quantized`.)
    const bool Q = model.quantized;
    auto LW = [&](std::size_t off, const QuantTensor* qt, int in, int out) -> LinearWeight {
        if (Q) return LinearWeight{nullptr, qt, in, out};
        return LinearWeight{model.at(off), nullptr, in, out};
    };

    for (int l = 0; l < c.n_layers; ++l) {
        const LayerWeights& L = model.w.layers[static_cast<std::size_t>(l)];
        const QuantLayerWeights* ql = Q ? &model.qlayers[static_cast<std::size_t>(l)] : nullptr;

        for (int i = 0; i < n_new; ++i)
            layernorm(x.data() + static_cast<std::size_t>(i) * d, model.at(L.ln_1_w),
                      model.at(L.ln_1_b), ln.data() + static_cast<std::size_t>(i) * d, d, c.ln_eps);

        const LinearWeight w_cattn = LW(L.c_attn_w, ql ? &ql->c_attn_w : nullptr, d, 3 * d);
        const LinearWeight w_cproj = LW(L.c_proj_w, ql ? &ql->c_proj_w : nullptr, d, d);
        attention_kv(ln.data(), w_cattn, model.at(L.c_attn_b), w_cproj, model.at(L.c_proj_b),
                     attn.data(), cache, l, n_new, n_past, d, c.n_heads);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += attn[i];

        for (int i = 0; i < n_new; ++i)
            layernorm(x.data() + static_cast<std::size_t>(i) * d, model.at(L.ln_2_w),
                      model.at(L.ln_2_b), ln.data() + static_cast<std::size_t>(i) * d, d, c.ln_eps);

        const LinearWeight w_fc = LW(L.mlp_fc_w, ql ? &ql->mlp_fc_w : nullptr, d, dm);
        const LinearWeight w_proj = LW(L.mlp_proj_w, ql ? &ql->mlp_proj_w : nullptr, dm, d);
        linear(ln.data(), w_fc, model.at(L.mlp_fc_b), mlp_h.data(), n_new);
        gelu_tanh(mlp_h.data(), static_cast<int>(mlp_h.size()));
        linear(mlp_h.data(), w_proj, model.at(L.mlp_proj_b), mlp.data(), n_new);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += mlp[i];
    }

    // The chunk is now cached; advance the fill level.
    cache.len = n_past + n_new;

    // ---- Final LayerNorm + tied LM head for the LAST position only. Greedy
    // decoding needs just the final token's logits, so -- unlike forward() -- we
    // skip the d x vocab head for every earlier position. wte is [vocab, d] =
    // [out, in], the tied head; it has no bias, and linear() guards the null. ----
    const float* x_last = x.data() + static_cast<std::size_t>(n_new - 1) * d;
    std::vector<float> lnf(static_cast<std::size_t>(d));
    layernorm(x_last, model.at(model.w.ln_f_w), model.at(model.w.ln_f_b), lnf.data(), d, c.ln_eps);

    std::vector<float> logits(static_cast<std::size_t>(V));
    const LinearWeight head =
        Q ? LinearWeight{nullptr, &model.q_head, d, V} : LinearWeight{wte, nullptr, d, V};
    linear(lnf.data(), head, nullptr, logits.data(), 1);
    return logits;
}

}  // namespace ie
