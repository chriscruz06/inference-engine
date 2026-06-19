// src/kv_cache.hpp
#pragma once

#include <cstddef>
#include <vector>

#include "model.hpp"

namespace ie {

// Per-layer key/value cache for incremental (autoregressive) decoding.
//
// Without it, generating each token re-runs the full forward over the whole
// growing sequence -- O(n^2) work, plus an LM head over every position when only
// the last is needed. With it, prefill computes K/V for every prompt position
// once; each later decode step computes Q/K/V for the single new token, appends
// its K/V here, and attends over everything cached so far. That is the
// unusable -> usable jump (gameplan Phase 3 / Week 4).
//
// Layout: K and V are each one flat fp32 buffer of [n_layers, n_ctx, d], so the
// K (or V) vector for layer l at position p is a contiguous d-float row at
// k_row(l, p) / v_row(l, p) -- the same [.., d] layout the fused QKV projection
// already produces, with head h occupying columns [h*head_dim, (h+1)*head_dim).
//
// Buffers are sized to the full context up front, so the row pointers handed to
// attention never move as the cache fills. For GPT-2 124M that is
// 2 * 12 * 1024 * 768 * 4 B ~= 75 MB; grouped-query attention shrinks it for the
// Llama port (fewer KV heads => narrower rows).
struct KVCache {
    int n_layers = 0;
    int n_ctx = 0;
    int d = 0;
    int len = 0;  // positions currently filled, in [0, n_ctx]

    std::vector<float> k;  // [n_layers, n_ctx, d]
    std::vector<float> v;  // [n_layers, n_ctx, d]

    // Allocate for a model config and reset to empty.
    void init(const ModelConfig& c) {
        n_layers = c.n_layers;
        n_ctx = c.n_ctx;
        d = c.d_model;
        len = 0;
        const std::size_t n = static_cast<std::size_t>(n_layers) *
                              static_cast<std::size_t>(n_ctx) * static_cast<std::size_t>(d);
        k.assign(n, 0.0f);
        v.assign(n, 0.0f);
    }

    // Reuse the allocation for a fresh sequence without freeing.
    void clear() { len = 0; }

    std::size_t layer_stride() const {
        return static_cast<std::size_t>(n_ctx) * static_cast<std::size_t>(d);
    }
    float* k_row(int layer, int pos) {
        return k.data() + static_cast<std::size_t>(layer) * layer_stride() +
               static_cast<std::size_t>(pos) * static_cast<std::size_t>(d);
    }
    float* v_row(int layer, int pos) {
        return v.data() + static_cast<std::size_t>(layer) * layer_stride() +
               static_cast<std::size_t>(pos) * static_cast<std::size_t>(d);
    }
    const float* k_row(int layer, int pos) const {
        return k.data() + static_cast<std::size_t>(layer) * layer_stride() +
               static_cast<std::size_t>(pos) * static_cast<std::size_t>(d);
    }
    const float* v_row(int layer, int pos) const {
        return v.data() + static_cast<std::size_t>(layer) * layer_stride() +
               static_cast<std::size_t>(pos) * static_cast<std::size_t>(d);
    }
};

}  // namespace ie
