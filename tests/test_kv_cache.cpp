// KV-cache equivalence test (Phase 3).
//
// Proves the incremental, cache-backed decode path (forward_cached) computes the
// same thing as the O(n^2) full re-forward (forward) it replaces: greedy
// decoding through both must pick the SAME token at every step. That is the
// gameplan's Phase-3 correctness gate -- "cached and uncached outputs are
// token-identical." It holds for ANY weights (it's a property of two code paths
// computing the same function, by causality: appending a token never changes an
// earlier position's activations), so this test is hermetic: it builds a tiny
// model with small deterministic weights in memory -- no real GPT-2, no Python.
// CTest runs it; see CMakeLists.txt.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "forward.hpp"
#include "kv_cache.hpp"
#include "model.hpp"

namespace {

constexpr int kLayers = 3, kHeads = 2, kDModel = 8, kHeadDim = 4;
constexpr int kNCtx = 32, kVocab = 32, kDMlp = 16;
constexpr float kLnEps = 1e-5f;

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "  FAIL: %s\n", what);
        ++g_failures;
    }
}

void put_i32(std::ofstream& f, std::int32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); }
void put_f32(std::ofstream& f, float v) { f.write(reinterpret_cast<const char*>(&v), 4); }

// Small, deterministic, finite weights -- enough to drive a real forward pass
// without overflowing (value==index, as in test_loader, would blow the matmuls
// up to Inf). A cheap integer hash keeps it stable across runs and platforms.
// Quality is irrelevant here; only path equivalence is being tested.
float tiny_weight(std::size_t i) {
    std::uint32_t x = static_cast<std::uint32_t>(i) * 2654435761u + 1013904223u;
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    return (static_cast<float>(x & 0xFFFFu) / 65536.0f - 0.5f) * 0.16f;  // in [-0.08, 0.08)
}

std::size_t tiny_total_floats() {
    const std::size_t D = kDModel, M = kDMlp, V = kVocab, C = kNCtx;
    std::size_t t = V * D + C * D;
    for (int l = 0; l < kLayers; ++l)
        t += 2 * D + 3 * D * D + 3 * D + D * D + D + 2 * D + M * D + M + D * M + D;
    t += 2 * D;
    return t;
}

void write_tiny_model(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    f.write("GPT2", 4);
    put_i32(f, 1);  // version
    put_i32(f, kLayers);
    put_i32(f, kHeads);
    put_i32(f, kDModel);
    put_i32(f, kHeadDim);
    put_i32(f, kNCtx);
    put_i32(f, kVocab);
    put_i32(f, kDMlp);
    put_f32(f, kLnEps);
    const std::size_t total = tiny_total_floats();
    for (std::size_t i = 0; i < total; ++i) put_f32(f, tiny_weight(i));
}

int argmax(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace

int main() {
    const std::string model_path = "ie_test_kv_model.bin";
    write_tiny_model(model_path);

    ie::Model model;
    if (!ie::load_model(model_path, model)) {
        std::fprintf(stderr, "  FAIL: load_model returned false\n");
        return 1;
    }

    const std::vector<int> prompt = {1, 5, 9, 2, 7};
    const int n_gen = 12;

    // --- Uncached: the Phase-2 path. Full forward over the growing sequence. ---
    std::vector<int> seq_uncached = prompt;
    std::vector<std::vector<float>> logits_uncached;
    for (int step = 0; step < n_gen; ++step) {
        std::vector<float> lg = ie::forward(model, seq_uncached);
        logits_uncached.push_back(lg);
        seq_uncached.push_back(argmax(lg));
    }

    // --- Cached: prefill once, then decode one token at a time. ---
    ie::KVCache cache;
    cache.init(model.config);
    std::vector<int> seq_cached = prompt;
    std::vector<std::vector<float>> logits_cached;
    std::vector<float> lg = ie::forward_cached(model, cache, prompt, 0);
    for (int step = 0; step < n_gen; ++step) {
        logits_cached.push_back(lg);
        const int next = argmax(lg);
        seq_cached.push_back(next);
        if (step + 1 < n_gen)
            lg = ie::forward_cached(model, cache, std::vector<int>{next}, cache.len);
    }

    // Cache bookkeeping: prefill (prompt) + (n_gen - 1) single-token decodes.
    check(cache.len == static_cast<int>(prompt.size()) + n_gen - 1, "cache length after decode");

    // The headline gate: identical generated tokens.
    const bool tokens_match = (seq_uncached == seq_cached);
    check(tokens_match, "cached and uncached generate identical tokens");
    if (!tokens_match) {
        std::fprintf(stderr, "    uncached:");
        for (int t : seq_uncached) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n    cached:  ");
        for (int t : seq_cached) std::fprintf(stderr, " %d", t);
        std::fprintf(stderr, "\n");
    }

    // And the logits themselves should agree to floating-point noise every step.
    float worst = 0.0f;
    for (int step = 0; step < n_gen; ++step)
        worst = std::max(worst, max_abs_diff(logits_uncached[step], logits_cached[step]));
    std::printf("[test_kv_cache] worst per-step logit diff: %.3e\n", static_cast<double>(worst));
    check(worst < 1e-4f, "per-step logits match within 1e-4");

    if (g_failures == 0) {
        std::printf("test_kv_cache: OK (%d steps, prompt %zu tokens)\n", n_gen, prompt.size());
        return 0;
    }
    std::fprintf(stderr, "test_kv_cache: %d check(s) failed\n", g_failures);
    return 1;
}
