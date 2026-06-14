// Loader + decode test for the Phase 1 plumbing.
//
// Two modes:
//   (no args)         self-contained: writes a tiny model.bin + tokenizer.bin
//                     (independent C++ writer), loads them, checks everything.
//                     Hermetic -- this is what CTest runs.
//   --fixture <dir>   loads <dir>/model.bin + <dir>/tokenizer.bin instead. Point
//                     it at the output of tests/make_fixtures.py to prove the
//                     Python writer and the C++ reader agree byte-for-byte.
//
// The tiny config and vocab below mirror tests/make_fixtures.py. Keep in sync.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "model.hpp"
#include "tokenizer.hpp"

namespace {

// --- mirror of tests/make_fixtures.py ---------------------------------------
constexpr int kLayers = 2, kHeads = 2, kDModel = 8, kHeadDim = 4;
constexpr int kNCtx = 16, kVocab = 32, kDMlp = 16;
constexpr float kLnEps = 1e-5f;
const char* kExpectedDecode = "Hi world!";

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "  FAIL: %s\n", what);
        ++g_failures;
    }
}

void put_i32(std::ofstream& f, int32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); }
void put_f32(std::ofstream& f, float v) { f.write(reinterpret_cast<const char*>(&v), 4); }
void put_str(std::ofstream& f, const std::string& s) {
    put_i32(f, static_cast<int32_t>(s.size()));
    f.write(s.data(), static_cast<std::streamsize>(s.size()));
}

// Compute the tiny model's total float count the same way the loader does,
// independently of the loader, so a size bug in either side is caught.
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
    for (std::size_t i = 0; i < total; ++i) put_f32(f, static_cast<float>(i));
}

void write_tiny_tokenizer(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    f.write("TOK1", 4);
    put_i32(f, 1);  // version
    put_i32(f, 4);  // n_vocab
    put_i32(f, 0);  // n_merges
    put_str(f, "H");
    put_str(f, "i");
    put_str(f, "\xC4\xA0world");  // 'Ġ' (U+0120) + "world" -> " world"
    put_str(f, "!");
}

void check_model(const ie::Model& m) {
    check(m.config.n_layers == kLayers, "n_layers");
    check(m.config.n_heads == kHeads, "n_heads");
    check(m.config.d_model == kDModel, "d_model");
    check(m.config.head_dim == kHeadDim, "head_dim");
    check(m.config.n_ctx == kNCtx, "n_ctx");
    check(m.config.vocab_size == kVocab, "vocab_size");
    check(m.config.d_mlp == kDMlp, "d_mlp");

    const std::size_t total = tiny_total_floats();
    check(m.weights.size() == total, "weight buffer size");

    // Offsets must point where the canonical order says.
    check(m.w.wte == 0, "wte offset");
    check(m.w.wpe == static_cast<std::size_t>(kVocab) * kDModel, "wpe offset");
    check(m.w.layers.size() == static_cast<std::size_t>(kLayers), "layer count");

    // Values were written as value == index; verify every float landed in
    // order (this catches any transposition, padding, or endianness bug).
    bool all_match = m.weights.size() == total;
    for (std::size_t i = 0; i < m.weights.size() && all_match; ++i)
        if (m.weights[i] != static_cast<float>(i)) all_match = false;
    check(all_match, "every weight float at correct offset (value == index)");

    // Spot-check addressing through the recorded offsets.
    check(m.at(m.w.wpe)[0] == static_cast<float>(kVocab * kDModel), "at(wpe)[0]");
    check(m.at(m.w.layers[0].c_attn_w) != nullptr, "layer0 c_attn_w pointer");
}

void check_tokenizer(const ie::Tokenizer& t) {
    check(t.loaded(), "tokenizer loaded");
    check(t.vocab_size() == 4, "tokenizer vocab size");
    const std::string got = t.decode({0, 1, 2, 3});
    const bool ok = (got == kExpectedDecode);
    check(ok, "decode([0,1,2,3]) == \"Hi world!\"");
    if (!ok) std::fprintf(stderr, "    got: \"%s\"\n", got.c_str());
    // Out-of-range ids are skipped, not crashes.
    check(t.decode({999, 0}) == "H", "decode skips out-of-range ids");
}

}  // namespace

int main(int argc, char** argv) {
    std::string fixture_dir;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--fixture") == 0 && i + 1 < argc) fixture_dir = argv[++i];

    std::string model_path, tok_path;
    if (!fixture_dir.empty()) {
        model_path = fixture_dir + "/model.bin";
        tok_path = fixture_dir + "/tokenizer.bin";
        std::printf("[test_loader] using fixtures in %s\n", fixture_dir.c_str());
    } else {
        model_path = "ie_test_model.bin";
        tok_path = "ie_test_tokenizer.bin";
        write_tiny_model(model_path);
        write_tiny_tokenizer(tok_path);
        std::printf("[test_loader] self-contained (wrote tiny fixtures to cwd)\n");
    }

    ie::Model model;
    if (!ie::load_model(model_path, model)) {
        std::fprintf(stderr, "  FAIL: load_model returned false\n");
        return 1;
    }
    check_model(model);

    ie::Tokenizer tok;
    if (!tok.load(tok_path)) {
        std::fprintf(stderr, "  FAIL: tokenizer.load returned false\n");
        return 1;
    }
    check_tokenizer(tok);

    if (g_failures == 0) {
        std::printf("test_loader: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_loader: %d check(s) failed\n", g_failures);
    return 1;
}
