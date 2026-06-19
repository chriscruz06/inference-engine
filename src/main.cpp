// src/main.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "forward.hpp"
#include "kv_cache.hpp"
#include "model.hpp"
#include "npy.hpp"
#include "tokenizer.hpp"

namespace {

void print_usage(const char* prog) {
    std::printf(
        "inference-engine - a from-scratch LLM inference engine (no ML frameworks)\n"
        "\n"
        "Usage:\n"
        "  %s --model <path> [--prompt \"...\"] [--tokens N] [--tokenizer <path>]\n"
        "     [--input-ids <path>] [--check-cache]\n"
        "\n"
        "Options:\n"
        "  --model      <path>   Path to a model exported by tools/export_weights.py\n"
        "  --prompt     <text>   Prompt to continue (ignored until encode() lands; the\n"
        "                        prompt currently comes from --input-ids)\n"
        "  --tokens     <N>      Number of tokens to generate (default: 64)\n"
        "  --tokenizer  <path>   tokenizer.bin from tools/convert_tokenizer.py, used to\n"
        "                        decode output to text (default: models/tokenizer.bin).\n"
        "                        If it can't be loaded, raw token ids are printed instead.\n"
        "  --input-ids  <path>   Token ids fed as the prompt -- encode() is still a stub, so\n"
        "                        the prompt is read from here for now\n"
        "                        (default: tests/reference_dumps/input_ids.npy)\n"
        "  --check-cache         Greedy-decode both with and without the KV cache and verify\n"
        "                        the token sequences are identical, then exit. The Phase-3\n"
        "                        correctness check, on the real model.\n"
        "  -h, --help            Show this help and exit\n"
        "\n"
        "Set IE_DUMP_DIR=<dir> to run a single forward pass that dumps every activation\n"
        "for tests/check_layers.py, instead of generating text.\n",
        prog);
}

// Greedy sampling: index of the largest logit.
int argmax(const std::vector<float>& v) {
    int best = 0;
    float best_val = v[0];
    for (int i = 1; i < static_cast<int>(v.size()); ++i) {
        if (v[i] > best_val) {
            best_val = v[i];
            best = i;
        }
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt = "The quick brown fox";
    std::string tokenizer_path = "models/tokenizer.bin";
    std::string input_ids_path = "tests/reference_dumps/input_ids.npy";
    int n_tokens = 64;
    bool check_cache = false;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        auto take_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires an argument\n", name);
                std::exit(2);
            }
            return argv[++i];
        };

        if (std::strcmp(arg, "--model") == 0) {
            model_path = take_value("--model");
        } else if (std::strcmp(arg, "--prompt") == 0) {
            prompt = take_value("--prompt");
        } else if (std::strcmp(arg, "--tokens") == 0) {
            n_tokens = std::atoi(take_value("--tokens"));
        } else if (std::strcmp(arg, "--tokenizer") == 0) {
            tokenizer_path = take_value("--tokenizer");
        } else if (std::strcmp(arg, "--input-ids") == 0) {
            input_ids_path = take_value("--input-ids");
        } else if (std::strcmp(arg, "--check-cache") == 0) {
            check_cache = true;
        } else if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n\n", arg);
            print_usage(argv[0]);
            return 2;
        }
    }

    ie::Model model;  // defaults to the GPT-2 124M config
    std::printf("inference-engine v0.1.0\n");
    std::printf("config: %d layers, %d heads, d_model=%d, vocab=%d\n", model.config.n_layers,
                model.config.n_heads, model.config.d_model, model.config.vocab_size);

    if (model_path.empty()) {
        std::printf(
            "\nNo --model provided. Pass --model models/gpt2-124m.bin (export it with\n"
            "tools/export_weights.py). See README.md.\n");
        return 0;
    }

    if (!ie::load_model(model_path, model)) {
        std::fprintf(stderr, "failed to load model from '%s'\n", model_path.c_str());
        return 1;
    }
    std::printf("[model] loaded '%s': %d layers, d_model=%d, vocab=%d\n", model_path.c_str(),
                model.config.n_layers, model.config.d_model, model.config.vocab_size);

    // Prompt tokens. encode() is still a stub, so the prompt is fed as reference
    // token ids from --input-ids rather than tokenized from --prompt. Once
    // encode() lands, this becomes `ids = tok.encode(prompt)`.
    std::vector<int> ids = npy::load_i32_1d(input_ids_path);
    if (ids.empty()) {
        std::fprintf(stderr, "[gen] no input ids at '%s'; run tools/reference.py first\n",
                     input_ids_path.c_str());
        return 1;
    }
    (void)prompt;  // unused until encode() is implemented

    const int n_ctx = model.config.n_ctx;
    if (static_cast<int>(ids.size()) >= n_ctx) {
        std::fprintf(stderr, "[gen] prompt (%zu tokens) is already at the context limit (%d)\n",
                     ids.size(), n_ctx);
        return 1;
    }

    // --- Verification mode (layer diff) ------------------------------------
    // With IE_DUMP_DIR set, run exactly ONE forward over the prompt ids so
    // forward() dumps every activation, then stop. This is the
    // tests/check_layers.py path; it uses the uncached forward (which dumps all
    // positions) and must stay a single forward so the dumps line up with the
    // fixed reference prompt.
    if (const char* dump_dir = std::getenv("IE_DUMP_DIR")) {
        std::printf("[verify] %zu prompt tokens from '%s'\n", ids.size(), input_ids_path.c_str());
        const std::vector<float> logits = ie::forward(model, ids);
        (void)logits;
        std::printf("[verify] forward pass complete; activations dumped to '%s'\n", dump_dir);
        std::printf("[verify] check with: python tests/check_layers.py\n");
        return 0;
    }

    // --- KV-cache equivalence check (Phase 3) ------------------------------
    // Greedy-decode the prompt both with the cache and with the old O(n^2) full
    // re-forward, and confirm the generated token ids are identical -- the
    // gameplan's Phase-3 "done when" on the real model. (The hermetic version
    // that CTest runs lives in tests/test_kv_cache.cpp.)
    if (check_cache) {
        const int prompt_n = static_cast<int>(ids.size());
        int budget = n_tokens;
        if (prompt_n + budget >= n_ctx) budget = n_ctx - prompt_n - 1;
        if (budget < 1) {
            std::fprintf(stderr, "[check-cache] prompt too long to compare any tokens\n");
            return 1;
        }
        std::printf("[check-cache] greedy-decoding %d tokens with and without the KV cache...\n",
                    budget);

        // Uncached: the Phase-2 path -- full forward over the growing sequence.
        std::vector<int> a = ids;
        for (int s = 0; s < budget; ++s) {
            const std::vector<float> lg = ie::forward(model, a);
            a.push_back(argmax(lg));
        }

        // Cached: prefill once, then one token per decode step.
        ie::KVCache cache;
        cache.init(model.config);
        std::vector<int> b = ids;
        std::vector<float> lg = ie::forward_cached(model, cache, ids, 0);
        for (int s = 0; s < budget; ++s) {
            const int next = argmax(lg);
            b.push_back(next);
            if (s + 1 < budget)
                lg = ie::forward_cached(model, cache, std::vector<int>{next}, cache.len);
        }

        int first_div = -1;
        for (int s = 0; s < budget; ++s) {
            if (a[static_cast<std::size_t>(prompt_n + s)] !=
                b[static_cast<std::size_t>(prompt_n + s)]) {
                first_div = s;
                break;
            }
        }
        if (first_div < 0) {
            std::printf("[check-cache] PASS: all %d generated tokens identical.\n", budget);
            return 0;
        }
        std::fprintf(stderr, "[check-cache] FAIL at generated token %d: uncached %d vs cached %d\n",
                     first_div, a[static_cast<std::size_t>(prompt_n + first_div)],
                     b[static_cast<std::size_t>(prompt_n + first_div)]);
        return 1;
    }

    // --- Generation mode ---------------------------------------------------
    // Greedy decoding with a KV cache: prefill the prompt once (filling the
    // cache), then each step take the argmax logit, append it, and decode just
    // that one new token. Per-token cost is now roughly flat in sequence length
    // instead of the old O(n^2) full re-forward.
    ie::Tokenizer tok;
    const bool have_tok = tok.load(tokenizer_path);
    if (!have_tok) {
        std::fprintf(stderr,
                     "[gen] couldn't load tokenizer from '%s'; printing raw token ids.\n"
                     "      build it with: python tools/convert_tokenizer.py --from-hf gpt2 "
                     "--out %s\n",
                     tokenizer_path.c_str(), tokenizer_path.c_str());
    }

    std::printf("[gen] generating up to %d tokens (KV cache enabled)...\n\n", n_tokens);

    // Echo the prompt, then stream each new token as it's produced.
    if (have_tok) {
        const std::string prompt_text = tok.decode(ids);
        std::fwrite(prompt_text.data(), 1, prompt_text.size(), stdout);
        std::fflush(stdout);
    }

    ie::KVCache cache;
    cache.init(model.config);

    // Prefill the whole prompt; the returned logits predict the first new token.
    std::vector<float> logits = ie::forward_cached(model, cache, ids, 0);

    for (int step = 0; step < n_tokens; ++step) {
        const int next = argmax(logits);
        ids.push_back(next);

        if (have_tok) {
            const std::string piece = tok.decode({next});
            std::fwrite(piece.data(), 1, piece.size(), stdout);
            std::fflush(stdout);
        } else {
            std::printf("%d ", next);
        }

        // Can't cache another position past the context window.
        if (cache.len >= n_ctx) {
            std::fprintf(stderr, "\n[gen] hit context limit (%d); stopping early.\n", n_ctx);
            break;
        }
        // Decode the token just emitted to get logits for the next one.
        if (step + 1 < n_tokens)
            logits = ie::forward_cached(model, cache, std::vector<int>{next}, cache.len);
    }
    std::printf("\n");
    return 0;
}
