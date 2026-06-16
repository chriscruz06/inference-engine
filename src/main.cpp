// src/main.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "forward.hpp"
#include "model.hpp"
#include "npy.hpp"
#include "tokenizer.hpp"

namespace {

void print_usage(const char* prog) {
    std::printf(
        "inference-engine - a from-scratch LLM inference engine (no ML frameworks)\n"
        "\n"
        "Usage:\n"
        "  %s --model <path> [--prompt \"...\"] [--tokens N] [--input-ids <path>]\n"
        "\n"
        "Options:\n"
        "  --model      <path>   Path to a model exported by tools/export_weights.py\n"
        "  --prompt     <text>   Prompt to continue (default: a built-in sample)\n"
        "  --tokens     <N>      Number of tokens to generate (default: 64)\n"
        "  --input-ids  <path>   Phase 2 bringup: feed these reference token ids to the\n"
        "                        forward pass instead of tokenizing --prompt\n"
        "                        (default: tests/reference_dumps/input_ids.npy)\n"
        "  -h, --help            Show this help and exit\n",
        prog);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt = "The quick brown fox";
    std::string input_ids_path = "tests/reference_dumps/input_ids.npy";
    int n_tokens = 64;

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
        } else if (std::strcmp(arg, "--input-ids") == 0) {
            input_ids_path = take_value("--input-ids");
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
            "\nNo --model provided. The engine is scaffolded, but generation is not\n"
            "wired up yet. See the roadmap in README.md for what lands next.\n");
        return 0;
    }

    if (!ie::load_model(model_path, model)) {
        std::fprintf(stderr, "failed to load model from '%s'\n", model_path.c_str());
        return 1;
    }
    std::printf("[model] loaded '%s': %d layers, d_model=%d, vocab=%d\n", model_path.c_str(),
                model.config.n_layers, model.config.d_model, model.config.vocab_size);

    // --- Phase 2 bringup (TEMPORARY) -------------------------------------
    // encode() is still a stub, so the --prompt path can't tokenize yet. Feed
    // reference token ids straight to the forward pass instead, and let
    // forward() dump each activation to $IE_DUMP_DIR for tests/check_layers.py.
    // Once encode() + the generation loop land, this whole block becomes:
    //   tokenize(prompt) -> generation loop (forward + KV cache) -> decode.
    const std::vector<int> ids = npy::load_i32_1d(input_ids_path);
    if (ids.empty()) {
        std::fprintf(stderr, "[bringup] no input ids at '%s'; run tools/reference.py first\n",
                     input_ids_path.c_str());
        return 1;
    }
    std::printf("[bringup] %zu input tokens from '%s'\n", ids.size(), input_ids_path.c_str());

    const std::vector<float> logits = ie::forward(model, ids);
    (void)logits;    // last-position logits unused until greedy decoding is wired
    (void)prompt;    // activates once encode() works
    (void)n_tokens;  // activates once the generation loop lands
    // --------------------------------------------------------------------

    std::printf("[bringup] forward pass complete; generation loop not yet implemented\n");
    return 0;
}