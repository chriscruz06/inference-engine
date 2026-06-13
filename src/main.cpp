#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "forward.hpp"
#include "model.hpp"
#include "tokenizer.hpp"

namespace {

void print_usage(const char* prog) {
    std::printf(
        "inference-engine - a from-scratch LLM inference engine (no ML frameworks)\n"
        "\n"
        "Usage:\n"
        "  %s --model <path> [--prompt \"...\"] [--tokens N]\n"
        "\n"
        "Options:\n"
        "  --model  <path>   Path to a model exported by tools/export_weights.py\n"
        "  --prompt <text>   Prompt to continue (default: a built-in sample)\n"
        "  --tokens <N>      Number of tokens to generate (default: 64)\n"
        "  -h, --help        Show this help and exit\n",
        prog);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt = "The quick brown fox";
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

    // TODO: tokenize prompt -> generation loop (forward + KV cache) -> decode.
    (void)prompt;
    (void)n_tokens;
    std::printf("generation loop not yet implemented\n");
    return 0;
}
