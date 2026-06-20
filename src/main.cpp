// src/main.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "forward.hpp"
#include "kv_cache.hpp"
#include "model.hpp"
#include "npy.hpp"
#include "profile.hpp"
#include "tokenizer.hpp"

namespace {

void print_usage(const char* prog) {
    std::printf(
        "inference-engine - a from-scratch LLM inference engine (no ML frameworks)\n"
        "\n"
        "Usage:\n"
        "  %s --model <path> [--prompt \"...\"] [--tokens N] [--tokenizer <path>]\n"
        "     [--input-ids <path>] [--check-cache] [--bench] [--quant q8|q4]\n"
        "     [--compare-quant]\n"
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
        "  --bench               Benchmark prefill (GEMM) and decode (GEMV) tokens/sec\n"
        "                        separately on synthetic input, then exit. The Phase-4\n"
        "                        baseline that the kernel work is measured against.\n"
        "  --bench-prefill <N>   Prefill length for --bench (default: 128)\n"
        "  --bench-decode  <N>   Decode steps for --bench (default: 64)\n"
        "  --bench-iters   <N>   Timed iterations for --bench; the median is reported\n"
        "                        (default: 5, plus one warmup)\n"
        "  --quant      <mode>   Weight-only quantization for the streamed matmuls:\n"
        "                        none (default), q8 (int8), or q4 (packed int4).\n"
        "                        Shrinks decode's per-token weight stream; applies to\n"
        "                        generation and --bench.\n"
        "  --compare-quant       Greedy-decode --tokens steps fp32, then teacher-force\n"
        "                        the quantized model on the same context and report the\n"
        "                        token-match rate, first divergence, and mean logit\n"
        "                        error (uses --quant's type; q8 if unset). Then exit.\n"
        "  --quant-group <N>     Weights per shared scale (default: q8 64, q4 16). A\n"
        "                        finer group trades footprint for accuracy.\n"
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

// Median of a list of timings (sorts a copy).
double median(std::vector<double> xs) {
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    const std::size_t n = xs.size();
    return (n % 2 == 1) ? xs[n / 2] : 0.5 * (xs[n / 2 - 1] + xs[n / 2]);
}

// Benchmark the two compute regimes separately:
//   prefill -> one forward_cached over the whole prompt   (matrix x matrix, GEMM)
//   decode  -> N single-token forward_cached steps        (matrix x vector, GEMV)
// Token VALUES don't affect timing (only the matmul shapes do), so a synthetic
// deterministic prompt stands in for real text -- no tokenizer or input-ids file
// is needed. Every per-step cost the engine pays today (including the per-call
// temporary allocations inside forward_cached) is included, so this is the
// honest baseline the Phase-4 optimizations are measured against.
int run_bench(const ie::Model& model, int prefill_len, int decode_steps, int iters) {
    using clock = std::chrono::steady_clock;
    const int n_ctx = model.config.n_ctx;
    const int V = model.config.vocab_size;

    if (prefill_len < 1) prefill_len = 1;
    if (iters < 1) iters = 1;
    if (decode_steps < 0) decode_steps = 0;
    // Keep the whole run inside the context window.
    if (prefill_len >= n_ctx) prefill_len = n_ctx - 1;
    if (prefill_len + decode_steps >= n_ctx) decode_steps = n_ctx - prefill_len - 1;
    if (decode_steps < 0) decode_steps = 0;

    // Deterministic, valid token ids. Values are irrelevant to timing.
    std::vector<int> prompt(static_cast<std::size_t>(prefill_len));
    for (int i = 0; i < prefill_len; ++i) {
        const unsigned h = static_cast<unsigned>(i) * 2654435761u + 12345u;
        prompt[static_cast<std::size_t>(i)] = static_cast<int>(h % static_cast<unsigned>(V));
    }

    std::printf("[bench] %d layers, d_model=%d, d_mlp=%d, vocab=%d, n_ctx=%d\n",
                model.config.n_layers, model.config.d_model, model.config.d_mlp, V, n_ctx);
    std::printf("[bench] prefill=%d  decode=%d  iters=%d (+1 warmup)\n", prefill_len, decode_steps,
                iters);

    // Per-shape matmul profiling, on only when IE_PROFILE is set. Prefill and
    // decode get separate breakdowns because their matmul mix differs sharply:
    // in prefill every position flows through the projections (big GEMM) but the
    // LM head runs once; in decode each projection sees one row yet the full
    // 768x50257 head still runs every step, so it dominates a decode step.
    const bool profile = std::getenv("IE_PROFILE") != nullptr;

    ie::KVCache cache;
    cache.init(model.config);

    double checksum = 0.0;  // accumulated so the work can't be dead-code-eliminated

    // ---- Prefill: fresh cache, one forward_cached over the whole prompt. ----
    {
        std::vector<double> secs;
        secs.reserve(static_cast<std::size_t>(iters));
        for (int it = 0; it < iters + 1; ++it) {  // iteration 0 is warmup
            cache.clear();
            ie::prof::registry().enabled = profile;
            if (it == 1) ie::prof::registry().reset();  // drop warmup from the buckets
            const auto t0 = clock::now();
            const std::vector<float> lg = ie::forward_cached(model, cache, prompt, 0);
            const auto t1 = clock::now();
            checksum += static_cast<double>(lg.empty() ? 0.0f : lg[0]);
            if (it > 0) secs.push_back(std::chrono::duration<double>(t1 - t0).count());
        }
        const double med = median(secs);
        const double tps = med > 0.0 ? prefill_len / med : 0.0;
        std::printf("[bench] prefill: %9.3f ms median  ->  %9.1f tok/s\n", med * 1e3, tps);
        if (profile) ie::prof::registry().report("prefill");
    }

    // ---- Decode: prefill (untimed) then time `decode_steps` single tokens. ----
    if (decode_steps > 0) {
        std::vector<double> secs;
        secs.reserve(static_cast<std::size_t>(iters));
        for (int it = 0; it < iters + 1; ++it) {  // iteration 0 is warmup
            cache.clear();
            ie::prof::registry().enabled = false;  // don't profile the setup prefill
            std::vector<float> lg = ie::forward_cached(model, cache, prompt, 0);
            int next = argmax(lg);
            if (it == 1) ie::prof::registry().reset();  // drop warmup's decode buckets
            ie::prof::registry().enabled = profile;     // profile only the decode steps
            const auto t0 = clock::now();
            for (int s = 0; s < decode_steps; ++s) {
                lg = ie::forward_cached(model, cache, std::vector<int>{next}, cache.len);
                next = argmax(lg);
            }
            const auto t1 = clock::now();
            ie::prof::registry().enabled = false;
            checksum += static_cast<double>(next);
            if (it > 0) secs.push_back(std::chrono::duration<double>(t1 - t0).count());
        }
        const double med = median(secs);
        const double tps = med > 0.0 ? decode_steps / med : 0.0;
        const double ms_per = med * 1e3 / decode_steps;
        std::printf("[bench] decode:  %9.3f ms median  ->  %9.1f tok/s  (%.3f ms/token)\n",
                    med * 1e3, tps, ms_per);
        if (profile) ie::prof::registry().report("decode");
    }

    ie::prof::registry().enabled = false;
    std::printf("[bench] (checksum %.6f -- ignore; defeats dead-code elimination)\n", checksum);
    return 0;
}

// Quantization accuracy probe. Greedy-decode `n` tokens with the fp32 model
// (capturing the per-step logits), quantize the model in place, then teacher-force
// the SAME context (prompt + the fp32-chosen tokens) through the quantized model
// so both see identical inputs at every step. This isolates the quantization
// perturbation from autoregressive drift and yields the end-to-end accuracy
// numbers (token-match rate, first divergence, mean per-logit abs error). Mutates
// `model` (it ends quantized); callers run it last.
int run_compare_quant(ie::Model& model, const std::vector<int>& ids, int n, ie::QuantType type,
                      int group) {
    const int V = model.config.vocab_size;
    const int n_ctx = model.config.n_ctx;
    if (n < 1) n = 1;
    if (static_cast<int>(ids.size()) + n >= n_ctx) n = n_ctx - static_cast<int>(ids.size()) - 1;
    if (n < 1) {
        std::fprintf(stderr, "[compare-quant] prompt too long to compare any tokens\n");
        return 1;
    }
    const char* name = (type == ie::QuantType::Q8) ? "q8" : "q4";

    ie::KVCache cache;
    cache.init(model.config);

    // ---- fp32 reference pass (model not yet quantized). ----
    std::vector<int> gen;
    gen.reserve(static_cast<std::size_t>(n));
    std::vector<std::vector<float>> logits_fp32;
    logits_fp32.reserve(static_cast<std::size_t>(n));
    {
        std::vector<float> lg = ie::forward_cached(model, cache, ids, 0);
        for (int s = 0; s < n; ++s) {
            logits_fp32.push_back(lg);
            const int t = argmax(lg);
            gen.push_back(t);
            if (s + 1 < n) lg = ie::forward_cached(model, cache, std::vector<int>{t}, cache.len);
        }
    }

    // ---- Quantize, then teacher-force the fp32 token stream. ----
    std::size_t f32_bytes = 0;
    const std::size_t q_bytes = ie::quantize_model(model, type, group, &f32_bytes);

    int match = 0, first_div = -1;
    double abs_err_sum = 0.0;
    std::size_t logit_count = 0;
    {
        cache.clear();
        std::vector<float> lg = ie::forward_cached(model, cache, ids, 0);
        for (int s = 0; s < n; ++s) {
            if (argmax(lg) == gen[static_cast<std::size_t>(s)])
                ++match;
            else if (first_div < 0)
                first_div = s;
            const std::vector<float>& lf = logits_fp32[static_cast<std::size_t>(s)];
            for (int v = 0; v < V; ++v)
                abs_err_sum += std::fabs(static_cast<double>(lg[static_cast<std::size_t>(v)]) -
                                         static_cast<double>(lf[static_cast<std::size_t>(v)]));
            logit_count += static_cast<std::size_t>(V);
            if (s + 1 < n)
                lg = ie::forward_cached(model, cache,
                                        std::vector<int>{gen[static_cast<std::size_t>(s)]},
                                        cache.len);
        }
    }

    const double match_rate = 100.0 * match / n;
    const double mean_logit_err = logit_count ? abs_err_sum / static_cast<double>(logit_count) : 0.0;
    const double ratio = q_bytes > 0 ? static_cast<double>(f32_bytes) / static_cast<double>(q_bytes)
                                     : 0.0;
    std::printf("[compare-quant] %s (group %d) vs fp32 over %d teacher-forced steps:\n", name, group,
                n);
    std::printf("[compare-quant]   token match:        %d/%d (%.1f%%)\n", match, n, match_rate);
    if (first_div < 0)
        std::printf("[compare-quant]   first divergence:   none\n");
    else
        std::printf("[compare-quant]   first divergence:   step %d\n", first_div);
    std::printf("[compare-quant]   mean per-logit error: %.4f\n", mean_logit_err);
    std::printf("[compare-quant]   footprint:          %.1f MB quant vs %.1f MB fp32 (%.2fx)\n",
                static_cast<double>(q_bytes) / 1e6, static_cast<double>(f32_bytes) / 1e6, ratio);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string prompt = "The quick brown fox";
    std::string tokenizer_path = "models/tokenizer.bin";
    std::string input_ids_path = "tests/reference_dumps/input_ids.npy";
    int n_tokens = 64;
    bool check_cache = false;
    bool bench = false;
    int bench_prefill = 128;
    int bench_decode = 64;
    int bench_iters = 5;
    std::string quant_mode = "none";  // none | q8 | q4
    bool compare_quant = false;
    int quant_group = 0;  // 0 = auto (q8: 64, q4: 32); >0 overrides

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
        } else if (std::strcmp(arg, "--bench") == 0) {
            bench = true;
        } else if (std::strcmp(arg, "--bench-prefill") == 0) {
            bench_prefill = std::atoi(take_value("--bench-prefill"));
        } else if (std::strcmp(arg, "--bench-decode") == 0) {
            bench_decode = std::atoi(take_value("--bench-decode"));
        } else if (std::strcmp(arg, "--bench-iters") == 0) {
            bench_iters = std::atoi(take_value("--bench-iters"));
        } else if (std::strcmp(arg, "--quant") == 0) {
            quant_mode = take_value("--quant");
        } else if (std::strcmp(arg, "--compare-quant") == 0) {
            compare_quant = true;
        } else if (std::strcmp(arg, "--quant-group") == 0) {
            quant_group = std::atoi(take_value("--quant-group"));
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

    // --- Quantization (Phase 5) --------------------------------------------
    // Resolve --quant and (unless we are comparing, which quantizes itself) build
    // the int8/int4 weights in memory so --bench and generation use the quantized
    // matmul kernels. The fp32 buffer is kept (biases, LayerNorm, embedding lookup
    // still read it); forward() (the dump oracle) is unaffected and stays fp32.
    ie::QuantType qtype = ie::QuantType::Q8;
    bool want_quant = false;
    if (quant_mode == "q8") {
        want_quant = true;
        qtype = ie::QuantType::Q8;
    } else if (quant_mode == "q4") {
        want_quant = true;
        qtype = ie::QuantType::Q4;
    } else if (quant_mode != "none") {
        std::fprintf(stderr, "error: --quant must be one of none|q8|q4 (got '%s')\n",
                     quant_mode.c_str());
        return 2;
    }
    // Group size: explicit --quant-group wins; otherwise int4 uses a much finer
    // group (16) than int8 (64). int4's 16 code levels are coarse enough that on a
    // small, quantization-sensitive model like GPT-2 124M, groups of 32/64
    // collapse into repetition; group 16 is the coherence floor here (measured via
    // --compare-quant, logged in BENCH.md).
    const int group = quant_group > 0 ? quant_group
                                      : (qtype == ie::QuantType::Q4 ? 16 : ie::kQuantGroup);
    if (want_quant && !compare_quant) {
        std::size_t f32_bytes = 0;
        const std::size_t q_bytes = ie::quantize_model(model, qtype, group, &f32_bytes);
        const double ratio = q_bytes > 0 ? static_cast<double>(f32_bytes) /
                                               static_cast<double>(q_bytes)
                                         : 0.0;
        std::printf(
            "[quant] %s: matmul weights %.1f MB quantized vs %.1f MB fp32 (%.2fx smaller)\n",
            quant_mode.c_str(), static_cast<double>(q_bytes) / 1e6,
            static_cast<double>(f32_bytes) / 1e6, ratio);
    }

    // --- Benchmark mode ----------------------------------------------------
    // Synthetic and self-contained (no tokenizer or input-ids file needed): time
    // prefill and decode separately so the Phase-4 kernel work has a baseline to
    // beat. Runs before the prompt is loaded so --bench works with just --model.
    if (bench) {
        return run_bench(model, bench_prefill, bench_decode, bench_iters);
    }

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

    // --- Quantization accuracy comparison (Phase 5) ------------------------
    if (compare_quant) {
        std::printf("[compare-quant] %zu prompt tokens; comparing %s against fp32...\n", ids.size(),
                    qtype == ie::QuantType::Q8 ? "q8" : "q4");
        return run_compare_quant(model, ids, n_tokens, qtype, group);
    }

    // --- Verification mode (layer diff) ------------------------------------
    if (const char* dump_dir = std::getenv("IE_DUMP_DIR")) {
        std::printf("[verify] %zu prompt tokens from '%s'\n", ids.size(), input_ids_path.c_str());
        const std::vector<float> logits = ie::forward(model, ids);
        (void)logits;
        std::printf("[verify] forward pass complete; activations dumped to '%s'\n", dump_dir);
        std::printf("[verify] check with: python tests/check_layers.py\n");
        return 0;
    }

    // --- KV-cache equivalence check (Phase 3) ------------------------------
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

        std::vector<int> a = ids;
        for (int s = 0; s < budget; ++s) {
            const std::vector<float> lg = ie::forward(model, a);
            a.push_back(argmax(lg));
        }

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

    if (have_tok) {
        const std::string prompt_text = tok.decode(ids);
        std::fwrite(prompt_text.data(), 1, prompt_text.size(), stdout);
        std::fflush(stdout);
    }

    ie::KVCache cache;
    cache.init(model.config);

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

        if (cache.len >= n_ctx) {
            std::fprintf(stderr, "\n[gen] hit context limit (%d); stopping early.\n", n_ctx);
            break;
        }
        if (step + 1 < n_tokens)
            logits = ie::forward_cached(model, cache, std::vector<int>{next}, cache.len);
    }
    std::printf("\n");
    return 0;
}