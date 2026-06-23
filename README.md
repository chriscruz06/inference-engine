# inference-engine

[![CI](https://github.com/chriscruz06/inference-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/chriscruz06/inference-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

`inference-engine` is a CPU language-model inference engine written in C++ with no machine-learning or math-library dependencies. It reads GPT-2 124M or TinyLlama 1.1B weights from disk, runs the full transformer forward pass directly, and generates text with greedy decoding and a KV cache. The standard library and, optionally, OpenMP are the only libraries linked. It provides a readable, framework-free implementation of transformer inference at the level of memory and arithmetic.

For the design rationale, the headline result, and the systems reasoning behind the numbers, see the project writeup in [`writeup.md`](writeup.md).

## Features

- Loads GPT-2 124M or TinyLlama 1.1B from a flat binary produced by the included Python exporter; the architecture is selected by the file's magic.
- Implements the full transformer forward pass by hand: token and learned position embeddings, pre-LayerNorm blocks, multi-head causal self-attention, a tanh-approximation GELU MLP, and a tied language-model head.
- Adds the Llama-2 architecture for the TinyLlama port: RMSNorm, rotary position embeddings (the rotate-half convention), grouped-query attention with a narrowed KV cache, a SwiGLU MLP, and an untied language-model head, all behind the same forward and KV-cached entry points selected by an architecture flag.
- Optimizes the single `linear` kernel that serves both compute regimes: an AVX2 path (8-wide FMA with four accumulators) parallelized across cores with OpenMP over output features, gated by work size so the bandwidth-bound decode GEMV stays single-threaded.
- Adds optional weight-only quantization (group-wise symmetric int8 and packed int4): the matmul weights are quantized in memory and the kernel dequantizes each group on the fly, shrinking the per-token weight stream that bounds decode.
- Provides a KV cache for incremental decoding, so per-token cost is independent of sequence length rather than quadratic.
- Encodes text to token ids for GPT-2 (byte-level BPE: pretokenization, the byte-to-unicode mapping, and the rank-ordered merge loop) and decodes ids back to text for both tokenizers, byte-level BPE (GPT-2) and SentencePiece (Llama). SentencePiece encoding is not implemented, so Llama prompts are supplied as token ids.
- Generates text with greedy (argmax) sampling and streams each token as it is produced.
- Includes a reference-diffing harness that dumps every intermediate activation and compares it against a HuggingFace reference.
- Links no PyTorch and no BLAS; uses only the C++17 standard library and optional OpenMP.
- Builds with CMake (with a Makefile wrapper), runs unit tests through CTest, and tests on GCC and Clang in CI.

## How it works

The forward pass repeats a single transformer block. Each block applies LayerNorm, multi-head causal self-attention, a second LayerNorm, and a two-layer MLP with a tanh-approximation GELU activation, with residual connections around the attention and MLP sublayers. GPT-2 uses learned absolute position embeddings, pre-LayerNorm placement, and a token embedding matrix tied to the output projection. A final LayerNorm precedes the language-model head.

The TinyLlama port swaps these GPT-2 specifics for the Llama-2 block: RMSNorm in place of LayerNorm, rotary position embeddings in place of learned ones, grouped-query attention (fewer key/value heads than query heads, which narrows the cache), a SwiGLU MLP, and an untied head. A dispatch on the loaded model's architecture selects the block; the `linear` kernel, the cache, and the verification harness are shared across both.

Two compute regimes appear during generation:

| Phase   | Operation                    | Limited by         |
|---------|------------------------------|--------------------|
| Prefill | matrix-matrix product (GEMM) | compute throughput |
| Decode  | matrix-vector product (GEMV) | memory bandwidth   |

Prefill processes the whole prompt at once. Decode processes one token per step. The KV cache stores each layer's keys and values as they are computed, so a decode step computes the query, key, and value for only the new token and attends over the cached history. The cached path (`forward_cached`) and the uncached path (`forward`) are checked for identical output.

A single `linear` kernel serves both regimes. It is vectorized with AVX2 (8-wide FMA, four independent accumulators to hide FMA latency) and parallelized with OpenMP over output features. A work-size gate keeps the parallel region off the small per-token GEMV of decode, where the bottleneck is memory bandwidth rather than cores.

## Benchmarks

Measured on GPT-2 124M (fp32), one socket, 8 physical cores, with the engine's `--bench` mode (median of several iterations after a warmup). Prefill and decode are reported separately because they are different problems: prefill is a compute-bound matrix-matrix product, decode a memory-bandwidth-bound matrix-vector product.

| Kernel                          | Prefill tok/s | Decode tok/s |
|---------------------------------|--------------:|-------------:|
| Naive scalar (`-O2`, 1 thread)  |           9.1 |          6.0 |
| AVX2 (8-wide FMA)               |          88.3 |         36.6 |
| AVX2 + OpenMP (work-size gated) |         224.7 |         36.8 |

The AVX2 step lifts both regimes (about 10x prefill, 6x decode). Multithreading then lifts prefill another ~2.5x but leaves decode flat: a decode step streams the entire weight set once per token and is capped by memory bandwidth, not cores, so the gate deliberately keeps it single-threaded. (Ungated, multithreading makes decode *slower* through fork/join and memory contention on one-row GEMVs.)

Cache-blocking the prefill GEMM over the token dimension was implemented and benchmarked, then dropped: it gave no improvement over the multithreaded kernel on this hardware, because the activation that would be tiled already fits in shared L3, so its re-reads are L3 hits rather than RAM traffic. The remaining prefill scaling gap is shared-L3 bandwidth and all-core clock throttling, which cache blocking cannot address. Decode's wall is the same memory-bandwidth limit, addressed in Phase 5 by quantization (fewer bytes moved per token).

```bash
./build/inference-engine --model models/gpt2-124m.bin \
    --bench --bench-prefill 256 --bench-decode 256 --bench-iters 3
```

Set `IE_PROFILE=1` on any run to break `linear` time down by matmul shape.

### Quantization (Phase 5)

Weight-only group-wise quantization shrinks the per-token weight stream that bounds decode. Same session, GPT-2 124M, prefill and decode 256:

| Weights         | Decode tok/s | Footprint | Token match vs fp32 |
|-----------------|-------------:|----------:|--------------------:|
| fp32            |         36.1 |        1x |                   - |
| int8 (group 64) |         46.6 |     3.76x |               95.8% |
| int4 (group 16) |         21.4 |     5.33x |               64.6% |

int8 lifts decode about 29% and is near-lossless (95.8% of greedy tokens match fp32, mean per-logit error 1.2). int4 shrinks the footprint further and stays coherent at a group of 16, but decode is *slower* than fp32 here: the nibble-unpack costs more ALU than an int8 load, and the finer group that int4 needs for coherence forces more per-group rescales than the bandwidth it saves on a model this small. On the larger TinyLlama port int4 becomes far more accurate (token-identical to fp32 even at a group of 64), but its decode is still not faster than fp32, so the int4 speed result holds at 1B (see below). The full accuracy and speed sweep (and the int4 negative result) is in [BENCH.md](BENCH.md).

```bash
./build/inference-engine --model models/gpt2-124m.bin --tokens 30 --quant q8        # generate (int8)
./build/inference-engine --model models/gpt2-124m.bin --compare-quant --quant q8    # accuracy vs fp32
```

### TinyLlama 1.1B (Phase 6)

The Llama port runs the same kernels on a model about nine times larger. Same session, prefill 128 and decode 64:

| Weights         | Prefill tok/s | Decode tok/s | Footprint | Token match vs fp32 |
|-----------------|--------------:|-------------:|----------:|--------------------:|
| fp32            |          43.6 |          5.4 |        1x |                   - |
| int8 (group 64) |          40.6 |          8.1 |     3.76x |                100% |
| int4 (group 64) |          29.5 |          5.1 |     7.11x |                100% |

int8 is again the decode win, and a larger one than on GPT-2: about +50% (5.4 to 8.1 tok/s), near-lossless, because a TinyLlama decode step streams roughly nine times the weight bytes and is more bandwidth-starved. int4 becomes dramatically more accurate at scale (token-identical to fp32 even at a group of 64, where GPT-2 needed a group of 16 and still diverged), giving a 7.11x footprint, but its decode is still not faster than fp32: the nibble-unpack ALU stays the wall even on a 1B model. So int8 is the all-around decode win and int4 is the footprint/accuracy choice. Full numbers in [BENCH.md](BENCH.md).

```bash
./build/inference-engine --model models/tinyllama-1.1b.bin --tokens 30 --quant q8
```

### Versus llama.cpp (the Phase 6 goal)

Tokens/sec against [`llama.cpp`](https://github.com/ggerganov/llama.cpp) on the same model and hardware. Same machine (AMD Ryzen 7 PRO 7840U, 8 physical cores), same TinyLlama 1.1B, both engines at 8 threads, prefill 128 / decode 64. `llama.cpp` (build 099b579, `GGML_NATIVE=ON`, so the same AVX2/FMA ISA as `-march=native`) is measured with its own `llama-bench`; the GGUF models are converted from the same checkpoint (F32, then `llama-quantize` to Q8_0/Q4_0). The quant schemes are not identical (llama.cpp uses 32-weight blocks and hand-tuned integer dot-product kernels; this engine uses group-64 dequant-on-the-fly), so the quant rows compare engines at the same bit width, not the same kernel.

| Quant (ours / llama) | Regime  | ours tok/s | llama.cpp tok/s | gap (llama / ours) |
|----------------------|---------|-----------:|----------------:|-------------------:|
| fp32 / F32           | prefill |       36.9 |           128.0 |              3.5x  |
| fp32 / F32           | decode  |        5.0 |             7.6 |          **1.5x**  |
| int8 / Q8_0          | prefill |       36.2 |           149.2 |              4.1x  |
| int8 / Q8_0          | decode  |        7.3 |            26.4 |              3.6x  |
| int4 / Q4_0          | prefill |       23.8 |           220.5 |              9.3x  |
| int4 / Q4_0          | decode  |        4.8 |            43.7 |              9.1x  |

The **fp32 row is the apples-to-apples comparison** (identical arithmetic, no quantization-kernel quality to confound it), and it lands in target: decode within **1.5x** and prefill within **3.5x** of llama.cpp, for an engine whose entire GEMM is one hand-written AVX2 kernel. Decode is the closer regime because it is memory-bandwidth-bound, so both engines stream the same fp32 weights per token and neither has much room to pull ahead; prefill is compute-bound, where llama.cpp's better register blocking and threading show through. The gap widens with quantization because quantized matmul is llama.cpp's specialty (years of hand-tuned integer-MAC kernels vs our per-group fp32 dequant); our int8 still beats our own fp32 decode and is correct, but closing the quant gap would mean writing integer-domain dot-product kernels, a different project than hand-writing the fp32 GEMM. The full per-regime analysis is in [BENCH.md](BENCH.md). (These are a separate benchmark session from the per-model sweeps above; cross-session variance is about 9% under all-core load.)

## Requirements

- A C++17 compiler and CMake 3.16 or newer. OpenMP is used if present.
- Python 3 with the packages in `requirements.txt` (`numpy`, `torch`, `transformers`, and `sentencepiece` for the Llama tokenizer) for weight export, reference dumps, and the diff scripts. The engine itself has no runtime dependencies.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

A Makefile wraps the common steps:

```bash
make        # configure and build
make test   # run the unit tests
make run    # build and print the CLI help
```

## Configuration

Build-time options, passed with `-D` (for example `cmake -S . -B build -DIE_SANITIZE=ON`):

| Option                  | Default | Effect                                   |
|-------------------------|---------|------------------------------------------|
| `IE_BUILD_TESTS`        | `ON`    | Build the unit tests                     |
| `IE_ENABLE_NATIVE`      | `ON`    | Compile with `-march=native` (AVX2/FMA)  |
| `IE_ENABLE_OPENMP`      | `ON`    | Link OpenMP if available                 |
| `IE_WARNINGS_AS_ERRORS` | `OFF`   | Treat compiler warnings as errors        |
| `IE_SANITIZE`           | `OFF`   | Enable Address and UB sanitizers         |

## Usage

The weights and tokenizer are not committed. Export them once with the Python tooling.

### Export the model and tokenizer

```bash
python -m pip install -r requirements.txt
python tools/export_weights.py   --model gpt2  --out models/gpt2-124m.bin
python tools/convert_tokenizer.py --from-hf gpt2 --out models/tokenizer.bin

# TinyLlama 1.1B (Llama-2 architecture, SentencePiece tokenizer)
python tools/export_weights.py    --arch llama --out models/tinyllama-1.1b.bin
python tools/convert_tokenizer.py --arch llama --out models/tokenizer-llama.bin
```

### Generate text

```bash
./build/inference-engine --model models/gpt2-124m.bin --prompt "The quick brown fox" --tokens 20
```

```
The quick brown foxes are a great way to get a little bit of a kick out of your dog.
```

GPT-2 prompts are encoded from text with `--prompt`. SentencePiece (Llama) encoding is not implemented, so TinyLlama prompts are supplied as token ids through `--input-ids`:

```bash
./build/inference-engine --model models/tinyllama-1.1b.bin \
    --input-ids tests/reference_dumps_llama/input_ids.npy --tokens 20
```

```
The quick brown fox jumps over the lazy dog.
```

The Llama prompt ids come from `tools/reference.py --arch llama`, which writes `tests/reference_dumps_llama/input_ids.npy`. Generation uses the KV cache by default and stops at the end-of-sequence token. The tokenizer matching the model is selected automatically.

### Command-line options

| Flag               | Default                                | Description                                                  |
|--------------------|----------------------------------------|--------------------------------------------------------------|
| `--model <path>`   | required                               | Model binary from `tools/export_weights.py`                  |
| `--tokens <N>`     | `64`                                   | Number of tokens to generate                                 |
| `--tokenizer <path>` | matches the model                    | Tokenizer binary used to encode the prompt (GPT-2) and decode output (GPT-2 or Llama, per model); falls back to raw ids |
| `--input-ids <path>` | `tests/reference_dumps/input_ids.npy` | Prompt token ids, used in place of `--prompt` (required for Llama, and the verification-harness path) |
| `--prompt <text>`  | `"The quick brown fox"`                | Prompt text. GPT-2: encoded with the byte-level BPE tokenizer. Llama: use `--input-ids` (SentencePiece encode is not implemented) |
| `--check-cache`    |                                        | Compare cached and uncached generation, then exit            |
| `--bench`          |                                        | Benchmark prefill and decode tok/s separately, then exit     |
| `--quant <mode>`   | `none`                                 | Weight-only quantization for the matmuls: `none`, `q8`, `q4` |
| `--quant-group <N>`| q8 `64`; q4 `16` (GPT-2) / `64` (Llama) | Weights per shared scale; finer trades footprint for accuracy |
| `--compare-quant`  |                                        | Report quantized-vs-fp32 token match, divergence, logit error, then exit |
| `-h`, `--help`     |                                        | Print usage and exit                                         |

In `--bench` mode, `--bench-prefill <N>`, `--bench-decode <N>`, and `--bench-iters <N>` set the prompt length, the number of generated tokens, and the count of timed iterations (the median is reported).

Setting the `IE_DUMP_DIR` environment variable runs a single forward pass that dumps every activation, instead of generating text. Setting `IE_PROFILE=1` reports a breakdown of `linear` time by matmul shape.

## Testing

Unit tests build with the project and run through CTest:

```bash
ctest --test-dir build --output-on-failure
```

| Test                  | Checks                                                       |
|-----------------------|-------------------------------------------------------------|
| `gemm_correctness`    | The naive matmul against a known result                     |
| `linear_equivalence`  | The AVX2 `linear` kernel against the scalar reference        |
| `quant_equivalence`   | The AVX2 int8/int4 kernels against their scalar references and a dequant cross-check |
| `loader_roundtrip`    | The binary loader and tokenizer decode on generated fixtures |
| `kv_cache_equivalence`| Cached and uncached generation produce identical tokens      |
| `encode_equivalence`  | The GPT-2 BPE encoder: decode/encode round-trips and merge/boundary behavior on a synthetic vocab |

`tests/make_fixtures.py` writes model and tokenizer fixtures in the on-disk format; `test_loader --fixture <dir>` then loads them, confirming the Python writers and the C++ readers agree on the byte layout. Likewise `tests/make_encode_fixtures.py` writes a golden set of (string, HuggingFace id) pairs, and `test_encode --tokenizer models/tokenizer.bin --fixtures tests/encode_fixtures.txt` checks the BPE encoder against them, the encode analogue of `check_argmax.py` (run manually, since it needs the exported tokenizer). `make format` runs clang-format. CI builds and tests on GCC and Clang, checks formatting with clang-format, and lints the Python tooling with ruff.

## Verification

The subtle failures in a project like this are not crashes but a single transposed weight or an off-by-one in the causal mask that makes the output almost correct. The repository includes a reference-diffing harness for catching these.

1. `tools/reference.py` runs HuggingFace GPT-2 on a fixed prompt and writes every intermediate tensor to `tests/reference_dumps/`.
2. Running the engine with `IE_DUMP_DIR` set writes the same tensors for the same input.
3. `tests/check_layers.py` reports the first tensor whose maximum absolute error exceeds a threshold.

Per-tensor error accumulates through the network because of fp32 rounding, so deep layers can exceed a tight threshold while still being correct. The decisive check is whether greedy decoding selects the same token. `tests/check_argmax.py` compares the argmax of the engine and reference logits at every position.

```bash
python tools/reference.py --model gpt2 --out-dir tests/reference_dumps
IE_DUMP_DIR=tests/engine_dumps ./build/inference-engine --model models/gpt2-124m.bin
python tests/check_layers.py --reference tests/reference_dumps --engine tests/engine_dumps
python tests/check_argmax.py --reference tests/reference_dumps --engine tests/engine_dumps
```

The KV cache has a separate check: cached generation must produce the same tokens as the uncached path. `--check-cache` runs both on the loaded model and compares them.

```bash
./build/inference-engine --model models/gpt2-124m.bin --tokens 20 --check-cache
```

The equivalent check on a small in-memory model runs as part of the test suite (`tests/test_kv_cache.cpp`).

## Project layout

```
inference-engine/
├── src/
│   ├── main.cpp             # CLI, generation loop, verification and cache-check modes
│   ├── model.{cpp,hpp}      # config and weight loading (GPT-2 and Llama formats)
│   ├── tokenizer.{cpp,hpp}  # byte-level BPE (encode + decode) and SentencePiece decode
│   ├── forward.{cpp,hpp}    # forward pass for GPT-2 and Llama, cached and uncached
│   ├── kv_cache.hpp         # per-layer key/value cache
│   ├── gemm.{cpp,hpp}       # linear/matmul kernels (scalar + AVX2 + OpenMP)
│   ├── profile.hpp          # zero-overhead RAII wall-clock profiler (IE_PROFILE)
│   ├── quant.{cpp,hpp}      # group-wise int8/int4 weight quantization
│   └── npy.hpp              # minimal .npy reader/writer for the harness
├── tools/
│   ├── export_weights.py    # HF weights to flat binary (GPT-2 or Llama)
│   ├── convert_tokenizer.py # HF vocab/merges or SentencePiece to tokenizer binary
│   ├── reference.py         # dump reference activations
│   └── format_spec.py       # shared on-disk format definitions
├── tests/
│   ├── test_gemm.cpp        # matmul unit test
│   ├── test_linear.cpp      # AVX2 vs scalar linear-kernel equivalence
│   ├── test_quant.cpp       # int8/int4 quantized-kernel equivalence
│   ├── test_loader.cpp      # loader and decode test
│   ├── test_kv_cache.cpp    # cached vs uncached equivalence
│   ├── test_encode.cpp      # BPE encoder: round-trips + HuggingFace parity
│   ├── make_fixtures.py     # cross-language loader fixtures
│   ├── make_encode_fixtures.py # golden (string, HF id) pairs for test_encode
│   ├── check_layers.py      # per-tensor diff against reference
│   └── check_argmax.py      # greedy-token match against reference
├── CMakeLists.txt
├── Makefile
├── BENCH.md
├── writeup.md
└── .github/workflows/ci.yml
```

## Roadmap

- [x] Build system, CI, naive GEMM, weight export
- [x] Phase 1: weight export and BPE decode
- [x] Phase 2: forward pass and greedy generation, verified against the reference
- [x] Phase 3: KV cache
- [x] Phase 4: GEMM optimization (AVX2 SIMD, multithreaded over output features)
- [x] Phase 5: weight-only int8 and int4 quantization (int8 the decode win; int4 tradeoffs documented)
- [x] Phase 6: TinyLlama 1.1B port (RMSNorm, RoPE, GQA, SwiGLU, untied head, SentencePiece decode, int8/int4), verified token-for-token against HuggingFace
- [x] Phase 6 (cont.): throughput comparison against llama.cpp on the same model and hardware (fp32 decode within 1.5x; full table in BENCH.md)
- [x] GPT-2 byte-level BPE encoder, so GPT-2 prompts can be given as text (`--prompt`)

## References

- Andrej Karpathy, [`llm.c`](https://github.com/karpathy/llm.c)
- [`llama.cpp`](https://github.com/ggerganov/llama.cpp)
- The GPT-2 and Llama papers for the architecture details

## License

MIT. See [LICENSE](LICENSE).