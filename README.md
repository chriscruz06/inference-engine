# inference-engine

[![CI](https://github.com/chriscruz06/inference-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/chriscruz06/inference-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

`inference-engine` is a CPU language-model inference engine written in C++ with no machine-learning or math-library dependencies. It reads GPT-2 124M weights from disk, runs the full transformer forward pass directly, and generates text with greedy decoding and a KV cache. The standard library and, optionally, OpenMP are the only libraries linked. It provides a readable, framework-free implementation of transformer inference at the level of memory and arithmetic.

> Status: the weight loader, byte-level BPE decoding, the GPT-2 124M forward pass, and KV-cached greedy generation are implemented and tested. The forward pass matches a HuggingFace GPT-2 reference token-for-token on the reference prompt. BPE encoding, the SIMD and multithreaded GEMM kernels, quantization, and the Llama port are not yet implemented.

## Features

- Loads GPT-2 124M from a flat binary produced by the included Python exporter.
- Implements the full transformer forward pass by hand: token and learned position embeddings, pre-LayerNorm blocks, multi-head causal self-attention, a tanh-approximation GELU MLP, and a tied language-model head.
- Provides a KV cache for incremental decoding, so per-token cost is independent of sequence length rather than quadratic.
- Decodes byte-level BPE token ids to text using the GPT-2 byte-to-unicode mapping. Encoding is not yet implemented.
- Generates text with greedy (argmax) sampling and streams each token as it is produced.
- Includes a reference-diffing harness that dumps every intermediate activation and compares it against a HuggingFace reference.
- Links no PyTorch and no BLAS; uses only the C++17 standard library and optional OpenMP.
- Builds with CMake (with a Makefile wrapper), runs unit tests through CTest, and tests on GCC and Clang in CI.

## How it works

The forward pass repeats a single transformer block. Each block applies LayerNorm, multi-head causal self-attention, a second LayerNorm, and a two-layer MLP with a tanh-approximation GELU activation, with residual connections around the attention and MLP sublayers. GPT-2 uses learned absolute position embeddings, pre-LayerNorm placement, and a token embedding matrix tied to the output projection. A final LayerNorm precedes the language-model head.

Two compute regimes appear during generation:

| Phase   | Operation                    | Limited by         |
|---------|------------------------------|--------------------|
| Prefill | matrix-matrix product (GEMM) | compute throughput |
| Decode  | matrix-vector product (GEMV) | memory bandwidth   |

Prefill processes the whole prompt at once. Decode processes one token per step. The KV cache stores each layer's keys and values as they are computed, so a decode step computes the query, key, and value for only the new token and attends over the cached history. The cached path (`forward_cached`) and the uncached path (`forward`) are checked for identical output.

## Requirements

- A C++17 compiler and CMake 3.16 or newer. OpenMP is used if present.
- Python 3 with the packages in `requirements.txt` (`numpy`, `torch`, `transformers`) for weight export, reference dumps, and the diff scripts. The engine itself has no runtime dependencies.

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
```

### Generate text

```bash
./build/inference-engine --model models/gpt2-124m.bin --tokens 20
```

```
The quick brown foxes are a great way to get a little bit of a kick out of your dog.
```

The BPE encoder is not implemented, so the prompt is supplied as token ids through `--input-ids` (which defaults to the reference prompt) rather than through `--prompt`. Generation uses the KV cache by default.

### Command-line options

| Flag               | Default                                | Description                                                  |
|--------------------|----------------------------------------|--------------------------------------------------------------|
| `--model <path>`   | required                               | Model binary from `tools/export_weights.py`                  |
| `--tokens <N>`     | `64`                                   | Number of tokens to generate                                 |
| `--tokenizer <path>` | `models/tokenizer.bin`               | Tokenizer binary used to decode output; falls back to raw ids |
| `--input-ids <path>` | `tests/reference_dumps/input_ids.npy` | Prompt token ids, used in place of `--prompt`               |
| `--prompt <text>`  | `"The quick brown fox"`                | Currently ignored; `encode()` is not implemented             |
| `--check-cache`    |                                        | Compare cached and uncached generation, then exit            |
| `-h`, `--help`     |                                        | Print usage and exit                                         |

Setting the `IE_DUMP_DIR` environment variable runs a single forward pass that dumps every activation, instead of generating text.

## Testing

Unit tests build with the project and run through CTest:

```bash
ctest --test-dir build --output-on-failure
```

| Test                  | Checks                                                       |
|-----------------------|-------------------------------------------------------------|
| `gemm_correctness`    | The naive matmul against a known result                     |
| `loader_roundtrip`    | The binary loader and tokenizer decode on generated fixtures |
| `kv_cache_equivalence`| Cached and uncached generation produce identical tokens      |

`tests/make_fixtures.py` writes model and tokenizer fixtures in the on-disk format; `test_loader --fixture <dir>` then loads them, confirming the Python writers and the C++ readers agree on the byte layout. `make format` runs clang-format. CI builds and tests on GCC and Clang, checks formatting with clang-format, and lints the Python tooling with ruff.

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
│   ├── model.{cpp,hpp}      # config and weight loading
│   ├── tokenizer.{cpp,hpp}  # byte-level BPE (decode implemented, encode pending)
│   ├── forward.{cpp,hpp}    # forward pass, cached and uncached
│   ├── kv_cache.hpp         # per-layer key/value cache
│   ├── gemm.{cpp,hpp}       # matmul and linear kernels
│   ├── quant.{cpp,hpp}      # int8 quantization (scaffolding)
│   └── npy.hpp              # minimal .npy reader/writer for the harness
├── tools/
│   ├── export_weights.py    # HF weights to flat binary
│   ├── convert_tokenizer.py # HF vocab and merges to tokenizer binary
│   ├── reference.py         # dump reference activations
│   └── format_spec.py       # shared on-disk format definitions
├── tests/
│   ├── test_gemm.cpp        # matmul unit test
│   ├── test_loader.cpp      # loader and decode test
│   ├── test_kv_cache.cpp    # cached vs uncached equivalence
│   ├── make_fixtures.py     # cross-language fixtures
│   ├── check_layers.py      # per-tensor diff against reference
│   └── check_argmax.py      # greedy-token match against reference
├── CMakeLists.txt
├── Makefile
└── .github/workflows/ci.yml
```

## Roadmap

- [x] Build system, CI, naive GEMM, weight export
- [x] Phase 1: weight export and BPE decode
- [x] Phase 2: forward pass and greedy generation, verified against the reference
- [x] Phase 3: KV cache
- [ ] Phase 4: GEMM optimization (AVX2, multithreading, cache tiling)
- [ ] Phase 5: int8 and int4 quantization
- [ ] Phase 6: port to a ~1B Llama model (RoPE, SwiGLU, GQA, RMSNorm) and benchmark against llama.cpp

## References

- Andrej Karpathy, [`llm.c`](https://github.com/karpathy/llm.c)
- [`llama.cpp`](https://github.com/ggerganov/llama.cpp)
- The GPT-2 and Llama papers for the architecture details

## License

MIT. See [LICENSE](LICENSE).