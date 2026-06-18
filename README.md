# inference-engine

[![CI](https://github.com/OWNER/inference-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/inference-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

A from-scratch LLM inference engine in C++. It loads a real model's weights,
runs the entire transformer forward pass by hand, and is optimized to compete
with [`llama.cpp`](https://github.com/ggerganov/llama.cpp) on CPU: **no PyTorch,
no BLAS, no ML or math frameworks**. The standard library and OpenMP are the only
things linked.

> **Status: generating text.** The full GPT-2 124M forward pass runs by hand and
> produces coherent English, verified token-for-token against HuggingFace (see
> [Verification](#verification-the-actual-hard-part)). The weight loader, the
> byte-level BPE `decode`, and the naive GEMM are in place and tested. Next up is
> the KV cache and the performance work — see the [roadmap](#roadmap).

## Why

Single-token LLM generation is, under the hood, a fixed sequence of matrix
multiplications plus a handful of other operations. Writing that out by hand,
and then making it fast, sits at the intersection of low-level systems work and
machine learning. This project is that exercise end to end: correctness first,
then the performance work (SIMD, cache blocking, multithreading, quantization)
that actually earns a benchmark.

## How it works

The forward pass is the path an input takes through the network to predict the
next token. For a GPT-style model it repeats one block many times:

- **Attention**: lets every token look at every other token and decide what
  matters.
- **MLP** (feed-forward): a couple of large matrix multiplies that do most of
  the compute.
- **LayerNorm / RMSNorm**: keeps activations from blowing up or vanishing.

Two performance regimes hide inside "make the matmul fast", and the engine
treats them differently:

| Regime    | Operation        | Bottleneck         | What helps                    |
|-----------|------------------|--------------------|-------------------------------|
| Prefill   | matrix × matrix  | compute (FLOPs)    | tiling / cache blocking, SIMD |
| Decode    | matrix × vector  | memory bandwidth   | SIMD, quantization (fewer bytes/token) |

A KV cache stores the keys and values from previous tokens so decode only
computes the new token's worth of work, instead of recomputing the whole
sequence every step. (Not yet implemented — generation today is correct but
O(n²); see the roadmap.)

## Verification (the actual hard part)

The tricky bugs here aren't crashes; they're a single transposed matrix or an
off-by-one in the attention mask that makes the output *almost* right. So the
project is built around a reference-diffing harness:

1. `tools/reference.py` runs the HuggingFace reference model on one frozen prompt
   and dumps **every** intermediate tensor.
2. The C++ engine dumps the same tensors for the same input (`IE_DUMP_DIR=<dir>`).
3. `tests/check_layers.py` diffs them and reports the **first** layer where they
   diverge.

That turns "stare at twelve layers of code" into "layer 4's attention output is
wrong, look there."

One honest subtlety: per-tensor error **accumulates** through the stack — fp32
rounding around GPT-2's high-magnitude features pushes the deepest layers past a
tight `1e-4` threshold even when the arithmetic is exactly right (the per-layer
error climbs and then *falls* as the final LayerNorm re-normalizes it). So the
real correctness test isn't raw max-abs-error; it's whether greedy decoding picks
the **same token**. `tests/check_argmax.py` checks exactly that against the
reference logits — and it matches at every position.

## Build

Requires a C++17 compiler and CMake ≥ 3.16. (OpenMP is used if present.)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Or use the convenience wrapper:

```bash
make        # configure + build
make test   # run the unit tests
make run    # build and print the CLI help
```

### Run

The weights and tokenizer aren't committed (the weights are large). Export them
once with the Python tooling:

```bash
python -m pip install -r requirements.txt
python tools/export_weights.py   --model gpt2  --out models/gpt2-124m.bin
python tools/convert_tokenizer.py --from-hf gpt2 --out models/tokenizer.bin
```

Then generate:

```bash
./build/inference-engine --model models/gpt2-124m.bin --tokens 20
```

```
The quick brown foxes are a great way to get a little bit of a kick out of your dog.
```

Sampling is greedy (argmax), and there's no KV cache yet, so generation is O(n²)
in sequence length — keep `--tokens` modest for now. The BPE `encode()` isn't
implemented either, so the prompt is read from a fixed token-id file
(`--input-ids`, defaulting to the reference prompt) rather than `--prompt`; that
switches over once the encoder lands.

To re-verify the forward pass against the reference instead of generating, set
`IE_DUMP_DIR` (this runs a single forward and dumps every activation):

```bash
IE_DUMP_DIR=tests/engine_dumps ./build/inference-engine --model models/gpt2-124m.bin
python tests/check_layers.py --reference tests/reference_dumps --engine tests/engine_dumps
python tests/check_argmax.py --reference tests/reference_dumps --engine tests/engine_dumps
```

### Useful build options

| Option                  | Default | Effect                                   |
|-------------------------|---------|------------------------------------------|
| `IE_ENABLE_NATIVE`      | `ON`    | Compile with `-march=native` (AVX2/FMA)  |
| `IE_ENABLE_OPENMP`      | `ON`    | Link OpenMP for multithreading if found  |
| `IE_SANITIZE`           | `OFF`   | Address + UB sanitizers (great for debugging the forward pass) |
| `IE_WARNINGS_AS_ERRORS` | `OFF`   | Treat warnings as errors                 |

## Roadmap

- [x] Repo scaffolding, CMake build, CI, naive GEMM + test
- [x] **Phase 1**: Weight export + BPE tokenizer (`decode` first)
- [x] **Phase 2**: Full forward pass, naive matmul, coherent GPT-2 text (verified against reference)
- [ ] **Phase 3**: KV cache (the unusable → usable jump)
- [ ] **Phase 4**: GEMM optimization (AVX2, multithreading, cache tiling)
- [ ] **Phase 5**: int8, then int4 quantization
- [ ] **Phase 6**: Port to a ~1B Llama (RoPE, SwiGLU, GQA, RMSNorm) + benchmark vs `llama.cpp`

## Benchmark

Measured per regime (prefill and decode are different problems and lumping them
hides the story), on the same model, machine, quantization, and thread count as
`llama.cpp`'s own `llama-bench`. Numbers land here once Phase 6 is done.

| Model | Regime  | engine (tok/s) | llama.cpp (tok/s) | ratio |
|-------|---------|---------------:|------------------:|------:|
| TBD   | prefill | TBD            | TBD               | TBD   |
| TBD   | decode  | TBD            | TBD               | TBD   |

**Honest target:** `llama.cpp` is years of hand-tuned kernels. A from-scratch
CPU engine landing within **2-4×** of its tokens/sec on the same hardware is a
strong result; "same order of magnitude" is the win. The writeup will focus on
*the gap and what would close it*, which is more interesting than a single
number.

## Project layout

```
inference-engine/
├── src/
│   ├── main.cpp         # CLI + generation loop
│   ├── model.{cpp,hpp}  # config, weight loading
│   ├── tokenizer.*      # byte-level BPE
│   ├── forward.*        # the forward pass + building blocks
│   ├── gemm.*           # matmul kernels: naive → SIMD → threaded → tiled
│   └── quant.*          # int8 / int4 quantization
├── tools/
│   ├── export_weights.py    # HF weights → flat binary
│   ├── convert_tokenizer.py # HF vocab + merges → flat tokenizer.bin
│   └── reference.py         # dump reference activations (verification harness)
├── tests/
│   ├── test_gemm.cpp     # C++ unit test (run by CTest/CI)
│   ├── test_loader.cpp   # loader + decode round-trip (run by CTest/CI)
│   ├── check_layers.py   # diff engine dumps vs reference (per-tensor error)
│   └── check_argmax.py   # the real check: greedy token matches the reference
├── CMakeLists.txt
└── .github/workflows/ci.yml
```

## References

- Andrej Karpathy's [`llm.c`](https://github.com/karpathy/llm.c): a clean
  reference implementation of exactly this idea.
- [`llama.cpp`](https://github.com/ggerganov/llama.cpp): the benchmark, and a
  goldmine of CPU kernel tricks.
- The original GPT-2 and Llama papers for the architecture details.

## License

MIT. See [LICENSE](LICENSE).
