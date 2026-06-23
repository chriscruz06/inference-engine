# Writing a CPU LLM inference engine from scratch, within 1.5x of llama.cpp

I wrote a language model inference engine in C++ with no machine learning or math library behind it. No PyTorch, no BLAS, no GGML. The standard library and OpenMP are the only things linked. It loads real GPT-2 124M and TinyLlama 1.1B weights off disk, runs the entire transformer forward pass by hand, and generates text. Then I made it fast enough to put next to llama.cpp on the same machine.

The headline: on identical hardware, the same model, and the same thread count, its fp32 decode throughput lands within **1.5x** of llama.cpp, and prefill within **3.5x**. That is the whole engine running on a single hand-written AVX2 kernel, measured against years of tuned production code. This post is about that number, what it actually means, and the three things I learned earning it.

All numbers below are on an AMD Ryzen 7 PRO 7840U, 8 physical cores.

## System Overview

Generating one token from a transformer is, underneath, a fixed sequence of matrix multiplies plus a handful of elementwise operations: attention, a feed-forward block, a normalization step, repeated once per layer. There is no magic in the control flow. The hard parts are getting the arithmetic exactly right, and then getting it to move through the memory system quickly.

The engine implements all of that directly. It starts with GPT-2 124M (learned position embeddings, pre-LayerNorm, GELU, a tied output head) and then ports to TinyLlama 1.1B, which swaps in the modern Llama-2 pieces: RMSNorm, rotary position embeddings, grouped-query attention with a narrowed KV cache, and a SwiGLU MLP. A single `linear` kernel serves both models and both compute regimes. On top of that sit a KV cache for incremental decoding and weight-only int8 and int4 quantization. The Python tooling exists only to export weights and to provide a reference to check against. The engine itself has no runtime dependencies.

## Correctness and Verification

A fast engine that produces slightly wrong text is just a fast bug, and the bugs in this kind of work are nasty: a single transposed weight matrix or an off-by-one in the causal mask makes the output *almost* right, which is the hardest failure to spot by reading code.

So the first thing I built was not the forward pass. It was a diffing harness. A Python script runs the HuggingFace reference model on a fixed prompt and dumps every intermediate tensor; the engine dumps the same tensors for the same input; a third script reports the first layer where they diverge. That turns "stare at twelve layers of code" into "layer 4's attention output is wrong, look there."

One subtlety that matters for the rest of the post: the correctness bar is **token identity, not bit identity**. AVX2's fused multiply-add reorders the summation in a dot product, so the engine's logits differ from the reference at the 1e-4 level. That is not a bug. The gate is whether greedy decoding picks the same token at every position, and it does, for both models, on the reference prompt. Floating point associativity is a thing you make peace with once and then stop fighting.

## Comparison with llama.cpp

Here is the comparison against llama.cpp on TinyLlama 1.1B, both engines at 8 threads, prefill 128 and decode 64 tokens. llama.cpp was built with `GGML_NATIVE=ON`, so it gets the same AVX2/FMA instruction set as my `-march=native`. The quantized rows are not the same kernel (llama.cpp uses 32-weight blocks and hand-tuned integer dot products; mine dequantizes group-wise on the fly), so they compare engines at the same bit width, not the same implementation.

| Precision (ours / llama) | Regime  | ours tok/s | llama.cpp tok/s | gap  |
|--------------------------|---------|-----------:|----------------:|-----:|
| fp32 / F32               | prefill |       36.9 |           128.0 | 3.5x |
| fp32 / F32               | decode  |        5.0 |             7.6 | **1.5x** |
| int8 / Q8_0              | prefill |       36.2 |           149.2 | 4.1x |
| int8 / Q8_0              | decode  |        7.3 |            26.4 | 3.6x |
| int4 / Q4_0              | prefill |       23.8 |           220.5 | 9.3x |
| int4 / Q4_0              | decode  |        4.8 |            43.7 | 9.1x |

The fp32 row is the apples-to-apples comparison. It is identical arithmetic with no quantization-kernel quality to confound the result, so it isolates the thing I actually wrote: the GEMM and how it touches memory. Landing fp32 decode within 1.5x of llama.cpp is the result the project was aiming for. The gap widens with quantization, and the rest of this post explains both halves of that picture: why fp32 is close, and why int8 and int4 are not.

## Two Compute Regimes: Prefill and Decode

The single most useful thing I learned is that "make the matrix multiply fast" hides two different workloads with different bottlenecks.

**Prefill** processes the whole prompt at once. Every token flows through the weight matrices together, so it is a matrix times matrix product, a real GEMM. It is compute-bound: the limit is how many fused multiply-adds per second the cores can retire, and each weight is reused across every row of the batch.

**Decode** generates one token at a time. With a KV cache, each step computes the query, key, and value for a single new token and attends over the cached history. That single token times a weight matrix is a matrix times *vector* product, a GEMV. It is memory-bandwidth-bound: every weight is touched exactly once per token and then thrown away, so the limit is how fast you can stream the weights out of RAM, not how fast you can multiply.

That one distinction explains the entire optimization arc. Here is the kernel progression on GPT-2 124M:

| Kernel                          | Prefill tok/s | Decode tok/s |
|---------------------------------|--------------:|-------------:|
| Naive scalar (`-O2`, 1 thread)  |           9.1 |          6.0 |
| AVX2 (8-wide FMA, 4 accumulators) |        88.3 |         36.6 |
| AVX2 + OpenMP (work-size gated) |         224.7 |         36.8 |

AVX2 lifts both regimes by roughly 10x and 6x, because vectorizing the dot product helps whether you are compute-bound or bandwidth-bound. The signature trick in that kernel is four independent accumulators:

```cpp
__m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
__m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
for (; i + 32 <= in_dim; i += 32) {
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + i),      _mm256_loadu_ps(wr + i),      a0);
    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + i + 8),  _mm256_loadu_ps(wr + i + 8),  a1);
    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + i + 16), _mm256_loadu_ps(wr + i + 16), a2);
    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + i + 24), _mm256_loadu_ps(wr + i + 24), a3);
}
```

FMA has about 4 to 5 cycles of latency but high throughput. A single accumulator would serialize on its own dependency chain, leaving the pipeline mostly idle. Four accumulators give the out-of-order engine enough independent work to stay full.

Then look at the third row. Adding OpenMP across output features lifts prefill another 2.5x, to 224 tok/s, but decode does not move at all: 36.6 to 36.8. That is not a disappointment, it is the prediction confirmed. Decode is already saturating memory bandwidth on one core, so handing the work to eight cores buys nothing, and ungated it actually makes decode *slower* through fork-join overhead and cross-core contention on a one-row GEMV. The kernel keeps decode single-threaded with a work-size gate on the parallel region:

```cpp
#pragma omp parallel for schedule(static) if (work >= kLinearParallelMinWork)
```

where `work` is `rows * out_dim * in_dim`. Prefill's big GEMM clears the threshold and rides all cores; decode's tiny per-token GEMV stays below it and runs serially.

This also explains the fp32 result above. Decode is the close regime (1.5x) precisely because it is bandwidth-bound: both engines stream the same fp32 weights per token, so there is little room for one to pull ahead of the other. Prefill is the wider regime (3.5x) because it is compute-bound, and that is where llama.cpp's better register blocking and threading actually show through.

## Negative Results Worth Reporting

Two optimizations I expected to win did not, and understanding why taught me more than the wins did.

**Cache tiling did nothing.** The textbook next step for a GEMM is to tile the loops so the working set fits in L1 or L2. I implemented it for the prefill path, benchmarked it, and got 224.7 to 228.8 tok/s, which is noise. The reason is specific and worth stating: the activation that tiling would keep resident already fits in the shared L3 on this model, so its re-reads were already cheap L3 hits, not RAM traffic. Tiling only added per-call fork-join overhead with no bandwidth to recover. I reverted it. The remaining prefill scaling gap is shared-L3 bandwidth and all-core clock throttling, neither of which cache blocking can touch.

**int4 was slower than fp32.** This is the counterintuitive one. Quantization shrinks the weight stream, and on a bandwidth-bound decode path fewer bytes should mean more throughput. int8 confirms that cleanly: on GPT-2 it lifts decode 29% (36.1 to 46.6 tok/s) and on TinyLlama 50% (5.4 to 8.1), near-lossless both times. But int4 on GPT-2, at the group size of 16 it needs to stay coherent, decodes at **21.4 tok/s, slower than fp32's 36.1**, even though it streams fewer bytes than int8.

The reason is that decode stops being purely bandwidth-bound once the per-weight work gets expensive enough. Unpacking two 4-bit values per byte is more ALU than loading an int8, and a group of 16 forces four times as many per-group horizontal-sum-and-rescale steps as int8's group of 64. On a model this small that overhead is not hidden behind memory latency, so it dominates the bandwidth saving and the kernel crosses back into being ALU-bound. The general lesson, which mirrors the tiling result: **a smaller weight stream only buys throughput while the path stays bandwidth-bound.** Past that point the dequant ALU becomes the wall, and you have optimized the wrong thing.

## Accounting for the Quantization Gap

So why is llama.cpp 3.6x to 9x ahead on the quantized rows? Because quantized matmul is exactly its specialty. Its Q8_0 and Q4_0 paths are years of hand-tuned SIMD integer dot-product kernels that do the multiply-accumulate in the integer domain with minimal unpack overhead. Mine dequantizes each weight group to fp32 on the fly and then does an fp32 dot product. That is the simplest correct approach, and it works: my int8 decode (7.3 tok/s) still beats my own fp32 decode (5.0) and stays token-identical to it. But it leaves throughput on the table that integer-domain kernels would recover.

The point I want to make is that this gap is the *width of a specific missing optimization*, not a flaw in the from-scratch approach. The engine is competitive with llama.cpp exactly where the comparison is about the GEMM and the memory system (fp32, especially decode), and it falls behind by precisely the amount that llama.cpp's specialized integer kernels are worth. Closing that gap means writing integer-domain dot products with no per-group fp32 dequant. That is a worthwhile next project, but it is a different one from "hand-write the fp32 GEMM," which is what this was.

## Conclusion

A from-scratch CPU inference engine, with no ML or math dependencies, landing within 1.5x of llama.cpp on fp32 decode is a result I am happy to stand behind, because I can account for every part of it. The close regimes are the ones that test what I actually built, the GEMM and the memory system, and the gaps are the ones that test optimizations I deliberately did not write. Knowing which is which, and being able to point at the exact reason for each number, was the real goal. The throughput chart is just the evidence.

The full benchmark log, including the per-model quantization sweeps and the negative results, is in the repository, along with the verification harness and the C++ kernels.

[https://github.com/chriscruz06/inference-engine]