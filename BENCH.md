## Benchmark log (GPT-2 124M, prefill=256 decode=256, <your CPU>)

| Step                        | Prefill tok/s | Decode tok/s | Decode ms/tok | Notes                                    |
|-----------------------------|--------------:|-------------:|--------------:|------------------------------------------|
| Naive scalar                |           9.1 |          6.0 |         165.8 | -O2 -march=native, 1 thread              |
| AVX2 (8-wide FMA, 4 accum)  |          88.3 |         36.6 |          27.3 | linear() vectorized; 9.7×/6.1×           |
| +OpenMP (gated by work size)|         224.7 |         36.8 |          27.2 | prefill rides cores                      |
| token-block tiling(reverted)|         228.8 |         36.1 |          27.7 | no gain vs +OpenMP; mlp_proj L3-resident |

## Phase 5: weight-only quantization (int8 / int4)

Weight-only, group-wise symmetric quantization of the streamed matmul weights
(per-layer c_attn / c_proj / mlp_fc / mlp_proj plus the tied LM head). Activations
stay fp32; the kernel dequantizes each weight group on the fly, so decode streams
fewer weight bytes per token. int8 uses a group of 64; int4 packs two values per
byte. fp32 weights are kept in memory for the embedding lookup, biases, and
LayerNorm, so this is an in-memory format, not an on-disk one.

All numbers same session (so A/B-valid), GPT-2 124M, prefill=256 decode=256,
median of 3 (+1 warmup), 8 physical cores. The fp32 row is this session's control.

### Speed (tok/s)

| Kernel        | Group | Prefill tok/s | Decode tok/s | Decode ms/tok | vs fp32 decode |
|---------------|------:|--------------:|-------------:|--------------:|---------------:|
| fp32 (AVX2+OMP)|     - |         240.7 |         36.1 |          27.7 |           1.00x |
| **int8**      |    64 |         212.6 |     **46.6** |          21.5 |       **1.29x** |
| int4          |    64 |         182.1 |         32.8 |          30.5 |           0.91x |
| int4          |    32 |         161.0 |         27.8 |          35.9 |           0.77x |
| int4          |    16 |         134.2 |         21.4 |          46.6 |           0.59x |

### Footprint and accuracy

Footprint is the quantized matmul weights vs the fp32 size of the same matrices.
Accuracy is from `--compare-quant` (48 teacher-forced steps on "The quick brown
fox": both models see identical context each step, isolating quant error from
autoregressive drift). "Match" is the fraction of steps whose argmax equals fp32's.

| Kernel | Group | Footprint     | Token match vs fp32 | Mean per-logit err | Coherent? |
|--------|------:|--------------:|--------------------:|-------------------:|-----------|
| int8   |    64 | 131.3 MB (3.76x) |          95.8%   |               1.23 | yes (~fp32) |
| int4   |    64 |  69.5 MB (7.11x) |          41.7%   |              18.79 | no (repeats) |
| int4   |    32 |  77.2 MB (6.40x) |          50.0%   |              25.53 | no (repeats) |
| int4   |    16 |  92.6 MB (5.33x) |          64.6%   |              12.52 | yes |

Sample continuations (30 tokens, prompt "The quick brown fox"):
- fp32: "...foxes are a great way to get a little bit of a kick out of your dog."
- int8: "...foxes are a great way to get a quick look at the world around you."
- int4 (g16): "...foxes will surely track you down, or track you down while you work."

### Read

- **int8 is the win:** +29% decode (36.1 -> 46.6 tok/s), 3.76x smaller, and
  near-lossless (95.8% token match, mean logit error 1.23). Decode streams 1 byte
  per weight instead of 4; the saved memory traffic is the bandwidth-bound decode
  path's bottleneck, so it shows up directly as throughput. Prefill dips ~12%:
  it is compute-bound and reuses each weight across all 256 rows, so the int8 ->
  fp32 conversion is pure added ALU with no bandwidth saving to offset it.

- **int4 is a documented negative speed result here.** It needs a group of 16 to
  stay coherent on this small, quantization-sensitive model (groups of 32/64 both
  collapse into repetition), and at group 16 decode is *slower* than fp32 (21.4 vs
  36.1) even though it streams fewer bytes than int8 (92.6 vs 131.3 MB). So decode
  is not purely bandwidth-bound for int4: the nibble-unpack is more ALU per weight
  than an int8 load, and group 16 forces 4x as many per-group horizontal-sum +
  rescale steps as int8's group 64. That overhead is not hidden behind memory
  latency on a 124M model, so it dominates the bandwidth saving. int4 keeps its
  footprint win (5.33x, coherent) but loses on speed.
  Lesson (mirrors the Phase 4 tiling finding): a smaller weight stream only buys
  decode throughput while the path stays bandwidth-bound; past a point the dequant
  ALU becomes the wall. int4's speed payoff is expected on the larger Llama port
  (Phase 6), where the matrices are bigger and decode is more bandwidth-starved.

- The `--bench` checksum changes under quantization (2238.66 -> 1201.89 int8 ->
  135315.56 int4), confirming the quantized kernels actually run.

