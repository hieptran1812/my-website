---
title: "The Open Kernel Challenge, Decoded: Two Boards, One Model, and Where the Money Is"
date: "2026-10-24"
publishDate: "2026-10-24"
description: "How the Open Kernel Challenge scores kernels and SGLang patches for Muse Spark Open on H200, what its rules forbid, how far SGLang sits from the speed-of-light floor, and the 44-post map of this series."
tags: ["gpu-kernel-engineering", "open-kernel-challenge", "h200", "sglang", "moe", "cuda-graphs", "llm-inference", "kernel-optimization", "benchmarking"]
category: "machine-learning"
subcategory: "GPU Kernel Engineering"
author: "Hiep Tran"
featured: true
readTime: 14
---

Most kernel competitions reward a single number on a single shape. The Open Kernel Challenge, run by Paradigm with Meta and Andromeda at [openkernelchallenge.dev](https://openkernelchallenge.dev/), asks for something closer to a production job: make one real model, Meta's **Muse Spark Open** (an 816B-parameter mixture of experts with 42B active, 256 experts, NVFP4 weights, an FP8 KV cache), serve faster on H200s. It splits that job across two boards with different units, different caps, and very different payoffs. If you read only the leaderboard you will optimize the wrong thing.

![Two tracks on one calendar: a 14-day kernel track feeding serving priority into a 15-day serving track with the prize money](/imgs/blogs/the-open-kernel-challenge-decoded-1.webp)

The diagram above is the mental model: two tracks, run back to back. The **kernel track** (Oct 22 to Nov 4) has five problems, each one Python file timed against SGLang's own kernel on one H200, and no prize money. The **serving track** (Nov 5 to Nov 19) takes a `git diff` against a pinned SGLang, runs it on a whole 8×H200 node, and pays $100,000. The bridge between them is the queue: the top 50 on the overall kernel board when it closes get priority on scarce serving nodes. This post decodes both scoring systems from the challenge's own published rules (`llms.txt`, each problem's `TASK.md`, and `GET /api/v1/serving`), derives how much headroom SGLang leaves on the table, and maps the 44 posts that follow.

## 1. Two boards, and why the money sits on one

The rules are blunt: "The prizes are paid on the serving board; the kernel problems' boards carry no prizes." The serving pot is $50,000 plus a signed H200 for first, then $30,000, $10,000, $3,000, $2,000, and $1,000 each for sixth through tenth.

That makes the kernel track look like a warm-up, and it is not. Three reasons:

1. **Queue priority is real money.** A serving submission holds a whole 8×H200 node for up to 90 minutes, and you get one scored run and three check runs per day. When nodes are contended, going first means more iterations in the same 15 days.
2. **The kernel problems are the serving patch, cut into pieces.** Each problem is one block of the same decoder layer that SGLang runs under `--tp-size 8 --ep-size 8`. A kernel that wins `muse_moe_w4a16` is a candidate to drop into the serving patch, provided it can ship as Python source.
3. **The kernel board is the cheapest feedback loop on this model:** ten scored and twenty check runs per problem per day, against one scored run a day on serving.

| | Kernel track | Serving track |
|---|---|---|
| Window | Oct 22, 11 am PT to Nov 4, 11:59 pm PT | Nov 5, noon PT to Nov 19, 11:59 pm PT |
| Submission | one Python file defining `kernel_fn` | a `git diff` to SGLang `a55ab7d5`, only under `python/sglang/` |
| Hardware | 1× H200 SXM | whole 8× H200 node, up to 90 min |
| Score | speedup vs SGLang's kernel, as CUDA graphs | tokens/s per GPU at stock SGLang's interactivity |
| Daily caps | 10 scored + 20 check runs per problem | 1 scored + 3 check runs |
| Prize | none; top 50 get serving priority | $100,000 across the top 10 |

## 2. The five kernel problems, and how far each sits from its floor

Each problem is one slice of what a single GPU does for Muse Spark Open under TP8/EP8. `okc init <problem>` writes a plain-PyTorch `submission.py`, SGLang's own kernel as `baseline.py`, and the judge's `reference.py` and `checks.py`. Every problem except attention also calls an untimed `prepare(weights)` once, for up to 300 seconds, so you can repack weights any way you like, as long as the repack is lossless.

| Problem | What it computes | SGLang time |
|---|---|---|
| `muse_bf16_proj` | the BF16 projections (qkv_gate, o, pooler, unpooler) and the FP32 router | 4,830.8 µs per decode step |
| `muse_attention_decode` | gated GQA decode over a paged FP8 cache, sliding 2,048-token or full | 3,766.3 µs per decode step |
| `muse_moe_w4a16` | top-8 of 256 experts, NVFP4 weights, SwiGLU, post-expert norm | 13,448.5 µs per decode step |
| `muse_decode_layer` | the whole layer: everything above plus every norm and residual | 22,620.3 µs per decode step |
| `muse_prefill_layer` | the whole layer at prefill shapes (753-token turns, 16K chunks) | 387,403.9 µs per prefill load |

These SGLang times are per step across the whole model, not per layer: a step runs 46 sliding layers and 16 full-attention layers, which is where the weights 46 and 16 in every shape table come from. Dividing the layer problem's time by 62 gives roughly 365 µs per layer per decode step, averaged over loads.

Every `TASK.md` also publishes a **floor** for each scored shape: "the time to move the data at 4.8 TB/s or do the math at 989 TFLOP/s, whichever is longer." Those two constants match the H200 SXM's HBM3e bandwidth and dense BF16 throughput on [NVIDIA's H200 page](https://www.nvidia.com/en-us/data-center/h200/). Since the floors come per shape, we can combine them exactly the way the judge combines times (weighted sum per load, geometric mean over loads) and compare with SGLang:

![Paired horizontal bars per problem: SGLang time per decode step against the geometric-mean speed-of-light floor](/imgs/blogs/the-open-kernel-challenge-decoded-2.webp)

The gap between each dark and light bar is the room a kernel can win. Here is the arithmetic for the whole layer, the problem that best predicts the serving board. The snippet is illustrative, uses only numbers from `TASK.md`, and the output below is from running it with `python3 -I`:

```python
import math

def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

# muse_decode_layer: per-shape floors (us) from TASK.md, as (sliding, full).
# A step runs 46 sliding layers and 16 full layers.
floors = {
    "decode 1": (40.0, 44.2), "decode 16": (76.7, 137.5), "decode 32": (84.5, 208.1),
    "verify 1": (62.3, 66.4), "verify 16": (126.6, 188.7), "verify 32": (143.8, 267.2),
}
load_floor = {load: 46 * s + 16 * f for load, (s, f) in floors.items()}
for load, t in load_floor.items():
    print(f"{load:10s} floor {t:9.1f} us")

sglang_us = 22620.3  # SGLang's time per decode step, averaged over loads the same way
gm_floor = geomean(load_floor.values())
print(f"geomean floor {gm_floor:.1f} us -> ceiling speedup {sglang_us / gm_floor:.2f}x")

# Two strategies, six loads each: per-load speedups over SGLang.
spiky = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
even = [1.1225] * 6
print(f"one 2x load, five flat: {geomean(spiky):.4f}x")
print(f"1.1225x on every load:  {geomean(even):.4f}x")
```

```console
decode 1   floor    2547.2 us
decode 16  floor    5728.2 us
decode 32  floor    7216.6 us
verify 1   floor    3928.2 us
verify 16  floor    8842.8 us
verify 32  floor   10890.0 us
geomean floor 5843.9 us -> ceiling speedup 3.87x
one 2x load, five flat: 1.1225x
1.1225x on every load:  1.1225x
```

The same computation over each problem's own floor table gives ceilings of 2.02× for the projections, 6.73× for attention, 4.40× for the routed experts, 3.87× for the whole decode layer, and 3.18× for prefill. Two readings follow. First, the projections are already close to bandwidth-bound: at m=1 a GEMV has nowhere to hide, and the win there is in fusing launches and reading weights once, not in clever math. Second, attention has the widest gap, but much of it sits on full-attention layers at batch 16 and 32, where an 80,000-token FP8 context must be streamed per sequence; the sliding layers' floors are a few microseconds and are dominated by launch and gather overhead rather than bytes. A floor is a bound, not a target.

The three parts sum to 22,045.6 µs, about 575 µs short of the whole layer: norms, residuals and launch gaps (approximately, since loads average geometrically). Post #3 takes that apart.

## 3. How the kernel board scores you

The scoring rule is three nested aggregations, and each one changes what is worth optimizing.

![Worked example of the score: paired ratios per shape, weighted sum per load, geometric mean over six loads, and two strategies with the same result](/imgs/blogs/the-open-kernel-challenge-decoded-3.webp)

1. **Per shape.** Your graph and SGLang's graph replay in turns on the same fresh inputs, each replay starting with a cold L2 (the judge first reads a buffer larger than L2). A shape's speedup is the median of paired ratios, taken two repetitions at a time with the order swapped so whichever side went first cancels. A fixed graph-replay cost of about 4 µs is subtracted from both sides.
2. **Per load.** A load such as "decode 16" is the sum of its shapes' times, each multiplied by its weight (how many times one step runs that shape).
3. **Per problem.** Your speedup is the geometric mean of SGLang's load time over yours, across loads. Every load counts equally.

The third rule is the one people underweight. As the script above shows, a 2× win on one of six loads and nothing elsewhere scores `2^(1/6) = 1.12×`, the same as a uniform 12% win. A kernel tuned for batch 32 that regresses at batch 1 can lose ground overall. The decode problems have six loads (decode 1, 16, 32 and verify 1, 16, 32, the last three being 8-token speculative verify blocks), so a strategy has to cover the whole shape range.

Each run reports its speedup with a 95% interval of at least ±1.5%. A 1% "improvement" between two submissions is noise, not progress, which matters a great deal when you only get ten scored runs a day.

## 4. What the judge checks before it times anything

A fast wrong kernel scores nothing, and the judge is designed so that "wrong" includes "right only on the scored shapes."

![Pipeline of a kernel submission from static checks through 13 tests, CUDA graph capture, interleaved timing, and final reruns](/imgs/blogs/the-open-kernel-challenge-decoded-4.webp)

The gates, in order:

- **Static checks.** A submission is rejected if it patches timing, reads harness frames, touches harness I/O or secrets, edits loaded modules, imports the reference, starts processes, or uses the network.
- **Thirteen correctness tests,** some harder than the scored shapes. The decode layer's list includes a 1,048,576-token full-attention context, a batch of 33, and a run with a KV scale of 1/64. A kernel that hard-codes the 80K scored context fails here.
- **Every timed call is also checked.** Errors are relative L2 against a reference. For the decode layer the residual update must be within 2.5× the FP32 reference's own error when its queries are rounded to FP8 (3–15% depending on shape), new cache entries within 8%, and no single row above 5× its bound. Routing gets its own check: every expert you pick must score within 0.05 of the reference's 8th best, and any of the reference's top 8 on this GPU that clearly wins (beating the 9th by more than 0.03) must be among yours.
- **CUDA-graph capture.** No `.item()`, `.cpu()`, or `torch.cuda.synchronize()`, and no allocation or control flow that depends on input values. Branching on shapes and `config` is allowed. The [CUDA graphs post](/blog/machine-learning/inference-engineering/cuda-graphs-and-torch-compile-for-the-decode-loop) covers why this matters for decode.

Then the numerics fence, which is the rule most likely to be broken by accident:

| Allowed | Forbidden |
|---|---|
| Repack weights losslessly in `prepare()` | Storing NVFP4 or BF16 weights at any other precision |
| Dequantize NVFP4 to BF16 inside your kernel | BF16 activations rounded to FP8 where SGLang keeps BF16 |
| Router FP32 weights as a BF16 hi + lo pair, FP32 accumulate | Any other lowering of the router |
| Queries rounded to FP8 E4M3, as FA3 does | Anything coarser than E4M3 for queries |
| Any attention schedule over every key | Top-k, sparse, or block-skipping attention |

Finally, every board position is provisional. After the challenge the organizers read the top submissions, rerun them several times on held-out data and on nearby shapes, and rank by the medians of those reruns. The rules name the patterns they remove: tampering with timing, doing work outside the timed calls, caching or replaying outputs, and computing below the allowed precision. The judge descends from GPU MODE's open [reference-kernels](https://github.com/gpu-mode/reference-kernels) harness, and post #5 walks through it.

## 5. How the serving board scores a patch

The serving track measures something a microbenchmark cannot: a patched SGLang serving multi-turn agentic-coding conversations end to end, with the DFlash drafter proposing 8 tokens per step, hierarchical caching, and 16,384-token chunked prefill. Each request returns exactly 220 tokens.

![Throughput per GPU against tokens per second per user: a stock SGLang curve through four concurrency rungs and a patched curve above it, scored at stock's interactivity](/imgs/blogs/the-open-kernel-challenge-decoded-5.webp)

The figure is an illustrative shape, not measured data. The ladder runs four rungs at concurrency 1, 4, 16 and 32 (three reps each, except one at 32). Each rung gives a point on a curve of tokens/s per user (interactivity) against tokens/s per GPU (throughput). Your score is **throughput at stock SGLang's interactivity, relative to stock**: you are compared at the same user experience, so you cannot win by simply raising batch size and letting each user's stream crawl. Two guards close the remaining loopholes. Time to first token may be at most 2× stock's at each load, and a load that is slower than that cannot raise your score. Outputs must agree with stock SGLang's nearly as well as stock agrees with itself.

The patch rules are tight and worth memorizing before you write a line:

- Only source and JSON files under `python/sglang/`. No renames, mode changes, or binaries; the prebuilt `sgl-kernel` is never rebuilt.
- Nothing under `python/sglang/srt/model_loader/`. At most 720 KB, no added line over 400 characters.
- The model stays the model: 8 routed experts per token, the 2,048-token window, every norm, and the registered parameter names and shapes.
- The drafter stays DFlash with its released weights. You may make it faster. Extra drafting may use only the requests themselves, such as prompt lookup or n-grams.

Kernels from the first track enter through runtime-compiled paths (Triton, TileLang, CuTe DSL, or `load_inline` with source in the patch), which is why post #40 exists. For the drafter itself, see the [DFlash mechanism](/blog/machine-learning/large-language-model/dflash-block-diffusion-speculative-decoding); for the engine, the [SGLang overview](/blog/machine-learning/inference-frameworks/sglang-inference).

## 6. The series map

The series is a curriculum aimed at the top of both boards. Every post ends with an "On the board" box saying which problem or track it moves and what the rules forbid around it.

![Grid of 44 cells in nine columns, one per track, colored by the board each track moves, with this post highlighted](/imgs/blogs/the-open-kernel-challenge-decoded-6.webp)

**Track A, the challenge and the model (Wave 1).** 1. The Open Kernel Challenge, Decoded (this post). 2. [Muse Spark Open for Kernel Engineers](/blog/machine-learning/gpu-kernel-engineering/muse-spark-open-for-kernel-engineers). 3. [One Layer on One H200: Where the Microseconds Go](/blog/machine-learning/gpu-kernel-engineering/one-layer-on-one-h200-where-the-microseconds-go). 4. [Roofline for Decode on H200: Bytes, Not FLOPs](/blog/machine-learning/gpu-kernel-engineering/roofline-for-decode-on-h200-bytes-not-flops). 5. [Inside the Judge: CUDA Graph Timing, Tolerances and Reward Hacking](/blog/machine-learning/gpu-kernel-engineering/inside-the-judge-cuda-graph-timing-tolerances-and-reward-hacking). 6. [Setting Up the Lab: okc, the SGLang Image and Nsight](/blog/machine-learning/gpu-kernel-engineering/setting-up-the-lab-okc-sglang-image-and-nsight).

**Track B, Hopper SM90 craft (Wave 2).** 7. Hopper SM90 for Kernel Writers: WGMMA, TMA, Clusters. 8. Warp Specialization and Producer-Consumer Pipelines. 9. CuTe Layouts, the Mental Model. 10. CuTe DSL: A Hopper GEMM in Python. 11. Triton, TileLang or CUDA on Hopper: When Each Wins. 12. Reading an Nsight Compute Profile.

**Track C, numerics under the rules (Wave 3).** 13. NVFP4 on a GPU with No FP4 Tensor Cores. 14. FP8 E4M3 KV Cache Scales and In-Kernel Dequant. 15. Accumulation Error and Tolerance Budgets. 16. The Router Is the Precision Trap. 17. Lossless Weight Repacking in `prepare()`.

**Track D, dense projections and attention decode (Wave 4).** 18. Skinny BF16 GEMM on Hopper: Split-K and Stream-K. 19. Fusing the Projections into One Launch. 20. Flash-Decoding Split-KV for 80K Contexts. 21. GQA Head Packing and a Paged FP8 Cache with Page Size One. 22. Fusing QK-Norm, RoPE and the Sigmoid Gate into Attention. 23. Speculative Verify Blocks: 8-Token Causal Decode Attention.

**Track E, routed experts (Wave 5).** 24. Inside FA3 and SGLang's Attention Baseline. 25. Marlin W4A16 Anatomy. 26. MoE Routing Kernels: Sigmoid Top-k, Sort and Align. 27. Grouped GEMM under Skewed Routing. 28. Fusing SwiGLU, the Post-Expert Norm and the Weighted Sum. 29. The Prefill Grouped NVFP4 GEMM Is Compute-Bound.

**Track F, the whole layer (Wave 6).** 30. Glue Kernels: Norms, Residuals and Why They Add Up. 31. Launch Overhead, PDL and CUDA Graphs. 32. Toward a Decode Megakernel. 33. Prefill Layer: Chunked Attention Plus the Big GEMM.

**Track G, the serving track (Wave 7).** 34. SGLang at the Pinned Commit, from Scheduler to Kernel. 35. Throughput at Fixed Interactivity: Reading the Serving Score. 36. DFlash in SGLang: Making the Drafter Faster Without Changing It. 37. Prompt Lookup and N-gram Drafting for Agentic Coding. 38. TP8/EP8 Communication on One H200 Node. 39. Radix and Hierarchical Cache for Multi-Turn Agents.

**Track H, shipping and strategy (Wave 8).** 40. Shipping JIT Kernels into SGLang Without Rebuilding sgl-kernel. 41. Iterating with One Scored Run per Day. 42. Real Speedups vs Gaming the Judge.

**Track I, capstone (after Nov 20).** 43. Open Kernel Challenge Post-Mortem, written only after final results. 44. A Top-1 Playbook for Kernel and Serving Challenges.

Tracks B and C come first because every problem needs Hopper's asynchronous machinery and has a precision fence. The routed experts get a whole track because they are about 59% of the decode layer (13,448.5 of 22,620.3 µs per step). Background on [MoE routing and expert parallelism](/blog/machine-learning/inference-engineering/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem) and on [FP8 and FP4 formats](/blog/machine-learning/inference-engineering/fp8-and-fp4-inference-what-the-hardware-actually-gives-you) is assumed, not re-derived.

## 7. Three ways to misread this challenge

**Chasing the biggest gap.** Attention has a 6.73× ceiling, so it looks like the best place to start. But in the decode layer it costs 3,766.3 µs of 22,620.3, about 17%. Amdahl says a 2× attention kernel alone moves the layer by roughly 9%, while the same 2× on the experts moves it by about 42%. Read the ceiling and the share together.

**Optimizing one load.** The geometric mean punishes regressions at shapes you did not tune for. A batch-32 specialist that loses 10% at batch 1 has given back more than half of a 20% win on its own load, since `(1.2 × 0.9)^(1/6)` is 1.013× against 1.031× had batch 1 held flat. Tune per shape, but check all six.

**Treating the kernel board as the goal.** The money is on serving, where a kernel only counts if it ships as Python source inside a 720 KB patch, captures in SGLang's CUDA graphs, and agrees with stock outputs. A kernel that needs a prebuilt binary is a kernel-board trophy and nothing more.

## On the board

> **On the board.** *Moves:* nothing directly; this post sets the target for all five kernel problems and the serving board. *Lever:* aim where share × headroom is largest (routed experts first, then the whole layer), and optimize every load, because the geometric mean counts them equally. *Rule fence:* weights keep their precision (lossless repacks only), activations stay BF16 where SGLang keeps them, the KV cache stays FP8, queries go no coarser than FP8 E4M3, attention covers every key, everything captures in a CUDA graph, and final places come from medians of post-challenge reruns on held-out and nearby shapes.
