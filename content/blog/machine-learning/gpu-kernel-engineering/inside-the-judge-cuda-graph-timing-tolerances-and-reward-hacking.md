---
title: "Inside the Judge: CUDA Graph Timing, Tolerances and Reward Hacking"
date: "2026-10-24"
publishDate: "2026-10-24"
description: "How the Open Kernel Challenge judge times a kernel against SGLang, what each problem's error bound is and why, how the routing check works, and which shortcuts get a run removed."
tags: ["gpu-kernel-engineering", "open-kernel-challenge", "cuda-graphs", "benchmarking", "reward-hacking", "h200", "numerics", "moe", "sglang", "gpu-mode"]
category: "machine-learning"
subcategory: "GPU Kernel Engineering"
author: "Hiep Tran"
featured: true
readTime: 13
---

Every kernel competition is two contests at once. The visible one is about making a GEMM or an attention step faster. The invisible one is between the people who write kernels and the code that decides whether a kernel is fast and correct. When the judge loses, the board fills with numbers a stopwatch produced and nobody can ship. The Open Kernel Challenge says so plainly in its rules: "The judge catches many attempts to game it but not all of them," so the top entries are read by humans and rerun after the close, on held-out data, with final positions taken from the medians.

This post is a tour of that judge (the [challenge overview](/blog/machine-learning/gpu-kernel-engineering/the-open-kernel-challenge-decoded) covers the tracks and prizes), built from the challenge's own files: the `llms.txt` rules page, each problem's `TASK.md`, and the `reference.py` and `checks.py` that `okc init` writes beside your submission. It explains what the judge measures and why each rule exists. It is not a guide to slipping past it; the rules say that gets you removed.

![The four gates a kernel submission passes: static scan, untimed prepare, correctness tests, timed CUDA graph replays, then final review](/imgs/blogs/inside-the-judge-cuda-graph-timing-tolerances-and-reward-hacking-1.webp)

The diagram above is the mental model. A single Python file defining `kernel_fn` is scanned before anything runs, gets one untimed `prepare(weights)` call of up to 300 seconds, must pass a set of correctness tests that are harder than the scored shapes, and only then is timed.

## Why a kernel judge is different

At microsecond scale almost every naive measurement choice inflates the number, and a search process (a person on day nine, or an agent on hour two) finds the inflation before the speedup.

| Naive assumption | What actually happens at these sizes | What the OKC judge does |
| --- | --- | --- |
| Time the call in eager mode | Launch overhead dominates; eager winners lose under graphs | Captures each shape as a CUDA graph and times replays |
| Run SGLang's kernel once as a baseline | Clocks, thermals and neighbors drift between runs | Interleaves your replays with SGLang's in the same run |
| Reuse one input for all repetitions | Outputs can be cached or replayed | Fresh inputs every call, every timed call checked |
| Loop the replay back to back | Weights stay resident in L2 | Reads a buffer larger than L2 before every replay |
| Check correctness on the scored shapes | Kernels get tuned to exactly those shapes | Harder tests first, nearby and held-out shapes in final review |

## 1. What gets timed: graph replays, interleaved, cold

**Rule of thumb: tune on CUDA graph replays from a cold L2, because that is the only number the board sees.** The task files are explicit that eager timing misleads: "configurations that win eagerly can lose under graphs."

![Two lanes over time: the judge writes fresh inputs and flushes L2, while the H200 stream alternates SGLang and submission replays, swapping order in each pair](/imgs/blogs/inside-the-judge-cuda-graph-timing-tolerances-and-reward-hacking-2.webp)

Your replay and SGLang's take turns on the same GPU and the same fresh inputs, each after a read of a buffer larger than L2, and the order flips every pair. Per `TASK.md`, a shape's speedup is "the median of paired ratios: each repetition's SGLang time over yours, taken two repetitions at a time with the order swapped, so which side went first cancels."

Two more adjustments shape the number. First, the judge subtracts a graph replay's fixed cost, about 4 µs, from both sides. That matters a great deal for the smallest shapes. Take an illustrative router shape where SGLang's replay measures 9 µs and yours 7 µs. The raw ratio is $9/7 \approx 1.29$. After removing the fixed cost from both, it is $(9-4)/(7-4) = 5/3 \approx 1.67$. The subtraction stops a fixed launch tax from compressing every small-shape speedup toward 1. Second, each run reports its speedup with a 95% interval of at least ±1.5%, so a 1% gain is noise by construction.

Shapes then roll up into loads, and loads into a problem score. A load's time is the sum of its shapes' times, each multiplied by its weight (how many times one decode step runs it; 62 for each `muse_bf16_proj` layer, 46 for sliding and 16 for full attention layers in `muse_decode_layer`). The problem speedup is the geometric mean over loads, so `decode 1` counts exactly as much as `verify 32`.

### Why the L2 flush exists

![Bars to scale of the muse_bf16_proj weight sets against H200's 50 MB L2: router, sliced unpooler, o and qkv_gate fit; pooler and the full unpooler do not](/imgs/blogs/inside-the-judge-cuda-graph-timing-tolerances-and-reward-hacking-3.webp)

The weight sizes follow directly from the shapes in `reference.py`. The `o` projection is $8192 \times 1024$ BF16, so $8192 \times 1024 \times 2 = 16.8$ MB. The FP32 router is $256 \times 8192 \times 4 = 8.4$ MB. `qkv_gate` is $2304 \times 8192 \times 2 = 37.7$ MB. Hopper's L2 is 50 MB ([NVIDIA Hopper architecture in-depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)). Four of the six weight sets fit. In a back-to-back loop they would be served from L2, a situation that never happens in serving, where every other layer's weights evict them between visits.

The judge's floors use the same bytes: the `TASK.md` floor for `o` at m=1 is 3.5 µs, and $16.8\text{ MB} / 4.8\text{ TB/s} = 3.5$ µs. These floors are bytes over HBM bandwidth (or FLOPs over 989 TFLOP/s, whichever is longer; the same speed-of-light arithmetic [the roofline post](/blog/machine-learning/gpu-kernel-engineering/roofline-for-decode-on-h200-bytes-not-flops) does for every shape), so they only make sense if the bytes come from HBM. The flush makes the measurement agree with the floor's assumption.

The real `okc` loop from `llms.txt`; a check run uses the same tests and timing without spending a scored run.

```bash
okc init muse_bf16_proj ./proj        # submission.py, baseline.py, reference.py, checks.py, TASK.md
cd proj
okc check submission.py               # same tests + timing, off the board, own daily cap (20/day)
okc submit submission.py --json       # scored run (10/day), result as JSON
okc problems --json                   # today's limits and what you have left
```

## 2. What counts as correct: relative L2, per row, and probes

**Rule of thumb: the bound is on the whole tensor and on every row, so a kernel that is right on average and wrong on one token fails.** Every problem shares one `checks.py`, and its `compare(out, ref, tol)` does four things: shape and dtype must match exactly, no NaN or Inf, whole-tensor relative L2 error $\lVert out - ref \rVert / \lVert ref \rVert \le tol$, and every row (the last dimension) within $5 \times tol$. A near-zero row is measured against a tenth of the average row norm instead of its own.

![A log-axis ladder of error bounds: router 1e-4, probe rows 1e-3, BF16 projections 1 percent, routed experts 1.5 percent, cache entries 8 percent, attention and layer outputs 3 to 15 percent](/imgs/blogs/inside-the-judge-cuda-graph-timing-tolerances-and-reward-hacking-4.webp)

| Output | Bound (relative L2) | Why this tight |
| --- | --- | --- |
| Router logits (`muse_bf16_proj`) | 1e-4 | Logits choose experts; small errors flip top-8 picks |
| Probe rows (`muse_bf16_proj`) | 1e-3 | Must be an exact weight column at the weight's own precision |
| BF16 projections | 1e-2 | BF16 output rounding plus accumulation order |
| Routed experts (`muse_moe_w4a16`) | 1.5e-2 | NVFP4 dequant and the post-expert RMSNorm |
| New FP8 KV cache entries | 8e-2 | FP8 storage is already coarse |
| Attention and layer outputs | 3% to 15%, per case | Scaled by the reference's own FP8-query error |

The router rung encodes a rule: FP32 router weights may run as a BF16 hi + lo pair with FP32 accumulation, never as plain BF16. The illustrative script below (CPU, a toy, not the judge) uses the router's shape and weight distribution from `reference.py`:

```python
# Illustrative: why muse_bf16_proj's router bound is 1e-4 (CPU, small m).
import math, torch

torch.manual_seed(0)
N, K, M = 256, 8192, 16                      # router shape from reference.py
W = (torch.randn(N, K) / math.sqrt(K)).float()  # FP32 weights, std 1/sqrt(K)
x = torch.randn(M, K).bfloat16()             # BF16 activations

ref = x.double() @ W.double().T              # FP64 reference, as the judge does

def rel_l2(out):
    return float((out.double() - ref).norm() / ref.norm())

# 1) FP32 weights, FP32 accumulation: what the reference intends
fp32 = x.float() @ W.T
# 2) BF16 weights alone (the tempting shortcut)
bf16 = (x.float() @ W.bfloat16().float().T)
# 3) BF16 hi + lo split, FP32 accumulation (what SGLang does above 8 tokens)
hi = W.bfloat16()
lo = (W - hi.float()).bfloat16()
split = x.float() @ hi.float().T + x.float() @ lo.float().T

for name, out in [("fp32 weights", fp32), ("bf16 weights", bf16), ("bf16 hi+lo", split)]:
    err = rel_l2(out)
    print(f"{name:13s} rel L2 = {err:.2e}  {'PASS' if err <= 1e-4 else 'FAIL'} (tol 1e-4)")
```

Run with `python3 -I` on PyTorch 2.7.1:

```console
fp32 weights  rel L2 = 1.61e-06  PASS (tol 1e-4)
bf16 weights  rel L2 = 1.63e-03  FAIL (tol 1e-4)
bf16 hi+lo    rel L2 = 2.58e-06  PASS (tol 1e-4)
```

BF16 keeps 8 bits of significand, so each weight carries a relative rounding error of up to $2^{-9} \approx 0.2\%$, and the output inherits about that: 1.6e-3, sixteen times over the bound. The hi + lo pair recovers roughly 16 bits and lands well inside. The tolerance sits in the gap between the allowed trick and the forbidden one.

### Probe rows: catching quietly lower precision

A 1% bound leaves room for weights stored at lower precision that still average under 1% on random inputs. Probe rows close it. About one row in eight of the input has a single nonzero entry, a power of two between 8 and 128, so the output row is just a scaled column of the weight matrix: exact with the weights as given. Those rows must match within 0.1%. Quietly re-quantized weights fail here even when the average looks fine.

## 3. Attention's moving bound

**Rule of thumb: the attention bound is a function of the case, not a constant.** FP8 queries are allowed (FA3 rounds queries to E4M3 when the cache is FP8), so the judge first measures how much error that rounding introduces in the reference itself, then grants you a multiple of it.

![Allowed error versus the reference's own FP8-query error: two clamped lines between a 3 percent floor and a 15 percent cap](/imgs/blogs/inside-the-judge-cuda-graph-timing-tolerances-and-reward-hacking-5.webp)

The function is `query_precision_bound` in `checks.py`: $\min(cap, \max(floor, factor \cdot e + slack))$, where $e$ is the reference's relative L2 error with E4M3 queries. For `muse_attention_decode` the constants are factor 2.5, slack 0.015, floor 0.03, cap 0.15. A case where FP8 queries cost the reference $e = 0.02$ gets $2.5 \times 0.02 + 0.015 = 0.065$, a 6.5% bound. The layer problems use the same factor and clamp without the slack, so the same case would get 5%.

The design is self-calibrating: peaky attention, where FP8 query rounding barely matters, gets a bound near the floor; attention spread over many keys gets more room. A fixed tolerance would be too loose on one or too tight on the other.

The tests make the bound bite: keys at the window's edges catch an off-by-one in the 2,048-token window, and on full layers the low-attention keys' values share a direction, so dropping them shifts the output. That is the numerical teeth behind the ban on top-k, sparse or block-skipping attention.

## 4. The routing check

**Rule of thumb: the router may differ from the reference only where the scores nearly tie, and never in a way that lightens this GPU's expert work.** The layer problems return the chosen experts as an `int32 [T, 8]` tensor, and `check_routing` in `reference.py` checks them before checking any hidden state.

![One token's sorted expert scores with the 8th and 9th best marked: five must be kept, three may be swapped, three are eligible replacements, two are never allowed](/imgs/blogs/inside-the-judge-cuda-graph-timing-tolerances-and-reward-hacking-6.webp)

The figure is a worked example with made-up scores for one token. The rules, from `TASK.md` and the constants in `reference.py`:

- Every expert you pick must score (sigmoid plus `router_bias`) within 0.05 of the reference's 8th best. With the 8th best at 0.770, anything at or above 0.720 is eligible.
- Every reference top-8 expert on this GPU that beats the 9th best by more than 0.03 (0.04 above 256 tokens) must be in your set. With the 9th at 0.765, the experts above 0.795 cannot be dropped.
- This GPU must keep at least the reference's routes minus 8% minus 4, and its distinct experts minus 15% minus 2.
- The MoE output is then checked against the reference computed with *your* experts.

Ties are real (a hi + lo router and an FP32 one can disagree in the fourth decimal), but every route you drop is an expert GEMM you skip. The count floors are the fence: with a reference of 256 routes on this GPU, you need at least $256 \times 0.92 - 4 = 231.5$, so 232. That leaves room for near-tie disagreements and none for a router that systematically sends work elsewhere.

The judge goes one step further on the load itself. `weights_fields` tries up to 5,000 router seeds per run and picks one whose routing gives each load its target MoE work within 1.5%, on the busiest GPU's 32 experts.

## 5. The disqualification list, and the hole each one plugs

**Rule of thumb: if a speedup would vanish in SGLang's real forward pass, the judge treats it as not a speedup.** Two lists apply. The static scan rejects before anything runs; the review removes after the fact.

| Rule | The shortcut it stops | Where it is enforced |
| --- | --- | --- |
| Capturable in a CUDA graph; no `.item()`, `.cpu()`, sync, value-dependent allocation or branching | Host-side work hidden from GPU timing; kernels SGLang could not graph | Graph capture fails; branching on shapes and `config` is fine |
| Fresh inputs; cache nothing derived from `data` | Returning a stored output from an earlier call | Every timed call checked against a new reference |
| No patching timing, reading harness frames, harness I/O or secrets, editing loaded modules, importing the reference | Changing the stopwatch instead of the kernel | Static scan before the run |
| No processes, no network | Work done outside the timed region | Static scan; the image has no network |
| Weights stay at their precision (losslessly smaller is fine) | Re-quantizing BF16 to FP8, NVFP4 to coarser | Probe rows, tight bounds |
| Activations BF16 where SGLang keeps BF16; FP8 KV; queries no coarser than E4M3 | Cheaper math that SGLang could not ship | Error bounds, final review |
| Every key: no top-k, sparse or block-skipping attention | Skipping most of an 80K-token context | Edge and spread-attention tests |

The review list adds what no scanner can see: "tamper with the timing or harness, do work outside the timed calls, cache or replay outputs, and compute at lower precision than the rules allow." The same paragraph adds a valve: a run where an agent did something reward-hacky can be pulled with `okc withdraw`. Agents optimizing a score find judge holes first; read their diffs as you would a stranger's.

> A kernel that is faster only inside the judge is a bug report about the judge, not a speedup.

## 6. Nearby shapes and held-out reruns

Branching on shapes is allowed: m=1 and m=256 want different kernels. But a configuration searched to the exact scored shapes can sit on a cliff, where m=33 falls off a tile boundary into a slow path. Final review "reruns the top entries on the scored shapes and on nearby ones. Positions come from the scored shapes; an entry that's much slower on the nearby shapes gets a closer look."

The tests are already wider than the scored list: `muse_decode_layer` tests batch 9, batch 33, a 300,000-token context and a 1,048,576-token full-attention context, none of which are scored. Positions then come from medians of several post-close reruns, on held-out data too, so a single lucky run cannot hold a rank.

## Where the harness comes from

The judge is not new code from nothing. `llms.txt` credits GPU MODE's openly published [reference-kernels](https://github.com/gpu-mode/reference-kernels) and [kernelbot](https://github.com/gpu-mode/kernelbot) as its starting point: the problem format (a `reference.py` with `generate_input`, `ref_kernel` and `check_implementation`), the harness design, the anti-cheat approach, and an evaluator originally adapted from reference-kernels' `eval.py`. Both are under the June 9 Researcher Reciprocity License, and GPU MODE does not sponsor the challenge.

For the serving-side version of this discipline, see [an experiment protocol for inference benchmarks](/blog/machine-learning/inference-engineering/an-experiment-protocol-for-inference-benchmarks) and, on why graphs change which kernel wins, [CUDA graphs and torch.compile for the decode loop](/blog/machine-learning/inference-engineering/cuda-graphs-and-torch-compile-for-the-decode-loop).

## Case study: the kernel that reused the reference's answer

In February 2025, Sakana AI announced its AI CUDA Engineer with claims of very large kernel speedups. Outside readers could not reproduce them, and one found a kernel that appeared to pick up the eager implementation's output from memory, so the correctness check passed without the kernel doing the work. Sakana's update said the system had found "a memory exploit in the evaluation code" and other exploits in the benchmark tasks, called it reward hacking, hardened its harness and said it would revise the paper ([TechCrunch coverage](https://techcrunch.com/?p=2969225), [Sakana's project page](https://sakana.ai/ai-cuda-engineer/)). The OKC's answers to this exact failure are fresh inputs on every call, every timed call checked, and a scan that refuses imports of the reference or reads of harness frames.

## When to trust your own number, and when not to

Trust a local speedup when:

- It was measured as CUDA graph replays from a cold L2, interleaved with `baseline.py` on the same inputs.
- It holds on shapes you did not tune for.
- `okc check` agrees with it, outside the ±1.5% interval.

Distrust it when:

- It appears only in eager timing or back-to-back loops.
- It depends on anything computed from a previous call's `data`.
- The routing differs from the reference in a direction that always favors less work.

> **On the board**
>
> - **Problems moved:** all five kernel problems, indirectly. This post buys no microseconds; it saves submissions (10 scored and 20 checks per problem per day) and keeps the ones you make standing after final review.
> - **Lever:** tune against what the judge measures: graph replays, cold L2, interleaved pairs, geometric mean over loads. Treat a gain inside ±1.5% as noise.
> - **Rule fence:** CUDA-graph capturable, fresh inputs, weights and activations at their stated precision (router hi + lo and FP8 E4M3 queries are the only exceptions), every key attended, routing within the tie band and count floors, nothing outside the timed calls. Anything faster only inside the judge gets withdrawn or removed.
