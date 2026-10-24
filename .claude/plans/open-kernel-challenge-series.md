# Winning the Open Kernel Challenge — Hopper kernels and SGLang serving for an MoE model

**Subcategory:** `gpu-kernel-engineering` (NEW folder → renders "GPU Kernel Engineering")
**Path:** `content/blog/machine-learning/gpu-kernel-engineering/<slug>.md`
**subcategory string in frontmatter:** `"GPU Kernel Engineering"`
**Category:** `machine-learning`
**Size:** **44 posts, 9 tracks (A–I), 8 waves of 4–6.** **STATUS: 📝 PLAN — awaiting approval (drafted 2026-10-24).**
**Language:** English (repo convention). Length 2,000–3,000 prose words (house rule), 5–7 figures, ≥½ pictorial.

---

## Angle

The challenge (openkernelchallenge.dev — Paradigm × Meta × Andromeda) asks for one thing: make **Muse Spark Open** (816B-A42B MoE, 256 experts top-8, NVFP4 weights, FP8 KV, hybrid sliding/full gated attention) serve faster on **H200**. Two tracks:

- **Kernel track** (Oct 22 → Nov 4, no prize money, top 50 get serving-node priority): 5 problems — `muse_bf16_proj`, `muse_attention_decode`, `muse_moe_w4a16`, `muse_decode_layer`, `muse_prefill_layer` — each one Python file timed against SGLang's own kernel as a CUDA graph on 1×H200.
- **Serving track** (Nov 5 → Nov 19, **$100k**, 1st = $50k + H200): a `git diff` to SGLang `a55ab7d5`, scored as tokens/s/GPU at stock SGLang's interactivity on 8×H200, concurrency 1/4/16/32, 220-token agentic-coding turns, DFlash drafter fixed.

The series is the **curriculum for top 1**: every post maps to a skill the competition actually tests, and every post ends with a *"On the board"* box — which problem/track it moves, and what the rule forbids. It is a case-study series built on one concrete model and one concrete harness, which is what separates it from the generic siblings.

**Relationship to existing posts — link, never re-derive (hard rule):**
| Existing post | Covered there; here we only apply it |
|---|---|
| `inference-engineering/gemm-for-decode-the-skinny-matrix-problem` | skinny GEMM basics → #18 |
| `inference-engineering/dequant-fused-gemm-int4-weights-on-the-fly` | dequant-in-kernel basics → #13, #25 |
| `inference-engineering/paged-attention-kernel-by-hand` | paged attention basics → #20–21 |
| `inference-engineering/moe-inference-routing-expert-parallel-and-the-load-imbalance-problem` | MoE/EP basics → #26–27 |
| `inference-engineering/fp8-and-fp4-inference-what-the-hardware-actually-gives-you` | format overview → #13–14 |
| `inference-engineering/triton-for-inference-kernels-and-when-to-stop-writing-cuda`, `cuda-graphs-and-torch-compile-for-the-decode-loop` | → #11, #31 |
| `inference-frameworks/sglang-inference`, `attention-backends-deep-dive-flashattention-flashinfer` | → #24, #34 |
| `frontier-models/deepseek-open-infra-deepep-deepgemm-flashmla` | DeepGEMM/DeepEP → #27, #38 |
| `large-language-model/dflash-block-diffusion-speculative-decoding`, `paper-reading/large-language-model/dflash-block-diffusion-drafter-v1-v2` | DFlash mechanism → #36 |
| `high-performance-computing/inside-the-gpu-sms-warps-and-the-simt-execution-model` | SM basics → #7 |

**Honesty rule (same as inference-engineering):** no "I ran this". Every number is (a) **from the challenge's public API** (`GET /api/v1/problems/{p}` → `task_md`, `sglang_us`, scored shapes, SGLang baseline source; `GET /serving` → launch args, rungs), (b) **derived** from a stated formula (roofline bytes/FLOPs for the stated shapes on H200: 4.8 TB/s HBM3e, ~989 TFLOPS dense BF16), or (c) **cited** with link + date (leaderboard snapshots, papers, SGLang source at the pinned commit). No competitor code reproduced; leaderboard figures quoted as dated snapshots only.

**Primary sources for drafters (no login needed):**
- `https://openkernelchallenge.dev/llms.txt` — rules, restrictions, CLI, prizes
- `https://openkernelchallenge.dev/api/v1/problems/<name>` — spec, TASK.md, AGENTS.md, starter + baseline files, `sglang_us`
- `https://openkernelchallenge.dev/api/v1/serving` — pinned commit, image, patch rules, rungs, launch args
- SGLang @ `a55ab7d5` (github.com/sgl-project/sglang), Marlin, FA3, CUTLASS/CuTe DSL, TileLang, DeepGEMM sources
- Snapshot baseline_us (2026-10-24): bf16_proj 4830.8 · moe_w4a16 13448.5 · attention_decode 3766.3 · decode_layer 22620.3 · prefill_layer 387403.9

**Recurring visual language (defined in #3, reused):**
- **The layer strip** — one decoder layer on one GPU as a horizontal timeline (proj → attn → glue → router → experts → glue), each block's width = its µs; every post highlights the block it attacks.
- **The roofline card** — H200 roofline with the problem's shapes plotted as points (m=1,8,16,32,128,256).
- **The rule fence** — what the post's technique may/may not do under the challenge rules (precision, CUDA-graph, no sparse attention).

---

## Track A — The challenge and the model (Wave 1)
1. `the-open-kernel-challenge-decoded` — **INTRO.** Two tracks, why the money is on the serving board, the scoring (speedup vs SGLang as CUDA graph; throughput at fixed interactivity), limits (10/day kernels, 1/day serving), validation by median reruns on held-out shapes. The map of the whole series.
2. `muse-spark-open-for-kernel-engineers` — the model as a kernel engineer sees it: 256 experts top-8, sigmoid router with selection-only bias + renormalize, post-expert norm, SwiGLU, gated attention with QK-norm, 2,048 sliding window + full layers, GQA 16q/2kv per GPU under TP8/EP8, NVFP4 + FP8 KV.
3. `one-layer-on-one-h200-where-the-microseconds-go` — the layer strip from `sglang_us`; which block dominates at decode vs prefill; Amdahl across the five problems; why decode_layer ≠ sum of parts.
4. `roofline-for-decode-on-h200-bytes-not-flops` — compute bytes and FLOPs for every scored shape; which problems are bandwidth-bound (all decode) and which approach compute-bound (prefill grouped GEMM); the speed-of-light bound for each.
5. `inside-the-judge-cuda-graph-timing-tolerances-and-reward-hacking` — how the harness times (interleaved, same inputs, CUDA graph), each problem's tolerance, the routing check, nearby-shape reruns, and the list of things that get you disqualified. Lineage from GPU MODE reference-kernels.
6. `setting-up-the-lab-okc-sglang-image-and-nsight` — okc workflow (init/check/submit), the `lmsysorg/sglang:dev-muse-spark-open` image, reproducing the timing locally, Nsight Systems + Compute in a container, an iteration loop that respects daily caps. **Include the no-H200 path:** developing on a different-architecture GPU (e.g. 2×L40S, Ada SM89: no WGMMA/TMA/FA3, ~864 GB/s vs 4.8 TB/s) for correctness + CUDA-graph capture + `checks.py`, while using `okc check` runs as the remote H200 timer; why tuned parameters don't transfer; why the serving track's ~450 GB model can't run locally and how to test patch logic on a small MoE instead.

## Track B — Hopper (SM90) craft (Wave 2)
7. `hopper-sm90-for-kernel-writers-wgmma-tma-clusters` — what changed from Ampere: async warpgroup MMA, TMA bulk copies, thread-block clusters + distributed shared memory, and why naive mma.sync code leaves half the chip idle.
8. `warp-specialization-producer-consumer-pipelines` — producer warps feeding consumer warpgroups through mbarrier rings; ping-pong vs cooperative schedules; the pattern behind FA3, Marlin-on-Hopper and DeepGEMM.
9. `cute-layouts-the-mental-model` — shape/stride algebra, tiling, partitioning, swizzles; reading CUTLASS 3 code without fear.
10. `cute-dsl-a-hopper-gemm-in-python` — building a WGMMA+TMA GEMM in CuTe DSL step by step; JIT inside a single submission file.
11. `triton-tilelang-cuda-on-hopper-when-each-wins` — the same kernel in three tools; what each exposes (TMA, warp specialization, layouts); a decision table per challenge problem. `load_inline` mechanics.
12. `reading-an-nsight-compute-profile` — memory throughput vs SOL, achieved occupancy, stall reasons, bank conflicts, tail effects; a profile-reading checklist for decode kernels.

## Track C — Numerics under the rules (Wave 3)
13. `nvfp4-on-a-gpu-with-no-fp4-tensor-cores` — E2M1 values, FP8 E4M3 block-16 scales, FP32 global scale; why Hopper must dequant to BF16 in registers; cost of the dequant per byte loaded; lossless repack allowed, precision change forbidden.
14. `fp8-e4m3-kv-cache-scales-and-in-kernel-dequant` — E4M3 range, per-tensor scales (1 scored, 1/64 in tests), FP8 queries as FA3 does, and why nothing coarser is allowed.
15. `accumulation-error-and-tolerance-budgets` — BF16 inputs + FP32 accumulate, split-K reduction order, the "2.5× reference's own error + 1.5%" rule and how to stay inside it with margin.
16. `the-router-is-the-precision-trap` — 0.01% tolerance, why outputs that choose experts are fragile, the BF16 hi+lo split with FP32 accumulation, tie-breaking and the top-8 routing check.
17. `lossless-weight-repacking-in-prepare` — using the untimed 300 s `prepare()`: Marlin layouts, interleaving, scale reordering, fusing weight matrices, what "lossless" means to the judge.

## Track D — Dense projections + attention decode (Wave 4)
18. `skinny-bf16-gemm-on-hopper-split-k-and-stream-k` — `muse_bf16_proj` at m=1..256; bandwidth-bound GEMV vs small GEMM; split-K, stream-K, tile choice per m.
19. `fusing-the-projections-into-one-launch` — qkv_gate, o, router, pooler as one kernel or grouped launch; reading weights once; the router's FP32 path inside.
20. `flash-decoding-split-kv-for-80k-contexts` — split the 80K context across SMs, reduce with LSE merge; full vs sliding layers have opposite parallelism problems.
21. `gqa-head-packing-and-a-paged-fp8-cache-with-page-size-one` — 8 query heads share one KV head: load KV once; page-size-1 gather patterns; coalescing tricks.
22. `fusing-qk-norm-rope-and-the-sigmoid-gate-into-attention` — the gated, normalized output path; what to fuse before/after the softmax loop.
23. `speculative-verify-blocks-8-token-causal-decode-attention` — L=8 verify steps change the arithmetic intensity; causal-inside-block masking; batching queries per KV load.

## Track E — Routed experts, the biggest block (Wave 5)
24. `inside-fa3-and-sglangs-attention-baseline` — what the baseline actually does (FA3 schedule metadata, FP8 path) and where it leaves time on the table. (Closes the attention track.)
25. `marlin-w4a16-anatomy` — the Marlin kernel: weight layout, in-register dequant, async pipeline, why it wins at small m and where it stops.
26. `moe-routing-kernels-sigmoid-topk-sort-and-align` — fused routing, `moe_align_block_size`, building per-expert token lists for only the 32 local experts, all CUDA-graph-safe.
27. `grouped-gemm-under-skewed-routing` — the busiest-GPU rule; load balancing across SMs when two experts take most tokens; persistent tile schedulers.
28. `fusing-swiglu-post-expert-norm-and-weighted-sum` — up/gate → SwiGLU → down → norm → weight → sum in as few passes as possible; epilogue fusion.
29. `the-prefill-grouped-nvfp4-gemm-is-compute-bound` — `muse_prefill_layer`: 753-token turns and 16K chunks; WGMMA-saturating dequant GEMM with BF16 activations; dequant cost hidden behind MMA.

## Track F — The whole layer (Wave 6)
30. `glue-kernels-norms-residuals-and-why-they-add-up` — the microseconds hiding between big kernels; fusing RMSNorm, residual add, gate.
31. `launch-overhead-pdl-and-cuda-graphs` — programmatic dependent launch, graph-captured decode, what's left after the graph.
32. `toward-a-decode-megakernel` — persistent single-kernel layer: promise, scheduling, and why few teams ship it; partial-megakernel compromise.
33. `prefill-layer-chunked-attention-plus-the-big-gemm` — assembling `muse_prefill_layer`: varlen attention over cached context + the expert GEMM; balancing the two.

## Track G — The serving track: SGLang (Wave 7)
34. `sglang-at-the-pinned-commit-from-scheduler-to-kernel` — request lifecycle in SGLang `a55ab7d5` for Muse Spark Open: scheduler, overlap, CUDA graph runner, MoE runner (`marlin`), attention backend; where a patch can plug in.
35. `throughput-at-fixed-interactivity-reading-the-serving-score` — the tokens/s/GPU vs tokens/s/user curve, the four rungs, TTFT ≤ 2× guard, output-agreement check; what kinds of wins move the score and which don't.
36. `dflash-in-sglang-making-the-drafter-faster-without-changing-it` — DFlash's 8-token block drafts, verify cost, acceptance; optimizing drafter runtime within the rules.
37. `prompt-lookup-and-ngram-drafting-for-agentic-coding` — the allowed extra drafting: n-grams from the conversation; why agentic coding (code echoes, tool output) is prime territory; combining with DFlash.
38. `tp8-ep8-communication-on-one-h200-node` — all-reduce / all-gather / dispatch-combine over NVLink, custom all-reduce, overlapping comm with compute, two-batch overlap.
39. `radix-and-hierarchical-cache-for-multi-turn-agents` — 80K-token conversations, prefix hits per turn, chunked prefill 16384, scheduling prefill vs decode at concurrency 32.

## Track H — Shipping and strategy (Wave 8)
40. `shipping-jit-kernels-into-sglang-without-rebuilding-sgl-kernel` — patch rules (python/sglang only, no binaries, 720 KB, 400-char lines); integrating Track D–F kernels via Triton/TileLang/CuTe DSL/load_inline at runtime; warmup and graph capture.
41. `iterating-with-one-scored-run-per-day` — experiment design under caps: local 8×H200 replica, check runs, variance and medians, deciding what to submit.
42. `real-speedups-vs-gaming-the-judge` — catalog of disqualifying patterns (timing tampering, out-of-timer work, cached outputs, lower precision) and why upstreamable is the north star.

## Track I — Capstone (Wave 8, after Nov 20)
43. `open-kernel-challenge-post-mortem` — **write only after final results** + public writeups: what the winners did, cited; which posts in this series predicted it.
44. `a-top-1-playbook-for-kernel-and-serving-challenges` — **CAPSTONE.** The ordered checklist from all 42 posts; transferable to GPU MODE and future challenges.

---

## Waves
| Wave | Posts | Note |
|---|---|---|
| 1 | #1–6 | foundation; #3 defines the layer strip |
| 2 | #7–12 | Hopper craft |
| 3 | #13–17 | numerics |
| 4 | #18–23 | proj + attention |
| 5 | #24–29 | attention baseline + MoE |
| 6 | #30–33 | whole layer |
| 7 | #34–39 | serving track — **prioritize before Nov 5 if the user competes** |
| 8 | #40–42, then #43–44 after 2026-11-20 | shipping + capstone |

**Time-sensitivity:** kernel track closes 2026-11-04, serving opens 2026-11-05. If the series is meant to support live participation, run Waves 1 → 7 → 5 first (orientation, serving, MoE are worth the most); otherwise run in order.

## Gates
- 2,000–3,000 prose words; 5–7 figures, ≥½ pictorial, box-flow ≤⅓ (max 2); no em dash.
- Verify: `bash .claude/skills/blog-writer/scripts/verify-post.sh <post.md> <slug>` via `post-verifier`.
- Each post ends with the **"On the board"** box (problem/track, expected lever, rule fence).
