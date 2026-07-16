# Profiling & Optimizing AI Services — Performance Engineering Series Plan

**Subcategory:** `performance-engineering` (NEW folder; auto-registers from frontmatter → renders "Performance Engineering")
**Path:** `content/blog/machine-learning/performance-engineering/<slug>.md`
**subcategory string in frontmatter:** `"Performance Engineering"`
**Category:** `machine-learning`
**Size:** 40 posts, 8 tracks, 8 waves of 4–6. **STATUS: 📝 PLAN — awaiting approval.**
**Language:** English (repo convention; verify gate enforces English-only). Request in VN; figures must be *trực quan* (highly visual/intuitive).
**Reference spine:** PyTorch Deep-Dive tutorials (https://docs.pytorch.org/tutorials/deep-dive.html) — Profiling PyTorch · CUDA Graph Kernel Annotations & Profiling · Inductor CPU Backend Debugging & Profiling · Channels Last Memory Format · Scaled Dot Product Attention.

**Angle (every post, 3-in-1):**
1. **Intuition-first & heavily visual** — a mental model of *where the time / memory / utilization actually goes* before any number. Figures are the point.
2. **Runnable, tool-driven code** — `torch.profiler`, Nsight Systems/Compute (`nsys`/`ncu`), NVTX, `nvidia-smi`/DCGM, CUDA Graphs (`torch.cuda.graph`), `torch.compile`/Inductor, `torch.cuda.memory._record_memory_history`, py-spy, `perf`. Real commands, real output.
3. **A measured before→after resource win** on named hardware (A100 80GB, H100 SXM, L4, RTX 4090, Jetson Orin) — GPU util %, SM occupancy, latency p50/p99, memory MB, throughput, host-overhead ms. Every claim traces to a profile.

**Why a new pillar (not a dupe):**
- `high-performance-computing/` = broad GPU internals + roofline + one profiling post.
- `model-serving/` = *how serving systems work* (batching, KV cache, schedulers) with a couple of profiling posts.
- `distributed-training/` = multi-GPU/node scaling + one `profiling-a-distributed-run`.
- **This series = the profiler-first *performance-debugging craft*** applied to a live AI service (training *and* inference, CPU *and* GPU): profile → hypothesize → fix → measure, over and over, tool by tool. Cross-link OUT to the above rather than re-deriving GPU architecture or serving mechanics.

**Kit:** `.cache/blog-writer/_performance-engineering-series-kit.md` (BUILD in infra step; READ FULLY before writing any post).
**Render helper:** `bash .cache/blog-writer/_render-pe.sh <slug>` (DSL/in.json → scene → PNG → lossless webp into `public/imgs/blogs/`).
**Verify:** `bash .claude/skills/blog-writer/scripts/verify-post.sh <post.md> <slug> deep-dive`
**Gates:** ≥ 6k words floor / target 9–11k; ≥ 5 figures floor / target 7 per post; ≥ 4 distinct figure kinds; ≥ 1–3 animated figures where motion carries meaning (async launch queue filling, capture→replay, memory fragmentation/compaction, GPU timeline gaps closing, roofline point sweeping).

---

## Cross-link spine (every post links these two)
- **Intro:** `why-your-ai-service-wastes-cpu-and-gpu` (Track A #1)
- **Capstone:** `the-performance-engineering-playbook` (Track H #40)
Plus 2–3 sibling links where ideas compose (profiler ↔ roofline ↔ CUDA graphs ↔ compile ↔ memory ↔ CPU ↔ case study).

---

## Track A — Foundations & the optimization loop (Wave 1)
1. `why-your-ai-service-wastes-cpu-and-gpu` — **INTRO.** The four wastes: idle GPU (host-bound), low occupancy (bad kernels), bandwidth wall (memory-bound), redundant work (no fusion/caching). The one-picture map + the profile→hypothesize→fix→measure loop. Why "GPU util 100%" lies.
2. `the-mental-model-of-a-gpu-service` — host (Python/CPU) enqueues kernels onto an async GPU stream; the launch queue; producer/consumer; where a request's time really goes (H2D copy · kernels · D2H · sync).
3. `metrics-that-actually-matter` — util% vs occupancy vs SM efficiency vs MFU; p50/p99 latency vs throughput; allocated vs reserved memory; what each metric lies about and which to trust.
4. `the-roofline-for-your-service` — arithmetic intensity, compute- vs memory-bound, placing *your* kernels on the roofline, why it dictates the fix. (X-link HPC roofline.)
5. `setting-up-a-reproducible-benchmark` — warmup, `torch.cuda.synchronize()`, CUDA events vs wall clock, locking clocks, isolating noise; how not to fool yourself. Grounds every before/after in the series.

## Track B — GPU profiling deep (Wave 2) ← the tools
6. `profiling-pytorch-with-torch-profiler` — `torch.profiler`, `record_function`, the wait/warmup/active schedule, `key_averages()`, exporting a Chrome trace. *(PyTorch: Profiling PyTorch.)*
7. `reading-a-chrome-trace` — the timeline: CPU-op vs CUDA-kernel lanes, gaps = host-bound, launch→execute lag; spotting bubbles, tiny kernels, sync stalls in perfetto.
8. `nsight-systems-for-ai-services` — `nsys profile`, system-wide timeline, CUDA API vs kernel rows, NVTX ranges, finding the wall across CPU+GPU+copies. (X-link model-serving nsight; go deeper on method.)
9. `nsight-compute-kernel-deep-dive` — `ncu`, occupancy, memory throughput, warp-stall reasons, the Speed-of-Light section; one kernel to the metal.
10. `nvtx-and-semantic-profiling-traces` — instrument code with NVTX ranges + CUDA-graph kernel annotations for readable traces; custom lanes; map profiler rows back to your handler. *(PyTorch: CUDA Graph Kernel Annotations & Profiling.)*

## Track C — CUDA graphs & launch overhead (Wave 3)
11. `the-kernel-launch-overhead-problem` — many tiny kernels starve the GPU; ~µs launch latency × thousands of kernels; the CPU-bound signature; measuring launch cost.
12. `cuda-graphs-from-first-principles` — capture vs replay, the graph as a recorded kernel DAG, eliminating per-launch CPU cost; stream-capture API; static-shape/pointer requirement.
13. `cuda-graphs-in-pytorch` — `torch.cuda.graph`, `make_graphed_callables`, the graph pool, warmup iters, static I/O tensors; wiring it into a forward pass. Runnable.
14. `cuda-graphs-gotchas-and-debugging` — dynamic shapes break capture, allocator interactions, sync inside the region, "why did my graph output garbage"; when NOT to graph.
15. `cuda-graphs-in-a-serving-loop` — batch-size bucketing, per-shape graphs, combining with continuous batching; a real inference service host-bound→GPU-bound. (X-link model-serving cuda-graphs.)

## Track D — torch.compile / Inductor (Wave 4)
16. `what-torch-compile-actually-does` — Dynamo capture, guards, Inductor codegen + fusion; the mental model of the stack; what a graph break is.
17. `debugging-graph-breaks` — `torch._dynamo.explain`, `TORCH_LOGS=graph_breaks`, recompilation storms, `fullgraph=True`; fixing the common causes.
18. `inductor-cpu-backend-debugging-and-profiling` — CPU codegen, vectorization, the Inductor CPU profiling workflow, host- vs compute-bound CPU inference. *(PyTorch: Inductor CPU Backend Debugging & Profiling.)*
19. `compile-plus-cuda-graphs-reduce-overhead` — `mode="reduce-overhead"` = compile + CUDA graphs; how they compose; the wins and the memory/dynamic-shape pitfalls.
20. `profiling-compiled-code` — reading a trace of compiled kernels, verifying fusion actually happened, compile-time vs runtime tradeoffs.

## Track E — Memory & data-movement (Wave 5)
21. `the-cuda-caching-allocator` — allocated vs reserved, fragmentation, OOM at 60% used, `PYTORCH_CUDA_ALLOC_CONF`, `expandable_segments`.
22. `memory-snapshot-and-leak-hunting` — `torch.cuda.memory._record_memory_history()`, the snapshot visualizer, retained graphs / caches, the slowly-growing service.
23. `killing-host-device-copies` — pinned memory, `non_blocking=True`, overlap copy with compute, the H2D/D2H tax; measuring copy time in the trace.
24. `channels-last-and-memory-formats` — NCHW vs NHWC, `channels_last`, why layout changes kernel selection + bandwidth; a vision-service win. *(PyTorch: Channels Last Memory Format.)*
25. `bandwidth-bound-and-fusion` — memory-bound kernels, elementwise fusion saving HBM round-trips; SDPA/FlashAttention as the canonical memory-wall fix. *(PyTorch: SDPA.)* (X-link HPC flashattention.)

## Track F — CPU-side & host bottlenecks (Wave 6)
26. `when-the-cpu-is-your-gpu-bottleneck` — the host-bound service: Python overhead, GIL, per-op dispatch; the empty-GPU-timeline signature.
27. `profiling-python-with-py-spy-and-cprofile` — py-spy flamegraphs, cProfile, the hot Python path in a serving handler, async vs sync overhead.
28. `the-dataloader-and-preprocessing-wall` — `num_workers`, `prefetch_factor`, pinned memory, CPU-side tokenization/augmentation as the wall; overlapping the input pipeline with compute.
29. `cpu-affinity-numa-and-threading` — `torch.set_num_threads`, OMP/MKL threads, NUMA placement, oversubscription, pinning; the CPU-contention slowdown in a multi-worker service.
30. `request-batching-and-queueing-overhead` — dynamic batching, scheduler CPU cost, ser/deser, tensor-copy at the request boundary; end-to-end latency accounting.

## Track G — War-story case studies (Wave 7) ← the debugging heart
31. `the-service-at-30-percent-gpu-util` — host-bound inference; profile reveals launch overhead; CUDA graphs + compile; util 30%→85%. Full numbers.
32. `the-memory-leak-that-oomed-every-6-hours` — slow-growing allocation; snapshot finds the retained graph/cache; the fix.
33. `the-p99-latency-tail` — fine p50, ugly p99; a periodic sync / GC / recompile stall; smoothing the tail.
34. `the-dataloader-that-halved-throughput` — GPU idle 40%; the trace shows loader starvation; workers/prefetch/pinning fix.
35. `the-torch-compile-recompilation-storm` — recompiling on every new shape; dynamic-shape guards; bucketing + `dynamic=True`; wall-clock recovered.
36. `the-multi-model-gpu-contention` — several models on one GPU fighting over SMs/memory; MPS/MIG, streams, scheduling; isolating the noisy neighbor.

## Track H — Advanced kernels & capstone (Wave 8)
37. `writing-a-fused-kernel-when-you-must` — profiler says "nothing fuses this"; a Triton/custom-CUDA fused norm/elementwise kernel; the measured win. (X-link model-serving custom-cuda-kernels, HPC cuda-programming.)
38. `overlapping-streams-and-concurrency` — multiple CUDA streams, events, concurrent independent kernels; the concurrency the single-stream default leaves on the table.
39. `the-full-optimization-loop-end-to-end` — one service from naive → profiled → graphed → compiled → memory-tuned → CPU-tuned, with the trace at every step; the compounding wins.
40. `the-performance-engineering-playbook` — **CAPSTONE.** The decision tree: symptom → which profiler → likely cause → fix. The checklist tying the series together. Resolves all forward-links.

---

## Wave → execution map
| Wave | Track | Slugs | Status |
|---|---|---|---|
| 1 | A | 1–5 | ✅ shipped 2026-07-15 (~37.5k words, 35 figs; all gates pass, all 35 figures C2-clean). Agents hit account rate-limit but completed deliverables to disk first (salvage pattern); rendered+verified+committed in main session. |
| 2 | B | 6–10 | ✅ shipped 2026-07-15 (~44.9k words, 35 figs; all gates pass, all 35 C2-clean). Agents finished cleanly (rate-limit reset). One figure (profiler fig3) recast graph→diamond after linear-flow reject. |
| 3 | C | 11–15 | ✅ shipped 2026-07-16 (~45.5k words, 35 figs incl. 2 ANIMATED: launch-queue-draining + capture→replay; all gates + C2 + check-anim + anim-source-review clean). Agents finished cleanly. Fixed 1 graph→diamond earlier-style + req/s glyph-clip in 2 before-afters (→ rps). |
| 4 | D | 16–20 | ⏳ pending |
| 5 | E | 21–25 | ⏳ pending |
| 6 | F | 26–30 | ⏳ pending |
| 7 | G | 31–36 | ⏳ pending |
| 8 | H | 37–40 | ⏳ pending |

Per wave: dispatch parallel agents (one per post) → verify each `.md` on disk (idle ≠ done; resume via SendMessage if empty) → `bash _render-pe.sh <slug>` per post → `verify-post.sh` per post → C2 visual pass → commit only that wave's `content/.../*.md` + `public/imgs/blogs/<slug>-*.webp` (explicit paths, never `git add -A`) → push to main. Update this tracker + MEMORY.md after each wave.

## Cross-links OUT (verify slug exists before linking; else describe)
- `high-performance-computing/profiling-gpu-workloads-finding-the-real-bottleneck`, `.../the-roofline-model-compute-bound-vs-memory-bound`, `.../inside-the-gpu-sms-warps-and-the-simt-execution-model`, `.../the-memory-hierarchy-registers-shared-memory-and-hbm`, `.../kernel-fusion-and-flashattention-beating-the-memory-wall`, `.../cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel`
- `model-serving/profiling-llm-serving-with-nsight`, `.../kernel-fusion-cuda-graphs-torch-compile`, `.../custom-cuda-kernels-for-inference`, `.../continuous-batching-and-pagedattention`, `.../gpu-scheduling-mig-and-autoscaling`, `.../kv-cache-optimization`
- `distributed-training/profiling-a-distributed-run`, `.../overlapping-compute-and-communication`, `.../the-data-pipeline-at-scale`
- `mlops/llm-gpu-benchmark`, `mlops/tensorrt-end-to-end-inference-compiler`
- `edge-ai/the-roofline-model-where-your-bottleneck-lives` (if slug present; else HPC roofline)
