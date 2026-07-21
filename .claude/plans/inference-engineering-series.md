# Inference Engineering — Build Your Own LLM Inference Engine

**Subcategory:** `inference-engineering` (NEW folder; auto-registers from frontmatter → renders "Inference Engineering")
**Path:** `content/blog/machine-learning/inference-engineering/<slug>.md`
**subcategory string in frontmatter:** `"Inference Engineering"`
**Category:** `machine-learning`
**Size:** **65 posts, 11 tracks (A–K), 11 waves of 5–7.** **STATUS: 📝 PLAN — awaiting approval.**
**Language:** English (repo convention; verify gate enforces English-only). Request came in VN; figures must be *trực quan*.

---

## Angle — why this is a new pillar, not a dupe

The repo already has 50 posts in `model-serving/` that **describe** engines other people wrote (vLLM, TGI, Triton, SGLang) and 40 in `performance-engineering/` about the **profiler-first craft**. This series is **constructive**: you write the engine.

A single toy repo — **`nanoserve`** — grows post by post: load weights → forward pass → KV cache → paged blocks → scheduler → samplers → grammar masks → CUDA kernels → tensor parallel → an OpenAI-compatible endpoint. Every post adds a file to it and re-measures. The capstone benchmarks `nanoserve` against vLLM and lists honestly what it still loses.

**Three things in every post:**
1. **Intuition-first & heavily visual** — a mental model of *where the tokens and the microseconds actually go* before any code.
2. **Code you can run** — real Python/CUDA/Triton, not pseudocode. Each post's snippet is a diff on `nanoserve`.
3. **A number with a provenance** — see the honesty rule below. Every claim traces to a formula, a cited public benchmark, or a runnable script.

**Non-negotiable honesty rule (applies to ALL numbers, especially Tracks E & H):**
No post may claim "I ran this and got X". Every number must be one of:
- **(a) Derived** — from a stated formula (KV bytes/token, roofline, arithmetic intensity, $/1M tokens). Show the arithmetic.
- **(b) Cited** — from a public benchmark, model card, paper, or vendor spec, with a link and a date.
- **(c) Reproducible-by-the-reader** — a labeled script plus an *expected range* on named hardware ("on a 4090 you should see ~X–Y tok/s; run `bench.py` to get yours").
Tables carry a `Source` column. This is a hard gate at review time.

---

## Gates (same tier as `performance-engineering`)

- ≥ **6k words** floor / target 9–11k.
- ≥ **5 figures** floor / target **7**; ≥ 4 distinct figure kinds.
- ≥ **1 animated figure** per post where motion carries meaning — KV blocks filling & fragmenting, the continuous-batching queue admitting/evicting, logit mask collapsing a distribution, warps coalescing vs strided reads, prefill chunks sliding, speculative tokens accepted/rejected.
- Verify: `bash .claude/skills/blog-writer/scripts/verify-post.sh <post.md> <slug> deep-dive`
- Kit: `.cache/blog-writer/_inference-engineering-series-kit.md` (BUILD in infra step; READ FULLY before writing any post).
- vLLM grounding refs: `.cache/blog-writer/_inference-engineering-vllm-references.md` — per-track digest of the official vLLM blog (mechanisms, numbers-with-setup, flags, limitations) mined 2026-07-20 covering tracks A–K; every wave's agents READ their track's section and CITE vLLM (benchmark target + contrast), never as first-hand.
- Render helper: `bash .cache/blog-writer/_render-ie.sh <slug>` (DSL/in.json → scene → PNG → lossless webp → `public/imgs/blogs/`).

## Hardware & model matrix (fixed across the series — use these names, don't invent)

- **GPUs:** RTX 4090 24GB (consumer baseline) · L4 24GB (cost tier) · A100 80GB SXM · H100 80GB SXM · Jetson Orin (only where edge is the point, else x-link `edge-ai/`).
- **Models:** Llama-3.1-8B (spine model, used in most posts) · Qwen3-8B & Gemma-3-12B (architecture comparison) · Qwen3-30B-A3B or gpt-oss-20b (MoE on consumer HW) · DeepSeek-V3-family (MLA + big-MoE reference, cited only) · Whisper-large-v3 + a Qwen-VL (multimodal track).
- Same prompt suite everywhere: chat (short in / long out), RAG (long in / short out), code completion, translation. Defined once in Track H post #40 and referenced by slug thereafter.

---

## Cross-link spine (every post links these two)
- **Intro:** `what-inference-engineering-is` (A #1)
- **Capstone:** `the-inference-engineering-playbook` (J #58)
Plus 2–3 sibling links where ideas compose (cache ↔ scheduler ↔ kernel ↔ quantization ↔ experiment ↔ case study).

---

## Track A — From weights to a token (Wave 1)
1. `what-inference-engineering-is` — **INTRO.** The layer map: weights → kernels → engine (cache/scheduler) → decoding layer → API → product. What each existing series owns and what we build here. The `nanoserve` roadmap picture. The honesty rule stated to the reader.
2. `loading-weights-safetensors-dtypes-and-device-placement` — safetensors vs GGUF vs HF pickle, mmap, lazy load, dtype casting cost, sharded checkpoints, load-time vs first-token time, why a 16GB model needs >16GB.
3. `a-forward-pass-by-hand-llama-from-scratch` — implement Llama-3 style forward in pure PyTorch, no `transformers`: RMSNorm, RoPE, GQA attention, SwiGLU, tied/untied embeddings. Verify logits against HF within tolerance — the correctness harness the whole series leans on.
4. `the-tokenizer-boundary-and-incremental-detokenization` — BPE encode/decode, special tokens, chat templates, the *streaming* problem: partial UTF-8, byte fallback, stop strings spanning token boundaries, offset mapping. Tokens-per-word differs per model → so does your $/request.
5. `the-naive-decode-loop-and-your-first-baseline` — the simplest greedy loop, then measure: TTFT, TPOT, tok/s, GPU util. Where the time goes (attention vs MLP vs sampling vs Python). This baseline is what every later post beats.

## Track B — The KV cache you write yourself (Wave 2)
6. `why-recompute-is-fatal-writing-a-kv-cache` — O(n²) recompute vs O(n) with cache; implement a contiguous cache; measure the speedup; the shape/layout choices (`[layer][B, H_kv, S, D]`) and what they cost later.
7. `the-memory-math-of-the-kv-cache` — bytes/token = `2 · layers · H_kv · D · dtype`; worked tables for Llama-3-8B/70B, Qwen3, MLA; how many concurrent users fit in 24GB / 80GB; why GQA was the biggest inference win of 2023.
8. `paged-kv-cache-implementing-blocks-and-a-block-table` — fragmentation in a contiguous cache; block allocator, block table, logical→physical mapping; a mini PagedAttention in Python. **Animated:** fragmentation → paging.
9. `prefix-sharing-radix-trees-and-copy-on-write` — a trie over block hashes; sharing system prompts and few-shot preambles; CoW on `n>1` sampling and on beam/branching; hit-rate math and when it does nothing.
10. `eviction-preemption-and-kv-swapping` — running out of blocks mid-generation: recompute vs swap-to-host vs reject; priority and starvation; the preemption thrash pattern.

## Track C — Batching & the scheduler (Wave 3)
11. `static-batching-and-the-padding-tax` — pad-to-longest waste, head-of-line blocking, why the "obvious" batch loop wastes half the GPU; measuring effective vs wasted FLOPs.
12. `writing-a-continuous-batching-loop` — the `step()` function: waiting queue, running set, admit/finish per iteration; ~60 lines that change everything. **Animated:** the batch as a churning set.
13. `chunked-prefill-and-the-ttft-tpot-tradeoff` — long prefills stall decoders; chunk them; the token-budget knob; the frontier plot of TTFT vs TPOT as chunk size moves.
14. `the-scheduler-as-a-policy-problem` — FCFS vs priority vs fair-share vs SLO-aware; short-job starvation; preemption policy; what "goodput" means and why throughput is the wrong objective.
15. `admission-control-backpressure-and-latency-collapse` — the queueing-theory picture (utilization → latency knee), queue caps, 429s, load shedding; the curve that explains every "it was fine until it wasn't".

## Track D — The decoding layer ⭐ *biggest gap in the repo* (Wave 4)
16. `from-logits-to-tokens-the-sampler-zoo` — temperature, top-k, top-p, min-p, typical-p, repetition/presence/frequency penalties, DRY; what each does to the distribution shape. **Animated:** the distribution collapsing under each knob.
17. `sampling-numerics-determinism-and-batch-invariance` — fp32 softmax, seeds and RNG placement, why the same prompt gives different tokens at different batch sizes, reduction-order nondeterminism, what "deterministic inference" actually requires.
18. `constrained-decoding-from-first-principles-masking-logits-with-an-fsm` — build JSON-mode by hand: schema → FSM → per-step allowed-token mask; the token-vs-character mismatch problem; mask cost on GPU.
19. `grammar-based-decoding-gbnf-pushdown-automata-and-xgrammar` — CFGs beyond regular languages; llama.cpp GBNF, Outlines, xgrammar, llguidance compared on expressiveness + compile time + per-token overhead; caching compiled masks.
20. `structured-output-in-production-streaming-json-and-tool-calls` — streaming partial JSON to a UI, parsing tool calls mid-stream, schema coercion vs retry, and the quality question: does constraining hurt accuracy, and when.
21. `stop-conditions-eos-handling-and-thinking-budgets` — EOS vs stop strings vs max_tokens, the runaway generation, reasoning-effort / thinking-budget control, budget forcing, early-exit on confidence.

## Track E — CUDA & kernels for inference (Wave 5) ← *added per request*
22. `the-inference-kernel-landscape-what-actually-runs` — take one decode step down to the kernel list with nsys/ncu; prefill (GEMM-heavy, compute-bound) vs decode (GEMV-heavy, memory-bound); the ~10 kernels that own your latency. (X-link `performance-engineering/nsight-*`.)
23. `writing-your-first-inference-cuda-kernel-rmsnorm-and-rope` — CUDA C++ + a PyTorch extension; block/warp reductions, vectorized `float4` loads, fusing norm+scale and RoPE into one pass; correctness test against the reference. (X-link `high-performance-computing/cuda-programming-*`.)
24. `the-kv-cache-append-and-gather-kernel` — `reshape_and_cache`: writing new K/V into paged blocks; coalescing under block-table indirection; layout choice (`[block, head, dim]` vs interleaved) and what it does to bandwidth. **Animated:** coalesced vs strided access.
25. `paged-attention-kernel-by-hand` — decode attention for one query against many KV blocks: online softmax, split-K over sequence, partial reductions; why the decode kernel is bandwidth-bound and how close to peak HBM you can get.
26. `gemm-for-decode-the-skinny-matrix-problem` — batch=1 is a GEMV: tensor cores idle, arithmetic intensity ≈ 1, weights dominate traffic; why batching is the only real fix; cuBLAS vs custom for tall-skinny; measuring achieved bandwidth as the honest metric.
27. `dequant-fused-gemm-int4-weights-on-the-fly` — unpack 4-bit weights in registers and feed tensor cores; the Marlin/Machete idea; why weight-only quant is a *bandwidth* win not a FLOP win; where it stops paying (large batch).
28. `triton-for-inference-kernels-and-when-to-stop-writing-cuda` — port the RoPE and the top-k/top-p sampling kernel to Triton, autotune, compare against hand-CUDA; the maintenance argument; a fused GPU sampler that removes a host sync.

## Track F — Precision & compression inside your engine (Wave 6)
29. `weight-only-quantization-in-your-engine-gguf-awq-gptq-at-load-time` — the formats, group size, zero-points, packing; loading a quantized checkpoint into `nanoserve`; memory before/after.
30. `fp8-and-fp4-inference-what-the-hardware-actually-gives-you` — E4M3/E5M2, per-tensor vs per-channel scales, Hopper/Blackwell support matrix, where the advertised 2× does and doesn't materialize.
31. `kv-cache-quantization-fp8-int8-and-the-accuracy-cliff` — quantizing the *cache* rather than the weights; context length gained; per-channel K vs per-token V; where the cliff is.
32. `activation-outliers-calibration-and-measuring-quality-loss` — why some channels break INT8, SmoothQuant/AWQ intuition, calibration sets; and how to *measure* damage: perplexity vs task evals vs vibes, with the eval harness.
33. `cuda-graphs-and-torch-compile-for-the-decode-loop` — the decode loop is host-bound; capture per batch-size bucket; interaction with paged cache pointers and dynamic shapes. (X-link `performance-engineering/cuda-graphs-in-a-serving-loop`.)

## Track G — Big models: parallelism, MoE, long context, multimodal (Wave 7)
34. `tensor-parallel-inference-by-hand` — column/row sharding of QKV, O, MLP; where the all-reduces land; per-token comm cost vs NVLink/PCIe; why TP hurts more at decode than prefill.
35. `pipeline-parallel-and-multi-node-inference` — micro-batching at inference, the bubble, when PP beats TP, KV placement across nodes, network as the new wall.
36. `moe-inference-routing-expert-parallel-and-the-load-imbalance-problem` — active vs total params, top-k routing, expert-parallel all-to-all, hot experts and stragglers, expert offload to CPU/host memory.
37. `mla-and-attention-variants-at-inference-time` — MHA → MQA → GQA → MLA: KV footprint vs quality; DeepSeek's absorb trick; sliding-window and hybrid Mamba layers; how the choice reshapes your cache math (back-refs post #7).
38. `long-context-inference-rope-scaling-sinks-and-the-prefill-cost-curve` — quadratic prefill cost, position-extension methods at inference, attention sinks / StreamingLLM, sparse & trainable-sparse attention at decode, the memory spike that OOMs a node.
39. `multimodal-inference-vlm-image-tokens-and-streaming-audio` — the image encoder as a prefill amplifier, token budgets per image/tile, mm-embedding cache; streaming ASR/TTS latency (chunking, lookahead, first-audio-out).

## Track K — Hybrid attention + SSM / linear-attention inference (Wave 8) ← *added per request*
*Reads as a continuation of Track G; numbered 59–65 so earlier IDs stay stable. Executed BEFORE the experiments track so Track H can benchmark hybrids.*
**Model list — VERIFY every name, version and claim against an official source at write time (assistant cutoff is Jan 2026; do not assert a release that isn't cited):** Nemotron-H / Nemotron Nano · Qwen3-Next and the Qwen 3.5 line · Kimi Linear (KDA) · LFM2 / LFM2.5 · Granite 4.0 · Falcon-H1 · Jamba · MiniMax lightning-attention models · Zamba. If a named model can't be verified, drop it and say so rather than describing it.

**PRIMARY REFERENCE for Track K (read + cite):** vLLM blog, "Disaggregated Serving for Hybrid SSM Models" — https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg (Lucchesi, Hu et al., 2026-04-21). Post-cutoff (Apr 2026); treat every claim as *cited*, quote it as "the vLLM team reports…", never as first-hand. It is the load-bearing source for posts 60, 62 and 63. Concrete, citable material it provides:
- The exact engine problem this track is about: FA layers use a uniform per-token KV layout; SSM layers hold a fixed-size **conv state + temporal state with NO per-token dimension** — so one descriptor/index format cannot address both. This is the "two-cache engine" (post 60) stated in production terms.
- **Dual descriptor views over the same physical memory** (one list indexes FA blocks, another SSM blocks) → heterogeneous-TP transfers with no reshuffle. Grounds post 60's two-allocator design and post 63's disaggregation section.
- **Physical vs logical block bridging**: FA blocks get subdivided to meet a kernel's token granularity (e.g. FlashInfer's 16-token requirement, HMA padding inflating an FA block to 400 tokens); SSM layers have no token dim so they use logical blocks directly. Real detail for post 62 (kernel) + post 60.
- **Conv-state decomposed into 3 sub-projections (x, B, C) + SSM state in a `(dim, state_len)` DS layout** so each decode rank RDMA-reads only its 1/TP slice; padding bytes never transferred (~50 MB/request saved on a bf16 setup, per their Fig 1). Cite as their measured result.
- Benchmark context (CITE, don't claim): `NVIDIA-Nemotron-3-Super-120B-A12B-FP8` on 8×H200, disaggregated P/D (prefill TP=4 + decode TP=4) "Pareto-dominates the co-located baseline at higher batch sizes"; worked config example `Nemotron-3-Nano-30B-A3B-FP8` TP=2 (52 layers alternating Mamba/FA, 5 HMA groups → 6 shared KV tensors).
- **Acknowledged limitations** (use in post 64's "sharp edges" + the honest-verdict framing): Mamba1 unsupported, GDN pending, **speculative-decoding interaction "not extensively validated"** (directly supports post 64's thesis that spec-dec + recurrent state is the hard open problem), mixed-block-size HMA not yet supported. Available vLLM v0.20.0+; `VLLM_SSM_CONV_STATE_LAYOUT=DS`, `NixlConnector` role `kv_both`.

59. `hybrid-models-and-the-end-of-the-kv-cache-assumption` — the landscape: Mamba-2/SSD, gated DeltaNet, gated linear attention, sliding-window+full-attention hybrids; the interleave ratio pattern (1:3, 1:7, "one full-attention layer every N"); and the single fact that reshapes an engine — **recurrent layers carry a fixed-size state, not a cache that grows with context.** The memory curve: O(S) vs O(1) per layer, and where the crossover is. (X-link `large-language-model/nemotron-h-hybrid-mamba-transformer`.)
60. `implementing-a-two-cache-engine-kv-blocks-plus-recurrent-state` — state tensor shape and bytes-per-request per layer; allocating state next to the paged KV blocks; two allocators, two lifetimes; capacity math when half your layers ignore sequence length; what `nanoserve` has to change.
61. `prefill-is-a-scan-decode-is-a-recurrence` — the dual form: chunked/parallel scan (matmul-rich, tensor-core friendly) for prefill vs a single-step state update for decode. Two code paths for one layer; chunk-size tuning; why hybrid prefill stays compute-bound while hybrid decode gets *very* small — and what that does to your batching strategy.
62. `the-selective-scan-and-delta-rule-decode-kernel` — CUDA/Triton for the one-step update (Mamba-2 SSD step, gated delta rule): state resident in registers/SMEM, fusing gates + norm + state write, why this kernel is latency- and occupancy-bound rather than bandwidth-bound. Companion to Track E.
63. `batching-and-scheduling-hybrid-models` — fixed-size state changes admission control (capacity stops depending on context length for those layers); **prefix caching largely dies for recurrent layers** — a state is not sliceable or shareable the way KV blocks are; preemption means save/restore state instead of recompute; what prefill/decode disaggregation means when the thing you hand over is a state vector.
64. `speculative-decoding-and-rollback-with-recurrent-state` — the sharp edge: rejecting k drafted tokens means **rewinding** the SSM state, and you can't rewind by truncating a cache. Checkpoint-and-restore vs re-scan vs keep-k-states; the memory/compute price of each; the same problem hits `n>1` sampling, beam search and copy-on-write branching.
65. `experiment-hybrid-vs-transformer-at-long-context` — under the Track H protocol: memory vs context-length curves, TTFT/TPOT at 4k / 32k / 128k, throughput at fixed VRAM, and the recall-sensitive tasks where hybrids give ground. Honest verdict per workload, every number sourced.

## Track H — Experiments on real models (Wave 9) ← *added per request*
40. `an-experiment-protocol-for-inference-benchmarks` — the template the rest of the track obeys: open- vs closed-loop load, warmup, clock locking, seed control, prompt suite definition, what to report (TTFT/TPOT/tok-s/p99/goodput/$), and how to state provenance. Ships `bench.py`.
41. `experiment-llama-3-8b-on-a-single-4090` — the full sweep: batch × context × dtype grid; memory ceiling vs KV math from post #7; where the roofline says you are; the tok/s curve as batch grows and where it flattens.
42. `experiment-qwen3-vs-gemma-3-vs-llama-3-at-equal-hardware` — architecture diffs that hit inference: KV heads, vocab & tokenizer efficiency (tokens per English/Vietnamese word!), context length, tied embeddings. Same prompts, same GPU, different economics.
43. `experiment-a-moe-model-on-consumer-hardware` — Qwen3-30B-A3B / gpt-oss-20b: active vs total params, VRAM ceiling, expert offload, why "3B active" doesn't mean "3B fast" on 24GB.
44. `experiment-quantization-quality-vs-speed-across-models` — INT8/INT4/FP8 across the model matrix: memory, tok/s, and measured quality (perplexity + 2 task evals). Where the cliff is per family, with sources.
45. `experiment-speculative-decoding-acceptance-rates-in-the-wild` — draft/target pairs, acceptance rate by workload (code ≫ chat > translation), the break-even formula, when spec-dec is a net loss at high batch. (X-link `speculative-decoding/` series.)

## Track I — The API, the platform edge & operations (Wave 10)
46. `designing-an-openai-compatible-inference-api` — the surface, SSE streaming contract, cancellation & client disconnect (and the GPU work you must stop), usage accounting, idempotency. (X-link `api-design/`.)
47. `prompt-caching-semantics-engine-side-and-provider-side` — engine prefix cache vs provider prompt caching: cache keys, prefix stability, TTL, what invalidates, the cost model and how to design prompts *for* the cache.
48. `serving-many-models-on-few-gpus-lora-swapping-and-cold-starts` — multi-LoRA batching, weight swapping, cold-start latency, model cascades and difficulty routing, small→large fallback.
49. `the-cost-model-of-inference-dollars-per-million-tokens` — GPU-hour → tok/s → $/1M tokens, derived; prefill vs decode cost asymmetry; batch economics; self-host vs API break-even, shown as arithmetic not opinion.
50. `reliability-timeouts-retries-hedging-and-degraded-modes` — retries that duplicate GPU work, hedged requests, partial-stream failure, graceful degradation (shorter context, smaller model, no spec-dec).
51. `observability-for-inference-goodput-not-throughput` — per-request token accounting, TTFT/TPOT histograms, queue-time vs compute-time split, cache hit rate, preemption counters; the dashboard that predicts the collapse in post #15.

## Track J — Case studies, hardening & capstone (Wave 11) ← *added per request*
52. `case-study-the-endpoint-that-was-fine-until-32-concurrent-users` — queueing collapse: p99 explodes while GPU util is *flat*; diagnosis via queue-time split; fix = admission control + chunked prefill.
53. `case-study-the-p99-that-doubled-after-turning-on-json-mode` — grammar compile on the request path + per-token mask overhead; fix = mask cache + precompiled schemas + GPU-side masking.
54. `case-study-the-prefix-cache-that-leaked-across-tenants` — shared prefix blocks + timing side channel = cross-tenant inference; isolation keys, hash salting, what to give up for safety.
55. `case-study-the-quantized-model-that-quietly-got-dumber` — INT4 shipped on a perplexity check alone; a task-specific regression found weeks later; the eval gate that should have existed.
56. `case-study-the-long-context-request-that-oomed-the-node` — one 200k-token prefill spikes activation memory and evicts everyone; fix = chunked prefill + context-aware admission + per-request memory budget.
57. `testing-and-hardening-an-inference-engine` — logit-parity tests, golden traces, shape fuzzing, cache-consistency assertions, chaos (kill a rank mid-decode), the CI that keeps `nanoserve` honest.
58. `the-inference-engineering-playbook` — **CAPSTONE.** `nanoserve` vs vLLM on the same prompt suite: an honest gap table (what we're within 20% of, what we're 5× behind on, and why). The symptom → cause → fix decision tree. Build vs buy. Resolves every forward-link.

---

## Wave → execution map
| Wave | Track | Posts | Status |
|---|---|---|---|
| 1 | A | 1–5 | ✅ shipped 2026-07-20 (commit 30a3c738; ~55k words, 35 webp + 7 animated figs; all gates pass; all 5 agents finished cleanly, no salvage needed; honesty-rule grep clean). |
| 2 | B | 6–10 | ✅ shipped 2026-07-20 (commit 9bf435d6; ~54.8k words, 35 webp + 7 animated figs; all gates pass; 66 vllm.ai citations — first wave using the grounding digest). First dispatch died instantly on a session limit (5 writers right after 6 research agents) with zero partial output; a single probe agent confirmed capacity had returned before re-dispatching the rest. |
| 3 | C | 11–15 | ✅ shipped 2026-07-20 (commit 9364a0d9; ~56.7k words, 35 webp + 5 animated figs; all gates pass; 50 vllm.ai citations). All 5 agents finished cleanly. |
| 4 | D | 16–21 | ✅ shipped 2026-07-21 (commit a51577c8; ~72.1k words, 42 webp + 8 animated figs; all gates pass). All 6 agents clean. Post-17 agent web-verified that vLLM's bitwise post does NOT name Thinking Machines → cited separately (+ SGLang as a third source); 3 out-of-digest citations spot-checked verbatim. |
| 5 | E (CUDA) | 22–28 | ⏳ pending |
| 6 | F | 29–33 | ⏳ pending |
| 7 | G | 34–39 | ⏳ pending |
| 8 | K (hybrid attn+SSM) | 59–65 | ⏳ pending |
| 9 | H (experiments) | 40–45 | ⏳ pending |
| 10 | I | 46–51 | ⏳ pending |
| 11 | J (case studies) | 52–58 | ⏳ pending |

Per wave: dispatch parallel agents (one per post) → **verify each `.md` on disk (idle ≠ done; resume via SendMessage if empty)** → `bash _render-ie.sh <slug>` per post → `verify-post.sh` per post → C2 visual pass → **commit only that wave's `content/.../*.md` + `public/imgs/blogs/<slug>-*.webp` (explicit paths, never `git add -A`)** → push to main. Update this tracker + MEMORY.md after each wave.

**Pre-flight per post:** check `git ls-tree HEAD` for the slug before rendering — an already-shipped post must not be clobbered by a same-slug re-render.

## Cross-links OUT (verify slug exists before linking; else describe in prose)
- `model-serving/`: `continuous-batching-and-pagedattention`, `kv-cache-optimization`, `prefix-caching-and-radixattention`, `prefill-decode-disaggregation`, `attention-backends-deep-dive-flashattention-flashinfer`, `custom-cuda-kernels-for-inference`, `grouped-gemm-moe-kernel`, `request-scheduling-and-preemption`, `vllm-deep-dive`, `quantization-for-llm-serving`, `fp8-fp4-low-precision-serving-deep-dive`, `multi-lora-and-adapter-serving`, `observability-for-llm-serving`, `cost-optimization-at-llm-scale`
- `performance-engineering/`: `nsight-systems-for-ai-services`, `nsight-compute-kernel-deep-dive`, `cuda-graphs-in-a-serving-loop`, `the-roofline-for-your-service`, `setting-up-a-reproducible-benchmark`, `what-torch-compile-actually-does`
- `high-performance-computing/`: `cuda-programming-for-ai-engineers-threads-blocks-and-a-first-kernel`, `the-memory-hierarchy-registers-shared-memory-and-hbm`, `kernel-fusion-and-flashattention-beating-the-memory-wall`, `the-roofline-model-compute-bound-vs-memory-bound`, `numerical-formats-and-mixed-precision-fp32-tf32-bf16-fp16-fp8`
- `speculative-decoding/`: all 8 posts (Track H #45, Track F)
- `large-language-model/`: `kv-cache`, `multi-head-latent-attention-mla`, `quantization-in-llm`, `how-quantization-works-gguf-quant-types-decoded`, `bpe-tokenizer`, `optimizing-llm-inference-complete-guide`, `sglang-inference`, `vllm-inference`, `trainable-sparse-attention-nsa-vs-dsa`
- `mlops/`: `tensorrt-end-to-end-inference-compiler`, `onnx-deep-dive-format-runtime-serving`, `llm-gpu-benchmark`
- `edge-ai/`, `api-design/`, `system-design/` for the product edge.

## Anti-dupe rules
- Never re-derive GPU architecture, roofline, or profiler UX — link out, then go straight to the inference-specific consequence.
- Never re-describe vLLM's architecture as documentation; only reference it as the **benchmark target** or as "here's how the real one differs from ours".
- Track K must stay *inference* — Mamba/SSM/linear-attention **training** and architecture design belong to `large-language-model/`; here the subject is the engine (two caches, scan-vs-recurrence, state rollback, scheduling).
- Track E must stay *inference* kernels (decode GEMV, paged attention, KV append, dequant-fused GEMM, sampling). Training kernels and general CUDA teaching belong to HPC.
