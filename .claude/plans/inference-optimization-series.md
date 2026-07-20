# The Inference Optimization Handbook — every technique, what it costs, how they stack

**Subcategory:** `inference-optimization` (NEW folder → renders "Inference Optimization")
**Path:** `content/blog/machine-learning/inference-optimization/<slug>.md`
**subcategory string in frontmatter:** `"Inference Optimization"`
**Category:** `machine-learning`
**Size:** **46 posts, 9 tracks (A–I), 9 waves of 3–6.** **STATUS: 📝 PLAN — awaiting approval.**
**Language:** English (repo convention). Figures must be *trực quan*.

---

## Angle — a handbook, not another pile of technique explainers

The repo already explains many individual techniques (`quantization-in-llm`, `pruning-in-llm`, `distillation-in-llm`, the 8-post `speculative-decoding/` series, 50 posts of `model-serving/`, 50 of `edge-ai/`). **This series does the thing none of them do:**

1. **A taxonomy** — every technique placed on one map: *which resource it buys back* (latency / throughput / memory / cost) × *which stage it acts at* (architecture → weights → runtime → kernel → system).
2. **A trade-off card per technique** — a fixed format: mechanism (1 figure) → what it saves → **what it costs** (quality, memory, complexity, engineering time) → when it wins → **when it loses** → interactions.
3. **Composition** ⭐ — the part nobody writes: which techniques stack multiplicatively, which cancel each other, and in what **order** to apply them. Quantization changes speculative-decoding acceptance. Batching erases the CUDA-graph win. Prefix caching makes chunked prefill matter less. Track G is the heart of the series.
4. **Recipes** — four end-to-end stack-ups for four different objectives, each with a measured step-by-step ledger of what every technique actually bought.

**Relationship to the sibling series `inference-engineering/`:**
- `inference-engineering/` = **how it works, write the code** (you implement paged attention, a scheduler, a CUDA kernel).
- `inference-optimization/` = **which lever to pull and what it costs** (you decide between INT4 and spec-dec given a p99 budget and a quality budget).
- Every technique card links to its mechanism post in the engineering series instead of re-deriving it. Hard rule.

**Honesty rule (identical to the sibling series):** no "I ran this". Every number is **(a) derived** from a stated formula, **(b) cited** with link + date, or **(c) reproducible** via a named script with an expected range on named hardware. Trade-off tables carry a `Source` column.

---

## Gates (same tier as `performance-engineering` / `inference-engineering`)
- ≥ **6k words** floor / target 9–11k · ≥ **5 figures** floor / target **7** · ≥ 4 distinct kinds · ≥ **1 animated figure** where motion carries meaning (a technique card's before/after resource bar, tokens dropping out of a KV cache under eviction, the stack-up ledger filling in).
- Verify: `bash .claude/skills/blog-writer/scripts/verify-post.sh <post.md> <slug> deep-dive`
- Kit: `.cache/blog-writer/_inference-optimization-series-kit.md` · Render: `bash .cache/blog-writer/_render-io.sh <slug>`

**The recurring visual language (define once in post #2, reuse in all 43):**
- The **resource radar** — 5 axes (latency, throughput, memory, quality, complexity) drawn for each technique.
- The **stage map** — where the technique sits: architecture → weights → runtime → kernel → system.
- The **win/lose band** — batch size on x, benefit on y; every technique has a region where it stops paying.

---

## Track A — The framework (Wave 1)
1. `the-inference-optimization-map` — **INTRO.** All ~40 techniques on one picture: resource bought × stage acted on. Why "make it faster" is four different questions. How to use this handbook.
2. `how-to-read-a-technique-card` — the fixed format, the resource radar, the win/lose band, the honesty rule. The template every later post fills in.
3. `find-your-bottleneck-before-you-optimize` — the decision tree: memory-capacity-bound? bandwidth-bound? compute-bound? host-bound? queue-bound? Each answer eliminates two-thirds of the handbook. (X-link `performance-engineering/`.)
4. `set-a-quality-budget-first` — you cannot optimize without an eval: perplexity vs task evals vs human preference, acceptable degradation as an explicit number, the regression gate. The post that prevents Track J's "quietly got dumber" story.
5. `why-optimization-wins-dont-add-up` — Amdahl at the technique level; the fraction-of-time-affected argument; overlapping wins; the honest way to report "3.2× total" when four techniques each claimed 2×.

## Track B — Architecture-level techniques (Wave 2)
6. `attention-variants-as-an-inference-lever-mha-mqa-gqa-mla` — KV footprint vs quality; the single biggest architectural inference decision; what you can and can't retrofit.
7. `sparse-and-windowed-attention-at-inference` — sliding window, attention sinks / StreamingLLM, block-sparse, trainable sparse (NSA/DSA); the long-context economics.
8. `distillation-as-an-inference-optimization` — the "just use a smaller model" baseline every other technique must beat; task-specific distillation; when a 3B student wins outright.
9. `pruning-that-actually-speeds-things-up` — structured (heads/channels/layers/depth) vs unstructured; 2:4 semi-structured on tensor cores; why 50% sparsity ≠ 2× speed.
10. `moe-as-sparse-computation-at-inference` — active vs total params; the memory-for-FLOPs trade; routing cost, load imbalance, expert offload; when MoE is a *pessimization* on one GPU.
11. `adaptive-computation-early-exit-and-layer-skipping` — confidence-based exit, depth-adaptive decoding, cascades inside one model; why it's rare in production and what would change that.

## Track C — Weight & numeric-level techniques (Wave 3)
12. `the-quantization-decision-tree` — PTQ vs QAT, weight-only vs W8A8 vs full low-precision, granularity (per-tensor/channel/group), what each buys on which bottleneck.
13. `int8-and-int4-methods-compared-gptq-awq-smoothquant-hqq` — the mechanisms side by side, calibration needs, accuracy at equal bits, ecosystem/kernel support as a real constraint.
14. `fp8-and-fp4-hardware-native-low-precision` — E4M3/E5M2, scaling recipes, the Hopper/Blackwell support matrix; where the advertised speedup evaporates.
15. `kv-cache-compression-quantization-eviction-and-low-rank` — FP8/INT8 cache, H2O/SnapKV-style eviction, low-rank/MLA-style compression; context length gained per point of quality lost.
16. `shrinking-the-non-weight-costs-vocab-embeddings-and-tokenizers` — embedding/lm-head memory at large vocabularies, vocab trimming, tokenizer efficiency as a *token-count* optimization (huge for Vietnamese and code).
17. `measuring-quality-loss-honestly` — perplexity's blind spots, task evals that catch what perplexity misses, long-tail regressions, the A/B that decides ship or not. Pairs with post #4.

## Track D — Runtime & decoding-level techniques (Wave 4)
18. `the-batching-family` — static, dynamic, continuous, chunked-prefill; batching as the technique that reshapes every other technique's payoff. (X-link engineering series.)
19. `the-caching-family-exact-prefix-and-semantic` — response cache vs prefix/KV cache vs semantic cache; hit-rate math, staleness, correctness risk, cost-per-hit.
20. `the-speculative-family-draft-medusa-eagle-lookahead-mtp` — one framework, five instantiations; acceptance rate as the master variable; the break-even formula; where each variant wins.
21. `scheduling-and-disaggregation-as-optimization` — priority, preemption, prefill/decode disaggregation, SLO-aware admission; the "free" wins that need no model change.
22. `generate-fewer-tokens-the-most-underrated-optimization` — output length is linear in cost: budget forcing, thinking budgets, stop conditions, structured output as a length cap, prompt design for terseness.
23. `offloading-cpu-nvme-and-layer-streaming` — when the model doesn't fit: weight offload, KV offload, layer streaming, PCIe as the ceiling; the honest latency price.

## Track E — Compilation & kernel-level techniques (Wave 5)
24. `graph-compilers-compared-tensorrt-llm-torch-compile-onnx-runtime` — what each actually optimizes, build-time vs run-time cost, portability vs peak; when the compiler beats hand-tuning.
25. `kernel-fusion-and-attention-backends` — FlashAttention/FlashInfer/SDPA, fused norm+residual+RoPE, paged-attention kernels; the memory-traffic argument in one figure.
26. `cuda-graphs-and-the-launch-overhead-family` — the host-bound signature; graphs, bigger kernels, fewer Python ops; **and the win/lose band: why this vanishes at large batch.**
27. `memory-layout-and-data-movement-tricks` — cache layouts, pinned memory & overlap, avoiding host syncs (a GPU-side sampler), contiguity vs paging.
28. `hardware-as-an-optimization-choosing-and-partitioning` — GPU selection by bandwidth-per-dollar, MIG partitioning, CPU inference viability, accelerators (Inferentia/TPU/Gaudi) — what changes about your technique stack on each.

## Track F — System & product-level techniques (Wave 6)
29. `routing-and-model-cascades` — difficulty routing, small→large escalation, the quality/cost frontier, the router's own latency and error cost.
30. `semantic-caching-and-request-deduplication` — embedding-based cache, threshold tuning, the false-hit failure mode, dedup of concurrent identical requests.
31. `prompt-and-context-optimization` — system-prompt size as a recurring bill, prefix-stability design for cache hits, context compression/summarization, RAG chunk budgets.
32. `batch-and-async-tiers` — offline/batch APIs, priority classes, time-shifting load into cheap capacity; the throughput-first stack that ignores latency entirely.
33. `autoscaling-capacity-and-the-cost-frontier` — replicas vs bigger batches, cold starts, spot/preemptible economics, the utilization target that minimizes $/token without breaking p99.

## Track G — Composition: how techniques interact ⭐ *the core of the series* (Wave 7)
34. `the-interaction-matrix` — a full N×N table: which pairs stack multiplicatively, which are redundant, which actively fight. The reference figure of the whole handbook.
35. `quantization-meets-speculative-decoding` — a quantized draft drifts from the target → acceptance falls; quantized target changes the verification distribution; the combined-speedup arithmetic that is *not* a product.
36. `batching-versus-the-latency-techniques` — CUDA graphs, spec-dec, and small-batch kernels all decay as batch grows; the win/lose bands overlaid on one plot; the single most common stacking mistake.
37. `caching-changes-what-else-is-worth-doing` — a 70% prefix-cache hit rate makes prefill optimizations nearly worthless and shifts everything to decode; re-deriving your bottleneck *after* every applied technique.
38. `the-stacking-order` — the recommended sequence (measure → free system wins → runtime → weights → kernels → architecture), why doing it backwards wastes weeks, and the re-measure loop between steps.

## Track I — Hybrid attention + SSM models (Wave 8) ← *added per request*
*Numbered 44–46 so earlier IDs stay stable; reads after Track G because it is fundamentally an interaction story.*
**Models (verify every name/version/claim against an official source at write time — cutoff Jan 2026; drop what you can't cite):** Nemotron-H / Nemotron Nano · Qwen3-Next and the Qwen 3.5 line · Kimi Linear (KDA) · LFM2 / LFM2.5 · Granite 4.0 · Falcon-H1 · Jamba · MiniMax lightning attention · Zamba.

**PRIMARY REFERENCE (read + cite, post-cutoff so treat as cited not first-hand):** vLLM blog, "Disaggregated Serving for Hybrid SSM Models" — https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg (2026-04-21). Load-bearing for post 45 (what breaks) and post 46 (production recipe): FA layers carry a per-token KV layout while SSM layers hold a fixed-size conv+temporal state with no token dimension, so one cache format can't serve both; the fix is dual descriptor views + physical/logical block bridging; ~50 MB/request of padding transfer eliminated; and their own caveat that **speculative-decoding interaction is "not extensively validated"** on hybrids — perfect for the interaction-matrix's hybrid column.

44. `hybrid-attention-ssm-models-as-a-memory-lever` — the technique card. **Buys:** memory flat in context length for the recurrent layers, cheap decode, longer contexts on the same GPU. **Costs:** in-context recall / retrieval-style tasks, immature kernel + quantization ecosystem, fewer serving-stack features. The win/lose band by context length and workload; the interleave ratio (1:3, 1:7) as the tuning knob; how to decide without retraining anything.
45. `what-breaks-when-half-your-layers-are-recurrent` — **the interaction post, and the reason this track exists.** Prefix/prompt caching, KV quantization, paged memory, prefill–decode disaggregation, preemption-by-recompute and speculative decoding all silently assume a *sliceable, shareable, truncatable* KV cache. Go through them one by one: which survive unchanged, which need redesign (state checkpoint/rollback), which become pointless (KV-cache compression on layers that have no KV cache), and which get *better*. Extends the Track G interaction matrix with a hybrid column.
46. `recipe-serving-a-hybrid-model-in-production` — choosing a hybrid (size, ratio, ecosystem support), the optimization stack that actually survives, the evals that catch recall regressions before users do, and migrating a live transformer deployment: what capacity planning looks like when memory stops scaling with context.

## Track H — Recipes & capstone (Wave 9)
39. `recipe-a-local-assistant-on-one-24gb-gpu` — objective: fit + feel fast at batch 1. The chosen stack, the rejected techniques, the ledger of each step's gain, the final quality check.
40. `recipe-a-cost-first-batch-pipeline` — objective: $/1M tokens. Huge batches, quantization, offline tier; every latency technique deliberately dropped.
41. `recipe-a-latency-first-interactive-api` — objective: p99 TTFT/TPOT under an SLO. Disaggregation, spec-dec, admission control, capacity headroom; the things you pay for to protect the tail.
42. `recipe-edge-and-on-device` — objective: it runs at all. Distill → quantize → prune → compile; the technique set that survives without a datacenter GPU. (X-link `edge-ai/`.)
43. `the-inference-optimization-playbook` — **CAPSTONE.** The full decision tree from symptom to technique; the one-page card deck; the "you are probably here" flowchart; when to stop optimizing. Resolves all forward-links.

---

## Wave → execution map
| Wave | Track | Posts | Status |
|---|---|---|---|
| 1 | A framework | 1–5 | ⏳ pending |
| 2 | B architecture | 6–11 | ⏳ pending |
| 3 | C weights/numerics | 12–17 | ⏳ pending |
| 4 | D runtime/decoding | 18–23 | ⏳ pending |
| 5 | E compile/kernel | 24–28 | ⏳ pending |
| 6 | F system/product | 29–33 | ⏳ pending |
| 7 | G composition ⭐ | 34–38 | ⏳ pending |
| 8 | I hybrid attn+SSM | 44–46 | ⏳ pending |
| 9 | H recipes/capstone | 39–43 | ⏳ pending |

Per wave: dispatch parallel agents (one per post) → verify each `.md` on disk (idle ≠ done; SendMessage to resume) → render → `verify-post.sh` → C2 visual pass → commit only that wave's explicit paths → push. Update this tracker + MEMORY.md per wave.

## Anti-dupe rules (strict — this series sits next to 4 overlapping pillars)
- **Never** re-derive a mechanism that has a post already. Open each card with a 1-figure recap + a link, then spend the post on **cost, boundaries, interactions, and evidence**.
- Existing posts to link rather than repeat: `large-language-model/{quantization-in-llm, pruning-in-llm, distillation-in-llm, how-quantization-works-gguf-quant-types-decoded, multi-head-latent-attention-mla, kv-cache, past-4-bit-wall-frontier-llm-quantization, trainable-sparse-attention-nsa-vs-dsa, optimizing-llm-inference-complete-guide}`, the whole `speculative-decoding/` series, `model-serving/{quantization-for-llm-serving, fp8-fp4-low-precision-serving-deep-dive, continuous-batching-and-pagedattention, prefix-caching-and-radixattention, prefill-decode-disaggregation, cost-optimization-at-llm-scale}`, `edge-ai/*`, `performance-engineering/*`, and the sibling `inference-engineering/*`.
- If a card can't say anything new about *cost / boundary / interaction*, it does not deserve a post — fold it into a neighbouring card.
- Track G posts may not exist as standalone technique explainers anywhere else; they are the series' reason to exist and must be written with the most care.
