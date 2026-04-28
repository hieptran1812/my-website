---
title: "Pretraining Large Reasoning Models: A Senior Engineer's Deep Dive"
publishDate: "2026-04-28"
category: "machine-learning"
subcategory: "Large Language Model"
tags:
  [
    "llm",
    "reasoning",
    "pretraining",
    "scaling-laws",
    "long-context",
    "rlvr",
    "data-curation",
    "muon",
    "deep-learning",
  ]
date: "2026-04-28"
author: "Hiep Tran"
featured: true
excerpt: "What changes when you pretrain a model that has to think — not just predict. A practitioner's tour of data curation for reasoning, pretraining objectives beyond next-token, mid-training and long-context extension, RLVR transitions, scaling laws specific to reasoning, and the failure modes that only appear at trillion-token scale."
---

The era of "scale up next-token prediction and hope" is over. Frontier reasoning models — o1, DeepSeek-R1, Qwen3-Thinking, Claude with extended thinking, Gemini 2.5 Deep Think — share a common shape: a strong pretrained base, a deliberate mid-training stage that shifts the data mixture toward reasoning, and a post-training loop dominated by reinforcement learning from verifiable rewards (RLVR). Pretraining is no longer a single monolithic phase; it is a staged pipeline whose every decision has downstream consequences for whether the model can think.

This article is a senior practitioner's tour of pretraining for reasoning models. It assumes you already know transformer mechanics, what AdamW does, and what perplexity measures. It focuses on the decisions that distinguish a base model that *trains well at RLVR* from one that *plateaus*: data curation under a reasoning lens, objectives beyond plain LM, the mid-training shift, long-context extension, scaling laws when the inference budget includes thinking tokens, and the failure modes that only appear at the trillion-token scale.

![Reasoning Pretraining Pipeline](/imgs/blogs/reasoning-pretraining-pipeline.png)

The diagram above shows the four-stage structure that recurs across publicly disclosed frontier reasoning model recipes (DeepSeek-V3 → R1, Qwen3-Thinking, the o-series-style buildups). Stage 2 (mid-training) is starred because, as we will argue with both analysis and case studies below, it is the stage where reasoning capability is *forged* — RLVR amplifies what is there, but cannot create it from a base that lacks the prior.

## 1. What "Reasoning" Pretraining Actually Means

A reasoning model is not just a base LM with a system prompt that says "think step by step." It is a model whose *latent capability* — the prior over sequences instilled during pretraining — biases it toward producing long, structured, self-correcting chains of intermediate computation. That bias is built almost entirely before RLVR starts. RLVR amplifies what is already there; it does not create capability from nothing.

This reframes pretraining from "minimize log-loss on web text" to "build a representational substrate where reasoning trajectories are likely under the prior." The practical consequences:

- The data mixture is engineered, not scraped. The fraction of code, math, and structured logical content matters at the 10–30% level, not the 1% level.
- The training stage is not uniform. Early pretraining rewards diversity; mid-training rewards quality and reasoning density; long-context extension rewards needle-in-haystack and multi-hop synthesis.
- The objective is not always next-token. Fill-in-the-middle, span corruption, and curriculum-aware loss weighting all show up.
- Evaluation during pretraining is reasoning-aware. Loss alone is misleading; you measure pass@k on math, code, and logic suites well before RL.
- Hyperparameters that worked for a 70B base model do not transfer cleanly when reasoning is the goal. Learning rate schedules, batch ramps, and weight-decay regimes all need re-tuning.

The rest of this article walks through the pipeline.

## 2. The Four-Stage Reasoning Pretraining Pipeline

Most frontier teams now describe their pretraining as four overlapping stages, each with its own data mixture, hyperparameter regime, and evaluation harness. The names vary; the structure is convergent.

```
Stage 1  Foundation pretraining        ~80% of compute
            broad web + code, 4K–8K context, primary loss=NLL
Stage 2  Mid-training (reasoning shift) ~10–15% of compute
            reasoning-dense corpora (math, code, scientific), curated
            tool-augmented synthetic data, instruction proxies
Stage 3  Long-context extension         ~3–5% of compute
            32K → 128K → 1M tokens via RoPE rescaling, YaRN, position
            interpolation; long-doc + retrieval-style data
Stage 4  Annealing / cooldown           ~2% of compute
            highest-quality subset, sharply decayed LR, no fresh
            knowledge — consolidate what already exists
```

A senior insight that distinguishes good teams from great ones: **mid-training is where reasoning capability is forged.** Stage 1 builds a competent imitator. Stage 2 makes that imitator a reasoner. Stage 4 polishes. RLVR then teaches the model when to deploy what mid-training instilled. Skipping or underinvesting in Stage 2 is the most common cause of "we ran RL and the curves are flat."

## 3. Data Curation Under a Reasoning Lens

If pretraining were a building, data would be the foundation, the steel, and the wiring. Architecture is the wallpaper. This section is therefore long.

### 3.1 The reasoning data axes

Stop thinking about data as one homogeneous river. Decompose it along axes that matter for reasoning:

| Axis | Why it matters for reasoning |
|---|---|
| Symbolic density | Density of math symbols, code, formal logic per token. High-density tokens train algorithmic priors. |
| Step length distribution | How long the typical chain of intermediate computation is. Short chains → shallow reasoners. |
| Verifiability | Is the correctness of the output checkable by a program? Verifiable data is RLVR-ready. |
| Diversity of structure | Proof, code, debate, plan-execute, dialogue, table reasoning. |
| Source provenance | Quality of the writer/community. Stack Overflow accepted answers > random forum posts. |
| Contamination risk | Overlap with downstream evals (MATH, GSM8K, HumanEval, AIME, etc.). |

A typical mid-training mixture for a frontier reasoning model:

```
Math (textbooks, papers, ProofPile, OpenWebMath)         12–18%
Code (filtered GitHub, StackOverflow, competitive prog)  18–25%
Scientific text (arXiv, PubMed, MedRxiv)                  6–10%
Reasoning synthetic (long CoT generated by stronger LM)   8–15%
General web (filtered, dedup'd, quality-classified)      30–40%
Books (curated, OCR cleaned)                              4–8%
Multilingual reasoning                                    3–6%
Tool-use / agentic traces (synthetic)                     2–5%
```

Compare this to a typical Stage-1 mixture (web ~70%, code ~10%, math ~3%) and the shift is dramatic. The mid-training mixture is not just "more math"; it is *structurally* different.

### 3.2 Filtering for reasoning quality

Standard pretraining filters (language ID, perplexity filter, near-dedup, NSFW, PII) are necessary but not sufficient. Reasoning-specific filters that pay off:

**Symbolic-density classifier.** A small fastText or distilbert-sized classifier trained on labeled high/low-reasoning pages. Throw away pages where reasoning density falls below threshold. On a 5T-token web corpus, this typically prunes 60–80% but keeps the reasoning lift.

**Math/code structural filters.** For math: AST/MathML parse, balanced parentheses, equation density per paragraph. For code: AST validity, ratio of comments to code, presence of test cases. Discard files that fail structural checks; they pollute training with broken priors.

**Step-extraction filter.** For documents that purport to "explain a solution," extract the step delimiters (numbered lists, "First… Then… Finally…", `<step>` tags). Documents with zero or one step are not reasoning content even if topical.

**Solution-correctness verification.** For math/code with verifiable answers (problem + solution pairs), run the solution. Discard pairs where the solution is wrong. Sounds obvious; very few teams do it at scale. The lift is real.

**De-contamination at every stage, not just at the end.** Build a 13-gram or MinHash-LSH index over your eval suites and run every new data shard through it. Recontamination via "synthetic" data is one of the silent killers of evaluation validity.

### 3.3 Synthetic reasoning data: how to do it without poisoning the well

Synthetic data is unavoidable for frontier reasoning training; the open web does not contain enough long, correct chains of thought. The risks are well-known: distribution collapse, error compounding, contamination of the eval set you forgot to dedup against. The patterns that work:

**Distillation from a stronger reasoner.** Use a model that already reasons (could be your previous generation, or an external model where licensing allows) to generate long CoT for a curated problem set. Verify each generation's answer against ground truth. Discard wrong answers. Discard chains shorter than a length threshold (lazy teacher signals).

**Self-play with a verifier.** Generate problem → solution → verifier-judges. Keep verified-correct (solution, chain-of-thought) pairs. The verifier can be a unit-test runner (for code), a symbolic math engine, or a strong critic LM with chain-of-verification.

**Backward generation.** Start from a verified theorem/proof, generate plausible-but-wrong paths, then teach the model to recognize and recover from them. This is how you instill a self-correction prior cheaply.

**Curriculum by difficulty.** Use the verifier's pass rate as a difficulty signal. Up-weight problems where the current model passes 30–70% — the zone where learning happens. Skip ones it always passes (no signal) and ones it always fails (too far for current capacity).

**Hard-negative-rich rationale data.** For each correct chain, generate a "subtly-wrong" chain that diverges late. Train on contrastive pairs. This is a reasoning-specific analog to hard-negative mining in retrieval.

A pragmatic ratio at mid-training: 30–60% of the reasoning slice can be synthetic, but only if every synthetic example has been *verified* by an external check. Unverified synthetic at >5% poisons the prior.

### 3.4 Deduplication, near-duplication, and the contamination minefield

Standard MinHash dedup at 13-gram is necessary but does not catch reasoning-specific contamination patterns:

**Translation contamination.** GSM8K problems translated to French, Chinese, etc. The same problem leaks in if your dedup is ASCII-Jaccard. Use a multilingual semantic dedup (sentence embedding nearest-neighbor at threshold 0.85+).

**Paraphrase contamination.** "Tom has 12 apples..." and "There are 12 apples and Tom owns them..." Same problem. Hash on parsed semantic structure (entity-relation triples) or use embedding dedup.

**Reformatted contamination.** A leaderboard problem in markdown vs. LaTeX vs. JSON. Normalize aggressively before dedup.

**Synthetic-loop contamination.** You generated 500K problems with model M; some are paraphrases of MATH problems M was trained on, which are paraphrases of test-MATH. Now you train M+1 on those paraphrases. M+1's MATH score looks great. Run hard de-contamination on test sets *before* you generate, and again on the generations.

A senior heuristic: maintain a "test set vault" of every benchmark you care about, normalized and embedded, and run every shard of every data stage through it. Failure to do this has invalidated entire model releases. Do not be that team.

### 3.5 Long-context data is its own beast

You cannot extend context to 128K+ by training on 4K-token web pages and rescaling RoPE at the end. The model has never *seen* what to do with a 64K-token prompt. Long-context data must be present, in volume, during the long-context extension stage:

- Curated long documents (books, theses, technical specifications, RFCs, court rulings)
- Synthetic multi-document QA: pack 5–20 retrieved passages plus a question whose answer requires multi-hop synthesis across them
- Code-repository-level data: full repos serialized with structure-preserving order
- Long-CoT: math/code problems with chains of 16K+ tokens

The mixture during long-context extension is typically ~30% true long documents, ~30% synthetic packed contexts, ~20% reasoning-dense long-CoT, ~20% from the regular mid-training mixture (to prevent forgetting). Without the last component you watch short-context capability degrade silently — a lesson many teams learn the hard way.

## 4. Pretraining Objectives Beyond Next-Token

Causal language modeling (next-token prediction) remains the dominant objective. But reasoning-aware pretraining increasingly mixes in supplementary objectives.

### 4.1 Fill-in-the-middle (FIM)

Originally for code (re-arrange a document so a middle span is generated last given prefix+suffix), FIM also helps for reasoning because it forces the model to reason given partial context — exactly the pattern at inference time when the model is mid-thought.

```
Original: <prefix> <middle> <suffix>
Rewritten: <PRE> <prefix> <SUF> <suffix> <MID> <middle>
```

A typical pretraining mix is 80–90% causal, 10–20% FIM. The FIM rate above 30% starts hurting straight causal completions; below 5% gives little benefit. Exact thresholds depend on tokenizer.

### 4.2 Document-level objectives

Span-corruption (T5-style) and prefix LM appear less in modern frontier pretraining, but a small admixture (5–10%) helps with structured outputs. The empirical evidence is mixed; treat as a knob worth ablating, not a default.

### 4.3 Loss masking and reasoning weighting

Pure NLL averaged over all tokens treats `the` and an inflection-point token in a chain of thought equivalently. Modern recipes use:

- **Reasoning-token weighting.** Tokens inside `<think>...</think>` regions get loss weight 1.5–2x. Encourages the model to learn the structure of reasoning explicitly during pretraining, not just at SFT.
- **Answer masking on demonstrations.** When training on `(problem, chain, final answer)` triples synthesized for mid-training, mask the loss on the problem statement. The model learns to *generate* reasoning given a problem, not to memorize problem statements.
- **Deduplication-aware down-weighting.** If a near-duplicate cluster has 1000 members, do not train on each as if independent. Down-weight by cluster size. Saves compute and prevents memorization of common phrasings.

### 4.4 The "thinking token" pretraining trick

A small but increasingly popular trick: introduce special tokens (`<think>`, `</think>`, or unused embedding slots) during pretraining that the model learns to associate with longer, lower-perplexity reasoning passes. Even if SFT later replaces them, the *embedding direction* is already there for RLVR to exploit. Cheap insurance.

## 5. Architecture Choices That Pay Off for Reasoning

Architecture is downstream of data; do not over-index on it. But several architectural decisions interact specifically with reasoning training.

### 5.1 Attention: GQA vs MQA vs MLA

Long reasoning chains stress KV cache. Standard MHA at 128K context with a 70B model demands ~30 GB of KV per request — unworkable at any throughput. Choices:

- **GQA (Grouped-Query Attention)** with 8 KV heads is the modern default; cuts KV by 8x with negligible quality loss.
- **MQA (single KV head)** trades more quality for max KV reduction; fine for small models, marginal at frontier scale.
- **MLA (Multi-head Latent Attention, DeepSeek)** projects KV to a low-rank latent — best ratio of quality to cache so far. Worth the extra implementation cost for any model serving long-CoT.

A senior takeaway: pick KV scheme based on the *expected inference shape*, not just training efficiency. A reasoning model spending 70% of inference tokens inside `<think>` blocks needs aggressive KV reduction or you cannot serve it.

### 5.2 Position encoding for long thinking

RoPE remains dominant. The variations that matter for reasoning:

- **NTK-aware scaling, YaRN, LongRoPE** for context extension. YaRN is the modern default — apply during Stage 3 with a small fine-tune (1–5B tokens) at the new context length.
- **Position-agnostic ALiBi** is mostly obsolete at frontier scale; reasoning models that need to attend to specific positions in a long chain do worse with linear bias.

### 5.3 MoE vs dense

Mixture-of-Experts (DeepSeek-V3, Qwen3, Mixtral) gives more parameters per FLOP, which translates to better world knowledge per inference budget. For reasoning specifically, MoE has a counter-intuitive pitfall: load-balance loss can suppress expert specialization in ways that hurt long-chain stability. Mitigations:

- Auxiliary-loss-free balancing (DeepSeek-V3 style, with bias-based routing)
- Lower load-balance coefficient during mid-training to let experts specialize on reasoning patterns
- Validate stability with long-CoT eval *during* pretraining, not only at the end

### 5.4 Normalization and activation choices

RMSNorm + SwiGLU is the modern default. The detail that occasionally matters: pre-norm vs post-norm vs the hybrid (Llama-style pre-norm with extra final norm). For reasoning models that produce 16K+ token outputs, training stability at long sequence lengths is sensitive to norm placement. Track gradient norm at long lengths, not just at training-time sequence length.

## 6. Scaling Laws When Reasoning Enters the Picture

Chinchilla scaling (compute-optimal at ~20 tokens per parameter) was derived for next-token loss. It does not directly apply when:

1. The downstream task budget includes inference-time thinking tokens.
2. The data mixture is non-uniform across stages.
3. Capability is measured by pass@k, not loss.

### 6.1 Inference-time scaling

The o1 paper formalized what reasoning teams already knew empirically: at fixed pretraining compute, allowing the model to "think longer" at inference improves accuracy on reasoning benchmarks logarithmically. The implication: a smaller model that thinks more can match a larger model that thinks less, *if* the smaller model's prior supports productive thinking.

The senior corollary: pretraining compute should be optimized jointly with expected inference budget. A model deployed to think 10K tokens per query justifies a different mid-training mixture than one deployed for 200-token answers. Frontier teams now plan pretraining around an explicit "thinking-token budget per query" target.

### 6.2 Data-mixture scaling

Mixture proportions are themselves a hyperparameter. A robust approach:

- Train smaller "scaling probes" (1B–7B params) at multiple mixtures, evaluate on reasoning suites.
- Fit a parametric model (DoReMi, RegMix, MixtureOfData) for how proportions affect downstream metrics.
- Extrapolate to the target scale.

This is non-trivial: mixture effects are not always monotonic at scale. Mixtures that look optimal at 1B can saturate at 70B. Always validate by running the proposed mixture at *at least* the next scale rung before committing trillions of tokens.

### 6.3 The "reasoning emergence" question

Some capabilities (multi-hop math, plan-and-execute) appear non-linearly with scale. This is partly a measurement artifact (binary pass/fail metrics make smooth log-loss improvements look like jumps), partly real (representational composition that requires sufficient capacity). Practical implication: do not declare a mixture "doesn't work for reasoning" based on a 1B probe. The probe might be below the emergence threshold for that capability.

## 7. Hyperparameters: Where Reasoning Pretraining Differs

Default Llama-style hyperparameters are a good starting point but rarely optimal for reasoning-focused training.

### 7.1 Learning rate schedule

The classic warmup → cosine decay still works, but reasoning training increasingly uses:

- **Trapezoidal / WSD schedule** (warmup → stable → decay): the stable phase makes mid-training data-mixture changes possible without re-warming. The decay phase aligns with the annealing stage.
- **Re-warming for new data stages.** When you transition from Stage 1 to Stage 2, the loss surface changes. A short re-warm (a few hundred steps) prevents the optimizer from over-correcting on the new mixture.
- **Lower peak LR for reasoning data.** Reasoning data is denser; the same step size that was stable on web text causes spikes. Reduce peak LR by 20–40% during mid-training.

### 7.2 Batch size

Reasoning sequences are long. Effective batch size in *tokens* matters more than in *sequences*. Aim for 4–8M tokens per batch for 70B-class models; ramp up batch size during training (small early to escape sharp minima, large late for stability).

### 7.3 Weight decay and gradient clipping

Standard 0.1 weight decay is fine. The detail: decoupled weight decay (AdamW) interacts with very long sequences in subtle ways — gradient norms at the end of long chains are larger, and aggressive clipping (max-norm 1.0) becomes the binding constraint instead of weight decay. Track gradient-norm distribution per sequence length; if 95th-percentile norm at 32K context is 5x the norm at 4K, your effective LR is sequence-length-dependent.

### 7.4 Optimizer choice

AdamW remains dominant. Notable alternatives gaining traction in 2025–26:

- **Muon (orthogonalizing momentum):** faster wall-clock convergence on reasoning benchmarks; modest adoption at frontier scale, growing.
- **Lion / Sophia:** lower memory footprint, mixed empirical evidence on reasoning specifically.
- **Shampoo / SOAP (second-order):** strong in-the-wild results from a few teams; engineering complexity is real.

A senior caution: optimizer choice matters less than data and compute. Do not let an optimizer hunt distract from data work, but do periodically re-evaluate. The 2024 default may not be the 2026 default.

## 8. Stability and Failure Modes at Scale

Trillion-token, hundred-billion-parameter training runs fail. Reasoning training is not gentler than vanilla pretraining; in some respects (long sequences, dense math) it is harder.

### 8.1 Loss spikes

Most spikes come from data: a shard of malformed text, a Unicode-heavy document tokenized strangely, a synthetic batch with zero variance. Defenses:

- **Per-shard loss tracking.** A shard whose mean loss diverges by >2σ from its peers is suspect. Quarantine and inspect.
- **Spike rollback discipline.** If loss spikes 3x and gradient clipping cannot absorb it, revert to a checkpoint, drop or shuffle the offending data, and resume. Do not let a run "recover on its own"; the underlying data issue still exists.
- **Lower LR at suspected boundaries.** Stage transitions (Stage 1 → Stage 2 in particular) are spike-prone. A short LR dip plus re-warm helps.

### 8.2 Loss going down, evals going down

The single most demoralizing failure mode. Causes seen in production:

- Mid-training mixture over-weighted easy synthetic data → loss falls (model memorizes templates) but generalization on novel problems regresses.
- Long-context extension done too aggressively → short-context evals (MMLU, HellaSwag) regress.
- Tokenizer drift (e.g., changed BPE for code) breaks downstream eval scoring; loss looks fine on the new tokenizer.
- Eval suite contamination: model "improves" on contaminated set, fails on held-out clean version.

A senior practice: maintain *two* eval suites — a public one for tracking-against-the-field, and a private clean one that has been carefully de-contaminated against your data. Trust the private one when they disagree.

### 8.3 Numerical instability at long sequence lengths

Attention logits scale with sequence length and head dimension. At 128K context with bf16 storage, softmax denominators can underflow. Mitigations:

- FP32 softmax (most frameworks default to this; verify).
- Stable attention kernels (FlashAttention v3, Triton custom kernels with online softmax).
- Activation checkpointing of attention specifically (the rest of the layer can be cheaper).

Track NaN frequency, not just loss. A run that produces 0.001% NaNs is degrading silently; the surviving updates are biased toward whatever sequences did not NaN.

### 8.4 Checkpoint and rollback discipline

A senior pretraining team treats checkpoints like banks treat backups: every N steps, multiple regions, automatic verification, well-rehearsed restore. The failure case is when you discover a regression at step 500K and want to roll back to step 480K but the only checkpoint you have is at step 200K.

Recommended cadence for a 70B-class run: full checkpoint every ~2 hours, verified by loading and computing a forward pass on a fixed prompt, replicated to a second region, with the last 24 hours kept hot and older ones tiered to cold storage. Cost is real (a 70B fp32 optimizer state is ~1.5 TB); the cost of *not* having the checkpoint is the entire run.

## 9. Mid-Training: The Phase That Decides Reasoning Capability

Many teams under-budget mid-training. This is the phase that, in retrospect, most differentiates strong reasoning models from average ones.

### 9.1 Goals of mid-training

1. Shift the model's prior toward reasoning-rich completions without destroying general capability.
2. Instill structured reasoning patterns (step-by-step, verification, backtracking) at the *prior* level, so RLVR has something to amplify.
3. Make tool-use, scratchpad, and long-CoT formats native — model "knows" what `<think>` blocks are by the time of post-training.
4. Set up the long-context extension that follows.

### 9.2 Practical mid-training recipe

A worked recipe at 70B scale, adaptable down or up:

```
Mid-training duration: 200–500B tokens (3–7% of total compute)
Data mix: as in §3.1, mid-training column
LR: re-warm to ~30% of Stage 1 peak, cosine decay over the stage
Batch: same token-batch as Stage 1 (~6M)
Sequence length: extend to 16K mid-stage (precursor to long-context)
Loss weighting: 1.5x on tokens inside <think>...</think>
Evaluation cadence: every 5B tokens on math (MATH-500), code (LCB),
   logic (BBH), plus short-context regression suite (MMLU, HellaSwag)
Stop criteria: improvement plateaus on reasoning suites OR regression
   on general suites exceeds 1 point
```

The *evaluation cadence* is non-negotiable. Mid-training is the stage where you can ruin a base model fastest. Catching regressions within 5B tokens means you can rollback and re-mix; catching them at 200B means you train an extra 50B on the wrong mix to "see if it recovers" (it usually doesn't).

### 9.3 Annealing as part of mid-training, not after

A subtle point: the annealing phase (sharply decayed LR, highest-quality data) often blends into mid-training rather than following Stage 3. You pick the very best curated subset — typically 20–40B tokens of verified math/code/scientific text — and run with LR decayed to ~5% of peak. The model "consolidates" these patterns. This is where the marginal reasoning capability per token is highest in the entire training run.

## 10. Long-Context Extension: Beyond Naive RoPE Scaling

A 4K-trained model does not become a 128K model by setting `max_position_embeddings=131072`. The naive failure modes are well-known: attention dilution, position-encoding extrapolation breakage, KV cache thrashing.

### 10.1 The recipe that works

```
1. Stage 3a: extend to 32K
   - YaRN or LongRoPE rescaling of RoPE base
   - 2–5B tokens of long documents + packed retrieval contexts
   - LR ~10% of mid-training peak
2. Stage 3b: extend to 128K (if target)
   - Re-rescale RoPE
   - Synthetic long-CoT dominates
   - Add needle-in-haystack training data explicitly
3. Stage 3c: extend to 1M (if target)
   - Sparse / sliding-window attention or hybrid
   - Heavily synthetic; real 1M-token coherent documents are rare
```

### 10.2 The "lost in the middle" pathology

Models extended naively under-attend to mid-context content; recall on facts placed in token 50K of a 100K context drops to chance. Fixes:

- Train explicitly on tasks where the answer is in the middle (force attention there).
- Use position-shuffled training (rotate which segment of a packed context is relevant).
- Validate with a "pin-test" suite: insert a unique fact at position p ∈ {start, 25%, 50%, 75%, end} and measure retrieval rate.

If your model passes pin-tests at start/end but fails at middle, you have not yet completed long-context extension regardless of what the loss says.

### 10.3 KV cache implications during training

Long-context training stresses memory. Sequence packing (concatenating multiple short docs to fill the context window with cross-doc attention masking) helps utilize the FLOPs but introduces subtle bugs if attention masks are wrong. Sequence parallelism (Megatron-style or Ulysses) becomes essential at ≥64K. Test the parallelism scheme on a known-loss reference run before committing to a real training run; off-by-one bugs in attention masking are a leading cause of "the long-context run trained but the model is incoherent."

## 11. Pretraining-RLVR Handoff

Pretraining ends, post-training begins. Reasoning models then go through SFT (small, optional) → RLVR (large, central) → preference learning. The pretraining team's job is not done at handoff; the *quality* of the handoff determines whether RLVR improves the model or thrashes.

### 11.1 What the post-training team needs from the base

- High pass@k at low k on the verifier-set (i.e., the model already produces correct answers sometimes; RL amplifies frequency).
- Long, structured generations under a thinking template (the format is already familiar to the model).
- Calibration: model's confidence is informative; this comes from diverse, high-quality pretraining, not from RLVR.
- Stable long-context behavior; otherwise RL rollouts are noisy.

If pass@8 on your math verifier is below 30% at the end of pretraining, RLVR will struggle to push pass@1 above ~50%. The reasoning prior must already be there.

### 11.2 The "RL-readiness" eval

A senior team runs an RL-readiness eval before declaring pretraining done:

```
For a held-out problem set with verifiable answers:
  Sample 16 completions per problem at temperature 0.8
  Measure: pass@1, pass@8, pass@16
  Measure: chain length distribution (mean, p99)
  Measure: chain self-consistency (do top-k samples agree?)
  Measure: format adherence (does <think>...</think> wrap reasoning?)
```

A model passing this eval at thresholds your post-training team set will respond to RLVR. A model failing this eval will not, regardless of how clever your RL recipe is.

## 12. Compute, Cost, and Engineering Reality

A senior section, briefly. Pretraining a 70B reasoning-capable model in 2026 is roughly:

- ~12T tokens total (Stage 1: 10T, Stage 2: 1.5T, Stage 3+4: 0.5T)
- ~2.5e24 FLOPs
- ~1024 H100s for ~60 days, or ~512 H200/B100s for proportionally less
- Wall-clock cost: $5–15M depending on cloud / on-prem

Cost dominators in order: data acquisition + curation (often underestimated, 20–30% of total budget when you include the verifier infrastructure), training compute, evaluation infrastructure, mistakes (failed runs, restart, ablations).

The senior advice: budget 1.5–2x your "model card cost." The model card reports the successful run. The successful run is preceded by 3–5 failed runs at smaller scale, mixture ablations, and a long tail of evaluation engineering. A real budget includes those.

## 13. Case Studies: What Frontier Teams Have Said Publicly

These are abridged from publicly available reports; details vary across releases.

### 13.1 DeepSeek-R1 / DeepSeek-V3 lineage

DeepSeek-V3 invested heavily in mid-training with a math/code-heavy mixture and used auxiliary-loss-free MoE balancing to enable expert specialization on reasoning. The R1 line then pushed RLVR aggressively from a base that was already strong at math. The pretraining lesson: a base model carefully tuned for *RLVR-readiness* converts a modest RL budget into outsized gains.

### 13.2 OpenAI o-series

OpenAI has disclosed less, but the public framing of o1/o3 as "scaling inference-time compute" only makes sense if pretraining already built a model whose long completions are productive. The implicit pretraining message: reasoning emerges from a *combination* of mid-training data shifts and RLVR; neither alone suffices.

### 13.3 Qwen3 and Qwen2.5-Math

Alibaba's Qwen team has been transparent about staged training: a math-dense mid-training phase, careful long-context extension, and a strong preference for synthetic reasoning data verified by external solvers. Qwen2.5-Math's strong performance derives from data engineering more than architectural cleverness.

### 13.4 What didn't work (from postmortems)

- Teams that deferred reasoning data to post-training found their RLVR runs plateaued.
- Teams that scaled context to 128K without dedicated data found long-context evals collapsed.
- Teams that under-evaluated during mid-training shipped models with subtle but persistent regressions on general capability that were only caught after release.

## 14. Senior Heuristics and Closing Thoughts

A short list of heuristics that recur across teams I have worked with or interviewed:

**Data work pays more than architecture work, by roughly an order of magnitude, at frontier scale.** If you have a week and one engineer, spend it on data filtering, not on a new attention variant.

**Mid-training is where reasoning is born.** Stage 1 builds a fluent imitator. Mid-training shifts the prior. Annealing consolidates. RLVR amplifies. Skipping mid-training and hoping RLVR will compensate is the most expensive mistake in current reasoning training.

**Verify everything verifiable.** Solutions that claim correctness, dedup against eval sets, synthetic chains, long-context retrievals. Verification infrastructure is part of pretraining, not an afterthought.

**Evaluate during, not after.** A 70B run is not a thing you launch and check at the end. It is a thing you steer, with a feedback loop measured in days, not months. Build the eval cadence first, then the training pipeline.

**Plan for the inference shape.** The model's optimal pretraining mixture depends on how much it will think at inference. A 200-token-answer model and a 16K-thinking-tokens model are different optimization problems.

**Track KV cache from day one.** Architecture choices (GQA, MLA, MoE) interact with serving cost in ways that can render a strong model unservable. Senior teams design for the deployment surface during pretraining, not after.

**Distrust your own evals.** Run a clean held-out suite alongside the public benchmarks. If they disagree, the public ones are wrong (probably contamination on your end).

**Scaling laws are local.** A mixture optimal at 1B may be wrong at 70B. Always validate at the next scale rung before committing the full budget.

**Reproducibility is non-negotiable for postmortems.** Pin everything: data hashes, code commits, library versions, kernel versions, hardware topology. The first time a key reviewer asks "is this reproducible?", the answer must already be yes.

Pretraining a reasoning model in 2026 is no longer a single craft; it is a pipeline of crafts. The teams that ship the strongest reasoning models are the ones that treat data curation, mid-training, long-context extension, and RLVR-readiness as first-class engineering disciplines — each with its own owners, evals, and deadlines. Architecture is the wallpaper. Compute is the substrate. Data is the medium of capability. And reasoning is what emerges when all three are aligned.

## 15. Decision Analysis: Why These Choices, Not Others

Senior engineers do not just follow recipes; they understand why each ingredient is in the recipe so they can substitute when constraints change. This section walks through the major decisions in reasoning pretraining and explains the reasoning behind the dominant choice versus credible alternatives.

### 15.1 Why staged training, not uniform mixing

**Alternative considered.** Mix all data (web + math + code + synthetic CoT + long docs) into one giant pool, shuffle, and train uniformly. Simpler engineering, no stage transitions, no mixture decisions per stage.

**Why staged wins.** Three reasons converge.

First, **learning rate matters and reasoning data needs a smaller LR than web text.** A uniform mix forces a single LR. Web text tolerates LR=3e-4 at 70B scale; math-dense documents at the same LR show loss spikes (denominators in derivations exhibit high gradient norm; rare token combinations push individual logits hard). Empirical evidence from the Qwen2.5-Math report and several internal postmortems: reducing LR by 30–40% during the math-heavy phase eliminates spikes without slowing overall convergence on web data, because web data has already been mostly absorbed by then.

Second, **early exposure to high-quality reasoning data is wasted.** A 1B-token-old model has not yet learned syntax; teaching it Lean proofs is throwing signal away. The model literally cannot represent the structure. Staging concentrates the expensive reasoning data in the window where the model has the capacity to absorb it.

Third, **annealing requires no new knowledge.** The most powerful per-token signal in the entire run comes from running the highest-quality 20B tokens at sharply decayed LR after the model is mature. This only works as a distinct stage; mixed in early, those tokens are seen at high LR and partially memorized rather than consolidated.

**Code: a practical staged-mixture controller.**

```python
# stage_controller.py — toggle data mixtures and LR per stage during pretraining
from dataclasses import dataclass

@dataclass
class StageConfig:
    name: str
    token_budget: int          # in billions
    mixture: dict[str, float]  # source -> weight, sums to 1
    lr_scale: float            # multiplier on peak LR
    seq_len: int
    rewarmup_steps: int        # short rewarmup at stage entry
    reasoning_loss_weight: float  # multiplier on tokens inside <think>...</think>

STAGES = [
    StageConfig(
        name="foundation",
        token_budget=10_000,
        mixture={"web_filtered": 0.70, "code_general": 0.10, "books": 0.05,
                 "math_general": 0.03, "scientific": 0.05, "multilingual": 0.07},
        lr_scale=1.0, seq_len=8192, rewarmup_steps=0, reasoning_loss_weight=1.0,
    ),
    StageConfig(
        name="midtrain_reasoning",
        token_budget=400,
        mixture={"web_filtered_top": 0.30, "code_curated": 0.22, "math_curated": 0.15,
                 "scientific_top": 0.08, "synth_cot_verified": 0.12, "books_top": 0.06,
                 "tool_traces": 0.04, "multilingual_reasoning": 0.03},
        lr_scale=0.65, seq_len=16384, rewarmup_steps=400, reasoning_loss_weight=1.5,
    ),
    StageConfig(
        name="long_context_32k",
        token_budget=30,
        mixture={"long_docs": 0.30, "packed_retrieval": 0.30, "long_cot": 0.20,
                 "midtrain_holdout": 0.20},
        lr_scale=0.10, seq_len=32768, rewarmup_steps=200, reasoning_loss_weight=1.5,
    ),
    StageConfig(
        name="long_context_128k",
        token_budget=20,
        mixture={"long_docs": 0.25, "packed_retrieval": 0.30, "long_cot": 0.30,
                 "needle_synth": 0.10, "midtrain_holdout": 0.05},
        lr_scale=0.05, seq_len=131072, rewarmup_steps=200, reasoning_loss_weight=1.5,
    ),
    StageConfig(
        name="anneal",
        token_budget=30,
        mixture={"math_top": 0.30, "code_top": 0.25, "scientific_top": 0.15,
                 "synth_cot_verified_top": 0.20, "instruction_proxy": 0.10},
        lr_scale=0.05, seq_len=16384, rewarmup_steps=100, reasoning_loss_weight=2.0,
    ),
]

class StageRunner:
    """Drive a pretraining loop through staged configs.

    The runner owns the mixture, LR scale, and per-token loss weights. It
    notifies the trainer at stage boundaries so checkpoints, eval cadence,
    and re-warmup behavior can branch on the stage name.
    """
    def __init__(self, stages, trainer):
        self.stages = stages
        self.trainer = trainer
        self.tokens_seen = 0

    def run(self):
        for stage in self.stages:
            self._enter(stage)
            target = self.tokens_seen + stage.token_budget * 1e9
            while self.tokens_seen < target:
                batch = self.trainer.sample_batch(
                    mixture=stage.mixture, seq_len=stage.seq_len)
                loss = self.trainer.step(
                    batch,
                    lr_scale=stage.lr_scale,
                    reasoning_weight=stage.reasoning_loss_weight)
                self.tokens_seen += batch.token_count
                if self._is_eval_boundary(stage):
                    self.trainer.run_evals(stage.name)
            self._exit(stage)

    def _enter(self, stage):
        self.trainer.checkpoint(f"enter_{stage.name}")
        self.trainer.set_lr_scale(stage.lr_scale)
        self.trainer.rewarmup(stage.rewarmup_steps)
        self.trainer.set_seq_len(stage.seq_len)
```

The structural lesson: stages are *data + LR + sequence-length + loss-weighting* tuples, not just data swaps. Treating them as such makes the boundary code explicit and safe to checkpoint around.

### 15.2 Why mixture-of-experts for reasoning, when

**Alternative considered.** Stick with dense transformers (Llama-style) for simplicity. Dense models avoid load-balance headaches, are easier to serve, and have well-understood scaling properties.

**Why MoE often wins for reasoning specifically.** Total parameter count drives world-knowledge capacity; active parameter count drives FLOPs. Reasoning queries spend many tokens *thinking* — meaning each user query consumes thousands of forward passes. MoE lets us spend a 230B-parameter knowledge bank's worth of capacity at the FLOPs of a 22B dense model (DeepSeek-V3's actual ratio). For workloads where inference is reasoning-heavy, this Pareto improvement is decisive.

**Why MoE sometimes loses for reasoning specifically.** Three counter-considerations.

First, **load-balance loss can suppress expert specialization.** Vanilla MoE training adds an auxiliary loss to keep tokens distributed across experts. If the loss coefficient is too high, experts converge to interchangeable functions, and the long chains you want them to specialize on never form. DeepSeek-V3's auxiliary-loss-free balancing (bias-shifted routing without an auxiliary loss term) was a deliberate response to this — measured improvements on long-CoT eval at the same active params.

Second, **expert routing instability hurts long sequences.** A 16K-token chain of thought routes 16K tokens through routing decisions. Small per-token routing noise compounds; a token mid-chain routed to the wrong expert degrades that step's representation, propagating downstream. Top-2 routing with loss-free balancing is more robust here than top-1 or auxiliary-loss-balanced.

Third, **MoE makes evaluation noisier.** Two runs with different random seeds can route the same eval prompt to different experts, producing different outputs. Reproducibility of long-CoT eval requires deterministic routing, which most production frameworks now support but is not the default.

**Decision matrix.**

| Constraint | Choose dense | Choose MoE |
|---|---|---|
| Inference budget per query | Short answers (<500 tokens) | Long thinking (>3K tokens) |
| Serving stack maturity | Strong (vLLM/SGLang on dense) | Need MoE-aware serving (vLLM 0.6+, SGLang) |
| Team experience with routing | Limited | Comfortable |
| Total param budget | <30B | 100B+ |
| Required global knowledge breadth | Narrow domain | Broad / general purpose |

### 15.3 Why GQA / MLA over MHA, when

The KV cache equation tells the whole story. For sequence length L, batch B, head dim D, num heads H, layers Λ, in fp16:

```
MHA  KV bytes = 2 * B * L * H * D * Λ * 2
GQA  KV bytes = 2 * B * L * (H / g) * D * Λ * 2     # g = group size
MQA  KV bytes = 2 * B * L * 1 * D * Λ * 2
MLA  KV bytes = 2 * B * L * R * Λ * 2               # R = latent rank
```

For a 70B reasoning model serving a 32K-token thinking chain at batch 8: MHA needs ~85 GB of KV alone (untenable on a single H100). GQA-8 cuts that to 11 GB. MLA (DeepSeek-V3 with R≈576) reaches ~6 GB, *with* slightly better quality than GQA-8 on long-CoT.

**Why MLA is not a free lunch.** It adds projection matrices (rank-R latent), so total parameter count grows by ~3% at the same hidden size. It also requires custom kernels for efficient inference; off-the-shelf vLLM and TRT-LLM took months to support it well. For a team without kernel expertise, GQA-8 is the safer choice; MLA pays off when serving is a first-party concern and long CoT is the dominant traffic shape.

### 15.4 Why YaRN over PI, NTK-aware, or training from scratch at 128K

Position interpolation (PI), NTK-aware scaling, and YaRN all address the same problem: a model trained at 4K context has RoPE frequencies that do not extrapolate to 128K naturally. The differences:

- **PI** linearly compresses positions. Simple, but forces the model to relearn the entire frequency basis. Quality regresses noticeably below 32K.
- **NTK-aware** preserves high-frequency components by scaling the RoPE base unevenly. Better than PI but still loses some long-distance structure.
- **YaRN** combines NTK-aware scaling with a temperature correction in attention logits. Empirically strongest at 32K–128K extension with the smallest tuning budget (1–5B tokens of long-context data).
- **Training from scratch at 128K** is the gold standard but ~4–8x more expensive in compute (attention is quadratic; even with FlashAttention, longer sequences cost more than training short and extending).

**Why YaRN became the default.** It is the cheapest method that retains short-context quality. The Qwen, Llama-3, and DeepSeek-V3 reports all converged on YaRN-style methods. The empirical loss on a 4K-context eval after YaRN extension is typically <0.3% worse than the original; PI loses 1–3%. For a frontier release, that gap matters.

**Code: YaRN configuration applied at the start of long-context extension.**

```python
# yarn_config.py — apply YaRN to a pretrained model before Stage 3
import math
import torch

def apply_yarn(model, original_max_pos: int, target_max_pos: int,
               beta_fast: float = 32.0, beta_slow: float = 1.0,
               attn_scale: bool = True):
    """Reconfigure RoPE with YaRN scaling.

    Why these defaults:
      beta_fast=32, beta_slow=1 are the values from the YaRN paper that
      preserve high-frequency RoPE channels (which encode local position)
      while uniformly scaling low-frequency channels (which encode
      long-range position). This shape is what protects short-context
      accuracy during extension.

      attn_scale=True multiplies attention logits by sqrt(1/scale_factor)
      which compensates for the entropy growth that comes with longer
      attention windows. Without it, softmax over 128K tokens spreads
      probability too thin and recall drops.
    """
    scale = target_max_pos / original_max_pos
    for layer in model.layers:
        rope = layer.self_attn.rotary_emb
        rope.reset_with_yarn(
            scale=scale,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            original_max_pos=original_max_pos,
        )
        if attn_scale:
            # Magnitude scaling per the YaRN paper: 0.1 * ln(scale) + 1.0
            layer.self_attn.attn_logit_scale = 0.1 * math.log(scale) + 1.0
    model.config.max_position_embeddings = target_max_pos
    return model
```

The detail to internalize: YaRN is not just a config flag, it is two coupled changes (frequency scaling + attention magnitude). Forgetting the magnitude scale is one of the most common silent regressions during extension.

### 15.5 Why RLVR over RLHF for reasoning

**RLHF (Reinforcement Learning from Human Feedback)** uses a learned reward model trained on human preference comparisons. It works well for general dialogue alignment but has fundamental limits for reasoning:

- Human preferences on reasoning are noisy and slow to collect (an annotator takes minutes to verify a long math proof).
- The reward model is itself an LLM and can be gamed (reward hacking on stylistic features).
- Long chains of thought have many ways to be wrong; preference data underspecifies which ones to penalize.

**RLVR (Reinforcement Learning from Verifiable Rewards)** sidesteps these issues by using *programmatic verifiers*: unit tests for code, symbolic equivalence for math, deterministic checkers for puzzles. The reward is binary (or near-binary), unbiased, and infinitely scalable.

**Why RLVR only works on top of strong pretraining.** RLVR exploits cases where the model already produces correct answers *sometimes*. The optimization objective is to increase the frequency of correct answers. If the base model has pass@1=2% on AIME, RLVR can push it to 30–50%. If the base model has pass@1=0% on AIME (never gets a correct answer in 100 samples), RLVR has no signal to amplify and the run plateaus immediately. This is the empirical reason why mid-training matters: it is what makes pass@k>0 across a wide problem distribution.

**Code: RL-readiness eval, the gate between pretraining and post-training.**

```python
# rl_readiness.py — measure whether the base model is ready for RLVR
import torch
from dataclasses import dataclass
from collections import Counter

@dataclass
class RLReadinessResult:
    pass_at_1: float
    pass_at_8: float
    pass_at_16: float
    mean_chain_len: float
    p99_chain_len: float
    self_consistency: float  # fraction of problems where top-k samples agree
    format_adherence: float  # fraction wrapping reasoning in <think>...</think>

def evaluate_rl_readiness(model, tokenizer, problems, verifier,
                          n_samples: int = 16, temperature: float = 0.8,
                          max_new_tokens: int = 8192) -> RLReadinessResult:
    """Run the RL-readiness eval before declaring pretraining done.

    A model passing thresholds set by your post-training team is RL-ready;
    a model failing this eval will not respond to RL regardless of the
    cleverness of the recipe.

    Recommended thresholds (calibrate per task):
      pass@1   >= 5%   (RLVR has signal to amplify)
      pass@8   >= 30%  (sample diversity is productive)
      format_adherence >= 0.6  (reasoning template is native)
      self_consistency 0.4-0.7 (too high = mode collapse, too low = noise)
    """
    correct_at_1, correct_at_8, correct_at_16 = 0, 0, 0
    chain_lens = []
    consistent_count = 0
    formatted_count = 0
    total = 0

    for problem in problems:
        completions = model.generate(
            prompt=problem.prompt,
            n=n_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        verdicts = [verifier(c, problem.ground_truth) for c in completions]

        if verdicts[0]: correct_at_1 += 1
        if any(verdicts[:8]): correct_at_8 += 1
        if any(verdicts[:16]): correct_at_16 += 1

        for c in completions:
            chain_lens.append(len(tokenizer.encode(c)))
            if "<think>" in c and "</think>" in c:
                formatted_count += 1

        # Self-consistency: do the answers extracted from top-k samples agree?
        answers = [problem.extract_answer(c) for c in completions[:8]]
        most_common = Counter(answers).most_common(1)[0][1]
        if most_common >= 4:  # majority of 8 agree
            consistent_count += 1

        total += n_samples

    n = len(problems)
    return RLReadinessResult(
        pass_at_1=correct_at_1 / n,
        pass_at_8=correct_at_8 / n,
        pass_at_16=correct_at_16 / n,
        mean_chain_len=sum(chain_lens) / total,
        p99_chain_len=sorted(chain_lens)[int(0.99 * total)],
        self_consistency=consistent_count / n,
        format_adherence=formatted_count / total,
    )
```

The code is short; the discipline is the hard part. Senior teams *gate* the pretraining-to-post-training transition on this eval. Failing the gate means rolling back to a checkpoint and adjusting mid-training, not pushing forward.

## 16. Code Walkthroughs: From Raw Web to Trainable Tokens

This section assembles the data-side pipeline end-to-end, with code that survives at scale. The shapes are illustrative; production systems use distributed compute (Spark, Ray Data, Apache Beam) but the per-record logic is identical.

### 16.1 Symbolic-density classifier for the reasoning lens

```python
# reasoning_filter.py — fastText-style classifier for reasoning density
from dataclasses import dataclass
import re

EQUATION_PATTERNS = [
    r"\\\(", r"\\\[", r"\$.+?\$",            # inline/block math
    r"\\frac", r"\\sum", r"\\int", r"\\prod",
    r"\\forall", r"\\exists", r"\\implies",
    r"=\\?[^=\n]+=\\?",                      # chained equalities
    r"\\theorem", r"\\proof", r"\\lemma",
]
CODE_PATTERNS = [
    r"^\s*def\s+\w+\(", r"^\s*class\s+\w+",
    r"^\s*function\s+\w+\(", r"^\s*async\s+def",
    r"```[a-z]+\n",                          # fenced code
    r"^\s*(if|else|elif|for|while|return)\s",
]
REASONING_PATTERNS = [
    r"\b(therefore|hence|thus|because|since)\b",
    r"\b(suppose|assume|let|consider)\b",
    r"\b(step\s*\d+|first|second|finally)\b",
    r"\b(proof|claim|theorem|lemma|corollary)\b",
    r"<think>", r"\\boxed\{",
]

@dataclass
class DensityScore:
    eq_density: float
    code_density: float
    reasoning_density: float
    step_count: int
    is_reasoning_dense: bool

def score_document(text: str, threshold: float = 0.015) -> DensityScore:
    """Score a document for reasoning density.

    Per-100-tokens normalization is the right unit because we want a
    fair comparison between a 200-token snippet of dense math and a
    20K-token textbook chapter — both can be reasoning-dense.

    threshold=0.015 (1.5 reasoning markers per 100 tokens) is empirically
    the inflection where downstream evals (math/code pass@1) start
    correlating with up-weighting; below it the up-weight is noise.
    """
    n_tokens = max(1, len(text.split()))

    eq_hits = sum(len(re.findall(p, text, re.M)) for p in EQUATION_PATTERNS)
    code_hits = sum(len(re.findall(p, text, re.M)) for p in CODE_PATTERNS)
    reason_hits = sum(len(re.findall(p, text, re.I | re.M))
                      for p in REASONING_PATTERNS)

    step_count = len(re.findall(r"(?im)^\s*(?:step\s*\d+|\d+\.)\s+\w", text))
    eq_d = eq_hits / n_tokens * 100
    code_d = code_hits / n_tokens * 100
    reason_d = reason_hits / n_tokens * 100

    is_dense = (eq_d + code_d + reason_d) >= threshold * 100

    return DensityScore(eq_d, code_d, reason_d, step_count, is_dense)
```

This is a hand-crafted prefilter. In production, train a fastText classifier on labeled samples (5K positive, 5K negative is enough) and run that — much faster, slightly better F1. The hand-crafted version is what you debug with when the classifier fights you.

### 16.2 Solution-correctness verification at scale

```python
# verify_solutions.py — run claimed solutions, keep only correct ones
import sympy
import subprocess
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout

def verify_math_pair(problem: str, solution: str, ground_truth: str,
                     timeout_s: float = 5.0) -> bool:
    """Verify a math problem solution by symbolic equivalence.

    Why symbolic, not exact-string: '1/2' and '0.5' and '\\frac{1}{2}' are
    all the same answer; string match would discard correct solutions.

    Why timeout: SymPy can hang on adversarial expressions; we'd rather
    discard than block the pipeline.
    """
    try:
        with ProcessPoolExecutor(max_workers=1) as exe:
            future = exe.submit(_sympy_equal, solution, ground_truth)
            return future.result(timeout=timeout_s)
    except (FTimeout, Exception):
        return False

def _sympy_equal(a: str, b: str) -> bool:
    expr_a = sympy.simplify(sympy.sympify(a))
    expr_b = sympy.simplify(sympy.sympify(b))
    return sympy.simplify(expr_a - expr_b) == 0

def verify_code_pair(problem: str, solution: str, tests: list[str],
                     timeout_s: float = 10.0) -> bool:
    """Verify a code solution by running its tests in a sandbox.

    Use a real sandbox (firejail / nsjail / docker --rm) in production.
    The version below is illustrative; never run untrusted code without
    isolation.
    """
    program = solution + "\n\n" + "\n".join(tests)
    try:
        proc = subprocess.run(
            ["python", "-c", program],
            capture_output=True, timeout=timeout_s,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False

def filter_synthetic_reasoning(records: list, kind: str = "math") -> list:
    """Drop unverified synthetic CoT before mid-training. Crucial.

    Empirically (Qwen2.5-Math, DeepSeekMath reports): mid-training on
    unverified synthetic at >5% of the mixture *poisons* the prior;
    verified synthetic at 30-60% lifts reasoning evals 5-15 points.
    Verification is the difference, not synthetic-ness itself.
    """
    out = []
    for r in records:
        ok = (verify_math_pair(r["problem"], r["solution"], r["answer"])
              if kind == "math"
              else verify_code_pair(r["problem"], r["solution"], r["tests"]))
        if ok:
            out.append(r)
    return out
```

The discipline this code enforces is the difference between "synthetic CoT helps" and "synthetic CoT poisons your prior." Always verify; never trust the teacher.

### 16.3 Contamination dedup against eval suites

```python
# decontaminate.py — block contamination before training
from datasketch import MinHash, MinHashLSH

class EvalContaminationIndex:
    """LSH-based dedup against held-out eval suites.

    Uses 13-gram MinHash by default. Hits at threshold 0.85 get logged
    and dropped from training data with the source recorded so the team
    can audit which eval (and how many examples) leaked in.
    """
    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self.eval_meta = {}

    def add_eval(self, eval_name: str, examples: list[str]):
        for i, text in enumerate(examples):
            mh = self._minhash(text)
            key = f"{eval_name}::{i}"
            self.lsh.insert(key, mh)
            self.eval_meta[key] = eval_name

    def _minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        ngrams = self._ngrams(text, n=13)
        for g in ngrams:
            m.update(g.encode())
        return m

    @staticmethod
    def _ngrams(text: str, n: int):
        toks = text.lower().split()
        return [" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)]

    def is_contaminated(self, doc: str) -> tuple[bool, list[str]]:
        mh = self._minhash(doc)
        hits = self.lsh.query(mh)
        return (len(hits) > 0, [self.eval_meta[h] for h in hits])
```

In production, run this against MATH, GSM8K, HumanEval, MBPP, AIME, BBH, MMLU, ARC, IFEval, and every other eval you ship metrics for. Add multilingual variants (translation contamination is real). Hit rates above 0.5% on a clean web crawl mean your dedup is broken; investigate.

### 16.4 Sequence packing with attention masking

```python
# pack.py — pack short documents into long sequences without cross-doc attn
import torch

def pack_documents(docs: list[list[int]], max_len: int, eos_token: int):
    """Pack documents into fixed-length sequences with proper attention masks.

    Why packing: padding short docs to max_len wastes 60-90% of FLOPs.
    Why masking: without per-doc masks, attention from doc[i+1] can
    "see" doc[i] and learn spurious cross-doc dependencies — a subtle
    quality regression that does not show up in loss but does show up
    in reasoning evals.
    """
    sequences, masks = [], []
    cur, cur_mask, cur_doc_id = [], [], 0
    for d in docs:
        d = d + [eos_token]
        if len(cur) + len(d) > max_len:
            # Pad and emit
            pad = max_len - len(cur)
            sequences.append(cur + [eos_token] * pad)
            masks.append(cur_mask + [-1] * pad)  # -1 = padding
            cur, cur_mask, cur_doc_id = [], [], 0
        cur.extend(d)
        cur_mask.extend([cur_doc_id] * len(d))
        cur_doc_id += 1
    if cur:
        pad = max_len - len(cur)
        sequences.append(cur + [eos_token] * pad)
        masks.append(cur_mask + [-1] * pad)
    return torch.tensor(sequences), torch.tensor(masks)

def make_attention_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """Build a [seq, seq] mask where positions can only attend within doc.

    Combine this with the standard causal mask in the attention layer:
        final_mask = causal_mask & doc_mask
    """
    return doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)
```

Cross-document attention masking is one of those details that distinguishes serious pretraining codebases from quick research scripts. The bug it prevents — model learning that document boundaries are predictive — is silent and accumulates.

### 16.5 Reasoning-token loss weighting in the training step

```python
# train_step.py — apply 1.5x loss to tokens inside <think> blocks
import torch
import torch.nn.functional as F

def compute_weighted_loss(logits: torch.Tensor,        # [B, L, V]
                          labels: torch.Tensor,         # [B, L]
                          think_mask: torch.Tensor,     # [B, L] bool
                          reasoning_weight: float = 1.5,
                          ignore_index: int = -100) -> torch.Tensor:
    """Cross-entropy with up-weighted loss inside <think>...</think>.

    Why up-weighting reasoning tokens during PRE-training (not just SFT):
    by the time SFT runs, the model already has a strong prior over what
    "reasoning text" looks like. If pretraining treats every token equally,
    the model spends most of its capacity on the 95% of tokens that are
    web text. Up-weighting during mid-training shifts capacity allocation
    toward the reasoning structure we care about — measured improvements
    of 2-5 points on math evals at the same token budget.

    Why 1.5x and not 2x or 5x: empirical sweet spot. At 2x the model
    starts producing <think> blocks even when not prompted (noise);
    below 1.3x the effect disappears in noise. 1.5x is the value the
    Qwen and DeepSeek mid-training reports converge on.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = think_mask[..., 1:].contiguous()

    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(shift_labels.shape)

    weights = torch.where(
        shift_mask,
        torch.full_like(loss_per_token, reasoning_weight),
        torch.ones_like(loss_per_token),
    )
    valid = (shift_labels != ignore_index).float()
    weighted = loss_per_token * weights * valid
    return weighted.sum() / valid.sum().clamp(min=1.0)
```

The normalization detail: dividing by `valid.sum()` rather than `(weights * valid).sum()` keeps the loss magnitude comparable across stages. If you divide by the weighted sum, the effective LR shifts when the reasoning-token fraction changes, which destabilizes optimizer state across the boundary into mid-training. Senior teams catch this in their LR-vs-loss curves — sudden flat regions immediately after stage transitions usually mean a normalization mistake.

## 17. Extended Case Studies with Analysis

The earlier brief case studies pointed at public reports. This section walks through three composite case studies in depth, drawn from public reports plus things I have personally seen happen in production training runs. Names anonymized where details are not public.

### 17.1 Case study: the team that "scaled compute, got nothing"

A team trained a 30B dense model on 6T tokens with a Llama-3-style mixture (web 80%, code 8%, math 3%, books 4%, multilingual 5%). Compute investment was substantial — comparable to public 70B-class budgets at the time. They then ran 10K H100-hours of GRPO-style RLVR on math and code verifiers.

**Result.** Pre-RL: pass@1 on AIME ≈ 4%, pass@1 on MATH-500 ≈ 38%. Post-RL: pass@1 on AIME ≈ 7%, pass@1 on MATH-500 ≈ 45%. The team had expected the gains DeepSeek-R1 reported; they got a fraction.

**Diagnosis.** Three issues, in order of magnitude.

First, **no mid-training stage.** The model went straight from foundation pretraining to RL. The reasoning prior was thin: pass@8 on AIME at the start of RL was only ~12%, meaning 88% of problems had zero correct samples and therefore zero RL signal. Comparable models with mid-training showed pass@8 in the 35–50% range.

Second, **no synthetic verified CoT.** The 3% math share was raw web math (often unverified solutions, terse, no explicit step structure). The model had not seen many long, well-structured chains during pretraining and could not produce them at temperature.

Third, **eval contamination drift.** They had not maintained a private clean eval; their MATH-500 numbers were inflated by ~3 points from web crawls of solutions. Post-RL the contamination effect saturated, masking the underlying small RL gain.

**Fix.** Roll back to a checkpoint at ~5T tokens. Add a 400B-token mid-training stage with mixture matching §3.1. Verify all synthetic CoT. Build a private eval. Re-run RL from the new base. Result was pass@1 AIME 22%, MATH-500 71% — within 5 points of frontier numbers at the same compute. The mid-training stage cost ~7% additional pretraining compute and unlocked the RL gains they thought they had been buying directly.

**Senior takeaway.** RL is leverage on existing capability, not a capability factory. If pass@8 is low at hand-off, *no amount of RL fixes it*. Diagnose pretraining first.

### 17.2 Case study: the long-context regression

A team extended a 70B base from 8K to 128K context using YaRN with 5B tokens of long-document data. Long-context evals (Needle-in-a-Haystack, RULER) were strong: 95%+ retrieval at 128K. They shipped.

**Result two weeks post-launch.** Customer reports of subtle degradation on short-context tasks: code completion at 4K context noticeably worse, MMLU on the team's internal regression suite down 2.3 points, instruction-following degraded.

**Diagnosis.** The long-context extension data was 80% true long documents, 20% packed retrieval contexts. *Zero percent* short-context data from the original mid-training mix. The model "forgot" how to handle short contexts because it had been fine-tuned for 5B tokens on exclusively long ones. Loss on short-context data had drifted up by 0.04 nats during extension — small enough to look like noise on the loss curve but cumulatively significant.

**Fix.** Re-extension with a 20–25% admixture of the original mid-training mixture during long-context training. This is now standard practice across teams. The fixed model retained 128K capabilities and recovered short-context performance to within 0.3 points of pre-extension numbers.

**Senior takeaway.** Long-context extension is a fine-tune; fine-tunes forget. The regularizer is mixing in the original distribution. Anyone extending context who does not include "regular" data in the mix is silently regressing short-context capability.

### 17.3 Case study: the synthetic data death spiral

A team building a 7B reasoning model used a self-distillation loop: generate synthetic CoT with the current best checkpoint, train on it, repeat. Each round, evals improved on internal benchmarks. After four rounds, internal MATH-500 hit 78% — flagship-class on a 7B base.

**External eval.** Independent evaluation on a private MATH variant: 38%. Half their internal number. AIME independent: 4%, vs claimed 22%.

**Diagnosis.** Three compounding issues.

First, **training-time contamination.** The synthetic generator was trained on MATH-500-like data that had leaked through web dedup. Each self-distillation round re-injected paraphrased versions. By round four, ~12% of synthetic problems were near-duplicates of MATH-500 problems with different surface form.

Second, **distribution collapse.** The synthetic generator developed stylistic patterns (specific phrasings of "let x = ...", specific formatting of final answers). Internal eval prompts triggered these patterns, externally-phrased problems did not. The model had memorized solution *templates*, not solution *strategies*.

Third, **verifier weakness.** The verifier accepted answers that matched the gold answer string. It did not check that the chain of reasoning was actually valid. ~15% of "verified" synthetic CoT had a wrong derivation arriving at the right number by accident. The model learned bad reasoning that happened to terminate correctly.

**Fix.** Three layers. (1) Hard de-contamination of synthetic generations against every eval set, run before each round. (2) Stronger verifier: SymPy-equivalent answer check *and* chain validity check (heuristic for math: each step must be a valid algebraic transformation). (3) Cap synthetic admixture at 50% and require ≥30% of mid-training to come from human-written sources to anchor distribution.

After fix: internal MATH-500 dropped to 68% (more honest), independent MATH-500 rose to 64% (much closer agreement). The model became dramatically more useful in production.

**Senior takeaway.** Synthetic data scales only if every layer of the pipeline (generator, verifier, dedup, mixture cap) is strong. Weakest link governs. The trap is that internal evals always look good on a self-distilled model — by the time you notice on external evals, you have spent millions of dollars in compute on an over-fit.

## 18. Operational Patterns: How Senior Teams Run These Pipelines

This is the chapter rarely written down because it is operational rather than scientific. It is also where most teams burn their first frontier-model attempt.

### 18.1 The "reasoning eval dashboard"

Treat reasoning evaluation as a dashboard, refreshed every few hours during pretraining, with a fixed set of metrics:

```
Reasoning health dashboard (refreshed every 5B tokens)

  Math
    GSM8K (pass@1, pass@8)
    MATH-500 (pass@1, pass@8)
    AIME-2024 (pass@1 at T=0.0, pass@16 at T=0.7)
    Private math holdout (pass@1)

  Code
    HumanEval+ (pass@1, pass@5)
    MBPP+ (pass@1)
    LiveCodeBench medium (pass@1)
    Private code holdout (pass@1)

  Logic
    BBH (avg)
    Private deduction holdout

  General regression (do NOT regress these)
    MMLU (5-shot)
    HellaSwag
    ARC-C
    Private regression suite

  Long-context (after Stage 3 begins)
    RULER (across positions: 25%, 50%, 75%, 100%)
    NIAH (single + multi-needle)
    Long-CoT pass@1 on 32K-prompt math problems

  RL-readiness (last ~200B of mid-training)
    pass@8 on RLVR verifier set
    Format adherence rate
    Mean / p99 chain length
```

The dashboard's value is not the metrics but the *cadence*. A team that evaluates monthly catches problems too late. Weekly is the minimum. Every 5B tokens (typically every 12–24 hours on a frontier-scale run) is the senior practice.

### 18.2 The shadow run

Run a smaller "shadow" of the main pretraining: same data pipeline, same code, ~2% of the parameters. Use it to:

- Test mixture changes before committing to the real run
- Validate new data shards before they hit production
- Reproduce loss spikes for diagnosis without freezing the real run
- Verify long-context extension recipe end-to-end at 1B before doing it at 70B

Shadow runs cost ~2–3% of the main compute and have caught data bugs at three of the four trillion-token training campaigns I have firsthand knowledge of. Skipping them is false economy.

### 18.3 Postmortem culture

Every loss spike, every eval regression, every checkpoint rollback gets written up. A typical postmortem has:

- Symptom: what was observed, when, severity
- Root cause: data shard, code commit, hardware, hyperparameter
- Detection delay: how long between cause and detection
- Recovery action: rollback, data fix, code fix
- Generalization: what other things in the pipeline have the same risk
- Action items: what to add to prevent recurrence

Frontier teams accumulate 20–50 such postmortems per training campaign. The compounding learning is what separates teams that ship strong second models from teams whose first model was their last.

## 19. Looking Forward: Open Questions in Reasoning Pretraining

A few questions where the field has not converged as of 2026, and where senior practitioners disagree:

**Should reasoning be a separate model or a mode?** Both o1-style "always reason" and Claude-3.5-style "reason when asked" architectures exist. The training implications differ — separate models can over-specialize but never have to balance modes.

**How much synthetic data is too much?** Empirical caps of 50–60% in mid-training are conservative. Some teams report stable training at 80%+ with sufficiently diverse generators. The community has not converged on a principled answer.

**Is RLVR the endgame or a transition?** RLVR works on tasks with verifiable answers. As models tackle longer-horizon tasks (multi-day software projects, scientific discovery), the verifier's reward sparsity becomes a problem. Process rewards, learned value functions, and curriculum-driven RL are all being explored.

**Will architectural changes redraw the map?** Linear-attention models, state-space models (Mamba/S5), and hybrid attention architectures are credible alternatives at long context. None has yet matched dense transformer reasoning quality at frontier scale, but "yet" is doing work in that sentence.

The questions move; the principles do not. Build for the reasoning prior. Verify every synthetic. Eval before, during, and after. Mid-training is where capability is born. RLVR amplifies. Compute is the substrate. Data is the medium. And the senior engineer's job is to keep all three honest.
