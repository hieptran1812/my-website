---
title: "OLMo 3: How a fully-open lab builds a 32B reasoning model — the data, training, and eval tricks worth stealing"
date: "2026-06-02"
publishDate: "2026-06-02"
description: "A deep paper-read of the OLMo 3 technical report: the model flow, Dolma 3 data recipe, swarm-based data mixing, the OLMo 3 architecture, Delta Learning DPO, OlmoRL, and the open evaluation suite — every reproducible technique pulled apart."
tags: ["olmo-3", "llm-pretraining", "post-training", "rlvr", "grpo", "dpo", "data-mixing", "long-context", "evaluation", "open-source", "reasoning-models"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 30
aiGenerated: true
---

> [!tldr] TL;DR
> - **What it claims.** OLMo 3 is a *fully open* 7B/32B family — Ai2 ships not just weights but "the entire model flow": every checkpoint, every data point, every dependency. Olmo 3-Think 32B is presented as the strongest fully-open thinking model to date.
> - **Why it matters.** Almost every "open" model hides the one thing that actually determines behavior — the data pipeline and the post-training recipe. OLMo 3 is the rare release where you can audit *why* the model behaves the way it does, not just *that* it does.
> - **The surprising bit.** With a *decontaminated* RL setup, "RLVR with random rewards" stops helping. The famous "spurious rewards work" results appear to have been a contamination artifact. That single ablation reframes a chunk of 2024–2025 RL folklore.
> - **The clever engineering.** Data mixing chosen by *swarms of 30M proxy models* + a fitted predictor; long-context filtering by *GZIP compressibility*; preference data manufactured by pairing a strong model against a weak one (*Delta Learning*); an *asynchronous* RL trainer that cuts wall-clock from 15+ days to ~6.
> - **Where it lags.** Knowledge benchmarks (MMLU and friends), especially at 7B — partly a *licensing* tax: some high-signal sources are dropped to keep the release commercially usable.

The OLMo 3 technical report ([arXiv 2512.13961](https://arxiv.org/abs/2512.13961)) is unusual in that the interesting content is not the model — it is the *recipe*. Most frontier reports give you a number and a vibe. This one gives you the pipeline, named and ablated, stage by stage. If you build LLMs for a living, it reads less like a paper and more like a runbook someone forgot to mark confidential.

This post walks the report end to end, following its own table of contents — model flow, Base (architecture + the three pretraining stages), Think (SFT → DPO → RLVR), Instruct, and RL-Zero — and pulls out the techniques that are worth copying into your own stack regardless of whether you ever touch OLMo weights.

![The OLMo 3 model flow: one base model branching into Think, Instruct, and RL-Zero](/imgs/blogs/olmo-3-training-finetuning-techniques-1.png)

The diagram above is the mental model for the entire report: a single pretraining trunk (Stage 1 → 2 → 3) produces **Olmo 3 Base**, and that one base model then forks into four released artifacts through different post-training recipes. Hold this picture; every section below is a zoom into one box.

## Context: what "fully open" actually buys you

There is a spectrum of openness that the marketing word "open" flattens into nothing.

- **Open weights** (Llama, Qwen, Gemma, Mistral): you get a checkpoint. You can fine-tune it, serve it, distill from it. You cannot reproduce it, audit its training data, or know why it refuses some prompts and hallucinates on others.
- **Open everything** (OLMo 1/2/3, partially Pythia, Marin, Apertus): weights *plus* the data, the data-processing code, the training code, the intermediate checkpoints, and the evaluation harness.

OLMo 2 established Ai2's lineage here: a credible fully-open model that was *useful*, not just a reproducibility demo. The gap OLMo 3 sets out to close is the reasoning gap — the post-2024 world where the headline number is no longer MMLU but AIME, LiveCodeBench, and long-horizon agentic tasks, all of which are driven by *post-training* (long chain-of-thought SFT + RL) rather than by raw pretraining.

So the bet of OLMo 3 is: **can a fully-open lab, with a fully-disclosed data pipeline (which legally forces them to drop some high-value sources), still land within striking distance of Qwen 3 on reasoning?** The answer, roughly, is "yes on math and code, not quite on knowledge" — and the *why* is the most instructive part.

## Contributions

In the authors' framing, tightened:

1. **The model flow** — a fully-released lifecycle (every stage, checkpoint, datapoint, dependency) for a 7B/32B family, not just final weights.
2. **Dolma 3** — a ~9.3T-token corpus and three derived mixes (Mix / Dolmino / Longmino) for pretraining, mid-training, and long-context, with the dedup/filter tooling open-sourced.
3. **Dolci** — a post-training data suite with separate, decontaminated mixes for SFT, DPO, and RLVR, across math, code, instruction following, and chat.
4. **OlmoRL** — an asynchronous, off-policy RL stack with an enhanced GRPO objective that makes RLVR ~4× more throughput-efficient.
5. **A decontaminated RL-Zero setup** — and with it, evidence that the "random/spurious rewards help" phenomenon was a contamination artifact.
6. **An open evaluation suite** — OlmoBaseEval (43 benchmarks, clustered), the OLMES harness, and the `decon` decontamination tool.

## The model flow

The report's organizing abstraction is the **model flow**: the full directed graph from raw data to released variant. Concretely:

| Stage | Data | Tokens (trained) | Output |
|---|---|---|---|
| 1 — Pretrain | Dolma 3 Mix | ~6T | broad capability base |
| 2 — Midtrain | Dolma 3 Dolmino | 100B (from 2.2T pool) | math/code/science/reasoning boost |
| 3 — Long-context | Dolma 3 Longmino | ~50B (from 639B pool) | context to ~65K tokens |
| → | | | **Olmo 3 Base 7B / 32B** |
| Post-train (Think) | Dolci Think | SFT → DPO → RLVR | **Olmo 3 Think 7B / 32B** |
| Post-train (Instruct) | Dolci Instruct | SFT → DPO → RL | **Olmo 3 Instruct 7B** |
| Post-train (RL-Zero) | Dolci RL-Zero | RLVR from Base | **Olmo 3 RL-Zero** |

Two design decisions are doing a lot of work here and are worth naming up front, because they recur:

- **Mid-training as a distinct stage.** Rather than one monolithic pretraining run, the hard, high-value material (math, code, reading comprehension, reasoning traces) is concentrated into a separate 100B-token stage *after* the bulk 6T run. This is the "anneal on the good stuff" pattern, formalized.
- **Long-context as the *last* pretraining stage,** not baked in from token zero. The model learns language cheaply at 8K context, then gets stretched to 65K on a curated diet of genuinely long documents. Paying the quadratic-ish attention cost only on the final ~50B tokens is a large compute saving.

On the hardware side, the numbers ground the scale: pretraining ran on **up to 1,024 H100s** at **~7,700 tokens/device/second** for the 7B base; mid-training on **128 H100s**; post-training on **256 H100s**. These are not frontier-lab numbers, which is precisely the point — the recipe is the moat, not the cluster.

## Olmo 3 Base — architecture

OLMo 3 keeps the OLMo 2 skeleton (a dense, decoder-only transformer with **post-norm** RMSNorm and **QK-norm**) and adds the modern efficiency machinery the 2024–2025 architectures converged on.

![OLMo 3 attention layout: three sliding-window layers per full-attention layer](/imgs/blogs/olmo-3-training-finetuning-techniques-4.png)

The load-bearing choices:

- **Sliding-Window Attention (SWA) in 3 of every 4 layers**, with a **4K-token window**; every fourth layer uses **full attention**. This is the same trick Gemma 2/3 and others use: most layers only need local context, so you cap their attention cost at the window size, and let the occasional full-attention layer carry global information. At 65K context, having 75% of layers bounded to a 4K window is the difference between a tractable and an intractable KV cache.
- **YaRN applied *only* to the full-attention layers.** Rotary position embeddings are stretched (via YaRN) to reach 65K context — but only where it matters. SWA layers never see tokens beyond their 4K window, so rescaling their RoPE would be pointless; leaving them untouched avoids perturbing the short-range behavior the model spent 6T tokens learning.
- **QK-norm** — RMSNorm applied to queries and keys *before* the attention dot-product. This bounds attention-logit magnitudes and is a major stability lever at 32B scale; it is one of the quiet reasons OLMo 2/3 train without the loss spikes that plagued earlier open models.
- **GQA at 32B** (40 attention heads, 8 KV heads); the **7B uses plain multi-head attention**. GQA shrinks the KV cache 5× at 32B, which is the bottleneck for long-context serving.
- **SiLU** activations, standard SwiGLU-style MLPs.

The 7B-vs-32B differences are exactly what you'd expect — head count, KV-head count, hidden dim, and layer count scale; the recipe does not.

The senior-engineer takeaway: **none of these are novel, and that is the design philosophy.** OLMo 3 spends its novelty budget on data and post-training, and deliberately rides the consensus architecture so that its results isolate the *recipe's* contribution rather than confounding it with architectural cleverness.

### OLMo 2 → OLMo 3: what actually changed

It's worth being precise about the architectural delta, because it tells you where the team believed the marginal returns were:

| Dimension | OLMo 2 | OLMo 3 |
|---|---|---|
| Norm placement | post-norm RMSNorm | post-norm RMSNorm (kept) |
| QK-norm | yes | yes (kept) |
| Attention pattern | full attention, all layers | **SWA 3-of-4 + full every 4th** |
| Long context | shorter native context | **8K → 65K via YaRN on full-attn layers** |
| 32B attention | — (no 32B in OLMo 2) | **GQA, 40 heads / 8 KV** |
| Post-training | SFT + DPO + RLVR (Tülu lineage) | **SFT + DPO (Delta) + OlmoRL**, decontaminated |
| Data | Dolma/Dolmino v1 | **Dolma 3 + swarm mixing + GZIP long-ctx filter** |

The architecture changes (SWA, YaRN, GQA) are *all* about making long-context training and serving affordable. Everything else that moved is data and post-training. Read that table as a thesis statement: **OLMo 3 is a data-and-recipe paper wearing a model's clothes.** If you came hoping for an attention-mechanism breakthrough, you're in the wrong report; if you came for how to *build* a competitive model from scratch, this is the densest 50 pages you'll read this year.

A subtle consequence of the SWA-heavy stack worth flagging for anyone planning to *serve* OLMo 3: the KV-cache footprint is wildly non-uniform across layers. The 3-of-4 SWA layers cap their cache at the 4K window regardless of sequence length, while the 1-of-4 full-attention layers grow linearly to 65K. A naïve serving stack that allocates a uniform per-layer cache will over-provision the SWA layers by an order of magnitude at long context. Paged/heterogeneous KV allocation is not optional here — it's the entire point of the design. (If you've read the [KV-cache deep-dive](/blog/machine-learning/large-language-model/kv-cache) on this blog, this is exactly the heterogeneous-block scenario it warns about.)

## Data: building Dolma 3

If you read OLMo 3 for one thing, read it for the data work. This is where the report is most generous and where the most transferable ideas live.

![The Dolma 3 data funnel: each stage draws a small sharp mix from a far larger pool](/imgs/blogs/olmo-3-training-finetuning-techniques-2.png)

**Dolma 3** is a ~**9.3T-token pool** assembled from web crawl, **238M unique academic PDFs** (converted with **olmOCR**, knowledge cutoff December 2024), code repositories, math problems-and-solutions, and encyclopedic text. From that pool, three mixes are drawn:

- **Dolma 3 Mix** — the **5.9T (~6T)** pretraining mix, deliberately heavier on code and math than a generic web mix.
- **Dolma 3 Dolmino** — **100B** training tokens sampled from a **~2.2T** pool of high-quality math, science, code, instruction-following, reading-comprehension data, *including reasoning traces*. This is the mid-training fuel.
- **Dolma 3 Longmino** — **~50B** training tokens drawn from a **639B** pool of genuinely long documents, mixed with mid-training data, to teach the model to track information across very long inputs.

### Deduplication and quality filtering

The dedup/filter stack is shipped as real tooling, not described in prose: `datamap-rs` (a Rust pipeline for large-scale quality filtering) and `duplodocus` (deduplication). The high-level moves:

- **Aggressive exact + near-dup removal** across the pool before any mixing — the single highest-ROI data operation, full stop.
- **Quality-aware upsampling.** Rather than a binary keep/drop, high-quality documents get a *monotonically increasing* upsampling factor: the top-5% percentile is upsampled ~**7×**. Crucially, the upsampling curve is fit *per topic* via a constrained parametric search, not globally — so "high quality" for code and "high quality" for prose get different curves.

### The GZIP long-context filter

This is the cleverest small idea in the data section. How do you select documents that actually exercise long-range dependencies (vs. a 50K-token log file that's locally repetitive and teaches nothing about long context)?

OLMo 3's answer: **heuristic GZIP compressibility filtering** — remove any document in the **top and bottom 20%** of GZIP compressibility. The intuition:

- **Too compressible** (bottom 20% of entropy) ⇒ highly repetitive (logs, boilerplate, templated tables). No genuine long-range structure to learn.
- **Too *in*compressible** (top 20%) ⇒ near-random (hashes, base64 blobs, minified assets). No learnable structure at all.
- The middle band is where real long documents live: books, papers, multi-file codebases — locally varied but globally coherent.

The report notes this **outperforms model-based perplexity filtering** for identifying long-range dependencies. A `gzip` call beating a forward pass is the kind of result that should make you suspicious of your own over-engineered pipelines.

```python
import gzip

def gzip_ratio(text: str) -> float:
    """Compressed-size / raw-size. Low ratio = repetitive, high ratio = random."""
    raw = text.encode("utf-8")
    comp = gzip.compress(raw, compresslevel=6)
    return len(comp) / max(1, len(raw))

def longctx_keep(docs, lo_pct=0.20, hi_pct=0.80):
    """Fit the 20th/80th-percentile thresholds on a sample, then keep the middle band."""
    ratios = sorted(gzip_ratio(d) for d in docs)
    lo = ratios[int(lo_pct * len(ratios))]
    hi = ratios[int(hi_pct * len(ratios))]
    return [d for d in docs if lo <= gzip_ratio(d) <= hi]
```

### Decontamination

A fully-open lab cannot quietly let test sets leak into training — every claim is auditable. OLMo 3 runs `decon`, an open decontamination tool, against its eval suite, and (as we'll see in RL-Zero) treats decontamination as a *first-class experimental variable*, not a checkbox.

## Data mixing: a swarm of proxy models

Choosing the *proportions* of a 6T-token mix is the highest-stakes pretraining decision and is usually made by intuition + a couple of expensive ablations. OLMo 3 instead turns it into an optimization problem.

![Swarm-based data mixing: proxy models and a fitted predictor choose the mix](/imgs/blogs/olmo-3-training-finetuning-techniques-3.png)

The procedure:

1. **Sample candidate mixtures.** Draw ~5× as many data sources as you'll ultimately use, and generate many candidate mixture weightings over them.
2. **Train a swarm of 30M-parameter proxy models**, each on **3B tokens**, one per candidate mixture. These are cheap enough to run by the dozen.
3. **Evaluate each proxy** on the *Base Easy* suite, using **bits-per-byte** (negative log-likelihood ÷ answer bytes) as the proxy metric — a smooth, low-variance signal that shows up even at 30M scale.
4. **Fit a generalized linear model** that predicts proxy performance from the mixture weights: `perf ≈ f(mix)`.
5. **Optimize the predicted performance under constraints** — total token budget, per-domain repetition caps — to pick the final 6T mix.

The trick that makes this practical at scale is **conditional mixing.** When you add or change a data source, you don't want to re-run the entire swarm. Instead, you treat the *previous optimal mixture* as a single "virtual" domain and only optimize over the new/modified sources against it. This turns "we got new data" from a multi-day re-search into a cheap incremental fit.

| Approach | Cost to choose a mix | Re-cost when data changes | Signal |
|---|---|---|---|
| Intuition + a few full ablations | days of full-scale runs | repeat from scratch | noisy, few points |
| **Swarm + GLM (OLMo 3)** | dozens of 30M × 3B runs | **incremental (conditional mixing)** | dense, fitted surface |

If you take one idea from the pretraining half of this paper into your own work, it's this: **proxy-model swarms make data mixing a measured optimization instead of a vibe.**

## Stage by stage: the pretraining curriculum

The three pretraining stages are not just "more data, then better data" — each has a distinct *job*, and the transitions between them are where the curriculum logic lives.

### Stage 1 — Pretraining (the 6T bulk run)

This is the broad-capability run on **Dolma 3 Mix (5.9T tokens)** at **8K context**. The job here is breadth: language, world knowledge, basic code and math fluency. The deliberate choices:

- **Code- and math-heavy mix from the start.** Unlike older recipes that treat code as a late-stage add-on, OLMo 3 front-loads it, on the now-well-supported theory that code and math pretraining improve *general* reasoning, not just code/math benchmarks.
- **8K context only.** Cheap attention for the expensive bulk of training; long context is deferred to Stage 3.
- **The swarm-optimized mixture** (from the section above) governs the proportions, so even this "bulk" stage is tuned, not default.

### Stage 2 — Mid-training (the 100B anneal)

Mid-training on **Dolma 3 Dolmino (100B tokens from a 2.2T pool)** is where capability gets concentrated. Think of it as a learning-rate-decay phase pointed at the *hardest, highest-value* data: competition math, science, code with solutions, instruction-following examples, reading comprehension, and — critically — **reasoning traces**. Several things make this stage load-bearing:

- **Signal density.** 100B tokens of curated hard material late in training, when the LR is decaying and the model is most plastic to high-quality signal, moves benchmarks far more per token than the same data would have moved them at the start of Stage 1.
- **It seeds the reasoning behavior** that post-training will later amplify. By the time SFT begins, the base model has already *seen* reasoning traces; SFT is teaching format and consistency, not the concept from scratch.
- **It's a separate, swappable stage.** Because mid-training is decoupled, you can iterate on the Dolmino mix (a 100B run) without touching the 6T base run. This modularity is the whole reason the team can ablate data choices at a sane cost.

### Stage 3 — Long-context extension (the 50B stretch)

The final stage trains on **Dolma 3 Longmino (~50B tokens from a 639B pool)** — the GZIP-filtered long documents from earlier — combined with mid-training data, while applying **YaRN** to stretch RoPE on the full-attention layers out to ~65K. Two details matter:

- **Mixing long docs *with* mid-training data**, rather than training purely on long documents, prevents the model from forgetting the hard-won Stage-2 capabilities while it learns to track long-range dependencies. Pure long-context fine-tuning is a known way to degrade short-context reasoning; the blend is the guardrail.
- **Only ~50B tokens.** Long-context ability is cheap to *install* once the underlying model is strong — it's largely a positional-encoding and attention-routing skill, not a knowledge skill. Spending 50B of your ~6.05T total here, at the very end, is the efficient allocation.

| Stage | Context | LR regime | Primary job | Failure if skipped |
|---|---|---|---|---|
| 1 Pretrain | 8K | warmup → high | breadth, fluency | no foundation |
| 2 Midtrain | 8K | decaying | concentrate hard skills | weak math/code/reasoning |
| 3 Long-ctx | 8K → 65K | low | track long inputs | context collapse beyond 8K |

## Experimental design and evaluation

OLMo 3 treats evaluation as infrastructure, which is the only way the data-mixing loop above can even function (you can't optimize against a metric you don't trust).

![OlmoBaseEval taxonomy: a cheap proxy tier and a full-scale decision tier](/imgs/blogs/olmo-3-training-finetuning-techniques-7.png)

**OlmoBaseEval** is a **43-benchmark** suite organized into task clusters — *MC STEM, MC non-STEM, Math, Code, Code FIM* — where the clustering itself is data-driven: they clustered **~23K scores from ~70 open-weight models** to discover which benchmarks actually move together. Benchmarks are then split into two tiers:

- **Base Easy** — tasks that already show signal at small scale, scored with **bits-per-byte**. This is the tier the proxy swarm optimizes against.
- **Base Main** — tasks not yet saturated at large scale, used to judge full-scale runs.

**Why bits-per-byte and not accuracy?** A 30M proxy model trained on 3B tokens scores near-random on multiple-choice accuracy — the signal is buried in noise, and you can't optimize against noise. Bits-per-byte (the model's negative log-likelihood of the *correct* answer string, normalized by its byte length) is a *continuous* surrogate: even a weak model assigns slightly higher probability to the correct answer of an easy question, and that gradient is visible long before accuracy moves off the floor.

```python
import math

def bits_per_byte(model, prompt, answer):
    """Lower is better. Smooth even when MC accuracy is at random chance."""
    nll_nats = model.sum_neg_log_likelihood(answer, given=prompt)  # natural log
    n_bytes = len(answer.encode("utf-8"))
    return (nll_nats / math.log(2)) / n_bytes        # nats -> bits, per byte
```

A worked intuition: suppose two candidate data mixes produce proxy models A and B. On accuracy both score 25% (random, 4-way MC) — indistinguishable. But model A averages **0.91 bits/byte** on the correct answers and B averages **0.97**. A is *measurably* assigning more probability mass to correct continuations; the swarm picks A's mix, and that 0.06-bit edge at 30M scale reliably predicts a real accuracy gap at 7B+ scale. This predictiveness — small-scale BPB → large-scale accuracy — is the empirical bet the entire fast inner loop rests on.

This two-tier split is the quiet enabler of the whole methodology: the cheap, smooth tier drives the fast inner loop (data mixing, ablations), while the harder tier is reserved for the expensive outer-loop decisions. Around it sits the **OLMES** harness and the **`decon`** tool, plus explicit **"vibe checks"** — researchers reading diverse generations by hand — acknowledged as a complement to, not a replacement for, metrics. (For instruct models they add the **Berkeley Function-Calling Leaderboard**, **LitQA2**, and **SimpleQA**.)

## Olmo 3 Think — the reasoning model

Think is the flagship. Its post-training is a strict three-stage pipeline: **SFT → DPO → RLVR**. Each stage has its own Dolci mix and its own load-bearing trick.

### Stage 1 — Supervised fine-tuning with Dolci Think SFT

The SFT stage teaches the model to produce long chain-of-thought *at all*. Specifics:

- **~2.3M examples** spanning math, science, coding, instruction following, chat, and safety.
- **Completions generated by DeepSeek-R1 or QwQ-32B**, *with reasoning traces* — i.e., OLMo 3 Think learns to reason by distilling from existing strong reasoning models, then surpasses them on some axes via RL.
- **Correctness filtering per domain** — test-case execution for code, verifiers for math/IF — so the SFT set is not just stylistically reasoning-shaped but actually *correct*.
- **Two epochs**, with the final model produced by **weight-merging checkpoints trained at different learning rates.** This LR-merge is a cheap variance-reduction / ensembling trick that recovers some of the robustness you'd otherwise get from a much longer hyperparameter search.
- Run on **OLMo-Core**, reported as **~8× faster** than the prior Open-Instruct implementation — an infrastructure win that makes the whole post-training loop iterable.

```python
def build_sft_example(prompt, teacher, verifier):
    """Distill a verified long-CoT trace from a strong teacher (R1 / QwQ-32B).

    Step 1: generate a chain-of-thought completion from the teacher.
    Step 2: verify correctness with a domain-specific checker.
    Step 3: keep only verified traces (drop incorrect reasoning).
    """
    completion = teacher.generate(prompt, reasoning=True)   # includes <think> trace
    if not verifier(prompt, completion):                    # tests / sympy / IF-constraints
        return None                                          # drop incorrect traces
    return {"prompt": prompt, "completion": completion}
```

### Stage 2 — Preference tuning with Delta Learning

This is the most quietly clever post-training idea in the report. Standard preference data is expensive: you need humans (or a strong judge) to rank pairs. OLMo 3's **Delta Learning** sidesteps the ranking problem by *manufacturing* a reliable quality gap.

![Delta Learning: a strong and weak model produce a clean preference gap for DPO](/imgs/blogs/olmo-3-training-finetuning-techniques-5.png)

The construction: for each prompt, generate the **chosen** response from a **strong** model (Qwen-3-32B) and the **rejected** response from a **weak** model (Qwen-3-0.6B). You don't need to judge which is better — you *know*, by construction, that the 32B output is on average higher quality than the 0.6B output. DPO then trains the policy to prefer the strong-model distribution over the weak-model one.

- **150K pairs for the 7B**, **200K pairs for the 32B**.
- The chosen/rejected gap (the "delta") is the learning signal; the absolute quality of either side matters less than the *reliability* of the gap.

Two findings from this stage are worth pinning to the wall:

1. **DPO gives substantial gains on math and reasoning benchmarks** — preference tuning is not just a "politeness/format" stage, it moves hard-capability numbers.
2. **DPO is a better *starting point for RL* than more SFT.** Given a fixed budget, the report finds it's better to do SFT → DPO → RL than to pour the DPO budget back into SFT. DPO seems to put the policy in a region of parameter space from which RL explores more productively.

### Stage 3 — Reinforcement learning with OlmoRL ("the cherry on top")

RLVR — Reinforcement Learning with Verifiable Rewards — is the final stage, and the report's section title ("The Cherry on Top") is honest about its marginal-but-real contribution: SFT + DPO does most of the work; RL squeezes out the last several points on the hardest benchmarks.

![OlmoRL asynchronous training loop with in-flight weight sync and TIS reweighting](/imgs/blogs/olmo-3-training-finetuning-techniques-6.png)

**The algorithm — enhanced GRPO.** OLMo 3 starts from Group Relative Policy Optimization and applies a stack of by-now-established fixes:

- **No KL-divergence term.** The classic GRPO/PPO KL-to-reference penalty is dropped — for verifiable-reward RL on reasoning, it mostly throttles useful exploration.
- **Token-level loss normalization** rather than sequence-level, so long correct traces aren't down-weighted relative to short ones.
- **Zero-gradient filtering and active sampling** — skip prompts whose entire group is all-correct or all-wrong (zero advantage, zero gradient), and actively oversample prompts in the productive difficulty band.
- **Adjusted clipping upper bound** (the "clip-higher" idea) to allow the policy to move further on confidently-good actions.
- **Truncated Importance Sampling (TIS).** This one is subtle and important: the *inference* engine (vLLM-style, for generating rollouts) and the *training* engine compute token log-probabilities slightly differently. Off-policy RL across two engines therefore optimizes a biased objective. TIS re-weights the GRPO objective by the (truncated) ratio of the two engines' log-probs, correcting the mismatch. As async RL with separate inference/training backends becomes standard, this engine-mismatch correction becomes a must-have.

```python
def grpo_tis_loss(logp_train, logp_infer_old, logp_train_old, A, eps_lo, eps_hi, c):
    """Enhanced GRPO objective with TIS reweighting (schematic, per token).

    logp_train      : token logprob under the training engine.
    logp_infer_old  : token logprob under the inference (rollout) engine.
    logp_train_old  : token logprob under the training engine at rollout time.
    A               : group-relative advantage = (reward - group mean) / group std.
    """
    ratio = (logp_train - logp_train_old).exp()                  # policy ratio
    clipped = ratio.clamp(1 - eps_lo, 1 + eps_hi)                # asymmetric clip
    pg = -torch.min(ratio * A, clipped * A)                     # PPO-style surrogate

    tis = (logp_train_old - logp_infer_old).exp().clamp(max=c)  # truncated IS weight
    return (tis * pg).mean()                                    # token-level mean, no KL
```

**The rewards** are domain-specific and, where possible, *verifiable*:

| Domain | Reward signal |
|---|---|
| Math | binary correctness via `sympy` normalization + equivalence checking |
| Code | test-case execution (binary, or pass-ratio) |
| Instruction following | constraint-specific verifiers (e.g., "answer in JSON", "≤ 3 sentences") |
| Chat (non-verifiable) | LLM judge (Qwen-3-32B), reference-based or reference-free |

**The infrastructure (OlmoRL)** is where the real engineering is. RLVR's cost is dominated by *generation*, not by gradient updates — the report measures rollouts consuming **5–14× more compute** than policy updates. So the wins are all about keeping the generator busy:

- **Fully asynchronous, off-policy** — the trainer and the rollout engine run concurrently rather than lock-stepping.
- **Continuous batching** — new generation requests slot into the rollout engine as soon as slots free up, instead of waiting for a full batch to finish.
- **In-flight weight updates** — the policy's new weights are streamed into the rollout engine *without pausing generation*. No "stop the world to sync weights" barrier.

The combined effect: **~4× throughput** and wall-clock for an RL run dropping from **15+ days to ~6 days.** That is the difference between RL being a once-a-quarter event and a routine iteration.

### Olmo 3 Think — key findings & results

The headline: **Olmo 3-Think 32B narrows the gap to Qwen 3 32B's thinking models while using roughly 6× fewer training tokens.** Selected 32B-Think scores:

| Benchmark | Olmo 3-Think 32B |
|---|---|
| MATH | 96.2 |
| AIME 2024 | 80.6 |
| OMEGA | 53.4 |
| BigBench-Hard | 88.6 |
| HumanEval+ | 91.5 |
| IFEval | 93.8 |
| MMLU | 86.4 |

Against the field: it ties or wins among **fully-open** models (Marin 32B, Apertus 70B), sits within ~2 points of the best **open-weight** model on MATH / OMEGA / BBH / HumanEval+, clears Gemma 3 27B comfortably on reasoning and coding — and trails Qwen 3 32B specifically on **knowledge** (MMLU and similar). That knowledge gap is the recurring theme, and we return to its cause in the critique.

## Olmo 3 Instruct — the chat/agent variant

Instruct (released at 7B) is the non-thinking, fast-response sibling: multi-turn chat, instruction following, and tool use, without the long visible reasoning trace. Structurally it mirrors Think — **Dolci Instruct SFT → Dolci Instruct DPO → Dolci Instruct-RL** — but the mixes are tuned for breadth and latency rather than for hard reasoning:

- **SFT** on instruction-following, chat, tool-use, and safety data (shorter, non-CoT completions).
- **DPO** with the same Delta-Learning-style strong-vs-weak pairing.
- **RL** (Dolci Instruct-RL) with verifiable rewards weighted toward instruction-following constraints and tool-call correctness rather than competition math.

The reported outcome: **Olmo 3-Instruct matches or outperforms Qwen 2.5, Gemma 3, and Llama 3.1** at similar scale, and is **competitive with Qwen 3** on several instruction and reasoning benchmarks — with function-calling measured on the Berkeley leaderboard and factuality on SimpleQA/LitQA2. The lesson mirrors Think: the same pipeline, re-pointed at a different reward profile, yields a different product. The flow is the asset.

## Olmo 3 RL-Zero — and the spurious-reward debunk

RL-Zero is the most scientifically interesting variant precisely because it is *not* trying to be the best model. It is a clean experimental apparatus: **apply RLVR directly to the base model, with no SFT or DPO in between.** Ai2 releases the **Dolci RL-Zero** dataset (decontaminated, spanning math, code, precise instruction following, and mixed objectives) plus **four series of checkpoints** from domain-focused RL.

Why does this matter? Because "RL from base" is the setting where a string of 2024–2025 papers reported a startling result: **RLVR with random or spurious rewards still improves benchmarks.** If true, that's deeply weird — it implies RL isn't teaching the model anything reward-specific, just eliciting latent capability (or worse, exploiting eval leakage).

OLMo 3's finding, with a properly **decontaminated** setup:

> **RLVR with random rewards no longer benefits model performance once the RL prompt set is decontaminated against the eval suite.**

In other words: the earlier "spurious rewards work" results appear to have been **contamination artifacts** — the base model had effectively seen the test items, and *any* RL signal (even noise) nudged it toward regurgitating them. Remove the contamination, and the free lunch disappears. This is the kind of negative result that only a fully-open, decontamination-first pipeline can credibly produce, and it should recalibrate how you read RL ablations elsewhere.

One more practical RL-data detail surfaces here: **offline difficulty filtering.** Before RL, prompts with **> 62.5% pass rate** on the base model are removed — they're too easy to provide useful gradient. You spend your rollout budget on the band of prompts the model gets right *sometimes*, which is where the learning signal concentrates. This pairs with the *online* zero-gradient filtering in the GRPO loop: difficulty filtering removes hopelessly-easy prompts up front, and zero-gradient filtering removes any prompt whose sampled group happens to come out all-right or all-wrong on a given step. Together they keep nearly every rollout in the informative middle band — the single biggest lever on RL sample-efficiency, and one that costs nothing but bookkeeping.

### The four RL-Zero series as a research instrument

The release ships **four separate checkpoint series** from domain-focused RL — math, code, precise instruction following, and general chat — each trained *from base* with only that domain's reward. This is not a product decision; it's an experimental one, and it's the kind of thing only a fully-open lab bothers to publish. It lets the community ask questions that are normally impossible to study cleanly:

- **How much of "reasoning" is domain-transferable?** With a math-only RL-Zero checkpoint and a code-only one, you can measure whether RL on math improves code (and vice versa) *without* the confound of a shared SFT stage that already mixed both.
- **What does RL add over the base model's latent capability?** Because there's no SFT/DPO in between, any benchmark movement is attributable to RL alone. This is the cleanest available setting for the "does RL teach or merely elicit?" debate.
- **Where does the verifiable-reward approach break down?** The general-chat series uses an LLM judge (non-verifiable reward); comparing its stability and reward-hacking behavior against the math/code series (verifiable) isolates exactly what you lose when you can't check the answer.

The non-verifiable chat case deserves a flag of its own: using **Qwen-3-32B as the reward judge** means OLMo 3's chat RL is, in part, optimizing toward *another model's preferences*. That's pragmatic and standard, but it's a reward-hacking surface — the policy can learn to produce outputs the judge likes rather than outputs that are good. The verifiable domains (math via `sympy`, code via test execution) have no such loophole, which is precisely why the report leans on them for its strongest claims and treats chat reward more cautiously. The general lesson for your own RL work: **verifiable rewards are not just convenient, they're the only rewards you can fully trust not to be gamed** — reserve LLM-judge rewards for the domains where nothing better exists, and watch them like a hawk.

## Costs

The report is unusually candid about cost, which matters because "fully open" should include the bill. The grounded figures: pretraining on **up to 1,024 H100s** at **~7.7K tok/device/s** (7B); mid-training on **128 H100s**; post-training on **256 H100s**; and the RL wall-clock improvement from **15+ days → ~6 days** via OlmoRL. The strategic claim underneath the numbers is that **Olmo 3-Think reaches its quality with ~6× fewer training tokens than the Qwen 3 comparison point** — i.e., the recipe (mid-training concentration + strong-teacher SFT + DPO + efficient RL) substitutes for raw token volume. For a lab without hyperscaler compute, that substitution is the whole game: you cannot out-spend Alibaba on tokens, but you *can* out-engineer them on token efficiency, and the cost table is the receipt showing it worked at a scale a well-funded university lab can actually afford.

## Critique — the senior-engineer lens

**What's genuinely strong.**
- *Reproducibility as a feature, not a slogan.* Every claim above is auditable against released artifacts. That is rare and valuable, and it's what makes the spurious-reward debunk credible — you can check it.
- *Methodological discipline.* Data mixing as measured optimization (swarm + GLM), evaluation as a two-tier instrument, decontamination as a controlled variable. This is how data-centric ML *should* look.
- *Engineering that compounds.* The 8× SFT speedup and 4× RL throughput aren't headline numbers, but they're what made the iteration loop fast enough to find the other results.

**What's weak / unfalsifiable / needs caution.**
- *The knowledge gap is partly self-inflicted by openness.* The report effectively concedes that **some high-signal sources are removed to retain a commercial license**, while open-weight competitors disclose nothing and pay no such tax. So "OLMo 3 trails Qwen 3 on MMLU" conflates *recipe quality* with *legal constraints* — you cannot cleanly attribute the gap. An ablation adding back the licensed data (even if unreleasable) would isolate this; it's absent.
- *Delta Learning's ceiling is the strong teacher.* Manufacturing preferences from Qwen-3-32B-vs-0.6B means the preference signal is bounded by Qwen-3-32B's taste. It's cheap and scalable, but it can't teach the policy anything the 32B teacher doesn't already prefer — a structural cap on how far DPO can push beyond the teacher.
- *"~6× fewer tokens" is a favorable framing.* Token count isn't the only cost axis; the strong-teacher distillation (R1/QwQ/Qwen-3 generations) imports capability that those teachers paid for in *their* token budgets. The comparison is fair for *Ai2's* compute, but the total societal compute behind the model is higher than the 6T headline.
- *RL is honestly labeled "the cherry on top"* — which is refreshing, but also means the most fashionable part of the pipeline is the least load-bearing. If your goal is capability-per-dollar, the report quietly argues you should invest in data and SFT/DPO before RL.

**What would change my mind:** a single controlled ablation holding architecture, SFT, and DPO fixed while varying *only* the pretraining data license-filter — showing how much of the Qwen-3 knowledge gap is the filter vs. the recipe. If most of the gap is the filter, OLMo 3's recipe is essentially at parity with closed-data frontier practice, and the headline should be much louder.

## What I'd build with this

1. **Steal the GZIP long-context filter today.** It's ~15 lines, needs no model, and the report says it beats perplexity filtering for long-range selection. Drop it in front of any long-context fine-tuning set you're assembling.
2. **Proxy-swarm your own data mixing.** You don't have 1,024 H100s, but you can afford a handful of 30M-param proxy runs on 3B tokens to fit a `perf = f(mix)` surface before committing your real budget. The conditional-mixing trick makes it cheap to maintain as your corpus changes.
3. **Manufacture preference data with Delta Learning.** For any DPO setup where labeling is the bottleneck, pair a strong and a weak model on shared prompts. No judge, reliable gap. Validate that the gap survives your domain before trusting it.
4. **Adopt TIS the moment you go async.** If you split rollouts (vLLM/SGLang) from training (FSDP/Megatron), the engine log-prob mismatch is real and silently biases your objective. Truncated importance sampling is the standard fix; bake it in from day one rather than debugging a mysterious reward plateau later.
5. **Decontaminate *before* you draw conclusions from RL.** The spurious-reward debunk is a warning: any RL-from-base result you can't reproduce on a decontaminated prompt set is suspect. Run `decon` (or equivalent) as a gate, not an afterthought.

## References

- **OLMo 3 Technical Report** — [arXiv 2512.13961](https://arxiv.org/abs/2512.13961) ([PDF](https://arxiv.org/pdf/2512.13961))
- **Ai2 blog: "Olmo 3: Charting a path through the model flow"** — [allenai.org/blog/olmo3](https://allenai.org/blog/olmo3)
- **Models & data on Hugging Face** — [allenai/Olmo-3-1125-32B](https://huggingface.co/allenai/Olmo-3-1125-32B)
- Open infrastructure: OLMo-Core, Open-Instruct, `datamap-rs`, `duplodocus`, OLMES, `decon`, OLMoTrace.

Sibling deep-dives on this blog:

- [Fine-tuning an LLM with GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) — the RL algorithm OLMo 3 enhances.
- [Fine-tuning an LLM with DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) — the preference-tuning method behind Delta Learning.
- [Pretraining large reasoning models](/blog/machine-learning/large-language-model/pretraining-large-reasoning-models) — the broader context for mid-training and long-CoT.
- [Distillation in LLMs](/blog/machine-learning/large-language-model/distillation-in-llm) — what the R1/QwQ-traced SFT stage is really doing.
