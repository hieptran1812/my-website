---
title: "MAI-Thinking-1: Building a Hill-Climbing Machine"
date: "2026-07-09"
description: "A full walkthrough of Microsoft AI's first from-scratch reasoning model — how they pre-train a 35B/1T MoE, judge every decision with Efficiency Gain, and climb it with GRPO from a base with no reasoning traces."
tags: ["paper-reading", "large-language-model", "mixture-of-experts", "reinforcement-learning", "grpo", "reasoning-models", "moe", "rlhf", "pretraining", "scaling-laws"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 41
paper:
  title: "MAI-Thinking-1: Building a Hill-Climbing Machine"
  authors: "The Microsoft AI Team"
  venue: "Technical report, 2026"
  url: "https://microsoft.ai/pdf/mai-thinking-1.pdf"
---

> [!tldr]
> - **What it is.** MAI-Thinking-1 is Microsoft AI's first in-house reasoning model: a **35B-active / 1T-total** sparse Mixture-of-Experts (MoE), pre-trained from scratch on 30 trillion tokens and then taught to reason with reinforcement learning. It is trained on clean, licensed human data with **no distillation from any third-party model**.
> - **The thesis.** Progress comes not from one model but from a *hill-climbing machine* — the integrated loop of data pipelines, training infrastructure, RL environments, rewards, and evals that turns model development into an empirical optimization problem. Every architectural and data decision is judged by a single number, *Efficiency Gain*, measured along a *scaling ladder*.
> - **The surprising part.** The RL climb starts from a base model that has *never seen a reasoning trace*, and still produces smooth, **log-linear** improvement over thousands of steps (Figure 1 below). Two small changes to GRPO — an entropy-controlled clip and a hard outer clip — plus self-distillation are what make climbs that long survivable.
> - **The results.** 97.0% AIME 2025, 94.5% AIME 2026, 87.7% LiveCodeBench v6, 52.8% SWE-Bench Pro, 73.5% SWE-bench Verified — broadly competitive with Claude Sonnet 4.6, and preferred over it in human side-by-sides.
> - **Where it falls short.** It does not lead the frontier: it trails Claude Opus 4.6 in human preference and on SWE-Bench Pro, scores only 46.0% on Terminal-Bench 2.0, and the report withholds the data-provider list, the exact mixture, and the fitted scaling-law constants — so the most load-bearing claims are hard to reproduce.

The diagram below is the whole paper at a glance — three benchmark-accuracy curves that climb steadily on a log-step axis. The rest of this post unpacks how a from-scratch base model and a carefully instrumented RL loop produce a line that keeps going up.

![Figure 1 from The Microsoft AI Team (2026): pass@1 during MAI-Thinking-1's reinforcement-learning climb on AIME 2025, a hard subset of LiveCodeBench v6, and SWE-bench Verified, plotted against training step on a log axis.](/imgs/blogs/mai-thinking-1-fig2.webp)

## The problem: optimize the process, not the model

Most technical reports are organized around *a model*: here is the architecture, here is the data, here are the numbers. MAI-Thinking-1 is organized around *a process*. The paper's framing is that a single checkpoint is a fixed point, but capability comes from **the rate at which you can improve the current best checkpoint** — and that rate is itself an engineering artifact you can design, measure, and optimize.

That reframing has teeth because the naive way to compare two ideas — train both, see which scores higher — is treacherous at frontier scale. The apparent benefit of an architecture tweak or a new dataset routinely *shrinks* as the compute budget grows: a change that looks like a 5% win at 1B parameters can wash out entirely at 100B. If you accept small-scale wins uncritically, you accumulate a pile of decisions that individually looked good and collectively buy nothing. The paper calls the disciplined alternative *hill climbing*, and commits to three principles:

1. **Capabilities should be learned, not inherited.** Distilling from a stronger model is faster but, the authors argue, imports that model's ceiling and its idiosyncrasies while giving up the steerability and robustness you need to keep climbing. So MAI-Thinking-1 is trained from scratch and, notably, its RL climb begins from a checkpoint that has *never* seen a chain-of-thought.
2. **Simplicity is sustainable.** Simple, scalable recipes and transparent infrastructure survive thousands of steps; clever-but-fragile ones collapse.
3. **Scientific rigor avoids shortcuts.** Every decision is tested with scaling ladders, ablations, and evaluations that expose whether an improvement *persists* as scale grows.

This lineage matters. The RL machinery is a direct descendant of [DeepSeek-R1's demonstration that reasoning can be incentivized by pure RL](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) — MAI even uses R1's `<think>...</think>` prompt template to bootstrap. The pre-training rests on the [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) framework and [MoE scaling laws](/blog/machine-learning/scaling-laws/moe-scaling-laws). And the whole pre-train → mid-train → RL decomposition echoes the [interplay between those three stages](/blog/paper-reading/large-language-model/pre-training-mid-training-and-rl-interplay) that recent open models have converged on. The gap MAI claims to fill is *methodological*: a report that documents the machine — the ladders, the metric, the failure recoveries — rather than just the model.

## What the paper actually contributes

Stripped to its load-bearing ideas, the report contributes:

1. **A scaling-focused pre-training methodology** — a *scaling ladder* of model sizes and a single scalar, *Efficiency Gain*, that says how much extra compute the baseline would need to match a candidate. Every architecture and data decision is accepted only if its Efficiency Gain *holds up* as you climb the ladder.
2. **An architecture co-designed with the hardware** — interleaved high-sparsity MoE and dense FFN layers, 5:1 local/global attention, and a *latent* MoE that compresses tokens before the all-to-all, chosen because it is fast on GB200 NVLink domains, not just cheap in FLOPs.
3. **A from-scratch RL recipe that sustains long climbs** — GRPO with two modifications (adaptive entropy control, an outer ratio clip), a difficulty-aware reward, and *self-distillation* that lets a run survive collapses and base-model swaps across thousands of steps.
4. **Three specialist climbs consolidated into one model** — separate RL runs for STEM/coding, agentic tool use, and helpfulness/safety, distilled into a single model with SFT and a final light RL pass.
5. **The surrounding infrastructure** — YOLO (a from-scratch, bitwise-deterministic training framework) and Rocket (an asynchronous off-policy RL system) that make all of the above run at 8K-GPU scale.

The figure below is the mental model for the whole thing: five sequential optimization stages, each an instance of the hill-climbing loop, turning raw human data into MAI-Thinking-1.

![A left-to-right pipeline of five stages — in-house data (30T tokens, no distillation), pre-training MAI-Base-1 (35B/1T MoE, 16K context, 8K GB200), mid-training (+3.55T tokens, context to 256K), three from-scratch RL climbs (8K to 128K rollouts, 4.6K GB300), and consolidation (SFT distill plus a final RL pass) producing MAI-Thinking-1.](/imgs/blogs/mai-thinking-1-2.webp)

We take the two halves in turn: Part I builds the base model **MAI-Base-1**; Part II climbs it into the reasoning model **MAI-Thinking-1**.

## Part I — Building MAI-Base-1 (the pre-training half)

MAI-Base-1 is a 35B-active / 1T-total sparse MoE, pre-trained from scratch on 8,192 GB200 GPUs. The corpus is 30T tokens of main pre-training plus 3.55T tokens of mid-training, all built in-house from public and licensed human data, with **no** language-model-generated synthetic data.

### The architecture: interleaved sparsity, periodic attention, latent MoE

**The problem.** A dense transformer spends the same FLOPs on every token, and a uniform-MoE spends the same *routing* everywhere. Neither is the FLOP- and bandwidth-optimal shape once you have 8K GPUs whose fast interconnect is an intra-rack NVLink domain and whose slow interconnect is cross-rack InfiniBand. The architecture question is not "what is theoretically best" but "what learns the most per wall-clock second *on this cluster*."

**The intuition.** Think of the network as a factory line. Some stations do generic work that every product needs (a *dense* feed-forward layer); others are a bank of specialists where each product visits only the two or three specialists it needs (a *sparse* MoE layer). MAI interleaves them: one dense station, then one very-sparse station with 512 specialists of which each token visits 8. Doing a *few* extremely-sparse stations plus dense stations turns out to learn as much as making *every* station moderately sparse — but it moves less data between racks, so it finishes sooner.

**The mechanism.** The figure below is the authors' own diagram. On the left is the transformer body: pairs of blocks, each block a pre-norm attention sublayer and a feed-forward sublayer, with RMSNorm applied at both the input *and* the output of each sublayer right before the residual add. The feed-forward alternates dense FFN and sparse MoE. Attention alternates too — five *local* (sliding-window) layers per one *global* layer. On the right is the MoE layer: a token's hidden vector is first **down-projected** into a compressed latent space, then **dispatched** (the all-to-all that sends each token to its chosen experts across GPUs), processed by the experts, **combined** back, and finally **up-projected** to the original width. Crucially, the *router* reads the original (uncompressed) representation, but the expensive all-to-all moves the *compressed* one.

![Figure 2 from The Microsoft AI Team (2026): the MAI-Base-1 architecture. Left, the transformer body interleaving sparse MoE and dense FFN blocks and local and global attention. Right, the latent MoE layer routing 8 of 512 experts in a compressed latent space.](/imgs/blogs/mai-thinking-1-fig1.webp)

**The math.** For a token with hidden vector $x \in \mathbb{R}^{D}$ (here $D = 6656$), the router computes a score over experts and picks the top-$k$:

$$
g_e(x) = \operatorname{softmax}_e\!\big(x^\top W_r\big),\qquad
\mathcal{E}(x) = \operatorname{top\text{-}k}\big(g(x),\, k\big),\quad k=8,\ \text{of } 512 \text{ experts}.
$$

Here $W_r \in \mathbb{R}^{D \times 512}$ is the router matrix and $g_e(x)$ is the gate weight for expert $e$. The layer's output is the gate-weighted sum over the chosen experts, but with the *latent* twist — a shared down-projection $W_\downarrow \in \mathbb{R}^{D \times d}$ (compression factor $2\times$, so $d = D/2$) is applied *before* dispatch and an up-projection $W_\uparrow$ after combine:

$$
\text{MoE}(x) = W_\uparrow\!\Bigg(\sum_{e \in \mathcal{E}(x)} g_e(x)\; \text{FFN}_e\big(W_\downarrow x\big)\Bigg).
$$

Each $\text{FFN}_e$ is a SwiGLU expert operating in the $d$-dimensional latent space and expanding it $3\times$ internally. The win is bandwidth: the all-to-all that shuffles tokens between GPUs moves $d$-dimensional vectors instead of $D$-dimensional ones — half the bytes — while routing quality is preserved because the *decision* still uses the full $x$.

Two more choices earn their place:

- **Periodic attention (5 local : 1 global).** Local layers use a 512-token sliding window with RoPE (base frequency 10,000); global layers use *no* positional encoding at all — the paper reports NoPE performs comparably to RoPE on global layers while being cheaper. Pairing five cheap local layers with one global layer slashes both training attention cost and the inference KV-cache. They use grouped-query attention with 8 KV heads and per-head dimension 128, and RMSNorm the queries and keys for stability.
- **Global-batch load balancing.** With 512 experts, tokens can pile onto a few experts and starve the rest. MAI uses a GShard-style auxiliary balancing loss, but the key finding is that **the aggregation scope matters far more than the loss type**: as long as expert frequencies are aggregated across *all* data-parallel workers and micro-batches in the global batch, the classic loss and the loss-free variant perform about the same. They also went fully **dropless** — every token reaches its experts via variable-size all-to-all — after discovering that finite expert capacity (token dropping) can subtly change which conclusions you draw from ablations.

**Why it works / when it fails.** The interleaved layout wins on wall-clock, not FLOPs (we quantify this below): a uniform-MoE plus shared experts matches it in FLOP-efficiency but loses once you account for how fast the kernels actually run. The failure mode is the dropless MoE's Achilles heel — under heavy imbalance the all-to-all message sizes swing wildly, threatening memory blow-ups and OOMs, which is exactly the pathology the training-loss spikes later trace back to.

### Efficiency Gain: how every decision is judged

**The problem.** You cannot A/B two architectures by training both to the finish — that is the whole run. You need a metric that (a) is cheap to measure at small scale and (b) *predicts* which choice will still be ahead at full scale. Raw loss does not do this: two models at different sizes and token counts are not comparable, and "lower loss at fixed FLOPs" ignores that implementations differ in how efficiently they turn FLOPs into wall-clock.

**The intuition.** Fit a "conversion rate" from compute to loss for your current baseline. Then, for any candidate, ask: *how much compute would the baseline have needed to reach the loss this candidate reached?* Divide that by what the candidate actually spent. If the answer is 1.3, the candidate is worth a 30% compute discount — the baseline would have to spend 30% more to catch up.

**The mechanism.** Build a **scaling ladder**: train several model sizes at a fixed number of tokens-per-active-parameter (TPP), fit a scaling law to the baseline ladder, and place every candidate against that curve.

**The math.** Fit the baseline loss as a power law in training cost $C$ (FLOPs or wall-clock time):

$$
L = f(C) = A\,C^{-\alpha} + E,
$$

where $A$ sets the magnitude of the *reducible* loss, $\alpha$ is the exponent (how fast loss falls with compute), and $E$ is the *irreducible* loss floor. For a candidate run that reaches loss $L'$ at cost $C'$, invert the baseline law to find the cost the baseline would need for that loss, $f^{-1}(L')$, and define **Efficiency Gain**:

$$
\mathrm{EG} = \frac{f^{-1}(L')}{C'}.
$$

$\mathrm{EG} > 1$ means the candidate is more efficient than the baseline. When $C$ is FLOPs the metric ignores implementation quality on purpose (so a new architecture is not penalized for having un-optimized kernels); when $C$ is wall-clock they write it $\mathrm{EG}_\text{Time}$ and it *does* reward a fast implementation.

**Worked micro-example.** Suppose the baseline fit is $L = 2.0\,C^{-0.1} + 0.8$ (with $C$ in some FLOPs unit). A candidate reaches $L' = 1.20$ at $C' = 1000$. Invert: $1.20 = 2.0\,C^{-0.1} + 0.8 \Rightarrow C^{-0.1} = 0.2 \Rightarrow C = 0.2^{-10} \approx 9.77\times 10^{6}$. Then $\mathrm{EG} = 9.77\times 10^{6} / 1000 \approx 9.8\times 10^{3}$ — an extreme number that only illustrates the mechanics; realistic architecture EGs sit near 1.0–1.4. The point is the *procedure*: read the candidate's loss, find the baseline's equivalent cost, take the ratio.

```python
import numpy as np

# Baseline scaling-law fit L = A * C**(-alpha) + E
A, alpha, E = 2.0, 0.10, 0.80

def baseline_loss(C):            # forward: cost -> loss
    return A * C**(-alpha) + E

def baseline_cost(L):            # inverse: loss -> cost the baseline needs
    return ((L - E) / A) ** (-1.0 / alpha)

def efficiency_gain(L_cand, C_cand):
    return baseline_cost(L_cand) / C_cand

# candidate reached L'=1.20 spending C'=1000
print(efficiency_gain(1.20, 1000))   # ~9.8e3 in this toy fit
```

**Why it works / when it fails.** EG is only as trustworthy as the assumption that a candidate ahead at small scale stays ahead at large scale — the *rank-invariance* hypothesis. The single most valuable empirical result in Part I (next section) is that this assumption **fails** for data mixtures, which is why MAI insists on measuring EG *across* the ladder, not at one point.

The paper reports EG in two flavors to make a specific point. When they compare their interleaved "high-sparsity MoE + dense FFN" layout against the more common "medium-sparsity MoE everywhere," a MoE-every-layer variant with shared experts reaches $\mathrm{EG}_\text{FLOPs} = 1.03$ — nominally *better* in FLOPs — but $\mathrm{EG}_\text{Time} = 0.82$, clearly *worse* once real kernel speed on GB200 is counted. That gap is the entire justification for the interleaved design.

### Why NLL, not accuracy

For day-to-day development MAI evaluates on ~40 held-out **negative-log-likelihood** (NLL) benchmarks grouped into Code, STEM, Math, General Knowledge, and Multilingual, and collapses them into one number with fixed weights:

$$
\text{Target} = 0.5\cdot\text{Code} + 0.175\cdot\text{STEM} + 0.175\cdot\text{Math} + 0.1\cdot\text{Gen} + 0.05\cdot\text{Multiling}.
$$

The weights encode priorities — coding and reasoning dominate. Why NLL rather than "how many questions did it get right"? Three reasons, each a real operational pain the authors call out. First, **cost**: generative benchmarks need autoregressive decoding (and often a judge model), which is slow in a training framework not tuned for inference; NLL is the same next-token objective as training and runs almost for free, so they can run *more* evals more consistently. Second, **robustness to confounders**: multiple-choice formats assume the model has learned to *do* multiple choice (an ability that emerges only at large scale), and benchmarks like MATH or MBPP are sensitive to formatting trivia (`\boxed{}`, `\r\n` vs `\n`) that can swamp the signal you care about. Under NLL the model is teacher-forced on the ground-truth prefix, so a small early error cannot compound. Third, **construction cost**: a good NLL benchmark can start from any topical corpus, whereas a good Q&A benchmark needs expert-authored questions and difficulty calibration.

The catch they are candid about: **contamination**. Benchmarks leak into training data — especially via GitHub, where evals are re-published with solutions — and this leakage produces counterintuitive results (coding data mysteriously boosting long-tail trivia). Their defenses: drop all of `huggingface.co` and mirrors, apply a universal 20-gram fuzzy dedup at 80% similarity across every source, and — because those are imperfect — commission *private* benchmarks they are confident exist nowhere on the web.

### The data pipeline: clean, deduplicated, no synthetic text

Every token is processed in-house from one of three base sources — web HTML/PDFs, books and journals, and public GitHub — with a deliberate choice to use **no** LLM-generated synthetic pre-training text and to actively remove AI-generated content. When an LLM *is* used in the pipeline (for hard extraction cases), it is only ever allowed to *keep or remove* source text, never to *add* content.

The extraction philosophy is "no single method for everything": schema-aware parsers for structured formats, hand-crafted BeautifulSoup extractors for high-value domains, LLM/agent extraction for the hard cases, and — memorably — *training on raw markup* for sources like Wikipedia, where wikitext is ~3× more verbose but stripping it loses infoboxes and tables, and the raw version simply trains better.

Deduplication is treated as first-order, because a sparse 1T-parameter model memorizes aggressively and repetition degrades scaling — larger models "exhaust the supply of new information earlier." The pipeline stacks five kinds of dedup: boilerplate removal (line-occurrence statistics), exact (byte/hash), fuzzy (MinHash LSH at 0.8 similarity), templated-page skeletonization, and *semantic* dedup with a Qwen3-Embedding model to collapse independently-authored-but-near-identical documents (rampant in code — everyone writes the same binary-search-tree traversal). A final **cross-dataset** dedup uses a global drop-order: a duplicate is kept only in the highest-priority dataset it appears in. This makes drop-order a genuinely load-bearing hyperparameter — changing one dataset can silently shift data into or out of another.

### Choosing a data mixture (and the rank-non-invariance surprise)

**The problem.** Hundreds of heterogeneous sources, a fixed token budget, and pairwise interactions (datasets substitute for and complement each other). You cannot run full post-training per candidate mixture, the search space is astronomical, and — the sting — a source that helps at small scale can *hurt* at large scale.

**The mechanism.** MAI uses forecasting-based search (train thousands of small 760M–4B models on sampled mixtures, fit a predictor, optimize the NLL Target) but then validates the winner *up the ladder*. The final selection is a hierarchical alternation: **local search** varies weights *within* a category (code files vs PRs vs commits) with everything else frozen; **global search** varies the relative weights *between* ~10 categories. Throughout, no dataset may be repeated more than 8× (to guard against overfitting a small high-quality source).

The result worth internalizing is the figure below — a clean refutation of the *rank-invariance hypothesis*. They built a `stem-heavy-mix` and a `code-heavy-mix`. At 5B scale, `stem-heavy` wins the held-out STEM eval (left panel, the circles). But train both at 23B for ~20T tokens and the curves **cross** partway through (right panel): `code-heavy` ends up better on STEM. The forensic cause: two STEM sources with high quality but heavy fuzzy-duplication and low diversity had 11.8% weight in `stem-heavy` versus 0.3% in `code-heavy` — great for a small model, but their lack of diversity capped the larger one.

![Figure 6 from The Microsoft AI Team (2026): rank non-invariance in data-mixture scaling. Left, stem-heavy-mix beats code-heavy-mix on held-out STEM at 5B but loses at 23B. Right, the two 23B STEM curves cross midway through 20T tokens of training.](/imgs/blogs/mai-thinking-1-fig4.webp)

**Why it matters.** This is the empirical justification for the entire EG-across-the-ladder discipline. If small-scale rank were preserved, cheap experiments would settle everything; because it is not, every mixture decision has to be checked at multiple scales. The final mixture (Table 5 in the paper) lands at 54.6% code, 15.8% STEM, 5.4% math (sampled 5.28× — the most-repeated family), with web text and PDFs each seen *less than once* over 30T tokens.

Mid-training then re-weights this same corpus (no new sources) toward STEM/math (35%) and code (55%), filters STEM PDFs by a "Bloom Analyze" cognitive-level heuristic, and applies **memorization-aware epoch capping**: sources whose late-pre-training loss improvement comes disproportionately from near-certain tokens (NLL < 0.01, a memorization signature) get stricter epoch caps.

### The training recipe: dropout, zero-init attention, FP8

A few recipe choices are unusual enough to flag:

- **AdamW** with $\beta_1 = 0.95,\ \beta_2 = 0.925$, weight decay 0.1 (reduced to 0.01 on attention and 0.005 on embeddings), gradient-norm clip 1.0. The LR warms up over ~12B tokens then cosine-decays from $2\times10^{-4}$ to $2\times10^{-5}$ — a **final-to-peak ratio of 0.1×**, not the more common 0.01×, because they found *decaying less* improved post-RL results.
- **Dropout 0.15 at every layer's output** before the residual add. High dropout is non-standard for LLM pre-training, but they report it gives regularization complementary to weight decay and improves ladder-measured evals.
- **Zero-initialized attention.** At init, attention softmax is nearly uniform, so it acts as causal mean-pooling — collapsing token representations and, worse, feeding highly-correlated inputs into the MoE routers so a few experts get slammed (imbalance that grows with depth). The fix is elegant: initialize the attention output to *zero* by zeroing the output RMSNorm gains. The model then starts as a stack of per-token feed-forwards, and cross-token attention "kicks in" gradually as training proceeds. This measurably cut initial expert imbalance and improved EG.
- **Mixed precision.** BF16 weights/activations by default; FP8 E4M3 for forward GEMMs, FP8 E5M2 for the data-gradient, BF16 compute with FP32 accumulation for the weight-gradient. FP32 is reserved for numerically sensitive spots: all pre-softmax activations (attention scores, router logits, output logits), the MoE combine, the entire residual stream, and optimizer state. Stochastic rounding is used wherever gradients are downcast.

The 30T-token loss curve is clean, with a few early spikes that recovered without intervention — spikes the runtime logs traced to coding data hitting high expert imbalance under dropless routing, the exact failure mode flagged above.

### YOLO: distributed training and the MFU–EG co-design

The training framework, **YOLO** ("You Only Launch Once"), is built from scratch on PyTorch and implements the full stack: custom FP8/grouped-GEMM Triton/CUTLASS kernels, descriptive sharding annotations (so different tensors can have different parallelism — e.g. tensor-parallel embeddings and attention but data-parallel MLP/MoE), from-scratch ZeRO stages 1–3, Ulysses-style context parallelism, and a dropless-MoE implementation with a *static-memory* mode that runs capped dispatch→compute→collect rounds to bound memory under imbalance.

Two engineering commitments stand out. **Determinism**: for fixed hardware, config, and software version, two runs produce *bitwise-identical* models. That requires deterministic reductions everywhere (fixed-order tiled accumulations in RMSNorm backprop, stable-sort tie-breaking in top-k routing, NVLink SHARP disabled so collectives are order-stable) at a real MFU cost — but it buys reproducibility, easier debugging, and "golden" regression tests. **Fault tolerance**: a rewritten distributed-checkpoint path (>10× faster saves and lower CPU overhead), asynchronous checkpointing off the training processes, and Ray-actor hot-standbys for fast in-job restart. The pre-training run hit **90.0% goodput** at 8K GPUs, with the largest remaining overhead being MFU drops rather than crashes.

The most instructive systems figure is the architecture-generation history below. Reading the top panel: each new model version (v2→v5) *raises* Efficiency Gain but, when first run on the *previous* generation's optimization stack, *drops* MFU — because a better architecture introduces new bottlenecks. The bottom panel shows EG climbing $1.00\times \to 1.40\times \to 1.69\times$. Twenty-plus kernel/comms/memory optimizations per generation dragged MFU back above 20% each time.

![Figure 11 from The Microsoft AI Team (2026): MFU (top) and Efficiency Gain over v2 (bottom) across pre-training generations v2 to v5. Each architecture change raises EG but initially degrades MFU on the old optimization stack; 20+ optimizations recover it above 20%.](/imgs/blogs/mai-thinking-1-fig6.webp)

Concretely: v2 (23B, first GB200 run) went 18%→22% MFU via GPU-Direct RDMA, a custom block-sparse attention backend, ZeRO-2, and a Triton expert-encode kernel (10%→80% HBM utilization). v4 introduced the latent-MoE and 512 experts, dropping initial MFU to 16% until FA4 deterministic kernels and CPU-overhead work recovered it to 20%. v5 (the 35B/1T final) hit a ZeRO-3 communication wall, solved by activation offloading so they could revert to ZeRO-2 and restore compute/comms overlap at ~20% MFU.

### Long-context extension

The 256K context is reached in stages, not by training long throughout (long sequences have terrible MFU). MAI pre-trains at 16K, mid-trains at 64K, then does a short dedicated 256K extension. The striking ablation: a *progressive* recipe — mid-train at 32K then extend to 256K with only ~100B extra tokens — **matches** a full 1T-token 128K mid-training run at 128K context, and all configurations converge to the same short-context quality. Adaptation is fast: most of the 256K NLL improvement lands within the first 1–10% of extension steps, which the authors read as evidence that long-context ability is mostly *present* from pre-training and the extension phase merely re-calibrates positional/attention behavior for out-of-distribution positions. The retrieval and generative-QA evals show the extended checkpoint holding near-flat NLL and near-perfect answer accuracy across the full 256K window, while the un-extended model degrades sharply past its training length — especially for evidence near the *end* of the context, the out-of-distribution region. The final model uses 64K mid-training + 140B tokens of 256K extension (a conservative choice over what the ablations strictly required).

## Part II — The RL Climb (teaching the base model to think)

Pre- and mid-training give broad competence but do not specify *how* to behave, solve long-horizon tasks, or spend inference compute. The RL climb does that — and it starts from a checkpoint with **no prior exposure to reasoning traces**, which makes long-term stability the central challenge. The overview figure: three specialist teachers are trained independently, distilled into one consolidated model with SFT, and finished with a light final RL pass.

![Figure 12 from The Microsoft AI Team (2026): the RL climbs. From the mid-trained model, three specialist teachers (SWE/agentic, STEM, helpfulness and safety) are trained by RL, distilled into a consolidated model by SFT, then finished with a final RL climb to produce MAI-Thinking-1.](/imgs/blogs/mai-thinking-1-fig3.webp)

### The GRPO objective, and MAI's two changes

**The problem.** Policy-gradient RL on language needs an advantage estimate. PPO uses a learned value network — expensive and one more thing to tune. [GRPO](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) replaces the critic with a *group baseline*: sample several answers to the same prompt and normalize each answer's reward against its group.

**The mechanism.** For a prompt $q$, sample a group of $G$ responses $y_1, \dots, y_G$ from the rollout policy $\pi_{\text{old}}$. Each gets a scalar reward $R_i = R(q, y_i)$ — from a code executor, a formal verifier, a trained reward model, or a prompted AI judge, depending on domain. The group's mean and standard deviation define a per-response advantage; every token in $y_i$ shares it.

**The math.** The advantage is a within-group z-score:

$$
A_i = \frac{R_i - \operatorname{mean}(R_{1:G})}{\operatorname{std}(R_{1:G})}.
$$

The per-token importance ratio between the current policy $\pi_\theta$ and the rollout policy $\pi_{\text{old}}$ is

$$
r_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t}\mid q,\, y_{i,<t})}{\pi_{\text{old}}(y_{i,t}\mid q,\, y_{i,<t})},
$$

and the objective is the clipped, token-level surrogate (normalization is over *all* tokens in the global batch so every token counts equally regardless of response length):

$$
\mathcal{J}(\theta) = \mathbb{E}\Bigg[\frac{1}{\sum_i |y_i|}\sum_{i=1}^{G}\sum_{t=1}^{|y_i|} \min\!\Big(r_{i,t}A_i,\ \operatorname{clip}(r_{i,t},\, 1-\epsilon,\, 1+\epsilon)\,A_i\Big)\Bigg].
$$

Every symbol: $|y_i|$ is response $i$'s token length; $\epsilon$ is the clip width. The $\min$ with the clipped branch is PPO's trust region — it refuses to reward the policy for moving a token's probability too far from where the rollout policy had it.

**MAI's two changes.** The report's stability contribution is two surgical edits to this objective. The figure below is the mental model for both: the importance ratio $r$ lives on a number line, and MAI clips it in three zones.

![A number line for the GRPO importance ratio r, annotated with a base clip interval [0.4, 2.5] at k=0, an entropy-controlled widening of the upper bound up to 5.0, and a hard outer clip at r_max = 50, with the adaptive-entropy controller (Eq. 8) and the two-clip rationale explained below.](/imgs/blogs/mai-thinking-1-3.webp)

### Adaptive entropy control

**The problem.** The clip width controls exploration. Too wide and entropy explodes (the policy thrashes); too narrow and entropy collapses (the policy becomes deterministic and stops learning). A single fixed $\epsilon$ cannot be right for the whole climb, and tuning it by hand across thousands of steps is hopeless.

**The intuition.** A thermostat. Set a target entropy $H^\star$; if the policy is getting too deterministic (entropy below target), *widen* the upper clip bound so the policy is allowed to raise the probability of alternative tokens more aggressively; if it is too random, *tighten*. Adjust continuously, automatically.

**The mechanism.** Use asymmetric clip bounds parameterized by a base $\epsilon$ and an *entropy-dependent relaxation* $k$ of the upper bound:

$$
r_{i,t}^{\text{tr}}(\theta) = \operatorname{clip}\!\Big(r_{i,t}(\theta),\ 1-\epsilon,\ (1-\epsilon)^{-1} + k\Big).
$$

At $k=0$ the bounds $1-\epsilon$ and $(1-\epsilon)^{-1}$ are multiplicative inverses — symmetric in log-ratio space. An integral controller then nudges $k$ after each step based on the gap between an importance-weighted entropy estimate $\hat H(\pi_\theta)$ and target $H^\star$:

$$
\hat H(\pi_\theta) = \frac{1}{|\mathcal{T}|}\sum_{(i,t)\in\mathcal{T}} -\log\pi_\theta(y_{i,t}\mid q, y_{i,<t})\cdot r_{i,t}(\theta),
\qquad
k \leftarrow \operatorname{clip}\!\Big(k + \delta\cdot\operatorname{sign}\big(H^\star - \hat H(\pi_\theta)\big),\ 0,\ k_{\max}\Big).
$$

Here $\mathcal{T}$ is the set of (response, token) pairs in the batch, $\delta$ is the step size, and $k_{\max}$ caps the widening. When $\hat H < H^\star$ (too deterministic), $\operatorname{sign}(\cdot) = +1$ so $k$ rises and the upper bound widens; when $\hat H > H^\star$, $k$ falls.

**Worked micro-example.** With $\epsilon = 0.6$: the lower bound is ${1 - 0.6 = 0.4}$ and the symmetric upper bound is ${1/0.4 = 2.5}$. With $k_{\max} = 2.5$, the upper bound can widen to ${2.5 + 2.5 = 5.0}$. If a step measures $\hat H = 0.22$ against $H^\star = 0.30$ with $\delta = 0.25$, then $k \leftarrow k + 0.25$ (widen); a later step with $\hat H = 0.35$ pulls $k$ back down by ${0.25}$.

```python
def entropy_controller(k, H_hat, H_star=0.30, delta=0.25, k_max=2.5):
    # integral controller: widen the upper clip when too deterministic
    step = delta * (1 if H_star > H_hat else -1)
    return max(0.0, min(k + step, k_max))
```

**Why it works.** It is an automatic entropy regularizer with *no explicit entropy-bonus term in the loss* — the authors report an explicit bonus underperformed. The figure below shows it holding observed entropy near $H^\star = 0.30$ over 800 steps by driving $k$ up and down.

![Figure 13 from The Microsoft AI Team (2026): adaptive entropy control. Top, observed policy entropy tracking a target of 0.3 over ~800 RL steps; bottom, the controller's value of k rising and falling to keep entropy on target.](/imgs/blogs/mai-thinking-1-fig7.webp)

### The outer ratio clip

**The problem.** GRPO/PPO deliberately leave two cases *unclipped*: when the advantage is negative and the new policy already assigns higher probability, and when the advantage is positive and the new policy assigns lower probability. The original rationale was to let the policy correct itself freely in the right direction. In practice, MAI found these unclipped branches occasionally produce **catastrophic gradient-norm spikes** — a single token with a wildly off-policy ratio can blow up the update.

**The mechanism and math.** Add a hard *outer* clip applied to *all* branches, sitting outside the trust-region clip:

$$
r_{i,t}^{\text{out}}(\theta) = \operatorname{clip}\big(r_{i,t}(\theta),\ r_{\min},\ r_{\max}\big),
$$

with $r_{\max}$ large (they use 50) and $r_{\min}$ left unconstrained ($=0$). This is the two-level strategy the number-line figure shows: the inner clip is the normal trust region for well-behaved ratios; the outer clip is a guillotine for the pathological ones (a ratio of 50 means the new policy assigns a token 50× the old probability — almost certainly a numerical artifact, not a real gradient signal). It is close in spirit to dual-clip PPO. The reported effect: fewer gradient-norm spikes, more stable climbing.

### Reward design

Across all three specialist climbs the reward shares one decomposition:

$$
R(q, y_i) = R_{\text{task}}(q, y_i) + w_{\text{lang}}\cdot R_{\text{lang}}(y_i) - w_{\text{len}}\cdot R_{\text{len}}(y_i).
$$

$R_{\text{task}}$ is the domain-specific signal (code tests, verifier, judge). The other two are stabilizers. **Language consistency** ($w_{\text{lang}} = 0.5$) penalizes non-English tokens in the chain-of-thought:

$$
R_{\text{lang}}(y_i) = \max\!\big(1 - \alpha\cdot n_{\text{non-english}}(y_i),\ 0\big),\quad \alpha = 0.005,
$$

because as context lengths grow, models start emitting mixed-language CoTs, which correlate with train/inference log-prob divergence and destabilize the climb. The **length penalty** ($w_{\text{len}} = 0.25$) is the clever one — it is scaled by *problem difficulty*:

$$
R_{\text{len}}(y_i) = \rho_q\cdot\frac{|y_i|}{\ell_{\max}},
$$

where $\rho_q$ is the empirical pass rate of problem $q$ and $\ell_{\max}$ is the max rollout length. Easy problems (high $\rho_q$) get a strong length penalty — be concise, cut the hedging; hard problems (low $\rho_q$) get a weak one — explore longer. Because the penalty is normalized by $\ell_{\max}$, it automatically weakens as the length curriculum grows the budget; they zero it entirely at the 128K stage.

### Sampling: which problems, which tokens

Two sampling ideas do a lot of quiet work. **Problem sampling** uses an early-exit filter: sample $G_{\text{early}} = 16$ responses first, and only if the early pass rate lands in $[0.05, 0.80]$ do they sample the full $G = 128$; a second filter on the full group ($[0.1, 0.8]$) discards low-variance groups (near-all-correct or near-all-wrong give no relative learning signal). **Rollout sampling** uses top-$p = 0.97$ nucleus sampling — and here is a subtle, important detail: back-propagating through the logits of tokens *outside* the sampled nucleus causes catastrophic off-policy mismatch that diverges within a few steps. The fix is **top-$p$ mask replay**: reuse the exact nucleus mask from sampling to set excluded logits to $-\infty$ before the training softmax. A length-extension **curriculum** caps rollouts at 8K early and doubles up to 128K — long reasoning is rarely needed while the model is weak, and this slashes inference cost in the low-performance regime.

### Self-distillation: how to climb for thousands of steps

**The problem.** RL from a fresh base over thousands of steps *will* hit trouble: numerical mismatches accumulate and diverge the run, and you periodically want to swap in a better base checkpoint without throwing away the reasoning behavior the climb has discovered. Resuming from a pre-collapse checkpoint often does not work — the instability is baked into the weights many steps before the visible collapse.

**The mechanism.** *Self-distillation*: collect rollouts generated during RL, run plain SFT on a mid-trained checkpoint using those rollouts, and resume the climb from the SFT'd model. It is a save-point that carries forward progress across collapses, infrastructure changes, and base-model upgrades. Their ablated best practices are worth quoting because they are counterintuitive:

- **~1M reasoning traces is enough** to match teacher performance; more over-constrains the policy and leaves no room to explore when RL resumes.
- **Including traces that reached wrong answers works about as well** as successful-only (though they ultimately used successful-only, since RL produces those in abundance).
- **Trace freshness matters**: use traces from *later, strong* checkpoints — very-early traces degrade and need many steps to recover — but from a *range* of strong checkpoints, not only the final one, for diversity.
- **Prompt diversity beats traces-per-prompt** at fixed budget, and plain random sampling beat several fancy selection heuristics.
- A high dropout (0.15) and a large MoE load-balancing coefficient ($10^{-2}$, versus $10^{-5}$ during RL) during self-distillation raise entropy and rebalance experts, preventing collapse when RL resumes.

The STEM-climb figure below makes this concrete: the star markers are self-distillation restarts, the sudden drops are collapses, and the colored segments are different base-model versions — all stitched into one continuous log-linear climb from ~50% to ~95% on AIME 2025, with the length curriculum (8K→128K) annotated at the bottom.

![Figure 15 from The Microsoft AI Team (2026): AIME 2025 (left) and hard LiveCodeBench v6 (right) pass@1 across the STEM climb. Star markers indicate self-distillation restarts, sudden drops are collapses, colored segments are different base-model versions, and the 8K-to-128K length curriculum is marked along the bottom.](/imgs/blogs/mai-thinking-1-fig8.webp)

### The STEM climb and its data pipeline

The STEM climb is the longest run, on >5M verifiable samples (the hardest tranche >550k). Every instance is either a $(q, a)$ pair (query + ground-truth answer, checked by SymPy or an AI judge) or a $(q, \{t_1,\dots,t_n\})$ pair (query + test cases, for competitive coding — 160k problems, 17 languages). The data pipeline that manufactures these pairs from textbooks and PDFs is a four-phase assembly line, and its design encodes hard-won lessons about noisy sources.

![Figure 17 from The Microsoft AI Team (2026): the four-phase pipeline that extracts question-answer pairs from textbooks and academic PDFs for the STEM Mix dataset — hierarchical parsing, Q-A pairing, curation, and scoring.](/imgs/blogs/mai-thinking-1-fig9.webp)

Phase 1 (**hierarchical parsing**) OCRs and cleans documents and has an LLM mark question/answer spans. Phase 2 (**Q-A pairing**) matches questions to answers when they live apart (exercises at chapter end, answer keys in appendices). Phase 3 (**curation**) classifies verifiability, question type, taxonomy, PII, and *answer leakage* (where the answer is trivially in the question), and crucially **converts** multiple-choice and proof questions to open-ended form — MCQs can be guessed (unreliable reward) and proofs are hard to verify — running the conversion three times and keeping only consensus. Phase 4 (**scoring**) solves each problem $k$ times with four model tiers (using their AIME-2025 skill as a proxy) to bin difficulty, and runs a **blind-grading** guard against faulty ground truth: for a hard item, show a judge the strong model's consensus answer and the stated ground truth in random order — if the judge prefers the consensus answer, the ground truth is suspect and the item is dropped. Stages sensitive to hallucination are run multiple times with consensus voting throughout.

### The agentic climb: environments, tools, and reward hacking

The agentic climb teaches multi-step interaction: decompose a request, call tools, observe, adapt. It reuses the single-step objective but extends a rollout to a full trajectory of policy steps and environment observations, orchestrated by a ReAct-style loop, with credit assigned uniformly across all tokens of all steps. Each environment is a task spec + a **Sandbox Execution Environment** (SEE) session (fresh, network-isolated container per task, destroyed after) + verifiable/judged rewards. Interestingly, they jointly climb agentic *and* STEM tasks: adding STEM *stabilizes* the agentic climb and transfers positively to it, while agentic tasks transfer neither way to single-pass STEM.

For software engineering, the environments are harvested from real GitHub pull requests — and the yield is brutal. The funnel below is the single best illustration in the paper of how much curation real-world RL data needs.

![A funnel showing how 102M public GitHub pull requests are filtered to 265k verified SWE-RL environments: 4.87M with linked issues, 2.08M that build and run (42.8%), 745k that pass reference grading (15.3%), and 265k (5.5%) verified across 94k repositories, with dropped-at-each-stage reasons.](/imgs/blogs/mai-thinking-1-5.webp)

Starting from 102M PRs, they keep only merged PRs that touch fewer than 15 files and include *both* code and test changes (so the test diff can serve as hidden tests) and have a linked issue — 4.87M. An LLM agent then writes Dockerfiles to build each into an executable image (2.08M succeed, 42.8%). Reference grading runs the test suite twice — once with only the test diff (pre-fix), once with test+code (post-fix) — and keeps problems with **fail-to-pass** (F2P) tests as the resolution signal and **pass-to-pass** (P2P) as the regression signal (745k, 15.3%). Final re-validation inside the *training* sandbox — empty patch must fail, golden patch must pass, across trials, filtering non-deterministic tests — leaves **265k** environments across 94k repos (5.5%). The two available tools are Bash and a string-replace editor.

Because the environments come from public repos, **reward hacking** is a live threat, and the mitigations are specific: disable/limit internet so the agent cannot search for the actual PR; scrub all git commits/branches after the base commit to "time-travel" the repo so the solution is not hiding in git history; reset any test files the agent modified before grading and hide test changes during inference (following SWE-Bench's protocol), backed by LLM monitors that flag subtler tampering like monkey-patching the test framework. General tool-use environments (>50 tools each, mocked stateful backends, 150+ synthetic environments, 130k tasks) add cross-environment graders that reward efficient tool use — parallel calls, no duplicates, valid arguments — and deliberately include tasks that *do not* require tools, to curb over-eager tool-calling.

### Helpfulness and safety: subjective rewards

This climb optimizes what is *not* machine-verifiable — human preference, instruction following, steerability, honesty, safety, style — and combines three reward types: a **reward model** (post-trained MAI-Base-1 fine-tuned on human preference data), **AI judges** (fast, rubric-guided, adjustable without retraining), and **verifiable rewards** (rule-checked constraints like "answer in one paragraph," which they find less hackable and more stabilizing than the others).

The reward model has a neat inference trick. It ingests $k$ candidate responses in one context (so scores are calibrated against each other) but, because of autoregressive position bias, later candidates get noisier scores. So they run the model $k$ times on cyclic permutations of the responses, and for each call only decode the *first* token, reading the probability that the leading candidate is rated highest (score 5). That per-candidate probability is the reward — more stable than the alternatives they tried.

The genuinely reusable idea is **how the rewards are combined**. Naive weighted summation is broken: rewards occupy different scales, so the largest-magnitude one dominates regardless of importance — and worse, a beautifully-written but unsafe answer can still net a positive score. The figure below contrasts the naive approach with the two MAI actually uses.

![A 3x4 matrix comparing three reward-aggregation strategies — additive (naive), lexicographic, and gated — across combination rule, cross-scale safety, whether safety can be traded away, and what MAI uses each for. Additive fails on every axis; lexicographic and gated make priority explicit.](/imgs/blogs/mai-thinking-1-4.webp)

**Lexicographic shaping**: a lower-priority reward influences the gradient only when all rollouts in a group *tie* on the higher-priority reward — a strict priority order that is invariant to absolute reward scales because it works on within-group comparisons. Used for instruction following (the IF reward is primary). **Gated application**: a higher-priority reward must clear a minimum before lower-priority rewards apply at all. Safety is the canonical case — an unsafe response gets the minimum reward and is *never graded on quality*. The paper backs the need for gating with an audit statistic: policy-compliance had only moderate correlation with a scalar safety Likert (Pearson 0.293), and **87.8% of policy-non-compliant responses still received reward-model scores ≥ 3** — i.e., a scalar mixture would happily reward unsafe-but-fluent answers. Honesty is handled by grading each response into one of five buckets (confident-correct → confident-incorrect) and weighting them so confident hallucinations get the steepest penalty and abstention is neutral, balancing factual precision against the risk of over-hedging.

### Consolidating three specialists into one model

The three teachers are merged in two stages. **Consolidation SFT** reuses the self-distillation pipeline on each teacher's rollouts (with teacher-specific filtering), balanced by *sample* weight — 56% STEM/coding, 11% agentic, 33% helpfulness/safety — even though by *token* weight STEM/coding dominates (89%) because its traces are long. This SFT runs 4 epochs. Then **consolidation RL**, a light pass based on the helpfulness-and-safety recipe, improves safety, over-refusals, and style while retaining a small fraction of STEM/coding data to keep reasoning from slowly decaying.

### Rocket: the asynchronous RL infrastructure

**The problem.** At MAI's scale, most GPUs are doing *inference* (rollout generation), not learning — in the largest job, 4096 of 4864 GB300 chips serve inference and only 768 learn, a ~5:1 ratio. Synchronous RL would leave the learner idle while rollouts generate, so the system must be asynchronous — which means the policy generating rollouts is *stale* relative to the learner, and staleness must be bounded or the importance-sampling correction breaks.

**The mechanism.** **Rocket** organizes this around a controller, problem/rollout workers, and an SGLang-based inference pool with the learner running YOLO. The figure below traces one iteration's data flow.

![A data-flow graph of Rocket: an async controller feeds problem workers (which early-exit-filter tasks, dropping out-of-band ones), rollout workers run a ReAct loop against an SGLang inference pool (~5:1 vs the learner), completed rollouts with advantages and top-p masks land in a store, the YOLO learner trains 5 steps per weight update, and weight transfer pushes fresh weights that are at most 8 updates stale.](/imgs/blogs/mai-thinking-1-6.webp)

The controller loads tasks and sends completed, filtered rollouts to the learner. Problem workers run the early-exit protocol (16 early rollouts, then 128 full ones, aborting tasks whose pass rate is out of band) and compute normalized advantages. Rollout workers run the ReAct loop against inference and grade. The learner does **5 gradient steps between inference-model updates** and discards any rollout whose generating policy is **more than 8 updates stale** (40 gradient steps) — the explicit staleness/throughput trade. The most important stability lever is the **numerics gap**: YOLO (learner) and SGLang (inference) use different kernels, and even tiny per-token log-prob discrepancies compound across a long rollout and destabilize the off-policy correction — so both sides run **bf16** in RL (smaller gap than lower-precision alternatives), plus MoE routing replay and the top-$p$ mask replay. Weight transfer is compiled once into a plan that intersects the learner's and inference's sharding layouts and emits per-sub-shard byte moves with dtype/layout transforms, run in parallel across DP-group slices to bound the failure blast radius. Defense-in-depth watchdogs (replica self-restart, router circuit-breaker, job liveness, a step-progress monitor for the "alive but not advancing" case) keep thousands of inference replicas from failing the whole job.

## Experiments & results

MAI-Thinking-1 is evaluated as the average of 4 runs at temperature 1.0, top-$p = 0.97$, with 256K max output. The headline table pits it against the frontier:

| Benchmark | MAI-Thinking-1 | Sonnet 4.6 | Opus 4.6 | GPT 5.4 | Kimi K2.6 | DeepSeek V3.2 | DeepSeek V4 | GLM-5.1 |
|---|---|---|---|---|---|---|---|---|
| AIME 2025 | **97.0** | 95.6 | 99.8 | — | — | 93.1 | — | — |
| AIME 2026 | 94.5 | — | — | — | 96.4 | — | — | 95.3 |
| HMMT Feb 2026 | 84.9 | — | — | — | 92.7 | — | 95.2 | 82.6 |
| GPQA Diamond | 84.2 | 89.9 | 91.3 | 92.8 | 90.5 | 82.4 | 90.1 | 86.2 |
| LiveCodeBench v6 | 87.7 | — | — | — | 89.6 | 83.3 | 93.5 | — |
| Terminal-Bench 2.0 | 46.0 | 59.1 | 65.4 | 75.1 | 66.7 | 46.4 | 67.9 | 69.0 |
| SWE-bench Verified | 73.5 | 79.6 | 80.8 | — | 80.2 | 73.1 | 80.6 | — |
| SWE-Bench Pro | 52.8 | — | 53.4 | 57.7 | 58.6 | — | 55.4 | 58.4 |

The honest read the authors give: MAI-Thinking-1 is "in the competitive range" — it does not lead, but it is consistently strong. It exceeds Sonnet 4.6 on AIME 2025 and nearly matches Opus 4.6 on SWE-Bench Pro. The Terminal-Bench 46.0% is the clear soft spot, and the authors flag *why*: all their SWE training used only Bash and string-replace tools with no terminal-specific environments, so that number reflects generalization, not training. On a broad second table (knowledge, instruction following, long context, safety, honesty, health, tool calling) they report parity with Sonnet 4.6 across most benchmarks (e.g. MMLU-Pro 85 vs 87, IFBench 69 vs 50, GraphWalks-128k 90 vs 96, BFCL v3 72 vs 76, AIR-Bench safety 88 vs 88).

Two results are more interesting than the leaderboard. First, the **base model** is genuinely strong for its active-parameter count. In bits-per-byte (a tokenizer-invariant NLL) on four held-out tasks, MAI-Base-1 (35B active) beats similarly-active open models across the board; only DeepSeek-V4-Pro (1.4× the active, 1.6× the total parameters) is clearly ahead.

![Figure 10 from The Microsoft AI Team (2026): bits-per-byte (lower is better) of base pre-trained models on four held-out tasks — code, QA, STEM, and math. MAI models (yellow bars) beat similarly-sized open models; only the larger DeepSeek-V4-Pro leads.](/imgs/blogs/mai-thinking-1-fig5.webp)

Second, **human side-by-sides** (1,276 English tasks, native-speaker raters). Raters preferred MAI-Thinking-1 to Sonnet 4.6 (overall +0.07 on a $[-1.5, 1.5]$ scale; 49% win, 6% tie, 45% loss) but preferred Opus 4.6 to it (−0.07; 43/5/52). The dimension breakdown is telling: MAI won on *conciseness/relevance* (+0.11) and *style/tone* (+0.08), and was within noise on instruction following, factuality, and completeness — a model tuned to *read well*, which aligns with how much of the helpfulness-and-safety climb is about style. On safety, internal jailbreak evals put its attack-success-rate comparable to Sonnet 4.6 and Opus 4.6 across foundational, compositional, and adaptive attack buckets, and red-teaming over 15 engagements and 2,170 scenarios drove a ~22% aggregate reduction in attack success from pre-mitigation to release.

**What is load-bearing that might not transfer.** The results rest on things a smaller lab cannot cheaply replicate: 30T tokens of *privately curated* clean data (the whole "no distillation, in-house everything" stance depends on owning that pipeline), 8K GB200s with a bitwise-deterministic framework, a ~5:1 inference-heavy RL cluster, and hundreds of thousands of *executable, re-validated* SWE environments. The recipe's specific numbers — dropout 0.15, LR ratio 0.1×, $\epsilon = 0.6$ with $k_{\max} = 2.5$, top-$p = 0.97$, 8-update staleness — are tuned to *this* stack; the report is candid that mixture and long-context decisions were made conservatively where scaling behavior was unclear.

## Critique

**What is genuinely strong.** The methodological spine — a ladder, one metric (EG), and the discipline to demand that gains persist *up* the ladder — is the most transferable idea here, and the rank-non-invariance result (Figure 6) is a concrete, honest demonstration of why cheap small-scale experiments mislead. The two GRPO edits are small, well-motivated, and clearly ablated. The self-distillation-as-save-point framing is a genuinely useful answer to "how do you run RL for thousands of steps without the run dying." And the safety-gating argument backed by the "87.8% of unsafe responses scored ≥ 3" audit is the rare safety claim with a falsifiable number attached.

**What is weak, unfalsifiable, or withheld.** The report is transparent about *methods* but opaque about the *artifacts* that would let anyone check them. It does not release weights, does not disclose the data providers or the exact final mixture beyond a coarse table, and — most frustrating for a paper whose central tool is a fitted scaling law — never reports the fitted $A$, $\alpha$, or $E$ for any ladder, nor the ladder's raw points. So "Efficiency Gain = 1.03 FLOPs but 0.82 time" is a conclusion we must take on faith. The baselines in Tables 11–12 are mostly self-reported model-card numbers evaluated under the labs' own harnesses (MAI even re-ran Sonnet 4.6 themselves for Table 12) — the comparison is apples-to-oranges in exactly the way the paper elsewhere warns benchmark comparisons are. The dates are also conspicuously future-stamped (DeepSeek V4, Kimi K2.6, GB300, "AIME 2026") in a way that makes external cross-checking impossible at time of reading.

**What ablation is missing.** The two GRPO changes are presented as jointly necessary, but there is no clean ablation isolating adaptive-entropy-control from the outer-clip — we cannot tell which one buys the stability. There is no ablation of the *consolidation* step showing how much capability is lost merging three specialists versus training one generalist directly. And the "no distillation" principle, while philosophically central, is never quantified against a distillation baseline — we are told inherited capability is less robust, but shown no experiment measuring that robustness gap.

**What would change my mind.** If MAI released the ladder's raw loss-vs-compute points and the fitted scaling-law constants, and an ablation isolating adaptive-entropy-control from the outer-clip on a public benchmark, I would upgrade this from "a persuasively-argued methodology report" to "a reproducible recipe." Conversely, if an independent eval of the released model (were one released) failed to reproduce the AIME/LCB numbers under a standard harness, the whole "clean data + honest climbing beats distillation" thesis would need re-examination.

## What I'd build with this

These are my extrapolations, not the paper's claims:

1. **EG-as-a-service for small labs.** The scaling-ladder + Efficiency-Gain loop is the most portable idea and needs no frontier cluster — a disciplined ladder at 100M–1B with EG-across-scale gating would catch a lot of "looked good at 300M, washed out at 3B" mistakes that smaller teams currently make blind. I would build it as a harness that fits the baseline law, computes EG for every candidate, and *refuses* to accept a change whose EG trend is flat or declining up the ladder.
2. **Difficulty-scaled length penalties everywhere.** The $\rho_q$-scaled length term is a clean, general trick for any verifiable-reward RL — it is a one-line change that gives you concise-on-easy, exploratory-on-hard for free. I would port it to math and code RL setups that currently use a flat length penalty.
3. **Self-distillation checkpoints as standard RL hygiene.** Treating periodic SFT-on-own-rollouts as a save-point (with the "diverse strong checkpoints, ~1M traces, high dropout" recipe) is the kind of operational practice that belongs in any long-horizon RL pipeline, not just MAI's.
4. **The numerics-gap discipline for open RL stacks.** The bf16-on-both-sides + mask-replay + routing-replay recipe for closing the learner/inference gap is directly actionable for anyone running asynchronous RL on vLLM/SGLang, where this exact mismatch silently diverges runs.
5. **A public rank-non-invariance benchmark.** Someone should build a small, open reproduction of the Figure-6 experiment — two mixtures whose small-scale ranking reverses at larger scale — as a test case for data-mixture methods that *claim* rank-invariance.

## References

- **Paper.** The Microsoft AI Team. *MAI-Thinking-1: Building a Hill-Climbing Machine.* Technical report, 2026. [microsoft.ai/pdf/mai-thinking-1.pdf](https://microsoft.ai/pdf/mai-thinking-1.pdf)
- **Foundational prior work.** GRPO and RL-from-base reasoning: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning).
- **Sibling posts on this blog.** [MoE scaling laws](/blog/machine-learning/scaling-laws/moe-scaling-laws) · [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) · [Expert parallelism for MoE](/blog/machine-learning/distributed-training/expert-parallelism-moe) · [The pre-training, mid-training, and RL interplay](/blog/paper-reading/large-language-model/pre-training-mid-training-and-rl-interplay).
