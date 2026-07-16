---
title: "Mellum 2: Designing a Code LLM Backwards From a Latency Budget"
date: "2026-07-16"
description: "A detailed, intuition-first walkthrough of JetBrains' Mellum 2 technical report — a 12B Mixture-of-Experts code model with 2.5B active parameters, where every architecture choice was ablated against a fixed single-H100 inference budget."
tags: ["paper-reading", "mixture-of-experts", "code-llm", "mellum", "grouped-query-attention", "speculative-decoding", "multi-token-prediction", "yarn", "rlvr", "grpo", "muon", "inference-efficiency"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 40
paper:
  title: "Mellum 2 Technical Report"
  authors: "Marko Kojic et al. (JetBrains)"
  venue: "arXiv 2026"
  url: "https://arxiv.org/abs/2605.31268"
---

> [!tldr]
> - **What it is.** Mellum 2 is JetBrains' open-weight (Apache 2.0) code model: a 12B-parameter Mixture-of-Experts (MoE) transformer that activates only **2.5B parameters per token**. It succeeds the 4B *dense* Mellum that shipped inside JetBrains IDEs.
> - **The one idea.** Every architecture decision — MoE vs dense, expert sparsity, KV-head count, the sliding-window pattern, the multi-token-prediction head — was chosen by ablation against a *fixed inference budget*: match the single-H100 speed of Qwen2.5-7B. Quality was optimized **subject to** that latency constraint, not the other way around.
> - **Why it matters.** The result runs at the per-token compute of a 2.5B dense model (192 tok/s single-request, +21% throughput vs Qwen2.5-7B under load) while carrying the knowledge envelope of a 12B model — competitive with 4B–14B dense baselines on code and reasoning.
> - **Most surprising bit.** With no self-identity data, the model spontaneously claimed to be *"an AI assistant developed by Google"* — even though no Google models were used to generate its training data. They had to oversample identity dialogues 3× to fix it.
> - **Where it's weak.** Broad world knowledge. On GPQA Diamond (graduate science), the 9B Qwen3.5 scores 79.8 against Mellum 2's 40.9 — a deliberate trade of encyclopedic breadth for code depth. The RL stage also relaxes some refusals (an "alignment tax").

The diagram below is the mental model for the whole report: one decoder layer, repeated 28 times, with a small extra head bolted on top. Everything else in this post is a justification for one of the boxes in it.

![A layered stack of the Mellum 2 decoder: input embedding, then a repeated block of RMSNorm plus 4-KV-head GQA attention, a 3-of-4 sliding-window pattern, and an 8-of-64 MoE feed-forward, with a single MTP head at the bottom serving as a speculative-decoding draft](/imgs/blogs/mellum2-code-moe-1.webp)

## The problem: quality per dollar, not quality per parameter

Coding assistants have quietly become the most demanding LLM workload in production. What started as inline autocomplete now has to write whole functions from a natural-language spec, edit and debug existing files, plan multi-step engineering tasks, call tools, navigate a repository as an agent, and hold a conversation about code — often several of those inside a single editing session. And it has to do all of that at a serving cost low enough that a company can put it in front of every developer, on every keystroke, without going bankrupt.

That last clause is the whole game. The open-weights landscape splits into two regimes on the quality-versus-cost curve. **Dense models in the 4–14B range** are cheap to serve but plateau on the harder coding and reasoning workloads. **Very large MoE models** (DeepSeek-V3, the Kimi-K2 family) reach frontier quality but at deployment costs that make per-keystroke serving absurd. Mellum 2 aims for the narrow lane in between — the one occupied by Qwen3-Coder-30B-A3B and Ling-Coder-Lite: enough total parameters to absorb the long tail of programming-language and reasoning knowledge, but enough sparsity that per-token compute stays in the 2–3B-dense range.

The predecessor, [the original 4B dense Mellum](https://arxiv.org/abs/2510.05788), was a *completion* model — it filled a single span inside your editor. Mellum 2 is a full assistant: it generates, edits, calls tools, plans agentic workflows, and (in its Thinking variant) emits explicit reasoning traces. The paper is the story of how you get there without blowing the latency budget that made the original deployable.

The framing to hold onto: this is not "we trained the best model we could and then optimized inference." It is **"we fixed the inference budget first — match Qwen2.5-7B on one H100 — and then searched the architecture space for the highest-quality model that fits."** That inverts the usual order, and it explains a lot of choices that would otherwise look conservative.

## What Mellum 2 actually is

Tightened into a list, the report's contributions are:

1. **An efficiency-aware architecture.** A systematic ablation of dense vs MoE backbones, GQA configurations, Multi-head Latent Attention, sliding-window patterns, and expert-sparsity ratios — settling on a 12B-total / 2.5B-active MoE that matches or beats a 7B dense baseline's throughput.
2. **A three-phase pre-training curriculum** over ~10.6T tokens, shifting the data mix from web-heavy to code-and-math-heavy (code ratio 23% → 42% → 59%) as training proceeds.
3. **A Muon + FP8 training recipe at production scale**, with a Warmup-Hold-Decay learning-rate schedule that decays linearly to zero, plus a candid diary of training-stability incidents.
4. **Long-context extension to 128K** via a *layer-selective* YaRN that only re-maps the full-attention layers.
5. **Two post-trained variants from one base** — an Instruct model that answers directly and a Thinking model that reasons first — each refined by reinforcement learning with verifiable rewards (RLVR).
6. **A full open release** — base, instruct, and thinking checkpoints under Apache 2.0, plus the report itself.

The rest of this post walks the architecture, then the pre-training recipe, then the long-context stage, then post-training, then the results — climbing the intuition-to-math ladder on each load-bearing technique.

## The architecture, chosen backwards from a latency budget

Mellum 2 is a decoder-only transformer that closely follows the Qwen3-MoE recipe: RMSNorm, SiLU-gated MLPs, Rotary Position Embeddings, Grouped-Query Attention with QK-Norm, and an MoE feed-forward in every layer. On top of that backbone it adds two latency-oriented modifications — sliding-window attention on most layers, and a multi-token-prediction head. Let me take the four load-bearing choices one at a time.

### Sparsity: 8 of 64 experts

**The problem it solves.** A dense transformer spends the same FLOPs on every token, and quality scales with parameter count — but so does serving cost. If you want the knowledge of a 12B model at the serving cost of a 2.5B model, a dense network cannot give it to you: the parameters you add are the parameters you pay for on every token. The authors confirm this empirically — none of their dense Qwen3-based configurations (depth 24–40, hidden 2304–4096), nor a DeepSeek-style Multi-head Latent Attention variant, could beat Qwen2.5-7B within the latency budget.

**The intuition.** A Mixture-of-Experts layer is a *panel of specialists with a receptionist*. Instead of one big feed-forward network that every token passes through, you keep 64 smaller networks (the experts) and a tiny router (the receptionist) that reads each token and sends it to the 8 experts most likely to help. The token only pays for the 8 it visits, so the *compute* is that of a much smaller model, but the *knowledge* is spread across all 64 — the model can specialize an expert on Rust borrow-checker errors, another on SQL, another on numerical Python, and only wake the relevant ones per token.

**The mechanism, step by step.** For each token with hidden state $h \in \mathbb{R}^{d}$ (here $d = 2304$):

1. The **router** computes a score for every expert: a linear projection $g = h W_r$ with $W_r \in \mathbb{R}^{d \times E}$, giving $E = 64$ logits.
2. Keep the **top-8** experts by score, discard the other 56.
3. Softmax over just those 8 scores to get mixing weights.
4. Run the token through those 8 expert FFNs, and take the weighted sum.

**The math.** Let $g_i = (h W_r)_i$ be the router logit for expert $i$, and let $\mathcal{T} = \text{top-}k(g)$ be the indices of the $k = 8$ highest. The layer output is

$$
y = \sum_{i \in \mathcal{T}} \frac{\exp(g_i)}{\sum_{j \in \mathcal{T}} \exp(g_j)} \; \text{FFN}_i(h),
$$

where each $\text{FFN}_i$ is a SiLU-gated MLP with intermediate size 896. Because only $k$ of $E$ experts run, the per-token FFN cost is $k/E = 8/64 = \tfrac{1}{8}$ of the dense-equivalent — which is exactly how 12B total parameters collapse to ~2.5B *active*.

**A worked micro-example.** Shrink it to $E = 4$ experts, $k = 2$. Say a token produces router logits $g = [2.0,\ 0.5,\ 1.5,\ -1.0]$. Top-2 are experts 0 and 2 (logits 2.0 and 1.5). Softmax over just those two: $\tfrac{e^{2.0}}{e^{2.0}+e^{1.5}} \approx 0.62$ and ${0.38}$. So the output is $0.62\,\text{FFN}_0(h) + 0.38\,\text{FFN}_2(h)$ — experts 1 and 3 never run for this token, and cost nothing.

There's a catch the router creates for itself, and it's the reason MoE training has extra loss terms:

- **Load imbalance.** Left alone, the router falls into a rich-get-richer collapse: a few popular experts get all the tokens (and all the gradient), the rest starve. Mellum 2 adds a **global auxiliary load-balancing loss** (coefficient $10^{-3}$) that pushes the token distribution toward uniform. Its standard form is $\mathcal{L}_{\text{bal}} = \alpha_{\text{bal}} \cdot E \sum_{i=1}^{E} f_i\, P_i$, where $f_i$ is the fraction of tokens dispatched to expert $i$ and $P_i$ is the mean router probability for expert $i$; the product is minimized when both are flat at ${1/E}$.
- **Logit blow-up.** A **router z-loss** (also $10^{-3}$) penalizes the log-partition-function $\big(\log\sum_i e^{g_i}\big)^2$, keeping the router logits from drifting to extreme magnitudes that destabilize training. The router itself runs in FP32.

Two more MoE choices worth flagging, because both are "we resisted the fashionable option":

- **No shared expert.** DeepSeek-style MoE adds one always-on expert. At Mellum 2's scale it gave *no* measurable quality gain and consistently hurt inference (the extra always-on FFN is pure per-token cost), so they dropped it.
- **Dropless routing.** Rather than cap each expert's token capacity and drop overflow, they use dropless routing (MegaBlocks-style) — no tokens dropped, no information lost. It was ~15% slower per step early in training while the router was still lopsided, but as the load-balancing loss converged, throughput climbed to match capacity-limited routing. (For the serving-side view of all this, see [serving MoE models at scale](/blog/machine-learning/model-serving/serving-moe-models-at-scale) and the [MoE scaling laws](/blog/machine-learning/scaling-laws/moe-scaling-laws) that motivate the sparsity sweet-spot.)

**Why 8 and not 2.** Higher sparsity is faster — 2 active experts ran ~1.5× lower latency than 8. But consistent with scaling-law work on fine-grained MoE, *high sparsity hurts quality at small scale*: 2 active experts were substantially worse on the eval benchmarks. 8-of-64 was the quality-latency optimum, and it lets the model reach ~15B total parameters while still matching Qwen2.5-7B latency.

**When it fails.** MoE's whole premise breaks if the router can't learn a good assignment. The failure modes are all routing pathologies: expert collapse (fixed by the balancing loss), token dropping under capacity limits (avoided by going dropless), and — as we'll see in the RL section — *routing non-determinism*, where the same hidden state gets dispatched to different experts by two copies of the "same" model, which turns out to be the single nastiest problem in the whole RL pipeline.

### Grouped-query attention: 4 KV heads

**The problem it solves.** At serving time under concurrency, the bottleneck usually isn't the model's FLOPs — it's the **KV cache**: the stored keys and values for every past token, for every attention head, held in GPU memory so you don't recompute them each step. The bigger the KV cache per token, the fewer concurrent requests fit in memory and the more bandwidth each decode step burns. The paper is blunt about it: *the number of KV heads is the single most significant factor affecting inference throughput under high concurrency.* (See [why LLM serving is memory-bound](/blog/machine-learning/model-serving/why-llm-serving-is-different) for the underlying roofline.)

**The intuition.** In vanilla multi-head attention (MHA), every query head carries its *own* key and value heads — 32 queries, 32 keys, 32 values. Grouped-Query Attention (GQA) says: let several query heads *share* one key/value head. Think of it as 32 readers (queries) all consulting a smaller set of 4 shared reference books (KV heads) instead of each hauling around a personal copy. The readers still ask different questions; they just look up answers in shared references. Multi-Query Attention (MQA) is the extreme — all 32 queries share a *single* KV head.

**The mechanism and the math.** The KV cache stores, per token per layer, both keys and values across $n_{kv}$ KV heads of dimension $d_{head}$. So its size is proportional to

$$
\text{KV bytes / token / layer} \;\propto\; 2 \cdot n_{kv} \cdot d_{head}.
$$

For Mellum 2, $n_{kv} = 4$ and $d_{head} = 128$, giving $2 \cdot 4 \cdot 128 = 1024$ stored values per token per layer. Full MHA with $n_{kv} = 32$ would store $2 \cdot 32 \cdot 128 = 8192$ — **8× more**. MQA with $n_{kv} = 1$ stores ${256}$, a further 4× reduction but at a quality cost the authors found unacceptable at this scale. The matrix below is the whole decision.

![A four-row by three-column matrix comparing MHA, GQA-4, and MQA on query heads (32/32/32), KV heads (32/4/1), relative KV cache (8x heavier, 1x baseline, 0.25x lighter), and the verdict at Mellum 2's scale — MHA too heavy, GQA-4 chosen, MQA insufficient quality](/imgs/blogs/mellum2-code-moe-2.webp)

**A worked micro-example.** Concretely, at 4 KV heads a token's KV footprint across all 28 layers is $1024 \times 28 = 28{,}672$ values; at 32 KV heads it would be $229{,}376$. Multiply by two bytes (BF16) and by sequence length and batch size, and the 8× difference is the difference between fitting 8 concurrent long-context requests on one GPU and fitting one. The report notes the punchline directly: Qwen2.5-7B with 4 KV heads gets roughly the *same* throughput as the predecessor Mellum-4B with 8 KV heads, despite being nearly twice the size — the KV-head count dominates.

**Why 4, not 2 or 32.** 8 heads caused significant throughput degradation; 2 heads gave insufficient quality. 4 was the knee. The report also observes the effect is asymmetric across serving modes: in synchronous single-request decoding the KV cache is under-utilized, so KV-head count barely matters and *depth* dominates latency; in throughput mode under concurrency, the KV-cache bottleneck bites hard and *wider* models (bigger hidden dimension) are disproportionately penalized. Design choices that look free in a single-request benchmark can be ruinous under load — a recurring theme in this report.

QK-Norm (RMSNorm applied to the query and key projections) and RoPE with base $\theta = 500{,}000$ round out the attention block; the large RoPE base is what makes the later context extension to 128K feasible.

### Sliding window attention (3:1)

**The problem it solves.** Full attention is $O(n^2)$ in sequence length — every token attends to every prior token. That quadratic cost is fine at 512 tokens and painful at 8K, and it grows the KV cache you must read on every decode step. But most of what a token needs is *local*: the previous few hundred tokens carry the bulk of the signal.

**The intuition.** Sliding Window Attention (SWA) caps each token's view to a fixed window of the most recent $w$ tokens, like reading with a ruler that only exposes the last paragraph. You lose nothing local, and you make the per-layer cost linear in $n$ instead of quadratic. The trick Mellum 2 borrows from the Gemma family is to not do this *everywhere*: apply SWA to **3 of every 4 layers** (window 1,024) and leave the 4th layer full-attention. The full-attention layers act as periodic "long-range relays" — information that needs to travel across the whole context hops through them — while the majority of layers stay cheap.

**The math.** A full-attention layer costs $O(n^2)$; an SWA layer with window $w$ costs $O(n \cdot w)$. Over a 4-layer block, the average per-layer attention work is

$$
\underbrace{\tfrac{3}{4} \cdot O(n w)}_{\text{SWA layers}} \;+\; \underbrace{\tfrac{1}{4} \cdot O(n^2)}_{\text{full layer}},
$$

so as $n$ grows past $w$, three-quarters of the layers stop scaling quadratically. Consistent with Gemma's findings, a window of 1,024 beat a window of 512 on quality benchmarks — 1,024 is wide enough to hold the local structure a code token depends on.

**The payoff, measured.** The authors' own figure makes the case: MoE models with SWA hold latency comparable to Qwen2.5-7B *even at double the context length*. At 8,192 input tokens Mellum 2 is 1.35× faster than Qwen2.5-7B, and the gap is a durable advantage in exactly the long-context workflows (repository-scale completion) that a code assistant lives in.

![Figure 3 from Kojic et al. (2026): mean latency versus input length for Mellum 2 with sliding-window attention against Qwen2.5-7B; Mellum 2 stays 1.35–1.49x faster across 2,304 to 8,192 input tokens](/imgs/blogs/mellum2-code-moe-fig1.webp)

**When it fails.** SWA assumes the important signal is local. For tasks that genuinely need dense long-range interaction on *every* layer — some retrieval-heavy or global-reasoning patterns — the 3:1 ratio can under-serve; the single full-attention layer per block is the only place cross-context information mixes freely. Mellum 2 bets that for code, local structure plus periodic global relays is enough, and the long-context results (below) mostly bear that out.

### Multi-token prediction

**The problem it solves.** Standard language modeling trains the model to predict only the *next* token. That's a myopic objective: the model never has to plan two tokens ahead, and at inference it can only emit one token per forward pass, which caps decode throughput.

**The intuition.** A Multi-Token Prediction (MTP) head is a small extra module — here a single transformer layer — that reads the main model's hidden states and predicts *the token after next* ($t{+}2$). During training this is a mild auxiliary objective that nudges the model to plan slightly ahead. But the same head has a second life at inference: it becomes a **draft model for speculative decoding**. And the beautiful part is that it's *one head doing both jobs.*

**The mechanism.** The MTP head sits on top of the main model's hidden states and is trained with a scaled loss (weight $\alpha = 0.1$) to predict $t{+}2$. The total objective is just

$$
\mathcal{L} = \mathcal{L}_{\text{next-token}} + \alpha \, \mathcal{L}_{\text{MTP}}, \qquad \alpha = 0.1.
$$

Because $\alpha$ is small, the main next-token objective is essentially untouched — the validation-loss curves with and without MTP are nearly identical. At evaluation time the head is simply removed; it does not change the main model's predictions at all. At *serving* time, the head drafts the next-next token, and the main model verifies the draft in a single forward pass — if the draft matches what the main model would have produced, you emit two tokens for the price of one step. (The mechanics of accept/reject are the subject of [speculative decoding in production](/blog/machine-learning/model-serving/speculative-decoding-in-production).)

![Two side-by-side columns titled "At training time" and "At inference time", each showing the same MTP head consuming main-model hidden states — on the left producing an auxiliary loss (weight 0.1) that sharpens pre-training, on the right drafting token t+2 for the main model to verify and emit up to 2 tokens per step](/imgs/blogs/mellum2-code-moe-3.webp)

**The worked payoff.** In an ablation on a 14B MoE trained for 105B tokens, adding the MTP head cost only **7% extra training time** and moved benchmarks substantially — HumanEval +10.4, MMLU +3.6, MMLU-Pro +3.3, GSM8K +3.0 (with a couple of tasks flat or slightly negative). So MTP is close to a free lunch here: a small planning-ahead signal that improves the model *and* gives you a matched draft head for speculative decoding, without a separate draft model to train, host, or keep in sync.

| Benchmark | Baseline | + MTP | Δ |
|---|---|---|---|
| HumanEval (pass@1) | 20.73 | 31.10 | **+10.37** |
| HumanEval+ (pass@1) | 18.29 | 26.22 | +7.93 |
| MMLU | 37.49 | 41.06 | +3.57 |
| MMLU-Pro | 19.07 | 22.32 | +3.25 |
| GSM8K | 30.63 | 33.59 | +2.96 |
| MBPP (pass@1) | 36.80 | 36.80 | 0.00 |
| Winogrande | 59.51 | 58.01 | −1.50 |

**When it fails.** MTP's inference speedup depends on the draft being *accepted* often — for hard, high-entropy positions the draft misses and you fall back to one token per step. And the training benefit is scale- and recipe-dependent: the report explicitly notes MTP "does not degrade" the primary objective, but the gains here are on a specific 14B/105B ablation, not a guaranteed constant.

### The final configuration

Putting the four decisions together yields the model summarized in the report's Table 2. It's worth having the numbers in one place, because the rest of the paper keeps referring back to them:

| Group | Setting | Value |
|---|---|---|
| **Scale** | Total / active parameters | ~12B / ~2.5B |
| | Vocabulary | 98,304 (untied embeddings) |
| | Context length | 8,192 → 131,072 (after extension) |
| **Backbone** | Layers · hidden | 28 · 2,304 |
| | Norm · activation | RMSNorm ($\epsilon = 10^{-6}$) · SiLU-gated |
| | Position encoding | RoPE, base $\theta = 500{,}000$ |
| **Attention** | Query / KV heads | 32 / 4 (GQA), head dim 128 |
| | QK-Norm · sliding window | Yes · 1,024 (3:1 pattern) |
| **MoE + MTP** | Experts total / active | 64 / 8 (top-8) |
| | Expert MLP size · shared expert | 896 · none |
| | MTP layers | 1 ($\alpha = 0.1$) |

One easy-to-miss detail from a footnote is a good example of the "inference-first" discipline: every matrix dimension — hidden 2,304, head dim 128, expert size 896 — is kept divisible by 128. Violating that alignment can cost up to a **2× slowdown** in GPU kernel execution, so it was treated as a hard constraint throughout the architecture search, not a nice-to-have.

## Pre-training: 10.6 trillion tokens in three acts

With the architecture fixed, the model is pre-trained on approximately **10.6 trillion tokens** drawn from three broad sources — web/general knowledge, source code, and mathematical content. The interesting part is not the sources; it's the *schedule*.

### The three-phase curriculum

**The problem it solves.** Not all data is equally valuable, and its value depends on *when* the model sees it. Curated high-quality data is scarce and expensive; noisy web data is abundant. Spending your best data early — while the learning rate is still warming up and the model is thrashing — wastes it. Spending it late, as the learning rate decays and weights settle, is where it sticks.

**The intuition.** This is the "web early, curated late" paradigm (Llama 3.1, DeepSeek-V3, SmolLM2). Think of it like an education: broad general reading first to build fluency and world model, then progressively more specialized, higher-quality material as you approach the exam. Mellum 2 splits pre-training into three phases whose boundaries line up with the learning-rate schedule, and steadily raises the code fraction from 23% to 59%.

| Phase | Tokens | % of total | LR state | Web % | Code % | Math % |
|---|---|---|---|---|---|---|
| 1: Foundation | 6.18T | 58.0 | Warmup → Hold | 70 | 23 | 6 |
| 2: Quality Uplift | 2.79T | 26.2 | Hold | 44 | 42 | 14 |
| 3: Capability Sharpening | 1.69T | 15.9 | Decay | 23 | 59 | 18 |

The reasoning behind each boundary is precise. **Phase 1** builds broad linguistic capability on mostly web data while the LR warms up and holds. **Phase 2** introduces the curated datasets — SFT-style data, reasoning QA, STEM instruction — *deliberately here rather than in Phase 1*, because curated data is more effective during a stable learning rate than during warmup. **Phase 3** is where the LR decays to zero and the model is most sensitive to data quality, so it maximizes code (59%) and reduces web to only the highest-quality curated sources. The raw code corpus is seen for three epochs across the run (contributing ~958B tokens), and no dataset is repeated more than 4× — the point past which the authors found repetition stops helping. Repetition matters *more* for MoE, because seeing high-quality data multiple times sharpens expert specialization in a way one noisy pass cannot.

### Fill-in-the-middle

There's a second objective running alongside next-token prediction. In an IDE, the model must complete code at the cursor conditioned on *both* the prefix and the suffix — the code after the cursor is right there. Standard left-to-right modeling can't use it. **Fill-in-the-Middle (FIM)** fixes this: split a document at two random points into (prefix, middle, suffix), reorder with sentinel tokens, and train the model to generate the middle given both ends. Mellum 2 uses a 50/50 mix of the two orderings (Prefix-Suffix-Middle and Suffix-Prefix-Middle).

The FIM *rate* varies across the curriculum, which is a nice touch: 50% in Phase 1 (expose the model to bidirectional context early), dropped to 10% in Phase 2 (so the freshly-introduced curated code and instruction data is consumed mostly under normal left-to-right modeling), then restored to 50% in Phase 3 but **restricted to source-code files only**. The schedule concentrates FIM on exactly the data distribution that matches the downstream completion setting, while keeping natural-language generation clean.

### Muon, FP8, and Warmup-Hold-Decay

This is the mathematically richest part of the recipe, and it's where Mellum 2 departs most from the standard AdamW playbook.

**The optimizer: Muon.** AdamW updates each weight independently using per-coordinate adaptive step sizes. **Muon** takes a different view for the 2D weight matrices of hidden layers: it treats the momentum-smoothed gradient as a *matrix* and orthogonalizes it before applying the update.

*Why orthogonalize?* The intuition is that a raw gradient-momentum matrix is often dominated by a few large singular directions — the update mostly pushes along one or two axes and neglects the rest. Orthogonalizing replaces that matrix with the nearest matrix whose singular values are all 1 (its "sign" in the matrix sense), so the update pushes *equally* along every direction the gradient identified. It's a more balanced, better-conditioned step.

The mechanics: keep a momentum buffer $M_t$ of the gradient, then compute an approximate orthogonalization $\text{msign}(M_t) \approx M_t (M_t^\top M_t)^{-1/2}$ using a fixed **Newton–Schulz** iteration — 5 steps of a cheap polynomial that converges to the orthogonal factor without any explicit matrix inverse or SVD:

$$
X_{k+1} = a\,X_k + b\,(X_k X_k^\top)\,X_k + c\,(X_k X_k^\top)^2 X_k,
$$

started from $X_0 = M_t / \lVert M_t \rVert$ and iterated $k = 0 \dots 4$. The coefficients $(a,b,c)$ are chosen so the iteration drives all singular values toward 1. Muon is applied only to the hidden 2D layers; embeddings and the output layer still use Adam, because orthogonalization only makes sense for the weight matrices that mix features.

The ablations here are worth reading as a cautionary tale about defaults. On a dense 7B, Muon with *Megatron default* scaling (extra scale factor 1.0) **diverged immediately** — the loss shot to 2.47 by 21B tokens — while the *Moonlight* configuration (extra scale factor 0.2) beat AdamW by 0.028 validation loss (~2.5%). On the 14B MoE, both Muon configs converged. They chose the Moonlight setup for stability across both architectures. The lesson: the orthogonalized update's magnitude has to be scaled correctly, and the "obvious" default is the one that blows up.

**The precision: FP8 hybrid.** Training uses BF16 as the base with FP8 hybrid mixed precision (tensorwise recipe, most-recent amax algorithm), while **gradient reduction is done in FP32** to preserve numerical stability. FP8 halves the memory and bandwidth of the matmuls that dominate training compute; keeping the reduction in FP32 avoids the accumulation errors that would otherwise corrupt the gradient. (The serving-time analog, quantizing weights for inference, is covered in [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving).)

**The schedule: Warmup-Hold-Decay.** Rather than a cosine decay to a nonzero floor, Mellum 2 uses a WHD schedule: warm up linearly over 2,000 steps to a peak of $3 \times 10^{-4}$, *hold* at peak through Phases 1 and 2, then **decay linearly to zero** over Phase 3 (~49,306 steps, ~15% of training). The linear-decay-to-zero choice follows recent findings that it beats cosine-to-nonzero-minimum, delivering equivalent loss at lower effective compute. The batch size ramps from 2,048 to 4,096 sequences early on; at full batch each step processes ~33.6M tokens.

![Figure 5 from Kojic et al. (2026): the Warmup-Hold-Decay learning-rate schedule holding at 3e-4 through Phases 1–2 then decaying linearly to zero in Phase 3, overlaid with the batch-size rampup from 2,048 to 4,096 and the three curriculum-phase boundaries](/imgs/blogs/mellum2-code-moe-fig2.webp)

One small optimizer detail the authors flag: setting Adam's $\epsilon$ as large as $10^{-5}$ (the LLaMA 2 value) disproportionately dampens updates; they use $\epsilon = 10^{-8}$.

### MoE-specific training and the stability diary

A few production-scale details separate "trained an MoE in a notebook" from "trained one on 256 GPUs across two continents":

- **Global-batch load balancing** (not per-sequence). Per-sequence balancing gave slightly better loss on short runs, but global-batch balancing is more flexible with variable batch sizes, so it won.
- **Sequence packing with best-fit.** Documents are packed into fixed 8,192-token sequences using best-fit packing, which minimizes intra-document truncation versus naive concatenate-and-chunk — and reduces hallucinations caused by spurious cross-document context bleeding across a truncation boundary.
- **Geographically split infrastructure.** Data processing runs offline in France; the GPU training fleet runs in the United States. A background streamer feeds pre-tokenized shards through an in-memory Redis queue, so the two systems share no filesystem and the transatlantic latency never touches the training loop.

The report's candor about **stability incidents** is one of its most useful contributions, because these are the failures every large run hits and few report:

1. **Loss spikes from low-diversity sequences.** Early spikes traced to sequences with pathologically low lexical diversity (e.g., a single token repeated across the whole context). Fix: filter samples with fewer than 82 unique tokens (1% of the 8,192 context).
2. **Periodic loss dips from hash-sorted duplicates.** The data pipeline sorted samples by a hash of the token sequence. Long documents sliced into 8,192-token chunks produced exact-duplicate chunks, and hash-sorting placed them at the same offset in every one of the 16 shards per phase — producing 16 evenly-spaced downward loss dips per phase. They verified the dips were modest and harmless (no effect on the load-balancing loss) and, since removing duplicates from already-prepared data was non-trivial, *chose to keep training with them in place.* An honest engineering call.
3. **A load-balancing-loss shift after cluster migration.** Halfway through, they migrated from 32 nodes to 16 while keeping the effective batch size fixed. The global load-balancing loss dropped noticeably — but this was an *accumulation-semantics artifact*, not a model change: Megatron-LM computes the auxiliary loss against a running per-expert token count that resets at gradient finalization, and with fewer data-parallel ranks there are more gradient-accumulation microbatches per step, so the running average converges closer to the true distribution before reset. Same optimization signal, systematically lower reported number. The kind of thing that looks alarming on a dashboard and is completely benign once you understand the implementation.

## Stretching the context to 128K: layer-selective YaRN

After the main run, a dedicated stage extends the effective context from 8,192 to 131,072 tokens (128K). The technique is a targeted twist on YaRN.

**The problem it solves.** RoPE encodes position by *rotating* each query and key vector by an angle proportional to its position. If you train at 8K and then feed 128K tokens, the positions at the far end correspond to rotation angles the model never saw — it extrapolates badly and quality collapses. YaRN fixes this by re-mapping the rotation frequencies so the extended range fits within the angles the model already understands.

**The intuition, and the twist.** RoPE's dimensions rotate at different frequencies — some fast (sensitive to nearby tokens), some slow (sensitive to distant structure). YaRN interpolates the slow, low-frequency dimensions (the ones that would "wrap around" at long context) while leaving the fast local ones alone. Mellum 2's insight: **apply YaRN only to the full-attention layers.** The sliding-window layers operate on a fixed local span of 1,024 tokens — they *never* attend beyond the window, so they never need to extrapolate to long positions, and re-mapping their frequencies only perturbs something that was already working. Only the full-attention layers have to reach across the new, longer sequence.

**The math sketch.** The context-extension factor is $s = L_{\text{new}} / L_{\text{orig}} = 131072 / 8192 = 16$. YaRN spreads that factor across RoPE's frequency bands with a ramp — high-frequency dims are left near their original rotation, low-frequency dims are interpolated toward ${1/s}$ of their original angle — plus a small attention-temperature scaling. In Mellum 2 this re-mapping is applied to the full-attention layers only; the SWA layers keep their original RoPE parameters unchanged.

**The evidence.** The ablation is clean. At a 64K evaluation context, layer-selective YaRN reaches a RULER score of **0.64**, versus **0.52** for a uniform RoPE-base bump on all layers and **0.33** for leaving the base unchanged. The gap widens with context: the unchanged-base run never adapts its full-attention layers and collapses past 32K, while the uniform bump *unnecessarily perturbs* the sliding-window layers that were already fine. (The authors caution the absolute RULER numbers are depressed by a prompt-formatting bug on the QA subsets — read the figure as a within-recipe ranking, not RULER's last word.)

![Figure 7 from Kojic et al. (2026): RULER score versus evaluation context length for three long-context recipes — layer-selective YaRN stays highest at every context length from 16K to 128K, the uniform theta-bump is middle, and the unchanged RoPE base collapses fastest](/imgs/blogs/mellum2-code-moe-fig3.webp)

**A surprising negative result.** They tried to reproduce OLMo 3's "Longmino" long-context data mix and *couldn't replicate its gains* — with everything else held constant, adding it *lowered* RULER by 2–3 points at every context length. More broadly, different mixtures produced very similar numbers, with their own base mix narrowly on top, and quality plateaued after only ~30B tokens of long-context training. The only thing that kept changing past 30B was the MoE router's load-balancing loss, as the router re-equilibrated to the new sequence-length regime — which is why they extended the run to ~117B tokens anyway, to let the router settle before annealing. A reminder that long-context data-mix gains reported by one group don't always transfer.

## Post-training: SFT then RLVR

The long-context base is turned into two deployable assistants through a two-stage pipeline: supervised fine-tuning, then reinforcement learning.

### Two SFT variants: Instruct and Thinking

Both variants start from the *same* long-context YaRN checkpoint and the same data mix; they differ only in chat template and how reasoning traces and loss masking are handled:

- **Instruct (no-thinking)** answers directly. Loss is computed on *every* assistant turn, all other tokens masked, and any reasoning fields in the source data are discarded.
- **Thinking** emits an internal chain of thought before its final answer. Only the *last* assistant turn plus its reasoning trace contributes to the loss; earlier turns are conditioning context, and conversations without a reasoning trace are excluded. To amplify the signal on multi-turn data, each conversation is "unfolded" — the loss target slides across successive assistant turns, producing up to five training samples per source conversation.

Both keep the MTP head active, use the same distributed Muon optimizer, run three epochs, pack sequences to the full 131,072 length (dropping samples that don't fit rather than truncating), and lower the peak LR to $3 \times 10^{-5}$ (a tenth of pre-training). The MoE auxiliary-loss coefficient is dropped from $10^{-3}$ to $10^{-4}$ — the router is already well-balanced, so a smaller coefficient avoids over-constraining expert use on the narrower SFT distribution. The Instruct run consumes ~47B tokens; Thinking, with its turn-unfolding, ~167B.

An amusing data-composition detail: they oversample self-identity dialogues **3×** so the model reliably introduces itself as Mellum 2. Without them, early runs had the model insisting it was *"an AI assistant developed by Google"* — despite no Google models being used to generate any synthetic data. A vivid illustration of how strongly identity priors leak in from the broader pre-training distribution.

### RLVR: GRPO with an MoE twist

The final stage refines each SFT checkpoint with **Reinforcement Learning with Verifiable Rewards (RLVR)**. The choice of RLVR over RLHF is deliberate: every prompt in their corpus admits an *unambiguous, programmatic* correctness check — run the unit tests, check the math answer symbolically — so there's no need to train a separate reward model whose noise could dominate the gradient. (For the RLHF-style alternative and the GRPO lineage, see the sibling analysis of [DeepSeek-R1's RL recipe](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning).)

**The system.** RL runs on a Kubernetes cluster split into two fixed roles: a small group of *training* nodes that own the policy weights and run gradient updates, and a larger group of *inference* nodes that host the generation engines (vLLM) and produce rollouts. Reward computation is decoupled into its own microservice cluster — a verification gateway routes each request to the right backend (a code-execution sandbox, a symbolic/numeric math verifier, an LLM-as-judge for free-form outputs) — so verifiers never contend for GPUs with the trainer or the generation engines.

![A left-to-right system graph: a GRPO trainer that owns the policy weights syncs weights to vLLM engines, whose rollouts flow through a verification gateway that fans out to a code sandbox, a math verifier, and an LLM-as-judge, whose graded rewards fan back into a trajectory buffer feeding the GRPO update](/imgs/blogs/mellum2-code-moe-4.webp)

The whole loop is **asynchronous**: trajectory collectors stream completed rollouts into a global buffer, the trainer pulls batches from it rather than waiting for generation, and after each weight push the inference engines recompute their KV cache so prefix logits stay consistent with the new policy. A rollout's tokens are allowed to be at most two training steps older than the policy used in the update — a bounded staleness window that keeps the trainer busy without letting rollouts drift too far off-policy.

The verification gateway also distinguishes two failure types so the trainer sees a clean signal: an *un-scoreable model response* gets a zero reward (and the model is shown the error string on its next rollout), while a *transient verifier outage* is retried rather than punished. Conflating those two would teach the model to avoid whatever tripped a flaky verifier.

#### The GRPO + IcePop objective

Both RL stages use a variant of GRPO (Group Relative Policy Optimization) adapted for asynchronous rollouts on a BF16 MoE policy. Let me build the loss up one piece at a time, because four separate ideas are stacked in it.

**1. The advantage: leave-one-out, no std normalization.** GRPO avoids a learned value function by sampling a *group* of $G$ responses per prompt and using the group's own rewards as the baseline. Mellum 2 uses a leave-one-out baseline (following Dr. GRPO):

$$
A_i = R_i - \frac{1}{G-1} \sum_{j \neq i} R_j,
$$

where $R_i$ is the verifier's reward for response $i$. The advantage is positive when a response beats the average of its siblings. Crucially they do *not* divide by the group's standard deviation — std-normalization biases the gradient toward prompts where the model is already inconsistent.

**2. The PPO ratio and clip-higher.** As in PPO, each token's update is scaled by the probability ratio between the current and pre-step policy, $r_{i,t} = \pi_\theta(y_{i,t} \mid y_{i,\lt t}) / \pi_{\theta_{\text{old}}}(y_{i,t} \mid y_{i,\lt t})$, and clipped to a trust region. The twist (from DAPO) is an **asymmetric** clip range $[1 - \epsilon_{\text{low}},\ 1 + \epsilon_{\text{high}}]$ with $\epsilon_{\text{low}} = 0.2 \lt \epsilon_{\text{high}} = 0.28$ — "clip-higher" lets positive-advantage updates (good behaviors) flow more freely than negative ones.

**3. IcePop: the MoE-specific fix.** Here is the subtle problem that makes MoE RL genuinely hard. The rollouts are generated by the inference engine (vLLM); the gradient is computed by the trainer. Even when the two are *nominally the same model with identical weights*, they disagree on per-token log-probabilities. The dominant cause is the **router itself**: for the same hidden state, the inference-time router may dispatch a token to a *different expert* than the trainer-side router, and the resulting log-probs differ even though the weights are identical. BF16 numerical noise adds to it. They track this train-versus-inference disparity as

$$
\rho_{i,t} = \frac{\pi_{\text{train}}(y_{i,t} \mid y_{i,\lt t};\, \theta_{\text{old}})}{\pi_{\text{infer}}(y_{i,t} \mid y_{i,\lt t};\, \theta_{\text{old}})},
$$

which is *not* exactly 1 even before any gradient step. Left unbounded, a handful of expert-flipped tokens with huge $\rho$ would dominate the gradient. **IcePop** truncation zeroes out any token whose disparity falls outside a band $[\alpha, \beta] = [0.5, 5.0]$:

$$
M(\rho) = \begin{cases} \rho & \text{if } \alpha \le \rho \le \beta, \\ 0 & \text{otherwise.} \end{cases}
$$

Unlike PPO's clip (which *caps* an out-of-band ratio at the clip edge), IcePop **drops the token entirely** — the right call when the cause of a large $\rho$ is an expert flip rather than a genuine on-policy update you want to apply.

**4. Put it together.** The per-step loss the trainer minimizes is

$$
\mathcal{L}_{\text{GRPO}} = -\frac{1}{N_{\text{tok}}} \sum_{i,t} M(\rho_{i,t}) \, \min\!\Big( r_{i,t} A_i,\ \text{clip}(r_{i,t},\, 1-\epsilon_{\text{low}},\, 1+\epsilon_{\text{high}})\, A_i \Big),
$$

where $N_{\text{tok}}$ is the total number of valid generated tokens in the batch — token-level normalization (from DAPO / Dr. GRPO) so every generated token contributes equally regardless of which response it came from. In pseudocode the whole step reads:

```python
# One GRPO+IcePop step for a group of G rollouts on one prompt
rewards = [verify(resp) for resp in group]          # programmatic, in [0, 1]
baseline = (sum(rewards) - r_i) / (G - 1)           # leave-one-out, per response
A_i = r_i - baseline                                # no std normalization

for t, tok in enumerate(response_i):
    r   = pi_train(tok) / pi_train_old(tok)         # standard PPO ratio
    rho = pi_train_old(tok) / pi_infer_old(tok)     # train-vs-inference disparity
    if not (0.5 <= rho <= 5.0):                      # IcePop: drop expert-flipped tokens
        continue                                     # contribution zeroed, not clipped
    clipped = clip(r, 1 - 0.2, 1 + 0.28)            # clip-higher (eps_low < eps_high)
    loss -= rho * min(r * A_i, clipped * A_i)
loss /= num_valid_tokens                             # token-level normalization
```

Notably, there is **no KL penalty** anchoring the policy to the SFT reference — recent large-scale open RL systems (DAPO, OLMo 3, Qwen3) have converged on dropping it, and Mellum 2 follows suit.

#### Reward shaping

Two shaping rules sit on top of the verifier's raw score:

- **Soft overlong penalty (from DAPO).** Rewards inside a buffer just below the max response length interpolate down to a floor at the cap; rollouts that *exceed* the cap are dropped from the loss entirely. This avoids training on samples that merely ran out of budget while preserving the signal on shorter ones.
- **Concision penalty on non-thinking responses.** During an early Instruct run, the policy started emitting inline reasoning *without* the `<think>` delimiters — long "wait, actually, hmm, let me reconsider" ramblings that violate the Instruct model's "answer briefly" contract. They follow an ARLCP-style penalty: multiplicatively shrink the reward on *correct* rollouts in proportion to the count of reflection-trigger words (bucketed into three tiers), applied only where that lexicon isn't legitimately part of the output (so Thinking-mode math isn't penalized). The effect is large: average reflection-trigger words per rollout fell from **7.3** (no penalty) to **0.6** (production) in late-training math rollouts.

The RL hyperparameters differ mainly in sequence budget: the Instruct stage runs 500 steps with a 16,384-token max sequence; the Thinking stage is a *cold restart* from the SFT-Thinking checkpoint, runs 100 steps, and lifts the max sequence to 40,960 to accommodate long chains of thought (forcing the trainer micro-batch down to 1). Both use 256 prompts × 16 generations per step, 1.5× oversampling with zero-variance groups discarded (dynamic sampling, à la DAPO), an IcePop band of $[0.5, 5.0]$, and a constant AdamW LR of $10^{-6}$.

## Results: where a 2.5B-active model wins and loses

The honest way to read the results is through the lens of the design constraint. Mellum 2 activates 2.5B parameters and is competing against dense baselines from 4B (Qwen3.5-4B) to 14B (Ministral-3-14B). It wins where the domain matches its training mix and loses where breadth is the point.

**Pre-training base model.** Despite 2.5B active parameters, the base is competitive with 7B dense models and *exceeds* them on several reasoning and code tasks:

| Benchmark | Mellum 2 (2.5B/12B) | OLMo-3-7B | Qwen2.5-7B | Qwen3-4B |
|---|---|---|---|---|
| MMLU-Pro | **59.3** | 34.5 | 48.6 | 51.5 |
| BBH | **74.9** | 63.6 | 69.0 | 71.3 |
| GSM8K | 81.7 | 73.5 | 81.9 | 82.0 |
| MBPP | 62.4 | 50.6 | 63.6 | 67.0 |
| HumanEval | 41.5 | 45.1 | 55.5 | 57.3 |
| MATH | 10.0 | 18.7 | 24.6 | 27.7 |
| GPQA Diamond | 31.3 | 28.8 | 32.8 | 36.9 |

MMLU-Pro (59.3, beating everything including the 7B baselines) and BBH (74.9) are standouts. HumanEval (41.5) is a weak spot the authors flag explicitly as a growth area — and indeed it lifts substantially after post-training. MATH is low at the base stage.

**Post-training, and where RL earns its keep.** The post-trained Instruct and Thinking variants show the clearest single-stage jumps from RL. A few load-bearing numbers (RL = after RLVR):

- **EvalPlus (robust function synthesis):** Mellum 2-RL leads the instruct panel at **78.4**, ahead of Qwen3.5-9B (71.8) and the code-specialized Seed-Coder-8B (73.8). This is the regime the pre-training mix targets directly.
- **LiveCodeBench (competitive-programming reasoning):** the *instruct* variant lags the Qwen3.5 series badly (37.2 vs 63.7), but the **Thinking** variant hits **75.1** — top of the panel, 6.8 points ahead of Qwen3.5-9B-Thinking. The read: algorithmic reasoning is in reach but needs an explicit thinking budget to unlock, whereas function synthesis transfers from pre-training without one.
- **Tool use (BFCL v3):** RL lifts instruct from 43.1 → 66.3 and thinking from 60.5 → 69.4. On BFCL v4 (adds agentic web-search/memory), Mellum 2-RL-Thinking *leads* the panel at 45.6 — the function-calling RL recipe transfers to held-out agentic settings.
- **Math (AIME):** 29.9 → 41.7 (instruct) and 20.0 → 58.4 (thinking) after RL — RL is transformative here.

**The principal weakness: knowledge.** On MMLU-Redux and GPQA Diamond, the Qwen3.5 series dominates: 91.1 / 79.8 at 9B against Mellum 2's 78.1 / 40.9 (instruct) and 86.2 / 57.6 (thinking). GPQA — graduate-level science outside computer science — is essentially a probe of factual depth the training mix deliberately traded away. For a code assistant this profile is defensible, but it bounds off-domain use and the authors surface it plainly to deployers.

**The efficiency headline.** All of this at the promised cost. On a single H100 with vLLM FP8 serving, at a production-representative workload (2,304 input / 256 output tokens), Mellum 2 matches Qwen2.5-7B's single-request latency to within one token (192 vs 193 tok/s) and pulls **21% ahead** under concurrency (5,179 vs 4,283 output tok/s), and 79% ahead of Qwen3-8B.

![Figure 13 from Kojic et al. (2026): output tokens/s on a single H100 — in sync mode Mellum 2 (192) matches Qwen2.5-7B (193); in throughput mode Mellum 2 (5,179) leads Qwen2.5-7B (4,283) and Qwen3-8B (2,897)](/imgs/blogs/mellum2-code-moe-fig4.webp)

**What's load-bearing that may not transfer.** The results rest on several choices that are specific to *this* deployment. The whole architecture is tuned to a single-H100, short-context, single-batch IDE workload — the report is explicit that hybrid attention (Gated DeltaNet) variants they tried *regressed* on exactly this short-context regime, even though they'd win on long-context large-batch serving. The pre-training mix is deliberately code-and-developer-doc heavy, which is why the model is strong on developer-flavored tasks and weak on GPQA. And the RL gains lean on programmatically verifiable rewards, which exist for code and math but not for most conversational quality. Read the numbers as "what an inference-first code MoE looks like," not "what a general 12B MoE looks like."

## Critique

**What's genuinely strong.** The disciplined thing here is the *methodology*, not any single trick. Fixing an inference budget and then searching the architecture space against it is the right way to build a deployable model, and the report follows it consistently — every choice (MoE-vs-dense, 8-of-64, 4 KV heads, 3:1 SWA, one MTP head) is tied to an ablation under that constraint, and the paper reports the negative results too (MLA insufficient, shared expert hurts, hybrid attention regresses on short context, Longmino mix didn't transfer). The stability diary — low-diversity spikes, hash-sorted-duplicate dips, the cluster-migration load-balancing artifact — is the kind of operational honesty that's rare and genuinely useful. IcePop's framing of MoE RL instability as *router non-determinism between the inference and training forward passes* is a clean articulation of a problem many teams have hit and few have named this precisely.

**What's weak or unfalsifiable.** Several of the headline efficiency claims come from *the ongoing production run* rather than a controlled from-scratch comparison, and the key ablations (Muon, MTP, optimizer) are on smaller proxy models (7B dense, 14B MoE, 105B tokens) — the report is careful to say so, but it means the specific numbers (MTP's +10.4 HumanEval, Muon's −2.5% loss) are not guaranteed to hold at the full 10.6T-token scale. The RULER long-context numbers are self-admittedly depressed by a prompt-formatting bug and usable only as a within-recipe *ranking*, which weakens the absolute 128K capability claim. And "competitive with 4–14B baselines" is doing a lot of work in the abstract — on knowledge benchmarks the gap to the 9B Qwen3.5 is not competitive, it's a rout (40.9 vs 79.8 on GPQA), and the RLVR safety regression (HarmBench 8.4 → 23.1 after RL) is a real alignment tax the report acknowledges but doesn't fix.

**What ablation is missing.** The most conspicuous gap is a clean, matched-compute **dense-vs-MoE quality comparison at the final scale** — the dense sweeps are described qualitatively ("none consistently outperformed Qwen2.5-7B") without a table, so the reader takes the central architectural decision partly on faith. I'd also want the speculative-decoding *acceptance rate* and the realized end-to-end speedup from the MTP head at serving time — the paper motivates MTP-as-draft but reports the throughput numbers without isolating how much of the win comes from the draft head versus the sparsity and KV-head choices.

**What would change my mind.** If a matched-parameter dense baseline, trained on the identical 10.6T-token curriculum with the same Muon+FP8 recipe, matched Mellum 2's quality within the same single-H100 latency budget, the case for the MoE backbone would largely collapse — the entire thesis rests on the claim that no dense config can, and that claim is asserted more than shown at final scale.

## What I'd build with this

These are my extrapolations, not the paper's claims:

1. **Repository-level SWE-RL.** The authors name this as the obvious next step, and it's the highest-leverage one: the RLVR harness (verification gateway + code sandbox + async rollouts) already exists, so pointing it at real repository-level edit tasks with test-suite rewards — SWE-bench-style — is mostly a data-and-environment problem, not a systems one.
2. **A measured speculative-decoding study.** Publish the MTP head's acceptance rate as a function of temperature and task, and the realized tokens/step. If the head accepts, say, 1.7 tokens/step on completion workloads, that reframes the whole efficiency story and would justify a *bigger* MTP head (2 layers, predicting $t{+}2$ and $t{+}3$) for a further multiplier.
3. **Router-consistency at inference.** IcePop treats train-vs-inference router disagreement as noise to truncate. The more ambitious fix is to make the two forward passes *agree* — deterministic routing, or having vLLM and the trainer share a routing implementation — which would recover the tokens IcePop currently discards and tighten the RL signal.
4. **A knowledge-recovery pass.** The GPQA/MMLU gap is the model's one glaring weakness and it's a data choice, not an architecture limit. A late-stage mid-training injection of dense encyclopedic and scientific data (a fourth curriculum phase) could close part of it without disturbing the code strengths — worth an ablation.
5. **The "larger, similarly inference-aware Mellum" the authors tease.** The recipe — pick architecture by ablation against a fixed inference budget — is scale-free. Applied to an H200 or a 2×H100 budget, it would produce a different but equally principled design point, and the interesting question is which choices (KV-head count, SWA ratio, sparsity) move and which stay put as the budget grows.

## References

- **Paper:** Kojic, Bondyrev, de Moor, Shtok, et al. *Mellum 2 Technical Report.* arXiv:2605.31268 (2026). [arxiv.org/abs/2605.31268](https://arxiv.org/abs/2605.31268)
- **Predecessor:** Pavlichenko et al. *Mellum: Production-Grade in-IDE Contextual Code Completion.* arXiv:2510.05788 (2025).
- **Weights & license:** base, instruct, and thinking checkpoints, Apache 2.0, on Hugging Face (see the paper's release links).
- **Related on this blog:**
  - [Serving MoE models at scale](/blog/machine-learning/model-serving/serving-moe-models-at-scale) — the deployment-side view of the sparsity Mellum 2 exploits.
  - [Speculative decoding in production](/blog/machine-learning/model-serving/speculative-decoding-in-production) — the accept/reject mechanics the MTP draft head plugs into.
  - [MoE scaling laws](/blog/machine-learning/scaling-laws/moe-scaling-laws) — why 8-of-64 sits where it does on the quality-sparsity curve.
  - [DeepSeek-R1: incentivizing reasoning via RL](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) — the GRPO lineage Mellum 2's RLVR stage builds on.
