---
title: "Monolith-1: Reading a 1.6-Trillion-Parameter Open MoE — Architecture, Compute, and the Fine Print"
date: "2026-07-18"
description: "A technique-by-technique walk through Basalt's Monolith-1 report: how a sparse 1.6T mixture-of-experts is built, why it activates only 49.5B parameters per token, how it stretches to a million-token context — and why the eye-popping benchmark numbers deserve a very careful read."
tags: ["paper-reading", "mixture-of-experts", "moe", "deepseekmoe", "grouped-query-attention", "yarn", "long-context", "multi-token-prediction", "rlvr", "ascend", "llm", "sparse-models"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 32
paper:
  title: "Monolith-1: A 1.6T Open Mixture-of-Experts Foundation Model for Reasoning at Scale"
  authors: "Chen Mingzhi, Li Wenjuan, et al. (The Basalt Research Team)"
  venue: "Basalt PBC technical report, April 2026"
  url: "https://basaltlabs.org/Monolith-1.0-2606.pdf"
---

> [!tldr]
> - **What it is.** Monolith-1 is a sparsely-activated transformer with **1.57T total parameters** but only **49.5B active per token** — a 32× sparsity ratio — trained for roughly ${1.78 \times 10^{25}}$ FLOPs on **60 trillion tokens**, then stretched to a **1,048,576-token** context.
> - **The mechanism in one line.** Every feed-forward block is a DeepSeekMoE-style layer that routes each token to the **top-2 of 128 experts** plus **one always-on shared expert**, with an *auxiliary-loss-free* balancer keeping the experts evenly loaded.
> - **Why it's interesting.** It's a frontier-scale MoE trained end-to-end on **12,288 Huawei Ascend 910C NPUs** (no FP8, all BF16), released under the MIT license — a rare open look at trillion-parameter training on non-NVIDIA silicon.
> - **The surprising number.** It claims **99.4% on Humanity's Last Exam**, where the strongest cited competitors sit near 40% — but that headline is produced under a **best-of-32, model-judges-its-own-answers** protocol, not a standard eval.
> - **The fine print (read this twice).** The report's own **Appendix E** states that the headline facts come from "the Basalt corporate website source," and that *almost every technical detail in the paper is inferred* by the authors for internal consistency — not measured from a real run. The techniques below are real and worth learning; the **results are not an independent measurement of capability**, and I'll treat them that way.

The diagram below is the mental model for the whole post: one decoder block, repeated eighty times, with the two pieces that do all the work — a memory-frugal attention layer and a sparse expert layer — wrapped in the standard pre-norm residual skeleton.

![The Monolith-1 decoder block: pre-norm RMSNorm feeds grouped-query attention and a Mixture-of-Experts FFN, with two residual skips carrying the input past each sublayer](/imgs/blogs/monolith-1-moe-1.webp)

This is a paper analysis with an unusual epistemic status, so let me be upfront about how I'm going to treat it. The **methods** in Monolith-1 are all real, published, load-bearing techniques — DeepSeekMoE routing, grouped-query attention, YaRN context extension, auxiliary-loss-free balancing, multi-token prediction, RLVR post-training. Explaining them carefully teaches you how a 2026-era frontier MoE is actually assembled, and that's most of this post. The **results and the specific measured claims** are a different matter, because the report itself tells us (in Appendix E) that they are drawn from a marketing source and reconstructed for consistency. So we'll climb the intuition-to-math ladder on every technique as if describing a real system — and then, in the Critique, we'll do the harder and more useful thing: read the numbers the way a skeptical reviewer must.

## The problem

For the last two years the very top of the language-model leaderboard has lived behind APIs. Open weights closed a lot of ground in 2024–2025 — Llama 3, DeepSeek-V3 — but the single best systems stayed private. Monolith-1 frames itself as an attempt to put a *frontier-scale* model into the open under a permissive (MIT) license, at "the scale of compute we can actually afford."

Underneath that framing is a technical problem that every lab building at this scale runs into: **capability scales with parameters, but inference cost scales with the parameters you actually use on each token.** If you make a dense model twice as big, every token now costs twice as many FLOPs — during training *and* forever afterward in serving. That's ruinous when your deployment is inference-bound, which at frontier scale it always is.

The mixture-of-experts (MoE) idea, going back to [Shazeer et al. (2017)](https://arxiv.org/abs/1701.06538) and scaled up by GShard and Switch Transformer, breaks that coupling. You store a huge bank of "expert" sub-networks but, for any given token, you **compute only a few of them**. Total parameters — which control how much the model can *know* and *route between* — grow freely; active parameters — which control per-token FLOPs — stay small. Mixtral brought MoE to the open frontier; DeepSeekMoE refined it into *fine-grained* experts plus a *shared* expert; DeepSeek-V3 combined that with an auxiliary-loss-free balancer at trillion-parameter scale. Monolith-1 sits squarely in that lineage, and its whole architecture is an answer to one question: **how do you get the knowledge of a 1.6-trillion-parameter model while paying the compute of a ~50-billion-parameter one?**

The rest of the report is the set of engineering choices that make that trade work at 60-trillion-token scale on domestic Chinese accelerators — and a long, unusually candid failure section about what didn't work.

## Contributions

Tightened into my own words, the report claims five things:

1. **An open 1.6T MoE under the MIT license**, redistributable commercially — the second openly-available model at the trillion-total-parameter scale as of April 2026.
2. **Evidence that fine-grained MoE keeps scaling**: 128 routed experts with top-2 routing plus a shared expert, giving 32× per-token sparsity "without observable destabilization in late training."
3. **A frontier-scale run on Huawei Ascend 910C NPUs** in CloudMatrix-384 super-pods, entirely in BF16, with a described parallelism plan and an HCCL port of the DualPipe schedule.
4. **A reproducible long-context recipe** stretching a 4K pretraining context to 1M tokens with only 2T extra tokens, via a two-stage YaRN curriculum.
5. **A complete release**: weights, tokenizer, config, an evaluation harness, a model card, and red-team summaries.

Keep contribution (2) and the phrase "without observable destabilization" in mind — the analysis section quietly qualifies it, and the critique section will pull on it hard.

## The architecture at a glance

Monolith-1 is a decoder-only transformer with **80 blocks**. Each block is the pre-norm sandwich you saw in the mental-model figure: RMSNorm → grouped-query attention → residual add → RMSNorm → MoE feed-forward → residual add. Input and output embeddings are tied. The full hyperparameter sheet is Table 1 from the report:

![Table 1 from the Monolith-1 report: the architecture specification — 80 layers, d_model 8192, 64 query heads over 8 KV groups, 128 routed experts plus 1 shared, top-2 routing, 1.048M context, 1.572e12 total parameters](/imgs/blogs/monolith-1-moe-fig1.webp)

A few of these numbers are worth internalizing before we unpack the mechanisms, because they recur throughout:

- **${d_{\text{model}}} = 8192$**, **80 layers** — a tall, wide trunk.
- **64 query heads, 8 KV heads, head dim 128** — grouped-query attention at an 8:1 ratio.
- **128 routed experts + 1 shared, top-2** — so **3 experts fire per token** (2 routed + 1 shared).
- **Expert intermediate width ${d_{\text{ff}}} = 6144$**, SwiGLU — each expert is a gated FFN.
- **Vocabulary 151,936**, tied embeddings (1.24B of the total parameters).
- **RoPE base 10,000 at pretraining, 5,000,000 after extension**; context **4,096 → 1,048,576**.
- **Total 1.572T, active 49.5B.**

Everything from here is one `###` per load-bearing technique. Each climbs the same ladder: the problem it solves, an intuition, the mechanism in words, the math with every symbol defined, a tiny worked example, and where it breaks.

## Method

### Sparsity: 1.6T stored, 49.5B active

**The problem.** A dense model pays FLOPs proportional to *all* its parameters on *every* token. At 1.57T parameters that would be about 9.4 TFLOP per token just for the forward pass — untenable to train on 60T tokens and worse to serve.

**Intuition.** Think of a large hospital. It employs hundreds of specialists, but any single patient sees only two or three of them — a cardiologist and a nurse, say — plus the general-practitioner they always see. The hospital's *capability* is the whole roster; the *cost of one visit* is the handful of people actually in the room. MoE is that hospital: 128 specialist experts on staff per layer, but each token books only its top two, plus one generalist who sees everyone.

**Mechanism.** For each token, a small router scores all 128 experts, the two highest-scoring experts are activated, their outputs are blended, and a single shared expert (always on) is added. The other 126 experts sit idle for that token — their parameters occupy memory but burn no FLOPs. Multiply that saving across 80 layers and the per-token compute collapses from "all 1.57T" to "49.5B."

**The math.** With ${N_{\text{active}}}$ active parameters per token and $D$ training tokens, the standard training-FLOP approximation ([Kaplan et al., 2020](https://arxiv.org/abs/2001.08361); [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) is

$$
C \;\approx\; 6 \, N_{\text{active}} \, D,
$$

where the factor of 6 counts one forward and two backward multiply-adds per parameter. For sparse models the *active* count is what enters the FLOP budget; the *total* count governs memory and routing capacity. Plugging in ${N_{\text{active}} = 4.95 \times 10^{10}}$ and ${D = 6.0 \times 10^{13}}$:

$$
6 \times (4.95 \times 10^{10}) \times (6.0 \times 10^{13}) \;\approx\; 1.78 \times 10^{25}\ \text{FLOP},
$$

matching the model card's ${1.8 \times 10^{25}}$. The sparsity ratio is ${1.572 \times 10^{12} / 4.95 \times 10^{10} \approx 32}$.

**Worked micro-example.** Compare the two worlds side by side. A dense 1.57T model would cost ${6 \times 1.572 \times 10^{12} \approx 9.4}$ TFLOP per token and about ${5.7 \times 10^{26}}$ FLOP to train on 60T tokens. The sparse MoE costs 0.30 TFLOP per token and ${1.78 \times 10^{25}}$ FLOP to train — 32× less, for (the claim goes) comparable capability.

![Why 1.6T parameters cost only 49.5B per token: the dense-equivalent path costs 32x the per-token compute and training FLOPs of the sparse MoE that Monolith-1 actually runs](/imgs/blogs/monolith-1-moe-2.webp)

**Why it works / when it fails.** The bet is that most tokens don't need most of the network — a token about French grammar and a token about Python syntax want different experts, and forcing them through the same dense FFN wastes compute. The failure mode is **load imbalance**: if the router sends 90% of tokens to a handful of experts, you've paid for 128 experts but trained 8, and the idle ones rot. Preventing that collapse is the job of the balancer we get to shortly. The second cost is memory: you still have to *store* 1.57T parameters somewhere, which is why serving needs 640 GB of GPU memory even at FP8.

### The DeepSeekMoE layer: fine-grained experts and a shared expert

**The problem.** The original MoE layers used a few big experts. That's coarse — an expert either fires or it doesn't, and with only 8 experts the router's decisions are lumpy. You want *finer* specialization (more, smaller experts) without losing the general-purpose capacity that a dense FFN provides for free.

**Intuition.** Two ideas stacked. First, **fine-grained experts**: slice the FFN capacity into many narrow experts (128 here) so routing can be surgical — a token can grab exactly the two sub-specialists it needs. Second, the **shared expert**: keep one expert that every token always uses, to absorb the "common knowledge" that would otherwise have to be redundantly learned by all 128. The routed experts specialize; the shared expert handles what everybody needs. Back to the hospital: 128 specialists *plus* one generalist every patient sees.

**Mechanism.** Each expert is a SwiGLU feed-forward block. For a token's hidden vector $x$, a linear router produces an affinity score for each of the 128 experts. The top-2 scores are selected; those two experts run and their outputs are weighted by softmax gates; the shared expert runs unconditionally; everything is summed onto the residual stream. The figure traces exactly this path — router on the left, two lit experts and a green always-on shared expert in the middle, the weighted combine on the right, and the dashed residual skipping the whole layer.

![Inside one MoE layer: the router scores 128 experts, activates the top-2 plus one always-on shared expert, and sums their gated outputs with the residual](/imgs/blogs/monolith-1-moe-3.webp)

**The math.** Each expert is
$$
\text{Expert}(x) \;=\; \big(\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\big)\, W_{\text{down}},
$$
where $x \in \mathbb{R}^{d_{\text{model}}}$ is the token's hidden vector (${d_{\text{model}} = 8192}$); $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ project up to the expert width (${d_{\text{ff}} = 6144}$); $\odot$ is elementwise product; $\text{SiLU}(z) = z\,\sigma(z)$ is the gating nonlinearity; and $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ projects back down. The router computes affinities ${s_i = (x W_R)_i}$ for expert $i$, where $W_R \in \mathbb{R}^{d_{\text{model}} \times N}$ and ${N = 128}$. The block output is

$$
y \;=\; x \;+\; \text{Expert}_{\text{shared}}(x) \;+\!\!\sum_{i \in \text{Top}_k(s + b)}\!\! g_i \cdot \text{Expert}_i(x),
$$

with ${k = 2}$. Here $g_i$ is the softmax gate over the *selected* experts' logits (so the two gates sum to 1), and $b \in \mathbb{R}^N$ is the per-expert bias from the balancer (next section) — note it's added to $s$ *only for the top-$k$ selection*, not for the gate weights $g_i$. The leading $x$ is the residual.

**Worked micro-example.** Suppose for some token the router's top two affinities are experts 17 and 88, with post-softmax gates ${g_{17} = 0.63}$ and ${g_{88} = 0.37}$. The layer computes ${y = x + \text{Shared}(x) + 0.63\,\text{Expert}_{17}(x) + 0.37\,\text{Expert}_{88}(x)}$. Experts 3, 52, and the other 123 contribute exactly zero for this token — they were never evaluated. That's the whole trick: three FFN evaluations (2 routed + 1 shared) out of 129 available.

**Why it works / when it fails.** Fine-grained experts let the model carve knowledge into narrow, reusable pieces, which the report's own probing confirms (experts in layers 10–40 fire up to 7× their baseline rate on a single domain). The shared expert stops the routed experts from each having to re-learn common features — and indeed its output norm runs about 1.6× the average routed expert in late layers, meaning it's doing a disproportionate share of the work. The failure mode is subtle: with a small $k$, a token that genuinely needs three or four kinds of processing gets clipped to two, and the shared expert becomes a bottleneck. Above layer 50 the report says specialization "is much weaker; experts there look more like a generic conditional FFN bank" — i.e. in the upper half of the network the fine-grained design buys less than it does lower down.

### Auxiliary-loss-free load balancing

**The problem.** Left alone, routers collapse. Early in training a few experts get slightly more traffic, so they learn slightly faster, so they attract even more traffic — a rich-get-richer spiral that ends with a handful of overworked experts and a hundred dead ones. The classic fix is an **auxiliary load-balancing loss** that penalizes imbalance, but it fights the main objective: you're adding a term whose only job is to make routing *worse* for the language-modeling loss in order to keep experts busy.

**Intuition.** Instead of punishing the model for imbalance, nudge the *thermostat*. Give each expert a bias that you turn *down* when it's overloaded and *up* when it's starved — like a maître d' who quietly steers the next few parties away from the slammed section and toward the empty one, without telling the kitchen to cook worse food. The bias steers *selection* only; it never corrupts the gradient of the actual loss.

**Mechanism.** Maintain a per-expert bias vector $b$. It's added to the affinities *for the top-$k$ decision*. At the end of every step, look at how much traffic each expert actually got, and adjust its bias in the opposite direction: overloaded experts get their bias lowered (so they're picked less next step), starved ones raised.

**The math.** The update, from [Wang et al. (2024b)](https://arxiv.org/abs/2408.15664), is
$$
b_i \;\leftarrow\; b_i \;-\; \gamma \cdot \left(\bar f_i - \tfrac{1}{N}\right),
$$
where $\bar f_i$ is the fraction of the step's tokens routed to expert $i$, ${1/N}$ is the ideal uniform share (${1/128}$), and ${\gamma = 10^{-3}}$ is the controller gain. If expert $i$ took more than its share (${\bar f_i > 1/N}$), the parenthetical is positive and $b_i$ drops; if it's starved, $b_i$ rises. Crucially, $b$ appears in the *argmax* that selects experts but **not** in the gate weights $g_i$ — so it changes *who* runs, never *how the loss flows through them*. Monolith-1 uses this bias controller alone and, unlike DeepSeek-V3, adds **no** complementary auxiliary loss and **no** z-loss.

**Worked micro-example.** Say expert 17 grabbed 2% of a step's tokens against an ideal of ${1/128 \approx 0.78\%}$. Then ${\bar f_{17} - 1/N = 0.0200 - 0.0078 = 0.0122}$, and its bias updates by ${-10^{-3} \times 0.0122 = -1.22 \times 10^{-5}}$. Tiny per step, but integrated over thousands of steps it firmly pushes an over-eager expert back toward its fair share. The report logs the **coefficient of variation** of per-expert load settling to **0.039 by step 10K** and staying below 0.05 for the rest of the run — a well-balanced bank.

**Why it works / when it fails.** The elegance is that balancing becomes a control-loop side-channel rather than a competing loss term, so it costs nothing in language-modeling quality. The risk is controller tuning: too large a $\gamma$ and the biases oscillate; too small and imbalance builds faster than the controller corrects. The report actually attributes its *avoidance* of DeepSeek-V3's late-training instabilities partly to running "a slightly more aggressive $\gamma$" than the cited value — which is a nice hypothesis, and also exactly the kind of claim that (as we'll see) is impossible to check against a real run here.

### Grouped-query attention

**The problem.** Attention's memory cost at inference is dominated by the **KV cache** — the keys and values you must keep for every past token so you can attend to them. In vanilla multi-head attention (MHA) with 64 heads, you store 64 keys and 64 values per token per layer. At a million-token context that cache is enormous, and it's read from HBM on *every* decode step, so it bottlenecks throughput.

**Intuition.** Do you really need 64 independent key/value streams? Multi-query attention (MQA) says "no, share *one* KV across all 64 query heads" — great for memory, but quality suffers because all heads are forced to look at the same keys. Grouped-query attention (GQA) is the compromise: split the 64 query heads into 8 groups, and let each group share one KV head. You keep most of MHA's expressiveness at one-eighth of its cache. The comparison matrix makes the trade explicit:

![Grouped-query attention as the middle seat between MHA and MQA: 64 query heads share 8 KV heads (8:1), shrinking the KV cache eightfold with near-MHA quality](/imgs/blogs/monolith-1-moe-4.webp)

**Mechanism.** Project the input into queries, keys, and values. There are 64 query heads but only 8 key/value heads; each KV head is *reused* by a group of 8 query heads. Compute scaled dot-product attention per head, with rotary position embeddings applied to Q and K, mask out the future, softmax, and take the value-weighted sum.

**The math.** For input ${x \in \mathbb{R}^{T \times d_{\text{model}}}}$ (a sequence of $T$ tokens):
$$
Q = x W_Q, \qquad K = x W_K, \qquad V = x W_V,
$$
$$
A_{ij} = \text{softmax}\!\left(\frac{Q_i K_j^\top}{\sqrt{d_h}} + M_{ij}\right), \qquad O_i = \sum_j A_{ij} V_j, \qquad y = O\,W_O,
$$
where $d_h = 128$ is the head dimension; the ${\sqrt{d_h}}$ divisor keeps the dot products from growing with dimension (which would push softmax into a near-one-hot regime and kill gradients); ${M_{ij}}$ is the causal mask (${-\infty}$ for future positions); and — the GQA part — $K$ and $V$ have only 8 heads while $Q$ has 64, so each KV head is broadcast across a group of 8 query heads. The kernel is FlashAttention-2 during training; inference adds a paged KV cache and switches to block-sparse attention beyond 256K tokens.

**Worked micro-example.** The KV cache per token is ${2 \times n_{\text{kv}} \times d_h \times L \times \text{bytes}}$. With BF16 (2 bytes), ${n_{\text{kv}} = 8}$ KV heads, ${d_h = 128}$, and ${L = 80}$ layers: ${2 \times 8 \times 128 \times 80 \times 2 \approx 328}$ KB per token. At the full ${2^{20}}$-token context that's roughly **340 GB** of cache. Under full MHA (64 KV heads) it would be 8× larger — about **2.7 TB** — which simply does not fit. GQA is not a nicety here; it's what makes a million-token context physically possible on the target hardware.

**Why it works / when it fails.** Grouping trades a little head diversity for a big cache reduction, and empirically the quality loss is small — GQA sits within a whisker of MHA on most tasks. It fails on the margin where those extra KV streams mattered: the report notes that beyond 256K tokens it switches to *block-sparse* attention (not full attention), and — foreshadowing the limitations — that the model's ability to *compare two distant facts* in a very long context degrades past 256K. GQA saves the memory; it doesn't buy you free long-range reasoning.

### RoPE and the two-stage YaRN stretch to a million tokens

**The problem.** The model pretrains at a **4,096-token** sequence length. But the deployed context is **1,048,576** — 256× longer. Rotary position embeddings (RoPE) encode position as rotations whose frequencies were tuned for 4K; feed the model a position it has never seen (say token 500,000) and those rotations spin into a regime the attention weights were never trained on, and quality craters. You need to *extend* the positional encoding without retraining from scratch.

**Intuition.** RoPE encodes each position as a set of clock hands rotating at different speeds — fast hands for fine local order, slow hands for coarse global position. When you stretch the context 256×, the fast hands whirl past every value they saw in training (extrapolation, which breaks) while the slow hands barely move. YaRN's insight: don't stretch all the hands equally. **Interpolate** the fast, high-frequency hands (so they stay inside their trained range) but let the slow, low-frequency hands **extrapolate** (they encode the genuinely new long-range positions). Do it "by parts," frequency band by frequency band. Then anneal into it in two stages rather than one violent jump.

**Mechanism.** After pretraining at RoPE base 10,000, raise the base to ${5 \times 10^{6}}$ and apply YaRN's NTK-aware-by-parts interpolation in **two stages** — first extend to 32K, then to 1M — training on a curriculum that mixes target-length documents with shorter ones in a **1:3 ratio** so the model doesn't forget how to handle short contexts. Total extension cost: about **2T additional tokens**, roughly 3% of pretraining.

**The math.** RoPE rotates the query/key components at position $m$ by angles ${m\theta_d}$, where ${\theta_d = \text{base}^{-2d/d_h}}$ sets the per-dimension frequency. Raising `base` from ${10^4}$ to ${5 \times 10^6}$ slows every frequency, buying headroom; YaRN then rescales *selectively* — dimensions whose wavelength is shorter than the context get interpolated (their effective position is divided down to stay in-range), while long-wavelength dimensions are left to extrapolate. The "by parts" qualifier means the interpolation factor is a smooth function of the dimension's frequency rather than a single global scale, which is what preserves local resolution while unlocking global reach.

**Worked micro-example.** Consider the fastest RoPE dimension. At 4K training length its clock hand sweeps through, say, its full ${2\pi}$ range many times — the model has seen every angle. Push a token to position 900,000 and, without interpolation, that hand has wound around 256× more than ever seen; the attention logits go out of distribution. YaRN divides that hand's effective position back down so its angle stays inside the trained range, while the slowest hand — which at 4K barely rotated at all — is allowed to keep turning, because *that's* the hand that now needs to distinguish position 900,000 from 4,000.

**Why it works / when it fails.** The two-stage curriculum plus the 1:3 short-document mix is what keeps the model from regressing on ordinary short prompts while it learns the long ones. Where it works: single-needle retrieval — the report claims above 99% at every probed depth out to 1M tokens. Where it fails, and the report is refreshingly blunt about this: tasks requiring the model to *compare two distant facts* degrade past 256K. "Distractor-free retrieval works; multi-fact reasoning at the full context length does not, and we did not find a fix in this release." A million-token context that can find a needle but can't reason across two of them is a real, honestly-stated limitation.

### Multi-token prediction and self-speculative decoding

**The problem.** Standard training teaches the model to predict only the *next* token. That's a weak signal for planning — the model never has to think about where a sentence is going. And at inference, next-token-only generation is inherently serial: one forward pass per token, which is slow.

**Intuition.** Two birds, one mechanism. During training, ask the model to predict not just token ${t+1}$ but also ${t+2}$, ${t+3}$, ${t+4}$ — this forces the hidden state to encode a little lookahead, a plan. At inference, reuse those same extra heads to **draft** several tokens at once, then let the main model **verify** the whole draft in a single forward pass, accepting the longest correct prefix. When the draft is right (which it usually is on predictable text), you get several tokens for the price of one forward pass.

**Mechanism.** Add three auxiliary prediction heads on top of the trunk, each a single transformer block sharing the embedding and unembedding matrices, predicting tokens ${t+2}$, ${t+3}$, ${t+4}$ alongside the standard next-token head. Weight the auxiliary losses lightly. At inference, run the heads to produce a 4-token draft, feed it back through the trunk once to check it, and keep the prefix that matches.

![Multi-token prediction: three auxiliary heads predict tokens t+2 through t+4 during pretraining, then draft ahead so the trunk verifies several tokens in one pass](/imgs/blogs/monolith-1-moe-5.webp)

**The math.** The pretraining loss sums four next-$n$ objectives:
$$
\mathcal{L} \;=\; \sum_{n=1}^{4} \lambda_n \cdot \mathbb{E}_t\big[-\log p_\theta(x_{t+n} \mid x_{\le t})\big],
$$
where ${p_\theta(x_{t+n} \mid x_{\le t})}$ is the probability the $n$-th head assigns to the true token $n$ positions ahead, and the weights are ${\lambda_1 = 1.0}$ (the real next-token objective) and ${\lambda_2 = \lambda_3 = \lambda_4 = 0.1}$ (the auxiliary lookahead heads, deliberately down-weighted so they help without dominating).

**Worked micro-example.** Suppose the trunk is generating "the capital of France is ___". The next-token head emits "Paris"; the ${t+2}$ head guesses ".", the ${t+3}$ head guesses "It", the ${t+4}$ head guesses "is". The draft "Paris . It is" is fed back through the trunk once; the trunk confirms "Paris ." and rejects the rest — so this single verification pass emitted two tokens instead of one. On highly predictable spans (code boilerplate, formulaic prose) the accepted prefix is longer, which is why the report measures **2.1× decode speedup on natural language and 2.7× on code**.

**Why it works / when it fails.** As a training signal, multi-token prediction improves the trunk's representations (following [Gloeckle et al., 2024](https://arxiv.org/abs/2404.19737)); as an inference trick, self-speculation is *free* because the draft model is the model's own heads — no separate draft network to train or serve. It fails when the draft is usually wrong: on high-entropy, genuinely unpredictable text the trunk rejects most of the draft and you're back to (slightly worse than) one-token-per-pass, having paid for the wasted draft. The speedup is real but content-dependent.

### The compute budget, from first principles

This section isn't a new technique — it's a worked example of reading a training report's numbers for internal consistency, which is a skill in itself and, for this particular paper, the *only* kind of verification available to an outside reader.

Start from the batch. Global batch size is 8,192 sequences of length 4,096, so **tokens per step** ${= 8192 \times 4096 = 3.355 \times 10^{7}}$. Over ${1.79 \times 10^{6}}$ steps that's ${3.355 \times 10^{7} \times 1.79 \times 10^{6} \approx 6.0 \times 10^{13}}$ — exactly the 60T-token budget. Feed that into ${6 N_{\text{active}} D}$ and you recover ${1.78 \times 10^{25}}$ FLOP.

Now turn FLOPs into wall-clock. Each Ascend 910C has a published BF16 peak of ${7.52 \times 10^{14}}$ FLOP/s; there are 12,288 of them; the realized model-FLOPs-utilization (MFU) is 32%. So

$$
\text{days} \;=\; \frac{1.78 \times 10^{25}}{12{,}288 \times 7.52 \times 10^{14} \times 0.32 \times 86{,}400} \;\approx\; 69.7,
$$

which rounds to the "70 days" the report logs for pretraining. Add 7 days of context extension and 14 days of post-training and you land at roughly **91 days** and **~27M Ascend 910C-hours** total.

The numbers are mutually consistent — batch × steps × length gives the token count, tokens × ${6 N_{\text{active}}}$ gives the FLOPs, FLOPs ÷ (chips × peak × MFU) gives the days. That consistency is a real (and deliberate) property of the report. It is also, as Appendix E will remind us, *the point*: the numbers were chosen to be consistent, which is not the same as being measured.

One more figure worth internalizing: because the 910C has no FP8, everything runs in BF16, and the report states this costs roughly **2.5× the chip-hours** an FP8-capable cluster of the same FLOP budget would need. That's the price of hardware sovereignty, stated plainly.

### Parallelism on 12,288 Ascend NPUs

**The problem.** A 1.57T-parameter model does not fit on one device, or one node, or one rack. You must split it across 12,288 accelerators along multiple axes at once, and keep them all busy despite the communication that splitting forces.

**Intuition.** Four cuts, each along a different grain. **Pipeline parallelism** slices the 80 layers into consecutive stages on different devices (like an assembly line). **Tensor parallelism** splits each big matrix multiply across devices within a stage (many hands on one matrix). **Expert parallelism** scatters the 128 experts across devices (each device hosts a few experts, and tokens are shuffled to wherever their chosen experts live). **Data parallelism** replicates the whole thing across independent token streams. The art is choosing which cut runs over which network link so the fast links carry the chatty traffic.

**Mechanism.** Monolith-1 uses **16-way pipeline × 8-way tensor × 96-way expert × 1-way data = 12,288 ranks.** Tensor-parallel collectives (which are bandwidth-hungry, reducing activations every layer) run *inside* a single 384-NPU super-pod where Huawei's UB bus is fast; expert all-to-all traffic (shuffling tokens to their experts) runs *across* pods over 400 Gbps RoCE. The DualPipe schedule from DeepSeek-V3 is used so that expert all-to-all overlaps with expert-FFN GEMMs — but since DualPipe was written for InfiniBand, it had to be ported onto Huawei's HCCL collective library. Activation recomputation is on for the MoE block, off for attention; the residual stream uses stochastic rounding to prevent silent BF16 underflow in late layers.

**Why it works / when it fails.** Matching the *communication pattern* to the *network topology* — keeping tensor-parallel reductions on the fast intra-pod bus and expert shuffles on the slower inter-pod fabric — is what lets a 32% MFU be achievable at all on this hardware. The two logged incidents show the failure surface: on day 11, a corpus shard with unusually dense code comments created a "hot routing path" and a 5% expert imbalance that didn't self-correct, fixed by *re-shuffling the data order* rather than touching the optimizer; on day 22, a datacenter power event took out a full super-pod and cost ~4 hours to the last checkpoint. These are the mundane, real-sounding operational details that make the report read like an engineering log — and, per Appendix E, are also inferred rather than recorded.

### Post-training: SFT → DPO → RLVR → constitutional

**The problem.** A pretrained base model completes text; it doesn't follow instructions, prefer helpful answers, reason reliably to a verified conclusion, or refuse harmful requests. Post-training installs those behaviors — but each stage can undo the last, so order and dosage matter.

**Intuition.** Four passes, from broad to sharp. **SFT** teaches the format of a good answer by imitation. **DPO** teaches *preference* — this answer is better than that one — without a separate reward model. **RLVR** teaches *correctness* on problems where the answer can be checked by a machine (math, code), rewarding verified success. **Constitutional refinement** repairs the safety behaviors that reasoning-focused RL tends to erode. The timeline shows how short this tail is relative to pretraining:

![One training run, five stages: pretraining dominates the wall clock while context extension and the three-part alignment pipeline are comparatively short tails](/imgs/blogs/monolith-1-moe-6.webp)

**Mechanism and math.** **SFT**: 4.2M instruction–response pairs, 3 epochs, peak LR ${5 \times 10^{-6}}$, loss masked on prompt tokens — and *deliberately light*, because "aggressive SFT degraded RLVR exploration in pilot runs." **DPO** ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)): 1.1M preference pairs (720K helpfulness + 380K safety), the DPO objective with ${\beta = 0.1}$, plus a length-debiasing regularizer ([Park et al., 2024](https://arxiv.org/abs/2403.19159)) — because an unregularized DPO run produced a "38% increase in median response length with no measurable quality gain." **RLVR** (following [DeepSeek-R1](https://arxiv.org/abs/2501.12948)): a rule-based reward of ${+1}$ for a verifiably correct final answer (math) or a passing test suite (code) and 0 otherwise, plus a small format reward; optimized with **GRPO** at group size 16, KL coefficient 0.001 against the SFT initialization, LR ${1 \times 10^{-6}}$, over 1.8M competition-math + 600K competitive-programming + 240K Lean/Coq + 80K logic problems, consuming ~3M NPU-hours — the majority of the post-training budget. **Constitutional refinement** ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)): a model-written critique-and-revision loop on 320K adversarial prompts, folded back through a short DPO pass.

**Why it works / when it fails.** The pipeline's logic is that verifiable rewards (RLVR) are the cleanest signal we have for reasoning — you can't reward-hack a passing unit test — but RL optimizing hard for math/code correctness drifts away from safety, so a constitutional pass at the end pulls it back. The stated failure surfaces are honest: keeping SFT light to preserve RL exploration, length-debiasing DPO to stop verbosity inflation, and (from the limitations) that the constitutional pass still leaves six unmitigated model-level jailbreaks. This is a textbook-modern alignment stack; the interesting engineering is in the *dosing*, which the report describes but — again — cannot demonstrate here.

## Experiments & results

Here is where the analysis has to change gears. The headline table:

![Table 2 from the Monolith-1 report: headline benchmark results with Monolith-1 leading GPT-5.4, Claude Opus 4.6, Gemini 3.1 Pro, and Kimi K2.6 on all four benchmarks, with footnotes on the HLE protocol and the MMLU-Pro extractor patch](/imgs/blogs/monolith-1-moe-fig2.webp)

Taken at face value, Monolith-1 leads all four reasoning benchmarks: **99.4% on Humanity's Last Exam, 100.0% on AIME 2025, 95.9% on GPQA Diamond, 96.2% on MMLU-Pro**, against GPT-5.4, Claude Opus 4.6, Gemini 3.1 Pro, and Kimi K2.6. Three things must be said before any of those numbers is repeated as a fact.

**First, the HLE number is produced under a non-standard protocol.** Read footnote † in the table, in the authors' own words: the 99.4% "uses code execution and best-of-32 with an off-policy LLM judge." Appendix B spells it out — the model generates **32 candidate answers** across a temperature sweep, then **the same checkpoint acts as judge** to pick the final answer, with a sandboxed Python kernel available. The competitor numbers (~40%) are single-attempt scores from their model cards. Comparing a best-of-32-with-self-judging pipeline against single-shot baselines is not a like-for-like comparison; it measures a *deployed system with a selection harness*, not the model's intrinsic accuracy. A 99.4%-vs-40% gap produced this way tells you almost nothing about relative capability.

**Second, MMLU-Pro's 96.2 depends on a custom answer extractor.** Footnote ‡: with the unmodified upstream `lm-evaluation-harness` extractor, the same checkpoint scores **95.3**. A ~1-point swing from an answer-parsing patch is not damning on its own — extraction genuinely is fiddly — but it means the headline is the *favorable* configuration, and it should be cited as "95.3–96.2 depending on the extractor," not 96.2 flat.

**Third — and this is the one that reframes everything — the numbers are not an independent measurement.** Section 6 says all Monolith-1 numbers were "produced with EleutherAI's lm-evaluation-harness ... run on the released checkpoint." But **Appendix E** states that "all four benchmark scores reported for both Monolith-1 and the competitor systems are taken directly from the codebase" — i.e. from the Basalt corporate website, not from a harness run an outsider could reproduce. Those two statements are in tension, and the honest resolution is the skeptical one: **treat the benchmark scores as claims sourced from a marketing artifact, not as reproduced measurements.**

The contamination check (Section 6.1) reports an aggregate 13-gram hit rate of 0.03% (GPQA 0.00%, AIME 0.00%, MMLU-Pro 0.05%, HLE 0.00%), with the headline unchanged after removing the single flagged MMLU-Pro item. As a *methodology* that's the right thing to do; as *evidence* it inherits the same problem — a contamination number computed over "our pretraining shards" is only as real as the run that produced those shards.

**What's load-bearing that might not transfer.** Even setting the epistemics aside, the strongest results lean on choices that wouldn't survive a change of setting: the 60T-token over-training regime (far above Chinchilla-optimal, justified only by amortizing over inference), the specific tokenizer's Chinese–English balance, the MMLU-Pro extractor patch, and — for HLE especially — the best-of-32 self-judging harness. Strip the harness and the single most dramatic number moves the most.

## Critique

**What's genuinely strong.** As a piece of *technical writing*, Monolith-1 is a coherent, well-organized synthesis of the 2026 frontier-MoE playbook, and it does several things I wish more reports did. It has a real **failure section** (Section 8) that leads with "we did not solve several problems we set out to solve" and then enumerates them — MLA beat GQA on validation loss but lost long-context retrieval after YaRN, so they shipped GQA and *say they don't know why*; the model is text-only; agentic robustness trails the closed frontier; low-resource languages lag 8–15 points; six jailbreaks remain unpatched. It states operational incidents with dates. It gives the FP8-versus-BF16 chip-hour tax honestly. And — most unusually — **Appendix E explicitly separates what is sourced from what is inferred.** That candor is exactly what lets a careful reader calibrate.

**What's weak, unfalsifiable, or cherry-picked.** The benchmark results, as discussed, are not independent measurements and the single headline number (HLE 99.4%) is produced by a self-judging harness that inflates it beyond any comparable baseline. More fundamentally, per Appendix E, the *entire technical apparatus* — architecture, optimizer, parallelism, YaRN recipe, MTP setup, contamination numbers, expert-specialization analysis, the two training incidents, the loss-curve story — is **inferred by the authors to be internally consistent with a set of headline facts pulled from a website**, not recorded from a run. This is not a criticism of the arithmetic (which checks out) but of what the arithmetic *demonstrates*: internal consistency, not physical reality. A report can be perfectly self-consistent and describe a model that was never trained the way it says, or trained at all.

**What ablation or baseline is missing.** The load-bearing empirical claim — contribution (2), that fine-grained MoE "scales further than current public results indicate ... without observable destabilization" — has no ablation behind it. There's no scaling curve, no comparison against a matched dense model, no expert-count sweep, no learning-curve plot. The MLA-vs-GQA comparison that *would* be the paper's most interesting ablation is mentioned only as prose ("MLA was slightly better on validation loss but produced systematically worse long-context retrieval ... We do not know why"). For a report whose second contribution is an empirical scaling claim, the absence of a single figure of measured loss is telling.

**What would change my mind.** I would move from "plausible reconstruction" to "credible frontier model" on the following, in order of weight: (1) an **independent third party reproducing even one benchmark** by running the released checkpoint through the *unmodified* `lm-evaluation-harness` — especially HLE under a standard single-attempt protocol, where I'd expect a number far below 99.4%; (2) a **released training loss curve** with the described day-4 and day-11 incidents actually visible in it; (3) a **matched-compute dense baseline** substantiating the "scales further" claim; and (4) any measurement — even one — that Appendix E lists as *inferred* being shown to match an artifact from a real run. Absent those, the correct posture is to learn the (genuinely well-explained) techniques from this document and to hold the capability claims at arm's length.

## What I'd build with this

These are my extrapolations, not the paper's claims — things I'd want to try or verify if I had the checkpoint.

1. **Run the honest eval.** The very first thing: pull the released weights and run HLE, GPQA, AIME, and MMLU-Pro through the pinned upstream harness under single-attempt, temperature-0 protocols, with no self-judging and no extractor patch. Publish the delta from the headline table. That one experiment settles most of the epistemic questions.
2. **Probe the 256K cliff.** The report admits multi-fact reasoning degrades past 256K while single-needle retrieval survives to 1M. I'd build a graded multi-hop long-context suite (2-fact, 3-fact, k-fact at increasing separations) to map exactly where the cliff is and whether it tracks the block-sparse-attention switch at 256K.
3. **Re-run the MLA-vs-GQA question they punted on.** They shipped GQA because MLA "produced systematically worse long-context retrieval after YaRN" and said they don't know why. I'd test whether MLA's latent compression interacts badly with YaRN's frequency-selective interpolation specifically — a genuinely open and interesting question that the report leaves on the table.
4. **Quantify the shared expert.** They observe the shared expert's output norm running 1.6× the mean routed expert in late layers. I'd ablate the shared expert entirely and measure where the loss goes — it would tell you how much of the "MoE" is really a dense FFN in disguise, especially above layer 50 where specialization is weak.
5. **Stress the self-speculation on adversarial text.** The 2.1×/2.7× speedups are on natural language and code. I'd measure the decode speedup (or slowdown) on high-entropy inputs where the MTP draft is usually rejected, to bound the worst case a latency-SLA deployment would actually see.

## References

- **The paper.** *Monolith-1: A 1.6T Open Mixture-of-Experts Foundation Model for Reasoning at Scale.* The Basalt Research Team, Basalt PBC, April 2026. [basaltlabs.org/Monolith-1.0-2606.pdf](https://basaltlabs.org/Monolith-1.0-2606.pdf). (Note Appendix E, which separates sourced facts from inferred detail — read it before citing any number.)
- **Core lineage.** DeepSeekMoE ([Dai et al., 2024](https://arxiv.org/abs/2401.06066)); auxiliary-loss-free balancing ([Wang et al., 2024](https://arxiv.org/abs/2408.15664)); DeepSeek-V3 ([Liu et al., 2024](https://arxiv.org/abs/2412.19437)); GQA ([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)); YaRN ([Peng et al., 2024](https://arxiv.org/abs/2309.00071)); multi-token prediction ([Gloeckle et al., 2024](https://arxiv.org/abs/2404.19737)); DPO ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)); DeepSeek-R1 / RLVR ([Guo et al., 2025](https://arxiv.org/abs/2501.12948)).
- **Sibling posts on this blog.** [DeepSeek-V4: the million-token-context MoE](/blog/paper-reading/large-language-model/deepseek-v4-million-token-context-moe) — the closest architectural cousin; [DeepSeek-R1: incentivizing reasoning with RL](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) — the RLVR recipe Monolith-1's post-training follows; [RoFormer / RoPE](/blog/paper-reading/large-language-model/roformer-enhanced-transformer-with-rotary-position-embedding) — the positional encoding YaRN extends; [Kimi K2.6](/blog/paper-reading/large-language-model/kimi-k2-6) — one of the systems in Monolith-1's comparison table.
