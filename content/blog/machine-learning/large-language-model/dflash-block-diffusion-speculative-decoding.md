---
title: "DFlash: Block Diffusion for Flash Speculative Decoding — Turning the Drafter Parallel"
date: "2026-06-02"
publishDate: "2026-06-02"
description: "A deep dive into DFlash, a speculative decoding framework that swaps the serial autoregressive drafter for a lightweight block-diffusion drafter conditioned on target hidden features — reaching 6× lossless acceleration and 2.5× over EAGLE-3."
tags: ["speculative-decoding", "diffusion-language-models", "llm-inference", "kv-cache", "eagle-3", "sglang", "vllm", "block-diffusion", "drafting", "inference-optimization"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
aiGenerated: true
---

Every speculative decoding paper for the last two years has optimized the same half of the loop: verification. Tree attention, dynamic draft trees, better acceptance criteria — all of it is about making the *target* model verify more tokens per forward pass. Almost nobody touched the *drafter*. And that is exactly where the latency hides.

The reason is structural. State-of-the-art methods like EAGLE-3 draft tokens **autoregressively** — one token per forward pass, just like the target model they are trying to accelerate. To keep that serial drafting cheap, the drafter is squeezed down to a single transformer layer. A one-layer model proposing tokens one at a time, with errors accumulating down the chain: that is why the whole field has been stuck at a 2–3× speedup ceiling for so long. We made verification embarrassingly parallel and left drafting embarrassingly serial.

[DFlash](https://arxiv.org/abs/2602.06036) (Chen, Liang & Liu, UC San Diego, ICML 2026) attacks the other half. It replaces the autoregressive drafter with a **lightweight block-diffusion model** that emits an entire block of draft tokens in a *single* forward pass, then conditions that drafter on hidden features pulled straight out of the target model. The result is over **6× lossless acceleration** on Qwen3-8B and up to **2.5× faster than EAGLE-3** across math, code, and chat benchmarks. This post is a tour of how it works, which techniques are worth stealing for your own systems, and where it breaks.

I want to be precise about what kind of result this is, because "6×" gets thrown around loosely in inference papers. This is a *lossless* speedup measured in end-to-end wall-clock decoding latency against a real autoregressive baseline on the same hardware, not a throughput number inflated by batching or a quality-traded approximation. The output distribution is provably identical to the target model's. That combination — large speedup, zero quality cost, drop-in into existing serving stacks — is rare, and it is worth understanding exactly how DFlash earns it.

## Why the drafter is the bottleneck nobody optimizes

Let us start by naming the mismatch between what most engineers assume about speculative decoding and what actually governs its speedup.

| Common assumption | The naive view | The reality DFlash exploits |
|---|---|---|
| "The drafter is small, so drafting is basically free." | Drafting cost ≈ 0; speedup is all about acceptance. | Serial drafting cost grows linearly with the speculation budget γ and often *dominates* the cycle. |
| "Speedup is bottlenecked by how many tokens the target can verify." | Bigger draft trees → more speedup. | Verification is already parallel; the serial drafter is the gate. EAGLE-3 with a 60-node tree gets *slower* at high concurrency. |
| "A bigger drafter would draft better tokens." | Just scale the draft model. | Autoregressive drafters must stay shallow (1 layer) or drafting latency explodes — so they *cannot* be scaled. |
| "Diffusion LLMs are too low-quality to be useful." | Diffusion can't match autoregressive quality. | True for end-to-end generation — but drafting only needs to be *good enough to be verified*, and verification makes the output lossless. |
| "Speculative decoding is a solved 2–3× trick." | We've hit the ceiling. | The ceiling was a property of *serial drafting*, not of speculation itself. Parallel drafting moves the frontier to 6×. |

The thread running through every row is the same: the community optimized the cheap-looking thing (verification) and ignored the expensive-looking-but-actually-fixable thing (serial drafting). DFlash's whole contribution is to take the drafter seriously.

It helps to remember why drafting *feels* free. The drafter is small — a single transformer layer in EAGLE-3 — so each draft forward pass is cheap relative to a target forward pass. The mistake is reasoning about one pass instead of the cycle. To speculate γ tokens you run γ draft passes, and even cheap passes are *serial* and *memory-bound*: each one loads the drafter's weights from HBM, does a thin slice of compute, and stalls. Eight serial memory-bound passes can easily cost more wall-clock than one fat parallel pass through a five-times-bigger model. The drafter is not free; it is the only serial component left in a loop you spent years making parallel.

> The drafter is not free. It is the only serial component left in a loop you spent years making parallel.

### A short history of why diffusion LLMs were "not good enough"

To see why DFlash's framing is clever, it helps to know the arc of diffusion language models. LLaDA (Nie et al., 2025) was the first to scale diffusion LLMs to billions of parameters and reach quality comparable to Llama-3.1-8B — a real milestone, but fully-parallel diffusion came with two structural problems: fixed-length generation and no efficient KV-cache support, both of which hurt practical inference. Block diffusion models (Arriola et al., 2025) fixed those by denoising sequences block-by-block, blending parallelism with autoregressive structure. Follow-ups like Fast-dLLM v2 (Wu et al., 2025) and SDAR (Cheng et al., 2025) adapted pretrained autoregressive LLMs into block-diffusion variants, preserving quality on specific tasks.

But the verdict stuck: standalone diffusion LLMs still underperform state-of-the-art autoregressive models on general benchmarks, and worse, they need *many* denoising steps to reach acceptable quality — which destroys their headline parallelism advantage, because each denoising step is another forward pass. A diffusion model that needs 10 denoising steps to write one block is not parallel in any useful sense; it is autoregression wearing a costume. This is the trap d3llm (Qian et al., 2026) and others tried to escape with distillation.

DFlash's insight sidesteps the whole debate. It does not need the diffusion model to be high-quality, and it does not need many denoising steps. It uses **one** denoising step (a single forward pass producing the block) and lets verification clean up the mistakes. The diffusion drafter can be aggressive about minimizing denoising steps — maximizing parallelism — precisely because it is not the final word on quality. The thing that made diffusion LLMs impractical standalone (low quality at few steps) is exactly the thing that does not matter when they draft.

## The mental model

![DFlash speculative decoding loop: a diffusion drafter proposes a 16-token block in one pass while the target verifies in parallel](/imgs/blogs/dflash-block-diffusion-speculative-decoding-1.png)

The diagram above is the mental model, and the rest of this article is a tour of it. A prompt enters the target model, which runs a standard prefill pass and produces the first token. During that pass DFlash extracts **context features** from a handful of the target's hidden layers. Those features are injected into a small **diffusion drafter**, which fills in a block of 16 masked positions *all at once* in a single forward pass. The target then verifies the whole 16-token block in parallel, accepts the longest correct prefix (on average around 7 tokens), and emits one bonus token that seeds the next block. Loop.

Two things make this fast that EAGLE-style drafting cannot match. First, the draft block is produced in **one** forward pass instead of 16 sequential ones. Second, because drafting cost no longer scales with block size, DFlash can afford a *five-layer* drafter — deep enough to produce genuinely good tokens — at lower latency than EAGLE-3's one-layer drafter running eight serial steps.

If you have read about [speculative decoding](/blog/machine-learning/large-language-model/speculative-decoding) and [diffusion language models](/blog/machine-learning/open-source-library/dllm-diffusion-language-models-deep-dive) before, DFlash is the synthesis: it confines the diffusion model to the one job it is uniquely good at — fast parallel proposal — and lets the autoregressive target keep the one job it is uniquely good at — high-quality verification. Neither paradigm has to win the whole pipeline. They split the work.

The naming is worth fixing in your head before we go deeper, because every section refers back to these three pieces. The **target** is the big frozen model you actually want to serve (Qwen3-8B, Llama-3.1-8B, and so on). The **drafter** is the small block-diffusion model DFlash trains — five transformer layers, sharing the target's embedding and LM head. The **block** is the unit of speculation: 16 masked positions the drafter fills in parallel and the target verifies in one pass. Hold those three and the rest is mechanics.

## 1. The speedup arithmetic

**Senior rule of thumb: before optimizing anything in a speculative loop, write down the latency equation and find which term you are actually paying for.**

![The speedup arithmetic: per-token latency falls only by raising acceptance length tau or cutting drafting cost](/imgs/blogs/dflash-block-diffusion-speculative-decoding-2.png)

Following the standard formulation (Sadhukhan et al., 2025), the average per-token latency of a speculative decoder is:

$$
L = \frac{T_{\text{draft}} + T_{\text{verify}}}{\tau}
$$

where $T_{\text{draft}}$ is the time spent generating draft tokens in a cycle, $T_{\text{verify}}$ is the cost of the parallel verification pass, and $\tau \in [1, \gamma + 1]$ is the expected number of accepted tokens per cycle (including the bonus token the target always produces). If $L_{\text{target}}$ is the plain autoregressive per-token latency, the speedup is simply:

$$
\eta = \frac{L_{\text{target}}}{L}
$$

This tiny equation tells you everything. There are exactly two levers. You can **raise $\tau$** (accept more tokens per cycle) or you can **cut $T_{\text{draft}}$** (make drafting cheaper). $T_{\text{verify}}$ is roughly one target forward pass over the block, and it is already parallel, so there is not much to win there at low concurrency.

The trap that EAGLE-style methods fall into is that their two levers are *coupled*. To raise $\tau$ they need a better drafter, but a better autoregressive drafter means more layers or more draft steps, which directly inflates $T_{\text{draft}}$. They are pushing on a lever that pushes back. DFlash's whole structural advantage is that diffusion drafting **decouples** these levers: it can make the drafter deeper (raising $\tau$) without making drafting slower (holding $T_{\text{draft}}$ flat). We will see exactly why in the next section.

### A worked example

Numbers make the levers concrete. Suppose the target's per-token latency is $L_{\text{target}} = 20$ ms. Take an EAGLE-3-style drafter where one draft pass costs $t_{\text{step}} = 2$ ms, the verification pass costs $T_{\text{verify}} = 6$ ms, you draft γ = 7 tokens, and you accept τ = 3.5 on average. Then:

$$
L_{\text{EAGLE}} = \frac{7 \times 2 + 6}{3.5} = \frac{20}{3.5} \approx 5.7\ \text{ms}, \qquad \eta \approx 3.5\times
$$

Now DFlash. The five-layer diffusion drafter is bigger per pass, say $t_{\text{parallel}} = 4$ ms, but it drafts all 16 tokens in *that one pass*. Verification grows a bit because the block is larger, $T_{\text{verify}} = 7$ ms, and acceptance jumps to τ = 6.5 because the deeper, KV-conditioned drafter produces better tokens. Then:

$$
L_{\text{DFlash}} = \frac{4 + 7}{6.5} = \frac{11}{6.5} \approx 1.7\ \text{ms}, \qquad \eta \approx 11.8\times
$$

The illustrative arithmetic overshoots the paper's measured ~5–6× (real verification and overheads are messier), but it shows the mechanism cleanly: DFlash *both* shrinks the numerator (one draft pass instead of seven) *and* grows the denominator (τ from 3.5 to 6.5). EAGLE can only do one at a time. When a method moves both terms of a ratio in your favor at once, you get a multiplicative, not additive, win.

### Second-order consequence: the high-concurrency cliff

Here is the non-obvious bit. At batch size 1, $T_{\text{verify}}$ is small and the equation is dominated by $T_{\text{draft}}/\tau$. But as concurrency rises, the target's verification pass becomes compute-bound — verifying a 16-token block for 32 concurrent requests is real FLOPs — and $T_{\text{verify}}$ starts to dominate. This is why every speculative method, DFlash included, sees its speedup *taper* at high batch sizes. EAGLE-3 with a 60-node tree actually drops *below* 1.0× at concurrency 32 on some tasks (it verifies 60 nodes per request and most get rejected). DFlash degrades more gracefully because its block of 16 is a tighter verification budget, but the cliff is real and we will return to it when we talk about block-size scheduling.

The intuition to keep is that speculative decoding spends *extra compute* (drafting plus verifying tokens you might throw away) to buy *lower latency*. At low concurrency the GPU has spare compute, so that trade is free money. At high concurrency the GPU is already saturated with useful work, so the extra speculative compute competes with real requests, and the trade gets worse. Speculation is a latency optimization that quietly becomes a throughput tax as you fill the machine.

This framing also tells you *which* number to optimize for. If you run an interactive product — a coding assistant, a chat UI, a reasoning agent where a human is waiting on the stream — you care about per-request latency and you run at low effective concurrency, so the full DFlash speedup is exactly the metric that matters. If you run a batch pipeline — offline evaluation, bulk document processing, synthetic data generation — you care about aggregate throughput and you run the GPU hot, so you should weigh the high-concurrency 2–3× against the option of simply batching harder without speculation. The same checkpoint serves both, but the *value* of the speedup is completely different between the two regimes, and conflating them is the most common way teams either over- or under-invest in speculative decoding. Decide which regime you are in before you read any speedup number, including the ones in this post.

## 2. Why diffusion drafting changes the design space

**Senior rule of thumb: a cost that is linear in your budget caps your budget; a cost that is flat in your budget removes the cap.**

![Autoregressive vs diffusion drafting cost: diffusion flattens cost in the speculation budget so a deeper drafter is free](/imgs/blogs/dflash-block-diffusion-speculative-decoding-3.png)

Write down the two drafting costs side by side. An **autoregressive drafter** generates γ tokens sequentially, so:

$$
T_{\text{draft}}^{\text{AR}} = \gamma \cdot t_{\text{step}}
$$

where $t_{\text{step}}$ is the latency of one forward pass through the drafter. Drafting cost grows *linearly* with the speculation budget γ. To keep that manageable, EAGLE-3 constrains the drafter to a single transformer layer. And here is the cruel part: increasing γ increases the cost, but acceptance length τ saturates quickly because a one-layer model simply does not have the capacity to predict far into the future. You pay more and get less.

A **diffusion drafter** generates all γ tokens in parallel within a single forward pass:

$$
T_{\text{draft}}^{\text{diff}} = t_{\text{parallel}}
$$

Modern GPUs execute that parallel block far more efficiently than γ sequential passes, so $t_{\text{parallel}} \ll \gamma \cdot t_{\text{step}}$ for models of comparable size. Critically, $T_{\text{draft}}$ is now *largely insensitive to γ*. The cost is flat.

This is the structural pivot of the whole paper. Once drafting cost stops scaling with the number of tokens you draft, you can spend your latency budget on **depth** instead. The paper's measurements are unambiguous: a five-layer DFlash drafter generating 16 tokens has both *lower latency* and *higher acceptance length* than a one-layer EAGLE-3 drafter generating 8 tokens. DFlash is strictly on a better point of the draft-quality-vs-cost Pareto frontier.

| | EAGLE-3 (autoregressive) | DFlash (block diffusion) |
|---|---|---|
| Drafting cost | $\gamma \cdot t_{\text{step}}$ (linear in γ) | $t_{\text{parallel}}$ (flat in γ) |
| Forward passes per block | γ (one per token) | 1 |
| Affordable drafter depth | ~1 layer | 5 layers (8 for MoE coder) |
| Acceptance length τ behavior | saturates with capacity | scales with depth |
| GPU utilization while drafting | poor (serial, memory-bound) | high (one parallel matmul) |

### The roofline view: why one fat pass beats many thin ones

If the flat-cost claim feels like a free lunch, the roofline model explains why it is not. A single token's forward pass through a small drafter is **memory-bound**: the GPU spends almost all its time streaming the drafter's weights from HBM, and the actual matrix multiplies finish long before the next batch of weights arrives. Arithmetic intensity (FLOPs per byte loaded) is low, so you are nowhere near the GPU's compute roofline — you are pinned to its memory-bandwidth roofline. Running γ such passes serially pays that memory cost γ times over.

A diffusion drafter loads its weights *once* and applies them to all 16 positions in one batched matmul. Arithmetic intensity goes up roughly 16×, pushing the operation off the memory roof and toward the compute roof, where the hardware is actually fast. This is the same reason prefill is faster per-token than decode in any LLM: batching positions amortizes the weight load. DFlash turns drafting from a decode-shaped (memory-bound, serial) workload into a prefill-shaped (compute-bound, parallel) one. That is the entire performance story in one sentence.

### Second-order consequence: depth is now a tunable

Because depth is decoupled from latency, "how many draft layers" becomes a genuine deployment knob rather than a hard constraint. The paper's ablation (Table 6) shows a 3-layer drafter, a 5-layer drafter, and an 8-layer drafter:

| Drafter depth | Math500 (speedup / τ) | HumanEval (speedup / τ) | MT-Bench (speedup / τ) |
|---|---|---|---|
| 3 layers | 4.69× / 5.64 | 3.90× / 4.61 | 2.38× / 3.18 |
| 5 layers | 4.71× / 5.99 | 3.96× / 4.94 | 2.35× / 3.37 |
| 8 layers | 4.64× / 6.33 | 3.96× / 5.29 | 2.23× / 3.50 |

The 8-layer model has the longest acceptance length (τ = 6.33 on Math500), but the 5-layer model wins on *end-to-end speedup* because it strikes a better balance between draft quality and the small-but-nonzero per-pass cost. Notice the shape: τ rises monotonically with depth, but speedup peaks in the middle and then falls as the deeper drafter's own latency eats into the win. The sweet spot is 5 layers for dense models — but the point is that there *is* a sweet spot to tune, which is something a serial drafter never gives you.

## 3. The target knows best: context features via KV injection

**Senior rule of thumb: a small model conditioned on a big model's internal state will beat a small model reasoning from scratch — but only if the conditioning survives all the way through the small model's depth.**

![Input fusion vs KV injection: injecting target features into every draft layer avoids the depth dilution of input-only fusion](/imgs/blogs/dflash-block-diffusion-speculative-decoding-4.png)

This is the technique most worth stealing, so we will spend real time on it.

The naive way to build a diffusion drafter is to train a small diffusion model and run it as a drafter. The paper tried exactly this (a five-layer block diffusion drafter with **no** conditioning from the target) and got modest results — speedups around 2–3×, no better than autoregressive methods (Table 10). The problem is that without access to the target's internal representations, the diffusion drafter has to predict future tokens *from scratch*, and a tiny model is bad at that.

The fix comes from an observation by Samragh et al. (2025): the hidden states of a large autoregressive model **implicitly encode information about multiple future tokens**. The target already "knows" roughly what it is going to say next — that knowledge is sitting in its activations. So instead of asking a tiny drafter to re-derive the future, DFlash hands it the target's hidden features and asks it to *decode* them in parallel. The drafter becomes, in effect, a **diffusion adapter** on top of the target's representation space.

Why are the hidden states richer than the logits? The logits are a projection down to vocabulary space — a lossy summary of "what comes next" collapsed to one distribution over the next token. The hidden states upstream of that projection still carry the un-collapsed information: syntactic plans, the entity being described, the structure of the sentence three tokens ahead. A drafter conditioned on logits alone sees only the next-token shadow; a drafter conditioned on hidden states sees the geometry that produced it. That is why feature-level conditioning (the EAGLE insight) beats logit-level drafting, and why DFlash leans on it even harder.

There is a deeper reason this works for *block* prediction specifically. Samragh et al. (2025) showed that an autoregressive model's hidden state at a given position carries usable signal not just about the immediate next token but about several tokens ahead — the model has, in effect, already "decided" the next few words and is committing them to its residual stream one at a time. An autoregressive drafter cannot exploit this efficiently, because it still emits those tokens serially. A block-diffusion drafter is the natural reader of that multi-token signal: it predicts all the positions the target has implicitly committed to, in one parallel shot. The architecture of the drafter (parallel block prediction) is matched to the structure of the information it is given (multi-token-ahead features). That match is not a coincidence — it is *why* the diffusion-drafter-plus-KV-injection combination beats either piece alone. Choosing the right hidden layers to read from matters for the same reason: shallow layers carry token-level and positional signal, deep layers carry semantic and planning signal, and the five uniformly-spaced taps give the drafter both ends of that spectrum.

### Why injection matters more than what you inject

EAGLE-3 also uses target hidden features. The difference — and this is the subtle, important part — is *how* the features are delivered. EAGLE-3 **fuses** target features with the drafter's token embeddings at the *input* and feeds them in once. As the drafter gets deeper, that signal is diluted through successive layers of self-attention and FFN, so the conditioning weakens with depth. This is why adding layers to an input-fusion drafter gives diminishing returns on acceptance length.

DFlash treats the fused target context as **persistent contextual information** and injects it directly into the Key and Value projections of *every* draft layer. The features live in the drafter's KV cache and are reused across drafting iterations. Concretely (from Appendix A.3), DFlash first concatenates hidden states from the selected target layers and projects them once:

$$
H_t = \text{RMSNorm}\left(W_c\, [H^{(l_1)}; \dots; H^{(l_5)}]\right)
$$

These projected features $H_t$ are shared by all draft layers. Then at draft layer $i$, the draft tokens produce queries, while *both* the target features and the draft tokens contribute keys and values:

$$
Q_i = W_i^Q H_d, \qquad
K_i = [W_i^K H_t; W_i^K H_d]_{\text{seq}}, \qquad
V_i = [W_i^V H_t; W_i^V H_d]_{\text{seq}}
$$

The target features only ever serve as *additional KV entries* for the masked-block draft tokens. They bypass the drafter's Q projection, output projection, self-attention update, and FFN entirely. Every layer gets to attend back to the full-strength target signal — no dilution. In code, a single KV-injected attention layer looks like this:

```python
import torch
import torch.nn.functional as F

def kv_injected_attention(h_draft, h_target_ctx, wq, wk, wv, wo, n_heads):
    """One DFlash draft-layer attention block.

    h_draft:      (B, L_block, D)  the 16 masked draft positions
    h_target_ctx: (B, L_ctx,   D)  projected target features H_t (shared, frozen at this layer)
    """
    B, Lb, D = h_draft.shape
    q = wq(h_draft)                                   # queries: ONLY from draft tokens
    k = torch.cat([wk(h_target_ctx), wk(h_draft)], 1) # keys:   target ctx ++ draft tokens
    v = torch.cat([wv(h_target_ctx), wv(h_draft)], 1) # values: target ctx ++ draft tokens

    def split(x):                                     # (B, Lx, D) -> (B, n_heads, Lx, D/n)
        return x.view(B, -1, n_heads, D // n_heads).transpose(1, 2)

    out = F.scaled_dot_product_attention(split(q), split(k), split(v))  # bidirectional within block
    out = out.transpose(1, 2).reshape(B, Lb, D)
    return wo(out)                                    # output is ONLY the 16 draft positions
```

The shape to internalize: `q` has length `L_block` (16), but `k` and `v` are length `L_ctx + L_block`. The draft positions attend back to the target context every single layer, and the target context is never updated by the drafter — it is read-only conditioning. That asymmetry is the whole trick.

The ablation that proves it matters is Table 9, and it is clean:

| Drafting style | Conditioning | GSM8K (τ / speedup) | HumanEval (τ / speedup) | MT-Bench (τ / speedup) |
|---|---|---|---|---|
| Autoregressive | Input fusion (EAGLE-3-5L) | 4.2 / 2.1× | 4.3 / 2.2× | 3.1 / 1.4× |
| Autoregressive | **KV injection** (DFlash-AR) | 4.8 / 2.4× | 4.6 / 2.3× | 3.4 / 1.5× |
| Block diffusion | Input fusion | 3.5 / 2.9× | 3.5 / 2.9× | 2.6 / 2.0× |
| Block diffusion | **KV injection** (DFlash) | 4.2 / 3.3× | 4.0 / 3.2× | 3.0 / 2.2× |

Read it twice. KV injection beats input fusion in *both* the autoregressive and the diffusion setting, so the gain is attributable to the injection mechanism itself, not to diffusion. And DFlash combines the two wins — KV injection *and* parallel diffusion drafting — which is why it lands highest on speedup even when its acceptance length is only comparable to EAGLE-3-5L.

> Conditioning a small model on a big one is not a one-time handshake at the input. It is a contract that has to be re-affirmed at every layer, or it gets forgotten.

### Second-order consequence: the conditioning is almost free in memory

You might worry that injecting features into every layer's KV cache blows up memory. It does not. The only extra parameterized component is the shared projection $W_c \in \mathbb{R}^{D \times 5D}$. For Qwen3.5-35B-A3B with $D = 2048$ in BF16, that is $5 \times 2048 \times 2048 \times 2 \approx 42$ MB — negligible against the ~70 GB target model. The runtime activation overhead during decoding with block size 16 is *under 400 KB*. The technique is cheap precisely because the features are computed once during prefill and then reused as static KV entries, not recomputed per layer. This is the same reuse logic behind a [KV cache](/blog/machine-learning/large-language-model/kv-cache): compute an expensive representation once, then read it many times.

## 4. The inference architecture

**Senior rule of thumb: in a two-model system, draw the data-flow and mark exactly which tensors cross the boundary — that boundary is where your performance and your bugs live.**

![DFlash inference architecture: features from five target layers condition the KV-injected draft stack for single-pass block speculation](/imgs/blogs/dflash-block-diffusion-speculative-decoding-5.png)

Now we can assemble the full inference path. Given a prompt, the target model runs a standard prefill pass to generate the first token. During that pass, DFlash extracts hidden representations from a **fixed set of five layers**, uniformly sampled from the second layer to the third-to-last layer of the target (shallow to deep, so the features span both low-level token statistics and high-level semantics). These hidden states are concatenated and passed through the lightweight projection $W_c$ to fuse cross-layer information into a single compact **target context feature**.

That feature is injected — as described above — into the KV cache of every draft layer. The drafter then takes a block of mask tokens (`<m>` × 16), runs **bidirectional attention** over the block (this is the diffusion part — masked positions attend to each other, not just left-to-right), and produces logits for all 16 positions at once through the **shared LM head** of the target. Those 16 speculative tokens go to verification.

Two design choices in this path are worth flagging:

- **Shared, frozen embedding and LM head.** The draft model reuses the target's token embedding layer and language modeling head, both kept frozen. Only the draft transformer layers are trained. This reduces trainable parameters dramatically and forces the drafter to live in the target's representation space — it is literally a small adapter, not an independent model.
- **The five-layer feature extraction is fixed at training time.** The choice of which target layers to read from is a hyperparameter (the paper uses five; an ablation in Table 7 shows three is meaningfully worse: τ drops from 5.64 to 5.38 on Math500). More features means richer conditioning and higher τ, but the offline-training storage cost grows linearly with the number of cached features, so five is the practical compromise.

Here is what wiring this up looks like with the released checkpoints on the Transformers backend (Qwen3 / Llama only). The `spec_generate` call owns the entire DFlash loop — prefill, feature extraction, diffusion drafting, target verification, accept-prefix-plus-bonus, and repeat — so from the caller's side it looks like ordinary generation:

```python
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

target_id = "Qwen/Qwen3-8B"
draft_id  = "z-lab/Qwen3-8B-DFlash-b16"   ## block size 16 drafter

tokenizer = AutoTokenizer.from_pretrained(target_id)
target = AutoModelForCausalLM.from_pretrained(
    target_id, torch_dtype=torch.bfloat16, device_map="cuda"
)
draft = AutoModel.from_pretrained(           ## carries W_c + the 5 draft layers
    draft_id, torch_dtype=torch.bfloat16, device_map="cuda"
)

messages = [{"role": "user", "content": "Prove that sqrt(2) is irrational."}]
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

output = draft.spec_generate(
    input_ids=input_ids,
    target=target,            ## target passed INTO the drafter — drafter orchestrates
    max_new_tokens=2048,
    block_size=16,            ## diffusion block; can be reduced at inference
    temperature=0.0,          ## greedy verification is lossless
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The thing to internalize: `spec_generate` lives on the *drafter*, and the *target* is passed in as an argument. The drafter is the orchestrator because it is the thing that knows about diffusion blocks; the target is just the verification oracle it calls.

On Apple Silicon the MLX backend has the same shape — load target, load drafter, stream:

```python
from dflash.model_mlx import load, load_draft, stream_generate

model, tokenizer = load("Qwen/Qwen3.5-4B")
draft = load_draft("z-lab/Qwen3.5-4B-DFlash")

for token in stream_generate(model, draft, tokenizer, prompt, max_tokens=512):
    print(token, end="", flush=True)
```

### Where the feature extraction cost lands

A fair question: if DFlash extracts hidden features from five target layers, does that not add a forward-pass cost? The answer is no, and the reason is timing. The features are extracted *during the prefill pass the target already runs* to produce the first token. Prefill happens regardless — it is how the model reads the prompt — so DFlash piggybacks on activations that are already in memory. The only added work is the single $W_c$ projection over those activations, which is one small matmul. During the decode loop, the context features are *not* recomputed: they live in the drafter's KV cache as static entries and are read on every draft pass. So the feature-extraction cost is a one-time, near-free addition to a pass you were running anyway, not a recurring per-block tax. This is the same amortization principle that makes the KV cache itself worth maintaining — pay once at prefill, read many times at decode.

There is a subtlety for multi-block generation. As the response grows, new clean tokens (the accepted prefixes and bonus tokens) get appended, and their target features must be computed too. But those features come for free as part of the verification pass — verification already runs the target over the new tokens — so again there is no extra forward pass dedicated to feature extraction. The accounting works out because every tensor DFlash needs is a byproduct of a pass the speculative loop was already going to run.

### How verification stays lossless

The lossless guarantee deserves its own paragraph because it is the property that makes the whole "use a low-quality drafter" bet safe. After the drafter proposes 16 tokens, the target runs *one* forward pass over the prompt-plus-draft and produces its own next-token distribution at every one of those 16 positions in parallel. Verification then walks the block left to right and applies the standard speculative-sampling acceptance test:

- **Greedy (temperature 0):** accept draft token at position $k$ iff it equals the target's argmax at position $k$. Stop at the first mismatch; the target's own argmax becomes the bonus token.
- **Sampling (temperature 1):** accept with probability $\min\big(1, p_{\text{target}}(x_k)/p_{\text{draft}}(x_k)\big)$, and on rejection resample from the residual distribution $\max(0, p_{\text{target}} - p_{\text{draft}})$ renormalized. This is the Leviathan et al. (2023) correction, and it makes the *marginal* output distribution exactly equal to sampling from the target alone.

Either way, every emitted token is one the target would have produced. The drafter's only influence is on *how many* tokens get accepted per cycle — i.e., on speed — never on *which* tokens come out. That is why a wrong draft costs you a shorter accepted prefix (a smaller speedup that cycle) and never a quality regression. It is the cleanest kind of optimization: the worst case is "no faster," not "wrong."

### Second-order consequence: lossless is not the same as deterministic

A subtle point that trips people up in production. "Lossless" means the output *distribution* matches the target's. At temperature 0 that also means bit-identical greedy outputs. But at temperature > 0, DFlash will produce *different samples* than the target run with the same seed, because the speculative-sampling correction draws from residual distributions in a different order than plain sampling does. The distribution is identical; the particular sample is not. If you have a regression test that pins exact sampled outputs against a fixed seed, it will "fail" under speculation even though nothing is wrong. Pin the *distribution* (or run greedy) in your tests, not the exact token stream.

## 5. Training the diffusion adapter

**Senior rule of thumb: train the drafter on exactly the distribution it will see at inference, including the part you are tempted to treat as a detail — the clean prefix token.**

![DFlash training: each block starts from a clean anchor token and predicts the rest in parallel, matching inference](/imgs/blogs/dflash-block-diffusion-speculative-decoding-6.png)

DFlash does not adopt standard block-diffusion training off the shelf. It makes a set of modifications that align training with the speculative-decoding inference behavior, and they matter a lot for acceptance length.

The setup: given a sequence of prompt tokens $p$ and response tokens $r$, the entire *clean* sequence is first passed through the frozen target model to extract and fuse the hidden features for all tokens. Those features are injected into the draft model as Key and Value entries, exactly as at inference. Then the response is broken into blocks for the masked-prediction objective.

The diagram above shows the key idea. Rather than uniformly dividing the response into blocks and masking random positions (the standard block-diffusion recipe), DFlash **randomly samples anchor tokens** from the response. Each sampled anchor becomes the *first* position of a block — a clean, known token — and the next (block size − 1) positions are masked. The drafter is trained to predict those masked positions in parallel.

Why anchors? Because at inference, the drafter *always* conditions on a clean token: the bonus token produced by the previous verification step. The first position of every real draft block is known-good. If you train with random masking that sometimes leaves the first position masked, you create a train/inference mismatch. Anchoring every block on a clean token removes that mismatch — the training distribution now matches the inference distribution exactly. This train/inference alignment is the single most important training decision in the paper, and we will see its measured impact in the next section.

### The attention mask and FlexAttention

The attention structure (shown in the figure) enforces the right information flow. Within a block, tokens attend **bidirectionally** to each other and to the injected target context features. Across blocks, attention is **disallowed** — block 2 cannot peek at block 1's clean tokens, because at inference those would not yet exist. This block-diagonal-plus-context mask is irregular: it is not a clean causal triangle, and it changes shape every step because anchors are resampled. That is exactly the kind of mask that kills throughput with a naive `(seq, seq)` boolean attention matrix.

The fix is [FlexAttention](https://pytorch.org/blog/flexattention/) (Dong et al., 2024), which lets you express the mask as a *function* `(b, h, q_idx, kv_idx) -> bool` and compiles it into a fused, block-sparse kernel. DFlash concatenates all draft blocks into one sequence and processes them jointly in a single forward/backward pass under this sparse mask — many blocks trained for the price of one. Without FlexAttention you would either materialize a huge mostly-zero mask or loop over blocks; with it, the irregular mask is essentially free. If you take one *infrastructure* lesson from this paper, it is that FlexAttention is what makes the anchor-based block training practical at all.

### Bidirectional within the block: the diffusion part

It is worth pausing on why the within-block attention is bidirectional rather than causal. In autoregressive drafting, token 5 can only see tokens 1–4. In DFlash, all 16 masked positions are predicted *simultaneously* and each one attends to all the others (plus the target context). This is the defining property of a diffusion / masked-prediction model: position 12 can use information that "comes after" it in the sequence, because at prediction time there is no left-to-right ordering — every masked slot is filled in one shot. Bidirectionality is what lets a single forward pass produce a coherent 16-token block instead of 16 independent guesses. The cost is that the drafter cannot use already-decoded tokens within the block as context — but that is precisely the job verification picks up.

### The training configuration that actually shipped

For reproducibility, here is the recipe from Appendix A.1:

- **Optimizer:** AdamW, learning rate $6 \times 10^{-4}$, gradient clip 1.0, cosine schedule, warmup ratio 0.04.
- **Epochs:** 6.
- **Sequence length:** 3072 tokens (4096 for Qwen3-Coder).
- **Anchors per sequence:** 512 randomly sampled positions.
- **Data:** ~800K samples from NVIDIA Nemotron Post-Training Dataset V2 + CodeAlpaca — but crucially, the *responses are regenerated by the target model itself* for better alignment, rather than using the original dataset responses.
- **Drafter:** 5 layers (8 for Qwen3-Coder), block size 16 (10 for Llama-3.1).
- **Features:** 5 target layers, uniformly spaced from layer 2 to the third-to-last.

That "regenerate responses with the target model" detail is easy to skip past and important: the drafter is learning to mimic *this specific target*, so training on the target's own outputs gives a much tighter distribution match than training on arbitrary human text. A drafter trained on human-written answers would keep proposing tokens the human would write but the target would not — and every such token is a guaranteed rejection. This is the same reasoning behind on-policy distillation: you want the student to be trained on the teacher's own trajectory, not on a reference trajectory the teacher would never take. The acceptance test is unforgiving about this — it only counts tokens that match the target exactly (greedy) or pass the probability ratio test (sampling), so any systematic gap between the training distribution and the target's distribution shows up directly as shorter acceptance.

### What is trainable, and how small the drafter really is

It is worth quantifying just how lightweight the trained component is, because "5 layers" can sound bigger than it is. The drafter shares the target's token embedding and LM head (both frozen) and adds only its own transformer layers plus the single $W_c$ projection. For a model like Qwen3-8B, the trainable drafter is on the order of a few hundred million parameters — roughly 5/36 of the target's layer stack, minus the embedding and head that dominate parameter count in smaller models. The practical consequences are concrete: training fits comfortably on a single H200, a full run is hours not days, and the released checkpoints are small enough to download and swap in seconds. Compare this to the 7B diffusion drafters of DiffuSpec and SpecDiff-2, which are nearly the size of the target itself and must be co-resident in GPU memory alongside it. DFlash's drafter is small enough that its weights are a rounding error in your memory budget — the 42 MB projection plus a sub-billion-parameter stack against a multi-gigabyte target.

### Online vs offline feature extraction

Training can run two ways. **Online**, the target hidden features are computed on the fly each step — simpler, no storage, but you pay a target forward pass per training step. **Offline**, the features are precomputed once and cached to disk, then streamed during draft optimization — far cheaper compute per step, but storage grows linearly with the number of extracted layers (which is the real reason five features, not ten, is the sweet spot). For a one-off training run, online is fine; for repeated experiments on the same target, offline amortizes. The storage math is the gotcha: caching five BF16 hidden states per token for 800K samples of ~1K tokens each runs into the tens of terabytes for a large-$D$ target, which is exactly why you do not cache ten.

## 6. Three training tricks worth stealing

Beyond the architecture, DFlash includes three training-time ideas that generalize well past this paper. Each is a small change with a measurable payoff.

### Trick 1: random anchor sampling as data augmentation

![Block construction: random anchor sampling augments the data and matches inference, lifting acceptance length](/imgs/blogs/dflash-block-diffusion-speculative-decoding-7.png)

We just covered *why* anchoring matters for train/inference alignment. The second benefit is **data augmentation**. Because anchor positions are resampled every epoch, the same training sequence produces different block boundaries on different passes, exposing the drafter to a far more diverse set of target context features. One sequence becomes many training examples. The ablation (Table 13) isolates this:

| Block construction | Math500 (speedup / τ) | HumanEval (speedup / τ) | MT-Bench (speedup / τ) |
|---|---|---|---|
| Standard (uniform split, fixed masks) | 4.13× / 4.94 | 3.29× / 3.86 | 2.13× / 2.80 |
| **Random anchor sampling** | 4.69× / 5.64 | 3.90× / 4.61 | 2.38× / 3.18 |

That is a ~14% relative bump in acceptance length on Math500 from a change that costs nothing at inference and almost nothing to implement. This is the kind of trick that quietly separates a good speculative system from a mediocre one. The general principle: when your training examples are *windows* over a longer sequence, randomizing the window boundaries every epoch is nearly-free augmentation that also lets you match a specific inference-time alignment.

### Trick 2: position-weighted loss decay

![Loss decay: an early-position error invalidates every later token, so DFlash weights early positions exponentially](/imgs/blogs/dflash-block-diffusion-speculative-decoding-8.png)

In speculative decoding, **not all token positions are equal**. Verification accepts the longest *correct prefix* of a block. So an error at position 2 invalidates positions 3 through 16 regardless of whether they were individually correct — the whole tail is discarded the moment an early token is rejected. Early predictions are disproportionately valuable for acceptance length.

DFlash reflects this asymmetry by weighting the cross-entropy loss to emphasize earlier positions. For a token at position $k$ within a block:

$$
w_k = \exp\!\left(-\frac{k-1}{\gamma}\right)
$$

where $\gamma$ controls the decay rate (set to 7 for block size 16, 5 for block 10, 4 for block 8). Position 1 gets weight 1.0, and weights fall off exponentially down the block. In Python this is a few lines, applied as a weighted reduction over the per-position cross-entropy:

```python
import torch

def block_loss_weights(block_size: int, gamma: float) -> torch.Tensor:
    k = torch.arange(block_size, dtype=torch.float32)  ## 0-indexed positions
    return torch.exp(-k / gamma)                        ## w_k = exp(-(k-1)/gamma)

def weighted_block_loss(logits, targets, gamma):
    ce = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), targets, reduction="none"
    )                                                   ## (B, block_size)
    w = block_loss_weights(logits.size(1), gamma).to(ce)
    return (w * ce).sum(dim=1).mean() / w.sum()         ## early positions weighted most

weights = block_loss_weights(16, 7.0)                   ## [1.00, 0.87, 0.75, 0.65, 0.56, ...]
```

The payoff (Figure 5 in the paper): loss decay makes training **converge faster and to a higher acceptance length** — roughly 6.3 vs 5.8 on Math500 after convergence, with the gap visible from the first few epochs. It is a few lines of change with a free lunch attached. The lesson generalizes to any system where outputs are accepted as a prefix: weight the loss toward the positions whose correctness gates everything downstream.

### Trick 3: bounded long-context training

Training speculative drafters on long contexts is expensive for methods like EAGLE-3 because of their costly "training-time test" procedure (they run the drafter autoregressively during training to simulate inference, which scales badly with sequence length). DFlash sidesteps this by **fixing the number of masked blocks per sequence** and randomly sampling anchor positions for each sequence every epoch. The training cost per sequence is therefore constant in context length — you always predict the same fixed number of blocks, regardless of whether the sequence is 4K or 32K tokens. The randomized anchors still provide augmentation across the long sequence. We will see in the case studies that the base 4K-context drafter can be adapted to 32K context with a tiny 1.6K-sample fine-tune — that cheap adaptation is a direct consequence of this bounded-cost training design.

## 7. Block size: the one knob that matters

**Senior rule of thumb: train at the largest block size you can afford, then scale down at inference — never the reverse.**

![Train-inference block-size transfer: a block-16 drafter generalizes down to block 8, but a block-8 drafter cannot scale up](/imgs/blogs/dflash-block-diffusion-speculative-decoding-9.png)

Block size is *the* dominant design choice for a DFlash drafter, and the paper's ablation (Table 8) surfaces a sharp asymmetry. The matrix above reports speedup / τ on Math500 for an 8-layer drafter trained and tested at every combination of block size 8 and 16.

Two findings, both actionable:

1. **When training and inference block sizes match, larger is better.** The block-16 drafter (4.64× / 6.33 τ) substantially beats the block-8 drafter (3.97× / 5.21 τ) on math and code. The acceptance histograms explain why: the block-8 model frequently accepts the *entire* block (35.7% of the time on Math500), which means block 8 is too small — it is leaving acceptance on the table. The block-16 model has a more spread-out acceptance distribution and a higher average, indicating it actually uses the larger budget.

2. **Generalization is one-directional.** A drafter trained at block 16 generalizes *down* to inference block 8 nearly as well as a natively-block-8 model (3.87× / 5.09 vs 3.97× / 5.21). But a drafter trained at block 8 *cannot* scale *up* to inference block 16 — it collapses to 3.78× / 5.02, worse than its native setting. The big block has seen the long-range patterns; the small block never learned them.

The practical recipe falls out immediately: **train once at block 16, deploy at whatever inference block size your serving conditions call for.** This enables dynamic block-size scheduling — a single checkpoint can serve large blocks at low concurrency (where you want maximum acceptance) and smaller blocks at high concurrency (where verification cost dominates and a tighter block is cheaper to verify). The paper leaves adaptive scheduling to future work, but the one-checkpoint-fits-all property is already there for you to exploit.

### Second-order consequence: block size interacts with the concurrency cliff

Recall the high-concurrency cliff from Section 1. Block size is the lever that controls it. A larger block raises acceptance length (good at low concurrency) but raises per-request verification cost (bad at high concurrency, where verification is compute-bound). Because a block-16 drafter degrades gracefully to block 8 at inference, you can in principle pick block size per-batch based on current load — large when the GPU is idle-ish, small when it is saturated. That is the single most impactful tuning you can do with a deployed DFlash drafter, and it costs you nothing at training time because you only ever train the one block-16 checkpoint.

## 8. The diffusion-drafting landscape

DFlash is not the first attempt to put a diffusion model in the drafting seat. Understanding why the earlier attempts plateaued is the fastest way to understand what DFlash got right. There are roughly four prior families, and each made a different trade.

| Method | Drafter | Conditioning | Practical speedup | Why it plateaued |
|---|---|---|---|---|
| **DiffuSpec** (2025) | ~7B pretrained dLLM | inference-time search | 3–4× | Massive drafter → high memory + drafting latency offsets the win |
| **SpecDiff-2** (2025) | ~7B pretrained dLLM | train–test alignment | 3–4× | Same: long acceptance but the 7B drafter is too slow/heavy to serve |
| **PARD** (2025) | small AR model | mimics parallel gen | ~3× | Tiny model lacks the target's capacity → short acceptance |
| **TiDAR** (2025) | joint diffusion+AR | trained together | n/a (not lossless) | "Think in diffusion, talk in AR," but final quality not yet lossless |
| **DFlash** (2026) | 5-layer block-diffusion | KV-injected target features | **6×** | — combines lightweight drafter + strong conditioning + lossless verify |

The pattern is a vise. On one side, DiffuSpec and SpecDiff-2 show that a *big* diffusion drafter produces high-quality, long-acceptance drafts — but a 7B drafter is so heavy that drafting latency and memory eat the speedup, capping it at 3–4× and making it impractical to co-serve. On the other side, PARD shows that a *small* model is cheap to run — but a tiny model reasoning from scratch lacks the target's capacity, so acceptance is short and the ceiling is ~3×. TiDAR tries to fuse the paradigms at the architecture level but does not reach losslessness.

DFlash escapes the vise by changing what the small model has to do. The drafter does not need 7B parameters' worth of world knowledge, because it is not reasoning from scratch — it is *decoding the target's hidden features*, which already contain the reasoning. So you get the long-acceptance benefit of the big-drafter approaches with the low-latency benefit of the small-drafter approaches, and verification keeps it lossless. That is the "no free lunch — actually, yes free lunch" move: the lunch was being paid for by the target model all along, and DFlash just stopped making the drafter re-buy it.

> The drafter doesn't need to be smart. It needs to be a fast decoder for a smart model's thoughts.

## Cross-cutting: memory cost and stack placement

![DFlash in the serving stack: speedup is largest at low concurrency and tapers as verification turns compute-bound at large batch](/imgs/blogs/dflash-block-diffusion-speculative-decoding-10.png)

We covered the memory cost in Section 3 — the $W_c$ projection adds ~42 MB and the runtime activation overhead is under 400 KB at block 16. That is the entire memory footprint of DFlash beyond the drafter's own (small) weights. Compare this to DiffuSpec and SpecDiff-2, which use 7B-parameter diffusion drafters: those carry a multi-gigabyte memory footprint that is often prohibitive for real serving, and their high drafting latency caps practical speedups at 3–4×. DFlash's lightweight drafter is the whole point.

Where does DFlash sit in your stack? It is a drop-in drafter for the speculative-decoding path your serving framework already has. The released implementation supports four backends — Transformers, [SGLang](/blog/machine-learning/large-language-model/sglang-inference), [vLLM](/blog/machine-learning/large-language-model/vllm-inference), and MLX. On SGLang, DFlash plugs into the Spec-v2 scheduling-overlap path; you enable it the way you would any speculative decoder. The serving-time benchmark is one command, which reports tokens/s, speedup over the autoregressive baseline, and average τ at each concurrency level:

```bash
python -m dflash.benchmark \
    --target Qwen/Qwen3-8B \
    --draft  z-lab/Qwen3-8B-DFlash-b16 \
    --backend sglang \
    --dataset math500 \
    --block-size 16 \
    --concurrency 1,4,8,16,32 \
    --temperature 0.0
```

(Supported datasets: `gsm8k`, `math500`, `humaneval`, `mbpp`, `mt-bench`.) A note on reading these benchmarks honestly: the tool reports both raw tokens/s and the speedup ratio, and you should look at both. The ratio tells you how much speculation helped; the raw throughput tells you whether you are even in a regime where the ratio matters. A 5× speedup at 1,175 tok/s and a 2.8× speedup at 16,076 tok/s are both real, but they describe different machines doing different jobs — one is a latency-optimized single stream, the other is a throughput-optimized fleet. Always pair the multiplier with the absolute number before you draw a conclusion. The full SGLang result on a single B200 with the FlashAttention-4 backend tells the throughput story end to end:

| Model / task | Metric | conc 1 | conc 4 | conc 8 | conc 16 | conc 32 |
|---|---|---|---|---|---|---|
| Qwen3-8B / Math500 | baseline tok/s | 230 | 861 | 1666 | 3133 | 5694 |
| Qwen3-8B / Math500 | DFlash tok/s | 1175 | 3884 | 7485 | 12268 | 16076 |
| Qwen3-8B / Math500 | speedup | 5.1× | 4.5× | 4.5× | 3.9× | 2.8× |

The diagram above is the operational mental model: requests flow into the scheduler, the DFlash drafter produces a block, the target verifies on the FA4 backend, and the *outcome depends on concurrency*. At batch 1 you see the full ~5.1× win; at batch 32 verification is compute-bound and you see ~2.8×. Both are real, and which one you get is a function of your load — so benchmark at *your* concurrency, not at the headline batch-1 number.

> Every speculative speedup number you read is at a specific concurrency. If the paper doesn't tell you which, assume batch 1 and halve it for production.

### A reproduction checklist

If you are standing this up and your numbers come in low, the paper's ablations effectively hand you a debugging checklist. Work it top to bottom:

1. **Is the target-feature injection actually wired in?** A drafter with no conditioning tops out at ~2–3× (Table 10). If you are near there, your KV injection is probably misconfigured or silently disabled.
2. **Do training and inference block sizes match (or train ≥ infer)?** Train-small/infer-large drops you ~20% (Table 8). Never infer at a larger block than you trained.
3. **Are you extracting five target layers, not fewer?** Three layers costs ~5% τ (Table 7).
4. **Did you train with random anchors and loss decay?** Each is worth ~10–14% τ (Tables 13 and Figure 5).
5. **Did you train on target-regenerated responses, not raw dataset text?** Mismatch here silently shortens acceptance.
6. **Are you benchmarking at the right concurrency?** A "disappointing" 2.8× at concurrency 32 is the *expected* number, not a bug.
7. **Did you regenerate training responses with the target?** Training on raw dataset text instead of the target's own outputs is a quiet acceptance-killer that no single ablation table isolates but every production team rediscovers the hard way.

## Case studies from production and evaluation

These are drawn from the paper's experiments (all on NVIDIA H200 or B200 GPUs) and framed as the deployment scenarios you would actually hit. Each one carries a lesson about *when* the headline number applies and when it does not.

### 1. Qwen3-8B chat on the Transformers backend

The flagship result. With greedy decoding (temperature 0) on the Transformers backend, DFlash with block 16 hits an average **4.86× speedup** across math, code, and chat — peaking at **6.08×** on MATH-500 with an acceptance length of 7.87 tokens per cycle. EAGLE-3 with a matched 16-node tree manages 1.76× average; even EAGLE-3 with a 60-node tree only reaches 2.02×. So DFlash is **2.4× faster than the fairest EAGLE-3 comparison** and still ahead of the verification-heavy 60-node configuration. Under non-greedy sampling (temperature 1) DFlash holds 4.03× average — sampling lowers acceptance because the target's distribution is softer, but the structural advantage survives. The lesson: the win is largest on structured, predictable outputs (math reasoning, where the next several tokens are highly determined) and smallest on open-ended chat (MT-Bench: 2.75×), because chat tokens are less predictable and acceptance length drops. Match your expectations to your workload's predictability — a math-tutoring product will see double what a free-form chat product sees, off the same checkpoint.

### 2. Reasoning mode with long chain-of-thought

The scenario DFlash is *made* for. With Qwen3 thinking mode enabled — long CoT traces, exactly the workload where inference latency hurts most — DFlash holds ~4.5× speedup at temperature 0 and ~3.9× at temperature 1 (Table 2: 4.64× on MATH-500 and 4.51× on AIME25 for Q3-8B, with τ around 5.7–5.8). Reasoning traces are long and internally consistent, so the drafter's acceptance length stays high across thousands of tokens — there is no short-output amortization problem here. The lesson: speculative decoding's value scales with output length, and reasoning models produce the longest outputs in production. A reasoning model that takes 30 seconds to think now takes ~7, with identical answers. If you are serving a reasoning model, this is where the money is, and it is also where users feel latency most acutely.

### 3. SGLang on B200 with FlashAttention-4: the concurrency story

The realistic-serving case. On a single B200 with the FA4 backend and Spec-v2 scheduling overlap, Qwen3-8B on MATH-500 shows the full concurrency curve: **5.1× at concurrency 1, 4.5× at 8, 3.9× at 16, and 2.8× at 32**. Throughput climbs from 230 tok/s (baseline, concurrency 1) to 16,076 tok/s (DFlash, concurrency 32). Read those two axes together: the *multiplier* shrinks as you batch, but the *absolute* throughput gain is enormous at every level — at concurrency 32 you are still nearly tripling tokens/s. The lesson is the one we keep returning to: the speedup decays with batch size as verification becomes compute-bound, but it never goes below ~2.8×. Do not quote the 5.1× to your capacity planner if you run at concurrency 32 — quote the 2.8×, and note that it is 2.8× on top of an already-batched baseline.

### 4. Qwen3-Coder-30B-A3B: the MoE coder that stays flat

The interesting outlier. The Qwen3-Coder MoE model (30B total, 3B active) uses an 8-layer drafter and shows a *different* concurrency profile: HumanEval speedup is 3.5× at concurrency 1 and stays around 3.0–3.2× all the way to concurrency 32 — on MBPP it is essentially flat at ~3.1–3.3×. The MoE structure means the active parameter count is small, so verification stays cheaper for longer and the concurrency cliff is gentler. The lesson: MoE targets and dense targets have different DFlash economics — MoE holds its speedup at high concurrency better, which is exactly where dense models struggle. If you serve MoE, DFlash is even more attractive than the dense numbers suggest, because the regime where dense models lose their edge is the regime where MoE keeps it. This complements the broader story on [optimizing MoE inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference).

### 5. Long-context adaptation to 32K with a 1.6K-sample fine-tune

The "will it survive my real prompts" case. The base DFlash drafters are trained at 4K context, and the paper shows their acceptance length *degrades* as context grows past 4K (on gov_report, τ falls from 4.53 at 1K to 2.09 at 32K — the drafter has never seen the long-range patterns). The fix is cheap: fine-tune the base Qwen3.5-27B drafter on just **1.6K samples** from LongAlign-10K for 3 epochs. After that, the long-context drafter *maintains or improves* acceptance — on hotpotqa it goes from 4.91 (base, 1K) to **6.05** (fine-tuned, 16K), and on gov_report it holds 3.56 at 32K versus the base model's 2.09. The lesson: the extracted target features remain representative at long context (the target's own long-context ability is intact); the drafter just needs to *see* long-range patterns once to learn to decode them. Budget a tiny, hours-long long-context fine-tune before deploying on long prompts, and do not trust the 4K checkpoint past 8K.

### 6. Llama-3.1-8B on EAGLE-3's own training data

The apples-to-apples case that rules out "DFlash just had better data." The authors trained a DFlash drafter (block size 10) for Llama-3.1-8B-Instruct on the *exact* same UltraChat + ShareGPT data EAGLE-3 used, and evaluated against the official EAGLE-3 checkpoint on SGLang. DFlash wins across math, code, and chat at every concurrency: on GSM8K, DFlash hits 2.4× at concurrency 1 (τ = 4.32) vs EAGLE-3's 1.6× — and at concurrency 32 EAGLE-3's 60-node tree collapses to **0.6×** (slower than baseline!) while DFlash holds 1.6×. That 0.6× is the most instructive number in the paper: a big draft tree verifies dozens of mostly-rejected nodes per request, and at high concurrency that wasted verification compute makes speculation a net *loss*. The lesson: EAGLE-3's tree-based verification actively hurts at high concurrency, whereas DFlash's tight block degrades gracefully. The advantage is structural, not data-driven — same data, same target, very different scaling behavior.

### 7. GPT-OSS-120B and the native-MTP comparison

The "does it scale to frontier-size models" case. DFlash is evaluated on GPT-OSS-20B and GPT-OSS-120B, and on Qwen3.5 models against their *native* multi-token-prediction (MTP) heads. On Qwen3.5-9B, DFlash hits 3.5× / 3.4× / 2.5× on math/code/chat versus native MTP's 1.7× / 1.7× / 1.3× — roughly double across the board. On GPT-OSS-120B, DFlash still delivers 1.6–1.7× even at concurrency 8 on a large MoE. The lesson: DFlash beats the model's own built-in speculative mechanism, because MTP heads are shallow input-fusion predictors (one extra head per future position, fed the same diluted input signal) and DFlash's deep KV-injected drafter simply produces better blocks. If your model ships with MTP, the natural assumption is "it already has speculation, I'm done" — but DFlash is still worth the swap, often for a 2× improvement over the built-in path.

### 8. vLLM at low vs high concurrency

The "my framework isn't SGLang" case. On vLLM with Qwen3.5-9B, DFlash delivers 4.0× / 4.6× / 3.0× (math/code/chat) at concurrency 1, decaying to 1.9× / 2.1× / 1.3× at concurrency 32. The shape matches SGLang — strong at low concurrency, tapering as verification saturates — confirming the behavior is a property of the method, not the framework. The absolute numbers differ slightly (scheduler overlap and kernel choices vary between frameworks), but the curve is the same. The lesson: DFlash is portable across serving stacks, and the concurrency curve is intrinsic. Pick your framework for other reasons — ecosystem, ops familiarity, kernel support — and DFlash will behave the same way on each. There is no framework you can switch to that escapes the high-concurrency taper.

### 9. The two cautionary tales: no-conditioning and block-size mismatch

The debugging case, assembled from two ablations. First, the *no-target-feature* drafter: a five-layer block-diffusion drafter trained without any target conditioning gets only 2.83× on GSM8K and 3.73× on Math500 (Table 10) — i.e., if you forget the KV injection, you have built a mediocre standalone diffusion drafter and thrown away the entire DFlash advantage. The drafter still *works*, which is what makes this insidious: it does not crash, it just quietly delivers EAGLE-class numbers and you wonder why the paper claimed 6×. Second, the *block-size mismatch*: train at block 8, infer at block 16, and you drop to 3.78× / 5.02 τ — worse than just training at block 8 natively. The lesson for anyone reproducing this: the two ways to silently lose most of the speedup are (a) misconfiguring or dropping the target-feature injection, and (b) running inference at a larger block size than you trained on. If your reproduction lands at ~2–3× instead of ~5×, check those two things before you touch anything else.

## When to reach for DFlash, and when not to

DFlash is a strong default for latency-sensitive single-stream and low-to-medium-concurrency serving, but it is not free of tradeoffs. Here is the honest decision guide.

**Reach for DFlash when:**

- You serve a **reasoning model** or any workload with long, structured outputs (math, code, long CoT). This is where acceptance length stays high and the 4–6× wins materialize.
- You run at **low-to-medium concurrency** (roughly 1–16) where verification is not yet compute-bound and the full drafting advantage shows.
- You want **lossless** acceleration — the output distribution must exactly match the target. DFlash guarantees this through verification; there is no quality regression to argue about with stakeholders.
- You are on a supported target (Qwen3 / Qwen3.5 / Qwen3-Coder, Llama-3.1, GPT-OSS) and backend (Transformers, SGLang, vLLM, MLX), and you can spend a modest one-time training run for the drafter.
- Your model ships with **native MTP** and you want something strictly better — DFlash roughly doubles MTP's speedup.
- You serve an **MoE target**, where DFlash holds its speedup at high concurrency better than dense models do.

**Skip DFlash (or temper expectations) when:**

- You are **throughput-bound at very high concurrency** (32+). Verification becomes the bottleneck, speedup tapers toward ~2×, and you may prefer to spend GPU on raw batched throughput rather than speculation. Benchmark at your real concurrency before committing.
- Your outputs are **short and open-ended** (a few dozen tokens of chat). Speculative decoding's overhead amortizes over output length; short outputs barely benefit (MT-Bench is the floor of every DFlash table for a reason).
- You **cannot train a drafter** for your target and no checkpoint exists. DFlash needs a target-specific drafter; it is not a zero-shot wrapper you can point at an arbitrary model.
- You need to deploy on **very long contexts** without any fine-tuning. The 4K base drafter degrades past 8K — you must budget the (cheap, ~1.6K-sample) long-context adaptation first.
- Your serving framework has **no speculative-decoding path** at all. DFlash slots into an existing spec-decode mechanism; it is not a from-scratch serving engine.

The meta-lesson of DFlash is bigger than any single number. For two years the field treated diffusion language models as failed competitors to autoregressive models — too low-quality to ship. DFlash reframes them: a diffusion model does not have to win end-to-end generation to be useful. Confined to drafting, where its parallelism is a superpower and its quality is backstopped by verification, it becomes the missing piece that finally makes the *drafter* parallel. That reframing — find the one sub-task where a "weak" model's strengths align with the task and let a strong model cover its weaknesses — is the part worth carrying into your own systems, well beyond speculative decoding. Most "this model isn't good enough" verdicts are really "this model isn't good enough *at the whole job*." Shrink the job to the part it is good at, backstop the rest, and the verdict flips.

## Further reading

- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) — the paper (Chen, Liang & Liu, ICML 2026).
- [z-lab/dflash on GitHub](https://github.com/z-lab/dflash) — reference implementation with Transformers, SGLang, vLLM, and MLX backends, plus released drafter checkpoints on Hugging Face.
- [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://arxiv.org/abs/2503.01840) — the autoregressive-drafting state of the art DFlash measures against.
- [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573) — the block-diffusion foundation DFlash builds on.
- [Your LLM Knows the Future](https://arxiv.org/abs/2507.11851) — the observation that target hidden states encode future-token information, which motivates KV injection.
- [FlexAttention](https://pytorch.org/blog/flexattention/) — the PyTorch programming model that makes DFlash's irregular block-sparse training mask practical.
- Sibling posts on this blog: [Speculative decoding](/blog/machine-learning/large-language-model/speculative-decoding), [Diffusion language models deep-dive](/blog/machine-learning/open-source-library/dllm-diffusion-language-models-deep-dive), [KV cache](/blog/machine-learning/large-language-model/kv-cache), and [SGLang inference](/blog/machine-learning/large-language-model/sglang-inference).
