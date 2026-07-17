---
title: "Kimi K3: How Moonshot Built the First Open 3-Trillion-Parameter Model"
date: "2026-07-17"
description: "A close read of Moonshot's Kimi K3 launch: what a 2.8T-parameter open MoE with Kimi Delta Attention, Attention Residuals, a 16-of-896 LatentMoE, and MXFP4 quantization-aware training actually buys you — built from intuition up to the math, with the numbers checked against the announcement."
tags:
  - paper-reading
  - kimi-k3
  - moonshot-ai
  - mixture-of-experts
  - linear-attention
  - kimi-delta-attention
  - attention-residuals
  - mxfp4-quantization
  - long-context
  - model-scaling
  - open-weights
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 32
image: "/imgs/blogs/kimi-k3-1.webp"
paper:
  title: "Kimi K3: Open Frontier Intelligence"
  authors: "Moonshot AI (Kimi Team)"
  venue: "Moonshot AI launch blog, July 2026 (technical report pending)"
  url: "https://www.kimi.com/blog/kimi-k3"
---

> [!tldr]
> - **What it is.** Kimi K3 is a **2.8-trillion-parameter** open Mixture-of-Experts model with native vision and a **1-million-token** context window — the first open model in the "3T class." It activates **16 of 896 experts** per token and is built on two new attention pieces: **Kimi Delta Attention (KDA)** and **Attention Residuals (AttnRes)**.
> - **The one-line mechanism.** Replace most of the quadratic full-attention layers with cheap fixed-state linear-attention (KDA) layers, replace the flat residual sum with a learned softmax over depth (AttnRes), and push MoE sparsity far higher — then hold it all stable with a bundle of routing and optimizer fixes (**Stable LatentMoE**).
> - **Why it matters.** Moonshot claims about **2.5× better scaling efficiency than [Kimi K2](/blog/paper-reading/large-language-model/kimi-k2)** — more intelligence per unit of compute — and prices the API aggressively (\$0.30 / \$3.00 / \$15.00 per MTok) precisely because KDA plus prefix caching makes 1M-context serving cheap.
> - **Most surprising result.** A from-scratch coding agent: in a single 48-hour autonomous run, K3 designed a chip — 1.46M standard cells, 100 MHz timing closure, an INT4 MAC array — that runs a nano model built on K3's own architecture.
> - **Where it fails (their words).** K3 trails **Claude Fable 5** and **GPT 5.6 Sol** overall, is unstable if the harness drops its thinking history, is prone to "excessive proactiveness," and has a "noticeable gap in user experience" versus the top closed models.
> - **The honest caveat.** This is a **launch announcement, not a technical report**. The weights and full report land by **July 27, 2026**. Every architectural claim below is grounded in the blog plus Moonshot's *prior* papers on the same components; I flag where I extrapolate.

![The KDA layer: one fixed 128x128 recurrent state, updated by a per-channel gate and a delta write, replaces a KV cache that grows with sequence length](/imgs/blogs/kimi-k3-1.webp)

The figure above is the whole cost argument in one picture. A standard attention layer keeps a key–value cache that grows with every token, so a million-token context costs a million tokens' worth of memory and re-reads. A KDA layer keeps **one fixed-size state** and folds each new token into it. The rest of this post unpacks that idea, the three others stacked next to it, and whether the benchmark numbers earn the "frontier" label Moonshot puts on them.

A note on sourcing before we start. Kimi K3 shipped on **July 16, 2026** as a product launch: a blog post, an API, and a promise of open weights by July 27. There is no technical report yet. That is unusual for a paper-reading post, so I have done two things. First, every *number* — parameter count, expert count, benchmark score, price — comes verbatim from [the launch blog](https://www.kimi.com/blog/kimi-k3). Second, every *mechanism* is grounded in the papers Moonshot already published on the exact same components: [Kimi Linear](/blog/paper-reading/large-language-model/kimi-linear) for KDA, [Attention Residuals](/blog/paper-reading/large-language-model/attention-residuals) for AttnRes, [Gated Delta Networks](/blog/paper-reading/large-language-model/gated-delta-networks) for the delta rule, and [Muon / Moonlight](/blog/paper-reading/large-language-model/muon-moonlight) for the optimizer. Where I reach past what Moonshot has actually stated for K3, I say so.

## The problem: three walls between you and a 3-trillion-parameter model

Scaling a dense-ish transformer from the hundred-billion regime to the trillions is not one problem. It is three, and each one bites harder the bigger you go.

**Wall one: attention is quadratic, and context is now huge.** A full-attention layer compares every token against every other token. That is $O(T^2)$ work and, at decode time, an $O(T)$ key–value cache that you re-read on every single step. At a 1M-token context — which K3 targets for agentic, long-horizon coding — that cache dominates both memory and latency. You cannot serve a trillion-parameter model at a competitive token price if every layer re-reads a million-token cache per generated token. Something has to give.

**Wall two: very deep residual streams dilute their own layers.** Every modern LLM since GPT-2 uses *pre-normalization* residuals: `x = x + f(norm(x))`. The input to layer $\ell$ is literally the unit-weight sum of every earlier layer's output. That sum grows in norm roughly like $\sqrt{L}$ with depth $L$, and the relative contribution of any single deep layer shrinks toward ${1/L}$ — it gets *diluted*. Push a model deep enough and each new layer buys you less. For a model meant to "scale well beyond the trillion-parameter regime," a flat residual sum is a structural tax.

**Wall three: extreme MoE sparsity makes training unstable.** Mixture-of-Experts is how you get trillions of *parameters* without trillions of *FLOPs*: each token is routed to a small subset of experts. But the sparser you make it — K3 activates only **16 of 896** experts, about 1.8% — the more the whole system depends on the router doing something sane. Routers collapse (a few experts hog all the tokens), load balancing turns into a fragile hyperparameter search, and the optimizer struggles to give hundreds of rarely-visited experts a stable gradient signal. At 2.8T parameters, "the router got weird at step 400k" is a very expensive sentence.

K3's architecture is, one-for-one, an answer to these three walls: **KDA** for wall one, **AttnRes** for wall two, and **Stable LatentMoE** for wall three. Then two more pieces — **MXFP4/MXFP8 quantization-aware training** and a **balanced expert-parallel serving stack** — turn the trained model into something you can actually run at \$3/MTok. Let us take them in order.

## What Kimi K3 actually claims

Stripped of the launch-day adjectives, here is the contribution list I read out of the blog:

1. **A 2.8T-parameter open MoE** with native vision and a 1M-token context — "the first open model to reach 2.8 trillion parameters," with weights promised by July 27, 2026.
2. **Two attention-stack changes**: Kimi Delta Attention (linear, fixed-state) as the workhorse, and Attention Residuals (a learned combination across depth) replacing the flat residual.
3. **Higher MoE sparsity, held stable**: 16-of-896 routing inside a "Stable LatentMoE" framework with four named stabilizers — Quantile Balancing, Per-Head Muon, SiTU, and Gated MLA.
4. **Low-precision by construction**: quantization-aware training from the SFT stage onward, using MXFP4 weights and MXFP8 activations.
5. **A serving story that makes the price real**: balanced expert-parallel training with static shapes, supernode deployment, a KDA-aware prefix cache contributed to vLLM, and Mooncake-style disaggregated inference achieving a >90% cache-hit rate on coding workloads.

The headline number that ties them together: an **approximate 2.5× improvement in overall scaling efficiency compared to Kimi K2** — meaning K3 converts a unit of training compute into roughly 2.5× as much benchmark performance as its predecessor did. Everything below is, in effect, where that 2.5× is supposed to come from.

## The architecture

### Kimi Delta Attention: fold the past into one fixed-size state

**The problem it solves.** Full attention's cost is the growing KV cache (wall one). Linear attention has been the obvious escape hatch for a decade: replace the softmax with a kernel, and attention collapses into a *recurrent state update* — a fixed-size matrix you carry forward token by token, $O(T)$ total work and $O(1)$ memory per step. The catch has always been quality: a finite-state memory cannot losslessly store an arbitrarily long context, so pure linear-attention models fail at exact recall and copying. KDA is Moonshot's attempt to keep the cheap recurrence while clawing back most of the quality.

**Intuition.** Think of the state as a small whiteboard with a fixed number of slots. Each new token does three things to the whiteboard: it **fades** what is already written (older, less-relevant marks dim), it **writes** its own key→value association into the slot its key points at — overwriting whatever was there rather than scribbling on top — and later, a query **reads** the slot its query points at. The magic is entirely in *how selectively it fades*: instead of dimming the whole board by one global factor, KDA fades each of the 128 dimensions at its own learned rate. A slot holding a long-lived fact (a variable name you will reference 50k tokens later) can stay bright while a slot holding a throwaway detail (the current indentation level) fades fast.

**The mechanism, step by step.** For each token $t$, the layer produces five things from the hidden state: a query $q_t$, a key $k_t$, a value $v_t$, a per-channel gate $\alpha_t$, and a scalar write rate $\beta_t$. It then updates a per-head state matrix $S_t$: multiply the old state by the channel-wise gate (the fade), subtract out whatever value the current key already retrieves (so the write *overwrites*), add the new key→value association scaled by $\beta_t$ (the write). Finally the query reads the updated state to produce the output. That is the entire layer — no attention matrix, no cache that grows.

**The math.** Let the per-head state be

$$
S_t \in \mathbb{R}^{d_k \times d_v}, \qquad d_k = d_v = 128,
$$

a matrix that is the *same size at token 1 and at token 1,000,000*. With key $k_t \in \mathbb{R}^{d_k}$, value $v_t \in \mathbb{R}^{d_v}$, per-channel gate $\alpha_t \in (0,1)^{d_k}$, and write rate $\beta_t \in (0,1)$, the KDA recurrence is the **gated delta rule**:

$$
S_t = \mathrm{Diag}(\alpha_t)\, S_{t-1} \;+\; \beta_t\,\bigl(v_t - S_{t-1}^\top k_t\bigr)\, k_t^\top .
$$

Read it left to right. $\mathrm{Diag}(\alpha_t)$ is a $128\times128$ diagonal matrix whose $i$-th entry is the forgetting rate for channel $i$ — this is the *per-channel* fade, and it is the single change that separates KDA from its scalar-gated predecessor. The term $\bigl(v_t - S_{t-1}^\top k_t\bigr)$ is the **delta**: $S_{t-1}^\top k_t$ is the value the state currently associates with $k_t$, so subtracting it means we write the *correction*, overwriting the old slot instead of accumulating on top of it. The output is a straight read:

$$
o_t = S_t^\top q_t \in \mathbb{R}^{d_v}.
$$

Every symbol here is per-head; a real layer runs many heads in parallel and concatenates their $o_t$. The point of the shapes: nothing in these equations depends on $t$. The state is $128\times128$ forever, which is exactly why memory and per-step compute stay flat as context grows.

There is a deeper reason KDA is fast, worth one sentence because it is where the efficiency lives. The general "diagonal-plus-low-rank" (DPLR) recurrence is $S_t = (D - a_t b_t^\top)S_{t-1} + k_t v_t^\top$ for arbitrary low-rank vectors $a_t, b_t$; it is expressive but its hardware kernel is slow and numerically delicate. KDA is the special case that sets $a_t = b_t = k_t$, which — as the [Kimi Linear paper](/blog/paper-reading/large-language-model/kimi-linear) shows — removes several chunked matmuls and a fragile log-domain step, running roughly 2× faster than the general kernel while staying faithful to the classical delta rule.

**A worked micro-example.** Take a toy head with $d_k = d_v = 2$, start from an empty state $S_0 = \begin{smallmatrix}0&0\\0&0\end{smallmatrix}$, and set $\beta_1 = 1$, $\alpha_1 = (1, 1)$ (no fade yet). Token 1 writes key $k_1 = (1,0)$, value $v_1 = (5,5)$. The delta is $v_1 - S_0^\top k_1 = (5,5) - (0,0) = (5,5)$, so

$$
S_1 = 0 + (5,5)^\top (1,0) = \begin{pmatrix} 5 & 5 \\ 0 & 0 \end{pmatrix}.
$$

Now a query $q = (1,0)$ reads $o = S_1^\top q = (5,5)$ — it recovers the value stored under that key. Token 2 arrives with the *same* key $k_2 = (1,0)$ but a new value $v_2 = (1,1)$, still $\beta_2 = 1$. The delta is $v_2 - S_1^\top k_2 = (1,1) - (5,5) = (-4,-4)$, and

$$
S_2 = \begin{pmatrix} 5 & 5 \\ 0 & 0 \end{pmatrix} + (-4,-4)^\top(1,0) = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}.
$$

The slot got *overwritten*, not summed — reading with $q=(1,0)$ now returns $(1,1)$, the fresh value. That single subtraction is the whole difference between a hash table that updates cleanly and a running sum that smears every write on top of the last.

In pseudo-PyTorch the recurrence is almost embarrassingly short:

```python
import torch

def kda_step(S, q, k, v, alpha, beta):
    # S:     (H, d_k, d_v)  fixed-size state per head — never grows with T
    # q,k:   (H, d_k)       query / key
    # v:     (H, d_v)       value
    # alpha: (H, d_k)       per-channel forget gate in (0, 1)
    # beta:  (H, 1)         scalar write rate in (0, 1)
    retrieved = torch.einsum("hkv,hk->hv", S, k)        # what S already stores for k
    delta     = (v - retrieved).unsqueeze(-2)           # (H, 1, d_v)  the correction
    S = alpha.unsqueeze(-1) * S \
        + beta.unsqueeze(-1) * (k.unsqueeze(-1) @ delta)  # fade, then overwrite
    o = torch.einsum("hkv,hk->hv", S, q)                # read with the query
    return S, o
```

Training does not actually loop token by token — it uses a chunked, matmul-heavy kernel so the work lands on Tensor Cores — but the loop above is the ground truth the kernel reproduces.

**Why it works / when it fails.** It works because a fixed-size state gives you constant memory and linear total compute, and the per-channel gate plus delta rule give you enough selectivity to survive exact-recall tasks that sink cruder linear-attention. It *fails* on the thing a finite state fundamentally cannot do: losslessly retrieve an arbitrary token from 500k tokens ago. That is a capacity limit, not a tuning problem — which is precisely why K3 does not use KDA alone.

### The 3:1 hybrid: keep a few full-attention layers on purpose

**The problem it solves.** KDA's failure mode — imperfect long-range retrieval — is exactly what full attention is good at. So don't choose. K3 interleaves them.

![K3's attention stack interleaves three cheap KDA layers per full-attention MLA layer, so only one layer in four pays the quadratic long-range cost](/imgs/blogs/kimi-k3-3.webp)

**The mechanism.** As the stack diagram shows, K3 alternates blocks: a majority of **KDA layers** (linear, fixed-state, carrying local memory) with a sparse minority of full **Multi-Head Latent Attention (MLA)** layers (quadratic, KV-cached, providing exact global recall). The [Kimi Linear](/blog/paper-reading/large-language-model/kimi-linear) work — which K3's architecture descends from directly — established a **3:1 ratio**: three KDA layers per one MLA layer, applied uniformly through depth. That means only one layer in four pays the $O(T^2)$ price; the other three run in $O(T)$.

The blog does not restate the exact K3 ratio (that will be in the report), but it states the design principle plainly: "KDA provides an efficient foundation for scaling attention, while AttnRes selectively retrieves representations across depth." The economic consequence is the whole reason the price is competitive. If $f$ is the fraction of full-attention layers, the decode-time KV cache shrinks by roughly $(1-f)$ relative to an all-attention model — at a 3:1 ratio, that is a ~75% reduction, which is the number Kimi Linear reported. Fewer bytes re-read per token is fewer dollars per token.

**Why it works / when it fails.** The hybrid works because the two layer types are *complementary*: KDA handles the cheap, high-frequency job of local recency, and the rare MLA layers preserve the long-range information flow that a finite state loses. It fails — or at least stops helping — if you push $f$ too low: drop the full-attention layers entirely and you are back to a pure linear model that cannot retrieve. The ratio is a dial, and 3:1 is where the prior work found the best quality-per-FLOP.

### Attention Residuals: stop adding every layer with equal weight

**The problem it solves.** Wall two — PreNorm dilution. The residual stream adds every layer's output with a fixed weight of 1, so the stream grows without bound and deep layers get drowned out.

![Attention Residuals replaces PreNorm's fixed-weight residual sum with a learned, input-dependent softmax over all preceding layer outputs](/imgs/blogs/kimi-k3-2.webp)

**Intuition.** The residual stream is a shared notebook that every layer reads from and writes to. PreNorm says: *when a new layer reads the notebook, it sees the plain sum of everything written so far, with no sense of which past entries matter for the token in front of it.* AttnRes says: *let the layer decide, per token, how much of each past entry to pull in* — using the one primitive we already trust for "decide how much of each thing to attend to": softmax attention. The twist is that it runs attention along the **depth** axis (over layers) rather than the usual **sequence** axis (over tokens).

**The mechanism.** Each layer $\ell$ gets a single learned vector — a "pseudo-query" $w_\ell$. To form its input, the layer scores every preceding layer's output $O_j$ against $w_\ell$, softmaxes those scores into weights, and takes the weighted combination. PreNorm falls out as the special case where all the weights are equal.

**The math.** Let $O_j$ be the output of layer $j$ and $w_\ell \in \mathbb{R}^{d}$ the pseudo-query for layer $\ell$. The depth-attention weight from layer $\ell$ back to layer $j$ is

$$
a_{\ell j} = \frac{\exp\!\bigl(w_\ell^\top O_j / \sqrt{d}\bigr)}{\sum_{i \,\lt\, \ell} \exp\!\bigl(w_\ell^\top O_i / \sqrt{d}\bigr)}, \qquad j \,\lt\, \ell,
$$

and the input to layer $\ell$ is the softmax-weighted combination

$$
\tilde h_\ell = \sum_{j \,\lt\, \ell} a_{\ell j}\, O_j .
$$

Here $d$ is the model width and the $\sqrt{d}$ divisor is the usual scaling that keeps the dot products from saturating the softmax. Compare this to PreNorm, whose input is $\sum_{j \,\lt\, \ell} O_j$ — identical except every $a_{\ell j}$ is forced to 1. Because the weights now sum to 1 and depend on the actual activations of *this* token, the hidden-state norm stays controlled with depth and the gradient reaching each layer stays uniform.

**The catch, and the fix.** Computing $a_{\ell j}$ for all pairs $(\ell, j)$ is $O(L^2)$ in depth — fine for a demo, a real cost at 60+ layers. The AttnRes paper's shippable version, **Block AttnRes**, attends over a handful (~8) of *block summaries* instead of all $L$ individual layers, recovering almost all the gain at $O(Nd)$ memory and marginal wall-clock overhead. Moonshot's own framing for K3 is the same: AttnRes "selectively retrieves representations across depth rather than accumulating them uniformly," at — per Moonshot's statements around the launch, though not in the blog body itself — roughly **25% higher training efficiency at under 2% additional cost**. Treat that specific number as reported-not-in-the-blog until the report lands.

**Why it works / when it fails.** It works because it turns the single most rigid component in a transformer — the `+` in the residual — into something learned and input-dependent, directly attacking the dilution that caps very deep models. It could fail if the depth-softmax collapses (every layer attends only to itself, recovering a degenerate stream) or if the block approximation is too coarse for a given depth; the paper's ablations suggest ~8 summaries is enough, but 2.8T is deeper water than the 48B model it was validated on.

### Stable LatentMoE: 16 of 896 experts, held together by four fixes

**The problem it solves.** Wall three. K3 activates only 16 of 896 experts per token. At that sparsity — about 1.8% — routing and optimization stop being background details and become the thing most likely to blow up your run.

![Stable LatentMoE pairs a 16-of-896 sparse router with four stabilizers, each removing a different instability: routing, optimization, activations, and attention](/imgs/blogs/kimi-k3-4.webp)

**The intuition.** A sparse MoE is a giant company where every incoming task is routed to 16 of 896 specialists. Three things reliably go wrong at scale: a few star specialists get every task while most sit idle (load imbalance); the training signal to rarely-picked specialists is so noisy they never improve (optimization); and the shared plumbing — activations and attention — drifts as the whole thing gets bigger. Stable LatentMoE is four targeted fixes, one per failure, shown fanning into the stable-training outcome in the figure above.

**The four stabilizers.**

- **Quantile Balancing.** The usual load-balancing trick adds an auxiliary loss with a hand-tuned coefficient that you pray is right. Quantile Balancing instead "derives expert allocation directly from router-score quantiles" — it reads where a token's router score falls in the distribution of scores and allocates from that, "eliminating heuristic updates and a sensitive balancing hyperparameter." In plain terms: it balances load by construction rather than by a fragile penalty term.
- **Per-Head Muon.** [Muon](/blog/paper-reading/large-language-model/muon-moonlight) is Moonshot's matrix-aware optimizer, already shown to scale on Moonlight. Per-Head Muon "extends Muon by optimizing attention heads independently," giving each head its own adaptive update instead of one shared schedule — more stable learning when hundreds of heads and experts each see very different gradient statistics.
- **SiTU (Sigmoid Tanh Unit).** An activation function that "improves activation control" — a gated nonlinearity in the spirit of SwiGLU/GeGLU, tuned to keep activation magnitudes in a trainable band at this scale.
- **Gated MLA.** A gate on the Multi-Head Latent Attention path that "improves attention selectivity," i.e. lets attention layers sharpen or suppress their output rather than passing everything through.

**A little math on the sparsity.** With $E = 896$ experts and $k = 16$ active per token, the activation ratio is

$$
\rho = \frac{k}{E} = \frac{16}{896} \approx 0.0179,
$$

so under 2% of the expert parameters fire on any given token. This is the whole MoE bargain: total capacity scales with $E$, but per-token FLOPs scale with $k$. The blog does not state K3's *active* parameter count; by analogy to K2's roughly 32B-active design and this sparsity, a few tens of billions of parameters are live per token — but I am extrapolating, and the report will have the real figure. The reason the four fixes matter is that as $\rho$ shrinks, the variance in how often each expert is trained *grows*, and without balancing and per-head optimization, the rarely-visited experts never converge.

**Why it works / when it fails.** It works if — and this is the load-bearing "if" — the four fixes actually remove the four instabilities they target, so you can run the sparse router hard without a collapse. It fails the way all aggressive-sparsity MoEs can fail: a distribution shift mid-training (say, when the data mix changes for a new stage) can still knock the router off its learned quantiles, and no amount of balancing saves you from experts that specialized on data you have stopped showing. This is the piece I most want to see ablated in the report.

### MXFP4 weights, MXFP8 activations: low precision on purpose

**The problem it solves.** A 2.8T-parameter model in 16-bit weights is ~5.6 TB — a serving and memory nightmare. You want it in 4-bit. But naively quantizing a model *after* training (post-training quantization) at 4 bits usually wrecks accuracy. So K3 trains *knowing* it will be low-precision.

**The intuition.** "Microscaling" (MX) formats are block floating point: instead of one scale for a whole tensor, you give every small block of values (typically 32 elements) its own shared exponent/scale, then store each element in very few bits. A block whose values are all tiny gets a small scale; a block with big values gets a large one. The shared scale absorbs the dynamic range so 4 bits per element is enough to capture the *relative* differences that matter. K3 uses **MXFP4 for weights** (4-bit elements, shared block scale) and **MXFP8 for activations** (8-bit, because activations have wilder outliers than weights).

**The mechanism, and why QAT.** Quantization-aware training (QAT) simulates the low-precision rounding in the forward pass while keeping higher-precision master weights for the gradient, so the model *learns* weights that are robust to being rounded. Crucially, K3 applies QAT "from the SFT stage onward" — not the whole pretraining, but from supervised fine-tuning through the end — so the final weights are shaped to the format they will ship in. For a block of weights $w$ with a shared scale $s$ (itself a low-bit float), the simulated forward value is

$$
\hat w = s \cdot \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{w}{s}\right),\, -q_{\max},\, q_{\max}\right),
$$

where $q_{\max}$ is the largest magnitude the 4-bit mantissa can represent. The backward pass uses a straight-through estimator — gradients flow as if $\hat w = w$ — so training can still move the master weights even though $\mathrm{round}$ has zero gradient almost everywhere.

**Why it works / when it fails.** It works because QAT lets the model pre-adapt to rounding error instead of being ambushed by it at deployment, and MX block scaling keeps 4-bit weights within tolerance across "broad hardware compatibility" (the blog's phrasing — MX formats are a cross-vendor standard). It fails if activation outliers exceed what MXFP8 blocks can absorb, which is exactly why weights get 4 bits and activations get 8 — an asymmetry that is itself a tell about where the numerical danger lives.

### The serving stack: why the price is what it is

The architecture would be academic if K3 cost \$30/MTok to serve. It does not — it is \$0.30 (cache-hit input) / \$3.00 (cache-miss input) / \$15.00 (output) per million tokens. Four infrastructure moves make that possible, and they are worth naming because they are where a lot of the real engineering went:

- **Fully balanced expert-parallel training** with "static shapes and no host synchronization on the critical path." Sparse MoE routing normally produces *ragged* per-expert batch sizes (some experts get many tokens, some few), which forces dynamic shapes and CPU↔GPU syncs that stall the pipeline. Forcing static shapes removes the stall — at the cost of some padding — and keeps throughput high even when the router is imbalanced.
- **Supernode deployment (64+ accelerators).** Because inference benefits from "larger high-bandwidth communication domains," Moonshot recommends running K3 on nodes of 64 or more accelerators — the expert-parallel all-to-all traffic wants fat, low-latency interconnect.
- **A KDA-aware prefix cache in vLLM.** Prefix caching (reusing the compute for a shared prompt prefix across requests) is standard for attention, but "KDA poses new challenges for conventional prefix caching" — a recurrent state is not a simple appendable KV cache, so you cannot naively slice and reuse it. Moonshot contributed an implementation to vLLM so KDA can still cache prefills, which is what "allows us to serve Kimi K3 at a highly competitive token price despite its scale and long context."
- **Mooncake disaggregated inference.** The official API runs on [Mooncake](/blog/paper-reading/large-language-model/mooncake), Moonshot's KV-cache-centric disaggregated architecture, and hits a **>90% cache-hit rate on coding workloads** — which is exactly why the *cache-hit* input price (\$0.30) is 10× cheaper than the cache-miss price (\$3.00). Coding agents re-send the same repository context constantly; a high cache-hit rate turns that repetition into savings.

## Experiments & results

Moonshot is refreshingly direct about the top line: K3's "overall performance still trails the most powerful proprietary models, Claude Fable 5 and GPT 5.6 Sol," while "consistently outperforming other tested models." The benchmark table backs a more precise version of that claim — K3 tracks the closed frontier within a few points, leading on some axes and trailing on others.

![On headline coding, agentic, and knowledge benchmarks, K3 lands within a few points of Claude Fable 5 and GPT 5.6 Sol](/imgs/blogs/kimi-k3-5.webp)

Here is the full table as reported, all six models. K3 is run at `max` reasoning effort, temperature 1.0, top-p 1.0; each model uses whichever agentic harness (KimiCode, Claude Code, or Codex) its footnotes specify. An asterisk marks scores the blog flags with methodology caveats.

| Benchmark | Kimi K3 (max) | Claude Fable 5 (max, w/ fallback) | GPT 5.6 Sol (max) | Claude Opus 4.8 (max) | GPT 5.5 (xhigh) | GLM-5.2 (max) |
|---|---|---|---|---|---|---|
| **Coding** | | | | | | |
| DeepSWE | 67.5 | 70.0 | **73.0** | 59.0 | 67.0 | 46.2 |
| Program Bench | **77.8** | 76.8 | 77.6 | 71.9 | 70.8 | 63.7 |
| Terminal-Bench 2.1 | 88.3 | 84.6 | **88.8** | 84.6 | 83.4 | 82.7 |
| FrontierSWE | 81.2 | **86.6** | 71.3 | 66.7 | 64.9 | 67.3 |
| SWE Marathon | **42.0** | 35.0 | 39.0 | 40.0 | 14.0 | 13.0 |
| PostTrain Bench | 36.6 | **41.4** | 34.6 | 34.1 | 28.4 | 34.3 |
| MLS Bench | 48.3 | **49.9** | 46.2 | 42.8 | 35.5 | 40.4 |
| Kimi Code Bench 2.0 (Internal) | 72.9 | **76.9** | 64.8 | 71.7 | 69.0 | 64.2 |
| **Agentic** | | | | | | |
| GDPval-AA v2 (Elo) | 1668 | **1760** | 1748 | 1600 | 1494 | 1514 |
| BrowseComp | **91.2** | 88.0 | 90.4 | 84.3 | 84.4 | — |
| DeepSearchQA (f1) | **95.0** | 94.2 | — | 93.1 | — | — |
| Toolathlon-Verified | 73.2 | **77.9** | 74.9 | 76.2 | 73.5 | 59.9 |
| MCP Atlas | 84.2 | **84.7** | 83.6 | 83.6 | 82.8 | 82.6 |
| Automation Bench | **30.8** | 29.1 | 29.7 | 27.2 | 22.7 | 12.9 |
| Job Bench | 52.9 | **57.4** | 46.5 | 48.4 | 38.3 | 43.4 |
| AA-Briefcase (Elo) | 1548 | **1583** | 1495 | 1354 | 1158 | 1260 |
| APEX-Agents | 37.6 | **43.3** | 39.9 | 39.4 | 38.5 | 35.6 |
| Office QA Pro | 63.3 | **69.9\*** | 63.2\* | 63.9\* | 60.9\* | 41.4 |
| SpreadsheetBench 2 | **34.8** | 34.7\* | 32.4\* | 31.6\* | 29.1\* | 28.1 |
| DECK-Bench (Internal) | 73.5 | 73.0 | **74.7** | 66.9 | 68.2 | 68.6 |
| **Reasoning & Knowledge** | | | | | | |
| GPQA-Diamond | 93.5 | 92.6 | **94.1** | 91.0 | 93.5 | 91.2 |
| HLE-Full | 43.5 | **53.3** | 44.5 | 49.8\* | 41.4\* | — |
| HLE-Full w/ tools | 56.0 | **63.0** | 58.0 | 57.9\* | 52.2\* | — |
| **Vision** | | | | | | |
| MMMU-Pro | 81.6 | 81.2 | **83.0** | 78.9 | 81.2 | — |
| CharXiv (RQ) | 84.8 | **88.9** | 84.6 | 80.5 | 84.1 | — |
| MathVision | 94.3 | 94.8 | **95.8** | 86.7 | 92.2 | — |
| MathVision w/ python | 97.8 | **98.6** | 97.8 | 97.1 | 96.8 | — |
| ZeroBench_main (pass@5) | 23.0 | 23.0 | 17.0 | 17.0 | 22.0 | — |
| WorldVQA ForceAnswer | 51.0 | **56.7** | 41.8 | 39.1 | 38.5 | — |
| OmniDocBench | **91.1** | 89.8 | 85.8 | 87.9 | 89.4 | — |
| PerceptionBench | 58.5 | 57.2 | **59.7** | 47.2 | 55.8 | — |

A derived read of the coding block, since it is the headline capability: across the eight coding benchmarks, K3 wins outright on **Program Bench (77.8)**, **SWE Marathon (42.0)**, and by the footnotes leads several others, but trails Fable 5 on FrontierSWE (81.2 vs 86.6, a 5.4-point gap) and on the two internal-ish suites. Averaging the eight, K3 and the two closed leaders sit within roughly 2–3 points of each other — which is the quantified version of "tracks the frontier." The single widest gap in the whole table is **HLE-Full**, the hardest closed-form reasoning eval: 43.5 for K3 versus 53.3 for Fable 5, a **9.8-point** shortfall that no amount of tool use fully closes (56.0 vs 63.0 with tools). If you want the honest one-sentence summary of where K3 is weakest, it is frontier reasoning, not coding.

**What is load-bearing in their setup that might not transfer.** Three things to hold in mind before treating these numbers as settled:

1. **Max effort by default.** "All Kimi K3 results are obtained with the reasoning effort set to `max`." Low- and high-effort modes "are to be introduced in subsequent updates." So the numbers reflect K3 thinking as hard as it can; the cheaper modes you would actually deploy at scale are not benchmarked here.
2. **Harness asymmetry.** K3 is frequently run under Moonshot's own **KimiCode** harness while competitors run under Claude Code or Codex. Agentic scores are notoriously harness-sensitive — the same model can swing several points on a different scaffold — so a couple of K3's coding wins may be as much about KimiCode as about the model.
3. **Internal and in-house benchmarks.** Kimi Code Bench 2.0, DECK-Bench, and PerceptionBench are Moonshot's own. They may be perfectly fair, but they are not independently reproducible the way DeepSWE or GPQA are, and they should carry less weight in your mental model until the report and weights let others re-run them.

## What K3 built: the case studies

The benchmark table is the *that*; the case studies are the *so what*. These are the parts of the launch that are hardest to fake and, honestly, the most impressive — long-horizon, open-ended builds that a benchmark score cannot capture. Two of the artifacts below are shown exactly as Moonshot published them.

**GPU kernel optimization.** In an identical sandbox with up to 24 hours per task, K3 optimized four GPU kernels — spanning AttnRes, KDA, and a 512-head-dimension MLA kernel — across NVIDIA H200 and a general-purpose GPU from an alternative vendor. Moonshot reports K3 "performed competitively with Fable 5 (with fallback) and substantially outperformed Opus 4.8, GPT 5.6 Sol, and GPT 5.5." The tell that this is real and not a demo: "in the late stages of Kimi K3 development, an early version of Kimi K3 handled the majority of the team's kernel optimization works." The model was optimizing the kernels for its own training.

**MiniTriton: a compiler from scratch.** K3 built **MiniTriton**, a compact Triton-like GPU compiler with "its own tile-level IR layer over MLIR, optimization passes, and a PTX code-generation pipeline." On supported roofline benchmarks it matches or beats Triton and `torch.compile`, and — the part that separates a compiler from a pile of kernels — it "sustains end-to-end nanoGPT training with stable convergence, the loss curve closely tracking the reference." Building a coherent DSL frontend → IR → PTX codegen → runtime is a systems task most human teams measure in months.

**A 3D open world, built with vision in the loop.** K3 built a fully procedural browser 3D exploration game on Three.js WebGPU with GPU compute, generating the terrain procedurally and using a 3D asset tool for the rider and horse. What makes this more than a coding demo is the workflow Moonshot calls "vision in the loop": K3 iterates between writing code and looking at live screenshots of the result, seeing and refining its own output. The screenshot below is that game.

![From the Kimi K3 blog (Moonshot AI): a procedural browser 3D open world — forests, a log-cabin village, snowy mountains, and dynamic weather — with a rider and horse, built with "vision in the loop"](/imgs/blogs/kimi-k3-fig1.webp)

**A chip, designed by a model, for a model.** This is the launch's most striking claim. In a single 48-hour autonomous run, K3 "built, optimized, and verified" a chip using open-source EDA tools on the Nangate 45nm library. Within 4 mm², the design closes timing at 100 MHz and sustains over 8,700 tokens/s decode throughput in simulation, packing 1.46M standard cells, 0.277 MB of SRAM, and an INT4 MAC array with fused dequantization. The chip serves a nano model built on K3's own architecture — a full loop from architecture to silicon, run by the model that the silicon runs.

**Two hours of astrophysics.** To reproduce the I–Love–Q universal relations (a result in neutron-star physics), K3 "reviewed and cross-validated 20+ papers, implemented the full numerical pipeline, evaluated 300+ equations of state, identified inconsistencies in published formulas, generated 3,000+ lines of Python code, and produced an interactive HTML dashboard" — in about two hours, versus the one-to-two weeks Moonshot estimates for an experienced researcher. The general-relativity black-hole renderer below is from the same family of physics-heavy artifacts on the launch page: a real-time null-geodesic ray-tracer with a Schwarzschild-metric HUD.

![From the Kimi K3 blog (Moonshot AI): a real-time general-relativity ray-tracer of a supermassive black hole ("Gargantua"), with a HUD reporting the Schwarzschild metric, observer distance, and geodesic step count](/imgs/blogs/kimi-k3-fig2.webp)

**Interactive research at scale.** In Kimi Work, K3 produced a 42-year interactive history of the AI-ASIC industry through "120+ rounds of recursive self-improvement," pulling data via 2,800+ web searches and 1,100+ terminal data pulls across 11,000+ pages, 87 quarterly reports, and 99 PDFs. A separate case ran a GWTC-5 gravitational-wave analysis of 391 events using 20+ concurrent subagents. Whatever you make of the benchmarks, the through-line of the case studies is *duration*: K3's pitch is that it can hold a coherent goal across dozens of hours and thousands of tool calls.

## Critique

**What is genuinely strong.** The architecture is not a scaling stunt; it is a coherent, mutually reinforcing set of answers to real problems. KDA attacks serving cost, AttnRes attacks depth dilution, Stable LatentMoE attacks routing instability, and each is grounded in a prior Moonshot paper with controlled experiments — this is not the first time the world has seen KDA or AttnRes. The willingness to state, on launch day, that the model trails Fable 5 and GPT 5.6 Sol and has a "noticeable gap in user experience" is more credible than the usual we-beat-everyone launch. And committing to open weights at 2.8T is a genuinely large gift to the field, whatever the benchmarks say.

**What is weak, unfalsifiable, or cherry-picked.** The elephant: **this is an announcement, not a report.** Every architectural claim — the 2.5× scaling efficiency, the four LatentMoE stabilizers, the MXFP4 QAT recipe — is currently unfalsifiable because there is no report and no weights to inspect. The **2.5× scaling-efficiency** figure in particular is doing enormous work and comes with no methodology: 2.5× measured how, against which K2 checkpoint, on what axis? The **benchmark harness asymmetry** (KimiCode for K3, other harnesses for competitors) means some coding wins are partly attributable to the scaffold, and the **max-effort-only** results describe a configuration you may never actually deploy. Several headline benchmarks are **internal**. And the case studies, while impressive, are curated best-of runs — we see the chip that closed timing, not the ten that did not.

**What ablation is missing.** The one I most want: an apples-to-apples **KDA-vs-full-attention** ablation *at K3 scale*, on long-context recall specifically, because a finite state's failure mode is exactly long-range retrieval and 2.8T is far past where Kimi Linear validated the 3:1 ratio. Second: a **routing-stability** ablation for Stable LatentMoE showing that all four fixes are load-bearing and not just decoration. Third: the **quantization tax** — how much accuracy does MXFP4 QAT cost versus a bf16 twin? A 4-bit model that is "as good as the report says" and a 4-bit model that quietly gave up two points on hard reasoning are very different products.

**What would change my mind.** I would move from "promising, provisionally believe the framing" to "this is a genuine step at the frontier" if, when the weights and report drop on July 27, an independent third party reproduces the coding numbers **under a neutral harness** (not KimiCode), and the report shows the KDA long-context recall ablation and the MXFP4-vs-bf16 gap at scale. Conversely, I would sharply discount the launch if the open weights underperform the reported numbers by more than a couple of points under neutral evaluation, or if the 2.5×-scaling-efficiency claim turns out to rest on a favorable choice of K2 baseline or metric.

## What I'd build with this

These are my extrapolations, not Moonshot's claims — directions the architecture makes newly cheap.

1. **A KDA-native long-context retrieval probe.** Because a KDA layer's state is a fixed $128\times128$ matrix, you can literally *read out* what it stores at any point in the sequence. I would build a diagnostic that visualizes the per-channel gate $\alpha_t$ over a long document to see which channels the model keeps "sticky" for long-range facts versus which it fades — a concrete window into how a hybrid model decides what to remember.
2. **Effort-aware agent routing.** With low/high/max effort modes coming, an agent orchestrator could run K3 at `low` for the 90% of tool calls that are mechanical (read a file, run a test) and escalate to `max` only for the hard planning steps — capturing most of the quality at a fraction of the token cost, exactly the way the cache-hit pricing already rewards repetition.
3. **A cache-residency-aware coding harness.** Given the 10× gap between cache-hit (\$0.30) and cache-miss (\$3.00) input pricing, the winning move is to structure prompts so the expensive, stable context (the repository) stays cache-resident while only the cheap, volatile part (the current instruction) changes. A harness that deliberately maximizes prefix reuse could cut real coding-agent bills substantially.
4. **An open replication of Quantile Balancing.** Of the four LatentMoE fixes, Quantile Balancing is the most portable idea — a hyperparameter-free load balancer usable in any MoE. I would lift it into a small open MoE and measure whether it actually removes the auxiliary-loss coefficient as cleanly as claimed.

When the report and weights arrive, I will come back and mark which of the critiques above survived contact with the evidence. Until then, the fair summary is: Kimi K3 is the most architecturally interesting open model of the year so far, priced to be used, and honest about trailing the closed frontier — but its most important claims are, for the next ten days, promises rather than proofs.

## References

- **Kimi K3: Open Frontier Intelligence** — Moonshot AI (Kimi Team), launch blog, July 2026. [https://www.kimi.com/blog/kimi-k3](https://www.kimi.com/blog/kimi-k3). Weights and full technical report expected by July 27, 2026.
- **Kimi Linear: An Expressive, Efficient Attention Architecture** — the source of KDA and the 3:1 hybrid, with controlled experiments. [arXiv:2510.26692](https://arxiv.org/abs/2510.26692) · [code](https://github.com/MoonshotAI/Kimi-Linear). My close read: [Kimi Linear](/blog/paper-reading/large-language-model/kimi-linear).
- **Attention Residuals** — Kimi Team (arXiv:2603.15031). My close read: [Attention Residuals](/blog/paper-reading/large-language-model/attention-residuals).
- Related deep-dives on this blog: [Gated Delta Networks](/blog/paper-reading/large-language-model/gated-delta-networks) (the delta rule KDA inherits), [Muon / Moonlight](/blog/paper-reading/large-language-model/muon-moonlight) (the optimizer behind Per-Head Muon), [Kimi K2](/blog/paper-reading/large-language-model/kimi-k2) (the predecessor K3's 2.5× is measured against), and [Mooncake](/blog/paper-reading/large-language-model/mooncake) (the disaggregated serving stack behind the API price).
