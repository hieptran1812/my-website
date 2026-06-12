---
title: "Attention Residuals: Replacing the Residual Stream with Learned Depth-Wise Attention"
publishDate: "2026-06-10"
date: "2026-06-10"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  - attention-residuals
  - residual-connections
  - prenorm
  - transformer-architecture
  - kimi-linear
  - moonshot-ai
  - scaling-laws
  - deep-learning
description: "A close read of Attention Residuals (Kimi Team, arXiv:2603.15031): how replacing the fixed unit-weight residual sum with a softmax over preceding layer outputs cures PreNorm dilution, and how Block AttnRes makes it a near-zero-overhead drop-in inside Kimi Linear 48B."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/attention-residuals-1.png"
readTime: 30
---

> [!tldr]
> - **The claim:** the residual connection — `x = x + f(x)`, untouched since 2015 — quietly destabilizes deep transformers. PreNorm adds *every* layer's output into the stream with **fixed unit weight**, so the hidden state grows without bound and each new layer's voice is diluted. Attention Residuals (AttnRes) replaces that flat sum with a **softmax over all preceding layer outputs**, scored by one learned pseudo-query per layer.
> - **Why it matters:** it's a *drop-in* change to the single most universal component in modern LLMs. No new attention variant, no new optimizer — just a different rule for how layers read the residual stream. Integrated into [Kimi Linear](/blog/paper-reading/large-language-model/kimi-linear) 48B (3B activated) over 1.4T tokens, it lifts every downstream benchmark.
> - **Most surprising finding:** the naive version is O(L²) in depth, but **Block AttnRes** — attend over ~8 block summaries instead of all L layers — recovers almost all the gain at **O(Nd)** memory and *marginal* wall-clock overhead, matching a baseline trained with **1.25× more compute**.
> - **Where it fails:** the gains are reported at one model family and scale; the pseudo-query is a strikingly low-capacity router, and the paper leans on a "PreNorm dilution" story that is intuitive but only partially measured.

Open any modern transformer and you will find one line that nobody questions:

```python
x = x + sublayer(norm(x))   # PreNorm residual — the same since 2015
```

We have rewritten attention four times — multi-head, multi-query, grouped-query, multi-head latent. We have swapped optimizers ([Muon](/blog/paper-reading/large-language-model/muon-moonlight) for AdamW), swapped position encodings (RoPE for NoPE), swapped dense FFNs for [mixture-of-experts](/blog/paper-reading/large-language-model/kimi-k2). But the residual connection — the literal backbone that carries information from layer 1 to layer 61 — has been a fixed `+`. Attention Residuals, from Moonshot's Kimi Team (arXiv:2603.15031), is the first production-scale paper I have seen that takes that `+` seriously and asks: *why should every layer's contribution be added with exactly the same weight?*

![Why PreNorm residuals dilute deep layers](/imgs/blogs/attention-residuals-1.png)

The diagram above is the mental model. On the left is what PreNorm actually does: it sums all `L` layer outputs with unit weight, the hidden-state norm grows roughly like $\sqrt{L}$, and the relative contribution of any single deep layer shrinks toward $1/L$ — it gets *diluted*. On the right is AttnRes: a learned, input-dependent softmax decides how much of each earlier output flows into the next layer, so the norm stays controlled and the gradients stay uniform across depth. This post is a close read of *how* that works, *why* the naive version is too expensive, and the block trick that makes it shippable.

## Context: what came before

To see why this is more than a cute tweak, it helps to remember what the residual connection is *for*. He et al. introduced it in ResNet to fix the degradation problem: stacking more layers made deep networks *worse*, not because of overfitting but because the optimizer could not even drive training error down. The residual shortcut $h_\ell = h_{\ell-1} + f_\ell(h_{\ell-1})$ gave gradients a highway back to early layers, and suddenly 100+ layer networks trained.

Transformers inherited the idea and then made one consequential choice: **pre-normalization**. The original "Attention Is All You Need" used PostNorm — $h_\ell = \mathrm{LN}(h_{\ell-1} + f_\ell(h_{\ell-1}))$ — which is unstable at depth because the normalization sits *outside* the residual path and the raw sum can blow up before it is normalized. Every serious LLM since GPT-2 moved the LayerNorm *inside*:

$$h_\ell = h_{\ell-1} + f_\ell\!\left(\mathrm{LN}(h_{\ell-1})\right).$$

PreNorm trains beautifully. The clean identity path $h_\ell = h_{\ell-1} + (\dots)$ means gradients never have to pass through a nonlinearity to reach layer 0, and you can stack 60, 80, 100 layers without warmup gymnastics. This is why it won.

But unroll that recurrence and a second-order consequence falls out. Define $v_i = f_i(\mathrm{LN}(h_{i-1}))$ as the *output* of sublayer $i$, and $v_0 = x_0$ as the token embedding. Then

$$h_\ell = x_0 + \sum_{i=1}^{\ell-1} v_i = \sum_{i=0}^{\ell-1} v_i.$$

The input to layer $\ell$ is literally **the unit-weight sum of every earlier layer's output**. There is no decision about which earlier representations matter — they all get coefficient 1. That is the gap Attention Residuals claims to fill: PreNorm is not just *a* way to combine layer outputs, it is the *most rigid possible* way, and the paper argues that rigidity costs you accuracy at depth.

There is a measurable side of this story too. The instability that PostNorm suffers is not folklore — Xiong et al. (2020) showed that PostNorm's gradients at initialization scale like $O(\sqrt{L})$ at the top layers, which is why it needs learning-rate warmup, while PreNorm's are $O(1)$. The fix worked so well that the field largely stopped looking. But a parallel literature kept flagging that *something* about deep residual stacks degrades: the "rank collapse" / "token uniformity" results (Dong et al., 2021) show that pure attention without the residual+MLP path drives all token representations toward a single vector exponentially fast in depth, and the residual path is what holds that collapse off. AttnRes can be read as a sharper tool for the same job: if the residual stream is what fights collapse, then *how* you combine into the stream should matter, and a flat sum is a blunt instrument.

The list of partial fixes is long, which is itself a signal that the community senses the problem. **DeepNorm** (Wang et al., 2022) rescales the residual branch by a depth-dependent constant to train 1000-layer transformers — a fixed, hand-derived weight. **Admin** and **Fixup** carefully initialize the residual branches so the early-training signal does not explode — again, a one-time static choice. **LayerScale** (Touvron et al., 2021), **ReZero** (Bachlechner et al., 2021), and **SkipInit** all learn a *scalar* multiplier $\lambda_\ell$ on each residual branch: $h_\ell = h_{\ell-1} + \lambda_\ell\, f_\ell(\cdot)$. These help, and they are nearly free, but every one of them shares two limitations: the weight is the *same for every token and every input*, and it modulates only the *new* branch, never how much of the *accumulated history* you keep.

**DenseNet** (Huang et al., 2017) went the other direction in vision: wire every layer to every later layer and *concatenate*. That gives genuine per-layer access to history, but concatenation grows the channel count linearly with depth, which is hopeless for a 4096-dim residual stream at 60 layers. **Hyper-connections** (Zhu et al., 2024) is the closest prior art — it learns a small matrix of connection weights between an expanded set of residual streams — but the weights are still input-independent learned constants. AttnRes is the first to make the mixing weights both **per-pair** (one weight for each ordered pair of layers) *and* **input-dependent** (computed from the actual activations of this specific token), which is exactly what "attention" buys you. The novelty is reusing the attention primitive along the *depth* axis instead of the *sequence* axis — and, crucially, doing it cheaply enough to survive contact with a 48B production run.

## Contributions

The paper's stated contributions, tightened:

1. **A diagnosis: PreNorm dilution.** Standard PreNorm residuals accumulate all layer outputs with fixed unit weight, causing uncontrolled hidden-state growth with depth and progressively diluting each layer's contribution. The paper frames this as the structural reason very deep transformers see diminishing returns per layer.
2. **Attention Residuals (AttnRes).** Replace the flat sum with a softmax attention over preceding layer outputs, using a single learned **pseudo-query** $w_\ell \in \mathbb{R}^d$ per layer. PreNorm becomes the special case where all softmax weights are equal.
3. **Block AttnRes for practicality.** The naive form attends over all $L$ prior outputs — $O(Ld)$ memory and $O(L^2)$ compute. Block AttnRes partitions depth into $\sim 8$ blocks, accumulates normally inside a block, and attends only over **block-level summaries** across blocks, dropping memory to $O(Nd)$ with marginal overhead via a two-phase schedule and cache-based pipeline communication.
4. **Scaling-law evidence.** The improvement is consistent across model sizes, gradients and output magnitudes become more uniform across depth, and Block AttnRes matches a PreNorm baseline trained with **1.25× more compute** — integrated end-to-end into Kimi Linear 48B (3B activated) on 1.4T tokens, improving every evaluated downstream task.

## Method

### The mechanism: one learned query attends over depth {#attnres-mechanism}

Here is the whole idea in one equation. Where PreNorm sets the input to layer $\ell$ to the unit-weight sum $\sum_{i<\ell} v_i$, AttnRes makes it a *weighted* sum:

$$h_\ell = \sum_{i=0}^{\ell-1} \alpha_{i\to\ell}\, v_i, \qquad \alpha_{\cdot\to\ell} = \mathrm{softmax}_i\!\left(\langle w_\ell, v_i\rangle\right).$$

Read the symbols carefully because the roles are unusual:

- $v_i$ — the output of layer $i$. It plays the part of **both the key and the value**. The thing we score is the same thing we mix.
- $w_\ell \in \mathbb{R}^d$ — a single learned vector per layer, the **pseudo-query**. There is no query *projection* of an input token; the query is a free parameter. This is what makes AttnRes cheap: the attention has one query, not $T$ queries.
- $\alpha_{i\to\ell}$ — the softmax weight describing how much of layer $i$'s output flows into layer $\ell$. Because it is a softmax, $\sum_i \alpha_{i\to\ell} = 1$: the new residual input is a **convex combination** of past outputs, not an unbounded sum. That single fact — convexity — is what controls the norm growth.

![Attention Residuals: a learned query mixes depth](/imgs/blogs/attention-residuals-2.png)

The figure traces the dataflow: outputs $v_0, v_1, v_2, \dots, v_{\ell-1}$ are scored by the pseudo-query $w_\ell$; the softmax over depth produces a weight profile $\alpha_{i\to\ell}$ (which the paper finds is *non-uniform* and learned, not collapsed back to flat); and the weighted sum becomes $h_\ell$, the input the next layer actually reads. Note what is *not* here: there is no extra per-token projection, no KV cache along sequence, no quadratic-in-sequence cost. The attention is purely along the depth axis, and the only new parameters are the $L$ pseudo-query vectors — a rounding error against a 48B model.

It is worth sitting with why softmax normalization is the load-bearing part. PreNorm's pathology is that $\|h_\ell\| \approx \sqrt{\sum_{i<\ell}\|v_i\|^2}$ grows like $\sqrt\ell$ if the outputs are roughly decorrelated. The LayerNorm inside each sublayer rescales the *input* to $f_\ell$, but it cannot change the fact that a fresh output $v_\ell$ of norm $\sim 1$ is being dropped into a stream of norm $\sim\sqrt\ell$ — so its fractional contribution is $\sim 1/\sqrt\ell$ and falls with depth. By forcing $\sum_i\alpha_{i\to\ell}=1$, AttnRes keeps $h_\ell$ a convex blend whose norm is bounded by the largest $\|v_i\|$, and the *learned* weights let the model spend that fixed budget on whichever layers are actually informative for this input. The diagnosis and the cure are the same mechanism.

#### A worked example of dilution

Put numbers on it, because the intuition is worth making concrete. Assume each sublayer output has unit norm $\|v_i\| = 1$ and is roughly decorrelated from the others — a standard first-order model of a trained residual stream. Then at the input to layer $\ell$, the accumulated stream has norm $\|h_\ell\| \approx \sqrt{\ell}$. The *fraction* of that stream contributed by any single layer $i$ is $\|v_i\| / \|h_\ell\| \approx 1/\sqrt{\ell}$.

Walk it up the stack. At layer 4, a fresh output is $1/2 = 50\%$ of the stream's magnitude — it is *heard*. At layer 16 it is $1/4 = 25\%$. At layer 36 it is $1/6 \approx 17\%$. At layer 64 — roughly the depth of a frontier model — it is $1/8 = 12.5\%$. So by the time you reach the layers that are supposed to do the hardest integration, each one can only nudge the stream by an eighth of its own magnitude, and the other seven-eighths is inherited history that the layer cannot choose to discount. The network is structurally biased toward *keeping* what early layers wrote, regardless of whether it is still relevant.

Now contrast AttnRes. Because the weights are a softmax, $h_\ell = \sum_i \alpha_{i\to\ell} v_i$ with $\sum_i \alpha = 1$, the stream norm is bounded: $\|h_\ell\| \le \max_i \|v_i\| \approx 1$ regardless of depth. A layer that wants to *overwrite* the stream can place most of its mass on its own predecessor's output; a layer that wants to *reach back* to an early feature can put mass there directly, paying nothing for the 30 intervening layers. The dilution is gone not because the model works harder, but because the combination rule stopped being a sum and started being an average it controls. This is also exactly why the reported gains skew toward GPQA-Diamond (deep, multi-step) over MMLU (shallower recall): the worked example says the damage is concentrated in deep layers, and so is the repair.

#### Depth attention is not sequence attention

It is easy to hear "attention over layers" and import all your intuitions about self-attention. Most of them do not apply, and the differences are what make AttnRes cheap. The table below lines up the two:

| Property | Sequence self-attention | Attention Residuals (depth) |
|---|---|---|
| Axis attended over | tokens ($T$, up to 1M) | layers ($L$, ~60) |
| Number of queries | $T$ (one per token) | **1** (a learned pseudo-query) |
| Query source | projected from input | a free parameter $w_\ell$ |
| Keys / values | projected from tokens | the raw layer outputs $v_i$ |
| Cost | $O(T^2 d)$ | $O(L d)$ naive, $O(N d)$ blocked |
| What it mixes | information across positions | information across depth |
| New parameters | $4 d^2$ per layer (QKVO) | $d$ per layer ($w_\ell$ only) |

The single most consequential row is "number of queries." Sequence attention has one query per token because every position needs its own view of the context. AttnRes has *one query for the whole layer* — the depth-mixing weights are the same for every token in the sequence (though still input-dependent through $v_i$). That collapses the cost from quadratic-in-sequence to linear-in-depth and reduces the new parameter count to a single vector per layer. It is the cheapest possible thing that still deserves the name "attention," and that minimalism is deliberate: a residual rule that doubled per-layer parameters would never survive the scaling-law comparison.

#### How much can one query learn?

The pseudo-query is a strikingly low-capacity router: a single $d$-dimensional vector deciding, via dot product with each layer's output, how much of that layer to keep. There is no per-token, per-head, or per-position adaptivity in the *query* — all the input-dependence flows through the keys $v_i$. Is one vector enough? The paper's answer is empirical (it works), but the mechanism is plausible: $w_\ell$ does not need to represent *content*, only a *direction in activation space that correlates with "this layer is worth reading at layer $\ell$."* If informative layer outputs share a rough subspace (a reasonable prior, given how features distribute across depth), one well-placed vector can separate them from the rest. The flip side, which the Critique returns to, is that this is also where AttnRes is most vulnerable: if a learned-but-*static* weight profile captures most of the benefit, the pseudo-query's input-dependence is doing less work than the framing implies.

Here is the naive version as a PyTorch module. It is faithful to the equation and genuinely runnable; it is also exactly the version you should *not* ship, for reasons we get to next.

```python
import torch
import torch.nn as nn

class AttnResStream(nn.Module):
    """Attention Residuals (naive form). Replace the unit-weight residual
    sum with a softmax mixture over all previous layer outputs, scored by
    one learned pseudo-query per layer.  Ref: Kimi Team, arXiv:2603.15031."""

    def __init__(self, num_layers: int, d_model: int):
        super().__init__()
        # The ONLY new parameters: one pseudo-query vector per layer.
        self.query = nn.Parameter(torch.empty(num_layers, d_model))
        nn.init.normal_(self.query, std=d_model ** -0.5)
        self.scale = d_model ** -0.5

    def mix(self, layer_idx: int, prev: torch.Tensor) -> torch.Tensor:
        # prev: (L_prev, B, T, D) — stack of outputs v_0 .. v_{l-1}
        w = self.query[layer_idx]                                  # (D,)
        logits = torch.einsum("d, l b t d -> l b t", w, prev)      # score each depth
        alpha = (logits * self.scale).softmax(dim=0)               # weights, sum_l = 1
        return torch.einsum("l b t, l b t d -> b t d", alpha, prev)  # weighted sum
```

And the forward loop that drives it. The structural change from a normal transformer is that we keep *every* layer output around and re-mix the whole history before each layer, instead of carrying a single running sum:

```python
def forward(self, x, layers, stream: AttnResStream):
    outputs = [x]                              # v_0 = token embedding
    for l, layer in enumerate(layers):
        history = torch.stack(outputs, dim=0)  # (l, B, T, D)  <-- O(l) memory
        h = stream.mix(l, history)             # softmax-mixed residual input
        v = layer(h)                           # attention or MLP sublayer
        outputs.append(v)
    return outputs[-1]
```

Stare at `torch.stack(outputs, ...)` and the problem is obvious: by layer $\ell$ we are holding $\ell$ activation tensors live and softmaxing over all of them. Summed over the network that is $O(L^2)$ compute and $O(Ld)$ activation memory for the depth-attention alone. For Kimi Linear's depth that is not a rounding error anymore — it is the kind of cost that turns a good idea into a benchmark footnote. Which is the whole reason for the next section.

### Block AttnRes: bounding the depth-attention memory {#block-attnres}

The fix is the same move that made [MoBA](/blog/paper-reading/large-language-model/moba) work for sequence attention: don't attend over everything, attend over *blocks*. Partition the $L$ layers into $N \approx 8$ contiguous blocks. Inside a block, just use the ordinary unit-weight residual — cheap, and over a short span the dilution problem is mild. Only at **block boundaries** do you run the depth-attention, and only over **one summary vector per block** rather than every layer's output.

![Block AttnRes keeps the cache at O(Nd)](/imgs/blogs/attention-residuals-3.png)

The figure lays out the two-level structure. Each block (here Block 1, Block 2, … Block 8) runs its `L/N` layers with a standard intra-block residual, then emits a single **block summary** — one vector. The cross-block Attention Residual then operates over just those $N$ summaries. The accounting is the point: instead of caching all $L$ layer outputs ($O(Ld)$), you cache $N$ summaries ($O(Nd)$), and with $N\approx 8$ that is essentially free regardless of how deep the model gets. The depth-attention compute drops from $O(L^2)$ to $O(N^2 + L)$.

There is a real design tension buried in "one summary per block." If the summary is just the block's final residual value, you've thrown away the within-block trajectory; if it's something richer, you pay more memory. The report's choice — accumulate normally and carry a **partial sum** alongside the block representation — keeps the summary to a single vector while still letting the cross-block softmax see a faithful proxy for the block. The boundary fires every `block_size // 2` attention+MLP pairs in their implementation, i.e. blocks are defined over sublayer-pairs, not raw layers.

A conceptual sketch (illustrative, not the exact kernel):

```python
def block_attnres(x, layers, block_size, stream):
    summaries = []              # one cached vector per finished block: O(N) total
    partial = x                 # running intra-block residual sum
    for l, layer in enumerate(layers):
        # At a block boundary, mix the block summaries with depth-attention.
        if l % block_size == 0 and summaries:
            history = torch.stack(summaries + [partial], dim=0)  # (<=N, B, T, D)
            partial = stream.mix(len(summaries), history)        # cross-block AttnRes
        v = layer(partial)
        partial = partial + v                                    # Phase 1: std residual
        if (l + 1) % block_size == 0:
            summaries.append(partial)                            # cache block summary
    return partial
```

Notice the memory profile flips from "grows with depth" to "grows with block count," and block count is a fixed small constant you choose. That is the difference between an architecture you can scale and one you cannot.

### Two-phase computation and the cache-based pipeline {#two-phase}

Bounding the *memory* is necessary but not sufficient — you also need the wall-clock overhead to be marginal, or nobody will adopt a residual connection that taxes every step. This is where the engineering, not the math, does the work.

![Two-phase schedule keeps AttnRes a drop-in](/imgs/blogs/attention-residuals-4.png)

The schedule has two phases, illustrated above as a timeline. **Phase 1** is the ordinary forward pass: the layers in block $k$ run normally and accumulate a standard intra-block residual sum. As each block finishes, its block representation plus partial sum is written to a small cache — this is the *cache-based pipeline communication* the paper calls out. **Phase 2** is the cross-block Attention Residual: the softmax over cached block summaries. Because Phase 2 only needs the *summaries* (already in cache) and not the live per-layer activations, it can be **overlapped** with the forward computation of the next block rather than blocking it. The depth-attention rides along in the gaps of the pipeline instead of sitting on the critical path.

That overlap is why the report can claim "marginal overhead" for what is, on paper, an extra attention operation at every block boundary. The cost is real but it is hidden behind compute you were already doing. It is the same philosophy that makes [Mooncake](/blog/paper-reading/large-language-model/mooncake)'s disaggregated serving and Kimi Linear's chunked kernels fast: the algorithmic idea is half the work; scheduling it so the new cost overlaps the old cost is the other half.

| Variant | Depth-attn memory | Depth-attn compute | Mixing weights | Overhead |
|---|---|---|---|---|
| PreNorm residual | $O(1)$ (running sum) | $O(L)$ | fixed, all = 1 | none |
| Naive AttnRes | $O(Ld)$ | $O(L^2)$ | learned, per-pair | large |
| **Block AttnRes** | $O(Nd)$, $N\approx 8$ | $O(N^2 + L)$ | learned, per-block | **marginal** |

The table is the whole argument in three rows. PreNorm is cheap but rigid. Naive AttnRes is expressive but expensive. Block AttnRes keeps almost all of the expressivity for almost none of the cost — and "almost all" is the empirical claim the next section has to defend.

## Experiments

The headline result is an apples-to-apples swap: take Kimi Linear 48B (3B activated), train on 1.4T tokens, change *only* the residual rule from PreNorm to Block AttnRes, and measure downstream.

![AttnRes gains on Kimi Linear 48B (1.4T tokens)](/imgs/blogs/attention-residuals-5.png)

| Benchmark | Baseline (PreNorm) | Block AttnRes | Δ |
|---|---|---|---|
| GPQA-Diamond | 36.9 | 44.4 | **+7.5** |
| HumanEval | 59.1 | 62.2 | +3.1 |
| MATH | 53.5 | 57.1 | +3.6 |
| MMLU | 73.5 | 74.6 | +1.1 |

Two things stand out. First, **every** benchmark moves up — there is no task where the learned mixing hurts, which is the minimum bar for a change to the universal residual path. Second, the gains are *largest on the hardest reasoning task*: +7.5 on GPQA-Diamond versus +1.1 on MMLU. That pattern is consistent with the dilution story — knowledge-recall benchmarks like MMLU lean on early/middle layers that are not badly diluted, while multi-step reasoning needs deep layers to integrate signal from across the stack, exactly where flat-weight accumulation hurts most. I would not over-read four numbers, but the *shape* of the gain matches the proposed mechanism, and that is the kind of internal consistency I look for.

Beyond raw scores, the paper reports two diagnostics that matter more to me than the benchmark table:

- **More uniform output magnitudes and gradient distribution across depth.** This is the direct fingerprint of curing dilution: if the problem was that deep layers contribute a shrinking fraction, the cure should show up as flatter per-layer norms and flatter gradients. They report it does.
- **A scaling-law statement: Block AttnRes matches a baseline trained with 1.25× more compute.** This is the claim worth the most, because it reframes AttnRes from "a small accuracy bump" to "a 25%-ish compute multiplier that compounds with scale." If it holds at larger $N$, larger $L$, and longer token budgets, it is a free lunch of the rare, real kind.

**What is load-bearing in their setup that might not transfer.** The evaluation is on the Kimi Linear family — a hybrid linear-attention architecture, not a vanilla dense transformer. It is plausible that AttnRes interacts specifically well with linear attention (whose per-layer outputs may have different norm statistics than softmax-attention layers), or specifically well with the 3:1 hybrid block structure that already thinks in terms of blocks. The 1.4T-token run is also short by frontier standards (Kimi K2 trained on 15.5T). A residual change that helps at 1.4T could wash out — or compound — at 15T. None of this is disqualifying; it is the standard "one architecture, one scale" caveat that every architecture paper carries until someone reproduces it elsewhere.

**What "1.25× compute" actually means.** This is the number to internalize, so let me unpack it. The claim is that a PreNorm baseline must be trained with 25% more tokens (or parameters, or steps — the paper frames it as compute) to reach the same loss that Block AttnRes reaches at the base budget. At the scale of a real pretraining run, 25% of compute is not a rounding error: for a model that costs, say, $5M to train, a compute-equivalent architectural change that costs only a handful of extra vectors and marginal step overhead is worth on the order of a million dollars of saved GPU time — *if* the equivalence holds at production scale. That "if" is the entire bet. Compute-equivalence multipliers measured at small scale notoriously shrink as you scale (the baseline catches up), which is why the most valuable follow-up is not another benchmark but a clean scaling-law curve at 10T+ tokens showing the gap stays open.

**A reproduction checklist.** If you want to trust this before betting a training run on it, here is what I would want to see reproduced, in priority order: (1) the *monotonic* gain — every benchmark up, no regressions — on a second architecture; (2) the flat per-layer gradient/norm diagnostic, which is the direct fingerprint of the proposed mechanism and is cheap to measure; (3) the GPQA-over-MMLU skew, because it is a *prediction* of the dilution story rather than a generic "it helps"; (4) the static-scalar ablation (more on this in the Critique). If 1–3 reproduce and 4 shows real input-dependence, the technique is solid. If only 1 reproduces, you have "a thing that helps a bit," which is worth far less than "the residual connection was wrong."

## Where AttnRes sits in the Kimi research program

It is tempting to read Attention Residuals as a one-off architecture paper, but it lands inside a remarkably coherent body of work, and the through-line tells you something about why Moonshot keeps finding these wins. Look at what the same lab has shipped: [MuonClip and QK-Clip](/blog/paper-reading/large-language-model/kimi-k2) to train a trillion-parameter MoE *without a single loss spike*; [Muon / Moonlight](/blog/paper-reading/large-language-model/muon-moonlight), which fixed two tiny things about an optimizer to double compute efficiency; [Kimi Linear](/blog/paper-reading/large-language-model/kimi-linear), which rebuilt attention around a gated delta rule; [MoBA](/blog/paper-reading/large-language-model/moba), which applied MoE-style top-k routing to attention itself. Every one of these is the same *kind* of move: take a component everyone treats as fixed — the optimizer's update rule, the attention pattern, now the residual connection — and ask whether the default is actually optimal or just *first*.

Attention Residuals fits that mold exactly, and it shares two design fingerprints with its siblings. The first is **stability as a first-class goal**. MuonClip exists to keep attention logits from exploding; QK-Clip rescales the offending query/key projections; AttnRes's softmax normalization is, at heart, another norm-control mechanism — it keeps the residual stream from growing unboundedly with depth the way QK-Clip keeps logits bounded across steps. Moonshot's papers read like a lab that has been badly burned by training instability and now treats "does this keep the activations well-conditioned?" as a primary design question, not an afterthought. That instinct is why a residual tweak gets evaluated on its *gradient uniformity*, not just its benchmark delta.

The second fingerprint is **the block trick as a recurring pattern**. MoBA bounds sequence-attention cost by attending over blocks of tokens; Block AttnRes bounds depth-attention cost by attending over blocks of layers. It is the same algorithmic idea — replace "attend over everything" with "attend over a small set of summaries" — transplanted from the sequence axis to the depth axis. Once you have built the machinery to make block-sparse routing fast (the kernels, the caching, the two-phase schedules), you can amortize it across every axis of the model. AttnRes is partly a story about *reuse of systems infrastructure*: the reason it can claim marginal overhead is that Moonshot has already paid down the cost of doing block-wise attention efficiently. A lab without that infrastructure would have shipped the O(L²) version and lost on the scaling-law comparison.

This is the synthesis worth carrying away even if you never train a model with AttnRes: the frontier-lab edge is increasingly *not* a single brilliant idea but a **portfolio of "question the default" experiments riding on shared stability-and-efficiency infrastructure**. AttnRes is one more entry, and it is exactly the kind of low-risk, high-optionality bet — tiny parameter cost, drop-in integration, compounding-if-it-holds upside — that such a portfolio is built from.

## Critique

**What's strong.** The framing is excellent: it identifies a genuinely unquestioned component, gives a clean mathematical statement of why it is suboptimal (PreNorm as the all-weights-equal special case), proposes the minimal fix (learned softmax weights, $L$ new vectors), and then does the unglamorous systems work to make the fix affordable. The Block AttnRes + two-phase pipeline is the part I respect most — plenty of papers would have shown the naive O(L²) version winning on a 1.5B model and called it a day. Carrying it into a 48B production run and reporting marginal overhead is the difference between a curiosity and a technique.

**What's weak or unfalsifiable.** The "PreNorm dilution" narrative is intuitive and partially measured, but the causal chain — flat weights → norm growth → diluted contribution → worse reasoning → AttnRes fixes it — has several links that are asserted more than isolated. The uniform-gradient diagnostic supports the norm-growth link; the benchmark gains support the end-to-end link; the middle is inference. A skeptic could argue AttnRes simply adds a tiny bit of useful per-layer adaptivity and the dilution story is a *post hoc* rationalization that happens to predict the GPQA-vs-MMLU pattern.

**The missing ablation.** The single experiment I want and do not see: **decouple "learned per-layer scalar" from "input-dependent softmax."** Hyper-connections and LayerScale already give you learned-but-input-independent layer weights. How much of the +7.5 on GPQA comes from merely *learning* the weights versus making them *depend on the activation*? If a learned-but-static weight profile captures most of the gain, AttnRes is a more expensive way to get most of a cheaper method's benefit, and Block AttnRes's whole apparatus is partly unnecessary. Relatedly: an ablation on $N$ (block count) would tell us whether 8 is a sweet spot or just the first value that worked.

**The summary is a lossy bottleneck.** Block AttnRes's entire efficiency argument rests on compressing each block to a single vector before the cross-block attention sees it. That is a real information bottleneck, and the paper does not (in what I can see) quantify how much is lost versus the naive all-layers form. With $N \approx 8$ blocks over a 60-ish-layer model, each summary stands in for ~7–8 layers. If the within-block trajectory carries signal the cross-block router would have wanted — say, an early-block feature that a late block needs to reach back to *specifically*, not via its block's averaged summary — Block AttnRes cannot express it, while naive AttnRes can. The reported "matches naive within marginal overhead" is reassuring, but I want to see the gap as a function of $N$: does it degrade gracefully as you coarsen, or is there a cliff? That curve decides whether 8 is principled or lucky.

**Generality and the inference story are under-discussed.** Two practical questions go unanswered. First, does AttnRes interact with quantization? The Kimi line cares a lot about [low-bit inference](/blog/paper-reading/large-language-model/kimi-k2-thinking) (K2 Thinking ships native INT4), and a softmax over a depth axis introduces a new set of activations whose dynamic range and quantization behavior are simply not characterized here. Second, the block-boundary computation introduces a structured non-uniformity into the forward pass — most layers are cheap, boundary layers do extra work — which complicates pipeline-parallel scheduling and could bite at very large tensor/pipeline-parallel degrees in ways a uniform PreNorm stack never does. Neither is a flaw in the idea; both are the kind of thing that turns "wins on the eval" into "wins in production," and a follow-up should address them.

**What would change my mind.** If a reproduction on a *dense* (non-linear-attention) transformer at a *different* scale showed the same monotonic, reasoning-skewed gains and the same flat-gradient diagnostic — and if the learned-static-scalar ablation came back showing the input-dependence is doing real work — I would upgrade this from "promising architecture tweak" to "the residual connection was wrong and this is the fix." Conversely, if the static-scalar ablation captured ~80% of the gain, I'd file AttnRes as a heavier reinvention of hyper-connections, useful but oversold.

## What I'd build with this

1. **The static-scalar ablation, first.** Before anything else, train three small models that are identical except for the residual rule: (a) PreNorm, (b) learned-but-input-independent per-pair weights (a softmax over depth with a *constant* logit table, no pseudo-query), (c) full AttnRes. The gap between (b) and (c) is the entire empirical justification for input-dependence. This is a one-week experiment that resolves the paper's biggest open question.
2. **AttnRes as an interpretability probe.** The learned weight profile $\alpha_{i\to\ell}$ is a depth-by-depth attention map over the whole network — a readout of *which layers each layer chooses to read*. Logging $\alpha$ across training and across inputs would be a cheap, novel lens on residual-stream structure, complementary to the SAE and steering work over in [interpretability](/blog/machine-learning/ai-interpretability). Do reasoning prompts induce visibly different depth-attention than recall prompts? The dilution story predicts yes.
3. **Block AttnRes on a long-context budget.** The 1.4T run is short. I'd want the scaling-law claim re-measured at a 10T+ budget and at 1M-token context, where Kimi Linear's decode advantages live, to see whether the 1.25×-compute equivalence holds or grows.
4. **Pair it with MoE routing signal.** AttnRes decides *how much of each past layer* to read; MoE decides *which experts* to fire. Both are input-dependent routers. Sharing or co-training the routing signal (does the pseudo-query correlate with expert choice?) is a natural follow-up that could either simplify both or reveal they are doing orthogonal work.
5. **A drop-in PyTorch/`torch.compile` block** that exposes `residual="prenorm" | "attnres" | "block-attnres"` as a config flag, so any existing training stack can A/B the residual rule without surgery. The whole point of this paper is that the change is local; the tooling should make adopting it equally local.
6. **A graceful-degradation curve over block count $N$.** The single experiment that most de-risks adoption: train the same model at $N \in \{2, 4, 8, 16, L\}$ (where $N=L$ recovers naive AttnRes) and plot final loss against depth-attention cost. This answers three questions at once — whether 8 is a real sweet spot, how lossy the per-block summary is, and where the cost/quality knee sits for *your* depth — and it is the missing axis that turns "we picked 8" into "8 is optimal because." Until that curve exists, every adopter is re-deriving the block count from scratch.

Taken together, these are not five disconnected ideas but a single research agenda: *prove the input-dependence matters, measure the bottleneck, and expose the knob.* If I were running a pretraining team and someone handed me AttnRes, that agenda is precisely the two-week spike I would fund before committing a flagship run to it — cheap to execute, and decisive either way.

## When to reach for Attention Residuals (and when not to)

**Reach for it when** you are training a *deep* model from scratch and depth is where you suspect you are leaving accuracy on the table — long-horizon reasoning, deep MoE stacks, anything where the top layers must integrate signal from far below. The cost is a handful of vectors and a marginal overhead; the upside, if the scaling-law claim holds, is a compute multiplier. If you are already paying for a custom training stack (you build your own kernels, you train past 1T tokens), the integration cost is noise and the experiment is cheap to run.

**Be cautious when** you are fine-tuning or continuing pretraining of an *existing* PreNorm checkpoint. AttnRes changes the residual semantics; you cannot simply bolt pseudo-queries onto a network that was trained to expect a flat sum and expect the +7.5 to appear. This is a from-scratch (or at least long-anneal) architectural choice, not a LoRA you can attach.

**Skip it when** your model is shallow, your bottleneck is data or sequence-length rather than depth, or you have not yet run the static-scalar ablation for your own setting. For a 12–24 layer model the dilution problem is mild and the bookkeeping may not pay for itself; a learned scalar gate (LayerScale/ReZero) might capture the available gain at a fraction of the complexity. And if you are inference-bound rather than quality-bound, note that AttnRes helps *training-time quality*, not decode latency — for that you want the architectural moves in [Kimi Linear](/blog/paper-reading/large-language-model/kimi-linear), not this.

To make that concrete, picture three teams. The first is pretraining a 64-layer reasoning model from scratch with a custom stack and a multi-trillion-token budget: this is the *ideal* case — depth is the lever, the integration cost is noise, and a compute-equivalence multiplier compounds across the whole run. They should run the two-week spike and, if it reproduces, adopt. The second is fine-tuning an off-the-shelf 32-layer PreNorm checkpoint for a downstream product: AttnRes is the *wrong* tool — they would have to re-pretrain to change the residual semantics, which dwarfs any plausible gain, so they should reach for LoRA, data, or a better base model instead. The third is a research team chasing interpretability rather than benchmarks: for them the *most* interesting thing about AttnRes is not the accuracy bump at all but the learned $\alpha_{i\to\ell}$ map, a free readout of which layers each layer chooses to read — they should adopt it as an instrument regardless of whether it wins on loss. Same technique, three completely different verdicts, and the variable that decides is *what gates your quality* — depth, deployment, or understanding.

The deeper lesson is the one I keep coming back to: the parts of the stack nobody questions are exactly where the next gains hide. We spent a decade tuning attention and optimizers while a literal `+` sat untouched in every model. Attention Residuals is a reminder to occasionally point the microscope at the boring lines.

## References

- **Attention Residuals** — Kimi Team, Moonshot AI. arXiv:2603.15031. [Paper](https://arxiv.org/abs/2603.15031) · [Code](https://github.com/MoonshotAI/Attention-Residuals)
- He et al., *Deep Residual Learning for Image Recognition* (ResNet), 2015 — the original residual connection.
- Xiong et al., *On Layer Normalization in the Transformer Architecture*, 2020 — the PreNorm-vs-PostNorm stability analysis.
- Sibling reads on this blog: [Kimi Linear](/blog/paper-reading/large-language-model/kimi-linear) (the architecture AttnRes ships inside), [Muon / Moonlight](/blog/paper-reading/large-language-model/muon-moonlight) (the optimizer half of the same research program), [MoBA](/blog/paper-reading/large-language-model/moba) (the block-sparse idea AttnRes echoes along the depth axis), and [Kimi K2](/blog/paper-reading/large-language-model/kimi-k2) (the trillion-parameter MoE this lineage feeds).
