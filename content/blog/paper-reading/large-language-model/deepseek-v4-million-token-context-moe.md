---
title: "DeepSeek-V4: How a 1.6T MoE Reads a Million Tokens at a Tenth of the KV Cache"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "A close read of DeepSeek-V4: hybrid Compressed Sparse and Heavily Compressed Attention, manifold-constrained hyper-connections, Muon at trillion scale, and a specialist-then-distillation post-training pipeline that serves one-million-token context at roughly 27% of V3.2's FLOPs."
tags: ["deepseek-v4", "large-language-model", "mixture-of-experts", "sparse-attention", "long-context", "kv-cache", "muon-optimizer", "hyper-connections", "post-training", "paper-reading"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: false
readTime: 28
---

> [!tldr]
> - **DeepSeek-V4** ships two mixture-of-experts models built around one goal: native, *cheap* million-token context. **V4-Pro** is 1.6T parameters / 49B activated; **V4-Flash** is 284B / 13B activated. Both are pretrained on 32–33T tokens.
> - The headline is an efficiency claim, not a benchmark claim: at a 1M-token context, **V4-Pro uses only ~27% of the single-token inference FLOPs and ~10% of the KV cache of DeepSeek-V3.2** (Flash drops to ~10% FLOPs and ~7% KV). Versus a vanilla GQA8/BF16 baseline the KV cache falls to roughly **2%**.
> - Three subsystems are genuinely new: a **hybrid attention** stack (Compressed Sparse Attention + Heavily Compressed Attention), **Manifold-Constrained Hyper-Connections** that pin the residual-mixing matrix to the doubly-stochastic manifold, and **Muon** used as the primary optimizer at trillion scale.
> - The most interesting bet is in *post-training*: instead of one model learning every domain, V4 trains **independent specialists** (math, code, agent, instruction-following) each through SFT then GRPO, then folds them into one student by **on-policy reverse-KL distillation**.
> - Where it still loses: raw **knowledge** (SimpleQA-style recall) and the **hardest reasoning**, where it sits a few months behind the very top closed frontier. The efficiency story is the moat; the capability story is "best open, closing on frontier."

A million-token context window is easy to put on a slide and brutally expensive to actually serve. The model weights are a fixed cost you pay once and amortize across every request, but the KV cache is a *per-request* tax that grows linearly with sequence length, and at a million tokens it is the KV cache — not the parameters — that decides how many concurrent users fit on a node. Past a certain length the question stops being "is the model good enough?" and becomes "can I afford to let it read this much at the throughput my users will tolerate?" Most long-context launches quietly answer that question with "only at batch size one, only if you pay for it."

[DeepSeek-V4](https://arxiv.org/abs/2606.19348) is an attempt to answer it differently. The paper's thesis is that long context is an *efficiency* problem before it is a *modeling* problem, and that you win it by attacking the two costs that scale with sequence length — the per-token attention FLOPs and the per-token KV cache — at the same time, with architecture rather than with quantization tricks bolted on at serving time.

![From DeepSeek-V3 to V4: three subsystems swapped](/imgs/blogs/deepseek-v4-million-token-context-moe-1.png)

The diagram above is the mental model for the whole report. DeepSeek-V4 is not a from-scratch redesign; it keeps the [DeepSeekMoE](/blog/machine-learning/large-language-model/deepseek-moe-lineage-fine-grained-shared-experts) backbone — fine-grained routed experts plus shared experts — that has carried the V2 and V3 lines. What changes is three load-bearing subsystems: the attention block becomes a hybrid of two compressed variants, the residual stream is rebuilt as a constrained hyper-connection, and the optimizer switches from AdamW to [Muon](/blog/paper-reading/large-language-model/muon-moonlight). The MoE FFN survives almost untouched — only its gating activation changes. Everything else in this post is a close look at those three swaps and at the post-training pipeline that turns the pretrained base into something you would actually deploy.

## The road from V3 to V4

To see why V4 looks the way it does, you have to see what each prior DeepSeek generation already spent its complexity budget on. V2 introduced [Multi-head Latent Attention (MLA)](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla), which compresses the KV cache into a low-rank latent and was the single biggest reason DeepSeek could serve long context cheaply at all. V3 scaled the MoE — 671B total parameters, 37B activated — and proved that an auxiliary-loss-free load-balancing scheme could keep hundreds of experts busy without the usual router pathologies. V3.2 layered *sparse* attention on top, in the lineage of trainable sparse-attention work like [NSA and DSA](/blog/machine-learning/large-language-model/trainable-sparse-attention-nsa-vs-dsa): instead of every query attending to every key, a cheap selector picks a subset of keys to read.

The gap all of this leaves is the *interaction* between length and cost. MLA shrinks the per-token KV footprint by a constant factor, but the cache still grows linearly with sequence length, so a million tokens is still a million latents. Sparse attention cuts the per-token FLOPs but, in its V3.2 form, still keeps the full KV around to *select* from — you save compute but not memory. And as you stack more layers and widen the residual stream to make the model strong enough to use a million tokens of context, you walk straight into the deep-network stability problems that AdamW papers over but never really solves at this scale.

DeepSeek-V4's claim is that you can close that gap by co-designing the three pieces. Compress the KV so aggressively that memory, not just compute, drops by an order of magnitude. Constrain the residual stream so widening it does not blow up. And switch to an optimizer whose update geometry is better matched to the matrix-shaped parameters that dominate a transformer, so the wider, deeper, more aggressively-compressed model still trains stably on 32T tokens. The rest of this article reads those three bets one at a time.

## What's new in V4

The paper lists its contributions in the usual numbered form. Tightened to what actually matters:

1. **Hybrid attention** — every attention layer is one of two compressed variants. **Compressed Sparse Attention (CSA)** compresses the KV stream and then *sparsely selects* which compressed blocks to read; **Heavily Compressed Attention (HCA)** compresses far more aggressively and reads *all* of its (now tiny) compressed entries densely. Together they drop both FLOPs and KV cache at long context.
2. **Manifold-Constrained Hyper-Connections (mHC)** — a widened residual stream whose mixing matrix is projected onto the doubly-stochastic manifold every step, guaranteeing its spectral norm stays at most 1 so signal can neither explode nor vanish through depth.
3. **Muon at trillion scale** — Muon, an optimizer that orthogonalizes the momentum matrix before each step, replaces AdamW as the primary optimizer, with engineering to make it stable and communication-cheap across a 1.6T-parameter MoE.
4. **A specialist-then-distillation post-training pipeline** — independent domain specialists (math, code, agent, instruction-following) are each trained with SFT then GRPO, then merged into one model by on-policy reverse-KL distillation, with explicit support for variable reasoning effort, generative reward models, and agentic tool use.
5. **FP4 quantization-aware training and a long-context RL stack** — MoE expert weights are trained FP4-aware, distillation runs over the full vocabulary with scheduled teachers, and the RL rollout service is built to be preemptible and to handle million-token episodes.

Two models come out of this:

| Model | Total params | Activated / token | Context | Pretraining tokens |
|---|---|---|---|---|
| DeepSeek-V4-Pro | 1.6T | 49B | 1M (native) | ~33T |
| DeepSeek-V4-Flash | 284B | 13B | 1M (native) | ~32T |

The post-trained "thinking" variants used in the evaluations are named **V4-Pro-Max** and **V4-Flash-Max**; throughout the paper "-Max" denotes the reasoning-tuned checkpoint with a large default thinking budget.

## The hybrid attention stack: CSA and HCA

Attention is where the long-context cost lives, so it gets the deepest redesign. The key idea is that not every layer needs the same kind of attention. Some layers need *precise recall* — the ability to reach back and read a specific earlier token more or less exactly. Other layers only need a *cheap global summary* — a low-resolution sense of the whole context. V4 builds two attention operators for those two jobs and interleaves them.

### Compressed Sparse Attention

CSA runs in three phases, and the diagram below traces a single decode step through all of them.

![Compressed Sparse Attention: compress, select, then attend](/imgs/blogs/deepseek-v4-million-token-context-moe-2.png)

**Phase 1 — KV compression.** Rather than keep one KV entry per token, CSA compresses every $m$ consecutive tokens into a single compressed entry using learned, softmax-normalized compression weights combined with learnable position biases. Writing $C^a_j, C^b_j$ for two streams of token-level KV features and $S^a_j, S^b_j$ for their (softmax-normalized) compression weights, a compressed entry is

$$
C^{\text{Comp}}_i = \sum_j \left( S^a_j \odot C^a_j \right) + \sum_j \left( S^b_j \odot C^b_j \right),
$$

where $\odot$ is elementwise product. The practical effect is that a context of $T$ tokens becomes $T/m$ compressed entries — the cache shrinks by a factor of $m$ before any precision tricks are applied.

**Phase 2 — sparse selection via the Lightning Indexer.** Even $T/m$ entries is a lot at a million tokens, so CSA does not attend to all of them. A deliberately cheap "Lightning Indexer" scores each compressed block and keeps only the top-$k$. The scoring is low-rank by construction: a down-projection $c^Q_t = h_t W^{DQ}$ maps the hidden state to a small indexer dimension $d_c$, and the index score between query token $t$ and compressed block $s$ is a ReLU-gated, per-head weighted sum,

$$
I_{t,s} = \sum_h w^I_{t,h}\,\operatorname{ReLU}\!\left( q^I_{t,h} \cdot K^{IComp}_s \right).
$$

Only the top-$k$ blocks by $I_{t,s}$ survive into the actual attention. The ReLU matters: it makes the score non-negative and sparse, so a block that is irrelevant contributes nothing rather than a small negative nudge.

Here is the indexer-plus-selection step in PyTorch-shaped code. It is the load-bearing trick, so it is worth seeing the tensor shapes:

```python
import torch
import torch.nn.functional as F

def csa_select(h, W_dq, Wq_idx, w_idx, K_idx, top_k):
    """CSA Lightning Indexer: cheaply score compressed blocks, keep the top-k.

    h      : (B, T, d)        decoder hidden states (the queries)
    W_dq   : (d, d_c)         down-projection to a small indexer dim d_c
    Wq_idx : (H, d_c, d_c)    per-head indexer query maps
    w_idx  : (B, T, H)        per-head, per-token index weights
    K_idx  : (B, S, d_c)      index keys, one per compressed block (S = T / m)
    returns: (B, T, top_k)    block indices each query token will attend to
    """
    c_q = h @ W_dq                                      # (B, T, d_c) low-rank query
    q = torch.einsum("btc,hcd->bthd", c_q, Wq_idx)      # (B, T, H, d_c)
    sim = F.relu(torch.einsum("bthd,bsd->bths", q, K_idx))  # (B, T, H, S)
    scores = torch.einsum("bth,bths->bts", w_idx, sim)  # (B, T, S) index scores
    return scores.topk(top_k, dim=-1).indices           # (B, T, top_k)
```

Why is this cheap enough to run at every CSA layer? The indexer never touches the full hidden dimension. The down-projection $W^{DQ}$ maps $d$ (thousands) down to $d_c$ (a few dozen), so each per-block score is a $d_c$-dimensional dot product, not a $d$-dimensional one — one to two orders of magnitude fewer multiplies than the attention it gates. The index keys $K^{IComp}$ are precomputed once per compressed block and stored in FP4, so scoring a million tokens' worth of blocks is a small, low-precision matmul whose cost is dwarfed by the MQA that follows on the selected $k$. The indexer is, deliberately, the cheapest thing in the layer — its only job is to decide what the expensive part is allowed to look at.

**Phase 3 — core attention and output.** The selected compressed entries are read with Multi-Query Attention (MQA) — one KV head shared across all query heads — and the output projection is *grouped*: the $n_h$ heads are split into $g$ groups whose outputs are projected together, cutting the projection cost. MQA plus grouped output is what makes the actual attention arithmetic cheap once the selection has narrowed the field.

### Heavily Compressed Attention

HCA is the cheap-global-summary counterpart. It uses the same MQA and grouped-output machinery as CSA but makes two opposite choices: it compresses *far* more aggressively, with a compression rate $m' \gg m$ and no overlap between compressed windows, and then it skips selection entirely and attends *densely* over the resulting handful of entries. Because $m'$ is large, "densely" is still cheap — there are simply not many entries to attend to.

![CSA vs HCA: two attention variants, two jobs](/imgs/blogs/deepseek-v4-million-token-context-moe-3.png)

The split is worth dwelling on because it is the heart of the design. A purely sparse-selective scheme (CSA everywhere) preserves recall but pays for an indexer and a top-$k$ at every layer. A purely heavy-compression scheme (HCA everywhere) is dirt cheap but blurry — it can tell you roughly what is in the context but not reach back and quote it. By interleaving the two across the layer stack, V4 gets recall where it needs it and global context almost for free everywhere else, and the *aggregate* KV cache is dominated by the cheap HCA layers. This is the same philosophy as hybrid linear/full-attention stacks like [Qwen3-Next](/blog/paper-reading/large-language-model/qwen3-next-hybrid-attention-ultra-sparse-moe) and [MiniMax-01](/blog/paper-reading/large-language-model/minimax-01-lightning-attention-hybrid-moe), but executed with two *compressed* variants rather than one linear and one quadratic.

### The KV-cache budget

The FLOPs story is mostly told by compression and selection. The *memory* story has a second, independent lever: numeric precision. V4 stores different parts of the KV in different formats, and this is where the most extreme numbers come from.

![Where the KV cache goes: fewer entries, then fewer bits](/imgs/blogs/deepseek-v4-million-token-context-moe-4.png)

The RoPE-carrying dimensions — the ones whose exact values encode position and are most sensitive to rounding — are kept in BF16. The bulk content dimensions are stored in FP8. The Lightning Indexer's keys, which only ever feed a ReLU-gated argmax-like selection, are stored in FP4, because the selection is robust to coarse quantization. Stacking the token-count compression (CSA's $m{\to}1$, HCA's $m' \gg m$) against this mixed-precision storage is what produces the report's most aggressive figure: relative to a vanilla GQA8/BF16 baseline, the KV cache at one million tokens falls to roughly **2%**.

It is worth doing the arithmetic, because two independent factors multiply here and the product is what gets you to single-digit percentages. (The paper reports ratios rather than per-token byte counts, so treat the following as illustrative of *how* the factors compose, not as the model's exact configuration.) Suppose a heavily-compressed HCA layer folds roughly 16 tokens into one entry — an ~16× cut in the number of entries. Independently, moving the stored values from BF16 (16 bits) to a blend dominated by FP8 content and FP4 index keys brings the average down to roughly 6 bits — another ~2.7× cut. The two compose multiplicatively: $16 \times \tfrac{16}{6} \approx 43\times$, so the KV cache shrinks to roughly 2% of the dense BF16 baseline. The point is that neither lever alone gets you there — token-count compression alone leaves you near 6%, precision alone near 37% — but stacked they reach an order of magnitude past either. This is also why the compression is computed once at prefill and the compressed, low-precision KV is what persists through decode: the per-token decode cost, not just the one-time prefill cost, is what the architecture is fighting.

| Metric @ 1M tokens | V4-Pro vs V3.2 | V4-Flash vs V3.2 |
|---|---|---|
| Single-token inference FLOPs | ~27% | ~10% |
| KV cache size | ~10% | ~7% |

A caveat the paper is honest about: the FP4 indexer and FP8 content paths lean on hardware that can multiply low-precision formats efficiently. The report notes that FP4×FP8 operations should buy roughly a third more efficiency *on future hardware* — which is a tell that some of the headline efficiency is latent, waiting on accelerators that schedule these mixed-precision ops well, rather than fully realized on today's fleet.

## Manifold-constrained hyper-connections

Once attention is cheap, the next bottleneck is depth. To use a million tokens of context the model has to be deep and wide, and deep residual networks have a chronic problem: the residual stream can amplify or attenuate signal as it passes through dozens of layers, and the usual fixes (careful init, normalization placement, residual scaling) are band-aids that you re-tune every time the architecture changes.

Hyper-Connections generalize the residual connection by widening the residual stream from $\mathbb{R}^d$ to $\mathbb{R}^{n_{hc} \times d}$ — keeping $n_{hc}$ parallel copies of the stream — and learning how each layer reads from, writes to, and mixes those copies:

$$
X_{l+1} = B_l X_l + C_l\, \mathcal{F}_l(A_l X_l),
$$

where $\mathcal{F}_l$ is the layer's sublayer (attention or MoE), $A_l$ is an input map that gathers the streams into the sublayer's input, $C_l$ writes the sublayer output back, and $B_l$ is a residual-mixing matrix that decides how the $n_{hc}$ streams recombine. The expressive power is real, but so is the danger: if $B_l$ has spectral norm greater than 1, the stream grows geometrically with depth, and a 100-layer stack turns a 1% per-layer gain into a blow-up.

DeepSeek-V4's contribution — **Manifold-Constrained Hyper-Connections** — is to forbid that outright by constraining $B_l$ to the **Birkhoff polytope** of doubly-stochastic matrices,

$$
\mathcal{M} = \left\{ M \in \mathbb{R}^{n \times n} \;\middle|\; M\mathbf{1}_n = \mathbf{1}_n,\; \mathbf{1}_n^\top M = \mathbf{1}_n^\top,\; M \ge 0 \right\}.
$$

A doubly-stochastic matrix has spectral norm exactly 1, so $\lVert B_l \rVert_2 \le 1$ is guaranteed by construction — the residual stream is a *mixture* of its previous states, never an amplification of them.

![Manifold-Constrained Hyper-Connections](/imgs/blogs/deepseek-v4-million-token-context-moe-5.png)

The mechanics, traced left to right in the figure: the raw parameters are produced from an input-dependent (dynamic) part plus a static learned part,

$$
\tilde{A}_l = \alpha^{\text{pre}}_l \cdot (\hat{X}_l W^{\text{pre}}_l) + S^{\text{pre}}_l,
\qquad
\tilde{B}_l = \alpha^{\text{res}}_l \cdot \operatorname{Mat}(\hat{X}_l W^{\text{res}}_l) + S^{\text{res}}_l,
$$

so the mixing adapts to the actual activations rather than being fixed. The raw $\tilde{B}_l$ is then projected onto $\mathcal{M}$ by running **20 iterations of the Sinkhorn–Knopp algorithm** — alternately normalizing rows and columns until the matrix is (numerically) doubly-stochastic:

```python
import torch

def sinkhorn(logits, iters=20):
    """Project a raw n x n matrix onto the doubly-stochastic Birkhoff polytope.

    Alternating row/column normalization converges to a matrix whose rows and
    columns each sum to 1; such a matrix has spectral norm exactly 1, so the
    residual-mixing step can never amplify the stream.
    """
    M = logits.exp()                            # non-negative entries
    for _ in range(iters):
        M = M / M.sum(dim=1, keepdim=True)      # row-normalize
        M = M / M.sum(dim=0, keepdim=True)      # column-normalize
    return M
```

The input and output maps are bounded with sigmoids — $A_l = \sigma(\tilde{A}_l)$ and $C_l = 2\sigma(\tilde{C}_l)$ — so every coefficient in the recurrence sits in a known range. The result is a residual stream that is strictly contractive in its mixing term and bounded in its read/write maps, which is exactly the property you want when you are about to train 100-plus layers on 32T tokens and cannot afford a divergence at step 400,000.

The cost is real but contained: the Sinkhorn iterations and the wider stream add work, but the paper reports the mHC wall-time overhead is held to about **6.7%** of the pipeline stage it overlaps with — cheap insurance against the failure mode that ends training runs.

## The MoE backbone

The MoE FFN is the part V4 changes *least*, which is itself a statement: the DeepSeekMoE design — many fine-grained routed experts plus a few always-on shared experts, balanced by an auxiliary-loss-free strategy — is treated as solved. The deltas are small and targeted:

- **Gating activation.** The router's gate moves from a sigmoid to $\sqrt{\operatorname{softplus}(\cdot)}$. Softplus is a smooth, always-positive function, and taking its square root tempers how sharply the gate saturates, which the paper credits with steadier routing early in training.
- **Hash routing in the first blocks.** The earliest transformer blocks replace the learned dense FFN with **hash routing** — a deterministic, parameter-free assignment of tokens to experts. Early layers do mostly low-level feature work where a learned router buys little, and a hash route removes a source of early-training instability.
- **Auxiliary-loss-free balance, relaxed.** V4 keeps the aux-loss-free balancing of V3 with a mild sequence-wise balance term, but **removes the constraint on the number of routing target nodes** and redesigns the parallelism around it — a scaling change more than a modeling one.
- **Multi-Token Prediction (MTP).** The MTP modules from V3 carry over **unchanged**, predicting more than one future token per position to densify the training signal and to enable self-speculative decoding at inference.

The sparsity is the whole economic argument. V4-Pro activates 49B of 1.6T parameters — roughly a **3% activation ratio** — and Flash activates 13B of 284B, about **4.6%**. You pay to *store* the full parameter count once and amortize it across every request, but you pay to *compute* only the activated count per token, so the marginal cost of a token tracks the 49B (or 13B) while the model's capacity tracks the 1.6T (or 284B). MoE buys capacity decoupled from per-token compute; the hybrid attention then keeps the *other* per-token cost — the KV cache — from quietly undoing that bargain once the context gets long. The two ideas are complementary: without compressed attention, a long-context MoE would have a cheap FFN and a ruinous cache.

If you want the deeper rationale for why fine-grained-plus-shared experts and aux-loss-free balancing work, the [DeepSeekMoE lineage post](/blog/machine-learning/large-language-model/deepseek-moe-lineage-fine-grained-shared-experts) covers it; V4 inherits those arguments rather than re-litigating them.

## Training V4

### Pretraining: 32–33T tokens

Both models are pretrained on more than 32T tokens (Flash on ~32T, Pro on ~33T) of "diverse and high-quality" data, and both support 1M context natively after pretraining rather than via a bolted-on context-extension stage. The report is thin on the exact long-context curriculum, which is the one place I wish it said more — "natively supports 1M" is doing a lot of work without a described data mixture or staged length schedule to back it.

Two stability measures are called out, and both are the kind of unglamorous fix that actually keeps a trillion-parameter run alive:

- **Anticipatory routing** — a mechanism that pre-empts extreme expert load imbalance before it happens, rather than only correcting it after a balance loss spikes. At this scale a single badly-imbalanced step can stall a pipeline; getting ahead of it matters.
- **SwiGLU clamping** — clamping the SwiGLU activations to prevent the occasional overflow that, in BF16/FP8 training, turns into a NaN that poisons the run.

### Muon at trillion scale

The optimizer swap is the most surprising pretraining choice, because AdamW is the safe default that nobody gets fired for picking. [Muon](/blog/paper-reading/large-language-model/muon-moonlight) makes a different bet: for a *matrix-shaped* parameter — which is to say nearly every weight in a transformer — the right update is not Adam's per-coordinate rescaling but an **orthogonalized** version of the momentum, so the update has balanced influence across all directions of the weight matrix rather than being dominated by a few large-gradient coordinates.

![One Muon step: orthogonalize the update before applying it](/imgs/blogs/deepseek-v4-million-token-context-moe-6.png)

A single Muon step, traced through the figure: accumulate momentum $M_t = \mu M_{t-1} + G_t$, orthogonalize $\mu M_t + G_t$ via a hybrid Newton–Schulz iteration, rescale the result so its RMS matches the parameter shape, and apply it with decoupled weight decay:

$$
O'_t = \operatorname{NewtonSchulz}(\mu M_t + G_t),
\qquad
O_t = O'_t \cdot \sqrt{\max(n,m)} \cdot \gamma,
\qquad
W_t = W_{t-1}(1 - \eta\lambda) - \eta O_t.
$$

The orthogonalization avoids an SVD (far too expensive per step) by using Newton–Schulz iterations, which drive a matrix toward the closest orthogonal matrix using only matrix multiplies. V4 uses a *hybrid* schedule of 10 iterations: the first 8 use aggressive coefficients $(a,b,c) = (3.4445, -4.7750, 2.0315)$ for fast convergence, the last 2 use gentler $(2, -1.5, 0.5)$ for numerical stability near the fixed point. Each iteration is the quintic

$$
M_k = a M_{k-1} + b\,(M_{k-1} M_{k-1}^\top) M_{k-1} + c\,(M_{k-1} M_{k-1}^\top)^2 M_{k-1}.
$$

In code:

```python
import torch

def newton_schulz(G, steps=10):
    """Orthogonalize G with a hybrid Newton-Schulz iteration (no SVD)."""
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)                  # spectral pre-scaling into [0, 1]
    transposed = G.size(0) > G.size(1)
    if transposed:                             # iterate on the smaller dimension
        X = X.T
    for i in range(steps):
        a, b, c = (3.4445, -4.7750, 2.0315) if i < 8 else (2.0, -1.5, 0.5)
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X

def muon_step(W, M, grad, lr, mu=0.95, wd=0.1, gamma=0.2):
    """One Muon update for a 2-D weight matrix W of shape (n, m)."""
    M.mul_(mu).add_(grad)                      # momentum accumulation
    O = newton_schulz(mu * M + grad)           # orthogonalized direction
    O = O * (max(W.shape) ** 0.5) * gamma      # RMS rescale to match W's shape
    W.mul_(1 - lr * wd).add_(O, alpha=-lr)     # decoupled decay + step
    return W, M
```

Two engineering details make Muon survive at 1.6T parameters. First, Muon is applied to *most* modules but not all — embeddings, RMSNorm parameters, and the static mHC parameters fall back to AdamW, because they are not the matrix-shaped weights Muon's geometry argument applies to. Second, MoE gradients are synchronized in **BF16**, halving the all-reduce communication volume — a meaningful saving when expert gradients dominate the cross-device traffic.

## Post-training: specialists, then distillation

This is the part of the report I would steal first. The conventional post-training recipe runs one model through SFT and then RL across all domains at once, which forces the optimizer to trade off math against creative writing against tool use in a single objective. V4 refuses the trade-off by splitting it in two stages.

![Post-training: train specialists, then distill into one model](/imgs/blogs/deepseek-v4-million-token-context-moe-7.png)

**Stage 1 — specialists.** Independent models are trained for each domain that has a clean success criterion: mathematics, coding, agentic tool use, and instruction-following. Each specialist gets its own SFT on high-quality domain data, then its own RL via **Group Relative Policy Optimization (GRPO)** — the same family of critic-free RL that powered [DeepSeek-R1](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) — with a reward model tuned to that domain. A math specialist can be rewarded on verified final answers; a code specialist on tests passing; an agent specialist on task completion. None of them has to compromise to be good at the others.

**Stage 2 — on-policy distillation.** A single student model is then trained to imitate all of the specialists at once, using a **reverse-KL** loss against the specialist teachers, *on-policy* — the student generates, and the teachers score the student's own samples. Reverse KL is mode-seeking: rather than smearing probability across everything every teacher might say (forward KL), it concentrates on the high-quality modes, which is what you want when distilling capability rather than calibration. The output is one unified model — Pro or Flash — that carries the union of the specialists' skills.

Two choices in Stage 2 are doing quiet work. *On-policy* — scoring the student's own generations rather than a fixed corpus of teacher outputs — sidesteps the exposure-bias trap of ordinary distillation, where a student trained only on text it would never have produced drifts off-distribution the moment it has to continue its own (imperfect) generations at inference. And *reverse* KL, $D_{\mathrm{KL}}(\pi_{\text{student}} \,\|\, \pi_{\text{teacher}})$, penalizes the student for putting probability mass where the teacher would not, sharpening it onto the teacher's best modes; forward KL would instead force it to *cover* everything the teacher might do, hedging included. When the goal is to consolidate hard-won capability from several strong specialists into one model — not to faithfully reproduce a teacher's full uncertainty — mode-seeking is the right bias, and on-policy is what keeps the distilled skill from being brittle.

Bolted onto this are the features that make V4 usable as a product rather than a benchmark artifact:

- **Reasoning efforts** — explicit, variable compute budgets so the model can be told to think harder or answer fast; this is the knob the "-Max" evaluations turn up.
- **A generative reward model** — a reward model that *generates* a judgment rather than emitting a scalar, for the open-ended tasks where a single number is a poor reward.
- **Tool-call schema and special tokens, interleaved thinking, and a Quick Instruction mode** — the agentic plumbing: a structured tool-call format, reasoning tokens that interleave with tool calls, and a fast-response path for when no deliberation is needed.

The infrastructure around this is its own contribution: FP4 quantization-aware training for the MoE expert weights, full-vocabulary on-policy distillation with an efficient teacher-scheduling scheme, a preemptible and fault-tolerant rollout service, an RL framework that scales to million-token episodes, and a sandbox with a unified interface over four execution substrates for agentic training. None of these are model ideas, but at this scale the infrastructure *is* the moat — a clean RL story is worth little if the rollout service falls over at 1M-token context.

## What the numbers say

The evaluation tells a consistent two-part story: V4 is the strongest *open* model across the board, and it has closed most of the gap to the closed frontier without quite reaching the very top.

![Where V4-Pro-Max sits, axis by axis](/imgs/blogs/deepseek-v4-million-token-context-moe-8.png)

Breaking it down by capability, with the baselines named:

| Capability | vs open SOTA (Kimi-K2.6, GLM-5.1) | vs prev frontier (GPT-5.2, Gemini-3.0-Pro) | vs top frontier (GPT-5.4, Gemini-3.1-Pro) |
|---|---|---|---|
| Knowledge (SimpleQA, MMLU-Pro, GPQA, HLE) | Ahead — significantly on SimpleQA, marginally elsewhere | Near par | **Trails** — knowledge is the weakest axis |
| Reasoning (large thinking budget) | Ahead | Ahead | Just behind (a few months) |
| Agentic tasks | Near par | Slightly behind | Behind |
| Long-context (1M, synthetic + real) | Ahead | Ahead | **Surpasses** Gemini-3.1-Pro on academic long-context |

A few specifics worth pulling out. On **knowledge**, V4-Pro-Max significantly outperforms leading open models on SimpleQA and Chinese-SimpleQA and holds a marginal lead on MMLU-Pro, GPQA, and HLE, but it still trails Gemini-3.1-Pro on knowledge tasks — the report frankly describes the gap as "significantly closed" rather than erased. On **reasoning**, with the thinking budget turned up, V4-Pro-Max beats GPT-5.2 and Gemini-3.0-Pro and lands marginally short of GPT-5.4 and Gemini-3.1-Pro; the authors estimate a roughly **3–6 month** development gap to the absolute frontier. On **agentic** tasks it is on par with the best open models and slightly behind the closed frontier, though an internal evaluation has it outperforming Claude Sonnet 4.5 and approaching Opus 4.5. **Long context** is where it is most clearly ahead — it surpasses Gemini-3.1-Pro on academic 1M-token benchmarks, which is the one place the efficiency architecture turns directly into a capability win.

V4-Flash is the predictable shape of a smaller sibling: lower on knowledge (you cannot fit as many facts in 284B), competitive on reasoning once you give it a larger thinking budget, matching Pro on several agent tasks, and trailing on the hardest problems.

**What's load-bearing in their setup that might not transfer.** The efficiency numbers are quoted at a 1M-token context, which is exactly where compressed/sparse attention pays off most; at 4–32K tokens — where most production traffic actually lives — the relative savings are smaller and the report does not lead with them. The most aggressive "~2% of GQA8-BF16" KV figure also assumes the FP4/FP8 storage path, which presumes hardware that handles those formats well. And the headline reasoning numbers use a large thinking budget, so the "near-frontier" claim is a near-frontier-*at-high-compute* claim. None of this is dishonest — it is all stated — but it means the right way to read the efficiency story is "an order of magnitude cheaper *at the long-context regime it was built for*," not "an order of magnitude cheaper everywhere."

Concretely: a chat workload whose median request is 2K tokens of prompt and 500 tokens of output spends almost none of its time in the regime where compressed attention pays off — the KV cache for 2.5K tokens is small however you store it, and the per-token cost is dominated by the MoE FFN, not attention. For that workload V4's attention architecture is roughly cost-neutral against a V3.2-style design; the savings only become decisive when the *same* model is handed a 400K-token codebase or a book-length document, where the KV cache would otherwise be the binding constraint on batch size. V4 is engineered for the tail of the length distribution, and the deployment decision follows directly: reach for it where your tail actually is — repository-scale code assistance, long-document analysis, multi-hour agent trajectories — and expect ordinary gains, not an order of magnitude, on short-prompt chat.

## Reading it as a skeptic

**What's genuinely strong.** The co-design is the real achievement. Plenty of papers cut attention FLOPs *or* KV memory; V4 cuts both at once, and then stacks numeric-precision compression on top of structural compression so the two multiply rather than compete. mHC is a clean, principled answer to residual-stream stability — "constrain the mixing matrix to a manifold whose spectral norm is bounded by construction" is the kind of fix that generalizes far beyond this model. And the specialist-then-distillation pipeline is a genuinely better factorization of post-training than the all-at-once recipe, because it lets each domain's RL be tuned to its own reward without cross-domain interference.

**What's weak or under-specified.**

- **The long-context curriculum is a black box.** "Natively supports 1M context" with no described length schedule or data mixture is the biggest omission. Long-context *ability* (not just the architecture admitting long inputs) usually comes from a carefully staged curriculum, and its absence makes the 1M results hard to reproduce.
- **The efficiency comparison is favorably framed.** Quoting FLOPs and KV at 1M tokens is the regime that flatters compressed attention most. A curve of relative cost across 4K → 1M would be more honest than a single best-case point.
- **Some efficiency is latent, not realized.** The "FP4×FP8 gives ~⅓ more on future hardware" line means part of the win is waiting on accelerators, not bankable today.
- **No clean ablation isolating each lever.** The report does not (in the read material) give a controlled study that turns CSA, HCA, mHC, and Muon on and off independently to attribute the gains. With four big simultaneous changes, you cannot tell from the headline numbers how much each one bought — and that is exactly the experiment that would tell a practitioner which idea to copy.

**What would change my mind.** If DeepSeek released a same-data, same-token-budget ablation showing V4's architecture beating a V3.2-style baseline at *short* context (≤ 8K) as well as long, I would upgrade my read from "an excellent long-context efficiency play" to "a strictly better transformer." Conversely, if independent reproductions found the 1M-context quality degrades sharply on adversarial needle-in-haystack tests — where heavy compression should hurt most — I would conclude that the KV savings are partly bought with recall the synthetic benchmarks don't stress. The single most decision-relevant missing experiment is the per-lever ablation; everything else I can infer, but attribution I cannot.

## What I'd build with this

1. **Port mHC into a small dense model first.** The doubly-stochastic residual-mixing constraint is architecture-agnostic and cheap to test. I would add it to a 1–3B dense transformer and measure training stability at increased depth before trusting it at scale — it is the most reusable idea in the paper and the easiest to validate in isolation.
2. **A CSA-only serving profile for retrieval workloads.** For long-document QA where recall dominates, I would deploy a configuration that biases toward CSA layers and tune $k$ (the top-$k$ block budget) per request, trading a little latency for recall on the queries that need it — effectively a per-request quality dial built on the indexer.
3. **Specialist-then-distillation for a narrow product.** Even without a frontier base model, the Stage-1/Stage-2 split is worth copying: train two or three small specialists for the domains your product actually serves, then distill on-policy into one deployable student. The win is that each specialist's RL reward can be clean and domain-specific.
4. **A Muon-vs-AdamW bake-off at my scale.** Muon's payoff is supposed to grow with model size, but the only way to know where the crossover is for a given budget is to run both. I would run a controlled comparison at the largest scale I can afford and treat the optimizer as a tunable, not a default.
5. **Measure the FP4 indexer's robustness directly.** Before trusting FP4 index keys in production, I would sweep the indexer precision (FP4 → FP8 → BF16) against a recall benchmark to find where coarse quantization starts dropping the *right* blocks — the selection is only robust until it isn't.

## References

- DeepSeek-AI. *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence.* arXiv:2606.19348. [abstract](https://arxiv.org/abs/2606.19348) · [PDF](https://arxiv.org/pdf/2606.19348)
- Related on this blog:
  - [Trainable sparse attention: NSA vs DSA](/blog/machine-learning/large-language-model/trainable-sparse-attention-nsa-vs-dsa) — the sparse-selection lineage CSA builds on.
  - [Multi-head Latent Attention (MLA)](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) and [KV cache](/blog/machine-learning/large-language-model/kv-cache) — the KV-compression background.
  - [The DeepSeekMoE lineage: fine-grained and shared experts](/blog/machine-learning/large-language-model/deepseek-moe-lineage-fine-grained-shared-experts) — the MoE backbone V4 keeps.
  - [Muon and Moonlight](/blog/paper-reading/large-language-model/muon-moonlight) — the optimizer, in depth.
  - [DeepSeek-R1: incentivizing reasoning via RL](/blog/paper-reading/large-language-model/deepseek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning) — the GRPO post-training V4 reuses per-domain.
