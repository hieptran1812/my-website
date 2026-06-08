---
title: "Muon is Scalable for LLM Training: Inside Moonlight"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "Large Language Model"
tags:
  - muon-optimizer
  - moonlight
  - mixture-of-experts
  - newton-schulz
  - scaling-laws
  - distributed-training
  - weight-decay
  - adamw
description: "How two tiny fixes to the Muon optimizer make matrix orthogonalization scale to a 16B-parameter MoE, doubling compute efficiency over AdamW."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/muon-moonlight-1.png"
readTime: 30
---

For the better part of a decade, the optimizer question in large-model training has had a boring answer: use AdamW, tune three numbers, and move on. AdamW is not the theoretically best optimizer anyone has proposed; it is the optimizer that refuses to blow up across enough orders of magnitude that nobody wants to risk the alternative on a run that costs a few million dollars. Every few months a paper claims a faster optimizer on a small benchmark, and every few months that optimizer fails to survive contact with a real pretraining run. The graveyard is large, and the bodies in it mostly died the same way: a clever update rule that looked great on a 100M-parameter toy diverged, drifted, or quietly underperformed once it met a real schedule, a real data mixture, and a real sharded cluster.

Muon is the latest candidate, and on paper it has a genuinely different idea: instead of normalizing each weight element independently the way Adam does, it orthogonalizes the entire momentum matrix before stepping, using a cheap polynomial iteration. On small language models, that buys real efficiency. The open question — the one that has killed every predecessor — was whether the trick survives at billions of parameters and trillions of tokens, in a sharded distributed setting, across both pretraining and fine-tuning. This paper, the Moonlight technical report from Moonshot AI, answers that question by actually training a 16B-parameter mixture-of-experts model on 5.7 trillion tokens and reporting what broke and what they did about it.

The punchline is unusually clean for an optimizer paper. Vanilla Muon does not survive at scale on its own — weights drift until they exceed the precision range of bf16. But two small fixes, neither of them clever in a way that requires a PhD to appreciate, make it work: add the same decoupled weight decay that AdamW already uses, and rescale each matrix's update so its root-mean-square magnitude matches what AdamW would have produced. With those in place, Muon matches AdamW's loss using roughly 52% of the training FLOPs — about a 2x improvement in compute efficiency — and the resulting Moonlight model sits on the Pareto frontier of benchmark performance versus training budget.

![One Muon update step for a weight matrix](/imgs/blogs/muon-moonlight-1.png)

The diagram above is the mental model: a single Muon step takes the gradient, accumulates Nesterov momentum, runs five Newton-Schulz iterations to orthogonalize that momentum into a step direction, rescales the step by `0.2 * sqrt(max(A,B))` so it lands in AdamW's update-RMS range, and finally subtracts both that step and a decoupled weight-decay term scaled by `lambda=0.1`. Everything in this article is an elaboration of that one picture — why each box exists, what happens when you remove it, and what it costs to run at scale.

> [!tldr] TL;DR
> - **What it claims:** Two fixes — decoupled weight decay and a per-matrix update-RMS rescale — make Muon work out-of-the-box at LLM scale, reusing AdamW's tuned hyperparameters with no Muon-specific search; the result is ~2x compute efficiency (matching AdamW at ~52% of FLOPs).
> - **Why it matters:** Optimizer efficiency is multiplicative with every other efficiency you have. A 2x sample-efficient optimizer is, to first order, a 2x cheaper pretraining bill on the same architecture and data, and it composes with MoE sparsity and better data.
> - **Most surprising finding:** Muon updates weights in more diverse singular directions than AdamW (higher SVD entropy for >90% of matrices), and the gap is largest for MoE router weights — which is a mechanistic reason MoE models benefit more from Muon than dense ones.
> - **Where it fails:** It is a hybrid, not a pure-Muon, optimizer — norms, embeddings, and the LM head still use AdamW. And the benefit does not transfer to an AdamW-pretrained checkpoint: applying Muon only at SFT gives no advantage, so you cannot retrofit it onto someone else's base model.

## Context: what came before

The lineage here is short but pointed. Adam (2014) and its decoupled-weight-decay variant AdamW (2017) define the default. Adam keeps a per-element running average of the gradient and of the squared gradient, then divides one by the square root of the other. The effect is a coordinate-wise rescaling: every scalar weight gets its own adaptive learning rate based on its own gradient history. This is robust precisely because it treats the weight tensor as a bag of independent scalars — there is no way for the optimizer to make a structurally bad assumption about how those scalars relate, because it makes no assumption at all.

Muon (Jordan et al., 2024) starts from the opposite premise: the structure of a weight matrix matters, and you should exploit it. A weight matrix maps one vector space to another, and its singular value spectrum tells you how it stretches different directions. If the optimizer keeps pushing updates along the same few dominant singular directions, the matrix collapses toward low effective rank — a handful of directions do all the work and the rest atrophy. Muon's answer is to orthogonalize the momentum matrix before taking a step, which makes the update "isomorphic" in the sense that it spreads energy across directions rather than concentrating it. The orthogonalization is done with a Newton-Schulz iteration, a matrix polynomial that drives the singular values of the momentum toward 1 without ever computing an SVD.

The framing that makes this principled, following Bernstein et al. (2024), is that Muon is steepest descent under a spectral-norm constraint on the update, whereas Adam is steepest descent under a max-of-max (RMS-like) norm. The spectral norm is the natural operator norm for a matrix that acts as a linear map, so the argument is that Muon's geometry is better matched to what a weight matrix actually does. This is a satisfying story, but stories do not train models. The gap this paper fills is empirical and engineering: Muon had only been shown to work on small-scale LM training, with no evidence it would survive the three things that kill optimizers in practice — extreme scale, distributed sharding, and the pretraining-to-fine-tuning handoff. Those are exactly the three open questions the paper sets out to close.

## Contributions

1. **Two minimal modifications that make Muon scale.** Adding AdamW-style decoupled weight decay and a per-matrix update-RMS rescale so Muon trains stably at 16B parameters / 5.7T tokens without the weight-norm blowup that breaks vanilla Muon. Crucially, the rescale is tuned so Muon's update RMS matches AdamW's, which means AdamW's learning rate and weight decay transfer directly with no Muon-specific hyperparameter search.
2. **Distributed Muon.** A ZeRO-1-style sharded implementation (Algorithm 1) that reconciles Newton-Schulz's need for the full gradient matrix with element-wise optimizer-state partitioning, at a memory cost of 0.5x AdamW's extra optimizer state and a communication cost of (1-1.25)x AdamW's.
3. **A scaling-law measurement of the efficiency gain.** Fitted loss-versus-compute curves across five model sizes (399M-1.5B) showing Muon needs ~52% of AdamW's FLOPs to reach the same loss — about 2x sample efficiency.
4. **Moonlight, an open 16B-A3B MoE.** A DeepSeek-V3-style mixture-of-experts model trained entirely with Muon, released with base and instruct checkpoints, that advances the performance-versus-training-budget Pareto frontier against Llama3.2-3B, Qwen2.5-3B, and DeepSeek-V2-Lite.
5. **A mechanistic and a cautionary empirical finding.** Muon-trained weights have higher SVD entropy (more diverse update directions), with the largest gap on MoE routers; and the Muon advantage requires optimizer consistency across pretraining and SFT — mismatched pairs lose the gain.

## Method

The method has four moving parts that build on each other: the Newton-Schulz core that defines Muon, decoupled weight decay to keep weights in range, the update-RMS rescale that both stabilizes shape-dependent updates and aligns Muon with AdamW, and the distributed implementation that makes all of it runnable on a sharded cluster. We take them in that order, defining each symbol as it appears.

### Newton-Schulz: the core operation

Muon operates only on matrix-shaped parameters — the 2D weight matrices inside attention and the FFN. Let $g_t$ be the gradient of such a matrix at step $t$. Muon first accumulates Nesterov-style momentum with coefficient $\mu = 0.95$, producing a momentum matrix $M_t$. The novelty is what happens next: instead of element-wise normalization, Muon orthogonalizes $M_t$.

Exact orthogonalization would mean computing the SVD $M_t = U \Sigma V^\top$ and replacing it with $U V^\top$ — the closest orthogonal matrix, with all singular values set to 1. SVD is too expensive to run every step on every matrix. Newton-Schulz approximates $UV^\top$ with a matrix polynomial. Starting from the Frobenius-normalized momentum $X_0 = M_t / \lVert M_t \rVert_F$, it iterates:

$$
X_k = a\,X_{k-1} + b\,(X_{k-1} X_{k-1}^\top) X_{k-1} + c\,(X_{k-1} X_{k-1}^\top)^2 X_{k-1}
$$

with coefficients $a = 3.4445$, $b = -4.7750$, $c = 2.0315$ (the same values as the original Muon). These are chosen so the scalar map $\sigma \mapsto a\sigma + b\sigma^3 + c\sigma^5$ pushes every singular value of $X_{k-1}$ toward 1. After $N$ iterations the singular spectrum of $X_N$ is approximately flat, so $X_N \approx U V^\top$ — an approximately orthogonal step direction $O_t$. The paper uses $N = 5$ iterations; $N = 10$ gives a more accurate orthogonalization but no performance gain, so the extra five iterations are wasted compute.

The intuition for why this helps: orthogonalizing the update prevents the weight matrix from collapsing into a few dominant singular directions. Each step pushes energy into many directions at once, keeping the matrix high-rank and the representation expressive. This is the spectral-norm geometry from Bernstein et al. made concrete — the orthogonalized step is the steepest-descent direction when you constrain the update's spectral norm rather than its element-wise magnitude.

It is worth being concrete about what "the singular values go to 1" buys you, because it is the crux of the whole method. Suppose the raw momentum $M_t$ has singular values $[8, 3, 0.5, 0.1]$ — a typical spiky spectrum where one direction dominates. A plain gradient step would move the weight mostly along that first direction, reinforcing whatever structure already dominates. Newton-Schulz instead returns a step whose singular values are all $\approx 1$: the directions are preserved (the left and right singular vectors $U$ and $V$ survive) but their relative magnitudes are equalized. So the weak directions — the $0.1$ direction that a raw step would have nearly ignored — get the same push as the strong one. Over thousands of steps this is the difference between a matrix that slowly degenerates toward rank-1 and one that keeps using its full capacity. The cost of this is five matrix-matrix products per matrix per step (each iteration is two products, $X X^\top$ and then a multiply), which is why it has to run in bf16 and why the team stops at $N=5$: the marginal accuracy of $N=10$ does not move the loss, so the extra products are pure overhead.

Two implementation details matter for anyone porting this. First, the iteration is run on the thinner orientation of the matrix (transpose if $A > B$) because the gram matrix $X X^\top$ is then the smaller of the two possible products, which saves compute. Second, Frobenius-normalizing $X_0$ is not cosmetic — the quintic polynomial only contracts singular values toward 1 inside a bounded region, so feeding it an un-normalized momentum with large singular values can push the iteration outside its region of convergence and produce garbage. The normalization guarantees the largest singular value starts at most 1, keeping every subsequent iterate well-behaved.

Here is the core loop in PyTorch-shaped pseudocode. Note that comments are indented inside the function so they never sit at column zero:

```python
import torch

def newton_schulz(M, steps=5, eps=1e-7):
    """Approximate the orthogonal factor U V^T of M via a quintic
    matrix polynomial. Runs in bf16; coefficients match Jordan et al.
    M is a 2D momentum matrix of shape [A, B]."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = M.bfloat16()
    X = X / (X.norm() + eps)               # Frobenius-normalize -> X_0
    transposed = X.size(0) > X.size(1)     # iterate on the thinner side
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T                        # gram matrix
        B = b * A + c * (A @ A)            # b*XX^T + c*(XX^T)^2
        X = a * X + B @ X                  # quintic update
    if transposed:
        X = X.T
    return X                               # approx orthogonal step O_t

def muon_step(W, grad, momentum_buf, lr, mu=0.95, wd=0.1, rms_scale=0.2):
    """One Muon update for a single matrix parameter W of shape [A, B]."""
    momentum_buf.mul_(mu).add_(grad)               # heavy-ball accumulate
    M = grad.add(momentum_buf, alpha=mu)           # Nesterov look-ahead
    O = newton_schulz(M, steps=5)                  # orthogonalize
    A, B = W.shape
    scale = rms_scale * (max(A, B) ** 0.5)         # 0.2 * sqrt(max(A,B))
    update = O * scale + wd * W                     # step + decoupled WD
    W.add_(update, alpha=-lr)                       # W <- W - lr * update
    return (O * scale).pow(2).mean().sqrt()        # update RMS, for logs
```

The two lines that turn vanilla Muon into the version that scales are `scale = rms_scale * (max(A, B) ** 0.5)` and `update = O * scale + wd * W`. The next two subsections explain why each is necessary.

### Decoupled weight decay

Vanilla Muon, including Jordan's original implementation, has no weight decay. At small scale this is fine. At LLM scale it is fatal: the paper reports that without weight decay, both the weight RMS and the layer-output RMS keep growing throughout training, and eventually the output RMS exceeds the high-precision range of bf16. Once activations live in the low-precision tail of bf16, rounding error corrupts the forward pass and performance degrades. Vanilla Muon also converges faster early in training, which is seductive, but some weights blow up over time and final performance suffers.

The fix is the same decoupled weight decay AdamW already uses. With $\lambda$ the weight-decay ratio, $\eta_t$ the learning rate, and $O_t$ the orthogonalized step, the update becomes (Eq. 3):

$$
W_t = W_{t-1} - \eta_t\,(O_t + \lambda\,W_{t-1})
$$

The term $\lambda W_{t-1}$ pulls weights back toward zero each step, counteracting the drift. Decoupled means the decay is applied directly to the weights rather than folded into the gradient, so it is not warped by the orthogonalization. Moonlight uses $\lambda = 0.1$ for every training stage. The payoff is visible in the over-train regime: on an 800M model trained to 100B tokens (roughly 5x the compute-optimal budget), validation loss orders as Muon+WD < vanilla-Muon < AdamW. Adding weight decay does not just rescue Muon — it makes Muon beat AdamW where vanilla Muon was losing.

![Why vanilla Muon broke at LLM scale](/imgs/blogs/muon-moonlight-2.png)

The before-after above is the failure-and-fix in one frame. On the left, the three pathologies of vanilla Muon at scale: weight RMS grows unboundedly, output RMS exceeds bf16's usable range, and the update magnitude is shape-dependent in a way we are about to make precise. On the right, the two fixes — decoupled weight decay and the RMS rescale — and the empirical result that the fixed version beats both AdamW and vanilla Muon on validation loss.

### Consistent update RMS

The third pathology on the left of that figure is subtler and is the more interesting of the two fixes. Lemma 1 in the paper makes a clean statement: for a full-rank matrix parameter of shape $[A, B]$, the theoretical RMS of Muon's orthogonalized update is $1 / \sqrt{\max(A, B)}$. Because $O_t$ is approximately orthogonal, its singular values are all $\approx 1$, so its Frobenius norm is $\sqrt{\min(A,B)}$ and its element-wise RMS is $\sqrt{\min(A,B)} / \sqrt{AB} = 1/\sqrt{\max(A,B)}$.

The consequence is that **update magnitude depends on matrix shape**, and that is a problem in a real model where matrices have wildly different shapes. A dense MLP weight has a large $\max(A,B)$, so its update RMS is tiny — that matrix is under-trained. A per-head key/value matrix in GQA or MLA attention has a small $\max(A,B)$, so its update RMS is large — that matrix is pushed hard enough to cause instability. The same nominal learning rate produces effective learning rates that vary by an order of magnitude across the layer types in a single model.

The fix is to multiply each matrix's update by $\sqrt{\max(A,B)}$, which cancels the shape dependence and gives every matrix the same update RMS. There is a second, equally important reason to rescale, beyond uniformity: choosing the constant lets us match AdamW. AdamW's empirical update RMS is roughly 0.2-0.4. The paper rescales Muon's update RMS into that range with the constant $0.2$. The final update rule (Eq. 4) is:

$$
W_t = W_{t-1} - \eta_t\,\big(0.2 \cdot O_t \cdot \sqrt{\max(A,B)} + \lambda\,W_{t-1}\big)
$$

Why does matching AdamW's update RMS matter so much in practice? Because it means Muon can directly reuse the learning rate and weight decay that were already tuned for AdamW. There is no Muon-specific hyperparameter search — and hyperparameter search at this scale is the single most expensive part of adopting a new optimizer. The paper notes that Jordan's original $\sqrt{\max(1, A/B)}$ scaling is equivalent up to a global scale when all matrices share their second dimension; the $\sqrt{\max(A,B)}$ form generalizes cleanly to the heterogeneous shapes in a real MoE.

The ablation behind this choice is Table 1 (800M model, 4B tokens, with the MLP shape deliberately set to $[H, 4H]$ to exaggerate the shape effect):

| Method | Train loss | Val loss | Query weight RMS | MLP weight RMS |
|---|---|---|---|---|
| Baseline ($\times 0.2\cdot\sqrt{H}$) | 2.734 | 2.812 | 3.586e-2 | 2.52e-2 |
| Update Norm (Eq. 6) | **2.720** | 2.789 | 4.918e-2 | 5.01e-2 |
| Adjusted LR (Eq. 7) | 2.721 | 2.789 | 3.496e-2 | 4.89e-2 |

Both "Update Norm" and "Adjusted LR" beat the baseline on validation loss (2.789 vs 2.812), and both pull the MLP weight RMS up from a starved 2.52e-2 to ~4.9e-2, closing the gap with the query weights. The paper chooses "Adjusted LR" because it is the cheaper of the two to compute — there is no measurable quality difference, so cost decides. One important scoping note: AdamW is still used for non-matrix parameters. RMSNorm scales, the LM head, and the token embeddings are 1D or otherwise not matrix-shaped in the relevant sense, so Muon does not apply to them; they keep AdamW, sharing the same learning rate and weight-decay schedule. This is why the resulting optimizer is honestly described as a hybrid.

The following table summarizes how the hybrid optimizer differs from plain AdamW and plain Muon:

| Aspect | AdamW | Vanilla Muon | This paper (Muon + fixes) |
|---|---|---|---|
| Matrix-param update | element-wise $m/\sqrt{v}$ | orthogonalized momentum | orthogonalized + RMS rescaled |
| Non-matrix params | AdamW | (n/a) | AdamW |
| Weight decay | decoupled, tuned | none | decoupled, $\lambda=0.1$ |
| Update RMS | ~0.2-0.4 empirically | $1/\sqrt{\max(A,B)}$, shape-dependent | rescaled to ~0.2-0.4, shape-free |
| Optimizer-state buffers | 2 (first + second moment) | 1 (momentum) | 1 (momentum) |
| Hyperparameter search | required | Muon-specific | reuse AdamW's directly |

### Distributed Muon

The last piece is making this run on a real cluster. The standard memory-efficient setup is ZeRO-1, which partitions optimizer states across the data-parallel (DP) group element-wise: each DP rank owns a slice of every parameter's optimizer state and only ever touches that slice. For AdamW this is perfect, because AdamW's update is element-wise — rank $r$ can compute its slice of the update from its slice of the gradient and its slice of the moments, with no cross-rank dependency.

Muon breaks this. Newton-Schulz operates on the **whole matrix** — the gram matrix $X X^\top$ couples every element to every other element, so you cannot compute a slice of the orthogonalized update from a slice of the gradient. Distributed Muon (Algorithm 1) reconciles the two by adding exactly two operations on top of ZeRO-1 AdamW: a DP-gather to assemble the full gradient matrix $G$, and a discard step that throws away all but the local partition of the resulting update.

![Distributed Muon on top of ZeRO-1](/imgs/blogs/muon-moonlight-4.png)

The dataflow above shows the full sequence. The DP ranks each hold a gradient shard; a reduce-scatter assembles the gradient $G$ and momentum is applied; a DP-gather collects $G$ into the full matrix (in bf16); Newton-Schulz runs once on the full $G$ to produce $U$; each rank keeps only its local slice $u$ of $U$ and discards the rest; the local update is applied; and a final all-gather reconstructs the full parameters. The algorithm also returns $\sqrt{\mathrm{mean}(u^2)}$, the update RMS, for logging — which is how the team watches the per-matrix update magnitudes stay in AdamW's range during the run.

The cost accounting is the part that decides whether anyone uses this:

- **Memory.** Muon keeps a single momentum buffer per matrix versus AdamW's two moment buffers. So Muon's extra optimizer memory is half that of Distributed AdamW — a real win, not a wash. The hybrid keeps AdamW only for the small set of non-matrix params, so the bulk of the parameter count enjoys the 0.5x.
- **Communication.** The worst-case Distributed Muon communication is (1, 1.25)x of Distributed AdamW. The upper bound comes from 4 fp32 gradient reduce-scatters + 2 bf16 gradient gathers + 4 fp32 parameter all-gathers, versus AdamW's 4 + 4. With multiple DP groups the empirical cost is closer to the lower bound of 1x. Newton-Schulz itself runs in bf16, and there is an extra bf16 tensor-parallel gather if TP is enabled.
- **Latency.** End-to-end optimizer latency is negligible compared to forward/backward — about 1-3% of a step. The paper states there is no noticeable latency overhead compared to the AdamW counterpart, and notes a planned PR to upstream Distributed Muon into Megatron-LM.

A worked sanity check on the memory claim: for the 15.29B non-embedding parameters that Muon governs, AdamW would carry two fp32 moment buffers, while Muon carries one fp32 momentum buffer. That is the difference between ~122 GB and ~61 GB of optimizer state for the matrix params alone (at 4 bytes/param/buffer), sharded across the DP group either way. Halving the optimizer-state footprint is the kind of thing that lets you fit a larger micro-batch or a bigger model on the same hardware, which is itself an efficiency lever on top of the sample-efficiency one.

The communication picture deserves a second look because it is where most people expect Muon to fall apart, and it mostly does not. The fear is reasonable: Newton-Schulz needs the full matrix, so naively you would gather every gradient matrix to every rank, which sounds like it should explode bandwidth. The reason it does not is that the gather is in bf16, not fp32, and it replaces work that ZeRO-1 was already doing rather than adding a whole new collective. Accounting the worst case as a multiple of Distributed AdamW's traffic: AdamW does 4 fp32 gradient reduce-scatters and 4 fp32 parameter all-gathers per step. Distributed Muon does the same 4 fp32 reduce-scatters and 4 fp32 all-gathers, plus 2 bf16 gradient gathers for the Newton-Schulz step. Because the extra gathers are half-precision, they add at most 0.25x to the byte count, giving the (1, 1.25)x upper bound. With multiple DP groups the gather can be overlapped and partially amortized, pushing the real cost toward 1x. The practical upshot is the latency line: optimizer time stays at ~1-3% of a step, which is below the noise floor of most training runs, so the comm overhead never becomes the bottleneck the naive analysis predicts. The one caveat is tensor parallelism: if TP is enabled there is an additional bf16 gather across the TP group to assemble the full matrix, which the cost model accounts for but which grows with TP degree.

## Experiments

There are two experiments that matter: the scaling-law measurement that quantifies the efficiency gain in a controlled setting, and the end-to-end Moonlight model that shows the gain survives all the way to a competitive 16B MoE. We also have apples-to-apples and ablation tables that isolate the contribution of Muon specifically.

### The scaling-law result

The headline efficiency number comes from fitting loss-versus-compute curves across a grid of small models. The grid (Table 2) spans 399M / 545M / 822M / 1.1B / 1.5B parameters (excluding embeddings), with 12-20 heads, 12-20 layers, hidden sizes 1536-2560, trained on 8.92B-38.91B tokens following a Chinchilla/Kaplan compute-optimal setup. These are Llama-architecture dense models, and critically, Muon reuses AdamW's compute-optimal hyperparameters — which is only legitimate because of the update-RMS matching from the method section.

The fitted curves (Table 3), giving LM loss $L$ at sequence length 8K as a function of compute $C$, are:

| Optimizer | Fitted loss-vs-compute curve |
|---|---|
| Muon | $L = 2.506 \cdot C^{-0.052}$ |
| AdamW | $L = 2.608 \cdot C^{-0.054}$ |

To match AdamW's loss, Muon needs only ~52% of the training FLOPs — Figure 1a in the paper annotates this as "0.519x FLOPs." That is the ~2x compute/sample efficiency claim, stated precisely. The exponents are nearly identical ($-0.052$ vs $-0.054$), so the two optimizers scale at almost the same rate; the win is in the coefficient, a near-constant-factor head start that persists across the measured range. This is the right shape for a believable optimizer result: a clean constant-factor improvement, not a magical change in the scaling exponent that would be too good to be true.

It is worth deriving the 0.519x for ourselves, because the two-curve presentation can hide how the number arises. Pick a target loss $L^\star$. AdamW reaches it at compute $C_A$ where $2.608\,C_A^{-0.054} = L^\star$, and Muon reaches it at $C_M$ where $2.506\,C_M^{-0.052} = L^\star$. Taking the ratio and solving, $C_M / C_A = (2.506/2.608)^{1/0.052} \cdot C_A^{(0.054/0.052) - 1}$, and over the measured compute range this evaluates to roughly 0.52 — Muon hits the same loss for about half the FLOPs. The mild dependence on $C_A$ (because the exponents are not exactly equal) is why the paper reports a single representative ratio rather than a constant: the savings drift slightly with scale, but stay near 2x across the grid. The honest caveat is that this is an extrapolation in two senses — the curves are fit on dense models up to 1.5B, and the "match at half the FLOPs" claim is read off the fitted curves, not measured by training a Muon model to exactly AdamW's loss and counting FLOPs directly. It is a strong result, but it is a curve-fit result.

### Moonlight at 5.7T tokens

The full-scale validation is Moonlight-16B-A3B, whose architecture we summarized earlier and reproduce here for the experiments discussion.

![Moonlight-16B-A3B architecture](/imgs/blogs/muon-moonlight-3.png)

Moonlight is a DeepSeek-V3-style ("deepseek-v3-small") MoE: 27 layers, hidden size 2048, MLA attention with 16 heads and a KV LoRA rank of 512, 64 routed experts plus 2 shared experts with top-6 routing, layer 0 dense and the rest MoE, vocabulary 163,840, 8K pretraining context. It has 15.29B total parameters excluding embeddings (16B including) and activates 2.24B (3B including embeddings) per token. It was pretrained from scratch on 5.7T tokens, entirely with the Muon optimizer described above.

The headline comparison (Table 5) puts Moonlight against three comparable open models. Bold marks the best in the paper:

| Benchmark (Metric) | Llama3.2-3B | Qwen2.5-3B | DSV2-Lite | Moonlight |
|---|---|---|---|---|
| Activated params | 2.81B | 2.77B | 2.24B | 2.24B |
| Training tokens | 9T | 18T | 5.7T | 5.7T |
| Optimizer | AdamW | Unknown | AdamW | Muon |
| MMLU (5-shot) | 54.7 | 65.6 | 58.3 | **70.0** |
| MMLU-pro (5-shot) | 25.0 | 34.6 | 25.5 | **42.4** |
| BBH (3-shot) | 46.8 | 56.3 | 44.1 | **65.2** |
| TriviaQA (5-shot) | 59.6 | 51.1 | 65.1 | **66.3** |
| HumanEval (pass@1) | 28.0 | 42.1 | 29.9 | **48.1** |
| MBPP (pass@1) | 48.7 | 57.1 | 43.2 | **63.8** |
| GSM8K (4-shot) | 34.0 | **79.1** | 41.1 | 77.4 |
| MATH | 8.5 | 42.6 | 17.1 | **45.3** |
| CMath | — | 80.0 | 58.4 | **81.1** |
| C-Eval (5-shot) | — | 75.0 | 60.3 | **77.2** |
| CMMLU (5-shot) | — | 75.0 | 64.3 | **78.2** |

![Moonlight vs comparable models at 5.7T tokens](/imgs/blogs/muon-moonlight-5.png)

The matrix figure above renders the same story visually, with Moonlight's winning column in green and the single loss — GSM8K — flagged. Moonlight wins **every** benchmark except GSM8K, where Qwen2.5-3B's 79.1 edges Moonlight's 77.4 — and Qwen2.5-3B was trained on 18T tokens, more than 3x Moonlight's 5.7T. That token-budget asterisk is the whole point: Moonlight uses the fewest tokens of any MoE peer and still lands on the Pareto frontier of performance versus training budget. The most striking gaps are on the reasoning-heavy benchmarks — MMLU-pro (42.4 vs the next-best 34.6) and BBH (65.2 vs 56.3) — which is consistent with the cooldown stage being trained on the highest-quality math, code, and reasoning data.

### Apples-to-apples: Muon vs AdamW, same everything

Table 5 conflates the optimizer with architecture and data choices, so the controlled comparison (Table 4) is more directly load-bearing for the optimizer claim. It uses an intermediate 1.2T-token Moonlight checkpoint (before the LR fully decays and before cooldown). "Moonlight-A" is the identical setup but with AdamW substituted for Muon; "DSV3-Small" is DeepSeek-V3-Small (AdamW, 1.33T tokens):

| Benchmark | DSV3-Small @1.33T | Moonlight-A @1.2T (AdamW) | Moonlight @1.2T (Muon) |
|---|---|---|---|
| MMLU | 53.3 | 60.2 | **60.4** |
| MMLU-pro | — | 26.8 | **28.1** |
| BBH | 41.4 | **45.3** | 43.2 |
| TriviaQA | — | 57.4 | **58.1** |
| HumanEval | 26.8 | 29.3 | **37.2** |
| MBPP | 36.8 | 49.2 | **52.9** |
| GSM8K | 31.4 | 43.8 | **45.0** |
| MATH | 10.7 | 16.1 | **19.8** |
| CMath | — | 57.8 | **60.2** |
| C-Eval | — | 57.2 | **59.9** |
| CMMLU | — | 58.2 | 58.8 |

Holding architecture, data, and token count fixed, Muon wins almost everywhere, and the gaps are largest on math and code: HumanEval 37.2 vs 29.3 (a 7.9-point swing) and MATH 19.8 vs 16.1. BBH is the one place AdamW edges Muon at this checkpoint (45.3 vs 43.2). This is the cleanest evidence in the paper that the optimizer, not the architecture or data, is responsible for the gain — same everything, swap only the optimizer, and code/math improve disproportionately.

### Why MoE benefits most: the SVD-entropy result

The mechanistic explanation ties back to the orthogonalization story. The paper measures the singular-value-decomposition entropy of trained weight matrices — a measure of how spread-out the singular spectrum is, where higher entropy means the matrix uses more of its directions rather than concentrating in a few. Across all six weight groups (AttnQO, AttnKV, Experts, SharedExperts, Router, Dense), Muon yields higher SVD entropy than AdamW. For more than 90% of weight matrices, Muon's SVD entropy exceeds AdamW's. And the gap is largest for router weights — which is the direct mechanistic reason MoE models benefit more from Muon than dense ones: a higher-entropy router spreads tokens across experts more diversely, which is exactly what you want from a router. This is the rare optimizer paper that offers a "why," not just a "what."

### What is load-bearing and what might not transfer

The 2x efficiency number is the most transferable claim, because it comes from a controlled scaling-law sweep with reused hyperparameters — but it is measured on dense Llama-architecture models up to 1.5B, and the single large-scale validation is one 16B MoE. Whether the 0.519x FLOPs ratio holds at, say, 100B+ activated parameters is genuinely unknown from this data. The Moonlight benchmark wins are real but confounded: Moonlight differs from the baselines in architecture (MoE vs dense for Llama/Qwen), data, and token count simultaneously, so Table 5 cannot by itself attribute the wins to Muon. Table 4 is the table to trust for the optimizer claim specifically. Finally, the data composition is not disclosed — the report defers pretraining data details to an external "K. Team 2025" reference — so anyone reproducing this is reproducing the optimizer recipe, not the data recipe, and the cooldown's math/code emphasis is a confound for the reasoning-benchmark gaps.

## Critique

**What is strong.** The two fixes are minimal and well-motivated, and the paper is honest that vanilla Muon does not work at scale without them. The update-RMS-matching argument is the best part: it is not just a stability fix, it is what makes the whole recipe usable, because it removes the need for a Muon-specific hyperparameter search at a scale where that search would cost more than the savings. The controlled Table 4 comparison is the right experiment to run, and the SVD-entropy analysis gives a genuine mechanism rather than hand-waving. Releasing the model and the distributed implementation makes the claims checkable.

**What is weak or unfalsifiable.** The scaling-law sweep tops out at 1.5B dense models, and the leap to "therefore it works at 16B MoE" is an empirical bet that happens to have paid off once. There is no scaling-law data on MoE models, which is awkward given that the central mechanistic claim is that MoE benefits most. The Moonlight-vs-baselines table confounds optimizer, architecture, data, and tokens, and the prose leans on it for the headline narrative more than the controlled table warrants. The "more natural induced norm" framing is a story that the paper itself only partially tests — the spectral-norm justification is borrowed, and the proposed extension to general Schatten-$p$ norms is mentioned but not done, so we do not actually know whether spectral is special or just convenient.

**What ablation is missing.** There is no ablation isolating the weight-decay value (0.1 is asserted for all stages, but we never see 0.05 or 0.2). There is no sensitivity analysis on the 0.2 rescale constant beyond the claim that it lands in AdamW's range. There is no scaling-law curve for the MoE setting. And there is no wall-clock or GPU-hours number anywhere — the latency claim is "~1-3% of a step," but we never see end-to-end training time, GPU type, or cluster size, which makes the "2x cheaper" framing partly a FLOPs argument rather than a dollars argument.

**What would change my mind.** If a replication on a different architecture and data pipeline — ideally a dense model at the 7B-13B scale where AdamW is extremely well-understood — failed to reproduce the ~0.5x FLOPs ratio, I would downgrade the efficiency claim from "robust" to "MoE-and-data-specific." Conversely, if someone showed the SVD-entropy router gap correlates with the math/code improvement across multiple models, I would upgrade the mechanistic story from "plausible" to "load-bearing."

## What I'd build with this

1. **A pure-Muon variant.** The honest limitation is that this is a hybrid — embeddings, the LM head, and norms still use AdamW. Building and ablating a version that orthogonalizes (or sensibly handles) the embedding and LM-head matrices would close the most obvious gap and is explicitly flagged as future work. The embedding matrix is huge and shape-extreme, so the RMS rescale would matter a lot there.
2. **An MoE-specific scaling-law sweep.** Given that the router SVD-entropy gap is the central mechanism, the missing experiment is a scaling-law grid of MoE models with varying expert counts and routing top-$k$, measuring whether the FLOPs-savings ratio grows with sparsity. If it does, that is a much stronger paper than the dense sweep we got.
3. **A Schatten-$p$ generalization.** The paper frames Muon as spectral-norm (Schatten-$\infty$) steepest descent and proposes general Schatten-$p$ as future work. Building a knob that interpolates between the element-wise RMS geometry of Adam ($p$ small) and the spectral geometry of Muon ($p = \infty$) would let you tune the optimizer's geometry to the layer, which the shape-dependence analysis suggests might matter.
4. **A checkpoint-bridging fix for the pretrain-SFT mismatch.** The mismatch penalty blocks reuse of AdamW-pretrained checkpoints with Muon fine-tuning. A short "optimizer-warmup" or weight-realignment phase that makes an AdamW checkpoint amenable to Muon SFT would unlock the enormous installed base of open AdamW base models — currently you have to pretrain with Muon from scratch to get the benefit.
5. **Upstreaming Distributed Muon into mainstream frameworks.** The paper mentions a planned Megatron-LM PR. Getting the gather/discard pattern into the standard distributed-training stacks, with the bf16 Newton-Schulz path optimized, is the difference between a research result and a default people actually reach for.

## When to reach for Muon (and when not to)

Reach for Muon when you are **pretraining a matrix-heavy model from scratch** and you control the full run — especially an MoE, where the router gains are largest, and especially when compute is your binding constraint. The recipe is genuinely low-risk to adopt because the update-RMS matching lets you reuse your existing AdamW learning rate and weight decay; you are not signing up for a hyperparameter search. The memory win (0.5x extra optimizer state) and the negligible latency overhead mean the downside is small even if the efficiency gain on your specific setup turns out to be less than 2x. If you are training a large MoE on a tight token budget, this is the most credible "free lunch" the optimizer literature has produced in a while.

![Three-stage 5.7T-token pretraining schedule](/imgs/blogs/muon-moonlight-7.png)

The schedule above is the concrete recipe if you do reach for it: a 33B-token warmup ramping the learning rate to 4.2e-4 over 2k steps at batch 2048, a long cosine decay to 4.2e-5 that doubles the batch to 4096 after 200B tokens, and a final 500B-token cooldown on high-quality math/code/reasoning data that decays the learning rate to zero. Weight decay is 0.1 throughout, Newton-Schulz runs $N=5$ iterations at momentum 0.95, and the update-RMS rescale uses the 0.2 constant.

Do **not** reach for Muon when you only have an **AdamW-pretrained checkpoint** and want to bolt the optimizer on at fine-tuning time. The optimizer-interchangeability result is unambiguous on this point.

![Optimizer must match across pretrain and SFT](/imgs/blogs/muon-moonlight-6.png)

The 2x2 matrix above shows the consistency requirement. Across the four pretrain-to-SFT optimizer pairings (Table 6, Moonlight-1.2T, tulu-3, 2 epochs), Muon-pretrain + Muon-SFT wins every benchmark — MMLU 55.7, HumanEval 57.3, MBPP 55.6, GSM8K 68.0 — while mismatched pairs like Muon-to-AdamW (MMLU 50.2) and AdamW-to-Muon (GSM8K 62.1) lose the advantage. And on a public AdamW-pretrained model (Table 7, Qwen2.5-7B), Muon-SFT is on par or slightly below Adam-SFT (MMLU 70.8 vs 71.4, GSM8K 85.8 vs 89.8). The lesson is sharp: **Muon's value is in pretraining**, and it only pays off if the optimizer is consistent end-to-end. If you are doing instruction-tuning on top of someone else's Llama or Qwen base, Muon buys you nothing — keep AdamW. The mechanism for this mismatch penalty is not understood and is flagged for theoretical investigation, which is the single most interesting open question the paper leaves on the table.

## References

- **Paper (arXiv abstract):** [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) — arXiv:2502.16982, v1, 24 Feb 2025 (cs.LG), Moonshot AI and UCLA.
- **Code and models:** [MoonshotAI/Moonlight on GitHub](https://github.com/MoonshotAI/Moonlight); base checkpoint [moonshotai/Moonlight-16B-A3B](https://huggingface.co/moonshotai/Moonlight-16B-A3B) and instruct [moonshotai/Moonlight-16B-A3B-Instruct](https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct).
- [Kimi K2: Open Agentic Intelligence](/blog/paper-reading/large-language-model/kimi-k2) — the production model that took the Muon-at-scale lesson to a trillion-parameter MoE, with MuonClip stabilizing attention logits.
- [Kimi Linear: An Expressive, Efficient Attention Architecture](/blog/paper-reading/large-language-model/kimi-linear) — a sibling efficiency lever from the same lab, on the attention side rather than the optimizer side.
- [Kimi K2 Thinking: An Open-Source Reasoning Model Built on K2](/blog/paper-reading/large-language-model/kimi-k2-thinking) — how the same training stack scales into a reasoning model.
- [MoBA: Mixture of Block Attention for Long-Context LLMs](/blog/paper-reading/large-language-model/moba) — a complementary sparsity idea applied to attention for long context.
