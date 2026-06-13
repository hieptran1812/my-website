---
title: "Layer Normalization From the Inside Out: Statistics, Gradients, and Where to Put It"
publishDate: "2026-06-13"
category: "machine-learning"
subcategory: "Deep Learning"
tags:
  [
    "layer-normalization",
    "batch-normalization",
    "rmsnorm",
    "transformers",
    "deep-learning",
    "pre-norm",
    "mixed-precision",
    "pytorch",
    "training-stability",
    "qk-norm",
  ]
date: "2026-06-13"
author: "Hiep Tran"
featured: true
readTime: 51
description: "A from-the-inside-out deep dive on Layer Normalization: the per-token statistics, the dense backward pass, why Pre-LN beats Post-LN at depth, how RMSNorm and QK-norm changed the modern transformer block, the mixed-precision numerics that bite in production, and eleven war stories."
excerpt: "Layer Normalization is treated as a one-liner. It is not. This is the per-token statistics, the dense Jacobian, the Pre-LN vs Post-LN decision that makes deep transformers trainable, RMSNorm and the modern norm zoo, the bf16 numerics that produce NaNs, and eleven production war stories — with runnable PyTorch throughout."
---

Open almost any transformer implementation and you will find a line that looks like an afterthought:

```python
x = self.norm(x + self.attn(x))
```

One call. No arguments worth mentioning. It sits between the parts everyone talks about — attention, the MLP, the residual add — like punctuation. And because it looks like punctuation, most engineers carry around a one-sentence mental model of it: *"LayerNorm keeps the activations in a nice range so training doesn't blow up."*

That sentence is not wrong so much as it is hiding almost everything that matters. LayerNorm is doing four separable jobs, only one of which is "keeping activations in a range." The learnable parameters inside it are load-bearing in a way that will silently wreck your model if you treat them carelessly. The exact position of that one `self.norm` call — before the sublayer or after the residual add — is the difference between a 12-layer model that trains and a 100-layer model that diverges on step 400. And the innocent-looking mean-and-divide is a numerical minefield in mixed precision that ships NaNs to production if you compute it the obvious way.

This article takes LayerNorm apart from the inside out. We start with the statistics it computes on a single token, walk through the dense backward pass that explains *why* it helps optimization (and why the textbook "internal covariate shift" story is mostly folklore), work out the Pre-LN versus Post-LN decision that governs deep training, then trace how the modern transformer block grew RMSNorm, QK-norm, and sandwich norms. We finish with the production numerics, a capability matrix for the whole normalization family, and eleven war stories where a misunderstanding of one of these points cost someone real time.

![What LayerNorm does to one token vector](/imgs/blogs/layer-normalization-inside-out-1.webp)

The diagram above is the mental model, and we will keep returning to it: LayerNorm takes one token's feature vector, reduces it to a mean and a variance *over its own features*, subtracts the mean, divides by the standard deviation, and then re-scales with two learned vectors. Six little steps, each independent of every other token in the batch. Hold onto the "over its own features, independent of the batch" part — it is the single fact that separates LayerNorm from BatchNorm and explains nine-tenths of why transformers use it.

## Why "normalization" is the wrong word

Before any math, it is worth dismantling the assumptions you probably carry, because each one corresponds to a section below.

| What people assume | The naive picture | What is actually true |
| --- | --- | --- |
| "LayerNorm keeps activations in a nice range" | a clipping or squashing step | it re-centers and re-scales per token; the affine $\gamma,\beta$ can stretch the range right back out |
| "It fixes internal covariate shift" | it stabilizes the input distribution each layer sees | the covariate-shift story is largely wrong; it smooths the loss landscape and makes the effective learning rate self-regulating |
| "Normalization means zero-mean, unit-variance" | subtracting the mean is essential | RMSNorm drops the mean entirely and matches LayerNorm — re-scaling, not re-centering, is the load-bearing half |
| "Where you put it is an implementation detail" | a norm is a norm | Pre-LN vs Post-LN decides whether a deep stack trains at all, and was the difference between needing learning-rate warmup and not |
| "$\gamma$ and $\beta$ are just extra parameters" | minor knobs | they are what lets the layer represent the identity; leave them in weight decay and the model's scale collapses |
| "It is numerically trivial" | a mean and a divide | reducing the variance in bf16 gives NaNs at $d=4096$; production kernels reduce in fp32 |

The word "normalization" makes it sound like the goal is a tidy distribution. The goal is nothing of the sort. The standardization step is a *means*; the $\gamma$ and $\beta$ that follow let the network undo it wherever a tidy distribution is the wrong thing. What LayerNorm actually buys you is a particular geometry on the gradients — and that is a story about the backward pass, not the forward one.

> A normalization layer is not a filter you put on activations. It is a reparameterization of the loss surface that happens to be implemented as an activation transform.

## 1. The statistics: per-sample, not per-batch

Take a single token's hidden vector $x \in \mathbb{R}^{d}$, where $d$ is the model dimension (768 for BERT-base, 4096 for Llama-2-7B, 8192 for the big ones). LayerNorm computes two scalars from *this vector alone*:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \qquad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2 .
$$

Note the $\frac{1}{d}$ in the variance — this is the *biased* estimator (it divides by $d$, not $d-1$). That detail is not pedantry; it is the source of a real porting bug we will hit in the case studies. With those two scalars it standardizes and then applies a learned affine map:

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y_i = \gamma_i\, \hat{x}_i + \beta_i .
$$

Here $\epsilon$ is a small constant (PyTorch defaults to $10^{-5}$) added *inside* the square root for numerical safety, and $\gamma, \beta \in \mathbb{R}^{d}$ are learned per-feature scale and shift vectors. After standardization, $\hat{x}$ has mean zero and variance one across its $d$ features, by construction. Then $\gamma$ and $\beta$ are free to move it anywhere — including all the way back to $x$ if $\gamma = \sqrt{\sigma^2+\epsilon}$ and $\beta = \mu$. That recoverability is the point: the layer never costs the network expressiveness, it only changes the *coordinates* the optimizer works in.

The whole thing is fifteen lines of NumPy with no surprises:

```python
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    # x: (..., d). Normalize over the LAST axis only — one statistic per row.
    mu  = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)          # biased: divides by d, matches PyTorch
    x_hat = (x - mu) / np.sqrt(var + eps)        # eps INSIDE the sqrt
    return gamma * x_hat + beta
```

The crucial line is `axis=-1`. Every leading axis — batch, sequence position, attention head — is left untouched. A `(batch=32, seq=512, d=768)` tensor produces `32 * 512 = 16384` independent `(mu, var)` pairs, one per token. No token's normalization depends on any other token's value, and nothing depends on the batch. That independence is what makes LayerNorm behave identically whether you run a batch of 1 or 1024, in training or in inference, and it is exactly the property BatchNorm lacks.

In PyTorch you would never hand-roll this for production — you would use `nn.LayerNorm` — but it is worth proving to yourself that the module is doing precisely the math above:

```python
import torch, torch.nn as nn

d = 768
x  = torch.randn(32, 512, d)
ln = nn.LayerNorm(d, eps=1e-5)        # elementwise_affine=True by default -> gamma, beta

def manual_ln(x, w, b, eps=1e-5):
    mu  = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)   # unbiased=False is the catch
    return (x - mu) / torch.sqrt(var + eps) * w + b

torch.testing.assert_close(ln(x), manual_ln(x, ln.weight, ln.bias))   # passes
```

If you swap `unbiased=False` for the default `unbiased=True`, that assertion fails — by a factor of $\sqrt{d/(d-1)}$, which at $d=768$ is about $0.065\%$. Tiny. Tiny enough that it sails through unit tests and quietly degrades a model you ported from another framework. We will come back to it.

### A worked example, by hand

Numbers make the invariances concrete. Take a tiny vector $x = [2, 4, 6, 8]$ with $d = 4$. Its mean is $\mu = 5$ and its (biased) variance is $\sigma^2 = \frac{1}{4}(9 + 1 + 1 + 9) = 5$, so the standard deviation is $\sqrt{5} \approx 2.236$. Standardizing gives

$$
\hat{x} = \frac{[2,4,6,8] - 5}{2.236} = [-1.342,\ -0.447,\ 0.447,\ 1.342],
$$

which has mean zero and unit variance by construction. With $\gamma = 1$ and $\beta = 0$ the output equals $\hat{x}$. Now the part worth internalizing: rescale and shift the input, and the output does not move.

```python
x = np.array([2., 4., 6., 8.])
print(layer_norm(x, gamma=1.0, beta=0.0))         # [-1.3416 -0.4472  0.4472  1.3416]
print(layer_norm(10 * x + 100, 1.0, 0.0))         # identical, to four decimals
```

Scaling by ten and adding a hundred changes $\mu$ and $\sigma$ in lockstep, and the standardization cancels both. This is the scale-and-shift invariance from section 3 made visible, and it is the property that turns into a self-regulating learning rate once a weight matrix feeds the norm. The affine $\gamma, \beta$ are then free to re-introduce whatever scale and offset the loss actually wants — but training starts from a canonical, well-conditioned point rather than from whatever scale the layer below happened to emit.

### `normalized_shape`, and what gets reduced

`nn.LayerNorm(768)` normalizes over the last dimension. `nn.LayerNorm([512, 768])` normalizes over the last *two* — every element of a `512 * 768` block shares one mean and variance. For NLP you almost always want the former (per-token). The two-dimensional form shows up in vision transformers and in some convolutional hybrids, and getting it wrong is a subtle way to leak information across positions. The rule of thumb: **`normalized_shape` should cover exactly the axes you want to share a statistic, and nothing more.**

### Second-order gotcha: where $\epsilon$ lives

There are two ways to add $\epsilon$, and they are not the same function:

$$
\frac{x-\mu}{\sqrt{\sigma^2 + \epsilon}} \qquad \text{vs.} \qquad \frac{x-\mu}{\sqrt{\sigma^2} + \epsilon}.
$$

PyTorch, TensorFlow, and JAX all use the first (inside). Some older and hand-rolled implementations use the second (outside). For typical activations the difference is negligible, but for near-constant inputs (where $\sigma^2 \to 0$) the two diverge sharply: the inside form caps the gain at $1/\sqrt{\epsilon}$, the outside form caps it at $1/\epsilon$ — a square difference. When you load weights trained under one convention into a runtime using the other, you get a model that is *almost* right, which is the most expensive kind of wrong. Always check $\epsilon$ placement, $\epsilon$ value, and biased-vs-unbiased variance when porting.

## 2. The axis question: BatchNorm vs LayerNorm

The single most clarifying picture in this whole topic is what happens when you lay the activations out as a grid and ask: *along which axis do we compute the statistics?*

![Two normalizations, opposite axes of one tensor](/imgs/blogs/layer-normalization-inside-out-2.webp)

Lay the activations as a matrix with samples down the rows and features across the columns. LayerNorm reduces along a *row* — one sample, all its features (the green band). BatchNorm reduces down a *column* — one feature, all the samples in the batch (the blue band). They are the same operation along orthogonal axes, and that orthogonality is the entire difference in their behavior.

[BatchNorm](https://arxiv.org/abs/1502.03167), introduced by Ioffe and Szegedy in 2015, normalizes each feature (channel) using statistics gathered across the batch dimension — and, for convolutions, across the spatial dimensions too. This works beautifully for image classification with large, i.i.d. batches, and it dominated computer vision for years. But the batch dependence comes with a tax that is ruinous for sequence models:

| Property | BatchNorm | LayerNorm |
| --- | --- | --- |
| Reduces over | batch (+ spatial) | feature dimension |
| Statistics depend on other samples? | yes | no |
| Behaves the same at batch=1? | no (stats are garbage) | yes |
| Train and eval identical? | no (uses running stats at eval) | yes |
| Handles variable-length sequences? | poorly (padding pollutes stats) | yes |
| Needs synchronization across GPUs? | yes, for correct stats (SyncBN) | no |
| Stateful (running buffers)? | yes | no |

Walk down that column. BatchNorm maintains *running* mean and variance buffers during training (an exponential moving average, momentum $0.1$ by default) and switches to them at inference, because at inference you may have a batch of one and cannot compute a meaningful batch statistic. That immediately creates a train/eval discrepancy — a model that behaves differently in `.train()` and `.eval()` mode, which is a famous source of "it worked in training, it's broken in prod" bugs. For sequences it is worse: a batch of sentences padded to a common length means the padding tokens pollute the per-feature statistics, and the statistics themselves fluctuate wildly from batch to batch because language is not i.i.d. the way shuffled ImageNet crops are.

LayerNorm sidesteps every one of these. Because the statistic is computed per token over the feature axis, there is no batch to depend on, no running buffer to maintain, no train/eval split, no padding contamination (a padded position normalizes itself and is masked out downstream anyway), and nothing to synchronize across data-parallel replicas. It was, in retrospect, the obviously correct choice for transformers — and the original [Layer Normalization paper](https://arxiv.org/abs/1607.06450) (Ba, Kiros, Hinton, 2016) was motivated precisely by making normalization work for recurrent networks, where BatchNorm had never fit cleanly.

The deeper point is that BatchNorm and LayerNorm encode *different assumptions about what is comparable*. BatchNorm assumes feature $j$ is comparable across samples — that the distribution of "edge-detector activations" is a stable thing worth normalizing. LayerNorm assumes the features within one sample are comparable to each other — that a token's representation should be calibrated against itself. For tokens whose meaning is contextual and whose batch composition is arbitrary, the per-sample assumption is the sane one.

### Why keep the affine at all?

If standardization is the point, why follow it with a learned scale and shift that can undo it? Because without $\gamma$ and $\beta$ the layer would *force* every token's representation to have mean zero and unit variance, and that is a genuine constraint — it removes two degrees of freedom per vector. The affine hands them back as *learnable* degrees of freedom. The network can keep the standardized scale, amplify a feature it cares about, or restore the original distribution entirely, on a per-feature basis. In the limit $\gamma_i = \sqrt{\sigma^2 + \epsilon}$ and $\beta_i = \mu$, the layer is the exact identity, so inserting LayerNorm can never shrink the function class the model can represent — it only changes the coordinates the optimizer searches in. That is the formal sense in which the layer is "free": it is a reparameterization, not a restriction. The one place this freedom bites back is weight decay, which (as we will see) pulls $\gamma$ toward zero and silently removes the scale the network learned — the reason norm parameters must be kept out of the decay group.

## 3. Gradients: what LayerNorm actually does to optimization

Here is where the one-sentence mental model breaks down completely. The forward pass tells you LayerNorm produces zero-mean, unit-variance vectors. That is true and almost beside the point. The reason normalization accelerates training lives in the *backward* pass, and it starts with a fact that surprises people: **the LayerNorm Jacobian is dense.**

![Why the LayerNorm Jacobian is dense](/imgs/blogs/layer-normalization-inside-out-3.webp)

Because $\mu$ and $\sigma^2$ are computed from *every* feature, each normalized output $\hat{x}_i$ depends on *every* input $x_j$, not just $x_i$. Perturb a single input feature and you move the mean and the variance, which moves every output. The gradient of any output therefore flows back through every input — the figure shows the two shared statistics as the hubs that couple the whole vector together.

Write it out. Let $g_i = \partial L / \partial \hat{x}_i$ be the gradient arriving at the normalized values (which is the upstream gradient times $\gamma_i$). The gradient with respect to the inputs works out to:

$$
\frac{\partial L}{\partial x_i} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}\left( g_i - \frac{1}{d}\sum_{k=1}^{d} g_k - \hat{x}_i \cdot \frac{1}{d}\sum_{k=1}^{d} g_k\,\hat{x}_k \right).
$$

Read the three terms. The first is the raw incoming gradient. The second subtracts its mean — so the gradient that reaches $x$ is always *mean-centered*. The third subtracts the component of the gradient that is correlated with $\hat{x}$ itself — so the gradient is also made *orthogonal* to the current activation direction. Normalization does not just scale the gradient; it projects out the two directions (uniform shift and radial scaling) that the normalization itself made irrelevant. In code:

```python
def ln_backward(grad_y, x, gamma, eps=1e-5):
    d = x.shape[-1]
    mu   = x.mean(-1, keepdim=True)
    var  = x.var(-1, unbiased=False, keepdim=True)
    rstd = torch.rsqrt(var + eps)                 # 1 / sqrt(var + eps)
    x_hat = (x - mu) * rstd

    g = grad_y * gamma                            # dL/dx_hat
    grad_x = rstd * (g
                     - g.mean(-1, keepdim=True)                     # center
                     - x_hat * (g * x_hat).mean(-1, keepdim=True))  # de-correlate
    grad_gamma = (grad_y * x_hat).sum(dim=(0, 1))
    grad_beta  = grad_y.sum(dim=(0, 1))
    return grad_x, grad_gamma, grad_beta
```

You can check this against autograd in three lines, and it matches to floating-point tolerance:

```python
x = torch.randn(4, 16, 768, requires_grad=True, dtype=torch.float64)
ln = nn.LayerNorm(768).double()
y = ln(x); y.sum().backward()
gx, gg, gb = ln_backward(torch.ones_like(x), x.detach(), ln.weight.detach())
torch.testing.assert_close(x.grad, gx)            # passes
```

### Scale invariance is the real mechanism

The deepest consequence falls out of one identity. For any positive scalar $a$ and any constant shift $c$,

$$
\mathrm{LN}(a x + c\mathbf{1}) = \mathrm{LN}(x).
$$

LayerNorm is invariant to the scale and the additive offset of its input. Now chain that with the weight matrix $W$ that *produces* $x$. If the layer feeding the norm scales its weights up by a factor $a$, the pre-norm activations scale by $a$, and the norm cancels it: the output is unchanged. The network's function is invariant to the norm of those weights. This has a startling implication for the gradient — the gradient with respect to $W$ is always *orthogonal* to $W$, and the effective step you take in the meaningful (directional) part of $W$ scales like $1/\lVert W \rVert^2$. As training pushes $\lVert W \rVert$ up, the effective learning rate on that weight automatically shrinks. Normalization gives you a free, per-weight, self-tuning learning rate.

This is also why **weight decay and normalization are coupled in a way that surprises people**: weight decay shrinks $\lVert W \rVert$, which *raises* the effective learning rate, until an equilibrium forms between the two forces (van Laarhoven, 2017). When people report that "weight decay matters more than I expected" in transformer training, this feedback loop is usually why.

#### The effective learning rate, worked out

It is worth seeing the $1/\lVert W \rVert^2$ claim explicitly, because it is the single most useful fact about why normalized networks train well. Suppose a row $w$ of a weight matrix feeds a LayerNorm, and write the loss as a function of that row, $L(w)$. Scale invariance says $L(cw) = L(w)$ for any $c > 0$ — the norm cancels the scaling of $w$. Differentiate that identity with respect to $c$ at $c = 1$ and you get $\langle \nabla L(w), w \rangle = 0$: **the gradient is always orthogonal to the weight vector.** A gradient step therefore only rotates $w$; to first order it cannot change $\lVert w \rVert$ at all. Decompose a step of size $\eta$ into its angular part — the only part that changes the function — and it scales like $\eta / \lVert w \rVert^2$. As $\lVert w \rVert$ drifts upward over training, the angular (effective) learning rate quietly decays with no scheduler involved. Weight decay is then the only force that can shrink $\lVert w \rVert$, so the steady-state norm, and hence the steady-state effective learning rate, is set by the *balance* of weight decay against the gradient's tendency to grow the norm. That equilibrium is why tuning weight decay in a normalized transformer feels like tuning a learning rate: mechanically, it is one.

### The internal-covariate-shift story is mostly wrong

The original BatchNorm paper justified normalization by appeal to "internal covariate shift" — the idea that each layer's input distribution keeps shifting as the layers below it update, and normalization stabilizes it. It is an intuitive story and it is largely incorrect. Santurkar et al.'s 2018 paper [*How Does Batch Normalization Help Optimization?*](https://arxiv.org/abs/1805.11604) showed you can *inject* covariate shift after the norm (by adding time-varying noise) and training is just as fast, and that the real effect is that normalization makes the loss landscape measurably smoother — smaller Lipschitz constants on both the loss and its gradients, which lets you take larger, more reliable steps. The same reasoning carries over to LayerNorm. So when you reach for a normalization layer, the honest justification is not "I am stabilizing the distribution"; it is "I am buying a better-conditioned loss surface and a self-regulating learning rate."

### Second-order gotcha: never decay $\gamma$ and $\beta$

Because $\gamma$ multiplies the whole normalized vector, it controls the *scale* of the signal leaving the layer. If your optimizer applies weight decay to $\gamma$, it pulls $\gamma$ toward zero, which shrinks the signal, which the network fights by growing other weights, and the equilibrium is a model with a degraded effective capacity — or, in the bad cases, representation collapse. The fix is universal and you should treat it as non-negotiable: **exclude all 1-D parameters (norm weights, norm biases, and any other biases) from weight decay.**

```python
def param_groups(model, weight_decay=0.1):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 1-D tensors are norms and biases. Never decay them.
        if p.ndim < 2 or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

opt = torch.optim.AdamW(param_groups(model), lr=3e-4, betas=(0.9, 0.95))
```

Every serious training codebase (nanoGPT, Megatron, the Llama recipes) does exactly this split. If yours does not, that is the first thing to fix.

## 4. Where you put it: Pre-LN vs Post-LN

Now the placement question — the one that looks like a detail and is actually the difference between a model that trains and a model that does not.

![Post-LN vs Pre-LN: one box moves, stability flips](/imgs/blogs/layer-normalization-inside-out-4.webp)

The original [Transformer](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) used **Post-LN**: the normalization comes *after* the residual add.

$$
x_{l+1} = \mathrm{LN}\big(x_l + \mathrm{Sublayer}(x_l)\big).
$$

The modern default, since roughly GPT-2 and crystallized by Xiong et al.'s 2020 analysis [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745), is **Pre-LN**: the normalization comes *first*, inside the residual branch, and the residual add is left clean.

$$
x_{l+1} = x_l + \mathrm{Sublayer}\big(\mathrm{LN}(x_l)\big).
$$

The figure puts the two side by side. In Post-LN the norm sits on the main path — every residual add is immediately re-normalized, so the layer cannot pass its input through untouched. In Pre-LN the norm sits on the *branch* and the residual is an unbroken identity connection from input to output.

That structural difference has a precise consequence for gradients at initialization. Xiong et al. showed that in Post-LN the expected gradient norm at the top layers scales like $\mathcal{O}(\sqrt{\ln N})$ relative to the bottom in a way that grows with depth $N$, so for deep stacks the gradients are badly imbalanced at step zero. The standard patch was **learning-rate warmup** — start the LR near zero and ramp it over the first few thousand steps — which is a band-aid over the fact that the early gradients are untrustworthy. In Pre-LN, the identity path keeps gradient magnitudes roughly constant across depth, and you can train deep models with no warmup and a higher peak learning rate.

### The residual stream is a highway

![Pre-LN keeps an identity path open](/imgs/blogs/layer-normalization-inside-out-5.webp)

The clean way to picture Pre-LN is as a highway with on-ramps. The residual stream runs straight from the embeddings to the final layer as an unbroken identity path; each block taps off it, runs `LayerNorm -> Attention/MLP` on a copy, and merges the result back in with an add. Crucially, the identity skip in the figure *bypasses normalization entirely* — nothing on the main path is ever rescaled — so a gradient flowing backward has a multiplicative-one route from the loss all the way to the embeddings. That is the structural reason Pre-LN trains at depth.

Pre-LN is not free. Because every block *adds* to the residual stream and nothing rescales it, the stream's magnitude grows with depth — by the top layers the activations can be large, and you need a **final LayerNorm** before the output projection to bring them back to a sane scale (which is why every Pre-LN model has that one trailing `ln_f`). There is also a quality wrinkle: when Post-LN *can* be trained (shallow enough, or with warmup), it sometimes reaches slightly better final loss, because the per-layer renormalization acts as a mild regularizer on the representations. For the depths and scales that matter today, the trade is not close — stability wins — but it is worth knowing the trade exists.

Here are both blocks side by side, so the one-box move is concrete:

```python
class PostLNBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mlp   = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x, need_weights=False)[0])  # norm AFTER add
        x = self.norm2(x + self.mlp(x))
        return x


class PreLNBlock(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mlp   = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]             # norm INSIDE branch
        x = x + self.mlp(self.norm2(x))
        return x
```

### DeepNorm: keeping Post-LN's quality at 1000 layers

The story does not end at "use Pre-LN." Microsoft's [DeepNet](https://arxiv.org/abs/2203.00555) (Wang et al., 2022) showed you can keep Post-LN's representational edge *and* train 1000-layer transformers by doing two things: scale the residual before the norm by a depth-dependent constant $\alpha$, and downscale the initialization of certain weights by $\beta$.

$$
x_{l+1} = \mathrm{LN}\big(\alpha\, x_l + \mathrm{Sublayer}(x_l)\big), \qquad \alpha = (2N)^{1/4}\ \text{(encoder)} .
$$

The intuition is that bounding the per-layer update magnitude keeps the whole stack inside a region where Post-LN's gradients stay tame. DeepNorm and its cousins (Fixup, T-Fixup, ReZero, ScaleNorm) are all variations on the same insight: depth stability is fundamentally about controlling how much each residual branch is allowed to perturb the stream, and the norm's placement is one knob among several. For the overwhelming majority of models you will build, plain Pre-LN with a final norm is the right default — reach for DeepNorm only when you are genuinely going past a few dozen layers and Pre-LN's quality gap starts to bite.

### Warmup, and why Pre-LN lets you skip it

Learning-rate warmup — ramping the LR from near zero over the first few thousand steps — is so standard that people apply it reflexively, but it is worth knowing exactly *what it patches*. At initialization, an Adam update is roughly the sign of the gradient scaled by the learning rate, and in Post-LN the top-layer gradients are large and badly scaled relative to the bottom. A full-size step on those untrustworthy early gradients overshoots; once a few activations blow up, the variance estimates inside the norms degrade, and the run is lost. Warmup keeps the early steps tiny while the gradient statistics settle into something the optimizer can trust. Pre-LN changes the gradient geometry so the magnitudes are roughly balanced across depth from step zero, which is why Pre-LN models tolerate a higher peak learning rate and frequently train with little or no warmup at all. The practical reading: if you find yourself needing an ever-longer warmup to keep a deep model alive, that is a signal your placement or your residual scaling is wrong, not that the warmup should be longer still. Warmup is symptom management; Pre-LN treats the underlying condition.

## 5. RMSNorm: dropping the mean

For years the field assumed the mean-subtraction in LayerNorm was essential — it is half of "zero-mean, unit-variance," after all. Then [RMSNorm](https://arxiv.org/abs/1910.07467) (Zhang and Sennrich, 2019) asked the obvious question nobody had pushed on: what if you only re-scale, and never re-center?

![RMSNorm is LayerNorm with two steps removed](/imgs/blogs/layer-normalization-inside-out-6.webp)

RMSNorm throws away the mean computation and the $\beta$ shift, keeping only a divide by the root-mean-square of the vector:

$$
\bar{x}_i = \frac{x_i}{\mathrm{RMS}(x)}\,\gamma_i, \qquad \mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}.
$$

That is the whole layer. No $\mu$, no subtraction, no $\beta$. The figure marks the two removed steps in green: mean-centering and the bias shift are simply gone. The hypothesis — which has held up across hundreds of models since — is that the **re-scaling invariance**, not the re-centering invariance, is what does the optimization work. Removing the mean costs a reduction and a subtraction across $d$ elements and removes one $d$-vector of parameters, which is why RMSNorm is measurably faster (the original paper reported 7–64% wall-clock reduction on the normalization op depending on the model) and why it has become the default in essentially every modern LLM: Llama, Llama-2, Llama-3, Mistral, Gemma, Qwen, PaLM, T5 (a variant), and most others.

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))   # gamma only — there is no beta
        self.eps = eps

    def forward(self, x):
        # compute the statistic in fp32, cast back to the input dtype: the safe pattern
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight
```

It is worth seeing the difference numerically. For a vector with a nonzero mean, RMSNorm and LayerNorm produce genuinely different outputs — RMSNorm leaves the offset in. The bet the field has made is that the network can absorb that offset just fine, and that the saved compute is worth more than the lost re-centering. At trillion-token training budgets, "the norm op is 10% faster" is a lot of GPU-hours.

### Does dropping the mean ever hurt?

The honest answer is "rarely, and not where it counts for language models." Mean-subtraction removes a single shared offset across the feature vector; RMSNorm leaves that offset in and lets the next layer absorb it. Empirically, across the large pretraining runs that made the switch — Llama and its descendants are the clearest public examples — removing re-centering cost nothing measurable in final quality while reliably speeding up the norm. There are corners where re-centering seems to help a little, some encoder-style and vision-adjacent setups among them, but for decoder-only language models the field has effectively concluded that mean-centering is optional. The theory lines up with section 3: the optimization benefit of normalization comes from *scale* invariance — the $1/\lVert W \rVert^2$ effective-learning-rate effect — and RMSNorm preserves scale invariance exactly, giving up only the shift invariance. You keep the half that does the work and drop the half that mostly does not, and you save a reduction and a parameter vector for it.

### Second-order gotcha: the Gemma `(1 + weight)` convention

Here is the porting bug that has bitten more people than any other in this section. [Gemma](https://arxiv.org/abs/2403.08295)'s RMSNorm stores its scale parameter centered at zero and applies `(1 + weight)`:

```python
class GemmaRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d))   # initialized at ZERO, not one
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        out = x.float() * rms
        return (out * (1.0 + self.weight.float())).type_as(x)   # the (1 + w) is load-bearing
```

The stored weights are therefore *offsets from one*, not the gain itself. Load a Gemma checkpoint into a standard RMSNorm that multiplies by `weight` directly, and you multiply by something near zero instead of near one — the model emits fluent-looking garbage or pure noise. Every framework that supports Gemma has a special-cased RMSNorm for exactly this reason. The general lesson, again: a normalization layer is a contract, and the contract includes the variance estimator, the $\epsilon$ placement, the dtype of the reduction, and whether the scale is `weight` or `1 + weight`. Read the source, do not assume.

## 6. The modern norm zoo

If you opened a 2017 transformer block you would find exactly two LayerNorms. Open a 2025 frontier-model block and normalization has colonized five different sites, each fixing a specific failure mode that showed up as models scaled.

![Where norms sit in a 2025 LLM block](/imgs/blogs/layer-normalization-inside-out-7.webp)

The figure walks a modern block. Reading down the spine: a pre-attention RMSNorm, attention (with its own QK-norm hanging off the side), an optional post-sublayer "sandwich" norm, a pre-MLP RMSNorm, the MLP, and a final pre-logits norm closing the stack. Five distinct norm roles, each earning its place.

**Pre-attention and pre-MLP norms** are the standard Pre-LN sites from the previous section — one before each sublayer.

**QK-norm** is the newest and most interesting addition. As models and learning rates scaled, a specific instability showed up: the attention logits — the dot products $q \cdot k$ before the softmax — would grow without bound, the softmax would saturate into a near one-hot distribution, gradients through it would vanish, and the loss would spike. The fix, popularized by Google's [scaling ViT to 22B parameters](https://arxiv.org/abs/2302.05442) (Dehghani et al., 2023) and now standard in Gemma-2, OLMo, and others, is to apply a normalization (LayerNorm or RMSNorm) to the queries and keys *before* the dot product:

```python
class AttentionWithQKNorm(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim)        # per-head normalization of Q
        self.k_norm = RMSNorm(self.head_dim)        # and K — this bounds the logits

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.q_norm(q.view(B, T, self.n_heads, self.head_dim))
        k = self.k_norm(k.view(B, T, self.n_heads, self.head_dim))
        # ... transpose, scaled-dot-product attention, project out
        return self.out(...)
```

By normalizing $q$ and $k$ to a fixed scale, the dot product is bounded regardless of how large the projection weights grow, and the logit-explosion instability disappears. It is a remarkably cheap insurance policy against a class of mid-training blowups that used to require babysitting.

**Sandwich norms** (a norm both before *and* after each sublayer) come from NormFormer and were adopted by Gemma-2. The extra post-sublayer norm rescales the residual branch's contribution before it merges back, which further tames the residual-stream growth that Pre-LN suffers at depth. It costs two extra norms per block — cheap relative to attention and the MLP — and buys a smoother training curve for very deep or very high-learning-rate runs.

**The final norm** is the trailing `ln_f` we already met: one normalization between the last block and the output projection, mandatory in any Pre-LN model to undo the residual-stream growth before the logits are computed.

### Embedding scale, the implicit sixth site

There is a sixth place scale gets managed that is not a norm but interacts directly with the first one. Several models — Gemma is the clearest example — multiply the token embeddings by $\sqrt{d_{\text{model}}}$ before they enter the residual stream, so the embeddings arrive on a scale comparable to what the rest of the network expects. This matters because the very first pre-attention norm sees those embeddings, and an embedding table initialized at unit scale would otherwise be re-scaled hard by that first norm, wasting part of the first block's capacity on fixing a scale mismatch instead of doing useful work. The embedding multiplier and the norms are two halves of one scale-management story: the multiplier sets the entry scale, the norms maintain it through the stack, and the final norm resets it before the logits. When you port a model and the early-layer behavior looks off, check the embedding scaling alongside the norm conventions — it is the easiest of the six sites to overlook precisely because it does not look like a normalization at all.

### The frontier: removing normalization entirely

The natural question is whether you need any of this. The 2025 line of work on [normalization-free transformers](/blog/paper-reading/large-language-model/stronger-normalization-free-transformers) says maybe not. Dynamic Tanh (DyT) replaces the whole norm with a learnable element-wise squashing function,

$$
\mathrm{DyT}(x) = \gamma \odot \tanh(\alpha x) + \beta,
$$

where $\alpha$ is a single learned scalar — no mean, no variance, no reduction at all. It matches LayerNorm's quality on a surprising range of tasks while removing the cross-feature reduction that makes normalization a synchronization point in distributed kernels. Whether DyT and its relatives displace RMSNorm at scale is still open, but the very fact that a parameter-light pointwise function can stand in for the statistics is the strongest possible evidence that what normalization buys you is geometry, not a tidy distribution. Where these norms physically sit in real architectures is covered in more detail in the [modern LLM architectures](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek) walkthrough, and the residual-stream view connects to the broader treatment of [attention and residuals](/blog/paper-reading/large-language-model/attention-residuals).

## 7. Numerics and kernels in production

Everything above assumed exact arithmetic. Real training runs in bf16 or fp16, and that is where the innocent mean-and-divide turns dangerous.

![Mixed precision needs an fp32 reduction island](/imgs/blogs/layer-normalization-inside-out-9.webp)

The failure mode and the fix are both in the figure. Naively, you would load the activations in bf16, sum them in bf16 to get the mean, square and sum again for the variance, and divide — all in bf16. Two things go wrong. First, summing $d = 4096$ bf16 values accumulates rounding error fast, because bf16 has only 8 bits of mantissa; the mean drifts and the variance, computed as a difference of large nearly-equal sums, suffers catastrophic cancellation. Second, in fp16 specifically, squaring a moderately large activation can overflow fp16's $\pm 65504$ range, and the sum of squares overflows even sooner — you get an `inf`, then a `NaN`, then a dead run a few hundred steps in. The fix is the **fp32 island**: upcast to fp32 *for the reduction only*, compute mean and variance there, normalize, and cast the result back to bf16. PyTorch's `nn.LayerNorm` does this internally under autocast, which is one more reason to use the built-in rather than hand-rolling.

```python
def layer_norm_mixed(x, gamma, beta, eps=1e-5):
    # The production-safe pattern, explicit about precision boundaries.
    in_dtype = x.dtype                  # bf16 / fp16
    x32 = x.float()                     # upcast: fp32 island begins
    mu  = x32.mean(-1, keepdim=True)
    var = x32.var(-1, unbiased=False, keepdim=True)
    x_hat = (x32 - mu) * torch.rsqrt(var + eps)
    y = x_hat * gamma.float() + beta.float()
    return y.to(in_dtype)               # downcast: island ends
```

There is a second numerical trap even in fp32: computing the variance as $E[x^2] - E[x]^2$ (the "one-pass" formula) is itself unstable when the mean is large relative to the spread — the two terms are big and nearly equal, and you lose precision in the subtraction. The two-pass formula in the code above ($E[(x-\mu)^2]$) is stable; production fused kernels use [Welford's online algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance), which gets numerically-stable variance in a single pass over the data. If you ever write a custom LayerNorm CUDA or Triton kernel, Welford is the algorithm you want, and apex's `FusedLayerNorm` and the reference Triton tutorial kernel both use it with fp32 accumulation regardless of the input dtype.

### Fused kernels and why they matter

A naive LayerNorm in eager PyTorch launches a handful of separate CUDA kernels (mean, subtract, variance, rsqrt, scale, shift) and reads the activation tensor from HBM several times. Since LayerNorm is memory-bandwidth bound — it does very little arithmetic per byte — those repeated round-trips dominate. A *fused* kernel does the whole thing in one launch, reading the input once and writing the output once. The speedup is large enough to matter at scale:

```python
import torch, torch.utils.benchmark as benchmark

x  = torch.randn(8, 2048, 4096, device="cuda", dtype=torch.bfloat16)
ln = torch.nn.LayerNorm(4096).cuda().to(torch.bfloat16)

t = benchmark.Timer("ln(x)", globals={"ln": ln, "x": x}).blocked_autorange()
print(f"{t.median * 1e6:.1f} us per call")     # fused path via cuDNN / ATen
```

Swap that `LayerNorm` for `RMSNorm` and you save the mean reduction and the bias; swap it for apex `FusedLayerNorm` and you collapse the kernel launches. On a forward+backward training step with hundreds of norm calls, these are not rounding-error savings — they are single-digit percentages of total step time, which at frontier scale is measured in GPU-years.

### Recomputation, checkpointing, and determinism

Gradient (activation) checkpointing trades memory for compute by discarding intermediate activations in the forward pass and recomputing them during the backward. For LayerNorm this means the mean and variance get computed twice — once in the forward, once in the recompute — and if the two computations do not match bit-for-bit (different reduction order, a different fused kernel, a different precision under autocast), you introduce a small nondeterminism into the gradients. It is usually harmless, but it is exactly the kind of thing that makes a run fail to reproduce and sends someone hunting for a phantom bug. The defenses are to keep the norm's reduction in fp32 — so the precision is identical on both passes — and to use the same kernel on the forward and the recompute. The broader point is that a normalization layer is a reduction, and reductions are where reduction order and numeric precision quietly leak into your results; checkpointing just makes that leak happen twice per step.

### Second-order gotcha: bias removal is nearly free

A theme runs through the modern architectures: **the bias terms barely matter.** PaLM removed all biases (including in the norms) and reported improved training stability with no quality loss; Llama uses RMSNorm, which has no $\beta$ at all. The reasoning is that the bias adds parameters and a memory write for a degree of freedom the network rarely needs, and removing it tightens the memory footprint and can improve large-batch stability. If you are designing an architecture from scratch today, starting from "RMSNorm, no biases anywhere" and adding complexity only where a measured problem demands it is the right default — which is, satisfyingly, the opposite of where the field started in 2017.

## The normalization family at a glance

Step back from transformers and the whole family lines up along a few axes. The matrix below is the one to keep on a sticky note.

![The normalization family at a glance](/imgs/blogs/layer-normalization-inside-out-8.webp)

The figure makes BatchNorm's oddness visual: it is the only row that is batch-dependent and stateful (those are the red cells), which is exactly why it is wrong for sequence models and awkward for small batches. LayerNorm and RMSNorm are per-sample and stateless. RMSNorm is the leanest of all — no mean-centering, no bias — which is the all-green row. GroupNorm and InstanceNorm sit in between, normalizing over channel groups or single channels respectively, and they exist mostly to give convolutional and generative-vision models a batch-independent option where BatchNorm fails.

| Norm | Reduces over | Domain where it wins |
| --- | --- | --- |
| BatchNorm | batch + spatial | CNN image classification, large stable batches |
| LayerNorm | feature dimension | transformers, RNNs, anything with variable-length or tiny batches |
| RMSNorm | feature dimension | modern LLMs — LayerNorm's speed-tuned successor |
| GroupNorm | channel groups + spatial | detection/segmentation with small per-GPU batch |
| InstanceNorm | per channel, spatial | style transfer, image generation |

The decision procedure is almost mechanical. Sequence model or small/variable batch? LayerNorm, or RMSNorm if you want the speed. Convolutional vision backbone with big batches? BatchNorm still earns its keep. Vision model that cannot afford a big batch (detection, segmentation)? GroupNorm. Building a new LLM? RMSNorm with Pre-LN placement, a final norm, and QK-norm if you intend to push the learning rate. Everything else is a tuning detail on top of those defaults.

## LayerNorm beyond transformers

Transformers made LayerNorm famous, but the per-sample, batch-independent property pays off anywhere the batch is awkward. It is worth a quick tour, because each domain stresses a different one of the properties above.

**Recurrent networks** were the original motivation. A BatchNorm in an RNN would need separate running statistics per timestep — the distribution at step 1 is nothing like the distribution at step 50 — which is unworkable for variable-length sequences. LayerNorm normalizes each hidden state against its own features and so applies identically at every timestep, with no per-step state. The original paper's headline experiments were on recurrent models for exactly this reason.

**Reinforcement learning** leans on LayerNorm harder than almost anywhere else, and for a subtle reason: the data distribution is non-stationary (the policy changes what it sees as it learns) and the effective batch is often tiny or even a single trajectory. BatchNorm's running statistics are meaningless when the distribution is a moving target, and its batch dependence is poison when the batch is one rollout. LayerNorm's statelessness and batch-independence make it the safe default in actor and critic networks, and several recent RL stabilization results hinge on adding LayerNorm to the critic to control value-function explosions.

**Graph neural networks** face variable-size inputs by nature — different graphs have different node counts — so batch statistics are ill-defined, and LayerNorm (or its graph-aware variants) is common. **Tabular and small-data settings** sometimes prefer it too, because a small batch makes BatchNorm's statistics noisy. The through-line is the same fact we keep returning to: the moment your batch is small, variable, or non-i.i.d., a batch-dependent normalization is the wrong tool, and a per-sample normalization is the right one. Transformers are simply the most visible member of a much larger class of models for which that is true.

## Case studies from production

The points above are abstract until they cost you a week. Here are eleven incidents — some I have debugged personally, others that are well-documented in the field — where a specific misunderstanding of one of these mechanisms had a specific, expensive consequence.

### 1. The deep Post-LN divergence

A team ports a working 12-layer encoder up to 48 layers, keeps the same Post-LN architecture and the same learning-rate schedule, and the loss explodes to NaN around step 300 — every time, regardless of seed. The first hypothesis is a data bug; the second is a bad weight init. Both are wrong. The actual cause is the Post-LN gradient imbalance: at 48 layers the gradients at the top of the stack are large enough at initialization that the first few high-learning-rate steps overshoot catastrophically. The original Transformer hid this with a long learning-rate warmup, which the team had shortened because "warmup is a relic." The fix is one of: reinstate a several-thousand-step warmup, switch to Pre-LN (which removes the imbalance and the need for warmup entirely), or adopt DeepNorm's residual scaling if Post-LN's quality is needed. The lesson: warmup was never a relic, it was a patch for a specific Post-LN pathology, and the moment you change depth or placement you have to revisit it.

### 2. The fp16 LayerNorm variance underflow

A mixed-precision run trains cleanly for a few hundred steps, then produces NaNs that propagate through the whole model within one step. Gradient clipping does not help; lowering the learning rate only delays it. The culprit is a hand-rolled LayerNorm computing the variance in fp16. As activations grow during training, squaring them for the variance overflows fp16's 65504 ceiling, the sum of squares becomes `inf`, the reciprocal-sqrt becomes zero or `NaN`, and the poison spreads. The fix is the fp32 island: upcast before the reduction. The deeper lesson is that "it trained fine for 300 steps" is not evidence of numerical safety — the failure is triggered by the activation magnitude crossing a threshold, which happens partway through training, not at the start. Anyone hand-rolling a norm in mixed precision should reduce in fp32, full stop.

### 3. BatchNorm in a sequence model

An engineer coming from computer vision builds a transformer-ish sequence classifier and reflexively reaches for BatchNorm, since that is what worked on images. Training metrics look great. Then the model is deployed behind an API that serves one request at a time, and accuracy collapses. The reason is BatchNorm's train/eval split colliding with batch-size-one inference: at batch one, the running statistics (accumulated over training batches whose composition does not match production traffic) are all the model has, and they are a poor fit. Worse, the padding tokens in training had been polluting the per-feature statistics all along. The fix is to replace BatchNorm with LayerNorm, which is batch-independent, stateless, and identical in train and eval. This is the single most common normalization mistake people make when crossing from vision into NLP, and it is exactly the failure that motivated LayerNorm in the first place.

### 4. The epsilon-placement porting bug

A model is ported from a research framework into a production runtime, the weights load without error, and the outputs are *almost* right — close enough that smoke tests pass, wrong enough that a downstream ranking metric drops two points and nobody can find why. Days later someone diffs the LayerNorm implementations and finds the source framework used $1/(\sqrt{\sigma^2} + \epsilon)$ (epsilon outside the root) while the runtime used $1/\sqrt{\sigma^2 + \epsilon}$ (inside). For most tokens the difference is negligible; for the low-variance tokens it is enough to shift the argmax of a close decision. The fix is trivial once found; finding it is the expensive part. The general defense is a numerical parity test on a fixed batch that asserts max-absolute-difference below a tight threshold *before* you trust a port — and that test is also how you would have caught the next one.

### 5. The Gemma `(1 + weight)` garbage outputs

Someone integrates Gemma into an inference stack that already has a generic RMSNorm, reuses it, and the model produces confident nonsense — grammatical text with no coherence, or pure token soup. No error, no NaN, just garbage. The cause is Gemma's `(1 + weight)` convention: its stored norm weights are offsets from one, centered at zero. The generic RMSNorm multiplied by `weight` directly, so every normalization scaled the signal by something near zero. The fix is one character of arithmetic — `(1.0 + weight)` — but you only find it by reading Gemma's reference implementation and noticing the init is `zeros`, not `ones`. The lesson is the one from section 5: a norm is a contract, and "RMSNorm" alone does not specify the contract. The scale convention is part of it.

### 6. QK-norm tames the attention-logit blowup

A large pretraining run is pushed to a higher learning rate to save wall-clock, and the loss develops periodic spikes — it trains fine, then jumps, then recovers, then jumps higher, and eventually one spike does not recover. Logging the attention logits reveals the cause: in a handful of heads the pre-softmax dot products grow into the hundreds, the softmax saturates to one-hot, and the gradient through those heads vanishes and then destabilizes. The fix is QK-norm — normalize the queries and keys per head before the dot product, which bounds the logits no matter how large the projection weights become. After the change the spikes vanish and the higher learning rate sticks. This is now standard practice for high-learning-rate or very large runs precisely because the failure is otherwise so hard to diagnose from the loss curve alone.

### 7. Gamma and beta caught in weight decay

A custom training loop applies AdamW with weight decay to `model.parameters()` — all of them, uniformly — and the model trains, but underperforms a reference implementation by a frustrating, stubborn margin that no amount of LR tuning closes. The cause is weight decay pulling every $\gamma$ toward zero. Since $\gamma$ sets the scale of each norm's output, decaying it shrinks the signal throughout the network, and the model spends capacity fighting its own regularizer. The fix is the parameter-group split from section 3: 1-D parameters (norm weights, norm biases, all biases) get zero weight decay. The gap closes immediately. Almost every reference codebase does this split; almost every from-scratch training loop forgets it the first time. If your loss is mysteriously a notch worse than a known-good baseline, check this before anything else.

### 8. The small-batch detection backbone

A detection model is trained with a per-GPU batch of two (large images, limited memory), using a BatchNorm backbone inherited from the classification pretraining. Convergence is unstable and the final mAP is well below expectations. The cause is BatchNorm computing statistics over a batch of two, which is far too small to estimate a feature's mean and variance — the statistics are dominated by noise, and they differ every step. The standard fixes in detection are either to *freeze* the BatchNorm statistics (use the pretrained running stats and stop updating them) or to replace BatchNorm with GroupNorm, which normalizes over channel groups within each sample and is therefore batch-independent. GroupNorm was introduced for exactly this regime. The lesson generalizes: whenever your batch is small or its composition is non-i.i.d., a batch-dependent norm is the wrong tool, and the family matrix tells you which batch-independent norm fits the domain.

### 9. The residual off-by-one

An engineer reimplements a Pre-LN block from a paper and, in a moment of inattention, normalizes the wrong tensor: instead of `x + sublayer(norm(x))` they write `x + norm(sublayer(x))`, or they apply the norm to the residual sum as well as the branch. The model trains — that is the cruel part — but it tracks the reference implementation's loss curve a hair worse and slowly diverges in quality over a long run. There is no crash to point at, just a quiet under-performance. The fix is to diff against a trusted reference block and assert the forward passes match on a fixed input. The broader lesson is that normalization placement is part of the architecture's *definition*, not a free-floating utility you can slot in anywhere; in Pre-LN the norm goes on the branch input and the residual stays clean, and one transposed line breaks that invariant without breaking the run.

### 10. Bias removal as a measured win

A team profiling a 7B model for inference notices that the norm biases and the various `Linear` biases collectively cost memory bandwidth on every token, for parameters that ablations suggest contribute almost nothing. They run the experiment that PaLM and Llama had already run at larger scale: remove the biases (and move to RMSNorm, which has no $\beta$). The result is a small but real reduction in memory and a measurable throughput improvement at long context, with no measurable quality loss — and, as a bonus, slightly better large-batch training stability, which matches the published reports. The lesson is the inverse of the usual one: not every parameter earns its keep, and the bias terms in particular are a place where "remove it and measure" usually comes back in favor of removal. Start lean and add complexity only when a metric demands it.

### 11. The accidental running statistics

A well-meaning engineer "improves" a LayerNorm by giving it BatchNorm-style momentum buffers — `running_mean` and `running_var` updated during training and used at eval — reasoning that smoothing the statistics over time must help. Training looks normal. Evaluation is subtly worse, and the gap between `.train()` and `.eval()` mode, which should be *exactly zero* for LayerNorm, is nonzero. The cause is the added state: LayerNorm is supposed to be stateless and identical in both modes, and the running buffers reintroduced the very train/eval discrepancy that LayerNorm exists to avoid. The fix is to delete the buffers and compute statistics from the current input every time, in both modes. The lesson is conceptual: LayerNorm's statelessness is a *feature*, not an omission. The momentum machinery is BatchNorm's workaround for not being able to compute a meaningful statistic at inference — LayerNorm has no such problem and needs no such workaround.

### 12. The forgotten final norm

A from-scratch Pre-LN implementation omits the final `ln_f` between the last block and the output projection — an easy thing to miss, since every *block* already has its norms and the trailing one looks redundant. Training proceeds, but the loss plateaus high and the logits have a strange, ever-growing magnitude. The cause is the Pre-LN residual-stream growth from section 4: with nothing rescaling the stream, its magnitude accumulates across every block, and by the final layer the activations feeding the unnormalized output projection are large and poorly scaled, producing over-confident, hard-to-calibrate logits and a sluggish loss. Adding the single final norm fixes it immediately. The lesson is that Pre-LN's clean residual highway has a cost — unbounded stream growth — and the final norm is the mandatory toll at the exit. It is not optional decoration; it is the counterpart to the on-path renormalization that Post-LN got for free.

### 13. The `normalized_shape` that leaked across positions

An engineer adapting a vision transformer reuses `nn.LayerNorm` but passes `normalized_shape=[seq_len, d]` instead of `d`, reasoning that "normalizing the whole token block" sounds more thorough. The model trains and even performs acceptably on the training distribution, then generalizes poorly and behaves erratically when the sequence length changes at inference. The cause is that the two-axis `normalized_shape` makes every position share one mean and variance, so a token's normalization now depends on every *other* token in the sequence — information leaks across positions, and the statistics shift when the length does. The intended per-token behavior needs `normalized_shape=d`, normalizing the feature axis alone. The lesson restates section 1's rule: `normalized_shape` must cover exactly the axes meant to share a statistic. One extra axis silently turns a per-token operation into a cross-token one, and the failure only surfaces when the sequence length you test on differs from the one you trained on.

### 14. The low-precision norm at inference

A quantized model is deployed with weights in int8 and activations in fp16 for throughput, and a subtle quality regression appears that was absent in the fp32 reference — not a crash, just a few points of degradation on hard inputs. Profiling shows the regression concentrates in tokens with large activation magnitudes, and the trail leads to the LayerNorm: the inference kernel computed the variance in fp16 to save a cast, and for the high-magnitude tokens the squared sum lost precision exactly where it mattered. The fix mirrors the training-side fp32 island — reduce the statistics in fp32 even when the surrounding compute is fp16 or lower — and the regression disappears. The lesson is that the numerical care LayerNorm demands does not stop at training: any time the reduction runs in a low-precision format, the variance is the term that breaks first, and an fp32 reduction is cheap insurance whether you are training or serving.

## An implementation checklist

When you add, port, or debug a normalization layer, this is the list to run down. Every item maps to a failure above.

- **Reduce over the right axis.** `normalized_shape=d` for per-token LayerNorm. One extra axis leaks information across positions (case 13).
- **Match the variance estimator.** PyTorch uses the biased variance (divide by $d$). `unbiased=True` introduces a $\sqrt{d/(d-1)}$ discrepancy that passes unit tests and degrades ported models (case 4).
- **Match the $\epsilon$ placement and value.** Inside the square root, not outside; the same numeric value as the source. The two forms diverge on low-variance inputs (section 1).
- **Reduce in fp32 under mixed precision.** Upcast before computing mean and variance, downcast the result. Reducing in bf16/fp16 produces NaNs at scale (cases 2 and 14).
- **Exclude $\gamma$, $\beta$, and all biases from weight decay.** 1-D parameters go in a zero-decay group; decaying $\gamma$ collapses the signal scale (case 7).
- **Use Pre-LN with a final norm** for anything past a handful of layers, and do not forget the trailing `ln_f` (cases 1 and 12).
- **Honor the scale convention when porting.** "RMSNorm" alone does not tell you whether the gain is `weight` or `1 + weight`; read the reference init (case 5).
- **Keep it stateless.** LayerNorm has no running buffers and must be identical in train and eval; a nonzero train/eval gap means someone added state (case 11).
- **Add QK-norm** if you intend to push the learning rate or the scale, before the logit blowup shows up as loss spikes (case 6).
- **Write a numerical parity test.** A fixed input, a tight max-absolute-difference threshold against the reference, run before you trust any port. It catches cases 4, 5, and 13 in one shot.

## When to reach for LayerNorm — and when not to

Reach for **LayerNorm (or RMSNorm)** when:

- You are building a transformer or any sequence model — this is the default, full stop.
- Your batch is small, variable in size, or non-i.i.d. (RL rollouts, online learning, batch-size-one inference).
- You need train and eval to behave identically, with no stateful buffers to manage.
- You are working with variable-length inputs where padding would pollute batch statistics.
- You want a self-regulating effective learning rate from the scale-invariance property.

Prefer **RMSNorm over LayerNorm** when you are designing a new LLM and want the ~10% norm-op speedup and one fewer parameter vector — which today means essentially always for new language models.

Use **Pre-LN placement with a final norm** for any model past a handful of layers, and add **QK-norm** if you intend to train at a high learning rate or very large scale. Reach for **DeepNorm or other residual-scaling schemes** only when you are genuinely going past a few dozen layers and Pre-LN's small quality gap starts to matter.

**Skip LayerNorm / reach for something else** when:

- You are training a convolutional image classifier with large, stable, i.i.d. batches — BatchNorm still tends to win there, and its batch dependence is not a liability when the batch is big and well-mixed.
- You have a small per-GPU batch in vision (detection, segmentation) — use GroupNorm, which is batch-independent but keeps the spatial structure BatchNorm exploits.
- You are doing style transfer or per-image generation — InstanceNorm's per-channel-per-sample statistics are the right inductive bias.
- You are chasing the absolute minimum of kernel synchronization at extreme scale and are willing to experiment — the normalization-free direction (DyT and relatives) is worth watching, though not yet a safe default.

The thread tying all of this together is the one we started with: normalization is not a tidiness filter on activations. It is a reparameterization of the optimization problem. The forward pass — mean, variance, scale, shift — is the easy half to see and the unimportant half to understand. The backward pass, the dense Jacobian, the scale invariance, and the placement relative to the residual stream are where the leverage is. Get those right and the `self.norm` one-liner does exactly what it is supposed to: disappear into the background and let the model train. Get them wrong and it will, very quietly, cost you a week.

## Interview questions worth being able to answer

These come up constantly in staff-level ML interviews, and each maps directly to a section above.

**Why do transformers use LayerNorm instead of BatchNorm?** Because the statistics are computed per token over the feature axis, LayerNorm is independent of batch size and composition, identical in train and eval, robust to variable-length and padded sequences, and needs no cross-device synchronization — all properties BatchNorm lacks, and all of which matter for sequence models. (Section 2.)

**Does LayerNorm behave differently in train and eval mode?** No. It is stateless — it computes statistics from the current input every time. If you observe a train/eval difference in a LayerNorm, someone has added running buffers it should not have. (Case study 11.)

**Why does normalization help optimization — is it really about internal covariate shift?** Largely not. The dominant effect is a smoother, better-conditioned loss landscape, plus scale invariance that yields a self-regulating effective learning rate. The covariate-shift story was shown to be mostly incidental. (Section 3.)

**What is the difference between Pre-LN and Post-LN, and which is the default?** Post-LN normalizes after the residual add and amplifies gradients with depth, needing learning-rate warmup; Pre-LN normalizes inside the branch, keeps the residual path as a clean identity, and trains deep models stably without warmup. Pre-LN is the modern default, at the cost of residual-stream growth that a final norm corrects. (Section 4.)

**What does RMSNorm remove, and why is it safe?** It removes mean-centering and the bias shift, keeping only RMS rescaling. It is safe because the optimization benefit comes from scale invariance, which RMSNorm preserves exactly; only the shift invariance is given up, and language models do not appear to need it. (Section 5.)

**Why must $\gamma$ and $\beta$ be excluded from weight decay?** Because $\gamma$ controls the output scale of the layer; decaying it toward zero shrinks the signal and degrades or collapses the representation. The universal fix is to put all 1-D parameters in a zero-weight-decay group. (Section 3.)

**Why does mixed-precision LayerNorm need special care?** Reducing the mean and especially the variance in bf16 or fp16 loses precision and can overflow, producing NaNs partway through training. Correct kernels upcast to fp32 for the reduction, then downcast. (Section 7.)

**What is QK-norm and what does it fix?** Normalizing queries and keys before the attention dot product bounds the pre-softmax logits, preventing the softmax-saturation instability that surfaces as loss spikes at high learning rates or large scale. (Section 6.)

**Why is LayerNorm's backward pass not just an element-wise scaling?** Because $\mu$ and $\sigma^2$ depend on every feature, the Jacobian is dense: the gradient reaching each input is centered and de-correlated with the normalized activation, projecting out the two directions the normalization made irrelevant. (Section 3.)

## Further reading

- Ba, Kiros, Hinton (2016), [*Layer Normalization*](https://arxiv.org/abs/1607.06450) — the original, motivated by recurrent networks.
- Ioffe, Szegedy (2015), [*Batch Normalization*](https://arxiv.org/abs/1502.03167) — the layer LayerNorm reacts against, and the "internal covariate shift" framing.
- Santurkar, Tsipras, Ilyas, Madry (2018), [*How Does Batch Normalization Help Optimization?*](https://arxiv.org/abs/1805.11604) — the landscape-smoothing explanation that displaces the covariate-shift story.
- Xiong et al. (2020), [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745) — the Pre-LN vs Post-LN gradient analysis.
- Zhang, Sennrich (2019), [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467) — RMSNorm.
- Wang et al. (2022), [*DeepNet: Scaling Transformers to 1,000 Layers*](https://arxiv.org/abs/2203.00555) — DeepNorm residual scaling.
- Dehghani et al. (2023), [*Scaling Vision Transformers to 22 Billion Parameters*](https://arxiv.org/abs/2302.05442) — QK-norm against attention-logit instability.
- On this blog: [modern LLM architectures](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek), [attention and residuals](/blog/paper-reading/large-language-model/attention-residuals), and the [normalization-free transformers](/blog/paper-reading/large-language-model/stronger-normalization-free-transformers) frontier.
