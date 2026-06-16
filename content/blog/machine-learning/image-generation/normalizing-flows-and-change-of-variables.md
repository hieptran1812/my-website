---
title: "Normalizing Flows and the Change of Variables: The Road to Flow Matching"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Derive the change-of-variables formula, build a RealNVP coupling flow in PyTorch, and see exactly why modern text-to-image models moved from diffusion SDEs to flow-matching ODEs."
tags:
  [
    "image-generation",
    "diffusion-models",
    "normalizing-flows",
    "flow-matching",
    "change-of-variables",
    "generative-ai",
    "deep-learning",
    "realnvp",
    "continuous-normalizing-flows",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/normalizing-flows-and-change-of-variables-1.png"
---

Here is a deceptively simple question that the whole field of generative modeling has been circling for a decade: if I can sample a vector $z$ from a plain Gaussian — `torch.randn(3, 256, 256)`, one line, milliseconds — and I have a function $f$ that maps that vector to a picture of a cat, *what is the probability of the cat I get out*? Not "is it a good cat." The actual number $p(x)$, the density the model assigns to that exact 196,608-pixel array. If you can answer that question exactly, you have an **exact-likelihood model**, and you can do things diffusion models cannot do cleanly: compute the log-likelihood of any image, detect out-of-distribution inputs by their density, and train by directly maximizing the probability of your data with no variational slack, no lower bound, no surrogate loss.

Normalizing flows are the family that answers that question exactly. They are built on one piece of calculus — the **change-of-variables formula** — and one hard constraint: the function $f$ must be **invertible**, and the determinant of its Jacobian must be cheap to compute. That single constraint is the whole story of this post. It is what makes flows mathematically beautiful, what made them lag GANs and diffusion on image quality for years, and — through a continuous-time generalization called the **neural ODE** — what set up the single most important architectural shift in text-to-image since latent diffusion: the move from diffusion SDEs to **flow matching**, the training objective behind Stable Diffusion 3 and FLUX.

By the end of this post you will be able to: derive the change-of-variables formula and the log-determinant-of-the-Jacobian term from scratch; explain *why* a triangular Jacobian makes a coupling layer's likelihood cheap; implement a working RealNVP-style affine-coupling flow in PyTorch — coupling layer, exact forward, exact inverse, and the negative-log-likelihood loss; reason about the MAF/IAF forward-versus-inverse speed asymmetry; and — the payoff — state precisely why continuous normalizing flows were beautiful but impractical, and how flow matching sidesteps their fatal cost to become the engine of the 2025–2026 frontier.

![A graph showing a base Gaussian distribution flowing through a stack of invertible maps into a complex data distribution and back, with exact density carried through each step](/imgs/blogs/normalizing-flows-and-change-of-variables-1.png)

This is the sixth post in the foundations track of our image-generation series. It sits alongside [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard), which laid out the four-family map (VAE, GAN, autoregressive, diffusion/flow) and the **generative trilemma** — sample quality versus mode coverage versus sampling speed. Flows occupy a fascinating corner of that trilemma: they nail mode coverage and give you exact likelihood, but historically paid for it in sample quality. Understanding *why* they paid that price, and how the continuous-time version escapes it, is the bridge to the modern frontier. Keep the trilemma in your head as we go; every design choice in this post is a move on that triangle.

A word on why this matters even if you never train a flow. The change-of-variables formula is not a niche tool — it is the *grammar* of generative modeling. The ELBO that trains a VAE, the score identity that trains a diffusion model, and the velocity regression that trains SD3 are all, at bottom, statements about how a density transforms when you push a random variable through a map. Flows are the family where that grammar is exposed in its purest, most literal form: an exact, invertible push-forward with explicit volume bookkeeping. Learn it here, where there is no variational slack to hide behind, and the rest of the series — the diffusion bound, the reverse SDE, flow matching — reads as variations on a theme you already understand. That payoff is the real reason this "old" family earns a foundations slot in a series about the 2026 frontier.

## 1. The change-of-variables formula, derived

Let me start with the one equation everything rests on, and let me actually derive it rather than assert it, because the derivation is where the intuition lives.

Suppose I have a random variable $z$ with a density I know — say a standard Gaussian, $z \sim p_Z(z) = \mathcal{N}(0, I)$. I push it through an invertible, differentiable function $x = f(z)$. The inverse exists: $z = f^{-1}(x)$. I want the density of $x$, call it $p_X(x)$. The question is how the probability mass redistributes when I bend space with $f$.

### The one-dimensional case first

Start in 1D, where you can see it. Probability mass is conserved: the probability that $z$ lands in a tiny interval $[z, z + dz]$ must equal the probability that $x$ lands in the image of that interval, $[x, x + dx]$. There is no mass created or destroyed by relabeling the axis; $f$ just stretches and squeezes it. So:

$$
p_X(x)\,|dx| = p_Z(z)\,|dz|
$$

The absolute values are there because a density is non-negative — direction does not matter, only how much the interval was stretched. Divide through:

$$
p_X(x) = p_Z(z)\left|\frac{dz}{dx}\right| = p_Z\big(f^{-1}(x)\big)\left|\frac{d f^{-1}(x)}{dx}\right|
$$

Read that carefully. If $f$ **stretches** a region (so a small $dz$ becomes a large $dx$), then $|dz/dx| < 1$ and the density goes *down* — the same mass is spread over more space, so it is less concentrated. If $f$ **compresses** a region, the density goes *up*. The term $|dz/dx|$ is exactly the local stretch factor, and it is the thing that keeps the total probability integrating to one. That factor is the seed of everything that follows.

### The multivariate case: enter the Jacobian

Now go to $D$ dimensions, $z, x \in \mathbb{R}^D$. The local stretch factor is no longer a single derivative — it is how a tiny *volume* element transforms, and volume in many dimensions is governed by the **Jacobian determinant**. The Jacobian of $f$ at a point is the matrix of partial derivatives,

$$
J_f(z) = \frac{\partial f(z)}{\partial z} = \begin{bmatrix} \frac{\partial f_1}{\partial z_1} & \cdots & \frac{\partial f_1}{\partial z_D} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_D}{\partial z_1} & \cdots & \frac{\partial f_D}{\partial z_D} \end{bmatrix}
$$

and the absolute value of its determinant, $|\det J_f(z)|$, is the factor by which $f$ scales an infinitesimal volume around $z$. (This is exactly the Jacobian you remember from multivariable calculus when changing variables in an integral — $dx = |\det J_f|\,dz$.) Conserving probability mass over volumes instead of intervals gives the **change-of-variables formula**:

$$
\boxed{\,p_X(x) = p_Z\big(f^{-1}(x)\big)\,\left|\det J_{f^{-1}}(x)\right|\,}
$$

Using the identity $\det J_{f^{-1}}(x) = \big(\det J_f(z)\big)^{-1}$ (the determinant of an inverse map is the reciprocal), and taking logs because we always work with log-densities in practice (products become sums, and we avoid numerical underflow on tiny densities), we get the form you will actually code:

$$
\boxed{\,\log p_X(x) = \log p_Z(z) - \log\left|\det J_f(z)\right|, \qquad z = f^{-1}(x)\,}
$$

This is the single most important equation in the post, so let me unpack every symbol:

- $\log p_Z(z)$ is the log-density of the **base distribution** evaluated at the latent $z$ you recover by inverting $f$. The base is something trivial — a standard Gaussian — for which the log-density is a closed form: $\log \mathcal{N}(z; 0, I) = -\frac{1}{2}\|z\|^2 - \frac{D}{2}\log(2\pi)$.
- $\log|\det J_f(z)|$ is the **log-determinant of the Jacobian** — the "log-det" term, often abbreviated *LDJ*. It is the running tally of how much $f$ stretched space on the way from $z$ to $x$. This is the term that costs money.

The word "normalizing" in *normalizing flow* refers to exactly this: the flow transforms the data into a **normal** (Gaussian) base distribution, and the log-det term is the bookkeeping that keeps the transformed density properly **normalized** to integrate to one.

### Why this gives you exact likelihood — and what it demands

Here is the punchline. If I can (a) **invert** $f$ to get $z = f^{-1}(x)$, and (b) **compute the log-determinant of the Jacobian** of $f$, then I can write down the exact log-likelihood of *any* data point $x$, with no approximation. Then training is dead simple: maximize $\log p_X(x)$ over your dataset — pure maximum likelihood, no ELBO gap like a [VAE](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch), no adversarial minimax like a GAN, no variational bound like diffusion. You are directly maximizing the probability your model assigns to real images. This is what we mean when we call flows the **exact-likelihood family**; the post on [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions) frames why an exactly-tractable likelihood is such a prize and how it relates to KL divergence and the ELBO.

But look at what the two requirements demand of $f$:

1. **$f$ must be invertible** — a bijection between $z$-space and $x$-space. This already rules out most neural networks, which routinely throw away information (a ReLU zeroes out half its inputs; a pooling layer collapses four pixels into one). An invertible network cannot lose any information, anywhere. It also forces the latent space to have the **same dimensionality** as the data — there is no bottleneck, no compression. A flow over $256\times256\times3$ images has a $196{,}608$-dimensional latent. That non-negotiable dimensionality match is one of the deepest costs flows pay.
2. **$|\det J_f|$ must be cheap** — and this is the killer. A general $D \times D$ determinant costs $O(D^3)$. For $D = 196{,}608$ that is around $7.5 \times 10^{15}$ operations *per training example per layer*. Utterly hopeless. The entire architectural history of normalizing flows is a search for clever invertible maps whose Jacobian determinant you can compute in $O(D)$ instead of $O(D^3)$.

Hold those two constraints — **invertibility** and a **tractable Jacobian** — in your mind. Everything from RealNVP to Glow to the neural ODE is an answer to "how do I build an expressive $f$ that satisfies both?" And flow matching, the destination of this post, is what you get when you finally stop fighting the second constraint and route around it entirely.

#### Worked example: a single squeeze, by hand

Let me make the change of variables concrete with numbers you can check, because the intuition is slippery until you see the bookkeeping balance. Take a 1D base $z \sim \mathcal{N}(0, 1)$ and the invertible map $x = f(z) = 2z + 1$. The inverse is $z = (x - 1)/2$ and the derivative is $df/dz = 2$, constant everywhere. The change-of-variables formula says $p_X(x) = p_Z(z)\,|df/dz|^{-1} = \frac{1}{2}\,\mathcal{N}\!\big((x-1)/2; 0, 1\big)$. Sanity-check the density at the mode: $z = 0$ maps to $x = 1$, and $p_Z(0) = \frac{1}{\sqrt{2\pi}} \approx 0.399$, so $p_X(1) = 0.399 / 2 \approx 0.199$. The factor of $\frac{1}{2}$ is the map *stretching* the axis by 2, so the density at the peak is *halved* — the same total mass, spread over twice the width. In log form, $\log p_X(1) = \log 0.399 - \log 2 = -0.919 - 0.693 = -1.612$, which is exactly $\log p_Z(0) - \log|df/dz|$: the base log-density minus the log-det. The log-det term subtracted off $0.693$ nats — the cost, in nats, of the stretch. Now picture that exact accounting happening in 196,608 dimensions, with the single derivative replaced by a Jacobian determinant. The arithmetic is identical; only the bookkeeping of the stretch factor gets harder. That difficulty is the entire engineering problem, and the rest of this post is about making the stretch-factor accounting cheap.

### A note on discrete pixels: dequantization

One subtlety that bites everyone who trains a flow on real images, and it is worth a paragraph because it explains a number in our results table. Pixel values are **discrete** integers in $\{0, 1, \dots, 255\}$, but the change-of-variables formula assumes a *continuous* density. Train a flow naively on integer pixels and it cheats: it piles arbitrarily high density spikes on the 256 allowed values and reports a meaningless (infinitely good) likelihood. The fix is **dequantization** — add uniform noise $u \sim U[0, 1)$ to each pixel before training, $x \leftarrow x + u$, which turns the discrete distribution into a continuous one whose density is a proper lower bound on the discrete model's likelihood. This is why flow likelihoods are always reported in **bits per dimension** (a per-pixel, dequantization-aware unit) and why a fair comparison must use the same dequantization scheme. Variational dequantization (a small learned flow for $u$, from the Flow++ paper) tightens the bound further and was worth a few hundredths of a bit. It is a small detail with an outsized effect on the headline number.

## 2. Composing simple flows into expressive ones

A single invertible map with a cheap Jacobian cannot be very expressive — there is a tension between "simple enough to invert and differentiate cheaply" and "flexible enough to turn a Gaussian into the distribution of natural images." Flows resolve this the way deep networks resolve everything: **composition**. Stack many simple invertible transforms and the composite can be arbitrarily expressive while *each piece* stays cheap.

![A stack of invertible transform layers turning a Gaussian into data, where the total log-determinant is the sum of each layer's contribution](/imgs/blogs/normalizing-flows-and-change-of-variables-2.png)

The mathematics of composition is exactly what makes this work, and it is beautiful. Let $f = f_K \circ \cdots \circ f_2 \circ f_1$ be a composition of $K$ invertible maps, with intermediate states $h_0 = z$, $h_k = f_k(h_{k-1})$, $x = h_K$. Two facts:

- **Invertibility composes.** The inverse of a composition is the composition of inverses in reverse order: $f^{-1} = f_1^{-1} \circ f_2^{-1} \circ \cdots \circ f_K^{-1}$. If each layer can be inverted, the whole stack can.
- **Log-determinants add.** By the chain rule, $J_f = J_{f_K} \cdots J_{f_1}$, and since $\det(AB) = \det(A)\det(B)$, the log-determinant of the whole is the *sum* of the per-layer log-determinants:

$$
\log\left|\det J_f(z)\right| = \sum_{k=1}^{K} \log\left|\det J_{f_k}(h_{k-1})\right|
$$

So the full log-likelihood of a deep flow is just the base log-density plus a sum of per-layer log-dets:

$$
\log p_X(x) = \log p_Z(z) - \sum_{k=1}^{K} \log\left|\det J_{f_k}(h_{k-1})\right|, \qquad z = f^{-1}(x)
$$

This is the architectural template for *every* normalizing flow: design one cheap, invertible, expressive-enough layer; stack a lot of them; sum the log-dets. The whole game reduces to designing that one layer. The rest of this post is a tour of the clever answers — coupling layers, 1×1 convolutions, autoregressive transforms — and then the continuous limit where $K \to \infty$ and the sum becomes an integral.

There is a useful "the way this works" intuition here that the figure above makes concrete: each layer bends the density a little, the way a single fold bends a sheet of paper. No single fold turns a flat sheet into a sculpture, but enough folds, each individually simple and reversible, can. The log-det sum is the total amount of bending, and maximizing likelihood means choosing the folds so the data ends up looking Gaussian in latent space.

### The two passes a flow must support

Every flow layer has to be usable in two directions, and it is worth being crystal clear about which direction is which because the terminology trips everyone up:

- **The normalizing / inference direction**, $z = f^{-1}(x)$: data → latent. You run this during **training** (and density evaluation), because to compute $\log p_X(x)$ you need to recover $z$ from a data point $x$ and accumulate log-dets along the way.
- **The generative / sampling direction**, $x = f(z)$: latent → data. You run this to **generate** — sample $z \sim \mathcal{N}(0, I)$, push it forward through $f$, get an image.

A flow is only practical if *both* directions are tractable. As we will see with autoregressive flows (MAF vs IAF), some designs make one direction fast and the other slow, and which one you make fast decides whether your flow is good for density estimation or for sampling — a direct collision with the generative trilemma's speed axis.

There is a second, quieter reason both passes must be cheap, and it is about *training dynamics*. Because the same network is used in both directions, the loss landscape couples them: a parameter update that sharpens the density (improves the normalizing pass) also reshapes the generative pass. In a [VAE](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) the encoder and decoder are separate networks that can be tuned somewhat independently; in a flow there is one weight-tied bijection, so there is nowhere to hide a modeling error. This is part of what makes flows *honest* — the density you report is the density the sampler actually realizes — and also part of what makes them *hard*, because a single network must simultaneously be a good encoder and a good decoder with shared weights. Diffusion models sidestep this entirely: they never invert a network, only iterate one, which is one of the structural advantages we will return to in Section 7.

## 3. Coupling layers: the RealNVP trick

Now the central engineering idea, the one that made flows work on images at all: the **affine coupling layer**, introduced in NICE (Dinh et al., 2014) and made affine and convolutional in RealNVP (Dinh et al., 2017). It is a genuinely clever piece of design, and once you see it you cannot unsee it.

The problem we are solving: we want an invertible $f$ whose Jacobian determinant is cheap. The insight: if the Jacobian is **triangular**, its determinant is just the product of its diagonal — $O(D)$ instead of $O(D^3)$. So how do we build an expressive invertible map with a triangular Jacobian? The coupling layer's answer is to split the input and transform half of it as a function of the other half.

![A graph of an affine coupling layer splitting the input, computing scale and shift from the untouched half, and transforming the other half](/imgs/blogs/normalizing-flows-and-change-of-variables-3.png)

### The construction

Split the $D$-dimensional input $h$ into two parts, $h = (h_a, h_b)$ — say the first half and the second half of the channels. The affine coupling layer does this:

$$
\begin{aligned}
x_a &= h_a \\
x_b &= h_b \odot \exp\!\big(s(h_a)\big) + t(h_a)
\end{aligned}
$$

In words: **leave the first half untouched** (the identity), and **affine-transform the second half** — scale it elementwise by $\exp(s(h_a))$ and shift it by $t(h_a)$, where $s$ and $t$ are *arbitrary* neural networks (call them the scale and translation nets) that look only at the untouched half $h_a$. The $\odot$ is elementwise multiplication. We use $\exp(s)$ rather than $s$ directly to guarantee the scale is strictly positive, which keeps the layer invertible and makes the log-det clean.

Now watch the magic. **Inverting this layer is trivial and exact** — no matrix inversion, no iterative solver, just algebra:

$$
\begin{aligned}
h_a &= x_a \\
h_b &= \big(x_b - t(x_a)\big) \odot \exp\!\big(-s(x_a)\big)
\end{aligned}
$$

Because $x_a = h_a$ is passed through unchanged, when we invert we still have $h_a$ available, so we can recompute the *exact same* $s(h_a)$ and $t(h_a)$ that were used in the forward pass — and undo the affine transform on $h_b$. Crucially, **$s$ and $t$ never need to be inverted.** They can be arbitrarily deep, nonlinear, non-invertible neural networks — convolutional stacks, ResNets, whatever — because we only ever *evaluate* them, never invert them. That is the whole trick: hide all the expressive, non-invertible computation inside $s$ and $t$, which sit on the easy side of the split.

### Why the Jacobian is triangular — and the log-det is free

Here is the part that makes coupling layers cheap. Compute the Jacobian of the forward map, ordering the output as $(x_a, x_b)$ and the input as $(h_a, h_b)$:

$$
J = \frac{\partial(x_a, x_b)}{\partial(h_a, h_b)} = \begin{bmatrix} \dfrac{\partial x_a}{\partial h_a} & \dfrac{\partial x_a}{\partial h_b} \\[2ex] \dfrac{\partial x_b}{\partial h_a} & \dfrac{\partial x_b}{\partial h_b} \end{bmatrix} = \begin{bmatrix} I & 0 \\[1ex] \dfrac{\partial x_b}{\partial h_a} & \operatorname{diag}\!\big(\exp(s(h_a))\big) \end{bmatrix}
$$

Look at the four blocks:

- $\partial x_a / \partial h_a = I$ — the identity, because $x_a = h_a$.
- $\partial x_a / \partial h_b = 0$ — because $x_a$ does not depend on $h_b$ at all. **This zero is the whole point.** It is what makes the matrix **lower-triangular**.
- $\partial x_b / \partial h_a$ — some complicated block full of derivatives of $s$ and $t$. We do not care what is in it.
- $\partial x_b / \partial h_b = \operatorname{diag}(\exp(s(h_a)))$ — diagonal, because $x_b = h_b \odot \exp(s(h_a)) + t(h_a)$ transforms each element of $h_b$ independently, scaled by the corresponding element of $\exp(s)$.

The determinant of a triangular matrix is the **product of its diagonal entries**, and the off-diagonal block (however messy) does not enter the determinant at all. The diagonal is $1$ from the top block and $\exp(s(h_a))$ from the bottom block. So:

$$
\log\left|\det J\right| = \sum_{j} s(h_a)_j
$$

The log-determinant is just the **sum of the scale network's outputs**. It costs nothing beyond evaluating $s$, which we had to evaluate anyway. No determinant computation. No $O(D^3)$. This is the entire reason coupling-based flows are practical — the triangular structure converts an intractable determinant into a free sum.

### Coupling layers leave half the input fixed — so we alternate

One coupling layer never transforms $h_a$. If you stack two identical ones, $h_a$ stays frozen forever — half your dimensions never move. The fix is to **alternate the split**: in the next layer, swap which half is the identity and which is transformed (or, in RealNVP for images, alternate a checkerboard mask and a channel-wise mask). After a few alternating layers every dimension has been transformed as a function of every other, and the composite is genuinely expressive. This is exactly the "stack many simple layers" principle from Section 2, now made concrete.

## 4. A working RealNVP flow in PyTorch

Enough theory. Here is a small but complete and runnable affine-coupling flow — the coupling layer with exact forward and inverse, a stack with alternating masks, and the exact negative-log-likelihood loss. This trains on a toy 2D distribution (the classic "two moons"), which is the right scale to see the mechanics without a GPU. The same code scales to images by making $s$ and $t$ convolutional and adding squeeze/multi-scale operations, but the heart is identical.

```python
import torch
import torch.nn as nn

class AffineCoupling(nn.Module):
    """One RealNVP affine coupling layer with an arbitrary binary mask.

    Masked dims pass through unchanged; unmasked dims are scaled and
    shifted by nets that see ONLY the masked dims. Jacobian is triangular,
    so log|det J| is just the sum of the scale-net outputs.
    """
    def __init__(self, dim, hidden=128, mask=None):
        super().__init__()
        self.register_buffer("mask", mask)            # 1 = pass through, 0 = transform
        # s and t share a trunk; s is tanh-bounded for stability.
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim * 2),               # outputs [s_raw, t]
        )
        self.scale = nn.Parameter(torch.zeros(dim))   # learned global scale on s

    def _s_t(self, x_masked):
        s_raw, t = self.net(x_masked).chunk(2, dim=-1)
        s = torch.tanh(s_raw) * self.scale            # bounded, stabilizes training
        return s, t

    def forward(self, x):
        """Generative direction is f; here forward = NORMALIZING (x -> z)."""
        x_m = x * self.mask                           # the untouched half
        s, t = self._s_t(x_m)
        s = s * (1 - self.mask); t = t * (1 - self.mask)
        z = x_m + (1 - self.mask) * (x - t) * torch.exp(-s)
        log_det = -(s.sum(dim=-1))                    # inverse direction: negate
        return z, log_det

    def inverse(self, z):
        """Generative direction (z -> x): sampling."""
        z_m = z * self.mask
        s, t = self._s_t(z_m)
        s = s * (1 - self.mask); t = t * (1 - self.mask)
        x = z_m + (1 - self.mask) * (z * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return x, log_det
```

Notice the two things the theory promised. The `inverse` is closed-form algebra — no matrix inversion. And `log_det` is a single sum over the scale outputs, exactly $\sum_j s_j$ as we derived. The `tanh`-times-learned-scale on $s$ is a standard stability trick (RealNVP and Glow both bound the scale this way) so $\exp(s)$ never explodes early in training.

Now the full model — a stack with alternating masks and the Gaussian base:

```python
import math

class RealNVP(nn.Module):
    def __init__(self, dim=2, n_layers=6, hidden=128):
        super().__init__()
        masks = []
        for i in range(n_layers):
            m = torch.arange(dim) % 2                  # alternating parity mask
            masks.append((m if i % 2 == 0 else 1 - m).float())
        self.layers = nn.ModuleList(
            AffineCoupling(dim, hidden, masks[i]) for i in range(n_layers)
        )
        self.dim = dim

    def log_prob(self, x):
        """Exact log-likelihood via change of variables: NORMALIZING pass."""
        z, log_det = x, torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:                      # data -> latent
            z, ld = layer.forward(z)
            log_det = log_det + ld
        # Standard-Gaussian base log-density, closed form.
        log_pz = -0.5 * (z ** 2).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi)
        return log_pz + log_det                        # exact log p_X(x)

    @torch.no_grad()
    def sample(self, n):
        """GENERATIVE pass: z ~ N(0, I) -> x."""
        z = torch.randn(n, self.dim)
        for layer in reversed(self.layers):            # latent -> data
            z, _ = layer.inverse(z)
        return z
```

And the training loop. The loss is just the negative mean log-likelihood — **maximum likelihood, no surrogate**:

```python
from sklearn.datasets import make_moons

model = RealNVP(dim=2, n_layers=6, hidden=128)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(5000):
    x, _ = make_moons(n_samples=512, noise=0.05)
    x = torch.tensor(x, dtype=torch.float32)
    loss = -model.log_prob(x).mean()                   # negative log-likelihood
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0:
        # loss is in nats; lower = data is more probable under the model.
        print(f"step {step:5d}  nll = {loss.item():.3f}")

samples = model.sample(1000)                           # new points from the flow
```

That is a complete generative model in about 70 lines. Run it and the loss drops from roughly $4$ nats to under $1$, and `model.sample(1000)` draws fresh points that trace out the two crescents. Three things are worth dwelling on because they are exactly what flows give you that other families do not:

1. **The loss is the actual negative log-likelihood**, in nats. There is no lower bound and no adversarial game. `loss.item()` is a real, comparable number — you can report bits-per-dimension (divide nats by $D \cdot \ln 2$) and compare directly against any other likelihood model.
2. **`log_prob` is exact and works on arbitrary inputs.** Feed it a point off the data manifold and it tells you, exactly, how improbable that point is. That is the out-of-distribution detection use case flows are genuinely good at.
3. **Sampling and density use the same parameters in opposite directions.** No separate decoder, no separate discriminator. One invertible network, run forward to evaluate density, backward to sample.

### A problem-solving narrative: when the flow's loss explodes

Let me walk through the failure mode you *will* hit if you train this code at scale, because debugging it teaches the mechanics better than any clean derivation. You scale the toy flow up — convolutional coupling layers, a few squeeze operations, CIFAR-10 — kick off training, and within a few hundred steps the loss goes to `NaN`. What happened? Trace it through the math. The log-likelihood is $\log p_Z(z) - (\text{LDJ})$, and the LDJ is $\sum_j s_j$. Maximum likelihood *wants to make $\log p(x)$ large*, which it can do by making the LDJ term very *negative* — i.e. by driving the scale outputs $s_j$ toward large negative values, which means $\exp(s_j) \to 0$. The flow discovers it can win cheap likelihood by collapsing the scale to near-zero, squashing the data into a sliver of latent space. But then the *inverse* pass divides by $\exp(s_j) \approx 0$, and you get an overflow. The density says "great," the sampler says `inf`. This is the single most common flow training failure, and it is a direct consequence of the weight-tied, two-direction coupling we flagged in Section 2.

The fix is exactly the line in the code you might have glossed over: `s = torch.tanh(s_raw) * self.scale`. Bounding $s$ with a `tanh` caps how negative the log-scale can go, so $\exp(s)$ stays in a safe range and the inverse never divides by something tiny. RealNVP and Glow both do a version of this. The second fix is **gradient clipping** (`torch.nn.utils.clip_grad_norm_`), because the LDJ gradient can spike when a scale starts to run away. The third is good **initialization** — actnorm's data-dependent init (Section 5) exists precisely so the first forward pass produces unit-variance activations and the LDJ starts near zero rather than at some wild value. None of these are cosmetic; each one is a direct countermeasure to the likelihood objective's incentive to cheat by collapsing the Jacobian. Understanding *why* the loss explodes — the model trading sampler stability for cheap density — is what turns "add a tanh and pray" into a principled fix. This is the kind of debugging that separates someone who has read the RealNVP paper from someone who has actually trained one at 2am.

Stress-test the boundary: what if you remove the `tanh` bound *and* clip gradients hard? You will often limp through training, but the model quietly learns a pathological flow with huge dynamic range in its scales, and your samples come out with blown-out, oversaturated regions where some channel's inverse scale exploded. The clean lesson is that the two-direction constraint is not just a training nuisance — it is a *modeling* constraint, and respecting it (bounded scales, careful init, normalized inputs) is the difference between a flow that samples cleanly and one that produces artifacts. Diffusion models never face this particular failure because they never invert the network; that immunity is one of the underrated reasons diffusion was easier to scale than flows.

#### Worked example: reading the bits-per-dimension number

Suppose you train this flow (scaled up with convolutional coupling layers) on $32\times32\times3$ CIFAR-10, so $D = 3072$. After training, `model.log_prob(x).mean()` over the test set reads, say, $-7100$ nats. Convert to **bits per dimension**, the standard likelihood metric for images: $\text{bpd} = -\frac{\log p(x)}{D \cdot \ln 2} = \frac{7100}{3072 \times 0.693} \approx 3.34$ bpd. For reference, RealNVP reported about **3.49 bpd** on CIFAR-10 and Glow about **3.35 bpd**; a strong autoregressive PixelCNN++ reached about **2.92 bpd**. Lower is better — it means the model needs fewer bits to encode a test image, i.e. assigns it higher probability. The gap between Glow ($3.35$) and PixelCNN++ ($2.92$) is the quiet headline of this whole post: **flows win at being exact and fast to sample, but they lose at raw likelihood to autoregressive models, and lose at sample sharpness to GANs and diffusion.** We will see exactly why in Section 7.

## 5. Glow: making coupling flows work on real images

RealNVP showed coupling flows could model images. Glow (Kingma & Dhariwal, 2018) is what made them *good* — sharp $256\times256$ faces, smooth latent interpolations between people, the works. Glow keeps the affine coupling layer and adds two components that fix specific weaknesses. Both are, predictably, invertible maps with cheap Jacobians.

![A stack showing one Glow step of flow: actnorm, then 1x1 invertible convolution, then affine coupling, repeated](/imgs/blogs/normalizing-flows-and-change-of-variables-4.png)

### Actnorm: a data-dependent, invertible normalization

Batch normalization is everywhere in vision, but it is a nightmare in a flow: its statistics depend on the whole batch, which muddies the per-example log-det, and it behaves differently at train and test time. **Actnorm** (activation normalization) replaces it with a per-channel affine transform, $x = h \odot \gamma + \beta$, where $\gamma$ (scale) and $\beta$ (bias) are learned parameters, **initialized** from the first batch so that post-actnorm activations have zero mean and unit variance — but thereafter just ordinary parameters, independent of the batch. It is trivially invertible ($h = (x - \beta) / \gamma$) and its log-det is $\sum_j \log|\gamma_j|$ per spatial location — cheap, per-example, identical at train and test. It is the flow-friendly substitute for batchnorm.

### The 1×1 invertible convolution: a learned channel mixer

Recall the weakness from Section 3: coupling layers only transform half the channels per layer, and RealNVP alternated *fixed* masks to mix information. Fixed permutations are crude. Glow's insight is that a **$1\times1$ convolution is just a matrix multiply across channels at every spatial position** — and a matrix multiply is invertible (if the matrix is) with a computable determinant. So Glow learns a $C \times C$ weight matrix $W$ and applies $x_{ij} = W h_{ij}$ at every pixel $(i, j)$. This is a *learned, generalized permutation* — it lets the network decide how to shuffle and mix channels before the next coupling layer, instead of being stuck with a fixed checkerboard.

The Jacobian log-det of a $1\times1$ conv over an $H \times W$ feature map is clean: each of the $H \cdot W$ spatial positions applies the same $W$, so $\log|\det J| = H \cdot W \cdot \log|\det W|$. The only cost is a single $C \times C$ determinant. Glow keeps even that cheap with an **LU decomposition** of $W = PL(U + \operatorname{diag}(s))$, which makes $\log|\det W| = \sum_j \log|s_j|$ — an $O(C)$ sum instead of an $O(C^3)$ determinant. The same triangular-structure trick, applied one level up.

A full Glow is a **multi-scale architecture**: a "squeeze" operation trades spatial resolution for channels ($H\times W\times C \to \frac{H}{2}\times\frac{W}{2}\times 4C$), then $K$ steps of [actnorm → 1×1 conv → affine coupling], then a "split" that factors out half the channels straight to the latent (so later, higher-resolution flow steps work on fewer channels). It is the coupling-flow idea, industrialized. Glow's headline demo — interpolating between two faces in latent space and getting a smooth, photorealistic morph — is a direct consequence of the exact, invertible latent: every point on the line between two latents decodes to a valid face, because the map is a bijection over the whole space.

#### Worked example: the cost of one Glow step

Take an SDXL-ish latent block, $C = 128$ channels at $16\times16$. One $1\times1$ conv applies a $128\times128$ matrix at each of $256$ positions. The forward cost is $256 \times 128^2 \approx 4.2$M multiply-adds — trivial. The log-det is $256 \times \log|\det W|$, and with the LU trick that determinant is a $128$-element sum, not a $128^3 \approx 2.1$M-operation determinant. Stack $L = 32$ such steps across 3 scales and the whole flow's per-image log-det is a few thousand cheap sums. **Contrast the naive route:** if you tried to compute the full $D\times D$ Jacobian determinant of the entire flow directly, with $D = 128\times16\times16 = 32{,}768$, you would face an $O(D^3) \approx 3.5\times10^{13}$-operation determinant per image. The structured-Jacobian design buys roughly a **ten-billion-fold** speedup. That ratio is *why* flows are built the way they are.

## 6. Autoregressive flows and the forward/inverse speed asymmetry

There is a second great family of flows, built not on coupling but on **autoregressive** structure, and it exposes a trade-off that turns out to be the deepest lesson in this whole post — a lesson that reappears, transformed, in flow matching.

The idea: instead of splitting the input in half, transform each dimension as a function of *all previous* dimensions. An autoregressive flow defines

$$
x_i = h_i \odot \exp\!\big(s_i(x_{1:i-1})\big) + t_i(x_{1:i-1})
$$

where the scale and shift for dimension $i$ depend on dimensions $1$ through $i-1$. Because $x_i$ depends only on *earlier* dimensions, the Jacobian $\partial x / \partial h$ is **triangular** again — the same trick as coupling, but now with a fine-grained ordering instead of a coarse two-way split. The log-det is again a cheap sum of the diagonal, $\sum_i s_i$. Autoregressive flows are strictly more expressive than coupling layers (every dimension conditions on every earlier one), which is why they hit better likelihoods. There is no free lunch, of course; the cost shows up as a brutal asymmetry between the two directions.

![A before-and-after comparison of MAF fast-density slow-sampling versus IAF slow-density fast-sampling](/imgs/blogs/normalizing-flows-and-change-of-variables-5.png)

### MAF: fast density, slow sampling

**Masked Autoregressive Flow** (MAF, Papamakarios et al., 2017) parameterizes the *normalizing* direction $x \to z$ autoregressively. Computing $z$ from $x$ is **fully parallel**: every $z_i$ depends on $x_{1:i}$, and all of $x$ is given, so you compute all the $s_i, t_i$ in one masked forward pass (the same masking trick as the MADE autoencoder). So **density evaluation is one parallel pass — fast.** But sampling ($z \to x$) is **sequential**: to get $x_i$ you need $x_{1:i-1}$, which you do not have until you have generated them one at a time. Sampling a $D$-dim vector takes $D$ sequential passes. For a $196{,}608$-pixel image, that is $196{,}608$ sequential network evaluations per sample. MAF is therefore a **density estimator** — great for computing likelihoods, painfully slow to generate from.

### IAF: fast sampling, slow density

**Inverse Autoregressive Flow** (IAF, Kingma et al., 2016) flips it. IAF parameterizes the *generative* direction $z \to x$ autoregressively, conditioning each $x_i$ on $z_{1:i-1}$ — but since the $z$'s are all available at sampling time (you drew them all from the Gaussian at once), **sampling is one parallel pass — fast**. The price: computing the density of an *external* data point $x$ (which you need for maximum-likelihood training) is sequential, because to invert you must recover the $z$'s one at a time. IAF is therefore a **fast sampler** that is slow at scoring arbitrary data — which is why IAF was originally used not for direct ML training but as a flexible *posterior* in a VAE, where you only ever score your own samples.

### The asymmetry is the lesson — and Parallel WaveNet saw it first

Here is the table that makes the trade-off unmissable:

| Flow type | Density $p(x)$ for given $x$ | Sampling $z \to x$ | Natural use |
|---|---|---|---|
| **Coupling (RealNVP/Glow)** | Parallel, 1 pass | Parallel, 1 pass | Both — the practical default for images |
| **MAF** | Parallel, 1 pass (fast) | Sequential, $D$ passes (slow) | Density estimation |
| **IAF** | Sequential, $D$ passes (slow) | Parallel, 1 pass (fast) | Fast sampling / VAE posterior |

Coupling layers look like they win outright — both directions parallel — and that is exactly why they dominated image flows. But MAF and IAF expose something deeper: **you can train the slow-sampling model and distill it into the fast-sampling one.** That is precisely what Parallel WaveNet (van den Oord et al., 2018) did for audio — train an autoregressive teacher with fast density, then distill it into an IAF student with fast sampling, getting real-time audio generation. *Train where scoring is cheap; deploy where sampling is cheap.* Hold that thought. The entire modern speed story — consistency models, distillation, and the reason flow matching is fast — is a continuous-time echo of this exact maneuver.

#### Worked example: the cost of autoregressive sampling at image scale

Put numbers on the asymmetry so the "sequential, $D$ passes" entry in the table stops being abstract. Take a modest $64\times64\times3$ image, so $D = 12{,}288$. A coupling flow samples it in **one** forward pass — say 5 ms on a GPU. MAF would sample the same image in $D = 12{,}288$ **sequential** network calls; even at an optimistic 0.1 ms per call that is $12{,}288 \times 0.1\,\text{ms} \approx 1.23$ seconds per image, and it does not parallelize across pixels because pixel $i$ literally needs pixel $i-1$ first. Scale to $256\times256\times3$ ($D = 196{,}608$) and a single MAF sample is on the order of **20 seconds** of strictly sequential compute. That is the wall that made autoregressive *sampling* impractical for images, and it is the same wall that PixelCNN hit — and it is exactly why the field wanted either coupling flows (parallel both ways) or, later, a continuous formulation whose sampler takes a *fixed small* number of steps regardless of $D$. The asymmetry is not a footnote; it is a first-order design constraint that shaped which models won.

## 7. The fundamental trade-off: why flows lagged

We now have enough to state plainly why, despite their mathematical elegance and exact likelihood, normalizing flows never won the image-generation crown — why by 2021 the field had moved decisively to GANs and then diffusion. There are three structural reasons, and they all trace back to the invertibility-plus-tractable-Jacobian constraint.

![A matrix comparing flows, VAEs, GANs, and diffusion across exact likelihood, sample quality, invertibility, and one-pass sampling](/imgs/blogs/normalizing-flows-and-change-of-variables-6.png)

**Reason 1: the constraints cap expressiveness.** A coupling layer leaves half its inputs untouched; an autoregressive layer is triangular; both are deliberately *restricted* maps. You buy a cheap Jacobian by limiting what each layer can do, then try to recover expressiveness through depth. But each layer is weaker than an unconstrained network of the same size, so flows need to be *deeper and wider* to match the capacity of a GAN generator or a diffusion U-Net. The dimensionality constraint compounds this: because the latent must match the data dimension, there is no compression, so a flow spends parameters modeling pixel-level structure that a [latent diffusion model](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions) gets to ignore by working in a compressed VAE latent. More parameters, harder optimization, and still a ceiling on flexibility.

**Reason 2: exact likelihood is not the same objective as perceptual quality.** This is subtle and important. Maximum likelihood is *mode-covering* — it heavily penalizes assigning low probability to any real data point, so a flow tries to put mass *everywhere* the data could plausibly be. The side effect is that it also puts mass in the gaps *between* modes — in regions of pixel space that are not realistic images. GANs, by contrast, are *mode-seeking*: the adversarial loss rewards producing samples that fool the discriminator, so a GAN happily ignores parts of the distribution to make the samples it *does* produce razor-sharp. The upshot is a real and measured phenomenon: **likelihood and sample quality are not the same axis.** A model can have excellent likelihood and mediocre samples. Flows, optimizing pure likelihood, landed exactly there — strong bits-per-dimension, samples that looked slightly soft and "off" compared to a GAN's. This is the mode-covering versus mode-seeking face of the generative trilemma, and flows sit firmly on the coverage side.

**Reason 3: the empirical FID gap.** The numbers told the story. On CIFAR-10, Glow's samples scored an FID around the high 40s; a contemporary GAN (StyleGAN-class) was in the single digits to low teens; DDPM diffusion reached **FID ≈ 3.17**. (FID, Fréchet Inception Distance, measures the distance between the Inception-feature statistics of generated and real images; lower is better — see [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions) for what it does and does not capture.) An order-of-magnitude FID gap is not a tie-breaker; it is a verdict. Flows were the most *principled* family and among the *least competitive* on the metric people cared about for images.

Here is the comparison that situates flows among the four families:

| Property | Normalizing flow | VAE | GAN | Diffusion |
|---|---|---|---|---|
| **Exact likelihood?** | Yes (the whole point) | No (ELBO lower bound) | No (implicit, none) | No (variational bound) |
| **Sample quality (image FID)** | Weak | Weak–medium (blurry) | Strong (sharp) | State of the art |
| **Invertible / encode any image?** | Yes, exactly | Approx (encoder) | No (no encoder) | Approx (DDIM inversion) |
| **One-pass sampling?** | Yes (one forward) | Yes (one decode) | Yes (one forward) | No (many steps) |
| **Mode coverage** | High (mode-covering) | High | Low (mode collapse) | High |
| **Latent = data dim?** | Yes (no compression) | No (bottleneck) | No (low-dim z) | Latent or pixel |

Read across the "exact likelihood" and "sample quality" rows together and the tragedy of flows is right there: they are the *only* family with truly exact likelihood and one-pass sampling, and they were the *weakest* on the quality metric that drove the field. So flows became a specialist tool — out-of-distribution detection, density estimation, problems where you genuinely need $\log p(x)$ — rather than the engine of text-to-image. **But the story does not end there**, because the discrete-flow framework has a continuous-time generalization that quietly fixes the expressiveness problem, and *that* is the road to flow matching.

## 8. Continuous normalizing flows: the neural ODE

What if, instead of stacking $K$ discrete invertible layers, you took the limit $K \to \infty$ — infinitely many infinitesimal transforms? Each layer becomes a tiny step, and the discrete composition $h_{k} = h_{k-1} + \epsilon \cdot v(h_{k-1})$ turns into a **differential equation**. This is the **continuous normalizing flow** (CNF), introduced in the Neural ODE paper (Chen et al., 2018), and it is the conceptual hinge of this entire post.

![A before-and-after of a discrete-layer flow versus a continuous-time flow defined by a velocity field ODE](/imgs/blogs/normalizing-flows-and-change-of-variables-7.png)

Define a **velocity field** $v_\theta(x, t)$ — a neural network that, given a point $x$ and a time $t \in [0, 1]$, returns a velocity (a vector telling the point which way and how fast to move). A sample's trajectory from base to data is the solution of the ODE:

$$
\frac{dx(t)}{dt} = v_\theta\big(x(t), t\big), \qquad x(0) = z \sim \mathcal{N}(0, I)
$$

To **sample**, you draw $z$ from the Gaussian and **integrate the ODE forward** from $t=0$ to $t=1$ using a numerical solver (Euler, Runge–Kutta, Dormand–Prince) — the solver takes many small steps, repeatedly calling $v_\theta$, and the endpoint $x(1)$ is your generated image. To **invert** (data → base), you integrate *backward*. Invertibility is automatic and exact in the continuous limit: ODE flows are reversible by construction, because you can always run time backward.

### The instantaneous change of variables

Now the beautiful part — the continuous analogue of the log-det term. In the discrete case we summed $\log|\det J_k|$ over layers. In the continuous limit, that sum becomes an *integral*, and remarkably, the change-of-variables formula collapses from a determinant into a **trace**. The **instantaneous change of variables** theorem (Chen et al., 2018) states that the log-density evolves along a trajectory as:

$$
\frac{\partial \log p(x(t))}{\partial t} = -\operatorname{Tr}\!\left(\frac{\partial v_\theta}{\partial x}\right) = -\nabla \cdot v_\theta\big(x(t), t\big)
$$

The right-hand side is the negative **divergence** of the velocity field — the trace of its Jacobian, *not* the log-determinant. Integrate from $0$ to $1$ to get the total log-density change:

$$
\log p(x(1)) = \log p(x(0)) - \int_0^1 \nabla \cdot v_\theta\big(x(t), t\big)\, dt
$$

This is genuinely elegant. The continuous formulation **escapes the structural constraints of discrete flows entirely**: $v_\theta$ can be *any* neural network — no triangular Jacobian required, no coupling split, no invertibility constraint on the architecture, because reversibility comes from the ODE itself, not from the network's structure. The expressiveness problem of Section 7 (Reason 1) simply evaporates. A CNF can be as flexible as any net, and it still gives exact likelihood via the trace integral.

To build intuition for why the determinant collapses to a trace, think of it this way. Over a discrete layer, the map applies a finite linear transform whose volume change is $|\det J|$ — a product over all the eigenvalue magnitudes. Over an *infinitesimal* time step $\epsilon$, the map is $x \mapsto x + \epsilon\, v(x)$, so its Jacobian is $I + \epsilon\, \partial v/\partial x$. The determinant of $I + \epsilon A$ for small $\epsilon$ is, to first order, $1 + \epsilon\,\operatorname{Tr}(A)$ — this is Jacobi's formula, and it is *why* the trace appears: in the infinitesimal limit, the log of the product of eigenvalues becomes the sum of eigenvalues, which is the trace. So $\log|\det(I + \epsilon\,\partial v/\partial x)| \approx \epsilon\,\operatorname{Tr}(\partial v/\partial x)$, and summing these infinitesimal contributions over the trajectory turns the sum-of-log-dets into the integral-of-trace above. The mental model is clean: the **divergence** of the velocity field is the instantaneous rate at which volume (and hence density) is expanding or contracting at each point, exactly as in fluid dynamics, where the divergence of a flow tells you whether the fluid is locally compressing or expanding.

So CNFs are the dream: arbitrarily expressive *and* exact-likelihood *and* invertible. Why, then, did they not immediately take over image generation? Because of one word: **simulation.**

### The simulation cost that killed CNFs in practice

Training a CNF by maximum likelihood means, for *every gradient step on every training image*, you must:

1. **Integrate the ODE** from data to base to compute the trajectory — many sequential calls to $v_\theta$ (a stiff solver might need dozens to hundreds of function evaluations, "NFEs").
2. **Integrate the divergence** $\nabla \cdot v_\theta$ along that trajectory to get the log-det — and the exact trace costs $O(D)$ vector-Jacobian products (you estimate it with the Hutchinson stochastic trace estimator to bring it down, but it is still expensive).
3. **Backpropagate through the entire ODE solve** — either store every intermediate state (huge memory) or use the adjoint method (a second backward ODE solve, doubling the integration cost).

Every single training step is a full numerical simulation of an ODE, forward and backward. CNFs were **one to two orders of magnitude slower to train** than a comparable discrete model, and the cost scaled badly with image resolution. The math was gorgeous; the wall-clock was a wall. CNFs remained a beautiful idea that did not scale to high-resolution image generation — a Ferrari with the engine of a lawnmower. The whole field needed a way to get the expressiveness and exactness of the continuous flow *without paying the simulation cost at training time.* That way is flow matching.

## 9. Flow matching: regress the velocity, skip the simulation

Here is the punchline the entire post has been building toward, and the reason this post exists in an image-generation series at all. The breakthrough — **flow matching** (Lipman et al., 2023) and the closely related **rectified flow** (Liu et al., 2023) — is a way to train a CNF's velocity field $v_\theta$ **without ever simulating the ODE during training.** It is the maneuver that turned the elegant-but-impractical CNF into the engine of Stable Diffusion 3 and FLUX.

![A graph contrasting CNF training by ODE simulation against flow matching regressing a velocity target directly](/imgs/blogs/normalizing-flows-and-change-of-variables-8.png)

The key realization: we do not actually need to integrate anything to *train* the velocity field. We need $v_\theta(x, t)$ to point, at every point and time, in the direction the probability mass should flow to carry the base distribution to the data distribution. So what if we just **specify that direction analytically** for every training pair, and **regress** $v_\theta$ onto it with a plain mean-squared-error loss? No ODE solve. No log-det. No adjoint. Just supervised regression of a vector field.

The simplest and most influential choice is a **straight-line path** (this is rectified flow / conditional optimal-transport flow matching). For a data point $x_1$ and a noise sample $x_0 \sim \mathcal{N}(0, I)$, define the interpolation and its constant velocity:

$$
x_t = (1 - t)\,x_0 + t\,x_1, \qquad \frac{dx_t}{dt} = x_1 - x_0
$$

The target velocity along this straight path is simply $x_1 - x_0$ — a constant, the same at every $t$. The flow-matching loss is then the most ordinary thing imaginable:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t \sim U[0,1],\; x_0 \sim \mathcal{N}(0,I),\; x_1 \sim p_{\text{data}}}\; \Big\| v_\theta\big(x_t, t\big) - (x_1 - x_0) \Big\|^2
$$

That is it. Sample a time $t$, sample a noise vector and a real image, linearly interpolate to get $x_t$, and train the network to predict the velocity $x_1 - x_0$ that would carry $x_t$ along the straight line. **No simulation, no Jacobian, no determinant, no integral — at training time.** You only integrate at *sampling* time, when you draw $z$ and solve the ODE forward, and because the learned paths are nearly straight, the solver needs few steps.

```python
import torch

def flow_matching_loss(model, x1):
    """Rectified-flow / conditional-OT flow matching. One training step.
    model(x_t, t) predicts the velocity field; target is the straight-line
    velocity x1 - x0. No ODE solve anywhere in this function.
    """
    x0 = torch.randn_like(x1)                          # base sample
    t = torch.rand(x1.shape[0], 1, device=x1.device)   # time in [0,1]
    x_t = (1 - t) * x0 + t * x1                         # linear interpolation
    target = x1 - x0                                   # constant velocity
    v_pred = model(x_t, t.squeeze(-1))                 # predicted velocity
    return ((v_pred - target) ** 2).mean()             # plain MSE

@torch.no_grad()
def sample(model, shape, steps=25):
    """Sampling DOES integrate the ODE — but only at inference, few steps."""
    x = torch.randn(shape)                             # z ~ N(0, I)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), i * dt)
        x = x + model(x, t) * dt                       # Euler step forward
    return x
```

Compare this to training a CNF by maximum likelihood (Section 8): that required a full forward-and-backward ODE solve *per training step*. Flow matching replaces all of it with one MSE evaluation. The training cost drops by **one to two orders of magnitude**, and what you get is a velocity field that defines exactly the kind of continuous flow CNFs promised — expressive, ODE-defined, invertible — but trainable at the scale of billion-parameter text-to-image models.

### Why regressing the conditional velocity is legitimate

There is a subtle question lurking here that deserves a real answer, because it is the theoretical heart of flow matching and the reason it is *correct* and not just a convenient hack. We want $v_\theta$ to match the **marginal** velocity field $u_t(x)$ that transports the whole base distribution to the whole data distribution. But that marginal field is intractable — it would require knowing the marginal probability path, which is exactly the hard object we are trying to model. What we *can* write down is the **conditional** velocity, $u_t(x \mid x_1) = x_1 - x_0$, the velocity of the straight line for one specific data point $x_1$. The deep result of Lipman et al. is that **regressing onto the conditional velocity yields the same gradients as regressing onto the marginal velocity.** Formally, the conditional flow-matching loss $\mathbb{E}\|v_\theta(x_t, t) - u_t(x_t \mid x_1)\|^2$ and the (intractable) marginal loss $\mathbb{E}\|v_\theta(x_t, t) - u_t(x_t)\|^2$ have *identical* gradients with respect to $\theta$, because the marginal velocity is the conditional-velocity expectation $u_t(x) = \mathbb{E}[u_t(x \mid x_1) \mid x_t = x]$ and MSE regression to a target equals MSE regression to its conditional mean up to a constant. So minimizing the easy, tractable, per-example conditional loss provably trains $v_\theta$ toward the correct marginal field. That equivalence is what makes flow matching rigorous: you regress onto a target you *can* compute and provably learn the target you *cannot*. It is the same trick, structurally, that makes denoising score matching work for diffusion — regress onto a tractable conditional (the noise direction for one example) and recover the intractable marginal (the score). Flow matching and diffusion training are siblings under the skin; they differ in the path and the parameterization, not in the regression principle.

### Why this is the bridge to the 2025–2026 frontier

Now connect it to diffusion, because this is *why* SD3 and FLUX moved off diffusion SDEs. A [diffusion model](/blog/machine-learning/image-generation/diffusion-from-first-principles) also learns to map noise to data, but it does so by learning the **score** (the gradient of the log-density) and sampling by integrating a *stochastic* differential equation along a *curved* probability path — the path implied by the variance-preserving noising schedule. Flow matching learns a **velocity** and samples by integrating an *ordinary* differential equation along a *straight* (or nearly straight) path. The straight path is the prize: a straighter trajectory means a numerical ODE solver needs **far fewer steps** to integrate accurately, because there is less curvature to track. That is the mechanism behind few-step sampling, and rectified flow's "reflow" procedure straightens the paths even further, enabling high-quality generation in a handful of steps — the topic the [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) post takes up in full, where SD3's logit-normal time sampling and FLUX's recipe live.

Here is the lineage in one line, and it is worth committing to memory: **change of variables → discrete normalizing flow (RealNVP/Glow) → continuous normalizing flow (neural ODE) → flow matching (regress the velocity, skip the simulation) → SD3/FLUX.** Every arrow is a response to a cost. Discrete flows pay with constrained architectures; CNFs lift the constraint but pay with simulation; flow matching pays neither, by giving up the explicit likelihood objective in favor of a regression that *implicitly* defines the same flow. The change-of-variables formula you derived in Section 1 is still the theoretical foundation under all of it — flow matching's velocity field is exactly the field whose ODE transports the base density to the data density per the continuous change of variables — but we no longer compute the log-det. We route around it. That is the move.

## 10. Case studies: real models and real numbers

Let me ground all of this in named models and measured numbers from the literature, honestly flagged where they are approximate.

**Glow on faces (Kingma & Dhariwal, 2018).** Glow generated $256\times256$ CelebA-HQ faces sharp enough to be a genuine demo at the time, and its semantic latent interpolations (smile ↔ no-smile, young ↔ old, by manipulating latent attribute directions) were the headline. On likelihood, Glow reached about **3.35 bits/dim on CIFAR-10** and **1.03 bits/dim on ImageNet 32×32** — competitive likelihoods. But its CelebA / CIFAR sample FID lagged GANs by a wide margin, and the model was large: the CelebA-HQ Glow used on the order of hundreds of millions of parameters for an image quality a far smaller StyleGAN beat. **Lesson: exact likelihood and pretty latent interpolations did not close the FID gap.**

**RealNVP / Glow likelihood vs autoregressive (the bpd table).** On CIFAR-10: RealNVP ≈ **3.49 bpd**, Glow ≈ **3.35 bpd**, PixelCNN++ ≈ **2.92 bpd**, image diffusion (DDPM, reported as a likelihood) ≈ **3.70 bpd** but with vastly better samples. Read this carefully: the autoregressive model wins on raw likelihood, the flow is middle of the pack, and diffusion has *worse* likelihood than the flow yet *far better* samples. This is the clean empirical proof that **likelihood ≠ sample quality** — the central caution of Section 7.

**Continuous flows: FFJORD (Grathwohl et al., 2019).** FFJORD made CNFs trainable with the Hutchinson trace estimator and got competitive densities on tabular data and small images — and demonstrated exactly the cost story: training required hundreds of function evaluations per step, and it did not scale to high-resolution images. It is the proof-of-concept that CNFs are beautiful and impractical, the gap flow matching closed.

**Flow matching at scale: Stable Diffusion 3 (Esser et al., 2024) and FLUX (Black Forest Labs, 2024).** Both are trained with a flow-matching / rectified-flow objective on a straight (conditional-OT) probability path, using an MM-DiT transformer backbone. SD3's paper reports that the rectified-flow formulation with their logit-normal timestep weighting outperformed the equivalent diffusion (ε/v-prediction) parameterizations on the same architecture and data — a controlled, apples-to-apples win for flow matching over the diffusion objective. FLUX.1 pushed the same recipe to a ~12B-parameter model with state-of-the-art prompt adherence and a few-dozen-step (and, in the schnell distilled variant, ~1–4 step) sampling profile. **This is the payoff of the whole post: the modern frontier is flow matching, and flow matching is the practical descendant of the change-of-variables formula.** The [MM-DiT recipe post](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) covers the architecture; this post covers *why the objective changed*.

**Rectified flow's reflow: straightening for one-step generation (Liu et al., 2023).** Worth one more case because it closes the loop on the speed story. Rectified flow's "reflow" procedure takes a trained flow, generates noise-data pairs *with it*, and retrains on those pairs — which provably **straightens** the trajectories (reduces their curvature) toward genuinely straight lines. A perfectly straight path can be integrated exactly in **one Euler step**, so reflow pushes the model toward one-step generation. The paper reported that two rounds of reflow plus distillation yielded competitive CIFAR-10 FID in a *single* network evaluation. This is the continuous-time fulfillment of the Parallel WaveNet idea from Section 6: train a flexible model, then distill it into a fast sampler — except now "fast" means one ODE step instead of one parallel pass, and the straightness of the path is the property that makes one step enough. The lineage from MAF/IAF distillation to reflow is not a coincidence; it is the same maneuver, replayed one level up in the continuous-time setting.

#### Worked example: steps versus quality on a straight path

Make the few-step claim concrete. Suppose a flow-matching model and a curved-path diffusion model are both trained to the same quality at many steps. Sampling with a fixed-step Euler solver, the *local* integration error per step scales with the path's curvature. On a nearly straight flow-matching path, dropping from 50 steps to **8 steps** might cost only a fraction of an FID point, because there is little curvature for the solver to miss — a usable few-step Pareto point you can ship. On a strongly curved diffusion ODE, the same drop to 8 steps can cost several FID points and visible artifacts, because the solver overshoots the bends. This is the mechanism, in one sentence: **straighter paths tolerate bigger steps, so flow matching buys few-step sampling almost for free** — and reflow buys even fewer. The exact numbers are model-specific (measure them yourself with a fixed seed, a 10k-sample FID against a held-out reference set, and a warmed-up GPU), but the *direction* is robust and is precisely why the frontier moved to ODEs on straight paths.

## 11. When to reach for flows (and when not to)

Be decisive about this, because flows are a specialist tool in 2026, not a default.

**Reach for a discrete normalizing flow (RealNVP/Glow-style) when** you genuinely need an **exact, tractable likelihood** $\log p(x)$ — and not a bound. The real use cases: **out-of-distribution / anomaly detection** (score a sample by its density; low density = anomaly), **density estimation** on tabular or scientific data, **lossless compression** (the exact likelihood gives you the optimal code length), simulation-based inference in physics, and as a **flexible prior or posterior** inside a larger model (the original IAF-in-a-VAE use). For these, a flow's exactness is the feature and sample FID is irrelevant.

**Do not reach for a discrete flow for high-resolution text-to-image.** The constraints (invertibility, tractable Jacobian, no latent compression) cap quality, the parameter cost is high, and a [latent diffusion model](/blog/machine-learning/image-generation/diffusion-from-first-principles) or a flow-matching DiT will beat it on FID at a fraction of the engineering pain. If image quality is the goal, the discrete flow is the wrong tool — its descendant, flow matching, is the right one.

**Do not train a continuous normalizing flow by maximum likelihood at image scale.** The per-step ODE simulation makes it impractical; if you find yourself reaching for a neural ODE for generation, you almost certainly want **flow matching** instead — same continuous-flow expressiveness, none of the simulation cost.

**Reach for flow matching (the modern descendant) when you are doing exactly what SD3 and FLUX do:** training a high-quality generative model where you want straight, few-step ODE sampling and a clean MSE objective. This is now the default for frontier text-to-image, and it is the direct beneficiary of everything in this post.

A quick stress test of the boundaries. *What if you need both exact likelihood and great samples?* You cannot have both in one model today at image scale — pick the axis that matters. *What if you only have 5 training images?* A flow will badly overfit (it is high-capacity and mode-covering); you want a fine-tuning method like LoRA on a pretrained diffusion model, not a flow from scratch. *What if you need the latent for editing?* A flow's exact invertible latent is genuinely nice for editing, but in practice DDIM inversion on a diffusion model or a flow-matching model gives you a usable latent without the flow's quality penalty.

## 12. Key takeaways

- The **change-of-variables formula**, $\log p_X(x) = \log p_Z(z) - \log|\det J_f(z)|$, is the entire foundation: it lets you compute the exact density of $x$ from a known base density and the Jacobian of an invertible map.
- Flows demand two things of $f$: it must be **invertible**, and its **Jacobian determinant must be cheap**. A general determinant is $O(D^3)$; every flow architecture is a trick to make it $O(D)$.
- **Coupling layers** (RealNVP) get a cheap Jacobian by transforming half the input as a function of the other half, yielding a **triangular Jacobian** whose log-det is just the sum of the scale-net outputs. The scale/shift nets never need to be inverted.
- **Glow** industrialized coupling flows with **actnorm** (flow-friendly normalization) and **1×1 invertible convolutions** (learned channel mixing), producing sharp faces and smooth latent interpolations.
- **Autoregressive flows** (MAF/IAF) are more expressive but expose a **forward/inverse speed asymmetry** — fast one way, sequential the other — and the train-where-scoring-is-cheap, deploy-where-sampling-is-cheap distillation idea (Parallel WaveNet) prefigures the modern speed story.
- Flows **lagged GANs and diffusion on image FID** for three structural reasons: the constraints cap expressiveness, the no-compression latent wastes capacity, and — most importantly — **exact likelihood is mode-covering, not mode-seeking, so likelihood ≠ sample quality.**
- **Continuous normalizing flows** (neural ODEs) lift the architectural constraints — any network can be the velocity field, with the log-det becoming the **trace/divergence** — but pay a fatal **per-step ODE-simulation cost** at training time.
- **Flow matching / rectified flow** regresses the velocity field directly with a plain MSE loss, **skipping the simulation entirely** at training time; the straight-line path means few-step sampling, which is precisely why **SD3 and FLUX moved from diffusion SDEs to flow-matching ODEs.**
- The whole lineage — change of variables → RealNVP/Glow → CNF → flow matching → SD3/FLUX — is a sequence of responses to the cost of computing or simulating the Jacobian. Modern models keep the continuous flow's expressiveness and route around its cost.

## Further reading

- **NICE / RealNVP** — Dinh, Krueger & Bengio, "NICE: Non-linear Independent Components Estimation" (2014); Dinh, Sohl-Dickstein & Bengio, "Density Estimation using Real NVP" (2017). The coupling-layer foundation.
- **Glow** — Kingma & Dhariwal, "Glow: Generative Flow with Invertible 1×1 Convolutions" (2018). Actnorm, the 1×1 conv, multi-scale faces.
- **MAF / IAF** — Papamakarios, Pavlakou & Murray, "Masked Autoregressive Flow for Density Estimation" (2017); Kingma et al., "Improving Variational Inference with Inverse Autoregressive Flow" (2016). The forward/inverse asymmetry.
- **Neural ODEs / CNF** — Chen, Rubanova, Bettencourt & Duvenaud, "Neural Ordinary Differential Equations" (2018); Grathwohl et al., "FFJORD" (2019). The instantaneous change of variables and its simulation cost.
- **Flow matching / rectified flow** — Lipman, Chen, Ben-Hamu, Nickel & Le, "Flow Matching for Generative Modeling" (2023); Liu, Gong & Liu, "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (2023). The objective that powers the frontier.
- **Flow matching at scale** — Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3, 2024). Flow matching beating diffusion parameterizations head-to-head.
- **Within this series** — start from [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard) and [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions); continue to [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) for the payoff; and see it all assembled in [building an image-generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
