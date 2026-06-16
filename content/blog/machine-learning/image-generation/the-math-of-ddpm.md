---
title: "The Math of DDPM: Deriving the Denoising Objective From the Variational Bound"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Derive the full DDPM training objective from scratch — the closed-form forward marginal, the variational bound, the tractable Gaussian posterior, the noise-prediction reparameterization, and why the field quietly dropped the weighting term to get L_simple — with PyTorch that matches every line of algebra."
tags:
  [
    "image-generation",
    "diffusion-models",
    "ddpm",
    "variational-bound",
    "elbo",
    "noise-prediction",
    "generative-ai",
    "deep-learning",
    "pytorch",
    "probability",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/the-math-of-ddpm-1.png"
---

There is a moment, the first time you read the DDPM paper, where you hit the loss function and it looks like a magic trick. You start with a fully rigorous variational bound — a tower of KL divergences, a marginal likelihood you cannot compute, the whole machinery of variational inference — and three pages later it has collapsed into

$$
\mathcal{L}_\text{simple} = \mathbb{E}_{t, x_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(x_t, t) \right\|^2 \right].
$$

A neural network that predicts the noise you added. That is it. That is the loss that trained Stable Diffusion, that trains every latent diffusion model shipping today, that turned "generate a 512×512 cat" from a research curiosity into a product. And it looks, on the page, like nothing — a plain squared error you could have written down on day one without any of the variational bound at all.

This post is the bridge between those two facts. We are going to derive $\mathcal{L}_\text{simple}$ from the variational bound, every non-trivial step, so that by the end you can reproduce the derivation on a whiteboard and — more importantly — you understand *why* each simplification is legal and *why* the final shortcut (dropping a weighting term that the math says should be there) makes the model **better**, not just simpler. This is the rigorous companion to [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles), which builds the intuition; here we do the algebra that intuition was standing on.

![A branching graph of the DDPM Markov chain showing the forward noising path from clean data x_0 to pure Gaussian noise x_T, and the tractable forward posterior conditioned on both x_t and x_0 that serves as the training target for the learned reverse model.](/imgs/blogs/the-math-of-ddpm-1.png)

The diagram above is the whole game in one picture, and figure 1 is worth pinning to the top of the page. The forward process (top row) walks a clean image $x_0$ into pure noise $x_T$ one small Gaussian step at a time. The reverse process $p_\theta$ — the thing we want to learn — tries to walk it back. The trick that makes the math tractable is the node in the middle: we never ask the network to match the true reverse $q(x_{t-1}\mid x_t)$ directly (that distribution is intractable). Instead we condition on $x_0$ and match the *posterior* $q(x_{t-1}\mid x_t, x_0)$, which turns out to be a clean closed-form Gaussian. Everything below is the consequence of that one move.

By the end you will be able to: derive the closed-form marginal $q(x_t\mid x_0)$ and use it to sample $x_t$ in one line; write down the variational bound and split it into its three roles; compute the forward posterior mean and variance from scratch; reparameterize that mean in terms of predicted noise; and implement `q_sample`, the posterior, and the loss in PyTorch with a numerical check that the closed form actually matches iterated sampling. We will tie all of it back to the series spine — the **generative trilemma** (quality × diversity × speed) and the **diffusion stack** — and end with the schedule choices ($\beta_t$ linear vs cosine, $\sigma_t$) that quietly decide whether your samples are crisp or gray mush.

One note on how to read this post. The derivation is the same whether your $x_0$ is a 32×32 CIFAR-10 image, a 1×28×28 MNIST digit, or a 64×64×4 Stable Diffusion latent — the math treats $x_0$ as a vector in $\mathbb{R}^d$ and never looks at its shape. So I will keep a single running example, **unconditional CIFAR-10 with $T=1000$ on the linear schedule**, because that is the exact setup of the original DDPM paper and lets us quote real numbers (FID 3.17) at the end. Everywhere the algebra appears, picture that 32×32×3 tensor being noised and denoised; everywhere a number appears, it is from a model someone actually trained. The point of a from-scratch derivation is that you could re-run it tonight on a single GPU and reproduce those numbers — and the PyTorch in each section is written so you can.

A word on prerequisites, since this is the math-heavy post in the series. You should be comfortable with: Gaussians and their density, the reparameterization trick (sampling $\mathcal N(\mu,\sigma^2)$ as $\mu + \sigma z$ with $z\sim\mathcal N(0,1)$), KL divergence as a distance-like quantity between distributions, and the ELBO as a lower bound on log-likelihood. If the ELBO is hazy, read [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) first — DDPM's bound is the *same* ELBO machinery applied to a length-$T$ chain instead of a single latent, so the VAE is genuinely the right warm-up. Everything else we build here.

## 1. Notation and the two processes we are juggling

Before any derivation, let us nail down the objects, because half the confusion in diffusion math is sloppy notation. We have a data point $x_0 \sim q(x_0)$ — for us, an image, flattened or kept as a tensor; the math does not care about its shape, only that it is a vector in $\mathbb{R}^d$. We define a sequence of **latents** $x_1, x_2, \dots, x_T$ of the *same dimension* as $x_0$ (this is unlike a VAE, where the latent is smaller — see [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) for that contrast). $T$ is typically 1000.

There are two processes, and you must always know which one you are in:

- The **forward process** (also "diffusion process") $q$ is *fixed*. It has no learnable parameters. It progressively adds Gaussian noise according to a **variance schedule** $\beta_1, \dots, \beta_T$ with each $\beta_t \in (0,1)$. We define it as a Markov chain:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t \mathbf{I}\right).
$$

- The **reverse process** $p_\theta$ is *learned*. It starts from pure noise $p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$ and walks back down:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\ \boldsymbol{\mu}_\theta(x_t, t),\ \boldsymbol{\Sigma}_\theta(x_t, t)\right).
$$

The whole reverse trajectory is $p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t)$, and the marginal we ultimately care about is $p_\theta(x_0) = \int p_\theta(x_{0:T})\, dx_{1:T}$ — the model's density over images. We want to maximize the likelihood it assigns to real data, but that integral is intractable, so we bound it. That is the entire plot.

Two terms to define now because they will appear everywhere. **SNR** (signal-to-noise ratio) at step $t$ is the ratio of squared signal coefficient to noise variance in $x_t$; it falls monotonically from huge (at $t=0$, all signal) to roughly zero (at $t=T$, all noise), and it is the single best lens for "how hard is denoising at this step." And the **ELBO** (evidence lower bound) is the variational lower bound on $\log p_\theta(x_0)$ that we maximize in place of the intractable likelihood — exactly the ELBO derived in [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions), reused here on a much longer chain.

A small but load-bearing design decision: $\sqrt{1-\beta_t}$, not $1$, multiplies the mean. Why scale the signal down as we add noise? Because it keeps the **variance of $x_t$ bounded**. If $x_0$ has unit variance and we kept the mean coefficient at 1 while adding $\beta_t$ of variance each step, the total variance would grow without bound over 1000 steps. With the $\sqrt{1-\beta_t}$ shrink, the variance stays near 1 throughout — a "variance-preserving" chain. That single algebraic choice is what makes the next section's closed form so clean.

This is worth one more sentence because the name shows up everywhere downstream. Song et al.'s SDE framework calls this the **VP-SDE** (variance-preserving) family, in contrast to the **VE-SDE** (variance-exploding) family used by the score-matching NCSN line, where the signal is *not* shrunk and the noise variance grows without bound instead. Both are valid forward processes and both have a reverse; DDPM happens to use the variance-preserving one, and that is why our marginal variance is $(1-\bar\alpha_t)$ — a clean number between 0 and 1 — rather than some growing quantity. If you ever wonder why a Stable Diffusion latent stays well-scaled across all 1000 steps while a naive "just keep adding noise" chain would blow up your activations and your loss, this is the reason: the $\sqrt{1-\beta_t}$ factor is doing quiet, essential numerical work on every single step. Get this factor wrong in an implementation (a surprisingly common bug) and your $\bar\alpha_t$ will not telescope correctly, your terminal state will not be standard normal, and your samples will be garbage in a way that is maddening to trace back to one missing square root.

## 2. The closed-form forward marginal $q(x_t \mid x_0)$

The first beautiful thing about DDPM is that you never have to actually run the forward chain. Even though $q(x_t \mid x_0)$ is defined as $t$ composed Gaussian steps, the composition has a closed form: it is *one* Gaussian. Let us derive it, because this is the workhorse — it is what makes training $O(1)$ per sample instead of $O(T)$.

![A branching graph showing how the per-step variance schedule beta combines into alpha equal to one minus beta, then a cumulative product alpha-bar, which together with the clean image and a single Gaussian noise draw produces the closed-form marginal for x_t in one jump.](/imgs/blogs/the-math-of-ddpm-2.png)

Define $\alpha_t = 1 - \beta_t$ and the cumulative product $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Figure 2 traces this collapse. Now apply the reparameterization trick to one forward step: sampling $x_t \sim q(x_t\mid x_{t-1})$ is the same as computing

$$
x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1-\alpha_t}\, \boldsymbol{\epsilon}_{t-1}, \qquad \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
$$

Substitute the same expansion for $x_{t-1}$:

$$
x_t = \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}\, x_{t-2} + \sqrt{1-\alpha_{t-1}}\, \boldsymbol{\epsilon}_{t-2}\right) + \sqrt{1-\alpha_t}\, \boldsymbol{\epsilon}_{t-1}.
$$

Collect terms:

$$
x_t = \sqrt{\alpha_t \alpha_{t-1}}\, x_{t-2} + \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\, \boldsymbol{\epsilon}_{t-2} + \sqrt{1-\alpha_t}\, \boldsymbol{\epsilon}_{t-1}}_{\text{sum of two independent Gaussians}}.
$$

Here is the key lemma: **the sum of two independent zero-mean Gaussians is Gaussian, with variances adding.** The first noise term has variance $\alpha_t(1-\alpha_{t-1})$, the second has variance $(1-\alpha_t)$. Their sum has variance

$$
\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = \alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t\alpha_{t-1}.
$$

So the two noise injections merge into a single $\sqrt{1-\alpha_t\alpha_{t-1}}\,\bar{\boldsymbol{\epsilon}}$ with a fresh $\bar{\boldsymbol{\epsilon}} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$. Notice the pattern: after folding in $x_{t-2}$, the signal coefficient is $\sqrt{\alpha_t\alpha_{t-1}}$ and the noise variance is $1 - \alpha_t\alpha_{t-1}$. Induct all the way down to $x_0$ and the products telescope into $\bar{\alpha}_t$:

$$
\boxed{\ x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}). \ }
$$

Equivalently, as a distribution:

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar{\alpha}_t}\, x_0,\ (1-\bar{\alpha}_t)\mathbf{I}\right).
$$

This is the single most-used equation in all of diffusion. Read it physically: $x_t$ is your clean image scaled down by $\sqrt{\bar{\alpha}_t}$ plus noise scaled by $\sqrt{1-\bar{\alpha}_t}$. As $t \to T$, $\bar{\alpha}_t \to 0$, so $x_T \to \boldsymbol{\epsilon}$: pure noise, matching the prior $p(x_T) = \mathcal{N}(\mathbf{0},\mathbf{I})$. As $t \to 0$, $\bar{\alpha}_t \to 1$, so $x_0$ comes back. The SNR is exactly $\bar{\alpha}_t / (1-\bar{\alpha}_t)$ — pure signal over pure noise, falling monotonically because $\bar{\alpha}_t$ is a product of numbers below 1. That ratio is the whole story of "how much is left to denoise," and we will return to it when we choose schedules.

#### Worked example: variance preservation in numbers

Take the linear schedule from Ho et al.: $\beta_1 = 10^{-4}$ rising linearly to $\beta_T = 0.02$ over $T=1000$. Then $\alpha_1 = 0.9999$, and $\bar{\alpha}_1 = 0.9999$ — at $t=1$ we have essentially the clean image with a whisper of noise, SNR $\approx 0.9999/0.0001 \approx 9999$. At $t=1000$, the cumulative product $\bar{\alpha}_{1000} \approx 4\times10^{-5}$, so the signal coefficient $\sqrt{\bar{\alpha}_{1000}} \approx 0.006$ — the image is gone, and SNR $\approx 4\times10^{-5}$, effectively zero. The variance of $x_t$ at every step is $\bar{\alpha}_t \cdot \text{Var}(x_0) + (1-\bar{\alpha}_t) \approx 1$ when $x_0$ is normalized to unit variance: bounded the whole way, exactly as the $\sqrt{1-\beta_t}$ scaling promised. This is the practical payoff of the variance-preserving choice — your activations never blow up across 1000 steps.

Here is `q_sample`, the closed-form forward, in PyTorch — one line of real work, called millions of times during training:

```python
import torch

def make_schedule(T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
    # Linear beta schedule from Ho et al. (2020).
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)  # alpha_bar_t = prod_{s<=t} alpha_s
    return betas, alphas, alpha_bars

def q_sample(x0, t, alpha_bars, noise=None):
    """Sample x_t ~ q(x_t | x_0) in closed form. t is a LongTensor of indices."""
    if noise is None:
        noise = torch.randn_like(x0)
    # gather sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t), broadcast over image dims
    a_bar = alpha_bars[t].view(-1, *([1] * (x0.dim() - 1)))
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise, noise
```

Notice we return the `noise` too: that draw $\boldsymbol{\epsilon}$ is the *label* the network will try to predict. That is the whole supervised signal of diffusion training, and it falls directly out of this equation. No iteration, no chain — one shot.

## 3. The variational bound: from intractable likelihood to a sum of KLs

We want to maximize $\log p_\theta(x_0)$ but cannot compute the integral over all the latents. The standard move — identical in spirit to the VAE ELBO, just on a length-$T$ chain — is to introduce the forward process $q$ as our (fixed, non-learned) approximate posterior and apply Jensen's inequality.

Start from the marginal and multiply by $1 = q(x_{1:T}\mid x_0)/q(x_{1:T}\mid x_0)$ inside the integral:

$$
\log p_\theta(x_0) = \log \int p_\theta(x_{0:T})\, dx_{1:T} = \log \mathbb{E}_{q(x_{1:T}\mid x_0)}\!\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right].
$$

Jensen's inequality ($\log \mathbb{E}[\cdot] \ge \mathbb{E}[\log \cdot]$, since $\log$ is concave) gives the bound:

$$
\log p_\theta(x_0) \ge \mathbb{E}_q\!\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right] =: -\mathcal{L}_\text{vlb}.
$$

So $\mathcal{L}_\text{vlb} = \mathbb{E}_q\!\left[\log \frac{q(x_{1:T}\mid x_0)}{p_\theta(x_{0:T})}\right]$ is an *upper bound* on the negative log-likelihood, and minimizing it pushes the model's likelihood up. So far this is just the ELBO. The DDPM-specific work is expanding it into per-step terms that are individually tractable.

![A vertical stack showing the reduction from the evidence lower bound down through the sum of expectations, a single KL between Gaussians, the closed-form squared-distance of means, and finally the noise-regression form that becomes L_simple.](/imgs/blogs/the-math-of-ddpm-3.png)

Figure 3 previews where this is heading: ELBO at the top, $\mathcal{L}_\text{simple}$ at the bottom, each layer a more trainable rewrite. Now the expansion. Substitute the factorized forms $p_\theta(x_{0:T}) = p(x_T)\prod_t p_\theta(x_{t-1}\mid x_t)$ and $q(x_{1:T}\mid x_0) = \prod_t q(x_t\mid x_{t-1})$:

$$
\mathcal{L}_\text{vlb} = \mathbb{E}_q\!\left[\log \frac{\prod_{t\ge1} q(x_t\mid x_{t-1})}{p(x_T)\prod_{t\ge1} p_\theta(x_{t-1}\mid x_t)}\right] = \mathbb{E}_q\!\left[-\log p(x_T) + \sum_{t\ge1}\log\frac{q(x_t\mid x_{t-1})}{p_\theta(x_{t-1}\mid x_t)}\right].
$$

If you stop here, you have a valid loss, but it is a disaster to optimize: each $\log\frac{q(x_t\mid x_{t-1})}{p_\theta(x_{t-1}\mid x_t)}$ is a ratio between a forward step and a reverse step that point in opposite directions, and Monte-Carlo estimating it has brutal variance. The fix — this is the crux of the whole derivation — is to **rewrite $q(x_t\mid x_{t-1})$ using Bayes' rule, conditioned on $x_0$**, so that we compare the reverse model against a forward *posterior* that points the same way.

### 3.1 The Bayes trick that conditions on $x_0$

Because the forward process is Markov, $q(x_t\mid x_{t-1}) = q(x_t\mid x_{t-1}, x_0)$ (adding $x_0$ to the conditioning changes nothing — $x_{t-1}$ already screens it off). Now apply Bayes:

$$
q(x_t\mid x_{t-1}, x_0) = \frac{q(x_{t-1}\mid x_t, x_0)\, q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}.
$$

Substitute this for every $t > 1$ in the sum. Let us actually do the telescoping rather than waving at it, because this is the one step most write-ups skip and it is where the elegance lives. Split the sum into the $t=1$ term and the $t \ge 2$ terms. For $t \ge 2$, the log of the substituted forward step becomes

$$
\log\frac{q(x_t\mid x_{t-1})}{p_\theta(x_{t-1}\mid x_t)} = \log\frac{q(x_{t-1}\mid x_t, x_0)}{p_\theta(x_{t-1}\mid x_t)} + \log\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)}.
$$

The first piece is a clean per-step comparison of reverse model against forward posterior — exactly what we wanted. The second piece is the telescoping ratio. Sum it over $t = 2, \dots, T$:

$$
\sum_{t=2}^{T} \log\frac{q(x_t\mid x_0)}{q(x_{t-1}\mid x_0)} = \log\frac{q(x_T\mid x_0)}{q(x_1\mid x_0)},
$$

because every numerator $q(x_t\mid x_0)$ cancels the denominator $q(x_t\mid x_0)$ of the next term — a textbook telescoping sum collapsing to its two endpoints. Now assemble everything. We had $\mathcal{L}_\text{vlb} = \mathbb{E}_q[-\log p(x_T) + \sum_{t\ge1} \log\frac{q(x_t\mid x_{t-1})}{p_\theta(x_{t-1}\mid x_t)}]$. Substitute the split for $t\ge2$, add the telescoped endpoint, and keep the $t=1$ term as $\log\frac{q(x_1\mid x_0)}{p_\theta(x_0\mid x_1)}$. The $q(x_1\mid x_0)$ in the telescoped denominator cancels the $q(x_1\mid x_0)$ in the $t=1$ numerator, leaving a $q(x_T\mid x_0)$ that pairs with $-\log p(x_T)$ to form a KL. Grouping the survivors:

$$
\mathcal{L}_\text{vlb} = \mathbb{E}_q\!\left[\log\frac{q(x_T\mid x_0)}{p(x_T)} + \sum_{t=2}^{T}\log\frac{q(x_{t-1}\mid x_t,x_0)}{p_\theta(x_{t-1}\mid x_t)} - \log p_\theta(x_0\mid x_1)\right].
$$

Finally, recognize that taking the expectation $\mathbb{E}_q$ of each $\log$-ratio over the appropriate conditional turns it into a KL divergence — $\mathbb{E}_{q(z)}[\log\frac{q(z)}{p(z)}] = D_\text{KL}(q\,\|\,p)$ by definition. The first ratio, under $\mathbb{E}_{q(x_T\mid x_0)}$, is $D_\text{KL}(q(x_T\mid x_0)\,\|\,p(x_T))$; each summand, under $\mathbb{E}_{q(x_t, x_{t-1}\mid x_0)}$, is $D_\text{KL}(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t))$. That gives the canonical three-part decomposition:

$$
\mathcal{L}_\text{vlb} = \underbrace{D_\text{KL}\!\left(q(x_T\mid x_0)\,\|\,p(x_T)\right)}_{\mathcal{L}_T}
+ \sum_{t=2}^{T} \underbrace{D_\text{KL}\!\left(q(x_{t-1}\mid x_t, x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\right)}_{\mathcal{L}_{t-1}}
- \underbrace{\log p_\theta(x_0\mid x_1)}_{\mathcal{L}_0}.
$$

Take a breath and read what each term *does*, because this decomposition is the skeleton of everything that follows. Figure 6 below tabulates it, but in words:

- $\mathcal{L}_T = D_\text{KL}(q(x_T\mid x_0)\,\|\,p(x_T))$ is the **prior-matching term**. It measures how far the fully-noised $x_T$ is from the standard Gaussian prior. Crucially, $q$ has no parameters and $p(x_T)$ is fixed, so **$\mathcal{L}_T$ has no trainable parameters at all** — it is a constant w.r.t. $\theta$. With a well-designed schedule $\bar\alpha_T \approx 0$, so $q(x_T\mid x_0) \approx \mathcal{N}(\mathbf 0, \mathbf I)$ and this KL is near zero anyway. We drop it.
- $\mathcal{L}_{t-1} = D_\text{KL}(q(x_{t-1}\mid x_t, x_0)\,\|\,p_\theta(x_{t-1}\mid x_t))$ for $t = 2,\dots,T$ is the **denoising-matching term**, and this is where *all* the learning happens. It asks the reverse model $p_\theta$ to match the tractable forward posterior $q(x_{t-1}\mid x_t, x_0)$ at every step. Both are Gaussians, so this KL has a closed form. This is the term we will grind on in Section 5.
- $\mathcal{L}_0 = -\log p_\theta(x_0\mid x_1)$ is the **reconstruction / decode term** — the negative log-likelihood of the clean image under the last reverse step. In Ho et al. it is handled by a discrete decoder mapping the continuous $x_1$ back to the $\{0,\dots,255\}$ pixel values. In practice it folds into the $t=1$ case of the same loss.

The decomposition is exact — no approximation has been made yet, only algebra. We turned an intractable likelihood into a sum of KLs between Gaussians, each of which we can write in closed form. That is the entire purpose of Section 3.

## 4. Why the reverse process is Gaussian (and why small steps matter)

Before computing the posterior, settle a question that quietly underpins the whole construction: *why is it legitimate to model $p_\theta(x_{t-1}\mid x_t)$ as a Gaussian at all?* The true reverse of a diffusion is generally not Gaussian — it can be a complicated multimodal mess (given a noisy $x_t$, many clean images could have produced it). So why does a single Gaussian per step work?

The answer is a classical result from the theory of diffusion processes, and it hinges on the step size $\beta_t$ being *small*. Feller (1949) showed that for a continuous-time diffusion, the reverse-time transition over an infinitesimal interval is Gaussian. Discretely: if each forward step adds only a little noise ($\beta_t \ll 1$), then the reverse step over that same small interval is *approximately* Gaussian — the true reverse posterior $q(x_{t-1}\mid x_t)$ (marginalized over $x_0$) is well-approximated by a Gaussian when the step is small. This is precisely why $T$ is large (1000) and each $\beta_t$ is tiny: it is not a hyperparameter you tune for fun, it is what *licenses the Gaussian reverse model*. Crank $\beta_t$ up so each step is big, and the per-step reverse becomes genuinely multimodal, a single Gaussian can no longer fit it, and your samples degrade. Many small steps, each individually Gaussian, is the deal diffusion strikes.

It is worth seeing *why* small steps make the reverse Gaussian, because the reason is not hand-waving — it is the structure of the continuous-time limit. As $T\to\infty$ and each $\beta_t\to0$, the forward Markov chain converges to a stochastic differential equation (SDE) of the form $dx = -\tfrac12\beta(t)\,x\,dt + \sqrt{\beta(t)}\,d\mathbf w$, where $\mathbf w$ is a Wiener process. Anderson's theorem (1982) states that *every* such diffusion SDE has a reverse-time SDE that is also a diffusion — meaning its infinitesimal transitions are Gaussian, with a drift that depends on the score $\nabla_x\log q_t(x)$ (the same score we connect $\boldsymbol\epsilon_\theta$ to in Section 9). A diffusion's reverse is Gaussian *in the infinitesimal limit*, exactly. The discrete DDPM step with small $\beta_t$ is a first-order approximation of that exactly-Gaussian infinitesimal reverse, and the approximation error shrinks as the step shrinks. So "model the reverse as a Gaussian" is not a convenient lie — it is the leading-order-correct truth, with error controlled by $\beta_t$. That is the rigorous content behind "small steps make it Gaussian," and it is why the SDE view (Track B's grand unification) and the DDPM view are the same theory.

This connects straight to the **generative trilemma**. Diffusion buys its excellent quality and mode coverage precisely by taking many small steps — but those many steps are exactly why naive DDPM sampling is *slow* (1000 network calls per image). That tension is the entire subject of the fast-sampling literature: [DDIM and fast deterministic sampling](/blog/machine-learning/image-generation/diffusion-from-first-principles) and the samplers that follow find ways to take *bigger* effective steps without paying the multimodality tax, often by switching to a deterministic ODE view. Hold that thought; here we stay in the small-step regime where everything is Gaussian and the math is exact.

There is a second simplification worth naming. Ho et al. fix the reverse **variance** to a schedule-derived constant, $\boldsymbol{\Sigma}_\theta(x_t,t) = \sigma_t^2 \mathbf{I}$, rather than learning it. They try two choices, $\sigma_t^2 = \beta_t$ and $\sigma_t^2 = \tilde\beta_t$ (the posterior variance from the next section), and find both work about equally well for sample quality. So the network only ever has to output the reverse **mean** $\boldsymbol{\mu}_\theta(x_t, t)$ — one prediction per step. (Later work, Nichol & Dhariwal's improved DDPM, *does* learn the variance via an interpolation, which buys better log-likelihood; we will note where that plugs in.) For the core derivation, fixed variance means the KL term collapses to a distance between *means*, which is the simplification that makes the noise-prediction objective fall out so cleanly.

## 5. The tractable forward posterior $q(x_{t-1}\mid x_t, x_0)$

Now the centerpiece. We need the distribution $q(x_{t-1}\mid x_t, x_0)$ in closed form so we can compute the KL in $\mathcal{L}_{t-1}$. We already have all three ingredients as Gaussians: $q(x_t\mid x_{t-1})$ (a forward step), $q(x_{t-1}\mid x_0)$ and $q(x_t\mid x_0)$ (closed-form marginals from Section 2). Bayes again:

$$
q(x_{t-1}\mid x_t, x_0) = \frac{q(x_t\mid x_{t-1})\, q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}.
$$

A product/quotient of Gaussians (in the same variable $x_{t-1}$) is Gaussian, so we know the answer has the form $\mathcal{N}(x_{t-1};\, \tilde{\boldsymbol{\mu}}_t(x_t,x_0),\, \tilde\beta_t \mathbf{I})$. To find $\tilde{\boldsymbol{\mu}}_t$ and $\tilde\beta_t$, expand the exponents (each Gaussian contributes a quadratic in $x_{t-1}$) and complete the square. Writing only the terms that depend on $x_{t-1}$:

$$
\propto \exp\!\left(-\frac{1}{2}\left[\frac{(x_t - \sqrt{\alpha_t}\,x_{t-1})^2}{\beta_t} + \frac{(x_{t-1} - \sqrt{\bar\alpha_{t-1}}\,x_0)^2}{1-\bar\alpha_{t-1}}\right]\right).
$$

Collect the coefficient of $x_{t-1}^2$ (gives the inverse variance) and the coefficient of $x_{t-1}$ (gives mean over variance). The $x_{t-1}^2$ coefficient is

$$
\frac{1}{\tilde\beta_t} = \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}}.
$$

Put over a common denominator and use $\alpha_t(1-\bar\alpha_{t-1}) + \beta_t = \alpha_t - \alpha_t\bar\alpha_{t-1} + (1-\alpha_t) = 1 - \bar\alpha_t$:

$$
\boxed{\ \tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\,\beta_t. \ }
$$

That is the **posterior variance** — note it is a pure function of the schedule, no $x$ in it at all, which is why we can precompute it once. For the mean, the $x_{t-1}$ coefficient gives $\tilde{\boldsymbol{\mu}}_t / \tilde\beta_t = \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}} x_0$, so multiplying through by $\tilde\beta_t$:

$$
\boxed{\ \tilde{\boldsymbol{\mu}}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\, x_t + \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t}\, x_0. \ }
$$

Read this: the posterior mean of $x_{t-1}$ is a **convex-ish blend of where you are ($x_t$) and where you started ($x_0$)**, with weights set entirely by the schedule. It is exact, it is cheap, and it is the target our network must learn to hit. Here it is in PyTorch, every coefficient matching the boxed formulas:

```python
def posterior_mean_variance(x_t, x_0, t, betas, alphas, alpha_bars):
    """Closed-form q(x_{t-1} | x_t, x_0) = N(mu_tilde, beta_tilde * I)."""
    def gather(v):
        return v[t].view(-1, *([1] * (x_t.dim() - 1)))
    beta_t      = gather(betas)
    alpha_t     = gather(alphas)
    abar_t      = gather(alpha_bars)
    # alpha_bar_{t-1}; define alpha_bar_0 = 1 so t=0 is well-behaved
    abar_prev   = gather(torch.cat([torch.ones(1, device=alpha_bars.device),
                                    alpha_bars[:-1]]))
    coef_x0 = (torch.sqrt(abar_prev) * beta_t) / (1.0 - abar_t)
    coef_xt = (torch.sqrt(alpha_t) * (1.0 - abar_prev)) / (1.0 - abar_t)
    mu_tilde   = coef_x0 * x_0 + coef_xt * x_t
    beta_tilde = (1.0 - abar_prev) / (1.0 - abar_t) * beta_t
    return mu_tilde, beta_tilde
```

#### Worked example: how the posterior blends $x_t$ and $x_0$

Make the blend concrete on the linear schedule. At an early step, say $t=50$ where $\bar\alpha_{50}\approx 0.91$ and $\bar\alpha_{49}\approx 0.91$ with $\beta_{50}\approx 1.1\times10^{-3}$: the $x_0$ coefficient $\frac{\sqrt{\bar\alpha_{49}}\,\beta_{50}}{1-\bar\alpha_{50}}\approx \frac{0.95\cdot0.0011}{0.09}\approx 0.011$, and the $x_t$ coefficient is $\approx 0.99$. So early in the chain, the posterior mean of $x_{t-1}$ is *almost entirely* $x_t$ scaled by ~1 — the network barely needs to move, because one step removes only a whisper of noise. Now at a late step $t=900$ where $\bar\alpha_{900}\approx 8\times10^{-3}$ and $\beta_{900}\approx 0.018$: the $x_0$ coefficient jumps to roughly $0.1$ and the $x_t$ coefficient drops correspondingly — the posterior now leans much harder on the (estimated) clean image $x_0$, because $x_t$ itself is mostly noise and carries little usable signal. This is the algebra encoding an obvious truth: **early in denoising, trust where you are; late in denoising, trust where you think you came from.** The schedule coefficients automate that shifting trust, and the network never has to learn it explicitly — it is baked into the posterior we are matching.

### 5.1 The KL term collapses to a distance between means

Now plug into $\mathcal{L}_{t-1}$. It pays to recall the general formula first so the simplification is not a black box. For two $d$-dimensional Gaussians $p = \mathcal N(\boldsymbol\mu_1, \boldsymbol\Sigma_1)$ and $q = \mathcal N(\boldsymbol\mu_2, \boldsymbol\Sigma_2)$:

$$
D_\text{KL}(p\,\|\,q) = \tfrac{1}{2}\!\left[\log\frac{|\boldsymbol\Sigma_2|}{|\boldsymbol\Sigma_1|} - d + \text{tr}(\boldsymbol\Sigma_2^{-1}\boldsymbol\Sigma_1) + (\boldsymbol\mu_2-\boldsymbol\mu_1)^\top\boldsymbol\Sigma_2^{-1}(\boldsymbol\mu_2-\boldsymbol\mu_1)\right].
$$

In our case both arguments are Gaussians with the **same** (fixed) isotropic variance $\sigma_t^2 \mathbf I$ — the posterior has variance $\tilde\beta_t\mathbf I$, and we set the model's variance $\boldsymbol\Sigma_\theta = \sigma_t^2 \mathbf I$ to match it (the $\sigma_t^2 = \tilde\beta_t$ choice). When $\boldsymbol\Sigma_1 = \boldsymbol\Sigma_2 = \sigma_t^2\mathbf I$, the log-det term is zero ($|\boldsymbol\Sigma_2|/|\boldsymbol\Sigma_1| = 1$), the trace term equals $d$ which cancels the $-d$, and the only survivor is the quadratic, which becomes $\frac{1}{2\sigma_t^2}\|\boldsymbol\mu_2-\boldsymbol\mu_1\|^2$. So the KL collapses to a scaled squared distance between means:

$$
D_\text{KL}\!\left(\mathcal{N}(\tilde{\boldsymbol\mu}_t, \sigma_t^2\mathbf I)\,\|\,\mathcal{N}(\boldsymbol\mu_\theta, \sigma_t^2\mathbf I)\right) = \frac{1}{2\sigma_t^2}\left\|\tilde{\boldsymbol\mu}_t - \boldsymbol\mu_\theta\right\|^2 + C,
$$

where $C$ absorbs the variance/log-det terms that do not depend on $\theta$. So, dropping the constant,

$$
\mathcal{L}_{t-1} = \mathbb{E}_q\!\left[\frac{1}{2\sigma_t^2}\left\|\tilde{\boldsymbol\mu}_t(x_t, x_0) - \boldsymbol\mu_\theta(x_t, t)\right\|^2\right].
$$

The intractable likelihood is now a weighted squared error between the true posterior mean and the network's predicted mean. We could stop here and train a network to output $\boldsymbol\mu_\theta$ directly. But there is one more reparameterization that makes it dramatically easier — and it is the move that defines DDPM.

## 6. Reparameterizing the mean as predicted noise

The network *could* predict $\boldsymbol\mu_\theta$ directly. But $\tilde{\boldsymbol\mu}_t$ depends on $x_0$, and at sampling time we do not have $x_0$ — we are trying to *produce* it. So the clever move is to express the target mean in terms of something the network can plausibly recover from $x_t$ alone. The closed-form marginal from Section 2 gives us the bridge. Recall

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon \quad\Longrightarrow\quad x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\left(x_t - \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon\right).
$$

Substitute this expression for $x_0$ into the posterior mean $\tilde{\boldsymbol\mu}_t = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t + \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0$. Let us do this cancellation in full, because the result is so clean that it looks like luck and is actually structure. Replacing $x_0$:

$$
\tilde{\boldsymbol\mu}_t = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t + \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\cdot\frac{1}{\sqrt{\bar\alpha_t}}\left(x_t - \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon\right).
$$

Use $\bar\alpha_t = \alpha_t\bar\alpha_{t-1}$, so $\sqrt{\bar\alpha_{t-1}}/\sqrt{\bar\alpha_t} = 1/\sqrt{\alpha_t}$. The second term's coefficient on $x_t$ becomes $\frac{\beta_t}{(1-\bar\alpha_t)\sqrt{\alpha_t}}$. Collect the two $x_t$ coefficients over the common denominator $(1-\bar\alpha_t)\sqrt{\alpha_t}$:

$$
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} + \frac{\beta_t}{(1-\bar\alpha_t)\sqrt{\alpha_t}} = \frac{\alpha_t(1-\bar\alpha_{t-1}) + \beta_t}{(1-\bar\alpha_t)\sqrt{\alpha_t}} = \frac{1-\bar\alpha_t}{(1-\bar\alpha_t)\sqrt{\alpha_t}} = \frac{1}{\sqrt{\alpha_t}},
$$

where we reused the identity $\alpha_t(1-\bar\alpha_{t-1}) + \beta_t = 1 - \bar\alpha_t$ from Section 5. The $x_t$ coefficient collapses to exactly $1/\sqrt{\alpha_t}$. The $\boldsymbol\epsilon$ coefficient is $-\frac{\beta_t}{(1-\bar\alpha_t)\sqrt{\alpha_t}}\cdot\sqrt{1-\bar\alpha_t} = -\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}$. So the $x_0$-dependence is gone, replaced entirely by $\boldsymbol\epsilon$-dependence, and the whole thing folds into a strikingly clean form:

$$
\boxed{\ \tilde{\boldsymbol\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol\epsilon\right). \ }
$$

This is gorgeous. The true posterior mean is just $x_t$, rescaled by $1/\sqrt{\alpha_t}$, with a correction proportional to the **noise $\boldsymbol\epsilon$ that was added to make $x_t$**. So if the network could predict $\boldsymbol\epsilon$ from $x_t$, it could reconstruct the posterior mean exactly. That motivates parameterizing the model's mean in the *same form*, replacing the true $\boldsymbol\epsilon$ with a network prediction $\boldsymbol\epsilon_\theta(x_t, t)$:

$$
\boldsymbol\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol\epsilon_\theta(x_t, t)\right).
$$

Now subtract: $\tilde{\boldsymbol\mu}_t - \boldsymbol\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\cdot\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\left(\boldsymbol\epsilon_\theta - \boldsymbol\epsilon\right)$. Plug into $\mathcal{L}_{t-1}$ and collect the scalar coefficients in front of the squared norm:

$$
\mathcal{L}_{t-1} = \mathbb{E}_{x_0, \boldsymbol\epsilon}\!\left[\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}\left\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta\!\left(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\ t\right)\right\|^2\right].
$$

We have arrived. The variational bound, term by term, is now an expectation of the squared error between the **true noise $\boldsymbol\epsilon$** and the **network's predicted noise $\boldsymbol\epsilon_\theta$**, with $x_t$ assembled from $x_0$ and $\boldsymbol\epsilon$ by the closed-form marginal. The network's job is the simplest supervised task imaginable: *given a noisy image and a timestep, predict the noise.* Everything variational has dissolved into noise regression.

![A matrix comparing three equivalent prediction targets — predicting epsilon, predicting x_0, and predicting the posterior mean — showing what the network outputs, how each recovers the others, and the implicit loss weighting each one carries.](/imgs/blogs/the-math-of-ddpm-4.png)

Figure 4 makes an important point: $\boldsymbol\epsilon$-prediction is not the *only* legal parameterization — it is one of three algebraically equivalent choices. You could predict $x_0$ directly ($x_0$-prediction), or predict $\boldsymbol\mu_\theta$ directly (the raw KL form), and they all share the same optimum because they are linear reparameterizations of one another. What differs is the *implicit loss weighting* each one imposes across timesteps, and that weighting is exactly what affects which steps the network prioritizes. We will see in Section 7 that $\boldsymbol\epsilon$-prediction plus a flat weight is a quietly excellent combination. (The broader parameterization landscape — $v$-prediction, the SNR-weighted choices — is the subject of [noise schedules and the parameterization zoo](/blog/machine-learning/image-generation/diffusion-from-first-principles); here we establish that $\boldsymbol\epsilon$-pred is the original and why it works.)

The conversion formulas are worth writing down explicitly, because in real code you constantly move between these three views — a scheduler might predict $\boldsymbol\epsilon$ but a guidance step might want $x_0$, and an editing method might want the mean. Given any one, the marginal $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$ ties them together:

$$
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon_\theta}{\sqrt{\bar\alpha_t}}, \qquad \boldsymbol\epsilon_\theta = \frac{x_t - \sqrt{\bar\alpha_t}\,\hat{x}_0}{\sqrt{1-\bar\alpha_t}}, \qquad \boldsymbol\mu_\theta = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t + \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\hat{x}_0.
$$

The first formula — recover $\hat{x}_0$ from predicted noise — is especially important in practice: it is the "predicted clean image" that tools like DDIM, classifier-free guidance, and most editing pipelines actually manipulate, because operating on $\hat{x}_0$ is more interpretable than operating on $\boldsymbol\epsilon$. The `diffusers` schedulers expose exactly this via their `prediction_type` flag (`"epsilon"`, `"sample"` for $x_0$, or `"v_prediction"`), and internally they apply these very conversions. So when you read scheduler source and see a line reconstructing `pred_original_sample`, that is this first boxed formula, no more and no less. One optimum, three coordinate systems, and a handful of lines of algebra to move between them — that is the entire parameterization story in code.

## 7. Dropping the weight: $\mathcal{L}_\text{simple}$ and why it helps

Look at that coefficient in front of the squared norm: $\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}$. It is a complicated, $t$-dependent weight. The single most consequential empirical finding in the DDPM paper is that you should **throw it away.** Ho et al. define the simplified objective by setting that weight to 1 for all $t$:

$$
\boxed{\ \mathcal{L}_\text{simple} = \mathbb{E}_{t\sim\mathcal U(1,T),\ x_0,\ \boldsymbol\epsilon}\!\left[\left\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta\!\left(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\ t\right)\right\|^2\right]. \ }
$$

Sample a timestep uniformly, sample a clean image, sample noise, form $x_t$, predict the noise, take the MSE. That is the entire training objective. And the punchline of this whole post: **this is not the variational bound. It is a *reweighting* of it — and the reweighting makes samples better, not worse.** Let us understand why, because "we dropped a term and it improved" should make any careful person suspicious.

![A before-and-after figure contrasting the exact variational loss L_vlb, whose per-timestep weight explodes at small t and yields high-variance gradients, against the flat-weight L_simple, which upweights the high-noise steps that matter for perceptual content and lowers gradient variance.](/imgs/blogs/the-math-of-ddpm-5.png)

Figure 5 lays out the mechanism. Examine the discarded weight $\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}$ as a function of $t$. At **small $t$** (low noise, $\bar\alpha_t \approx 1$, $1-\bar\alpha_t$ tiny), the denominator is tiny so the weight is **huge** — the variational bound puts enormous emphasis on the easy, near-clean steps where the network only has to remove a whisper of noise. At **large $t$** (high noise), the weight is small, so the bound *under*-emphasizes the hard steps where the network must hallucinate coarse structure out of near-pure noise. That is backwards for *sample quality*. The high-noise steps are where the network decides global composition — is this a cat or a dog, where is the horizon — and those are perceptually the steps that matter most. The low-noise steps are near-trivial denoising that contributes little to perceived quality (though a lot to *likelihood*, which is why this trade exists).

So dropping the weight **rebalances training toward the harder, more perceptually-important high-noise steps.** The flat weight is, relatively, an up-weighting of large $t$. It also has a pure optimization benefit: the exploding small-$t$ weight makes the gradient estimator high-variance (a few easy samples dominate the batch), and flattening it tames that variance. The cost is real and worth stating plainly: $\mathcal{L}_\text{simple}$ is no longer a bound on the log-likelihood, so DDPM trained this way reports *worse* NLL than the full-bound version. **It trades likelihood for sample quality** — exactly the kind of honest trade this series insists on naming. For a generative model whose job is to produce good-looking images, that is the right side of the trade, and the FID numbers prove it.

This connects to a general principle in the [parameterization and schedule literature](/blog/machine-learning/image-generation/diffusion-from-first-principles): the "right" loss weighting across timesteps is a *design knob*, and the noise-prediction-with-flat-weight choice is equivalent to a specific, empirically excellent weighting in $x_0$-space. Min-SNR weighting, introduced later, is essentially a principled interpolation between the $\mathcal{L}_\text{vlb}$ weight and the flat one — capping the weight so neither the easy nor the hard steps dominate. But the DDPM lesson stands: the flat $\boldsymbol\epsilon$-prediction loss is a remarkably strong default, and it is *why* the loss looks like a one-liner.

#### Worked example: the FID payoff on CIFAR-10

On unconditional CIFAR-10 (32×32), Ho et al. report that the model trained with $\mathcal{L}_\text{simple}$ reaches **FID 3.17**, the best reported at the time and competitive with the strongest GANs of 2020 — while the model trained with the *exact* weighted bound $\mathcal{L}_\text{vlb}$ produces noticeably worse samples (higher FID) despite achieving better codelength/NLL. Same network, same data, same schedule; the only difference is the weighting term, and removing it is the difference between state-of-the-art samples and mediocre ones. That is as clean a demonstration as you will find that *the objective's weighting, not just its form, decides the model you get.* (FID here is computed against the standard 50k CIFAR-10 reference statistics with 50k generated samples — always state your sample size and reference set, because FID is meaningless without them.)

The training step is now trivial to write, and it matches the boxed $\mathcal{L}_\text{simple}$ line for line:

```python
import torch.nn.functional as F

def ddpm_loss(model, x0, alpha_bars, T):
    """L_simple: predict the noise added by the closed-form forward process."""
    b = x0.shape[0]
    t = torch.randint(0, T, (b,), device=x0.device)      # t ~ U(1, T)
    x_t, noise = q_sample(x0, t, alpha_bars)              # closed-form forward
    pred = model(x_t, t)                                  # eps_theta(x_t, t)
    return F.mse_loss(pred, noise)                        # ||eps - eps_theta||^2

# Training loop sketch
# for x0 in loader:
#     loss = ddpm_loss(model, x0.to(device), alpha_bars, T)
#     loss.backward(); opt.step(); opt.zero_grad()
#     ema.update(model)   # EMA of weights is standard and matters for sample quality
```

Two practitioner notes the equations do not tell you but a 2am debugging session will. First, **normalize $x_0$ to roughly $[-1, 1]$** before noising — the schedule assumes unit-ish variance, and feeding $[0,1]$ data shifts the SNR and quietly hurts everything. Second, **keep an EMA (exponential moving average) of the weights** for sampling; Ho et al. and essentially every diffusion model since use EMA, and the difference in sample quality between the raw and EMA weights is large and free. Neither is in the loss; both are non-negotiable in practice.

There is a quiet elegance here worth pausing on: the entire training procedure touches the forward chain *only through `q_sample`*. We never simulate the reverse process during training, never call the network 1000 times, never even materialize $x_{t-1}$. Every gradient step is one `q_sample`, one network forward, one MSE — $O(1)$ in the chain length $T$. That is what makes diffusion training tractable at scale, and it is a direct dividend of the closed-form marginal from Section 2: because we can jump to any noise level in one shot, training cost is decoupled from the number of steps. You could set $T=4000$ tomorrow and your per-step training cost would not change at all (only the schedule resolution would). The expensive 1000-step loop lives *entirely* on the sampling side, which is exactly the right place for the cost to be, because sampling is where the fast-sampler research then attacks it. Training is cheap and parallel; sampling is sequential and is where the trilemma's speed tax is paid. Keeping those two facts separate in your head is most of what it takes to reason clearly about diffusion performance.

## 8. The sampling algorithm: putting $\boldsymbol\epsilon_\theta$ to work

Training gave us $\boldsymbol\epsilon_\theta$. Sampling is just the reverse process run with our parameterized mean. Start from $x_T \sim \mathcal{N}(\mathbf 0, \mathbf I)$ and, for $t = T, T-1, \dots, 1$, sample $x_{t-1} \sim p_\theta(x_{t-1}\mid x_t) = \mathcal{N}(\boldsymbol\mu_\theta(x_t,t),\ \sigma_t^2\mathbf I)$. Using the reparameterized mean:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol\epsilon_\theta(x_t, t)\right) + \sigma_t\, \mathbf z, \qquad \mathbf z \sim \mathcal{N}(\mathbf 0, \mathbf I)\ \text{(and}\ \mathbf z = \mathbf 0\ \text{at}\ t=1).
$$

![A branching graph showing the training loop, where a random timestep and noise form x_t and feed the U-Net to produce the L_simple regression target, and the sampling loop, where the same trained network is reused inside an ancestral sampler that walks pure noise back to a generated image.](/imgs/blogs/the-math-of-ddpm-8.png)

Figure 8 shows the symmetry: one network $\boldsymbol\epsilon_\theta$, two loops. The training loop fits it by random-timestep regression; the sampling loop reuses it inside the ancestral sampler. Here is the sampler, matching the equation exactly:

```python
@torch.no_grad()
def ddpm_sample(model, shape, betas, alphas, alpha_bars, device):
    """Ancestral DDPM sampling: x_T -> ... -> x_0, one network call per step."""
    T = betas.shape[0]
    x = torch.randn(shape, device=device)                # x_T ~ N(0, I)
    for i in reversed(range(T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        eps = model(x, t)                                # eps_theta(x_t, t)
        beta_t, alpha_t, abar_t = betas[i], alphas[i], alpha_bars[i]
        # posterior mean in noise-prediction form
        mean = (x - beta_t / torch.sqrt(1.0 - abar_t) * eps) / torch.sqrt(alpha_t)
        if i > 0:
            sigma_t = torch.sqrt(beta_t)                 # sigma_t^2 = beta_t choice
            x = mean + sigma_t * torch.randn_like(x)
        else:
            x = mean                                     # no noise on the last step
    return x  # x_0, in [-1, 1]; rescale to [0, 1] for display
```

That `for i in reversed(range(T))` with `T = 1000` is the trilemma's bill coming due: **1000 sequential network calls per image.** On a single A100, a 256×256 model at 1000 steps is on the order of tens of seconds per image — unusably slow for anything interactive. This is *the* motivation for the entire fast-sampling track: DDIM reinterprets this same trained $\boldsymbol\epsilon_\theta$ under a non-Markovian forward process to sample deterministically in 50 steps, and higher-order solvers push it to 20 — all *without retraining*, because they reuse this exact network. The math we just derived is what makes that reuse legal: $\boldsymbol\epsilon_\theta$ is an estimate of the score (up to a known scaling), and any sampler that integrates the same underlying ODE/SDE can drive it. See [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) for where this thread goes.

#### The $\sigma_t$ choice and the stochasticity knob

Notice we set $\sigma_t^2 = \beta_t$ in the sampler, but the posterior variance we *derived* was $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$. Ho et al. found both upper and lower bounds on the reverse variance — $\beta_t$ and $\tilde\beta_t$ respectively — give comparable sample quality, so either is fine. The choice matters more than it looks: $\sigma_t$ is the **stochasticity knob** of the sampler. Set it to zero entirely and you get a *deterministic* sampler — that is the DDIM limit, the bridge to the probability-flow ODE. Keep it at $\beta_t$ and you get fully stochastic ancestral sampling. The whole family of samplers lives on this dial between "inject fresh noise each step" (more diverse, more robust, slower to converge) and "deterministic" (fewer steps, exactly reproducible, interpolatable in latent space). The fact that one trained network supports the entire dial is, again, a gift of the noise-prediction parameterization.

## 9. Why $\boldsymbol\epsilon_\theta$ is secretly the score function

There is a deep identity hiding in the noise-prediction objective, and it is the single most important bridge out of DDPM into the rest of modern diffusion. The claim: **predicting the noise is, up to a fixed known scaling, the same as estimating the score** $\nabla_{x_t} \log q(x_t)$ — the gradient of the log-density of the noised data. Once you see this, the entire score-based / SDE framework opens up, and the reason "one trained network drives every sampler" stops being a happy accident and becomes a theorem.

Here is the derivation, and it is short. The marginal $q(x_t\mid x_0) = \mathcal N(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)\mathbf I)$ has a log-density whose gradient with respect to $x_t$ is, for any Gaussian $\mathcal N(\boldsymbol\mu,\sigma^2\mathbf I)$, the familiar $-\frac{x-\boldsymbol\mu}{\sigma^2}$:

$$
\nabla_{x_t} \log q(x_t\mid x_0) = -\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t}.
$$

Now recall the reparameterization $x_t - \sqrt{\bar\alpha_t}\,x_0 = \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$. Substitute:

$$
\nabla_{x_t} \log q(x_t\mid x_0) = -\frac{\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon}{1-\bar\alpha_t} = -\frac{\boldsymbol\epsilon}{\sqrt{1-\bar\alpha_t}}.
$$

The score *is* the noise, negated and divided by $\sqrt{1-\bar\alpha_t}$. So the network trained to predict $\boldsymbol\epsilon$ is implicitly learning the (marginal) score:

$$
\boxed{\ \boldsymbol\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar\alpha_t}\ \nabla_{x_t} \log q(x_t). \ }
$$

(The step from the conditional score $\nabla\log q(x_t\mid x_0)$ to the marginal score $\nabla\log q(x_t)$ is the denoising-score-matching identity: minimizing the expected squared error to the *conditional* score, averaged over $x_0$, has the same minimizer as matching the *marginal* score. The expectation does the marginalization for you — that is the whole content of Vincent's 2011 result, and it is why $\mathcal{L}_\text{simple}$ recovers the true marginal score despite only ever seeing conditional targets.)

Why does this matter so much? Because the reverse-time SDE that undoes diffusion is written entirely in terms of the score:

$$
dx = \left[f(x,t) - g(t)^2 \nabla_x \log q_t(x)\right]dt + g(t)\,d\bar{\mathbf w},
$$

and its deterministic twin, the **probability-flow ODE**, is the same drift with the noise term dropped. Plug our trained $\boldsymbol\epsilon_\theta$ in for the score and you can integrate *either* of these with *any* numerical solver — Euler, Heun, a multistep DPM-Solver, an adaptive RK method. The 1000-step ancestral sampler we wrote in Section 8 is just one particular discretization (the Euler-Maruyama scheme on the SDE). DDIM is the probability-flow ODE. DPM-Solver is a high-order exponential integrator. **They are all integrating the same vector field that $\boldsymbol\epsilon_\theta$ defines.** This is the rigorous reason a model trained with $\mathcal{L}_\text{simple}$ can be sampled in 1000, 50, or 20 steps without retraining — the network learned a *field*, not a *sampler*, and the field is the score. The full SDE/ODE machinery is the subject of [score-based models and the SDE view](/blog/machine-learning/image-generation/diffusion-from-first-principles); this identity is the door into it, and it falls out of the DDPM algebra for free.

#### Worked example: reading the score off a trained model

Suppose you have a trained pixel DDPM and you want to visualize the score at $t=500$ on the linear schedule, where $\bar\alpha_{500}\approx 0.13$ so $\sqrt{1-\bar\alpha_{500}}\approx 0.93$. Take a noisy image $x_{500}$, run $\boldsymbol\epsilon_\theta(x_{500}, 500)$, and the score estimate is simply $-\boldsymbol\epsilon_\theta / 0.93$. A single Langevin step "up the score" — $x \leftarrow x + \tfrac{\eta}{2}\,(-\boldsymbol\epsilon_\theta/0.93) + \sqrt{\eta}\,\mathbf z$ for a small $\eta$ — nudges the noisy image toward higher density, i.e. toward something more image-like at that noise level. That is exactly what NCSN (Song & Ermon, 2019) did *before* DDPM, from the score side; the identity above is why the two literatures turned out to be the same algorithm in different clothes. If you ever need to debug whether your $\boldsymbol\epsilon_\theta$ is sane, this is a one-line sanity check: the score it implies should point "uphill" toward cleaner-looking images.

## 10. The schedule: $\beta_t$ linear vs cosine, and what SNR decides

The variance schedule $\{\beta_t\}$ is the one piece of the forward process you, the engineer, choose. Section 2 showed it determines $\bar\alpha_t$ and therefore the SNR at every step. It is tempting to treat it as a minor hyperparameter; it is not. A bad schedule wastes a large fraction of your 1000 steps doing nothing useful.

![A before-and-after figure contrasting the linear beta schedule, whose cumulative signal collapses to near zero in the last quarter of the chain leaving low-SNR wasted steps, against the cosine schedule, whose signal-to-noise ratio decays smoothly so the late steps still carry learnable structure.](/imgs/blogs/the-math-of-ddpm-7.png)

The original DDPM uses the **linear** schedule: $\beta_t$ increases linearly from $10^{-4}$ to $0.02$. Figure 7 shows its flaw, identified by Nichol & Dhariwal (Improved DDPM, 2021): with the linear schedule, $\bar\alpha_t$ crashes toward zero *too fast at the end of the chain*. By the last ~25% of steps, $x_t$ is already indistinguishable from pure noise — SNR is effectively zero — so those steps contribute almost nothing to training or sampling. You are paying for 250 network evaluations that are doing busywork. Worse, the abrupt drop near the end means the network never learns a smooth denoising curve over the high-noise regime.

The **cosine** schedule fixes this by defining $\bar\alpha_t$ directly (then back-deriving $\beta_t$) so that SNR decays smoothly across the whole chain:

$$
\bar\alpha_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^2\!\left(\frac{t/T + s}{1 + s}\cdot\frac{\pi}{2}\right),
$$

with a small offset $s = 0.008$ to prevent $\beta_t$ from being too tiny near $t=0$. The cosine shape holds SNR meaningfully higher through the middle and late steps, so every step carries learnable signal. Nichol & Dhariwal report this improves log-likelihood and makes fewer-step sampling viable. Here is the cosine schedule in code, matching the formula:

```python
import math

def cosine_alpha_bars(T=1000, s=0.008, device="cpu"):
    """Nichol & Dhariwal cosine schedule: define alpha_bar directly."""
    steps = torch.arange(T + 1, device=device, dtype=torch.float64)
    f = torch.cos(((steps / T + s) / (1 + s)) * math.pi * 0.5) ** 2
    alpha_bars = f / f[0]                       # normalize so alpha_bar_0 = 1
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    betas = torch.clamp(betas, max=0.999)       # numerical guard on the last steps
    return betas.float(), alpha_bars[1:].float()
```

#### Worked example: SNR at the midpoint, linear vs cosine

Let us quantify the gap. The SNR at step $t$ is $\bar\alpha_t/(1-\bar\alpha_t)$. Consider the *halfway* point, $t = 500$ of $T = 1000$. On the **linear** schedule, the cumulative product has already crashed: $\bar\alpha_{500}\approx 0.13$, so SNR $\approx 0.13/0.87 \approx 0.15$ — at the literal midpoint of the chain, signal is already roughly one-seventh of noise. On the **cosine** schedule, $\bar\alpha_{500} = \cos^2(\frac{0.5+s}{1+s}\cdot\frac{\pi}{2})/f(0) \approx \cos^2(0.5\cdot\frac{\pi}{2}) = \cos^2(\pi/4) = 0.5$, so SNR $\approx 0.5/0.5 = 1.0$ — signal and noise in balance, nearly *seven times* the linear SNR at the same step. The linear schedule has spent half its steps getting to near-noise; the cosine schedule still has meaningful signal at the midpoint and reserves the genuine noise-destruction for the final stretch. That is the entire content of figure 7 in one ratio: $0.15$ vs $1.0$ at the midpoint. When people say "the cosine schedule uses its steps better," this is the number they mean — and it is why a cosine model tolerates fewer sampling steps before quality falls off, because no large block of its steps is wasted on already-destroyed signal.

The schedule choice and the SNR curve it induces are the seam between this post and the [parameterization-zoo post](/blog/machine-learning/image-generation/diffusion-from-first-principles), where v-prediction, zero-terminal-SNR, and min-SNR weighting all turn out to be different ways of fixing what the SNR curve gets wrong. The headline for now: **the schedule is not free, and the cosine schedule is a strict improvement for most pixel-space setups.** (In latent space — Stable Diffusion — the optimal schedule shifts again, because the VAE latents have different statistics than pixels; another reason the choice deserves real attention rather than a copied default.)

## 11. Mapping every loss term to what it trains

Let us consolidate the three-part decomposition from Section 3 against the simplified objective, because understanding *which term does what* is the difference between treating $\mathcal{L}_\text{simple}$ as a black box and knowing exactly what you are optimizing.

![A matrix mapping each variational-bound term — the prior term L_T, the denoising terms L_{t-1}, and the decode term L_0 — to its formula, what it trains, and how it appears in the simplified objective.](/imgs/blogs/the-math-of-ddpm-6.png)

Figure 6 is the reference table; here it is in prose and then as a markdown table you can keep.

- **$\mathcal{L}_T$, the prior term.** Formula $D_\text{KL}(q(x_T\mid x_0)\,\|\,\mathcal N(\mathbf 0,\mathbf I))$. Trains *nothing* — no parameters appear. With a sane schedule it is also numerically near zero. Dropped entirely from $\mathcal{L}_\text{simple}$. Its only practical role is a sanity check: if it is *not* small, your terminal SNR is too high (the chain did not fully noise the image), which is the "zero-terminal-SNR" bug that plagued early Stable Diffusion and produced washed-out, never-fully-dark images.
- **$\mathcal{L}_{t-1}$, the denoising terms ($1 < t \le T$).** Formula $D_\text{KL}(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta)$, which we reduced to $\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta\|^2$ (times a weight). Trains the **entire denoiser** — every parameter of $\boldsymbol\epsilon_\theta$. This is essentially *all* of $\mathcal{L}_\text{simple}$: sampling $t \sim \mathcal U(1,T)$ is a Monte-Carlo estimate of this sum over $t$.
- **$\mathcal{L}_0$, the decode term.** Formula $-\log p_\theta(x_0\mid x_1)$ — the final continuous-to-discrete pixel decode. In the simplified objective it is absorbed into the $t=1$ case of the same noise-prediction loss (Ho et al. show the noise-prediction loss at $t=1$ is a reasonable surrogate for the discrete decoder NLL).

| Term | Formula | Trains | Parameters? | In $\mathcal{L}_\text{simple}$ |
|---|---|---|---|---|
| $\mathcal{L}_T$ | $D_\text{KL}(q(x_T\mid x_0)\,\|\,\mathcal N(0,I))$ | nothing (prior match) | none | dropped ($\approx 0$) |
| $\mathcal{L}_{t-1}$ | $D_\text{KL}(q\text{-posterior}\,\|\,p_\theta) = \|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta\|^2$ | the denoiser $\boldsymbol\epsilon_\theta$ | all of $\theta$ | the whole loss |
| $\mathcal{L}_0$ | $-\log p_\theta(x_0\mid x_1)$ | final pixel decode | shared with $\boldsymbol\epsilon_\theta$ | folded into $t=1$ |

The table is the answer to "what am I actually training?" — a single noise-prediction network, supervised by the denoising terms, with the prior term free and the decode term riding along at $t=1$. If you internalize one figure from this post for practical work, it is figure 6.

One subtlety the table compresses deserves a paragraph, because it is where a lot of beginners get confused about what a single gradient step is doing. The denoising loss is a *sum over all $T$ timesteps*: $\sum_{t} \mathbb{E}[\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(x_t, t)\|^2]$. We do not evaluate all 1000 terms per image — that would be absurdly expensive. Instead we use the standard Monte-Carlo trick: replace the sum over $t$ with a single *uniformly-sampled* $t$ per image, and the expectation over that sampling recovers the sum (up to the constant factor $T$, which is absorbed into the learning rate). So each minibatch is a noisy, unbiased estimate of the full denoising loss, where every image in the batch is corrupted to a *different, random noise level*. This is why a diffusion training batch is so heterogeneous: one image might be at $t=12$ (nearly clean), the next at $t=931$ (nearly pure noise), and the network must handle the entire range with shared weights. The timestep embedding $t$ fed into $\boldsymbol\epsilon_\theta$ is what lets a single network specialize its behavior across that range — at low $t$ it learns fine-detail denoising, at high $t$ it learns coarse structure synthesis, and the embedding tells it which regime it is in. That single mechanism — one network, conditioned on $t$, trained on random noise levels — is the entire reason diffusion needs only one model instead of 1000 per-step models. The variational bound told us we needed a term per step; the timestep conditioning lets one network *be* all those terms.

## 12. A numerical check: closed form vs iterated sampling

A derivation you cannot test is a derivation you do not trust. The central claim of Section 2 is that the closed-form marginal $q(x_t\mid x_0)$ — one Gaussian — produces the *same distribution* as iterating the chain $q(x_t\mid x_{t-1})$ $t$ times. Let us assert it numerically: both should give $x_t$ with the same mean ($\sqrt{\bar\alpha_t}\,x_0$) and the same variance ($1-\bar\alpha_t$ around that mean). We check it in expectation over many samples.

```python
import torch

torch.manual_seed(0)
T = 1000
betas, alphas, alpha_bars = make_schedule(T)

x0 = torch.full((50000, 1), 0.7)   # a fixed "image": scalar value 0.7, many samples
t_idx = 300                         # check the marginal at t = 300

# --- Method A: closed-form single jump ---
a_bar = alpha_bars[t_idx]
xt_closed = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * torch.randn_like(x0)

# --- Method B: iterate the Markov chain t+1 times ---
xt_iter = x0.clone()
for s in range(t_idx + 1):
    xt_iter = torch.sqrt(alphas[s]) * xt_iter + torch.sqrt(betas[s]) * torch.randn_like(xt_iter)

# The theory predicts mean = sqrt(alpha_bar_t) * x0, var = 1 - alpha_bar_t
mean_pred = (torch.sqrt(a_bar) * 0.7).item()
var_pred  = (1 - a_bar).item()

print(f"closed: mean={xt_closed.mean():.4f} var={xt_closed.var():.4f}")
print(f"iter:   mean={xt_iter.mean():.4f}  var={xt_iter.var():.4f}")
print(f"theory: mean={mean_pred:.4f} var={var_pred:.4f}")

# Assert all three agree to Monte-Carlo tolerance (~1e-2 with 50k samples)
assert abs(xt_closed.mean() - xt_iter.mean()) < 2e-2
assert abs(xt_closed.var()  - xt_iter.var())  < 2e-2
assert abs(xt_closed.mean() - mean_pred)      < 2e-2
assert abs(xt_closed.var()  - var_pred)       < 2e-2
print("PASS: closed-form marginal matches iterated sampling in mean and variance")
```

Run it and the three rows agree to two decimals — the closed form, the 301-step iteration, and the theoretical $(\sqrt{\bar\alpha_t}\cdot 0.7,\ 1-\bar\alpha_t)$ all coincide. That is the Gaussian-composition lemma from Section 2, verified. This little test is also a great regression guard: if you ever refactor your schedule code and this assertion breaks, you have a bug in `alpha_bars` (usually an off-by-one in the cumulative product or a missing $\bar\alpha_0 = 1$) *before* it silently corrupts a week of training. I have caught exactly that bug twice; the check costs a millisecond and pays for itself.

## 13. Case studies: the numbers behind the derivation

The math is only worth trusting if the models it produced actually work. Three real, citable data points anchor the derivation to reality.

**DDPM on CIFAR-10 (Ho et al., 2020).** The original paper, unconditional CIFAR-10 at 32×32, $T=1000$, linear schedule, $\boldsymbol\epsilon$-prediction with $\mathcal{L}_\text{simple}$: **FID 3.17**, Inception Score 9.46 — the best FID reported at the time, beating the strongest GANs of that moment. The same architecture trained on $\mathcal{L}_\text{vlb}$ achieves better NLL (codelength) but visibly worse samples, which is the empirical core of the "drop the weight" lesson. This single result is what launched the diffusion era.

**Improved DDPM (Nichol & Dhariwal, 2021).** Introduces the cosine schedule and a *learned* reverse variance (interpolating between $\beta_t$ and $\tilde\beta_t$ in log-space), recovering the likelihood that $\mathcal{L}_\text{simple}$ throws away. Result: competitive NLL *and* good samples, plus the cosine schedule enabling sampling in far fewer steps. This is the post that says "you can have the likelihood back if you learn the variance" — the variance we fixed to a constant in Section 4 becomes a small extra network output.

**Scaling to the real world: Stable Diffusion.** The exact $\mathcal{L}_\text{simple}$ derived here is the training objective of Stable Diffusion 1.5 and 2 — only it runs in the **VAE latent space** rather than pixel space, which is the entire point of latent diffusion: the [VAE](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) compresses a 512×512×3 image to a 64×64×4 latent (a 48× reduction), and the diffusion math we derived runs unchanged on those latents. The noise-prediction loss, the closed-form forward, the posterior — every equation in this post is what trains the U-Net inside SD. The frontier moved to v-prediction, flow matching, and transformer backbones, but $\mathcal{L}_\text{simple}$ is the foundation every one of them is a refinement *of*. When you read "SD3 uses flow matching" or "FLUX is a rectified-flow transformer," they are replacing the *forward process and parameterization* derived here, not the variational-inference skeleton — that skeleton is permanent.

#### Worked example: cost of the 1000-step bill

Put a number on the trilemma's speed cost. A 256×256 pixel-space DDPM at $T=1000$ needs 1000 sequential forward passes through the U-Net per image. On an A100 80GB, a single U-Net forward at that resolution is on the order of ~25 ms, so one image is roughly **25 seconds** — and batching helps throughput but not the 25-second *latency* of a single image. Switch the *same trained network* to DDIM at 50 steps and you are at ~1.25 s; a DPM-Solver++ at 20 steps lands near ~0.5 s. Nothing about the model changed — only the sampler integrating the ODE that $\boldsymbol\epsilon_\theta$ defines. That 50× latency swing, for free, off one trained network, is the practical reason the noise-prediction derivation matters: it produced a *reusable* score estimate, not a sampler-locked one. (These are order-of-magnitude figures on a named GPU to make the trade concrete; exact latency depends on resolution, precision, and attention kernel.)

Here is the objective-choice trade-off as a table you can keep next to your training config:

| Objective | What it optimizes | Sample quality (FID) | Likelihood (NLL) | When to use |
|---|---|---|---|---|
| $\mathcal{L}_\text{vlb}$ (exact bound) | true ELBO weighting | worse (over-weights easy steps) | best | density estimation, compression |
| $\mathcal{L}_\text{simple}$ (flat weight) | perceptually-balanced MSE | best | worse (not a bound) | image generation (the default) |
| Learned-variance hybrid | $\mathcal{L}_\text{simple} + \lambda\,\mathcal{L}_\text{vlb}$ | best | near-best | when you need both samples and NLL |
| Min-SNR weighting | capped per-step weight | best, faster convergence | good | unstable / slow training runs |

### 13.1 Stress test: where the derivation's assumptions break

A derivation is only as trustworthy as its failure modes are understood. Let us deliberately break each assumption and watch what happens — this is the part that turns "I followed the algebra" into "I can debug the model at 2am."

**Stress the step size: what if $\beta_t$ is large?** The entire Gaussian-reverse argument (Section 4) hinges on small steps. Crank $\beta_t$ up — say, use $T=10$ instead of 1000, forcing each step to remove a huge amount of noise — and the true reverse posterior $q(x_{t-1}\mid x_t)$ becomes genuinely multimodal: given a very-noisy $x_t$, wildly different clean images are plausible, and a single Gaussian $p_\theta$ cannot represent that. The symptom is **blurry, averaged samples** — the network outputs the *mean* of all plausible reconstructions because that is the best a unimodal Gaussian can do. This is precisely why naive few-step DDPM fails and why the fast-sampling literature had to invent cleverer integrators rather than just using fewer Markov steps. The math told us this would happen; the gray mush confirms it.

**Stress the terminal SNR: what if $\bar\alpha_T \not\approx 0$?** If the schedule does not drive $\bar\alpha_T$ close to zero, then $x_T$ retains a faint ghost of $x_0$, the prior term $\mathcal{L}_T$ is *not* negligible, and there is a train/test mismatch: at sampling time you start from pure $\mathcal N(\mathbf 0,\mathbf I)$, but the model was trained expecting a slightly-signal-bearing $x_T$. The famous symptom in early Stable Diffusion was an **inability to generate very dark or very bright images** — the model could never reach pure black because it was never trained on a truly mean-zero terminal state. The fix, "zero-terminal-SNR," forces $\bar\alpha_T = 0$ exactly. The lesson: $\mathcal{L}_T$ being "free" only holds *if the schedule earns it*.

**Stress the data normalization: what if $x_0 \in [0,1]$?** The schedule's variance-preserving design assumes unit-variance data. Feed it $[0,1]$ images (variance ~0.08, mean ~0.5) and the effective SNR at every step is shifted — the network sees a different noise regime than the one the loss weighting was tuned for, and you get washed-out, low-contrast samples. The one-line fix is `x0 = 2 * x0 - 1` to map to $[-1, 1]$. This bug produces *plausible-looking* training curves and *subtly wrong* samples, which makes it the worst kind: it does not crash, it just quietly degrades.

**Stress the timestep sampling: what if $t$ is sampled non-uniformly?** $\mathcal{L}_\text{simple}$ samples $t \sim \mathcal U(1,T)$, which Monte-Carlo estimates the uniform sum over the denoising terms. Sample $t$ with a different distribution and you have silently re-introduced a weighting — sometimes deliberately (importance-sampling the high-loss timesteps speeds convergence), sometimes by accident (an off-by-one that never samples $t=T$). If your samples are great except for coarse global structure, suspect that the high-$t$ steps are under-trained because your $t$-sampler is biased low.

## 14. When to reach for the full bound (and when $\mathcal{L}_\text{simple}$ is all you need)

A decisive section, because the derivation hands you a choice and you should know how to make it.

**Use $\mathcal{L}_\text{simple}$ (the default, ~95% of the time).** If your goal is good-looking samples — text-to-image, image editing, any generative product — train with the flat-weight noise-prediction loss. It gives the best FID, the lowest gradient variance, and the simplest code. Every shipped diffusion model uses it (or a close relative like v-prediction with a comparable weighting). Do not reach for the full bound to "be more correct"; you will get worse samples.

**Use the full $\mathcal{L}_\text{vlb}$ (or learned variance) when you genuinely need likelihood.** If you are doing density estimation, compression, anomaly detection, or reporting NLL/bits-per-dim for a paper, the weighting term matters and you want the bound — or, better, Nichol & Dhariwal's learned-variance hybrid, which recovers likelihood without tanking sample quality. This is a minority case, but a real one.

**Reach for a *reweighting between them* (min-SNR) when training is unstable or slow to converge.** If $\mathcal{L}_\text{simple}$ is giving you noisy gradients or the high-noise steps dominate to the point of instability, min-SNR weighting caps the loss weight per timestep and often speeds convergence measurably. It is the principled middle ground and a good first thing to try when a from-scratch training run is misbehaving.

**Do NOT** hand-tune the $\beta$ schedule before trying cosine — cosine is a strict improvement for pixel diffusion and a sane default for latent. **Do NOT** train without EMA and normalized $[-1,1]$ inputs — those two free wins dwarf most loss-function fiddling. And **do NOT** sample with 1000 DDPM steps in production — the same network does it in 20–50 with a modern solver, and the quality is indistinguishable at a fraction of the cost. The derivation gives you the objective; these are the guardrails that turn it into a working model.

## 15. Key takeaways

- **The forward process is fixed and has a closed form.** $q(x_t\mid x_0) = \mathcal N(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)\mathbf I)$ — sample any noise level in one shot, no chain iteration. This single equation makes training $O(1)$ per step.
- **The variational bound splits into three roles.** A free prior term $\mathcal{L}_T$ (no parameters), the denoising terms $\mathcal{L}_{t-1}$ (all the learning), and a decode term $\mathcal{L}_0$ (folds into $t=1$). Only the middle one trains the network.
- **The tractable posterior is the keystone.** Conditioning $q(x_{t-1}\mid x_t)$ on $x_0$ gives a closed-form Gaussian with mean $\tilde{\boldsymbol\mu}_t$ and variance $\tilde\beta_t$ — the target that turns an intractable likelihood into a KL between Gaussians.
- **$\boldsymbol\epsilon$-prediction is a reparameterization, not a new model.** Predicting noise, predicting $x_0$, and predicting the mean share one optimum; they differ only in implicit loss weighting. Noise-prediction is the original and a strong default.
- **Dropping the weight improves samples by trading likelihood.** $\mathcal{L}_\text{simple}$ is $\mathcal{L}_\text{vlb}$ reweighted toward the perceptually-important high-noise steps; it raises FID quality and lowers gradient variance at the cost of NLL.
- **The schedule is a real design choice.** The linear schedule wastes the last quarter of the chain at near-zero SNR; the cosine schedule decays SNR smoothly and is a strict improvement for pixel diffusion.
- **One trained network, many samplers.** Because $\boldsymbol\epsilon_\theta$ is a (scaled) score estimate, the same weights drive 1000-step ancestral DDPM, 50-step DDIM, or 20-step DPM-Solver — the $\sigma_t$ dial controls stochasticity. This is why fast sampling needs no retraining.
- **Test the closed form.** Assert that the one-shot marginal matches iterated sampling in mean and variance; it catches the off-by-one $\bar\alpha$ bug before it costs you a training run.

## 16. Further reading

- **Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models" (2020)** — the source of every equation in this post; read Section 3 alongside this derivation.
- **Sohl-Dickstein et al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (2015)** — the original diffusion-as-generative-model paper that DDPM simplified and scaled.
- **Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021)** — the cosine schedule, learned variance, and the likelihood-vs-quality trade made precise.
- **Song et al., "Score-Based Generative Modeling through SDEs" (2021)** — the unifying view that recasts $\boldsymbol\epsilon_\theta$ as a score and DDPM as a discretized SDE; the bridge to deterministic sampling.
- **Kingma et al., "Variational Diffusion Models" (2021)** — diffusion through the lens of SNR, with the cleanest statement of the loss-weighting design space.
- Within this series: [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) (the intuition this post formalizes), [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions) (the ELBO and FID), [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) (the ELBO and reparameterization, and the latent space SD diffuses in), and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
- 🤗 `diffusers` documentation — the `DDPMScheduler` and `DDIMScheduler` source is the cleanest reference implementation of everything derived here; read it next to your own `q_sample` and `posterior_mean_variance`.
