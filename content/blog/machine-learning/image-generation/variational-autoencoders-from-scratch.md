---
title: "Variational Autoencoders From Scratch: The Latent Space Diffusion Is Built On"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Derive the ELBO and the reparameterization trick, train a convolutional VAE in PyTorch, then load Stable Diffusion's AutoencoderKL to see the exact 48x compression that makes latent diffusion possible."
tags:
  [
    "image-generation",
    "diffusion-models",
    "vae",
    "variational-autoencoder",
    "vq-gan",
    "latent-diffusion",
    "generative-ai",
    "deep-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/variational-autoencoders-from-scratch-1.png"
---

Here is a number that should bother you. Stable Diffusion generates a 512x512 image, but the diffusion model at its heart never once looks at a 512x512 image. It works entirely inside a tensor of shape `4x64x64` — sixteen thousand numbers instead of the seven hundred eighty-six thousand a full RGB image carries. The diffusion U-Net denoises that little latent for twenty or thirty steps, and only at the very end does a separate network expand it back into pixels. That expansion-and-compression network is a **variational autoencoder**, and it is the most under-appreciated component in the entire image-generation stack. Everyone talks about the denoiser, the sampler, the guidance scale. Almost nobody talks about the VAE — and yet without it, latent diffusion does not exist, training costs explode by roughly an order of magnitude, and you cannot fit Stable Diffusion on a consumer GPU.

This post builds the VAE from the ground up and connects it, precisely, to that 48x compression number. We will start with the plain autoencoder you may already know, make the one conceptual leap that turns it *variational*, and derive — not assert — the loss function that falls out of that leap: the **evidence lower bound**, or ELBO. We will see why you cannot just sample a latent and backpropagate through it, and why the **reparameterization trick** `z = μ + σ⊙ε` is the small piece of algebra that makes the whole thing trainable. We will train a real convolutional VAE in PyTorch, watch it produce slightly blurry samples, understand *exactly why* those samples are blurry (it is not a bug — it is the Gaussian likelihood doing precisely what you asked it to), and then meet the two fixes the field invented: the **discrete** branch (VQ-VAE, VQ-GAN) that produces the tokens autoregressive image models consume, and the **perceptual-plus-adversarial** branch that finally makes reconstructions crisp.

![Diagram of the VAE forward path from an input image through the encoder to a mean and standard deviation, a reparameterized latent sample, and a decoder reconstruction.](/imgs/blogs/variational-autoencoders-from-scratch-1.png)

By the end you will be able to: derive the ELBO and explain every term to a skeptical reviewer; implement a VAE encoder, decoder, reparameterization function, and combined KL-plus-reconstruction loss from a blank file; diagnose and fix posterior collapse; load Stable Diffusion's `AutoencoderKL` from 🤗 `diffusers`, encode and decode a real image, and report the measured compression factor and VRAM; and — the punchline this whole post drives toward — explain why **latent diffusion is just diffusion run in a VAE's latent space**, and why that single architectural choice is what made text-to-image generation affordable enough to ship.

This post sits in the foundations track of the series. It is the third pillar of the four-family map laid out in [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard), and it leans directly on the likelihood-and-ELBO primer in [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions). Everything here is the substrate that [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) and [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) build on, and it is a load-bearing component of the [end-to-end image-generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack) we assemble at the end of the series.

## 1. From autoencoder to variational autoencoder: the one leap that matters

Start with the thing you probably already know. An **autoencoder** is a pair of neural networks. An encoder $f_\phi$ squeezes an input $x$ — say a 64x64 RGB image, so $x \in \mathbb{R}^{12288}$ — down to a small vector $z = f_\phi(x)$, the *code* or *latent*. A decoder $g_\theta$ tries to rebuild the original: $\hat{x} = g_\theta(z)$. You train both together to minimize reconstruction error, typically squared error $\|x - \hat{x}\|^2$. The bottleneck — $z$ being much smaller than $x$ — forces the network to learn a compressed representation that keeps whatever is most important for reconstruction and throws away the rest. This is just nonlinear, learned compression. It works fine. PCA is the linear special case.

Now ask the question that breaks it: **can you generate new images with a plain autoencoder?** The naive idea is to skip the encoder, pick a random $z$, and run it through the decoder. Try it and you get garbage. The reason is subtle and important. The encoder has learned to map your training images to *some* set of points in latent space, but you have no idea what that set looks like. It might be a thin curved sheet, a scatter of disconnected blobs, a region centered at $z = 1000$ with wild variance. There is no structure you can sample from. The decoder has only ever seen latents that the encoder produced, so it has no obligation to do anything sensible with a $z$ you invented. The latent space of a plain autoencoder is a filing cabinet with no index: every training image has a drawer, but the empty space between drawers decodes to nonsense.

The **variational autoencoder**, introduced by Kingma and Welling in 2013, fixes exactly this. The leap is to stop treating the encoder as producing a single point and start treating it as producing a *probability distribution*. Instead of $z = f_\phi(x)$, the encoder outputs the parameters of a distribution over $z$ — concretely, a mean vector $\mu_\phi(x)$ and a (log-)variance vector $\log\sigma^2_\phi(x)$, defining a Gaussian $q_\phi(z \mid x) = \mathcal{N}(z; \mu_\phi(x), \mathrm{diag}(\sigma^2_\phi(x)))$. We call $q_\phi(z\mid x)$ the **approximate posterior** or **inference distribution**. To get a latent we *sample* from it. And — the crucial second half — we add a term to the loss that forces all those per-image posteriors to collectively look like a fixed, simple **prior** $p(z) = \mathcal{N}(0, I)$, a standard multivariate normal.

That second constraint is the whole game. If every image's posterior is pulled toward $\mathcal{N}(0, I)$, then the *aggregate* of all latents — what the decoder actually sees during training — fills out a smooth, connected, unit-Gaussian-shaped cloud with no holes. Now sampling generates real images: draw $z \sim \mathcal{N}(0, I)$, decode, and because the decoder was trained on latents densely covering that exact region, it produces something coherent. The filing cabinet gets an index, and the index is the standard normal distribution. You have turned a compressor into a *generative model*.

So the VAE is two ideas welded together: **(1)** the encoder emits a distribution, not a point, and **(2)** a regularizer drags those distributions toward a known prior you can sample from. The price you pay is that you now need a loss function that balances "reconstruct the input well" against "keep the posterior close to the prior." That balance is not something we get to invent by taste. It falls out, exactly, of trying to do maximum likelihood on $p(x)$ — and deriving it is the next section.

## 2. The science: deriving the ELBO

We want a generative model $p_\theta(x)$ that assigns high probability to real images. The model is defined by the decoder plus the prior: sample $z \sim p(z) = \mathcal{N}(0, I)$, then sample $x \sim p_\theta(x \mid z)$, where $p_\theta(x \mid z)$ is the decoder's output distribution (a Gaussian or Bernoulli over pixels, parameterized by the decoder network). The marginal likelihood of a data point is

$$
p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz .
$$

This integral is the problem. It marginalizes over every possible latent $z \in \mathbb{R}^{128}$ that could have produced $x$. There is no closed form, and you cannot Monte-Carlo it efficiently, because for almost every $z$ drawn from the prior, $p_\theta(x \mid z)$ is essentially zero — random latents rarely decode to your specific image, so the estimator has astronomically high variance. We need a smarter way to get a gradient on $\log p_\theta(x)$.

Here is the trick that defines variational inference. Introduce the approximate posterior $q_\phi(z \mid x)$ — the encoder — and write $\log p_\theta(x)$ by multiplying and dividing inside an expectation. For *any* distribution $q_\phi(z\mid x)$ with the same support:

$$
\log p_\theta(x) = \log \int p_\theta(x, z)\, dz
= \log \int q_\phi(z\mid x)\, \frac{p_\theta(x, z)}{q_\phi(z\mid x)}\, dz
= \log\, \mathbb{E}_{q_\phi(z\mid x)}\!\left[ \frac{p_\theta(x, z)}{q_\phi(z\mid x)} \right].
$$

Now apply **Jensen's inequality**. The logarithm is concave, so $\log \mathbb{E}[Y] \ge \mathbb{E}[\log Y]$. Pushing the log inside the expectation can only decrease the value:

$$
\log p_\theta(x) \;\ge\; \mathbb{E}_{q_\phi(z\mid x)}\!\left[ \log \frac{p_\theta(x, z)}{q_\phi(z\mid x)} \right] \;=\; \mathcal{L}(\theta, \phi; x).
$$

That right-hand side is the **evidence lower bound**, the ELBO. The name is literal: $\log p_\theta(x)$ is called the *evidence*, and $\mathcal{L}$ is a *lower bound* on it. We cannot maximize the evidence directly because of that intractable integral, so we maximize a bound we *can* compute. Push the bound up and you push the evidence up with it.

Now factor $p_\theta(x, z) = p_\theta(x\mid z)\, p(z)$ and split the log:

$$
\mathcal{L}(\theta, \phi; x)
= \mathbb{E}_{q_\phi(z\mid x)}\big[ \log p_\theta(x\mid z) \big]
+ \mathbb{E}_{q_\phi(z\mid x)}\!\left[ \log \frac{p(z)}{q_\phi(z\mid x)} \right].
$$

The second expectation is, by definition, the negative **Kullback-Leibler divergence** between the posterior and the prior. So the ELBO splits cleanly into two named terms:

$$
\boxed{\;\mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[ \log p_\theta(x\mid z) \big]}_{\text{reconstruction}} \;-\; \underbrace{D_{\mathrm{KL}}\!\big( q_\phi(z\mid x) \,\|\, p(z) \big)}_{\text{KL to prior}}\;}
$$

![Diagram showing the evidence lower bound on log p of x splitting into a reconstruction term and a KL-to-prior term that combine into the training loss.](/imgs/blogs/variational-autoencoders-from-scratch-5.png)

Read those two terms back into the architecture and the whole VAE clicks into place.

The **reconstruction term** $\mathbb{E}_{q_\phi}[\log p_\theta(x \mid z)]$ says: sample a latent from the encoder's posterior, decode it, and reward the model when the decoded distribution assigns high probability to the *actual* input $x$. This is "rebuild the image accurately." If the decoder models pixels as Gaussians with fixed variance, then $\log p_\theta(x\mid z) = -\frac{1}{2\sigma^2}\|x - \hat{x}\|^2 + \text{const}$, and maximizing it is exactly minimizing squared reconstruction error — the same objective as a plain autoencoder. Remember that identity; it is the seed of the blurriness problem in Section 5.

The **KL term** $D_{\mathrm{KL}}(q_\phi(z\mid x) \| p(z))$ says: keep each image's posterior close to the standard-normal prior. This is the regularizer that organizes the latent space. It is the price of generation. Without it the encoder would set variance to zero and place every image at its own arbitrary point — back to the indexless filing cabinet. With it, the posteriors are squeezed toward $\mathcal{N}(0, I)$, the latent cloud becomes samplable, and you get a generative model.

There is one more identity worth knowing because it tells you exactly how loose the bound is. The gap between the evidence and the ELBO is itself a KL divergence:

$$
\log p_\theta(x) - \mathcal{L}(\theta, \phi; x) = D_{\mathrm{KL}}\!\big( q_\phi(z\mid x) \,\|\, p_\theta(z\mid x) \big) \;\ge\; 0.
$$

The bound is tight exactly when the approximate posterior $q_\phi$ matches the *true* posterior $p_\theta(z\mid x)$. Maximizing the ELBO over $\phi$ therefore does double duty: it improves the generative model *and* tightens the bound by making the encoder a better approximation of the true posterior. This is why we can get away with maximizing a lower bound instead of the real thing — we are simultaneously shrinking the gap. That is the entire conceptual payload of variational inference, and you just derived it from one application of Jensen's inequality.

#### Worked example: the KL term has a closed form

Because both $q_\phi(z\mid x) = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$ and the prior $p(z) = \mathcal{N}(0, I)$ are Gaussians, the KL divergence is analytic — you never have to estimate it. For a $d$-dimensional diagonal Gaussian against the unit normal:

$$
D_{\mathrm{KL}}\big(\mathcal{N}(\mu, \mathrm{diag}(\sigma^2)) \,\|\, \mathcal{N}(0, I)\big) = \frac{1}{2}\sum_{j=1}^{d}\Big( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \Big).
$$

Plug in a single latent dimension to build intuition. If the encoder outputs $\mu_j = 0$ and $\sigma_j = 1$ — exactly the prior — the bracket is $0 + 1 - 0 - 1 = 0$. Zero KL, no penalty, but also zero information carried by that dimension. If it outputs $\mu_j = 2, \sigma_j = 0.5$, the bracket is $4 + 0.25 - \log(0.25) - 1 = 4 + 0.25 + 1.386 - 1 = 4.636$ nats. That dimension is now carrying real information about $x$ — it has moved away from the prior — and it pays for it in KL. The optimizer's job is to spend KL budget only where it buys reconstruction. Sum this over all $d = 128$ dimensions and you have a single scalar you can compute in one line of PyTorch, which we do in Section 4. That is the practical gift of choosing Gaussians on both sides: the regularizer is free to evaluate.

## 3. The reparameterization trick: why you cannot just sample

We have a loss. Now we have to train it with backpropagation, and there is a wall directly in the path. The reconstruction term is an expectation over $q_\phi(z\mid x)$ — to compute it we *sample* $z \sim \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))$ and decode. But sampling is a stochastic operation. You cannot backpropagate a gradient through a random draw. The node that produces $z$ has no derivative with respect to $\mu$ and $\sigma$, because "draw a sample" is not a differentiable function of the parameters it samples from. The gradient $\nabla_\phi$ hits the sampling step and dies. Without that gradient, the encoder never learns. This is the problem that stalled latent-variable deep learning for years.

The fix is almost embarrassingly simple, and it is the single most important piece of engineering in the VAE. **Move the randomness out of the path between the parameters and the loss.** Instead of sampling $z$ directly from $\mathcal{N}(\mu, \sigma^2)$, sample a *fixed* noise variable $\epsilon \sim \mathcal{N}(0, I)$ from a distribution with no learnable parameters, and then construct $z$ deterministically:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I),
$$

where $\odot$ is elementwise multiplication. This is the **reparameterization trick**. The distribution of $z$ is identical — a draw from $\mathcal{N}(\mu, \sigma^2)$ is, by the standard location-scale property of the Gaussian, exactly $\mu + \sigma\epsilon$ with $\epsilon$ standard normal. Nothing about the model's statistics changed. But the *computational graph* changed completely. Now $z$ is a smooth, differentiable function of $\mu$ and $\sigma$, with $\epsilon$ sitting outside as a constant input — noise you feed in, like a data augmentation. The gradients flow straight through:

$$
\frac{\partial z}{\partial \mu} = 1, \qquad \frac{\partial z}{\partial \sigma} = \epsilon.
$$

The randomness is still there — every forward pass draws a fresh $\epsilon$, so the same image maps to slightly different latents each time — but it no longer blocks the gradient, because the path from $\mu, \sigma$ to the loss is now made of nothing but addition and multiplication. Look at figure 1 again: the noise $\epsilon$ enters from the side, the $\mu$ and $\sigma$ flow in from the encoder, and the $z$ node is a clean deterministic combination of the three. That sidecar entry of $\epsilon$ is the trick, drawn.

#### Worked example: why the alternative fails

The classical way to get a gradient through a sampling operation is the **score-function estimator**, also called REINFORCE. It does not require reparameterization; it works for any distribution. The catch is variance. The REINFORCE estimator of $\nabla_\phi \mathbb{E}_{q_\phi}[f(z)]$ is $\mathbb{E}_{q_\phi}[f(z)\,\nabla_\phi \log q_\phi(z\mid x)]$, which weights the (often large) score $\nabla_\phi \log q_\phi$ by the loss value $f(z)$. For a 128-dimensional continuous latent that estimator has enormous variance — empirically often one to two orders of magnitude higher than the reparameterized "pathwise" gradient — because it does not exploit the fact that $f$ is differentiable. The reparameterized gradient threads the derivative of $f$ all the way back through the deterministic $z = \mu + \sigma\epsilon$, using far more information per sample. In practice this is the difference between a VAE that converges in an afternoon and one that never trains at all. The reparameterization trick is not a convenience; for continuous latents it is the only thing that makes the gradient usable. (It is also why VAEs are easy and discrete latents in Section 7 are hard — you cannot reparameterize a categorical draw, so VQ-VAE needs a different gradient hack entirely.)

One more subtlety, because it matters when you read code. We parameterize the encoder to output $\log \sigma^2$ — the **log-variance** — not $\sigma$ directly. Two reasons. First, variance must be positive, and a raw network output is unconstrained; exponentiating a log-variance guarantees positivity without a clamp. Second, it is numerically gentler: $\sigma = \exp(\frac{1}{2}\log\sigma^2)$ stays stable across many orders of magnitude, and the KL formula's $\log \sigma_j^2$ term is then just the raw network output, no extra log. When you see `std = torch.exp(0.5 * logvar)` in the next section, that is this choice cashed out in one line.

## 4. Building a convolutional VAE in PyTorch

Enough theory. Here is a complete, runnable convolutional VAE for 64x64 RGB images — the kind you would train on CelebA or a downscaled subset of ImageNet. It is small enough to train on a single GPU in an hour and faithful to every piece of math above. The encoder is a stack of strided convolutions that halve resolution and grow channels; it ends in two linear heads, one for $\mu$ and one for $\log\sigma^2$. The decoder mirrors it with transposed convolutions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, in_ch=3, latent_dim=128, base=64):
        super().__init__()
        # Encoder: 64 -> 32 -> 16 -> 8 -> 4, channels grow as we downsample
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1), nn.BatchNorm2d(base), nn.SiLU(),       # 64 -> 32
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.SiLU(), # 32 -> 16
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.BatchNorm2d(base * 4), nn.SiLU(), # 16 -> 8
            nn.Conv2d(base * 4, base * 8, 4, 2, 1), nn.BatchNorm2d(base * 8), nn.SiLU(), # 8 -> 4
        )
        self.flat_dim = base * 8 * 4 * 4
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)       # mean head
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)   # log-variance head
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        # Decoder mirrors the encoder with transposed convolutions
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1), nn.BatchNorm2d(base * 4), nn.SiLU(), # 4 -> 8
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.SiLU(), # 8 -> 16
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.BatchNorm2d(base), nn.SiLU(),         # 16 -> 32
            nn.ConvTranspose2d(base, in_ch, 4, 2, 1),                                             # 32 -> 64
        )
        self.base = base

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * eps, with sigma = exp(0.5 * logvar). This is the whole trick.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)        # fresh N(0, I) noise, no grad through it
        return mu + std * eps

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.base * 8, 4, 4)
        return torch.sigmoid(self.dec(h))  # pixels in [0, 1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

The `reparameterize` method is the entire content of Section 3 in three lines. Notice `torch.randn_like(std)` draws the fresh $\epsilon$ with no gradient attached, and the returned `mu + std * eps` is a differentiable function of the encoder outputs. That is the trick, in code.

Now the loss. It is the negative ELBO — we minimize negative-ELBO, which is the same as maximizing the ELBO. The reconstruction term, under a Gaussian pixel likelihood, becomes mean squared error (or binary cross-entropy for `[0,1]` data); the KL term is the closed form from the worked example in Section 2. We add a $\beta$ weight on the KL — set it to 1 for the plain ELBO, and you will see in the next section why you might want it different.

```python
def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    # Reconstruction: sum over pixels, mean over batch. Per the Gaussian likelihood
    # this is equivalent to maximizing log p(x|z) up to a constant.
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    # KL( N(mu, sigma^2) || N(0, I) ) in closed form, summed over latent dims.
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kl, recon, kl
```

And the training step — vanilla PyTorch, nothing exotic:

```python
model = ConvVAE(latent_dim=128).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

for epoch in range(num_epochs):
    for x, _ in loader:                       # x: [B, 3, 64, 64] in [0, 1]
        x = x.cuda()
        x_hat, mu, logvar = model(x)
        # KL warm-up: ramp beta from 0 to 1 over the first few epochs to avoid collapse
        beta = min(1.0, epoch / 10.0)
        loss, recon, kl = vae_loss(x, x_hat, mu, logvar, beta=beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch}: recon={recon.item():.1f}  kl={kl.item():.2f}  beta={beta:.2f}")
```

The `beta = min(1.0, epoch / 10.0)` line is **KL annealing**, and it is not optional cosmetics — it is the single most common fix for the failure mode we cover next. Sampling from the trained model is then trivial: draw `z = torch.randn(n, 128).cuda()`, call `model.decode(z)`, and you have $n$ fresh faces. That four-line sample loop is the entire reason we did all the work of the previous two sections. The latent space is now a place you can draw from.

A note on what to expect when you run this. Reconstructions converge fast and look reasonable within a few epochs — recognizable faces, right pose, right hair color. *Samples* from the prior take longer and are the real test, because they exercise the smoothness of the whole latent cloud, not just the points your encoder mapped. And even at convergence, both will be a little soft. That softness is not a bug in your code. It is the next section.

### What the KL term actually buys you: a smooth, interpolatable latent

There is a property of the trained VAE that is worth dwelling on, because it is both a sanity check and the thing that distinguishes a VAE from a plain autoencoder in practice: the latent space is **smooth and interpolatable**. Encode two real images to their means $\mu_1$ and $\mu_2$, walk a straight line between them, $z_t = (1-t)\mu_1 + t\mu_2$ for $t \in [0, 1]$, and decode each point. In a well-trained VAE you get a *smooth morph* — one face gradually becoming another, every intermediate frame a plausible face, with no nonsense in between. In a plain autoencoder the same walk passes through dead regions and produces garbage partway across, because nothing forced the in-between space to decode to anything sensible. The KL-to-prior term is exactly what fills that space: by squeezing every posterior toward the same overlapping $\mathcal{N}(0, I)$, it forces neighboring images to occupy neighboring, *connected* latent regions, so the line between them stays on the manifold of plausible images.

```python
# Latent interpolation: encode two images, walk between their means, decode.
mu1, _ = model.encode(img_a.cuda())
mu2, _ = model.encode(img_b.cuda())
frames = [model.decode((1 - t) * mu1 + t * mu2) for t in torch.linspace(0, 1, 8)]
# Each frame is a plausible face; the morph is smooth because the KL term
# made the in-between latent region decode to something sensible.
```

This is not a party trick — it is the operational definition of "the latent space is organized." It is also why diffusion can later *navigate* this latent space: a connected, smooth latent is one where small steps produce small, sensible changes, which is exactly the property an iterative denoiser needs. A holey, disconnected latent would make diffusion's job far harder. So the KL term is doing double duty: it makes the space samplable (Section 1) *and* it makes the space smooth enough for a downstream generator to move through. Hold onto that — it is part of why the lightly-regularized SD VAE in Section 8 still keeps *some* KL even though it is not used as a sampler.

### Amortized inference: why one encoder beats per-image optimization

One more idea that is easy to miss but explains why the VAE is *efficient*, not just *correct*. Classical variational inference fits a separate posterior $q(z)$ for *every* data point by running an optimization loop per example — slow and impractical for a million images. The VAE's encoder does something cleverer called **amortized inference**: instead of optimizing a fresh posterior for each image, it trains a single shared network $q_\phi(z\mid x)$ that *predicts* the posterior parameters $(\mu, \log\sigma^2)$ directly from $x$ in one forward pass. The cost of inference is "amortized" across the whole dataset — you pay once to train the encoder, then get the posterior for any image, including images never seen in training, for free at test time. This is why a VAE can encode a new photo instantly while classical VI would need a fresh optimization. It is also a subtle source of looseness in the bound: the shared encoder cannot perfectly match every image's true posterior the way per-image optimization could (this gap is called the *amortization gap*), which is one reason real VAEs do not saturate the ELBO. But the trade — a tiny bit of bound looseness for billions-fold faster inference — is overwhelmingly worth it, and it is the only reason VAEs scale to ImageNet-sized data at all.

## 5. Why vanilla VAE samples are blurry

Train the VAE above to convergence, sample from the prior, and the faces will be plausible but *soft* — like a photograph with a thin layer of fog, or a JPEG saved at low quality. Skin is smooth where it should have pores, hair is a brown mass where it should have strands, eyes lack the sharp catchlight that makes a face look alive. This is the single most famous property of VAEs, and it is worth understanding precisely, because the fix is what makes VQ-GAN and the Stable Diffusion VAE work — and because "VAE blur" is the reason latent diffusion needs a *good* VAE, not just any VAE.

The blur is not a failure of capacity or training. It is the **Gaussian likelihood doing exactly what you told it to.** Recall from Section 2 that a Gaussian pixel likelihood makes the reconstruction term equal to squared error, $\log p_\theta(x \mid z) = -\frac{1}{2\sigma^2}\|x - \hat{x}\|^2 + \text{const}$. Now think about what squared error rewards when the decoder is uncertain. Suppose, given a latent $z$, the true conditional distribution over images is *multimodal* — maybe this latent could correspond to a face with the hair parted left or parted right, but not in between. The decoder can only output one image $\hat{x}$. Which single output minimizes expected squared error against a multimodal target? The **mean** of the modes. And the mean of "hair left" and "hair right" is "hair blurred down the middle." Squared error is **mean-seeking**: when the model is unsure between several sharp possibilities, the loss-minimizing prediction is their average, which is sharp in none of them. Every fine, high-frequency detail the model is uncertain about gets averaged into a smooth gray compromise.

![Before-and-after comparison of a soft mean-seeking VAE reconstruction versus a sharp VQ-GAN reconstruction with restored texture and edges.](/imgs/blogs/variational-autoencoders-from-scratch-3.png)

There are two compounding reasons the VAE is *especially* uncertain, and therefore especially blurry. First, the **bottleneck**: the latent is tiny, so a lot of pixel-level detail genuinely is not encoded — the decoder has to hallucinate it, and "hallucinate the safe average" is what MSE rewards. Second, the **KL regularizer injects noise**: every latent is a *sample* $z = \mu + \sigma\epsilon$, so the decoder must produce a reconstruction that is good *on average over a small cloud of nearby latents*. Averaging over a cloud of latents is, again, an averaging operation that smooths out anything high-frequency. The KL term that gives you a samplable latent space is the very same term that costs you sharpness. That is the central tension of the vanilla VAE, and you cannot tune your way out of it with a fixed per-pixel Gaussian loss.

This is exactly where the **$\beta$-VAE** knob enters, and it is a trilemma in miniature. The $\beta$ in our loss weights the KL term. Crank $\beta$ **up** (say 4 or higher, the original $\beta$-VAE regime) and you force a stronger match to the prior: the latent dimensions become more disentangled and the space more smoothly samplable, but you starve the reconstruction of information and the blur gets *worse*. Crank $\beta$ **down** toward zero and reconstructions sharpen — the latent carries more information — but the latent space drifts away from the prior, develops holes, and your *samples* (as opposed to reconstructions) degrade because you are now sampling from regions the decoder never saw. There is no setting of $\beta$ that gives you crisp samples with a per-pixel Gaussian loss. You are trading reconstruction fidelity against sample quality along a single axis. To actually get sharp images you have to change the *loss itself* — replace or augment the mean-seeking pixel likelihood. That is Section 7's entire reason for existing.

#### Worked example: reconstruction quality versus latent dim and beta

To make the trade-off concrete, here is the shape of results you get sweeping `latent_dim` and `beta` on a 64x64 face VAE, reporting reconstruction MSE (lower is sharper) and a qualitative read on sample diversity. These are illustrative magnitudes from this class of model, not a benchmark table — but the *directions* are robust and reproducible.

| latent_dim | beta | Recon MSE (rel.) | Sharpness | Sample quality |
| --- | --- | --- | --- | --- |
| 32 | 1.0 | high | very soft | smooth but bland |
| 128 | 1.0 | medium | soft | balanced |
| 256 | 1.0 | lower | softer-medium | balanced |
| 128 | 0.1 | low | sharper recon | samples drift / holes |
| 128 | 4.0 | high | very soft | disentangled, blurry |

Read the table as two gradients crossing. Down the `latent_dim` axis, more capacity lowers reconstruction error — but past a point it mostly helps *reconstruction* (where you give it the real image to encode) and barely helps *sampling* (where you draw $z$ from the prior), because the prior cannot exploit extra dimensions it does not know how to fill. Across the `beta` axis, low $\beta$ sharpens reconstructions at the cost of a holey latent, high $\beta$ smooths the latent at the cost of sharpness. The honest conclusion an engineer should draw: **the vanilla VAE is a fine compressor and a mediocre image sampler, and no hyperparameter rescues the sampling sharpness.** Which is precisely why, for generation, the field went two directions at once — discrete latents and adversarial losses — and why for *diffusion* we do not ask the VAE to be a good sampler at all. We only ask it to be a good autoencoder, and let the diffusion model handle the sampling. Hold that thought; it is the resolution in Section 8.

## 6. Posterior collapse: the failure mode that eats your latent

Before we leave the continuous VAE, we have to confront its most insidious training failure, because if you train VAEs you *will* hit it, and the symptoms are confusing. It is called **posterior collapse**, and it is the reason the training loop in Section 4 has that `beta` warm-up.

Here is what happens. Early in training the decoder is bad at using the latent — it has not learned much yet. Meanwhile the KL term has a very easy, very tempting way to drop to zero: just set the posterior equal to the prior. If the encoder outputs $\mu = 0$ and $\sigma = 1$ for *every* image, the KL term is exactly zero (plug into the closed form: $0 + 1 - 0 - 1 = 0$ per dimension). The optimizer notices this is a free lunch — the KL part of the loss is satisfied at zero cost — and if the decoder can drive down reconstruction loss *without* relying on $z$ (for instance by modeling the dataset's average image, or by exploiting a powerful autoregressive decoder), it will happily do so. The result: the encoder ignores the input entirely, the latent carries no information, and $q_\phi(z\mid x) = p(z)$ for all $x$. The posterior has *collapsed* onto the prior.

![Before-and-after diagram contrasting a collapsed posterior where the KL term goes to zero and the decoder ignores the latent with a healthy latent maintained by free-bits and KL annealing.](/imgs/blogs/variational-autoencoders-from-scratch-6.png)

The symptom is unmistakable once you know it: your KL term, logged each epoch, falls to nearly zero and stays there, while reconstruction loss plateaus at a mediocre value. Samples from the prior all look roughly the same — they are essentially the dataset mean, because the decoder has learned to ignore $z$. You have, in effect, trained a very expensive averaging function. The model technically maximized the ELBO (a low KL is part of that), but it found the degenerate solution where the latent is useless. This is most acute when the decoder is very expressive — it is the central problem in text VAEs with autoregressive decoders, and it shows up in image VAEs too whenever the decoder is strong relative to the bottleneck.

The fixes all share one idea: **stop the KL term from winning the race before the decoder learns to use the latent.** There are three standard ones, and they compose.

**KL annealing** (the warm-up you already saw): start with $\beta = 0$ so the model is a pure autoencoder for the first few epochs — it is *forced* to use the latent, because that is the only way to reconstruct — and ramp $\beta$ up to 1 over several epochs. By the time the KL penalty bites, the decoder already depends on an informative latent and will not abandon it. This is the cheapest fix and usually the first one to reach for.

**Free bits** is more surgical. The idea: do not penalize KL until each latent dimension carries at least $\lambda$ nats of information. You clamp the per-dimension KL at a floor:

$$
\mathcal{L}_{\text{free-bits}} = \mathbb{E}_{q_\phi}[\log p_\theta(x\mid z)] - \sum_{j} \max\big(\lambda,\; D_{\mathrm{KL}}(q_\phi(z_j\mid x)\,\|\,p(z_j))\big).
$$

Because the $\max$ is flat below $\lambda$, there is *no gradient* pushing a dimension's KL toward zero once it is already below $\lambda$ — the optimizer cannot collapse that dimension to claim a reward that does not exist. You reserve a minimum information budget per dimension. A typical $\lambda$ is on the order of 0.5 to a few nats per dimension. This directly attacks the mechanism of collapse rather than just delaying it.

**$\beta$-VAE in the other direction** — using $\beta < 1$ — is the blunt instrument: weaken the KL term globally so it cannot dominate. It works, but it also loosens the prior match and can hurt sample quality, so it is a worse tool than free bits for this specific problem. In practice the modern recipe for image autoencoders is KL annealing plus free bits, and — crucially — Section 7's perceptual and adversarial losses, which give the decoder a strong reason to use the latent (you cannot fool a discriminator with the dataset mean). The Stable Diffusion VAE sidesteps collapse almost entirely by using a *very* small KL weight (its KL barely regularizes at all — more on that in Section 8) precisely because in latent diffusion we do not need the VAE's latent to match a clean prior; we need it to be a faithful, information-rich autoencoder, and the diffusion model learns the latent distribution separately.

## 7. The discrete branch: VQ-VAE and VQ-GAN

So far the latent has been continuous — a point in $\mathbb{R}^{128}$. There is an entirely different and enormously consequential design: make the latent **discrete**. This is the **vector-quantized VAE** (VQ-VAE), introduced by van den Oord and colleagues in 2017, and its sharper descendant **VQ-GAN**. This branch is not a footnote — it produces the *tokens* that autoregressive image models like DALL-E 1, Parti, MUSE, and the 2025 wave of GPT-Image-style models generate one at a time, and it is the conceptual sibling of the continuous VAE that Stable Diffusion ultimately chose. You need both branches to understand the modern landscape.

The motivation is direct. Many of the most powerful generative models — transformers — operate on sequences of discrete tokens, the way a language model predicts the next word. If you want to generate images *the way GPT generates text*, you first need to turn an image into a sequence of discrete tokens from a finite vocabulary. A continuous latent will not do; you need a codebook. (This is the bridge to [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models), where those tokens become the language the transformer speaks.)

Here is how vector quantization works. The encoder produces a grid of continuous vectors — say $16\times16$ vectors, each of dimension $D$ — exactly as before. But instead of using them directly, we maintain a **codebook**: a learned dictionary of $K$ embedding vectors $\{e_1, \dots, e_K\}$, each also of dimension $D$ (a typical codebook has $K = 1024$ to $16384$ entries). For each encoder output vector $z_e(x)$, we find the *nearest* codebook entry by Euclidean distance and snap to it:

$$
z_q(x) = e_k, \qquad k = \arg\min_j \, \| z_e(x) - e_j \|_2 .
$$

The quantized latent $z_q$ is what the decoder receives, and the *index* $k$ — an integer in $\{1, \dots, K\}$ — is the discrete token. A $16\times16$ grid of indices is 256 tokens, an image written as a short sequence in a 1024-word visual vocabulary. That is exactly what an autoregressive transformer can model.

But $\arg\min$ has zero gradient almost everywhere — snapping to the nearest codebook entry is a step function, and you cannot backpropagate through it. This is the discrete analogue of the sampling problem from Section 3, and reparameterization does not apply (you cannot reparameterize a hard nearest-neighbor lookup). The fix here is the **straight-through estimator**: in the forward pass, use the quantized $z_q$; in the backward pass, *pretend the quantization was the identity* and copy the gradient from the decoder input straight back to the encoder output. In PyTorch this is the famous one-liner:

```python
# Straight-through estimator: forward uses z_q, backward flows as if z_q == z_e.
# (z_q - z_e).detach() has the right forward value but zero gradient,
# so d(z_q_st)/d(z_e) = 1 — the gradient passes through quantization unchanged.
z_q_st = z_e + (z_q - z_e).detach()
```

The `.detach()` makes the difference `(z_q - z_e)` a constant in the graph: the forward value of `z_q_st` equals `z_q` exactly, but its gradient equals that of `z_e`. The decoder's gradient lands on the encoder as if quantization were transparent. It is a biased estimator — the gradient is not the true gradient of the quantized function, which does not exist — but it works remarkably well in practice.

The codebook itself still needs to learn, and the encoder needs to be encouraged to produce outputs near codebook entries (so the straight-through approximation stays accurate). The VQ-VAE loss has three parts:

$$
\mathcal{L}_{\text{VQ}} = \underbrace{\|x - \hat{x}\|^2}_{\text{reconstruction}} + \underbrace{\|\,\mathrm{sg}[z_e] - e_k\,\|^2}_{\text{codebook}} + \beta\,\underbrace{\|\,z_e - \mathrm{sg}[e_k]\,\|^2}_{\text{commitment}}
$$

where $\mathrm{sg}[\cdot]$ is stop-gradient (`.detach()`). The first term is ordinary reconstruction. The **codebook loss** $\|\mathrm{sg}[z_e] - e_k\|^2$ moves each chosen codebook vector toward the encoder outputs assigned to it (the stop-gradient on $z_e$ means only $e_k$ moves) — it is essentially online k-means on the codebook. The **commitment loss** $\|z_e - \mathrm{sg}[e_k]\|^2$ does the reverse: it pulls the encoder output toward its chosen codebook entry (stop-gradient on $e_k$ means only the encoder moves), with weight $\beta \approx 0.25$. Commitment keeps the encoder from oscillating wildly between codebook entries, which would make the discrete representation unstable. Notice there is **no KL term** at all — the discrete uniform prior over codes makes the KL a constant, so it drops out. That is a structural difference from the continuous VAE worth remembering.

VQ-VAE solves discreteness but, trained with plain L2 reconstruction, it is still **blurry** for the same mean-seeking reason as the continuous VAE — squared error does not care about perceptual sharpness. Enter **VQ-GAN** (Esser, Rombach, Ommer, 2021, the "Taming Transformers" paper), which keeps the codebook and straight-through machinery but replaces the loss with two much better terms:

```python
# VQ-GAN reconstruction objective (schematic): the L2 term is replaced/augmented
# by a perceptual loss and an adversarial loss for sharp, realistic texture.
recon_loss = lpips(x, x_hat)                  # perceptual: distance in a VGG feature space
g_loss     = -discriminator(x_hat).mean()      # adversarial: fool a patch discriminator
loss = recon_loss + lambda_adv * g_loss + vq_loss   # vq_loss = codebook + commitment
```

The **perceptual loss** (LPIPS) compares $x$ and $\hat{x}$ not pixel by pixel but in the feature space of a pretrained network (VGG), where "perceptually similar" means "similar deep features." This penalizes the loss of *texture* and *structure* that pixel MSE ignores — two images can have identical MSE while one is sharp and one is blurred, but they have very different VGG features. The **adversarial loss** adds a patch discriminator (a small CNN trained to tell real from reconstructed patches); the autoencoder is trained to fool it. Because a blurry patch is trivially distinguishable from a real one, the discriminator *forces* the decoder to produce sharp, realistic texture — it cannot win by averaging. Together, perceptual plus adversarial losses are what finally break the mean-seeking blur. The reconstructions become crisp. Figure 3 is this exact contrast: the same architecture, the same compression, but a loss that rewards realism instead of pixel-averaging.

This is the lineage that matters for generation, and figure 8 traces it: deterministic autoencoder → VAE (adds a sampleable prior via the ELBO) → VQ-VAE (adds a discrete codebook) → VQ-GAN (adds perceptual and adversarial sharpness) → and then the Stable Diffusion VAE, which takes the VQ-GAN *training recipe* (perceptual + adversarial losses for crisp reconstruction) but goes back to a **continuous, KL-regularized** latent because — as we are about to see — diffusion wants a continuous latent to add Gaussian noise to.

![Timeline of the autoencoder lineage from deterministic autoencoder to VAE to VQ-VAE to VQ-GAN to the Stable Diffusion KL-regularized VAE.](/imgs/blogs/variational-autoencoders-from-scratch-8.png)

#### Worked example: counting the tokens an image becomes

Make the token paradigm concrete. Take a 256x256 image and a VQ-GAN with an 16x downsampling factor. The encoder produces a $16\times16$ grid of latent vectors, each quantized to one of $K = 16384$ codebook entries. That is $16 \times 16 = 256$ tokens, each an integer in $[0, 16383]$ — so the image is now a length-256 sequence over a 16384-symbol vocabulary, roughly $256 \times \log_2(16384) = 256 \times 14 = 3584$ bits, about 448 bytes. A transformer can autoregressively predict those 256 tokens left-to-right, top-to-bottom, exactly the way a language model predicts 256 words — and decoding the predicted token grid through the VQ-GAN decoder paints the image. That is the whole trick behind autoregressive image generation: the VQ-GAN turns "model a 196608-dimensional pixel distribution" into "model a length-256 sequence over a 16384-word vocabulary," a problem transformers already solve beautifully. The autoencoder did the hard part — it found the *language*. Crank the downsampling to 8x and you get a $32\times32 = 1024$-token sequence: more tokens, sharper images, slower generation. That token-count-versus-fidelity knob is one of the central design choices in autoregressive image models.

## 8. The punchline: latent diffusion is diffusion in a VAE's latent space

Now we collect the debt. Everything in this post — the ELBO, the reparameterization trick, the blur problem, the VQ-GAN sharpening recipe — converges on one architectural decision that defines modern image generation: **run the diffusion process not on pixels, but inside a pretrained VAE's latent space.** This is the **latent diffusion model** (LDM) of Rombach et al., 2022 — the architecture that became Stable Diffusion. And the VAE is the unsung hero of the whole thing.

Here is the problem latent diffusion solves. Diffusion models (covered from first principles in [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles)) work by gradually adding Gaussian noise to data and training a network to reverse that process. The catch is cost: the denoising network has to process the full data tensor at every one of many sampling steps. On 512x512 RGB pixels — a $3\times512\times512 = 786432$-dimensional tensor — that is brutally expensive, both to train (you need enormous compute to model that many dimensions) and to sample (every step is a full forward pass over 786k values). Pixel-space diffusion models like the original ADM and Imagen needed massive resources precisely because they paid the full pixel cost at every step.

Latent diffusion's insight: **most of those pixels are perceptually redundant.** A natural image lives near a much lower-dimensional manifold; you do not need 786k numbers to capture its content, you need its *semantics* plus enough detail to reconstruct it. So: train a VAE (with the VQ-GAN recipe — perceptual plus adversarial losses for crisp reconstruction) to compress 512x512 images into a small latent, *freeze it*, and then run the entire diffusion process in that latent space. Decode once at the very end.

![Stack diagram of the latent diffusion pipeline showing a frozen VAE encode, diffusion denoising in the compressed latent, and a final VAE decode back to pixels.](/imgs/blogs/variational-autoencoders-from-scratch-7.png)

The numbers are the whole argument. Stable Diffusion's VAE uses an 8x spatial downsampling factor and a 4-channel latent. A $3\times512\times512$ image becomes a $4\times64\times64$ latent. Count the values:

$$
\frac{3 \times 512 \times 512}{4 \times 64 \times 64} = \frac{786432}{16384} = 48 .
$$

**Forty-eight times fewer values.** The diffusion model operates on 16384 numbers instead of 786432. That is why the denoiser is so much cheaper to train and sample — every one of its forward passes is over a tensor 48x smaller. The spatial resolution the U-Net or DiT attends over drops from 512 to 64 per side (an 8x reduction in each dimension, hence the 64x reduction in spatial positions, partly offset by going from 3 to 4 channels — net 48x in raw values). Self-attention, which scales quadratically with the number of spatial positions, gets *dramatically* cheaper. This single change is the reason Stable Diffusion trains on academic-scale compute and runs on a consumer GPU, while pixel-space diffusion at the same resolution does neither.

![Stack diagram showing a 512-pixel RGB image compressed eightfold per side into a four-channel latent with forty-eight times fewer values for diffusion.](/imgs/blogs/variational-autoencoders-from-scratch-2.png)

And here is the resolution to the blur paradox from Section 5. Earlier we said the vanilla VAE is a mediocre *sampler* — its prior samples are blurry. But in latent diffusion **we never sample from the VAE's prior.** We only ever use the VAE as an autoencoder: encode real images to latents during training, and decode diffusion-generated latents to pixels at inference. The *sampling* — the hard generative-modeling job — is done by the diffusion model in latent space, not by the VAE. So the VAE's weakness (bad prior samples) is completely sidestepped, and we only need its strength (faithful, sharp reconstruction, which the VQ-GAN recipe gives us). The blur problem evaporates because we changed the VAE's job description: from "be a generative model" to "be a great codec." That is the elegant division of labor at the heart of latent diffusion, and it is why understanding the VAE is non-negotiable for understanding Stable Diffusion.

One more detail that trips people up. Stable Diffusion's VAE is technically a **KL-regularized continuous VAE** (`AutoencoderKL` in `diffusers`), but with a *tiny* KL weight — the regularization is so weak that the latent is, for practical purposes, an information-rich continuous code rather than a clean unit-Gaussian sample. Why keep any KL at all? Because the diffusion model adds Gaussian noise to the latent, and it helps enormously if the latent is roughly standardized and well-conditioned — not living at scale 1000 with weird per-channel statistics. In fact SD multiplies the encoded latent by a **scaling factor** (0.18215 for SD 1.x) precisely to bring its standard deviation near 1 so the noise schedule behaves. The KL is not there to make the latent sampleable; it is there to keep it tame enough for diffusion to noise it cleanly. Continuous (so you can add continuous Gaussian noise), lightly KL-regularized (so it is well-conditioned), VQ-GAN-trained (so it is sharp): that is the precise recipe of the autoencoder under Stable Diffusion.

### Loading the real Stable Diffusion VAE

Theory into practice. Here is how you load Stable Diffusion's actual VAE from 🤗 `diffusers`, encode a real image to the SD latent, and decode it back — measuring the compression and VRAM yourself.

```python
import torch
from diffusers import AutoencoderKL
from diffusers.utils import load_image
import torchvision.transforms as T

# The exact VAE Stable Diffusion 1.5 ships with.
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16
).to("cuda")
vae.eval()

img = load_image("cat.png").resize((512, 512))
x = T.ToTensor()(img).unsqueeze(0).to("cuda", torch.float16) * 2 - 1   # SD expects [-1, 1]

with torch.no_grad():
    posterior = vae.encode(x).latent_dist        # a diagonal Gaussian, like our q(z|x)
    z = posterior.sample() * vae.config.scaling_factor   # 0.18215 for SD 1.x
    print("image tensor:", tuple(x.shape))       # (1, 3, 512, 512)
    print("latent tensor:", tuple(z.shape))      # (1, 4, 64, 64)
    print("compression:", x.numel() / z.numel()) # 48.0

    recon = vae.decode(z / vae.config.scaling_factor).sample  # back to pixels
    recon = ((recon.clamp(-1, 1) + 1) / 2)        # [-1,1] -> [0,1]
```

Notice `vae.encode(x).latent_dist` returns a `latent_dist` — a diagonal Gaussian with a `mean` and `logvar`, exactly the $q_\phi(z\mid x)$ we built by hand in Section 4, and `.sample()` applies the reparameterization trick under the hood. The `scaling_factor` is the standardization constant from the last paragraph. Run this and `x.numel() / z.numel()` prints `48.0` — the number this whole post has been building toward, measured on the real model. The VAE itself is small: SD 1.5's `AutoencoderKL` is about 84M parameters and occupies roughly 160 MB in fp16, a footnote next to the ~860M-parameter U-Net, yet it is doing the 48x heavy lifting that makes the U-Net affordable.

A practical serving note. The VAE decode is itself a chunk of the inference cost and VRAM — decoding a $4\times64\times64$ latent to 512x512 pixels spikes memory because the activations grow back to full resolution. For high resolutions or batched generation, `diffusers` offers `vae.enable_slicing()` (decode one image at a time) and `vae.enable_tiling()` (decode in spatial tiles), which trade a little speed for a large VRAM reduction — essential when you are generating 1024x1024 SDXL images on a 24 GB card. And if you want the decode nearly free, swap in `AutoencoderTiny` (TAESD), a distilled ~2M-parameter VAE that decodes in milliseconds at a small quality cost — invaluable for fast previews during interactive generation.

#### Worked example: what the 48x latent saves the diffusion model

Put a number on why this matters for the part everyone *does* talk about — the denoiser. The dominant cost in a diffusion U-Net or DiT is self-attention over spatial positions, and self-attention scales *quadratically* with the number of positions. In pixel space at 512x512 there are $512 \times 512 = 262144$ spatial positions to attend over; in the SD latent at 64x64 there are $64 \times 64 = 4096$. The ratio of positions is 64x, and because attention is quadratic, the cost of a full global attention map scales like $64^2 = 4096$x in the worst case — though in practice U-Nets use windowed/multi-resolution attention so the realized saving is smaller, the latent still cuts the attention bill by orders of magnitude. Now compound that over sampling steps: a diffusion model runs the denoiser 20 to 50 times per image. Every one of those passes is over the 48x-smaller latent. So the latent does not save you 48x once — it saves you 48x on *every* training forward pass and *every* one of the dozens of sampling steps, across millions of training images. This is the difference between a model that costs hundreds of A100-days to train in pixel space and one that trains in a fraction of that in latent space — and the difference between sampling an image in a second on an RTX 4090 versus tens of seconds. The VAE pays for itself a hundred million times over the course of training. That compounding is the real reason latent diffusion won, and it is entirely the VAE's doing.

There is a second, quieter saving: the VAE lets you *decouple* perceptual compression from generative modeling. The VAE handles the imperceptible, high-frequency pixel detail (the "perceptual" bits that humans barely register but that dominate the raw pixel count), so the diffusion model can spend its entire capacity on the *semantic* content — composition, objects, structure — that actually requires generative modeling. Rombach et al. called this the "perceptual compression versus semantic compression" split, and it is the conceptual heart of latent diffusion. The diffusion model is freed from re-learning how to draw sharp edges and textures (the VAE decoder already knows that) and can focus on *what* to draw. Splitting the problem this way is why a latent diffusion model of a given size beats a pixel-space model of the same size on both quality and speed: it is not wasting parameters on a job the VAE already does.

## 9. Case studies: real autoencoders in shipped models

Let us ground all of this in the autoencoders that actually run in production systems, with numbers you can cite. These are the VAEs and VQ models underneath models you have used.

**Stable Diffusion 1.5 / 2.x — the 8x KL-VAE.** The canonical `AutoencoderKL`: 8x downsampling, 4 latent channels, ~84M parameters, the 48x compression we measured. Trained with the LDM recipe — pixel reconstruction plus LPIPS perceptual loss plus a patch-GAN adversarial loss plus a small KL term. This is the workhorse. Its known weakness: at 4 channels it sometimes struggles with fine high-frequency detail and faces at small sizes, which is *the VAE being the bottleneck*, not the diffusion model. When you see a Stable Diffusion image with slightly mushy text or eyes, the VAE decode is often the culprit, not the U-Net.

**SDXL — a retrained, better VAE.** SDXL kept the same 8x / 4-channel *shape* but retrained the VAE with a larger batch and an EMA of weights, measurably improving reconstruction fidelity (lower reconstruction FID) over the SD 1.5 VAE. Same compression factor, better codec. It is a clean illustration that the VAE is an independent quality lever: you can upgrade the autoencoder without touching the diffusion model and get sharper outputs. The lesson shows up again in [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion), where the SDXL recipe is dissected in full.

**SD3 and FLUX — 16-channel latents.** The 2024 frontier models made a pointed change: they widened the latent from 4 to **16 channels** (while keeping 8x spatial downsampling). More channels means the latent carries more information per spatial position, which directly improves reconstruction of fine detail and text — the exact failure mode of the 4-channel VAE. The cost is a less aggressive *value* compression (16 channels instead of 4 cuts the 48x down to 12x), so the diffusion model has more to chew on, but the fidelity ceiling rises. This is the field explicitly trading some of the VAE's compression back for reconstruction quality, having learned that the 4-channel VAE was the limiting factor on detail.

**SANA — the deep-compression 32x autoencoder.** SANA (2024) pushed the *other* way: a deep-compression autoencoder with a **32x** spatial downsampling factor (versus the usual 8x), shrinking the latent grid 16x further in area. A 1024x1024 image becomes a tiny $32\times32$ latent. That makes the diffusion transformer dramatically cheaper — fewer tokens to attend over — and is a big part of why SANA generates high-resolution images so fast. It is the purest demonstration of the thesis of this post: **push more work into the autoencoder's compression, and the generative model gets cheaper.** The autoencoder is a compute lever, and SANA pulled it hard.

**DALL-E 1 and the Taming/MUSE line — the discrete branch in production.** DALL-E 1 used a discrete VAE (a VQ-VAE variant) to tokenize images into a $32\times32$ grid of tokens from an 8192-entry codebook, then modeled them autoregressively with a transformer. The Taming Transformers (VQ-GAN) line and Google's MUSE used VQ-GAN codebooks the same way. These are the discrete-branch autoencoders from Section 7, shipping at scale — the proof that the codebook path is not academic. They are the substrate of [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models).

Here is the comparison that organizes the whole design space — continuous versus discrete, the loss that determines sharpness, and what consumes each one.

![Matrix comparing vanilla VAE, VQ-VAE, and VQ-GAN across latent type, reconstruction crispness, key loss, and downstream model.](/imgs/blogs/variational-autoencoders-from-scratch-4.png)

| Autoencoder | Latent | Compression | Crisp? | Loss | Consumed by |
| --- | --- | --- | --- | --- | --- |
| Vanilla VAE | continuous | varies | no (blurry) | ELBO: KL + MSE | — (mediocre sampler) |
| SD 1.5 KL-VAE | continuous, 4ch | 48x | yes | LPIPS + GAN + small KL | SD U-Net (latent diffusion) |
| SD3 / FLUX VAE | continuous, 16ch | 12x | very | LPIPS + GAN + small KL | MM-DiT (latent diffusion) |
| SANA DC-AE | continuous | ~192x (32x spatial) | yes | LPIPS + GAN | SANA DiT (latent diffusion) |
| VQ-VAE | discrete codebook | varies | no (blurry) | recon + codebook + commit | DALL-E 1 (autoregressive) |
| VQ-GAN | discrete codebook | varies | yes | LPIPS + GAN + codebook | Taming/MUSE (autoregressive) |

Read the table top to bottom and the entire post is one row at a time. The continuous, KL-regularized, GAN-trained VAE is what latent diffusion eats. The discrete, codebook, GAN-trained VQ-GAN is what autoregressive transformers eat. The vanilla VAE — the one we derived from the ELBO — is the conceptual root of both and a production model of neither, which is exactly the right thing to understand first.

## 10. When to reach for a VAE (and when not to)

A decisive section, because the VAE is a tool with sharp edges and the right choice depends on what you are actually building.

**Reach for a continuous KL-VAE when you are building latent diffusion.** This is the default and overwhelmingly the most common case. If you are training or fine-tuning a Stable-Diffusion-class model, you want a frozen, pretrained, VQ-GAN-recipe KL-VAE doing 8x (4 or 16 channel) compression. Do not train it from scratch unless you have a specific reason — the SD/SDXL/FLUX VAEs are excellent and freely available, and a from-scratch VAE is a multi-day project that rarely beats them. The right move is almost always "load the pretrained VAE, freeze it, train the diffusion model in its latent."

**Reach for a VQ-GAN when you are building an autoregressive or masked token model.** If your generator is a transformer that predicts discrete tokens (the GPT-Image / Parti / MUSE paradigm), you need discrete latents, which means a VQ-VAE or VQ-GAN tokenizer. Use VQ-GAN, not plain VQ-VAE — the perceptual and adversarial losses are not optional if you want sharp output. The codebook size and downsampling factor are your main knobs (more tokens = sharper but slower generation).

**Do not use a vanilla VAE as your final generative model** if you care about image quality. It is the right pedagogical object and a fine compressor, but its prior samples are blurry and no amount of $\beta$-tuning fixes the sharpness. If someone proposes shipping a plain VAE as an image generator in 2026, that is a flag — the field moved past it for generation a decade ago, for the mean-seeking reason in Section 5. (VAEs are still excellent for representation learning, anomaly detection, and as the codec inside diffusion — just not as a standalone sampler.)

**Do not train your VAE from scratch when the bottleneck is elsewhere.** A common mistake: a team gets blurry diffusion outputs and assumes they need a bigger diffusion model, when the real ceiling is the 4-channel VAE losing high-frequency detail. Diagnose first — encode and decode a real image through your VAE *with no diffusion at all* and look at the reconstruction. If the VAE-reconstructed image is already blurry or loses text, no amount of diffusion-model improvement will fix it; the VAE is the bottleneck. Switch to a 16-channel VAE (SD3/FLUX-style) or accept the ceiling. This single diagnostic — "is my VAE round-trip already lossy?" — saves enormous wasted effort.

**Do not over-compress when fidelity matters.** SANA's 32x compression is a fantastic speed win, but every step up in compression is information thrown away that the decoder must hallucinate. For photorealistic faces or readable text, the aggressive-compression VAE can be the wrong call; the 16-channel SD3/FLUX direction (less compression, more fidelity) exists precisely because the field discovered the 4-channel VAE was over-compressing for detailed content. Compression is a lever with a real cost on the other end.

#### Worked example: diagnosing a blurry generation pipeline

You fine-tune a Stable Diffusion model on your product photos and the outputs look soft. Where is the blur coming from — the diffusion model or the VAE? Run the diagnostic. Take a real product photo, encode it with the VAE, immediately decode it (no diffusion, no noise), and compare to the original. Suppose the round-trip image is *already* soft — text on the packaging is mushy, fine fabric texture is gone. That tells you the 4-channel SD 1.5 VAE is the bottleneck: it physically cannot represent that detail in 4 channels at 8x compression, so your diffusion model, however good, is decoding through a lossy codec. The fix is not more diffusion training — it is a better VAE (move to a 16-channel SD3-class autoencoder, or fine-tune the VAE decoder on your domain). Conversely, if the VAE round-trip is crisp but your *generated* images are soft, then the blur is in the diffusion model or sampler (too few steps, wrong scheduler, weak conditioning) and the VAE is innocent. This five-minute test — round-trip a real image through the VAE alone — is the most useful debugging move in the entire stack, and it falls directly out of understanding that the VAE is a separable codec.

## 11. Key takeaways

- **A VAE is an autoencoder whose encoder outputs a distribution, not a point, plus a regularizer that drags those distributions toward a samplable prior.** Those two changes turn a compressor into a generative model — that is the whole leap.
- **The loss is not invented; it is derived.** The ELBO falls out of one application of Jensen's inequality to the intractable $\log p(x)$, and it splits exactly into a reconstruction term (rebuild the image) minus a KL term (match the prior). The bound's looseness is itself a KL between the approximate and true posterior, so maximizing the ELBO tightens the bound for free.
- **The reparameterization trick `z = μ + σ⊙ε` is what makes the VAE trainable.** It moves the randomness into an external $\epsilon$ so gradients flow through deterministic arithmetic instead of dying at a sampling node. For continuous latents it is the only practical option — the alternative (REINFORCE) has crippling variance.
- **Vanilla VAE samples are blurry because the Gaussian likelihood is mean-seeking.** When the decoder is uncertain, squared error rewards the average of the plausible images, which is sharp in none of them. No $\beta$ setting fixes it; you must change the loss.
- **Posterior collapse is the KL term winning the race before the decoder learns to use the latent.** KL annealing (warm up $\beta$ from 0) and free bits (a per-dimension KL floor with no gradient below it) are the standard fixes; a strong perceptual/adversarial decoder loss also discourages collapse.
- **The discrete branch — VQ-VAE and VQ-GAN — produces tokens, not continuous codes.** Vector quantization snaps encoder outputs to a learned codebook; the straight-through estimator passes gradients through the non-differentiable $\arg\min$; commitment loss keeps the encoder near its codes. VQ-GAN adds perceptual + adversarial losses for sharpness. This is the substrate of autoregressive image models.
- **Latent diffusion is diffusion run in a VAE's latent space, and the VAE is the unsung hero.** Stable Diffusion's 8x / 4-channel VAE compresses 786432 pixel values into 16384 latent values — 48x fewer — so the diffusion model never touches a full-resolution image. This single choice is why latent diffusion trains on academic compute and runs on consumer GPUs.
- **The VAE's "bad sampler" weakness is irrelevant for diffusion** because we use it only as a codec — encode real images, decode generated latents — and let the diffusion model do the sampling. We need the VAE's strength (faithful, sharp reconstruction) and never its prior samples.
- **The autoencoder is an independent quality and speed lever.** SDXL upgraded the VAE alone for sharper output; SD3/FLUX widened it to 16 channels for fine detail; SANA pushed 32x compression for speed. When generations are blurry, round-trip a real image through the VAE alone to check whether the VAE is the bottleneck before touching the diffusion model.

## 12. Further reading

- **Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)** — the original VAE paper. The ELBO and the reparameterization trick in their first form. Dense but foundational; read it after this post and the derivations will be familiar.
- **Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (2017)** — the $\beta$ knob and the disentanglement-versus-reconstruction trade-off made explicit.
- **van den Oord, Vinyals & Kavukcuoglu, "Neural Discrete Representation Learning" (VQ-VAE, 2017)** — vector quantization, the codebook, the straight-through estimator, and the commitment loss.
- **Esser, Rombach & Ommer, "Taming Transformers for High-Resolution Image Synthesis" (VQ-GAN, 2021)** — the perceptual + adversarial recipe that makes discrete autoencoders crisp, and the token paradigm for autoregressive image generation.
- **Rombach, Blattmann, Lorenz, Esser & Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models" (LDM / Stable Diffusion, 2022)** — the paper that runs diffusion in the VAE latent and quantifies the perceptual-compression argument. The punchline of this post in its original form.
- **🤗 `diffusers` `AutoencoderKL` documentation** — the API reference for loading, encoding, decoding, slicing, and tiling the production VAE; the `AutoencoderTiny` (TAESD) docs for fast previews.
- **Within this series:** [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard) (the four-family map and the generative trilemma), [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions) (the likelihood/ELBO primer this post builds on), and forward to [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) and [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models), with the full pipeline assembled in [building an image-generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
