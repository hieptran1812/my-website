---
title: "Latent Diffusion and Stable Diffusion: Making Image Generation Cheap Enough to Run Anywhere"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How latent diffusion cut the cost of image generation by roughly 48x and made Stable Diffusion possible: the perceptual-compression argument, the frozen-autoencoder two-stage design, cross-attention text conditioning, SD1.5 through SDXL with runnable diffusers code, and the VAE failure modes everyone hits."
tags:
  [
    "image-generation",
    "diffusion-models",
    "latent-diffusion",
    "stable-diffusion",
    "sdxl",
    "vae",
    "cross-attention",
    "generative-ai",
    "deep-learning",
    "text-to-image",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/latent-diffusion-and-stable-diffusion-1.png"
---

In 2021, training a competitive pixel-space diffusion model was a job for an industrial lab. Dhariwal and Nichol's ADM ran the denoiser directly on pixels, and at 256×256 — never mind 512 or 1024 — that meant pushing a 256×256×3 tensor, almost 200,000 values, through a heavy U-Net a thousand times per sample, for hundreds of thousands of training steps. The compute bill ran to hundreds of TPU- or GPU-days. Image generation was a spectator sport: you could admire the samples in a paper, but you could not train the model, you could not fine-tune it, and you certainly could not run it on the 3090 under your desk. Then in late 2021 a group at LMU Munich (Rombach, Blattmann, Lorenz, Esser, Ommer) asked an almost embarrassingly simple question: *why are we running the expensive part on pixels at all?* Their answer — run the diffusion in a compressed latent space — cut the cost by roughly an order of magnitude, and within a year it had become Stable Diffusion, the model that put text-to-image on a laptop and opened the floodgates.

This post is about that idea and what it became. The thesis is blunt and worth stating up front: **latent space is the single biggest efficiency lever in the entire image-generation stack.** Not a better sampler, not a smaller network, not quantization — those all help, and later posts cover them, but none of them moves the needle like deciding *what space the diffusion runs in*. Latent diffusion took a 512×512×3 image (786,432 numbers) and ran the diffusion on a 64×64×4 latent (16,384 numbers) instead — an 8× spatial downsample per side, a roughly 48× reduction in tensor size, and a comparable cut in the U-Net's per-step FLOPs. Everything else about Stable Diffusion — the consumer-GPU inference, the explosion of LoRAs and ControlNets, the fact that you can read this and then go generate a cat — follows from that one decision.

![A layered stack figure showing pixels at 512x512x3 encoded by a frozen VAE into a 64x64x4 latent that is 48x smaller, the U-Net denoiser running on that small latent, and a single decode back to a 512x512 image](/imgs/blogs/latent-diffusion-and-stable-diffusion-1.png)

By the end you will understand: the **perceptual-compression argument** (why most of the bits in a photograph are imperceptible high-frequency detail you can throw away before diffusing); the **two-stage design** (a frozen KL- or VQ-regularized autoencoder plus a latent-space diffusion U-Net) and why training them separately is not just convenient but principled; how **cross-attention** from a CLIP text encoder turns this into a text-to-image model; the concrete configs of **Stable Diffusion 1.x, 2.x, and SDXL** — the encoder swaps, v-prediction, the dual text encoders, the **micro-conditioning** trick that fixed cropping and low-res artifacts, the optional refiner; and the **VAE itself** — its perceptual+adversarial loss, why a small KL keeps the latent diffusable, and the failure modes (latent artifacts, the famous "SDXL VAE fix") that bite you in production. We will ground all of it in the math, in runnable [🤗 `diffusers`](https://huggingface.co/docs/diffusers) code, and in measured numbers from the papers.

Where does this sit in our running frame? Recall the **diffusion stack**: data → **VAE/latent** → forward noising → denoiser net → ODE/SDE sampler → guidance → image. This post is about the second box — the one that, it turns out, decides how expensive every later box is. And recall the **generative trilemma** (quality × diversity × speed): latent diffusion does not obviously buy you any one corner of that triangle, but it makes the *whole triangle affordable*, which is what actually mattered. If the VAE and the latent are unfamiliar, the [variational autoencoders post](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) builds the autoencoder this whole post stands on, and [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) is the denoising process we are about to relocate into latent space.

## The problem: pixel-space diffusion is paying for the wrong bits

Start with the cost model, because the whole idea falls out of it. A diffusion model's expensive operation is one forward pass of the denoiser U-Net, and you run it once per sampling step — 50 times for a typical DDIM sample, up to 1000 for full DDPM — and millions of times during training. The cost of one forward pass scales with the size of the tensor it operates on. For a convolutional U-Net on an $H \times W \times C$ input, the dominant terms scale roughly with the number of spatial positions $H \times W$ (the convolutions and, worse, the self-attention layers, whose cost is quadratic in the number of positions). So if you halve $H$ and $W$, you cut the convolution work by ~4× and the attention work by ~16×. The pixel grid is the thing you are paying for.

Now ask: how many of those pixels carry information you actually care about? A 512×512 photograph has 786,432 channel-values. But the *perceptually meaningful* content — the semantic layout, the objects, the rough textures — lives in a much smaller subspace. Most of the raw bits encode high-frequency detail: the exact value of each pixel relative to its neighbor, sensor noise, fine grain. Your visual system barely registers a lot of it. This is the same insight JPEG exploits: transform to the frequency domain, quantize the high frequencies hard, and you can throw away most of the data with little perceptual loss. The pixel representation is *perceptually redundant*. Rombach et al. made this precise with a two-phase view of what a generative model spends its capacity on. In the **perceptual compression** phase, a model removes high-frequency detail but learns little semantic structure — this is cheap information-theoretically but expensive in pixels. In the **semantic compression** phase, the model learns the actual content and composition of images — this is where the hard, interesting modeling lives. Pixel-space diffusion forces a single network to do *both*, and it spends most of its FLOPs on the boring perceptual-compression part, denoising imperceptible detail at full resolution.

The latent-diffusion move is to **split those two phases across two models**. Let an autoencoder handle perceptual compression once: it learns to map a 512×512×3 image into a small 64×64×4 latent and back, throwing away the imperceptible high-frequency detail in the process. Then let the diffusion model do *only* the semantic part, operating entirely in that compact latent space where every dimension carries meaningful, perceptually relevant signal. The U-Net never sees a raw pixel during training or sampling — it denoises latents. The decode to pixels happens exactly once, at the very end. We pay the perceptual-compression cost once (training the autoencoder) and reuse it across every diffusion training step and every sample.

How big is the win? The Stable Diffusion VAE downsamples by a factor of 8 in each spatial dimension (512→64) and maps 3 color channels to 4 latent channels. So the tensor goes from $512 \times 512 \times 3 = 786{,}432$ values to $64 \times 64 \times 4 = 16{,}384$ values. That is a $786{,}432 / 16{,}384 = 48\times$ reduction in the number of values the U-Net processes per step. Because convolution cost is roughly linear in spatial positions and attention is quadratic, the *per-step* compute reduction lands in the same order of magnitude — Rombach et al. report training and sampling speedups in the range of roughly an order of magnitude versus comparable pixel-space models, and the headline framing the community latched onto was "~48× cheaper" from the value-count alone. The exact FLOP ratio depends on architecture details (how much of the cost is attention vs convolution, how the channel counts compare), so treat 48× as the *tensor-size* compression and "roughly an order of magnitude" as the honest end-to-end compute statement.

#### Worked example: counting the compression

Take a concrete 512×512 RGB image and walk it through the SD VAE. Input: $512 \times 512 \times 3 = 786{,}432$ scalar values. The encoder applies a stack of strided convolutions that downsample $\times 2$ three times (so $2^3 = 8\times$ per side), ending at $64 \times 64$ spatial resolution with $4$ latent channels: $64 \times 64 \times 4 = 16{,}384$ values. Compression ratio: $786{,}432 / 16{,}384 = 48$. Equivalently: $8 \times 8 = 64$ fewer spatial positions, partly offset by going from 3 to 4 channels ($64 \times 3/4 = 48$). Now the cost story. Suppose, to first order, the U-Net's work is dominated by self-attention at the top resolution, which is quadratic in the number of spatial positions. Pixel-space attention at $512^2 = 262{,}144$ positions versus latent attention at $64^2 = 4{,}096$ positions is a ratio of $(262{,}144 / 4{,}096)^2 = 64^2 = 4{,}096\times$ for that layer — wildly more than 48×. In practice the U-Net mixes convolution (linear in positions) and attention (quadratic), and operates at multiple resolutions, so the realized speedup is nowhere near 4,096× but comfortably an order of magnitude. The lesson: the *minimum* you save is the 48× tensor-size factor; the *attention* layers save far more, which is exactly why this trick scales so well as you push to higher resolutions where attention dominates.

This is the punchline figure of the whole post, and it is figure 1: the pixel tensor is enormous, the VAE squeezes it 48× into a latent, the U-Net does all its expensive work on that tiny grid, and a single decode produces the image. Hold that picture; everything else is detail on top of it.

## The autoencoder: a good latent has to be small, smooth, and diffusable

The autoencoder is the foundation the entire model stands on, and getting it right is subtle. You want three things from the latent, and they are in tension.

First, you want it **small** — that is the whole point, the 48× compression. Second, you want it to be a **faithful** compression: the decoder must reconstruct a sharp, detailed image from the latent, or you have just thrown away the very high-frequency detail you wanted to keep in the *final output* (you only wanted to skip diffusing it, not lose it). Third — and this is the part that distinguishes a latent-diffusion autoencoder from a generic one — you want the latent space to be **smooth and well-behaved enough to run diffusion in**. A latent with a wild, spiky, high-variance distribution is hard to add Gaussian noise to and denoise; the noise schedule assumes a roughly normalized signal. So the latent must be regularized toward something Gaussian-ish, but only *gently*, because over-regularizing destroys the reconstruction quality you need.

The standard plain autoencoder loses on the second point: a pixel-MSE reconstruction loss produces blurry images, because MSE in pixel space averages over the high-frequency detail it cannot predict exactly. (This is the same blur that plagues vanilla VAEs — see the [VAE post](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) for why the Gaussian-likelihood objective causes it.) Latent diffusion fixes this with a **perceptual + adversarial** reconstruction loss, borrowed from the VQGAN line of work (Esser et al. 2021). The full autoencoder objective has three parts:

$$\mathcal{L}_\text{AE} = \underbrace{\mathcal{L}_\text{rec}}_{\text{pixel + LPIPS}} \;+\; \lambda_\text{adv}\, \underbrace{\mathcal{L}_\text{adv}}_{\text{patch GAN}} \;+\; \lambda_\text{reg}\, \underbrace{\mathcal{L}_\text{reg}}_{\text{KL or VQ}}.$$

The **reconstruction term** $\mathcal{L}_\text{rec}$ is *not* plain pixel MSE — it combines an L1 or L2 pixel loss with a **perceptual loss (LPIPS)**, which compares deep features of the original and reconstruction through a pretrained VGG network. LPIPS rewards getting the *texture and structure* right, not the exact pixel value, so it does not punish the model for hallucinating plausible high-frequency detail. The **adversarial term** $\mathcal{L}_\text{adv}$ adds a patch-based discriminator that tries to tell reconstructions from real images; training the decoder to fool it pushes the reconstructions to be *sharp and realistic* rather than blurry-but-safe. Together, perceptual + adversarial losses are why the SD VAE can decode a crisp 512×512 image from a 16,384-value latent — far sharper than a pixel-MSE autoencoder at the same bottleneck.

The **regularization term** $\mathcal{L}_\text{reg}$ is the diffusability knob, and latent diffusion offers two flavors. The **KL variant** treats the encoder as producing a Gaussian posterior $q(z|x) = \mathcal{N}(\mu(x), \sigma(x))$ and adds a small KL penalty toward a standard normal prior, exactly like a VAE — but with a *tiny* weight (think $\lambda_\text{reg} \sim 10^{-6}$). This is the crucial detail: the KL is small enough that it barely constrains the latent's content (so reconstruction stays sharp) but large enough to keep the latent's *scale* bounded and roughly centered, so it does not drift to a wild distribution that diffusion cannot handle. The **VQ variant** instead quantizes the latent to a learned codebook (VQ-GAN style), giving a discrete latent; this is the basis of the VQ-regularized LDMs and, later, of token-based models. Stable Diffusion uses the **KL variant** (a continuous latent), which is why you load it with `AutoencoderKL`.

Why pick KL over VQ for a *diffusion* model specifically? Diffusion adds *continuous* Gaussian noise and predicts a *continuous* score; it wants a continuous, smooth latent where small perturbations map to small image changes. A KL-regularized continuous latent is exactly that — locally smooth, no quantization boundaries to fall off. A VQ latent is discrete: every position snaps to one of $K$ codebook vectors, so the latent manifold is a lattice of points, not a smooth space, and adding small Gaussian noise to a code is not naturally meaningful. VQ latents shine when the *downstream model is autoregressive* (predict the next discrete token, like an LLM over image tokens — the VQGAN/Parti/MaskGIT line), because then discreteness is a feature. For diffusion the continuous KL latent is the right match, and that is the historical fork: VQ → autoregressive/masked token models; KL → diffusion. Stable Diffusion sits firmly on the KL branch.

A concrete look at what the latent actually *contains* makes the scaling factor click. After training, the SD KL-VAE's raw latent (before the scaling factor) has a per-channel standard deviation in the rough vicinity of 5 — far from unit scale. If you fed *that* to a noise schedule tuned for unit-variance data, the signal would swamp the noise at most timesteps and the SNR-vs-$t$ curve the U-Net trained against would be wrong. Multiplying by $1/\text{std} \approx 0.18215$ rescales the latent to roughly unit standard deviation, putting it where the noise schedule expects. That is the *entire* origin of the magic constant: it is one over the empirical latent standard deviation, computed once on the training set and frozen. (SDXL's VAE has a different empirical std, hence its different 0.13025.) The four latent channels are not interpretable as "red/green/blue/alpha" — they are a learned, entangled basis — but they *are* roughly decorrelated and unit-scaled after the factor, which is all diffusion needs. If you ever train a custom VAE for diffusion, computing this scaling factor on your own data is a mandatory, easy-to-forget step; skip it and your loss will be mysteriously high and your samples washed out.

Why does a *small* KL keep the latent diffusable while a *large* one would hurt? Diffusion adds noise according to $z_t = \sqrt{\bar\alpha_t}\, z_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ with $\epsilon \sim \mathcal{N}(0,I)$. For this to be well-conditioned, $z_0$ should have a roughly unit-scale, roughly isotropic distribution — otherwise the signal-to-noise ratio at each timestep is mismatched to the schedule the U-Net was trained against. A *large* KL would force $z_0 \approx \mathcal{N}(0,I)$ exactly, which is great for diffusion conditioning but collapses the latent toward the prior and destroys reconstruction (posterior collapse). A *small* KL leaves the latent free to carry rich content but nudges its scale toward order-1, which is "Gaussian enough" for the noise schedule. In practice SD goes one step further: it computes a fixed **scaling factor** (0.18215 for the SD1.x/2.x VAE) and multiplies the latent by it so the latent has approximately unit variance before diffusion. That magic constant you see in every SD codebase is exactly this normalization — it is the bridge between "the VAE's natural latent scale" and "the unit-ish scale the noise schedule assumes."

![A two-stage dataflow figure where a frozen autoencoder encodes an image into a latent, noise is added to the latent at step t, and a U-Net is trained to predict that noise with an MSE loss, all inside the latent space](/imgs/blogs/latent-diffusion-and-stable-diffusion-2.png)

Here is the two-stage structure made concrete (figure 2). **Stage one** trains the autoencoder on its own, with the perceptual + adversarial + small-KL loss above, until it reconstructs images well. Then you **freeze it**. **Stage two** trains the diffusion U-Net entirely inside the frozen latent: encode each training image once to get $z_0$, add noise to get $z_t$, and train $\epsilon_\theta(z_t, t)$ to predict the noise with the same simple MSE objective from ordinary DDPM, just on latents:

$$\mathcal{L}_\text{LDM} = \mathbb{E}_{z_0 = E(x),\, t,\, \epsilon \sim \mathcal{N}(0,I)} \left[ \left\| \epsilon - \epsilon_\theta(z_t, t, c) \right\|^2 \right], \qquad z_t = \sqrt{\bar\alpha_t}\, z_0 + \sqrt{1-\bar\alpha_t}\,\epsilon.$$

This is *literally the DDPM loss* from [the math of DDPM](/blog/machine-learning/image-generation/diffusion-from-first-principles), with $x_0$ replaced by $z_0 = E(x)$ and an optional condition $c$. Nothing about the diffusion math changes — that is the elegance. You just moved the whole process into a smaller, smoother space.

## Why training the two stages separately actually works

The two-stage design looks like it might be a compromise — wouldn't it be better to train the autoencoder and the diffusion model jointly, end to end, so the latent is optimized *for* diffusion? In practice, no, and understanding why is worth a paragraph because it is a recurring pattern in the field.

The autoencoder's job (perceptual compression — discard imperceptible detail, keep a faithful, smooth latent) and the diffusion model's job (semantic modeling — learn the distribution over latents) are **separable objectives** with very different optimization dynamics. The autoencoder is trained adversarially and converges to a good reconstruction relatively quickly and cheaply. The diffusion model is a long, stable MSE regression that benefits from a *fixed* target distribution — if the latent space were shifting underneath it (because you were still training the encoder), the diffusion objective would be chasing a moving target, and the adversarial autoencoder loss would inject instability into what is otherwise a beautifully stable regression. Freezing the autoencoder gives the diffusion model a clean, stationary space to model. It also means you can train *many* diffusion models (different resolutions, different conditioning, fine-tunes) against *one* autoencoder — which is exactly what happened: SD1.1 through 1.5 all share essentially the same VAE, and a huge ecosystem of fine-tunes reuses it. The autoencoder is infrastructure; the U-Net is the model.

There is a deeper reason too. The information the diffusion model needs to generate is the *semantic* information, and the autoencoder has already stripped away the perceptual redundancy that would otherwise waste the diffusion model's capacity. By the time the U-Net sees the latent, every dimension is "worth modeling." A pixel-space model spends enormous capacity learning to denoise sensor grain that nobody will notice; the latent-space model spends none. This is why latent diffusion is not just *faster* but often *better per FLOP* at the semantic task — it is not distracted by the perceptual busywork.

#### Worked example: encode and decode a latent with diffusers

Let's make the latent tangible. The fastest way to internalize the compression is to encode a real image and look at the shapes. With 🤗 `diffusers`, the VAE is an `AutoencoderKL`:

```python
import torch
from diffusers import AutoencoderKL
from diffusers.utils import load_image

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sdxl-vae", torch_dtype=torch.float16
).to("cuda")

# a 1024x1024 RGB image -> tensor in [-1, 1], shape (1, 3, 1024, 1024)
img = load_image("photo.png").resize((1024, 1024))
x = (torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1)
       .unsqueeze(0).half().cuda() / 127.5 - 1.0)

with torch.no_grad():
    posterior = vae.encode(x).latent_dist        # a diagonal Gaussian q(z|x)
    z = posterior.sample()                        # (1, 4, 128, 128)  <-- 8x down
    z = z * vae.config.scaling_factor             # ~0.13025 for SDXL VAE
    print("pixels:", x.numel(), " latent:", z.numel(),
          " ratio:", x.numel() / z.numel())       # ratio ~ 48

    z = z / vae.config.scaling_factor             # undo before decoding
    x_rec = vae.decode(z).sample                  # (1, 3, 1024, 1024)
```

Two things to notice. First, `vae.encode(...).latent_dist` returns a *distribution*, not a tensor — the encoder is variational, outputting a per-position mean and variance, and you `.sample()` (or `.mode()`) to get the latent. That is the KL-VAE structure in code. Second, the `scaling_factor` (0.18215 for SD1.x/2.x, 0.13025 for the SDXL VAE) is multiplied *after* encoding and divided *before* decoding — forget it and your latents are mis-scaled, the noise schedule is wrong, and you get noisy or washed-out output. This is one of the most common "why is my custom pipeline broken" bugs. For a 1024² image the ratio is again 48× (a 4×128×128 latent vs a 3×1024×1024 image: $1024^2 \cdot 3 / (128^2 \cdot 4) = 48$), independent of resolution — the 8× spatial and 3→4 channel factors are fixed.

## The science: why diffusing in latent space is still valid

It is worth being rigorous about *why* you are allowed to do this, because "just run diffusion on the latent" can sound like a hack. It is not — it is a principled change of variables, and the diffusion theory transfers cleanly. Here is the argument in three steps.

**Step 1: diffusion learns a score, and the score is defined on whatever space you noise.** Recall from the [score-based view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view) that a diffusion model is, at heart, a *score estimator*: the denoiser $\epsilon_\theta(x_t, t)$ is (up to a known scaling) an estimate of $\nabla_{x_t} \log p_t(x_t)$, the gradient of the log-density of the noised data at time $t$. The reverse process integrates this score to walk noise back to data. Nothing in this construction cares whether $x$ is a pixel grid or a latent — it only needs a continuous space with a tractable forward-noising process. So if we define our data distribution on the *latent* variable $z = E(x)$, the diffusion model learns $\nabla_{z_t} \log q_t(z_t)$, the score of the *latent* distribution, and the reverse process walks latent-noise back to a clean latent. The math is identical; only the variable's name changed.

**Step 2: the encoder pushes the data distribution forward to a latent distribution.** When you encode every image, you induce a new distribution $q(z) = \int p(x)\, q(z \mid x)\, dx$ — the distribution of latents over the dataset. This is a perfectly well-defined probability distribution on $\mathbb{R}^{64 \times 64 \times 4}$, and it is *exactly* what we want to model. The diffusion model's job is to learn to sample from $q(z)$; then the decoder $D(z)$ maps each sampled latent back to a sample image. As long as the decoder is a good (approximately deterministic, low-distortion) inverse of the encoder on the support of $q(z)$, sampling $z \sim q(z)$ and decoding gives $x \approx p(x)$. We have factored generation into "model the latent distribution" (diffusion) and "decode" (the frozen VAE), and the composition reproduces the data distribution. This is why the two-stage design is *correct*, not just convenient.

**Step 3: the small KL is what makes $q(z)$ actually diffusable.** Here is the one place the choice of regularizer enters the theory. The forward-noising process $q_t(z_t \mid z_0)$ is Gaussian and assumes $z_0$ lives at a controlled scale; the noise schedule $\bar\alpha_t$ was tuned for roughly unit-variance data. If $q(z)$ were heavy-tailed, multimodal at wildly different scales, or had near-zero variance in some dimensions and huge variance in others, the per-timestep signal-to-noise ratio would be mismatched and the denoiser would struggle — at high noise the signal in the small-variance dimensions would be swamped, and at low noise the large-variance dimensions would barely be perturbed. The small KL penalty (toward $\mathcal{N}(0, I)$) plus the `scaling_factor` normalization keep $q(z)$ roughly isotropic and unit-scaled, so a *single* noise schedule works across all latent dimensions. This is the quantitative content of "a small KL keeps the latent diffusable": it is not about making $q(z)$ exactly Gaussian (that would collapse content), but about *conditioning the scale* so the Gaussian forward process is well-matched. A pixel image, conveniently, is already roughly in $[-1,1]$ per channel; the VAE's job is to give the latent the same courtesy.

There is a subtle cost hiding here that is worth naming. The decoder is *not* a perfect inverse — it is lossy. So the achievable image distribution is $D_\#\, q(z)$ (the decoder pushforward of the latent distribution), which is band-limited by the decoder's reconstruction quality. No diffusion model, however good, can produce an image the decoder cannot represent. This is the **reconstruction ceiling** we will keep returning to: the VAE sets an upper bound on final image fidelity, and the diffusion model can only approach it, never exceed it. The whole game of choosing a downsampling factor is balancing this ceiling against the compute savings.

### The downsampling-factor ablation: there is a sweet spot

This balance is the empirical heart of the LDM paper, and it is worth internalizing because it is *the* design knob. Define the downsampling factor $f$: an $f$-autoencoder maps a $512 \times 512$ image to a $(512/f) \times (512/f)$ latent. So $f=4$ gives a $128 \times 128$ latent, $f=8$ gives $64 \times 64$ (Stable Diffusion), $f=16$ gives $32 \times 32$, and so on. There are two opposing forces:

- **Smaller $f$ (less compression):** the latent is large, the reconstruction ceiling is high (the decoder has plenty of capacity to represent detail), but the diffusion model is expensive — at $f=1$ you are back to pixel-space diffusion, slow and capacity-wasting. You keep all the perceptual redundancy that diffusion does not need to model.
- **Larger $f$ (more compression):** the latent is tiny, the diffusion model is cheap and fast, but the reconstruction ceiling drops — at $f=32$ the autoencoder must throw away so much that fine detail becomes unrecoverable, and final FID *worsens* even though the diffusion part is easy. You have over-compressed.

Rombach et al. swept $f \in \{1, 2, 4, 8, 16, 32\}$ and found a clear basin: $f=4$ and $f=8$ are the sweet spot, balancing a high-enough reconstruction ceiling against a small-enough latent for efficient, high-quality diffusion. $f=1$ and $f=2$ are needlessly slow; $f=16$ and $f=32$ start losing to the reconstruction ceiling. Stable Diffusion picked $f=8$ (the more aggressive end of the basin) to maximize the compute savings while staying on the good side of the ceiling — a defensible choice that traded a slightly lower ceiling for the consumer-GPU speed that made it famous. The modern move (SD3/FLUX) keeps $f=8$ but *widens the channels* from 4 to 16, which raises the reconstruction ceiling without shrinking the spatial grid — a different way to climb the same trade-off. The lesson that generalizes: **the autoencoder's compression factor is a first-class hyperparameter of the whole system**, as important as the U-Net's depth, and it has an interior optimum you can find empirically.

#### Worked example: estimating where over-compression bites

Suppose you are designing a new latent-diffusion model and considering $f=8$ (the SD choice, $64^2 \times 4 = 16{,}384$ latent values for a 512² image) versus a more aggressive $f=16$ ($32^2 \times 4 = 4{,}096$ values). The $f=16$ latent is 4× smaller, so the diffusion U-Net is roughly 4× cheaper per step in the convolution terms and ~16× cheaper in the top-resolution attention — tempting. But ask: can a 4-channel $32 \times 32$ latent reconstruct a sharp 512² image? That is 4,096 latent values for 786,432 pixels, a 192× compression. Empirically that is past the basin — reconstructions get noticeably soft, fine detail and small text degrade, and the FID floor rises enough that the cheaper diffusion cannot recover it. The honest engineering call: $f=16$ only pays off if your *images* are low-detail (think simple icons, not photographs), or if you compensate by widening channels (e.g. $32^2 \times 16 = 16{,}384$ values — same total as SD's $f=8$ at 4 channels, but with more channels carrying the detail and fewer spatial positions for attention to chew on). That last option — fewer spatial positions, more channels — is precisely the direction the deep-compression autoencoders (e.g. SANA's 32× AE) push, and it is why they can run linear-attention transformers fast. The number to remember: SD's $f=8$, 4-channel latent is a 48× value compression and sits comfortably in the basin; go much past ~100× total value compression on photographic content and the reconstruction ceiling starts to dominate your final quality.

## Conditioning: cross-attention turns this into text-to-image

So far we have an *unconditional* latent diffusion model — sample noise in latent space, denoise, decode, get a random image from the training distribution. To make it text-to-image, we need to feed it a prompt. Latent diffusion's contribution here, beyond the latent itself, was a clean general conditioning mechanism: **cross-attention** from any conditioning sequence into the U-Net.

The recipe: encode the prompt into a *sequence* of token embeddings with a frozen text encoder (CLIP for SD1.x, OpenCLIP for SD2, both for SDXL), then let the U-Net's spatial features attend to those tokens at every attention block. Mechanically, at each cross-attention layer the latent's spatial positions produce the **queries** $Q$, and the text tokens produce the **keys** $K$ and **values** $V$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V, \qquad Q = W_Q\, \phi(z_t), \;\; K = W_K\, \tau(c), \;\; V = W_V\, \tau(c),$$

where $\phi(z_t)$ are the U-Net's intermediate latent features and $\tau(c)$ is the text encoder's token sequence. Each latent position computes an attention distribution over the prompt tokens and pulls in a weighted mix of their values — so the position that will become the horse can attend to "horse" while the position that will become the moon attends to "moon." This **spatial selectivity** is what lets one image obey a multi-part prompt, and it is also exactly where **attribute-binding failures** live: if attention routes "red" to the wrong object, you get a red moon instead of a red horse. (The [text-encoders post](/blog/machine-learning/image-generation/classifier-free-guidance) and the dedicated conditioning posts go deeper on those failure modes; here we just need the mechanism.)

![A dataflow figure where a prompt is encoded by a CLIP text encoder into 77 tokens that serve as cross-attention keys and values while the noisy latent provides queries, producing a conditioned latent passed to the next U-Net block](/imgs/blogs/latent-diffusion-and-stable-diffusion-3.png)

The generality matters (figure 3). Because conditioning enters through cross-attention on an arbitrary sequence $\tau(c)$, the *same* mechanism handles text, but also class labels, layout maps, or — with a small adapter — images. ControlNet and IP-Adapter, which we cover later in the series, are both bolted onto this cross-attention pathway. Latent diffusion did not just make diffusion cheap; it made it *conditionable* in a modular way, which is half of why the ecosystem exploded.

Two practical notes. First, the text encoder is **frozen** during diffusion training (just like the VAE) — you train only the U-Net's cross-attention projections $W_Q, W_K, W_V$ to read the fixed text embeddings. Second, this conditional model is exactly what [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) operates on: during training you randomly drop the prompt (replace $c$ with the empty string ~10% of the time) so the model learns both the conditional and unconditional scores, and at sampling time you extrapolate between them with the guidance scale. Latent diffusion, cross-attention conditioning, and CFG are the three legs that together make a working text-to-image model; this post is the first leg, and it assumes the third.

## Stable Diffusion 1.x and 2.x: the actual configuration

Latent diffusion was the *method*; Stable Diffusion was the *product* — a specific instantiation, trained on a large filtered subset of LAION-5B, released openly by Stability AI / CompVis / Runway in 2022. The configurations are worth knowing precisely because they pin down what "an SD model" actually is and they explain a lot of community lore.

**Stable Diffusion 1.x** (1.1 → 1.4 → 1.5, the last being the most-used checkpoint in history). The denoiser is a `UNet2DConditionModel` with about **860M parameters**, operating on a $4 \times 64 \times 64$ latent from the KL-VAE (scaling factor 0.18215), at a native training resolution of **512×512**. The text encoder is **CLIP ViT-L/14** (from OpenAI's CLIP), which encodes a prompt into a $77 \times 768$ sequence (77 tokens, 768-dim). The prediction target is **ε-prediction** (predict the added noise). The whole model — VAE + U-Net + text encoder — is about 1.06B parameters and fits comfortably in fp16 on a consumer GPU. SD1.5 was the moment image generation became a commodity: trainable LoRAs, the DreamBooth craze, ControlNet, the entire ecosystem, was built on this checkpoint. Its weaknesses are real (512² native resolution shows; faces and hands are rough; it inherits CLIP's text-encoding limits), but its accessibility was decisive.

**Stable Diffusion 2.x** changed two things that are instructive. First, it swapped the text encoder from OpenAI's CLIP ViT-L to **OpenCLIP ViT-H/14** (a larger, open-data CLIP from LAION), encoding to a $77 \times 1024$ sequence. The motivation was openness and a stronger encoder, but in practice many users found SD2's prompt-following *different* and sometimes harder to control — partly because the new encoder had been trained with an aggressive NSFW filter on the data, which changed what concepts it represented well. Second, SD2 (the 768 model) switched the prediction target to **v-prediction** and trained at **768×768** native resolution. V-prediction predicts a velocity-like target $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$ instead of the raw noise $\epsilon$; it has better-behaved signal-to-noise properties across timesteps and pairs well with zero-terminal-SNR fixes (covered in the [noise-schedules post](/blog/machine-learning/image-generation/noise-schedules-and-the-parameterization-zoo)). The community reaction to SD2 is a useful lesson: a "better" encoder and a "better" parameterization do not automatically produce a model people prefer — the data filtering and the shift in prompt behavior mattered more to users than the cleaner math. SDXL would later get the encoder story right by *adding* rather than replacing.

Here is the minimal SD1.5 generation call so the config is not abstract — the practical flow you would actually run:

```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
# a fast, high-quality multistep ODE sampler instead of the default
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

image = pipe(
    prompt="an astronaut riding a horse on the moon, dramatic lighting",
    negative_prompt="blurry, low quality, extra limbs",
    num_inference_steps=25,      # DPM-Solver++ converges by ~20-25 steps
    guidance_scale=7.5,          # classifier-free guidance scale
    height=512, width=512,
).images[0]
image.save("astronaut.png")
```

That is the entire surface a user touches: a prompt, a negative prompt, a step count, a guidance scale, a resolution. Under it sits the whole machine — CLIP encodes the prompt to $77 \times 768$, the U-Net denoises a $4 \times 64 \times 64$ latent 25 times under DPM-Solver++ with CFG, and the VAE decodes once. The 48× latent compression is the only reason these five lines run in a couple of seconds on a 4090 instead of needing a datacenter.

To see the latent mechanics explicitly — and to prove there is no magic in the pipeline — here is the same generation written as a manual loop over the three components (text encoder, U-Net, VAE). This is the actual flow `StableDiffusionPipeline.__call__` runs, unrolled:

```python
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

repo = "runwayml/stable-diffusion-v1-5"
dev, dt = "cuda", torch.float16
tok  = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer")
te   = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder",  torch_dtype=dt).to(dev)
unet = UNet2DConditionModel.from_pretrained(repo, subfolder="unet",   torch_dtype=dt).to(dev)
vae  = AutoencoderKL.from_pretrained(repo, subfolder="vae",           torch_dtype=dt).to(dev)
sched = DDIMScheduler.from_pretrained(repo, subfolder="scheduler")

def embed(p):
    ids = tok(p, padding="max_length", max_length=77, truncation=True,
              return_tensors="pt").input_ids.to(dev)
    return te(ids)[0]                                  # (1, 77, 768)

cond, uncond = embed("an astronaut riding a horse"), embed("")
ctx = torch.cat([uncond, cond])                        # batch the two CFG branches

sched.set_timesteps(25)
z = torch.randn(1, 4, 64, 64, device=dev, dtype=dt)    # the latent, 16384 values
z = z * sched.init_noise_sigma
for t in sched.timesteps:
    zin = sched.scale_model_input(torch.cat([z, z]), t)         # duplicate for CFG
    with torch.no_grad():
        eps_u, eps_c = unet(zin, t, encoder_hidden_states=ctx).sample.chunk(2)
    eps = eps_u + 7.5 * (eps_c - eps_u)                 # classifier-free guidance
    z = sched.step(eps, t, z).prev_sample               # one denoise step, in latent

with torch.no_grad():
    img = vae.decode(z / vae.config.scaling_factor).sample   # decode once, at the end
```

Read the loop carefully and the whole post is in it: the U-Net is called 25 times on a $4 \times 64 \times 64$ latent (the expensive part, all in latent space), CFG runs both branches in one batched call, and `vae.decode` fires exactly *once* after the loop — the single pixel-space operation in the entire generation. The `scaling_factor` division before decode is the same normalization from earlier; drop it and the decode sees mis-scaled latents and returns garbage. Every modern pipeline, SDXL included, is a variation on this skeleton — the only differences are two text encoders, a wider latent, and the micro-conditioning kwargs we get to next.

#### Worked example: FLOPs and VRAM, pixel-space vs latent

Put hard numbers on "intractable becomes consumer-GPU." Take a U-Net whose dominant cost at the top resolution is self-attention. At 512², pixel-space attention runs over $512^2 = 262{,}144$ tokens; the attention score matrix alone is $262{,}144^2 \approx 6.9 \times 10^{10}$ entries — *per head, per layer*. Materializing that in fp16 is on the order of 100+ GB for a single attention map, which is why nobody ran full-resolution pixel attention at 512²; it does not fit on any GPU. The latent U-Net attends over $64^2 = 4{,}096$ tokens, a $4{,}096^2 \approx 1.7 \times 10^7$ score matrix, about 34 MB — roughly four thousand times smaller, and trivially resident. That single ratio is the difference between "needs a research cluster and exotic checkpointing tricks" and "fits in 8 GB on a 4090." For the full forward pass (convolutions plus attention across all resolutions), a 512² SD1.5 U-Net step is on the order of a few hundred GFLOPs and peaks under ~4 GB of activation memory in fp16 with attention slicing; the hypothetical pixel-space equivalent at the same architecture is one to two orders of magnitude more FLOPs and does not fit in consumer VRAM at all. Scale to 1024²: the latent is $4 \times 128 \times 128$ (still only 65,536 values, 4× the 512² latent), SDXL runs in ~10–16 GB with CPU offload, and a single image costs a few seconds on a 4090. Frame it in money: at a rented A100 around \$2/hr and ~3 s/image, that is roughly \$0.0017 per image — fractions of a cent. The pixel-space counterpart, if it ran at all, would be far more expensive and require a much larger, much pricier accelerator. The 48× value compression is what moved image generation from a capex line item to a rounding error.

## SDXL: bigger, dual encoders, and the conditioning fixes that mattered

By 2023 the SD recipe was mature enough to scale deliberately, and SDXL (Podell et al. 2023) is the carefully-engineered version. It kept the latent-diffusion core unchanged — same KL-VAE family, same cross-attention conditioning, same ε-prediction — and pushed on four axes: a much bigger U-Net, two text encoders, micro-conditioning, and an optional refiner.

**The bigger U-Net.** SDXL's base U-Net is about **2.6B parameters** — roughly 3× SD1.5's 860M. The architecture is rebalanced: SDXL moves transformer (attention) blocks to the lower-resolution stages where they are cheaper per-position and more effective, using a heterogeneous distribution of attention across resolutions rather than SD1.5's more uniform layout. It also trains at a native **1024×1024**, which (at the same 8× VAE downsample) means a $4 \times 128 \times 128$ latent. The bigger network plus higher resolution is most of why SDXL's images are sharper and more coherent.

**Dual text encoders.** This is SDXL's cleverest conditioning choice. Instead of *replacing* SD1.5's CLIP with a bigger one (the SD2 mistake), SDXL uses **both** — CLIP ViT-L (the SD1.5 encoder) *and* the larger **OpenCLIP ViT-bigG** — and concatenates their token outputs into a $77 \times 2048$ sequence for cross-attention. It additionally pools the bigG output into a single vector that feeds the model FiLM-style alongside the micro-conditioning embeddings (below). Keeping CLIP-L preserves the prompt behavior the community had learned, while bigG adds capacity and better concept coverage. When you run classifier-free guidance on SDXL, the *unconditional* branch must drop **both** encoders' tokens and the pooled vector consistently — null the bigG tokens but forget to null CLIP-L and the difference vector $\epsilon_\text{cond} - \epsilon_\text{uncond}$ is corrupted and guidance misbehaves. `diffusers` handles this for you, but hand-rolled SDXL CFG must null *every* conditioning input.

![A matrix comparing Stable Diffusion 1.5, 2.1, and SDXL across U-Net parameters, text encoder, native resolution, prediction target, and headline quality](/imgs/blogs/latent-diffusion-and-stable-diffusion-4.png)

Figure 4 lays the three generations side by side. The pattern is clear: the latent-diffusion recipe held constant while the U-Net grew (860M → 2.6B), the encoder evolved (CLIP-L → OpenCLIP-H → CLIP-L+bigG), and native resolution climbed (512 → 768 → 1024). What did *not* change — the latent space, the cross-attention conditioning, the basic denoising objective — is the load-bearing part that justifies calling all three "latent diffusion."

### Micro-conditioning: the trick that fixed cropping and low-res blur

SDXL's most underappreciated contribution is **micro-conditioning**, and it is a genuinely clever fix to a data problem that had quietly hurt every prior model. The problem: when you train on a large web-scraped image dataset, you have images of every resolution and aspect ratio. The standard pipeline resizes and **random-crops** them to a fixed training size (512² or 1024²). This does two damaging things. First, **random crops cut off subjects** — a centered portrait, after a random crop, might lose the top of the head, and the model learns that "people sometimes have their heads cut off," which then shows up in samples as awkwardly cropped compositions. Second, models trained on a mix of native-resolution and upscaled-small images learn an *average* sharpness; if you discard all sub-512 images you throw away a huge fraction of your data, but if you keep and upscale them, the model learns some blur. Both problems come from the model not *knowing* the provenance of each training image.

SDXL's fix: **tell the model**. During training, SDXL conditions on two extra signals (encoded as Fourier-feature embeddings added FiLM-style alongside the pooled text vector):

- `original_size` $= (h_\text{orig}, w_\text{orig})$ — the resolution of the source image *before* any resizing. So a small upscaled image is labeled as small, and a native-high-res image is labeled large. The model learns that low `original_size` correlates with blur, and high `original_size` with sharpness.
- `crops_coords_top_left` $= (c_\text{top}, c_\text{left})$ — the pixel coordinates of the top-left corner of the crop that was taken. So a crop that cut off the top of someone's head is labeled with its actual offset, and the model learns that nonzero crop coordinates correlate with cut-off subjects.

How are these integers actually fed to the network? Each scalar (a height, a width, a crop offset) is mapped through a **sinusoidal Fourier embedding** — the same positional-encoding trick used for diffusion timesteps. For a scalar value $s$ and embedding dimension $d$, you compute $\gamma(s) = [\sin(s\,\omega_1), \cos(s\,\omega_1), \dots, \sin(s\,\omega_{d/2}), \cos(s\,\omega_{d/2})]$ with a geometric series of frequencies $\omega_k$, exactly as in $\gamma(t)$ for the timestep. The four scalars (original height, original width, crop-top, crop-left — and at training the two target-size scalars) each get their own $\gamma(\cdot)$ vector; these are concatenated and passed through a small MLP, and the result is **added to the timestep embedding**. So the micro-conditioning rides the *same* additive FiLM channel that already carries $t$ and the pooled text vector — it modulates every ResBlock's scale-and-shift, globally, at every step. There is no architectural surgery: SDXL just widens the conditioning vector that was already there. That is why micro-conditioning was nearly free to add and why it composes cleanly with everything else.

Why does conditioning on the *defect* fix the *symptom*? Because it breaks a spurious correlation the model would otherwise bake in. Without the crop coordinate, "subject cut off at the top" is an unexplained, irreducible feature of ~20% of training images, so the model learns to reproduce it ~20% of the time unconditionally. *With* the coordinate, "cut off at the top" becomes *explained* by `crop-top > 0` — the model learns a conditional rule ("cut off iff the crop offset is nonzero") instead of an unconditional habit. At inference you set the offset to zero and the cut-off behavior simply does not fire. The same logic applies to `original_size`: low-res blur becomes explained by small `original_size` rather than being an averaged-in property of all outputs, so requesting a large `original_size` requests the sharp mode. This is the general principle of *conditioning on a nuisance variable to control it* — and it is the cleanest fix in the whole SD lineage for a problem (cropped, soft, badly-framed outputs) that had quietly degraded every prior model.

At **inference**, you simply *ask for what you want*: set `original_size = (1024, 1024)` to request a sharp, native-resolution image, and set `crops_coords_top_left = (0, 0)` to request a centered, uncropped composition. You have decoupled "what was true in the messy training data" from "what I want at test time." This is a beautiful piece of engineering: instead of cleaning the data perfectly (impossible at web scale), you condition on the data's imperfections and then steer around them at sampling time.

There is a companion training trick that pairs with micro-conditioning and explains why SDXL handles non-square aspect ratios so much better than SD1.5: **aspect-ratio bucketing.** Rather than force every training image into a square 1024×1024 crop (which is what mangled portraits and landscapes in the first place), SDXL sorts images into a set of buckets of *equal total pixel area* but different aspect ratios — for example 1024×1024, 1152×896, 896×1152, 1216×832, 832×1216, and so on, all near $\sim$1.05M pixels so the latent grids stay a similar size for efficient batching. Each batch is drawn from a single bucket so all latents in it share a shape. The effect: SDXL *sees* wide and tall images during training at their natural aspect ratio, rather than only center-cropped squares, so it learns to compose for 16:9 and 9:16 frames natively. Combined with the crop-coordinate conditioning, this is why you can ask SDXL for a 1216×832 cinematic landscape and get a sensibly-composed one, where SD1.5 (square-only training) would awkwardly cram or duplicate the subject. Bucketing is a *data-pipeline* change, not a model change — another instance of the SDXL philosophy that the cheapest big wins come from treating the data honestly rather than enlarging the network.

![A matrix showing SDXL micro-conditioning inputs, what each encodes, which artifact it fixes, the train-time value, and the test-time value you request](/imgs/blogs/latent-diffusion-and-stable-diffusion-5.png)

Figure 5 lays out the three micro-conditioning inputs (`original_size`, `crops_coords_top_left`, and `target_size`) and what to pass at test time. In `diffusers`, these are first-class pipeline kwargs:

```python
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
).to("cuda")
pipe.enable_model_cpu_offload()   # fit on a 12-16 GB GPU

image = pipe(
    prompt="a studio portrait of a corgi astronaut, 85mm, soft light",
    negative_prompt="cropped, blurry, jpeg artifacts",
    num_inference_steps=30,
    guidance_scale=7.0,
    # --- micro-conditioning: ask for sharp, centered, full-frame ---
    original_size=(1024, 1024),
    crops_coords_top_left=(0, 0),
    target_size=(1024, 1024),
).images[0]
image.save("corgi.png")
```

Drop the micro-conditioning kwargs and `diffusers` defaults them sensibly, but if you *deliberately* set `original_size=(256, 256)` you can watch the model produce a softer, lower-fidelity image on demand — a direct, visible confirmation that the conditioning works. Set `crops_coords_top_left=(256, 0)` and you can make it generate as if the frame were cropped downward. The control is real and steerable (figure 6 below contrasts the two regimes).

![A before-and-after figure contrasting SDXL output without micro-conditioning, with cropped subjects and soft detail, against output with size and crop conditioning that is centered and sharp](/imgs/blogs/latent-diffusion-and-stable-diffusion-6.png)

### The refiner: a second specialist for the last few steps

SDXL ships as two models: the **base** (2.6B) and an optional **refiner** (~6.6B but used only for a fraction of the steps). The refiner is a separate latent-diffusion model trained to specialize in the *low-noise, high-detail* end of the denoising trajectory — it operates in the same latent space and is meant to take over for the final ~20% of denoising steps to add fine texture. The two-model "ensemble of experts" flow runs the base for the high-noise steps (composition, layout) and hands the latent to the refiner for the low-noise steps (detail). In `diffusers` you express this with `denoising_end` on the base and `denoising_start` on the refiner:

```python
from diffusers import StableDiffusionXLImg2ImgPipeline

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
).to("cuda")

# base produces the latent for the first 80% of the schedule...
latent = pipe(prompt=prompt, num_inference_steps=40,
              denoising_end=0.8, output_type="latent").images
# ...refiner finishes the last 20% in the same latent space
image = refiner(prompt=prompt, num_inference_steps=40,
                denoising_start=0.8, image=latent).images[0]
```

Honest assessment: the refiner gives a modest detail bump and is frequently *skipped* in practice because it adds latency and VRAM for a small gain, and many fine-tuned SDXL checkpoints subsume its benefit. Worth knowing it exists; not worth always running it. It is a good example of a feature that is real but optional — the base model is where almost all the value is.

## The VAE in production: latent artifacts and the "SDXL VAE fix"

The autoencoder is infrastructure, but infrastructure breaks, and the SD VAE has two failure modes that every practitioner eventually meets. Both come from the same root cause: the VAE is a *learned, lossy* codec, and it has blind spots.

**Failure mode 1: latent / decode artifacts.** Because the latent is only 4 channels and the decoder is finite-capacity, certain content decodes badly. The classic symptoms are subtle color shifts, a faint grid or checkerboard texture in flat regions, and a characteristic loss of fine high-frequency detail (tiny text, dense patterns, fine fur) that the 48× compression simply cannot represent. The VAE is the *reconstruction ceiling* of the whole system: no matter how good your U-Net is, the final image can be no sharper than what the VAE can decode. When you see an SD image that is great except for mangled tiny text or a weirdly smooth patch, that is usually the VAE, not the U-Net.

A couple of these artifacts have specific, recognizable signatures worth naming because they tell you *where* to fix the problem. The **checkerboard / grid texture** in flat regions (a clear sky, a plain wall) traces to the decoder's transposed convolutions — upsampling convolutions can leave a periodic stamp, and the SD VAE's adversarial training mostly but not entirely suppresses it; it shows up most on large smooth areas where there is no detail to hide it. The **slight global color or contrast drift** — outputs that come out a touch desaturated or with a faint tint compared to the prompt's intent — comes from the latent's color subspace being imperfectly calibrated, and it is one reason people sometimes apply a small post-decode color correction or swap in a VAE tuned for color fidelity. The **fine-detail collapse** (tiny faces in a crowd, distant text, a watch face) is the pure reconstruction-ceiling failure: 16,384 latent values cannot encode a thousand legible glyphs, full stop. The fix for the first two is often a better-trained VAE (the SDXL VAE is noticeably cleaner than the SD1.5 one on flat regions); the fix for the third is a *wider* latent (more channels), which is the 16-channel move SD3 and FLUX made. Knowing the signature tells you which lever to pull — decoder retrain versus latent width — instead of blaming the U-Net or the prompt.

**Failure mode 2: the fp16 NaN problem — the "SDXL VAE fix."** The original SDXL VAE has activations that, for some inputs, exceed the range of **float16** and overflow to NaN or Inf during the decode, producing black images or colored garbage. This bit a huge number of early SDXL users running in fp16 (the default on consumer GPUs). The community fix, later adopted officially, was a **re-trained / numerically-stabilized VAE** (`madebyollin/sdxl-vae-fp16-fix`) whose activations stay in fp16 range, plus the workaround of running just the VAE in fp32 or bf16 while the U-Net stays in fp16:

```python
from diffusers import AutoencoderKL

# option A: the community fp16-safe VAE
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
pipe.vae = vae.to("cuda")

# option B: keep the original VAE but decode in fp32 (slower, always safe)
pipe.vae.to(dtype=torch.float32)
# or upcast just the problematic path:
pipe.upcast_vae()   # diffusers helper that runs the VAE in fp32
```

This is the single most common "my SDXL outputs are black" bug, and the fix is one of `pipe.upcast_vae()`, swapping in the fp16-fix VAE, or decoding in bf16. Knowing it saves an afternoon. The broader lesson: the VAE is a *separate trained component* with its own numerics, and you can — and sometimes must — swap it independently of the U-Net. People routinely swap in alternative VAEs (the SDXL "fp16-fix," or higher-channel VAEs in newer models) to fix artifacts without retraining the diffusion model at all. That modularity is a direct gift of the frozen two-stage design.

#### Worked example: when the VAE is the bottleneck

Suppose you generate a 1024×1024 SDXL image of a "newspaper front page with a headline," and the layout, the photo, the column structure all look great — but the headline text is gibberish. Is the U-Net failing to model text, or is the VAE failing to decode it? Quick test: take a *real* newspaper image, run it through `vae.encode` then `vae.decode` (no diffusion at all), and look at the text. If the round-tripped real image *also* has mangled text, the VAE is your ceiling — the 4-channel latent cannot represent fine glyph detail at that size, and no amount of better prompting or more steps will fix it. If the round-trip is clean but generation is not, the U-Net (or the text encoder's tokenization of the letters) is the culprit. In my experience the VAE round-trip is the limiter for tiny text far more often than people assume; it is *the* reason early SD models could not write, and it is why newer models (SD3, FLUX) moved to **higher-channel VAEs** (16 latent channels instead of 4) — more latent capacity, a higher reconstruction ceiling, better fine detail and text. That single change — 4→16 channels — is one of the biggest quiet quality jumps between the SD-era and the 2024+ frontier, and it is *entirely* a VAE decision, not a U-Net one. The latent is still the lever; widening it was the next move.

## Putting numbers on it: the comparison tables

Let me consolidate the measured story. First, the model-config comparison across the three SD generations (params and resolution are from the model cards / papers; FID figures are approximate and dataset-dependent, so treat them as directional, not precise):

| Model | U-Net params | Text encoder(s) | Native res | Latent | Pred. target | Quality / notes |
|---|---|---|---|---|---|---|
| SD 1.5 | ~860M | CLIP ViT-L/14 ($77\times768$) | 512² | $4\times64\times64$ | ε-pred | The commodity checkpoint; vast ecosystem |
| SD 2.1 | ~865M | OpenCLIP ViT-H/14 ($77\times1024$) | 768² | $4\times96\times96$ | v-pred (768) | Cleaner math, mixed reception |
| SDXL base | ~2.6B | CLIP-L + OpenCLIP-bigG ($77\times2048$) | 1024² | $4\times128\times128$ | ε-pred | Sharper, preferred in human eval |
| SDXL + refiner | ~2.6B + ~6.6B | same | 1024² | $4\times128\times128$ | ε-pred | Modest detail gain, often skipped |

Second — and this is the table that justifies the whole post — pixel-space versus latent-space diffusion *compute per step*, for a 512×512 target:

| Quantity | Pixel-space diffusion | Latent-space (SD) | Ratio |
|---|---|---|---|
| Denoiser input tensor | $512\times512\times3 = 786{,}432$ | $64\times64\times4 = 16{,}384$ | **48× smaller** |
| Spatial positions (top res) | $262{,}144$ | $4{,}096$ | 64× fewer |
| Top-res self-attention cost | $\propto 262{,}144^2$ | $\propto 4{,}096^2$ | ~4,096× cheaper |
| Realized end-to-end speedup | baseline | — | ~order of magnitude |
| VAE encode/decode overhead | none | one encode + one decode | amortized once |

The honest framing: the *tensor* is 48× smaller; the *attention* layers are far cheaper than that; the *realized* end-to-end speedup (which mixes convolution, attention, multiple resolutions, and the one-time VAE cost) is "roughly an order of magnitude," which is what Rombach et al. report and what made consumer-GPU training and inference possible. The VAE adds a fixed encode (at training time, once per image) and one decode (at sample time, once per image) — negligible against 25–50 U-Net steps.

![A before-and-after figure contrasting pixel-space diffusion operating on 786432 values per step against latent-space diffusion on 16384 values, a 48x cut in per-step compute](/imgs/blogs/latent-diffusion-and-stable-diffusion-7.png)

Figure 7 is the cost contrast in one frame: pixel-space diffusion pays for the full 786,432-value grid every single step and needs a datacenter; latent-space diffusion pays for 16,384 values and runs on the GPU you already own. That 48× is not a micro-optimization — it is the difference between "a research artifact" and "a thing millions of people run." How would you *measure* this honestly if you were verifying the claim? Fix everything except the diffusion space: same U-Net architecture, same step count (say 50 DDIM steps), same batch, same GPU (an A100 80GB, warmed up, fp16), and time a single denoising step on a 512² pixel tensor versus a 64² latent. Report median over 100 steps after a warm-up to exclude compilation. The ratio you measure will land in the order-of-magnitude range, with the gap *widening* as you raise the target resolution (because attention's quadratic term grows). Never report a single un-warmed timing; the first call includes kernel compilation and lies.

## Case studies: real numbers from the literature

A few concrete, citable results to anchor the claims (numbers are from the respective papers; where a figure is dataset-specific or approximate I say so).

**Latent Diffusion (Rombach et al., 2022).** On class-conditional ImageNet 256×256, the LDM with a moderate downsampling factor (their LDM-4, an $f=4$ autoencoder) reaches competitive FID while training and sampling roughly an order of magnitude cheaper than comparable pixel-space diffusion models like ADM. Their central ablation is the **downsampling-factor study**: too little compression ($f=1$, basically pixel-space) is slow and gains nothing; too much ($f=32$) over-compresses and the reconstruction ceiling hurts final FID; the sweet spot is $f=4$ to $f=8$. That ablation is the empirical core of "the latent is the lever" — there is an optimum, and both SD ($f=8$) and the LDM paper ($f=4$) sit near it. The paper also showed LDM matching or beating GANs (which had dominated high-res synthesis) while keeping diffusion's mode coverage — a direct hit on the generative trilemma.

**Stable Diffusion 1.5 (CompVis / Runway, 2022).** Trained on a filtered LAION subset at 512², ~860M U-Net, CLIP-L text encoder, the full checkpoint ~1B params in fp16. The headline practical fact is not an FID number but a *deployment* number: it runs a 512² image in a few seconds on a single consumer GPU (e.g. ~2–4 s for 25 DPM-Solver++ steps on an RTX 4090, fp16), in well under 8 GB of VRAM with attention slicing. That accessibility — not a benchmark — is what opened the floodgates.

**SDXL (Podell et al., 2023).** ~2.6B base U-Net, dual CLIP-L + OpenCLIP-bigG encoders, 1024² native, micro-conditioning, optional 6.6B refiner. In the paper's human-preference study, SDXL is preferred over SD1.5 and SD2.1 by a large margin, and over Midjourney v5.1 on a meaningful fraction of prompts — the gains attributed to the bigger U-Net, the dual encoders, and the micro-conditioning fixes (which specifically removed the cropped-subject and soft-detail artifacts). FID alone *understates* SDXL's gains because FID does not capture prompt-following or framing; human preference and CLIP-score do, which is why the paper leans on them. (This is a recurring eval lesson covered in the [evaluation post](/blog/machine-learning/image-generation/why-generating-images-is-hard) line of the series.)

**The refiner's marginal contribution.** The SDXL paper's own human-preference numbers show the base+refiner ensemble preferred over base-only — but by a *modest* margin, and concentrated on close-up, high-detail prompts (skin texture, foliage, fabric) where the last few low-noise steps matter most. On typical mid-shot or compositional prompts the difference is hard to see, which is exactly why production setups so often skip it: it adds a second ~6.6B model to load (more VRAM) and a second short denoising pass (more latency) for a gain that many users cannot pick out in a blind test. The honest cost-benefit: the base U-Net captures essentially all of the layout, prompt-following, and most of the detail; the refiner is a polish pass that earns its keep only when you are producing a hero asset and have the VRAM to spare. As a measurement discipline, *always A/B the refiner on your own prompt distribution with a fixed seed before adopting it in a pipeline* — do not assume the paper's average gain transfers to your use case, because the gain is so prompt-dependent. This is the same lesson as the whole post in miniature: the latent and the base model are where the structural wins live; the refiner is a small, optional increment bolted on at the low-noise end.

**The higher-channel-VAE jump (SD3 / FLUX, 2024).** Both moved from a 4-channel to a **16-channel** VAE latent. The reported effect is a meaningfully higher reconstruction ceiling — better fine detail, far better small-text rendering — at the cost of a larger latent (4× the channels, so the diffusion model processes more per position). This is the clearest modern demonstration that the *latent design* is still the dominant lever: a VAE channel change, with the diffusion architecture also evolving (to MM-DiT), produced one of the visible quality jumps of the SD3/FLUX era. We cover the backbone change in [diffusion transformers](/blog/machine-learning/image-generation/diffusion-from-first-principles)-adjacent posts; the VAE half of the story starts here.

## When to reach for each model (and when not to)

A decisive recommendation section, because every choice is a cost.

**Use SD1.5 when** you need speed, a small footprint, or the enormous SD1.5 ecosystem (a specific LoRA, a ControlNet, a fine-tune that only exists for 1.5). It runs in under 8 GB, samples in ~2 s, and has more community assets than anything else. *Don't* use it when you need 1024² native sharpness, good hands, or strong prompt-following on complex multi-object prompts — it will fight you.

**Use SDXL when** you want the best open quality-per-dollar in the SD family: 1024² native, much better composition and detail, micro-conditioning for framing control, and a still-large (if smaller than 1.5's) ecosystem. It needs ~10–16 GB in fp16 with CPU offload. *Don't* reach for the **refiner** by default — it adds VRAM and latency for a modest gain; run base-only first and add the refiner only if you can see the difference on your prompts. And *don't* run the original SDXL VAE in fp16 without the fix — you will get black images; use `upcast_vae()` or the fp16-fix VAE.

**Skip SD2.x** for most new work — it sits awkwardly between 1.5's ecosystem and SDXL's quality, and its prompt behavior surprised enough people that the community largely moved past it. It is worth understanding (v-prediction, OpenCLIP) but rarely worth deploying.

**Reach past all of these to SD3 / FLUX when** you need the best text rendering, the highest fidelity, and the 16-channel-VAE / MM-DiT recipe — at the cost of more VRAM and (for FLUX) a more restrictive license on some variants. Those are the frontier posts; the point here is that they are *still latent diffusion* — the lever did not change, the backbone and the latent width did.

**Use a distilled / Turbo model when latency is the constraint, not absolute quality.** SDXL-Turbo (ADD-distilled) and the LCM-LoRA route can produce a usable image in **1 to 4 steps** instead of 25–40 — a 10× wall-clock win — by distilling the multi-step denoising trajectory (and often the guidance) into a few-step student. The trade is real: 1-step output is slightly softer, less diverse, and less precisely steerable than a 30-step base SDXL sample, and high guidance scales no longer apply the same way (the distilled model bakes a fixed guidance in). The decision rule is clean. If you are generating interactively (a live canvas, a real-time preview, a high-volume API where p99 latency and cost dominate), reach for a distilled few-step model — the quality gap is invisible to most users and the latency win is enormous. If you are generating a hero asset where every detail and the full diversity matter, run the full-step base (or base+refiner) and pay the seconds. And note what *does not* change: the distilled model **still runs in the same VAE latent space** — distillation attacks the *number of steps* (one lever of the trilemma's speed face), while latent diffusion already paid down the *cost per step* (the other). They compose. A latent-space, few-step, distilled SDXL is the cheapest path to a good image that exists in the open ecosystem today, and it is cheap precisely because both levers — small latent *and* few steps — are pulled at once. The dedicated distillation posts in this series cover the consistency and DMD math; here the point is just the recommendation, and that it sits *on top of* the latent, not instead of it.

**A few hard "don'ts" that follow from this post:**

- **Don't forget the VAE `scaling_factor`** in any custom pipeline — mis-scaled latents are the most common silent corruption.
- **Don't run pixel-space diffusion** for high-res text-to-image in 2026. There is no scenario where it wins; the latent is strictly better for this task. (Pixel-space *super-resolution* cascades are a different, niche story.)
- **Don't expect the VAE to render fine text or dense fine patterns** at 4 channels — that is a reconstruction-ceiling problem, not a prompting problem; reach for a 16-channel-VAE model instead.
- **Don't co-train the autoencoder and the diffusion U-Net** — freeze the AE first; joint training destabilizes the regression for no benefit.

![A timeline from Latent Diffusion in 2022 through Stable Diffusion 1.5, SD2 with OpenCLIP and v-prediction, SDXL with dual encoders, and onward to the DiT and flow-matching frontier that still operates in latent space](/imgs/blogs/latent-diffusion-and-stable-diffusion-8.png)

Figure 8 traces the lineage and makes the closing point concrete: latent diffusion did not get *replaced* by the frontier — it got *absorbed* into it. SD1.5, SD2, SDXL, and then DiT-based SD3 and FLUX all run diffusion in a compressed VAE latent. The backbone changed (U-Net → transformer), the conditioning got richer (CLIP → CLIP+T5/LLM), the latent got wider (4 → 16 channels), and the objective evolved (ε-pred → v-pred → flow matching). But the one decision that opened the floodgates — *don't diffuse in pixels* — is still the foundation every modern open text-to-image model is built on. That is what it means to be the single biggest efficiency lever in the stack.

## Key takeaways

1. **Latent space is the biggest efficiency lever in image generation.** Running diffusion on a 64×64×4 latent instead of 512×512×3 pixels is a 48× tensor reduction and roughly an order-of-magnitude end-to-end speedup — the reason Stable Diffusion runs on consumer GPUs.
2. **The win comes from separating perceptual from semantic compression.** A frozen autoencoder discards imperceptible high-frequency detail once; the diffusion model then spends all its capacity on the semantic modeling, never on perceptual busywork.
3. **The autoencoder needs a perceptual+adversarial loss and a *small* KL.** LPIPS + patch-GAN losses keep reconstructions sharp; a tiny KL (and the `scaling_factor`) keeps the latent unit-scaled and diffusable without collapsing it.
4. **Train the two stages separately.** Freezing the AE gives the diffusion regression a stationary target and lets one AE serve many U-Nets — the modularity behind the whole fine-tuning ecosystem.
5. **Cross-attention is the conditioning mechanism.** Text tokens become keys/values, latent positions become queries; this same pathway carries text, ControlNet, and IP-Adapter, and it is where attribute-binding failures live.
6. **The SD generations changed the periphery, not the core.** SD1.5→SD2→SDXL grew the U-Net (860M→2.6B), evolved the encoder (CLIP-L→OpenCLIP→both), and raised resolution (512→1024) — but kept the latent, cross-attention, and denoising objective.
7. **SDXL's micro-conditioning is the underrated fix.** Conditioning on `original_size` and `crops_coords_top_left` lets the model learn from messy crops yet be steered to sharp, centered, full-frame output at test time.
8. **The VAE is a separate, swappable component with real failure modes.** Latent/decode artifacts set the reconstruction ceiling; the fp16 NaN bug needs `upcast_vae()` or the fp16-fix VAE; widening the latent to 16 channels (SD3/FLUX) was a major quiet quality jump.

## Further reading

- **Rombach, Blattmann, Lorenz, Esser, Ommer (2022), "High-Resolution Image Synthesis with Latent Diffusion Models"** — the LDM paper; the perceptual-compression argument, the downsampling-factor ablation, the cross-attention conditioning. The foundation of this entire post.
- **Esser, Rombach, Ommer (2021), "Taming Transformers for High-Resolution Image Synthesis" (VQGAN)** — where the perceptual+adversarial autoencoder loss comes from.
- **Podell et al. (2023), "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis"** — the dual encoders, micro-conditioning, refiner, and human-preference results.
- **Esser et al. (2024), "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3)** — the 16-channel VAE and the MM-DiT recipe that the latent-diffusion idea evolved into.
- **🤗 `diffusers` documentation** — [`AutoencoderKL`](https://huggingface.co/docs/diffusers), `StableDiffusionXLPipeline`, and the micro-conditioning kwargs used above.
- Within this series: [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) (the latent this post stands on), [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) (the denoising process we relocated into latent space), [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) (the steering that makes conditioning work), and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
