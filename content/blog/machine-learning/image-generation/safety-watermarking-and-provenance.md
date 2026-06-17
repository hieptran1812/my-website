---
title: "Safety, Watermarking, and Provenance: Telling Real From Generated"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Build the defense layer that ships around every serious image model — invisible watermarks, C2PA provenance, synthetic-image detection, and the safety pipeline — and understand exactly where each one breaks."
tags:
  [
    "image-generation",
    "diffusion-models",
    "watermarking",
    "provenance",
    "ai-safety",
    "deepfakes",
    "c2pa",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/safety-watermarking-and-provenance-1.png"
---

A bank's fraud team forwards you a photo. A "customer" has submitted a passport selfie to open an account; the image looks real, the lighting is plausible, the skin has pores. Their question is brutally simple: was this made by a camera or by a model? You open it in your tooling. There is no EXIF that helps. A passive "AI detector" you grabbed off GitHub returns "73% synthetic" — which is to say, it has no idea. You wish, very badly, that whoever generated this image had left a fingerprint inside it that you could read.

That wish is the entire subject of this post. As image generators crossed the line from "obviously fake" to "indistinguishable in a thumbnail," the interesting engineering moved one layer out — away from *making* images and toward *attributing* them. Every serious image model shipped in 2024–2026 now arrives wrapped in a safety, watermarking, and provenance stack: a training-data filter, a runtime safety classifier, an invisible watermark baked into every output, and a cryptographically signed manifest of where the file came from. None of these layers is sufficient alone. Each fails to a different attack. So production systems stack them, which is exactly why the figure below shows four layers and not one.

![A vertical stack diagram of the four-layer deployment safety pipeline from training-data filter through runtime classifier, watermark, and C2PA manifest to the released image](/imgs/blogs/safety-watermarking-and-provenance-1.png)

By the end of this post you will be able to: reason about the risk surface and *why* after-the-fact detection is structurally hard; embed and detect a Tree-Ring-style watermark in the initial noise with PyTorch, and explain the statistics (bit-accuracy, p-value, false-positive rate) that make a detector trustworthy; fine-tune a VAE decoder à la Stable Signature so every image carries a fixed message; read and write a C2PA Content Credentials manifest; wire a real NSFW/safety checker into a 🤗 `diffusers` pipeline; and — crucially — say honestly where each of these defenses breaks, so you do not oversell a fragile guarantee to a fraud team that is counting on it.

This sits at the far end of our diffusion stack. We have spent the series turning noise into photographs: the [VAE](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) that gives us a latent, [DDIM](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling) that makes sampling deterministic and invertible, [latent diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) that made it all cheap enough to ship. The defense layer reuses every one of those parts. Stable Signature fine-tunes the *same* VAE decoder. Tree-Ring relies on the *same* DDIM inversion. The provenance story is about the *same* edits we covered in [instruction editing](/blog/machine-learning/image-generation/instruction-and-in-context-image-editing). Defense is not a bolt-on; it is the diffusion stack viewed from the security side.

One framing note before we start, because it matters. This is a *defender's* document. I describe attacks — cropping, JPEG, regeneration, watermark removal — only to explain why robustness is the hard part and how to measure it. The goal throughout is detection and provenance, not evasion. If you are building any of this into a product, that distinction is also a legal and ethical one, and I will flag where it bites.

## 1. The risk surface, and why detection-after-the-fact is hard

Start with the threats, because they decide everything downstream. The harms from generative imagery cluster into a handful of categories, and they are not equally tractable:

- **Non-consensual intimate imagery (NCII)** — synthetic nudes of real people, overwhelmingly women. This is the single most common abuse of open image models, and it is devastating to victims.
- **CSAM** — child sexual abuse material. This is the bright-line, zero-tolerance category. The defense here is upstream: keep it out of training data, refuse to generate it, and report. There is no "robustness trade-off" conversation for CSAM; the only acceptable rate is zero.
- **Deepfakes and impersonation** — a real person depicted doing or saying something they did not, for political, reputational, or fraud purposes.
- **Fraud and forged documents** — synthetic IDs, fake receipts, fabricated "evidence." The passport selfie from the intro lives here.
- **Misinformation at scale** — fabricated "photographs" of events that never happened, generated cheaply enough to flood a platform.

Now the uncomfortable structural fact: **detecting a generated image after the fact is, in the limit, impossible by construction.** A diffusion model is explicitly trained to make its output distribution match the distribution of real images. The training objective *is* "be indistinguishable from real." As models improve, the residual statistical tells that a passive classifier latches onto — GAN-era checkerboard artifacts, frequency-spectrum anomalies, telltale up-sampling fingerprints — shrink toward the noise floor of real-camera variation. You are in an arms race where the generator's loss function is, almost literally, the negative of your detector's accuracy.

This is why the field pivoted from *passive* detection (look at the pixels and guess) to *proactive* attribution (plant something at generation time that you can read later). If the model itself leaves a mark, you no longer have to win a distribution-matching war; you just have to read a signal you put there.

To see why the arms race is structurally unfavorable to the passive defender, think about who updates faster. The generator improves on a roughly annual cadence (new architecture, more data, better training), and each release closes the gap your detector exploited. Your detector, trained on last year's outputs, decays the moment the new model ships — and you can only retrain it *after* you have a corpus of the new model's images, which means you are permanently one generation behind. The defender is reactive by construction; the attacker sets the pace. Worse, the generator's training objective is *adversarial to your detector by accident*: every improvement in FID or human-preference score is, almost definitionally, a reduction in the statistical distance between real and generated that your classifier needs. You are trying to measure a quantity the entire field is racing to drive to zero. No amount of detector cleverness escapes that; only changing the game — moving the signal from "tells the generator accidentally left" to "bits you deliberately planted" — does. The figure below is the taxonomy that organizes the rest of this post: proactive methods planted at generation, versus reactive detection applied afterward, and only one branch survives an adversary who strips metadata.

![A tree diagram splitting attribution into a proactive branch with watermarking and C2PA and a reactive branch with passive classifiers](/imgs/blogs/safety-watermarking-and-provenance-6.png)

There is a second structural problem even for proactive marks, and it is worth stating early so the trade-offs later make sense. A watermark is a *covert channel*: you are hiding bits inside an image without changing how it looks. The channel has a capacity, the channel is noisy (the image gets cropped, recompressed, screenshotted), and an adversary can deliberately add noise to jam it. Everything that follows is information theory dressed in pixels.

### What "hard" means precisely

Let me make the detection-is-hard claim quantitative, because hand-waving here leads to bad product promises. Suppose your passive detector achieves a true-positive rate (TPR) of 95% and a false-positive rate (FPR) of 5% on a benchmark. Sounds great. Now deploy it on a platform where, say, 1 in 1,000 uploaded images is actually synthetic (a base rate of 0.1%). Of 1,000,000 uploads, 1,000 are synthetic and you catch 950. But 999,000 are real, and at 5% FPR you flag **49,950** of them as fake. Your "fakes" pile is 49,950 false alarms to 950 true catches — a precision of under 2%. The detector is useless at that base rate, not because it is bad, but because base rates are unforgiving and real images vastly outnumber synthetic ones in most pipelines.

This is the core reason watermarking wins where it can be deployed: a *keyed* watermark detector can be tuned to an astronomically low false-positive rate (think $10^{-9}$), because a random natural image has essentially zero probability of accidentally correlating with your secret key. You trade the impossible passive problem for a tractable cryptographic one — at the cost of needing control of the generator.

## 2. Invisible watermarking: the two families

A watermark embeds a payload — anywhere from a single "this is synthetic" bit (a *0-bit* or *detection* watermark) to a multi-bit message identifying the model, user, or session — into an image such that (a) humans can't see it and (b) a detector can read it back even after the image has been mangled. There are two fundamentally different places to do this.

**Post-hoc watermarking** takes a finished image and perturbs it. Classic methods work in a transform domain — `DwtDct` embeds bits by tweaking DCT coefficients of DWT sub-bands; `RivaGAN` trains an encoder/decoder network to hide and recover a payload robustly. These are model-agnostic (they work on any image, including real photos and outputs from models you don't control), which is their great strength and their great weakness: because the mark is applied *after* generation, an attacker who has the raw model output, or who simply re-encodes the file, can often strip it.

**In-generation watermarking** bakes the mark into the generative process itself, so there is no "clean" version of the output that ever leaves the model. This is where the interesting 2023–2026 work lives:

- **Stable Signature** (Fernandez et al., 2023) fine-tunes the *VAE decoder* so that a fixed, frozen extractor network reads a chosen $k$-bit message from every image the decoder produces. The message is literally a property of the decoder's weights.
- **Tree-Ring** (Wen et al., 2023) embeds a pattern in the *initial noise* of the diffusion process — specifically in the Fourier domain of the starting latent — and detects it by DDIM-inverting a suspect image to recover that noise and correlating. It's a 0-bit (detection) watermark in its base form.
- **Gaussian Shading** (Yang et al., 2024) goes further: it maps the message bits into the *sampling of the initial noise itself* in a way that is provably distribution-preserving — the watermarked noise is still standard Gaussian — so the watermark is theoretically *performance-lossless* (no FID/quality hit) while carrying a real multi-bit payload.
- **SynthID** (Google DeepMind) is a production system: an imperceptible watermark embedded during generation plus a paired detector, deployed across Imagen/Gemini image outputs (and extended to text and audio). The exact method is proprietary, but it sits in the in-generation family and is designed to survive common transforms.

The comparison matrix below is the one you should internalize before reading further. The honest column is "Regen attack" — what happens when an adversary runs your watermarked image back through a diffusion model.

![A comparison matrix of watermarking methods against whether they are in-generation, robustness to crop and JPEG, survival of regeneration, and capacity](/imgs/blogs/safety-watermarking-and-provenance-2.png)

The pattern is clear: in-generation methods crush post-hoc methods on robustness to ordinary degradations (crop, JPEG, resize), because the mark is woven into image structure rather than sprayed on top. But every method in the table degrades or dies under *regeneration* — feeding the image through an image-to-image diffusion pass, which resamples the pixels and discards the planted signal. We'll quantify that in the attacks section. Hold the thought.

### A note on terminology: detection vs identification

Two different jobs hide under "watermarking," and conflating them causes grief:

- **Detection (0-bit):** "Was this made by *a* watermarked model?" One yes/no bit. Tree-Ring's base form does this. You only need to distinguish "carries our pattern" from "doesn't."
- **Identification (multi-bit):** "*Which* model / user / session made this?" Stable Signature's 48-bit message does this. Now you can attribute a leak to a specific API key, or trace abuse.

Multi-bit is strictly harder: more bits means a weaker per-bit signal for the same imperceptibility budget, which means worse robustness. This is the first face of the no-free-lunch trade-off we'll formalize in §6.

### The distribution-preserving idea, in brief

There is a third place to mark, distinct from "mark the seed" (Tree-Ring) and "mark the decoder" (Stable Signature), and it's the cleverest of the bunch: **mark the way you sample the seed.** Gaussian Shading observes that the initial latent $z_T$ is supposed to be drawn from $\mathcal{N}(0, I)$, and that there is enormous freedom in *which* sample you draw. Instead of drawing a fresh random sample, you draw one whose bits, after a fixed deterministic transform, encode your message — while ensuring the resulting sample is still *exactly* a standard Gaussian sample as far as any statistical test can tell.

The mechanism is a chain of three steps. First, the $k$-bit message is replicated and spread across the latent for redundancy (this is the error-correction budget that buys robustness). Second, those bits are encrypted with a stream cipher keyed by a secret, which makes the bit pattern statistically indistinguishable from uniform random bits — so an adversary without the key sees noise. Third, the uniform bits are mapped through the inverse Gaussian CDF into latent values, so the watermarked $z_T$ is distributed *identically* to an unwatermarked draw. The punchline: because the input noise distribution is mathematically unchanged, the *output* image distribution is unchanged, so FID and every quality metric are provably untouched. The watermark costs zero image quality — it dodges the imperceptibility corner of the trilemma entirely. Detection runs the same DDIM inversion as Tree-Ring to recover $\hat z_T$, decrypts, and majority-votes the recovered bits. It still pays the robustness price under regeneration, because Shannon doesn't care how clever your encoding is — but among "free lunch on quality" methods it is the state of the art, and it's the one to study if you need real payload bits at no FID cost.

## 3. Tree-Ring: a watermark in the noise, read by inverting diffusion

Tree-Ring is my favorite of these methods to teach, because it weaponizes a property of diffusion we already built earlier in the series: **DDIM sampling is (approximately) invertible.** If you understand why DDIM is a deterministic ODE integrator, you already understand why Tree-Ring works.

### The science

Recall the diffusion-stack picture. We sample a starting latent $z_T \sim \mathcal{N}(0, I)$, then run the reverse process — with DDIM, a deterministic update — down to $z_0$, and decode to an image $x$. DDIM's update is an Euler step on the probability-flow ODE, so it has an inverse: given $x$ (hence $z_0$), you can integrate the ODE *forward* in time to recover an estimate $\hat z_T$ of the very noise you started from. We covered this inversion when we discussed editing; Tree-Ring uses it for detection.

Here is the trick. Before generation, take the initial noise $z_T$ and modify it in the **Fourier domain**. Compute $Z_T = \mathcal{F}(z_T)$ (a 2D FFT over spatial dimensions). Choose a set of concentric rings of radii in frequency space, and overwrite the Fourier coefficients on those rings with a fixed *key* pattern $k$. Rings are chosen because they are rotation-invariant: if the image is rotated, the ring structure in the magnitude spectrum is preserved. Invert the FFT to get the watermarked noise $z_T^* = \mathcal{F}^{-1}(Z_T^*)$, and generate from it.

Why is this invisible? Because $z_T^*$ is still, to the diffusion model, just a plausible sample of noise. The rings perturb a small set of frequency coefficients; the marginal distribution stays close to Gaussian; the generated image looks completely normal. The mark lives in the *seed*, not the pixels.

Detection inverts the logic. Given a suspect image $x'$:

1. Encode to latent $z_0' = \mathcal{E}(x')$ and DDIM-invert to recover $\hat z_T \approx z_T^*$.
2. FFT it: $\hat Z_T = \mathcal{F}(\hat z_T)$.
3. On the ring locations, compare $\hat Z_T$ against the key $k$. The test statistic is the distance (or correlation) between the measured ring coefficients and the key.

The figure below traces this as two branches — the suspect image is inverted to recover its noise spectrum, the secret key produces the expected ring, and a correlation test on the ring frequencies yields a calibrated p-value.

![A graph diagram with two branches, one inverting the suspect image to recover its noise spectrum and one deriving the expected ring from the key, merging at a correlation and p-value test](/imgs/blogs/safety-watermarking-and-provenance-3.png)

### The detection statistic, made precise

This is where most tutorials wave their hands, and where you must not. The reason a keyed watermark can hit a $10^{-9}$ false-positive rate is a clean statistical argument.

Under the **null hypothesis** $H_0$ — the image is *not* watermarked — the inverted noise $\hat z_T$ is an ordinary sample of Gaussian noise (DDIM inversion of any image lands you near a Gaussian latent). On the ring locations, the Fourier coefficients are therefore complex Gaussian. The squared distance between the measured ring and the (zero-mean reference under $H_0$) key, suitably normalized, follows a **chi-squared distribution** with degrees of freedom equal to the number of marked frequency components $|R|$:

$$
\eta \;=\; \frac{1}{\sigma^2}\sum_{f \in R} \big|\hat Z_T(f) - k(f)\big|^2 \;\sim\; \chi^2_{|R|} \quad \text{under } H_0 .
$$

Under the **alternative** $H_1$ — the image *is* watermarked — the inverted ring coefficients sit near the key $k$, so $\eta$ is small. You declare "watermarked" when $\eta$ falls below a threshold $\tau$. The false-positive rate is then *exactly computable* from the chi-squared CDF:

$$
\text{FPR}(\tau) \;=\; P\big(\chi^2_{|R|} \le \tau\big) \;=\; F_{\chi^2_{|R|}}(\tau).
$$

That is the whole game. You pick $\tau$ to set FPR to whatever you can tolerate — $10^{-6}$, $10^{-9}$ — *analytically*, without a held-out set of "real" images, because the null is a known distribution. A passive classifier can never give you this; its FPR is an empirical estimate that drifts as the world changes. A keyed watermark's FPR is a number you compute from a CDF. That is the entire reason this approach is trustworthy where passive detection is not.

### Why the inversion works: the DDIM forward step

It's worth pausing on the one assumption everything above rests on — that we can recover $\hat z_T$ from an image at all — because if you skip it, Tree-Ring feels like magic, and magic is a bad foundation for a fraud-detection tool. Recall the DDIM update from the [sampling post](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling). The deterministic reverse step that turns $z_t$ into $z_{t-1}$ is

$$
z_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0(z_t) + \sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta(z_t, t),
\quad
\hat x_0(z_t) = \frac{z_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(z_t,t)}{\sqrt{\bar\alpha_t}} ,
$$

where $\bar\alpha_t$ is the cumulative noise-schedule product and $\epsilon_\theta$ is the denoiser's noise prediction. Because there is no stochastic term — no $\sigma_t z$ added — this is a deterministic map from $z_t$ to $z_{t-1}$. A deterministic map can be run backward. The DDIM *inversion* update is the same equation solved for the forward-time variable, marching $t$ up instead of down:

$$
z_{t+1} = \sqrt{\bar\alpha_{t+1}}\,\hat x_0(z_t) + \sqrt{1-\bar\alpha_{t+1}}\,\epsilon_\theta(z_t, t) .
$$

Geometrically, DDIM is an Euler discretization of the probability-flow ODE $\mathrm{d}z = f(z,t)\,\mathrm{d}t$, and an ODE is reversible: integrate the velocity field forward in time from $z_0$ and you arrive back near $z_T$. The catch is the word "near." Euler integration accumulates discretization error, the denoiser is queried at slightly off-distribution points during inversion, and classifier-free guidance breaks the exact reversibility (the guided velocity field is not the true score). In practice, inverting with the *empty* prompt and a modest step count (50) lands $\hat z_T$ close enough that the low-dimensional ring signal — which spans only a few dozen frequency coefficients and is highly redundant — survives the round-trip error. That redundancy is not incidental; it is *why* Tree-Ring picks rings (many coefficients carrying one bit of "is the pattern here") rather than a high-capacity dense pattern (every coefficient carrying its own bit, none robust to inversion error). The design is a direct concession to the imperfection of inversion.

### The code

Here is a compact, runnable sketch of the Tree-Ring embed-and-detect loop. It is deliberately minimal — a real implementation handles channels and latent shapes carefully — but every line maps to the math above. This is defensive tooling: it lets a model owner verify their *own* outputs.

```python
import torch
import numpy as np

def make_ring_key(shape, radii, device):
    """Build a boolean ring mask and a fixed complex key on those rings."""
    c, h, w = shape
    yy, xx = torch.meshgrid(
        torch.arange(h) - h // 2,
        torch.arange(w) - w // 2,
        indexing="ij",
    )
    r = torch.sqrt(xx.float() ** 2 + yy.float() ** 2)
    mask = torch.zeros(h, w, dtype=torch.bool)
    for r_in, r_out in radii:           # e.g. [(8, 10), (16, 18)]
        mask |= (r >= r_in) & (r < r_out)
    mask = mask.to(device)
    # Deterministic key drawn once from a seed = the secret.
    g = torch.Generator(device="cpu").manual_seed(1234)
    key = torch.randn(c, h, w, generator=g) + 1j * torch.randn(c, h, w, generator=g)
    return mask.to(device), key.to(device)

def embed(z_T, mask, key):
    """Stamp the ring key into the FFT of the initial noise."""
    Z = torch.fft.fftshift(torch.fft.fft2(z_T), dim=(-2, -1))
    Z[:, mask] = key[:, mask]           # overwrite ring coefficients
    z_star = torch.fft.ifft2(torch.fft.ifftshift(Z, dim=(-2, -1))).real
    return z_star

@torch.no_grad()
def detect(z_T_hat, mask, key):
    """Chi-squared distance on the ring; smaller = more likely watermarked."""
    Z = torch.fft.fftshift(torch.fft.fft2(z_T_hat), dim=(-2, -1))
    diff = (Z[:, mask] - key[:, mask]).abs() ** 2
    eta = diff.sum().item()             # the test statistic
    dof = 2 * mask.sum().item() * z_T_hat.shape[0]   # real+imag per channel
    return eta, dof
```

And the inversion step — recovering $\hat z_T$ from a suspect image with a 🤗 `diffusers` pipeline whose scheduler supports DDIM. The key call is running the scheduler's update *in reverse time order*:

```python
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
inverse = DDIMInverseScheduler.from_config(pipe.scheduler.config)

@torch.no_grad()
def ddim_invert(pipe, inverse, image_latent, prompt_embeds, steps=50):
    """Integrate the PF-ODE forward in time: x_0 -> z_T_hat."""
    inverse.set_timesteps(steps, device="cuda")
    latents = image_latent
    for t in inverse.timesteps:
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=prompt_embeds).sample
        latents = inverse.step(noise_pred, t, latents).prev_sample
    return latents       # ~ z_T_hat, ready for detect()
```

Two honest caveats. First, inversion is only *approximate*: classifier-free guidance and the VAE encode/decode round-trip inject error, so $\hat z_T \ne z_T^*$ exactly. Tree-Ring's robustness comes from the rings being a low-dimensional, redundant signal that survives this error. Second, you need a prompt (or an empty/null prompt) to invert; the original Tree-Ring paper inverts with the empty prompt and it works because the ring signal dominates.

## 4. Stable Signature: a message baked into the decoder

Tree-Ring marks the *seed*. Stable Signature marks the *decoder* — and that difference gives it a clean multi-bit channel.

### The science

Recall from the [VAE post](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) that latent diffusion generates a latent $z_0$ and then a decoder $\mathcal{D}$ maps it to pixels: $x = \mathcal{D}(z_0)$. Stable Signature's insight: the decoder is the *last* thing that touches every image, and it has spare capacity. So fine-tune $\mathcal{D}$ into $\mathcal{D}_m$ such that for a *fixed* secret message $m \in \{0,1\}^k$ and a *frozen, pretrained* extractor network $W$,

$$
W\big(\mathcal{D}_m(z_0)\big) \approx m \quad \text{for all } z_0 .
$$

The extractor $W$ is trained once (a HiDDeN-style watermark decoder), then frozen. The fine-tuning optimizes only the decoder weights, on two losses that encode exactly the trilemma:

$$
\mathcal{L} \;=\; \underbrace{\mathcal{L}_{\text{message}}\big(W(\mathcal{D}_m(z)),\, m\big)}_{\text{can the extractor read } m?} \;+\; \lambda\, \underbrace{\mathcal{L}_{\text{perceptual}}\big(\mathcal{D}_m(z),\, \mathcal{D}(z)\big)}_{\text{does it still look the same?}}
$$

The message loss is binary cross-entropy between the extractor's logits and the target bits. The perceptual loss (LPIPS plus an MSE/Watson-VGG term) keeps $\mathcal{D}_m$'s output close to the original decoder's, so quality doesn't drift. The weight $\lambda$ is your imperceptibility-vs-robustness knob. The figure below shows the two-branch training graph: one branch enforces image quality, the other enforces that the frozen extractor reads the secret bits.

![A graph diagram of Stable Signature fine-tuning where the latent goes through the fine-tuned decoder to an image that branches into a perceptual loss and an extractor that together form the joint loss](/imgs/blogs/safety-watermarking-and-provenance-5.png)

Why is this beautiful engineering? Three reasons. First, **fine-tuning is cheap and fast** — you touch only the decoder (tens of millions of params), for ~1 hour on a single GPU, not the whole diffusion model. Second, **the diffusion model is untouched**, so FID and prompt-following are essentially unchanged. Third, **every image is marked the same way**, by construction, because the mark is in the weights — there is no per-image embedding step to forget or skip. The reported numbers are strong: bit-accuracy near 99% on the 48-bit message with negligible FID change, and robustness to JPEG, crops, and brightness shifts well above the post-hoc baselines.

### The robustness training trick

The single most important detail: to make the mark survive degradations, you insert a **differentiable augmentation layer** between the decoder output and the extractor *during training*. JPEG (a differentiable approximation), random crops, resizes, blur, and color jitter are applied to $\mathcal{D}_m(z)$ before $W$ reads it. The extractor must then learn to read $m$ *through* those degradations, which forces the decoder to write the mark in a way that survives them. This is the same idea that makes adversarially-trained classifiers robust: train on the perturbations you expect to face.

```python
import torch, torch.nn.functional as F
import kornia.augmentation as K

# Frozen, pretrained 48-bit extractor (HiDDeN-style). Decoder is trainable.
extractor.requires_grad_(False)
augment = torch.nn.Sequential(
    K.RandomJPEG(jpeg_quality=(50.0, 90.0), p=0.5),   # differentiable JPEG
    K.RandomResizedCrop((256, 256), scale=(0.5, 1.0), p=0.5),
    K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.3),
)
target = message.float()                               # fixed 48-bit code, in {0,1}

for z0, clean in loader:                               # z0 = sampled latent
    x = decoder(z0)                                    # marked image
    x_aug = augment(x)                                 # survive-this layer
    logits = extractor(x_aug)                          # predicted bits
    loss_msg = F.binary_cross_entropy_with_logits(logits, target.expand_as(logits))
    loss_perc = lpips(x, clean).mean()                 # imperceptibility
    loss = loss_msg + lambda_perc * loss_perc
    loss.backward(); opt.step(); opt.zero_grad()
```

Detection at serve time is then trivial and cheap — no diffusion inversion required, unlike Tree-Ring. You just run the frozen extractor on the suspect image and compare bits:

```python
@torch.no_grad()
def verify(image, extractor, message, threshold_bits=42):
    logits = extractor(image)              # [B, 48]
    bits = (logits > 0).int()
    correct = (bits == message).float().mean(dim=1)   # bit-accuracy per image
    matches = (bits == message).sum(dim=1)            # bits matching
    return matches >= threshold_bits, correct          # decision, accuracy
```

### The p-value for a multi-bit mark

The statistics here are even cleaner than Tree-Ring's. Under $H_0$ (unmarked image), the extractor's output bits are ~uniform, so the number of matching bits $M$ out of $k=48$ is $\text{Binomial}(48, 0.5)$. The false-positive rate for a threshold of $\tau$ matching bits is the binomial tail:

$$
\text{FPR}(\tau) \;=\; P(M \ge \tau) \;=\; \sum_{i=\tau}^{48} \binom{48}{i} 2^{-48}.
$$

Plug in $\tau = 42$: the probability that a random image matches at least 42 of 48 bits by chance is about $2 \times 10^{-9}$. That is your false-positive rate, computed from a binomial, no held-out set needed. If a suspect image matches 46/48 bits, the p-value is around $10^{-11}$ — you can state, with cryptographic-grade confidence, that this image came from your decoder. *This* is what attribution should feel like.

#### Worked example: tuning the bit threshold

Your platform serves 10 million images/day and you want at most **one** false accusation per day. With $k = 48$ bits, you need $\text{FPR} \le 10^{-7}$. Walking up the binomial tail: $\tau = 40$ gives FPR $\approx 3\times10^{-7}$ (too loose), $\tau = 41$ gives $\approx 9\times10^{-8}$ (just under). So set the decision threshold at 41 matching bits. Now check your robustness budget: if regeneration-free degradations (a screenshot at 70% JPEG, a 20% crop) cost you ~4 bits of accuracy on average, your marked images land around 44/48 — comfortably above 41, so true positives still fire. The gap between "true positives land at 44" and "threshold at 41" is your entire safety margin. Spend it wisely; don't set the threshold so high that mild degradation drops a real mark below it.

## 5. The attacks, and why regeneration is the boss fight

You cannot reason about a watermark without reasoning about its adversary. I'll describe attacks at the level needed to *measure robustness* — the defender's job — and stop there. The point is always: which degradations does the mark survive, and how would you quantify it?

**Valuemetric distortions** are the easy adversary: JPEG recompression, resizing, cropping, brightness/contrast shifts, mild blur, additive noise. These are what an image suffers just by being posted to a social platform. In-generation watermarks (Stable Signature, Tree-Ring, Gaussian Shading) are explicitly trained or designed to survive these, and they largely do — bit-accuracy stays in the 90s under moderate JPEG and crops. Post-hoc methods like `DwtDct` degrade much faster.

**Geometric distortions** — rotation, perspective warp, flips — are harder. Tree-Ring's ring choice buys rotation-invariance in the magnitude spectrum; Stable Signature relies on having trained through crops/affine warps. Strong geometric attacks still hurt.

**Regeneration / paraphrase attacks** are the boss fight, and the honest center of this whole topic. The attack: take the watermarked image, run it through an image-to-image diffusion pass (encode to latent, add noise to some intermediate timestep, denoise back) or a strong VAE round-trip / autoencoder. The output looks ~identical to a human but is, pixel-for-pixel, a *fresh* sample from the model's distribution — and the planted watermark, whether in the seed (Tree-Ring) or the decoder's pixel-level signal (Stable Signature), is largely resampled away. This is not a bug in any particular scheme; it is a near-fundamental limit. A regeneration attack is, in effect, "ask a clean generator to redraw this," and a clean generator does not know your secret.

The research consensus by 2025 is sobering and worth stating plainly: **no current watermark fully survives a determined regeneration/diffusion-purification attack.** The 2024 "robustness of AI-image detectors" line of work showed that for any watermark with non-trivial capacity, an adversary with access to *a* diffusion model can drive detector accuracy toward chance, at the cost of slightly more visible distortion. There is a quantifiable trade-off — a stronger attack leaves more visible artifacts — but a motivated adversary can win.

So why deploy watermarks at all? Because the threat model is not "stop a determined nation-state adversary." It is:

- **Raise the cost.** Casual misuse (someone screenshots a deepfake and posts it) is fully covered. The attacker now needs a diffusion model and the know-how to run a purification pass.
- **Cover the honest majority.** Most images on most platforms are *not* being adversarially laundered. A watermark that survives JPEG and crops attributes the vast majority of real-world cases.
- **Layer with provenance.** When the watermark is stripped, C2PA (next section) may still carry signed origin; when C2PA is stripped, the watermark may still be readable. They fail to *different* attacks, which is the entire argument for stacking them.

The before/after figure below makes the imperceptibility side concrete: a clean output and a watermarked one are visually identical, yet the detector reads near-random bits from the clean image and the full 48-bit message from the marked one.

![A before and after comparison showing a clean image with no readable mark beside a watermarked image that is visually identical but yields the full message to the detector](/imgs/blogs/safety-watermarking-and-provenance-4.png)

### A problem-solving narrative: which watermark do you ship?

Let me walk a real decision, because the methods above only mean something against a concrete threat model. You run image generation for a stock-photography marketplace. Two requirements land on your desk. (1) *Attribution:* when a customer reports that a competitor stole an image generated on your platform, you must prove it came from you, and ideally identify *which account* generated it. (2) *Compliance:* the EU AI Act requires every output be machine-readably labeled as AI-generated. You have one quarter and one engineer. What do you ship?

Step one — separate the two jobs. Compliance (1 bit, "this is AI") is a *detection* problem; attribution-to-account (which of ~10⁵ accounts, so ~17 bits) is an *identification* problem. They have different capacity needs, so don't force one mechanism to do both.

Step two — pick per job. For compliance, C2PA signing is the cheapest, most standards-aligned answer: ~10 ms to sign on the way out, it carries the exact IPTC `trainedAlgorithmicMedia` label regulators want, and it's the format the ecosystem reads. For attribution, you need bits that *survive stripping*, because the whole scenario is "someone took my image and re-posted it" — so a strippable manifest alone won't do. Stable Signature is the fit: fine-tune the decoder once (~1 hour, single GPU), and let the per-account ID ride in the 48-bit message. 17 bits of account ID fit comfortably inside 48 with room for error-correction.

Step three — stack them, because each covers the other's failure. The C2PA manifest gives rich, human-readable, signed provenance *when the file is preserved*; the Stable Signature watermark gives a surviving account ID *when the manifest is stripped*. Your shipped pipeline: generate → embed account-ID via the fine-tuned decoder → sign a C2PA manifest → return. Both layers, ~25 ms combined overhead on a multi-second generation. Done.

Now stress-test the decision, because that's where you find out if you oversold it:

- **What if the thief screenshots the image?** The C2PA manifest dies (screenshots strip metadata). But the Stable Signature mark survives a screenshot's JPEG round-trip at high bit-accuracy. Attribution still works; compliance label is gone — acceptable, since the *thief* removing the AI label is the thief's violation, not yours.
- **What if the thief crops to 60% and re-uploads?** Stable Signature trained through crops holds up — bit-accuracy drops a few points but stays above your 41/48 threshold. You still attribute.
- **What if the thief runs it through an img2img "enhance" tool (regeneration)?** Now you're in the boss fight. The mark is largely resampled away; bit-accuracy falls toward chance; attribution *fails*. This is the honest limit, and you must tell the legal team up front: "We attribute screenshots, crops, and recompression with cryptographic confidence; we do *not* attribute an image that's been re-run through a diffusion model." If they need that case covered, no current watermark delivers it — that's a research frontier, not a Q3 deliverable.
- **What if an account ID collides with random chance?** With a 41/48 threshold, FPR ≈ $9\times10^{-8}$ per comparison. But you're comparing against 10⁵ accounts, so by the union bound your effective false-attribution rate across the whole account space is ~$10^{-2}$ — uncomfortably high. The fix: don't scan all accounts; the watermark *is* the account ID (read the 48 bits, look up the matching account), so you make one comparison, not 10⁵. Design the lookup as "decode bits → index," never "test every account," or the multiple-comparisons math eats your confidence.

That last point is the kind of thing that only surfaces when you stress-test, and it's the difference between a system that holds up in a dispute and one that produces a false accusation. The watermark is not just a signal; it's a signal embedded in a *protocol*, and the protocol's statistics are as load-bearing as the bits.

#### Worked example: measuring robustness honestly

You ship Stable Signature and claim "robust to JPEG." Here is how to back that claim with a number instead of a vibe. Take a fixed set of 5,000 generated images. For each attack — JPEG at quality $\{90, 70, 50, 30\}$, center-crop at $\{90\%, 70\%, 50\%\}$, and a regeneration pass (SDEdit at denoising strength 0.3) — run the extractor and record **mean bit-accuracy** and the **true-positive rate at your chosen FPR threshold** (41/48 bits from the earlier example). A typical honest result table looks like: JPEG-90 → 99% acc / 100% TPR; JPEG-50 → 96% / 99%; crop-70% → 95% / 98%; SDEdit-0.3 → **62% / 7%**. Report all of it, especially the last row. The right sentence in your docs is "robust to recompression and cropping; *not* robust to diffusion regeneration." A vendor who omits the regeneration row is selling you a false guarantee.

## 6. The no-free-lunch trade-off, formalized

By now the shape of the constraint is visible, and it rhymes with the generative trilemma that runs through this whole series. A watermark has three properties in tension:

- **Robustness** — survives degradation and attack.
- **Imperceptibility** — humans (and quality metrics like FID/PSNR/LPIPS) can't see the mark.
- **Capacity** — how many bits the payload carries.

You can have any two, paid for by the third. The figure below lays this out as a matrix across three representative methods.

![A matrix showing three watermarking methods scored on robustness, imperceptibility, and capacity, illustrating that each spends its budget on two of the three axes](/imgs/blogs/safety-watermarking-and-provenance-7.png)

### Why it's a real bound, not just an observation

The trade-off isn't folklore; it falls out of information theory. Model the watermark channel: the embedder writes a payload by perturbing the image by some signal of power $P$ (the *watermark energy*). Imperceptibility caps $P$ — beyond some PSNR/LPIPS budget, humans see it. The channel adds noise of power $N$ — every degradation and attack is noise injected into your covert channel. By Shannon, the bits you can reliably push through a Gaussian channel are bounded:

$$
C \;\le\; \tfrac{1}{2}\log_2\!\Big(1 + \tfrac{P}{N}\Big) \quad \text{bits per channel use.}
$$

Read that formula as the whole trade-off in one line:

- Want more **capacity** $C$? You need more signal power $P$ (hurts **imperceptibility**) or less channel noise $N$ (i.e., only survive gentle attacks — hurts **robustness**).
- Want more **robustness** (survive bigger $N$)? For fixed imperceptibility ($P$ capped), $C$ must drop — fewer bits. This is *exactly* why Tree-Ring's most robust variant is 0-bit (detection only) and why pushing Stable Signature past ~48 bits costs robustness.
- Want perfect **imperceptibility** (tiny $P$)? Then $C/N$ collapses and you can carry almost nothing through any real attack.

Gaussian Shading is the clever attempt to dodge one corner of this: by embedding the message *as* the noise sampling (keeping it exactly Gaussian), it pays *zero* imperceptibility cost in the FID sense — the watermarked images are statistically the same distribution — while still carrying multi-bit payload. But it doesn't repeal Shannon; its robustness to regeneration still degrades, because regeneration is a high-$N$ attack that no finite $P$ survives. The trilemma is a wall. Engineering is about choosing which face of the wall to lean on for *your* threat model.

#### Worked example: putting numbers on the bound

Make the Shannon bound concrete so the trade-off stops being abstract. Suppose your imperceptibility budget allows a watermark perturbation at about 40 dB PSNR — a signal power $P$ that is roughly $10^{-4}$ of the image power. Against a *gentle* channel (JPEG-90, mild crop) the effective noise power $N$ injected into your covert channel might be comparable to $P$, giving signal-to-noise ratio $P/N \approx 1$ and a per-coefficient capacity of $\tfrac{1}{2}\log_2(2) = 0.5$ bits. Spread that over a few thousand usable latent coefficients with heavy redundancy and you comfortably push 48 reliable bits — which is exactly the regime Stable Signature operates in. Now switch to a *regeneration* channel: an img2img pass at denoising strength 0.3 effectively *replaces* most of the planted signal, so $N \gg P$ and $P/N$ collapses toward $10^{-2}$ or worse. The capacity $\tfrac{1}{2}\log_2(1.01) \approx 0.007$ bits per coefficient — multiply by your coefficient budget and redundancy and you are below one reliable bit. That is the math behind "regeneration kills it," stated as a number rather than a vibe: the channel SNR crashes by orders of magnitude, and no encoding recovers bits the channel destroyed. The only knobs you have — raise $P$ (visible mark) or reduce redundancy (fewer, more fragile bits) — both make things worse, which is precisely why it's a wall and not a tuning problem.

## 7. Provenance: C2PA and Content Credentials

Watermarking hides a signal *in the pixels*. Provenance does the opposite: it attaches a signed, tamper-evident *record* of where a file came from and what was done to it. The two are complementary precisely because they have opposite failure modes — **a watermark survives stripping but carries few bits and dies to regeneration; a manifest carries rich signed metadata but is trivially stripped by re-saving the file.** Stack them.

### What C2PA actually is

The **Coalition for Content Provenance and Authenticity (C2PA)** is an open standard (backed by Adobe, Microsoft, Google, OpenAI, the BBC, Sony, and others) for cryptographically signed provenance. The user-facing brand is **Content Credentials** ("CR" icon). The data structure is a **manifest** attached to the asset:

- **Assertions** — claims about the asset: who created it, when, with what tool, whether it was AI-generated (`c2pa.actions` with a `created` action of type `aiGenerative`), what edits followed (crop, color, composite), and a hash of the pixel data.
- **A claim** — binds the assertions together with a hash.
- **A signature** — the claim is signed with an X.509 certificate from a known signer (the model provider, the camera maker, the editing tool). This makes it *tamper-evident*: change the pixels or the assertions, and the hash no longer matches, and/or the signature fails to verify.

Crucially, C2PA does **not** prove an image is *true* or *real*. It proves *provenance*: this manifest was signed by this key, and the asset matches the hash in the manifest. A camera with a C2PA signing chain (Leica, Sony, Nikon have shipped these) can assert "captured by this device, unedited." A diffusion provider can assert "generated by FLUX on this date." An editor can append "cropped and color-graded." You get a signed *chain of custody*, not a truth oracle.

### Reading a manifest in code

The reference implementation is Adobe's open-source `c2pa` library (Rust core, with Python/JS/C bindings). Reading the manifest from a suspect image is straightforward:

```python
from c2pa import Reader

# Read and verify the C2PA manifest embedded in an image file.
with Reader.from_file("suspect.jpg") as reader:
    manifest_json = reader.json()          # full manifest store as JSON
    print(manifest_json)

# Typical fields you parse out:
#   manifests[active].claim_generator        -> "FLUX.1 / c2pa-python 0.x"
#   manifests[active].signature_info.issuer  -> the signing cert's org
#   manifests[active].assertions[*].label    -> "c2pa.actions", "c2pa.hash.data"
#   validation_status                        -> [] means signature + hash OK
```

An empty `validation_status` means the signature verified and the asset's hash matches the manifest — i.e., the file has not been altered since signing. A non-empty status lists exactly what failed (untrusted signer, hash mismatch, etc.). Writing/signing a manifest is symmetric: you build assertions, point the SDK at your signing certificate and private key, and it embeds the signed manifest into the output during your generation pipeline.

```python
from c2pa import Builder, create_signer, SigningAlg

# Assert "AI-generated by our model" and sign on the way out of the pipeline.
manifest = {
    "claim_generator": "our-image-service/1.0",
    "assertions": [
        {"label": "c2pa.actions",
         "data": {"actions": [{"action": "c2pa.created",
                               "digitalSourceType":
                               "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"}]}},
    ],
}
signer = create_signer(cert_chain, private_key, SigningAlg.ES256, tsa_url=None)
with Builder(manifest) as builder:
    builder.sign_file(signer, "generated.png", "generated_signed.png")
```

That `digitalSourceType` of `trainedAlgorithmicMedia` is the IPTC code that says "this was made by a trained AI model" — the exact label the EU AI Act's transparency provisions are pushing providers to attach.

### Provenance for *edited* images

The single most underrated property of C2PA is that it handles *editing*, which is exactly where pure watermarking gets awkward. Recall from [instruction and in-context editing](/blog/machine-learning/image-generation/instruction-and-in-context-image-editing) that the 2025 wave of models edits an existing photo conversationally — "remove the person on the left," "make it night." Now the provenance question is subtler than "real or fake." A real photograph that has been AI-edited is *partly* both: the camera captured a scene, then a model altered it. A binary "is this synthetic?" answer is wrong for this image. C2PA represents it correctly as a **chain**: the camera signs the original capture assertion; the editor appends a signed `c2pa.edited` action describing the change and re-signs, embedding the previous manifest as a parent (the *ingredient*). The result is a signed lineage — "captured by this camera, then edited by this tool with this action" — that a viewer can inspect step by step.

This is something watermarking simply cannot express. A watermark says "a watermarked generator touched these pixels"; it cannot say "the left third of this image is original camera capture and the right third was inpainted by a model on this date." Edit provenance is inherently a structured, signed record, not a hidden bit-pattern — which is the deepest reason the two layers are complementary rather than competing. The hard part in practice is *preservation*: most edit tools, social platforms, and messaging apps strip the manifest on re-encode, breaking the chain. The C2PA "durable credentials" work-in-progress addresses this by storing a hard binding (a content hash plus a watermark) so the chain can be recovered from a registry even after the soft metadata is gone — again, watermark and provenance leaning on each other.

### Why provenance is complementary, not redundant

Here is the clean mental picture, and it's the reason both layers ship together. The watermark **travels with the pixels**: crop it, screenshot it, re-encode it, and the mark (mostly) rides along, but it carries only ~48 bits and regeneration kills it. The C2PA manifest **carries a rich signed record** — full edit history, signer identity, timestamps — but it lives in the file's metadata container and is **trivially stripped** the moment someone re-saves through a tool that doesn't preserve it (which is most of them, including a screenshot). One survives stripping; one survives regeneration-ish scenarios where the file is preserved but you need rich signed detail. Neither survives everything. Together they cover far more of the attack surface than either alone — which is, once more, the four-layer stack from Figure 1.

There's an active research direction to *bind* the two: store a hash of the C2PA manifest (or a pointer to it) inside the watermark, so that even after metadata stripping, the surviving watermark bits let you look up the original signed provenance from a registry. That's the most promising path to durable provenance, and it's exactly where SynthID-style detection plus C2PA Content Credentials are converging in production.

## 8. Passive detection: useful, fragile, and not a guarantee

For images you *didn't* generate — outputs from models you don't control, or legacy content — you fall back to **passive detection**: a classifier trained to separate real from synthetic by their statistical fingerprints. This is the reactive branch of the taxonomy tree, and you should deploy it with clear eyes about its fragility.

What signals do these classifiers exploit?

- **Frequency-domain artifacts.** Up-sampling layers (in GANs especially) leave periodic patterns in the high-frequency spectrum. CNNs and DCT-based detectors latch onto these. Diffusion models leave subtler but real spectral signatures.
- **Local texture statistics.** Synthetic images have characteristic noise-residual patterns (the "fingerprint" of the generator's last layers), which methods isolate with high-pass filters before classifying.
- **Semantic/physical inconsistencies.** Bad hands, impossible reflections, inconsistent shadows, garbled text. Useful for forensics by humans, but increasingly rare as models improve, and not something a single classifier reliably catches.

The hard truth: **passive detectors generalize poorly across generators and degrade as models improve.** A detector trained on SD1.5 outputs often fails on FLUX or Midjourney v6; a detector trained on today's models will be obsolete against next year's. Worse, simple post-processing (JPEG, slight blur) erases many of the artifacts these classifiers depend on. The base-rate math from §1 compounds this: even a "95% accurate" detector is near-useless at a 0.1% synthetic base rate.

A second, sharper failure mode is *adversarial* fragility. A passive detector is just a classifier, and classifiers have adversarial examples — imperceptible perturbations that flip the prediction. An attacker who wants their synthetic image to read as "real" can run a few steps of gradient ascent against an open-source detector (or a surrogate) and reliably fool it, all while the image stays visually identical. This is strictly worse than the watermark-removal problem: removing a watermark at least requires degrading or regenerating the image, whereas fooling a passive detector can be done with a perturbation no human would notice. The asymmetry is the whole point — a keyed watermark forces the attacker to *damage the image* to win; a passive detector lets them win for free.

So where does passive detection earn its place? As **one weak signal in an ensemble**, never as a sole arbiter. A platform might combine: (1) C2PA check — if a valid "AI-generated" manifest is present, done; (2) watermark check — run known detectors (SynthID, Stable Signature) for participating models; (3) passive classifier — a low-weight prior when the first two come up empty; (4) human review for high-stakes decisions. Anyone shipping a single passive "AI detector" as a yes/no oracle — especially in an academic-integrity or legal context — is shipping a false-confidence machine. Say so.

```python
# A passive detector is a prior, not a verdict. Combine signals; never trust one.
def attribution_decision(image_path, image_tensor):
    c2pa = read_c2pa(image_path)          # signed manifest? authoritative if valid
    if c2pa and c2pa.says_ai_generated and c2pa.signature_valid:
        return "synthetic", 1.0, "c2pa-signed"

    wm = run_known_watermark_detectors(image_tensor)   # SynthID/StableSig/...
    if wm.detected:                       # keyed detector: FPR ~ 1e-9
        return "synthetic", 0.999, f"watermark:{wm.method}"

    p = passive_classifier(image_tensor)  # fragile; treat as a weak prior only
    if p > 0.97:
        return "likely-synthetic", p, "passive-low-confidence"
    return "unknown", p, "inconclusive"   # the honest default
```

Notice the default is **"unknown,"** not "real." That is the correct epistemic posture: absence of a watermark is not proof of authenticity, because real images never carried one and adversaries strip them. Never let your system claim "verified real" from a negative watermark result.

## 9. The deployment safety pipeline: filtering, classifiers, unlearning

Watermarking and provenance answer "who made this?" The safety pipeline answers "should we have made it at all?" This is the part with zero tolerance for the CSAM category and serious obligations for the rest. The runtime gating is two checkpoints bracketing generation, as the figure shows: a prompt filter before, an output classifier after, and either one can divert to a blocked response.

![A graph diagram of runtime safety gating where a prompt flows through a prompt filter and generation to an output classifier that routes either to a blocked response or to a marked and signed release](/imgs/blogs/safety-watermarking-and-provenance-8.png)

The layers, in order of when they act:

**1. Training-data filtering (pre-training).** The most important and least glamorous layer. You cannot generate what the model never learned. The 2023 disclosure that the LAION-5B web-scrape contained CSAM URLs forced a reckoning: serious labs now run known-CSAM hash matching (PhotoDNA-style), NSFW classifiers, and aggressive filtering over training data, and re-released cleaned datasets. Filtering training data is *the* highest-leverage safety intervention, because it shapes the model's capabilities at the root rather than patching outputs.

**2. Prompt filtering (runtime, pre-generation).** Block or rewrite requests that ask for prohibited content. This ranges from simple blocklists (crude, easy to evade with euphemism) to learned classifiers over the prompt embedding. It's a first line, not a wall — natural language is too flexible for blocklists to be airtight — which is why you also need:

**3. Output safety classification (runtime, post-generation).** Run the *generated image* through an NSFW/safety classifier before returning it. This is the `safety_checker` that ships with Stable Diffusion in `diffusers`: a CLIP-based classifier that compares the output's embedding against a set of "unsafe concept" embeddings and blanks the image if it trips. It's blunt (false positives on art/medical content, false negatives on novel cases) but it's a real backstop. Wiring it is one line — and, importantly, *disabling* it is a deliberate choice you should log:

```python
from diffusers import StableDiffusionPipeline
import torch

# The safety checker is ON by default. Keep it on for served endpoints.
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    # safety_checker=None,   # <- disabling this is a policy decision; log it.
).to("cuda")

out = pipe("a portrait of a person in a park", num_inference_steps=30)
image = out.images[0]
if out.nsfw_content_detected[0]:          # the checker fired
    image = serve_placeholder()           # blank/blocked image, log the event
```

**4. Concept erasure / unlearning (model surgery).** Filters catch content at the gate; *unlearning* removes the capability from the weights so it can't be produced even by jailbroken prompts. Two representative methods:

- **ESD (Erased Stable Diffusion)** fine-tunes the model to *push its own predictions away* from a target concept. The elegant trick: use the model's own conditional noise prediction for the concept as a negative guidance signal, so the fine-tuned model's score for that concept is steered toward the unconditional (or away from it). It erases styles ("Van Gogh"), objects, or unsafe concepts directly in the weights, with no extra data.
- **SafeGen** and related methods target sexual content specifically by editing the model's self-attention so it cannot represent explicit imagery, regardless of the adversarial prompt, while preserving benign generation.

Unlearning is strictly stronger than prompt filtering against jailbreaks (the capability is *gone*, not gated), but it's imperfect: erased concepts can sometimes be recovered by fine-tuning or clever prompting, and aggressive erasure dents benign quality. It's a layer, not a guarantee — the recurring theme.

**5. Red-teaming and monitoring (continuous).** Before launch and continuously after, adversarially probe the stack: automated jailbreak prompt generation, NCII/CSAM test suites (run by trained, authorized teams under strict protocols), and abuse monitoring on production traffic. Safety is not a checkbox you tick at launch; it's a process you run forever.

The mechanics of red-teaming an image model are worth a beat, because "we red-team" is often hand-waved. A real program has three legs. First, a **curated prompt suite** — a versioned set of adversarial prompts covering each harm category, including the obvious euphemisms and the known jailbreak patterns (roleplay framings, encoded requests, multi-turn priming, prompts that ask for the unsafe concept "in the style of" a benign one). You re-run this suite on every model and filter change and track the pass rate as a regression metric, exactly like a test suite. Second, **automated adversarial search** — use a language model to *generate* novel jailbreak prompts against your own filters, then feed the ones that succeed back into training and filtering. This is cheaper and more thorough than humans alone, and it scales with the attack surface. Third, **production abuse signals** — rate-limit anomalies, accounts that hammer the prompt filter with near-miss variants (a tell for someone searching for a bypass), and a feedback channel for the trust-and-safety team to escalate novel abuse. The output classifier's *block* events are themselves a dataset: a sudden spike in blocks for a particular prompt pattern is an early warning that someone found a partial bypass.

A subtle but important design choice: **what you do on a block matters as much as the block itself.** Returning a blank placeholder leaks information ("that prompt was unsafe"), which an attacker uses as a gradient to search around the filter. Returning a generic refusal for *both* prompt-filter and output-classifier blocks, with no detail about *why*, denies the attacker that signal — at the cost of a worse experience for the false-positive case (the user generating legitimate medical or art content who gets blocked). That tension — security-through-opacity versus user transparency — has no universally right answer; it's a policy decision you make per surface, and it's the kind of thing that should be reviewed, not defaulted.

#### Worked example: the cost of a safety classifier

A serving question: what does the output classifier add to latency and bill? On an A100, generating a 1024×1024 SDXL image at 30 steps takes roughly 3–4 s. A CLIP-based safety classifier is a single forward pass over one image — on the order of 5–15 ms on the same GPU, i.e. well under 1% of generation time. So the output classifier is *nearly free* in latency and cost: there is no performance excuse for omitting it on a served endpoint. Prompt filtering is similarly cheap (one text-classifier forward, single-digit ms). The expensive safety layer is training-data filtering (a one-time pre-training cost) and red-teaming (human time). The runtime classifiers — the ones a user actually hits — are a rounding error. Budget accordingly and leave them on.

## 10. Case studies: real systems and real numbers

Concrete, cited reference points. Where I give a figure, it's from the cited paper or the system's published material; where I'm approximating, I say so.

**Stable Signature (Fernandez et al., Meta AI, 2023).** Fine-tunes the LDM decoder to embed a 48-bit message read by a frozen HiDDeN extractor. Reported bit-accuracy is near 99% on clean images with FID essentially unchanged from the base model, and robustness to JPEG, crops, brightness, and contrast far exceeding post-hoc baselines. Fine-tuning is fast (the paper reports on the order of an hour on a single GPU) because only the decoder is touched. The honest limit: it degrades sharply under strong regeneration. This is the cleanest *identification* (multi-bit) watermark to study and reproduce.

**Tree-Ring (Wen et al., Maryland, 2023).** A 0-bit detection watermark in the Fourier domain of the initial noise, detected by DDIM inversion. Reported detection AUC near 0.999 against common distortions, and notably stronger robustness to a range of attacks than pixel-space watermarks — including, in their evaluation, better resistance to some regeneration than methods that mark pixels directly, because the signal lives in the seed. The trade-off: base Tree-Ring carries ~0 bits (detection only); multi-bit variants exist but spend robustness for capacity, exactly per the Shannon bound.

**Gaussian Shading (Yang et al., 2024).** A *performance-lossless* multi-bit watermark: the message is encoded into the sampling of the initial Gaussian noise such that the watermarked noise distribution is provably identical to the unwatermarked one. Because the distribution is unchanged, FID and image quality are unaffected by construction — a genuinely clever way to escape the imperceptibility cost. It carries a real payload and is robust to common distortions; like everything else, regeneration is its weak point.

**SynthID (Google DeepMind).** A production in-generation watermark + detector deployed across Imagen and Gemini image outputs, later extended to text and audio. Method details are proprietary, but it's designed for imperceptibility plus robustness to common edits, and it's the largest real-world deployment of generative watermarking. It's also the clearest signal of where industry is betting: watermark every output by default, and pair with provenance.

**C2PA / Content Credentials (industry consortium).** Now embedded by Adobe (Firefly, Photoshop), increasingly by OpenAI and Google for their image outputs, and supported in cameras from Leica/Sony/Nikon. The standard is mature; the open question is *durability* — manifests get stripped by ordinary re-saving — which is precisely why pairing C2PA with a surviving watermark (and a lookup registry) is the active frontier.

Here is the consolidated method-comparison table — the one to keep. "Regen survival" is the column nobody should hide.

| Method | In-generation? | Robust to crop/JPEG | Regen survival | Capacity | Detector cost |
|---|---|---|---|---|---|
| DwtDct (post-hoc) | No | Weak | None | ~32 bits | Cheap |
| RivaGAN (post-hoc) | No | Medium | None | ~32 bits | Cheap (NN) |
| Stable Signature | Yes (decoder) | Strong | Poor | 48 bits | Cheap (1 NN fwd) |
| Tree-Ring | Yes (noise/FFT) | Strong | Partial | 0-bit (base) | Expensive (DDIM invert) |
| Gaussian Shading | Yes (noise) | Strong | Poor | Multi-bit | Medium |
| SynthID | Yes | Strong | Degrades | Multi-bit | Cheap (detector) |
| C2PA manifest | N/A (metadata) | N/A (strippable) | N/A | Rich signed | Cheap (verify sig) |
| Passive classifier | No | Fragile | N/A | 1 bit (guess) | Cheap |

And the latency/cost reference for the runtime layers, on a named GPU, so you can budget:

| Runtime layer | Where | Approx cost (A100) | Notes |
|---|---|---|---|
| Prompt filter | Pre-gen | ~3–8 ms | One text-classifier forward |
| SDXL generation | — | ~3–4 s (30 steps) | The actual work |
| Output safety classifier | Post-gen | ~5–15 ms | One CLIP forward; <1% of total |
| Stable Signature mark | In decoder | ~0 ms extra | Mark is in the weights |
| Tree-Ring detect | Verification | ~1–2 s | Needs full DDIM inversion |
| C2PA sign | Post-gen | ~5–20 ms | Hash + sign + embed |

## 11. The policy backdrop, briefly

The technical stack does not exist in a vacuum; regulation is dragging it toward "watermark and label by default."

- **EU AI Act (in force, phasing in through 2025–2026).** Article 50 transparency obligations require providers of generative systems to mark synthetic audio, image, video, and text in a machine-readable way and ensure it's detectable as artificially generated — explicitly nodding to watermarks and metadata like C2PA. Deployers of deepfakes must disclose. This is the regulatory engine behind in-generation watermarking moving from "nice to have" to "required."
- **US.** No single federal mandate as of this writing, but a patchwork: state laws on election deepfakes and NCII, the NIST work on synthetic-content provenance, and voluntary White House commitments from major labs to watermark and red-team. C2PA adoption is partly an industry move to get ahead of regulation.
- **Provider commitments.** The big labs publicly committed to provenance (watermarking + Content Credentials), CSAM filtering, and red-teaming. These are not laws, but they shape the de facto baseline a serious product is measured against.

The practical upshot for a builder: if you ship an image model to users, plan for *labeling* (C2PA + watermark) and *filtering* (CSAM at minimum) as table stakes, not optional extras. The cost is modest (we measured it: runtime classifiers are <1% of generation cost; watermarking a decoder is an hour of fine-tuning); the downside of skipping it is regulatory, reputational, and — in the CSAM case — criminal.

## 12. When to reach for each layer (and when not to)

A decisive recommendation section, because every layer is a cost and some are over-applied.

- **Always filter training data for CSAM.** Non-negotiable, no trade-off, no exceptions. Use known-hash matching and report. This is the one place "robustness vs imperceptibility" is not a conversation.
- **Always run an output safety classifier on served endpoints.** It costs <1% of generation latency. Disabling it is a deliberate, logged policy decision — appropriate for a closed research environment, not for a public API. Don't ship a public endpoint with `safety_checker=None` and no replacement.
- **Watermark by default if you control the generator and have any abuse surface.** Stable Signature is the pragmatic choice for *identification* (cheap to fine-tune, cheap to detect, real multi-bit payload, no quality hit). Tree-Ring/Gaussian Shading are stronger for *detection* robustness but Tree-Ring's per-image DDIM-inversion detection cost (~1–2 s) makes bulk verification expensive — reach for it when robustness matters more than detection throughput.
- **Sign with C2PA if your outputs flow into a provenance-aware ecosystem** (journalism, stock imagery, regulated contexts). It's cheap to add and the EU AI Act is pushing it toward mandatory. But do *not* rely on it as your only attribution layer — it's stripped by a screenshot. Pair it with a watermark.
- **Don't trust a passive detector as a sole oracle — ever.** Especially not for high-stakes decisions (academic integrity, legal evidence, account bans). The base-rate math makes it unreliable, and it ages out as generators improve. Use it as a *weak prior* in an ensemble, with human review for consequences.
- **Don't oversell robustness.** If a stakeholder (like the fraud team from the intro) needs a guarantee, the honest answer is: "Our watermark attributes the honest majority and casual misuse with cryptographic confidence; a determined adversary with a diffusion model can launder it out, and no current method fully prevents that." Saying this up front is the difference between a tool people trust and one that fails them at the worst moment.
- **Reach for concept erasure (ESD/SafeGen) when filters aren't enough** — i.e., when jailbreaks are getting through prompt filters and you need the capability *gone* from the weights. But test benign quality after erasure, and know that determined fine-tuning can sometimes recover the concept.

## 13. Key takeaways

- **Passive detection is structurally losing.** A diffusion model is trained to be indistinguishable from real; the residual tells shrink as models improve, and base rates make a "95% accurate" detector nearly useless. Pivot to proactive attribution.
- **In-generation watermarks beat post-hoc ones** on robustness because the mark is woven into the generative process — the seed (Tree-Ring), the decoder (Stable Signature), or the noise sampling (Gaussian Shading) — with no clean version to strip.
- **Tree-Ring marks the noise, read by DDIM inversion; Stable Signature marks the decoder, read by a frozen extractor.** Both give you an *analytically computable* false-positive rate (chi-squared / binomial), which is exactly why keyed watermarks are trustworthy where passive detectors aren't.
- **The trilemma is a Shannon bound, not folklore.** Robustness × imperceptibility × capacity trade off; you get two, the third pays. That's why the most robust watermark is 0-bit, and why pushing capacity costs robustness.
- **Regeneration is the boss fight.** No current watermark fully survives a determined diffusion-purification attack. Watermarks raise the cost and cover the honest majority; they are not an unbreakable guarantee, and you should never claim they are.
- **Watermarking and C2PA are complementary, not redundant.** One survives stripping but is low-capacity; one carries rich signed provenance but is trivially stripped. Stack them — and binding a manifest hash into the watermark is the durable-provenance frontier.
- **The runtime safety layers are nearly free.** A prompt filter and an output classifier together cost <1% of generation latency. There is no performance excuse to omit them on a public endpoint.
- **Training-data filtering is the highest-leverage safety intervention,** and CSAM filtering is a hard zero-tolerance line with no trade-off.
- **Default to "unknown," not "real."** Absence of a watermark is not proof of authenticity.

This closes the defense layer of the stack. With provenance, watermarking, detection, and the safety pipeline understood, you have the full picture of what ships *around* a model — not just the model itself. The capstone, [building an image-generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack), assembles every piece of this series — VAE, diffusion, sampler, guidance, control, distillation, and this defense layer — into one end-to-end, deployable pipeline.

## 14. Further reading

- **Fernandez, Couairon, Jégou, Douze, Furon (2023), "The Stable Signature: Rooting Watermarks in Latent Diffusion Models."** The decoder-fine-tuning watermark; the cleanest multi-bit method to reproduce.
- **Wen, Kirchenbauer, Geiping, Goldstein (2023), "Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust."** The noise-domain watermark detected by DDIM inversion.
- **Yang, Zeng, et al. (2024), "Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models."** The distribution-preserving multi-bit watermark.
- **C2PA Technical Specification + Content Credentials (c2pa.org), and Adobe's open-source `c2pa` library.** The provenance standard and its reference implementation.
- **Zhao, Saberi, et al. (2024), "Invisible Image Watermarks Are Provably Removable Using Generative AI."** The regeneration-attack result — read this before you promise robustness.
- **Gandikota, Materzyńska, Fiotto-Kaufman, Bau (2023), "Erasing Concepts from Diffusion Models" (ESD).** Concept unlearning via the model's own negative guidance.
- **Within this series:** [Variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) (the decoder Stable Signature fine-tunes), [DDIM and fast deterministic sampling](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling) (the inversion Tree-Ring relies on), [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion), and the capstone [building an image-generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
- **Related:** [best resources for AI safety](/blog/machine-learning/ai-safety/best-resources-for-ai-safety) for the broader safety landscape these defenses sit within.
