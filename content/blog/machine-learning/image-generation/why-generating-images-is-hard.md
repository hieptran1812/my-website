---
title: "Why Generating Images Is Hard: A Map of the Whole Field"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A first-principles map of image generation: the high-dimensional pixel distribution, the manifold hypothesis, the generative trilemma, the four model families, and a runnable hello-world that paints a photo from noise in ten lines."
tags:
  [
    "image-generation",
    "diffusion-models",
    "generative-ai",
    "deep-learning",
    "vae",
    "gan",
    "autoregressive-models",
    "flow-matching",
    "stable-diffusion",
    "manifold-hypothesis",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/why-generating-images-is-hard-1.png"
---

Type "a photo of a corgi wearing sunglasses on a beach at golden hour" into FLUX or Midjourney, wait two seconds, and a 1024×1024 photograph appears that has never existed. The fur catches the light correctly. The shadows fall the right way. The sunglasses reflect a tiny beach. Nobody drew it; nobody photographed it. A neural network sampled it out of pure noise. The first time you see this work it feels like a magic trick, and like most magic tricks the interesting part is not the reveal but the years of engineering that make it look effortless.

This series is about that engineering. We are going to take you from "I have seen Midjourney" to "I understand the probability math, can read the papers, and can build, fine-tune, and serve a text-to-image model myself." Not a survey. A working understanding: the variational bound that produces the denoising loss, the reverse SDE that the sampler integrates, the exact extrapolation that classifier-free guidance performs, the flow-matching velocity field that lets FLUX sample in four steps. By the end of the series you will have run a training loop, swapped a sampler, fine-tuned a LoRA, and quantized a model to fit a 24 GB GPU. This first post is the map. It frames the *whole problem* and shows you where every later post fits.

![A tree diagram splitting the goal of learning the image distribution into explicit-likelihood and implicit-sampling branches, then into the four families VAE, autoregressive, GAN, and diffusion plus flow](/imgs/blogs/why-generating-images-is-hard-1.png)

Here is the thesis of the whole field in one sentence: **generating an image means drawing a sample from the probability distribution of natural images, $p(x)$, and that distribution lives in a space so vast and so strangely shaped that learning to sample from it is one of the hardest problems in machine learning.** Everything else — VAEs, GANs, autoregressive transformers, diffusion, flow matching — is a different bet on *how* to learn and sample from $p(x)$. To see why those bets diverge so wildly, and why diffusion won the 2022–2025 era while autoregressive models are surging back in 2025–2026, you first have to feel just how hard the underlying problem is. So let's make it concrete.

## What "generate an image" actually means

A digital image is just a grid of numbers. A modest 512×512 RGB image is $512 \times 512 \times 3 = 786{,}432$ numbers, each an 8-bit integer from 0 to 255. So a single image is a point in a 786,432-dimensional space — call it $\mathbb{R}^{786432}$, the **pixel space**. (A 1024×1024 image, the FLUX default, lives in a space of $3{,}145{,}728$ dimensions. The corgi photo above is one point in a space with over three million coordinates.)

Now the central question. If I asked you to pick a point in that space *uniformly at random* — every coordinate an independent random byte — what would you get? You would get television static. Salt-and-pepper noise. You would get it every single time, and you could sit there sampling random points until the heat death of the universe and never once stumble onto something that looks like a corgi, a face, a landscape, or anything a human would call an image. Natural images are an astronomically tiny, structured subset of pixel space.

This is the first hard truth, and it has a name.

### The manifold hypothesis

The **manifold hypothesis** says that real, natural data does not fill its ambient high-dimensional space. Instead it concentrates on a much lower-dimensional surface — a **manifold** — embedded inside that space. A manifold is, loosely, a smooth surface that locally looks flat: the 2D surface of a globe sits inside 3D space; the set of all photographs of human faces is a (very curved, very high-dimensional, but still comparatively thin) sheet sitting inside the 786,432-dimensional pixel cube.

Why must this be true? Pixels in a real photo are massively correlated. The pixel at $(100, 100)$ is almost always close in value to the pixel at $(101, 100)$ next to it — edges and smooth regions, not independent noise. There are strong global constraints too: faces have two eyes above a nose above a mouth; the sky is at the top; lighting is consistent across a scene. Each constraint collapses the effective number of free dimensions. The set of *valid* images is the intersection of millions of such constraints, and that intersection is a thin, curved, lower-dimensional manifold. Estimates of the **intrinsic dimension** of natural-image datasets put it in the tens to low hundreds, not the hundreds of thousands of the ambient pixel space (Pope et al., "The Intrinsic Dimension of Images and Its Impact on Learning," ICLR 2021, found intrinsic dimensions roughly in the 20–60 range for common image datasets). The ambient space is enormous; the manifold the data actually lives on is comparatively tiny.

![A graph showing pixel space as a large cube where almost every random point is static, a thin image manifold sheet holds real photos, points just off the manifold look broken, and the generator must learn the sheet](/imgs/blogs/why-generating-images-is-hard-2.png)

This single picture explains why naive approaches fail and why the whole field is shaped the way it is. Consider what it means for the generator's job:

- **Almost all of the space is off-manifold.** If your model puts any probability mass on the wrong region, it generates garbage — static, color blobs, melted faces. A point even a little off the manifold does not look like a *slightly worse* image; it looks broken, because the manifold is thin and the off-manifold neighborhood is structureless.
- **The manifold is curved and disconnected.** "Corgi on a beach" and "diagram of a transformer" are both natural images, but they live in very different regions, possibly different connected components. A good generator has to cover *all* the regions (this is the *diversity* or *mode coverage* requirement) and stay *on* the surface everywhere (the *quality* requirement). Those two goals fight each other, which we will formalize as the trilemma in a moment.
- **You only have samples, never the density.** Your training set (LAION, ImageNet, your own photos) is a finite pile of points *on* the manifold. You never get told "here is $p(x)$ as a formula." You have to *infer* the shape of an unimaginably high-dimensional surface from a few hundred million example points sitting on it. That is the learning problem.

#### Worked example: how thin is the manifold?

Let's put numbers on "astronomically tiny." Suppose, generously, that the set of natural 512×512 images forms a manifold of intrinsic dimension 50 (already an over-estimate for a single visual domain). The ambient pixel space has $786{,}432$ dimensions. The fraction of "directions" that keep you on the manifold versus the directions that take you off it is roughly $50 / 786{,}432 \approx 6 \times 10^{-5}$. Move randomly and you leave the manifold with probability essentially 1. Now count *configurations*: pixel space has $256^{786432}$ possible images, a number with about 1.9 million digits. The number of natural images a human would accept is mind-bogglingly smaller — a rounding error of a rounding error. **The generator's entire job is to assign nearly all of its probability to that rounding error and nearly zero to everything else.** That is why a model that "almost works" produces convincing-looking garbage: it is sitting just off the sheet.

There is a second, subtler consequence of high dimensionality that trips people up, and it is the key reason diffusion's gradual approach works. In high dimensions, almost all of a Gaussian's probability mass sits not at the center but in a thin *shell* at radius roughly $\sqrt{d}$ from the mean — the "soap-bubble" phenomenon. So when you sample pure noise as your starting point, you are not starting "near" any image; you are starting on a vast sphere from which the data manifold is a faint, distant target. A one-shot generator has to hit that target in a single leap across a hostile, mostly-empty space. A diffusion model instead lays down a *trail of breadcrumbs* — a sequence of intermediate distributions, each only slightly noisier than the last — so that at every step the model only has to take a small, well-posed denoising stride toward slightly-cleaner data. The geometry of high dimensions is precisely what makes the one-shot problem brutal and the many-small-steps problem tractable, which is the deepest reason the iterative families dominate on quality. Keep that picture in mind; the diffusion track formalizes it as a stochastic differential equation, but the geometric intuition is the whole story.

So the problem is not "produce a 786,432-number array." A random number generator does that instantly. The problem is "produce a 786,432-number array that lands on an unknown, thin, curved manifold you have only ever seen samples from." That is the entire game, and the four families are four strategies for playing it.

### Why the naive approaches all fail

It is worth seeing exactly why the obvious ideas break, because each failure motivates a family. The first instinct of anyone who has done classical statistics is to *estimate the density directly*: fit $p(x)$ with a histogram, a kernel density estimate, or a Gaussian mixture, then sample from the fit. In one or two dimensions this works fine. In 786,432 dimensions it is hopeless. A histogram with even two bins per dimension has $2^{786432}$ cells — you could never fill more than a vanishing fraction of them with any finite dataset, so almost every cell is empty and your density is zero almost everywhere. Kernel density estimation suffers the same **curse of dimensionality**: the volume of a high-dimensional ball grows so fast that any fixed-bandwidth kernel either smears across the whole space (useless) or covers nothing (useless). Gaussian mixtures need an astronomical number of components to wrap a curved manifold and still produce blurry averages between data points. Classical density estimation simply does not survive contact with this many dimensions.

The second instinct is to *interpolate the training set*: store all the images and blend between them. But the manifold is curved, so a straight-line average of two on-manifold images is off-manifold — average a photo of a face turned left with one turned right and you get a ghostly double-exposure, not a forward-facing face. Linear interpolation in pixel space cuts a chord *through* the manifold rather than following its surface. (This is precisely why VAEs and diffusion models interpolate in a *learned latent space* where the manifold is unfolded into something closer to flat — a point we will make rigorous in the VAE post.)

The third instinct is to *learn a deterministic function* from a fixed input to an image, like a regressor. But generation is inherently *one-to-many*: the prompt "a cat" corresponds to billions of valid images, not one. A deterministic network trained with a pixel loss to map "cat" to images will, under that one-to-many uncertainty, output the *average* of all valid cats — a brown blur. The only fix is to inject randomness (a noise input $z$) and learn a *distribution*, which is exactly what all four families do. They differ only in how they shape and sample that distribution.

So every family is a response to the same three failures: density estimation dies of dimensionality, interpolation leaves the curved manifold, and deterministic regression averages away the diversity. The winning idea, across all four families, is to learn a *mapping from simple noise to the data manifold* — and the families are four different ways to learn and invert that mapping.

## The generative trilemma: the frame for everything

Before we meet the families, install the mental model that recurs in every post of this series. A good image generator wants three things at once, and for most of the field's history you could pick two:

1. **High sample quality** — generated images are sharp, coherent, and indistinguishable from real ones. Measured by FID (Fréchet Inception Distance — lower is better), human preference, and "does it have six fingers?"
2. **High mode coverage / diversity** — the model can produce *every* kind of image in the data, not just a few it has latched onto. A model that only ever makes one perfect cat has terrible coverage. Measured by recall, by FID's diversity component, and by whether it ever surprises you.
3. **Fast sampling** — you get an image in one or a few neural-network evaluations, not a thousand. Measured in steps and in seconds-per-image on a named GPU.

![A graph linking the three trilemma corners quality, diversity, and speed to the three classic families, with GAN reaching quality and speed, VAE reaching diversity and speed, and diffusion reaching quality and diversity](/imgs/blogs/why-generating-images-is-hard-3.png)

This is the **generative trilemma**, named in the diffusion literature (Xiao, Kreis, Vahdat, "Tackling the Generative Learning Trilemma with Denoising Diffusion GANs," ICLR 2022). The classic positioning:

- **GANs** nail *quality* and *speed* (one forward pass, razor-sharp images) but struggle with *diversity* — they drop modes, sometimes catastrophically.
- **VAEs** nail *diversity* and *speed* (one forward pass, full coverage by construction) but lose *quality* — they blur.
- **Diffusion** nails *quality* and *diversity* (sharp *and* full coverage) but historically lost on *speed* — it needed hundreds or thousands of network evaluations per image.

The reason diffusion came to dominate is partly that the *speed* corner turned out to be fixable by engineering — better samplers (DPM-Solver, UniPC) cut 1000 steps to 20, and distillation (LCM, Turbo, DMD) cut 20 to 1–4 — whereas the *quality* and *diversity* corners that GANs and VAEs sacrificed turned out to be much harder to recover. **It is easier to make a high-quality-but-slow model fast than to make a fast-but-low-quality model good.** That single asymmetry explains a lot of the last five years. Keep the trilemma in your head; every technique in this series is a move on this triangle.

### Why the three goals genuinely conflict

The trilemma is not just an empirical observation that "you can pick two"; there are real mechanisms that make the corners pull against each other, and understanding them tells you which knob hurts which axis. Take **quality versus diversity** first. A model that perfectly covers the data distribution will, by definition, also place a little probability on the low-density tails — the weird, rare, slightly-off images. A model that *only* generates flawless images is implicitly truncating those tails, which means it is no longer covering the full distribution. This is exactly what the **truncation trick** in GANs does: sample $z$ from a *narrowed* Gaussian and image quality goes up while diversity goes down, on a smooth, measurable curve. Classifier-free guidance in diffusion does the same thing — turning up the guidance scale sharpens prompt adherence and perceived quality while provably shrinking the diversity of outputs (we derive the exact extrapolation in the guidance post). The quality–diversity axis is a *truncation dial*, and every family has one.

Now **quality versus speed**. Quality, in the sense of landing precisely on the thin manifold, requires the generator to make a *complicated, curved* trip from noise to data. A one-step model has to approximate that entire curved trip with a single function evaluation, which is a brutally hard regression — the target is a multi-valued, sharply curved map. An iterative model gets to take many small, locally-easy steps, correcting course as it goes, which is why diffusion's per-step problem is so well-posed and why more steps buy more quality up to a point. Fewer steps means each step must do more, which means a harder learning problem, which (without distillation tricks) means lower quality. The step count is a *compute-for-quality* dial.

Finally **diversity versus speed** interacts through the *stochasticity* of sampling. SDE (ancestral) samplers inject fresh noise at every step, which helps coverage but costs steps; deterministic ODE samplers (DDIM) are faster and more reproducible but can slightly reduce diversity. So even the choice of sampler trades a little coverage for a little speed. Three axes, three dials, all coupled — that is what makes "just make it better" an underspecified instruction and why every paper in this field reports *where on the trilemma* it sits, not a single scalar.

#### Worked example: reading a quality-diversity curve

Concretely, suppose you sweep classifier-free guidance scale $w$ from 1 to 12 on a fixed model, generating 10,000 images per setting against the COCO validation prompts, and you measure two numbers at each setting: FID (lower = better quality + coverage match) and CLIP-score (higher = better prompt adherence). What you will see, reproducibly, is a *U-shaped FID curve* and a monotonically rising CLIP-score. At $w=1$ (no guidance) CLIP-score is low — the image only loosely matches the prompt — but diversity is high. As $w$ rises to roughly 3–7, CLIP-score climbs and FID *drops* to a minimum: you are buying prompt adherence cheaply. Past $w \approx 7$–8, CLIP-score keeps rising but FID turns *upward* again — the images are now over-saturated and mode-collapsed toward a few "prototypical" renderings, so coverage (and thus FID) gets worse even as each individual image screams the prompt louder. The sweet spot is the bottom of the FID U, typically $w \approx 5$–7 for SDXL-class models. That curve *is* the quality–diversity trade-off made measurable, and reading it is a skill every later post sharpens.

## The four families: four bets on p(x)

Now the map. There are exactly four families of deep generative model that have produced state-of-the-art images, and they split cleanly by *how they model $p(x)$*. The split at the top is **explicit vs implicit**: does the model give you a number for the likelihood $p(x)$ (explicit) or only the ability to draw samples (implicit)?

### Family 1: Variational Autoencoders (VAEs)

**Core idea.** Compress each image into a small **latent** vector $z$ (an *encoder*), then learn to reconstruct the image from $z$ (a *decoder*), while forcing the latents to follow a simple prior like a unit Gaussian. To generate, you sample $z \sim \mathcal{N}(0, I)$ and run the decoder. Because the latent space is low-dimensional and smooth, you are sampling near the manifold by construction. The training objective is the **ELBO** (Evidence Lower BOund), a tractable lower bound on $\log p(x)$ — so the VAE is an *explicit, approximate-likelihood* model. We will derive the ELBO from scratch in [the VAE post](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch).

**One strength.** Stable training and a clean, structured latent space you can interpolate and manipulate. VAEs essentially never collapse, and you can walk the latent space to morph one image into another.

**One weakness.** Blur. A plain VAE optimizes a pixel-wise reconstruction loss, which under uncertainty averages plausible outputs together, and averaging sharp images gives a soft one. VAE samples look like someone smeared the photo.

**Current status.** Nobody ships a *plain* VAE as a final image generator — but the VAE is the unsung hero of the entire modern stack. Stable Diffusion, SDXL, SD3, and FLUX all run diffusion *inside a VAE's latent space*. The VAE compresses 512×512×3 pixels down to a 4×64×64 latent (a 48× reduction), the diffusion model does its work in that small space, and the VAE decoder paints the final pixels. VQ-VAE and VQ-GAN, which use *discrete* latent codes, are the tokenizers that make autoregressive image models possible. So the VAE didn't lose; it became infrastructure. That is why it is the third post in the series.

### Family 2: Generative Adversarial Networks (GANs)

**Core idea.** Two networks play a game. A **generator** $G$ maps noise $z$ to an image; a **discriminator** $D$ tries to tell real images from generated ones. They train adversarially: $G$ tries to fool $D$, $D$ tries to catch $G$. At the equilibrium of this minimax game, $G$'s outputs are indistinguishable from real data. GANs are *implicit* — there is no $p(x)$ formula, only a sampler. The objective is the famous minimax:

$$\min_G \max_D \; \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))].$$

**One strength.** Speed and sharpness. A GAN generates in a *single* forward pass, and the adversarial loss pushes hard toward the realistic manifold, so outputs are crisp. StyleGAN faces hit FID around 2.8 on FFHQ and were, for years, the sharpest generative images anywhere.

**One weakness.** Training instability and **mode collapse**. The minimax game has no stable loss curve to watch; it can oscillate, diverge, or collapse to producing a handful of near-identical outputs (covering only part of the data — the diversity corner of the trilemma). Getting a GAN to train on an open-domain dataset is famously finicky.

**Current status.** GANs lost the headline text-to-image race to diffusion around 2021–2022 — they couldn't scale to messy, open-domain, text-conditioned generation without collapsing. But the adversarial loss came back through the side door: modern **distillation** methods (ADD in SDXL-Turbo, DMD2) use a discriminator as a *training signal* to compress a slow diffusion model into a one- or four-step generator. So GANs, like VAEs, became a component. We cover [why GANs lost and where they still win](/blog/machine-learning/image-generation/generative-adversarial-networks-and-why-they-lost) as the fourth post.

### Family 3: Autoregressive / token models

**Core idea.** Treat an image as a *sequence* and predict it one piece at a time, exactly like a language model predicts text one token at a time. Factor the joint distribution by the chain rule:

$$p(x) = \prod_{i=1}^{n} p(x_i \mid x_1, \dots, x_{i-1}).$$

Early versions (PixelRNN, PixelCNN, Image GPT) predicted raw pixels one at a time. Modern versions first *tokenize* the image with a VQ-VAE/VQ-GAN into a grid of discrete codes, then a transformer predicts those codes autoregressively — a "language model for image tokens." Newer variants change the *order*: VAR predicts coarse-to-fine "next-scale" instead of raster order; MAR predicts a *masked* set of tokens without a discrete codebook.

**One strength.** Exact likelihood and a clean training signal — it is just next-token cross-entropy, the most battle-tested objective in deep learning, and it scales predictably with model size and data. It also unifies trivially with text: the same transformer can model text *and* image tokens, which is the whole premise of native-multimodal models.

**One weakness.** Slow sampling. Generating an image means a sequential loop over hundreds or thousands of tokens, each requiring a forward pass — inherently serial, unlike diffusion's parallel-per-step denoising. Tokenization with a discrete codebook also caps reconstruction quality.

**Current status.** Surging back. After diffusion's dominance, 2025–2026 brought a strong autoregressive revival: OpenAI's GPT-Image (the "4o image generation" model) is a native-multimodal autoregressive system; VAR (NeurIPS 2024 best paper) showed next-scale prediction beating diffusion transformers on ImageNet with better scaling; MAR removed the need for vector quantization; and Tencent's HunyuanImage-3.0 (2025) is a large autoregressive multimodal generator. The bet is that the same transformer recipe that scaled language will scale image generation and unify the two. We cover [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models) as the fifth post and revisit the AR-vs-diffusion showdown near the end of the series.

### Family 4: Diffusion and flow matching (the current frontier)

**Core idea.** Don't try to jump from noise to image in one shot. Instead, define a gradual **forward process** that destroys an image by adding a little Gaussian noise at a time over many steps until it is pure static — and then *learn to reverse it*, denoising step by step from static back to an image. The network's job at each step is small and well-posed: given a noisy image, predict the noise (or equivalently, predict a slightly cleaner image). Stack hundreds of these tiny, easy denoising steps and you walk from $\mathcal{N}(0, I)$ all the way onto the image manifold. **Flow matching** is the continuous-time cousin: instead of a noising chain it learns a *velocity field* that transports noise to data along (ideally straight) paths, which is what SD3 and FLUX use.

**One strength.** It hits *both* quality and diversity — sharp images *and* full mode coverage — with a *stable* training loss (a simple regression, no adversarial game). The denoising objective is essentially mean-squared error on predicted noise; you can watch it go down. That stability plus quality plus coverage is why diffusion took over.

**One weakness.** Speed, historically. Walking from noise to image took hundreds or thousands of network evaluations. This is the trilemma corner diffusion sacrificed — and the one the field spent 2022–2025 furiously fixing with better samplers and distillation.

**Current status.** The frontier. From 2022 to 2025 essentially every state-of-the-art open and closed text-to-image model was diffusion or flow-matching: Stable Diffusion, SDXL, SD3, FLUX, SANA, DALL·E 3, Imagen, Midjourney. The slow-sampling weakness got engineered away (FLUX.1-schnell and SDXL-Turbo generate in 1–4 steps; SANA generates a 1024px image in well under a second). [Diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) is the foundation of the entire second track of this series.

![A before-and-after diagram contrasting the GAN era's unstable minimax game and mode collapse with the diffusion era's stable denoising loss, full coverage, and open-domain text-to-image](/imgs/blogs/why-generating-images-is-hard-4.png)

The before/after above captures the regime change. The GAN era fought an unstable adversarial game and dropped modes; the diffusion era traded that for a stable regression loss that covers the whole distribution and accepts text conditioning naturally. That trade is *the* reason the field's center of gravity moved.

## What each family actually optimizes

Keeping the math light is the rule for this orientation post, but you should see the *one defining objective* of each family side by side — because the differences in those objectives are the entire reason the families behave so differently on the trilemma. Four losses, four personalities.

**VAE — maximize a lower bound on the log-likelihood.** Because computing $\log p(x) = \log \int p(x \mid z)\, p(z)\, dz$ is intractable (you cannot integrate over all latents), the VAE introduces an approximate posterior $q_\phi(z \mid x)$ and optimizes the **ELBO**:

$$\log p(x) \;\ge\; \mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(x \mid z)\big] \;-\; D_{\mathrm{KL}}\!\big(q_\phi(z\mid x)\,\|\,p(z)\big).$$

The first term is reconstruction (decode $z$ back to $x$); the second pulls the encoder's latent distribution toward the prior $p(z) = \mathcal{N}(0,I)$ so you can later sample from it. This is a *principled likelihood* objective, which is why VAEs are stable and cover all modes — but the reconstruction term is typically a Gaussian (pixel MSE) likelihood, and the KL term forces the latent to be lossy, and *both* push toward averaged, blurry outputs. The blur is baked into the objective, not a bug.

**GAN — minimize a divergence implicitly, through a game.** The generator never sees a likelihood; it only sees the discriminator's gradient. At the optimal discriminator, the minimax objective is equivalent to minimizing the Jensen–Shannon divergence between $p_\text{data}$ and the generator's distribution. The sharpness comes from the fact that the discriminator can punish *any* off-manifold detail — there is no pixel-averaging term to blur things. The instability and mode collapse come from the same source: the objective is a *saddle point*, not a minimum, and a generator can lower its loss by perfectly nailing a *subset* of modes and ignoring the rest (the discriminator can be fooled mode-by-mode). Quality and collapse are two faces of the same adversarial coin.

**Autoregressive — minimize next-token cross-entropy (exact maximum likelihood).** Factor $p(x) = \prod_i p(x_i \mid x_{<i})$ and the loss is just $-\sum_i \log p_\theta(x_i \mid x_{<i})$ — the same cross-entropy that trains every language model. It is *exact* maximum likelihood (no bound, no game), which is why it is rock-stable and covers modes well, and why it scales so predictably with data and parameters. The cost is purely at sampling time: you must generate token by token, $O(n)$ serial passes, and a discrete tokenizer caps fidelity.

**Diffusion — minimize a denoising regression (a re-weighted likelihood bound).** The training objective reduces, after the DDPM derivation we cover in Track B, to a strikingly simple form:

$$\mathcal{L}_\text{simple} = \mathbb{E}_{x_0,\,\epsilon,\,t}\Big[\,\big\|\,\epsilon - \epsilon_\theta(x_t, t)\,\big\|^2\,\Big],$$

where $x_t$ is the clean image $x_0$ with a known amount of Gaussian noise $\epsilon$ added, and the network $\epsilon_\theta$ predicts that noise. It is mean-squared error — a regression — which is why training is so stable (you can watch the curve fall). Yet because it is, at its mathematical core, a particular weighting of the variational bound on $\log p(x)$, it inherits full mode coverage. And because it never tries to make the *whole* jump from noise to image in one shot — only to predict a little noise at each level — it sidesteps the one-to-many blur that sinks deterministic regressors. That combination of a simple stable loss with full coverage and sharp samples is the reason diffusion sits where it does on the trilemma. Flow matching swaps the noise-prediction target for a *velocity* target along a straight path, keeping the regression simplicity while shortening the trip.

Lay these four objectives next to each other and the whole landscape clicks into place: likelihood bound that blurs (VAE), adversarial game that's sharp but collapses (GAN), exact likelihood that's stable but serial (AR), and denoising regression that's stable, sharp, *and* covers — at the cost of many steps (diffusion). The trilemma is not arbitrary; it falls directly out of what each family is allowed to optimize.

## A results table: the four families head to head

Words are imprecise; let's make the comparison concrete. Here is how the four families stack up on the dimensions that matter, with the caveat that every cell is a *typical* characterization, not a law — a well-tuned member of a "weak" family can beat a careless member of a "strong" one.

![A matrix comparing VAE, GAN, autoregressive, and diffusion across quality, diversity, speed, training stability, and likelihood, showing diffusion strong on quality and diversity and autoregressive strong on likelihood](/imgs/blogs/why-generating-images-is-hard-7.png)

| Family | Sample quality | Diversity / coverage | Sampling speed | Training stability | Likelihood |
| --- | --- | --- | --- | --- | --- |
| **VAE** | Low (blurry) | High (by construction) | Fast (1 step) | Very stable | Explicit lower bound (ELBO) |
| **GAN** | High (sharp) | Low (mode collapse risk) | Fast (1 step) | Fragile (minimax) | None (implicit) |
| **Autoregressive** | High | High | Slow (sequential, $O(n)$ passes) | Stable (cross-entropy) | Exact |
| **Diffusion / flow** | State of the art | High | Iterative (1–50 steps after distillation) | Stable (regression) | Variational bound / exact ODE |

Read this table through the trilemma lens. VAE: diversity + speed, weak quality. GAN: quality + speed, weak diversity. Autoregressive: quality + diversity + exact likelihood, but slow. Diffusion: quality + diversity + stability, originally slow but now fixed. The autoregressive and diffusion rows are the only two that get quality *and* diversity, which is exactly why those two are the families fighting for the 2026 frontier — and why the older two became components rather than headline generators.

## The timeline: how we got here

The field moved fast. Here is the arc, which doubles as a reading order for the rest of the series.

![A timeline from 2020 to 2025 marking DDPM with 1000 steps, the score SDE unification, latent diffusion and SD1.5, the DiT transformer backbone, SD3 and FLUX flow matching, and Turbo and SANA few-step models](/imgs/blogs/why-generating-images-is-hard-5.png)

- **2014–2020, the GAN era.** GANs (Goodfellow et al., 2014) and then StyleGAN (Karras et al., 2019) define state-of-the-art image synthesis. VAEs (Kingma & Welling, 2013) develop in parallel as the stable-but-blurry alternative. Autoregressive PixelCNN and Image GPT show the token paradigm but are too slow to compete on large images.
- **2020, DDPM.** Ho, Jain & Abbeel publish "Denoising Diffusion Probabilistic Models." It works beautifully but needs **1000** sampling steps. The denoising objective is shown to reduce to a clean noise-prediction MSE. This is the spark.
- **2021, the score-SDE unification.** Song et al. ("Score-Based Generative Modeling through Stochastic Differential Equations") show diffusion is one instance of a reverse-time SDE driven by the *score* $\nabla_x \log p(x)$, unifying DDPM and score matching and introducing the probability-flow ODE. Also 2021: DDIM (Song, Meng, Ermon) makes sampling deterministic and cuts steps from 1000 to ~50 *without retraining*. Diffusion beats GANs on ImageNet (Dhariwal & Nichol).
- **2022, latent diffusion = Stable Diffusion.** Rombach et al. ("High-Resolution Image Synthesis with Latent Diffusion Models") move diffusion into a VAE's latent space, cutting compute ~48× and making text-to-image runnable on a single consumer GPU. Stable Diffusion 1.5 is released openly and the floodgates open. Classifier-free guidance (Ho & Salimans) becomes the standard fidelity knob.
- **2023, DiT and SDXL.** Peebles & Xie ("Scalable Diffusion Models with Transformers") replace the U-Net with a transformer (DiT) and show a clean compute↔FID scaling law. SDXL scales latent diffusion with dual text encoders and a refiner. Consistency models (Song et al.) introduce few-step generation. Flow matching (Lipman et al., 2023) and rectified flow (Liu et al.) lay the continuous-time foundation.
- **2024, flow-matching transformers.** Stable Diffusion 3 (Esser et al., MM-DiT) and FLUX.1 adopt flow matching on transformer backbones and become the new open frontier. FLUX.1-schnell distills to 1–4 step sampling.
- **2025–2026, few-step and the AR resurgence.** SANA (NVIDIA/MIT) uses a 32× deep-compression autoencoder plus linear attention to generate 1024px images in well under a second. Distillation (LCM, Turbo, DMD2) makes one- to four-step generation routine. And autoregressive/native-multimodal models — GPT-Image, VAR, MAR, HunyuanImage-3.0 — surge back, betting that the transformer recipe will unify image and text generation.

The pattern is worth internalizing: **almost every leap came from changing the representation or the training objective, not from raw scale alone.** Latent space (LDM) changed *where* you diffuse. DiT changed the *backbone*. Flow matching changed the *path*. Distillation changed the *step count*. Each is a separate post because each is a separate, learnable idea.

### Why diffusion won the 2022–2025 era

Step back and ask *why* diffusion, specifically, ran the table for those four years. Four reasons, in rough order of importance.

First, **the loss is a stable regression.** Training a GAN is an adversarial balancing act that diverges if the generator and discriminator fall out of step; training a diffusion model is minimizing a mean-squared error you can watch fall monotonically. That stability is not a minor convenience — it is what let the field scale models to billions of parameters and train on billion-image datasets without the runs collapsing. You cannot scale what you cannot stabilize.

Second, **it covers modes by construction.** Because diffusion optimizes (a weighting of) the likelihood, it has no incentive to drop modes the way a GAN does. Point it at an open-domain dataset of every conceivable image and it learns the *whole* distribution, which is exactly what open-ended text-to-image demands. GANs choked on this breadth; diffusion thrived on it.

Third, **conditioning is trivial to bolt on.** The denoiser takes the timestep and the text embedding as just more inputs, and cross-attention to text tokens slots in cleanly. There is no architectural fight to make generation prompt-controllable — and once it is, the same hook accepts images (IP-Adapter), structure (ControlNet), and edits. The control ecosystem that makes diffusion *useful* in production grew so fast precisely because the architecture welcomed it.

Fourth, **the speed weakness was engineerable.** This is the asymmetry from the trilemma section, and it is the linchpin. Diffusion's one real liability was step count, and step count yielded to better numerical solvers and then to distillation. A model that is stable, covers modes, and conditions easily — and whose only flaw is fixable — is a model that wins. By 2025 a distilled flow-matching model gave you GAN-like single-digit step counts with diffusion-like quality and coverage, collapsing the trilemma into a single point. That is what dominance looks like.

### Why autoregressive models are surging back in 2025–2026

And yet the story did not end with diffusion. Through 2025 and into 2026, autoregressive image generation came roaring back, and it is worth understanding why a "slower" paradigm is gaining, because it tells you where the field is heading.

The driver is **unification**. A diffusion model generates images but cannot natively *converse*, *reason*, or *interleave* text and images in a single stream. An autoregressive transformer can: if images are just tokens, then one transformer can read a paragraph, look at an image, reason about it in text, and emit a new image — all as one sequence, with one objective. OpenAI's GPT-Image (the 4o image-generation model) demonstrated that this native-multimodal route produces strikingly coherent, instruction-following images and edits, with the conversational, in-context behavior that diffusion models had to bolt on. That coherence-from-unification is the prize.

The second driver is **scaling pedigree**. Next-token prediction is the single best-understood scaling recipe in all of deep learning — we have years of laws telling us exactly how loss falls with parameters and data. VAR (Visual Autoregressive modeling, a NeurIPS 2024 best-paper award) showed that a *next-scale* autoregressive formulation not only beats diffusion transformers on ImageNet generation but exhibits clean, language-model-like scaling laws and zero-shot generalization. MAR (masked autoregressive) removed the long-standing fidelity ceiling of discrete tokenizers by doing autoregression in a *continuous* space with a small per-token diffusion head — quietly erasing AR's biggest quality handicap. Tencent's HunyuanImage-3.0 (2025) scaled a unified autoregressive multimodal generator to the frontier of open models.

The third driver is **the boundary is blurring**. MAR uses a tiny diffusion loss inside an autoregressive loop; some diffusion models borrow autoregressive ordering; unified architectures like Transfusion and Chameleon train one transformer on *both* a next-token loss for text and a diffusion loss for images. The 2026 reality is less "diffusion versus autoregressive" and more "a transformer backbone with a choice of generative head," and the interesting research is in the convergence. We give this its own showdown post late in the series. For now, the headline is: **diffusion won the image-quality race of 2022–2025; autoregressive and unified multimodal models are the bet on the next era because they fold image generation into the same engine that already scaled language.**

## The diffusion stack: the spine of the series

Because diffusion (and flow matching) is the frontier, the series is organized around a single pipeline that every modern text-to-image model instantiates. Learn this stack and you can place any new model, any new paper, and any new optimization onto it.

![A vertical stack diagram of the latent diffusion pipeline from text encoder through VAE latent, forward noising, denoiser network, ODE or SDE sampler, classifier-free guidance, to the decoded final image](/imgs/blogs/why-generating-images-is-hard-6.png)

Top to bottom:

1. **Text encoder** (CLIP, T5, or an LLM) turns the prompt into conditioning vectors. *Track D covers this.*
2. **VAE encoder** compresses pixels to a small latent (4×64×64, a 48× reduction). Diffusion happens *here*, not in pixel space. *Track A (VAE) and Track C (LDM) cover this.*
3. **Forward noising** $q(x_t \mid x_0)$ gradually corrupts the latent to Gaussian noise — this is fixed, not learned, and defines the training targets. *Track B covers this.*
4. **Denoiser network** (U-Net or DiT) predicts the noise to remove at each step, conditioned on the timestep and the text. This is the only big learned component. *Track C covers this.*
5. **ODE/SDE sampler** integrates the reverse process, turning a thousand-step ideal into 4–50 practical steps (DDIM, DPM-Solver, UniPC, Euler). *Track B covers this.*
6. **Guidance** (classifier-free guidance) sharpens prompt adherence by extrapolating between conditional and unconditional predictions. *Track B covers this.*
7. **VAE decoder** paints the final pixels from the denoised latent. *Back to the VAE.*

Every post in this series is, in effect, a deep zoom into one of these seven boxes — or a technique for making one of them faster (Track E), more controllable (Track D), or properly evaluated (Track F). Hold this stack in your head and the rest of the series has a place to live.

Two properties of this stack are worth stating now, because they are not obvious and they recur constantly. First, **only one box is heavy with learned parameters**: the denoiser. The VAE is comparatively small and trained once; the text encoders are frozen pretrained models; the forward noising is a *fixed* mathematical schedule with no parameters at all; the sampler is a numerical integrator (also no learned parameters); guidance is an arithmetic operation on two denoiser outputs. So when someone says a model "has 12B parameters," they almost always mean the *denoiser* has 12B — the rest of the stack is shared infrastructure. That is why architecture (Track C) is about the denoiser, and why scaling laws for image models are scaling laws for that one box.

Second, **the stack is modular and swappable**, which is the entire reason the ecosystem moves so fast. You can keep the denoiser and swap the *sampler* (DDIM for DPM-Solver) to cut steps with zero retraining. You can keep everything and swap the *VAE* for a more aggressive 32× one (SANA) to speed up decode. You can freeze the denoiser and add a small *adapter* (LoRA, ControlNet, IP-Adapter) to change the style or add structural control without touching the base weights. You can replace the *text encoder* (CLIP for T5 or an LLM) to improve prompt understanding. Each of these swaps is a thriving subfield and a post in this series, and they compose because the interfaces between boxes are clean tensors. Internalize the seven boxes and you will be able to read any new image-generation paper as "they changed box $k$ and here is what it bought."

## Hello world: paint an image from noise in ten lines

Enough theory for one section. Let's actually generate an image. The cleanest way to feel the whole stack in action is 🤗 `diffusers`, the standard library for this work. We'll use **FLUX.1-schnell**, a flow-matching transformer distilled to sample in just **four steps** — so you produce an image in seconds rather than minutes. (If your GPU is tight on VRAM, SDXL-Turbo is an even smaller one-to-four-step alternative; swap the commented lines.)

```bash
pip install -U diffusers transformers accelerate sentencepiece torch
```

```python
import torch
from diffusers import FluxPipeline

# Load FLUX.1-schnell: a flow-matching transformer distilled to 4 steps.
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()  # fits in ~12-16 GB by streaming weights to GPU

image = pipe(
    "a photo of a corgi wearing sunglasses on a beach at golden hour",
    num_inference_steps=4,       # schnell is distilled for ~4 steps
    guidance_scale=0.0,          # schnell is guidance-distilled; no CFG needed
    height=1024, width=1024,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

image.save("corgi.png")
print("Saved corgi.png")
```

Ten lines of real code and you have walked the entire stack: the prompt went through a text encoder, the transformer denoised a latent over four flow-matching steps, and the VAE decoded it to a 1024×1024 photo. On an RTX 4090 (24 GB) this runs in roughly **0.5–1 second** per image at four steps; on an A100 it is faster still. Compare that to original DDPM, which needed **1000** steps and tens of seconds for a *much smaller* image. The 250×-plus speedup between those two snippets *is* the engineering story of this series, compressed into one benchmark.

Walk the call line by line, because every argument is a knob a later post turns:

- `from_pretrained("black-forest-labs/FLUX.1-schnell")` downloads four components in one bundle: the flow-matching transformer (the denoiser), the VAE (encoder + decoder), and two text encoders (CLIP and T5). The `FluxPipeline` wires them together so you do not have to.
- `torch_dtype=torch.bfloat16` loads weights in 16-bit brain-float instead of 32-bit, halving VRAM and roughly doubling throughput on modern GPUs at no visible quality cost — the single highest-leverage one-liner in the whole snippet.
- `enable_model_cpu_offload()` keeps idle components on the CPU and streams each onto the GPU only while it runs, so the ~12B-parameter model fits in 12–16 GB instead of needing 24+. It trades a little latency for a lot of memory headroom; on a big GPU you would drop it.
- `num_inference_steps=4` is the *step count* — the heart of the speed corner. FLUX.1-schnell was distilled to produce a good image in four denoiser evaluations; push it to 1 and quality drops, push it to 50 and you waste time for no gain (the model was not trained for that regime).
- `guidance_scale=0.0` turns classifier-free guidance *off*. The schnell variant baked guidance into its weights during distillation, so re-applying it would double the cost and hurt quality. The non-distilled FLUX.1-dev, by contrast, wants `guidance_scale` around 3.5 — a Track B subject.
- `generator=...manual_seed(0)` fixes the random noise the process starts from, so the result is reproducible. Change the seed and you get a different valid sample from the same distribution — that is the *diversity* corner in action: same prompt, many images.

Notice what you did *not* have to do: no training, no loss function, no sampler implementation. The pipeline hides the entire stack behind six arguments. The point of the rest of the series is to open that box, so when your own generations come out blurry (steps too low), fried (guidance too high), or out of memory (offload off), you know exactly which line to change.

A quick swap to feel the alternative — the SDXL-Turbo path, also one to four steps:

```python
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

image = pipe(
    "a photo of a corgi wearing sunglasses on a beach at golden hour",
    num_inference_steps=4,
    guidance_scale=0.0,   # Turbo is trained for guidance_scale 0
).images[0]
image.save("corgi_turbo.png")
```

Both produce a sharp 1024px (or 512px for Turbo) photo in a heartbeat. Neither is magic — both are the same denoising idea, distilled. Once you understand *why* four steps suffice (the subject of Track E), you will know which knobs to turn when your own generations come out blurry, over-saturated, or too slow.

One more snippet to make the *modularity* of the stack tangible. Recall that the sampler is a swappable, learning-free box. With a non-distilled model you can change the numerical solver in one line and feel the step↔quality trade-off directly — the same denoiser, a different integrator:

```python
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16",
).to("cuda")

# Swap the default sampler for DPM-Solver++ — a higher-order ODE solver
# that reaches comparable quality in far fewer steps. No retraining.
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, algorithm_type="dpmsolver++"
)

prompt = "a photo of a corgi wearing sunglasses on a beach at golden hour"
for steps in (50, 20, 8):
    img = pipe(prompt, num_inference_steps=steps, guidance_scale=6.0).images[0]
    img.save(f"corgi_sdxl_{steps}steps.png")
```

Run that loop and you will *see* the trilemma: the 50-step image is crisp, the 20-step image is nearly identical (DPM-Solver++ is high-order, so it converges fast), and the 8-step image starts to show artifacts. The whole sampler track (Track B) is about finding the smallest step count that still lands on the manifold — and this five-line swap is the experiment that motivates it.

You can measure the speed side of that trade with nothing more than the standard library, which is the honest way to report any latency number:

```python
import time, torch

def benchmark(pipe, prompt, steps, n=10):
    # Warm-up run (first call compiles kernels and is not representative).
    _ = pipe(prompt, num_inference_steps=steps).images[0]
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n):
        _ = pipe(prompt, num_inference_steps=steps).images[0]
    torch.cuda.synchronize()
    return (time.time() - start) / n  # seconds per image

print(f"{benchmark(pipe, prompt, steps=20):.3f} s/image at 20 steps")
print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
```

Two details in there matter and are routinely gotten wrong: the **warm-up** call (the first run pays one-time compilation and caching costs, so timing it inflates your number) and the `torch.cuda.synchronize()` calls (GPU work is asynchronous, so without synchronizing you would time how fast Python *queued* the work, not how fast the GPU *finished* it). Report seconds-per-image, the GPU name, the step count, and peak VRAM, and your benchmark means something. Skip the warm-up and the sync and it does not.

## A problem-solving narrative: which family would you actually pick?

Theory is cheap; let's reason like an engineer with a real decision. Suppose you are building a product feature and need a generative image model. Walk the trilemma.

**Constraint 1: open-domain, text-conditioned, high quality.** This immediately rules out a plain VAE (too blurry) and a plain GAN (mode collapse on open-domain text-to-image is brutal — nobody has shipped a competitive open-domain text-to-image *pure* GAN). You are choosing between **diffusion/flow** and **autoregressive**.

**Constraint 2: latency budget.** If you need sub-second generation on a single GPU, a distilled diffusion or flow-matching model (FLUX.1-schnell, SDXL-Turbo, SANA) is the path of least resistance in 2026 — the tooling (`diffusers`, ComfyUI), the LoRA ecosystem, the ControlNets, and the distillation methods are all mature. If you need tight *text-and-image* unification (a single model that converses, edits, and generates in one autoregressive stream, like GPT-Image), the autoregressive route is increasingly viable but heavier and slower per image.

**Constraint 3: control and fine-tuning.** Do you need to fine-tune on a brand's style with 20 images (LoRA), enforce a pose or depth map (ControlNet), or edit existing images (inpainting, instruction editing)? The diffusion ecosystem's control tooling is, as of 2026, far richer. That tips most production decisions toward diffusion/flow.

**The decision.** For 90% of "generate or edit an image from a prompt, fast, on a budget" problems in 2026, the answer is a distilled latent flow-matching model (FLUX or SD3 lineage) with LoRA for customization and ControlNet/IP-Adapter for control. The autoregressive route is the one to watch and the right bet if your real requirement is *unified multimodal* interaction. This is exactly the decision tree the [capstone post](/blog/machine-learning/image-generation/building-an-image-generation-stack) formalizes.

Now stress-test that decision, because every choice has a failure mode:

- **What happens at 1 step?** Push FLUX.1-schnell or Turbo to a single step and quality drops — fine for thumbnails, visibly degraded for hero images. The step↔quality Pareto (Track E) tells you where to sit.
- **What happens when guidance is too high?** Crank classifier-free guidance to 15 and images over-saturate, lose diversity, and get "fried" — the guidance trade-off (Track B) made concrete.
- **What happens with three objects and counting?** "Three red cubes and two blue spheres" is where current text-to-image still fails — attribute binding and counting break (Track D, evaluated honestly in Track F). No family has solved compositionality cleanly.
- **What happens when the VAE is the bottleneck?** At very high resolution the VAE decode dominates latency and can introduce artifacts; SANA's 32× autoencoder is a direct answer (Track C/E). The "denoiser is fast but the VAE is slow" surprise bites people who only optimized the transformer.

Naming the failure modes up front is the honest way to teach this. Every later post will pose its problem, reason to a decision, and then stress-test it the same way.

## Case studies: landmark models with real numbers

Let's ground the timeline in specific, citable models so the ideas have weight. These are the milestones the series dissects.

| Model | Year | Params | Key idea | Notable number |
| --- | --- | --- | --- | --- |
| **DDPM** | 2020 | ~35M–550M (CIFAR/LSUN) | Denoising diffusion; noise-prediction loss | 1000 sampling steps; FID 3.17 on CIFAR-10 |
| **Stable Diffusion 1.5** | 2022 | ~860M U-Net (+VAE, +CLIP) | Latent diffusion: diffuse in a VAE latent | ~48× compute cut; runs on a consumer GPU |
| **SDXL** | 2023 | ~2.6B U-Net (+two text encoders) | Larger latent diffusion + refiner | Big quality jump over SD1.5 at 1024px |
| **Stable Diffusion 3** | 2024 | 800M–8B (MM-DiT) | Flow matching on a multimodal transformer | Joint text+image attention; improved typography |
| **FLUX.1** | 2024 | ~12B (rectified-flow transformer) | Flow-matching DiT; schnell variant distilled | schnell samples in ~1–4 steps |
| **SANA** | 2024–2025 | 0.6B–1.6B | 32× deep-compression AE + linear attention | 1024px image in well under 1 s |

A few things to notice. First, parameter counts went *up* (FLUX is ~12B) but so did *efficiency* — FLUX-schnell and SANA generate faster than SD1.5 did, because the gains came from latent compression, better backbones, and distillation, not just from making the net bigger. Second, the recurring move is **compression**: SD compressed pixels 48×, SANA compresses 32× *in the autoencoder alone* and adds linear attention. Compression is how you buy back the speed corner of the trilemma without giving up quality. Third, every one of these is a post in this series; the table is a syllabus.

#### Worked example: the step-count vs latency trade

Here is a concrete Pareto point you can act on, the kind every later post makes precise. Take a latent diffusion model on an A100 80GB. At the original schedule you might run **50 steps** with a DPM-Solver++ sampler and classifier-free guidance, costing two network evaluations per step (one conditional, one unconditional) — so **100 forward passes** and, say, ~2.5 s per 1024px image. Distill it to a four-step Turbo/LCM model that no longer needs guidance (one pass per step) and you are at **4 forward passes** — a 25× reduction in network calls, dropping to roughly **0.1–0.2 s** per image, at a modest, measurable FID cost (Turbo/LCM trade a few FID points for the speed; the exact delta is a published, model-specific number we will report honestly in Track E). The lesson: *sampling cost is steps × passes-per-step, and both factors are attackable.* Guidance doubles passes-per-step; distillation collapses both. Hold this formula and the entire speed track makes sense before you read it.

## How we measure all this (and where the numbers lie)

Every claim in this post and the rest of the series is anchored to a number — FID, CLIP-score, steps, seconds. You cannot reason about the trilemma without knowing what those numbers actually measure and where they mislead, so let's pin them down. (Track F devotes a whole post to evaluating image generation honestly; this is the working summary you need to read the tables.)

**FID — Fréchet Inception Distance.** This is the headline quality-and-diversity metric. You take a few thousand real images and a few thousand generated ones, push both through a pretrained Inception-v3 network, grab a 2048-dimensional feature vector for each image, and fit a Gaussian to each set. FID is the Fréchet distance between those two Gaussians:

$$\mathrm{FID} = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}\!\left(\Sigma_r + \Sigma_g - 2\big(\Sigma_r \Sigma_g\big)^{1/2}\right),$$

where $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$ are the mean and covariance of the real and generated feature distributions. Lower is better; 0 means the feature statistics match perfectly. FID is sensitive to *both* quality (the mean term, roughly "are the images realistic?") and diversity (the covariance term, roughly "do they spread the way real images do?"), which is exactly why it became the standard — it punishes mode collapse *and* blur in one number.

But FID lies in several well-documented ways, and you should distrust headline FIDs that do not state their protocol. It depends heavily on the sample count (FID is biased upward at small sample sizes — always report how many images, conventionally 10k–50k), on the exact reference set (FID on ImageNet is not comparable to FID on COCO), on the Inception model's quirks (it was trained on ImageNet, so it "sees" images through an ImageNet lens and can miss failure modes outside that distribution — text rendering, faces, fingers), and on image preprocessing (resize and JPEG artifacts shift it). Two papers reporting "FID 2.3" on different reference sets are not comparable. This is why the field increasingly reports **FID-DINOv2** (the same recipe but with a DINOv2 feature extractor, which correlates better with human judgment) alongside classic FID.

**CLIP-score — does the image match the prompt?** FID says nothing about whether your "corgi on a beach" actually shows a corgi on a beach. CLIP-score fills that gap: encode the prompt and the generated image with CLIP and take their cosine similarity. Higher means better prompt adherence. It is the natural metric for the *conditional* part of text-to-image, and it is what rises monotonically with guidance scale in the worked example above. Its blind spot: CLIP is bag-of-concepts-ish, so it rewards "the right objects are present" but barely penalizes wrong *relationships* ("a red cube *on top of* a blue sphere" scores almost the same as the swapped version). That is why compositionality benchmarks like GenEval and T2I-CompBench exist.

**Inception Score (IS) — the old guard.** IS predates FID: it rewards images that Inception classifies confidently (sharp, recognizable) *and* whose label distribution is spread out (diverse). Higher is better. It is largely deprecated now because it does not compare against a real reference set at all — you can score well by generating crisp ImageNet-class images even if they look nothing like your target distribution — but you will still see it in older papers, so know what it claims.

**Human preference scores — HPSv2, ImageReward, PickScore.** Because automatic metrics all have blind spots, the frontier increasingly reports models trained to predict *human* preferences between image pairs. These correlate better with "which image would a person pick?" than FID or CLIP-score alone, and they are how modern text-to-image models are actually ranked. They have their own biases (they inherit the aesthetic preferences of the annotators), but they are the closest proxy to the thing we actually care about.

**Speed and cost — steps, seconds, VRAM.** The speed corner of the trilemma is measured in *number of function evaluations* (NFE = steps × passes-per-step), wall-clock *seconds per image* on a **named** GPU (an A100 80GB and an RTX 4090 are very different machines, and a benchmark without the GPU named is meaningless), peak **VRAM** in GB (which decides whether the model even fits), and sometimes \$ cost per image (GPU-hour rate divided by images-per-hour — e.g., at roughly \$2/hr for an A100 and 1000 images/hour you are near \$0.002 per image). When this series quotes a speed number it will name the GPU, the step count, the precision (fp16/bf16), and whether guidance was on, because changing any of those changes the answer by multiples.

The honest-measurement rule for the whole series: **a number without a protocol is a rumor.** When you read "FID 2.3," ask: which reference set, how many samples, what preprocessing? When you read "0.5 s per image," ask: which GPU, how many steps, what precision, batch size one? We will always state these, and you should demand them of any model you evaluate yourself.

## When to reach for each family (and when not to)

A decisive recommendation section, because "it depends" is not an answer.

- **Reach for diffusion / flow matching** when you want the best open-domain quality and diversity with mature control and fine-tuning tooling — i.e., almost always in 2026 for text-to-image and image editing. Distill it if you need speed. This is the default.
- **Reach for autoregressive / token models** when you need *unified multimodal* generation — one model that interleaves text and images, converses, and edits in a single stream (GPT-Image style) — or when you want exact likelihood and language-model-grade scaling. Accept the slower per-image sampling.
- **Reach for a VAE** as a *component*, never as the final generator: it is the latent space under your diffusion model, or the tokenizer (VQ-VAE) under your autoregressive model. Don't ship raw VAE samples; they blur.
- **Reach for a GAN** as a *distillation loss*, not a primary model: the adversarial signal is how you compress a slow diffusion model into a one-step generator (ADD, DMD2). Don't try to train an open-domain text-to-image GAN from scratch; you will fight mode collapse and lose.
- **Don't** use 1000-step DDPM when DPM-Solver hits the same quality in 20–50 steps, or a distilled model in 4. **Don't** push classifier-free guidance above ~7 (it over-saturates and kills diversity). **Don't** fine-tune full weights when a LoRA on a few images suffices. **Don't** optimize only the denoiser and forget the VAE decode and text encoder — at high resolution they become the bottleneck. Each of these is a full post later; the rule is here so you have it now.

A word on the economics, because the trilemma's speed axis is, in production, a cost axis. Suppose you serve a 1024px text-to-image feature. A non-distilled SDXL at 30 steps with guidance is ~60 forward passes; on an A100 at roughly \$2/hr you might push perhaps 600–1200 images per GPU-hour, which is on the order of \$0.002–0.003 per image. Distill to a four-step model and you can serve several times as many images per GPU-hour, dropping the marginal cost proportionally and — just as importantly — cutting *latency* from a couple of seconds to a fraction of one, which is the difference between an interactive product and a batch job. That is why the speed track is not an academic exercise: every step you remove either lets you serve more users on the same hardware or gives each user a faster response. The trilemma's speed corner is where the cloud bill lives, and distillation is the single biggest lever on it. We make these numbers exact, on named GPUs with named precisions, in Track E and the capstone.

There is also a *build-versus-buy* axis the capstone treats in full. For a one-off or low-volume need, a hosted API (the closed frontier models) is cheapest in engineering time even if pricier per image. For high volume, brand-specific style, on-prem/privacy requirements, or heavy control (ControlNet, custom LoRAs), running an open model (FLUX, SD3, SDXL, SANA) on your own GPUs pays off — and gives you the whole control ecosystem this series teaches. The decision hinges on volume, control needs, and whether your data can leave your perimeter. None of that is a model-quality question; it is a systems question, and it is the last mile between "I understand image generation" and "I shipped it."

## Key takeaways

- **Image generation is sampling from $p(x)$, a distribution on an astronomically high-dimensional pixel space.** A 512×512 image is a point in ~786,432 dimensions, and almost every point in that space is noise.
- **The manifold hypothesis is the key constraint.** Real images occupy a thin, curved, low-dimensional manifold (intrinsic dimension in the tens) inside that huge ambient space; the generator's job is to put nearly all its probability on that thin sheet and almost none off it.
- **The generative trilemma — quality × diversity × speed — is the recurring frame.** For most of the field's history you could pick two; diffusion got quality + diversity, then engineered back the speed it sacrificed.
- **There are four families, split by how they model $p(x)$:** VAE (explicit ELBO, blurry, now infrastructure), GAN (implicit, sharp + fast but unstable + collapsing, now a distillation loss), autoregressive (explicit exact likelihood, slow, surging back for multimodal), and diffusion/flow (quality + diversity + stable, the 2022–2025 frontier).
- **Diffusion won because the corner it sacrificed (speed) was fixable** — better samplers and distillation took 1000 steps to 4 — while the corners GANs and VAEs sacrificed (diversity, quality) were not.
- **The latent diffusion stack is the spine:** text encoder → VAE latent → forward noising → denoiser (U-Net/DiT) → ODE/SDE sampler → guidance → VAE decode. Every post zooms into one box.
- **The frontier is two-horse:** flow-matching transformers (SD3, FLUX, SANA) and a resurgent autoregressive/native-multimodal line (GPT-Image, VAR, MAR, HunyuanImage-3.0). Watch them converge.

## The map of the rest of the series

This post is the anchor. Here is the territory it opens, organized into six tracks.

![A tree diagram mapping the series from this intro post outward through foundations, the diffusion engine, and architecture tracks to the build-the-stack capstone](/imgs/blogs/why-generating-images-is-hard-8.png)

- **Track A — Foundations.** The math of image distributions (likelihood vs sampling, FID and what it lies about), VAEs from scratch (the ELBO, the latent space diffusion is built on), GANs and why they lost, autoregressive image models, and normalizing flows (the bridge to flow matching). You are reading the first of these.
- **Track B — The diffusion engine.** [Diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles), the full DDPM derivation, the score-based SDE unification, DDIM and fast deterministic sampling, classifier-free guidance, the noise-schedule/parameterization zoo, and a samplers deep dive. This is the core.
- **Track C — Architecture.** The diffusion U-Net, latent diffusion and Stable Diffusion (the model that opened the floodgates), diffusion transformers (DiT), flow matching and rectified flow, and the modern MM-DiT recipe (SD3, FLUX, SANA).
- **Track D — Conditioning, control, and editing.** Text encoders and prompt conditioning, ControlNet, image editing, personalization (DreamBooth/LoRA), IP-Adapter, and the 2025 wave of conversational instruction editing.
- **Track E — Speed and distillation.** Why diffusion is slow and the four levers to fix it, consistency models and few-step generation, distribution-matching/adversarial distillation, and quantization/caching for consumer GPUs.
- **Track F — Evaluation, frontier, and practice.** Evaluating image generation honestly, the autoregressive-vs-diffusion 2026 showdown, safety/watermarking/provenance, and the capstone: [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack) end to end.

If you have ever trained a classifier but never a generative model, you now have the whole map: the problem (sample from a thin manifold in a huge space), the frame (the quality × diversity × speed trilemma), the four bets (VAE, GAN, autoregressive, diffusion/flow), the winner and why (diffusion, because its weakness was fixable), and the stack every modern model instantiates. For background on the transformer that powers DiT, MM-DiT, and autoregressive image models, see the large-language-model track's introduction to attention; the rest of this series assumes only that you know what a transformer block does. Next, we make the trilemma's first axis precise: what $p(x)$ really is, and what FID, Inception Score, and CLIP-score actually measure — and where they lie.

## Further reading

- Goodfellow et al., "Generative Adversarial Networks," NeurIPS 2014 — the GAN minimax game.
- Kingma & Welling, "Auto-Encoding Variational Bayes," ICLR 2014 — the VAE and the reparameterization trick.
- van den Oord et al., "Pixel Recurrent Neural Networks," ICML 2016, and "Neural Discrete Representation Learning" (VQ-VAE), NeurIPS 2017 — autoregressive and discrete-latent image models.
- Ho, Jain & Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS 2020 — DDPM, the spark.
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations," ICLR 2021 — the SDE unification.
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022 — latent diffusion / Stable Diffusion.
- Peebles & Xie, "Scalable Diffusion Models with Transformers," ICCV 2023 — the DiT backbone.
- Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023 — the flow-matching framework SD3/FLUX use.
- Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis," ICML 2024 — Stable Diffusion 3 / MM-DiT.
- Pope et al., "The Intrinsic Dimension of Images and Its Impact on Learning," ICLR 2021 — evidence for the manifold hypothesis.
- Xiao, Kreis & Vahdat, "Tackling the Generative Learning Trilemma with Denoising Diffusion GANs," ICLR 2022 — the trilemma framing.
- The 🤗 `diffusers` documentation — the practical toolchain used throughout this series.
- Within this series: [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles), [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch), [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions), and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
