---
title: "Diffusion From First Principles: How Adding Noise Teaches a Network to Paint"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Build the diffusion mental model from zero: the forward process that destroys an image into noise, the reverse process that learns to rebuild it, why predicting noise is the same as pointing toward real images, and a runnable DDPM you can train on MNIST tonight."
tags:
  [
    "image-generation",
    "diffusion-models",
    "ddpm",
    "denoising",
    "score-matching",
    "generative-ai",
    "deep-learning",
    "pytorch",
    "stable-diffusion",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/diffusion-from-first-principles-1.png"
---

Here is a recipe that should not work. Take a photograph. Add a little Gaussian noise to it. Add a little more. Keep going for a thousand small steps, each one sprinkling in a bit more static, until the photo is gone and you are left with a screen of pure television snow — every pixel an independent random number, no trace of the original cat or face or beach. You have *destroyed* information on purpose, completely and irreversibly, at least from the point of view of any single step. Now train a neural network to undo *one* of those steps: given a slightly-too-noisy image, predict the small dab of noise that was added to make it that way. That is the entire training task. It is a regression problem you could explain to an intern in two minutes.

And yet, once that network is trained, you can throw away every photograph, start from a fresh screen of pure random noise, and ask the network to undo the noise step by step — a thousand tiny corrections — and out the other end comes a brand-new photograph that has never existed. A cat that was never photographed. A face that belongs to no one. This is **diffusion**, the engine inside Stable Diffusion, FLUX, Midjourney, DALL-E 3, and Sora. The first time you implement it and watch a digit emerge from noise on your own machine, it genuinely feels like cheating. The point of this post is to make it stop feeling like cheating and start feeling *inevitable* — to build the mental model so completely from the ground up that, by the end, the question is not "how could this possibly work?" but "what else could it have been?"

![A directed graph showing a fixed forward process that turns a clean image into pure noise on one side and a learned reverse denoiser that rebuilds an image from noise on the other](/imgs/blogs/diffusion-from-first-principles-1.png)

This is a *foundation* post. The whole rest of the diffusion track — the full variational derivation, the score-SDE unification, fast samplers, guidance — formalizes and accelerates the intuition we build here. So we will keep the derivations light but never wrong: every equation you see is the real thing, just explained before it is manipulated. The heavy algebra (the variational bound that *proves* the noise-prediction loss is the right objective) is the very next post, [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm), which we forward-link at each spot it picks up. By the time you finish *this* post you will understand the two processes, the training loop, and the sampling loop well enough to read and modify a real DDPM — and there is a runnable one below that trains on MNIST in a few minutes on a single GPU.

Let me anchor this in the spine of the whole series. The hard problem, established in [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard), is that natural images occupy a vanishingly thin, strangely-shaped manifold inside an astronomically large pixel space, and we only ever see *samples* from the distribution $p(x)$, never the distribution itself. Every generative family is a different bet on how to learn to sample from $p(x)$. Diffusion's bet — the one that won the 2022–2025 era — is the strangest and the most beautiful: *don't try to model $p(x)$ directly at all. Instead, learn to reverse a process that gradually turns $p(x)$ into pure noise, because reversing noise, step by tiny step, turns out to be a stable supervised-learning problem with a labeled target at every single noise level.*

## The two processes: a fixed road out and a learned road back

Everything in diffusion is two processes, and if you hold these two clearly in your head the rest follows.

The **forward process** (also called the *diffusion* or *noising* process) takes a real image $x_0$ and corrupts it into noise over $T$ steps, producing a sequence $x_0, x_1, x_2, \dots, x_T$. Each step adds a small amount of Gaussian noise. By the final step $x_T$ is essentially indistinguishable from a draw of pure Gaussian noise, $\mathcal{N}(0, I)$ — all the structure of the original image has been washed away. Crucially, **the forward process is fixed**. It has no learnable parameters. It is just a recipe — "blend the current image a little toward zero and add a little Gaussian noise" — that we apply mechanically. We *choose* it; we never train it.

The **reverse process** (the *denoising* or *generative* process) goes the other way: it starts from pure noise $x_T \sim \mathcal{N}(0, I)$ and removes noise step by step — $x_T \to x_{T-1} \to \dots \to x_1 \to x_0$ — until it lands back on a clean image. Here is the whole trick in one sentence: **the reverse process is the only part we learn.** A single neural network is trained to undo one forward step, and we apply it $T$ times to walk all the way back from noise to image.

Why split it this way? Because the forward process, being fixed and known, hands us *exactly the training signal we need* to learn the reverse. At every step of the forward process we know precisely what noise we added — we added it ourselves. So we can show the network the noised image and ask it to predict the noise, and we can grade its answer exactly, because we have the answer key. We *manufactured* a supervised learning problem out of an unsupervised one (modeling $p(x)$) by deliberately destroying data in a controlled way. That is the conceptual pivot the entire field rests on, and it is worth sitting with: **we destroy information to create a label.**

Contrast this with the families it beat. A GAN (covered in [GANs and why they lost](/blog/machine-learning/image-generation/generative-adversarial-networks-and-why-they-lost)) pits a generator against a discriminator in a minimax game with no stable target — the two networks chase each other, and training is famously a knife-edge. A VAE ([VAEs from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch)) optimizes a variational bound but tends to blur because its Gaussian decoder averages over the posterior. Diffusion sidesteps both pathologies: there is no adversary, just a clean regression target, and there is no single-shot decoder averaging away detail — each of the $T$ steps only has to make a tiny correction, and tiny corrections are easy to get sharp.

Why does breaking the generation into many tiny steps buy us so much? This is the single most important *structural* reason diffusion works, so let me dwell on it. The hard part of generative modeling is that $p(x_0)$ is wildly multimodal — there are cats and faces and landscapes and diagrams, all in different regions of the manifold, and a one-shot generator has to capture that entire complicated shape in a single forward pass. That is what a GAN's generator and a VAE's decoder are each asked to do, and it is brutally hard. Diffusion's insight is that you do not have to model that complicated distribution all at once. The forward process gives you a *ladder* of distributions: $p(x_0)$ at the bottom (complicated), then slightly-noised versions, then more-noised, all the way up to $\mathcal{N}(0, I)$ at the top (trivially simple). Adjacent rungs differ only slightly, so the map from one rung to the next is *simple even though the endpoints are not.* The network never has to learn the complicated full distribution in one shot; it only ever learns "given this rung, nudge toward the rung just below," a thousand times. Complexity is amortized across the ladder. That decomposition — turning one impossibly-hard problem into a thousand easy ones — is the conceptual engine, and it is why the framework is so robust.

### The forward process, precisely

Let me write the forward process down, because it is simpler than the prose. We define a small **noise schedule**, a sequence of numbers $\beta_1, \beta_2, \dots, \beta_T$ with each $\beta_t$ a little bigger than the last, typically running from about $10^{-4}$ to $0.02$ over $T = 1000$ steps. At step $t$, the forward process takes $x_{t-1}$ and produces $x_t$ by:

$$
x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, I).
$$

Read this literally. We scale the current image down by $\sqrt{1-\beta_t}$ (just under 1, so we shrink it slightly toward zero) and we add fresh Gaussian noise scaled by $\sqrt{\beta_t}$ (a small amount). The two coefficients are chosen so that the *variance* of $x_t$ stays at 1 if $x_0$ was standardized — the signal slowly leaks out and noise slowly leaks in, but the total "energy" is conserved. In probability notation, this single step is a Gaussian:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\right).
$$

Because each step depends only on the previous one, the whole forward sequence is a **Markov chain**: the future depends on the present, not the past. That is figure 1's left half — a clean chain $x_0 \to x_1 \to \dots \to x_T$ with no learnable parameters, just the fixed schedule.

Now, applying a thousand steps one at a time to generate a training example would be painfully slow. Here is the first piece of magic that makes diffusion *practical*: because each step is a linear Gaussian, you can compose all $t$ steps in closed form and jump straight from $x_0$ to $x_t$ in a single draw. Define $\alpha_t = 1 - \beta_t$ and the **cumulative product** $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Then:

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar{\alpha}_t}\, x_0,\ (1 - \bar{\alpha}_t) I\right).
$$

This is **the forward marginal**, and it is the single most important equation in the post. In sampling form it says:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).
$$

Look at what it gives you. To corrupt $x_0$ to any noise level $t$ you want, you draw *one* Gaussian noise vector $\epsilon$, scale the clean image by $\sqrt{\bar{\alpha}_t}$, scale the noise by $\sqrt{1 - \bar{\alpha}_t}$, and add them. No loop. One shot. The coefficient $\sqrt{\bar{\alpha}_t}$ is how much of the original image survives; $\sqrt{1 - \bar{\alpha}_t}$ is how much noise has accumulated. At $t = 0$, $\bar{\alpha}_0 = 1$, so $x_0 = x_0$, untouched. As $t$ grows, $\bar{\alpha}_t \to 0$, so $x_t \to \epsilon$, pure noise. The whole forward process is a smooth dial from "your image" to "static," and the dial position is set by a single number $\bar{\alpha}_t$ you precompute once. (The derivation that two composed Gaussians give this closed form is short and lives in [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm); for now, take it as the gift it is.)

![A timeline showing the cumulative product alpha-bar shrinking from one toward zero across timesteps as the image goes from clean to pure noise](/imgs/blogs/diffusion-from-first-principles-2.png)

Figure 2 plots $\bar{\alpha}_t$ across the schedule. Early on it barely moves — the first fifty steps leave the image almost intact, $\bar{\alpha}_{50} \approx 0.97$. Then it falls off a cliff in the middle, and by $t \approx 1000$ it has collapsed to essentially zero. This shape matters: most of the *interesting* denoising work — where the network has to hallucinate real structure from near-noise — happens in the high-$t$ regime where $\bar{\alpha}_t$ is small. We will revisit this when we talk about loss weighting in [noise schedules and the parameterization zoo](/blog/machine-learning/image-generation/noise-schedules-and-the-parameterization-zoo); for now just notice that the schedule is a *design choice* that decides how much time the model spends at each difficulty level.

#### Worked example: noising a single MNIST pixel

Let me make the marginal concrete with arithmetic you can check by hand. Take one pixel of an MNIST digit, normalized so the white stroke has value $x_0 = 1.0$. Suppose at timestep $t = 250$ the schedule gives $\bar{\alpha}_{250} = 0.60$. Then the noised pixel is

$$
x_{250} = \sqrt{0.60}\cdot 1.0 + \sqrt{0.40}\cdot \epsilon = 0.775 + 0.632\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0,1).
$$

So the pixel's value is now a Gaussian centered at $0.775$ with standard deviation $0.632$. The original signal ($0.775$) is still the *mean*, so a denoiser that knows $\bar{\alpha}_{250}$ and sees a batch of such pixels can still recover the underlying stroke — the signal-to-noise ratio is $\sqrt{\bar{\alpha}_t / (1 - \bar{\alpha}_t)} = \sqrt{0.6/0.4} \approx 1.22$, comfortably above 1. Now push to $t = 900$ with $\bar{\alpha}_{900} = 0.02$: the pixel becomes $0.141 + 0.99\,\epsilon$, an SNR of about $0.14$. The stroke is almost entirely buried; recovering it requires the network to lean on everything it learned about what digits look like, not just local averaging. This is why high-$t$ denoising is the hard, *generative* part and low-$t$ denoising is the easy, *refining* part — a theme that recurs everywhere in the series.

## Why destroy data on purpose? Because it creates a target at every level

The deliberate-destruction move deserves its own beat, because it is the part newcomers find most counterintuitive. We have a finite pile of real images and we want a model that can produce new ones. The obvious instinct is to model $p(x)$ directly — fit a density, then sample it. But in a million-dimensional pixel space, fitting a normalized density is brutally hard: you cannot even write down the normalizing constant, and the manifold is so thin that almost every direction in space points *off* it into garbage. Direct density modeling is exactly the wall that VAEs, autoregressive models, and normalizing flows each climb a different way (see [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions)), and each pays a real cost.

Diffusion refuses the direct approach. Instead it observes: *I may not know how to model $p(x_0)$, but I know exactly how to corrupt it, and corruption is reversible if I learn the reverse one small step at a time.* The forward process bridges the impossibly complex distribution $p(x_0)$ to the trivially simple distribution $\mathcal{N}(0, I)$ through a chain of $T$ intermediate distributions, each only slightly noisier than the last. And here is the key structural fact: **adjacent distributions in the chain are close enough that a single reverse step is approximately Gaussian and therefore learnable by a network predicting a mean.** If we tried to jump from noise to image in one step, the reverse would have to model the full, multimodal complexity of $p(x_0)$ — exactly the hard problem we were avoiding. By breaking the journey into a thousand small steps, each individual step is a gentle, nearly-Gaussian nudge, and the network only ever has to learn gentle nudges.

The supervised-learning payoff is enormous. For *any* real image $x_0$ and *any* timestep $t$, we can produce the training pair $(x_t, \epsilon)$ in one line: draw $\epsilon$, form $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$, and the label is the $\epsilon$ we just drew. Every image in the dataset, at every one of $T$ noise levels, is a labeled example. A 60,000-image MNIST set with $T = 1000$ is, in effect, 60 million labeled denoising examples — and we can resample the noise each epoch for effectively infinite data. There is no labeling team, no adversary, no fragile equilibrium. The label *is* the corruption we applied.

![A before and after comparison contrasting a lightly noised image at a small timestep with a heavily noised image at a large timestep, showing how each noise level yields its own labeled training target](/imgs/blogs/diffusion-from-first-principles-3.png)

Figure 3 contrasts the two ends of the difficulty spectrum that the same network must handle. At low $t$ (left) the image is barely corrupted, the SNR is high, and the network only has to find and remove a faint dusting of noise — easy, almost a sharpening operation. At high $t$ (right) the image is mostly gone, the SNR is near zero, and to "remove the noise" the network has to *commit to what the image should have been* — which digit, which stroke, which orientation. The astonishing thing is that one network, conditioned on the timestep $t$, learns both regimes at once: it behaves like a denoiser at low $t$ and like a generator at high $t$. Conditioning on $t$ is what lets it switch behaviors, which is why the timestep is always an input to the network.

This is also why diffusion **covers modes** so well, in sharp contrast to GANs. A GAN's generator can quietly drop entire categories — "I'll just make convincing 1s and 7s and skip the hard 8s" — and still fool a weak discriminator; this is *mode collapse*, the chronic GAN disease. Diffusion cannot do this, because the training loss forces the network to denoise *every* image in the dataset at *every* noise level. There is no way to ignore the 8s: every noised 8 is a training example with a concrete target, and dropping it just raises the loss. The objective is a simple per-example MSE, summed over the whole data distribution, so the model is pressured to allocate probability mass everywhere the data is. We will make this precise — the loss is a bound on the data log-likelihood — but the intuition is already exact: **you cannot collapse modes when your loss grades you on reconstructing every one of them.**

## The denoising objective: predict the noise, and why that is the right thing

We now know the network's job at a high level — undo one forward step — but what exactly should it *output*, and what should the loss be? This is where diffusion's design becomes startlingly clean.

The reverse step we want to learn is $p_\theta(x_{t-1} \mid x_t)$: given the noisier $x_t$, produce a distribution over the slightly-cleaner $x_{t-1}$. Because adjacent forward distributions are close, this reverse conditional is well-approximated by a Gaussian, so the network only needs to predict its **mean** (the variance can be fixed to a schedule-derived constant — DDPM uses exactly this simplification). So really the network predicts "where should $x_{t-1}$ be centered, given $x_t$?"

Now the elegant part. There are several equivalent things the network could predict to pin down that mean — the clean image $x_0$, the reverse mean directly, or the noise $\epsilon$ — and they are algebraically interchangeable through the forward marginal. Ho et al.'s 2020 DDPM paper found, both theoretically and empirically, that predicting the **noise** $\epsilon$ gives the best-behaved objective. Why noise? Because of the marginal we already wrote down. We have $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$, so if the network can predict $\epsilon$ from $x_t$ and $t$, it can recover the clean image by rearranging:

$$
\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\,\hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}.
$$

Predicting $\epsilon$ is therefore *equivalent* to predicting $x_0$, but it has nicer numerical properties: $\epsilon$ is always unit-variance Gaussian regardless of $t$, so the target has a consistent scale across all noise levels, which keeps the loss well-conditioned. (At high $t$, predicting $x_0$ directly means predicting a tiny, heavily-attenuated signal — numerically nasty; predicting $\epsilon$ stays well-scaled.)

So the network is $\epsilon_\theta(x_t, t)$: it takes a noised image and a timestep, and outputs a prediction of the noise that was added. And the loss is the most boring thing imaginable — mean squared error between the true noise and the predicted noise:

$$
\mathcal{L}_\text{simple} = \mathbb{E}_{x_0,\, t,\, \epsilon}\left[\big\lVert \epsilon - \epsilon_\theta(x_t, t)\big\rVert^2\right], \qquad x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon.
$$

Let me unpack every term, because this one equation *is* diffusion training:

- $\mathbb{E}_{x_0, t, \epsilon}$ — we average over three random draws: a real image $x_0$ from the dataset, a timestep $t$ uniformly from $\{1, \dots, T\}$, and a noise vector $\epsilon \sim \mathcal{N}(0, I)$. Each draw is one training example.
- $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$ — we build the noised image in one shot from the marginal. This is the input to the network.
- $\epsilon_\theta(x_t, t)$ — the network's predicted noise, the only learned object.
- $\lVert \epsilon - \epsilon_\theta(x_t, t)\rVert^2$ — squared error between the noise we actually added and the noise the network guessed. Minimizing it makes the network a good noise predictor.

That is the whole objective. No KL terms to tune, no adversary, no reparameterization gymnastics in the loss itself. Ho et al. famously showed that this *simplified* loss — just the MSE, with the theoretically-motivated per-timestep weighting dropped — works *better* in practice than the full weighted variational bound, which is one of those rare cases where the simpler thing is also the better thing. The reason the simple MSE is even a valid objective at all is that it is, term by term, a (reweighted) lower bound on the data log-likelihood; that derivation — turning $\log p_\theta(x_0)$ into a sum of per-step denoising losses via the ELBO — is the entire subject of [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm). For this post, the takeaway is that the loss you would have invented as the obvious dumb thing to try is, provably, the right thing.

![A vertical stack of the training step from sampling an image, timestep, and noise, through forming the noisy image and predicting the noise, to the mean squared error loss](/imgs/blogs/diffusion-from-first-principles-4.png)

Figure 4 lays out the training step as a stack you can read top to bottom — and it is worth noting how *short* it is. Sample, noise, predict, MSE. Compare that to a GAN training step (alternate generator and discriminator updates, balance their learning rates, watch for collapse, tune the gradient penalty) and you start to feel why practitioners breathed a sigh of relief when diffusion arrived. The stability is not an accident; it is a direct consequence of having a fixed regression target instead of a moving adversarial one.

### Why predicting noise points toward real images: the score connection

Here is the question that, once it clicks, makes diffusion feel inevitable rather than magical. *Why* does predicting noise let you generate images? Removing noise from a noisy image gives you a cleaner image, sure — but how does that turn pure static into a coherent, novel photograph? The answer is the deepest idea in the whole post, and I will state it intuitively now and let [score-based models and the SDE view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view) make it fully rigorous later.

There is a quantity in statistics called the **score** of a distribution: the gradient of the log-density, $\nabla_x \log p(x)$. It is a vector field over the space, and at any point it **points in the direction of steepest increase in probability density** — uphill toward where the data is. If you stand at a random point in pixel space (off the manifold, in the static) and you repeatedly take small steps in the direction of the score, you climb toward higher-probability regions — toward the image manifold. The score is, quite literally, a compass that always points "more like a real image."

Now the punchline. For the diffused distribution at noise level $t$, the score has a closed form in terms of the noise:

$$
\nabla_{x_t} \log q(x_t) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}.
$$

The score is, up to a known scaling constant, *the negative of the noise that was added.* So when the network predicts $\epsilon_\theta(x_t, t)$, it is — up to that constant — predicting the score. **Noise prediction and score estimation are the same task wearing different clothes.** This is why removing predicted noise generates images: subtracting $\hat{\epsilon}$ is, up to scale, stepping along the score, which is stepping uphill toward higher density, which is stepping toward the manifold of real images. A thousand small uphill steps, starting from random noise, walk you from the static all the way onto the manifold — and where you land depends on the random starting point and the small noise injected along the way, which is why you get a *different* novel image every time instead of always the same one.

![A directed graph linking the predicted noise to the score gradient to a denoising step that moves a sample toward higher density and a more image like result](/imgs/blogs/diffusion-from-first-principles-5.png)

Figure 5 traces this equivalence as a chain: the noisy sample sits off the manifold; the network predicts the noise; that prediction equals the score up to a constant; the score points uphill; stepping along it moves the sample toward higher $p(x)$, i.e. toward looking like a real image. Hold onto this picture, because it is the bridge between the practical recipe (predict noise, MSE) and the theory (the reverse-time SDE that the score parameterizes). When people say diffusion models are "score-based generative models," *this* is what they mean — the two phrasings describe the same network doing the same thing. Song & Ermon's score-matching line of work and Ho et al.'s DDPM line of work were independently invented and then discovered to be the same idea, which is one of the great unifications in modern ML and the subject of an entire later post.

### Where the loss comes from: a sketch of the variational bound

I have been asserting that the simple MSE is "provably the right thing." Let me sketch *why*, at the level of intuition this foundation post calls for, and leave the full algebra to [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm). The honest goal of generative modeling is to maximize the likelihood the model assigns to real data, $\log p_\theta(x_0)$ — make the real images probable under the model. But $p_\theta(x_0)$ is an intractable integral over all the latent noised states $x_1, \dots, x_T$, so we cannot maximize it directly. The standard move, the same one that powers VAEs, is to maximize a tractable *lower bound* instead — the **evidence lower bound** (ELBO). If you push the bound up, you push the true likelihood up with it.

For diffusion, writing out that bound and simplifying produces something remarkable: $\log p_\theta(x_0)$ is bounded below by a sum of terms, *one per timestep*, and each term turns out to be a **KL divergence between the true reverse step and the model's reverse step**. Because both of those are Gaussians (the forward process is constructed so the true reverse conditional, given $x_0$, is Gaussian with a closed-form mean), the KL between them reduces to a *squared difference of their means*. And when you express those means in terms of noise — using the same forward-marginal substitution we keep leaning on — the squared difference of means becomes a squared difference of *noises*: $\lVert \epsilon - \epsilon_\theta(x_t, t)\rVert^2$, weighted by a per-timestep constant. So the whole variational bound collapses, term by term, into a weighted sum of the exact MSE we coded.

The final simplification — Ho et al.'s "simple" loss — is to *drop the per-timestep weights* (set them all to 1). Strictly, this is no longer the tightest bound; it up-weights the hard high-noise timesteps relative to the bound's prescription. But empirically it trains better and produces higher-quality samples, because it stops the easy low-noise steps (which contribute little to perceptual quality) from dominating the gradient. This is the rare and delightful case where the theoretically-motivated objective and a slightly-cruder practical objective disagree, and the practical one wins. The takeaway for your mental model: $\mathcal{L}_\text{simple}$ is not an ad-hoc choice that happens to work — it is a *reweighting* of a genuine likelihood bound, which is exactly why it covers modes (it is grading log-likelihood across the whole data distribution) and why it is stable (it is a sum of well-behaved Gaussian KLs, not an adversarial game). Every claim I made intuitively about the loss has this derivation underneath it; the next post makes it line-by-line rigorous.

## The network: a time-conditioned U-Net (or, for toys, an MLP)

We have said "a network predicts the noise" without saying what the network *is*. For the math, it does not matter — $\epsilon_\theta$ is just any function with enough capacity. For practice, the architecture matters a great deal, and the workhorse for pixel-space diffusion is the **U-Net**, a convolutional network with a symmetric encoder-decoder shape and skip connections. (The architecture gets its own deep dive in [the diffusion U-Net](/blog/machine-learning/image-generation/the-diffusion-unet); here we just need enough to write working code.)

The U-Net's shape is dictated by the task. The input is a noised image and the output is a noise map of *exactly the same height and width* — we predict one noise value per pixel. So the network must map an $H \times W$ image to an $H \times W$ output. The U-Net does this by first **downsampling** (a series of convolutional blocks that halve resolution while growing channel count, building up a compressed, semantic representation in a low-resolution bottleneck), then **upsampling** back to full resolution. The genius detail is the **skip connections**: at each resolution, the downsampling path's feature maps are concatenated into the matching upsampling block. Without skips, all the fine spatial detail would have to squeeze through the low-resolution bottleneck and would be lost; the skips give the upsampling path a direct line to high-frequency information, which is exactly what you need to predict a *pixel-precise* noise map.

![A directed graph of a U-Net denoiser with a downsampling path, an attention bottleneck, an upsampling path, and skip connections preserving detail across the bottleneck](/imgs/blogs/diffusion-from-first-principles-6.png)

Figure 6 shows the U-Net's branch-and-merge topology: input flows down through the encoder, through an attention-equipped bottleneck, back up through the decoder, while skip connections shortcut across the U. Two more ingredients make it a *diffusion* U-Net rather than a generic one. First, **time conditioning**: the timestep $t$ is embedded (typically a sinusoidal embedding passed through a small MLP) and injected into every block, usually by modulating the block's normalization — this is how one network behaves differently at different noise levels, the switch we discussed earlier. Second, **self-attention** in the lower-resolution blocks, which lets distant parts of the image coordinate (so the model can keep a face symmetric or a horizon level). For a real model on 512×512 images this U-Net might be 860M parameters (SD 1.5) or 2.6B (SDXL's U-Net); for the MNIST toy below, a tiny U-Net of a few hundred thousand parameters is plenty, and you can even use a plain MLP for 2D point data.

It is worth saying clearly: **the architecture is not the idea.** Diffusion is the forward/reverse framework and the noise-prediction objective; the U-Net is just today's best $\epsilon_\theta$ for pixels. Swap it for a transformer and you get a Diffusion Transformer ([DiT](/blog/machine-learning/image-generation/diffusion-transformers-dit)); run it in a compressed latent space instead of pixels and you get Latent Diffusion / Stable Diffusion. The framework is invariant to the choice of $\epsilon_\theta$, which is precisely why the field has been able to swap architectures (U-Net → DiT → MM-DiT) without changing the underlying diffusion math at all.

## The training loop and the sampling loop, in code

Enough words. Here is a complete, runnable DDPM in PyTorch — the noise schedule, the closed-form `q_sample`, a compact U-Net $\epsilon$-predictor, the training step, and an ancestral sampling loop. It trains on MNIST and generates digits. I have kept it minimal but real: every line corresponds to an equation above, and you can paste it into a file and run it. The point is not production polish; it is that the whole idea fits on a couple of screens.

First, the **noise schedule and the forward `q_sample`**. This is the marginal $q(x_t \mid x_0)$ turned into three lines of tensor code:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 1000
betas = torch.linspace(1e-4, 0.02, T)  # linear schedule, Ho et al. 2020: 1e-4 -> 0.02
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)          # cumulative product alpha-bar_t

def q_sample(x0, t, noise):
    # Forward marginal: x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) noise.
    ab = alpha_bars[t].view(-1, 1, 1, 1)            # gather per-sample, broadcast over C,H,W
    return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
```

That `q_sample` is the entire forward process. Given a batch of clean images `x0`, a batch of timesteps `t`, and a batch of noise vectors, it returns the noised images in one shot. No loop over steps — the cumulative product `alpha_bars` already composed all of them.

Next, a **tiny time-conditioned U-Net**. I keep it small and readable: two downsampling stages, a bottleneck, two upsampling stages with skip concatenation, and a sinusoidal timestep embedding injected into each block. On a real project you would reach for `diffusers`' `UNet2DModel`, but seeing the wiring explicitly once is worth more than any number of wrapper libraries.

```python
import math

def timestep_embedding(t, dim):
    # Standard sinusoidal embedding of the integer timestep.
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class Block(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(8, cout)
        self.temb = nn.Linear(tdim, cout)           # time conditioning per block
    def forward(self, x, temb):
        h = self.norm(self.conv(x))
        h = h + self.temb(temb)[:, :, None, None]   # inject timestep, broadcast over H,W
        return F.silu(h)

class TinyUNet(nn.Module):
    def __init__(self, tdim=128):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.d1 = Block(1, 64, tdim)
        self.d2 = Block(64, 128, tdim)
        self.mid = Block(128, 128, tdim)
        self.u2 = Block(128 + 128, 64, tdim)        # +128 from the d2 skip connection
        self.u1 = Block(64 + 64, 64, tdim)          # +64  from the d1 skip connection
        self.out = nn.Conv2d(64, 1, 1)
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
    def forward(self, x, t):
        temb = self.tmlp(timestep_embedding(t, self.tdim))
        h1 = self.d1(x, temb)                       # 28x28
        h2 = self.d2(self.pool(h1), temb)           # 14x14
        m = self.mid(self.pool(h2), temb)           # 7x7
        u2 = self.u2(torch.cat([self.up(m), h2], 1), temb)   # skip from h2
        u1 = self.u1(torch.cat([self.up(u2), h1], 1), temb)  # skip from h1
        return self.out(u1)                         # predicted noise, 1x28x28
```

The `torch.cat([..., h2], 1)` and `torch.cat([..., h1], 1)` lines *are* the skip connections from figure 6 — the encoder features `h1`, `h2` are spliced back into the decoder so detail survives the bottleneck. Now the **training step**, which is figure 4 turned into code. Notice how short it is:

```python
def training_step(model, x0, optimizer):
    optimizer.zero_grad()
    b = x0.shape[0]
    t = torch.randint(0, T, (b,), device=x0.device)        # sample t ~ U(0, T)
    noise = torch.randn_like(x0)                            # sample epsilon ~ N(0, I)
    x_t = q_sample(x0, t, noise)                            # build the noisy image
    pred = model(x_t, t)                                    # predict the noise
    loss = F.mse_loss(pred, noise)                          # L_simple = || eps - eps_theta ||^2
    loss.backward()
    optimizer.step()
    return loss.item()
```

Five meaningful lines: sample `t`, sample `noise`, build `x_t`, predict, MSE. That is $\mathcal{L}_\text{simple}$ exactly. There is no second network, no balancing, no special tricks — which is *why diffusion is stable to train*. The gradient always points at a fixed, well-scaled target. The training driver is the usual boilerplate:

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
tfm = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize([0.5], [0.5])])   # map pixels to [-1, 1]
loader = DataLoader(datasets.MNIST(".", download=True, transform=tfm),
                    batch_size=128, shuffle=True)

model = TinyUNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-4)
alpha_bars = alpha_bars.to(device)

for epoch in range(20):
    for x0, _ in loader:
        loss = training_step(model, x0.to(device), opt)
    print(f"epoch {epoch}  loss {loss:.4f}")
```

After a handful of epochs the loss settles and the model is a competent MNIST noise predictor. A few practical notes that this minimal loop omits but a real training run needs. First, **the loss does not go to zero and should not** — even a perfect denoiser cannot recover the exact noise vector at high $t$, because the signal is genuinely gone, so the loss plateaus at a positive value; watch for it *stabilizing*, not vanishing. Second, **EMA (exponential moving average) of the weights** is nearly always used at sampling time: you keep a slowly-updated running average of the model parameters and sample from *those*, because the EMA weights are far smoother and produce noticeably better images than the raw training weights — this single trick is worth a meaningful FID improvement and is essentially free. Third, **GroupNorm and SiLU are not arbitrary**: diffusion U-Nets are sensitive to normalization choice (BatchNorm interacts badly with the per-timestep conditioning and the wide range of activation scales across noise levels), which is why the field standardized on GroupNorm. Fourth, the timestep embedding *must* reach every block — a common beginner bug is embedding $t$ once at the input and letting it wash out, which leaves the network unable to tell low-noise from high-noise inputs and produces uniformly mediocre samples. These are the unglamorous details that separate a loop that technically runs from one that actually generates clean digits, and they scale directly up to the tricks real models use.

The last piece is the **sampling loop** — the reverse process — which is where noise becomes a digit. We start from pure noise and apply the learned reverse step $T$ times. The reverse step uses the predicted noise to compute the mean of $x_{t-1}$ and adds a small amount of fresh noise (except at the final step), which is the **ancestral sampler** from the DDPM paper:

```python
@torch.no_grad()
def sample(model, n=16, size=(1, 28, 28)):
    x = torch.randn(n, *size, device=device)               # start from pure noise x_T
    for t in reversed(range(T)):                            # walk T -> T-1 -> ... -> 0
        ts = torch.full((n,), t, device=device, dtype=torch.long)
        eps = model(x, ts)                                  # predicted noise = (neg) score
        ab = alpha_bars[t]
        a = alphas[t]
        # Mean of the reverse step, from x_t and the predicted noise:
        mean = (x - (1 - a) / torch.sqrt(1 - ab) * eps) / torch.sqrt(a)
        if t > 0:
            noise = torch.randn_like(x)                     # inject stochasticity
            sigma = torch.sqrt(betas[t])
            x = mean + sigma * noise
        else:
            x = mean                                        # last step: no noise, return clean x_0
    return x.clamp(-1, 1)
```

Walk through it once against the equations. At each step we predict the noise `eps`; we use it to form the reverse mean (the formula is the rearranged forward marginal plus the DDPM posterior coefficients); for every step except the last we add Gaussian noise scaled by $\sigma_t = \sqrt{\beta_t}$ to keep the process stochastic; on the final step we return the mean directly as our clean image. Run `sample(model)` and you get a grid of MNIST digits hallucinated from noise. The same code, with a bigger U-Net, a latent VAE, and a text-conditioning path, *is* Stable Diffusion — the core loop does not change; everything else is scaling and conditioning.

#### Worked example: tracing one reverse step

Let me trace a single reverse step with numbers so the mean formula stops being opaque. Suppose we are at $t = 500$ with $\beta_{500} = 0.01$, so $\alpha_{500} = 0.99$, and say $\bar{\alpha}_{500} = 0.30$. We have a current sample $x_{500}$ and the network predicts noise $\hat{\epsilon}$. The reverse mean is

$$
\mu = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\,\hat{\epsilon}\right) = \frac{1}{\sqrt{0.99}}\left(x_{500} - \frac{0.01}{\sqrt{0.70}}\,\hat{\epsilon}\right) = 1.005\,\big(x_{500} - 0.0120\,\hat{\epsilon}\big).
$$

The correction term $0.0120\,\hat{\epsilon}$ is *tiny* — each reverse step removes only about 1% of the noise's worth of signal. That is the whole point: a single step is a gentle nudge, almost a no-op, which is exactly why it is learnable as a near-Gaussian. The magic is in the accumulation: a thousand of these 1% nudges, each pointing along the (predicted) score, compound into a complete journey from static to a sharp digit. Then we add $\sigma_{500}\,\text{noise} = \sqrt{0.01}\cdot\text{noise} = 0.1\,\text{noise}$ to keep the trajectory stochastic, and move to $t = 499$. Cut the number of steps and each remaining step has to do more work — bigger, less-Gaussian jumps — which is the source of the quality loss we examine next.

One question the code should provoke: *why do we add fresh noise back at every step?* We just spent the whole post learning to remove noise, and now the sampler re-injects some on the way out. The reason is subtle and important. The reverse step is not deterministic — it is a *distribution*, $p_\theta(x_{t-1} \mid x_t)$, a Gaussian with the mean we computed and a variance $\sigma_t^2$. Sampling from that distribution means drawing the mean *plus* a noise term scaled by $\sigma_t$; if we only ever took the mean, we would not be sampling from the model, we would be taking the mode of a chain of conditionals, which collapses diversity and tends toward over-smooth, "average" images. The injected noise is what makes each run produce a *different* sample — it is the source of the stochasticity that lets one trained model generate endless variety from the same starting distribution. It is also why two runs from different random seeds give different images, and why diffusion covers modes: the stochastic exploration visits different regions of the manifold on different runs.

But here is a beautiful loose thread, and it is the seam where the next chapter of the story opens. What if we set $\sigma_t = 0$ — take the mean every time, no injected noise? You might expect garbage, but you get something subtler: a *deterministic* mapping from the initial noise $x_T$ to a final image, where the same starting noise always yields the same image. This deterministic variant is essentially **DDIM**, and it turns out to correspond to integrating an *ordinary differential equation* (the probability-flow ODE) rather than a stochastic one. The deterministic ODE view is what lets you take far fewer, larger steps without the quality falling apart — because integrating a smooth ODE accurately needs far fewer evaluations than simulating a noisy stochastic process. That single observation — *the reverse process can be made deterministic, and then it is an ODE you can integrate cheaply* — is the entire foundation of fast sampling, and it is the subject of [DDIM and fast deterministic sampling](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling). For now, just register that the noise injection is a *choice*, not a necessity, and that turning it off opens the door to the 50-step and 20-step samplers that made diffusion practical.

## Results: quality climbs with steps, and how diffusion stacks up

A foundation post owes you measured results, not just code. Two questions matter most for building intuition: how does sample quality depend on the number of sampling steps, and how does diffusion compare to the other generative families on the axes that actually decide which one you reach for?

First, **steps versus quality**. Because every reverse step is a small correction, the number of steps directly controls how faithfully you integrate the reverse trajectory. Too few steps means each step must take a large, less-accurate jump, and error accumulates as blur, blotches, and artifacts. More steps means smaller, more-accurate jumps and cleaner samples — at the cost of more sequential network evaluations (one full U-Net forward pass per step). For vanilla DDPM ancestral sampling, quality keeps improving up to many hundreds of steps; the classic recipe uses the full $T = 1000$, and you can *see* the difference on MNIST: at 5 steps the digits are smeary ghosts, at 50 they are recognizable but rough, at 1000 they are crisp.

![A before and after comparison of few step sampling that is fast but blurry against many step sampling that is slow but sharp, illustrating the speed quality trade-off](/imgs/blogs/diffusion-from-first-principles-7.png)

Figure 7 frames this as the trade-off it is: few steps buys speed at the price of quality; many steps buys quality at the price of latency. Here is an honest, reproducible way to *measure* it rather than eyeball it: fix a random seed, generate a fixed number of samples (say 50,000 for a stable FID) at each step budget, and compute FID against a held-out reference set of real images. You will get a curve of FID versus steps that falls steeply at first (each early step helps a lot) and then flattens (diminishing returns). That curve is the steps↔quality Pareto frontier, and picking a point on it is one of the most consequential decisions in a real deployment.

| Sampling steps (DDPM) | Relative quality | Sequential net evals | Latency on one A100 (toy) | When to use |
| --- | --- | --- | --- | --- |
| 5 | Poor — blurry, artifacted | 5 | ~0.05 s | Debugging the loop only |
| 50 | Decent — recognizable | 50 | ~0.5 s | Quick previews |
| 250 | Good | 250 | ~2.5 s | Reasonable default for DDPM |
| 1000 | Best (DDPM ceiling) | 1000 | ~10 s | Final quality, no time limit |

(Latencies are illustrative order-of-magnitude figures for a small MNIST U-Net on an A100, scaling linearly with steps; absolute numbers depend entirely on model size and resolution. The *shape* — linear cost in steps, diminishing quality returns — is the durable lesson.) The obvious pain point jumps out: a thousand sequential forward passes is *slow*, and it is slow in the worst way, because the steps are sequential and cannot be parallelized. This is **the** headline weakness of vanilla diffusion, and the entire reason later posts exist. [DDIM and fast deterministic sampling](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling) shows how to reinterpret the reverse process as an ODE and cut 1000 steps to 50 *without retraining*; the [samplers deep dive](/blog/machine-learning/image-generation/samplers-deep-dive) pushes to 20-step high-order solvers; and Track E's distillation methods get all the way to 1–4 steps. Hold the speed problem in mind as the thread that pulls the rest of the series forward.

Second, **diffusion versus GAN versus VAE**. These are the three workhorses of the pre-2022 and 2022-onward eras, and the comparison reveals exactly why diffusion took over. The axes that matter are training stability, mode coverage (does it represent the full diversity of the data, or drop modes?), the number of sampling steps, and sample quality.

![A comparison matrix of diffusion, GAN, and VAE across training stability, mode coverage, sample steps, and sample quality](/imgs/blogs/diffusion-from-first-principles-8.png)

Figure 8 is the comparison at a glance; the table below says it in words you can cite:

| Property | Diffusion | GAN | VAE |
| --- | --- | --- | --- |
| Training objective | MSE on predicted noise (a likelihood bound) | Adversarial minimax | Variational bound (ELBO) |
| Training stability | High — single fixed regression target | Low — fragile equilibrium, can diverge | High — stable, but bound is loose |
| Mode coverage | Excellent — loss grades every mode | Poor — prone to mode collapse | Good — but over-smooth |
| Sample quality | Excellent — sharp, detailed | Excellent — sharp (when it works) | Lower — characteristically blurry |
| Sampling cost | High — many sequential steps | Low — single forward pass | Low — single forward pass |
| One-shot likelihood | Approximate (via the bound) | None (implicit model) | Approximate (via the bound) |

Read the table as a story. GANs and diffusion both produce sharp images, but GANs pay for it with brutal training instability and chronic mode collapse — you can get a gorgeous generator or a collapsed one, and the difference is sometimes a learning-rate tweak. VAEs are stable and cover modes well, but their single Gaussian decoder averages over uncertainty and the result is blurry — the chronic VAE disease. Diffusion is the rare combination of *all three goods*: stable training (like a VAE), full mode coverage (like a VAE), and sharp samples (like a GAN). Its one real cost is sampling speed — the many sequential steps — and that cost turns out to be *attackable* by better samplers and distillation, whereas GAN instability and VAE blur are more intrinsic. That asymmetry — diffusion's weakness is fixable, the others' weaknesses are structural — is the deepest reason diffusion won.

This maps directly onto the **generative trilemma** that threads the whole series: sample quality × mode coverage × sampling speed, pick (at most) all-but-one cheaply. GANs sacrifice coverage; VAEs sacrifice quality; vanilla diffusion sacrifices speed. The reason diffusion became the dominant paradigm is that speed is the easiest of the three corners to buy back after the fact — you can take a slow-but-excellent diffusion model and *make it fast* with a better sampler or a distilled student, but you cannot easily make a mode-collapsed GAN cover modes or a blurry VAE sharp. We will spend Track E doing exactly that buying-back, and you will see the speed gap close from 1000 steps to a single step while keeping the quality and coverage diffusion gives you for free.

## Stress-testing the model

A model you cannot break is a model you do not understand. Let me poke at the diffusion picture from a few angles, because the failure modes are as instructive as the successes.

**What happens with too few steps?** As the table showed, quality degrades, but the *way* it degrades is informative. With vanilla DDPM ancestral sampling, dropping from 1000 to (say) 10 steps forces each step to take a huge jump, and the near-Gaussian approximation of the reverse step breaks down badly — the result is over-smoothed, washed-out mush, because the sampler cannot commit to sharp high-frequency detail in so few coarse jumps. Importantly, this is a *sampler* limitation, not a *model* limitation: the same trained network, sampled with a smarter ODE solver (DDIM, DPM-Solver), produces good images in 20–50 steps. The lesson — which trips up newcomers constantly — is that **the trained model and the sampler are separable.** You train once; you can swap samplers freely at inference. A "bad at low steps" diffusion model is usually just being sampled with a low-order method.

**What if the noise schedule is wrong?** The schedule decides how the model's capacity is spread across difficulty levels. A schedule that adds noise too fast (large $\beta$ early) destroys structure before the model gets to learn the easy refinement steps; too slow and you waste capacity on near-clean images that barely need denoising. The linear schedule we used works fine for MNIST but is known to be suboptimal at high resolution, where it leaves too little "hard" signal at the end — the **cosine schedule** (Nichol & Dhariwal, 2021) fixes this by keeping more signal late, and **zero-terminal-SNR** fixes a subtler bug where the final step is not *quite* pure noise. These are exactly the quiet, consequential choices [noise schedules and the parameterization zoo](/blog/machine-learning/image-generation/noise-schedules-and-the-parameterization-zoo) is about. The point for now: the schedule is not a throwaway hyperparameter; it shapes where the model spends its effort.

**Why predict $\epsilon$ and not $x_0$ directly?** We claimed they are equivalent, so why does the choice matter? Numerically, predicting $\epsilon$ keeps the target at unit variance for all $t$, while predicting $x_0$ at high $t$ means predicting a signal scaled down by $\sqrt{\bar{\alpha}_t} \approx 0$ — a tiny, ill-conditioned target. But $\epsilon$-prediction has its own weakness at *low* $t$: when there is almost no noise, predicting the small noise is nearly trivial and contributes little learning signal, yet the loss still weights it. This tension is why later models often use **v-prediction** (a clever interpolation between $\epsilon$ and $x_0$ that stays well-conditioned across all $t$) — another topic for the parameterization post. That such a "trivial" choice of prediction target has real consequences is a hint at how much engineering hides under the clean math.

**What about the cost — is it really that bad?** Yes and no. A thousand sequential U-Net evaluations to make one image is genuinely expensive: at SDXL scale that is real seconds per image and real dollars at volume. But the cost is *latency*, not *compute-per-step* — each individual step is just one network forward pass, the same cost as one GAN or VAE generation. The slowness is purely the *count* of steps. And because the steps trace an ODE trajectory (the deterministic view from DDIM), you can integrate that trajectory with far fewer, larger, smarter steps — which is why the field went from 1000 steps in 2020 to single-step distilled models by 2024. The cost was never fundamental; it was the *first, dumbest* way to integrate the reverse process, and four years of better integration schemes brought it to heel.

**Can it overfit or memorize?** With a small dataset and a big model, yes — diffusion models can and do memorize training images, reproducing them at sampling time, which is a real privacy and copyright concern at the frontier. The same mode-coverage property that makes diffusion faithful to the data also makes it faithful to *specific* training images if there are too few of them or they are duplicated. Large, deduplicated datasets and regularization mitigate this, but it is a live issue, not a solved one — worth knowing the method has this sharp edge.

## Case studies: from the toy to the systems that shipped

The MNIST loop above is the *same algorithm* that powers the models you have actually used. To make that concrete — and to ground the abstract framework in real numbers — here are four landmarks where the diffusion idea from this post became a shipped system. The throughline: nothing about the core forward/reverse/noise-prediction story changes; what changes is the scale, the space the diffusion runs in, and the cleverness of the sampler.

**DDPM (Ho et al., 2020) — the first time it worked at photo scale.** The paper this post unpacks trained a U-Net of roughly 35M–114M parameters (depending on the dataset) with exactly the $\epsilon$-prediction MSE loss we coded, on $T = 1000$ steps with a linear $\beta$ schedule. On unconditional CIFAR-10 ($32\times32$) it reported a Fréchet Inception Distance (FID) of about 3.17 — a strong result for 2020 that beat many GANs of the era on that benchmark, and, crucially, did so *without* a discriminator or any adversarial instability. The headline cost was the one we keep flagging: generating a single batch required 1000 sequential network evaluations. That single number — 1000 — is the seed of every speed-up in the rest of the series. The paper's lasting contribution was less the FID and more the demonstration that the boring, stable, simplified objective was enough to match adversarial methods, which redirected the whole field's attention.

**Improved DDPM and ADM (Nichol & Dhariwal, 2021) — diffusion beats GANs on ImageNet.** The follow-up work introduced the cosine noise schedule, learned reverse-process variances, and importance-sampled timesteps, then scaled the U-Net up with more attention resolutions and width. The result, *Ablated Diffusion Model* (ADM), reported an ImageNet $256\times256$ FID around 4.59 (and lower with classifier guidance), beating the best GANs of the time (BigGAN-deep) on the same benchmark — the paper's title was, pointedly, "Diffusion Models Beat GANs on Image Synthesis." This is the moment diffusion stopped being a curiosity and became the front-runner. The relevant lesson: every gain here came from the *quiet* choices — schedule, variance parameterization, loss weighting — exactly the knobs we flagged in the stress-test section, not from changing the fundamental forward/reverse framework.

#### Worked example: reading an FID improvement honestly

Suppose you read that switching from a linear to a cosine schedule on a $256\times256$ model moved FID from, say, 6.0 to 4.6. What does that *mean*, and how would you trust it? FID measures the Fréchet distance between the feature distributions (Inception-V3 activations) of generated and real images — lower is closer to real, and a drop of 1.4 FID at this range is a clearly visible quality improvement, not noise. But FID is famously sensitive to sample count and reference set: a "valid" comparison fixes the number of generated samples (the standard is 50,000), uses the *same* reference statistics for both runs, fixes the random seed for the generation, and keeps every other hyperparameter constant so the schedule is the only changed variable. If a paper computes FID on 5,000 samples instead of 50,000, the number inflates and the comparison across papers breaks. The discipline — fix sample count, fix reference set, change one thing — is exactly how you would A/B a schedule change in your own training run before believing the win. Numbers without that protocol are decoration; the protocol is what makes a measured result a *proof*.

**Latent Diffusion / Stable Diffusion (Rombach et al., 2022) — moving the diffusion off the pixels.** The breakthrough that put diffusion on every laptop was not a change to the diffusion math at all — it was *where* the diffusion runs. Stable Diffusion trains a VAE to compress a $512\times512\times3$ image (786,432 numbers) into a $64\times64\times4$ latent (16,384 numbers, a 48× reduction) and runs the *exact same* DDPM/denoising process we built, but on those latents instead of raw pixels. The U-Net ($\sim$860M parameters for SD 1.5) never touches a full-resolution image during diffusion; the VAE decoder lifts the final latent back to pixels in one pass at the end. This is a direct application of every idea in this post — same forward marginal, same $\epsilon$-loss, same ancestral sampler — composed with the VAE from [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch). The payoff was an order-of-magnitude reduction in training and inference cost, which is what made open, runnable text-to-image possible. The full story is [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion); the point here is that our toy loop is *structurally complete* — you make it Stable Diffusion by changing the space it runs in, not the algorithm.

**The fast-sampler lineage (DDIM 2021 → DPM-Solver 2022 → Turbo/LCM 2023–2024) — killing the 1000-step cost.** The single weakness we keep circling — sequential step count — was attacked relentlessly, and the progression is a clean illustration of "the model and the sampler are separable." DDIM (Song et al., 2021) reinterpreted the reverse process as a deterministic ODE and got high-quality samples from a DDPM-trained model in 50 steps with *no retraining* — a 20× latency cut for free. DPM-Solver and DPM-Solver++ (Lu et al., 2022) used higher-order ODE integration to reach comparable quality in roughly 15–25 steps. Then distillation methods — Latent Consistency Models, SDXL-Turbo, DMD2 — pushed all the way to 1–4 steps by *training a student* to take giant strides, trading a one-time training cost for a permanent inference speed-up, with one-step FIDs on the same order as the multi-step teacher. Across four years the headline step count fell from 1000 to 1, while the *trained model* in each case was still doing the same noise-prediction job we coded above. That is the strongest possible evidence that diffusion's slowness was never fundamental — it was an integration choice — and it is the entire thesis of [why diffusion is slow and how to fix it](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling) and Track E.

| Model / method | Year | Space | ~Params | Reported result | Sampling cost |
| --- | --- | --- | --- | --- | --- |
| DDPM | 2020 | Pixels ($32^2$) | 35M–114M | CIFAR-10 FID ~3.17 | 1000 steps |
| ADM (Improved DDPM) | 2021 | Pixels ($256^2$) | ~500M | ImageNet FID ~4.59 (beats BigGAN) | 250–1000 steps |
| Stable Diffusion 1.5 | 2022 | VAE latent (48× smaller) | ~860M U-Net | Open text-to-image at $512^2$ | 50 steps (DDIM) |
| SDXL-Turbo / LCM | 2023–24 | VAE latent | ~2.6B (distilled) | Near-teacher quality | 1–4 steps |

(Headline FIDs are as reported in the respective papers on the stated benchmark and sample protocol; treat cross-paper comparisons cautiously since reference sets and sample counts vary. The *trend* — diffusion matching then beating GANs, then moving to latents for efficiency, then collapsing the step count via better sampling and distillation — is the durable narrative.)

What I want you to take from these four is a single structural observation: **every one of them is the loop you just wrote, with one thing changed.** DDPM is the loop. ADM is the loop with a better schedule and a bigger net. Stable Diffusion is the loop run in a compressed latent. Turbo is the loop with a distilled few-step sampler. The framework is so stable that four years of frontier progress amounts to *swapping one component at a time* — the space, the schedule, the architecture, the sampler — while the forward marginal, the $\epsilon$-objective, and the score connection sit untouched at the center. That stability is not a coincidence; it is the payoff of having built the method on a clean probabilistic foundation rather than an adversarial hack.

## When to reach for diffusion (and when not to)

Having built the model and stress-tested it, here is the decisive guidance — because every technique is a cost, and the engineer's job is knowing when the cost is worth paying.

**Reach for diffusion when** you need high-quality, diverse samples and you can tolerate (or fix) multi-step sampling. This is the default for essentially all modern text-to-image, image editing, super-resolution, and increasingly video and audio. The training stability alone is worth a lot: a team can train a diffusion model without the black-magic babysitting GANs demand, and the simple MSE loss means fewer ways to shoot yourself in the foot. If you are starting a generative-image project in 2026, diffusion (or its flow-matching cousin) is the right starting point unless you have a specific reason otherwise.

**Do not reach for vanilla 1000-step DDPM** in production — that is a teaching baseline, not a deployment target. Use a modern sampler (DPM-Solver++, UniPC, or Euler for flow models) at 20–50 steps, or a distilled few-step model if latency is critical. Sampling at the full 1000 steps when a 25-step DPM-Solver gives equivalent quality is simply wasting 40× the compute and latency for nothing.

**Reach for a GAN instead when** you need genuinely single-shot, real-time generation and can accept the training pain and coverage risk — there are still niches (some real-time avatars, certain super-resolution pipelines) where a well-tuned GAN's one-pass speed wins, and notably GANs have *returned* inside diffusion as a *distillation loss* (the adversarial signal in SDXL-Turbo / DMD2 that enables one-step diffusion sampling). The GAN did not die; it got absorbed.

**Reach for a VAE instead when** you need a fast, smooth, structured *latent space* more than you need photorealistic samples — and note that this is exactly the role the VAE plays *inside* latent diffusion: Stable Diffusion uses a VAE to compress images into a latent space and runs the *diffusion* in that space, getting the VAE's efficiency and the diffusion's quality together. The families are not really competitors so much as components; the frontier stacks them.

**Do not use pixel-space diffusion at high resolution** — it is wasteful. The whole reason latent diffusion exists is that running the U-Net on raw 512×512 pixels is far more expensive than running it on a 64×64 compressed latent with near-identical quality. The MNIST toy here is pixel-space because MNIST is tiny; any real high-resolution model should diffuse in a VAE latent, as [latent diffusion and Stable Diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion) covers.

## The whole method, assembled

Step back and look at what we built, because the pieces now form a single coherent picture. We started with an impossible-looking goal — sample from the distribution of natural images — and we refused to attack it head-on. Instead we defined a *fixed* forward process that walks any image up a ladder of ever-noisier distributions to pure Gaussian noise, and we trained a *single* network to walk one rung back down. The training target fell out for free because we add the noise ourselves and therefore always know the answer; the objective turned out to be plain MSE on the predicted noise; and that noise prediction turned out to *be* the score, the compass that points toward real images. Generation is then nothing more than starting from noise and following that compass downhill-in-noise, uphill-in-probability, a thousand small steps onto the manifold. Training is stable because the target is fixed; modes are covered because the loss grades every example; quality is high because each step only makes a tiny, sharp correction. The one cost — sequential steps — is the loose thread the rest of the series pulls.

That is the whole method, and it is worth noticing how *few* moving parts it has: a schedule, a marginal, an MSE, a sampler. Everything else in modern image generation — latent spaces, transformers, text conditioning, guidance, distillation — is a modification of *one* of those four parts, layered onto this skeleton. When you read about MM-DiT or flow matching or ControlNet later in the series, trace each one back here: which of the four parts is it changing, and what does that buy? If you can answer that, you understand the frontier, because the frontier is this post's skeleton wearing better clothes. The capstone, [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack), assembles all those modifications into a serving pipeline — but the heart of that pipeline is the loop you wrote above.

## Key takeaways

- **Diffusion is two processes.** A *fixed* forward process gradually adds Gaussian noise until an image becomes pure static; a *learned* reverse process removes noise step by step to turn static back into an image. Only the reverse is trained.
- **We destroy data on purpose to create labels.** Because we add the noise ourselves, we always know the answer, so every image at every noise level becomes a labeled supervised example — converting an impossible density-modeling problem into easy regression.
- **The forward marginal is the workhorse.** $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\,x_0,\ (1-\bar{\alpha}_t)I)$ lets you jump to any noise level in one draw, so training never loops over steps.
- **The objective is just MSE on the noise.** $\mathcal{L}_\text{simple} = \mathbb{E}\,\lVert \epsilon - \epsilon_\theta(x_t, t)\rVert^2$. No adversary, no fragile balance — which is *why* diffusion is stable to train.
- **Predicting noise equals estimating the score.** $\nabla_{x_t}\log q(x_t) = -\epsilon / \sqrt{1-\bar{\alpha}_t}$, so removing predicted noise is stepping along the score, uphill toward higher-probability (more image-like) regions. That is *why* it generates images.
- **Diffusion covers modes** because the loss grades it on reconstructing every example — it cannot quietly drop categories the way a GAN can.
- **The one real cost is sampling speed** — many sequential steps — and unlike GAN instability or VAE blur, that cost is fixable with better samplers and distillation, which is the deepest reason diffusion won.
- **The model and the sampler are separable.** Train once; swap samplers freely. "Bad at low steps" is usually a sampler choice, not a model limitation.

## Further reading

- **Ho, Jain, & Abbeel, "Denoising Diffusion Probabilistic Models," NeurIPS 2020** — the DDPM paper. The forward marginal, the $\epsilon$-prediction objective, and $\mathcal{L}_\text{simple}$ all come from here; it is the paper this post unpacks.
- **Sohl-Dickstein et al., "Deep Unsupervised Learning using Nonequilibrium Thermodynamics," ICML 2015** — the original diffusion idea, five years before it worked at scale.
- **Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution," NeurIPS 2019** — the score-matching view that DDPM was later shown to be equivalent to; the source of the noise-equals-score intuition.
- **Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models," ICML 2021** — the cosine schedule and learned variances; practical improvements over vanilla DDPM.
- **Next in this track: [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm)** — the full variational-bound derivation that *proves* the noise-prediction loss, with the closed-form posterior and the schedule math.
- **Foundations: [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard)** and **[the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions)** — the manifold hypothesis and the generative trilemma this post builds on; **[variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch)** for the ELBO connection and the VAE that latent diffusion is built on.
- **Composing forward: [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance)** (how to steer the score with a prompt), **[DDIM and fast deterministic sampling](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling)** (cutting 1000 steps to 50), and the capstone **[building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack)** (where all of this becomes a serving pipeline).
