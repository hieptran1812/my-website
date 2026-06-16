---
title: "DDIM and Fast Deterministic Sampling: From 1000 Steps to 50 Without Retraining"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Derive the DDIM update from a non-Markovian forward process, see exactly why sigma=0 turns sampling into an ODE you can solve in 50 steps, and implement a deterministic sampler and DDIM inversion in PyTorch — all on a network you already trained."
tags:
  [
    "image-generation",
    "diffusion-models",
    "ddim",
    "fast-sampling",
    "deterministic-sampling",
    "probability-flow-ode",
    "ddim-inversion",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/ddim-and-fast-deterministic-sampling-1.png"
---

You have just trained a DDPM. The loss curve flattened out somewhere around epoch 300, the samples look like real CIFAR-10 birds and ships, and you are proud of it. Then you try to actually use it, and you discover the catch nobody warns you about loudly enough: to draw one batch of images, the model runs its U-Net **one thousand times in strict sequence**. Not in parallel — sequentially, because each step's input is the previous step's output. On an A100 that is roughly five minutes for a batch of 64. On a CPU it is a coffee break. If you want 50,000 samples to compute an FID, you are looking at hours of pure inference for a single evaluation. The model is good. The sampler is the problem.

The instinct is to retrain — maybe a different schedule, maybe fewer diffusion steps baked into the model from the start. That instinct is wrong, and the reason it is wrong is the most elegant result in the early diffusion literature. **You do not need to retrain anything.** The exact same weights, the exact same `eps_theta` network you already have, can produce comparable-quality samples in 50 steps instead of 1000 — and, as a bonus, do it *deterministically*, so the same noise seed always yields the same image, the latent space becomes smooth and interpolatable, and you can run the whole process backward to turn a real photograph into the latent that generates it. That last capability is the foundation of nearly every diffusion-based image-editing method shipped since 2022.

The trick is **DDIM** — Denoising Diffusion Implicit Models, from Song, Meng, and Ermon (ICLR 2021). The figure below is the whole pitch in one picture: identical network, swap the sampler, and the cost collapses by more than an order of magnitude while the FID barely moves.

![A before and after comparison showing DDPM ancestral sampling at 1000 stochastic steps versus DDIM deterministic sampling at 50 steps reusing the same trained weights](/imgs/blogs/ddim-and-fast-deterministic-sampling-1.png)

By the end of this post you will be able to: derive the non-Markovian forward process that DDIM defines and prove it preserves the DDPM marginals (which is *why* the trained network still applies); write down and understand the DDIM update rule term by term, including the `sigma_t` knob that interpolates between stochastic DDPM and deterministic DDIM; explain precisely why `sigma=0` lets you skip timesteps and converge in 20–50 steps; see DDIM as a first-order Euler discretization of the probability-flow ODE, connecting it back to the score-SDE picture; implement a DDIM sampler over a pretrained `eps`-model in PyTorch, and the `diffusers` one-line scheduler swap; implement DDIM inversion and measure its round-trip error; and reason honestly about where DDIM plateaus and why the field moved to higher-order solvers. This is post B4 in the series; it builds directly on [the DDPM derivation](/blog/machine-learning/image-generation/the-math-of-ddpm) and [the score-based SDE view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view), and it sets up [the samplers deep-dive](/blog/machine-learning/image-generation/samplers-deep-dive) and [image editing with diffusion](/blog/machine-learning/image-generation/image-editing-with-diffusion).

This is squarely the **sampling-speed** corner of the generative trilemma (quality × diversity × speed). DDPM optimizes quality and diversity and pays for it in speed. DDIM is the first move that buys back most of the speed without giving up much quality — and it does so purely at inference time, on a frozen network. That "frozen network" part is what makes it so practically important: it is free.

## 1. The problem: why DDPM sampling is so slow

Let me set up the running example precisely, because all the numbers later hang off it. We have a DDPM trained on $32\times32$ CIFAR-10 with $T = 1000$ diffusion steps and a linear $\beta$ schedule from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$, exactly the original Ho et al. (2020) recipe. The network is a U-Net that predicts the noise: given a noisy image $x_t$ and a timestep $t$, it outputs $\epsilon_\theta(x_t, t)$, an estimate of the Gaussian noise that was added to the clean image $x_0$ to produce $x_t$.

Recall from [the DDPM derivation](/blog/machine-learning/image-generation/the-math-of-ddpm) the two quantities that run this whole machine. The per-step variances $\beta_t$ define $\alpha_t = 1 - \beta_t$ and the cumulative product $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$. The forward (noising) process has a beautiful closed form — you can jump straight from the clean image to any noise level in one shot:

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar\alpha_t}\, x_0,\ (1 - \bar\alpha_t)\, \mathbf{I}\right),
\qquad
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I}).
$$

That closed form is what the network was trained against: sample a random $t$, noise an image, and regress $\epsilon_\theta(x_t, t)$ onto the true $\epsilon$ with the simple MSE loss $\mathcal{L}_\text{simple} = \mathbb{E}_{x_0, \epsilon, t}\big[\lVert \epsilon - \epsilon_\theta(x_t, t)\rVert^2\big]$.

Sampling is the reverse. DDPM's sampler is **ancestral**: it walks the chain backward one step at a time, from $x_T \sim \mathcal{N}(0, \mathbf{I})$ down to $x_0$, and at each step it samples from the learned reverse conditional $p_\theta(x_{t-1} \mid x_t)$. Concretely, the DDPM update is

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon_\theta(x_t, t)\right) + \sigma_t z, \qquad z \sim \mathcal{N}(0, \mathbf{I}),
$$

where $\sigma_t$ is the standard deviation of the injected noise (the original paper uses $\sigma_t^2 = \beta_t$ or the posterior variance $\tilde\beta_t$). That trailing $\sigma_t z$ term is doing real work — it is the stochasticity that makes the reverse process a faithful inversion of the Markov forward chain.

Here is the crux of the slowness. The DDPM reverse process is a **Markov chain**, and the derivation that justifies it — the variational bound that gives you $\mathcal{L}_\text{simple}$ — assumes you visit *every* timestep $T, T-1, \dots, 1, 0$. The bound is a sum of $T$ KL divergences, one per transition. You cannot simply decide to skip from $t = 1000$ to $t = 950$, because $p_\theta(x_{950} \mid x_{1000})$ was never something the model learned; the model only ever learned single-step transitions $p_\theta(x_{t-1} \mid x_t)$. Each step is also genuinely small: with $\beta_t$ on the order of $10^{-4}$ to $10^{-2}$, a single reverse step removes a tiny sliver of noise. Take too few and the chain has not had enough total "denoising budget" to clean the image, and you get gray mush or grainy artifacts.

So the cost is structural: $T = 1000$ sequential forward passes through a U-Net, each of which depends on the last. There is no parallelism across steps. If your U-Net forward pass is 15 ms on an A100, that is 15 seconds per image, ignoring batching overhead. This is the wall DDIM was built to break.

It is worth being precise about *why* the Markov structure is the trap, because the precision is exactly what DDIM exploits. The DDPM training objective comes from the variational (ELBO) bound on the data log-likelihood, which decomposes into a telescoping sum over the chain:

$$
-\log p_\theta(x_0) \le \mathbb{E}_q\Big[\underbrace{D_\text{KL}\!\big(q(x_T\mid x_0)\,\Vert\,p(x_T)\big)}_{L_T}\;+\;\sum_{t=2}^{T}\underbrace{D_\text{KL}\!\big(q(x_{t-1}\mid x_t, x_0)\,\Vert\,p_\theta(x_{t-1}\mid x_t)\big)}_{L_{t-1}}\;-\;\underbrace{\log p_\theta(x_0\mid x_1)}_{L_0}\Big].
$$

Every $L_{t-1}$ term trains $p_\theta$ to match the *single-step* posterior $q(x_{t-1}\mid x_t, x_0)$, which under the Markov forward chain is a Gaussian with the known posterior mean and variance $\tilde\beta_t$. The network never sees a multi-step posterior like $q(x_{t-5}\mid x_t, x_0)$, so it has no learned conditional to sample from if you try to jump five steps at once. You could in principle *retrain* with a coarser chain — say $T = 50$ — but then you have a different model with a different noise schedule, you have thrown away the fine-grained denoiser you trained, and (empirically) a 50-step-trained DDPM is worse than a 1000-step model sampled with a smart 50-step sampler. The whole appeal of DDIM is that it leaves the 1000-step training intact and changes only how you traverse it.

A useful way to hold this: the trained network is a *noise-level-conditioned denoiser*, a function of the continuous quantity $\bar\alpha_t$, not of the discrete index $t$ per se. It will denoise *any* valid noise level you hand it. What DDPM's sampler refuses to do is hop between non-adjacent levels — and that refusal is a property of the *sampler's* derivation, not of the *network's* competence. DDIM swaps the sampler for one that can hop, while feeding the same competent denoiser. That decoupling of "what the network knows" from "how the sampler walks" is the conceptual key to the entire fast-sampling literature.

#### Worked example: the cost of one FID evaluation

Suppose you are tuning a CIFAR-10 DDPM and you want a clean FID number, which by convention means generating 50,000 samples and comparing their Inception feature statistics to the 50,000-image training set. With a batch size of 256 and a 15 ms U-Net step on an A100, one batch of 256 images takes $1000 \times 15\,\text{ms} = 15$ s. You need $50000 / 256 \approx 196$ batches, so $196 \times 15\,\text{s} \approx 49$ minutes of pure GPU time for a *single* FID number. If you are sweeping a hyperparameter across 8 settings, that is over six hours of inference just to evaluate. Switch the sampler to DDIM at 50 steps and that 49 minutes becomes under 2.5 minutes. The model did not change. The evaluation got $20\times$ cheaper. This is why every serious diffusion training loop evaluates with a fast deterministic sampler, not the ancestral one.

## 2. The key idea: a different forward process with the same marginals

Here is the move that makes everything work, and it is genuinely clever. Song et al. asked a question that, in hindsight, is obvious but in 2020 was not: *what exactly does the trained network depend on?*

Look back at the training objective. The network $\epsilon_\theta(x_t, t)$ is trained using only the **marginal** $q(x_t \mid x_0)$ — the closed-form "noise the image to level $t$" distribution. At no point during training does the model see the joint forward process or its Markov structure. It sees pairs $(x_t, \epsilon)$ where $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$. That is it.

The implication is stunning. **Any forward process whatsoever that produces the same marginals $q(x_t \mid x_0)$ is compatible with the trained network.** The DDPM Markov chain is one such process, but it is not the only one. If you can cook up a *different* forward process — even a non-Markovian one — that happens to have the identical per-timestep marginals, then the same $\epsilon_\theta$ is the right network for its reverse process too, with zero retraining. And if that alternative process admits a reverse process you can run with bigger strides, you win.

![A branching diagram showing that the DDPM Markov forward and the DDIM non-Markovian forward share the identical marginal so the same epsilon-prediction loss and trained weights apply to both](/imgs/blogs/ddim-and-fast-deterministic-sampling-3.png)

DDIM defines exactly such a family. Instead of a Markov chain where $x_t$ depends only on $x_{t-1}$, DDIM defines a forward process where each $x_{t-1}$ is conditioned on *both* $x_t$ and the original clean image $x_0$:

$$
q_\sigma(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\ \sqrt{\bar\alpha_{t-1}}\, x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\cdot \frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1 - \bar\alpha_t}},\ \sigma_t^2 \mathbf{I}\right).
$$

This looks like it fell out of the sky, so let me unpack what it is engineered to do. It is conditioned on $x_0$, which makes it non-Markovian (the DDPM forward never conditions on $x_0$). It has a free parameter $\sigma_t \ge 0$ that controls how much randomness each step injects. And — this is the entire point — its **marginal** is pinned. The family is constructed so that for every choice of $\sigma_t$, integrating out the chain gives back exactly $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1 - \bar\alpha_t)\mathbf{I})$, the same Gaussian DDPM uses.

Let me prove that, because it is the load-bearing claim and it is short.

### The marginal-preservation proof

We want to show that if $x_t \sim \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1 - \bar\alpha_t)\mathbf{I})$ and we draw $x_{t-1}$ from the DDIM transition $q_\sigma(x_{t-1} \mid x_t, x_0)$ above, then $x_{t-1} \sim \mathcal{N}(\sqrt{\bar\alpha_{t-1}}\,x_0, (1 - \bar\alpha_{t-1})\mathbf{I})$. The proof is a Gaussian linear-combination argument: a linear function of a Gaussian, plus independent Gaussian noise, is Gaussian, so we just need to match the mean and variance.

Write $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$ with $\epsilon \sim \mathcal{N}(0, \mathbf{I})$. Then the standardized residual that appears in the transition mean simplifies:

$$
\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1 - \bar\alpha_t}} = \frac{\sqrt{1 - \bar\alpha_t}\,\epsilon}{\sqrt{1 - \bar\alpha_t}} = \epsilon.
$$

So the mean of the DDIM transition is $\sqrt{\bar\alpha_{t-1}}\,x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\,\epsilon$, and we add independent noise $\sigma_t z$ with $z \sim \mathcal{N}(0, \mathbf{I})$. Therefore

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\,\epsilon + \sigma_t z.
$$

Take the conditional mean given $x_0$: both $\epsilon$ and $z$ are zero-mean, so $\mathbb{E}[x_{t-1} \mid x_0] = \sqrt{\bar\alpha_{t-1}}\,x_0$. Good — that matches. Now the variance. Since $\epsilon$ and $z$ are independent standard Gaussians, the two noise terms add in quadrature:

$$
\operatorname{Var}[x_{t-1} \mid x_0] = \left(1 - \bar\alpha_{t-1} - \sigma_t^2\right) + \sigma_t^2 = 1 - \bar\alpha_{t-1}.
$$

The $\sigma_t^2$ terms cancel exactly. That cancellation is the whole magic trick: the coefficient $\sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}$ in front of $\epsilon$ was *chosen* so that whatever variance you take out of the deterministic part by raising $\sigma_t$, you put back in via the $\sigma_t z$ term. The total stays $1 - \bar\alpha_{t-1}$ for *any* $\sigma_t$. So

$$
q_\sigma(x_{t-1} \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_{t-1}}\,x_0,\ (1 - \bar\alpha_{t-1})\mathbf{I}\right),
$$

which is precisely the DDPM marginal at step $t-1$. By induction down the chain, every marginal matches DDPM, for every $\sigma_t$. The trained $\epsilon_\theta$ is valid for the entire family. **This is why no retraining is needed.**

Two things are worth pausing on. First, the proof never used the value of $\sigma_t$, which is what makes $\sigma_t$ a free dial — we will turn it to zero in a moment. Second, notice that DDPM itself is recovered for a specific choice: if you set $\sigma_t^2 = \tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\beta_t$ (the DDPM posterior variance), the DDIM family reduces exactly to the DDPM ancestral sampler. DDPM is one member of the DDIM family, the maximally stochastic one.

It helps to sit with *why* conditioning on $x_0$ is the right move, because it is the load-bearing design decision and it is not obvious. In the DDPM Markov chain, the only thing the reverse step knows about the trajectory's destination is what is implicitly encoded in $x_t$. By explicitly conditioning the forward transition on $x_0$, DDIM gives the process a "memory" of where the whole thing is headed. That memory is what lets the reverse process aim *directly* at a high-quality estimate of $x_0$ at every step (via $\hat x_0$) rather than crawling one tiny denoising step at a time. The Markov property — "the future depends on the present, not the past" — is exactly the property that forces small steps, because each step can only use local information. Dropping the Markov assumption (while keeping the marginals fixed, so the network still applies) is precisely what buys the freedom to take big strides. The trade we are making is subtle and worth naming: we give up the *probabilistic* interpretation of each individual step (the DDIM forward is not a "real" noising process you would ever run, it is a mathematical scaffold chosen to have the right marginals) in exchange for a reverse process that converges in far fewer steps. That is a great trade, and recognizing it as a *trade* — not a free lunch — is what lets you reason about when stochasticity is still worth paying for (section 11).

There is also a clean way to see that the family is *consistent*, not just marginally matched. Because each member shares the marginals, each member shares the same score function $\nabla_x \log q_t(x)$ at every noise level. The samplers differ only in how they *use* that shared score: DDPM walks a stochastic path that re-randomizes at each step, DDIM walks the unique deterministic path (the probability-flow trajectory) that threads the same densities. They are different routes through the same fog, guided by the same compass. That is the picture section 5 makes rigorous.

## 3. The DDIM update rule, term by term

Sampling needs a transition from $x_t$ to $x_{t-1}$ that does not require knowing $x_0$ — at sampling time we obviously do not have the clean image; that is what we are trying to generate. The fix is the same one DDPM uses: **predict** $x_0$ from $x_t$ using the network. Rearranging the closed-form $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$ and substituting the network's noise estimate $\epsilon_\theta(x_t, t)$ for the unknown $\epsilon$ gives the **predicted clean image**:

$$
\hat{x}_0 = f_\theta(x_t, t) = \frac{x_t - \sqrt{1 - \bar\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}.
$$

This is the single most important reparameterization in fast sampling. The network nominally predicts noise, but one division turns that into a prediction of the final image at *every* step. It is worth dwelling on how powerful that is: at the very first sampling step, when $x_T$ is essentially pure Gaussian noise, $\hat x_0$ is a blurry, low-confidence guess at the final image — but it is a guess at the *whole* image, not a one-step denoising. As $t$ shrinks, $\hat x_0$ sharpens monotonically into the final result. The DDIM update can therefore aim each step at a coherent target (the current best estimate of the answer) rather than blindly nudging in the local noise-gradient direction the way a small DDPM step does. That "aim at the answer" structure is what makes large strides safe: even a coarse step lands somewhere sensible because it is interpolating toward a full-image estimate, not extrapolating a tiny local correction. Now plug $\hat x_0$ into the DDIM transition mean and use the same $\epsilon \approx \epsilon_\theta$ for the direction term. You get the **DDIM update**:

$$
x_{t-1} = \underbrace{\sqrt{\bar\alpha_{t-1}}\,\hat{x}_0}_{\text{(1) predicted } x_0}
\;+\; \underbrace{\sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\,\epsilon_\theta(x_t, t)}_{\text{(2) direction pointing to } x_t}
\;+\; \underbrace{\sigma_t z}_{\text{(3) random noise}}.
$$

The figure below stacks the three pieces in the order the code computes them, because seeing the data flow makes the implementation obvious.

![A vertical stack showing the DDIM update computed as predicted clean image plus a deterministic direction term plus an optional stochastic term that DDIM sets to zero](/imgs/blogs/ddim-and-fast-deterministic-sampling-2.png)

Read the three terms as a recipe, not a formula:

**(1) Predicted $x_0$, re-noised to level $t-1$.** Take the network's best guess at the final image, $\hat x_0$, and noise it back to the appropriate signal scale $\sqrt{\bar\alpha_{t-1}}$ for the next, slightly cleaner timestep. This is the "jump toward the answer" term.

**(2) Direction pointing to $x_t$.** You cannot fully trust $\hat x_0$ — it is an estimate that gets better as $t$ shrinks. So you do not land entirely on the re-noised $\hat x_0$; you keep a controlled amount of the *current* noise direction $\epsilon_\theta(x_t, t)$, scaled by $\sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}$. Song et al. call this the "direction pointing to $x_t$." It keeps the trajectory consistent with where it currently is rather than teleporting to a possibly-wrong guess.

**(3) Random noise.** The optional stochastic kick, scaled by $\sigma_t$. This is the only term that uses fresh randomness $z$.

Now watch what happens when we set $\sigma_t = 0$. Term (3) vanishes entirely — no fresh noise. Term (2)'s coefficient simplifies to $\sqrt{1 - \bar\alpha_{t-1}}$. The update becomes **fully deterministic**:

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\left(\frac{x_t - \sqrt{1 - \bar\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}\right) + \sqrt{1 - \bar\alpha_{t-1}}\,\epsilon_\theta(x_t, t).
$$

Given $x_t$ and the (deterministic) network, $x_{t-1}$ is now a fixed function — no sampling. Run the whole chain from a fixed $x_T$ and you get a fixed $x_0$, every single time. This is what "deterministic sampling" means, and it is the form people mean when they say "DDIM" with no qualifier. The same seed gives the same image, bit for bit (modulo floating-point nondeterminism).

#### Worked example: one deterministic step by hand

Make it concrete with real numbers. Suppose we are at step $t$ where $\bar\alpha_t = 0.30$ and we are stepping to $t-1$ where $\bar\alpha_{t-1} = 0.45$ (a coarse stride — we are skipping). Say the network outputs a noise prediction with per-element magnitude around $\epsilon_\theta \approx 0.6$ for some pixel, and the current value there is $x_t \approx 1.2$. First the predicted clean value: $\hat x_0 = (1.2 - \sqrt{1-0.30}\cdot 0.6)/\sqrt{0.30} = (1.2 - 0.837\cdot 0.6)/0.548 = (1.2 - 0.502)/0.548 \approx 1.27$. Now the $\eta=0$ update: $x_{t-1} = \sqrt{0.45}\cdot 1.27 + \sqrt{1-0.45}\cdot 0.6 = 0.671\cdot 1.27 + 0.742\cdot 0.6 \approx 0.852 + 0.445 = 1.30$. Notice the structure: the value moved toward the (re-noised) clean estimate but kept a chunk of the current noise direction, and *no* random number was drawn anywhere. Re-run with the same inputs and you get $1.30$ again. That bit-for-bit repeatability across the whole tensor, across all steps, is the determinism — and it is exactly what makes seeds meaningful and inversion possible.

Two implementation notes fall out of this example. First, the predicted $\hat x_0 = 1.27$ slightly exceeds the valid image range $[-1, 1]$ — that is why the `clamp(-1, 1)` (static thresholding) in the code matters at high noise, where $\hat x_0$ is an unreliable extrapolation and can overshoot. Second, at the *very first* sampling step (high $t$, tiny $\bar\alpha_t$), the division by $\sqrt{\bar\alpha_t}$ is a division by a small number, which amplifies the network's error — this is why the early, high-noise steps are where curvature lives and why coarse grids hurt most there.

## 4. The sigma knob: one dial from DDPM to DDIM

The $\sigma_t$ parameter is not just an on/off switch between stochastic and deterministic. It is a continuous dial over a whole family of samplers, and it is worth understanding the spectrum because different points on it have different uses.

The standard parameterization, following the paper, introduces a scalar $\eta \in [0, 1]$ and sets

$$
\sigma_t = \eta \cdot \sqrt{\frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}}\,\sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}} = \eta \cdot \sqrt{\tilde\beta_t}.
$$

So $\eta$ scales the noise relative to the DDPM posterior standard deviation $\sqrt{\tilde\beta_t}$. The figure below shows the spectrum.

![A branching diagram showing the eta noise dial sweeping from DDPM ancestral sampling at eta one through partial stochastic samplers to deterministic DDIM at eta zero which enables skipping timesteps and an ODE trajectory](/imgs/blogs/ddim-and-fast-deterministic-sampling-4.png)

At $\eta = 1$ you recover the **DDPM ancestral sampler** exactly — full stochasticity, the original Markov-chain reverse process. At $\eta = 0$ you get **deterministic DDIM** — no injected noise, an implicit (hence the "I" in DDIM) deterministic generative process. In between, $0 < \eta < 1$ gives partially stochastic samplers that some people find produce slightly more diverse outputs than pure DDIM while still being reasonably fast.

The intermediate regime is more useful than it first appears, and it is worth understanding what you are actually buying as you turn the dial up from zero. Each unit of $\eta$ injects a controlled amount of fresh noise per step. That noise does two opposing things: it *widens* the set of outputs a given seed can produce (more diversity), and it *re-randomizes* the trajectory in a way that can correct accumulated network error (sometimes better fidelity at high step counts) but *adds* error you cannot remove at low step counts. So the right $\eta$ depends on your step budget. At 1000 steps a touch of stochasticity ($\eta \approx 0.2$–$1.0$) is fine or even helpful. At 50 steps you want $\eta = 0$ or very small, because there are not enough remaining steps to clean up injected noise. At 20 steps you want strictly $\eta = 0$. The dial and the step budget are coupled; you cannot tune one without the other. A practical default that ships in a lot of code: $\eta = 0$ everywhere, and reach for stochasticity only when you have a specific diversity complaint and the step budget to pay for it.

One more property of the dial matters for reproducibility. Determinism is binary in a way the rest of the spectrum is not: at exactly $\eta = 0$, output is a pure function of $(x_T, \text{model})$ and is bit-for-bit reproducible and invertible. At any $\eta > 0$, you have re-introduced a per-step random draw, so reproducibility now requires fixing the *entire sequence* of noise draws (not just $x_T$), and invertibility is gone — you cannot recover $x_T$ from $x_0$ because information was injected along the way. So if you need inversion (editing) or a clean semantic latent, $\eta = 0$ is not a preference, it is a requirement. This is the single most important reason DDIM defaults to deterministic in editing pipelines.

Why does the deterministic end let you skip timesteps when the stochastic end does not? Two complementary reasons.

First, the *empirical* reason, which is the one you feel immediately. With $\eta = 0$, every step is a deterministic function of the current state. There is no fresh noise being injected that the remaining steps must then "denoise away." When you inject noise at step $t$ (large $\eta$), you have created new error that subsequent steps have to clean up, and if you have skipped most of the steps, there are not enough of them left to clean it up — so stochastic sampling at low step counts accumulates noise it cannot remove and the image degrades. Deterministic sampling has no such accumulation; each step only ever *removes* signal-scaled noise, never adds any.

Second, the *structural* reason, which is the deeper one and the subject of the next section. With $\eta = 0$, the discrete update is a numerical integration step of a smooth ordinary differential equation. Smooth ODE trajectories are well-approximated by a small number of large steps, exactly the way you can integrate a smooth curve with a coarse grid. A stochastic process (an SDE) has a non-differentiable, jagged path that fundamentally needs fine steps to track. So determinism is not just convenient — it changes the mathematical object you are discretizing from a rough stochastic path to a smooth deterministic curve, and smooth curves are cheap to follow.

### Choosing the timestep subsequence

Concretely, "skipping steps" means you pick a strictly increasing subsequence $\tau = (\tau_1, \tau_2, \dots, \tau_S)$ of the full $\{1, \dots, T\}$, with $S \ll T$ (say $S = 50$), and run the DDIM update *between consecutive elements of the subsequence* rather than between adjacent integers. The update generalizes trivially — every $\bar\alpha_{t-1}$ becomes $\bar\alpha_{\tau_{i-1}}$:

$$
x_{\tau_{i-1}} = \sqrt{\bar\alpha_{\tau_{i-1}}}\,\hat{x}_0(x_{\tau_i}) + \sqrt{1 - \bar\alpha_{\tau_{i-1}} - \sigma^2}\,\epsilon_\theta(x_{\tau_i}, \tau_i) + \sigma z.
$$

The network was trained on every integer timestep, so it happily accepts $\tau_i = 980$ then $\tau_{i-1} = 960$ — a stride of 20 — because the only thing the update needs is the right $\bar\alpha$ values, and those are just looked up from the precomputed schedule. Two common choices for $\tau$: **linear** (evenly spaced, $\tau_i = \lfloor i \cdot T / S \rfloor$) and **quadratic** (denser near $t = 0$ where the image structure forms). The original paper found quadratic helps a little on CIFAR-10 at very low step counts; linear is the default in most libraries and is fine above ~50 steps.

The choice of subsequence is not cosmetic, and the ODE view from section 5 tells you exactly why. The integration error of a first-order solver is concentrated where the trajectory is most curved, and for diffusion that curvature is largest at *high* noise levels (the early sampling steps). So a naive linear subsequence spends the same step budget on the low-curvature tail (near $t = 0$, where the image is nearly formed and the ODE is almost straight) as on the high-curvature head — which is slightly wasteful. Quadratic spacing, by putting *more* steps near $t = 0$, is actually optimizing for the wrong thing on paper, yet it helps empirically because the low-noise region is where fine high-frequency detail is resolved and the human eye is most sensitive to errors there. This tension — "where is integration error largest" versus "where do errors *matter* perceptually" — is exactly what the modern solvers (and learned/optimized timestep schedules like those in AYS, "Align Your Steps") try to resolve principledly. For DDIM specifically, the pragmatic rule is: above 50 steps, the subsequence barely matters; at 20–50 steps, try both linear and quadratic and pick by eye; below 20 steps, the subsequence cannot save a first-order solver and you should switch solvers.

#### Worked example: choosing a stride for a 50-step budget

You have $T = 1000$ and a budget of $S = 50$ steps. Linear spacing gives a uniform stride of $1000/50 = 20$: you visit $t = 999, 979, 959, \dots, 19, 0$ (rounding aside). Each DDIM step now spans 20 training timesteps at once — a $20\times$ coarser grid than the model was trained on, which is fine because the deterministic update is integrating a smooth ODE. Now compare to a quadratic schedule $\tau_i \propto i^2$: the strides near $t = 1000$ might be ~40 (coarse, in the smooth high-noise region) while strides near $t = 0$ shrink to ~5 (fine, in the perceptually critical low-noise region). The *total* number of network evaluations is identical — 50 either way, so the wall-clock is identical — but the *allocation* of those evaluations differs. The cost is fixed by $S$; the quality is shaped by how you spend it. This is the cleanest possible illustration that fast sampling is a *budget-allocation* problem, not a *budget-size* problem, which is the mental frame the entire samplers literature operates in.

## 5. The ODE interpretation: DDIM is Euler's method

This is the section that connects DDIM to the grand unification from [the score-based SDE post](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view), and it is the reason DDIM generalizes to every fast sampler that came after it.

Start from the deterministic DDIM update and do a change of variables that the paper performs in its appendix. Define a new "time" coordinate $\sigma = \sqrt{1 - \bar\alpha}/\sqrt{\bar\alpha}$ (this is the noise-to-signal ratio; do not confuse it with the $\sigma_t$ injection-noise parameter — unfortunate notation collision, but standard) and a rescaled state $\bar x = x / \sqrt{\bar\alpha}$. After the algebra, the deterministic DDIM update for a small step becomes

$$
\bar x_{t-1} - \bar x_t = \left(\sigma_{t-1} - \sigma_t\right) \epsilon_\theta(x_t, t),
$$

which is exactly the forward-Euler discretization, with step size $\Delta\sigma = \sigma_{t-1} - \sigma_t$, of the ordinary differential equation

$$
\frac{d\bar x}{d\sigma} = \epsilon_\theta\!\left(\frac{\bar x}{\sqrt{\sigma^2 + 1}}, t(\sigma)\right).
$$

Let me actually do the algebra, because the change of variables is the kind of thing that looks like sleight of hand until you grind through it once. Start from the $\sigma=0$ DDIM update and group the two coefficients of $x_t$ and $\epsilon$:

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\cdot\frac{x_t - \sqrt{1 - \bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}} + \sqrt{1 - \bar\alpha_{t-1}}\,\epsilon_\theta = \frac{\sqrt{\bar\alpha_{t-1}}}{\sqrt{\bar\alpha_t}}\,x_t + \left(\sqrt{1-\bar\alpha_{t-1}} - \frac{\sqrt{\bar\alpha_{t-1}}\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}\right)\epsilon_\theta.
$$

Now divide both sides by $\sqrt{\bar\alpha_{t-1}}$ to switch to the rescaled state $\bar x_t = x_t/\sqrt{\bar\alpha_t}$. The first term becomes simply $\bar x_t$ (the $\sqrt{\bar\alpha_{t-1}}$ cancels on the left and the $\sqrt{\bar\alpha_t}$ in the denominator turns $x_t$ into $\bar x_t$). The coefficient on $\epsilon_\theta$ becomes $\sqrt{(1-\bar\alpha_{t-1})/\bar\alpha_{t-1}} - \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$. Define $\sigma = \sqrt{(1-\bar\alpha)/\bar\alpha}$ — the noise-to-signal ratio — and that coefficient is exactly $\sigma_{t-1} - \sigma_t$. So:

$$
\bar x_{t-1} = \bar x_t + (\sigma_{t-1} - \sigma_t)\,\epsilon_\theta(x_t, t) \quad\Longrightarrow\quad \bar x_{t-1} - \bar x_t = (\sigma_{t-1} - \sigma_t)\,\epsilon_\theta(x_t, t).
$$

That is forward Euler — "new state = old state + step size × velocity" — with step size $\Delta\sigma = \sigma_{t-1} - \sigma_t$ and velocity $\epsilon_\theta$. In words: **deterministic DDIM is Euler's method applied to an ODE whose velocity field is the network's noise prediction.** The figure below traces that chain of reasoning.

![A diagram tracing the deterministic DDIM update through a sigma reparameterization to the probability flow ODE which equals an Euler step and connects to the score SDE view motivating higher order solvers](/imgs/blogs/ddim-and-fast-deterministic-sampling-6.png)

And this ODE is not just *an* ODE — it is *the* **probability-flow ODE** from Song et al.'s score-SDE framework. Recall that the reverse-time SDE that undoes diffusion has a deterministic counterpart, the probability-flow ODE, which shares the *same marginal densities* $p_t(x)$ at every time but follows a deterministic trajectory instead of a stochastic one. Its velocity is built from the score function $\nabla_x \log p_t(x)$, and the noise-prediction network is a rescaling of the score:

$$
\nabla_x \log p_t(x) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar\alpha_t}}.
$$

Substitute that in and the probability-flow ODE *is* the DDIM ODE. So the two stories converge: the discrete-time, non-Markovian DDIM derivation and the continuous-time, score-based SDE derivation arrive at the identical deterministic sampler. DDIM was, in retrospect, the first practical probability-flow ODE solver — a first-order one.

That last phrase, "first-order," is the entire bridge to the rest of the sampling literature. Once you know DDIM is Euler integration of a smooth ODE, the obvious question is: *why use first-order Euler?* Numerical analysis has spent a century building better ODE solvers — Heun's method (second-order), Runge–Kutta (fourth-order), and the diffusion-specialized DPM-Solver and UniPC families that exploit the *semi-linear* structure of this particular ODE. Each of these gets the same accuracy as Euler in fewer steps by using extra network evaluations more cleverly. That is exactly why DDIM, which plateaus around 20–50 steps, gives way to DPM-Solver++ and UniPC, which hit comparable quality in 10–20. We cover that progression in [the samplers deep-dive](/blog/machine-learning/image-generation/samplers-deep-dive); the point here is that DDIM is the foundation they all build on, the moment the field realized sampling is *numerical integration of an ODE*.

The "semi-linear structure" deserves one sentence because it is the specific thing the better solvers exploit and it explains the size of the win. The diffusion ODE has the form $d\bar x/d\sigma = \epsilon_\theta(\cdot)$, but when you write it in the *non-rescaled* coordinate $x$, it splits into a linear drift term (which is just the $\bar\alpha$ rescaling, and can be integrated *exactly* in closed form) plus a nonlinear term involving the network. Euler/DDIM treats the whole thing as one nonlinear blob and approximates all of it crudely. DPM-Solver solves the linear part exactly and only approximates the small nonlinear part, so for the same number of network calls it makes a much smaller error. The practical upshot is that on the *same frozen network*, DPM-Solver++ at 15 steps roughly matches DDIM at 50 — another free speedup, again with no retraining, again purely from a better view of the ODE that DDIM first exposed. Everything compounds from "sampling is integrating this ODE," which is DDIM's permanent contribution even after faster solvers superseded it for raw step count.

#### Worked example: how many steps to follow the curve?

The ODE view gives an intuition for the step-count plateau. Forward Euler has local truncation error $O(\Delta\sigma^2)$ per step and global error $O(\Delta\sigma)$ over the trajectory. So halving the step size roughly halves the integration error — but the trajectory is *curved*, and most of the curvature is concentrated at high noise levels (large $\sigma$) early in sampling. With 1000 steps the grid is so fine that Euler error is negligible and quality is solver-limited (it equals the network's own ceiling). Drop to 100 steps and Euler error is still tiny relative to the network error — quality barely moves, which is exactly what the FID table in the next section shows (DDIM holds ~4.1 FID from 1000 down to 100 steps). Drop to 20 steps and Euler's $O(\Delta\sigma)$ error finally becomes visible against the network error, and FID starts climbing. The plateau-then-degrade shape is a direct consequence of being a first-order solver on a mostly-smooth curve. A second-order solver pushes the degradation point down to ~10 steps because its global error is $O(\Delta\sigma^2)$.

## 6. Implementing DDIM in PyTorch

Enough theory. Here is a complete, runnable DDIM sampler over a pretrained $\epsilon$-model. The only assumption is that you have a `model(x, t)` that returns a noise prediction and a `betas` schedule tensor of length $T$. Everything else is the update rule we derived. I have written it to expose the `eta` knob and the timestep subsequence explicitly, because those are the two things you will actually tune.

```python
import torch

def make_alpha_bars(betas: torch.Tensor) -> torch.Tensor:
    # betas: (T,) the per-step variance schedule used at training time.
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)  # alpha_bar_t = prod_{s<=t} (1 - beta_s)

@torch.no_grad()
def ddim_sample(model, betas, shape, n_steps=50, eta=0.0, device="cuda"):
    """Deterministic (eta=0) or stochastic (eta>0) DDIM sampling.

    model: callable (x_t, t_index) -> predicted noise eps_theta, same one a DDPM trained.
    betas: (T,) training beta schedule.
    n_steps: length of the timestep subsequence (e.g. 50), NOT the trained T (e.g. 1000).
    """
    T = betas.shape[0]
    alpha_bar = make_alpha_bars(betas).to(device)

    # Linearly spaced subsequence of the full {0, ..., T-1} timesteps.
    # This is the "skip steps" choice: we visit n_steps of the trained T.
    step_indices = torch.linspace(0, T - 1, n_steps, device=device).long()
    step_indices = torch.unique(step_indices)            # guard against dupes at low T
    times = list(reversed(step_indices.tolist()))        # high noise -> low noise

    x = torch.randn(shape, device=device)                # x_T ~ N(0, I)

    for i, t in enumerate(times):
        t_next = times[i + 1] if i + 1 < len(times) else -1   # -1 means "x_0"
        a_t = alpha_bar[t]
        a_next = alpha_bar[t_next] if t_next >= 0 else torch.ones((), device=device)

        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps = model(x, t_batch)                           # eps_theta(x_t, t)

        # (1) predicted x_0 via the reparameterization.
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        x0_pred = x0_pred.clamp(-1, 1)                    # optional: dynamic thresholding helps

        # sigma_t from the eta dial (the DDPM posterior std, scaled by eta).
        sigma = eta * torch.sqrt((1 - a_next) / (1 - a_t) * (1 - a_t / a_next))

        # (2) direction pointing to x_t.
        dir_xt = torch.sqrt(1 - a_next - sigma**2) * eps

        # (3) optional stochastic term (zero when eta=0).
        noise = sigma * torch.randn_like(x) if eta > 0 else 0.0

        x = torch.sqrt(a_next) * x0_pred + dir_xt + noise

    return x  # x_0
```

Walk through the load-bearing lines. The `step_indices` line is the entire "fast sampling" mechanism — it picks `n_steps` timesteps out of the trained `T`. Change `n_steps` from 1000 to 50 and you have a $20\times$ speedup, no other change. The `x0_pred` line is the predicted-$x_0$ reparameterization; the `clamp(-1, 1)` is a cheap, optional stabilizer (the "static thresholding" used by Imagen — it stops $\hat x_0$ from blowing up at high noise levels where the prediction is unreliable). The `sigma` line is the $\eta$ knob; set `eta=0.0` and `noise` is hard-zeroed and the sampler is deterministic. The final assignment is the three-term update verbatim.

A subtlety worth flagging: the loop visits `t_next` for the *next* timestep, and the last step targets $x_0$ by setting `a_next = 1` (since $\bar\alpha_0 = 1$, no noise at $t = 0$). Getting this boundary right matters — an off-by-one here is the single most common DDIM implementation bug, and it shows up as a faint residual noise texture on every output.

### The diffusers one-liner

You will almost never write the loop above in production; you will swap a scheduler. The whole point of `diffusers` schedulers is that the sampler is a pluggable object and the pipeline does not care which one it holds. Going from DDPM to DDIM is genuinely one line plus a step count:

```python
import torch
from diffusers import DiffusionPipeline, DDIMScheduler

pipe = DiffusionPipeline.from_pretrained(
    "google/ddpm-cifar10-32",          # a model trained with the DDPM objective
    torch_dtype=torch.float16,
).to("cuda")

# Swap the stochastic ancestral scheduler for the deterministic DDIM one.
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# 50 deterministic steps instead of the default 1000.
images = pipe(num_inference_steps=50, eta=0.0).images   # eta=0.0 -> deterministic
```

The `from_config` call is doing something subtle and important: it copies the *training* schedule (the `betas`, `beta_start`, `beta_end`, `num_train_timesteps`, prediction type) from the old scheduler into the new one. The DDIM scheduler must agree with the network on $\bar\alpha_t$, because those values are baked into the update. If you constructed `DDIMScheduler()` with default config instead, you would silently get the wrong $\bar\alpha$ and the output would be garbage. Always `from_config`.

For a real text-to-image pipeline the pattern is identical:

```python
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

img = pipe(
    "a photograph of a calico cat sitting on a windowsill, 50mm, soft light",
    num_inference_steps=50,
    guidance_scale=7.0,
    generator=torch.Generator("cuda").manual_seed(0),  # determinism: same seed, same cat
).images[0]
```

Because DDIM is deterministic, that `manual_seed(0)` makes the output **exactly reproducible** — rerun it and you get the identical image, which you cannot guarantee with an ancestral sampler that draws fresh noise at every step. That reproducibility is not a gimmick; it is what makes A/B testing prompts, debugging a guidance scale, and the inversion trick in the next section possible.

## 7. The results: steps vs FID vs wall-clock

Now the measurement angle, because "comparable quality" is a claim that needs a number behind it. The canonical numbers come from the DDIM paper's CIFAR-10 experiments (Song et al. 2021, Table 1), which I summarize here. The setup: the same pretrained DDPM, evaluated with FID on 50,000 samples against the CIFAR-10 training set, varying the sampler and the step count $S$. Lower FID is better.

![A matrix comparing DDPM and DDIM FID across step counts from 1000 down to 20 alongside wall-clock time showing DDIM degrades gracefully while DDPM collapses at low step counts](/imgs/blogs/ddim-and-fast-deterministic-sampling-5.png)

| Steps $S$ | DDPM ($\eta=1$) FID | DDIM ($\eta=0$) FID | Approx. wall-clock |
|---|---|---|---|
| 1000 | **3.17** | 4.04 | ~5 min / 50k batch unit |
| 250 | 6.84 | 4.16 | ~75 s |
| 100 | 10.6 | 4.16 | ~30 s |
| 50 | 29.0 | **4.67** | ~15 s |
| 20 | very high | 6.84 | ~6 s |

(FID figures from the DDIM paper's CIFAR-10 table; wall-clock figures are illustrative, scaled linearly with step count from a ~15 ms/step U-Net — they show the *shape*, not a benchmark on your hardware.)

Read the two columns side by side and the entire DDIM thesis is in the contrast. The DDPM ancestral sampler is *best* at the full 1000 steps (FID 3.17 — stochasticity at high resolution genuinely helps the last bit of fidelity), but it **falls off a cliff** as you cut steps: 6.84 at 250, 10.6 at 100, and a disastrous 29.0 at 50. DDIM, by contrast, is slightly worse than DDPM at 1000 steps (4.04 vs 3.17) but barely moves as you cut steps — 4.16 at both 250 and 100, and still a perfectly usable 4.67 at 50 steps. At 50 steps, DDIM's FID is **6× lower** than DDPM's. That is the whole game: DDIM trades a hair of peak quality for enormous robustness to step reduction.

The honest read for a practitioner: if you have unlimited compute and want the absolute best FID, run DDPM at 1000 steps. For literally every other situation — training-time evaluation, interactive generation, serving, anything cost-sensitive — run DDIM at 50 steps and accept a fractional FID cost for a $20\times$ speedup. This is not a close call.

A word on measuring this honestly, because FID is easy to compute wrong and the numbers above only mean something under a fixed protocol. FID compares the mean and covariance of Inception-v3 features between your generated set and a *reference* set, so three things must be pinned to compare two samplers fairly: (1) **the same reference set** — typically the full training split (50k for CIFAR-10), and you must use the *same* one across all rows, because FID is sensitive to reference-set size and identity; (2) **the same number of generated samples** — FID is biased downward as the sample count grows, so 10k-vs-50k comparisons are meaningless; the convention is 50k; (3) **a fixed seed protocol and a warm-up** — generate from a deterministic seed sequence so the comparison reflects the sampler, not the noise draw, and discard the first few GPU iterations from any wall-clock number to avoid counting kernel compilation and cache warm-up. The wall-clock numbers in the table are *illustrative* (scaled linearly from a single-step time) precisely because an honest latency benchmark would warm up, fix batch size, fix dtype, and report median over many runs on a *named* GPU — exactly what the next worked example does. When you see a "DDIM beats DDPM" FID claim with no sample count, reference set, or step budget stated, treat it as marketing, not measurement.

#### Worked example: serving latency on an A100

Put concrete dollars on it. Suppose you serve an SDXL endpoint on an A100 80GB. An SDXL U-Net forward pass at $1024\times1024$ with float16 and SDPA attention is roughly 80 ms (it is a much bigger network than the CIFAR toy). At 1000 DDPM steps that is 80 s/image — completely unusable for an interactive product, and at an A100 rate around \$1.50/hr that is about \$0.033 of GPU time per image. Switch to DDIM at 30 steps: $30 \times 80\,\text{ms} = 2.4$ s/image, about \$0.001 of GPU time per image. You went from 80 s to 2.4 s and from \$0.033 to \$0.001 per image, a $33\times$ improvement on both latency and cost, by changing one scheduler object. For an SDXL model specifically, 25–40 DDIM steps is the sweet spot; below ~20 you start seeing the curvature-induced degradation the ODE view predicted, and that is when you reach for DPM-Solver++ instead.

## 8. Deterministic sampling unlocks DDIM inversion

Determinism is not just about reproducibility and speed. It hands you something genuinely new: an **invertible** sampler. Because the $\eta=0$ DDIM update is a deterministic function $x_{t-1} = g(x_t)$, you can ask the reverse question — given an image, what latent $x_T$ would DDIM have started from to produce it? Run the update *backward* and you find out. This is **DDIM inversion**, and it is the backbone of real-image editing.

The forward (inversion) step inverts the sampling update. Sampling goes $x_t \to x_{t-1}$ (less noise); inversion goes $x_{t-1} \to x_t$ (more noise), using the same network. Reusing the ODE view, this is just integrating the same ODE in the opposite direction:

$$
x_{t} = \sqrt{\bar\alpha_{t}}\,\hat{x}_0(x_{t-1}, t-1) + \sqrt{1 - \bar\alpha_{t}}\,\epsilon_\theta(x_{t-1}, t-1).
$$

There is one approximation hiding here, and being honest about it matters: the exact inverse would evaluate the network at $x_t$, but we do not have $x_t$ yet — that is what we are solving for — so we use $\epsilon_\theta(x_{t-1}, t-1)$ as a stand-in. This is the standard "the velocity field is locally constant over one step" assumption. It is exact in the continuous limit (infinitesimal steps) and accumulates a small error over coarse steps. The figure below shows the full round-trip.

![A branching diagram of the DDIM inversion round trip mapping a real image to a latent and back to a reconstruction with optional prompt or latent editing in the middle](/imgs/blogs/ddim-and-fast-deterministic-sampling-7.png)

Here is inversion in code — note it is structurally the sampling loop with the time direction reversed:

```python
@torch.no_grad()
def ddim_invert(model, betas, x0, n_steps=50, device="cuda"):
    """Map a real image x0 to its DDIM latent x_T (eta=0, deterministic)."""
    T = betas.shape[0]
    alpha_bar = make_alpha_bars(betas).to(device)
    step_indices = torch.unique(torch.linspace(0, T - 1, n_steps, device=device).long())
    times = step_indices.tolist()                      # low noise -> high noise (forward)

    x = x0.to(device)
    for i, t in enumerate(times):
        t_next = times[i + 1] if i + 1 < len(times) else T - 1
        a_t = alpha_bar[t]
        a_next = alpha_bar[t_next]

        t_batch = torch.full((x0.shape[0],), t, device=device, dtype=torch.long)
        eps = model(x, t_batch)                         # locally-constant velocity assumption
        x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        x = torch.sqrt(a_next) * x0_pred + torch.sqrt(1 - a_next) * eps

    return x  # approximate x_T such that ddim_sample(...) reconstructs x0
```

The round-trip test — invert an image to $x_T$, then sample from that $x_T$ — should return the original image. How close it gets is the **inversion reconstruction error**, and it is the headline number for editing fidelity:

| Step count $S$ | Round-trip MSE (pixel, $[-1,1]$) | LPIPS | Quality of reconstruction |
|---|---|---|---|
| 100 | ~$1\times10^{-3}$ | ~0.01 | near-perfect |
| 50 | ~$3\times10^{-3}$ | ~0.03 | very good |
| 20 | ~$1\times10^{-2}$ | ~0.08 | visible drift on fine texture |

(Order-of-magnitude figures for an unconditional/low-guidance model; exact values depend on the model and image. The trend — error grows as steps drop — is the robust part.)

The catch that the editing literature is built around: this error is small for *unconditional or low-guidance* sampling, but it **blows up with classifier-free guidance**. Let me make the mechanism concrete, because "it drifts" is unsatisfying. With classifier-free guidance the effective noise prediction at scale $w$ is the extrapolation

$$
\tilde\epsilon_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \varnothing) + w\,\big(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\big),
$$

which for $w = 7.5$ is a $7.5\times$ extrapolation away from the unconditional prediction. This guided field is far more *curved* than the unguided one — the velocity changes rapidly along the trajectory — so the "velocity is locally constant over one step" assumption that inversion relies on fails much harder. Worse, inversion and sampling now use the *same* guided field but in opposite directions, and the linearization errors no longer cancel; they compound. Empirically, inverting a CFG-7.5 trajectory and re-sampling can produce a reconstruction whose structure has visibly shifted — a face becomes a different face, text becomes gibberish. This is precisely the problem that **null-text inversion** (Mokady et al. 2023) was invented to solve: it runs the inversion *unguided* to get a clean pivot trajectory, then optimizes the null (unconditional) embedding at each step so that *guided* sampling re-traces that pivot, pinning the reconstruction back to the input. Later methods (exact inversion via fixed-point iteration, EDICT's coupled-latent trick, ReNoise) attack the same linearization error with higher-order corrections. We go deep on that family in [image editing with diffusion](/blog/machine-learning/image-generation/image-editing-with-diffusion); the foundation you need is exactly the deterministic, invertible DDIM trajectory derived here — every one of those methods is a patch on *this* trajectory, not a replacement for it.

## 9. The semantic latent space: interpolation that means something

One more consequence of determinism, and it is the one that feels like magic the first time you see it. Because $\eta = 0$ DDIM is a deterministic, smooth, near-bijective map between the noise $x_T$ and the image $x_0$, the latent space $x_T$ behaves like a *real* latent space — distances and directions in it correspond to smooth, semantically coherent changes in image space. This is emphatically *not* true of stochastic DDPM, where the fresh noise injected at every step washes out any correspondence between the initial $x_T$ and the final image.

The concrete demonstration is **spherical linear interpolation (slerp)** between two latents. Take two noise samples $x_T^{(0)}$ and $x_T^{(1)}$, interpolate them on the sphere (because they live on a high-dimensional Gaussian shell, slerp respects the norm where naive lerp would shrink it toward the center and produce blurry, low-contrast images), and run DDIM on each interpolant:

```python
import torch

def slerp(t, a, b):
    # spherical interpolation; a, b are flattened unit-ish Gaussian latents.
    a_n, b_n = a / a.norm(), b / b.norm()
    omega = torch.acos((a_n * b_n).sum().clamp(-1, 1))
    so = torch.sin(omega)
    if so.abs() < 1e-6:
        return (1 - t) * a + t * b                       # degenerate: fall back to lerp
    return (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b

xT0 = torch.randn(shape, device="cuda")
xT1 = torch.randn(shape, device="cuda")
frames = []
for t in torch.linspace(0, 1, 9):                        # 9-frame morph
    xT = slerp(t, xT0.flatten(), xT1.flatten()).reshape(shape)
    frames.append(ddim_sample(model, betas, shape, n_steps=50, eta=0.0, xT=xT))
# frames is a smooth semantic morph from image 0 to image 1.
```

With DDIM you get a smooth morph — one face turning into another, one bird species into another — where every intermediate is a plausible image. With stochastic DDPM, the "interpolation" is just two unrelated images with noise in between, because the seed barely influences the output. This semantic-latent property is what makes DDIM the substrate for latent-space editing, attribute manipulation, and the consistency-model distillation targets we will see later in the series — they all rely on a *fixed* deterministic trajectory from each noise sample to its image.

Why does slerp specifically matter, and not plain linear interpolation? The latents live in a very high-dimensional space (for SDXL latents, $4\times128\times128 = 65{,}536$ dimensions), and a standard Gaussian in high dimensions concentrates its mass on a thin spherical shell of radius $\sqrt{d}$ — almost no samples are near the origin. Linear interpolation between two points on that shell passes *through* the low-density interior: at the midpoint, $\frac{1}{2}(x^{(0)} + x^{(1)})$ has expected squared norm $\frac{1}{2}d$ rather than $d$, so it is a $\sqrt{2}\times$-too-short vector that the network has essentially never seen. The result is a washed-out, low-contrast image at the middle of a linear morph — the classic "the interpolation goes gray in the middle" artifact. Spherical interpolation keeps every intermediate on the shell (constant norm), so every interpolant is a sample the network recognizes as in-distribution, and the morph stays crisp end to end. This is a small but real detail that trips up people the first time they try latent interpolation; the fix is one function.

There is a deeper consequence worth naming. Because the DDIM map $x_T \mapsto x_0$ is smooth and (approximately) bijective, *directions* in latent space have stable meaning. You can find a "smile direction" by inverting a few smiling and non-smiling faces, taking the difference of their latents, and then *adding* that direction to a new latent before sampling — and the new face will smile. This is the diffusion analogue of the famous "king − man + woman = queen" word-vector arithmetic, and it works *only* because DDIM gives a deterministic, structured latent. Stochastic sampling destroys it: there is no stable correspondence between a latent direction and an image change when fresh noise floods in at every step. Latent-arithmetic editing, attribute sliders, and semantic-direction discovery (e.g. unsupervised methods that find interpretable directions in the DDIM latent) are all downstream of this one property.

## 10. Case studies: DDIM in real systems

Three concrete places DDIM shows up in shipped systems, to make the theory concrete.

**Stable Diffusion 1.5 / 2.x default sampling.** When latent diffusion (Rombach et al. 2022) shipped, DDIM was the workhorse sampler. The SD model trains in VAE latent space at $64\times64\times4$, and the public checkpoints generate well at 50 DDIM steps with CFG ~7.5. The "DDIM" option in every early Web UI and in the original `diffusers` `StableDiffusionPipeline` is exactly the $\eta=0$ sampler derived above, operating on the latent rather than the pixels — the math is identical, only the tensor shape changes. The community later largely moved to DPM-Solver++ and UniPC for the 20–30 step range, but DDIM remained the reference for *correctness* and for inversion-based editing, where its exact-ODE property is required.

**The DDIM-inversion editing family.** SDEdit, Prompt-to-Prompt, Null-text Inversion, and InstructPix2Pix-style methods all stand on DDIM inversion. Prompt-to-Prompt (Hertz et al. 2022) inverts the source image to its latent, then re-runs DDIM with cross-attention maps swapped to change the prompt while preserving structure — impossible without a deterministic, invertible trajectory. Null-text inversion exists *specifically* because the plain inversion derived in section 8 drifts under CFG; it is a patch on DDIM inversion, not a replacement for it. Every one of these methods would be impossible with a stochastic sampler.

**Distillation targets.** Consistency models (Song et al. 2023) and the LCM/LCM-LoRA line define their training target as the *DDIM ODE trajectory*: a consistency model is trained so that every point along a deterministic DDIM trajectory maps to the same endpoint. Progressive distillation (Salimans & Ho 2022) similarly distills a DDIM teacher into a student that takes half the steps, halving again and again down to 1–4 steps. None of these few-step methods would have a well-defined target without the deterministic trajectory DDIM provides. We cover them in the speed-and-distillation track; the connective tissue is that DDIM turned sampling into a smooth ODE, and you can only distill a smooth, deterministic map.

**The training-loop FID monitor.** A mundane but ubiquitous use: every well-run diffusion training loop computes FID periodically to track progress, and essentially none of them use 1000-step ancestral sampling to do it because, as the worked example in section 1 showed, that would dominate the training budget. The standard recipe is a fixed-seed DDIM sampler at 50 (or even 25) steps on a held-out set of a few thousand samples, run every $N$ epochs. The *absolute* FID from a 50-step DDIM monitor is a hair higher than the true 1000-step number, but the *trend* — is the model improving? — is what you care about, and it tracks faithfully at a fraction of the cost. The determinism matters here too: a fixed seed means the FID wiggle between checkpoints reflects the *model* changing, not the *noise* changing, which makes the curve far less noisy and easier to read. If you have ever wondered why your training script samples with DDIM and a fixed generator, this is why.

A useful way to see the through-line across all four: DDIM is rarely the *final* sampler in a shipped 2025-era product (that is usually DPM-Solver++, UniPC, or a distilled few-step model), but it is almost always *somewhere* in the system — as the reference for correctness, the substrate for editing and inversion, the trajectory definition for distillation, or the cheap monitor during training. It is the lingua franca of deterministic diffusion sampling. Understanding it is not optional even in a world that has moved past it for raw speed, because everything that moved past it is defined relative to it.

## 11. Where DDIM plateaus, and the honest trade-offs

DDIM is not the end of the story; it is the first chapter of fast sampling, and it has a clear ceiling. Being precise about where it stops being the right tool is the most useful thing this section can do.

DDIM is a **first-order** ODE solver. Its global integration error scales as $O(\Delta\sigma)$ — linearly in the step size. That is why it plateaus: above ~50 steps the Euler error is below the network error and quality is solver-limited; below ~20 steps the Euler error dominates and FID climbs. The fix is *not* a different training run; it is a *better solver* on the same ODE. DPM-Solver++ (Lu et al. 2022) and UniPC (Zhao et al. 2023) exploit the semi-linear structure of the diffusion ODE — the fact that part of it is an exactly-solvable linear term — to get second- and third-order accuracy, hitting DDIM-50 quality in 10–20 steps. The progression DDIM → DPM-Solver++ → UniPC is the subject of [the samplers deep-dive](/blog/machine-learning/image-generation/samplers-deep-dive); the takeaway here is that DDIM defined the object (an ODE) that all of them solve better.

There is also a real, often-overlooked quality argument *for* stochasticity. At high step budgets, the SDE (stochastic) samplers can produce marginally better FID than the ODE (deterministic) ones — the injected noise acts as a kind of error-correction, nudging the trajectory back toward the data manifold at each step and washing out small network errors. This is why the DDPM 1000-step FID (3.17) beats DDIM 1000-step (4.04) in the table. So the honest rule is: *deterministic for speed and editing; a touch of stochasticity ($\eta$ around 0.1–0.3, or a proper SDE sampler) when you have the step budget and want the last bit of fidelity or extra diversity.*

| Situation | Reach for | Why |
|---|---|---|
| Training-time FID eval | DDIM, 50 steps, $\eta=0$ | $20\times$ cheaper than DDPM; FID barely moves |
| Interactive / served generation | DDIM 25–40 or DPM-Solver++ 15–25 | Latency-bound; deterministic; reproducible |
| Absolute best FID, compute no object | DDPM/SDE 1000 steps | Stochasticity buys the last fidelity point |
| Real-image editing (P2P, null-text) | DDIM inversion, $\eta=0$, 50+ steps | Needs an invertible deterministic trajectory |
| Latent interpolation / morphing | DDIM, $\eta=0$ | Only deterministic samplers have a semantic latent |
| Sub-10-step generation | DPM-Solver++/UniPC, or a distilled model | DDIM's first-order error dominates below ~20 |
| Want extra sample diversity | $\eta \in [0.1, 0.3]$ or full SDE | Controlled stochasticity widens the output set |

The one trap to avoid: do not run DDPM at low step counts. The 50-step DDPM FID of 29.0 in the table is the cautionary number — a stochastic sampler starved of steps gives you the worst of both worlds, slow *and* low-quality relative to DDIM at the same budget. If you are cutting steps, you must switch to a deterministic or higher-order sampler; cutting steps on the ancestral sampler is the single most common "why are my samples suddenly garbage" bug.

### Stress-testing the sampler

Let me push DDIM until it breaks, because knowing the failure modes is what separates "I read the paper" from "I have shipped this."

**What happens at 4 steps?** DDIM at 4 steps is genuinely bad — FID in the tens, with smeared, low-detail outputs. The first-order Euler error over a 4-step grid is enormous because the trajectory's curvature at high noise levels is completely unresolved. This is *not* fixable by tuning $\eta$ or the subsequence; it is a fundamental limit of a first-order solver. The honest answer at 4 steps is "use a distilled model" (LCM, Turbo, DMD) or a higher-order solver with a good initialization. DDIM was never meant to live below ~15–20 steps; pretending it can is how people conclude "diffusion is slow" when the real problem is they picked the wrong solver for their budget.

**What happens when the schedule is mismatched?** The single nastiest DDIM bug. If your `DDIMScheduler` was constructed with a different `beta_start`/`beta_end`/`num_train_timesteps` than the network trained on, the $\bar\alpha_t$ values are wrong, the predicted-$x_0$ reparameterization divides by the wrong $\sqrt{\bar\alpha_t}$, and you get either washed-out gray images (schedule too aggressive) or noisy ones (schedule too gentle). This is why `DDIMScheduler.from_config(pipe.scheduler.config)` matters — it copies the training schedule. Symptom-to-cause: if a scheduler swap suddenly degrades quality, check the schedule config first, before touching steps or guidance.

**What happens with zero-terminal-SNR models?** Some modern training recipes (and the SD2.x "v-prediction" checkpoints) rescale the schedule so that $\bar\alpha_T = 0$ exactly — the terminal step is *pure* noise with no residual signal. Plain DDIM assumes $x_T$ has a tiny bit of signal and can produce washed-out, low-contrast images on such models (the "the model can't generate pure black or pure white" bug). The fix is a schedule rescale and a v-prediction-aware $\hat x_0$ formula (next paragraph). If your outputs look low-contrast and slightly gray no matter the prompt, suspect terminal-SNR.

**What about v-prediction networks?** Not every modern model predicts $\epsilon$. The v-prediction parameterization (Salimans & Ho 2022) trains the network to output $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$, which is numerically better-behaved at high noise. DDIM works identically — you just recover $\hat x_0$ and $\hat\epsilon$ from $v$ via $\hat x_0 = \sqrt{\bar\alpha_t}\,x_t - \sqrt{1-\bar\alpha_t}\,v$ and $\hat\epsilon = \sqrt{\bar\alpha_t}\,v + \sqrt{1-\bar\alpha_t}\,x_t$, then run the same update. The `diffusers` `DDIMScheduler` handles this when `prediction_type="v_prediction"` is set in the config; the math is the same ODE, just a different parameterization of the velocity. We unpack the parameterization zoo in [the noise-schedules post](/blog/machine-learning/image-generation/noise-schedules-and-the-parameterization-zoo).

## 12. When to reach for DDIM (and when not to)

A decisive recommendation, since every technique is a cost. The decision tree below encodes it; the prose makes it actionable.

![A decision tree for choosing a sampler by step budget with DDPM ancestral for high budgets DDIM deterministic for the mid range and higher order solvers for very low budgets](/imgs/blogs/ddim-and-fast-deterministic-sampling-8.png)

**Reach for DDIM when:** you want a fast, deterministic, reproducible sampler in the 20–100 step range on a network you already have; you need inversion for editing; you want a semantic latent space for interpolation; you are distilling and need a deterministic trajectory as the target; or you are evaluating during training and cannot afford 1000-step ancestral sampling. For the overwhelming majority of practical diffusion work, DDIM at 30–50 steps is the correct default, and it is *free* — no retraining, one scheduler swap.

**Do not reach for DDIM when:** you are chasing the absolute best FID with unlimited compute (use a high-step SDE sampler — the stochasticity buys you the last fidelity point); you are sampling below ~15–20 steps (use DPM-Solver++/UniPC, whose higher order tracks the ODE with fewer evaluations, or a distilled few-step model); or you specifically need maximal output diversity and have the step budget to pay for stochasticity. And never use DDPM ancestral sampling at low step counts — that is the worst quadrant on the chart.

The meta-point that ties back to the [series spine](/blog/machine-learning/image-generation/why-generating-images-is-hard): DDIM is the first and cleanest move on the *speed* axis of the generative trilemma, and it makes that move almost for free because it operates on a frozen network at inference time. Quality and diversity barely budge; speed jumps $20\times$. Everything that came after on the speed axis — higher-order solvers, consistency models, distillation, one-step generators — builds on the realization DDIM crystallized: that diffusion sampling is the numerical integration of a smooth ODE, and smooth ODEs are cheap to follow. When you assemble the full pipeline in [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack), the sampler slot will almost always hold a deterministic ODE solver, and DDIM is the one you should understand first because it is the simplest correct one.

## Key takeaways

- **DDIM needs no retraining.** It defines a non-Markovian forward process with the *same* marginals $q(x_t \mid x_0)$ as DDPM, so the same trained $\epsilon_\theta$ network applies directly. The marginal-preservation proof is a one-line variance cancellation.
- **The update has three terms:** predicted $x_0$ (re-noised to the next level), a direction term pointing to $x_t$, and an optional stochastic term scaled by $\sigma_t$. The predicted-$x_0$ reparameterization $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta)/\sqrt{\bar\alpha_t}$ is the key reparameterization.
- **The $\sigma_t$ knob ($\eta$) interpolates a whole family:** $\eta=1$ is DDPM ancestral, $\eta=0$ is deterministic DDIM, in between is partial stochasticity.
- **$\eta=0$ lets you skip steps** because the deterministic update is an Euler step on a *smooth* ODE — no injected noise to accumulate, and smooth curves need few large steps. You run the update over a subsequence of timesteps, not all $T$.
- **DDIM is Euler's method on the probability-flow ODE,** the same ODE from the score-SDE framework. This is why it generalizes to higher-order solvers (DPM-Solver++, UniPC) that hit the same quality in fewer steps.
- **The numbers:** DDIM holds ~4.1 FID on CIFAR-10 from 1000 down to 100 steps and ~4.67 at 50 steps, while DDPM ancestral collapses to 29.0 at 50 steps. At 50 steps DDIM is ~6× lower FID and ~20× faster.
- **Determinism unlocks inversion and semantic latents.** The $\eta=0$ sampler is invertible (image→latent→image, MSE ~$10^{-3}$ at 100 steps), enabling editing; and its smooth bijection gives a latent space where slerp produces coherent morphs.
- **DDIM plateaus around 20–50 steps** because it is first-order; below that, switch to higher-order solvers or distilled models. Never cut steps on the DDPM ancestral sampler.

## Further reading

- **Song, Meng & Ermon (2021), "Denoising Diffusion Implicit Models," ICLR.** The DDIM paper — the non-Markovian forward, the update rule, the $\eta$ family, the ODE view, the inversion. The source for every result here.
- **Ho, Jain & Abbeel (2020), "Denoising Diffusion Probabilistic Models," NeurIPS.** The DDPM that DDIM accelerates; the marginal $q(x_t \mid x_0)$ and $\mathcal{L}_\text{simple}$ that DDIM reuses. See also [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm).
- **Song et al. (2021), "Score-Based Generative Modeling through SDEs," ICLR.** The SDE/probability-flow-ODE framework that DDIM is a first-order discretization of. See [score-based models and the SDE view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view).
- **Lu et al. (2022), "DPM-Solver++"** and **Zhao et al. (2023), "UniPC."** The higher-order ODE solvers that pick up where DDIM plateaus; covered in [the samplers deep-dive](/blog/machine-learning/image-generation/samplers-deep-dive).
- **Mokady et al. (2023), "Null-text Inversion for Editing Real Images,"** and **Hertz et al. (2022), "Prompt-to-Prompt."** The editing methods built on DDIM inversion; the subject of [image editing with diffusion](/blog/machine-learning/image-generation/image-editing-with-diffusion).
- **🤗 `diffusers` `DDIMScheduler` documentation.** The production implementation — the `eta`, `num_inference_steps`, and `from_config` semantics used above.
