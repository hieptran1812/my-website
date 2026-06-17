---
title: "Consistency Models and Few-Step Generation: From 50 Steps to 4"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn the consistency property that collapses a 50-step diffusion sampler into a single jump, then distill SDXL into a 4-step LCM and a portable LCM-LoRA you can snap onto any checkpoint."
tags:
  [
    "image-generation",
    "diffusion-models",
    "consistency-models",
    "latent-consistency-models",
    "distillation",
    "few-step-sampling",
    "lcm-lora",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/consistency-models-and-few-step-generation-1.png"
---

You type a prompt, hit generate, and wait. On a good GPU with a tuned sampler, a 1024×1024 SDXL image takes maybe two and a half seconds — twenty-five denoiser forwards, each one a full pass through a 2.6-billion-parameter U-Net, each one doubled because classifier-free guidance runs the conditional and unconditional branches together. Two and a half seconds is fine for a hobby. It is a disaster for a product where a user drags a slider and expects the image to move with their hand, or a server that has to fan out a thousand requests a second, or a phone that has neither the memory nor the patience for fifty network calls.

The obvious question is: why fifty? The forward diffusion process turns an image into noise over a thousand tiny steps, and the classical sampler walks that path backward, one cautious step at a time, because each step's denoiser estimate is only locally accurate. Fast solvers like DPM-Solver++ and UniPC (covered in the [samplers deep dive](/blog/machine-learning/image-generation/samplers-deep-dive)) get you down to roughly ten to twenty steps by being smarter about the numerical integration. But there is a floor. A solver is still solving an ordinary differential equation, and an ODE solver's error grows as you take fewer, bigger steps. Below about eight steps the images turn to blur and then to mush. We met this wall in the sibling post on [why diffusion is slow and how to fix it](/blog/machine-learning/image-generation/why-diffusion-is-slow-and-how-to-fix-it): the iterative sampling loop is the dominant cost, and shrinking the step count is the single most powerful lever you have.

This post is about a different idea — one that does not try to solve the ODE faster but instead *learns to skip it entirely*. Consistency models (Song, Dhariwal, Chen, and Sutskever, 2023) train a network that takes any noisy point on a sampling trajectory and maps it **directly** to the trajectory's clean origin in one shot. No integration, no fifty steps — one function call. The latent-consistency family (Luo et al., 2023) brought the idea into Stable Diffusion's latent space and into a portable LoRA you can fuse onto any checkpoint, and it is the reason you can now get a usable SDXL image in four steps and roughly four hundred milliseconds. Figure 1 shows the core picture: where a sampler painstakingly walks the probability-flow ODE, the consistency map jumps from anywhere on that curve straight to the answer.

![A graph showing a probability-flow ODE trajectory from noise, with two points on it both mapped by a single consistency function to the same clean origin x_0](/imgs/blogs/consistency-models-and-few-step-generation-1.png)

By the end you will understand the consistency property and *why* it yields a one-step map, the self-consistency objective and the boundary condition that pin it down, the skip-connection parameterization that enforces that boundary for free, the difference between consistency distillation (with a teacher) and consistency training (from scratch), how latent-consistency models and the CFG-augmented ODE move all of this into Stable Diffusion, how LCM-LoRA turns the acceleration into a portable adapter, and how multistep sampling buys back the quality you lose in a single step. You will also see real code — a `diffusers` LCM pipeline at four steps, an LCM-LoRA fuse on SDXL, and a consistency-distillation loss in PyTorch — plus an honest accounting of the quality and diversity you trade away. We keep tying back to the series' spine: the generative trilemma of quality × diversity × speed, where consistency models buy enormous speed and pay, measurably, in diversity.

## 1. The wall: why fast solvers still need a handful of steps

Let me set the stage with the object every fast sampler is secretly working on. Score-based diffusion (the [SDE view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view)) gives us a *probability-flow ODE* — a deterministic trajectory that carries a noise sample $x_T$ down to a data sample $x_0$ while passing through exactly the same sequence of marginal distributions as the stochastic reverse process. In the variance-exploding parameterization the model is conditioned on a noise level $\sigma$ rather than a discrete timestep, and the ODE reads

$$
\frac{dx}{d\sigma} = \frac{x - D_\theta(x, \sigma)}{\sigma},
$$

where $D_\theta(x,\sigma)$ is the denoiser's estimate of the clean image given the noisy input $x$ at noise level $\sigma$. The trajectory is the curve $\{x_\sigma\}$ that satisfies this from $\sigma = \sigma_\text{max}$ (essentially pure noise) down to $\sigma = 0$ (the clean image). DDIM, which we covered in [DDIM and fast deterministic sampling](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling), is the discretization of exactly this ODE.

The classical sampler integrates this ODE numerically. Euler's method takes a step $x \leftarrow x + (d x/d\sigma)\,\Delta\sigma$; higher-order methods like Heun or the multistep DPM-Solver use clever curvature corrections to take bigger steps with less error. But the fundamental scaling law of any solver still bites. For a $p$-th order method over a fixed integration interval split into $N$ steps, the *global truncation error* scales as

$$
\text{error} \sim C \left(\frac{1}{N}\right)^{p},
$$

where $C$ is a constant set by the magnitude of the trajectory's curvature (more precisely, by the $(p{+}1)$-th derivative of the solution). The consequences are stark when you read them off this formula. A first-order method ($p=1$, Euler/DDIM): halve $N$ and error doubles. A second-order method ($p=2$, Heun/DPM++): halve $N$ and error *quadruples*. Higher order helps in the regime where the steps are small enough that the Taylor expansion the order argument relies on is valid — but at four or two steps the steps are so big that the local expansion is meaningless, the error constant $C$ dominates, and the order barely matters. That is the deep reason a 2nd-order solver at 4 steps is not much better than a 1st-order one: you are outside the regime where order buys anything. There is real curvature in the diffusion ODE — the denoiser's estimate changes a lot as you cross the noise levels where the image's coarse structure forms — and once your steps are big enough to overshoot that curvature, the sample lands off the data manifold and you get the gray mush every practitioner has seen at four-step DDIM. No amount of solver cleverness fixes a problem that is fundamentally "too few function evaluations to resolve the curve." The only escape is to stop solving the ODE step by step and learn the endpoint directly — which is the consistency move.

#### Worked example: where the solver wall sits

Take SDXL on an RTX 4090. With DPM-Solver++ (a 2nd-order multistep solver) you get clean, detailed 1024×1024 images at about 20–25 steps. Drop to 15 and a trained eye starts to notice softness in fine texture; drop to 10 and it is obvious; drop to 6 and the global composition is roughly right but the details are smeared; drop to 4 and the image is a blurry impression of what you asked for. Each denoiser forward with CFG enabled costs roughly two network passes (conditional and unconditional), so 25 steps is about 50 forwards at maybe 24 ms each, which is the ~2.4 s headline. The arithmetic is unforgiving: latency is essentially *linear in steps*, so the only way to a sub-second image is to genuinely need fewer steps — not to compute the same steps faster.

This is the wall consistency models walk through. They do not make the solver better. They replace it.

## 2. The consistency property: one function to the origin

Here is the central move. Fix a single probability-flow ODE trajectory — one curve from noise to a particular clean image. Every point on that curve, $x_\sigma$ for any $\sigma$, *belongs to the same image*. If you could integrate the ODE perfectly from any of those points, you would land on the same $x_0$. So define a function $f(x, \sigma)$ whose job is to do exactly that: map any point on the trajectory directly to the trajectory's origin. The defining requirement is **self-consistency**:

$$
f(x_\sigma, \sigma) = f(x_{\sigma'}, \sigma') \quad \text{for all } \sigma, \sigma' \text{ on the same trajectory.}
$$

Read it slowly. Two different points, at two different noise levels, on the *same* ODE path, must be sent by $f$ to the *same* output. The output is, by definition, the clean image $x_0$ at the end of that path. The function is constant along each trajectory. That is the whole idea, and it is why $f$ is called a consistency function — it is consistent across the trajectory.

There is one more constraint that anchors the whole thing, the **boundary condition**:

$$
f(x_0, 0) = x_0.
$$

At the very end of the trajectory, where the noise level is zero and the point already *is* the clean image, the function must return it unchanged. Without this anchor, self-consistency alone is degenerate: a function that maps every point on every trajectory to the constant zero vector is perfectly self-consistent and perfectly useless. The boundary condition forces $f$ to actually land on the right image, because it pins the trajectory's endpoint to the true $x_0$ and self-consistency then propagates that correct value backward to every other point on the curve.

Now the payoff. If we have such an $f$, sampling is trivial. Draw $x_{\sigma_\text{max}} \sim \mathcal{N}(0, \sigma_\text{max}^2 I)$ — pure noise at the top of the trajectory — and compute

$$
\hat{x}_0 = f(x_{\sigma_\text{max}}, \sigma_\text{max}).
$$

One forward pass. No integration. The function *is* the solver, learned end to end. This is the sense in which the consistency property "collapses the ODE into one jump": the model has internalized the entire trajectory so that evaluating it once is equivalent to solving the ODE all the way down. Figure 1 (above) is exactly this collapse — the noisy point and an earlier point on the trajectory both land, via $f$, on the identical origin.

It is worth being precise about *why* this works and is not magic. The ODE trajectory is a deterministic, invertible map between the noise marginal and the data marginal. A consistency model learns the *terminal value* of that map as a function of position along the curve. Because the ODE is deterministic, the terminal value is well defined from any starting point — there is exactly one origin per trajectory — so the target $f$ is trying to fit is a genuine function, not a relation. The hard part is purely learning: you have to teach the network to be consistent across the trajectory without ever simulating the whole trajectory at training time, which is what the next two sections solve.

There is a tighter way to see why the one-step map is even *possible* to learn, and it is worth a paragraph because it dispels the "this is too good to be true" reflex. Define the ground-truth consistency function $f^\star(x, \sigma)$ as: integrate the probability-flow ODE from $(x, \sigma)$ down to $\sigma = 0$ and return the endpoint. This $f^\star$ trivially satisfies both the boundary condition (integrating from $\sigma=0$ returns the input) and self-consistency (two points on the same trajectory integrate to the same endpoint, because the ODE solution is unique). So a perfect consistency function *exists* — it is the exact ODE flow map — and the only question is whether a neural network can approximate it. The self-consistency loss is precisely a way to fit $f^\star$ *without* computing it: instead of integrating all the way down for every training point (expensive), you only require that $f_\theta$ agree on infinitesimally adjacent points, and the boundary condition supplies the anchor. If $f_\theta$ matches $f^\star$ at $\sigma=0$ (boundary) and has zero "consistency error" between every adjacent pair (the loss), then by induction along the trajectory it equals $f^\star$ everywhere. That induction — local agreement plus a correct anchor implies global correctness — is the mathematical heart of the method, and it is why a *local* loss can teach a *global* map.

One more subtlety that trips people up: the consistency function is a map from the *noisy* input, not from a latent code. It is not an encoder-decoder in the VAE sense; there is no separate bottleneck. The network sees a noisy image (or latent) and a noise level, and it outputs a clean image (or latent). The "compression" of the fifty-step trajectory lives entirely in the weights — the network has memorized, in a smooth interpolating way, where each trajectory goes. This is also why consistency models are sometimes described as learning the *flow map* of the diffusion ODE rather than its *vector field*: a normal diffusion model learns the local derivative $D_\theta(x, \sigma)$ that a solver then integrates, while a consistency model learns the integrated result directly. The solver is amortized into the network.

## 3. The skip-connection parameterization: boundary condition for free

How do you make a neural network satisfy $f(x_0, 0) = x_0$ exactly? You could add a penalty term to the loss and hope, but "hope" is not how you build something that has to be exactly right at the boundary. The clean trick — borrowed straight from Karras et al.'s EDM preconditioning — is to *parameterize* the model so the boundary holds by construction. Write the consistency function as a noise-level-dependent blend of the input itself and a learned residual:

$$
f_\theta(x, \sigma) = c_\text{skip}(\sigma)\, x + c_\text{out}(\sigma)\, F_\theta(x, \sigma),
$$

where $F_\theta$ is the raw network (your U-Net or DiT), and $c_\text{skip}, c_\text{out}$ are scalar functions of the noise level. The "skip" term $c_\text{skip}(\sigma)\,x$ is a direct connection from the input to the output — the network's job is only to produce the correction $F_\theta$ that the blend adds on top. Figure 2 shows this skip-blend structure and how it pins the boundary.

![A stack diagram showing the input noisy latent feeding both a skip connection and the network F, blended by c_skip and c_out, with the t equals zero case forcing the identity](/imgs/blogs/consistency-models-and-few-step-generation-3.png)

The boundary condition becomes a constraint on the two coefficients. We need $f_\theta(x, \sigma) \to x$ as $\sigma \to 0$, which holds for *any* network output if

$$
c_\text{skip}(0) = 1, \qquad c_\text{out}(0) = 0.
$$

At the origin the skip term passes the input through unchanged and the learned correction is multiplied by zero, so $f_\theta(x_0, 0) = 1 \cdot x_0 + 0 \cdot F_\theta = x_0$ — exactly, regardless of what the network learned. The boundary is satisfied by construction; the optimizer never has to fight for it. In the EDM/consistency parameterization the standard differentiable choice (with a small reference noise level $\sigma_\text{data}$ and a minimum noise $\sigma_\text{min}$) is

$$
c_\text{skip}(\sigma) = \frac{\sigma_\text{data}^2}{(\sigma - \sigma_\text{min})^2 + \sigma_\text{data}^2}, \qquad
c_\text{out}(\sigma) = \frac{\sigma_\text{data}\,(\sigma - \sigma_\text{min})}{\sqrt{\sigma_\text{data}^2 + \sigma^2}},
$$

which you can check satisfies $c_\text{skip}(\sigma_\text{min}) = 1$ and $c_\text{out}(\sigma_\text{min}) = 0$. The exact algebra matters less than the structure: a noise-weighted blend where, at the bottom of the trajectory, the input dominates and the correction vanishes.

This parameterization does a second, subtler job. At high noise levels $c_\text{skip}$ shrinks and $c_\text{out}$ grows, so the network's correction does most of the work — which is right, because a point that is almost pure noise needs a big jump to reach the image. At low noise levels the input is already close to the answer, so the skip term carries most of the signal and the network only nudges. The blend automatically scales how much "denoising" the network is responsible for to match how far the point is from the origin. This is the same insight that makes v-prediction and EDM preconditioning stabilize ordinary diffusion training (see [noise schedules and the parameterization zoo](/blog/machine-learning/image-generation/noise-schedules-and-the-parameterization-zoo)); here it is load-bearing because the boundary condition is the only thing standing between you and the degenerate constant solution.

A useful sanity check on the parameterization: it also keeps the network's *output magnitude* well-conditioned across noise levels. Without the $c_\text{out}$ scaling, the raw network would have to produce a very large correction at high noise (to bridge the big gap to the image) and a near-zero one at low noise, spanning orders of magnitude — a brutal regression target. The $c_\text{out}(\sigma)$ factor absorbs that scale, so $F_\theta$ outputs a roughly unit-variance signal at every noise level, which is exactly what neural networks train well on. This is not incidental tidiness; it is part of *why* a single network can learn the flow map across the entire trajectory at all. A network asked to output wildly different magnitudes at different inputs tends to underfit the extremes; the preconditioning normalizes the target so every noise level is equally learnable. When people port consistency models to new domains and skip the preconditioning, this is usually where it falls apart — the boundary "works" via a penalty, but the unnormalized target makes high-noise jumps train poorly, and the one-step samples are mush exactly where the network had to make the biggest correction.

## 4. Consistency distillation: borrow a teacher's trajectory

Now the training. We want self-consistency — $f$ should give the same answer at adjacent points on the same trajectory — but we cannot afford to integrate full trajectories during training. The fix is to enforce consistency *locally*, between a point and its immediate neighbor on the trajectory, and let that local constraint propagate. If $f$ agrees on every adjacent pair along the curve, it agrees on the whole curve.

To get a neighbor we need to know which way the trajectory goes from a given point — and that is exactly what a pretrained diffusion model tells us. **Consistency distillation** uses a frozen, pretrained diffusion model as a *teacher* that supplies the local direction of the ODE. The recipe per training step:

1. Sample a clean image $x_0$ from the data and a noise level $\sigma_n$ from a discretized schedule $\sigma_\text{min} = \sigma_1 < \sigma_2 < \dots < \sigma_N = \sigma_\text{max}$.
2. Add noise to get $x_{\sigma_n} = x_0 + \sigma_n \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$ — a point on the trajectory through $x_0$.
3. Use the teacher to take **one ODE solver step** from $\sigma_n$ to the adjacent, lower level $\sigma_{n-1}$, producing the neighbor $\hat{x}_{\sigma_{n-1}}$. (One Euler or Heun step with the teacher's denoiser.)
4. Push both points through the consistency model: the *online* student computes $f_\theta(x_{\sigma_n}, \sigma_n)$, and an EMA copy of the weights — the *target network* $f_{\theta^-}$ — computes $f_{\theta^-}(\hat{x}_{\sigma_{n-1}}, \sigma_{n-1})$ with a stop-gradient.
5. Minimize a distance between the two estimates:

$$
\mathcal{L}_\text{CD} = \mathbb{E}\big[\, \lambda(\sigma_n)\, d\big(f_\theta(x_{\sigma_n}, \sigma_n),\; f_{\theta^-}(\hat{x}_{\sigma_{n-1}}, \sigma_{n-1})\big)\big],
$$

where $d$ is a distance like squared $\ell_2$, $\ell_1$, or LPIPS, and $\lambda(\sigma_n)$ is a per-level weight. Figure 3 lays out the loop.

![A graph of the consistency distillation loop showing a noised point fed to a frozen teacher that steps back to a neighbor, the online student and EMA target each producing origin estimates, and a consistency loss between them](/imgs/blogs/consistency-models-and-few-step-generation-5.png)

Two design choices make this stable, and both are worth understanding rather than cargo-culting.

The **EMA target network** is the same trick that stabilizes deep Q-learning and BYOL-style self-distillation. If you regressed the student against *itself* — student at $x_{\sigma_n}$ versus student at the neighbor, both with live gradients — you would chase a moving target and the whole thing could collapse to a constant (the trivial self-consistent solution). By holding the target as a slowly updated exponential moving average $\theta^- \leftarrow \mu \theta^- + (1-\mu)\theta$ with the stop-gradient, you give the student a stable thing to match. The target drifts slowly toward the student, the boundary condition keeps pulling the endpoint to the true $x_0$, and the correct (non-degenerate) consistency function is the only stable fixed point. (Later work, "improved consistency training," found you can sometimes drop the EMA and use the student itself as the target with the right schedule, but the EMA version is what LCM uses and what is easiest to reason about.)

The **teacher only ever takes one step**, so distillation never integrates a full trajectory either — it just needs the local direction. The student's burden is to *globally* connect every local constraint into a single map to the origin, but the supervision is entirely local and cheap. This is why distillation converges fast: it inherits an already-good ODE from the teacher and only has to learn the collapse.

There is one more knob that quietly decides whether distillation succeeds: the **discretization schedule** $N$, the number of noise levels in the grid $\sigma_1 < \dots < \sigma_N$. This sets how far apart adjacent points are, and therefore how big each teacher ODE step is. Make $N$ too small (coarse grid, big steps) and the teacher's single Euler step is inaccurate — the "neighbor" it hands you is off the true trajectory, and the student learns to be consistent with a *wrong* curve. Make $N$ too large (fine grid, tiny steps) and adjacent points are nearly identical, so the consistency constraint is trivially satisfied and carries almost no learning signal — the student barely moves. The original consistency-models work used a fixed, fairly large $N$ for distillation (the teacher is accurate, so fine is fine), while consistency *training* (next section) needed $N$ to *grow* over the course of training: start coarse so the constraint bites, then refine so the final model is accurate. The Karras-style power-law spacing of the $\sigma$ levels — denser near $\sigma_\text{min}$ where the image's fine detail forms — is the standard choice, the same $\rho = 7$ schedule used in EDM sampling. The practical lesson: the discretization grid is not a throwaway hyperparameter; it is the resolution at which you are teaching the trajectory, and getting it wrong shows up as either blur (too coarse) or non-convergence (too fine).

The **EMA decay $\mu$** is the other hyperparameter that earns attention. Too small (target drifts fast, tracking the student closely) and you reintroduce the moving-target instability the EMA was meant to fix; too large (target frozen) and learning stalls because the student is matching a stale teacher that never improves. Values around $0.999$–$0.9999$ are typical, and LCM-scale runs often schedule it — looser early, tighter late — mirroring the discretization-refinement idea. The point worth internalizing is that *both* the EMA decay and the discretization schedule are about the same thing: controlling the gap between the thing the student predicts and the thing it is asked to match, so that gap is small enough to be a stable target but large enough to carry signal.

#### Worked example: why the loss can't just be "match the teacher's image"

A natural but wrong idea: skip consistency, and just train $f_\theta(x_{\sigma_n}, \sigma_n)$ to regress the teacher's *full multi-step denoising result* from $x_{\sigma_n}$. That works, but it requires running the teacher's full sampler for every training example — fifty teacher forwards per student step — which is brutally expensive and is essentially the old "progressive distillation" cost. Consistency distillation's cleverness is that it needs only *one* teacher forward per step (to find the neighbor), because the self-consistency constraint chains the local agreements together for free. You pay one teacher step and a single distance comparison; the global collapse emerges from the EMA fixed point. That asymmetry — one teacher step instead of fifty — is the entire efficiency argument for the method.

It is worth contrasting with **progressive distillation** (Salimans & Ho, 2022), the method consistency models partly displaced, because the comparison sharpens what is new. Progressive distillation halves the step count repeatedly: a teacher that samples in $2K$ steps trains a student to match its output in $K$ steps, then that student becomes the teacher for a $K/2$-step student, and so on down. It works and produces good few-step models, but each halving is a full training run, and the student is always chasing a *fixed multi-step teacher output* — so it inherits the teacher's exact trajectory and cannot do better than a careful interpolation of it. Consistency distillation collapses all of that into a single training run with a local objective, and because it learns the flow map directly rather than imitating $K$-step outputs, it reaches *one* step in one run rather than $\log_2$ runs. The lineage is real — both are distillation, both shrink steps — but the consistency formulation is what made one-step and four-step sampling a single cheap training job instead of a staircase of them.

## 5. Consistency training: the same map, no teacher

What if you do not have a pretrained teacher, or you want to train a consistency model from scratch on a new dataset? **Consistency training** removes the teacher entirely. The only thing the teacher provided was the neighbor point $\hat{x}_{\sigma_{n-1}}$ on the trajectory; consistency training replaces it with an *unbiased estimator* of that neighbor built from the same image and noise.

The key identity: if $x_{\sigma_n} = x_0 + \sigma_n \epsilon$, then the adjacent point on the same trajectory can be written, in expectation, using the *same* image $x_0$ with the *same* noise direction $\epsilon$ at the lower level: $x_{\sigma_{n-1}} = x_0 + \sigma_{n-1}\epsilon$. So you noise the *same* clean image to *two* adjacent levels with one shared $\epsilon$, and you have your pair — no teacher needed. The loss is the same shape:

$$
\mathcal{L}_\text{CT} = \mathbb{E}\big[\, \lambda(\sigma_n)\, d\big(f_\theta(x_0 + \sigma_n \epsilon,\, \sigma_n),\; f_{\theta^-}(x_0 + \sigma_{n-1}\epsilon,\, \sigma_{n-1})\big)\big].
$$

The catch is variance. The teacher's one ODE step gave a *low-variance* neighbor that respects the learned trajectory's curvature; the from-scratch estimator uses the raw noised image, which is a noisier (higher-variance) proxy for the true neighbor, especially at large gaps between adjacent levels. The original paper showed consistency training works but needs a carefully *increasing* schedule of discretization steps $N$ over training — start coarse and refine the grid — plus tuned noise-level distributions, to keep the variance manageable. Figure 4 contrasts the two regimes.

![A before-after diagram contrasting consistency distillation, which needs a pretrained teacher to supply neighbor points and converges fast, with consistency training, which needs no teacher but uses a higher-variance noise estimator and more compute](/imgs/blogs/consistency-models-and-few-step-generation-6.png)

In practice, for image generation, distillation wins almost every time — pretrained diffusion teachers are abundant, and reusing one is far cheaper and more stable than training a consistency model from noise. Consistency *training* matters as a matter of principle (it proves you do not *need* diffusion to get a consistency model — it is a standalone generative family) and it matters when you genuinely have no teacher. For the rest of this post, when we get to real systems like LCM, we are talking about distillation.

It is worth dwelling on *why* the teacher's neighbor has lower variance, because it explains the whole distillation-vs-training quality gap. The teacher's one ODE step computes the neighbor as $x_{\sigma_{n-1}} = x_{\sigma_n} + (\text{teacher derivative}) \cdot \Delta\sigma$, where the derivative uses the teacher's *learned, smooth* estimate of where the trajectory goes. The from-scratch estimator instead uses $x_0 + \sigma_{n-1}\epsilon$ with the *same raw noise* $\epsilon$ — which is correct in expectation but conflates "the trajectory direction" with "this particular noise sample." When the gap $\Delta\sigma$ is small the two agree; when it is large (coarse grid) the raw-noise estimator's neighbor can be far from the true trajectory, injecting variance into the target. The teacher, having already learned to average over noise, hands you a denoised direction with that variance removed. This is the same reason a trained denoiser beats the raw noisy image as an estimate of $x_0$: averaging over the conditional distribution removes variance. Distillation inherits that averaging for free; consistency training has to fight it with schedule tricks. The conceptual payoff is that consistency training reveals consistency models are a *primitive* generative family — you can build one with nothing but data and noise — while distillation is the *practical* path that rides an existing model's learned denoising.

## 6. Multistep sampling: buying back quality with 2–4 steps

A one-step consistency sample is fast but not free of cost. The single jump from pure noise to the image is the hardest thing the model has to do, and a single network is never going to be as good at it as fifty careful steps. The fix is elegant and is the reason you see "4 steps" and not "1 step" in most production LCM configs: **multistep consistency sampling** alternates the one-step jump with a re-noising, a few times, to refine.

The recurrence is short. Start at the top noise level and jump to an origin estimate; then add a controlled amount of noise back to bring it to an *intermediate* level; then jump again from there; repeat for a handful of steps. Concretely, with a chosen sequence of decreasing noise levels $\tau_1 > \tau_2 > \dots > \tau_K$:

1. $x \leftarrow \mathcal{N}(0, \sigma_\text{max}^2 I)$.
2. $\hat{x}_0 \leftarrow f_\theta(x, \sigma_\text{max})$ — the first jump to a (rough) origin estimate.
3. For $k = 1 \dots K-1$:
   - Re-noise: $x \leftarrow \hat{x}_0 + \tau_k\,\epsilon_k$ with fresh $\epsilon_k \sim \mathcal{N}(0, I)$ — push the estimate back up to level $\tau_k$.
   - Re-estimate: $\hat{x}_0 \leftarrow f_\theta(x, \tau_k)$ — jump to a *better* origin estimate from a *lower* noise level.
4. Return $\hat{x}_0$.

Why does this help? Each jump from a lower noise level is an easier problem — the point is closer to the manifold, the model's estimate is sharper, and the error is smaller. The re-noising is what lets you start the next, easier jump from the current estimate rather than from pure noise. After two to four rounds the estimate has been refined from "rough impression" to "crisp image." It is the consistency analogue of taking a few solver steps, except each step is a *full jump to the origin and back*, not a tiny local move. The trade-off is linear and obvious: more steps, more quality, more latency. The sweet spot for latent-consistency models on SDXL is empirically 4 steps; 2 is faster but noticeably softer; 1 is for real-time previews where you will refine later; 8 squeezes out a touch more fidelity for double the cost. Figure 5 puts numbers on it.

![A matrix of step counts one two four and eight against DDIM, DPM-Solver, and LCM showing standard solvers collapse below ten steps while LCM holds usable quality down to one step](/imgs/blogs/consistency-models-and-few-step-generation-4.png)

The pattern in that matrix is the headline result of the whole field: at 4 steps the consistency-distilled model is in the same quality neighborhood as the 25-step baseline, while the standard solvers at 4 steps are unusable. At 1 step LCM is rough but coherent — a recognizable image — where DDIM at 1 step is pure noise. This is not a free lunch; section 9 is honest about what you pay. But as a speed-for-quality trade it is one of the best deals in the diffusion stack.

Why does re-noising specifically help, and not just "evaluate the network more times"? The key is that re-noising moves you *back onto the trajectory manifold at a controlled level*. After the first jump you have $\hat{x}_0$, a rough clean estimate that is not a perfect image — it sits slightly off the data manifold. If you fed $\hat{x}_0$ straight back into $f_\theta$ at a low noise level, the model would see an input it was never trained on (a slightly-wrong clean image, not a properly-noised one) and behave unpredictably. Adding fresh Gaussian noise $\tau_k \epsilon_k$ to $\hat{x}_0$ produces a point that *looks like* a genuine sample at noise level $\tau_k$ — exactly the kind of input $f_\theta$ was trained on — so the next jump is well-posed. The noise also injects a little stochasticity that lets the trajectory correct toward a sharper mode. This is why multistep consistency sampling is, in spirit, a *stochastic* sampler even though each jump is deterministic: the re-noising is the stochastic step, and it is what distinguishes it from naively chaining deterministic jumps (which would just compound the first jump's error). The number of rounds $K$ is your quality dial, and the noise levels $\tau_k$ are usually a few points spaced down from $\sigma_\text{max}$ toward $\sigma_\text{min}$ — `LCMScheduler` picks them for you from `num_inference_steps`.

#### Worked example: walking the 4-step recurrence

Make it concrete with $K=4$. You start at $\sigma_\text{max}$ with pure noise and jump: $\hat{x}_0^{(1)} = f_\theta(x, \sigma_\text{max})$ — a blurry but globally-correct cat. Re-noise to $\tau_1$ (say a high-ish level): $x \leftarrow \hat{x}_0^{(1)} + \tau_1 \epsilon_1$, and jump again: $\hat{x}_0^{(2)} = f_\theta(x, \tau_1)$ — now the cat's pose and color are right, edges firming up. Re-noise to a lower $\tau_2$, jump: $\hat{x}_0^{(3)}$ — fur texture and whiskers appear. Re-noise to a still-lower $\tau_3$, jump: $\hat{x}_0^{(4)}$ — the final crisp image. Each round started from a *lower* noise level than the last, so each jump was an easier, sharper denoising problem, and the error shrank monotonically. Four network calls, no CFG doubling, and you are at teacher-comparable quality. If you watch this happen frame by frame in a live UI it looks like the image "developing" — which is exactly the iterative refinement of a normal sampler, compressed into four big jumps instead of fifty small ones.

## 7. Latent Consistency Models: moving it into Stable Diffusion

Everything so far was about pixel-space (or noise-level) consistency in the abstract. The reason consistency models became a daily tool is **Latent Consistency Models** (LCM, Luo et al., 2023), which made two changes that turned the theory into something that runs SDXL at four steps.

**Change one: work in the VAE latent space.** Stable Diffusion does not diffuse pixels; it diffuses an 8×-downsampled latent produced by a VAE (the foundation of [latent diffusion](/blog/machine-learning/image-generation/latent-diffusion-and-stable-diffusion)). LCM applies consistency distillation directly to a pretrained SD/SDXL model's latent-space ODE, with the SD model as the teacher. The consistency function $f_\theta$ is the SD U-Net (or SDXL's), re-tuned to map any noisy latent straight to the clean latent; the VAE decoder turns that clean latent into the final image, once, at the end. Nothing about the consistency math changes — it is the same self-consistency loss with the same EMA target and skip parameterization — but operating in the compact latent space makes every forward cheap and lets a small distillation run (a few thousand A100-hours, often far less) accelerate a billion-parameter model.

The economics of latent-space distillation deserve a sentence, because they are *why* this happened in latent space and not in pixels. A 1024² image is roughly 3 million numbers; its SDXL latent is 4×128×128, about 65 thousand — a ~48× compression. Every teacher forward, every student forward, every distance computation in the loss runs on the small latent, so a distillation step is ~48× cheaper in the spatial dimension than the pixel-space equivalent. The distillation also reuses the teacher's already-trained weights as the student's initialization — you are not learning a denoiser from scratch, only re-tuning it to be a flow map — so convergence is fast. Stack the compression and the warm start and you get the headline: a model that took thousands of GPU-days to pretrain can be accelerated to four steps in single-digit GPU-days. That cheapness is what made LCM, and then LCM-LoRA, something the open-source community could produce and share rather than a frontier-lab-only capability.

**Change two: the CFG-augmented ODE.** This is the subtle one. Stable Diffusion's quality comes from [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance), which at sampling time extrapolates between the conditional and unconditional predictions:

$$
\hat{\epsilon}_\text{cfg} = \hat{\epsilon}_\text{uncond} + w\,(\hat{\epsilon}_\text{cond} - \hat{\epsilon}_\text{uncond}),
$$

with guidance scale $w$ (typically 7.5). A naive consistency model distilled on the *un-guided* ODE would lose all of CFG's prompt adherence and saturation, because the guided and unguided trajectories are different curves. LCM's fix is to **bake guidance into the trajectory it distills**. It defines an *augmented* probability-flow ODE whose velocity field already includes the CFG extrapolation, treats the guidance scale $w$ as an extra conditioning input fed to the network (so one model can serve a *range* of guidance scales), and distills consistency along *that* guided ODE. The result: the student internalizes guided generation. At inference you therefore use a *low or unit* `guidance_scale` (~1) because the guidance is already inside the weights — running CFG again on top would double-count it and over-saturate. This is the single most common LCM gotcha: people set `guidance_scale=7.5` out of habit and get blown-out, contrasty garbage. With LCM you want roughly 1.0–2.0.

Here is the practical flow in 🤗 `diffusers` — load a pre-distilled LCM-distilled SDXL model and sample in four steps:

```python
import torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler

# A model that was already LCM-distilled from SDXL.
pipe = StableDiffusionXLPipeline.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Swap in the LCM multistep scheduler.
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

image = pipe(
    prompt="a photo of a tabby cat sitting on a windowsill, golden hour, 50mm",
    num_inference_steps=4,      # the whole point: 4, not 25
    guidance_scale=1.0,         # guidance is baked in — keep this low
).images[0]
image.save("cat_lcm_4steps.png")
```

`LCMScheduler` implements exactly the multistep consistency recurrence from section 6 — jump, re-noise, jump — and `num_inference_steps=4` sets $K = 4$. The `guidance_scale=1.0` is the load-bearing change from a normal SDXL call. Figure 6 (above, the matrix) is what you get if you sweep `num_inference_steps`.

#### Worked example: the speedup arithmetic for LCM-SDXL

Baseline SDXL with DPM-Solver++ at 25 steps and CFG 7.5: each step is two forwards (cond + uncond), so 50 U-Net forwards, ~24 ms each on a 4090 → ~1.2 s in the loop plus VAE decode → call it ~2.4 s end to end (the larger figure accounts for attention overhead at 1024² and Python). LCM at 4 steps with `guidance_scale=1.0`: CFG is baked in so each step is a *single* forward → 4 U-Net forwards → ~0.1 s in the loop plus the same one-time VAE decode and text encode → ~0.4 s end to end. That is the ~6× speedup shown below, and notice *two* multipliers stacked: 25→4 steps is ~6× on its own, and dropping the doubled CFG forward is another ~2× per step that partially offsets the larger per-step work. The net on a 4090 is roughly 0.4 s versus 2.4 s.

![A before-after diagram contrasting a 25-step DDIM baseline at 2.4 seconds with a 4-step distilled latent-consistency model at 0.4 seconds, roughly six times faster](/imgs/blogs/consistency-models-and-few-step-generation-2.png)

The before/after above is the trade in a single picture: the left column is the DDIM baseline — 25 sequential denoiser calls, each doubled by CFG, landing at ~2.4 s — and the right column is the distilled LCM — 4 single forwards with guidance already in the weights, landing at ~0.4 s. Both columns produce a comparable image; what changed is the number of times you had to run the network. That is the whole pitch of the method, and it is why a distilled four-step sampler is the first thing reached for when an image pipeline has to feel interactive rather than batch.

## 8. LCM-LoRA: distill the acceleration into a portable adapter

The first LCM models were *full* distilled checkpoints — you downloaded a new 6.9 GB SDXL whose weights had been re-tuned for four-step sampling. That is fine for the base model, but the entire ecosystem runs on *fine-tuned* checkpoints: a thousand community SDXL variants, anime models, photoreal models, your own DreamBooth of a product. Re-distilling each one is absurd. LCM-LoRA (Luo et al., 2023, the follow-up) solves this beautifully.

The observation is that the *difference* between a standard SD checkpoint and its LCM-distilled version is a low-rank update — it is an acceleration *direction* in weight space, and it is largely *independent of the specific fine-tune*. So instead of distilling the full weights, you distill the consistency objective into a **LoRA** (low-rank adapter, the same mechanism covered in [personalization with DreamBooth, Textual Inversion, and LoRA](/blog/machine-learning/image-generation/personalization-dreambooth-textual-inversion-lora)): a small set of rank-decomposed matrices added to the attention layers, on the order of 67M parameters for SDXL versus 2.6B in the base. You train this LoRA once, against the base SDXL teacher, and the result is a *portable acceleration module*. Figure 7 shows the plug-and-play stack.

![A stack diagram showing any SDXL checkpoint plus an LCM-LoRA adapter fused into the weights, with the LCM scheduler at four steps producing a fast image](/imgs/blogs/consistency-models-and-few-step-generation-7.png)

Now the magic: load *any* compatible SDXL checkpoint — the base, a photoreal fine-tune, your custom LoRA-stacked model — fuse the LCM-LoRA on top, switch to `LCMScheduler`, and that checkpoint becomes a four-step sampler. One adapter, every checkpoint. Here is the flow:

```python
import torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler

# Start from ANY SDXL checkpoint — base, or your favorite fine-tune.
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# Load the portable LCM-LoRA and fuse it into the weights.
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.fuse_lora()

image = pipe(
    prompt="an astronaut riding a horse on mars, cinematic, dramatic light",
    num_inference_steps=4,
    guidance_scale=1.0,
).images[0]
image.save("astronaut_lcm_lora_4steps.png")
```

The `fuse_lora()` call folds the low-rank update into the base weights so there is no per-forward adapter overhead — you pay the LoRA cost once, at load. You can also *stack* an LCM-LoRA with a *style* LoRA (use `set_adapters` with weights and skip the fuse) to get a four-step sampler of your styled model, which is how a lot of real-time ComfyUI workflows are built. The portability is the product: the community trained the acceleration once and the entire SDXL zoo got four-step sampling for free.

#### Worked example: LCM-LoRA on a custom fine-tune

You have a DreamBooth SDXL of a specific sneaker, sampled at 30 steps / CFG 6 for product shots, ~3.0 s each on a 4090. You want a live preview slider. Re-distilling a full LCM of your sneaker model would cost a real training run. Instead you `load_lora_weights("latent-consistency/lcm-lora-sdxl")` on your existing fine-tune, set `num_inference_steps=4`, `guidance_scale=1.5`, and you are at ~0.4 s — a ~7–8× speedup — with the sneaker identity preserved because your DreamBooth weights are untouched; the LCM-LoRA only changed *how many steps* the sampler needs, not *what* the model draws. The previews are slightly softer than your 30-step product renders, so you keep the slow path for the final hero shot and the fast path for the interactive browse. That split — fast for interaction, slow for the final frame — is the standard way to deploy these.

### Stacking LCM-LoRA with a style LoRA, and ComfyUI

The fuse-on flow above assumes the LCM-LoRA is the only adapter. In practice you often want to keep a *style* LoRA active too — your model already has a watercolor or product-photography LoRA you do not want to lose. In `diffusers` you load both and balance them with `set_adapters` instead of fusing, so the acceleration and the style compose at sampling time:

```python
# Keep a style LoRA AND the acceleration LoRA live, weighted.
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
pipe.load_lora_weights("path/to/your-style-lora", adapter_name="style")
pipe.set_adapters(["lcm", "style"], adapter_weights=[1.0, 0.8])

image = pipe(
    prompt="a watercolor of a fox in a snowy forest",
    num_inference_steps=4,
    guidance_scale=1.5,
).images[0]
```

The LCM adapter weight near 1.0 keeps the four-step acceleration intact; the style weight is your usual style dial. Push the LCM weight much below 1.0 and the few-step quality degrades — the acceleration is not fully applied — so treat 1.0 as the floor and tune the *style* weight instead. In **ComfyUI**, the same composition is a graph: a `Load LoRA` node for the LCM-LoRA feeding the model input of your `KSampler`, the sampler set to the `lcm` sampler with `cfg` near 1, `steps` at 4, and a normal scheduler. The node-graph form is what most real-time and live-paint workflows use, precisely because you can hot-swap the base checkpoint behind a fixed LCM-LoRA node and keep the four-step behavior. The portability that makes LCM-LoRA a downloadable file is exactly what makes it a single reusable node.

A final practical note on **which step the LoRA accelerates**. LCM-LoRA is trained against a *specific base family* — there is an SD-1.5 LCM-LoRA and an SDXL LCM-LoRA, and they are not interchangeable, because the underlying ODE and latent shapes differ. Fuse the SDXL LCM-LoRA onto an SDXL checkpoint, the SD-1.5 one onto SD-1.5. Mixing them produces noise. The rank-64 / ~67M-parameter SDXL adapter is the common one; the SD-1.5 adapter is smaller still. This is the one compatibility rule that bites people: portability is *within* a base family, not across architectures.

## 9. The honest trade-off: diversity and the few-step quality ceiling

Now the part the marketing leaves out. Few-step consistency sampling buys speed with two real costs, and you should name them before you ship.

**Diversity loss.** This is the big one. A one-step (and to a lesser degree four-step) consistency model produces noticeably *less diverse* outputs than the multi-step teacher for the same prompt across different seeds. The mechanism is the same one that makes high CFG less diverse: the model has learned a sharp, mode-seeking map to high-probability images, and the baked-in guidance pushes it further toward the prompt-typical mode. The probability-flow ODE the teacher integrates can, over many small steps, wander to the less-typical parts of the distribution; the one-shot map tends to snap to the center of the mode. In FID terms this often shows up as *precision* staying high (images look good) while *recall* drops (you cover fewer of the data modes) — exactly the quality-for-diversity trade that sits at one corner of the generative trilemma. If you need genuine variety across a batch — a grid of meaningfully different compositions for the same prompt — few-step LCM will disappoint you, and you should use the multi-step teacher or raise the step count.

Let me make the precision/recall mechanism concrete, because "diversity loss" is easy to wave at and harder to reason about. Precision and recall for generative models (Kynkäänniemi et al., 2019) split the single FID number into two: *precision* asks what fraction of generated samples land inside the support of the real-data manifold (are they realistic?), and *recall* asks what fraction of the real-data manifold is covered by the generated distribution (do you reach all the modes?). A mode-seeking sampler trades these against each other. When the consistency map snaps every seed toward the high-density center of the prompt's mode, almost everything it produces is realistic — precision is high — but it stops generating the rare, tail samples that would have covered the edges of the manifold, so recall falls. Concretely: ask a 25-step teacher for "a dog in a park" across 16 seeds and you get different breeds, poses, lighting, and framings; ask a 1-step LCM the same and you get 16 variations on the *single most prompt-typical* dog-in-a-park, with smaller deltas between seeds. Four steps recovers a lot of this — the re-noising in multistep sampling reintroduces seed-dependent variation at each round — which is another reason 4 is the default rather than 1. The honest framing for a stakeholder is: few-step consistency does not make worse images, it makes *fewer kinds* of images, and whether that matters depends entirely on whether your product needs variety or just one good result fast.

There is a deployment trick that partially buys diversity back without leaving the fast path: **vary the prompt or seed-plus-noise structure**, not just the seed. Because the map is mode-seeking on the prompt, small prompt perturbations move you between modes more effectively than seed changes do. If you need a varied grid from an LCM, perturb the prompt ("a dog in a park, autumn" / "...overcast" / "...low angle") rather than spinning seeds — you are steering the mode-seeker to different modes instead of asking it to sample variation it has been trained to suppress.

**A quality ceiling below the teacher.** Even at 4–8 steps, a distilled consistency model is typically a hair below its teacher's best multi-step quality on fine texture and small-text fidelity. The distillation is a lossy compression of the teacher's sampling behavior; you cannot in general exceed the teacher you distilled from (later methods like DMD2 and adversarial distillation, covered in the forward-linked post on [distribution matching and adversarial distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation), add a GAN-style loss to *close* this gap and sometimes match the teacher in one step — that is the next chapter of this story). LCM at 4 steps is "very good, ship it for most use"; it is not "indistinguishable from 50-step DPM-Solver under a microscope."

**Guidance sensitivity and prompt range.** Because guidance is baked in, your control over the quality/diversity dial is narrower. You can still nudge `guidance_scale` between ~1 and ~2, but the wide CFG range of normal SD (1–15) is gone, and prompts that needed very high guidance to bind a tricky attribute may bind less reliably in four steps. The CFG-augmented distillation conditions on $w$ over a *range*, so it is not zero control, but it is reduced control.

Here is the comparison the way I would actually present it to a team deciding whether to adopt LCM:

| Method | Steps | Rel. latency (SDXL, 4090) | Quality vs teacher | Diversity | Guidance control | When to use |
| --- | --- | --- | --- | --- | --- |
| DPM-Solver++ (teacher) | 25 | 1.0× (~2.4 s) | reference | full | full (CFG 1–15) | final renders, max diversity |
| DPM-Solver++ | 8 | ~0.35× | slightly soft | full | full | quality-leaning fast path |
| LCM / LCM-LoRA | 8 | ~0.2× | near-teacher | reduced | narrow (~1–2) | best LCM quality |
| LCM / LCM-LoRA | 4 | ~0.15× (~0.4 s) | very good | reduced | narrow (~1–2) | the default fast path |
| LCM / LCM-LoRA | 2 | ~0.08× | soft | low | narrow | quick previews |
| LCM / LCM-LoRA | 1 | ~0.05× (~0.15 s) | rough but coherent | low | minimal | real-time scrub, refine later |

Latencies are relative and approximate, measured as denoiser-loop-dominated wall time on a single RTX 4090 at 1024×1024 with fp16; absolute numbers vary with attention backend and batch. The shape — roughly 6× at four steps, with diversity as the price — is the robust takeaway.

### Stress-testing the decision: where few-step sampling breaks

The way to trust a method is to push it until it fails and know *why*. A few failure modes you will actually hit:

**You set `guidance_scale=7.5` out of habit.** The single most common LCM bug. Because the CFG-augmented ODE baked guidance into the weights, running real CFG on top double-counts it: contrast explodes, colors over-saturate, faces go waxy, and fine detail burns out. The fix is one line — drop to ~1.0–1.5. If your LCM images look "deep-fried," check the guidance scale first.

**You demand a varied 16-image grid and get 16 near-twins.** Covered above — this is the recall drop, not a bug. The fix is to perturb prompts rather than seeds, or to fall back to the multi-step teacher for that specific "give me variety" request.

**You fuse the wrong base family's LCM-LoRA.** SD-1.5 LCM-LoRA on an SDXL checkpoint (or vice versa) produces noise, because the latent shapes and ODE differ. Match the adapter to the base architecture.

**You push below the model's floor — 1 step on a model distilled for 4.** Some LCMs are distilled with a target step count in mind; ask for fewer and quality falls off a cliff faster than the matrix suggests. Respect the model card's recommended step range, and if you genuinely need one-step quality, use a method *built* for one step — SDXL-Turbo or DMD2 — rather than under-stepping an LCM.

**The VAE decode becomes your new bottleneck.** Once you have cut the denoiser loop from 50 forwards to 4, the one-time VAE decode (~120 ms at 1024²) is suddenly a *large* fraction of your ~400 ms total — it was noise before and is now a fifth of the budget. This is the lesson from [why diffusion is slow](/blog/machine-learning/image-generation/why-diffusion-is-slow-and-how-to-fix-it): when you remove the dominant cost, the next cost surfaces. The fix is a tiny VAE (`AutoencoderTiny`/TAESD) for previews, which decodes in single-digit milliseconds at a small quality cost — and at that point your four-step pipeline is genuinely latency-bound by the four forwards, which is exactly where you wanted to be.

## 10. A consistency-distillation loss in PyTorch

To make the training concrete rather than abstract, here is a stripped-down consistency-distillation step. It is a sketch — a real LCM training loop adds the CFG augmentation, the latent VAE encode, a proper sigma schedule, and an EMA update — but it shows the load-bearing parts: the skip parameterization, the teacher's one-step neighbor, the EMA target with stop-gradient, and the distance loss.

```python
import torch
import torch.nn.functional as F

def c_skip(sigma, sigma_data=0.5, sigma_min=0.002):
    return sigma_data**2 / ((sigma - sigma_min)**2 + sigma_data**2)

def c_out(sigma, sigma_data=0.5, sigma_min=0.002):
    return sigma_data * (sigma - sigma_min) / (sigma_data**2 + sigma**2).sqrt()

def consistency_f(net, x, sigma):
    # f = c_skip * x + c_out * F_theta(x, sigma)  -> boundary f(x,0)=x for free
    cs = c_skip(sigma).view(-1, 1, 1, 1)
    co = c_out(sigma).view(-1, 1, 1, 1)
    return cs * x + co * net(x, sigma)

def teacher_ode_step(teacher, x, sigma_n, sigma_nm1):
    # One Euler step of the PF-ODE: dx/dsigma = (x - D(x,sigma)) / sigma
    d = (x - teacher(x, sigma_n).detach()) / sigma_n.view(-1, 1, 1, 1)
    return x + d * (sigma_nm1 - sigma_n).view(-1, 1, 1, 1)

def cd_loss(student, target_net, teacher, x0, sigmas):
    # sigmas: a discretized increasing schedule, shape [N]
    b = x0.shape[0]
    n = torch.randint(1, len(sigmas), (b,), device=x0.device)
    sigma_n = sigmas[n]
    sigma_nm1 = sigmas[n - 1]

    eps = torch.randn_like(x0)
    x_n = x0 + sigma_n.view(-1, 1, 1, 1) * eps          # point on the trajectory

    # Teacher supplies the neighbor one step closer to the data.
    x_nm1 = teacher_ode_step(teacher, x_n, sigma_n, sigma_nm1)

    online = consistency_f(student, x_n, sigma_n)        # student, with gradient
    with torch.no_grad():                                # EMA target, stop-gradient
        target = consistency_f(target_net, x_nm1, sigma_nm1)

    return F.mse_loss(online, target)                    # or LPIPS / Huber

@torch.no_grad()
def ema_update(target_net, student, mu=0.999):
    for pt, ps in zip(target_net.parameters(), student.parameters()):
        pt.mul_(mu).add_(ps, alpha=1 - mu)
```

The pieces map one-to-one onto the math: `consistency_f` is the skip-blend parameterization that makes the boundary condition free; `teacher_ode_step` is the single Euler step that finds the neighbor; `cd_loss` compares the online student at $\sigma_n$ against the EMA target at the neighbor $\sigma_{n-1}$ with a stop-gradient; and `ema_update` is the slow target drift that keeps the fixed point stable. A production LCM swaps `mse_loss` for a Huber or LPIPS distance (more robust to the occasional bad teacher step), encodes images to latents first, and threads the CFG scale $w$ through `net(x, sigma, w)` so one model serves a guidance range. But this is genuinely the skeleton.

A note on the distance $d$. Plain $\ell_2$ in latent space works but tends to over-smooth — it punishes high-frequency disagreement the same as structural disagreement. LPIPS (a learned perceptual distance) or a pseudo-Huber loss gives sharper distilled results because it tolerates small pixel-level differences while still penalizing the structural ones that matter. The improved-consistency-training paper made the pseudo-Huber loss and a tuned noise-level weighting central to closing the gap to distillation, and it is a cheap upgrade.

## 11. TCD and the broader few-step family

LCM is not the only consistency-flavored accelerator, and it helps to place it in the family so you pick the right one. Figure 8 sketches the taxonomy.

![A tree of the few-step distillation family branching into training methods CD, CT, and LCM and deployment forms LCM-LoRA and TCD](/imgs/blogs/consistency-models-and-few-step-generation-8.png)

**TCD (Trajectory Consistency Distillation, Zheng et al., 2024)** is the most useful sibling. It reformulates the distillation so the model is consistent along the *trajectory* in a way that gives an explicit "stochasticity" knob ($\gamma$) at sampling time, decoupling how many steps you take from how much the sampler explores. In practice TCD often produces somewhat crisper detail than vanilla LCM at the same low step count, and the $\gamma$ knob lets you trade a little of the diversity back. In `diffusers` it ships as `TCDScheduler` and is a drop-in alternative to `LCMScheduler` (often with a TCD-LoRA), so trying both on your prompts costs one line.

The wider landscape, which the next post picks up, splits by *what extra signal* you add to the distillation:

- **Pure consistency** (LCM, TCD): match a trajectory's self-consistency. Cheap, stable, ~4 steps, slight diversity loss. This post.
- **Adversarial distillation** (ADD / SDXL-Turbo, LADD): add a GAN discriminator so the student's one-step outputs are pushed onto the real-image manifold, not just toward the teacher. Gets to *one* step at higher quality, at the cost of GAN training instability.
- **Distribution matching** (DMD, DMD2): match the *distribution* of one-step outputs to the data distribution via a score-distillation objective. DMD2 reaches one-step FID competitive with the multi-step teacher.

All three live in [distribution matching and adversarial distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation). The thing to hold onto is that consistency models opened this door: they were the first to show you can collapse the sampling ODE into a handful of jumps *without* a GAN, purely from the trajectory structure, and everything after is about closing the last quality gap.

Here is the family side by side, so the choice is legible:

| Method | Extra signal | Best step count | Quality vs teacher | Training cost | Portable LoRA? |
| --- | --- | --- | --- | --- | --- |
| Progressive distillation | none (imitate K-step output) | ~4–8 | matches teacher | high (log₂ runs) | no |
| LCM | self-consistency + CFG-aug | 4 | near-teacher | low | yes (LCM-LoRA) |
| TCD | trajectory consistency + γ | 4 | near-teacher, crisper | low | yes (TCD-LoRA) |
| ADD / SDXL-Turbo | GAN discriminator | 1 | competitive @ 1 step | medium (GAN) | partial |
| DMD2 | distribution matching | 1 | matches teacher @ 1 step | medium-high | partial |

The columns that matter for a decision: *best step count* (how few you can go), *quality vs teacher* (what you give up), and *portable LoRA* (whether you can fuse it onto an existing fine-tune without re-training). LCM and TCD are the cheap, portable, four-step workhorses; the GAN/distribution-matching methods buy one-step at the cost of heavier training and weaker portability. For most pipelines you start at LCM-LoRA, and you only graduate to the one-step methods when four steps is still too slow.

## 12. Case studies: the real numbers

A few measured results from the literature and shipped models, with sources, so the trade-offs are concrete rather than asserted. Where I do not have an exact figure I say so — never trust a suspiciously precise number you cannot trace.

**Consistency models on CIFAR-10 / ImageNet 64×64 (Song et al., 2023).** The original paper reported that consistency *distillation* on CIFAR-10 reached single-step FID in the low single digits (around 3.5) and two-step FID a touch better — a large jump over prior single-step distillation methods at the time, and within striking distance of the multi-step teacher. On ImageNet 64×64 the one-step and two-step numbers were similarly the best single/few-step results reported then. The headline was qualitative: *one network call* gave a genuinely good sample, where prior one-step methods gave mush. (Exact FID values depend on the eval setup; treat them as the right order of magnitude.)

**LCM on Stable Diffusion (Luo et al., 2023).** LCM distilled from SD-v1.5 and SDXL produced usable images at 1–4 steps where the base models need 25–50. The reported user studies and FID on LAION-subset benchmarks showed 4-step LCM roughly matching the 25-step DDIM baseline in human preference, with a measurable but acceptable diversity reduction. The distillation itself was cheap — on the order of a single-digit number of A100-days — because it operates in latent space and reuses the teacher.

**LCM-LoRA portability (Luo et al., 2023).** The LCM-LoRA paper's central demonstration was that *one* LoRA (~67M params for SDXL) trained against base SDXL accelerated *arbitrary* SDXL fine-tunes to 4 steps, without per-model re-distillation. This is the result that made few-step sampling a community default rather than a per-checkpoint research project — the speedup became a downloadable file you fuse on.

**SDXL-Turbo / ADD (Sauer et al., 2023), for contrast.** The adversarial-distillation line reached *single-step* 512² generation with quality the authors argued was competitive with multi-step SDXL in human preference at one step — a step beyond what pure consistency hits in one step. The cost is GAN training and a quality profile that is excellent at the typical mode but, again, narrower in diversity. It is the natural next rung and the reason the forward-linked distillation post exists.

#### Worked example: how to measure the LCM vs teacher gap honestly

Suppose you want to *verify* on your own data that 4-step LCM is "close enough" to your 25-step teacher before you ship it. The honest protocol, not the cherry-picked one: fix a held-out reference set of, say, 10,000 real images representative of your domain; generate 10,000 samples from each model using a *fixed, shared* set of prompts and a *fixed* set of seeds (so the only variable is the sampler); compute FID against the reference set with the same Inception feature extractor for both, and — crucially — compute *precision and recall* separately, because FID alone will hide the diversity story. You will typically see the teacher and the 4-step LCM within a small FID gap, with the LCM's precision equal-or-higher and its recall lower; that recall gap *is* the diversity cost, quantified. Report all three numbers, the exact sample count (FID is biased at small $N$ — 10k+ is the usual floor), the reference set, and the seeds, because an FID with none of that context is unfalsifiable. If you only have budget for a human study instead, randomize and blind the side-by-side pairs and report the win rate with a confidence interval. The discipline here is the same one a dedicated evaluation post in this series will go deeper on: a single number with no protocol is marketing, not measurement.

The honest summary across these: consistency models reliably deliver ~4-step sampling at near-teacher quality with a diversity cost, on a cheap distillation budget; the adversarial/distribution-matching successors push toward one step and close the quality gap at the cost of training complexity. For most teams the LCM/LCM-LoRA point — four steps, ~6× faster, fuse-on portability — is the pragmatic default, and you reach for the heavier one-step methods only when the last 2× and the absolute one-step latency genuinely matter.

## 13. When to reach for this (and when not to)

A decisive recommendation section, because every acceleration is a cost.

**Reach for LCM / LCM-LoRA when:**

- You need *interactive* latency — a live slider, a paint-and-see canvas, a real-time preview — where 0.4 s beats 2.4 s decisively and slight softness is invisible at preview size.
- You serve *many* requests and step count is your dominant cost; 6× fewer forwards is 6× more throughput per GPU, which is real money.
- You want to accelerate an *existing fine-tune* without re-training; LCM-LoRA fuses on and leaves your subject/style weights untouched.
- You are on a *constrained device* (consumer GPU, even mobile) where 50 forwards is infeasible but 4 is fine.

**Do not reach for it when:**

- You need *maximum diversity* across seeds — a grid of genuinely different compositions for one prompt. Few-step consistency snaps to the mode; use the multi-step teacher.
- You are producing *final hero renders* where the last few percent of fine-texture fidelity matters and latency does not; run the 25-step teacher.
- You need *wide guidance control* — strong attribute binding via high CFG, or precise CFG tuning per prompt. The baked-in guidance narrows your dial to ~1–2.
- A *fast solver already suffices*. If DPM-Solver++ at 8–10 steps hits your quality and latency budget, you may not need distillation at all — it has zero diversity cost and no extra training. Don't reach for a distilled model when a better sampler closes the gap.

The pattern that works in production is the **two-path split**: a fast LCM/LCM-LoRA path for interaction and previews, and a slow multi-step teacher path for the final committed frame. The user scrubs in real time on the fast path, then you spend the two seconds once on the path that maximizes quality. You get the best of both corners of the trilemma by choosing per-moment which corner you are optimizing.

## 14. Key takeaways

- A **consistency model** learns a function $f(x,\sigma)$ that maps *any* point on a probability-flow-ODE trajectory directly to that trajectory's clean origin $x_0$ — collapsing a 50-step ODE solve into one network call.
- The training target is **self-consistency**: $f$ must give the same answer at adjacent points on the same trajectory, anchored by the **boundary condition** $f(x_0, 0) = x_0$ that rules out the degenerate constant solution.
- The **skip-connection parameterization** $f = c_\text{skip}(\sigma)\,x + c_\text{out}(\sigma)\,F_\theta$ with $c_\text{skip}(0)=1, c_\text{out}(0)=0$ makes the boundary condition hold *by construction*, so the optimizer never fights for it.
- **Consistency distillation** uses a frozen diffusion teacher to supply a one-step neighbor on the trajectory and is cheap and stable; **consistency training** drops the teacher for a higher-variance estimator and trains from scratch. For images, distillation wins.
- The **EMA target network with stop-gradient** is what keeps training from collapsing to a constant — it is the same stabilizer as in self-distillation and deep Q-learning.
- **Multistep consistency sampling** (jump → re-noise → jump, 2–4 rounds) buys back the quality a single jump loses; 4 steps is the SDXL sweet spot.
- **LCM** distills consistency into Stable Diffusion's latent space along a **CFG-augmented ODE**, so you sample at `guidance_scale ≈ 1` — running normal CFG on top double-counts guidance and over-saturates.
- **LCM-LoRA** ships the acceleration as a ~67M-parameter portable adapter that fuses onto *any* compatible checkpoint, turning the whole SDXL zoo into 4-step samplers for free.
- The cost is real and measurable: **reduced diversity** (precision stays high, recall drops), a slight quality ceiling below the teacher, and narrowed guidance control. Name it before you ship, and keep a slow path for final renders.

## 15. Further reading

- **Song, Dhariwal, Chen, Sutskever (2023), "Consistency Models."** The original — the consistency property, distillation vs training, the skip parameterization, and the multistep sampler. The paper to read first.
- **Song & Dhariwal (2024), "Improved Techniques for Training Consistency Models."** The pseudo-Huber loss, noise-schedule weighting, and dropping the EMA target — how to close consistency *training*'s gap to distillation.
- **Luo et al. (2023), "Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference."** LCM, the latent-space distillation, and the CFG-augmented ODE that makes it work on Stable Diffusion.
- **Luo et al. (2023), "LCM-LoRA: A Universal Stable-Diffusion Acceleration Module."** The portable-adapter result that made few-step sampling a community default.
- **Zheng et al. (2024), "Trajectory Consistency Distillation (TCD)."** The trajectory-consistency reformulation and the sampling-time stochasticity knob; `TCDScheduler` in `diffusers`.
- **Karras et al. (2022), "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM).** Where the $c_\text{skip}/c_\text{out}$ preconditioning and the sigma parameterization come from.
- 🤗 **`diffusers` documentation** — `LCMScheduler`, `TCDScheduler`, LCM-LoRA usage, and the inference examples this post's code is built on.
- Within this series: [why diffusion is slow and how to fix it](/blog/machine-learning/image-generation/why-diffusion-is-slow-and-how-to-fix-it) (the cost model and the four levers), [DDIM and fast deterministic sampling](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling) and [the samplers deep dive](/blog/machine-learning/image-generation/samplers-deep-dive) (the ODE this method collapses), [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance) (the CFG that LCM bakes in), the forward-linked [distribution matching and adversarial distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation) (the one-step successors), and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
