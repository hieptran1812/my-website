---
title: "Flow Matching and Rectified Flow: The Straight-Line Successor to Diffusion"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Derive the conditional flow-matching loss and the straight-line velocity target v = x1 - x0, write a runnable PyTorch training loop and Euler sampler, and see exactly why SD3 and FLUX replaced the diffusion objective with flow matching."
tags:
  [
    "image-generation",
    "diffusion-models",
    "flow-matching",
    "rectified-flow",
    "continuous-normalizing-flows",
    "stable-diffusion-3",
    "flux",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/flow-matching-and-rectified-flow-1.png"
---

In 2024 something quietly broke with the consensus that diffusion was the way to train text-to-image models. Stability AI released **Stable Diffusion 3**, and Black Forest Labs released **FLUX.1**, and both of them — the two most important open text-to-image releases of the year — were not trained with the DDPM noise-prediction loss that powered SD1.5, SD2, and SDXL. They were trained with a different objective entirely: **flow matching**. Not a new sampler bolted onto the same model, not a schedule tweak — a different *training target*. Where SDXL regresses a U-Net onto the noise $\epsilon$ added to a latent, SD3 regresses a transformer onto a **velocity** $x_1 - x_0$, a constant vector pointing in a straight line from a noise sample to a data sample. That one change is the subject of this post, and understanding it is the difference between treating SD3 as a black box and knowing why it trains more stably, samples in fewer steps, and scales more cleanly than the diffusion models it replaced.

Here is the scenario that makes it concrete. You have a `torch.randn(1, 16, 128, 128)` tensor of pure Gaussian noise and you want a 1024×1024 photograph of a golden retriever in a kitchen. A DDPM-trained SDXL takes that noise and walks it back to a latent over roughly 30–50 denoising steps along a path through latent space that **curves**, because the variance-preserving forward process bends the trajectory; take too few steps and a coarse Euler integrator overshoots the curve and you get artifacts or mush. A flow-matching model takes the same noise and walks it to a latent along a path that is **straight** — a literal line segment — so a handful of steps, sometimes as few as one after distillation, can trace it almost exactly. The whole game of this post is *why* the flow-matching path is straight, *why* you can train for it without ever simulating a trajectory, and *why* straightness is worth so much at sampling time.

![A graph showing how continuous normalizing flow training was replaced by a simulation-free velocity regression onto a straight-line target, which then samples in a few Euler steps and powers SD3 and FLUX](/imgs/blogs/flow-matching-and-rectified-flow-1.png)

By the end you will be able to: explain why **continuous normalizing flows** — neural ODEs that transport noise to data — were mathematically beautiful but computationally hopeless to train, and pinpoint the exact cost (a full ODE solve and a divergence trace per training step) that killed them at image scale; derive the **conditional flow matching** objective of Lipman, Chen, Ben-Hamu, Nickel & Le (2023) and prove that regressing onto a *per-pair* conditional velocity gives you the correct *marginal* velocity field for free, with no marginal ever computed; derive the **straight-path** probability path $x_t = (1-t)x_0 + t\,x_1$ and read off its constant velocity target $v = x_1 - x_0$; explain **rectified flow** (Liu, Gong & Liu, 2023) and the **reflow** procedure that iteratively straightens trajectories until you can sample in one step; place flow matching next to diffusion and see that they are both ODEs over the same kind of object, flow matching just picks the straight path and the simpler loss, with DDPM falling out as a *curved* special case; understand the **logit-normal timestep sampling** and **resolution shift** that SD3 uses; and write a runnable flow-matching training loop and Euler sampler in PyTorch plus the 🤗 `diffusers` `FlowMatchEulerDiscreteScheduler` call that drives SD3 and FLUX in production.

Keep the series' spine in mind: the **generative trilemma** (sample quality × mode coverage × sampling speed) and the **diffusion stack** (data → VAE/latent → forward noising → denoiser net → ODE/SDE sampler → guidance → image). Flow matching is a move on two corners of that triangle at once. It does not sacrifice quality or coverage relative to diffusion — SD3 and FLUX prove that — but it buys a large chunk of the *speed* corner by replacing a curved sampling trajectory with a straight one, and it makes the *training* of the denoiser-net slot more stable. This post is the natural sequel to two earlier ones: [normalizing flows and the change of variables](/blog/machine-learning/image-generation/normalizing-flows-and-change-of-variables), which built the continuous-normalizing-flow setup and left flow matching as a teaser, and [score-based models and the SDE view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view), which showed that every diffusion model is secretly an ODE — the **probability-flow ODE**. Flow matching is what happens when you stop deriving that ODE from a noising SDE and instead *design the ODE you want* and train for it directly.

## 1. Continuous normalizing flows: the beautiful, impractical ancestor

To understand why flow matching is such a relief, you have to feel the pain it removed. That pain has a name: the **continuous normalizing flow** (CNF), and to set it up I'll recall the one fact from the [normalizing-flows post](/blog/machine-learning/image-generation/normalizing-flows-and-change-of-variables) that everything here rests on.

A discrete normalizing flow stacks $K$ invertible layers $f_1, \dots, f_K$ to turn a base Gaussian sample $z$ into data $x$, and tracks how the density changes through each layer with the change-of-variables formula. A CNF takes $K \to \infty$ and replaces the stack of discrete maps with a **continuous-time flow**: a vector field $v_\theta(x, t)$, a neural network, that defines an ordinary differential equation

$$
\frac{dx}{dt} = v_\theta(x, t), \qquad x(0) = x_0 \sim \mathcal{N}(0, I).
$$

You start at a noise sample $x_0$ at time $t = 0$, and you let the ODE carry it forward to time $t = 1$. The endpoint $x_1 = x(1)$ is your generated sample. The vector field $v_\theta$ is the *velocity* of the moving point: at every position $x$ and time $t$ it says "move this way, this fast." Integrate the velocity field over $[0, 1]$ and a Gaussian blob is transported, smoothly and invertibly, into the data distribution. This is a genuinely elegant picture — a single network parameterizes an entire continuous transformation between distributions, and because the flow is a diffeomorphism for any smooth $v_\theta$, you keep the exact-likelihood property of flows without the architectural straitjacket of hand-designed invertible layers.

The density evolves by the **continuous change of variables**, also called the *instantaneous change-of-variables* formula (Chen et al., Neural ODEs, 2018):

$$
\frac{d \log p_t(x(t))}{dt} = -\,\nabla \cdot v_\theta(x(t), t) = -\operatorname{tr}\!\left(\frac{\partial v_\theta}{\partial x}\right).
$$

The change in log-density along a trajectory is minus the **divergence** of the velocity field — the trace of its Jacobian. Integrate that from $0$ to $1$ alongside the state, and you get the exact log-likelihood of the generated sample. So the maximum-likelihood training objective is in principle clean: pick $v_\theta$ to maximize $\log p_1(x)$ on your data. And it works — on 2D toy distributions and small images, CNFs (and their scalable cousin FFJORD, which estimates the trace with Hutchinson's stochastic estimator) produce gorgeous exact-likelihood models.

Now the catch, and it is fatal. Look at what one training step costs. To evaluate the likelihood of a single data point you must:

1. **Solve the ODE** $dx/dt = v_\theta(x, t)$ — numerically integrate the neural network from $t=1$ back to $t=0$ (or forward, depending on your parameterization). A black-box adaptive solver like `dopri5` calls the network *dozens to hundreds of times* to hit its error tolerance, and each call is a full forward pass of the network.
2. **Integrate the divergence** $\nabla \cdot v_\theta$ along that same trajectory, which for an exact trace costs $D$ extra backward passes per step ($D$ = the data dimension), or one extra vector-Jacobian product per step with the Hutchinson estimator — noisy, but the cost is still tied to the solve.
3. **Backpropagate through the solver** to get $\partial \mathcal{L} / \partial \theta$, either by storing the whole solve graph (huge memory) or by the adjoint method (a second ODE solve backward in time).

So a single gradient step is *two or three full ODE solves of a neural network*. For a $256 \times 256 \times 3$ image, with a network you need to call a hundred times per solve, training is simply not viable — you would spend orders of magnitude more compute per step than a diffusion model spends, and diffusion models already need a lot. This is the wall. CNFs gave us the right *object* — a learned ODE transporting noise to data — and the wrong *training procedure*: simulation. Every step required simulating the very dynamics you were trying to learn.

FFJORD (Grathwohl et al., 2019) softened one of the three costs — the divergence — with **Hutchinson's trace estimator**, which replaces the exact $D$-pass trace with a single vector-Jacobian product against a random probe $\eta$: $\operatorname{tr}(A) = \mathbb{E}_\eta[\eta^\top A \eta]$ for $\eta$ with identity covariance. That turns the per-step divergence from $D$ backward passes into one, and it is what let CNFs touch real images at all. But notice what it did *not* fix: you still solve the ODE, still call the network dozens of times per example per step, still backprop through (or adjoint-solve) the whole trajectory. The dominant cost — *simulating the dynamics* — survived. FFJORD made an unaffordable method merely very expensive. The Hutchinson trick is a good lesson in the limits of attacking the wrong cost: it optimized the cheap part (the trace) and left the expensive part (the solve) untouched, so the method still lost to diffusion on compute-per-quality by a wide margin. The right move was not a better trace estimator; it was to stop solving altogether.

There is one more reason simulation-based training is fragile that's worth naming, because it motivates flow matching's design directly. When you backprop through an adaptive ODE solver, the *number of function evaluations is itself data-dependent and changes during training* — a stiff region of the field makes the solver take more steps, which changes the compute and the gradient noise from batch to batch. Training dynamics become coupled to solver dynamics in a way that is genuinely hard to debug: a model that's learning a stiffer field silently gets slower and noisier gradients. Flow matching severs that coupling completely. Training cost is constant and known (one network call per example per step, like any regression), because there is no solver in the loop at all. Decoupling the optimizer from the integrator is a real engineering win on top of the raw FLOP savings.

The escape, when it came, was almost embarrassingly simple in hindsight: *what if you never simulate at all?* What if, instead of solving the ODE to find out where the field should point and then correcting it, you could write down — in closed form — a velocity field that you *know* transports noise to data, and just regress the network onto it with plain mean-squared error? No solver, no divergence trace, no adjoint. That is flow matching, and the first thing to establish is that such a closed-form target exists.

## 2. The probability path: designing the trajectory you want

Flow matching flips the CNF logic on its head. A CNF *learns* a vector field and *discovers* what probability path it induces. Flow matching *chooses* the probability path first, derives the vector field that produces it, and then trains the network to match that field. So before any loss, we need to specify the path.

A **probability path** is a time-indexed family of distributions $p_t(x)$ for $t \in [0, 1]$ that interpolates between a simple base distribution and the data distribution:

$$
p_0 = p_{\text{noise}} = \mathcal{N}(0, I), \qquad p_1 = p_{\text{data}}.
$$

(I use the SD3/rectified-flow convention where $t=0$ is noise and $t=1$ is data; the original flow-matching paper uses the reverse, $t=0$ data and $t=1$ noise. The math is identical up to flipping $t$; I'll be consistent with $t=1$ = data throughout, because that matches what `diffusers` does.) The path is the *plan*: it says how the noise distribution should morph into the data distribution as $t$ runs from 0 to 1. A **vector field** $u_t(x)$ *generates* the path if pushing $p_0$ along the ODE $dx/dt = u_t(x)$ produces exactly $p_t$ at every time. The link between a path and the field that generates it is the **continuity equation** (a.k.a. the transport or Liouville equation), which is just conservation of probability mass in flow form:

$$
\frac{\partial p_t(x)}{\partial t} + \nabla \cdot \big( p_t(x)\, u_t(x) \big) = 0.
$$

Read it as a fluid: $p_t$ is the density of a fluid, $u_t$ is its velocity field, and the equation says mass is neither created nor destroyed — whatever flows out of a region shows up as a drop in density there. Given a path $p_t$, any $u_t$ satisfying this equation generates it. Our job is to (a) pick a convenient path and (b) find a $u_t$ that generates it. The genius of conditional flow matching is making both of those trivial by *conditioning on a single data point*.

![A before-after figure contrasting the curved variance-preserving DDPM trajectory that overshoots under coarse Euler steps with the straight optimal-transport flow-matching path that a few steps trace exactly](/imgs/blogs/flow-matching-and-rectified-flow-2.png)

### The conditional path: one noise sample to one data sample

Here is the move that makes everything tractable. Instead of trying to write down the full marginal path $p_t$ — which mixes all the data together and is hopeless in closed form — we condition on a single data sample $x_1$ and define a **conditional probability path** $p_t(x \mid x_1)$ that we *can* write down. The natural, almost trivial choice is a Gaussian that starts wide (the noise) and collapses onto the data point:

$$
p_t(x \mid x_1) = \mathcal{N}\big(x \,;\, \mu_t(x_1),\, \sigma_t^2 I\big),
$$

with boundary conditions $\mu_0 = 0, \sigma_0 = 1$ (so $p_0(\cdot \mid x_1) = \mathcal{N}(0, I)$, the noise, *independent* of $x_1$) and $\mu_1 = x_1, \sigma_1 \approx 0$ (so $p_1(\cdot \mid x_1)$ is a point mass at the data sample $x_1$). The whole conditional path is just a Gaussian blob whose mean slides from the origin to $x_1$ while its variance shrinks from $1$ to $0$. The marginal path you actually care about is recovered by averaging the conditional paths over the data:

$$
p_t(x) = \int p_t(x \mid x_1)\, p_{\text{data}}(x_1)\, dx_1.
$$

That integral is intractable — but, as we'll prove in the next section, *we never have to compute it*. For now, just note that this marginal $p_t$ has the right endpoints by construction: at $t=0$ every conditional is $\mathcal{N}(0,I)$ so the marginal is $\mathcal{N}(0,I)$; at $t=1$ every conditional is a spike at its own $x_1$ so the marginal is $p_{\text{data}}$. Good.

### The straight-line (optimal-transport) path

We have freedom in choosing $\mu_t$ and $\sigma_t$. Different choices give different known diffusion and flow models. The choice that defines flow matching's headline variant — and the one SD3 and FLUX use — is the **linear / optimal-transport path**, where the mean moves linearly and the variance shrinks linearly:

$$
\mu_t(x_1) = t\, x_1, \qquad \sigma_t = 1 - t.
$$

Sampling from this conditional path is then a one-liner. If $x_0 \sim \mathcal{N}(0, I)$ is a fresh noise sample, then

$$
x_t = \mu_t + \sigma_t\, x_0 = t\, x_1 + (1-t)\, x_0 = (1-t)\, x_0 + t\, x_1.
$$

This is the crucial formula. **The point at time $t$ is just a linear interpolation between the noise sample $x_0$ and the data sample $x_1$.** At $t=0$ it is the noise $x_0$; at $t=1$ it is the data $x_1$; in between it slides along the straight line connecting them. There is no curvature anywhere — for a *fixed pair* $(x_0, x_1)$, the trajectory $t \mapsto x_t$ is a literal straight line segment in $\mathbb{R}^D$. This is the optimal-transport displacement interpolation between the two points (the path that minimizes transport cost is a straight line at constant speed), which is where the "OT" name comes from.

### Reading off the velocity: the derivation

Now derive the velocity that moves a point along this conditional path. For a *fixed* pair $(x_0, x_1)$, the position is $x_t = (1-t)x_0 + t x_1$, and the velocity is simply the time derivative:

$$
u_t(x_t \mid x_0, x_1) = \frac{d x_t}{dt} = \frac{d}{dt}\big[(1-t)x_0 + t x_1\big] = -x_0 + x_1 = x_1 - x_0.
$$

That is the whole derivation, and the result is breathtakingly simple:

$$
\boxed{\; u_t = x_1 - x_0. \;}
$$

The target velocity is a **constant vector** — it does not depend on $t$ at all. The point moves from noise to data at constant speed along a straight line, so its velocity is the displacement $x_1 - x_0$, the same at every instant. This is the closed-form target that CNFs never had. We do not need to solve any ODE to find out which way the field should point for this pair; we can write it down. Train a network $v_\theta(x_t, t)$ to output $x_1 - x_0$ when fed the interpolated point $x_t$ and the time $t$, and you are regressing onto a known target with plain MSE. No solver, no divergence, no adjoint. *That* is what killed the CNF training cost.

There is a subtlety worth stating now and proving in the next section: $x_1 - x_0$ is the *conditional* velocity, the right answer when you happen to know both endpoints. At sampling time the network only sees $x_t$ and $t$, not the pair — it cannot know which $(x_0, x_1)$ produced a given $x_t$, because many pairs pass through the same point. What the network actually learns is the *expectation* of $x_1 - x_0$ over all pairs that route through $x_t$, and the beautiful result is that this expectation is exactly the **marginal** velocity field that generates the marginal path. Conditioning on a single pair to get a trivial target, then letting the regression average them into the correct marginal — that is the entire trick.

#### Worked example: the velocity at the midpoint

Make it concrete with numbers. Take a 1D toy: noise $x_0 = -2$, data $x_1 = 3$. The conditional path is $x_t = (1-t)(-2) + t(3) = -2 + 5t$. At $t=0$, $x_t = -2$ (the noise); at $t=1$, $x_t = 3$ (the data); at the midpoint $t=0.5$, $x_t = 0.5$. The velocity target is $x_1 - x_0 = 3 - (-2) = 5$, constant for all $t$. Check it against the path: $dx_t/dt = d(-2 + 5t)/dt = 5$. The point starts at $-2$ and moves right at speed $5$, covering the distance of $5$ in unit time to land exactly on the data at $t=1$. A network fed $(x_t = 0.5, t = 0.5)$ should output $5$. Now imagine a *second* pair, $x_0' = 1, x_1' = 0$, whose path is $x_t' = 1 - t$, passing through $x_t' = 0.5$ at $t = 0.5$ with target velocity $0 - 1 = -1$. Both pairs pass through the point $0.5$ at $t=0.5$ — so the network, which sees only $(0.5, 0.5)$, cannot output both $5$ and $-1$. It outputs the *average* weighted by how likely each pair is to be there, and that average is the marginal velocity at $(0.5, 0.5)$. The next section proves this averaging is exactly correct.

## 3. Conditional flow matching: the loss, and why the marginal falls out for free

We now have a closed-form conditional velocity. The remaining question is whether regressing onto it actually trains the *marginal* field we need for sampling. The answer is yes, and the proof is the heart of Lipman et al. (2023). It is one of those results that looks like it cannot possibly be that easy, and then the algebra makes it inevitable.

![A before-after figure showing the intractable marginal velocity field that transports the whole noise cloud versus the closed-form conditional velocity for one pair whose gradient matches the marginal in expectation](/imgs/blogs/flow-matching-and-rectified-flow-6.png)

### The two losses

The objective we *want* to minimize is the **flow matching (FM)** loss — regress the network onto the true marginal field $u_t(x)$ that generates the marginal path:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\; x \sim p_t(x)}\Big[\, \big\| v_\theta(x, t) - u_t(x) \big\|^2 \,\Big].
$$

This is uncomputable as written, for the same reason explicit score matching was uncomputable: we do not have the marginal field $u_t(x)$ in closed form (it is that intractable integral over all data). So instead we minimize the **conditional flow matching (CFM)** loss, which regresses onto the conditional velocity we *do* have:

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\; x_1 \sim p_{\text{data}},\; x \sim p_t(x \mid x_1)}\Big[\, \big\| v_\theta(x, t) - u_t(x \mid x_1) \big\|^2 \,\Big].
$$

Everything in $\mathcal{L}_{\text{CFM}}$ is samplable: draw a data point $x_1$, draw a noise $x_0$, form $x = x_t = (1-t)x_0 + t x_1$, and the target $u_t(x \mid x_1) = x_1 - x_0$ is a closed form. The claim that makes flow matching work is:

$$
\nabla_\theta\, \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta\, \mathcal{L}_{\text{CFM}}(\theta).
$$

**The two losses have identical gradients.** They differ only by a constant independent of $\theta$. So minimizing the tractable CFM loss minimizes the intractable FM loss exactly — the network you get is the one that matches the marginal field, even though you only ever regressed onto per-pair conditional targets. Let me prove it, because the proof is short and the mechanism is the whole point.

### The proof

Expand both squared norms. For the FM loss:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x \sim p_t}\big[\, \|v_\theta(x,t)\|^2 - 2\,\langle v_\theta(x,t),\, u_t(x)\rangle + \|u_t(x)\|^2 \,\big].
$$

For the CFM loss, the conditional expectation expands the same way:

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_1, x \sim p_t(\cdot|x_1)}\big[\, \|v_\theta(x,t)\|^2 - 2\,\langle v_\theta(x,t),\, u_t(x|x_1)\rangle + \|u_t(x|x_1)\|^2 \,\big].
$$

The third term in each ($\|u_t\|^2$, $\|u_t(\cdot|x_1)\|^2$) does not contain $\theta$, so it drops out of the gradient — it is the constant by which the two losses differ. That leaves the first and second terms, and we show those match between the two losses.

**First term.** We need $\mathbb{E}_{x \sim p_t}\|v_\theta(x,t)\|^2 = \mathbb{E}_{x_1, x \sim p_t(\cdot|x_1)}\|v_\theta(x,t)\|^2$. This is immediate: the marginal $p_t(x) = \int p_t(x|x_1) p_{\text{data}}(x_1)\,dx_1$ is *defined* as the average of the conditionals, so taking $\mathbb{E}$ of any function of $x$ under the marginal equals taking $\mathbb{E}$ over $x_1$ and then over the conditional. The first terms are equal.

**Second (cross) term.** This is the one that matters. We need

$$
\mathbb{E}_{x \sim p_t}\big[\langle v_\theta(x,t),\, u_t(x)\rangle\big] \;\stackrel{?}{=}\; \mathbb{E}_{x_1, x \sim p_t(\cdot|x_1)}\big[\langle v_\theta(x,t),\, u_t(x|x_1)\rangle\big].
$$

The key is the **definition of the marginal velocity** in terms of the conditionals. By the continuity equation, the marginal field that generates $p_t$ must be the conditional fields *averaged with the right weights* — specifically, weighted by how much each conditional path contributes density at $x$:

$$
u_t(x) = \int u_t(x \mid x_1)\, \frac{p_t(x \mid x_1)\, p_{\text{data}}(x_1)}{p_t(x)}\, dx_1 = \mathbb{E}_{x_1 \sim p_t(x_1 | x)}\big[\, u_t(x \mid x_1)\,\big].
$$

This is not an assumption; it is forced by the continuity equation. (Sketch: plug the right-hand side into the continuity equation for $p_t$. The numerator $u_t(x|x_1)p_t(x|x_1)$ is exactly the conditional flux, each conditional satisfies its own continuity equation, and integrating over $x_1$ gives the marginal continuity equation. So this weighted average is *the* field that transports $p_t$.) In words: the marginal velocity at $x$ is the **posterior-weighted average of the conditional velocities** of all the pairs whose path passes through $x$ — exactly the "average of $5$ and $-1$" from the worked example.

Now substitute this into the FM cross term and expand the marginal expectation as a double integral:

$$
\mathbb{E}_{x \sim p_t}\langle v_\theta(x,t), u_t(x)\rangle = \int p_t(x)\, \Big\langle v_\theta(x,t),\, \int u_t(x|x_1) \tfrac{p_t(x|x_1)p_{\text{data}}(x_1)}{p_t(x)}dx_1 \Big\rangle dx.
$$

The $p_t(x)$ outside cancels the $1/p_t(x)$ inside, leaving

$$
= \int\!\!\int \big\langle v_\theta(x,t),\, u_t(x|x_1)\big\rangle\, p_t(x|x_1)\, p_{\text{data}}(x_1)\, dx_1\, dx = \mathbb{E}_{x_1, x \sim p_t(\cdot|x_1)}\big[\langle v_\theta(x,t), u_t(x|x_1)\rangle\big].
$$

That is exactly the CFM cross term. The two cross terms are equal, the two first terms are equal, the two third terms are $\theta$-independent constants, so $\nabla_\theta \mathcal{L}_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{CFM}}$. **QED.**

Stand back and appreciate what just happened. We wanted to regress onto a field defined by an intractable integral over all data. We instead regressed onto a trivial per-pair target $x_1 - x_0$, and the *averaging built into the expectation* of the MSE loss reconstructs the correct marginal field automatically. The $1/p_t(x)$ that would have made the marginal velocity expensive to evaluate cancels against the $p_t(x)$ weight in the marginal expectation. You never compute the marginal, never solve an ODE, never touch a divergence. You sample a pair, interpolate, regress. The cancellation is the whole reason flow matching is cheap, and it is the exact analogue of the cancellation that made *denoising* score matching cheap in the [score-based post](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view) — both turn an intractable field-matching problem into a plain regression by conditioning on a sample.

For the straight-line path specifically, $u_t(x|x_1) = x_1 - x_0$, so the final training target is the constant displacement and the loss becomes the clean

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1}\Big[\, \big\| v_\theta\big((1-t)x_0 + t x_1,\, t\big) - (x_1 - x_0) \big\|^2 \,\Big].
$$

This is the entire SD3/FLUX training objective, modulo the conditioning (text embeddings) and the timestep weighting we'll get to. Five symbols of expectation, one MSE. Compare that to the DDPM variational bound with its KL terms and you start to feel why flow matching is "simpler."

## 4. Train by regression, sample by ODE: the runnable loop

Theory is cheap; let me show you the actual code, because the gap between "regress onto $x_1 - x_0$" and a working trainer is where the subtleties live (the time convention, the sampler direction, the velocity sign). Here is a complete, runnable flow-matching trainer and Euler sampler for a toy 2D problem — small enough to run on a laptop CPU in seconds, structured exactly like the real thing.

![A graph showing that flow-matching training fits a velocity network by simulation-free MSE regression while only sampling integrates the same network as an ODE with an Euler solver](/imgs/blogs/flow-matching-and-rectified-flow-3.png)

```python
import torch
import torch.nn as nn

# A tiny velocity network v_theta(x_t, t): takes a point and a time, returns a velocity.
class VelocityNet(nn.Module):
    def __init__(self, dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x_t, t):
        # t is a (B, 1) tensor in [0, 1]; concatenate it as an extra input feature.
        return self.net(torch.cat([x_t, t], dim=-1))


def sample_data(n):
    # Target distribution: two moons-ish blobs. Stand-in for "real latents x1".
    theta = torch.rand(n, 1) * 2 * torch.pi
    r = 2.0 + 0.1 * torch.randn(n, 1)
    x = torch.cat([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
    x[: n // 2, 1] += 1.0          # split into two arcs
    x[n // 2 :, 1] -= 1.0
    return x


def logit_normal_t(n, m=0.0, s=1.0):
    # SD3's logit-normal timestep sampling: draw in logit space, squash to (0,1).
    # Peaks near t=0.5 (the hard middle of the path), thins the easy ends.
    return torch.sigmoid(m + s * torch.randn(n, 1))


def train(steps=4000, batch=512, lr=2e-3):
    model = VelocityNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        x1 = sample_data(batch)               # data sample  (t = 1 end)
        x0 = torch.randn(batch, 2)            # noise sample (t = 0 end)
        t = logit_normal_t(batch)            # timestep ~ logit-normal
        x_t = (1 - t) * x0 + t * x1          # straight-line interpolation
        target = x1 - x0                     # constant velocity target
        pred = model(x_t, t)                 # v_theta(x_t, t)
        loss = ((pred - target) ** 2).mean()  # plain MSE -- no ODE solve anywhere
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")
    return model
```

Read the training step against the loss we derived. `x1` is the data sample (the $t=1$ end), `x0` is a fresh Gaussian (the $t=0$ end), `t` is the timestep, `x_t` is the linear interpolation $(1-t)x_0 + t x_1$, `target` is the constant velocity $x_1 - x_0$, and the loss is `((pred - target) ** 2).mean()`. There is no solver, no `odeint`, no divergence trace — the entire forward process that DDPM derives from an SDE and that a CNF would simulate is here a single line of interpolation. That is the simulation-free training that flow matching bought.

Sampling is where the ODE finally appears — and *only* here. We integrate the learned velocity field forward from a fresh noise sample with a plain Euler solver:

```python
@torch.no_grad()
def euler_sample(model, n=2000, steps=8):
    x = torch.randn(n, 2)                    # start at noise, t = 0
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n, 1), i * dt)       # current time
        v = model(x, t)                      # velocity at (x, t)
        x = x + v * dt                       # Euler step forward
    return x                                  # approximately distributed as p_data
```

Eight Euler steps. Each step asks the network "which way, how fast?" and nudges the point along. Because the *true* per-pair paths are straight lines, the learned marginal field is close to straight over most of the space, so even a handful of steps lands the samples on the data manifold. Run `model = train()` then `samples = euler_sample(model, steps=8)` and you will see the Gaussian blob transported onto the two arcs. Crank `steps` down to 2 and the two-moons shape is still recognizable, just coarser — try that with a curved DDPM path and 2 Euler steps and you get garbage. The contrast between the curved and straight paths in the figure above is not a metaphor; it is the difference you can watch on this toy in five lines.

A couple of implementation notes that bite people. First, **direction and sign**: I integrate *forward* in $t$ from noise ($t=0$) to data ($t=1$) with velocity $v = x_1 - x_0$. If you adopt the opposite time convention (data at $t=0$), you flip the sign of the velocity and integrate the other way; mixing conventions silently produces a model that runs the flow backward and outputs noise. Pick one and be ruthless about it. Second, **the network sees only $(x_t, t)$**, never the pair — that is what forces it to learn the marginal field, as the proof requires. Third, the `logit_normal_t` sampler is the one production detail that matters at scale, and §7 explains why.

### The velocity, the score, and ε are the same object in three coordinates

If you have trained a diffusion model you already have a network that predicts the noise $\epsilon$, and it is worth making explicit how that relates to the velocity a flow-matching network predicts — because they are, once again, the same field written in different coordinates, exactly as $\epsilon$-prediction and the score were in the [score-based post](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view). This is not academic: it tells you how to convert a flow-matching velocity into a classifier-free-guidance update, how to plug a flow model into a sampler written for $\epsilon$-prediction, and why a flow model and a diffusion model trained on the same data learn fundamentally the same vector field.

Start from the straight path $x_t = (1-t)x_0 + t x_1$ and solve it two ways. The velocity target is $v = x_1 - x_0$. But I can also write $x_0$ and $x_1$ each in terms of $x_t$ and $v$, because the path is linear. From the path and the velocity:

$$
x_1 = x_t + (1-t)\,v, \qquad x_0 = x_t - t\,v.
$$

Check: $x_t + (1-t)(x_1 - x_0) = (1-t)x_0 + t x_1 + (1-t)x_1 - (1-t)x_0 = t x_1 + (1-t)x_1 = x_1$. Good. So *given the predicted velocity*, you can read off the model's implied prediction of the clean data $x_1$ (the "$x_0$-prediction" of the diffusion world, confusingly — names clash across conventions) and its implied prediction of the noise $x_0$ (the "$\epsilon$-prediction"). A flow-matching velocity head is therefore interconvertible with an $\epsilon$-head and an $x$-head by these two linear formulas; `diffusers` uses exactly this to let flow models share guidance and sampler code with diffusion models.

The connection to the **score** $\nabla_{x_t} \log p_t(x_t)$ closes the loop. For a Gaussian conditional path, the score points from $x_t$ back toward the conditional mean, and a few lines of algebra (substitute the path, take the gradient of the log-Gaussian) give a linear relation between the velocity field and the score field of the form $v_t(x) = a(t)\,x + b(t)\,\nabla \log p_t(x)$ for known scalar functions $a, b$ that depend only on the schedule. The upshot is the same three-way equivalence we keep finding: **velocity ⇔ score ⇔ ε**, each a linear reparameterization of the others for a given path. Flow matching does not learn a *different* thing than score-based diffusion; it learns the *same* underlying field, parameterized so the regression target is the clean, $t$-stable displacement $x_1 - x_0$ instead of the scale-varying noise $\epsilon$. That reparameterization — not a new kind of physics — is the source of flow matching's better-conditioned loss.

#### Worked example: converting a velocity to a denoised image

Suppose at $t = 0.25$ the network outputs velocity $v_\theta = [1.0, -0.5]$ at the point $x_t = [0.3, 0.2]$. What clean image does the model think it's heading toward, and what noise does it think it started from? Apply the formulas. Implied data: $x_1 = x_t + (1-t)v = [0.3, 0.2] + 0.75\,[1.0, -0.5] = [0.3 + 0.75,\, 0.2 - 0.375] = [1.05, -0.175]$. Implied noise: $x_0 = x_t - t\,v = [0.3, 0.2] - 0.25\,[1.0, -0.5] = [0.3 - 0.25,\, 0.2 + 0.125] = [0.05, 0.325]$. Sanity check that these reconstruct the path: $(1-t)x_0 + t x_1 = 0.75[0.05, 0.325] + 0.25[1.05, -0.175] = [0.0375 + 0.2625,\, 0.24375 - 0.04375] = [0.3, 0.2] = x_t$. Exactly back to $x_t$. So at *any* point along sampling you can ask a flow-matching model "what's your current best guess of the final image?" — it's $x_1 = x_t + (1-t)v_\theta$ — which is precisely the quantity you'd preview in a progressive-decode UI, and precisely the quantity classifier-free guidance extrapolates. The interconversion is not a curiosity; it's how guidance and previewing work on flow models.

### Stress test: what breaks, and when

Walk the failure modes, because knowing where a method cracks is half of using it well. **Too few steps without straightening.** Plain flow matching (1-rectified) at 2 Euler steps on a complex distribution still shows artifacts, because the marginal field has residual curvature from crossing pairs — the cure is reflow (§6), not just fewer steps on the base model. **Wrong time convention.** As noted, flipping $t$ without flipping the velocity sign runs the flow backward; the symptom is a model that confidently turns noise into noise (loss looks fine, samples are garbage), and it is the single most common flow-matching bug. **No resolution shift at high res.** Train a flow at 1024×1024 with a 256-tuned schedule and you get locally sharp but globally incoherent images — three-armed people, floating objects — because the high-noise steps that set composition were under-trained (§7). **Over-reflowing.** Each reflow slightly degrades the endpoint distribution if your synthetic coupling dataset is too small or your previous flow's sampler was too coarse; the symptom is mode shrinkage (less diversity) after the second or third reflow. Stop at the fewest reflows that hit your step budget. **CFG too high on a flow model.** Flow models guide the same way diffusion models do (extrapolate the conditional velocity away from the unconditional), and the same over-saturation failure appears past a guidance scale of ~7 — which is why FLUX bakes a *distilled* guidance constant into the weights rather than exposing the raw knob. Each of these is a direct consequence of something in the derivation; none is mysterious once you hold the straight-path picture.

## 5. Flow matching versus diffusion: the same ODE, a straighter path

A natural objection at this point: "isn't this just diffusion with extra steps?" It is a fair question, and the honest answer is that flow matching and diffusion are deeply related — they are both ways of learning an ODE that transports noise to data — but flow matching makes two specific choices that diffusion does not, and those choices are why it wins. Let me make the relationship precise, because half-understanding it is where a lot of confusion lives.

![A matrix comparing DDPM, flow matching, and 2-rectified flow across objective, path shape, velocity target, and steps to good FID](/imgs/blogs/flow-matching-and-rectified-flow-4.png)

Recall from the [score-based / SDE post](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view) that every diffusion model has an associated **probability-flow ODE** — a deterministic ODE whose marginals match the noising SDE's marginals at every time. DDIM is exactly the discretization of that ODE. So diffusion, at sampling time, is *already an ODE that transports noise to data*. Flow matching is also an ODE that transports noise to data. The difference is not "ODE versus not-ODE." The difference is **which path** the ODE follows and **what objective** trained it.

The DDPM variance-preserving forward process is, in the interpolation form, $x_t = \sqrt{\bar\alpha_t}\, x_1 + \sqrt{1 - \bar\alpha_t}\, x_0$ (using my data-at-$t{=}1$ convention, with $\bar\alpha_t$ the cumulative product of $1 - \beta$). Compare it term by term with the flow-matching path $x_t = t\, x_1 + (1-t)\, x_0$:

- **DDPM coefficients** are $\sqrt{\bar\alpha_t}$ and $\sqrt{1 - \bar\alpha_t}$ — these satisfy $\bar\alpha_t + (1 - \bar\alpha_t) = 1$, so the *squared* coefficients sum to one. That is the "variance-preserving" property, and it makes the path **curved**: as $t$ varies, the point traces an arc (a quarter-circle-like trajectory in the $(x_0, x_1)$ plane), because you are moving along a *circle* of constant radius, not a line.
- **Flow-matching coefficients** are $t$ and $1-t$ — these sum to one *linearly*, so the path is the straight chord between $x_0$ and $x_1$.

So **DDPM is a flow with a curved schedule.** It is not a different kind of object; it is the same noise-to-data ODE with the coefficients chosen so the path bends. You could even *retrain* a DDPM-style model as a flow by picking $\mu_t, \sigma_t$ to match $\sqrt{\bar\alpha_t}, \sqrt{1-\bar\alpha_t}$ — flow matching is a *generalization* that contains the diffusion path as one (curved) special case and the straight OT path as another. This is exactly the unification Lipman et al. point out: Gaussian conditional paths with different $(\mu_t, \sigma_t)$ recover variance-preserving diffusion, variance-exploding diffusion, *and* the straight OT flow, all from the one CFM objective.

Why does the straight path matter so much? Because of **discretization error in the sampler**. An Euler step assumes the velocity is constant over the step. If the true trajectory is a straight line at constant velocity, that assumption is *exact* — for the conditional path, Euler with one step would be perfect. The marginal field is not perfectly straight (it is an average of straight conditionals that can cross), but it is far straighter than the DDPM arc, so Euler error is small and few steps suffice. The curved DDPM path violates the constant-velocity assumption at every step, so coarse Euler overshoots, and you need many small steps (or a fancy higher-order solver like DPM-Solver from the [samplers post](/blog/machine-learning/image-generation/samplers-deep-dive)) to stay on the arc. Straightness is a *property of the path that directly buys steps*. That is the entire speed argument in one sentence.

The two objectives also differ in feel. DDPM's $\epsilon$-prediction loss comes out of a variational bound on the log-likelihood, with a per-timestep weighting $\lambda_t$ that you can derive but that is fiddly (min-SNR weighting, zero-terminal-SNR fixes — see the [noise-schedules post](/blog/machine-learning/image-generation/noise-schedules-and-the-parameterization-zoo)). Flow matching's velocity loss is just MSE onto $x_1 - x_0$, with a timestep weighting you choose directly rather than inherit from a bound. In practice this makes flow matching's loss *better-conditioned across timesteps* — the target magnitude $\|x_1 - x_0\|$ is roughly constant over $t$, whereas $\epsilon$-prediction's effective signal varies enormously between low-noise and high-noise timesteps, which is exactly why DDPM needs SNR-based loss reweighting to train stably. Less reweighting machinery, more stable gradients: that is the training-stability argument SD3's authors make.

Here is the comparison in one table — the kind I'd put on a slide to argue for flow matching in a model-architecture review.

| Property | DDPM / VP diffusion | Flow matching (OT path) | 2-rectified flow |
| --- | --- | --- | --- |
| Training objective | $\epsilon$-prediction (variational bound) | velocity MSE onto $x_1 - x_0$ | velocity MSE (re-paired data) |
| Forward / path | $\sqrt{\bar\alpha_t}x_1 + \sqrt{1-\bar\alpha_t}x_0$ (curved) | $t\,x_1 + (1-t)x_0$ (straight chord) | straightened, near-linear marginals |
| Velocity / target | noise $\epsilon$ (varies in scale over $t$) | $x_1 - x_0$ (constant, $t$-stable) | $x_1 - x_0$ on re-coupled pairs |
| Loss weighting | needs SNR / min-SNR reweighting | direct, simpler $t$-weighting | direct |
| Sampler | prob-flow ODE / SDE; curved | Euler / Heun on near-straight ODE | Euler, very few steps |
| Steps to good FID | ~30–50 (Euler) or ~15–20 (DPM++) | ~10–30 | 1–4 |
| Sampling speed corner | slowest of the three | faster (straighter) | fastest (one step possible) |

The takeaway is not "diffusion is bad." DPM-Solver++ and UniPC make curved-path diffusion sample in 15–20 steps with excellent quality, and a well-tuned SDXL is a formidable model. The takeaway is that flow matching gives you straighter paths *for free from the objective*, which stacks with good solvers and, crucially, sets up rectified flow's reflow to push toward one step. It is the better *foundation* to build speed on.

## 6. Rectified flow and reflow: straightening until one step works

Flow matching gives you a path that is straight *for each conditional pair*. But the **marginal** field — the thing the network actually learns and integrates at sampling time — is the average of those straight conditionals, and averages of straight lines are not themselves straight wherever the lines **cross**. Two pairs whose segments intersect at a point produce, at that point, a velocity that is the average of two different directions, and the resulting marginal trajectory bends to avoid the collision. So the marginal flow has some residual curvature, which is why naive flow matching still wants ~10–30 steps rather than one. Rectified flow (Liu, Gong & Liu, 2023) is the procedure that removes that residual curvature, and its core idea — **reflow** — is clever enough to deserve its own section.

![A stack figure showing reflow iterations re-pairing noise with the flow's own ODE endpoints to straighten trajectories until a single distilled step generates the image](/imgs/blogs/flow-matching-and-rectified-flow-5.png)

### Where the crossing comes from

The reason the marginal curves is that flow matching pairs each noise sample with a data sample **independently and at random** ($x_0 \sim$ noise, $x_1 \sim$ data, drawn separately). With independent pairing, a noise point on the left might be assigned a data point on the right, and a noise point on the right might be assigned a data point on the left, so their straight segments cross in the middle. The marginal ODE cannot let two trajectories actually pass through the same point going different directions (that would make it not a function), so it deforms — the trajectories curve around each other. Crossings $\Rightarrow$ curvature $\Rightarrow$ Euler error $\Rightarrow$ more steps. If we could pair each noise point with a data point such that the segments *don't* cross, the marginal flow would itself be (nearly) straight and one Euler step would nearly suffice.

### Reflow: re-pair using the flow's own transport

Rectified flow's insight: the trained flow already gives you a *non-crossing* coupling — its own ODE map. Run the trained flow's ODE forward from a noise sample $x_0$ and it deterministically lands on some data-side point $x_1' = \text{ODE}(x_0)$. This map $x_0 \mapsto x_1'$ is a deterministic transport, so by construction *its* trajectories do not cross (an ODE flow is a bijection at each time; trajectories never intersect). Now **re-train** a fresh flow on the re-coupled pairs $(x_0, x_1')$ — same straight-line interpolation, same velocity MSE, but the pairs are now the flow's own endpoints instead of random matches. This is one **reflow** iteration, and the new flow's marginal trajectories are *straighter* than the old one's, because the coupling it learned from has no crossings.

Formally: let $Z_0 \sim$ noise and let $\text{Flow}^{(k)}$ be the $k$-th rectified flow. Reflow defines

$$
(X_0, X_1)^{(k+1)} = \big(Z_0,\ \text{Flow}^{(k)}\text{-ODE}(Z_0)\big), \qquad \text{Flow}^{(k+1)} = \text{train CFM on } (X_0, X_1)^{(k+1)}.
$$

Liu et al. prove that each reflow step does not increase the transport cost and provably **reduces the trajectory curvature** (the paths get straighter in a precise sense — the marginal field's deviation from straight decreases). The endpoints' *distribution* is preserved (you still generate $p_{\text{data}}$), but the *coupling* between noise and data gets rectified into a near-deterministic, near-straight map. The naming follows the count: the flow-matching model is **1-rectified flow**, one reflow gives **2-rectified flow**, another gives 3-rectified flow, and so on. Two reflows is usually enough to get the paths straight enough that a single Euler step is a good approximation.

### Distillation: collapse to one step

Once the trajectories are nearly straight, the ODE map $x_0 \mapsto x_1$ is nearly a *linear* function of $t$ along each path, so you can **distill** it into a one-step generator: train a student network to map $x_0$ directly to the flow's endpoint $\text{ODE}(x_0)$ in a single forward pass. Because reflow made the teacher's paths straight, the one-step student has an easy target — it is approximating a near-affine map rather than a wildly curved one — and the quality drop from one-step generation is small. This is how rectified flow reaches **1-step** sampling: straighten with reflow, then distill the straightened map. (The closely related consistency-model line, which we cover in [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation), reaches few-step sampling a different way — by enforcing self-consistency along the ODE rather than straightening it first — but they target the same speed corner.)

### Why reflow provably helps

It is tempting to treat "reflow straightens paths" as a hopeful heuristic, but Liu et al. make it a theorem, and the structure of the argument is worth holding because it tells you exactly *what* reflow optimizes. Two properties are guaranteed at each reflow step. First, **the marginal distributions are preserved**: the new flow still maps noise to exactly $p_{\text{data}}$, because it is trained on pairs whose endpoints $\text{ODE}(x_0)$ are, by construction, distributed as $p_{\text{data}}$ (the previous flow already generated $p_{\text{data}}$). So you never lose the target distribution by reflowing — a critical guarantee, since a "faster" model that quietly drifts off the data distribution would be useless. Second, **the convex transport cost does not increase**: for any convex cost $c$, the expected cost $\mathbb{E}[c(X_1 - X_0)]$ under the reflowed coupling is no larger than under the previous coupling. Random independent pairing has high transport cost (you ship noise on the left to data on the right and vice versa); the deterministic ODE coupling has lower cost because it never crosses. Lower transport cost under a convex measure is, precisely, *straighter on average*.

The connection between "low transport cost" and "few sampling steps" is the punchline. The error of a one-step Euler integration of an ODE is governed by how much the velocity *changes* along the trajectory — the path's curvature, or equivalently the time-derivative of the velocity. For a perfectly straight constant-velocity path that derivative is zero and one Euler step is exact. Reflow drives the average squared deviation of the trajectories from straight lines monotonically toward zero, so it drives the one-step Euler error toward zero. That is the rigorous version of "straighter paths need fewer steps." It is not that reflow makes the model *better* in some vague sense; it makes the specific quantity that controls Euler discretization error — trajectory curvature — provably smaller, and that is exactly the quantity standing between you and one-step sampling.

#### Worked example: a crossing that reflow removes

Take the simplest case where independent pairing fails. In 1D, suppose the data is two points $\{-1, +1\}$ with equal mass, and noise is symmetric around 0. Independent pairing assigns, half the time, noise $x_0 = +0.5$ (right of center) to data $x_1 = -1$ (left), and noise $x_0 = -0.5$ (left) to data $x_1 = +1$ (right). Those two straight segments **cross** near the origin — at $x \approx 0$, one wants velocity $\approx -1.5$ (heading left) and the other $\approx +1.5$ (heading right). The marginal field at the origin averages them to roughly $0$, so a particle arriving at the origin *stalls* and the true marginal trajectory has to bend sharply to escape — high curvature, many steps. Now reflow: run the trained flow's ODE and it deterministically sends $x_0 = +0.5$ to $x_1 = +1$ and $x_0 = -0.5$ to $x_1 = -1$ (the *non-crossing* assignment — each noise point goes to the nearest mode, because that's what the smoothed marginal flow actually did). Retrain on those pairs and the segments no longer cross; the velocity field is now smooth and nearly constant along each path, and a single Euler step from $\pm 0.5$ lands near $\pm 1$. The crossing that forced curvature is gone, purely from re-coupling — no architecture change, no extra capacity, just a better pairing the flow handed you for free.

#### Worked example: the step-count collapse on a real model

The headline rectified-flow result, from Liu et al. (2023), is on CIFAR-10. Their **1-rectified flow** (plain flow matching) needs many Euler steps for good quality, like any flow. After **one reflow** (2-rectified flow), a *single* Euler step already produces recognizable, reasonable samples, and after distillation the **1-step** 2-rectified flow reaches an FID around **4.85** on CIFAR-10 — competitive with multi-step diffusion models that take 50–1000 network evaluations, from a single network call. The text-to-image version, **InstaFlow** (Liu et al., 2023), applied reflow + distillation to Stable Diffusion 1.5 and produced a **one-step** SD-quality text-to-image generator — the first one-step model derived from SD by straightening — with an FID on MS-COCO in the low-to-mid 20s from a single function evaluation, versus the 25+ steps SD1.5 normally needs. The number to internalize: reflow + distillation turned a ~25-step model into a ~1-step model with a modest quality cost, *purely by making the paths straight first.* That is the payoff of straightness, made quantitative.

A blunt caveat so you don't over-sell this: each reflow iteration costs a full retraining run *and* you must generate a large synthetic dataset of $(x_0, \text{ODE}(x_0))$ pairs from the previous flow (which means running the slow multi-step sampler many times to build the training set). Reflow trades offline compute for online speed. It is worth it when you serve a model at scale and amortize the one-time straightening cost over billions of fast inferences; it is *not* worth it for a one-off research model you sample a few hundred times. Know which regime you're in.

## 7. The production details: logit-normal sampling and the resolution shift

The math above gives you a model that trains. Getting a model that trains *well at megapixel resolution* — the SD3/FLUX regime — needs two more choices that look like footnotes but materially move FID. Both concern *how you sample the timestep $t$ during training* and *how you place the noise schedule as resolution grows*. SD3's paper (Esser et al., 2024, "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis") ablates these carefully, and the results are worth knowing because they are the difference between a flow-matching model that's merely fine and one that beats diffusion.

![A matrix of timestep-sampling and schedule choices in SD3 mapping each to its effect on training and the models that use it](/imgs/blogs/flow-matching-and-rectified-flow-7.png)

### Why uniform $t$ wastes capacity

The simplest thing is to sample $t \sim \mathcal{U}[0,1]$ uniformly, as in the toy loop's default. The problem: not all timesteps are equally hard or equally important. Near $t = 0$ (almost pure noise) and near $t = 1$ (almost pure data), the velocity target $x_1 - x_0$ is easy to predict — at $t \approx 1$ the input $x_t$ is basically the data, and at $t \approx 0$ it's basically the noise, so the network has lots of signal. The **middle** of the path, $t \approx 0.5$, is the hard part: the input is a 50/50 blend of noise and data, the network has the least information about which way to push, and errors here propagate most through the sampling ODE. Uniform sampling spends equal training budget on the easy ends and the hard middle, which under-trains the part that decides quality.

### Logit-normal sampling

SD3's fix is **logit-normal timestep sampling**: instead of uniform, draw $t$ so that it concentrates near the middle and thins toward the ends. Concretely, sample a Gaussian and squash it through a sigmoid:

$$
u \sim \mathcal{N}(m, s^2), \qquad t = \frac{1}{1 + e^{-u}} = \sigma(u).
$$

With $m = 0, s = 1$ this is a bell-shaped density on $(0,1)$ **peaked at $t = 0.5$** and tapering toward $0$ and $1$ — exactly the "spend more on the hard middle" weighting we want. (`logit_normal_t` in the training loop above implements precisely this.) SD3 found logit-normal timestep sampling consistently beat uniform and several alternatives (mode sampling, cosine-map sampling) in their large ablation. The parameters $m$ and $s$ let you slide the peak: a negative $m$ shifts emphasis toward $t=0$ (high noise), useful at high resolution, as we'll see. This is the flow-matching analogue of min-SNR loss weighting in diffusion — both say "don't train all timesteps equally" — but flow matching gets to express it as a clean sampling distribution over $t$ rather than a derived loss multiplier.

### The resolution shift

The second detail is subtler and very practical. As you scale to higher resolution, the *same* amount of added noise destroys *less* of the image's perceptual content, because high-resolution images have more redundant, locally-correlated pixels — a noise level that fully obscures a 256×256 image leaves a 1024×1024 image's coarse structure partly visible. Concretely, at fixed $t$ the signal-to-noise ratio of $x_t$ is *higher* at higher resolution, so the model under-trains the high-noise (low-$t$) regime that actually matters for getting global composition right, and large images come out with weak global structure — the classic "high-res but incoherent layout" failure.

SD3's fix is a **resolution-dependent timestep shift**: shift the sampled $t$ toward the noisy end (smaller $t$) as resolution grows, so the model spends more training and sampling budget on the high-noise steps that set global structure. They derive a shift that scales the timesteps by a factor tied to the sequence length (number of latent tokens), and FLUX uses a similar dynamic shift baked into its scheduler. The 🤗 `diffusers` `FlowMatchEulerDiscreteScheduler` exposes this directly — `use_dynamic_shifting`, a `base_shift`/`max_shift`, and a `base_image_seq_len`/`max_image_seq_len` so the shift is computed from the actual token count of the image you're generating. If you generate a 1024×1024 image with the shift turned off, you'll often see weaker composition than with it on; this is not a corner case, it is the standard high-res recipe.

```python
from diffusers import FlowMatchEulerDiscreteScheduler

# The scheduler that drives SD3 and FLUX sampling. It does NOT add noise the
# DDPM way; it places timesteps on the [0,1] flow path and Euler-integrates v_theta.
scheduler = FlowMatchEulerDiscreteScheduler(
    num_train_timesteps=1000,
    shift=3.0,                  # static schedule shift toward high noise
    use_dynamic_shifting=True,  # compute the shift from image token count
    base_shift=0.5,
    max_shift=1.15,
    base_image_seq_len=256,
    max_image_seq_len=4096,
)

# Resolution-aware: more latent tokens -> larger shift -> more high-noise budget.
mu = scheduler._sigma_to_t  # internals differ by version; the point is the shift
scheduler.set_timesteps(num_inference_steps=28, mu=1.0)  # 28 steps is SD3's default
print(scheduler.timesteps[:5])  # the discrete flow timesteps, high-noise first
```

The practical headline: SD3 ships with **28 inference steps** as its default with this scheduler, and FLUX.1-dev defaults to around **50 guidance-distilled steps** while FLUX.1-schnell — a *timestep-distilled* flow model — generates in **1–4 steps**. Those step counts are low *because the underlying paths are straight and the schedule is shifted to spend budget where it matters*, not because of a clever solver alone. The schedule choices in this section are how the clean theory of §2–§6 survives contact with a 1024×1024 image.

## 8. Case studies: SD3, FLUX, and the numbers that mattered

Theory and toy code are necessary but not sufficient; the reason flow matching is the default now is that it *shipped and won* at scale. Here are the real models and the numbers I'd cite in a design review, with sources, and with honest flags where a figure is approximate.

![A timeline from neural ODEs through FFJORD, flow matching, and rectified flow to the SD3 and FLUX frontier models built on the objective](/imgs/blogs/flow-matching-and-rectified-flow-8.png)

**Stable Diffusion 3 (Esser et al., Stability AI, 2024).** SD3 is the proof of concept at scale that rectified-flow training beats diffusion training for text-to-image. It pairs the flow-matching objective with the **MM-DiT** architecture (a multimodal diffusion transformer with separate text and image streams — the subject of the [next post](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe)), trained at sizes from 800M up to **8B** parameters. Their paper's central scaling result is the one that convinced the field: across model sizes, the rectified-flow models showed a *clean, monotone* relationship between validation loss and model scale, and validation loss tracked human-preference and benchmark scores — meaning you can predict quality from scale, the hallmark of a healthy objective. On the timestep-sampling ablation, **logit-normal** sampling was the consistent winner. SD3 uses **28 inference steps** by default with the flow-match Euler scheduler. The headline qualitative claim — better prompt adherence and typography than SDXL — is borne out on GenEval and human studies in the paper; I won't quote a single FID because FID famously undersells these models (the [evaluation post](/blog/machine-learning/image-generation/why-generating-images-is-hard) frame applies: FID and human preference diverge here), but the scaling-law cleanliness is the result that matters.

**FLUX.1 (Black Forest Labs, 2024).** FLUX is the current open-weights frontier and is also a **rectified-flow transformer**, from the team that built SD3's predecessor. It comes in three relevant flavors that map directly onto this post's ideas: **FLUX.1-pro** (the closed, highest-quality model), **FLUX.1-dev** (open weights, **12B** parameters, *guidance-distilled*, ~50 steps), and **FLUX.1-schnell** (open weights, **timestep-distilled** to generate in **1–4 steps**). "Schnell" — German for "fast" — is the flow-matching speed story made into a product: it is a flow model distilled down the step axis, the spiritual descendant of InstaFlow's reflow-and-distill, producing usable 1024×1024 images in a handful of network calls. FLUX uses the dynamic resolution shift discussed in §7; turn it off and high-res composition degrades. The 12B parameter count is itself a data point — flow matching scales to models far larger than SDXL's 2.6B U-Net while staying stable, which is part of why the frontier moved to it.

**InstaFlow / rectified flow (Liu et al., 2023).** The research lineage that proved one-step generation from straightening. On CIFAR-10, 2-rectified-flow distilled to **1 step** reaches FID ≈ **4.85** (Liu et al.), competitive with multi-step diffusion at a single network evaluation. **InstaFlow** applied the same reflow-then-distill recipe to SD1.5 and produced the first **one-step** SD-quality text-to-image model, with COCO FID in the low-to-mid 20s from one function evaluation — a roughly 25× reduction in network calls versus standard SD1.5 sampling, at a quality cost small enough to be useful for latency-bound serving. These are the numbers that establish that straightness *causes* the step reduction; the SD3/FLUX results establish that the same objective scales to the frontier.

For completeness, here is what *running* a flow-matching model looks like in 🤗 `diffusers` — the production inference path for both SD3 and FLUX. Notice that nothing in the call site screams "flow matching"; the objective lives in how the model was trained and in the scheduler, and the pipeline API is the same shape you'd use for SDXL. That uniformity is deliberate and is part of why flow matching slid into the ecosystem with so little friction.

```python
import torch
from diffusers import StableDiffusion3Pipeline, FluxPipeline

# --- SD3: a rectified-flow MM-DiT, ~28 steps by default ---
sd3 = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
)
sd3.enable_model_cpu_offload()           # fit on a 24GB consumer GPU
img = sd3(
    prompt="a golden retriever in a sunlit kitchen, photo",
    num_inference_steps=28,              # low because the path is straight
    guidance_scale=7.0,
).images[0]

# --- FLUX.1-schnell: timestep-distilled, 1-4 steps, no CFG knob ---
flux = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
)
flux.enable_model_cpu_offload()
img_fast = flux(
    prompt="a golden retriever in a sunlit kitchen, photo",
    num_inference_steps=4,               # schnell = "fast"; straightened + distilled
    guidance_scale=0.0,                  # guidance is baked in, not a runtime knob
).images[0]
```

The `num_inference_steps=4` on FLUX.1-schnell is the whole post in one line: four network calls produce a 1024×1024 image because the underlying flow was straightened (reflow lineage) and distilled down the step axis, and the scheduler places those four timesteps on the flow path with the resolution shift. The `guidance_scale=0.0` is the other tell — schnell has guidance distilled into the weights, so there is no second unconditional pass per step, halving the per-step cost on top of the step reduction. A unifying observation across all three models: flow matching did not win by being a better *sampler* (DPM-Solver already made diffusion fast) or a better *architecture* (DiT is orthogonal — you can train a DiT with either objective). It won by being a better *training objective* — straighter paths, a $t$-stable target, stable scaling — that makes every downstream choice (sampler, distillation, scaling) easier. That is why it propagated through the whole frontier in a single year.

## 9. When to reach for flow matching (and when not to)

A technique is only as good as your judgment about when to use it. Here is the decisive version.

**Reach for flow matching when** you are training a new text-to-image (or image, or latent) generative model from scratch in 2025–2026. It is the current default for good reasons: straighter sampling paths (fewer steps for the same quality), a velocity target whose scale is stable across timesteps (less loss-reweighting fuss than $\epsilon$-prediction's SNR juggling), and clean, predictable scaling that SD3 demonstrated up to 8B and FLUX past 12B. If you're picking an objective for a serious model and have no specific reason to do otherwise, pick flow matching with the OT path, logit-normal $t$ sampling, and a resolution shift.

**Reach for rectified flow / reflow when** you need *very* few sampling steps (1–4) and you can afford the one-time straightening cost: generating the synthetic $(x_0, \text{ODE}(x_0))$ coupling dataset and retraining once or twice, then distilling. This is a serving-scale decision — it pays off when you amortize the offline straightening over enormous inference volume (a hosted product), and it is the path FLUX.1-schnell walked. The break-even is roughly: if you'll sample the model millions of times, straighten; if hundreds, don't bother.

**Do *not* reach for flow matching / reflow when:**

- **You already have a fine SDXL/diffusion model and a fast solver.** A DDPM-trained SDXL with DPM-Solver++ or UniPC at 15–20 steps is excellent and battle-tested, with a massive ControlNet/LoRA ecosystem. Switching objectives to shave a few steps is not worth abandoning that ecosystem unless you're building fresh. Flow matching is a *training-time* choice; you can't retrofit it onto an $\epsilon$-trained checkpoint without retraining.
- **You only ever sample a model a few times.** Reflow's offline cost (synthetic dataset + retrains) dwarfs any online savings if you're not serving at scale. Use the multi-step flow directly.
- **You're chasing one-step quality and can't accept the drop.** One-step rectified/distilled models trade a real, if modest, quality cost for speed. If you need maximum fidelity and can spend 20–28 steps, the multi-step flow (or a strong diffusion model) still edges out the one-step student. Match the step count to the latency budget, not to bragging rights.
- **Your bottleneck is the VAE or the text encoder, not the denoiser.** If decoding the latent or encoding a long T5 prompt dominates your latency, making the denoiser one-step buys you little. Profile before you straighten.

The meta-rule: flow matching is the better *foundation* for a new model and the better *substrate* for aggressive step reduction, but it is a training-time commitment, not a drop-in sampler swap. Decide at the start of a training run, not at inference time.

## 10. Key takeaways

- **Continuous normalizing flows had the right object (a learned ODE transporting noise to data) and the wrong training procedure (simulating that ODE every step).** The per-step cost — a full ODE solve plus a divergence trace plus backprop through the solver — killed them at image scale.
- **Flow matching trains the same ODE without simulating it.** Pick a probability path, write down a closed-form conditional velocity, and regress the network onto it with plain MSE. No solver, no divergence, no adjoint at training time.
- **The straight (optimal-transport) path is $x_t = (1-t)x_0 + t x_1$, and its velocity target is the constant $v = x_1 - x_0$.** The point moves from noise to data along a line at constant speed, so its velocity is just the displacement — derivable in one line.
- **Conditional flow matching gives the correct marginal field for free.** Regressing onto per-pair conditional velocities has the *same gradient* as regressing onto the intractable marginal field, because the MSE's expectation averages the conditionals into the marginal — the same cancellation that made denoising score matching cheap.
- **DDPM is flow matching with a curved schedule.** Both are noise-to-data ODEs; flow matching just chooses straight coefficients ($t$, $1-t$) instead of curved ones ($\sqrt{\bar\alpha_t}$, $\sqrt{1-\bar\alpha_t}$), and straightness directly cuts the number of Euler steps a sampler needs.
- **Rectified flow's reflow straightens the marginal trajectories** by re-pairing each noise sample with its own ODE endpoint and retraining; iterate, then distill, to reach 1–4 step (even 1-step) sampling. It trades offline compute for online speed.
- **Production flow matching needs logit-normal timestep sampling and a resolution shift.** Sample $t$ concentrated on the hard middle, and shift toward high noise as resolution grows, or large images lose global structure. The `FlowMatchEulerDiscreteScheduler` implements both.
- **SD3 and FLUX adopted flow matching because it is a better objective, not a better sampler or architecture:** straighter paths, a $t$-stable target, less reweighting fuss, and clean scaling to 8B–12B+ parameters. That is why the whole frontier moved to it in one year.

## 11. Further reading

- **Lipman, Chen, Ben-Hamu, Nickel & Le (2023), "Flow Matching for Generative Modeling."** The paper that introduced conditional flow matching and proved the marginal-from-conditional gradient identity. The source for §2–§3.
- **Liu, Gong & Liu (2023), "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow."** Rectified flow and the reflow procedure; the CIFAR-10 one-step results.
- **Liu et al. (2023), "InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation."** Reflow + distillation applied to Stable Diffusion 1.5 for one-step text-to-image.
- **Esser et al. (Stability AI, 2024), "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis."** The SD3 paper: the timestep-sampling ablation, the resolution shift, and the scaling-law evidence. The source for §7–§8.
- **Chen, Rubanova, Bettencourt & Duvenaud (2018), "Neural Ordinary Differential Equations,"** and **Grathwohl et al. (2019), "FFJORD."** The continuous-normalizing-flow background and the simulation cost flow matching removed.
- **🤗 `diffusers` documentation:** `FlowMatchEulerDiscreteScheduler`, `StableDiffusion3Pipeline`, and `FluxPipeline` — the production APIs that run flow matching.
- **Within this series:** [normalizing flows and the change of variables](/blog/machine-learning/image-generation/normalizing-flows-and-change-of-variables) (the CNF setup), [score-based models and the SDE view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view) (the probability-flow ODE), [DDIM and fast deterministic sampling](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling), [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles), the forward-looking [MM-DiT and the modern text-to-image recipe](/blog/machine-learning/image-generation/mmdit-and-the-modern-text-to-image-recipe) (where SD3/FLUX put flow matching to work), [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation), and the capstone [building an image-generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
