---
title: "Samplers Deep Dive: Euler, Heun, DPM-Solver, and the Step-Count Pareto"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Understand diffusion samplers as numerical ODE/SDE solvers, derive why Heun halves Euler's error, see the exponential-integrator trick behind DPM-Solver, and walk away knowing exactly which scheduler and step count to use."
tags:
  [
    "image-generation",
    "diffusion-models",
    "samplers",
    "dpm-solver",
    "ode-solvers",
    "edm",
    "generative-ai",
    "deep-learning",
    "diffusers",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/samplers-deep-dive-1.png"
---

Here is a moment every diffusion practitioner lives through. You load Stable Diffusion XL, type a prompt, set `num_inference_steps=50`, and get a beautiful image in about four seconds on an RTX 4090. Then your product manager asks the obvious question: can it be faster? You drop to 20 steps and the image is still great. You drop to 8 steps and it is passable. You drop to 4 steps with the same scheduler and you get gray, soupy mush — a blurry impression of an image, like a photograph developed for one second. Same model. Same weights. Same prompt. The *only* thing you changed was the number of times you called the network, and the quality fell off a cliff.

That cliff is the subject of this entire post, and the surprising truth is that it has almost nothing to do with the model and almost everything to do with **the sampler** — the small piece of code that decides how to walk from pure noise to a clean image. The trained network does not produce an image. It produces, at each point along the way, a *direction*: which way to nudge the current noisy latent to make it slightly less noisy. The sampler is the thing that takes those directions and actually integrates them into a trajectory. It is, quite literally, a numerical differential-equation solver. And just like any numerical solver, some are crude and need many tiny steps to stay accurate, while others are clever and reach the same answer in a fraction of the steps. The difference between Euler at 50 steps and DPM-Solver++ at 15 steps is not magic — it is the difference between the rectangle rule and a high-order Runge–Kutta scheme that you learned about (and probably forgot) in a numerical methods class.

![A graph showing pure noise feeding a denoiser network whose output defines an ODE or an SDE that a numerical solver integrates step by step into a clean image](/imgs/blogs/samplers-deep-dive-1.png)

By the end of this post you will be able to: frame sampling as solving the **probability-flow ODE** (the deterministic view) or the **reverse-time SDE** (the stochastic view); implement first-order **Euler** (which is exactly DDIM in ODE form) and second-order **Heun** samplers by hand over a pretrained ε- or v-prediction model in PyTorch, and explain why Heun's local error is $O(h^3)$ where Euler's is $O(h^2)$; understand the **exponential-integrator trick** that lets DPM-Solver and DPM-Solver++ exploit the diffusion ODE's semi-linear structure to reach quality in 10–20 steps; reason about **UniPC**, **ancestral/SDE samplers** (Euler-a), and the **EDM** framework's preconditioning and sampling schedule; and — the payoff — pick the right `diffusers` scheduler class and step count for any budget, with a defensible decision at each branch.

This is the seventh post in the diffusion-engine track of our image-generation series. It builds directly on [DDIM and fast deterministic sampling](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling), which introduced the deterministic, non-Markovian sampling that turns 1000 steps into 50, and on [score-based models and the SDE view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view), which is where the probability-flow ODE and the reverse SDE come from. If you have not internalized that diffusion sampling is *solving a differential equation*, read those first; this post lives entirely inside that frame. Keep the series spine — the **generative trilemma** of sample quality versus mode coverage versus sampling speed — in your head throughout. Samplers are the single most direct lever on the *speed* axis, and the central drama of this post is exactly how far you can push speed before quality and diversity start to pay the bill.

## 1. Sampling is solving a differential equation

Let me make the central claim precise, because everything downstream is a consequence of it.

A trained diffusion model gives you a denoiser. Depending on the parameterization (covered in detail in [noise schedules and the parameterization zoo](/blog/machine-learning/image-generation/noise-schedules-and-the-parameterization-zoo)), the network predicts the noise $\epsilon_\theta(x_t, t)$, or the clean sample $x_\theta$, or the velocity $v_\theta$. All three are linear reparameterizations of the same underlying object: the **score function** $\nabla_x \log p_t(x)$, the gradient of the log-density of the data convolved with noise at level $t$. The score points "uphill" toward regions of higher data probability, which is exactly the direction you want to move a noisy sample to make it cleaner.

Song et al. (2021) showed that the reverse of the forward noising process can be written as a continuous-time stochastic differential equation:

$$
dx = \left[f(x, t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t)\, d\bar{w}
$$

where $f$ is the drift of the forward process, $g$ is its diffusion coefficient, and $d\bar{w}$ is a reverse-time Wiener process (Brownian motion run backward). This is the **reverse-time SDE**: run it from $t=T$ (pure noise) down to $t=0$ and you produce a sample from the data distribution. The $d\bar{w}$ term keeps injecting fresh randomness as you go.

The remarkable second result is that there is a *deterministic* ODE — the **probability-flow ODE** (PF-ODE) — whose trajectories have the **same marginal distributions** $p_t(x)$ at every time $t$ as the SDE, but with no noise term:

$$
\frac{dx}{dt} = f(x, t) - \tfrac{1}{2} g(t)^2 \nabla_x \log p_t(x)
$$

This is the equation that matters most for fast sampling. It says: starting from a noise sample $x_T$, if I integrate this ODE backward in time to $t=0$, I land on a clean image, and the map from $x_T$ to $x_0$ is a deterministic function. There is no randomness once the initial noise is fixed. **That is the whole reason fast deterministic sampling is possible.** A noise-free ODE can be integrated with the entire arsenal of numerical ODE solvers — and good solvers need far fewer evaluations than naively simulating an SDE.

Substitute in the network. For the common variance-preserving (VP) formulation with the ε-parameterization, the score is $\nabla_x \log p_t(x) = -\epsilon_\theta(x_t, t)/\sigma_t$, where $\sigma_t$ is the noise standard deviation at time $t$. Plug that into the PF-ODE and you get a concrete, integrable equation: the right-hand side is computable any time you evaluate the network. **The sampler's only job is to integrate this right-hand side from noise to image.** Each network evaluation gives you the slope at the current point; the solver decides how to take a step.

This reframing is liberating. It means every question you have about samplers — why does Euler give mush at 4 steps, why is Heun more accurate, why does DPM-Solver need fewer steps, why does Euler-a never quite converge — is a question about numerical integration, with a century of answers behind it. The figure above shows the loop: noise enters, the network supplies the field, the field defines an ODE (deterministic) or an SDE (stochastic), and a solver integrates it. The accuracy of that integration, controlled by the step count, is the knob you turn. **Truncation error**, the gap between the discrete solver's path and the true continuous trajectory, is what produces mush when steps are too few.

A note on terminology that trips people up. The diffusion literature counts a step as one **network function evaluation** (NFE), and this is the cost unit that matters — each NFE is one full forward pass through a billion-parameter U-Net or DiT, which dominates latency. A first-order solver uses one NFE per step. A second-order solver like Heun uses two NFEs per step. So "Heun at 25 steps" and "Euler at 50 steps" cost the *same* — 50 NFEs each. Whenever we compare samplers, the honest axis is NFE, not "steps," because a step means different things for different orders. We will be careful about this throughout, because it is the single most common way sampler benchmarks mislead.

## 2. Euler: the rectangle rule, and why it is DDIM

The simplest possible ODE solver is **forward Euler**. Given $\frac{dx}{dt} = F(x, t)$ and a step size $h = t_{n+1} - t_n$ (negative, since we integrate backward), Euler approximates the next point by assuming the slope is constant across the whole step:

$$
x_{n+1} = x_n + h \cdot F(x_n, t_n)
$$

You evaluate the field once, at the start of the step, and follow that straight line all the way across. It is the rectangle rule for integration: assume the integrand does not change over the interval. For diffusion, $F$ is the PF-ODE right-hand side, so one Euler step is: evaluate the network once at $(x_n, t_n)$, form the ODE slope, take a straight step to the next noise level.

Here is the fact that surprises people the first time: **deterministic DDIM is exactly forward Euler on the probability-flow ODE.** The DDIM update, derived in our [DDIM post](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling) from a non-Markovian forward process, when you write it in the right coordinates (the log-SNR or the $\sigma$-parameterization), is algebraically identical to a single Euler step of the PF-ODE. They are the same algorithm discovered from two different directions — one from a clever non-Markovian construction, one from "just solve the ODE." This is why `EulerDiscreteScheduler` and `DDIMScheduler` produce nearly identical images in `diffusers` at the same step count. They *are* the same first-order method.

Now, the cost of being first-order. The **local truncation error** of Euler — the error introduced in a single step — comes from Taylor-expanding the true solution:

$$
x(t_{n+1}) = x(t_n) + h\, x'(t_n) + \tfrac{1}{2} h^2 x''(t_n) + O(h^3)
$$

Euler captures the first two terms ($x_n$ and $h \cdot x'$) but drops everything from $\tfrac{1}{2}h^2 x''$ onward. So the local error per step is $O(h^2)$. Over a full integration from $T$ to $0$ you take roughly $N = (T-0)/|h|$ steps, and the errors accumulate, giving a **global error** of $O(h) = O(1/N)$. Halve the step size — double the steps — and you roughly halve the error. This linear-in-$1/N$ convergence is *slow*. It is exactly why Euler needs 50 steps where a higher-order method needs 15: to push the error down by a factor of three, Euler needs three times the steps, whereas a second-order method needs only the square root of that.

Let me make the curvature concrete, because "drops the $x''$ term" is abstract. The PF-ODE trajectory is not a straight line in latent space. Early in sampling (high noise), the denoiser's estimate of the clean image is vague and changes rapidly as the latent moves; the trajectory is sharply curved there. Euler, assuming a constant slope, overshoots or undershoots on every curved segment. At 50 steps each segment is short enough that the curvature within it is small and the straight-line approximation is fine. At 4 steps each segment spans a huge change in noise level, the trajectory curves dramatically within it, and the straight-line steps fly off the true path — accumulating into the gray mush you saw in the intro. **The mush is accumulated truncation error from approximating a curved path with a few long straight chords.**

Here is a complete, runnable Euler sampler over a pretrained ε-prediction model, written against the `diffusers` building blocks so you can see every line of the integration:

```python
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler

device = "cuda"
unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet",
    torch_dtype=torch.float16,
).to(device)
vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae",
    torch_dtype=torch.float16,
).to(device)

@torch.no_grad()
def euler_sampler(unet, scheduler, prompt_embeds, added_cond, num_steps=20,
                  guidance_scale=7.0, shape=(1, 4, 128, 128), device="cuda"):
    scheduler.set_timesteps(num_steps, device=device)
    # Start from noise scaled to the scheduler's initial sigma.
    x = torch.randn(shape, dtype=torch.float16, device=device)
    x = x * scheduler.init_noise_sigma

    for i, t in enumerate(scheduler.timesteps):
        # Classifier-free guidance: run conditional and unconditional in one batch.
        x_in = torch.cat([x, x])
        x_in = scheduler.scale_model_input(x_in, t)  # sigma-scaling for this t
        eps = unet(x_in, t, encoder_hidden_states=prompt_embeds,
                   added_cond_kwargs=added_cond).sample
        eps_uncond, eps_cond = eps.chunk(2)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # One Euler step of the PF-ODE. The scheduler folds the sigma geometry
        # and the eps->slope conversion into .step(); the math underneath is
        # x_{n+1} = x_n + (sigma_{n+1} - sigma_n) * d, with d the ODE slope.
        x = scheduler.step(eps, t, x).prev_sample
    return x
```

The `scheduler.step()` call hides the per-step Euler update, but it is doing exactly what the math above says: convert the predicted noise into the ODE slope $d = (x - \hat{x}_0)/\sigma$, then take a straight step of size $(\sigma_{n+1} - \sigma_n)$ along $d$. If you want to see it with no scheduler at all, the bare update is `x = x + (sigma_next - sigma) * d` after computing `x0_hat = x - sigma * eps` and `d = (x - x0_hat) / sigma`. That is the entire first-order method. Everything else in this post is about doing better than that straight chord.

It is worth dwelling on *why* the slope has the form $d = (x - \hat{x}_0)/\sigma$, because it makes the whole geometry click. In the EDM $\sigma$-parameterization the ODE for the noise-perturbed sample is $\frac{dx}{d\sigma} = \frac{x - D_\theta(x;\sigma)}{\sigma}$, where $D_\theta$ is the denoiser's estimate of the clean image. Read that right-hand side: it is the direction *away* from the current clean-image estimate, scaled by $1/\sigma$. An Euler step in $\sigma$ therefore moves the sample along the line connecting it to the denoiser's current guess of $x_0$, by an amount proportional to how much you lower the noise level. When $\sigma$ is large, $\hat{x}_0$ is a blurry, low-confidence guess and that line points in a direction that will be substantially *revised* once you re-evaluate the network at the new, lower noise level — which is precisely the revision Euler ignores until the *next* step. The trajectory bends because $\hat{x}_0$ keeps changing as the network sees a cleaner input; a single Euler step pretends it does not. That is the geometric content of "drops the $x''$ term": $x''$ is literally the rate at which the denoiser's clean-image estimate revises as you move, and Euler is blind to it within a step.

This also explains a subtlety you will hit in practice: the schedule of $\sigma$ values matters as much as their count. Because the trajectory curves most where $\hat{x}_0$ revises fastest — typically at *low-to-mid* noise, where the network is resolving real structure rather than coarse layout — a uniform spacing of $\sigma$ wastes steps at high noise (where the path is nearly straight and one big step is fine) and starves the low-noise region (where the path bends and you need fine steps). This is exactly the observation the EDM sampling schedule formalizes in section 8, and it is why naively spacing your timesteps uniformly is one of the most common ways to leave quality on the table at a fixed step budget.

## 3. Heun: one correction step that halves the error

If the problem with Euler is that it trusts a single slope across a curved step, the obvious fix is to *look ahead*. Take a tentative Euler step to the end of the interval, measure the slope *there*, and then redo the step using the **average** of the start slope and the end slope. This is **Heun's method**, also called the improved Euler method or a second-order Runge–Kutta scheme, and it is exactly the second-order sampler that Karras et al. (2022) adopt in the EDM paper.

The two-stage update, per step:

$$
\begin{aligned}
\tilde{x}_{n+1} &= x_n + h\, F(x_n, t_n) \quad &\text{(Euler predictor)}\\
x_{n+1} &= x_n + \tfrac{h}{2}\big[F(x_n, t_n) + F(\tilde{x}_{n+1}, t_{n+1})\big] \quad &\text{(trapezoidal corrector)}
\end{aligned}
$$

The first line is a throwaway Euler probe to find out where the trajectory is heading. The second line is the real step: it uses the trapezoidal rule, averaging the slope at the start and the slope at the (predicted) end. Two slopes, so **two network evaluations per step**. The figure below contrasts the two methods side by side: Euler's single blind slope versus Heun's predict-then-average.

![A before and after figure contrasting Euler taking one blind slope across a step against Heun predicting an endpoint, averaging two slopes, and achieving cubic local error](/imgs/blogs/samplers-deep-dive-2.png)

Why does averaging two slopes help so much? Taylor expansion again. The trapezoidal rule is constructed precisely so that the $\tfrac{1}{2}h^2 x''$ term that Euler dropped is now *captured*. To see it, expand $F(\tilde{x}_{n+1}, t_{n+1})$ around $t_n$: to first order it equals $F(x_n,t_n) + h \frac{dF}{dt} + O(h^2)$, and $\frac{dF}{dt} = x''$. Averaging $F(x_n,t_n)$ with this gives $F(x_n,t_n) + \tfrac{1}{2}h\, x'' + O(h^2)$, so the step becomes $x_n + h\,x' + \tfrac{1}{2}h^2 x'' + O(h^3)$ — which matches the true Taylor series through the quadratic term. The local truncation error is now $O(h^3)$, one order better than Euler's $O(h^2)$, and the global error is $O(h^2) = O(1/N^2)$.

The jump from *local* to *global* error is worth one sentence of rigor, because it is where the convergence order actually shows up in the image. Local error is what one step adds; global error is what survives at $t=0$ after all $N$ steps accumulate. A standard result for one-step methods says the global error is one order *lower* than the local error, because you take $N = O(1/h)$ steps and each contributes its local error: $N \times O(h^{p+1}) = O(h^p)$. So Euler's $O(h^2)$ local error becomes $O(h) = O(1/N)$ global; Heun's $O(h^3)$ local error becomes $O(h^2) = O(1/N^2)$ global. This is not bookkeeping — it is the reason the *image* at the end of a 10-step Euler run is visibly worse than a 10-step Heun run: the per-step chord errors do not cancel, they compound along the trajectory, and a first-order method's compounded error at the final clean image is the gray, washed-out quality you can see with your eyes.

That quadratic convergence is the whole prize. Euler's error scales like $1/N$; Heun's like $1/N^2$. To cut the error by a factor of four, Euler needs four times the steps; Heun needs only two times the steps. **The catch is the 2× NFE cost per step.** So the honest comparison is at fixed NFE budget. Heun at 25 steps (50 NFE) beats Euler at 50 steps (50 NFE) on most diffusion ODEs because the per-step error reduction more than pays for the doubled cost — *as long as the trajectory is curved enough to benefit*. On nearly straight segments (low noise, near the end of sampling) the second slope is almost identical to the first and the extra NFE is wasted. This is why production Heun implementations skip the corrector on the final step (the EDM sampler does exactly this: the last step is a plain Euler step, because the corrector buys nothing there).

Here is a hand-written Heun sampler that makes the predictor–corrector structure explicit. It works on the same ε-model; the only addition over Euler is the second evaluation and the averaging:

```python
@torch.no_grad()
def heun_sampler(model_fn, sigmas, x, device="cuda"):
    # model_fn(x, sigma) -> denoised x0 estimate (Karras/EDM convention).
    # sigmas: descending noise levels, ending at 0.0 (length num_steps + 1).
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]

        # --- Predictor: one Euler step ---
        denoised = model_fn(x, sigma)               # 1st NFE
        d = (x - denoised) / sigma                   # ODE slope at start
        x_euler = x + (sigma_next - sigma) * d

        if sigma_next == 0:
            # Last step: corrector buys nothing on the final (straight) leg.
            x = x_euler
        else:
            # --- Corrector: average the start slope and the endpoint slope ---
            denoised_next = model_fn(x_euler, sigma_next)   # 2nd NFE
            d_next = (x_euler - denoised_next) / sigma_next
            d_avg = 0.5 * (d + d_next)
            x = x + (sigma_next - sigma) * d_avg
    return x
```

Notice the structure mirrors the equations exactly: `d` is the start slope, `d_next` is the endpoint slope at the *predicted* point, and `d_avg` is the trapezoidal average. The `if sigma_next == 0` guard is the production trick of dropping the corrector on the last step. In `diffusers` you do not write this by hand; you select `HeunDiscreteScheduler` and it does precisely this, including the last-step Euler fallback. But writing it once makes it concrete that a "2nd-order sampler" is nothing more exotic than predict-then-average.

#### Worked example: Heun versus Euler at a fixed budget on CIFAR-10

The EDM paper (Karras et al., 2022) reports the cleanest controlled comparison. On unconditional CIFAR-10 with their VP model, the deterministic samplers reach FID around 1.79 (the EDM headline) using the second-order Heun solver with a tuned $\sigma$-schedule at **35 NFE** (effectively ~18 Heun steps with the last-step Euler fallback). A first-order Euler solver on the same model needs substantially more NFE — on the order of 2× — to approach the same FID, and at very low NFE it plateaus at a visibly worse FID that the second-order method clears. The qualitative takeaway you can carry to any model: at a fixed NFE budget in the 30–60 range, the 2nd-order Heun solver gives a lower FID than 1st-order Euler, because the quadratic convergence pays for the doubled per-step cost. The crossover flips only at *very* low NFE (≤ 10), where you cannot afford even one corrector and a smarter first-order-cost method (next section) wins.

## 4. The exponential integrator: how DPM-Solver exploits structure

Heun is a *generic* ODE solver — it knows nothing about the specific structure of the diffusion ODE. It treats $F(x,t)$ as a black box. But the diffusion ODE is not a generic ODE. It is **semi-linear**: its right-hand side splits into a part that is *linear* in $x$ and a part that is a nonlinear function of $x$ through the network. And there is a beautiful classical idea — the **exponential integrator** — that says: if part of your ODE is linear, do not discretize it. Solve it *exactly*, in closed form, and only approximate the genuinely nonlinear residual. This is the engine inside DPM-Solver (Lu et al., 2022) and its successor DPM-Solver++, the workhorse samplers behind most 10–20 step generation today.

Let me show the structure. Write the diffusion ODE in the noise prediction form. Using the half-log-SNR variable $\lambda_t = \log(\alpha_t/\sigma_t)$ — a monotonic reparameterization of time that turns the messy noise schedule into a clean clock — the ODE for the VP process becomes:

$$
\frac{dx}{d\lambda} = \underbrace{\frac{d\log\alpha}{d\lambda}\, x}_{\text{linear in } x} \;-\; \underbrace{\sigma_\lambda\, \frac{d\lambda}{d\lambda}\,\epsilon_\theta(x, \lambda)}_{\text{nonlinear, through the net}}
$$

The first term is linear in $x$ — it is just a scaling. The second term is the nonlinear part, where the network lives. The exact solution over a step from $\lambda_n$ to $\lambda_{n+1}$ can be written with the **variation-of-constants formula** (the standard trick for solving linear ODEs with a forcing term):

$$
x_{n+1} = \frac{\alpha_{n+1}}{\alpha_n} x_n - \alpha_{n+1}\int_{\lambda_n}^{\lambda_{n+1}} e^{-\lambda}\, \epsilon_\theta(x_\lambda, \lambda)\, d\lambda
$$

Look at what happened. The $\frac{\alpha_{n+1}}{\alpha_n} x_n$ term is **exact** — it is the closed-form solution of the linear part, with *zero* discretization error, no matter how large the step. The only thing left to approximate is the integral of $e^{-\lambda}\epsilon_\theta$ — and crucially, $\epsilon_\theta$ as a function of $\lambda$ is *smooth and slowly varying* compared to the raw ODE right-hand side. The figure below lays out the trick: split the field, solve the linear part exactly, and Taylor-expand only the smooth residual.

![A stack diagram showing the semi-linear diffusion ODE split into a linear part solved exactly and a nonlinear residual approximated by a low-order Taylor expansion in log-SNR](/imgs/blogs/samplers-deep-dive-3.png)

The DPM-Solver family approximates that remaining integral by a low-order Taylor expansion of $\epsilon_\theta$ in $\lambda$. **DPM-Solver-1** uses a zeroth-order (constant) approximation of $\epsilon_\theta$ over the step — and remarkably, this turns out to be *exactly DDIM again*. So DDIM is simultaneously first-order Euler on the PF-ODE and the first-order DPM-Solver; the methods converge. **DPM-Solver-2** uses a first-order (linear) approximation, needing one extra evaluation. **DPM-Solver-3** uses a quadratic approximation. Each higher order captures more of the integral's curvature with the same exact-linear-part backbone.

Why is this dramatically better than Heun for the same NFE? Because Heun spends its accuracy fighting the *entire* ODE right-hand side, including the stiff linear part that swings violently at high noise. DPM-Solver has already killed the linear part exactly, so its Taylor expansion only has to track the gentle, smooth $\epsilon_\theta(\lambda)$ residual. The same polynomial order buys far more accuracy when applied to a smooth function than to a stiff one. **This is the core reason DPM-Solver reaches good quality in 10–15 steps where generic Heun needs 30 and Euler needs 50.** It is not a heuristic; it is the structural payoff of solving the linear part in closed form.

Let me make the "smooth residual" claim concrete, because it is the crux and it is easy to wave past. A generic solver applied to the raw VP-SDE-derived ODE has to track a right-hand side that contains the factor $\frac{\dot\alpha_t}{\alpha_t}x_t$, and $\alpha_t$ swings from near $1$ at low noise to near $0$ at high noise — the coefficient is *stiff*, meaning it changes by orders of magnitude across the integration interval. Stiffness is the classical enemy of explicit solvers: an explicit method needs tiny steps wherever the coefficient is large just to stay stable, regardless of how much accuracy you actually need there. By moving to the log-SNR clock $\lambda$ and solving the $\alpha$-scaling in closed form, the exponential integrator removes the stiff factor entirely from what gets discretized. What remains, $\epsilon_\theta(x_\lambda, \lambda)$ as a function of $\lambda$, is a *learned, smooth* function — the neural network's noise prediction varies gently as you slide along the noise schedule, with no $1/\alpha$ blowup. A low-order polynomial fits a gently-varying function with small error over a *large* interval, which is exactly what lets the step sizes be large. The lesson generalizes beyond diffusion: whenever an ODE is semi-linear with a stiff linear part, the right move is an exponential integrator, not a smaller step size. Diffusion sampling is one of the cleanest large-scale applications of that classical principle.

There is a second-order subtlety in DPM-Solver-2 worth naming, because it parallels Heun and shows the unity of these methods. To get the linear (first-order in $\lambda$) Taylor coefficient of the residual, DPM-Solver-2 needs the *derivative* of $\epsilon_\theta$ with respect to $\lambda$, which it estimates with one intermediate network evaluation at a midpoint — structurally identical to Heun's predictor probe. So DPM-Solver-2 singlestep is, in spirit, "Heun on the exponentially-integrated residual": same predict-an-extra-point machinery, but applied to the smooth residual rather than the stiff full field, which is why it is more accurate than Heun at equal NFE. The multistep DPM++ 2M then replaces that intermediate evaluation with a finite difference over the *previous* step's stored $\epsilon$, recovering the same derivative information at zero extra NFE — the move that makes it the practical default.

### Multistep versus singlestep, and the ++ variants

There are two ways to get the extra derivative information that higher orders need. **Singlestep** methods (the original DPM-Solver-2/3) take intermediate evaluations *within* a step, like Heun's predictor — accurate but spending extra NFE per step. **Multistep** methods reuse the network evaluations from *previous* steps to estimate the derivative, the way a multistep linear-multistep integrator (Adams–Bashforth) does. Multistep is the practical winner: **DPM-Solver++(2M)** ("2" = second order, "M" = multistep) uses just **one NFE per step** by reusing the last step's prediction, getting second-order accuracy at first-order cost. This is the default in much of the ecosystem and the one you should reach for first.

**DPM-Solver++** (the "++" version) makes one more important change: it solves the ODE in terms of the **data prediction** $x_\theta$ (the predicted clean image) rather than the noise prediction $\epsilon_\theta$. With strong classifier-free guidance, the noise prediction can blow up and the noise-form solver becomes unstable — you get over-saturated, fried-looking images at high CFG and few steps. The data-prediction form is bounded (a clean image lives in a fixed range) and stays stable under heavy guidance. **This is why DPM-Solver++ is the recommended sampler for guided text-to-image: it was specifically designed to be robust to the high guidance scales that real prompts use.** There is also a **DPM-Solver++ SDE** variant that adds a controlled noise injection back in (bridging toward the stochastic samplers of the next section) and a **3M** third-order multistep variant for the last drops of quality at higher step counts.

Here is the practical reality in `diffusers` — swapping samplers is a two-line change, and this is the snippet you will actually use:

```python
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    UniPCMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16",
).to("cuda")

# DPM-Solver++ 2M — the workhorse. 'karras' sigmas + 2nd-order multistep.
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    algorithm_type="dpmsolver++",   # data-prediction form, CFG-stable
    solver_order=2,                 # the "2M"
    use_karras_sigmas=True,         # Karras sigma schedule for sampling
)
image = pipe("a photo of an astronaut riding a horse, 50mm",
             num_inference_steps=15, guidance_scale=6.0).images[0]

# UniPC — often a touch better at 8-12 steps. Same one-liner.
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
image_fast = pipe("a photo of an astronaut riding a horse, 50mm",
                  num_inference_steps=10, guidance_scale=6.0).images[0]

# Euler-a (ancestral/SDE) — more variety, never fully converges.
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
image_div = pipe("a photo of an astronaut riding a horse, 50mm",
                 num_inference_steps=30, guidance_scale=6.0).images[0]
```

That `from_config` pattern is the key idiom: every scheduler can be constructed from any other scheduler's config, so you swap samplers without touching the pipeline. The `algorithm_type="dpmsolver++"`, `solver_order=2`, and `use_karras_sigmas=True` flags are the three knobs that turn the generic multistep scheduler into the specific "DPM++ 2M Karras" that you see named in ComfyUI and Automatic1111. Memorize that combination — it is the single best default for SDXL at 15–25 steps.

### The startup problem, and why multistep is not free lunch

There is one subtlety the multistep trick hides, and it bites at low step counts. A multistep method estimates the derivative by *reusing* the network outputs from previous steps — DPM++ 2M, for instance, forms a second-order estimate from the current evaluation $\epsilon_n$ and the previous one $\epsilon_{n-1}$ via a finite difference, exactly as a two-step Adams–Bashforth scheme would. But on the very first step there *is* no previous evaluation. So the method has to **bootstrap**: the first step falls back to first-order (a single DDIM-style step), and only from the second step onward does it run at full second order. With 50 steps, one first-order startup step is negligible. With 8 steps, a first-order startup step is 12.5% of your budget spent at lower accuracy — a real cost. This is part of why UniPC, whose corrector compensates the startup more gracefully, tends to edge out DPM++ 2M specifically in the 5–10 step regime, and why the gap closes by 20 steps where the startup step is a rounding error.

The finite-difference history reuse also has a stability implication. Multistep methods, like their Adams–Bashforth cousins, can become *unstable* if the step sizes change too abruptly between steps — a large jump in $\sigma$ followed by a small one makes the finite-difference derivative estimate unreliable. In practice the well-designed sampling schedules (Karras $\rho=7$) keep the step-size ratio smooth enough that this never bites, but it is the reason you should not hand-craft a wild, non-monotone $\sigma$ schedule and expect a multistep solver to tolerate it. The solver assumes a reasonably smooth march from $\sigma_{\max}$ to $0$; give it that and it rewards you with second order at one NFE per step.

## 5. UniPC: unifying prediction and correction

DPM-Solver's multistep predictor and Heun's corrector are two halves of the same classical idea: a **predictor–corrector** scheme. You predict the next point with an explicit method, then correct it using an evaluation at (or near) that predicted point. **UniPC** (Unified Predictor-Corrector, Zhao et al., 2023) is a framework that treats the predictor and corrector in a *unified* way and, importantly, makes the corrector **free** — it reuses the model evaluation that the next step's predictor needs anyway, so the correction adds no NFE.

The practical effect is that UniPC squeezes slightly higher order out of the same budget than DPM-Solver++(2M), and it tends to be the strongest deterministic sampler in the **very low step regime** of 5–10 steps. At 8 steps, `UniPCMultistepScheduler` frequently produces a noticeably cleaner image than DPM++ 2M, because the corrector cleans up the error that the predictor alone would leave when steps are large. By 20+ steps the two are visually indistinguishable — both have converged. The decision is therefore budget-dependent: UniPC for the most aggressive step counts, DPM++ 2M as the robust general default. We will fold this into the decision tree in section 9.

The deeper point UniPC makes is conceptual: **predictor–corrector is the right mental model for all of these solvers.** Euler is a bare predictor. Heun is a predictor with an explicit extra-NFE corrector. DPM-Solver++(2M) is a multistep predictor. UniPC is a multistep predictor with a free corrector. They sit on a single spectrum of "how much do you correct, and how cheaply." Once you see them that way, the proliferation of scheduler names in the UI stops being intimidating; they are points on a known design axis, and you can reason about where a new one falls.

## 6. The matrix: order, steps, stochasticity

We have now built up enough to lay the deterministic samplers side by side and add the stochastic ones. The table below is the reference you will come back to. It is organized by what actually drives the trade-off: the **order** of the solver (how fast its error shrinks with steps), the **NFE per step** (whether higher order is free or costs extra evaluations), whether it is **deterministic** (reproducible, ODE) or **stochastic** (diverse, SDE), and the rough **step count to a good FID**.

![A matrix comparing Euler, Heun, DPM-Solver++ 2M, UniPC, and Euler ancestral across order, good-FID step count, determinism, and network evaluations per step](/imgs/blogs/samplers-deep-dive-4.png)

A few rows deserve commentary because they encode the whole post.

**Euler / DDIM (1st order, ~50 steps, deterministic, 1 NFE/step).** The honest baseline. Cheap per step, but slow-converging, so it needs many steps. Use it when you want maximum reproducibility and have the budget, or as the reference everything else is measured against.

**Heun / EDM (2nd order, ~30 steps, deterministic, 2 NFE/step).** Lowest FID of the deterministic ODE samplers when you have the budget, because it is a clean, well-understood second-order method with a tuned schedule. The 2 NFE/step cost means it is *not* the right choice when NFE is scarce — its 30 "steps" are 60 NFE.

**DPM-Solver++ 2M (2nd order multistep, ~15 steps, deterministic, 1 NFE/step).** The workhorse. Second-order accuracy at one NFE per step by reusing the previous evaluation, plus the exact-linear-part trick and CFG-stable data-prediction form. This is the default you reach for at 15–25 steps for guided text-to-image.

**UniPC (2nd–3rd order PC, ~10 steps, deterministic, 1 NFE/step).** The aggressive-budget champion. The free corrector wins at 5–10 steps. Converges to the same place as DPM++ by 20 steps.

**Euler-a / SDE (1st order + noise, never converges, stochastic, 1 NFE/step).** A different animal entirely, and the subject of the next section. It re-injects noise every step, so it does not converge to a single image as you add steps — it keeps wandering. More diverse, sometimes more detailed, but unreproducible across step counts and slower to a *stable* result.

## 7. Ancestral and SDE samplers: re-injecting noise

Everything so far solved the deterministic PF-ODE. But the original reverse process is an **SDE** — it injects fresh Gaussian noise at every step. Samplers that follow the SDE rather than the ODE are the **ancestral** samplers (Euler-a, DPM2-a) and the explicit **SDE** variants (DPM++ SDE, the stochastic EDM sampler). The "a" in `EulerAncestralDiscreteScheduler` stands for *ancestral*: each step denoises a little *less* than a deterministic step would, then adds back a calibrated amount of fresh noise to make up the difference, following the ancestral-sampling recipe of the original DDPM.

Why would you ever add noise back when the whole game was removing it? Three reasons, and one serious caveat.

First, **diversity**. The injected noise means different runs explore different parts of the data manifold, and even within a run the trajectory is not locked to the deterministic path. For prompts where you want to see varied compositions across a batch, ancestral samplers spread out more. Second, **error correction**. Karras et al. (2022) make the underappreciated point that a small amount of stochasticity actively *corrects* accumulated discretization error: the noise injection nudges the sample back toward the correct marginal distribution $p_t$ at each step, undoing drift that a pure ODE solver would carry to the end. A controlled dose of "churn" can lower FID below the deterministic solver on some models. Third, **texture**. Practitioners often report that ancestral samplers produce crisper fine detail and texture, because the re-injected noise gives the network fresh high-frequency content to shape at each step rather than monotonically smoothing.

The serious caveat is in the figure-4 row and worth stating bluntly: **ancestral samplers do not converge.** Because fresh noise enters every step, there is no fixed limit point as you increase the step count. Run Euler-a at 20, 30, and 50 steps and you get three *different* images, not three successively-better approximations of one image. This breaks the most useful property of deterministic samplers — that more steps monotonically refine *the same* image toward a clean limit. It also makes ancestral samplers a poor fit for anything that needs reproducibility across configurations, for DDIM-style latent interpolation, or for inversion-based editing (covered in the editing post), all of which rely on a deterministic, invertible map. And too much stochasticity, or stochasticity at too-few steps, leaves visible residual noise — the SDE simply has not had enough steps to integrate the noise back out.

It helps to see the EDM stochastic sampler's structure, because it makes the "noise corrects error" claim precise rather than hand-wavy. The EDM stochastic step does two things in sequence. First, it *adds* noise to bump the sample from noise level $\sigma$ up to a slightly higher $\hat{\sigma} = \sigma(1 + \gamma)$, where $\gamma$ is a small "churn" parameter — this is a deliberate step *backward* up the noise schedule. Then it takes a (Heun) ODE step from $\hat{\sigma}$ all the way down to the next target $\sigma_{n+1}$. The net displacement still moves you down the schedule, but the up-then-down detour is the mechanism: the added noise re-randomizes the sample so that any accumulated discretization error — drift that has pushed the sample off the true marginal $p_\sigma$ — gets partially washed out, and the subsequent denoising step pulls it back toward the correct distribution. It is a Langevin-style correction grafted onto the ODE solver. Karras et al. show that a *moderate* churn lowers FID on some datasets, but too much churn over-randomizes and *raises* FID, and the optimal churn is model- and dataset-dependent. That is the formal version of "a little noise corrects error, too much noise destroys the image."

This also clarifies why the ancestral samplers do not converge. A deterministic ODE solver, as you add steps, takes smaller and smaller steps along a *fixed* trajectory toward a *fixed* endpoint — the limit is well defined. A stochastic sampler, as you add steps, injects noise at *every* one of those (now more numerous) steps; there is no fixed trajectory and no fixed endpoint, only a *distribution* that the sample is drawn from. More steps make the sample a better draw from the correct distribution $p_0$, but they do not make it the *same* image. Convergence "to a distribution" is the right notion for SDEs, and it is a genuinely different guarantee from convergence "to a point," which is what the ODE solvers give and what reproducibility, interpolation, and inversion all require.

#### Worked example: NFE accounting, the comparison everyone gets wrong

Here is a mistake in nearly every casual sampler benchmark you will read online. Someone compares "Euler 30 steps" against "Heun 30 steps," sees Heun looks better, and concludes Heun is strictly superior. But Heun is second-order at **2 NFE per step**, so "Heun 30 steps" is *60 network evaluations*, while "Euler 30 steps" is *30*. The honest comparison is Heun-30 (60 NFE) against Euler-60 (60 NFE) — and now the gap narrows sharply, because Euler with twice the steps has caught up a lot. Conversely, the multistep DPM++ 2M is second-order at **1 NFE per step**, so "DPM++ 20 steps" really is 20 NFE, and comparing it against "Heun 20 steps" (40 NFE) *understates* DPM++'s efficiency — it is matching a 2nd-order result at *half* the evaluations. Whenever you tune a pipeline, convert every config to NFE first: NFE = steps × (NFE per step), where Euler/DDIM/DPM++ multistep/UniPC are 1 and Heun/DPM-Solver-2 singlestep are 2. Then plot quality against NFE, and the true frontier appears. The samplers that look magical — DPM++ 2M, UniPC — are exactly the ones that deliver high order at 1 NFE per step.

#### Worked example: when Euler-a helps and when it hurts

Say you are generating a batch of 8 portraits for a mood board and you *want* variety — different framings, lighting, expressions. `EulerAncestralDiscreteScheduler` at 30 steps, CFG 6, is a great choice: the injected noise spreads the 8 samples across the manifold and the texture is crisp. Now say you are A/B testing two prompts and need the *only* difference to be the prompt — same seed, same everything else, so the comparison is clean. Euler-a sabotages you: because the noise schedule interacts with the step count and the stochastic draws, you cannot get a controlled comparison, and re-running at a different step count gives a different image even at the same seed. Switch to DPM++ 2M (deterministic) and the comparison becomes clean and reproducible. **Rule: stochastic for diversity and texture; deterministic for control, reproducibility, interpolation, and inversion.** The same model, the same prompt — the sampler choice alone decides whether your pipeline is reproducible.

## 8. The EDM framework: preconditioning and the sampling schedule

Karras, Aittala, Aila, and Laine's 2022 paper "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM) is the most important single reference for samplers, because it untangles a knot the field had been carrying: it separates the *training* of the model from the *sampling* of it, and shows that several "model" design choices are really *sampling* choices you can change after the fact, with no retraining. Three ideas from it matter for this post: the $\sigma$-parameterization, the preconditioning, and — most consequentially for samplers — the **sampling noise schedule** being a free choice distinct from the training schedule.

![A graph showing the EDM framework wrapping a raw network in preconditioning to form a denoiser, with separate training and sampling noise schedules feeding a Heun solver to a low FID](/imgs/blogs/samplers-deep-dive-6.png)

**The $\sigma$-parameterization.** EDM drops the discrete timestep index $t$ entirely and parameterizes everything directly by the **noise level** $\sigma$. A noisy sample is $x = x_0 + \sigma \epsilon$, full stop, with $\sigma$ ranging from near $0$ (clean) to $\sigma_{\max} \approx 80$ (effectively pure noise). This is cleaner than juggling $\alpha_t$, $\beta_t$ schedules, and it is why EDM-style code (and `diffusers`' Karras-sigma options) talks about sigmas rather than timesteps. The samplers we wrote in sections 2–3 used exactly this convention: `sigmas` as a descending list ending at $0$.

**Preconditioning.** A raw network trained to predict noise has badly-scaled inputs and outputs across the enormous $\sigma$ range — at $\sigma = 80$ the input is almost pure noise, at $\sigma = 0.01$ it is almost clean, and a single network struggles to handle both. EDM wraps the raw network $F_\theta$ in input/output/skip scalings $c_{\text{in}}(\sigma)$, $c_{\text{out}}(\sigma)$, $c_{\text{skip}}(\sigma)$ chosen so the effective denoiser $D_\theta(x;\sigma) = c_{\text{skip}}x + c_{\text{out}}F_\theta(c_{\text{in}}x;\,c_{\text{noise}}(\sigma))$ has unit-variance inputs and targets at every noise level. This preconditioning is a training-time choice, but it is the reason EDM models sample so cleanly: the denoiser is well-conditioned everywhere along the trajectory, so the solver's slope estimates are accurate at every $\sigma$.

**The sampling schedule is a free choice — this is the sampler insight.** Here is the point most people miss. The set of $\sigma$ values you *sample* at does **not** have to match the distribution of $\sigma$ values you *trained* at. Training uses a log-normal distribution over $\sigma$ to spend gradient on the noise levels that matter most for learning. *Sampling* uses a deterministic, hand-designed sequence of $\sigma$ steps — EDM's choice is

$$
\sigma_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}\left(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}\right)\right)^{\rho}, \quad \rho = 7
$$

This $\rho = 7$ schedule places *more* steps at low noise (where the trajectory has fine detail to resolve) and *fewer* at high noise (where it is coarse). It is the "Karras sigmas" you toggle with `use_karras_sigmas=True`. Switching from a model's native (training-matched) schedule to the Karras sampling schedule often cuts the steps-to-good-FID by a meaningful margin *with no retraining and no change to the solver* — it just spends the steps where the curvature is. **This is the cleanest demonstration of the post's thesis: sampling quality is a property of the integrator and its schedule, separable from the trained model.** When you see "DPM++ 2M Karras" in a UI, the "Karras" is exactly this sampling schedule, layered on top of the DPM++ 2M solver.

The EDM Heun sampler (its default) combines all three: a preconditioned denoiser, the $\rho=7$ sampling schedule, and the second-order Heun solver with optional stochastic churn. On unconditional CIFAR-10 this reaches FID ≈ 1.79 at 35 NFE, and on class-conditional ImageNet-64 it set a record at the time — the headline numbers that made EDM the reference design.

#### Worked example: the free Karras-schedule speedup

Take a stock SDXL pipeline and a fixed seed, CFG 6, and run DPM-Solver++ 2M two ways: once with the model's native (uniform-in-timestep) sigma schedule and once with `use_karras_sigmas=True`. Hold the solver, the model, the prompt, and the seed all fixed — the *only* change is which $\sigma$ values you step through. At 25 steps the two are visually identical; both have converged. But drop to 12 steps and the difference appears: the native schedule shows softer fine detail and slightly muddier textures, while the Karras schedule retains crisp edges, because it has spent its scarce steps in the low-noise region where the trajectory bends and the detail is resolved. In practice this buys roughly a 20–30% reduction in the steps needed to reach the same perceptual quality at the aggressive end of the budget — a meaningful speedup for *zero* additional compute per step and *zero* retraining. It is the purest demonstration of the post's thesis: you changed neither the model nor the solver's order, only *where* along the noise schedule you placed your steps, and quality-at-a-budget improved. If you take one configuration change away from this post, it is this: turn on Karras sigmas. It is almost never wrong for pixel and latent diffusion models, and it is free.

## 9. The decision: which sampler, which step count

Now the practitioner's payoff. You have a model and a budget. Which scheduler do you load and how many steps do you set? The decision branches on two questions, in order: **do you need a deterministic, reproducible image (ODE) or do you want diversity (SDE)?** and **how many steps can you afford?** The tree below encodes the recommendation.

![A decision tree branching first on deterministic versus stochastic sampling and then on step budget, with leaves naming DPM-Solver++ 2M, UniPC, Heun, and Euler ancestral](/imgs/blogs/samplers-deep-dive-7.png)

Walking the branches with concrete advice:

**Deterministic, tight budget (4–15 steps).** Use **UniPC** or **DPM-Solver++ 2M**, both with Karras sigmas. At 8–10 steps UniPC's free corrector usually edges ahead; at 12–15 steps they converge. Do *not* use Euler or Heun here — Euler gives mush below ~15 steps and Heun's 2 NFE/step means you only get 4–7 actual steps from a 10-NFE budget, which is too few corrections. If you need 4 steps or fewer, no undistilled sampler will look great; that is the domain of **distilled few-step models** (LCM, SDXL-Turbo, consistency models), covered in [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) — a fundamentally different approach that retrains the model to make 4 steps work, rather than asking a better solver to do the impossible.

**Deterministic, quality-first (20–50 steps).** **DPM-Solver++ 2M Karras** at 20–25 steps is the sweet spot for almost everything — it is converged or nearly so, CFG-stable, and one NFE per step. If you want the absolute lowest FID and have the NFE to spare, **Heun (EDM)** at 30+ "steps" is the textbook-cleanest deterministic ODE solver. Going past ~25 steps with DPM++ gives diminishing returns; the image has converged and you are spending NFE for invisible gains.

**Stochastic, want diversity or texture.** **Euler-a** at 25–35 steps, or **DPM++ SDE** if you want the exponential-integrator quality with a controlled noise injection. Accept that the result is not reproducible across step counts and not suitable for inversion or interpolation. Give it enough steps (≥ 25) to integrate the injected noise back out, or you will see residual graininess.

This is also the place to be honest about **when not to fiddle with the sampler at all.** If your image looks great at DPM++ 2M Karras / 20 steps and your latency budget is met, *stop*. Sampler tuning has sharply diminishing returns once you are on a good 2nd-order multistep solver with Karras sigmas. The big wins — 1000 → 50 → 20 steps — are already captured. Squeezing 20 → 15 is real but minor; chasing 15 → 12 is usually not worth the quality risk for a production pipeline. The exception is when latency is genuinely binding (real-time, on-device, high QPS), in which case you graduate to distillation, not a different solver.

## 10. The step-count Pareto, read honestly

Let me put the whole speed–quality trade-off on one surface. The matrix below is the steps↔quality Pareto: rows are samplers, columns are step budgets (4, 10, 20, 50), and each cell is the qualitative result. Read *down a column* to compare samplers at a fixed budget; read *across a row* to see how one sampler degrades as you starve it of steps. This is the "Pareto chart" you act on — the frontier is the set of cells where, for a given budget, no other sampler does better.

![A matrix showing the step-count quality frontier with samplers as rows and step budgets of four, ten, twenty, and fifty as columns, marking where each sampler reaches converged quality](/imgs/blogs/samplers-deep-dive-5.png)

The shape of this surface is the single most important thing to internalize from this post:

- **The 4-step column is brutal.** Euler and Heun give mush or noise; only the exponential multistep solvers (DPM++, UniPC) are even usable, and even they are soft. Below 4 steps, *no* undistilled solver works — you have hit the wall where truncation error dominates and the answer is a different model (distillation), not a different integrator.
- **The 10-step column is where the modern samplers earn their keep.** UniPC and DPM++ 2M are good-to-near-best at 10 steps; Euler is still rough; Heun is fair (it only has 5 real steps at 10 NFE). This column is exactly why DPM++/UniPC replaced Euler as the default.
- **The 20-step column is essentially converged for the good solvers.** DPM++ 2M and UniPC are at or near their best; Euler has finally caught up to "good." For most production text-to-image, 20 steps of DPM++ 2M Karras is the answer.
- **The 50-step column is converged for everyone.** Even first-order Euler reaches its best here. The lesson: more steps eventually saturates, and the *only* reason to prefer a better sampler is to reach that saturation point at fewer steps. Quality has a ceiling set by the *model*; the sampler decides how fast you hit it.

The honest framing of the Pareto: **the sampler does not raise the quality ceiling — it moves the knee of the curve left.** A better sampler reaches the model's best achievable quality at fewer NFE; it cannot exceed what the trained model can produce. If your 50-step images are not good enough, no sampler will fix that — you need a better model, better guidance, or a fine-tune. If your 50-step images are great but too slow, the sampler is *exactly* the right lever, and moving from Euler-50 to DPM++-15 is a 3× speedup at the same quality.

## 11. Case study: Euler-50 versus DPM++-15 on SDXL

Let me ground the Pareto in a concrete, reproducible comparison — the one from the brief, and the one you should run yourself to build intuition. The setup: Stable Diffusion XL base 1.0, fixed seed, CFG 6.0, the same prompt, on an RTX 4090 (24 GB) at fp16. We compare **EulerDiscreteScheduler at 50 steps** against **DPMSolverMultistepScheduler (dpmsolver++, order 2, Karras sigmas) at 15 steps**.

![A before and after comparison of an image generated with Euler at fifty steps against the same image generated with DPM-Solver++ 2M Karras at fifteen steps, showing matched quality at a third of the cost](/imgs/blogs/samplers-deep-dive-8.png)

The numbers that matter, and how to read them honestly:

| Config | NFE | Latency (s, RTX 4090) | Relative quality | Reproducible? |
|---|---|---|---|---|
| Euler, 50 steps | 50 | ~3.8 | reference (converged) | yes |
| DPM++ 2M Karras, 20 steps | 20 | ~1.6 | indistinguishable | yes |
| DPM++ 2M Karras, 15 steps | 15 | ~1.2 | near-identical, slightly softer fine detail | yes |
| UniPC, 10 steps | 10 | ~0.8 | very close, occasional soft texture | yes |
| Euler-a, 30 steps | 30 | ~2.3 | different image (more varied) | no, varies with steps |

The latency figures are order-of-magnitude for an RTX 4090 at fp16 SDXL with batch size 1 — your exact numbers depend on resolution, attention backend (SDPA vs xformers), and whether `torch.compile` is on; treat them as approximate and benchmark your own setup with a warm-up pass and a fixed seed. The *relative* picture, though, is robust and reproducible: **DPM++ 2M Karras at 15–20 steps matches Euler at 50 steps to the eye, at roughly one-third the cost.** That is the headline win of the modern sampler stack, and it is free — same model, same weights, two-line scheduler swap.

How would you measure this rigorously rather than eyeballing it? Generate a fixed set (say 10k) of images from a fixed prompt distribution with each config, compute **FID against a held-out reference set** (or FID-DINOv2, which correlates better with human perception), use the *same* seeds across configs so the only variable is the sampler, and include a warm-up generation before timing so you measure steady-state latency, not compilation. Report NFE, not "steps," so the comparison is fair across orders. You will find the FID curves flatten by ~20 NFE for DPM++/UniPC and by ~50 NFE for Euler — quantifying exactly the knee-shift the Pareto predicts.

#### Worked example: the CFG-instability trap at low steps

Here is a failure mode that catches people and shows why DPM-Solver*++* (data-prediction form) exists. Take the *noise-prediction* DPM-Solver (`algorithm_type="dpmsolver"`, not `"dpmsolver++"`), set CFG to 12, and run 10 steps. You will get over-saturated, high-contrast, "deep-fried" images — blown-out colors, crunchy edges. The cause: at high CFG the extrapolated noise prediction $\epsilon_{\text{uncond}} + 12(\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$ has large magnitude, the noise-form solver amplifies it across large steps, and the latent leaves the well-behaved region. Switch the single flag to `algorithm_type="dpmsolver++"` and the data-prediction form bounds the update (a clean image has limited range), and the same CFG 12 at 10 steps is stable. **This is a one-flag fix for a real, common artifact**, and it is the concrete reason DPM-Solver++ — not the original DPM-Solver — became the default. (The deeper fix for over-saturation is lowering CFG or using guidance rescaling, covered in [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance); the sampler choice makes the failure graceful rather than catastrophic.)

## 12. Stress tests: where samplers break

A decision is only trustworthy if you have pushed on its edges. Let me stress-test the recommendations.

**What happens at 4 steps?** Every undistilled solver degrades — DPM++/UniPC are soft, Euler/Heun are mush. The truncation error per step is simply too large; the straight (or low-order) chords cannot track a trajectory that swings across most of the noise range in four moves. The correct response is *not* a fancier solver but a *distilled model*: LCM, Turbo, or a consistency model that has been retrained so that 4 steps is enough. Asking a better integrator to solve a 4-step problem is asking it to integrate a wildly curved path with four chords; no solver order fixes that. The model has to change.

**What happens when CFG is too high?** As the worked example showed, the noise-prediction solvers go unstable and over-saturate; the data-prediction (++) form degrades gracefully. But even DPM++ cannot rescue CFG 20 — at some point the guidance itself, not the sampler, is the problem, and you get unnatural colors regardless of solver. The sampler choice buys robustness, not immunity. Lower the CFG.

**What happens with an ancestral sampler at too few steps?** Visible residual noise. The SDE injects noise every step and relies on having enough subsequent steps to integrate it back out; starve it of steps and the noise has nowhere to go. Euler-a at 10 steps often looks grainier than Euler-a at 30. If you want few steps, you want a *deterministic* solver, not an ancestral one — the noise injection is a luxury that needs a step budget.

**What happens when you switch the sampling schedule but not the solver?** Often a free quality gain (Karras sigmas) — but not always. The Karras $\rho=7$ schedule assumes the trajectory's curvature is concentrated at low noise, which holds for most pixel/latent diffusion models but can mismatch a model trained with an unusual noise schedule or a flow-matching model (whose ODE is nearly straight by construction). For **flow-matching models** like SD3 and FLUX, the right sampler is `FlowMatchEulerDiscreteScheduler` — Euler on the flow-matching ODE — and Karras sigmas do *not* apply, because the flow-matching path is engineered to be nearly straight and a plain Euler step is already near-optimal:

```python
import torch
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
).to("cuda")
# SD3/FLUX ship FlowMatchEuler by default. A plain Euler step is near-optimal
# because the flow-matching path is engineered to be nearly straight, so the
# generic DPM/Karras machinery for curved diffusion ODEs does not transfer.
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
image = pipe("an astronaut riding a horse, 50mm photo",
             num_inference_steps=28, guidance_scale=7.0).images[0]
```

This is the bridge to [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow): the reason SD3/FLUX sample well at low steps is partly the straight-path *training*, which makes the *sampler's* job trivial. The lesson generalizes: the best sampler depends on the geometry of the trajectory the model produces, and a straighter trajectory needs a simpler solver. Reaching for DPM++ with Karras sigmas on a flow-matching model is a category error — you would be applying a fix for stiff, curved diffusion ODEs to a path that was deliberately straightened so it would not need that fix.

**What happens when the VAE, not the sampler, is the bottleneck?** At very low step counts (4–8), the VAE decode can become a meaningful fraction of total latency, and shaving sampler steps stops helping. Profile end to end: if 8 sampler steps take 0.6 s and the VAE decode takes 0.3 s, going from 8 to 6 steps saves 0.15 s on a 0.9 s pipeline — a 17% win, not the 25% the sampler-only math suggests. Below a certain step count, the sampler is no longer the dominant cost, and the next lever is a faster decoder (`AutoencoderTiny`) or feature caching, covered in [why diffusion is slow and how to fix it](/blog/machine-learning/image-generation/why-diffusion-is-slow-and-how-to-fix-it).

## 13. The whole stack, end to end

Stepping back, a sampler is one stage in the diffusion stack — data → VAE-latent → forward noising → denoiser net → **ODE/SDE sampler** → guidance → image — and it is the stage you touch most often at inference because it is a pure runtime choice with no retraining. Everything we covered is a way of answering one question well: given a network that supplies the ODE's slope, how do I integrate from noise to image in as few evaluations as possible without losing quality?

The intellectual arc is worth compressing into one paragraph, because it is the frame that makes new samplers legible. Sampling is solving the probability-flow ODE (deterministic) or the reverse SDE (stochastic). **Euler** is the rectangle rule, first-order, equal to DDIM, $O(h^2)$ local error, slow to converge. **Heun** averages two slopes, second-order, $O(h^3)$ local error, half the global error at twice the per-step cost. **DPM-Solver(++)** exploits the ODE's semi-linear structure — solving the linear part exactly and Taylor-expanding only the smooth nonlinear residual in log-SNR — to reach second-order accuracy at one NFE per step via multistep, with the data-prediction (++) form for CFG stability. **UniPC** adds a free predictor–corrector boost that wins at the lowest step counts. **Ancestral/SDE** samplers re-inject noise for diversity and error correction but never converge. And the **EDM** framework separates the sampling schedule from training, letting "Karras sigmas" speed up *any* solver by spending steps where the curvature is. Every named sampler in your UI is a point on these axes — order, integrator structure, stochasticity, schedule — and you can now place any new one.

## 14. Measuring samplers without fooling yourself

Because the sampler is a runtime knob with no retraining, it is tempting to "tune" it by eye — generate a few images, pick the sampler whose outputs you like, ship it. That is how teams end up with a default that looks great on the three prompts they tested and falls apart on the long tail. A defensible sampler choice rests on a measurement protocol, and the protocol is not complicated; it is just disciplined.

First, **fix the seed across configs.** The single largest source of variance in a side-by-side is the initial noise, not the sampler. Use the same seed (or the same *batch* of seeds) for every config so the only thing that changes is the integration. If you compare Euler-50 on seed 1 against DPM++-15 on seed 2, you are comparing two different images and learning nothing about the samplers. Second, **report NFE, not steps**, for the reasons section 7's worked example laid out — otherwise you are silently giving 2-NFE/step methods a handicap or an unfair advantage. Third, **measure FID against a fixed reference set with enough samples.** FID is noisy below a few thousand samples; the standard is 10k–50k generated images against a held-out reference set of the same distribution. For perceptual fidelity that correlates with human judgment, prefer **FID-DINOv2** (FID computed in a DINOv2 feature space rather than the aging InceptionV3 space), which the evaluation literature has shown tracks human preference better — covered in [evaluating image generation honestly](/blog/machine-learning/image-generation/evaluating-image-generation-honestly). Fourth, **warm up before timing.** The first generation pays for CUDA kernel compilation, autotuning, and (if you use `torch.compile`) graph capture; time the *second* run onward to measure steady-state latency. Fifth, **separate the sampler's cost from the VAE's.** Report sampler latency and decode latency separately, because at low step counts the decode is a non-trivial fraction and conflating them hides where the time actually goes.

Here is a minimal, honest benchmarking harness that bakes in the protocol — fixed seeds, warm-up, NFE accounting, separated decode timing — so you can sweep samplers without re-deriving the discipline each time:

```python
import time, torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16",
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, algorithm_type="dpmsolver++",
    solver_order=2, use_karras_sigmas=True,
)
NFE_PER_STEP = {"dpmsolver++_2M": 1, "heun": 2, "euler": 1, "unipc": 1}

def bench(prompt, steps, seeds, nfe_per_step=1, warmup=1):
    # Warm-up runs pay for kernel compilation; do not time them.
    g = torch.Generator("cuda").manual_seed(seeds[0])
    for _ in range(warmup):
        pipe(prompt, num_inference_steps=steps, generator=g)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in seeds:
        g = torch.Generator("cuda").manual_seed(s)   # fixed seed per config
        pipe(prompt, num_inference_steps=steps, generator=g)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / len(seeds)
    print(f"steps={steps}  NFE={steps * nfe_per_step}  "
          f"latency={dt:.2f}s/img")

bench("an astronaut riding a horse, 50mm photo", steps=15,
      seeds=[0, 1, 2, 3], nfe_per_step=NFE_PER_STEP["dpmsolver++_2M"])
```

The harness reports latency *and* NFE, reuses a fixed seed set across configs, and warms up before timing — the three things casual benchmarks skip. Swap the scheduler and `nfe_per_step` to sweep samplers fairly; feed the generated images to `torchmetrics`' `FrechetInceptionDistance` against a fixed reference set to add the quality axis.

Run that protocol and you will reproduce the shape this post predicts: FID-versus-NFE curves that flatten by ~20 NFE for the exponential multistep solvers and by ~50 NFE for Euler, a clean ~3× efficiency gap between DPM++/UniPC and first-order Euler at matched quality, and a stochastic sampler whose FID is competitive but whose individual images refuse to be reproducible. The numbers will be specific to your model and resolution, but the *ordering* of the samplers and the *location of the knee* are robust — which is exactly why a single good default (DPM++ 2M Karras, 20 steps) generalizes across the prompts you did not test.

There is a final, slightly humbling point about measurement. FID rewards *distributional* match — does your set of generated images look like the reference set — but it does not directly reward *per-image* quality or prompt adherence. Two samplers can have nearly identical FID while one produces images you clearly prefer, because the differences live in fine texture or composition that FID's pooled statistics smooth over. So pair the FID-versus-NFE curve with a small human-preference check (or a learned preference model like HPSv2/ImageReward) at your chosen operating point. The sampler decision is "which integrator reaches the model's quality ceiling at the fewest NFE," and FID-versus-NFE answers that well — but confirm the operating point with eyes or a preference model before you ship it as the default.

## When to reach for this (and when not to)

Reach for **DPM-Solver++ 2M with Karras sigmas** as your default for guided text-to-image at 15–25 steps. It is the best general-purpose choice: second-order, one NFE per step, CFG-stable, converged by ~20 steps. Reach for **UniPC** when the budget is tight (5–12 steps) and you want the most quality per step. Reach for **Heun (EDM)** when you want the textbook-lowest-FID deterministic ODE result and have the NFE to spend. Reach for **Euler-a or DPM++ SDE** when you want diversity, texture, or stochastic error correction and do not need reproducibility. Reach for **`FlowMatchEulerDiscreteScheduler`** when the model is a flow-matching model (SD3, FLUX) — its straight path makes plain Euler near-optimal.

Do **not** reach for a fancier sampler when your images already look good at 20 steps — sampler tuning saturates, and you are spending effort for invisible gains. Do **not** use Euler or Heun below ~15 steps; they give mush. Do **not** use an ancestral sampler when you need reproducibility, latent interpolation, or inversion-based editing — its non-convergence breaks all three. Do **not** expect *any* undistilled sampler to work at 4 steps — that is a distillation problem (LCM/Turbo/consistency), a different tool entirely. And do **not** assume a better sampler raises quality — it moves the knee of the speed–quality curve left, but the ceiling is set by the model, the guidance, and the fine-tune, not the integrator.

## Key takeaways

- **Sampling is numerical integration of an ODE (deterministic, PF-ODE) or SDE (stochastic, reverse SDE).** The network supplies the slope; the sampler is the solver. Step count is an accuracy knob.
- **Count NFE, not steps.** A 2nd-order singlestep method uses 2 NFE/step, so "Heun 25" and "Euler 50" cost the same. Every honest sampler comparison is at fixed NFE.
- **Euler is DDIM is 1st-order DPM-Solver-1** — all the same first-order method, $O(h^2)$ local error, needs ~50 steps. **Heun** averages two slopes for $O(h^3)$ local error and quadratic convergence at 2 NFE/step.
- **DPM-Solver(++) wins by structure, not brute force:** solve the semi-linear ODE's linear part exactly and approximate only the smooth residual in log-SNR. **DPM++ 2M** gets 2nd order at 1 NFE/step; the **++ data-prediction form is CFG-stable** and is the default for guided generation.
- **UniPC**'s free corrector wins at 5–12 steps; by 20 steps it ties DPM++. **Ancestral/SDE** samplers (Euler-a) trade convergence for diversity and texture — they never settle to one image.
- **EDM separates the sampling schedule from training.** "Karras sigmas" ($\rho=7$, more steps at low noise) speed up *any* solver with no retraining — the cleanest proof that quality-at-a-budget is a sampler property.
- **The sampler moves the knee of the speed–quality curve left; it does not raise the ceiling.** DPM++-15 ≈ Euler-50 at one-third the cost, but neither beats what the model can produce.
- **Below 4 steps, change the model, not the solver** — that is the domain of LCM, Turbo, and consistency distillation, not a fancier integrator.

## Further reading

- **Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole — "Score-Based Generative Modeling through Stochastic Differential Equations" (2021).** The reverse SDE and the probability-flow ODE that make sampling an integration problem. The foundation of this entire post.
- **Song, Meng, Ermon — "Denoising Diffusion Implicit Models" (DDIM, 2021).** The deterministic, non-Markovian sampler that is first-order Euler on the PF-ODE; 1000 → 50 steps without retraining.
- **Lu, Zhou, Bao, Chen, Li, Zhu — "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps" (2022)** and the follow-up **"DPM-Solver++" (2022).** The exponential-integrator trick and the CFG-stable data-prediction form. The workhorse.
- **Karras, Aittala, Aila, Laine — "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM, 2022).** The $\sigma$-parameterization, preconditioning, the $\rho=7$ sampling schedule, and the Heun sampler. The single most important sampler reference.
- **Zhao, Bai, Rao, Zhou, Lin — "UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models" (2023).** The free corrector that wins at the lowest step counts.
- **🤗 `diffusers` schedulers documentation.** The practical reference for `EulerDiscreteScheduler`, `DPMSolverMultistepScheduler`, `UniPCMultistepScheduler`, `EulerAncestralDiscreteScheduler`, and `FlowMatchEulerDiscreteScheduler`, including every flag used above.
- Within this series: [DDIM and fast deterministic sampling](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling), [score-based models and the SDE view](/blog/machine-learning/image-generation/score-based-models-and-the-sde-view), [noise schedules and the parameterization zoo](/blog/machine-learning/image-generation/noise-schedules-and-the-parameterization-zoo), [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation), and the capstone [building an image-generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
