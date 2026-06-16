---
title: "Noise Schedules and the Parameterization Zoo: The Quiet Choices That Decide Quality"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The under-discussed knobs — what the network predicts (eps, x_0, or v) and how noise is scheduled (linear, cosine, sigmoid) — quietly decide training stability and FID. Derive the parameterization equivalences, read every schedule through SNR(t), fix the zero-terminal-SNR brightness bug, and shift the schedule for high resolution, with runnable PyTorch and diffusers code."
tags:
  [
    "image-generation",
    "diffusion-models",
    "noise-schedule",
    "v-prediction",
    "snr-weighting",
    "zero-terminal-snr",
    "cosine-schedule",
    "generative-ai",
    "deep-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/noise-schedules-and-the-parameterization-zoo-1.png"
---

Here is a bug that has shipped in nearly every Stable Diffusion model and probably bit you without you noticing. Prompt it for "a solid black background" and you get charcoal grey. Prompt it for "a pure white studio backdrop" and you get a soft, slightly dirty off-white. Crank the brightness of any output and the midtones feel oddly compressed, the blacks never crush, the whites never blow out. People blame the VAE, or the prompt, or the CFG scale. The real culprit is a single number deep in the noise schedule — the amount of signal that leaks through at the very last, supposedly-pure-noise timestep — and the fix is four lines of code that almost nobody applied for two years. That number is `$\bar\alpha_T$`, and when it is not exactly zero, the model is trained on inputs it never sees at sampling time, so it can only ever produce images near the dataset's mean brightness.

This post is about the two quietest design choices in a diffusion model: **what the network is asked to predict**, and **how the noise is scheduled over time**. Neither shows up in a press release. Both are usually copied verbatim from whatever repo you forked. And both move FID by more than the architecture changes people write papers about. A staff engineer can spend a month tuning a U-Net and get a 0.3 FID improvement; switching from `$\varepsilon$`-prediction to `$v$`-prediction and fixing the terminal SNR can move it by several points and rescue an entire class of failure modes. These are the choices that decide whether your high-resolution fine-tune converges or diverges, whether your distilled few-step model is stable, and whether your samples can render a real black.

The thread that ties all of it together is the **signal-to-noise ratio**, `$\text{SNR}(t) = \bar\alpha_t / (1 - \bar\alpha_t)$`. Every schedule is a curve of SNR against time. Every prediction target is well-conditioned at a different end of that curve. Every loss-weighting trick is a function of SNR deciding which noise levels the gradient pays attention to. Once you learn to read a diffusion model through its SNR curve, these scattered tricks — cosine schedules, min-SNR weighting, v-prediction, zero-terminal-SNR, the SD3 resolution shift — stop looking like a grab-bag of hacks and snap into a single coherent picture. Figure 1 is that picture: two orthogonal knobs, both routed through SNR, both feeding into stability and quality.

![A branching diagram showing the prediction target and the noise schedule as two orthogonal knobs routed through the signal-to-noise ratio into training stability and sample quality](/imgs/blogs/noise-schedules-and-the-parameterization-zoo-1.png)

By the end you will be able to: derive the algebraic equivalence of `$\varepsilon$`-, `$x_0$`-, and `$v$`-prediction and explain exactly why `$v$` is the stable one; read any schedule as an SNR curve and say where it spends its training budget; express the simple DDPM loss as an SNR-weighted variational bound and reweight it with min-SNR-`$\gamma$`; diagnose and fix the zero-terminal-SNR brightness bias; and shift a schedule correctly when you move to high resolution. This post sits between [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm), which derives the loss we are about to reweight, and [the samplers deep-dive](/blog/machine-learning/image-generation/samplers-deep-dive), which consumes the schedule we are about to design. If you have not internalized the forward process, start with [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles). Everything here is one layer below the architecture and one layer above the sampler — the layer that quietly decides quality.

## 1. SNR is the right lens on a schedule

Let me set notation once and then never re-derive it. The forward process in DDPM takes a clean image `$x_0$` and produces a noised version at timestep `$t$`:

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I).$$

Here `$\bar\alpha_t = \prod_{s=1}^{t} \alpha_s = \prod_{s=1}^{t}(1 - \beta_s)$` is the cumulative product of the per-step signal retention. At `$t=0$`, `$\bar\alpha_0 = 1$` and `$x_t = x_0$` (pure signal). As `$t \to T$`, `$\bar\alpha_t \to 0$` and `$x_t \to \varepsilon$` (pure noise). The whole forward process is just a schedule for how `$\bar\alpha_t$` slides from one to zero.

The coefficient `$\sqrt{\bar\alpha_t}$` is the amount of clean signal in `$x_t$`; the coefficient `$\sqrt{1-\bar\alpha_t}$` is the amount of noise. Because `$x_0$` is (after normalization) roughly unit-variance and `$\varepsilon$` is exactly unit-variance, the *power* in the signal component is `$\bar\alpha_t$` and the power in the noise component is `$1 - \bar\alpha_t$`. Their ratio is the signal-to-noise ratio:

$$\text{SNR}(t) = \frac{\bar\alpha_t}{1 - \bar\alpha_t}.$$

This is the single most useful number in the whole subject. It runs from `$\text{SNR}(0) = \infty$` (no noise) to `$\text{SNR}(T) \approx 0$` (no signal). Crucially, **SNR is what the model actually experiences**. The network does not see `$t$` directly in any meaningful sense — two different schedules that produce the same SNR at a given step give the model an identical task. The timestep index `$t$` is a coordinate; SNR is the physics. This is why, when you compare a linear schedule to a cosine schedule, you should not compare their `$\beta_t$` curves (which look wildly different and tell you nothing) — you compare their `$\text{SNR}(t)$` curves, ideally on a log scale, because that is the curve the model lives on.

A few facts make SNR even more central:

- **Log-SNR is the natural variable.** Define `$\lambda_t = \log \text{SNR}(t) = \log \bar\alpha_t - \log(1 - \bar\alpha_t)$`. Variational-diffusion-model papers (Kingma et al., 2021) showed that the continuous-time diffusion loss, written correctly, depends on the schedule *only through the endpoints* `$\lambda_0$` and `$\lambda_T$` of this log-SNR — the shape in between is reparameterization, not a different model. That is a deep statement: in the continuous limit, the schedule's interior is a sampling distribution over noise levels, not a change to the objective.
- **SNR sets the difficulty.** At high SNR (small `$t$`) the denoising task is nearly trivial — the image is almost clean, the model barely has to do anything. At low SNR (large `$t$`) the task is nearly impossible — the input is almost pure noise and the best the model can do is predict the dataset mean. The interesting learning happens in the middle band, and a good schedule and weighting put the model's budget there.
- **SNR is parameterization-invariant.** Whether the net predicts `$\varepsilon$`, `$x_0$`, or `$v$`, the SNR at a step is the same. The parameterization changes the *target* the model regresses to and therefore the *conditioning of the loss*, but not the underlying noise level. That orthogonality — schedule on one axis, parameterization on the other — is exactly Figure 1.

There is one more reason SNR is the right coordinate, and it is worth a paragraph because it dissolves a lot of confusion. In the discrete DDPM view you have a thousand timesteps and a `$\beta_t$` array, and it feels like the schedule is a thousand-dimensional object you are choosing. In the continuous-time view (Song et al.'s SDE framework, and Variational Diffusion Models), the forward process is a stochastic differential equation and the "schedule" is a single smooth function `$\lambda(t) = \log\text{SNR}(t)$`. Training samples a noise level by sampling `$t$`, evaluating `$\lambda(t)$`, and corrupting the image to that log-SNR. Two schedules that are reparameterizations of each other — same set of log-SNR values reached, just at different `$t$` indices — define *the same model* and differ only in how often each noise level is visited during training. That is why people increasingly specify a schedule as a **distribution over log-SNR** directly (EDM samples `$\ln\sigma$` from a Gaussian; the Laplace schedule samples log-SNR from a double-exponential) rather than as a `$\beta_t$` array. The array is an implementation detail of one particular discretization; the log-SNR distribution is the actual design.

This reframing also tells you what is *invariant* and what is a *choice*. The endpoints `$\lambda_0 = \log\text{SNR}(0)$` (how clean is the cleanest training input) and `$\lambda_T = \log\text{SNR}(T)$` (how noisy is the noisiest) are genuine modeling choices that change the objective — in particular, whether `$\lambda_T = -\infty$` (true zero terminal SNR) is the entire subject of §6. The shape *between* the endpoints is, in the continuous limit, a sampling distribution over noise levels: it changes *training efficiency* (where you spend gradient steps) but not the optimal model. That is a liberating fact. It means you can tune the interior of the schedule aggressively for convergence speed without worrying that you are changing what the model is fundamentally learning, while you must treat the endpoints with care because they *do* change the model.

So the rest of this post is two questions asked through one lens. Given the SNR curve, *what should the network predict* so the regression target is well-behaved at every point on the curve? And *what SNR curve* should we use, and how should we weight the loss across it, so the model spends its capacity where learning actually happens? Let us take the prediction target first.

## 2. The parameterization zoo: eps, x_0, and v

The forward equation `$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$` contains two unknowns, `$x_0$` and `$\varepsilon$`, tied together by the known `$x_t$` and the known schedule coefficients. Knowing any one of `$\{x_0, \varepsilon, x_t\}$` plus the schedule lets you solve for the others. That single linear relationship is the source of the entire "parameterization zoo." The network is a function of `$x_t$` and `$t$`; we get to choose which quantity it outputs. Figure 2 lays out the three standard choices and where each is well-conditioned.

![A matrix comparing eps-prediction, x_0-prediction, and v-prediction across what the network outputs, when each is numerically stable, and where it is used in practice](/imgs/blogs/noise-schedules-and-the-parameterization-zoo-2.png)

### 2.1 eps-prediction (DDPM)

The original DDPM (Ho et al., 2020) trains the network `$\varepsilon_\theta(x_t, t)$` to predict the noise that was added. The loss is the famous "simple" objective:

$$\mathcal{L}_\text{simple} = \mathbb{E}_{x_0, \varepsilon, t}\left[\, \lVert \varepsilon - \varepsilon_\theta(x_t, t) \rVert^2 \,\right].$$

Why predict the noise rather than the image? Two reasons. First, at low noise (`$t$` small) `$x_t \approx x_0$`, so predicting `$x_0$` is trivial and gives almost no gradient — but predicting `$\varepsilon$` is a meaningful task even there, because the model has to find the small amount of noise hiding in a clean image. Predicting `$\varepsilon$` keeps the target non-trivial across more of the range. Second, the `$\varepsilon$` target has fixed unit variance by construction (`$\varepsilon \sim \mathcal{N}(0,I)$`), so the regression target is normalized — the loss is on the same scale at every timestep, which is friendly to optimization.

But `$\varepsilon$`-prediction has a fatal weakness at the *other* end. As `$t \to T$`, `$x_t$` becomes almost pure noise, so `$x_t \approx \varepsilon$`, and predicting `$\varepsilon$` from an input that *is* essentially `$\varepsilon$` is again nearly trivial — but now in a dangerous way. To recover `$x_0$` from `$\varepsilon_\theta$`, you invert the forward equation:

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon_\theta}{\sqrt{\bar\alpha_t}}.$$

That `$1/\sqrt{\bar\alpha_t}$` factor explodes as `$\bar\alpha_t \to 0$`. A tiny error in `$\varepsilon_\theta$` at high noise gets amplified by an enormous factor when you map it back to image space. So `$\varepsilon$`-prediction is well-conditioned for recovering the noise but badly-conditioned for recovering the image precisely when the image content matters most for the sample's global structure (the high-noise steps set the layout). This is the seed of `$v$`-prediction.

### 2.2 x_0-prediction

The opposite choice: train `$x_{0,\theta}(x_t, t)$` to predict the clean image directly. The loss is `$\lVert x_0 - x_{0,\theta} \rVert^2$`. This is beautifully conditioned at high noise — recovering `$x_0$` from near-pure-noise is exactly the hard, meaningful task, and there is no exploding factor. To get the noise back you invert the same equation the other way:

$$\hat\varepsilon = \frac{x_t - \sqrt{\bar\alpha_t}\,x_{0,\theta}}{\sqrt{1-\bar\alpha_t}},$$

and now the dangerous factor is `$1/\sqrt{1-\bar\alpha_t}$`, which explodes as `$t \to 0$` (low noise). So `$x_0$`-prediction simply *swaps* which end is ill-conditioned. At low noise, predicting `$x_0 \approx x_t$` is trivial (no gradient) and recovering `$\varepsilon$` from it amplifies error. Neither pure parameterization is good at both ends. You can already feel the resolution coming: average them.

### 2.3 v-prediction: the stable compromise

Salimans and Ho introduced `$v$`-prediction in "Progressive Distillation for Fast Sampling of Diffusion Models" (2022). The velocity target is defined as

$$v = \sqrt{\bar\alpha_t}\,\varepsilon - \sqrt{1-\bar\alpha_t}\,x_0.$$

At first glance this is an arbitrary linear combination. It is not. Here is the geometry that makes it inevitable, and it is worth carrying in your head. Define an angle `$\phi_t$` by `$\cos\phi_t = \sqrt{\bar\alpha_t}$` and `$\sin\phi_t = \sqrt{1-\bar\alpha_t}$` — legitimate, because `$\bar\alpha_t + (1-\bar\alpha_t) = 1$`, so `$(\sqrt{\bar\alpha_t}, \sqrt{1-\bar\alpha_t})$` is a unit vector and lives on a quarter-circle. Then the forward process is a *rotation*:

$$x_t = \cos\phi_t\, x_0 + \sin\phi_t\, \varepsilon.$$

As `$t$` runs from `$0$` to `$T$`, `$\phi_t$` sweeps from `$0$` to `$\pi/2$`, rotating the "state" from pure `$x_0$` toward pure `$\varepsilon$`. The velocity `$v$` is then exactly the *tangent* to this circular path — the derivative of `$x_t$` with respect to the angle:

$$v = \frac{dx_t}{d\phi} = -\sin\phi_t\, x_0 + \cos\phi_t\, \varepsilon = \sqrt{\bar\alpha_t}\,\varepsilon - \sqrt{1-\bar\alpha_t}\,x_0.$$

Figure 3 draws this circle and the tangent.

![A diagram showing v-prediction as the tangent vector on the circle swept by signal and noise as the timestep advances, keeping the target magnitude bounded at every noise level](/imgs/blogs/noise-schedules-and-the-parameterization-zoo-6.png)

Why is this stable? Because `$x_t$` and `$v$` form an *orthonormal frame*: `$x_t$` points along the radius (the current state) and `$v$` points along the tangent (the direction of motion), and they are perpendicular unit-magnitude combinations. The target `$v$` therefore has bounded, roughly-constant magnitude at *every* noise level — it never explodes and never collapses to triviality. At `$t \to 0$` (`$\phi \to 0$`), `$v \to \varepsilon$`, so `$v$`-prediction behaves like `$\varepsilon$`-prediction where that is good. At `$t \to T$` (`$\phi \to \pi/2$`), `$v \to -x_0$`, so it behaves like `$x_0$`-prediction where *that* is good. v-prediction is the parameterization that automatically uses the well-conditioned target at each end and interpolates smoothly between them. That is the whole reason it is the default for distillation (where you take huge steps and error amplification is deadly), for high-resolution models (where the high-noise steps carry the global composition), and for SD 2.x.

Let me make the conditioning argument quantitative, because "well-conditioned" should not be a vibe. Define the conditioning of a parameterization at a noise level as the factor by which an error in the network's output gets amplified when you map it back to the quantity the sampler actually needs — the clean image `$x_0$`. For `$\varepsilon$`-prediction, that amplification is `$\partial \hat x_0 / \partial \varepsilon_\theta = -\sqrt{1-\bar\alpha_t}/\sqrt{\bar\alpha_t} = -\sqrt{1/\text{SNR}(t)}$`. As `$\text{SNR}(t) \to 0$` at high noise, this blows up like `$1/\sqrt{\text{SNR}}$` — an unbounded amplification exactly where global structure is decided. For `$x_0$`-prediction the amplification of `$x_0$`-error into `$x_0$` is, trivially, 1 — but the amplification of `$\varepsilon$`-error (which the SDE/ODE sampler also uses) goes as `$\sqrt{\text{SNR}(t)}$`, blowing up at low noise. For `$v$`-prediction, `$\partial \hat x_0/\partial v_\theta = -\sqrt{1-\bar\alpha_t} = -\sin\phi_t$`, whose magnitude is bounded by 1 *everywhere* on the circle, and the companion `$\partial \hat\varepsilon/\partial v_\theta = \sqrt{\bar\alpha_t} = \cos\phi_t$` is *also* bounded by 1. Both recovery directions are contractions, never expansions. That is the precise sense in which `$v$` is the conditioned choice: every quantity a sampler reads out of the network is a bounded linear combination of the network's output, with coefficients that are sines and cosines and therefore live in `$[-1, 1]$`. No division by a vanishing `$\sqrt{\bar\alpha_t}$` ever appears.

There is a second, subtler reason `$v$` helps that distillation papers lean on. When you distill a model into fewer steps, each student step has to emulate several teacher steps, which means it takes a *large* jump in `$x_t$`. Under `$\varepsilon$`-prediction, a large jump near the high-noise end multiplies the `$1/\sqrt{\bar\alpha_t}$` factor by the step size, and the compounding across distillation generations (you distill the student again and again, each time halving steps) makes the error grow geometrically. Under `$v$`-prediction the per-step error stays bounded, so the geometric compounding has a bounded base and the student stays close to the teacher even at one or two steps. This is not a marginal effect — it is the difference between a distilled model that converges and one that collapses to grey, which is why the progressive-distillation and consistency-model literatures standardized on `$v$` (or EDM's mathematically-equivalent preconditioning).

#### Worked example: the eps↔x_0↔v conversion triangle

Suppose at some timestep `$\bar\alpha_t = 0.36$`, so `$\sqrt{\bar\alpha_t} = 0.6$` and `$\sqrt{1-\bar\alpha_t} = 0.8$` (a clean 3-4-5 triangle). Say the true clean pixel value is `$x_0 = 1.0$` and the noise sample is `$\varepsilon = 0.5$`. Then:

- `$x_t = 0.6(1.0) + 0.8(0.5) = 0.6 + 0.4 = 1.0$`.
- The `$\varepsilon$`-target is `$0.5$`; the `$x_0$`-target is `$1.0$`; the `$v$`-target is `$v = 0.6(0.5) - 0.8(1.0) = 0.3 - 0.8 = -0.5$`.

Now check the conversions a sampler relies on. Given a perfect `$v_\theta = -0.5$`, recover `$x_0$` via `$\hat{x}_0 = \sqrt{\bar\alpha_t}\,x_t - \sqrt{1-\bar\alpha_t}\,v = 0.6(1.0) - 0.8(-0.5) = 0.6 + 0.4 = 1.0$`. Correct. Recover `$\hat\varepsilon = \sqrt{1-\bar\alpha_t}\,x_t + \sqrt{\bar\alpha_t}\,v = 0.8(1.0) + 0.6(-0.5) = 0.8 - 0.3 = 0.5$`. Correct. The point of the example: the three targets are *exactly* inter-convertible through the schedule coefficients, so any model can be evaluated in any parameterization at sampling time — but the *gradient* the model receives during training depends on which target you regress, and that is what changes stability. Notice that `$v = -0.5$` has the same magnitude as `$\varepsilon = 0.5$` here; at `$\bar\alpha_t \to 0$` the `$\hat x_0$` recovery from `$\varepsilon$` would have divided by `$\sqrt{\bar\alpha_t} \to 0$`, while the `$v$` recovery `$\hat x_0 = \sqrt{\bar\alpha_t}x_t - \sqrt{1-\bar\alpha_t}v$` uses *multiplication* by small numbers — bounded, never an explosion.

### 2.4 v-prediction in PyTorch

Here is the training target and loss, written the way you would drop it into a real training loop. The conversions are exactly the algebra above.

```python
import torch

def get_v_target(x0, eps, alpha_bar_t):
    # x0, eps: (B, C, H, W); alpha_bar_t: (B,) cumulative product at sampled t.
    a = alpha_bar_t.sqrt().view(-1, 1, 1, 1)        # sqrt(alpha_bar)
    b = (1.0 - alpha_bar_t).sqrt().view(-1, 1, 1, 1) # sqrt(1 - alpha_bar)
    return a * eps - b * x0                           # v = sqrt(ab) eps - sqrt(1-ab) x0

def v_prediction_loss(model, x0, t, alpha_bar):
    eps = torch.randn_like(x0)
    abar_t = alpha_bar[t]                             # gather per-sample alpha_bar
    a = abar_t.sqrt().view(-1, 1, 1, 1)
    b = (1.0 - abar_t).sqrt().view(-1, 1, 1, 1)
    x_t = a * x0 + b * eps                            # forward diffusion
    v_target = a * eps - b * x0                       # the regression target
    v_pred = model(x_t, t)                            # network predicts v
    return torch.nn.functional.mse_loss(v_pred, v_target)

# Recover x0 / eps from a v-prediction at sampling time:
def v_to_x0_eps(x_t, v, abar_t):
    a = abar_t.sqrt().view(-1, 1, 1, 1)
    b = (1.0 - abar_t).sqrt().view(-1, 1, 1, 1)
    x0  = a * x_t - b * v
    eps = b * x_t + a * v
    return x0, eps
```

That is the entire change to go from `$\varepsilon$`-prediction to `$v$`-prediction: compute a different target, regress it, and convert back at sample time. In 🤗 `diffusers` it is a one-word flag on the scheduler, which we will see in §7.

## 3. From the variational bound to L_simple as an SNR weighting

The simple loss `$\mathcal{L}_\text{simple} = \mathbb{E}\lVert \varepsilon - \varepsilon_\theta \rVert^2$` looks like it fell from the sky, but it is a *reweighting* of the principled variational bound. Understanding that reweighting is the key that unlocks every loss-weighting trick in §5, so let us derive it carefully. This is the science block.

The diffusion ELBO decomposes into a sum of per-timestep KL divergences `$L_{t-1} = D_\text{KL}\!\left(q(x_{t-1}\mid x_t, x_0) \,\Vert\, p_\theta(x_{t-1}\mid x_t)\right)$`. Both distributions are Gaussian. The true posterior `$q(x_{t-1}\mid x_t, x_0)$` has a closed-form mean `$\tilde\mu_t(x_t, x_0)$`, and the model `$p_\theta$` has mean `$\mu_\theta(x_t, t)$` with the same (fixed) variance. The KL between two Gaussians with equal variance is just the scaled squared distance between their means:

$$L_{t-1} = \frac{1}{2\sigma_t^2}\,\lVert \tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\rVert^2 + C.$$

Now substitute the `$\varepsilon$`-parameterization. The true posterior mean and the model mean, both written in terms of `$\varepsilon$` and `$\varepsilon_\theta$`, differ only in the noise term, and after the algebra (Ho et al., 2020, eq. 12) the per-step loss becomes

$$L_{t-1} = \underbrace{\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}}_{w_t}\, \lVert \varepsilon - \varepsilon_\theta(x_t, t)\rVert^2.$$

So the *correct* variational loss is `$\sum_t w_t \lVert \varepsilon - \varepsilon_\theta\rVert^2$` with a timestep-dependent weight `$w_t$`. DDPM's key empirical finding was that **dropping `$w_t$` entirely** — setting all weights to 1 — gives *better samples* than keeping the principled weight. That stripped-down version is `$\mathcal{L}_\text{simple}$`. It is not a different objective; it is the variational bound with a particular, deliberately-flat weighting.

Here is the crucial reinterpretation. The weight `$w_t$` is large at small `$t$` (low noise) and small at large `$t$` (high noise) — the variational bound cares most about the easy, low-noise steps. By flattening the weights to 1, `$\mathcal{L}_\text{simple}$` *up-weights the high-noise steps* relative to the bound. And it turns out that what your samples look like is dominated by the high-noise steps (they set the global structure), so this reweighting trades a tiny bit of likelihood for a large gain in perceptual quality. **`$\mathcal{L}_\text{simple}$` is the variational loss reweighted toward the steps humans care about.** Once you see it this way, "which loss weighting?" becomes the real design question, and min-SNR and P2 are just better answers to it.

To make the SNR connection explicit, rewrite the same loss in terms of SNR. A clean way to see it: the `$\varepsilon$`-prediction MSE, when converted to an `$x_0$`-prediction MSE, picks up an SNR factor. Specifically `$\lVert \varepsilon - \varepsilon_\theta\rVert^2 = \text{SNR}(t)\,\lVert x_0 - \hat{x}_0\rVert^2$`, because the map from `$x_0$`-error to `$\varepsilon$`-error multiplies by `$\sqrt{\bar\alpha_t}/\sqrt{1-\bar\alpha_t} = \sqrt{\text{SNR}(t)}$` (squared in the norm). So:

$$\mathcal{L}_\text{simple} = \mathbb{E}_t\big[\, \text{SNR}(t)\,\lVert x_0 - \hat{x}_0(x_t,t)\rVert^2\,\big].$$

Read in `$x_0$`-space, the "simple" loss is **an SNR-weighting of the clean-image reconstruction error**. High-SNR (easy) steps are weighted heavily; low-SNR (hard) steps barely at all. That is the opposite of where you want the model to focus, which is exactly the problem min-SNR fixes. Every weighting scheme in §5 is a choice of a function `$w(\text{SNR})$` multiplying this same reconstruction error.

It is worth being explicit about the change-of-variables between parameterizations, because it is the source of endless confusion and silent bugs. The *same physical loss* has a different algebraic form depending on which space you write it in, and the conversion factor is always a power of SNR:

$$\lVert \varepsilon - \varepsilon_\theta\rVert^2 \;=\; \text{SNR}(t)\,\lVert x_0 - \hat x_0\rVert^2 \;=\; \frac{1}{1 + \text{SNR}(t)}\,\lVert v - v_\theta\rVert^2.$$

The last equality is the one people get wrong. The `$v$`-target's norm relates to the `$\varepsilon$`-target's by a factor `$1/(1+\text{SNR})$`, which is why the min-SNR weight in `$v$`-space is `$\min(\text{SNR},\gamma)/(1+\text{SNR})$` and not `$\min(\text{SNR},\gamma)/\text{SNR}$` (the `$\varepsilon$`-space form). If you implement min-SNR by copying the `$\varepsilon$`-space formula into a `$v$`-prediction run, you mis-weight every timestep and your FID lands worse than the paper for no obvious reason. We will return to this exact footgun in the §5 code. The general lesson: **always know which space your loss lives in**, because the "natural" unit-weight loss in one space is a steep SNR weighting in another. There is no parameterization-free notion of "uniform weighting" — uniform in `$\varepsilon$`-space is `$\text{SNR}$`-weighted in `$x_0$`-space is `$\text{SNR}/(1+\text{SNR})$`-weighted in `$v$`-space. Picking the weighting *is* the design decision; the parameterization just renames where the SNR factor lives.

One more derivation makes the variational connection airtight and explains *why* the field tolerates throwing away the principled weight. The true per-step weight `$w_t = \beta_t^2 / (2\sigma_t^2\alpha_t(1-\bar\alpha_t))$`, when you substitute the DDPM variance choice `$\sigma_t^2 = \beta_t$` and simplify, is proportional to the *derivative of SNR* across the step, `$w_t \propto \text{SNR}(t-1) - \text{SNR}(t)$`. So the variational bound weights each step by how much SNR it removes. Because the linear schedule removes most of its SNR early (at low `$t$`), the bound piles weight on the low-noise steps — the perceptually least important ones, the ones that only sharpen already-formed images. `$\mathcal{L}_\text{simple}$`'s flat weighting redistributes that budget toward higher noise, where the perceptually decisive global structure forms. That is not a hack; it is a principled statement that *likelihood and perceptual quality disagree about which noise levels matter*, and a generator should optimize for the latter. min-SNR and P2 are simply more surgical redistributions of the same budget.

## 4. Noise schedules: linear, cosine, and beyond

Now the second knob. A schedule is a choice of how `$\bar\alpha_t$` (equivalently SNR) slides from 1 to 0 over the `$T$` steps. The schedule decides *which noise levels the model trains on and how often*, because during training you sample `$t$` uniformly and apply the schedule's noise level — so a schedule that spends many steps near pure noise trains the model heavily on near-pure-noise inputs, where almost nothing is learnable. Figure 4 contrasts the two canonical schedules through their SNR profiles.

![A before-and-after diagram comparing the linear beta schedule against the cosine schedule, showing the cosine schedule holds signal-to-noise ratio higher for longer and wastes less training on near-pure-noise inputs](/imgs/blogs/noise-schedules-and-the-parameterization-zoo-3.png)

### 4.1 Linear (DDPM)

The original schedule sets `$\beta_t$` to increase linearly from `$\beta_1 = 10^{-4}$` to `$\beta_T = 0.02$` over `$T = 1000$` steps, with `$\alpha_t = 1 - \beta_t$` and `$\bar\alpha_t$` the cumulative product. The problem, identified by Nichol and Dhariwal ("Improved Denoising Diffusion Probabilistic Models", 2021), is that the linear `$\beta$` schedule destroys information *too quickly at the end*. By the time you reach the last 10-20% of timesteps, `$\bar\alpha_t$` is already so close to zero that `$x_t$` is indistinguishable from pure noise — those steps contribute almost nothing because there is no signal left to denoise. The model wastes a large fraction of its training budget on steps where the task is "predict noise from noise." On low-resolution images (CIFAR, ImageNet 64) this is especially wasteful because there is less signal to begin with.

### 4.2 Cosine (Improved DDPM)

The cosine schedule fixes this by defining `$\bar\alpha_t$` directly (rather than via `$\beta_t$`):

$$\bar\alpha_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^2\!\left(\frac{t/T + s}{1 + s}\cdot\frac{\pi}{2}\right),$$

with a small offset `$s = 0.008$` to prevent `$\beta_t$` from being too small near `$t = 0$`. The key property: `$\bar\alpha_t$` follows a `$\cos^2$` curve, which is nearly linear in the middle and *flattens at both ends*. Concretely, it keeps SNR higher for longer and rolls off gently into the high-noise regime instead of crashing. The result is that the model spends far more of its training budget in the productive middle band of noise levels. Nichol and Dhariwal reported the cosine schedule improving FID on ImageNet 64×64 (roughly 19.5 → 17.5 in their ablations) and, more importantly, improving the *log-likelihood* substantially. The cosine schedule is the default for many models trained at modest resolution.

### 4.3 Sigmoid, Laplace, and the continuous view

Once you accept that "a schedule is an SNR curve," you can design the curve directly. Several families do exactly this:

- **Sigmoid schedule.** Parameterize log-SNR as a sigmoid of `$t$`, giving an S-shaped roll-off you can tune with a temperature and a bias. Used in some high-resolution recipes because the sigmoid's tails control how much time is spent at the very-clean and very-noisy ends independently.
- **Laplace schedule.** Recent work (the "Improved Noise Schedule" line, 2024) chooses the *sampling distribution over log-SNR* to be a Laplace (double-exponential) centered at the mid-noise region, concentrating training steps where the loss is most informative. Reported FID gains on ImageNet come purely from where the noise levels are sampled, with no architecture change.
- **The continuous reframing (EDM).** Karras et al. ("Elucidating the Design Space of Diffusion-Based Generative Models", 2022) threw out the `$\bar\alpha_t$` parameterization entirely and worked directly with a noise level `$\sigma$`, where `$x = x_0 + \sigma\,\varepsilon$`. The "schedule" becomes a distribution over `$\log\sigma$` (they use a Gaussian, `$\ln\sigma \sim \mathcal{N}(P_\text{mean}, P_\text{std}^2)$` with `$P_\text{mean}=-1.2$`, `$P_\text{std}=1.2$`), and the network has a principled preconditioning that makes its effective target well-scaled at every `$\sigma$` — essentially `$v$`-prediction's stability generalized. EDM's design is the cleanest modern statement of "the schedule is a sampling distribution over noise levels, choose it to put steps where they matter."

The unifying message: stop thinking of a schedule as a `$\beta_t$` array and start thinking of it as a curve of log-SNR over time, plus a distribution over where you sample noise levels during training. Everything good follows from shaping that curve sensibly.

There is a practical question hiding here that the literature rarely states plainly: *how do you actually choose a schedule for a new dataset or resolution?* The honest workflow is not "derive it from first principles" — it is "look at your per-timestep loss curve and rebalance." Train for a few thousand steps with whatever schedule you inherited, then bin the loss by timestep (or by log-SNR) and plot it. You will see one of two pathologies. If the loss is flat-and-low across a big chunk of high-`$t$` steps, those steps have already saturated — the model learned the trivial "predict noise from noise" task and is now getting zero useful gradient there, so your schedule is wasting budget at the destroyed end (the linear-schedule disease) and you should move toward cosine or sample fewer steps there. If instead the loss is steep and still-improving in the mid-SNR band while the clean end converged long ago, your schedule is fine but your *weighting* is over-counting the easy end — reach for min-SNR. The per-timestep loss curve is the single most informative diagnostic in diffusion training, and almost nobody plots it. It turns schedule selection from guesswork into a closed-loop adjustment: measure where the gradient is wasted, move budget away from there, repeat.

A note on the sigmoid and Laplace families specifically, since they are the 2024-era refinements you will see in recent papers. Both are answers to the same question — "where in log-SNR should I concentrate training steps?" — and both beat the cosine schedule at high resolution for the same reason cosine beats linear at low resolution: they push more of the training budget into the band of noise levels that are actually informative for *that* data and resolution. The sigmoid schedule gives you two tunable knobs (a center and a width on the log-SNR axis), which is enough to match most datasets; the Laplace schedule's sharper peak concentrates even harder on the center and tends to win when compute is the binding constraint (it converges to a given FID in fewer steps because almost none of the budget is wasted at the tails). The lesson is not "always use Laplace" — it is that the schedule's interior shape is a tunable hyperparameter with real FID consequences, not a constant of nature, and the right shape depends on your resolution and compute budget.

### 4.4 Plotting SNR for linear vs cosine

Seeing it beats describing it. Here is the code to compute and plot both schedules' SNR curves. (Note the escaped dollar signs in the mathtext labels.)

```python
import numpy as np
import matplotlib.pyplot as plt

T = 1000
t = np.arange(T)

# Linear beta schedule (DDPM)
beta_lin = np.linspace(1e-4, 0.02, T)
abar_lin = np.cumprod(1.0 - beta_lin)

# Cosine schedule (Improved DDPM)
s = 0.008
f = np.cos(((t / T + s) / (1 + s)) * np.pi / 2) ** 2
abar_cos = f / f[0]

snr_lin = abar_lin / (1 - abar_lin)
snr_cos = abar_cos / (1 - abar_cos)

plt.figure(figsize=(7, 4))
plt.plot(t, np.log(snr_lin), label="linear")
plt.plot(t, np.log(snr_cos), label="cosine")
plt.axhline(0.0, ls="--", c="gray", lw=0.8)  # log-SNR = 0  (SNR = 1)
plt.xlabel("timestep t")
plt.ylabel(r"log-SNR  $\log(\bar\alpha_t / (1-\bar\alpha_t))$")
plt.title("SNR profile: linear vs cosine")
plt.legend()
plt.tight_layout()
plt.savefig("snr_curves.png", dpi=120)
```

Run it and you will see the cosine curve sitting *above* the linear curve through most of the range and crossing log-SNR = 0 (the "half signal, half noise" line) much later — concrete evidence that cosine holds signal longer and spends less time in the dead zone. The crossing point of log-SNR = 0 is a good single-number summary of "where is the middle of this schedule's difficulty."

## 5. Loss weighting: rebalancing which noise levels matter

We established in §3 that the loss is `$\mathbb{E}_t[\,w(\text{SNR}(t))\,\lVert x_0 - \hat x_0\rVert^2\,]$` for some weight function, and that `$\mathcal{L}_\text{simple}$` corresponds to `$w = \text{SNR}$` (in `$x_0$`-space) — which over-weights easy steps. The schedule controls *which* noise levels appear and how often; the weighting controls *how much each one counts in the gradient*. They are complementary knobs on the same SNR axis. Figure 5 stacks the standard weighting choices.

![A vertical stack showing the loss-weighting ladder from the variational bound through L_simple, min-SNR-gamma, P2 weighting, and EDM sigma-weighting, each as the same MSE reweighted by a function of the signal-to-noise ratio](/imgs/blogs/noise-schedules-and-the-parameterization-zoo-4.png)

### 5.1 The conflict between objectives

Hang et al. ("Efficient Diffusion Training via Min-SNR Weighting Strategy", 2023) framed the problem cleanly: diffusion training is *multi-task learning*, where each noise level is a task, and the tasks *conflict*. Easy (high-SNR) tasks have large, confident gradients; hard (low-SNR) tasks have small, noisy gradients. If you weight them all the same in `$\varepsilon$`-space (`$\mathcal{L}_\text{simple}$`), the model still ends up dominated by certain levels because the gradient *magnitudes* differ across SNR. Plot the loss per timestep during training and you will see it is wildly uneven — some timesteps converge in a few thousand steps and then contribute only noise to the gradient, while others are still improving. That imbalance slows convergence and the empirically-observed result is that naive training is slower and lands at a worse FID than it should.

### 5.2 min-SNR-γ

The min-SNR-`$\gamma$` weighting clamps the SNR-based weight at a ceiling `$\gamma$`:

$$w_t = \frac{\min\{\text{SNR}(t),\, \gamma\}}{\text{SNR}(t)} \quad \text{(in } \varepsilon\text{-space)}, \qquad \text{or equivalently} \qquad w_t = \min\{\text{SNR}(t), \gamma\} \;\text{(in } x_0\text{-space)}.$$

The intuition: high-SNR steps would otherwise dominate (their `$x_0$`-space weight is huge), so cap their contribution at `$\gamma$`; below the cap, leave the weighting proportional to SNR. The paper recommends `$\gamma = 5$`. This single change gave a **3.4× speedup** in training convergence to a target FID on ImageNet 256 with a ViT-based diffusion model, and improved the final FID — purely from rebalancing the loss across noise levels, no architecture change. It also makes `$v$`-prediction and `$\varepsilon$`-prediction converge to similar quality, because the clamp neutralizes the conditioning differences between parameterizations. min-SNR is one of the highest-leverage, lowest-effort changes you can make to a diffusion training run.

### 5.3 P2 weighting

Choi et al. ("Perception Prioritized Training of Diffusion Models", 2022) proposed P2 weighting:

$$w_t = \frac{1}{(1 + \text{SNR}(t))^{\gamma}},$$

with `$\gamma = 1$` as a common choice. P2 down-weights the high-SNR (clean) steps — which it argues correspond to *imperceptible* fine details the model does not need to spend capacity on — and concentrates the loss on the mid-SNR steps where *perceptually important content* (object shapes, coarse structure) is formed. It is the same philosophy as min-SNR (rebalance toward the informative middle) with a smooth weighting instead of a clamp. On its target datasets P2 improved FID meaningfully (e.g., on CelebA-HQ and several others) by reallocating capacity to the perceptually-relevant noise band.

### 5.4 Implementing min-SNR weighting in PyTorch

Here is the weighting applied to a `$v$`-prediction loss — the combination most people actually use in a modern run.

```python
import torch

def min_snr_weight(alpha_bar_t, gamma=5.0, pred_type="v"):
    # SNR(t) = alpha_bar / (1 - alpha_bar)
    snr = alpha_bar_t / (1.0 - alpha_bar_t)
    clamped = torch.clamp(snr, max=gamma)        # min(SNR, gamma)
    if pred_type == "eps":
        return clamped / snr                      # eps-space weight
    elif pred_type == "v":
        return clamped / (snr + 1.0)              # v-space weight
    elif pred_type == "x0":
        return clamped                            # x0-space weight
    raise ValueError(pred_type)

def weighted_loss(model, x0, t, alpha_bar, gamma=5.0):
    eps   = torch.randn_like(x0)
    abar  = alpha_bar[t]
    a     = abar.sqrt().view(-1, 1, 1, 1)
    b     = (1.0 - abar).sqrt().view(-1, 1, 1, 1)
    x_t   = a * x0 + b * eps
    v_tgt = a * eps - b * x0
    v_pred = model(x_t, t)
    per_elem = (v_pred - v_tgt) ** 2
    per_sample = per_elem.mean(dim=[1, 2, 3])     # MSE per image
    w = min_snr_weight(abar, gamma=gamma, pred_type="v")
    return (w * per_sample).mean()                # SNR-rebalanced loss
```

The `pred_type` branch matters: the *same* min-SNR ceiling produces a different multiplier depending on whether your loss is measured in `$\varepsilon$`-, `$x_0$`-, or `$v$`-space, because the parameterization already bakes in an SNR factor. Getting this branch wrong silently mis-weights your training, which is a classic "my FID is worse than the paper" footgun. The 🤗 `diffusers` training scripts implement exactly this (`snr = ...; mse_loss_weights = torch.stack([snr, gamma * torch.ones_like(t)]).min(dim=0)[0] / snr` for eps, with the `+1` adjustment for v).

## 6. Zero-terminal-SNR: the brightness bug nobody noticed

Now the bug from the intro. This is the single most important practical takeaway in the post, because it is real, it is in shipped models, and the fix is small. The reference is Lin et al., "Common Diffusion Noise Schedules and Sample Steps are Flawed" (2023). Figure 6 shows the before and after.

![A before-and-after diagram contrasting a schedule that leaks signal at the final timestep against a rescaled schedule that enforces exactly zero terminal signal-to-noise ratio, removing the train-test brightness mismatch](/imgs/blogs/noise-schedules-and-the-parameterization-zoo-5.png)

### 6.1 The mismatch

At sampling time you start from `$x_T \sim \mathcal{N}(0, I)$` — pure Gaussian noise, zero signal. The implicit assumption is that the forward process at `$t = T$` produces pure noise too, so that train and test see the same distribution at the start. But check the numbers. For SD 1.5's schedule (scaled-linear, `$\beta_1 = 0.00085$`, `$\beta_T = 0.012$`, `$T = 1000$`), the terminal cumulative product is

$$\bar\alpha_T \approx 0.0047, \qquad \sqrt{\bar\alpha_T} \approx 0.068.$$

That is **not zero**. At the last forward step, `$x_T = 0.068\,x_0 + 0.998\,\varepsilon$` — about 7% of the clean image *leaks through*. So during training, the model at `$t = T$` always sees a faint ghost of the real image. At sampling time, it sees pure noise with no ghost. The input distributions do not match. The model has learned to expect a little bit of signal at the start and, finding none, it falls back on the safest possible guess: the *mean of the training data*. And the mean of a natural-image dataset is a medium grey. That is precisely why these models cannot render a true black or a true white — the very first denoising step is biased toward the dataset's mean luminance, and that bias propagates all the way to the final image.

### 6.2 The leak is a mean leak

To see *why* it specifically biases brightness, look at what the ghost carries. The leaked term is `$\sqrt{\bar\alpha_T}\,x_0$`. Averaged over a patch, the dominant low-frequency component of `$x_0$` is its mean brightness. So the leak is, to first order, a leak of the *image's average luminance* into the starting point. A model trained to use that leak will, at sampling time, hallucinate a luminance close to the *dataset average* (since it has no real `$x_0$` to read). Datasets skew toward mid-luminance, so outputs cluster around medium brightness. Request "pure black" and the model literally cannot place enough mass below the leaked mean it expects. This is not a contrast-curve cosmetic issue — it is a distributional bug at the very first step.

The reason it is specifically the *lowest* spatial frequency — the overall brightness — that leaks, rather than fine detail, is worth spelling out because it explains the exact visual signature. Diffusion's forward process destroys high spatial frequencies first and low spatial frequencies last: noise is high-frequency by nature, so as you add it the fine textures vanish into the noise floor long before the broad gradients and the overall luminance do. At the terminal step, after almost everything is gone, what survives in `$\sqrt{\bar\alpha_T}\,x_0$` is the most robust, lowest-frequency component — and the single lowest-frequency component of any image is its DC term, the mean. So the leak is not a leak of "a bit of the image"; it is a leak of *precisely the image's mean brightness and broadest color cast*, the components that survive longest under noising. That is why the symptom is so specifically a brightness-and-mean-color bug and not, say, a blurry-texture bug. The model learns to read the mean off the leak, and at sampling time, deprived of it, defaults to the dataset's mean. A model trained on LAION (which skews toward well-lit, mid-luminance web images) will pull every output toward that mid-luminance, and the failure is most visible exactly when the prompt asks for an *extreme* mean — pure black, pure white, a single saturated color filling the frame — because those are the means furthest from the dataset average the model fell back on.

This also explains a confusing observation people report: the bias is *worse* for solid-color or high-key/low-key images than for busy, mid-toned scenes. A busy scene's mean is already near the dataset mean, so the leak's pull lands roughly where the image wanted to be and you never notice. A pure-black request's mean is as far from the dataset mean as possible, so the pull is maximally wrong and maximally visible. The bug was hiding in plain sight for years precisely because the images people usually generate — portraits, landscapes, products on neutral backgrounds — have means close enough to the dataset average that the leak's pull is invisible. It is only when you ask for an extreme that the schedule's failure to reach pure noise becomes a failure you can see.

### 6.3 The fix: rescale to enforce SNR(T) = 0

The fix is to rescale the `$\sqrt{\bar\alpha_t}$` schedule so that `$\sqrt{\bar\alpha_T} = 0$` exactly, while leaving `$\sqrt{\bar\alpha_1}$` (the low-noise end) essentially unchanged. Lin et al. give a clean linear rescaling of `$\sqrt{\bar\alpha_t}$`:

```python
import torch

def enforce_zero_terminal_snr(betas):
    # betas: (T,) the original beta schedule
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    a0 = alphas_bar_sqrt[0].clone()    # sqrt(abar) at t=0  (~1)
    aT = alphas_bar_sqrt[-1].clone()   # sqrt(abar) at t=T  (~0.068, the leak)

    # Shift so the last value is exactly 0, then scale so the first is unchanged.
    alphas_bar_sqrt -= aT
    alphas_bar_sqrt *= a0 / (a0 - aT)

    # Convert back to betas.
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1.0 - alphas
    return betas
```

Two subtleties the paper insists on, and they matter as much as the rescale:

1. **You must also use a sampler that hits `$t = T$`.** The default DDIM step spacing in many libraries *skips* the final timestep (it uses "trailing" or "leading" spacing that never evaluates `$t=T$`), so even with a fixed schedule the model never starts from true pure noise. Set the timestep spacing to `trailing` so the first sampled step is the terminal one.
2. **You must rescale classifier-free guidance.** With the corrected schedule, high guidance scales over-expose the image (because the now-correct dynamic range interacts with CFG's extrapolation). Lin et al.'s "rescale CFG" trick renormalizes the guided prediction's standard deviation back to the unguided one, with a `guidance_rescale` factor around 0.7. Without this, the brightness fix trades grey-mush for blown-out highlights.

Do all three — zero-terminal-SNR schedule, trailing timesteps, CFG rescale — and the model can finally render true black, true white, and the full dynamic range. The paper shows it qualitatively (dark and bright prompts that previously failed now work) and reports improved FID, because the corrected dynamic range matches the reference distribution better. Figure 7 tabulates which schedules leak and what symptom each produces.

![A matrix mapping each common noise schedule to its terminal signal-to-noise ratio, how much mean brightness leaks through, and the resulting visible symptom in generated samples](/imgs/blogs/noise-schedules-and-the-parameterization-zoo-7.png)

#### Worked example: how much black can SD 1.5 actually render

Take a target output of a pure-black region, `$x_0 = -1$` in the `$[-1, 1]$` normalized pixel convention (black). At the first denoising step the model starts from `$x_T \sim \mathcal{N}(0, I)$`, mean 0. It has learned that at `$t=T$` the input mean should be `$\sqrt{\bar\alpha_T}\,\bar{x}_0 \approx 0.068 \times \bar{x}_0$`, where `$\bar x_0$` is the dataset mean luminance — say a mid-grey `$\bar x_0 \approx 0$` in the normalized convention. The model's best `$\hat x_0$` at the first step is pulled toward that learned expectation rather than toward `$-1$`. Quantitatively, the achievable mean of the first-step `$\hat x_0$` is bounded away from `$-1$` by roughly the leaked-mean term; empirically Lin et al. found the darkest the broken model reliably produces sits well above true black, and the brightest well below true white — a compressed range of maybe 30-70% of full. After the fix, `$\sqrt{\bar\alpha_T} = 0$`, the learned expectation at `$t=T$` is exactly the zero-mean prior, and the model is free to drive `$\hat x_0$` all the way to `$-1$`. The dynamic range opens up to the full `$[-1, 1]$`. The "bug" was a 0.068 coefficient.

## 7. Putting it together in 🤗 diffusers

Enough theory. Here is how every one of these choices surfaces in the actual toolchain, so you can flip them on a real pipeline. This is the practical flow.

### 7.1 Setting the prediction type and schedule

The scheduler object carries both the prediction parameterization and the noise schedule. Switching `$\varepsilon$`→`$v$` or linear→cosine is a config change, not a retrain of the architecture (though it *is* a retrain of the weights — the parameterization changes the target).

```python
from diffusers import DDPMScheduler

# eps-prediction, scaled-linear schedule (SD 1.5 default)
sched_eps = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085, beta_end=0.012,
    beta_schedule="scaled_linear",
    prediction_type="epsilon",
)

# v-prediction, same schedule (SD 2.1 style) — the stable target
sched_v = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085, beta_end=0.012,
    beta_schedule="scaled_linear",
    prediction_type="v_prediction",
)

# cosine-style schedule via squaredcos_cap_v2 (improved-DDPM cosine)
sched_cos = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",
    prediction_type="v_prediction",
)

# diffusers computes the v-target for you during training:
import torch
x0 = torch.randn(4, 4, 64, 64)                 # latents
noise = torch.randn_like(x0)
t = torch.randint(0, 1000, (4,))
noisy = sched_v.add_noise(x0, noise, t)
v_target = sched_v.get_velocity(x0, noise, t)  # built-in v target
# loss = F.mse_loss(model(noisy, t), v_target)
```

### 7.2 Enabling zero-terminal-SNR and the sampler fixes

This is the configuration that ships the brightness fix. The scheduler exposes `rescale_betas_zero_snr` and `timestep_spacing`, and the guidance rescale is a pipeline call argument.

```python
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# Rebuild the scheduler with the three fixes from Lin et al. (2023):
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config,
    rescale_betas_zero_snr=True,   # enforce SNR(T) = 0
    timestep_spacing="trailing",   # actually sample t = T
    prediction_type="v_prediction" # v-pred pairs naturally with zero-terminal-SNR
)

image = pipe(
    "a solid pure black background, high contrast studio photo",
    num_inference_steps=30,
    guidance_scale=7.5,
    guidance_rescale=0.7,          # renormalize CFG std (avoids over-exposure)
).images[0]
```

A practical caveat: applying `rescale_betas_zero_snr=True` and `timestep_spacing="trailing"` to a model that was *trained without* the fix will not magically fix it — the model never learned to start from true pure noise. The fix is a *training-time* change; the inference flags only realize its benefit on a model that was trained (or fine-tuned) with the corrected schedule. If you are fine-tuning an existing model, fine-tune with the corrected schedule for a few thousand steps to adapt the terminal-step behavior, then enable the inference flags.

### 7.3 A full training step with all the choices

Here is a complete, idiomatic training step that combines v-prediction, a cosine schedule, and min-SNR weighting — the modern default stack — using `diffusers` for the scheduler math.

```python
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="squaredcos_cap_v2",   # cosine
    prediction_type="v_prediction",
    rescale_betas_zero_snr=True,         # zero terminal SNR at train time
)
abar = scheduler.alphas_cumprod          # (1000,) precomputed

def training_step(unet, latents, encoder_hidden_states, gamma=5.0):
    b = latents.shape[0]
    noise = torch.randn_like(latents)
    t = torch.randint(0, scheduler.config.num_train_timesteps, (b,), device=latents.device)

    noisy = scheduler.add_noise(latents, noise, t)
    target = scheduler.get_velocity(latents, noise, t)     # v-target
    pred = unet(noisy, t, encoder_hidden_states).sample

    # min-SNR-gamma weighting in v-space
    snr = abar[t] / (1.0 - abar[t])
    snr = snr.to(latents.device)
    clamped = torch.clamp(snr, max=gamma)
    w = clamped / (snr + 1.0)                               # v-space weight

    loss = (w * F.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])).mean()
    return loss
```

That single function encodes four of this post's decisions — v-prediction, cosine schedule, zero-terminal-SNR, and min-SNR weighting — and it is the recipe a 2024-2026 latent-diffusion fine-tune would actually use. Swap `squaredcos_cap_v2` for `scaled_linear` to go back to the SD 1.5 schedule, or set `gamma=float("inf")` to recover unweighted `$\mathcal{L}_\text{simple}$`.

## 8. Resolution-dependent schedule shift

There is one more quiet choice, and it is the one that separates a schedule tuned at 256×256 from one that works at 1024×1024. Stable Diffusion 3 (Esser et al., 2024) made it explicit, but the effect is general. Figure 8 shows the logic.

![A branching diagram showing how higher resolution raises the effective signal-to-noise ratio because noise averages out over more pixels, so the schedule must be shifted toward higher noise to stay balanced](/imgs/blogs/noise-schedules-and-the-parameterization-zoo-8.png)

### 8.1 Why more pixels means more effective signal

Here is the physics. Add the same per-pixel noise level `$\sigma$` to a 256×256 image and to a 1024×1024 image. The 1024 image has 16× more pixels. The *signal* — the actual scene content — is highly spatially correlated (neighboring pixels are similar), so it does not get diluted by having more pixels. But the *noise* is independent per pixel, so when the network attends to a region or downsamples, the noise *averages out* faster the more pixels you have. Concretely, if you look at any fixed spatial frequency band, a higher-resolution image at the same per-pixel `$\sigma$` has a *higher* signal-to-noise ratio in that band, because there are more independent noise samples to average over for the same amount of correlated signal. So the *same* nominal noise level is *effectively less destructive* at high resolution.

The consequence: a schedule tuned at 256px is, at 1024px, *too easy* — it never noises the image enough at the high-noise end to actually destroy the global structure, because the structure survives the averaging. The model trained at 1024 with a 256-tuned schedule under-trains the high-noise (global-composition) steps, and you get incoherent global layout — beautiful textures, wrong composition. You need to *shift the schedule toward higher noise* at higher resolution to compensate.

### 8.2 The SD3 timestep shift

SD3 applies a *shift* to the (flow-matching) noise schedule, parameterized by a single number `$s$` (the "shift"). In the flow-matching `$\sigma \in [0, 1]$` parameterization the shift maps:

$$\sigma' = \frac{s\,\sigma}{1 + (s - 1)\,\sigma}.$$

With `$s = 1$` nothing changes; with `$s > 1$` the schedule is pushed toward higher noise (more time spent in the high-noise regime). SD3 uses a resolution-dependent shift — roughly `$s \approx 3$` at 1024px — and the value is chosen so the *effective* noise distribution at high resolution matches the well-tuned one at the base resolution. The flow-matching post in this series, [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow), covers the `$\sigma$` parameterization the shift acts on; here the point is just that the shift is the resolution-aware version of "choose your SNR curve." In `diffusers` it surfaces as the `shift` argument on `FlowMatchEulerDiscreteScheduler`:

```python
from diffusers import FlowMatchEulerDiscreteScheduler

# base schedule tuned at low resolution
sched_lowres = FlowMatchEulerDiscreteScheduler(shift=1.0)

# high-resolution: shift toward higher noise so global structure is trained
sched_hires = FlowMatchEulerDiscreteScheduler(shift=3.0)

# SD3/FLUX pipelines also support resolution-aware dynamic shifting:
sched_dyn = FlowMatchEulerDiscreteScheduler(
    use_dynamic_shifting=True,   # compute shift from the image sequence length
    base_shift=0.5, max_shift=1.15,
)
```

The takeaway for practitioners: **if you change resolution, you must re-examine the schedule.** Copying a 512px schedule to a 1024px fine-tune and wondering why global coherence is worse is one of the most common silent failures in the field. The shift parameter is the knob; `$s \approx 3$` at 1024px relative to a 256px base is the order-of-magnitude to start from, then tune.

#### Worked example: picking a shift for a 1536px fine-tune

You are fine-tuning an SD3-style model from a 1024px base (`$s = 3$`) up to 1536px. The pixel count goes from `$1024^2$` to `$1536^2$`, a factor of `$2.25\times$`. The effective-SNR argument says noise averages out roughly with the linear resolution ratio (more precisely with the square root of the pixel ratio for the standard-deviation of averaged noise), so the noise needs to be pushed further. A defensible starting shift is to scale by the resolution ratio: `$s_\text{new} \approx 3 \times (1536/1024) = 4.5$`, or use dynamic shifting keyed to the latent sequence length (which SD3/FLUX do automatically). Then validate: generate a batch of multi-object prompts and check *global composition*, not texture. If objects are misplaced or duplicated, the high-noise steps are still under-trained — push the shift higher. If textures degrade while composition is fine, you have over-shifted and starved the low-noise steps. The symptom tells you which way to move the single knob. This is the kind of decision that does not appear in any config template and that you have to reason about from the SNR picture.

## 9. The continuous-time picture: EDM as the unified view

Everything so far has been phrased in DDPM's discrete `$\bar\alpha_t$` language because that is what most code uses. But the cleanest way to see *all* of these choices as one design space is Karras et al.'s EDM framework, and it is worth a section because the modern frontier (SD3, FLUX, consistency models) lives closer to EDM's formulation than to vanilla DDPM. The payoff: EDM shows that the parameterization, the schedule, the weighting, and the terminal-SNR question are not four separate tricks but four faces of one set of continuous design choices, and it gives a principled default for each.

EDM throws away `$\bar\alpha_t$` and works with a noise level `$\sigma$` directly: a noised sample is `$x = x_0 + \sigma\,\varepsilon$`, with `$\sigma$` ranging from near-zero (clean) to large (noise dominates). The SNR is simply `$\text{SNR} = 1/\sigma^2$` (signal variance over noise variance), so `$\sigma$` *is* the SNR axis, inverted and square-rooted. The discrete DDPM schedule maps onto this via `$\sigma_t = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t} = 1/\sqrt{\text{SNR}(t)}$` — every DDPM schedule is just a particular curve of `$\sigma$` over time, and you can convert between the two views mechanically.

In this language the four choices become:

- **The schedule** is the *distribution you sample `$\sigma$` from during training*. EDM samples `$\ln\sigma \sim \mathcal{N}(P_\text{mean}, P_\text{std}^2)$` with `$P_\text{mean} = -1.2$`, `$P_\text{std} = 1.2$` — a log-normal concentrated around `$\sigma \approx 0.3$` (the mid-noise band where learning is most informative). This is the EDM analogue of "cosine spends less time at the destroyed end" and "min-SNR up-weights the middle": instead of weighting, EDM *samples* more steps in the informative band. Sampling and weighting are interchangeable ways to redistribute the training budget — you can move emphasis by changing where you draw noise levels, or by changing how much each drawn level counts. EDM chooses the sampling route, which has lower gradient variance.
- **The parameterization** is EDM's *preconditioning*. EDM writes the denoiser as `$D_\theta(x, \sigma) = c_\text{skip}(\sigma)\,x + c_\text{out}(\sigma)\,F_\theta(c_\text{in}(\sigma)\,x, c_\text{noise}(\sigma))$`, where the `$c$` coefficients are chosen so the network `$F_\theta$` always sees unit-variance inputs and predicts a unit-variance target *at every `$\sigma$`*. Work through the algebra and EDM's preconditioning is exactly the continuous generalization of `$v$`-prediction: it interpolates between predicting the image (at low noise) and predicting the noise (at high noise) with `$\sigma$`-dependent coefficients, keeping the effective target's magnitude constant. EDM and `$v$`-prediction are the same idea expressed in two notations.
- **The weighting** is EDM's `$\lambda(\sigma)$` loss weight, chosen alongside the preconditioning so the effective loss magnitude is uniform across `$\sigma$`. EDM's `$\lambda(\sigma) = (\sigma^2 + \sigma_\text{data}^2)/(\sigma\,\sigma_\text{data})^2$` is the principled analogue of min-SNR — it equalizes the gradient contribution of each noise level by construction rather than by an empirical clamp.
- **Terminal SNR** is, in EDM, the choice of the maximum `$\sigma_\text{max}$` you train and sample at. EDM uses `$\sigma_\text{max} = 80$`, large enough that `$\text{SNR} = 1/\sigma_\text{max}^2 \approx 1.6\times10^{-4}$` is negligible — effectively zero terminal SNR by construction. The DDPM community had to *discover* and *patch* the leaky terminal SNR (§6) precisely because the `$\bar\alpha_t$` parameterization hid it; EDM's `$\sigma$` parameterization makes "how noisy is the noisiest training input" an explicit, prominent hyperparameter you cannot accidentally leave leaking.

#### Worked example: converting an SD schedule into EDM sigmas

Take SD 1.5's terminal step, `$\bar\alpha_T \approx 0.0047$`. Its EDM-equivalent noise level is `$\sigma_T = \sqrt{(1 - 0.0047)/0.0047} = \sqrt{0.9953/0.0047} \approx \sqrt{211.8} \approx 14.6$`. Compare that to EDM's `$\sigma_\text{max} = 80$`. SD 1.5's "maximum noise" is *five and a half times smaller* than EDM's — concretely, the terminal SNR is `$1/14.6^2 \approx 4.7\times10^{-3}$` versus EDM's `$1/80^2 \approx 1.6\times10^{-4}$`, a 30× gap. That gap *is* the brightness bug from §6, now expressed as a noise level: SD 1.5 simply does not train at a high enough `$\sigma$` to make the terminal input pure noise, and `$\sigma_T \approx 14.6$` still leaves visible signal. The zero-terminal-SNR fix is, in EDM language, "push `$\sigma_T$` up until `$1/\sigma_T^2$` is negligible" — which is exactly what `$\sigma_\text{max} = 80$` does by default. Seeing the same bug from two coordinate systems is the best confirmation that SNR (or its inverse `$\sigma$`) is the real variable and the rest is notation.

The practical upshot: if you are building on the modern stack (SD3, FLUX, EDM-style training, consistency distillation), you are already in a world where the parameterization is well-conditioned, the schedule is a sampling distribution over log-SNR, the weighting is principled, and the terminal noise is an explicit hyperparameter. The DDPM-era tricks in §§2-7 are the patches that retrofit those properties onto the `$\bar\alpha_t$` formulation. Knowing both views lets you read any codebase — when you see `betas` and `prediction_type`, you are in DDPM-land and should check the terminal SNR; when you see `sigma_min`, `sigma_max`, and a preconditioned denoiser, you are in EDM-land and most of these issues are handled by construction.

## 10. Case studies: real numbers from real models

Let me ground all of this in shipped results, because the whole premise of this post is that these choices move measured FID more than people expect. Cite-and-verify mode: where I give a precise number it is from the named paper; where I am summarizing a trend I say so.

**Improved DDPM (Nichol & Dhariwal, 2021) — cosine schedule.** On ImageNet 64×64, switching from the linear to the cosine schedule improved both the negative log-likelihood and the FID in their ablations (FID roughly 19.5 → 17.5 in the relevant table), and the gain was larger on lower-resolution data where the linear schedule's end-of-range waste is most acute. The headline lesson the field absorbed: the schedule is not a fixed constant of nature, and the obvious default was leaving quality on the table.

**Min-SNR weighting (Hang et al., 2023).** On ImageNet 256×256 with a ViT-based diffusion backbone, min-SNR-`$\gamma$` (`$\gamma=5$`) accelerated convergence to a target FID by about **3.4×** versus the uniform `$\mathcal{L}_\text{simple}$` weighting, and reached a better final FID. The change is ~10 lines of code and no extra compute per step. This is the single best effort-to-payoff ratio in the post.

**Zero-terminal-SNR (Lin et al., 2023).** The paper demonstrated that nearly all then-current open models (SD 1.x, SD 2.x in their default configs) could not generate true black or true white, and that the combined fix (rescaled schedule + trailing timesteps + CFG rescale) restored the full dynamic range and improved FID on their evaluation. The qualitative result — dark-scene and bright-scene prompts that previously produced grey mush now rendering correctly — is the one most practitioners can reproduce in five minutes on their own model.

**v-prediction for distillation (Salimans & Ho, 2022).** Progressive distillation halves the sampler step count repeatedly, and the authors found `$\varepsilon$`-prediction *unstable* under distillation because the error amplification at high noise compounds across distillation generations, while `$v$`-prediction's bounded target stayed stable down to very few steps. This is *why* nearly every modern few-step / distilled model (and high-res model) uses `$v$` or the equivalent EDM preconditioning rather than raw `$\varepsilon$`. The [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) post builds directly on this stability.

**SD3 resolution shift (Esser et al., 2024).** SD3 reported that the resolution-dependent timestep shift was necessary for high-resolution sample quality — without it, high-resolution training produced worse global coherence — and that flow-matching with the shift outperformed the alternatives they ablated. It is the production-scale confirmation that schedule design is resolution-dependent.

Here is the consolidated comparison, the kind of table you should keep next to your training config:

| Choice | Default (naive) | Better choice | Where it bites | Reported effect |
|---|---|---|---|---|
| Prediction target | `$\varepsilon$` | `$v$` (or EDM precond.) | high-noise steps, distillation, hi-res | stable distillation; bounded target |
| Schedule | linear `$\beta$` | cosine / sigmoid / EDM | low-res; end-of-range waste | FID ~19.5→17.5 ImageNet 64 |
| Loss weighting | `$\mathcal{L}_\text{simple}$` (flat) | min-SNR-`$\gamma=5$` | convergence speed, all res | ~3.4× faster convergence |
| Terminal SNR | leaks (`$\sqrt{\bar\alpha_T}\approx0.068$`) | rescale to 0 | brightness/contrast range | true black/white restored |
| Resolution | copy base schedule | shift `$s\!\approx\!3$` @1024 | global composition at hi-res | needed for SD3 hi-res quality |

And the diagnostic table — the symptom-to-cause map that turns a vague "my samples look off" into a specific fix:

| Symptom in samples | Likely cause | Fix |
|---|---|---|
| Can't render pure black/white; dull midtones | nonzero terminal SNR (mean leak) | zero-terminal-SNR + trailing + CFG rescale |
| Over-exposed / blown highlights after the SNR fix | CFG std mismatch | `guidance_rescale ≈ 0.7` |
| Training diverges/oscillates at high noise | `$\varepsilon$`-pred error blow-up | switch to `$v$`-prediction |
| Slow convergence, uneven per-timestep loss | flat weighting over-weights easy steps | min-SNR-`$\gamma=5$` |
| Great textures, wrong global layout at hi-res | schedule not shifted for resolution | increase `shift` (`$s$`) |
| Distilled few-step model unstable | `$\varepsilon$`-pred amplification compounds | `$v$`-prediction for distillation |

## 11. When to reach for each (and when not to)

A technique is only useful if you know when to skip it. Decisive recommendations:

- **Use `$v$`-prediction** when you are doing distillation, training at high resolution, or training a few-step model — anywhere the high-noise steps and large sampler steps make `$\varepsilon$`'s error amplification dangerous. **Skip it** if you are reproducing an existing `$\varepsilon$`-prediction model exactly (SD 1.5 LoRAs, for instance) — switching parameterization means retraining the base, and a LoRA on top of an `$\varepsilon$` model must stay `$\varepsilon$`. The parameterization is a property of the base weights, not a flag you flip at inference.
- **Use a cosine (or sigmoid/EDM) schedule** at low-to-moderate resolution where the linear schedule's end-of-range waste hurts most. At high resolution the resolution *shift* matters more than linear-vs-cosine; get the shift right first.
- **Use min-SNR-`$\gamma=5$`** essentially always when training from scratch or doing a substantial fine-tune — it is nearly free and reliably helps convergence. **Skip it** for a tiny LoRA fine-tune on a handful of images, where the dynamics are dominated by overfitting, not by cross-timestep loss balance, and the weighting change is noise.
- **Apply the zero-terminal-SNR fix** if you control training and care about dynamic range (product photography, high-contrast art, anything with true blacks/whites). **Skip it** — or rather, know it is irrelevant — if you are only doing inference on a model trained without it; the flags alone will not help and `timestep_spacing="trailing"` on a leaky model can even hurt. It is a train-time fix.
- **Re-examine the schedule shift** whenever you change resolution. **Skip the agonizing** if you are at a single fixed resolution that matches your base model — just inherit the base's schedule.
- **Do not over-tune.** These knobs interact. Change one, measure FID and eyeball dynamic range and global coherence, then change the next. Changing all five at once and reporting a single FID number tells you nothing about which one helped, and you will not be able to debug a regression.

The meta-rule: every one of these is a *training-time* decision routed through SNR. If you only do inference, your levers are the sampler and guidance (covered in [the samplers deep-dive](/blog/machine-learning/image-generation/samplers-deep-dive) and [classifier-free guidance](/blog/machine-learning/image-generation/classifier-free-guidance)); the schedule and parameterization are baked into the weights you downloaded.

## 12. Key takeaways

- **Read every schedule through `$\text{SNR}(t) = \bar\alpha_t/(1-\bar\alpha_t)$`**, ideally as log-SNR over time. The `$\beta_t$` array is a distraction; the SNR curve is the physics the model lives on.
- **The three prediction targets are algebraically equivalent but not numerically equivalent.** `$\varepsilon$` blows up recovering `$x_0$` at high noise; `$x_0$` blows up recovering `$\varepsilon$` at low noise; `$v = \sqrt{\bar\alpha_t}\varepsilon - \sqrt{1-\bar\alpha_t}x_0$` is the bounded tangent that is well-conditioned everywhere — use it for distillation and high-res.
- **`$\mathcal{L}_\text{simple}$` is the variational bound reweighted.** In `$x_0$`-space it is an SNR-weighting of reconstruction error, which over-weights easy steps. Every weighting trick (min-SNR, P2, EDM) is a different `$w(\text{SNR})$` fixing that imbalance.
- **min-SNR-`$\gamma=5$` is the highest-leverage cheap win** — ~3.4× faster convergence and better FID for ~10 lines of code, by capping the easy steps' dominance.
- **Zero-terminal-SNR is a real, shipped bug.** Nonzero `$\sqrt{\bar\alpha_T}$` leaks dataset-mean brightness into the start of sampling, so models can't render true black/white. Fix it with rescaled betas + trailing timesteps + CFG rescale — and it is a *train-time* fix.
- **Schedules are resolution-dependent.** More pixels average out noise, raising effective SNR, so high-resolution training needs the schedule shifted toward higher noise (SD3's `$s\approx3$` at 1024px). Copying a low-res schedule to high-res silently breaks global composition.
- **These quiet choices move FID more than the architecture changes people publish.** Tune them deliberately, one at a time, measuring both FID and dynamic range and global coherence.

## 13. Further reading

- **Ho, Jain, Abbeel — "Denoising Diffusion Probabilistic Models" (2020).** The `$\varepsilon$`-prediction objective and `$\mathcal{L}_\text{simple}$`; derives the variational bound this post reweights.
- **Nichol & Dhariwal — "Improved Denoising Diffusion Probabilistic Models" (2021).** The cosine schedule and learned variances; the end-of-range-waste argument.
- **Salimans & Ho — "Progressive Distillation for Fast Sampling of Diffusion Models" (2022).** Introduces `$v$`-prediction and shows its stability under distillation.
- **Kingma, Salimans, Poole, Ho — "Variational Diffusion Models" (2021).** The continuous-time view where the loss depends on the schedule only through the log-SNR endpoints.
- **Karras, Aittala, Aila, Laine — "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM, 2022).** The `$\sigma$`-parameterization, preconditioning, and the noise-level sampling distribution.
- **Hang et al. — "Efficient Diffusion Training via Min-SNR Weighting Strategy" (2023).** Diffusion-as-multi-task-learning and the min-SNR-`$\gamma$` clamp.
- **Choi et al. — "Perception Prioritized Training of Diffusion Models" (P2, 2022).** The perceptual argument for down-weighting high-SNR steps.
- **Lin et al. — "Common Diffusion Noise Schedules and Sample Steps are Flawed" (2023).** The zero-terminal-SNR fix, trailing timesteps, and CFG rescale.
- **Esser et al. — "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3, 2024).** The resolution-dependent timestep shift in a flow-matching schedule.
- Within this series: [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) and [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm) set up the forward process and loss; [the samplers deep-dive](/blog/machine-learning/image-generation/samplers-deep-dive) and [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) consume the schedule; [consistency models and few-step generation](/blog/machine-learning/image-generation/consistency-models-and-few-step-generation) builds on `$v$`-prediction's stability; and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack) puts every knob in one pipeline.
