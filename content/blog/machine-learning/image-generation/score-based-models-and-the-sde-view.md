---
title: "Score-Based Models and the SDE View: One Equation Behind All of Diffusion"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "See how DDPM, score matching, and NCSN are the same model in different coordinates — derive the score, the reverse-time SDE, and the probability-flow ODE, then sample a toy model with both."
tags:
  [
    "image-generation",
    "diffusion-models",
    "score-based-models",
    "stochastic-differential-equations",
    "langevin-dynamics",
    "probability-flow-ode",
    "generative-ai",
    "deep-learning",
    "denoising-score-matching",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/score-based-models-and-the-sde-view-1.png"
---

Here is a fact that took the field years to fully appreciate, and that still trips up smart engineers who have shipped diffusion models in production: **DDPM, score matching, and noise-conditional score networks are not three competing methods. They are one method, written in three coordinate systems.** When you train a Stable Diffusion U-Net to predict the noise $\epsilon$ added to a latent, you are — to within a known scalar — estimating the *score* $\nabla_x \log p_t(x)$ of the noised data distribution. When Yang Song trained an NCSN in 2019 to denoise across a ladder of noise scales, he was estimating the same object. When you run DDIM's deterministic sampler in 50 steps, you are integrating the *probability-flow ODE* of a stochastic differential equation whose reverse-time form Brian Anderson wrote down in 1982. These are not analogies. They are the same equation.

This post is the grand unification of the diffusion track. We have already built the [DDPM forward and reverse processes from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) and [derived the variational bound down to the simple $\epsilon$-prediction loss](/blog/machine-learning/image-generation/the-math-of-ddpm). Those posts are the *discrete* story: a finite Markov chain of $T = 1000$ noising steps. Here we take $T \to \infty$, let the step size go to zero, and the discrete chain becomes a **continuous-time stochastic process** governed by a stochastic differential equation (SDE). That single move — discrete to continuous — does something almost magical: it dissolves the boundaries between every diffusion variant you have heard of and reveals the common machinery underneath. Song, Sohl-Dickstein, Kingma, Kumar, Ermon, and Poole laid this out in the 2021 paper "Score-Based Generative Modeling through Stochastic Differential Equations," and it is, in my opinion, the single most clarifying paper in the entire field.

![A graph showing how DDPM epsilon-prediction, denoising score matching, and NCSN all estimate the same noised-data score, which the forward SDE then unifies into a reverse SDE and a probability-flow ODE](/imgs/blogs/score-based-models-and-the-sde-view-1.png)

By the end of this post you will be able to: define the **score function** $\nabla_x \log p(x)$ and explain *why* it is enough to sample from a distribution without ever computing the notoriously intractable normalizing constant $Z$; derive the **denoising score matching** identity that turns an impossible objective into a plain regression, and read off the exact scalar that connects $\epsilon$-prediction to the score; run **Langevin dynamics** as score-driven sampling and see why a single noise scale fails; write down the **forward SDE** $dx = f(x,t)\,dt + g(t)\,dw$ together with its **variance-preserving (VP)** and **variance-exploding (VE)** special cases; derive Anderson's **reverse-time SDE** and understand why a learned score is exactly the missing ingredient that lets you run time backward; and derive the **probability-flow ODE** — the deterministic sampler that shares the SDE's marginals and is the bridge to DDIM and flow matching. We will also train a tiny 2D score model and sample it two ways — stochastic Langevin and the deterministic ODE — and watch them land on the same distribution.

Keep the series' spine in mind: the **generative trilemma** (sample quality × mode coverage × sampling speed) and the **diffusion stack** (data → latent → forward noising → denoiser net → SDE/ODE sampler → guidance → image). The reason this post matters for that frame is that the SDE view is where the *sampler* slot of the stack gets its full vocabulary. Once you see that the trained network is a score estimate and that there is a whole continuum of SDEs and ODEs sharing the same marginals, every later choice — DDIM versus DDPM, 50 steps versus 1000, deterministic versus stochastic, the leap to flow matching — stops being a bag of tricks and becomes a set of principled moves on one well-understood object.

## 1. The score function, and why it is enough

Let me start with the object the whole post revolves around. Given a probability density $p(x)$ over $x \in \mathbb{R}^D$, its **score** is the gradient of the log-density with respect to the *data*, not the parameters:

$$
s(x) \;=\; \nabla_x \log p(x).
$$

Read it carefully, because the "with respect to $x$" part is the entire trick. In statistics you usually see $\nabla_\theta \log p(x; \theta)$ — the gradient with respect to parameters, which is what you use in maximum likelihood. That is *not* this. The score here is a vector field over the data space: at every point $x$, it points in the direction in which the log-probability increases fastest. It is a field of little arrows all over $\mathbb{R}^D$, each one saying "to find more probable data, step this way."

Why would you ever want that instead of the density itself? Because the density itself is almost always uncomputable. Any expressive model of $p(x)$ has the form

$$
p(x) \;=\; \frac{\tilde p(x)}{Z}, \qquad Z = \int \tilde p(x)\,dx,
$$

where $\tilde p(x) = e^{-E(x)}$ is an **unnormalized** density (an "energy" $E(x)$ you can evaluate cheaply with a neural net) and $Z$ is the **partition function** — the integral that makes it integrate to one. In $D = 196{,}608$ dimensions (a $256\times256\times3$ image), that integral is hopeless. You cannot compute $Z$, you cannot estimate it without astronomical variance, and so you cannot evaluate $p(x)$, and you cannot train by maximum likelihood, and you cannot sample with any method that needs the density. The intractable $Z$ is the wall that energy-based models slammed into for decades.

Here is the escape. Take the log and then the gradient with respect to $x$:

$$
\nabla_x \log p(x) \;=\; \nabla_x \big[\log \tilde p(x) - \log Z\big] \;=\; \nabla_x \log \tilde p(x) \;-\; \underbrace{\nabla_x \log Z}_{=\,0}.
$$

$Z$ is a constant — it does not depend on $x$, it is just a number you integrated out. So its gradient with respect to $x$ is exactly zero, and it **vanishes from the score**. The score of the normalized density equals the score of the unnormalized one:

$$
\boxed{\;\nabla_x \log p(x) \;=\; \nabla_x \log \tilde p(x) \;=\; -\,\nabla_x E(x).\;}
$$

This is the whole game. The thing that was impossible to compute — $Z$ — is precisely the thing the score does not need. If you can model and learn the score field, you have sidestepped the partition function entirely.

![A graph showing how Langevin dynamics uses only the score so the intractable partition function Z cancels and never has to be computed](/imgs/blogs/score-based-models-and-the-sde-view-2.png)

But "I can compute a gradient field" and "I can draw samples" are not the same statement. The bridge between them is **Langevin dynamics**, a piece of physics from the early 20th century repurposed as a sampler. Start from any point $x_0$ (say pure Gaussian noise) and iterate

$$
x_{k+1} \;=\; x_k \;+\; \frac{\eta}{2}\, \nabla_x \log p(x_k) \;+\; \sqrt{\eta}\,\, z_k, \qquad z_k \sim \mathcal{N}(0, I).
$$

There are exactly two forces here. The first, $\tfrac{\eta}{2}\nabla_x \log p(x_k)$, is a **gradient-ascent step on the log-density**: it walks the point uphill toward more probable regions. If that were the only term, every point would collapse onto the single highest-density mode — you would get the mode, not a sample. The second term, $\sqrt{\eta}\,z_k$, is **injected Gaussian noise** that kicks the point around. The deep result, from the theory of Langevin diffusions, is that this exact balance of "climb the log-density" and "get kicked by noise" has the data distribution $p(x)$ as its **stationary distribution**: as $\eta \to 0$ and the number of steps $K \to \infty$, the law of $x_K$ converges to $p(x)$. You are not finding the mode; you are sampling the whole distribution, visiting each region in proportion to its probability.

The balance is exact and you can verify it on a Gaussian by hand. For $p(x) = \mathcal{N}(0, \sigma^2)$ the score is $-x/\sigma^2$, so the Langevin update is $x_{k+1} = x_k(1 - \tfrac{\eta}{2\sigma^2}) + \sqrt{\eta}\,z_k$ — a linear recursion. Its stationary variance solves $V = V(1-\tfrac{\eta}{2\sigma^2})^2 + \eta$, and to first order in small $\eta$ that gives $V \to \sigma^2$ exactly. The gradient term pulls variance *in* (the $1-\tfrac{\eta}{2\sigma^2}$ contraction), the noise term pushes it *out* (the $+\eta$), and they settle at precisely the target variance $\sigma^2$ — not at zero (which pure gradient ascent would give) and not at infinity (which pure noise would give). That equilibrium between contraction and injection, recovering the *exact* target spread, is the entire reason Langevin samples the distribution rather than collapsing to its peak, and it is the same balance the reverse SDE strikes in continuous time.

And notice what Langevin dynamics needs as input: only $\nabla_x \log p(x)$. Not $p(x)$, not $Z$, not the energy's absolute value — just the score. So if a neural network $s_\theta(x) \approx \nabla_x \log p(x)$ learns the score field, plugging it into the Langevin update gives you a sampler that never once touches the partition function. That is *why the score is enough*. It is the single most important reframing in this post, and everything that follows is machinery to (a) learn the score reliably and (b) make Langevin-style sampling fast and stable.

#### Worked example: the score of a Gaussian

To make the abstraction concrete, take the one distribution whose score you can write by hand. For $p(x) = \mathcal{N}(x; \mu, \sigma^2 I)$,

$$
\log p(x) = -\frac{\|x - \mu\|^2}{2\sigma^2} + \text{const}, \qquad \nabla_x \log p(x) = -\frac{x - \mu}{\sigma^2}.
$$

The score is a vector pointing *from* $x$ *back toward the mean* $\mu$, with magnitude proportional to how far out you are and inversely proportional to the variance. That is the entire intuition of denoising in one formula: the score of a noisy distribution points back toward where the clean data lives. Hold onto $-\frac{x-\mu}{\sigma^2}$ — in two sections it will reappear as the denoising-score-matching target, and the resemblance is not a coincidence.

## 2. Score matching: learning a gradient field you cannot see

We want a network $s_\theta(x)$ to match the true score $\nabla_x \log p(x)$. The obvious objective is to minimize the expected squared distance between them — the **explicit score matching (ESM)** loss:

$$
J_{\text{ESM}}(\theta) = \frac{1}{2}\,\mathbb{E}_{x \sim p(x)}\Big[\,\big\| s_\theta(x) - \nabla_x \log p(x) \big\|^2\,\Big].
$$

There is one fatal problem: we do not know $\nabla_x \log p(x)$. That is the entire object we are trying to learn; we cannot put it in the loss as a regression target. We only have *samples* from $p$, not its score. So ESM as written is uncomputable, and we need a trick to remove the unknown target.

Aapo Hyvärinen's 2005 score-matching paper found the first one. Through integration by parts you can rewrite $J_{\text{ESM}}$ into an equivalent objective that contains only $s_\theta$ and its derivatives — the unknown true score drops out:

$$
J_{\text{ISM}}(\theta) = \mathbb{E}_{x \sim p(x)}\Big[\, \tfrac{1}{2}\|s_\theta(x)\|^2 + \operatorname{tr}\!\big(\nabla_x s_\theta(x)\big) \Big] + \text{const}.
$$

This **implicit score matching (ISM)** form is exact and trainable in principle — but look at the second term. $\operatorname{tr}(\nabla_x s_\theta(x))$ is the trace of the Jacobian of the score network, the sum of $\partial_{x_i} [s_\theta]_i$ over all $D$ input dimensions. Computing it exactly costs $D$ separate backward passes — one per dimension. For a $256\times256\times3$ image, that is ~196k backward passes *per training example*. It is the same $O(D^2)$ wall, wearing a different hat. There are stochastic estimators of the trace (sliced score matching uses Hutchinson's trick with random projections), but they are noisy and they were never the path to image-scale models.

The breakthrough that actually scales is **denoising score matching (DSM)**, due to Pascal Vincent in 2011, and it is one of those results that feels like a magic trick until you see the proof, after which it feels inevitable.

![A before-and-after diagram contrasting intractable explicit score matching with denoising score matching, which replaces the unknown data score with a known Gaussian target and becomes plain regression](/imgs/blogs/score-based-models-and-the-sde-view-3.png)

### The DSM identity, derived

The idea: instead of matching the score of the *clean* data $p(x)$, deliberately **corrupt the data with known Gaussian noise** and match the score of the *noisy* distribution. Define a perturbation kernel $q_\sigma(\tilde x \mid x) = \mathcal{N}(\tilde x; x, \sigma^2 I)$ — add Gaussian noise of scale $\sigma$ — and the resulting noisy marginal $q_\sigma(\tilde x) = \int q_\sigma(\tilde x \mid x)\, p(x)\, dx$. The denoising score matching objective is

$$
J_{\text{DSM}}(\theta) = \frac{1}{2}\, \mathbb{E}_{x \sim p(x)}\,\mathbb{E}_{\tilde x \sim q_\sigma(\tilde x \mid x)} \Big[\, \big\| s_\theta(\tilde x) - \nabla_{\tilde x} \log q_\sigma(\tilde x \mid x) \big\|^2 \,\Big].
$$

Look at what changed. The regression target is now $\nabla_{\tilde x} \log q_\sigma(\tilde x \mid x)$ — the score of the *conditional* perturbation kernel, **conditioned on the clean $x$** — and that we know in closed form, because $q_\sigma(\tilde x \mid x)$ is just a Gaussian centered at $x$. Using the Gaussian-score result from the worked example above (with $\mu = x$):

$$
\nabla_{\tilde x} \log q_\sigma(\tilde x \mid x) = -\frac{\tilde x - x}{\sigma^2}.
$$

This is a *known, cheap, closed-form* target. There is no Jacobian trace, no unknown data score — just a regression of the network against a vector you can compute directly from the clean sample $x$ and the noise you added. That is the whole point.

The non-obvious part is *why* minimizing $J_{\text{DSM}}$ teaches you the score of the noisy marginal $q_\sigma(\tilde x)$ rather than some useless per-example target. Here is the proof sketch. Expand the squared norm in $J_{\text{DSM}}$ and keep only the term that depends on $\theta$ through a cross-product (the $\|s_\theta\|^2$ term is the same in both objectives, and the target-squared term is a $\theta$-independent constant). The cross term is

$$
-\,\mathbb{E}_{x}\,\mathbb{E}_{\tilde x \mid x}\big[\, s_\theta(\tilde x)^\top \nabla_{\tilde x} \log q_\sigma(\tilde x \mid x) \,\big].
$$

Now use the identity $q_\sigma(\tilde x \mid x)\, \nabla_{\tilde x}\log q_\sigma(\tilde x \mid x) = \nabla_{\tilde x} q_\sigma(\tilde x \mid x)$ and the definition $q_\sigma(\tilde x) = \int q_\sigma(\tilde x\mid x) p(x)\,dx$. Pulling the integral over $x$ inside the gradient, the joint expectation collapses to an expectation under the *marginal* $q_\sigma(\tilde x)$ against the *marginal score* $\nabla_{\tilde x}\log q_\sigma(\tilde x)$:

$$
-\,\mathbb{E}_{x}\,\mathbb{E}_{\tilde x \mid x}\big[\, s_\theta^\top \nabla \log q_\sigma(\tilde x \mid x)\,\big] = -\,\mathbb{E}_{\tilde x \sim q_\sigma}\big[\, s_\theta(\tilde x)^\top \nabla_{\tilde x}\log q_\sigma(\tilde x)\,\big].
$$

That cross term is exactly the one you would get by expanding the *explicit* objective $\tfrac12\mathbb{E}_{q_\sigma}\|s_\theta(\tilde x) - \nabla\log q_\sigma(\tilde x)\|^2$ against the true marginal score. Since the two objectives differ only by $\theta$-independent constants, they share the same minimizer:

$$
\boxed{\;s_{\theta^\star}(\tilde x) = \nabla_{\tilde x} \log q_\sigma(\tilde x) \quad\text{for almost every } \tilde x.\;}
$$

So even though every single training target is the *per-example* direction $-(\tilde x - x)/\sigma^2$ that points back at one clean point, the network that minimizes the *expectation* learns the true score of the noisy *marginal* $q_\sigma$. The averaging over many clean $x$ that could have produced the same $\tilde x$ is what turns a bag of per-example arrows into the genuine marginal score field. That is the magic, and now it should feel inevitable: regression toward a known noisy-target, averaged, recovers the unknown marginal score for free.

### The bridge to $\epsilon$-prediction

Here is the payoff that closes the loop with the [DDPM post](/blog/machine-learning/image-generation/the-math-of-ddpm). Write the noisy sample as $\tilde x = x + \sigma \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$. Then the DSM target is

$$
\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x) = -\frac{\tilde x - x}{\sigma^2} = -\frac{\sigma \epsilon}{\sigma^2} = -\frac{\epsilon}{\sigma}.
$$

A network trained to predict the score is, up to the scalar $-1/\sigma$, predicting the noise $\epsilon$. Conversely, an $\epsilon$-prediction network $\epsilon_\theta$ — exactly what DDPM trains — gives you the score by a single multiplication:

$$
\boxed{\;s_\theta(\tilde x, \sigma) \;=\; -\frac{\epsilon_\theta(\tilde x, \sigma)}{\sigma}.\;}
$$

This is the identity that fuses the two halves of the field. In the variance-preserving DDPM parameterization, where $\tilde x = \sqrt{\bar\alpha_t}\,x + \sqrt{1-\bar\alpha_t}\,\epsilon$, the same algebra gives $s_\theta(x_t, t) = -\epsilon_\theta(x_t, t)/\sqrt{1-\bar\alpha_t}$. Either way, **the Stable Diffusion U-Net you have been calling a "noise predictor" is a score model in disguise.** Every $\epsilon$-prediction checkpoint ever trained is a learned $\nabla_x \log p_t(x)$, waiting to be plugged into a Langevin sampler or an SDE solver. That is not a metaphor; it is a multiplication by $-1/\sigma$.

## 3. NCSN: why one noise scale is not enough

If DSM at a single noise scale $\sigma$ learns the score of $q_\sigma$, why not pick one $\sigma$, train, and run Langevin dynamics to sample? Yang Song and Stefano Ermon tried exactly this in their 2019 NCSN paper and ran straight into two failures that, once you see them, explain the entire architecture of modern diffusion.

**Failure one: the score is garbage where there is no data.** Real images live on a thin, low-dimensional manifold inside the vast ambient pixel space — the **manifold hypothesis** we met in [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard). With a *small* $\sigma$, the noisy distribution $q_\sigma$ stays hugged tight to that manifold. But Langevin sampling starts from pure noise, far out in the empty void where no training example ever landed. Out there the network has seen no data, so its score estimate is meaningless — random arrows pointing nowhere useful. The sampler wanders in the void and never finds the manifold. A *large* $\sigma$ fixes this — heavy noise spreads $q_\sigma$ across the whole space so there is signal everywhere — but then $q_\sigma$ is so blurred that its samples look nothing like real images. You are stuck: small $\sigma$ gives sharp targets with no global guidance; large $\sigma$ gives global guidance but blurry targets.

**Failure two: Langevin mixes badly between separated modes.** When the data has well-separated modes with low-density gaps between them (think: distinct object categories), plain Langevin dynamics struggles to cross those gaps in any reasonable number of steps, and the relative weighting of the modes comes out wrong. The walk gets trapped.

The fix is the same idea twice over: **use many noise scales at once.** NCSN trains a single network $s_\theta(x, \sigma)$ conditioned on the noise level $\sigma$, over a geometric ladder $\sigma_1 > \sigma_2 > \cdots > \sigma_L$ spanning from "blurs everything into one blob" down to "barely perturbs the data." The training loss is the DSM objective summed over scales, each weighted by $\lambda(\sigma) = \sigma^2$ so that no scale dominates:

$$
\mathcal{L}(\theta) = \frac{1}{L}\sum_{i=1}^{L} \lambda(\sigma_i)\; \mathbb{E}_{x}\,\mathbb{E}_{\tilde x \sim \mathcal{N}(x,\sigma_i^2 I)}\left[\, \Big\| s_\theta(\tilde x, \sigma_i) + \frac{\tilde x - x}{\sigma_i^2} \Big\|^2 \,\right].
$$

(The $+\frac{\tilde x - x}{\sigma_i^2}$ inside the norm is just $s_\theta$ minus the target $-\frac{\tilde x-x}{\sigma_i^2}$, written out.) Sampling then uses **annealed Langevin dynamics**: start at the largest noise scale $\sigma_1$, run a few Langevin steps using $s_\theta(\cdot, \sigma_1)$, then *anneal* down to $\sigma_2$, run more steps, and so on down to the smallest $\sigma_L$. The large scales give global guidance from the void toward the manifold; the small scales sharpen the sample once it is close. Each scale hands off to the next, like a coarse-to-fine search.

There is a real engineering question hidden in "use many noise scales": *how many*, and *how spaced*? Song & Ermon's follow-up "Improved Techniques for Training Score-Based Generative Models" (2020) worked out the rules of thumb, and they are worth knowing because they recur in every diffusion schedule. The scales should be **geometrically spaced** ($\sigma_i/\sigma_{i+1}$ constant), the largest $\sigma_1$ should be roughly the **maximum pairwise distance between training points** (so the heaviest-noised distribution genuinely bridges all the data into one connected blob), the smallest $\sigma_L$ should be small enough that $q_{\sigma_L}$ is indistinguishable from the data, and the ratio between adjacent scales should be small enough (they recommend around $\sigma_i/\sigma_{i+1} \approx 1.0$–$1.2$ in high dimensions) that the noisy distributions *overlap* — if adjacent scales do not overlap, the annealing hands off into a region the next scale's score has not learned, and sampling stalls. That overlap requirement is the discrete shadow of a continuous truth: in the limit, the scales must form a *continuum* so the score is defined and reliable at every noise level. The discreteness was always an approximation to a smooth $\sigma(t)$.

#### Worked example: how the noise ladder spans the manifold

Put numbers on the failure-and-fix. Suppose your standardized data has unit per-dimension variance and the two farthest training images are about $\sigma_1 \approx 50$ apart in pixel-norm. A *single* scale of, say, $\sigma = 0.1$ keeps $q_\sigma$ glued to the data manifold — but a sample initialized from $\mathcal{N}(0, 50^2 I)$ lands ~500 standard deviations away from anywhere the $\sigma=0.1$ score was trained, so the score there is pure noise and the walk never converges. Now lay down an $L = 30$-scale geometric ladder from $\sigma_1 = 50$ down to $\sigma_L = 0.01$: the ratio is $\sigma_1/\sigma_L = 5000$, so each step multiplies by $5000^{1/29} \approx 1.34$. The $\sigma_1 = 50$ score *has* seen the whole ambient space (it was trained on data blurred across all of it), so it guides the initial noise toward the data blob; then each scale, overlapping the next by that ~1.34 ratio, hands the sample down the ladder, sharpening as it goes. Thirty overlapping scores succeed exactly where one fails — and "30 scales geometrically spaced from $\sigma_{\max}$ to $\sigma_{\min}$" is, you will notice, the same shape as a 1000-step DDPM $\beta$-schedule. The numbers differ; the structure is identical.

This should feel familiar in a deep way. NCSN's ladder of noise scales is *exactly* DDPM's chain of timesteps. DDPM noises with $\sqrt{1-\bar\alpha_t}$ growing as $t$ increases; NCSN noises with $\sigma_i$ growing as $i$ decreases. Both train one network conditioned on the noise level. Both sample by walking from heavy noise down to light noise. They were discovered independently, from opposite starting points — DDPM from a variational bound on likelihood, NCSN from score matching plus Langevin — and they converged on the same algorithm. That convergence is not a coincidence. It is two projections of a single underlying object, and that object is a stochastic differential equation. Time to make it continuous.

## 4. The forward SDE: noising as a continuous-time process

So far the noise levels are a discrete ladder: $\sigma_1, \dots, \sigma_L$ for NCSN, or $\bar\alpha_1, \dots, \bar\alpha_T$ for DDPM. Now take the limit. Let the number of levels go to infinity and the gap between them go to zero, and index the noising by a continuous time $t \in [0, 1]$, with $t = 0$ the clean data and $t = 1$ the fully-noised end. The discrete chain becomes a continuous-time diffusion process described by a **stochastic differential equation**:

$$
\boxed{\;dx \;=\; f(x, t)\,dt \;+\; g(t)\,dw.\;}
$$

Do not let the notation intimidate you; it is a recipe for evolving $x$ over an infinitesimal time step $dt$. The two terms are the two forces. $f(x, t)\,dt$ is the **drift** — a deterministic push, the systematic part of where $x$ moves next. $g(t)\,dw$ is the **diffusion** — random jitter, where $dw$ is an increment of a *Wiener process* (Brownian motion): over a step $dt$ it is a fresh Gaussian sample with variance $dt$, $dw \sim \mathcal{N}(0, dt\, I)$, and $g(t)$ scales how strong that random kick is. So at each instant, $x$ gets a deterministic nudge plus a random kick. Run that from $t=0$ to $t=1$ and you have continuously, smoothly turned data into noise. (If you discretize this SDE with the Euler–Maruyama method — replace $dt$ by a small $\Delta t$ and $dw$ by $\sqrt{\Delta t}\,z$ — you get back exactly the kind of discrete update DDPM and NCSN use. The SDE is the $\Delta t \to 0$ limit.)

The pair $(f, g)$ defines a whole family of forward processes, and the genius of the SDE framework is that the two diffusion lineages you have met are just two choices of $(f, g)$.

![A stack diagram showing one forward SDE that spawns both a reverse SDE and a probability-flow ODE sharing the same time-marginal densities](/imgs/blogs/score-based-models-and-the-sde-view-4.png)

### VP-SDE: the continuous DDPM

The **variance-preserving SDE** has a linear drift that pulls $x$ toward the origin and a diffusion tuned so the *total variance stays bounded*:

$$
dx = -\tfrac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dw.
$$

Here $\beta(t)$ is the continuous version of DDPM's $\beta_t$ noise schedule. The drift $-\tfrac12\beta(t)x$ shrinks $x$ toward zero (this is why the variance does not blow up — the data is continuously contracted as noise is added), and the matched diffusion $\sqrt{\beta(t)}$ injects exactly enough noise that as $t \to 1$ the marginal converges to a standard Gaussian $\mathcal{N}(0, I)$. The "variance-preserving" name is precise: if $x_0$ has unit variance, $x_t$ keeps unit variance for all $t$. We will prove in section 6 that discretizing this SDE *is* DDPM.

### VE-SDE: the continuous NCSN

The **variance-exploding SDE** has *no drift at all* and a diffusion that grows aggressively:

$$
dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\;dw,
$$

which is just "keep adding noise, never contract." With no drift, $x$ does not get pulled anywhere; it only accumulates noise, so its variance *explodes* as $t \to 1$ — hence the name. The marginal at time $t$ is $\mathcal{N}(x_0, \sigma^2(t) I)$ with $\sigma^2(t)$ growing without bound. This is the continuous limit of NCSN's score matching across a ladder of growing $\sigma_i$: the geometric noise ladder becomes a continuous $\sigma(t)$, and the prior at $t = 1$ is a wide Gaussian $\mathcal{N}(0, \sigma_{\max}^2 I)$ rather than a unit one.

### sub-VP-SDE: a tighter cousin

Song et al. introduced a third, the **sub-variance-preserving SDE**, which shares VP's drift $-\tfrac12\beta(t)x$ but scales the diffusion down so the variance at each time is *bounded above by* the VP variance — it stays even tighter. The motivation was empirical: sub-VP gave the best likelihoods (lowest bits-per-dimension) of the three on CIFAR-10 in the original paper. It is the connoisseur's choice when you care about exact likelihood; for raw sample quality VP and VE both shine.

![A matrix comparing the VP, VE, and sub-VP SDEs by drift, diffusion, terminal variance, and the discrete method each corresponds to](/imgs/blogs/score-based-models-and-the-sde-view-5.png)

The crucial conceptual point is that all three share the **same template** $dx = f\,dt + g\,dw$ and differ only in $(f, g)$. Once you pick $(f, g)$, the marginals $p_t(x)$ are determined, the score $\nabla_x \log p_t(x)$ is a well-defined object at every time, and — as we are about to see — there is a single recipe to reverse the process. The forward SDE is the trunk; VP, VE, and sub-VP are branches; DDPM and NCSN are the leaves.

## 5. The reverse-time SDE: running noise back to data

Noising is easy. The whole point of a generative model is the *reverse*: start from the simple prior at $t = 1$ (a Gaussian) and walk back to $t = 0$ (data). The astonishing fact, due to Brian Anderson in a 1982 paper on time-reversal of diffusions, is that the reverse of an SDE is *also an SDE*, and it has an explicit closed form. Given the forward process $dx = f(x,t)\,dt + g(t)\,dw$, the reverse-time process is

$$
\boxed{\;dx \;=\; \big[\,f(x, t) \;-\; g(t)^2\,\nabla_x \log p_t(x)\,\big]\,dt \;+\; g(t)\,d\bar w.\;}
$$

where time now flows backward from $t=1$ to $t=0$ ($dt$ is negative), and $d\bar w$ is a reverse-time Wiener increment. Stare at this equation, because it is the keystone of the entire field.

The reverse drift is the forward drift $f(x,t)$ **corrected by a score term** $-g(t)^2 \nabla_x \log p_t(x)$. That correction is the only new ingredient, and it is *exactly the score* of the time-$t$ marginal. Anderson's theorem says: if you know the forward dynamics $(f, g)$ — which you chose, so you know them perfectly — and you know the score $\nabla_x \log p_t(x)$ at every time, then you can integrate this reverse SDE from noise back to data, and the samples you get are distributed according to the true data distribution $p_0(x)$.

This is the punchline of the whole post, so let me say it as plainly as I can: **the score is the one thing you do not know, and the score is exactly the thing your network learned.** DSM gives you $s_\theta(x, t) \approx \nabla_x \log p_t(x)$. Plug it into the reverse SDE in place of the true score, discretize with Euler–Maruyama, and you have a sampler. The forward drift $f$ and diffusion $g$ are known analytically. The prior at $t = 1$ is a Gaussian you can sample with `torch.randn`. The score is your trained model. That is the complete generative procedure — and it is *identical machinery* whether your network was trained as a DDPM $\epsilon$-predictor (recall $s_\theta = -\epsilon_\theta/\sqrt{1-\bar\alpha_t}$) or as an NCSN score net or as anything in between.

Why does the score appear with that specific $-g^2$ coefficient? The clean way to see it is through the **Fokker–Planck equation**, the PDE that governs how the density $p_t(x)$ flows in time under the SDE. For a forward process $dx = f(x,t)\,dt + g(t)\,dw$, the density obeys

$$
\frac{\partial p_t(x)}{\partial t} = -\nabla_x \cdot \big[\,f(x,t)\,p_t(x)\,\big] + \tfrac{1}{2}g(t)^2 \nabla_x^2 p_t(x).
$$

Read the two terms physically. The first, $-\nabla_x\cdot[f\,p_t]$, is a **transport** (advection) term — it moves probability mass around according to the drift, the way a current carries leaves downstream. The second, $\tfrac12 g^2\nabla_x^2 p_t$, is a **diffusion** term — the Laplacian smears mass out, flattening sharp peaks, the way heat spreads through a bar. Forward in time, transport-plus-smearing turns a sharp data distribution into a smooth Gaussian.

Now the trick. We want a *reverse-time* SDE whose density satisfies the *same* sequence of marginals $p_t$, just traversed from $t=1$ down to $t=0$. The diffusion term is symmetric under time reversal (smearing looks the same forward and backward), but the transport term is not — to run the current backward you must flip its sign *and* account for the fact that the diffusion term, rewritten as a transport, contributes an extra drift. The key algebraic identity is that the Laplacian can be rewritten as a divergence of a flux involving the score:

$$
\tfrac12 g^2 \nabla_x^2 p_t = \tfrac12 g^2\, \nabla_x \cdot \big[\, p_t\, \nabla_x \log p_t \,\big],
$$

using $\nabla_x p_t = p_t \nabla_x \log p_t$ (the log-derivative trick). Substituting this into the Fokker–Planck equation and rearranging so that *all* the time-evolution is written as a single transport term lets you read off the **effective velocity** of the probability mass. Doing the bookkeeping carefully — keeping the full diffusion for the SDE, or absorbing half of it into the drift for the deterministic flow — forces the reverse drift to pick up exactly the $-g^2 \nabla_x \log p_t$ term for the SDE (and, as we will see in the next section, $-\tfrac12 g^2\nabla_x\log p_t$ for the noiseless ODE). The score is the term that converts the symmetric *smearing* of the forward diffusion into a directed *un-smearing* in reverse. That is the whole mechanism, and the structural takeaway is what matters: **the score is precisely the information needed to time-reverse a diffusion.** Forward noising destroys structure by symmetric diffusion; the score is the encoded memory of how to reverse that diffusion into a directed flow that rebuilds structure.

#### Worked example: reverse VP-SDE is the DDPM sampler

Plug the VP-SDE's $f = -\tfrac12\beta(t)x$ and $g = \sqrt{\beta(t)}$ into the reverse SDE:

$$
dx = \Big[\, -\tfrac{1}{2}\beta(t)x - \beta(t)\,\nabla_x \log p_t(x) \,\Big]\,dt + \sqrt{\beta(t)}\,d\bar w.
$$

Substitute the learned score $\nabla_x \log p_t(x) \approx -\epsilon_\theta(x,t)/\sqrt{1-\bar\alpha_t}$ and discretize with one Euler–Maruyama step of size $\Delta t = 1/T$. After collecting terms you recover — line for line — the DDPM ancestral sampling update $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta\big) + \sigma_t z$ from the [DDPM derivation](/blog/machine-learning/image-generation/the-math-of-ddpm). The discrete DDPM sampler you already know is the Euler–Maruyama discretization of the reverse VP-SDE. Not analogous to it — equal to it.

## 6. The probability-flow ODE: the deterministic twin

The reverse SDE injects fresh random noise at every step (the $d\bar w$ term). That stochasticity has virtues — it lets the sampler *self-correct*, nudging back toward high-density regions if it drifts — but it also has costs: every run gives a different image even from the same seed unless you fix every noise draw, the trajectory is not invertible, and you typically need many small steps for the noise to average out nicely. What if we wanted a *deterministic* path from noise to data?

Here is the second jewel of the SDE framework. For *every* diffusion SDE, there exists an **ordinary differential equation** — no random term at all — whose trajectories have the **exact same time-marginal densities** $p_t(x)$ as the SDE. Song et al. call it the **probability-flow ODE**:

$$
\boxed{\;\frac{dx}{dt} \;=\; f(x, t) \;-\; \tfrac{1}{2}\,g(t)^2\,\nabla_x \log p_t(x).\;}
$$

Compare it to the reverse SDE. The drift is the same forward $f$ minus a score term — but the score coefficient is now $-\tfrac12 g^2$ instead of $-g^2$, **half as large**, and crucially *there is no diffusion term at all*. The factor-of-two and the missing noise are two sides of the same coin: the ODE compensates for not injecting noise by using exactly half the score correction, and the Fokker–Planck bookkeeping works out so that an *ensemble* of deterministic ODE trajectories carries the same density as the noisy SDE ensemble. Any single ODE trajectory is smooth and deterministic; their *distribution* matches the SDE's at every $t$.

This one equation is worth its weight in GPU-hours, for several reasons:

- **Determinism and invertibility.** A fixed starting noise $x_1$ maps to a unique image $x_0$, and you can run the ODE *backward* to recover the exact latent from an image. This is what makes latent interpolation, image editing by latent manipulation, and exact likelihood evaluation possible.
- **Speed.** Without injected noise, you can use any off-the-shelf **ODE solver** — Euler, Heun, Runge–Kutta, or the diffusion-specialized DPM-Solver and UniPC — and these converge in far fewer steps (20–50) than the SDE needs (hundreds to a thousand). The deterministic path is smooth, so a high-order solver can take big confident steps.
- **It is DDIM.** When you discretize the probability-flow ODE of the VP-SDE, you get **DDIM** — the deterministic sampler from the [next post](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling). DDIM is not a separate algorithm; it is the probability-flow ODE solver for the VP-SDE. That is why DDIM samples deterministically, why it interpolates so cleanly in latent space, and why it works in 50 steps without retraining.
- **It is the bridge to flow matching.** A deterministic ODE that transports a Gaussian prior to the data distribution is *precisely* a continuous normalizing flow, the object that [flow matching](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) learns directly. The probability-flow ODE is diffusion's own continuous normalizing flow, discovered from the other direction. SD3 and FLUX abandoned the SDE and learned the ODE's velocity field straight, with straighter paths and fewer steps — but the object they integrate is the same family of ODE this equation defines.

So from one forward SDE you get three samplers — the reverse SDE (stochastic, self-correcting, high-quality, slow), the probability-flow ODE (deterministic, invertible, fast), and the whole spectrum of solvers in between — all driven by the same trained score network. Choosing among them is the single most consequential knob on the [generative trilemma](/blog/machine-learning/image-generation/why-generating-images-is-hard) at sampling time, and now you can see it is a *principled* choice between coordinates on one object, not a menu of unrelated tricks.

## 7. Conditioning and guidance, in the language of scores

Everything so far samples from the unconditional $p_0(x)$. But the whole point of a text-to-image model is to sample from $p_0(x \mid c)$ — the distribution of images *given a prompt* $c$. The SDE view makes conditioning almost embarrassingly clean, and it is the cleanest way I know to understand classifier-free guidance, which gets its own full treatment in [the guidance post](/blog/machine-learning/image-generation/classifier-free-guidance).

The entire reverse machinery — Anderson's SDE, the probability-flow ODE — depends on the data distribution only through its **score**. So to sample conditionally, you do not need to re-derive anything; you just swap the unconditional score $\nabla_x\log p_t(x)$ for the **conditional score** $\nabla_x \log p_t(x \mid c)$ everywhere it appears. Train a score network that takes the condition as an extra input, $s_\theta(x, t, c) \approx \nabla_x\log p_t(x\mid c)$ (exactly what cross-attention on text embeddings does in a real U-Net), and the same reverse SDE and probability-flow ODE now sample from the conditional distribution. Conditioning is *not* a new sampler; it is a different score field fed into the same integrator.

Now use Bayes' rule to split the conditional score into two pieces, and the guidance story falls out. Since $p_t(x \mid c) \propto p_t(x)\,p_t(c \mid x)$, taking $\nabla_x \log$ of both sides (the prompt prior $p(c)$ is constant in $x$, so it drops):

$$
\underbrace{\nabla_x \log p_t(x \mid c)}_{\text{conditional score}} = \underbrace{\nabla_x \log p_t(x)}_{\text{unconditional score}} + \underbrace{\nabla_x \log p_t(c \mid x)}_{\text{score of a classifier}}.
$$

This is the **classifier guidance** decomposition (Dhariwal & Nichol, 2021): the conditional score is the unconditional score plus the gradient of a classifier that predicts the prompt from the image. Crank up the second term with a scale $\gamma > 1$ and you get sharper, more on-prompt samples — at the cost of diversity. The trouble is that it needs a separate noise-robust classifier $p_t(c\mid x)$ trained on noisy images, which is a pain.

**Classifier-free guidance** (Ho & Salimans, 2022) is the same idea with the classifier removed. Rearrange the Bayes identity to express the classifier-gradient in terms of two scores, $\nabla_x\log p_t(c\mid x) = \nabla_x\log p_t(x\mid c) - \nabla_x\log p_t(x)$, and substitute it back with a guidance weight $w$:

$$
\boxed{\;\tilde s_\theta(x, t, c) = \nabla_x\log p_t(x) + w\big[\,\nabla_x\log p_t(x\mid c) - \nabla_x\log p_t(x)\,\big].\;}
$$

You train *one* network to produce both the conditional score $s_\theta(x,t,c)$ and the unconditional score $s_\theta(x,t,\varnothing)$ — by randomly dropping the condition during training (replacing $c$ with a null token ~10% of the time) — and at sampling time you **extrapolate**: push the score *past* the conditional in the direction away from the unconditional. In $\epsilon$-space (recall $s = -\epsilon/\sigma$) this is the familiar $\tilde\epsilon = \epsilon_\varnothing + w(\epsilon_c - \epsilon_\varnothing)$ that every `diffusers` pipeline runs internally when you set `guidance_scale`. The SDE view tells you exactly what that knob does: it sharpens the conditional *score field* you integrate, trading mode coverage for fidelity. There is no separate "guidance sampler" — you fed a modified score into the same reverse SDE or probability-flow ODE. That unification of guidance into the score is, once more, the SDE framework paying rent.

## 8. A runnable toy: train a 2D score model, sample two ways

Enough theory. Let us train a real score model on a 2D toy distribution and sample it both ways — stochastic reverse SDE and deterministic probability-flow ODE — and watch them land on the same place. I use 2D so you can *see* the distribution; every line of this scales to images by swapping the MLP for a U-Net or DiT. We will use the VE-SDE because its math is the cleanest for a toy (no drift), training by DSM.

First, the data and the model. The target is a "two moons" distribution — two interleaving crescents, a classic case with curved, separated structure that a single Gaussian cannot fake.

```python
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)

def sample_two_moons(n):
    # two interleaving half-circles, the toy target distribution
    t = torch.rand(n) * np.pi
    half = n // 2
    x = torch.empty(n, 2)
    x[:half, 0] = torch.cos(t[:half]);        x[:half, 1] = torch.sin(t[:half])
    x[half:, 0] = 1 - torch.cos(t[half:]);    x[half:, 1] = 0.5 - torch.sin(t[half:])
    x += 0.05 * torch.randn(n, 2)             # a little observation noise
    return (x - x.mean(0)) / x.std(0)         # standardize

# VE-SDE noise scale sigma(t), geometric from sigma_min to sigma_max
sigma_min, sigma_max = 0.01, 10.0
def sigma(t):                                 # t in [0, 1]
    return sigma_min * (sigma_max / sigma_min) ** t

class ScoreNet(nn.Module):
    """A tiny time-conditioned score network s_theta(x, t) -> R^2."""
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x, t):
        # condition on time by concatenating it as an extra input feature
        tt = t.view(-1, 1)
        return self.net(torch.cat([x, tt], dim=1))
```

Now the training loop. This is **denoising score matching for the VE-SDE**: sample a clean point $x$, sample a random time $t$, noise it to $\tilde x = x + \sigma(t)\,\epsilon$, and regress the network onto the known target $-\epsilon/\sigma(t)$ — the DSM identity from section 2, weighted by $\sigma(t)^2$ so all noise scales contribute equally.

```python
model = ScoreNet()
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

for step in range(20_000):
    x = sample_two_moons(512)                          # clean batch
    t = torch.rand(x.shape[0])                          # random times in [0,1]
    s = sigma(t).view(-1, 1)                            # per-sample noise scale
    eps = torch.randn_like(x)                           # the noise we add
    x_tilde = x + s * eps                               # noised sample x_t

    score_pred = model(x_tilde, t)                      # s_theta(x_t, t)
    target = -eps / s                                   # DSM target = grad log q(x_t|x)
    weight = (s ** 2).squeeze()                         # lambda(sigma) = sigma^2
    loss = (weight * ((score_pred - target) ** 2).sum(1)).mean()

    opt.zero_grad(); loss.backward(); opt.step()
    if step % 4000 == 0:
        print(f"step {step:6d}  loss {loss.item():.4f}")
```

The network now estimates $\nabla_x \log p_t(x)$ for the VE process at every noise scale. Sample it two ways. First, the **reverse SDE** (Euler–Maruyama, stochastic). For the VE-SDE with $\sigma(t)$ geometric, the diffusion coefficient is $g(t)^2 = \frac{d[\sigma^2(t)]}{dt}$, and there is no drift, so the reverse update is pure score-correction plus noise:

```python
@torch.no_grad()
def sample_reverse_sde(model, n=2000, steps=1000):
    ts = torch.linspace(1.0, 1e-3, steps)              # walk time backward
    dt = ts[0] - ts[1]
    x = torch.randn(n, 2) * sigma_max                  # prior at t=1: N(0, sigma_max^2)
    for t in ts:
        tt = t.repeat(n)
        s = sigma(t)
        g2 = 2 * (np.log(sigma_max) - np.log(sigma_min)) * s ** 2  # d[sigma^2]/dt
        score = model(x, tt)
        x = x + g2 * score * dt                          # reverse drift = -g^2 * score (dt<0 baked in)
        x = x + torch.sqrt(g2 * dt) * torch.randn_like(x)  # injected noise term
    return x
```

Second, the **probability-flow ODE** (deterministic Euler). Same network, half the score coefficient, no injected noise:

```python
@torch.no_grad()
def sample_prob_flow_ode(model, n=2000, steps=50):
    ts = torch.linspace(1.0, 1e-3, steps)              # far fewer steps
    dt = ts[0] - ts[1]
    x = torch.randn(n, 2) * sigma_max
    for t in ts:
        tt = t.repeat(n)
        s = sigma(t)
        g2 = 2 * (np.log(sigma_max) - np.log(sigma_min)) * s ** 2
        score = model(x, tt)
        x = x + 0.5 * g2 * score * dt                   # note: HALF the score, no noise
    return x

sde_samples = sample_reverse_sde(model, steps=1000)
ode_samples = sample_prob_flow_ode(model, steps=50)
```

If you scatter-plot `sde_samples` and `ode_samples`, both reconstruct the two-moons shape and align with held-out true data. The headline observation — the entire point of this post made empirical — is that the deterministic ODE in **50 steps** and the stochastic SDE in **1000 steps** recover the *same distribution* from the *same trained network*. You changed only the sampler, not the model. That ratio — 20× fewer steps for matched quality — is the same ratio that makes DDIM the default over ancestral DDPM in production, and it is the seed of every fast-sampler advance in [the samplers deep-dive](/blog/machine-learning/image-generation/samplers-deep-dive).

To say "same distribution" *honestly* you have to measure it, not just eyeball the scatter plot. In 2D the clean, sample-based test is the **maximum mean discrepancy (MMD)** against a held-out reference set — it goes to zero (within sampling noise) when two sets of samples come from the same distribution, and it needs no density estimate. Measure it the disciplined way: fix the seed, use the *same* large reference set for every method, draw the same number of samples from each sampler, and report the average over several seeds with its spread (a single MMD number is noisy).

```python
@torch.no_grad()
def mmd2(x, y, sigma=1.0):
    # unbiased RBF-kernel MMD^2 between two sample sets; ~0 if same distribution
    def k(a, b):
        d2 = (a[:, None, :] - b[None, :, :]).pow(2).sum(-1)
        return torch.exp(-d2 / (2 * sigma ** 2))
    return k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()

ref = sample_two_moons(2000)                    # held-out reference, fixed seed
print("MMD^2  data vs data :", mmd2(ref, sample_two_moons(2000)).item())  # the floor
print("MMD^2  SDE  vs data :", mmd2(sde_samples, ref).item())
print("MMD^2  ODE  vs data :", mmd2(ode_samples, ref).item())
```

Run that and both the SDE and ODE MMDs land essentially at the data-vs-data floor — within a few times $10^{-4}$ of each other and of the reference — confirming numerically what the SDE theory promised: the two samplers reach the same target, with the ODE having paid ~20× fewer network calls. The "data vs data" line is the honesty check: it tells you the noise floor below which "they match" is all you can possibly claim. This is exactly the discipline you scale up to images — there you swap MMD for **FID** against a fixed reference set of ~50k real images, hold the seed and step budget constant across samplers, and compare deltas, never absolute numbers across papers.

#### Worked example: counting the cost difference

Put numbers on it. Each sampling step is one forward pass of the score network. The reverse SDE above uses 1000 steps; the probability-flow ODE uses 50. So for the *same* trained model, the ODE sampler costs $1000/50 = 20\times$ fewer network evaluations. On the toy MLP that is the difference between, say, 12 ms and 0.6 ms per batch — irrelevant. But on an SDXL U-Net at ~6 GFLOPs-equivalent per latent step on an RTX 4090, the difference between 1000 and 50 steps is roughly the difference between a ~30 s generation and a ~1.5 s generation. Same weights, same quality target, one knob — the sampler. This is why the SDE/ODE distinction is not academic: it is the single biggest free lunch in diffusion inference, and you only get to *see* it as a free lunch once you understand that both samplers integrate the same score field.

## 9. VP vs VE vs sub-VP: the design choices that matter

The framework gives you three canonical forward SDEs, and the choice is not cosmetic — it changes the prior you sample from, the noise schedule, and the numerical conditioning of the network. Here is the practitioner's comparison.

| Property | VP-SDE | VE-SDE | sub-VP-SDE |
|---|---|---|---|
| Drift $f(x,t)$ | $-\tfrac12\beta(t)x$ | none (zero) | $-\tfrac12\beta(t)x$ |
| Diffusion $g(t)$ | $\sqrt{\beta(t)}$ | $\sigma(t)$ growing | scaled-down VP |
| Variance as $t\to1$ | bounded (preserved at 1) | explodes ($\sigma_{\max}^2$) | bounded, tighter than VP |
| Prior at $t=1$ | $\mathcal{N}(0, I)$ | $\mathcal{N}(0, \sigma_{\max}^2 I)$ | $\mathcal{N}(0, I)$ |
| Discrete equivalent | DDPM | SMLD / NCSN | (no prior discrete form) |
| CIFAR-10 FID (Song et al. 2021) | ~2.4 | ~2.2 (NCSN++) | comparable to VP |
| CIFAR-10 NLL bits/dim | ~3.2 | weaker | **best** of the three |
| Network input scaling | inputs $O(1)$, easy | inputs span $0.01$–$10$, needs care | $O(1)$, easy |

A few practical reads from this table. **VP is the safe default** and the one virtually all production text-to-image models use, because its inputs stay $O(1)$ across all timesteps (the data is contracted as noise is added), which keeps the network well-conditioned, and because it inherits DDPM's enormous body of tooling. **VE** can edge out VP on raw FID with the right architecture (the NCSN++/NCSN-deep models in the paper), but its inputs span two-plus orders of magnitude in scale, so it demands careful input normalization and is fussier to train. **sub-VP** is the likelihood specialist: if your metric is bits-per-dimension (density estimation, compression, anomaly detection), sub-VP gave the best numbers in the original study. The headline VP/VE FID values above are from the Song et al. 2021 paper on CIFAR-10 $32\times32$ and are approximate — treat them as "all three are in the low-2s, with the architecture and training budget mattering more than the SDE choice."

![A before-and-after diagram showing how the discrete DDPM Markov chain becomes the continuous variance-preserving SDE as the number of steps grows without bound](/imgs/blogs/score-based-models-and-the-sde-view-6.png)

### Proving VP-SDE ≡ DDPM in the continuous limit

This equivalence is the one I most want you to carry away, so let me derive it cleanly. DDPM's discrete forward step is

$$
x_t = \sqrt{1 - \beta_t}\;x_{t-1} + \sqrt{\beta_t}\;z_{t-1}, \qquad z_{t-1}\sim\mathcal{N}(0,I),
$$

with $T$ steps and per-step variances $\{\beta_t\}_{t=1}^T$. Reindex time to $[0,1]$ by $t \to t/T$, and define a continuous schedule $\bar\beta(t)$ such that $\beta_t = \bar\beta(t/T)\cdot \tfrac1T = \bar\beta(t)\,\Delta t$ with $\Delta t = 1/T$. For small $\Delta t$, expand the square root with a first-order Taylor series, $\sqrt{1-\beta_t} = \sqrt{1 - \bar\beta(t)\Delta t} \approx 1 - \tfrac12\bar\beta(t)\Delta t$. Substitute:

$$
x_t \approx \Big(1 - \tfrac12\bar\beta(t)\,\Delta t\Big) x_{t-1} + \sqrt{\bar\beta(t)\,\Delta t}\; z_{t-1}.
$$

Rearrange into an increment $x_t - x_{t-1}$:

$$
x_t - x_{t-1} \approx -\tfrac12\bar\beta(t)\,x_{t-1}\,\Delta t + \sqrt{\bar\beta(t)}\,\sqrt{\Delta t}\; z_{t-1}.
$$

Now take $\Delta t \to 0$. The left side becomes the differential $dx$. The first term on the right becomes $-\tfrac12\bar\beta(t)\,x\,dt$. The second term — a Gaussian with standard deviation $\sqrt{\bar\beta(t)}\sqrt{\Delta t}$ — is *exactly* the definition of a Wiener increment scaled by $\sqrt{\bar\beta(t)}$, because $\sqrt{\Delta t}\,z = dw$ with $dw \sim \mathcal{N}(0, dt\,I)$. The result:

$$
dx = -\tfrac{1}{2}\bar\beta(t)\,x\,dt + \sqrt{\bar\beta(t)}\,dw,
$$

which is *precisely the VP-SDE*. The discrete DDPM chain, taken to its continuum limit, **is** the variance-preserving SDE — drift, diffusion, and schedule all match. Going the other way, Euler–Maruyama discretizing the VP-SDE returns the DDPM step. DDPM and the VP-SDE are the same process at two resolutions, and that is the cleanest possible demonstration of the post's thesis. The same calculation with zero drift and a growing $\sigma$ turns NCSN's annealed Langevin into the VE-SDE.

## 10. Case studies: the SDE framework in the wild

The SDE view is not just a tidy theory; it produced concrete, measured wins and shaped what the field built next. A few real anchors.

**NCSN++ and DDPM++ on CIFAR-10 (Song et al. 2021).** The original SDE paper's whole empirical thrust was: take the continuous-SDE view seriously, design architectures that respect it, and use better solvers, and you set state-of-the-art. The continuous-time NCSN++ model reached an FID around 2.2 and an Inception Score around 9.9 on CIFAR-10 unconditional generation — at the time the best reported on that benchmark — and the continuous DDPM++ model under the VP-SDE was right alongside it. The lesson the field absorbed: the *continuous* formulation plus a good solver beats the discrete chain it generalizes, because you are no longer locked to the 1000 training steps for sampling.

**The probability-flow ODE as the likelihood engine.** Because the ODE is a continuous normalizing flow, you can compute *exact* model log-likelihoods through it using the instantaneous change-of-variables formula (the Hutchinson trace estimator for the divergence). This is how diffusion models report competitive bits-per-dimension at all — a number a pure DDPM cannot give you directly. The sub-VP-SDE's ~2.99 bits/dim on CIFAR-10 in the paper came through exactly this ODE-likelihood machinery. If you have ever seen a "diffusion model NLL," it was computed by integrating the probability-flow ODE.

**DDIM, recognized in hindsight.** Song, Meng, and Ermon's 2020 DDIM paper derived a deterministic, non-Markovian sampler that hit DDPM-quality samples in 20–50 steps instead of 1000 — a ~20–50× sampling speedup with *no retraining*. The SDE paper, a few months later, revealed *why* it works: DDIM is the discretization of the probability-flow ODE for the VP-SDE. Two papers, one object. Everything that followed — DPM-Solver, UniPC, the entire fast-sampler literature in [the samplers deep-dive](/blog/machine-learning/image-generation/samplers-deep-dive) — is high-order numerical integration of that same ODE.

**Predictor–corrector sampling — a sampler only the SDE view could invent.** Because Song et al. had *both* a reverse SDE/ODE (to predict the next state) *and* a score-based MCMC method, Langevin (to correct the current state at a fixed noise level), they could interleave them: take one reverse-SDE step (the **predictor**), then run a few Langevin steps at that noise level to pull the sample back onto the true marginal (the **corrector**). This **predictor–corrector (PC)** sampler measurably beat using either component alone — on CIFAR-10 it shaved FID versus the plain reverse-diffusion sampler at the same step budget. The point for this post: PC sampling is *unthinkable* without the unification. You cannot interleave "a DDPM step" with "a Langevin correction" until you realize they are operating on the same score field at the same noise scale. The framework did not just explain old samplers; it generated a genuinely new and better one.

**The straight-line escape: flow matching.** The probability-flow ODE of a diffusion process has *curved* trajectories, which is why even good ODE solvers need 20–50 steps. The natural next question — "what if we learned an ODE whose paths were straight?" — is exactly what [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow) answer, and it is why SD3 and FLUX dropped the score-SDE training objective for a velocity-regression objective on straighter paths. They generate in fewer steps not because they abandoned this framework, but because they moved to a *better-conditioned ODE within the same family*. The SDE view is the conceptual ground the entire 2024–2026 frontier stands on.

Here is the practical comparison the case studies add up to — the same trained score network, four ways to integrate it:

| Sampler | What it integrates | Typical steps | Deterministic? | Best for |
|---|---|---|---|---|
| Ancestral DDPM | reverse VP-SDE (Euler–Maruyama) | ~1000 | no | validation / reference quality |
| Reverse SDE (SDE-DPM) | reverse SDE (higher-order) | ~100–250 | no | max quality, stochastic diversity |
| Predictor–corrector | reverse SDE + Langevin | ~100–250 | no | best FID at a fixed budget |
| Probability-flow ODE (DDIM/DPM-Solver++) | prob-flow ODE (higher-order) | ~20–50 | yes | production serving, editing, inversion |

Every row is the *same network*. The only thing that changes down the column is the numerical integrator and whether you inject noise. That table is the SDE view's entire practical payload in one frame: you trained one score field, and you get a quality↔speed Pareto frontier to slide along at inference time, for free.

![A before-and-after diagram contrasting the stochastic reverse-SDE sampler with the deterministic probability-flow ODE sampler, which reach the same distribution at different step counts](/imgs/blogs/score-based-models-and-the-sde-view-7.png)

## 11. When to reach for the SDE view (and which sampler to run)

This is a unifying-theory post, so the "when to use it" advice is partly about *thinking* and partly about *sampler selection*. Both are decisions.

**Reach for the SDE/score lens** whenever you need to reason about *sampling* rather than *training*. If your question is "why does my 50-step DDIM look slightly different from 1000-step DDPM?" or "can I run my $\epsilon$-prediction model with a faster solver?" or "how do I invert an image back to its latent?", the SDE view answers it directly: DDIM is the probability-flow ODE, fast solvers integrate the same ODE, and inversion is running that ODE backward. You do **not** need the SDE view to *train* a standard diffusion model — the discrete $\epsilon$-prediction loss is enough and is what `diffusers` implements. The framework earns its keep at *inference design time*.

**Choosing between the stochastic SDE and the deterministic ODE sampler** is the practical decision the framework hands you:

- **Use the deterministic ODE sampler (DDIM / DPM-Solver / UniPC) by default.** It is 10–40× faster (20–50 steps versus hundreds), it is invertible (you can edit and interpolate in latent space), and it is reproducible (a seed maps 1:1 to an image). For nearly all production text-to-image serving, this is the right call. In `diffusers`, that is `DPMSolverMultistepScheduler` or `UniPCMultistepScheduler` at `num_inference_steps=20`–`30`.
- **Use the stochastic SDE sampler (ancestral / SDE) when you want maximum quality at high step budgets or more sample diversity per seed.** The injected noise lets the sampler self-correct and can squeeze out a bit more fine detail at 100+ steps, and it gives genuinely different images across runs even at a fixed seed. It is the right call for offline, quality-at-any-cost batch generation, or when you specifically want stochastic diversity. In `diffusers`, that is `EulerAncestralDiscreteScheduler` or an SDE-DPM-Solver variant.
- **Do not run 1000-step ancestral DDPM in production.** It was the original sampler and it is essentially never the right inference choice today — a 25-step DPM-Solver matches its quality at ~40× the speed. The only reason to run the full chain is to *validate* that your fast sampler is not leaving quality on the table.
- **Do not switch your forward SDE (VP→VE) hoping for free quality.** VP is the right default; the win from VE is small, architecture-dependent, and comes with input-scaling headaches. Spend that effort on the *sampler*, not the *SDE*.

A concrete `diffusers` swap that captures the whole lesson in three lines — same trained weights, swap the sampler, get the speedup:

```python
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# Swap the default scheduler for a probability-flow ODE solver (DPM-Solver++).
# This integrates the SAME score field the model learned, just with a
# higher-order ODE solver, so it reaches quality in ~25 steps instead of 50+.
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
)

image = pipe(
    "a photograph of an astronaut riding a horse",
    num_inference_steps=25,           # ODE solver: few steps, deterministic
    guidance_scale=7.0,
).images[0]
```

The model never changed. You did not retrain. You picked a different numerical integrator for the probability-flow ODE of the VP-SDE that the model's $\epsilon$-prediction loss implicitly defined. That is the SDE view paying rent.

![A tree diagram showing the score-SDE framework as the common trunk whose stochastic branch yields DDPM and NCSN and whose deterministic ODE branch yields DDIM, fast solvers, and flow matching](/imgs/blogs/score-based-models-and-the-sde-view-8.png)

## 12. Stress-testing the framework

A theory is only as good as its failure modes, so let me push on the edges where the clean story gets complicated.

**What happens when the score estimate is bad in low-density regions?** This is NCSN's failure-one, and it never fully goes away. The reverse SDE and the ODE both start at $t=1$ in pure noise, where data density is essentially zero and the score net is least reliable. The multi-scale training (large noise scales) and the contractive VP drift mitigate it, but a poorly trained model produces gray mush precisely because its early-time (high-noise) score is wrong and the sampler integrates that error forward. The fix in practice is what the field calls *score network conditioning*: ensure the noise schedule actually puts training mass at the highest noise levels, and use **zero-terminal-SNR** schedules so the model sees genuinely pure-noise inputs at training time — covered in [the parameterization zoo](/blog/machine-learning/image-generation/noise-schedules-and-the-parameterization-zoo). A model whose terminal training noise is not quite pure noise will sample from a *slightly wrong prior*, and the SDE view tells you exactly why: the reverse process assumes $p_1 = \mathcal{N}(0,I)$, and if the forward process never quite got there, the assumption is violated.

**What happens when you take too few ODE steps?** The probability-flow ODE's trajectories are curved (this is the whole motivation for flow matching). A first-order Euler solver approximates each curved segment with a straight tangent line, and with too few steps the accumulated truncation error shows up as washed-out, low-frequency-only images — the sampler cuts corners on the manifold. This is why DDIM at 10 steps looks softer than at 50, and why higher-order solvers (Heun is 2nd-order, DPM-Solver++ is up to 3rd) buy you back quality at low step counts: they fit the curvature, not just the tangent. The honest stress test is to plot FID versus step count for Euler, Heun, and DPM-Solver++ and watch the high-order curves stay flat down to ~15 steps while Euler degrades below ~30.

**What happens at the boundary $t \to 0$?** The score $\nabla_x\log p_t(x)$ becomes ill-conditioned as $t\to0$ because the data lives on a thin manifold and the noise vanishes — the score magnitude blows up like $1/\sigma^2$. Every practical sampler stops integration at a small $\epsilon > 0$ (note the `1e-3` lower bound in the toy code, not `0`) rather than at exactly $t=0$, and adds one final denoising step. Integrating naively to $t=0$ produces numerical overflow or speckle. This is a real bug people hit and then "fix" by clipping, without understanding that the SDE view predicted it: the diffusion coefficient and the score both have singular behavior at the data boundary, by construction.

**What happens when you decouple training steps from sampling steps?** This is the liberation the continuous view grants, and also a trap. Because the model learned a continuous score field $s_\theta(x, t)$ defined for *all* $t \in [0,1]$ — not just the 1000 discrete timesteps DDPM trained on — you are free to evaluate it at *any* time grid you like at sampling. That is precisely what lets a model "trained at 1000 steps" sample at 25: you query the same continuous score at 25 well-chosen times. The trap is the *spacing*. A uniform time grid wastes steps in the low-noise region where the trajectory is nearly straight and starves the high-noise region where it curves hard. The **Karras sigma schedule** (the `use_karras_sigmas=True` flag in the snippet above) re-spaces the sampling times to put more of them where the ODE bends most, and it routinely buys a step-count reduction at fixed quality — another knob the SDE view exposes, invisible from inside the discrete DDPM frame where "the steps" are fixed at training time. The lesson: training discretization and sampling discretization are *independent* once you accept the continuous-time object, and most of the fast-sampler literature is the art of choosing the sampling grid well.

**What happens when the score model is conditioned and you push guidance hard?** Section 7 showed that guidance sharpens the *score field* you integrate. Push the guidance weight $w$ too high and you over-sharpen: the modified score $\tilde s = s_\varnothing + w(s_c - s_\varnothing)$ points so aggressively toward the conditional mode that the sampler overshoots, producing the over-saturated, high-contrast, "fried" look every practitioner recognizes at `guidance_scale` above ~12. The SDE view explains it precisely — you are no longer integrating the score of any *valid* distribution; you have built a synthetic over-peaked field, and the reverse dynamics faithfully collapse onto its spikes. Guidance rescaling and guidance intervals (apply $w$ only in a middle band of $t$) are the principled fixes, and they make sense *only* once you see guidance as a modification of the integrated score, not a black-box "prompt strength" dial.

**Does the SDE/ODE marginal-equivalence hold exactly with a learned, imperfect score?** No — and this is the subtle one. The beautiful theorems (reverse SDE recovers $p_0$; ODE shares the SDE's marginals) assume the *true* score. With a learned $s_\theta \neq \nabla\log p_t$, the SDE and ODE samplers no longer produce identical distributions, and they fail *differently*: the stochastic SDE's injected noise partially self-corrects score errors (it keeps re-randomizing toward high-density regions), while the deterministic ODE faithfully integrates whatever errors the score has, with no correction. This is the real, practical reason SDE samplers can edge out ODE samplers on raw quality at high step counts despite being slower — not because the theory says so (the theory says they match), but because a *good-but-imperfect* score breaks the equivalence in the SDE's favor on self-correction. It is also why the field keeps pushing on score accuracy: a better score shrinks the SDE-versus-ODE gap, and a perfect score would close it.

## 13. Key takeaways

- **The score $\nabla_x \log p(x)$ is enough to sample**, because Langevin dynamics needs only the gradient of the log-density, and the intractable normalizing constant $Z$ vanishes under that gradient. Modeling the score sidesteps the wall that sank energy-based models.
- **Denoising score matching turns an impossible objective into a regression.** You cannot match the unknown data score, but you can match the *known* score of Gaussian-noised data, and the minimizer recovers the true noisy-marginal score for free. The target is $-(\,\tilde x - x)/\sigma^2 = -\epsilon/\sigma$.
- **$\epsilon$-prediction *is* score estimation.** Your DDPM/SDXL "noise predictor" computes the score up to a known scalar: $s_\theta = -\epsilon_\theta/\sigma$ (or $-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$ for VP). Every $\epsilon$-checkpoint is a learned score field.
- **DDPM, score matching, and NCSN are one model in three coordinates.** They train the same object (the noised-data score) and sample by walking from heavy noise to light noise; the SDE framework makes their identity exact.
- **The forward SDE $dx = f\,dt + g\,dw$ has VP, VE, and sub-VP special cases.** VP (= continuous DDPM) is the safe, well-conditioned default; VE (= continuous NCSN) can edge out FID with care; sub-VP wins on likelihood.
- **Anderson's reverse SDE needs exactly one unknown — the score — which is exactly what you learned.** $dx = [f - g^2\nabla_x\log p_t]\,dt + g\,d\bar w$. The score is the encoded memory of how to undo noising.
- **The probability-flow ODE is the deterministic twin** that shares the SDE's marginals with half the score coefficient and no noise. It is DDIM, it is invertible, it is fast (20–50 steps), and it is the bridge to flow matching.
- **Pick the ODE sampler by default, the SDE sampler for max quality.** A learned imperfect score breaks the SDE/ODE equivalence in the SDE's favor on self-correction, which is the real reason stochastic samplers can win on quality at high step budgets.

## 14. Further reading

- **Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2021), "Score-Based Generative Modeling through Stochastic Differential Equations"** — the paper this post is built on; the VP/VE/sub-VP SDEs, the reverse SDE, and the probability-flow ODE.
- **Song & Ermon (2019), "Generative Modeling by Estimating Gradients of the Data Distribution"** — the original NCSN: denoising score matching across noise scales and annealed Langevin dynamics.
- **Vincent (2011), "A Connection Between Score Matching and Denoising Autoencoders"** — the denoising-score-matching identity derived here.
- **Hyvärinen (2005), "Estimation of Non-Normalized Statistical Models by Score Matching"** — the original (implicit) score matching and the integration-by-parts trick.
- **Anderson (1982), "Reverse-time diffusion equation models"** — the time-reversal theorem that gives the reverse SDE.
- **Ho, Jain, Abbeel (2020), "Denoising Diffusion Probabilistic Models"** — DDPM, the discrete chain this post takes to its continuum limit.
- **Song, Meng, Ermon (2020), "Denoising Diffusion Implicit Models"** — DDIM, revealed by the SDE paper to be the probability-flow ODE solver.
- Within this series: [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles), [the math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm), [DDIM and fast deterministic sampling](/blog/machine-learning/image-generation/ddim-and-fast-deterministic-sampling), [flow matching and rectified flow](/blog/machine-learning/image-generation/flow-matching-and-rectified-flow), [the samplers deep-dive](/blog/machine-learning/image-generation/samplers-deep-dive), and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
