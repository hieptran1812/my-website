---
title: "The Mathematics of Image Distributions: Likelihood, Manifolds, and the Metrics That Judge Us"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Build the probabilistic foundation under every image generator — what p(x) means, why images live on a manifold, the three ways to model a distribution, the ELBO derived from scratch, and what FID, CLIP-score, and human-preference metrics actually measure (and where they lie)."
tags:
  [
    "image-generation",
    "diffusion-models",
    "generative-modeling",
    "elbo",
    "fid",
    "evaluation-metrics",
    "manifold-hypothesis",
    "generative-ai",
    "deep-learning",
    "probability",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/the-mathematics-of-image-distributions-1.png"
---

You type "a red cube to the left of a blue sphere, studio lighting" into two different text-to-image models. Both return a crisp, photoreal render. You run the standard benchmark and both land at **FID 8.0** — a near-perfect tie. Ship the one with the better marketing? Here is the catch: model A drew three blue spheres and no cube, and model B drew exactly what you asked for. The headline number could not tell them apart, because **FID never reads the caption**. If you are going to train, fine-tune, or even just *choose* an image model, you need to know precisely what the math under it is doing and what the numbers on the leaderboard are — and are not — measuring.

This post is the probabilistic spine of the whole series. Everything that comes later — the [forward and reverse processes of diffusion](/blog/machine-learning/image-generation/diffusion-from-first-principles), the VAE that latent diffusion is built on, classifier-free guidance, flow matching — is a particular answer to one question: *how do you model and sample from the distribution of natural images?* And every claim that "model X is better than model Y" is a particular answer to a second question: *how do you measure that you got the distribution right?* Get these two foundations clean and the rest of the series reads like applied consequences. Get them muddy and you will fine-tune toward a metric that is lying to you.

Here is the plan. First half — **the distribution**: images as samples from an unknown density `$p(x)$`; the manifold hypothesis with the dimension-counting argument that makes it inevitable; the three ways to model a distribution (explicit likelihood, implicit/sample-only, and score/energy) and what each trades away; a clean derivation of KL divergence and the **ELBO = reconstruction − KL** that every later post optimizes; and the curse of dimensionality that kills naive density estimation. Second half — **the judgment**: what FID, Inception Score, CLIP-score, precision/recall, and learned preference metrics actually compute, derived from the definitions, and exactly where each one lies. Figure 1 maps the three modeling families we will keep returning to.

![A taxonomy tree showing how explicit likelihood, implicit sample-only, and score-based families each model the data density with different trade-offs](/imgs/blogs/the-mathematics-of-image-distributions-1.png)

By the end you will be able to: write down what "learning a generative model" formally means; derive the ELBO without looking it up; compute FID and CLIP-score with real `torchmetrics`/`open-clip` code; and read a leaderboard with the skepticism it deserves. This is the frame the rest of the series formalizes: the **generative trilemma** (sample quality × mode coverage × sampling speed) lives entirely inside the distribution we are about to define, and the metrics we derive are how we *see* each corner of it. If you have not read [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard), that post sets up the problem; this one supplies the math.

## 1. An image is a sample from a distribution

Start with the most literal description possible. A `$512 \times 512$` RGB image is a point in `$\mathbb{R}^{D}$` where `$D = 512 \times 512 \times 3 = 786{,}432$`. If you quantize each channel to 8 bits, the image is one specific element of a finite set with `$256^{786432}$` members. That number has more than a million digits. It dwarfs the number of atoms in the observable universe by an unfathomable margin. Almost every element of that set is pure static — uniform noise with no structure. The images a human would call "a photograph" form a vanishingly thin slice.

The generative-modeling claim is that this slice is not arbitrary. There is some probability distribution `$p_\text{data}(x)$` over `$\mathbb{R}^{D}$` that assigns high density to "things that look like real photographs" and essentially zero density to static. We never get to see `$p_\text{data}$` directly. What we get is a finite dataset `$\{x_1, x_2, \dots, x_N\}$` — LAION, ImageNet, your scraped corpus — which we treat as **i.i.d. samples** drawn from `$p_\text{data}$`. The entire game of generative modeling is:

> Given samples from an unknown `$p_\text{data}(x)$`, learn a model `$p_\theta(x)$` (with parameters `$\theta$`) that is close to `$p_\text{data}$`, and from which we can **draw new samples**.

Two verbs hide in there, and the rest of this series turns on the difference between them. **Evaluating** the density — being able to compute `$p_\theta(x)$` for a given `$x$` — is one capability. **Sampling** from the density — drawing a fresh `$x \sim p_\theta$` — is a different capability. A model might do one, the other, both, or neither cheaply. A histogram lets you evaluate but is useless for high-dimensional sampling. A GAN lets you sample but cannot evaluate `$p_\theta(x)$` at all. Keep this distinction in your head; it is the seam along which the four generative families split.

#### Worked example: how big is the gap between "all grids" and "real images"

Take a tiny case so the numbers are graspable: a `$2 \times 2$` grayscale image, 8 bits per pixel. The ambient space has `$256^4 \approx 4.3 \times 10^9$` possible grids — about four billion. Of those, how many are "natural"? There is no exact answer, but suppose the natural ones form a smooth 2-parameter family (say, a soft gradient with a tunable direction and contrast). A 2-D family discretized at 256 levels per axis has about `$256^2 = 65{,}536$` members. So the natural fraction is roughly `$65{,}536 / 4.3\times10^9 \approx 1.5 \times 10^{-5}$` — fifteen in a million. Now scale `$D$` from 4 to 786,432. The ambient space grows as `$256^D$` while the natural manifold grows only as `$256^d$` with `$d \ll D$`. The natural fraction is `$256^{d-D}$`, which for any realistic `$d \ll D$` is so close to zero that "the model must place essentially all its probability mass on a measure-zero set" stops being hyperbole and becomes the literal engineering problem.

That last sentence is the manifold hypothesis, and it deserves its own section.

## 2. The manifold hypothesis and a dimension-counting argument

The **manifold hypothesis** states that real-world high-dimensional data (images, audio, natural language embeddings) concentrate near a low-dimensional manifold embedded in the high-dimensional ambient space. For images: the *ambient* dimension is `$D = 786{,}432$`, but the *intrinsic* dimension `$d$` — the number of genuine degrees of freedom you would need to specify a natural image — is dramatically smaller. Figure 2 is the picture to hold in your head: a thin, curved sheet (the manifold) winding through a vast cube (the ambient pixel space), with real photographs sitting *on* the sheet and random static sitting *off* it.

![A graph contrasting the thin curved image manifold against the vast ambient pixel space with real photos on the sheet and random noise off it](/imgs/blogs/the-mathematics-of-image-distributions-2.png)

Why believe `$d \ll D$`? A dimension-counting argument makes it concrete. Consider photographs of a single rigid object — say, a teapot — under controlled conditions. The degrees of freedom are roughly: 3 for camera position, 3 for camera orientation, 1 for focal length, a handful for lighting direction and intensity, a few for material and color. Call it `$d \approx 15$`–`$50$`. Yet each such photo is a point in a space of dimension `$D \approx 10^6$`. The set of teapot images is a `$\sim 30$`-dimensional manifold inside a million-dimensional box. Empirically, **intrinsic dimension estimators** (nearest-neighbor scaling, maximum-likelihood estimation of local dimension, two-NN) on real image datasets return values in the range of tens to low hundreds — ImageNet's intrinsic dimension is estimated at roughly `$d \approx 40$`–`$60$` depending on the estimator, against `$D$` in the hundreds of thousands. The gap is enormous and it is the entire reason generative modeling is possible at all: you are not learning a function over a million-dimensional cube, you are learning the *shape and density of a thin sheet*.

This has three immediate consequences that shape every later architecture decision in the series.

**Consequence 1: the true density is degenerate.** If the data truly lives on a `$d$`-dimensional manifold with `$d < D$`, then `$p_\text{data}(x)$` as a density over `$\mathbb{R}^D$` is **zero almost everywhere and infinite on the manifold** — it is not absolutely continuous with respect to Lebesgue measure. Likelihood is ill-defined in the strict sense. This is not a pedantic worry; it is *why* we add noise. Diffusion models and score matching both work by **convolving the data with Gaussian noise**, which "puffs up" the measure-zero manifold into a full-dimensional, smooth density that *is* well-defined everywhere. The noise is not a bug to tolerate; it is the mathematical device that rescues likelihood. We will see this again in [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles).

**Consequence 2: a good model puts mass *on the sheet*.** The failure mode of a bad generator is putting probability mass *off* the manifold — in the near-manifold region you get blurry, slightly-wrong images (a VAE's signature); far off it you get static. A good model's samples land on the sheet. This is exactly what fidelity metrics try to detect.

**Consequence 3: distances in pixel space are nearly meaningless.** Two images that are perceptually identical (same scene, shifted one pixel) can be far apart in `$L_2$` pixel distance, while two perceptually different images can be close. The manifold is curved, so the straight-line ambient distance is the wrong ruler. This is *why* FID and friends measure distance in a **learned feature space** (Inception, CLIP, DINOv2) rather than pixel space — feature space approximately "flattens" the manifold so that Euclidean distance there tracks perceptual similarity. Hold that thought; it is the load-bearing assumption under FID, and also FID's deepest weakness.

## 3. Three ways to model a distribution

There are exactly three structurally distinct strategies for representing `$p_\theta(x)$`, and every generative family is one of them. The choice is not cosmetic — it decides whether you can compute likelihoods, how you sample, how stable training is, and where on the trilemma you land. The three are: **explicit likelihood**, **implicit (sample-only)**, and **score/energy**. (Figure 1 is this taxonomy.)

### 3.1 Explicit likelihood models

These parameterize `$p_\theta(x)$` directly and train by **maximizing the log-likelihood** `$\sum_i \log p_\theta(x_i)$` of the data. The subfamilies differ in *how they make `$p_\theta(x)$` computable*:

- **Autoregressive models** (PixelCNN, Image GPT, the modern token-based image models) factor the joint exactly via the chain rule: `$p_\theta(x) = \prod_{j} p_\theta(x_j \mid x_{<j})$`. Each conditional is a tractable softmax. Likelihood is *exact*. Sampling is *sequential and slow* — one pixel/token at a time. Covered in [autoregressive image models](/blog/machine-learning/image-generation/autoregressive-image-models).
- **Normalizing flows** (RealNVP, Glow) push a simple base density through an invertible map `$f_\theta$` and use the **change-of-variables formula** to get an exact density. Likelihood is exact, sampling is a single forward pass — but the invertibility constraint limits expressiveness. See [normalizing flows and change of variables](/blog/machine-learning/image-generation/normalizing-flows-and-change-of-variables).
- **Variational autoencoders** introduce a latent `$z$` and an intractable marginal `$p_\theta(x) = \int p_\theta(x \mid z) p(z)\, dz$`. They cannot compute the likelihood exactly, so they optimize a **lower bound** (the ELBO, §5). Likelihood is *approximate*; sampling is one forward pass; samples are *blurry* because the bound and the Gaussian decoder smooth the manifold. The autoencoder that latent diffusion is built on — [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch).

The shared advantage: you have a number, `$\log p_\theta(x)$` (exact or bounded), that you can optimize and report. The shared disadvantage: forcing the model to assign a valid, normalized density everywhere is a heavy constraint, and it tends to **spread mass off the manifold** — the model "wastes" probability on near-manifold blur to keep the density normalized. That is the textbook reason VAE samples look soft.

### 3.2 Implicit (sample-only) models

A **generative adversarial network** does not represent `$p_\theta(x)$` at all. It defines a generator `$G_\theta: z \mapsto x$` that transforms noise `$z \sim \mathcal{N}(0,I)$` into an image, and that is the *entire* model of the distribution: `$p_\theta$` is whatever push-forward `$G_\theta$` induces. You cannot evaluate `$p_\theta(x)$` for a given `$x$` — there is no formula, no integral, nothing. You can only **sample**. Training avoids likelihood entirely: a discriminator `$D_\phi$` learns to tell real from generated, and `$G$` is trained to fool it, a minimax game

$$\min_\theta \max_\phi \; \mathbb{E}_{x \sim p_\text{data}}[\log D_\phi(x)] + \mathbb{E}_{z}[\log(1 - D_\phi(G_\theta(z)))].$$

At the optimum, this minimizes the Jensen–Shannon divergence between `$p_\text{data}$` and `$p_\theta$`. The derivation is short and illuminating: for a *fixed* generator, the optimal discriminator is `$D^*(x) = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_\theta(x)}$` (just maximize the discriminator's objective pointwise). Substitute that back into the generator's objective and the value function reduces, after a few lines, to `$-\log 4 + 2\,D_{\mathrm{JS}}(p_\text{data} \,\|\, p_\theta)$`. Since `$D_{\mathrm{JS}} \ge 0$` with equality iff the two distributions match, the generator's global optimum is `$p_\theta = p_\text{data}$`, achieved at value `$-\log 4$`. So the GAN is, in the idealized limit, a JS-divergence minimizer — a symmetric, bounded cousin of the KL we have been using.

Because there is no likelihood normalization constraint, GANs can put *all* their mass tightly on the manifold — which is why a good GAN's samples are razor-sharp. The cost is twofold. First, training is unstable: the idealized "optimal discriminator then optimal generator" alternation does not hold in practice, the two networks chase a moving target, and when the discriminator gets too good its gradients to the generator vanish (the JS divergence saturates), so the generator stops learning. Second, and more fundamental, the model can **drop modes** — cover only part of the data distribution while looking great on the part it covers — because a generator that captures one mode perfectly can already fool a discriminator that is not specifically penalizing missing modes. This is the implicit family's defining trade: spectacular fidelity, fragile coverage. The mode-dropping is the reverse-KL-flavored zero-forcing behavior from §4 showing up in practice, and it is exactly what *recall* (§11) was invented to measure. The full story — including why GANs nonetheless came roaring back as a *distillation* loss for one-step diffusion — is in [generative adversarial networks and why they lost](/blog/machine-learning/image-generation/generative-adversarial-networks-and-why-they-lost).

### 3.3 Score / energy models

The third family sidesteps both the normalization headache of explicit models and the no-density problem of implicit ones by modeling **the gradient of the log-density** instead of the density itself:

$$s_\theta(x) \approx \nabla_x \log p_\text{data}(x).$$

This object is called the **score**. Why is modeling the score brilliant? Because the score *does not depend on the normalizing constant*. Write any density as `$p(x) = \tilde{p}(x) / Z$` where `$\tilde{p}$` is unnormalized and `$Z = \int \tilde{p}$` is the (usually intractable) partition function. Then

$$\nabla_x \log p(x) = \nabla_x \log \tilde{p}(x) - \underbrace{\nabla_x \log Z}_{=\,0},$$

because `$Z$` is a constant in `$x$`. **The score is invariant to `$Z$`.** This is the single most important reason diffusion and energy-based models exist: they get to ignore the partition function that makes explicit likelihood so painful in high dimensions. Once you have the score, you sample by following it — **Langevin dynamics**:

$$x_{t+1} = x_t + \tfrac{\epsilon}{2}\, s_\theta(x_t) + \sqrt{\epsilon}\,\eta_t, \qquad \eta_t \sim \mathcal{N}(0, I),$$

a noisy gradient ascent on log-density that, run long enough, produces samples from `$p$`. Diffusion models are the scaled-up, multi-noise-level version of this idea (they learn the score at *every* noise level), which we derive in [score-based models and the SDE view](/blog/machine-learning/image-generation/diffusion-from-first-principles). Likelihood is available but indirect (via the probability-flow ODE); sampling is iterative; quality and coverage are both strong. That combination is why diffusion currently dominates.

There is a subtlety worth naming, because it recurs across the series: the score `$\nabla_x \log p(x)$` is exactly the quantity that appears in the **reverse-time SDE** that turns noise back into data. When you run the forward noising process (add Gaussian noise gradually), there is a matching reverse process that *undoes* it, and the only unknown in that reverse process is the score of the noisy data at each time. So "learn the score at every noise level" is not an arbitrary objective — it is precisely what you need to reverse the corruption. This is the bridge between the abstract "model the gradient of log-density" and the concrete "train a network to predict the noise," and it is why score matching, denoising, and noise-prediction all turn out to be the same thing up to a reweighting. We make that equivalence rigorous later; here, just register that the score is the load-bearing object.

### 3.4 A fourth lens: exact likelihood by change of variables

Before the trade-off table, one more piece of machinery that makes the explicit family's promise concrete — the **change-of-variables formula**, the engine of normalizing flows and the cleanest example of "exact likelihood in high dimensions is possible if you constrain the model." Suppose you build an *invertible* map `$f_\theta : \mathbb{R}^D \to \mathbb{R}^D$` that transforms a simple base density `$p_z(z) = \mathcal{N}(0, I)$` into your data density. If `$x = f_\theta(z)$` and `$f_\theta$` is a smooth bijection, then probability mass is conserved under the transformation, and the density of `$x$` follows exactly:

$$\log p_\theta(x) = \log p_z\big(f_\theta^{-1}(x)\big) + \log \left| \det \frac{\partial f_\theta^{-1}(x)}{\partial x} \right|.$$

The first term says "how likely is the latent that maps to this `$x$`," and the second — the log-absolute-determinant of the **Jacobian** of the inverse map — is the volume-correction factor that accounts for how `$f_\theta$` stretches and squeezes space. Where the map expands a region, density thins; where it contracts, density concentrates; the determinant tracks exactly that. This is an *exact* likelihood — no bound, no approximation — which is why flows can report a true `$\log p_\theta(x)$`.

The catch, and it is the whole story of why flows did not win, is the determinant. A general `$D \times D$` Jacobian determinant costs `$\mathcal{O}(D^3)$` to compute — hopeless at `$D \approx 10^6$`. So flows constrain `$f_\theta$` to architectures with **cheap, triangular Jacobians** (coupling layers in RealNVP/Glow split the input, transform half conditioned on the other half, and produce a triangular Jacobian whose determinant is just the product of diagonal entries — `$\mathcal{O}(D)$`). That constraint buys tractability but caps expressiveness: every layer must be invertible and volume-trackable, which is a real straitjacket compared to an unconstrained U-Net. The lesson generalizes — *exact likelihood in high dimensions is achievable only by constraining the model*, and the constraint always costs you expressiveness somewhere. The full treatment is in [normalizing flows and change of variables](/blog/machine-learning/image-generation/normalizing-flows-and-change-of-variables); the reason it matters here is that flow matching (the technique behind SD3 and FLUX) is the continuous-time descendant of exactly this idea, with the determinant replaced by an integral that never needs to be computed.

### 3.5 The trade-off, side by side

| Family | Models | Likelihood? | Sampling | Coverage | Fidelity | Stability |
|---|---|---|---|---|---|---|
| Autoregressive | `$p(x_j\mid x_{<j})$` exact | Exact | Sequential, slow | Strong | Strong | Stable |
| Normalizing flow | invertible `$f_\theta$` | Exact | 1 pass | Strong | Moderate | Stable |
| VAE | latent + ELBO | Lower bound | 1 pass | Strong | Blurry | Stable |
| GAN (implicit) | sampler `$G_\theta$` only | **None** | 1 pass, fast | **Weak (mode drop)** | **Razor-sharp** | **Fragile** |
| Diffusion (score) | `$\nabla_x\log p$` | Via ODE | Iterative, slow | Strong | Strong | Stable |

Read this table as the generative trilemma made explicit. GANs buy one-step speed and sharpness by sacrificing coverage and stability. VAEs and flows buy exact/bounded likelihood and stability by sacrificing sharpness or expressiveness. Diffusion buys quality *and* coverage *and* stability by sacrificing **sampling speed** — and the entire Track E of this series ([why diffusion is slow and how to fix it](/blog/machine-learning/image-generation/building-an-image-generation-stack)) is about buying that speed back through distillation and few-step samplers. There is no free lunch; there is only which corner you pay for.

## 4. KL divergence: the ruler for "close to p_data"

We keep saying we want `$p_\theta$` "close to" `$p_\text{data}$". Close in what sense? The default ruler is the **Kullback–Leibler divergence**. For two densities `$p$` and `$q$`,

$$D_{\mathrm{KL}}(p \,\|\, q) = \mathbb{E}_{x \sim p}\!\left[\log \frac{p(x)}{q(x)}\right] = \int p(x) \log \frac{p(x)}{q(x)}\, dx.$$

Three facts make KL the natural choice and also flag its quirks.

**Fact 1 — it is the loss of maximum likelihood.** Minimizing `$D_{\mathrm{KL}}(p_\text{data} \,\|\, p_\theta)$` over `$\theta$` is *identical* to maximizing the expected log-likelihood. Expand:

$$D_{\mathrm{KL}}(p_\text{data}\|p_\theta) = \underbrace{\mathbb{E}_{p_\text{data}}[\log p_\text{data}(x)]}_{\text{const in }\theta} - \mathbb{E}_{p_\text{data}}[\log p_\theta(x)].$$

The first term is the negative entropy of the data, fixed. So `$\min_\theta D_{\mathrm{KL}}(p_\text{data}\|p_\theta) = \max_\theta \mathbb{E}_{p_\text{data}}[\log p_\theta(x)]$`, and the empirical version `$\frac{1}{N}\sum_i \log p_\theta(x_i)$` is exactly what explicit-likelihood models optimize. **Maximum likelihood is KL minimization.** That is why this divergence, and not some other, sits under the whole explicit family.

**Fact 2 — it is asymmetric, and the direction matters.** `$D_{\mathrm{KL}}(p\|q) \ne D_{\mathrm{KL}}(q\|p)$`, and the two directions induce different failure modes:

- **Forward KL** `$D_{\mathrm{KL}}(p_\text{data}\|p_\theta)$` (what MLE uses) is **mass-covering / mean-seeking**. The `$\log p_\theta$` term is multiplied by `$p_\text{data}$`, so the loss explodes wherever `$p_\text{data}$` has mass but `$p_\theta$` does not. The model is *forced to cover every mode* of the data — even if that means smearing mass into low-density valleys between modes. This is *why* MLE models (VAEs especially) produce blurry, "averaged" samples: they would rather hedge across modes than miss one.
- **Reverse KL** `$D_{\mathrm{KL}}(p_\theta\|p_\text{data})$` is **mode-seeking / zero-forcing**. It penalizes putting mass where the data has none, so the model is happy to **lock onto a single sharp mode** and ignore the rest. GAN-flavored objectives lean this way — sharp samples, dropped modes.

This single asymmetry explains a huge amount of generative-model behavior: the blur of likelihood models and the mode collapse of adversarial ones are the *same coin*, seen from opposite KL directions.

**Fact 3 — it is non-negative and zero iff equal.** `$D_{\mathrm{KL}}(p\|q) \ge 0$` with equality iff `$p = q$` almost everywhere. This is Gibbs' inequality, and it is what makes KL a valid "distance-like" quantity (though not a metric — it violates symmetry and the triangle inequality). The non-negativity is the lever that turns the intractable log-likelihood into the optimizable ELBO, which is the next section.

A small but useful relative: the **Jensen–Shannon divergence**, `$D_{\mathrm{JS}}(p\|q) = \frac{1}{2}D_{\mathrm{KL}}(p\|m) + \frac{1}{2}D_{\mathrm{KL}}(q\|m)$` with `$m = \frac{1}{2}(p+q)$`, is symmetric and bounded, and it is the divergence the original GAN minimizes at its optimum. We will not need JS again, but knowing GANs minimize JS while VAEs minimize a forward-KL bound is a clean way to remember why their samples look so different.

## 5. The ELBO, derived from scratch

This is the most important derivation in the post, because **every later post in the series uses the ELBO** — the VAE optimizes it directly, diffusion's training loss is an ELBO in disguise, and even flow matching can be read as tightening a bound. We will derive it twice, two ways, and land on the form **ELBO = reconstruction − KL** that you should be able to reproduce on a whiteboard.

The setup: a latent-variable model. We posit a latent `$z$` with a simple prior `$p(z) = \mathcal{N}(0, I)$`, a decoder `$p_\theta(x \mid z)$` (a neural net that turns latents into images), and we want to maximize the data log-likelihood

$$\log p_\theta(x) = \log \int p_\theta(x \mid z)\, p(z)\, dz.$$

That integral is intractable — it averages over all possible latents, a high-dimensional integral with no closed form. The trick is to introduce an **approximate posterior** (the *encoder*) `$q_\phi(z \mid x)$`, a neural net that guesses which latents are plausible for a given `$x$`.

**Derivation 1 — via Jensen's inequality.** Multiply and divide by `$q_\phi$` inside the integral, then apply the concavity of `$\log$` (Jensen):

$$\log p_\theta(x) = \log \int q_\phi(z\mid x)\, \frac{p_\theta(x\mid z)\,p(z)}{q_\phi(z\mid x)}\, dz = \log \mathbb{E}_{q_\phi}\!\left[\frac{p_\theta(x\mid z)\,p(z)}{q_\phi(z\mid x)}\right].$$

Since `$\log$` is concave, `$\log \mathbb{E}[\cdot] \ge \mathbb{E}[\log \cdot]$`, so

$$\log p_\theta(x) \;\ge\; \mathbb{E}_{q_\phi}\!\left[\log \frac{p_\theta(x\mid z)\,p(z)}{q_\phi(z\mid x)}\right] \;\equiv\; \mathcal{L}(\theta, \phi; x).$$

That right-hand side `$\mathcal{L}$` is the **Evidence Lower BOund** — a lower bound on the log-evidence `$\log p_\theta(x)$`. We maximize it as a tractable proxy for the intractable likelihood.

**Derivation 2 — via KL, which reveals the gap.** Start from an exact identity. For any `$q_\phi(z\mid x)$`,

$$\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi}\!\left[\log \frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]}_{\mathcal{L}(\theta,\phi;x)} + \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\big)}_{\ge\, 0}.$$

You can verify this in one line: expand `$p_\theta(x,z) = p_\theta(z\mid x)\,p_\theta(x)$` inside the first term, the `$\log p_\theta(x)$` pulls out of the expectation, and the leftover is exactly the negative KL. Since the KL term is `$\ge 0$` (Fact 3 from §4), `$\mathcal{L} \le \log p_\theta(x)$` — same bound — and now we can *see the gap*: **the bound is tight exactly when the approximate posterior `$q_\phi(z\mid x)$` equals the true posterior `$p_\theta(z\mid x)$`.** Maximizing the ELBO does double duty: it pushes up `$\log p_\theta(x)$` *and* drives `$q_\phi$` toward the true posterior.

**The form you will actually use.** Now split the ELBO. Expand the joint `$p_\theta(x,z) = p_\theta(x\mid z)\,p(z)$`:

$$\mathcal{L} = \mathbb{E}_{q_\phi}\!\left[\log \frac{p_\theta(x\mid z)\,p(z)}{q_\phi(z\mid x)}\right] = \underbrace{\mathbb{E}_{q_\phi(z\mid x)}\big[\log p_\theta(x \mid z)\big]}_{\text{reconstruction}} \;-\; \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{KL to prior}}.$$

There it is — **ELBO = reconstruction − KL.** Figure 3 lays out the two terms and the gap.

![A stacked decomposition of the evidence lower bound into a reconstruction reward minus a KL penalty against the prior](/imgs/blogs/the-mathematics-of-image-distributions-3.png)

Read the two terms physically:

- **Reconstruction term** `$\mathbb{E}_{q_\phi}[\log p_\theta(x\mid z)]$`: encode `$x$` to a latent `$z$`, decode it back, and reward the model for reconstructing `$x$` faithfully. For a Gaussian decoder this is just (negative) squared error in pixel/feature space; for a Bernoulli decoder it is cross-entropy. *This pulls the model toward faithful detail.*
- **KL term** `$D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p(z))$`: a regularizer that pulls the encoder's posterior toward the prior `$\mathcal{N}(0,I)$`. It keeps the latent space *organized and samplable* — without it, the encoder could scatter latents anywhere and you could not draw `$z \sim p(z)$` at generation time and expect a valid image. *This pulls the model toward a clean, generative latent space.*

The tension between these two terms is the entire drama of the VAE — too much weight on the KL and you get **posterior collapse** (the model ignores `$z$` and the latent carries no information); too little and the latent space is not samplable. The [VAE post](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) lives inside this trade-off. And when we get to diffusion, the DDPM training objective will turn out to be a *sum of ELBO terms*, one per noise level, which collapses (after the famous reweighting) into the simple `$\epsilon$`-prediction loss. So derive this once, carefully, and you have the scaffolding for the whole series.

#### Worked example: the closed-form Gaussian KL term

The KL term is not just a symbol — for the standard VAE choice it has a closed form, which is *why* the ELBO is cheap to optimize. Let the encoder output a diagonal Gaussian `$q_\phi(z\mid x) = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$` and the prior be `$p(z) = \mathcal{N}(0, I)$`, both in `$\mathbb{R}^d$`. Then

$$D_{\mathrm{KL}}\big(\mathcal{N}(\mu,\mathrm{diag}(\sigma^2)) \,\|\, \mathcal{N}(0,I)\big) = \tfrac{1}{2}\sum_{j=1}^{d}\big(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\big).$$

Plug in numbers. Suppose `$d = 4$` and the encoder outputs `$\mu = (0.5, -0.5, 0, 1)$` and `$\sigma^2 = (1, 1, 0.25, 4)$`. Term by term: `$j=1$`: `$0.25 + 1 - 0 - 1 = 0.25$`. `$j=2$`: `$0.25 + 1 - 0 - 1 = 0.25$`. `$j=3$`: `$0 + 0.25 - \log 0.25 - 1 = 0.25 + 1.386 - 1 = 0.636$`. `$j=4$`: `$1 + 4 - \log 4 - 1 = 4 - 1.386 = 2.614$`. Sum `$= 3.75$`, times `$\tfrac{1}{2}$` gives `$D_{\mathrm{KL}} = 1.875$` nats. The big contributor is dimension 4, where `$\sigma^2 = 4$` is far from the prior's unit variance — the KL penalizes both over-confident (`$\sigma^2 \ll 1$`) and over-diffuse (`$\sigma^2 \gg 1$`) posteriors, nudging every latent dimension toward a clean unit Gaussian. This is the term you literally compute, in closed form, on every VAE training step.

## 6. The curse of dimensionality, and why naive density estimation fails

We have a target `$p_\text{data}(x)$` and three families of models. A reasonable question: why not just estimate the density the classical way — a histogram, or a kernel density estimator — and be done? The answer is the **curse of dimensionality**, and seeing exactly *how* it kills each classical method tells you why we need deep generative models at all. Figure 7 is the scorecard across methods and dimensions.

![A matrix showing how histograms, kernel density, nearest neighbors, and manifold models each break down as dimension grows from two to nearly two hundred thousand](/imgs/blogs/the-mathematics-of-image-distributions-7.png)

**Histograms.** Partition each axis into `$b$` bins; the joint histogram has `$b^D$` cells. At `$D = 2$` and `$b = 100$` that is 10,000 cells — fine. At `$D = 786{,}432$`, even `$b = 2$` (a single bit per dimension) gives `$2^{786432}$` cells. You would need more samples than there are atoms in the universe to put even one sample in a meaningful fraction of cells. The histogram is **empty everywhere** — every test image falls in a cell that no training image touched, so its estimated density is zero. Dead on arrival.

**Kernel density estimation.** KDE places a smooth kernel (a little Gaussian bump) on each training point and sums them: `$\hat{p}(x) = \frac{1}{N}\sum_i K_h(x - x_i)$`. In low dimensions this works beautifully. In high dimensions it dies because of a more subtle effect — **distance concentration**. Here is the precise statement: for i.i.d. data in `$\mathbb{R}^D$` under mild conditions, the ratio of the spread of pairwise distances to their mean shrinks as `$D$` grows:

$$\frac{\mathrm{Std}[\,\|x_i - x_j\|\,]}{\mathbb{E}[\,\|x_i - x_j\|\,]} \;\xrightarrow[D \to \infty]{}\; 0.$$

In words: in high dimensions, **all pairs of points are nearly the same distance apart**. The notion of "nearby points" — which KDE relies on to share probability mass — evaporates. Every training point looks equidistant from your test point, so the kernel bandwidth `$h$` either captures everything (uniform, useless) or nothing (spiky, zero between points). There is no good `$h$`. KDE degenerates.

To feel the effect with a number, take points drawn uniformly from the unit hypercube `$[0,1]^D$`. The expected squared distance between two random points is `$\mathbb{E}\|x_i - x_j\|^2 = D \cdot \mathbb{E}[(u-v)^2] = D/6$` (each coordinate contributes `$\mathbb{E}[(u-v)^2] = 1/6$` for independent uniforms), so the typical distance grows like `$\sqrt{D/6}$`. Meanwhile the *standard deviation* of that distance grows only like `$\sqrt{D}$` times a constant that does not increase, so the *ratio* of spread to typical distance falls like `$1/\sqrt{D}$`. At `$D = 4$` the spread is a sizeable fraction of the typical distance — neighbors are meaningfully nearer than non-neighbors. At `$D = 10{,}000$` the ratio has shrunk by `$\sqrt{2500} = 50\times$`; at the `$D \approx 786{,}432$` of a real image it is utterly negligible. The nearest neighbor and the farthest neighbor of any query point sit at almost the same distance. Every method that relies on "closer points are more similar" — KDE, `$k$`-NN, mean-shift, naive manifold learning without the right inductive bias — inherits this collapse. The only escape is to stop measuring distance in the raw ambient space and instead learn a representation where the manifold is flattened, which is exactly what the encoders inside VAEs, the feature spaces inside FID, and the latent spaces of every modern generator do.

**Nearest neighbors.** The same distance-concentration result wrecks nearest-neighbor density estimation and `$k$`-NN classifiers: when all distances concentrate, the "nearest" neighbor is barely nearer than the farthest, so it carries almost no information. A nearest-neighbor density estimate becomes effectively random.

**The escape hatch — exploit the manifold.** Every classical method assumes the data fills the ambient `$\mathbb{R}^D$`. It does not — it lives on a `$d$`-dimensional sheet with `$d \ll D$` (§2). The methods that *win* are the ones that implicitly or explicitly model that low-dimensional structure: a VAE learns a `$d$`-dimensional latent code, a flow learns a `$d$`-effective invertible map, a diffusion model learns the score *on and near* the manifold. The deep generative model is not "a fancier histogram" — it is a method that **sidesteps the curse by learning the manifold's coordinates** instead of gridding the ambient cube. That is the whole reason the field exists, and it is why every later post is about *parameterizing the manifold cleverly* rather than estimating a raw density.

That closes the modeling half. We can define `$p_\text{data}$`, we know why it lives on a manifold, we know the three ways to model it and what each trades, we can derive the ELBO that several of those models optimize, and we know why the naive methods fail. Now the harder, more practical question: once you have a trained model, **how do you measure whether it is any good** — and which of those measurements are lying to you?

## 7. The evaluation problem: you cannot just look

Why is evaluating a generative image model so hard? Because the thing you actually want to measure — "is `$p_\theta$` close to `$p_\text{data}$`?" — requires comparing two *distributions over a million-dimensional space*, from finite samples, with no access to either density. You cannot compute KL (no densities). You cannot eyeball 50,000 images. And the property you care about is multi-axial: a generator can be **high-fidelity** (every sample is sharp and realistic) but **low-coverage** (it only ever draws golden retrievers), or **high-coverage** but **low-fidelity** (it draws every breed, all blurry), or perfectly realistic and diverse but **ignoring the prompt entirely**. No scalar captures all three. Figure 4 is the map of which metric sees which axis — and, crucially, what each one is *blind* to.

![A matrix mapping FID, Inception Score, CLIP-score, precision-recall, and learned preference metrics to what each captures, ignores, and fails on](/imgs/blogs/the-mathematics-of-image-distributions-4.png)

The honest framing for the rest of this post: **every metric is a lossy projection of "is the distribution right" onto one axis, and every metric can be gamed once you optimize against it.** Knowing the projection and the failure mode is the difference between a number that informs you and a number that fools you. We will take them in order — Inception Score, FID (derived in full), precision/recall, CLIP-score, and learned preference — and for each one say exactly what it computes and exactly where it lies.

## 8. Inception Score: the first try, and its blind spots

The **Inception Score (IS)** was the first widely used automatic metric, and understanding it sets up FID. The idea: run each generated image through a pretrained Inception-V3 classifier, getting a conditional label distribution `$p(y \mid x)$` over the 1000 ImageNet classes. A *good* sample should be **confidently classifiable** — `$p(y\mid x)$` should be peaked (sharp, recognizable object). A *good set* should be **diverse** — the marginal `$p(y) = \mathbb{E}_x[p(y\mid x)]$` should be spread across many classes (the model is not just drawing one thing). IS rewards both at once:

$$\mathrm{IS} = \exp\!\Big( \mathbb{E}_{x \sim p_\theta}\big[\, D_{\mathrm{KL}}\big(p(y\mid x)\,\|\,p(y)\big)\,\big] \Big).$$

The KL inside is large when `$p(y\mid x)$` (peaked) differs a lot from `$p(y)$` (spread) — i.e. each image is decisive *and* the set is varied. Higher IS is better; ImageNet real images score around `$\mathrm{IS} \approx 11$`–`$12$`.

Now the blind spots, and they are severe:

1. **No reference set.** IS never compares to the real data distribution. It only checks "are images classifiable and varied?" A model that produces 1000 perfect, distinct ImageNet exemplars — one flawless image per class, and *only* those — scores a near-perfect IS while having catastrophically collapsed mode coverage *within* each class. **IS cannot see intra-class mode collapse.**
2. **It only knows ImageNet's 1000 classes.** Generate a beautiful image of something not in ImageNet — a specific cartoon character, a UI mockup — and Inception's `$p(y\mid x)$` is garbage, so IS is meaningless. It does not generalize beyond the classifier's label set.
3. **It is trivially gameable.** Optimize directly for IS and you get adversarial high-confidence images that fool Inception but look wrong to humans.

IS measured a real thing in 2016, but it compares your samples to a *classifier's expectations*, not to the *real data*. That is the gap FID closes.

## 9. FID: the Fréchet distance between two Gaussians, derived

The **Fréchet Inception Distance** is the field's default metric, and you should know exactly what it computes — both because you will report it constantly and because its specific construction is the source of all its biases. Figure 6 is the pipeline; let us derive the formula.

![A dataflow graph showing real and generated images passed through Inception, summarized as two Gaussians, then compared by Frechet distance](/imgs/blogs/the-mathematics-of-image-distributions-6.png)

**Step 1 — features, not pixels.** Pass every real image and every generated image through Inception-V3 and read off the 2048-dimensional activation of the final average-pooling layer (`pool3`). This maps each image to a feature vector. Recall from §2 that pixel distance is meaningless on the curved manifold; feature space approximately flattens it, so Euclidean distance there tracks perceptual similarity. Call the real features `$\{f_i^r\}$` and generated features `$\{f_j^g\}$`.

**Step 2 — fit a Gaussian to each set.** Model each feature cloud as a multivariate Gaussian: compute the sample mean and covariance of the real features `$(\mu_r, \Sigma_r)$` and of the generated features `$(\mu_g, \Sigma_g)$`. This is a *modeling assumption* — that the 2048-dim feature distributions are approximately Gaussian — and it is a big part of why FID behaves the way it does.

**Step 3 — Fréchet distance between the two Gaussians.** The **Fréchet distance** (a.k.a. the 2-Wasserstein distance) between two distributions `$P$` and `$Q$` is `$W_2(P,Q)^2 = \inf_\gamma \mathbb{E}_{(u,v)\sim\gamma}\|u-v\|^2$` over all couplings `$\gamma$` with marginals `$P, Q$`. For two Gaussians `$\mathcal{N}(\mu_r,\Sigma_r)$` and `$\mathcal{N}(\mu_g,\Sigma_g)$`, this infimum has a closed form, and that closed form *is* FID:

$$\boxed{\;\mathrm{FID} = \lVert \mu_r - \mu_g \rVert^2 + \mathrm{Tr}\!\left(\Sigma_r + \Sigma_g - 2\big(\Sigma_r \Sigma_g\big)^{1/2}\right)\;}$$

Read the two terms. The first, `$\lVert \mu_r - \mu_g \rVert^2$`, is the squared distance between the **mean feature** of real vs generated images — it catches gross shifts ("all your images are too dark / too saturated / the wrong style"). The second, the trace term, compares the **covariance structure** — the spread and correlation of features. The matrix square root `$(\Sigma_r \Sigma_g)^{1/2}$` is the geometric-mean coupling that makes the trace term zero exactly when `$\Sigma_r = \Sigma_g$`. When both means and both covariances match, FID `$= 0$`; the further apart the two Gaussians, the larger FID. **Lower is better.** State-of-the-art text-to-image models report FID in the low single digits on standard benchmarks (e.g. roughly `$\mathrm{FID} \approx 2$`–`$9$` on MS-COCO `$256\times256$`, 30k samples, depending on model and protocol); a weak or broken model can be in the tens or hundreds.

Here is the derivation sketch for the closed form, so it is not a black box. For two Gaussians, the optimal coupling is itself jointly Gaussian, and the `$W_2^2$` between Gaussians decomposes additively into a mean part and a covariance part. The mean part is just `$\|\mu_r-\mu_g\|^2$`. The covariance part is the squared **Bures distance** between `$\Sigma_r$` and `$\Sigma_g$`, `$\mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2})^{1/2})$`, which (using the trace's cyclic property and that the relevant matrices are PSD) equals the symmetric-looking `$\mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$` written above. The key takeaway: **FID is `$W_2^2$` under a Gaussian approximation of Inception features.** Every word of that sentence is a knob you can turn — and a place FID can mislead you.

#### Worked example: FID by hand in two dimensions

Numbers make the formula stop being a black box. Forget 2048 dimensions; take a `$2$`-D toy "feature space" so we can compute FID with arithmetic. Suppose your real features have mean `$\mu_r = (0, 0)$` and covariance `$\Sigma_r = I$` (the `$2\times 2$` identity), and your generated features have mean `$\mu_g = (0.3, 0.4)$` and covariance `$\Sigma_g = 4I$` (the generator's features are shifted *and* four times as spread out — a plausible "mode-collapse plus drift" signature). Compute the two terms.

The **mean term**: `$\lVert \mu_r - \mu_g \rVert^2 = 0.3^2 + 0.4^2 = 0.09 + 0.16 = 0.25$`. The **covariance term**: because both covariances are diagonal (in fact scalar multiples of the identity), the matrix square root is easy — `$(\Sigma_r \Sigma_g)^{1/2} = (4 I)^{1/2} = 2 I$`. So `$\mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}) = \mathrm{Tr}(I + 4I - 2\cdot 2I) = \mathrm{Tr}(I + 4I - 4I) = \mathrm{Tr}(I) = 2$`. Total: `$\mathrm{FID} = 0.25 + 2 = 2.25$`. Now notice what the decomposition tells you: of the 2.25, only `$0.25$` came from the *mean shift* (the drift) and a full `$2.0$` came from the *covariance mismatch* (the over-spread). If you only looked at the FID number you would not know which problem dominated — but the two terms, computed separately, point straight at "your generator's feature variance is wrong," not "your generator's average image is off." In 2048-D you cannot eyeball this, but the structure is identical: FID is a sum of a drift penalty and a spread/correlation penalty, and when FID is high it is worth asking *which term* is responsible, because the fixes differ. This is also why a generator that produces *too little* diversity (collapsed, `$\Sigma_g \ll \Sigma_r$`) is penalized by the same trace term as one that produces too much — FID measures *mismatch* in both directions.

### 9.1 Computing FID for real, with torchmetrics

Here is the actual code. `torchmetrics` wraps the Inception backbone, the running mean/covariance accumulation, and the matrix square root, so you feed it batches of `uint8` images and read off the number.

```python
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

# feature=2048 selects the pool3 layer; normalize=False expects uint8 in [0,255]
fid = FrechetInceptionDistance(feature=2048, normalize=False).to("cuda")

# real_loader and gen_loader yield uint8 image batches of shape (B, 3, H, W)
for real_batch in real_loader:                       # e.g. 50k real images
    fid.update(real_batch.to("cuda"), real=True)

for gen_batch in gen_loader:                         # e.g. 50k generated images
    fid.update(gen_batch.to("cuda"), real=False)

score = fid.compute()
print(f"FID = {score.item():.2f}")                   # lower is better
```

Two correctness traps that quietly corrupt reported FID, both worth burning into memory:

```python
# TRAP 1: dtype/range. If you pass float images, set normalize=True and feed [0,1].
# Mismatching the range silently shifts every feature and poisons the score.
fid_float = FrechetInceptionDistance(feature=2048, normalize=True).to("cuda")
# now feed float tensors in [0, 1]

# TRAP 2: resizing. FID is sensitive to the resize/interpolation used to reach
# Inception's 299x299 input. Use the SAME resize for real and generated sets,
# or you are measuring your resampler, not your model.
```

### 9.2 Where FID lies — the biases you must know

FID is the best general-purpose metric we have *and* it is biased in at least five ways that change leaderboard rankings. Be honest about all of them.

1. **Sample-size bias (the big one).** FID is a *biased estimator*: it systematically *decreases* as you use more samples, because the Gaussian fit gets tighter. The standard is **50,000 generated samples** against a fixed reference set. Reporting FID at 5k vs 50k can shift the number by several points — enough to flip a ranking. **Always state the sample count.** A "FID 4.1" with no `$N$` is not a comparable number.
2. **The Gaussian assumption.** Inception features are *not* actually Gaussian. FID only sees the first two moments (mean, covariance); it is **blind to any difference in the third moment and beyond**. Two distributions with identical mean and covariance but wildly different higher-order structure get FID `$= 0$`. A model can match the Gaussian summary while producing subtly wrong images.
3. **The Inception backbone.** FID inherits all of ImageNet-Inception's blind spots and texture biases. Inception was trained for 2015-era object classification; it is weirdly sensitive to texture and weirdly insensitive to some structural errors humans notice instantly. This is why **FID-DINOv2** was proposed (Stein et al., 2023): swap the frozen Inception backbone for a self-supervised DINOv2 backbone, and the resulting distance correlates *much* better with human judgments and is less gameable. If you can, report both FID and FID-DINOv2.
4. **It cannot read prompts.** This is the one from the intro. FID compares *image distributions*. It has no idea what caption produced each image. **Two text-to-image models can tie on FID while one follows prompts and the other ignores them.** FID is necessary but radically insufficient for text-to-image. Figure 5 is exactly this trap.
5. **It conflates fidelity and coverage.** A single FID number cannot tell you *whether* a high score comes from blurry-but-diverse or sharp-but-collapsed samples. Two very different failure modes can produce the same FID. To separate them you need precision and recall (§11).

![A before-and-after comparison where two image models tie on FID yet one mis-renders the prompt and the other follows it, scoring higher on CLIP and human preference](/imgs/blogs/the-mathematics-of-image-distributions-5.png)

#### Worked example: two models tie on FID, split on everything else

Concrete scenario, the one from the intro, with numbers. You evaluate two text-to-image checkpoints on 30k MS-COCO prompts at `$256\times256$`, 50k-sample protocol. Model A: `$\mathrm{FID} = 8.1$`. Model B: `$\mathrm{FID} = 8.0$`. By FID they are indistinguishable — a 0.1 gap is within run-to-run noise. Now compute **CLIP-score** (§10) on the same prompt-image pairs: Model A scores `$0.26$`, Model B scores `$0.32$` — a 6-point gap, far outside noise. You inspect the compositional prompts ("red cube to the left of a blue sphere") and find Model A systematically botches counting and spatial relations while producing gorgeous individual objects; Model B gets the layout right. Finally you run **HPSv2** human-preference scoring: B wins ~68% of head-to-head comparisons. The lesson is not "FID is useless" — FID correctly certified both models produce realistic-looking images. The lesson is that **FID measured the axis where they agreed and was blind to the axis where they differed.** You needed CLIP-score and a preference model to see the real gap. Report all three or you will ship the wrong model.

## 10. CLIP-score: does the image match the prompt?

FID's biggest hole for text-to-image is that it ignores the caption. **CLIP-score** plugs that hole. CLIP (Contrastive Language-Image Pretraining) is a model with an image encoder and a text encoder trained so that matching image–text pairs have high cosine similarity in a shared embedding space. CLIP-score for a generated image `$x$` and its prompt `$c$` is simply that similarity:

$$\mathrm{CLIPScore}(x, c) = \max\!\big(0,\; w \cdot \cos\!\big(E_\text{img}(x),\, E_\text{txt}(c)\big)\big),$$

with `$w$` a fixed scaling constant (commonly `$2.5$` in the original formulation; many practitioners just report the raw cosine in `$[-1, 1]$`, typically landing around `$0.25$`–`$0.35$` for good text-to-image alignment). Higher means the image better matches the prompt. It measures the axis FID is blind to — **prompt alignment** — and that is exactly why you report the two *together*.

```python
import torch, open_clip
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.eval().to("cuda")

def clip_score(image_path: str, prompt: str) -> float:
    image = preprocess(Image.open(image_path)).unsqueeze(0).to("cuda")
    text = tokenizer([prompt]).to("cuda")
    with torch.no_grad():
        img_emb = model.encode_image(image)
        txt_emb = model.encode_text(text)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        cosine = (img_emb * txt_emb).sum(dim=-1)          # in [-1, 1]
    return cosine.item()

print(clip_score("gen_0001.png", "a red cube to the left of a blue sphere"))
```

CLIP-score's own blind spots, because every metric has them:

- **It is coarse.** CLIP embeddings capture overall semantic gist, not fine compositional structure. "Red cube left of blue sphere" and "blue cube left of red sphere" can score nearly identically — CLIP is famously weak at **attribute binding** and **spatial relations** and **counting**. So a high CLIP-score does *not* certify correct composition; for that you need a structured metric like GenEval or T2I-CompBench that decomposes the prompt and checks each requirement.
- **It says nothing about realism.** A blurry, artifacted image of the right content can outscore a photoreal image of slightly-wrong content. CLIP-score and FID are genuinely orthogonal — you need both.
- **It saturates and can be hacked.** Optimize images directly against CLIP (as some guidance schemes do) and you get high CLIP-score images that look subtly adversarial. And above a certain alignment level CLIP-score plateaus — it cannot distinguish "good" from "great" alignment.

The practical rule: **report FID and CLIP-score as a pair**, and plot them against each other. Raising classifier-free guidance, for example, typically *increases* CLIP-score (better prompt adherence) while *worsening* FID past a point (over-saturated, less diverse) — the FID–CLIP curve *is* the fidelity-vs-alignment trade-off, and finding the knee of that curve is how you actually set the guidance scale. We derive the guidance mechanism itself in [classifier-free guidance](/blog/machine-learning/image-generation/diffusion-from-first-principles); here, just know the two metrics together draw the trade-off curve.

## 11. Precision and recall: splitting fidelity from coverage

FID gives you one number that mashes fidelity and coverage together. **Improved precision and recall** (Kynkäänniemi et al., 2019) separate them, and the separation is exactly the distinction the manifold picture predicted. The idea: estimate the *support* of the real feature distribution and the generated feature distribution (using the same Inception/feature embedding), then ask two questions.

- **Precision** = the fraction of *generated* samples that fall inside the *real* manifold's support. High precision means "my samples look real" — **fidelity**. Low precision means the model produces off-manifold junk.
- **Recall** = the fraction of *real* samples that fall inside the *generated* manifold's support. High recall means "my model can produce the full variety of real images" — **coverage**. Low recall means **mode collapse** — the model only covers part of the data.

The support is estimated nonparametrically: build a `$k$`-nearest-neighbor hypersphere around each point in feature space, and a query point is "inside the support" if it lands within any of those spheres. Formally, with feature sets `$\Phi_r$` (real) and `$\Phi_g$` (generated),

$$\mathrm{precision} = \frac{1}{|\Phi_g|}\sum_{f \in \Phi_g} \mathbf{1}\big[f \in \mathrm{manifold}(\Phi_r)\big], \qquad \mathrm{recall} = \frac{1}{|\Phi_r|}\sum_{f \in \Phi_r} \mathbf{1}\big[f \in \mathrm{manifold}(\Phi_g)\big].$$

This is the single most useful diagnostic when FID disappoints, because **it tells you *which way* you failed**. High precision + low recall = sharp but collapsed (the GAN signature; raise diversity, lower guidance). Low precision + high recall = diverse but blurry/artifacted (the under-trained VAE signature; improve fidelity). FID alone cannot distinguish these, and they demand opposite fixes. When someone reports only FID, your first question should be "what are the precision and recall?" — the answer often reverses the conclusion.

A modern refinement, **density and coverage** (Naeem et al., 2020), fixes the original precision/recall's sensitivity to outliers (a single weird real sample can inflate the support hypersphere and let junk count as "precise"). If you are reporting these, prefer density/coverage for robustness — but the conceptual split, **fidelity vs coverage**, is the durable idea.

One more reason this split is worth the extra compute: it makes the *direction* of progress legible across a training run. Early in training a diffusion model, you typically see precision climb fast (samples start looking real) while recall lags (the model has not yet learned the full variety) — so FID drops mostly because fidelity improved. Later, recall catches up as coverage fills in, and FID's remaining gains come from diversity. If you only watch FID you see a single number falling and cannot tell *which* axis is improving on any given epoch; with precision and recall plotted separately, a stalled recall curve immediately tells you "the model is sharp but not diverse — it may be memorizing or mode-collapsing," which is an entirely different intervention (more data, stronger augmentation, lower guidance) than "the model is diverse but blurry" (train longer, improve the decoder/VAE, raise capacity). The metric you choose decides whether you can even *see* the problem you need to fix.

## 12. Learned preference metrics: regressing on what humans actually like

FID, IS, CLIP-score, and precision/recall are all *proxies*. The thing we ultimately care about — for text-to-image especially — is **which images humans prefer**. The 2023–2024 wave of metrics tackles this head-on: collect large datasets of human pairwise preferences over generated images, then train a model to predict them. These **learned preference metrics** are now the most human-correlated automatic scores available.

- **ImageReward** (Xu et al., 2023): a reward model fine-tuned on ~137k expert comparisons, trained to score (prompt, image) pairs by predicted human preference. It correlates with human ranking substantially better than CLIP-score or aesthetic predictors, and it doubles as a *reward signal* for fine-tuning (RLHF-style alignment of diffusion models).
- **HPSv2** (Human Preference Score v2, Wu et al., 2023): a CLIP-based preference predictor trained on a large, cleaned human-preference dataset (HPD v2) across multiple models and styles. Reported as a percentage-style preference score; widely used to rank text-to-image models on prompt sets.
- **PickScore** (Kirstain et al., 2023): trained on Pick-a-Pic, a dataset of *in-the-wild* user preferences (real users picking between real generations), making it especially well-aligned with what everyday users like.

How are these trained? The recipe is the **Bradley–Terry preference model**, the same one behind RLHF for language models. Show annotators two images for the same prompt and record which they prefer; model the probability that image `$i$` beats image `$j$` as `$P(i \succ j) = \sigma\big(r_\psi(x_i, c) - r_\psi(x_j, c)\big)$`, where `$r_\psi$` is the learned reward (a CLIP-style network with a scalar head) and `$\sigma$` is the logistic sigmoid; then maximize the log-likelihood of the observed human choices. The result is a scalar `$r_\psi(x, c)$` that ranks images by predicted preference. Because it is trained on *real human comparisons* rather than a hand-designed formula, it captures things no closed-form metric can — aesthetic quality, prompt fidelity, the absence of the subtle artifacts humans flag instantly — which is why these scores now top the correlation-with-humans tables.

These are the strongest available proxies for human judgment — and they have their *own* failure mode, which is the most dangerous one in the whole post: **reward hacking**. The moment you optimize *against* a learned reward model (fine-tuning a diffusion model to maximize ImageReward or PickScore), the generator learns to exploit the reward model's quirks — over-saturated colors, a particular "AI aesthetic," high-contrast skies, repeated compositional tropes the reward model happens to score highly — producing images that win the metric but drift from genuine human preference. The mechanism is precise: the reward model `$r_\psi$` was fit on a *fixed distribution* of images (the ones the annotators saw). Fine-tuning pushes the generator's output distribution *off* that training distribution, into a region where `$r_\psi$` was never validated and its predictions are unreliable — and gradient ascent will happily march straight into that region because that is where the (now-spurious) reward is highest. You end up maximizing the reward model's *extrapolation error*, not human preference. This is Goodhart's law in its sharpest form: **a measure optimized hard enough stops being a good measure.** The practical defenses are the same as in language-model RLHF: add a KL penalty that keeps the fine-tuned model close to the base model (so it cannot wander too far off-distribution), ensemble multiple reward models, stop early, and — non-negotiably — keep a fresh human-eval holdout that the reward model never saw. Use these metrics to *evaluate* freely; use them to *train* with a hand on the brake. Figure 8 traces how the whole metric lineage evolved, each one patching the previous one's blind spot.

![A timeline of image-generation metrics from Inception Score through FID, precision-recall, CLIP-score, learned preference, and FID-DINOv2, each fixing a prior blind spot](/imgs/blogs/the-mathematics-of-image-distributions-8.png)

## 13. Case studies: real numbers, honestly read

Four concrete reads from the literature and shipped models, with the caveat that exact figures depend on protocol (sample count, reference set, resolution, guidance) — so treat these as order-of-magnitude anchors and always check the source paper's exact setup.

**Case 1 — SDXL vs SD 1.5: more params, and why FID barely moves.** SD 1.5 has a ~860M-parameter U-Net; SDXL has a ~2.6B-parameter U-Net plus a second text encoder and a refiner. SDXL produces visibly better images — sharper, better composition, better text adherence. Yet on a raw MS-COCO FID comparison the gap is *small and can even invert*, because SD 1.5 was, in effect, tuned toward COCO-style FID while SDXL optimized for human preference and high-resolution aesthetics. The Stability team's own SDXL report leaned on **human preference win-rates**, not FID, to show the improvement — a direct admission that FID had stopped tracking what mattered. The lesson: when a clearly-better model does not show a clearly-better FID, suspect the metric, not the model.

**Case 2 — the FID-DINOv2 correction.** Stein et al. (2023) showed that FID's rankings of modern generative models *disagree with human judgments* in a meaningful fraction of cases, and that swapping Inception for a DINOv2 backbone (FID-DINOv2, a.k.a. FD-DINOv2) restores agreement with human raters. Models that looked tied or mis-ordered under Inception-FID separate correctly under DINOv2-FID. This is the cleanest evidence that **a chunk of "FID differences" between strong modern models is Inception-backbone noise, not real quality difference.** Report FID-DINOv2 alongside classic FID when comparing frontier models.

**Case 3 — guidance scale moves FID and CLIP in opposite directions.** A reproducible result across diffusion models: sweep the classifier-free-guidance scale `$w$` from low to high and FID traces a **U-shape** (best at moderate guidance, e.g. `$w \approx 2$`–`$4$` for many models) while CLIP-score **rises monotonically** (more guidance = more prompt adherence) before saturating. The two metrics literally pull apart. There is no single "best" guidance scale — there is a **fidelity–alignment trade-off curve**, and where you sit on it is a product decision (a stock-photo product wants different guidance than a creative tool). This is the metric-level shadow of the generative trilemma.

**Case 4 — few-step distillation: the metric you report decides the story.** One-step and few-step models (SDXL-Turbo via ADD, DMD2, LCM) trade some FID for enormous speedups. DMD2 reports one-step FID competitive with multi-step teachers on some benchmarks; LCM/Turbo hit usable quality in 1–4 steps versus 25–50 for the teacher. But "competitive FID" can hide a real diversity drop (lower recall) that FID's fidelity-coverage conflation masks. **Report precision/recall, not just FID, when you distill** — the speed win is real, but so is the coverage cost, and only the split metric shows it. We dig into the distillation mechanics in the [speed track and capstone](/blog/machine-learning/image-generation/building-an-image-generation-stack).

## 14. A metric × what-it-captures × failure-mode table

Pull it all together into the one table you should keep open while evaluating any image model. This is the operational summary of the whole second half.

| Metric | What it captures | Reference? | Reads prompt? | Where it lies |
|---|---|---|---|---|
| **Inception Score** | Per-image confidence + class spread | No | No | No real reference; blind to intra-class collapse; ImageNet-only; gameable |
| **FID** | Realism + diversity (Gaussian feature match) | Yes (real set) | No | Sample-size biased; Gaussian-only (1st/2nd moment); Inception-backbone bias; ignores caption; conflates fidelity/coverage |
| **FID-DINOv2** | Same, better backbone | Yes | No | Same structural limits as FID; still ignores caption — but far less backbone noise |
| **CLIP-score** | Prompt–image alignment | No | **Yes** | Coarse — weak on binding/spatial/counting; says nothing about realism; saturates; hackable |
| **Precision** | Fidelity (samples on real manifold) | Yes | No | Sensitive to support outliers; needs feature embedding choice |
| **Recall** | Coverage (real modes reproduced) | Yes | No | Same; low recall = mode collapse |
| **ImageReward / HPSv2 / PickScore** | Predicted human preference | Learned | Yes | **Reward hacking** under optimization; distribution-shift fragile; encodes annotator bias |

The discipline this table enforces: **never report a single number.** A defensible evaluation of a text-to-image model reports, at minimum, *FID (with sample count) + CLIP-score + precision/recall*, and ideally *FID-DINOv2 + a learned preference score + a structured compositional metric (GenEval/T2I-CompBench) + a human-eval holdout*. Each plugs a hole the others leave open. That is not bureaucracy; it is the only way to avoid optimizing your model into a metric's blind spot.

## 15. How to measure honestly — a protocol

Knowing the formulas is half of it; the other half is a reproducible protocol, because most FID disagreements in practice are *protocol* disagreements, not real quality differences. The checklist:

1. **Fix the reference set and state it.** FID against COCO-30k is a different number than FID against your private validation set. Name it.
2. **Fix the sample count at 50k** (the field standard) and *report it*. If compute forces fewer, say so and remember the number is optimistically biased low.
3. **Fix the resize/interpolation** used to reach Inception's input, identically for real and generated. A bilinear-vs-bicubic mismatch alone can move FID by a point.
4. **Fix the seed and the sampler config** (steps, guidance scale, scheduler) — and report them. "FID 4.1" at 50 steps `$w=3$` is not comparable to "FID 4.1" at 25 steps `$w=7$`.
5. **Report a pair, minimum**: FID + CLIP-score, ideally + precision/recall. One number is a red flag.
6. **Keep a small human-eval holdout** (even 100–200 pairwise comparisons) as ground truth, especially before trusting any learned preference metric you also fine-tuned against.
7. **When you distill or quantize, re-check recall**, not just FID — the coverage cost hides in the conflated number.

This protocol is the difference between an evaluation that informs a decision and a leaderboard entry that fools you and everyone downstream. The math we derived is what makes the protocol *make sense*: you fix the sample count because FID is a biased estimator; you report CLIP-score because FID cannot read prompts; you check recall because FID conflates fidelity and coverage. The protocol is the formulas, applied.

## 16. When to reach for which metric (and when not to)

A decisive recommendation section, because every metric is a cost and using the wrong one wastes your time or misleads your team.

- **Comparing unconditional or class-conditional generators (no text)?** FID is your primary, paired with precision/recall. Skip CLIP-score (no prompt). Prefer FID-DINOv2 if the models are close.
- **Comparing text-to-image models?** FID alone is *insufficient and will mislead you* (Case 1). Report FID + CLIP-score + a learned preference score, and run GenEval/T2I-CompBench if composition matters. Always keep a human holdout.
- **Debugging *why* a model underperforms?** Reach for precision/recall first — it tells you fidelity-vs-coverage, which dictates the fix (raise diversity vs improve sharpness). FID will not tell you which way you failed.
- **Setting the guidance scale?** Plot the FID–CLIP curve across `$w$` and pick the knee for your product (Case 3). Do not chase a single FID minimum.
- **Distilling to few steps or quantizing?** Re-measure recall, not just FID (Case 4). The speed win is real; verify the coverage cost is acceptable.
- **Fine-tuning toward a learned reward (ImageReward/PickScore)?** Use it as a *signal*, never as the *sole objective*. Hold out human eval, watch for reward-hacked saturation, and stop early. This is where metrics do the most damage.
- **When NOT to compute FID at all:** on a handful of samples (the estimator is meaningless below a few thousand), on out-of-distribution content the Inception backbone cannot encode (illustrations, UI, text-heavy images — the backbone is an ImageNet object classifier), or as your *only* signal for anything text-conditioned. In those cases FID is not just unhelpful, it is actively misleading.

## 17. Key takeaways

- **An image is a sample from an unknown density `$p_\text{data}(x)$` over a million-dimensional space.** Generative modeling is learning a `$p_\theta$` close to it that you can sample from. "Evaluate the density" and "sample from it" are different capabilities — the four families split along that seam.
- **Natural images live on a low-dimensional manifold** (`$d \ll D$`), which is why density is degenerate (we add noise to fix it), why bad models blur (mass off the sheet), and why metrics measure distance in *feature* space, not pixels.
- **Three ways to model `$p(x)$`:** explicit likelihood (VAE/flow/AR — you get a number, but mass spreads off-manifold), implicit/sample-only (GAN — sharp but no density, fragile, mode-drop), and score/energy (diffusion — model `$\nabla_x\log p$`, sidestep the partition function, win on quality+coverage at the cost of speed).
- **KL is the ruler, and its asymmetry explains everything:** forward KL (MLE) is mass-covering and causes blur; reverse KL is mode-seeking and causes collapse. Same coin, opposite faces.
- **ELBO = reconstruction − KL**, derivable from Jensen *or* from the exact `$\log p = \mathrm{ELBO} + \mathrm{KL}(q\|p_\text{post})$` identity. The bound is tight when the approximate posterior matches the true one. Every later post optimizes a version of this.
- **The curse of dimensionality kills naive density estimation** (empty histograms, degenerate KDE, distance concentration). Deep generative models win by learning the manifold's coordinates, not by gridding the cube.
- **FID = Fréchet distance between two Gaussians fit to Inception features** = `$\lVert\mu_r-\mu_g\rVert^2 + \mathrm{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$`. It is the best general metric *and* it is sample-size biased, Gaussian-only, backbone-biased, prompt-blind, and conflates fidelity with coverage.
- **No single number is enough.** Report FID (with `$N$`) + CLIP-score + precision/recall at minimum; prefer FID-DINOv2; treat learned preference metrics as evaluation signals, not training targets — they reward-hack the moment you optimize them hard.
- **Two models can tie on FID and split on human preference.** The metric measured the axis where they agreed and was blind to the one where they differed. Match the metric to the axis you actually care about.

With the distribution defined, the modeling families mapped, the ELBO derived, and the metrics demystified, you have the probabilistic and evaluative spine for everything that follows. The next posts build specific models on this foundation — the [VAE](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) that latent diffusion stands on, then [diffusion from first principles](/blog/machine-learning/image-generation/diffusion-from-first-principles) — and the [capstone](/blog/machine-learning/image-generation/building-an-image-generation-stack) wires them into a serving stack where every one of these metrics earns its keep.

## Further reading

- Kingma & Welling (2013), *Auto-Encoding Variational Bayes* — the original VAE and the ELBO/reparameterization derivation.
- Heusel et al. (2017), *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* — introduces FID.
- Salimans et al. (2016), *Improved Techniques for Training GANs* — introduces the Inception Score.
- Kynkäänniemi et al. (2019), *Improved Precision and Recall Metric for Assessing Generative Models* — the fidelity/coverage split; Naeem et al. (2020), *Reliable Fidelity and Diversity Metrics* — density & coverage.
- Hessel et al. (2021), *CLIPScore: A Reference-free Evaluation Metric for Image Captioning* — the CLIP-score formulation; Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP).
- Stein et al. (2023), *Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models* — the FID-DINOv2 correction.
- Xu et al. (2023), *ImageReward*; Wu et al. (2023), *Human Preference Score v2 (HPSv2)*; Kirstain et al. (2023), *Pick-a-Pic / PickScore* — learned preference metrics.
- Bengio et al. (2013), *Representation Learning: A Review and New Perspectives* — the manifold hypothesis in context; 🤗 `torchmetrics` FID docs and `open_clip` for the runnable evaluation code.
- Within this series: [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard), [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch), and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
