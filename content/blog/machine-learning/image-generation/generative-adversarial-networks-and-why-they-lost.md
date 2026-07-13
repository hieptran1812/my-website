---
title: "Generative Adversarial Networks and Why They Lost (and Where They Still Win)"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A first-principles tour of GANs — the minimax game, the proof that the optimal discriminator minimizes Jensen-Shannon divergence, why training is so unstable, the Wasserstein fix, the StyleGAN lineage, and the honest post-mortem of why diffusion won and where GANs quietly came back."
tags:
  [
    "image-generation",
    "diffusion-models",
    "gan",
    "stylegan",
    "wasserstein-gan",
    "adversarial-distillation",
    "generative-ai",
    "deep-learning",
  ]
category: "machine-learning"
subcategory: "Image Generation"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/generative-adversarial-networks-and-why-they-lost-1.png"
---

In 2018, if you wanted to generate a photorealistic human face that had never existed, there was exactly one answer: a GAN. StyleGAN's faces were so good they spawned a website — *thispersondoesnotexist.com* — that refreshed a new fake stranger on every page load, and for years almost nobody could tell. GANs were the undisputed state of the art in image synthesis. Then, over the course of about eighteen months between 2021 and 2022, they lost. Stable Diffusion, DALL-E 2, and Imagen arrived, and the entire field pivoted to diffusion so completely that a 2024 ML engineer can have a productive career having never trained a GAN at all.

This is a post-mortem, but not a dismissive one. GANs lost the *general* text-to-image war for reasons that are precise and instructive — and understanding exactly *why* they lost is one of the cleanest ways to understand why diffusion *won*. The two failures of GANs (unstable adversarial training and mode collapse) map directly onto the two things diffusion does effortlessly (stable regression-style training and full mode coverage). So this is really two stories braided together: how a beautiful idea worked, and how its specific weaknesses defined the shape of its successor.

![Diagram of the adversarial training loop showing noise feeding a generator, real and fake images feeding a discriminator, and the two loss signals updating each network](/imgs/blogs/generative-adversarial-networks-and-why-they-lost-1.png)

And here is the twist that makes GANs worth a full chapter in a diffusion series rather than a footnote: they came back. Not as a model class, but as a *loss*. The fastest diffusion samplers in 2025 — SDXL-Turbo, the DMD line, GigaGAN-style upscalers — all reintroduce a discriminator to compress a 25-step diffusion model into one or two steps. The adversarial game that lost the war became the secret weapon that made its conqueror fast. By the end of this post you will be able to derive the GAN objective from scratch, prove that the optimal discriminator turns the game into Jensen-Shannon divergence minimization, explain on a whiteboard exactly why that breaks, write a DCGAN training loop in PyTorch, and reason precisely about when a GAN — or a GAN *loss* — is still the right tool. Throughout, keep the series' spine in view: the **generative trilemma** of quality, coverage, and speed. GANs are the model that bet everything on speed and quality and lost coverage; that single trade is the whole story.

## 1. The adversarial idea: a forger and a detective

Ian Goodfellow's 2014 framing is one of those ideas that sounds like a party anecdote and turns out to be a rigorous probability statement. You have two networks. The **generator** $G$ is a forger: it takes a random vector $z$ — pure noise, say a 100-dimensional Gaussian — and tries to paint a convincing fake. The **discriminator** $D$ is a detective: it looks at an image and outputs a single number, the probability that the image is *real* (from the training set) rather than *forged* (from $G$). The two are trained against each other. The detective gets better at spotting fakes; the forger, in response, gets better at making them; and at the hypothetical end of this arms race the forgeries are indistinguishable from reality and the detective can do no better than a coin flip.

The crucial structural fact — the thing that makes GANs different from every model that came before them — is that **$G$ never sees the data directly and never computes a likelihood.** A VAE writes down $p(x)$ (or a bound on it) and maximizes it. An autoregressive model factorizes $p(x) = \prod_i p(x_i \mid x_{\lt i})$ and maximizes that. A GAN does neither. The generator's *only* learning signal is the gradient that flows back through the discriminator. $G$ learns to make images that $D$ thinks are real, and $D$ defines "real" implicitly by what it has learned to accept. This is why GANs are called **implicit generative models**: they define a distribution you can sample from but whose density you cannot evaluate.

That implicitness is the source of both the magic and the misery. The magic: $G$ is free to put all its capacity into producing sharp, plausible images, unconstrained by the blurring pressure that a pixel-space reconstruction loss imposes on a VAE. (This is exactly why GAN samples were sharp and VAE samples were soft — a VAE's $L_2$ reconstruction term averages over plausible pixels and the average of sharp images is a blur. We unpacked that failure in [variational autoencoders from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch).) The misery: because there is no likelihood and no fixed target, the loss landscape that $G$ descends is *defined by a second network that is itself moving.* You are not minimizing a function; you are searching for an equilibrium of a game. Games do not have to converge, and this one frequently does not.

Let me set up the notation we will use for the rest of the post. Let $p_{\text{data}}(x)$ be the true data distribution and $p_g(x)$ the distribution induced by pushing the noise prior $p_z(z)$ through the generator $x = G(z)$. The discriminator $D(x) \in [0,1]$ estimates $\Pr[x \text{ is real}]$. We want $p_g$ to converge to $p_{\text{data}}$, and we want to achieve this *without ever writing down either density.*

It is worth dwelling on why this was such a radical move in 2014. Every generative model before GANs was, in one way or another, a likelihood model. A VAE maximizes a tractable lower bound on $\log p(x)$. An autoregressive model maximizes an exact factorized likelihood. A normalizing flow uses the change-of-variables formula to make $\log p(x)$ exactly computable. All of them pay a tax for that tractability: the VAE's bound forces a Gaussian decoder that blurs; the autoregressive model pays in sequential, $O(N)$ sampling over $N$ pixels; the flow constrains its architecture to invertible layers with cheap Jacobians. GANs paid none of these taxes because they refused to compute a likelihood at all. That refusal bought sharpness and a single-pass sampler — and the bill came due as instability. The generative trilemma is, at bottom, the statement that you cannot get all three of {tractable likelihood, sharp samples, full coverage} for free; GANs picked sharpness and one-step sampling and mortgaged the rest.

One more framing worth installing before the math. People describe GANs as "two networks competing," which is true but undersells the structure. What is really happening is that the discriminator is a *learned, adaptive loss function* for the generator. A fixed loss like $L_2$ says the same thing about every image — "be close to this target in pixel space." The discriminator instead *learns what to criticize*: early in training it complains about gross color statistics, later about texture, later still about fine semantic inconsistencies (a face with mismatched earrings, a hand with six fingers). This adaptivity is exactly why GAN samples can be so sharp — the loss keeps finding new, subtler defects to penalize. It is also exactly why training is hard — your loss function is itself a neural network being trained at the same time, and if it learns too fast or too slow relative to the generator, the whole system destabilizes. Keep both halves of that sentence in mind; they recur for the rest of the post.

## 2. The minimax objective and the proof it minimizes JS divergence

Here is the science block — the derivation that every GAN paper assumes you know and most blog posts skip. It is worth doing carefully because the conclusion is exactly what predicts the training instability later.

The original GAN value function is

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}\big[\log D(x)\big] + \mathbb{E}_{z \sim p_z}\big[\log(1 - D(G(z)))\big].
$$

Read it as a competition over the same scalar $V$. The discriminator wants to *maximize* it: push $D(x) \to 1$ on real data (so $\log D(x) \to 0$, its max) and push $D(G(z)) \to 0$ on fakes (so $\log(1 - D(G(z))) \to 0$). The generator wants to *minimize* it: it can only touch the second term, and it minimizes by making $D(G(z)) \to 1$ — fooling the detective.

**Step 1: solve the inner maximization for a fixed $G$.** Rewrite the expectation over $z$ as an expectation over the generated samples $x = G(z)$, which have density $p_g$. Then

$$
V(D, G) = \int_x \Big[ p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x)) \Big]\, dx.
$$

For each fixed $x$ the integrand has the form $a \log y + b \log(1 - y)$ with $a = p_{\text{data}}(x)$, $b = p_g(x)$, and $y = D(x)$. Differentiate with respect to $y$ and set to zero:

$$
\frac{a}{y} - \frac{b}{1 - y} = 0 \quad\Longrightarrow\quad y^* = \frac{a}{a + b}.
$$

So the **optimal discriminator** for a fixed generator is

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}.
$$

This is a clean, interpretable result: the ideal detective reports the posterior probability that a sample is real under a 50/50 prior. Where $p_g = p_{\text{data}}$ it returns $\tfrac12$ — it genuinely cannot tell. Note already a warning sign for later: this formula assumes both densities are defined and comparable at $x$. If $p_g(x) > 0$ but $p_{\text{data}}(x) = 0$ — the generator produces something the data never contains — then $D^*(x) = 0$, a perfect, confident rejection. Hold that thought.

**Step 2: substitute $D^*$ back and see what $G$ is actually minimizing.** Plug $D^*$ into $V$:

$$
V(D^*, G) = \mathbb{E}_{x \sim p_{\text{data}}}\!\left[\log \frac{p_{\text{data}}}{p_{\text{data}} + p_g}\right] + \mathbb{E}_{x \sim p_g}\!\left[\log \frac{p_g}{p_{\text{data}} + p_g}\right].
$$

Now insert a factor of $2$ inside each log (adding $-\log 2$ twice, i.e. $-2\log 2$ total, which we add back outside):

$$
V(D^*, G) = -\log 4 + \mathrm{KL}\!\left(p_{\text{data}} \,\Big\|\, \frac{p_{\text{data}} + p_g}{2}\right) + \mathrm{KL}\!\left(p_g \,\Big\|\, \frac{p_{\text{data}} + p_g}{2}\right).
$$

Those two KL terms against the mixture $m = \tfrac12(p_{\text{data}} + p_g)$ are, by definition, twice the **Jensen-Shannon divergence**:

$$
\mathrm{JS}(p_{\text{data}} \,\|\, p_g) = \tfrac12 \mathrm{KL}(p_{\text{data}} \| m) + \tfrac12 \mathrm{KL}(p_g \| m).
$$

So the whole minimax game, *at the discriminator's optimum*, reduces to

$$
\min_G V(D^*, G) = -\log 4 + 2\,\mathrm{JS}(p_{\text{data}} \,\|\, p_g).
$$

This is the punchline of GAN theory. **A GAN, with an optimal discriminator, is minimizing the Jensen-Shannon divergence between the data distribution and the generated distribution.** JS is non-negative and zero only when $p_g = p_{\text{data}}$, so the global minimum is exactly $V = -\log 4$, achieved when the generator perfectly matches the data and $D^* \equiv \tfrac12$. The math says the game has the fixed point we want. We define JS, KL, and why a divergence is the natural object here in the sibling post on [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions); here we just need the consequence.

Everything that follows — the instability, the vanishing gradients, the Wasserstein fix — is a consequence of this one identity. Because the moment you ask *how well-behaved is JS divergence as an optimization objective when $p_g$ and $p_{\text{data}}$ live on different manifolds?* the answer is: catastrophically badly. And images always live on different manifolds.

Before we get there, two caveats that honest GAN theory always carries, because they are where the elegant proof quietly departs from reality. **First, the proof assumes $D$ reaches its optimum $D^*$ for each fixed $G$.** In practice you never train $D$ to convergence at every generator step — that would be absurdly expensive and, as we'll see, it would *starve* the generator of gradient. Real training takes a few discriminator steps then a generator step, so $D$ is always a lagging, approximate $D^*$. The clean JS interpretation holds only in the limit; away from it you are optimizing something fuzzier. **Second, the proof treats $G$ and $D$ as arbitrary functions in distribution space (it optimizes over the densities $p_g$ and the function $D(x)$ directly).** Real networks are finite-capacity parametric families optimized by gradient descent, so even the "global minimum exists" guarantee does not translate into "gradient descent finds it." The existence of a good equilibrium and the *reachability* of that equilibrium by alternating SGD are completely different claims, and the gap between them is the entire practical difficulty of GANs.

That gap has a name in game theory: we are looking for a **Nash equilibrium** of a two-player game, a point where neither player can improve by unilaterally changing its strategy. The trouble is that simultaneous-gradient dynamics on a minimax objective is *not* guaranteed to converge to a Nash equilibrium even when one exists — the canonical toy example is $\min_x \max_y x \cdot y$, whose only equilibrium is $(0,0)$ but whose gradient dynamics *orbit* the origin forever, never settling. GAN training inherits exactly this rotational, non-convergent character. When practitioners say a GAN "oscillates" or "cycles," they are watching this game-theoretic phenomenon, not a bug in their code. The two-time-scale update rule (TTUR) and a hundred stabilizer tricks exist to damp those orbits into a spiral that actually reaches the center. None of this is a problem for a regression loss, which is just convex-ish descent toward a fixed target.

## 3. Why GAN training is unstable — three failures, one root cause

The JS result is beautiful and almost useless as a training signal, for a reason that is geometric, not numerical. Real images do not fill pixel space; they lie on a thin, curved, low-dimensional **manifold** inside it (the manifold hypothesis, which the [foundation post on why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard) develops). A freshly initialized generator produces a *different* thin manifold somewhere else in the space. Two thin manifolds in a high-dimensional space generically do not intersect — they have measure-zero overlap, like two random lines in a million-dimensional room. And on disjoint supports, JS divergence is *constant.*

#### Worked example: JS divergence on disjoint supports

Take the simplest possible case. Let $p_{\text{data}}$ be a point mass at $0$ and $p_g$ a point mass at $\theta \neq 0$. Their supports are disjoint. The mixture $m$ puts mass $\tfrac12$ at each point. Compute:

$$
\mathrm{KL}(p_{\text{data}} \| m) = 1 \cdot \log\frac{1}{1/2} = \log 2, \qquad \mathrm{KL}(p_g \| m) = \log 2,
$$

so $\mathrm{JS} = \tfrac12(\log 2 + \log 2) = \log 2$, **for every $\theta \neq 0$.** The divergence is $\log 2$ whether the generated point is $0.001$ away or $1{,}000$ away from the data. Its gradient with respect to $\theta$ is *exactly zero everywhere except at $\theta = 0$,* where it is discontinuous. There is no slope to descend. The generator gets no information about which direction would bring it closer to the data, because JS does not measure *distance* — it measures *overlap*, and overlap is zero until the supports touch, at which point it snaps. This is the analytic core of GAN instability, and it manifests as three named failures.

![Before-and-after comparison contrasting Jensen-Shannon divergence on disjoint supports, where gradients vanish, with the smooth Wasserstein distance under a Lipschitz critic](/imgs/blogs/generative-adversarial-networks-and-why-they-lost-6.png)

**Failure one: vanishing gradients.** As the discriminator gets good — and it gets good fast, because separating two disjoint manifolds is easy — $D(G(z)) \to 0$ on fakes. The generator's loss is $\log(1 - D(G(z)))$, whose gradient with respect to $G$'s output scales like $\frac{-1}{1 - D}$ times $\nabla D$, and as $D \to 0$ the *useful* part of that gradient collapses. The better the detective, the less the forger learns. You get the worst possible coupling: a strong discriminator starves the generator of signal. This is the direct training-time symptom of the JS-on-disjoint-supports result.

**Failure two: the non-saturating heuristic and its own pathology.** Goodfellow's own 2014 paper anticipated the vanishing-gradient problem and proposed a fix that is still standard: instead of having $G$ minimize $\log(1 - D(G(z)))$, have it *maximize* $\log D(G(z))$ — the so-called **non-saturating loss**. These have the same fixed point but very different gradients: when $D(G(z))$ is near zero (fakes easily caught), $-\log D(G(z))$ has a *large* gradient, so the generator still learns early. This is what almost every GAN actually trains with, and it is why GANs work at all in practice. But it is a heuristic, not a divergence minimizer — it can be shown to minimize a reverse-KL-minus-JS combination that *rewards mode-seeking*, which feeds directly into the next failure.

**Failure three: mode collapse.** This is the famous one. The generator discovers that it does not need to cover the whole data distribution to fool the discriminator — it only needs to produce *some* images the discriminator can't reject. So it collapses onto a few high-quality modes and abandons the rest. A face GAN that only ever generates young women facing the camera. A digit GAN that produces gorgeous 3s and 8s and never a 7. The samples look great; the *coverage* is a disaster. We will dissect mode collapse in its own section because it is the failure that ultimately cost GANs the war.

The root cause unifying all three: **the adversarial objective with JS gives the generator no smooth, global notion of "how far am I from the data, and in which direction?"** A regression loss — the kind diffusion uses, where you literally predict the noise that was added and take the $L_2$ error — has a smooth gradient everywhere and a fixed target that does not move. That structural difference is the entire reason diffusion training is boring and stable and GAN training is a knife-edge. When people say "diffusion just works," this is the thing they mean.

## 4. The Wasserstein fix: trading JS for Earth-Mover distance

If the problem is that JS divergence is flat on disjoint supports, the fix is to use a divergence that *isn't.* That is the entire idea of the Wasserstein GAN (Arjovsky, Chintala, Bottou, 2017), and it is the most important theoretical advance in the GAN line.

The **Wasserstein-1 distance**, also called the **Earth-Mover (EM) distance**, measures the minimum "cost" of transporting the mass of one distribution to match the other, where moving a unit of mass a distance $d$ costs $d$:

$$
W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x, y) \sim \gamma}\big[\,\|x - y\|\,\big],
$$

where $\Pi$ is the set of all joint distributions (transport plans) with the right marginals. Go back to the disjoint-point example: with $p_{\text{data}}$ at $0$ and $p_g$ at $\theta$, the only transport plan moves all the mass from $\theta$ to $0$ at cost $|\theta|$, so $W = |\theta|$. Unlike JS, which was stuck at $\log 2$, the Wasserstein distance is *exactly the distance you'd want* — it shrinks smoothly to zero as $\theta \to 0$, and its gradient points straight at the data. **It gives the generator a usable signal even when the supports are completely disjoint.** That is the whole game.

The catch is that the infimum over all transport plans is intractable to compute directly. WGAN uses the **Kantorovich-Rubinstein duality**, which rewrites the EM distance as a maximization over 1-Lipschitz functions:

$$
W(p_{\text{data}}, p_g) = \sup_{\|f\|_L \le 1} \; \mathbb{E}_{x \sim p_{\text{data}}}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)].
$$

Here $f$ is the **critic** (WGAN deliberately renames the discriminator, because $f$ no longer outputs a probability in $[0,1]$ — it outputs an unbounded real-valued score). The constraint $\|f\|_L \le 1$ means $f$ is **1-Lipschitz**: $|f(x) - f(y)| \le \|x - y\|$, i.e. its outputs cannot change faster than its inputs. So the WGAN training procedure is: train a critic $f$ to maximize the difference in its average score between real and fake (approximating the sup), then train the generator to *raise* its own average critic score, which minimizes the EM distance. No more $\log$, no more saturating sigmoid — just a smooth, well-conditioned regression-like signal.

The only real engineering question is *how do you enforce 1-Lipschitz?* The original WGAN paper clipped every critic weight to $[-c, c]$, which works but is crude — it caps the critic's capacity and is sensitive to the clip value. The decisive improvement is **WGAN-GP** (Gulrajani et al., 2017), which adds a **gradient penalty**: a 1-Lipschitz function has gradient norm $\le 1$ everywhere, so penalize the critic when its gradient norm strays from $1$ at points $\hat{x}$ sampled uniformly along straight lines between real and fake pairs:

$$
L_{\text{critic}} = \underbrace{\mathbb{E}_{x \sim p_g}[f(x)] - \mathbb{E}_{x \sim p_{\text{data}}}[f(x)]}_{\text{negative EM estimate}} + \lambda \, \mathbb{E}_{\hat{x}}\Big[\big(\|\nabla_{\hat{x}} f(\hat{x})\|_2 - 1\big)^2\Big],
$$

with the penalty weight $\lambda = 10$ being the canonical value that essentially everyone uses. WGAN-GP was the recipe that made GAN training reliable enough to scale — ProGAN, StyleGAN, and BigGAN all build on Wasserstein-style or hinge-loss critics rather than the original log-loss. If you only remember one practical thing about training a GAN, remember $\lambda = 10$ and a gradient penalty.

#### Worked example: WGAN-GP critic step in PyTorch

Here is the gradient-penalty computation as you'd actually write it. This is the single most error-prone part of a GAN implementation, so it is worth seeing concretely.

```python
import torch

def gradient_penalty(critic, real, fake, device, lambda_gp=10.0):
    # sample a random interpolation point between each real/fake pair
    batch = real.size(0)
    eps = torch.rand(batch, 1, 1, 1, device=device)          # one alpha per image
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)

    scores = critic(interp)                                   # f(x_hat), shape [batch, 1]
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,                                    # need 2nd-order grad for backprop
        retain_graph=True,
    )[0]
    grads = grads.view(batch, -1)
    grad_norm = grads.norm(2, dim=1)                          # ||grad f(x_hat)||_2
    return lambda_gp * ((grad_norm - 1.0) ** 2).mean()        # push norm toward 1
```

Two subtleties that bite everyone the first time. First, `create_graph=True` is mandatory — you are about to backprop *through* this gradient norm, so the gradient computation itself must be part of the graph. Second, in a WGAN-GP critic you must **remove BatchNorm** (it correlates samples within a batch, breaking the per-sample Lipschitz assumption) and use LayerNorm or InstanceNorm instead. Forget either and your "WGAN" silently trains as something else.

A subtle but important point about *why* the gradient penalty enforces "gradient norm exactly 1" rather than "gradient norm at most 1." Strictly, 1-Lipschitz only requires $\|\nabla f\| \le 1$. But the optimal critic that achieves the Wasserstein sup has gradient norm *exactly* 1 almost everywhere along the optimal transport paths between real and fake samples — that is a property of the dual solution. So penalizing toward 1 (a two-sided penalty) rather than just clamping below 1 pushes the critic toward the *actual* optimal critic, not merely a feasible one, and empirically trains better. This is the kind of detail that looks like a hack until you trace it back to the duality theorem, at which point it is exactly right. It is also why a one-sided penalty (penalize only when norm exceeds 1) sometimes underperforms the symmetric version on hard datasets — you are no longer steering toward the true optimum.

Here is the comparison of the three Lipschitz-enforcement strategies you will encounter, so you can pick one deliberately:

| Method | How it enforces Lipschitz | Cost | Failure mode |
| --- | --- | --- | --- |
| Weight clipping (WGAN) | Clamp every weight to $[-c, c]$ | Cheap, one line | Caps capacity; sensitive to $c$; pushes weights to extremes |
| Gradient penalty (WGAN-GP) | Penalize $(\|\nabla f\| - 1)^2$ at interpolates | 1 extra backward pass | Per-sample, so no BatchNorm in critic; $\lambda$ tuning |
| Spectral norm (SN-GAN) | Divide each layer's weight by its largest singular value | Cheap, via power iteration | Slightly conservative bound; often the default in modern code |

In modern GAN code you most often see **spectral normalization** (Miyato et al., 2018) because it is the cheapest reliable option — it constrains each layer to be 1-Lipschitz by construction via a power-iteration estimate of the top singular value, with no extra loss term and no per-sample restriction. Many production GANs combine spectral norm with a hinge loss rather than the Wasserstein loss; the hinge loss $\mathbb{E}[\max(0, 1 - D(x_{\text{real}}))] + \mathbb{E}[\max(0, 1 + D(x_{\text{fake}}))]$ is what StyleGAN2 and BigGAN actually use. The lineage here is "WGAN identified the disease (JS flatness), WGAN-GP and spectral norm provided practical cures, and the field settled on spectral-norm-plus-hinge as the default antibiotic."

## 5. Mode collapse: the failure that cost GANs the war

Vanishing gradients are an annoyance you can mostly engineer around. Mode collapse is the structural flaw that diffusion simply does not have, and it is the deepest reason the field switched.

Recall the asymmetry baked into the adversarial objective. The discriminator's job is to catch fakes; the generator's job is to produce fakes that aren't caught. **Neither objective contains any term that rewards diversity.** A generator that produces one single perfect image of a cat — the same image every time, regardless of $z$ — would fool any discriminator that has been trained only to ask "is this a plausible cat?" The discriminator can object "but you always produce the *same* cat," yet to express that objection it would need to look at the *distribution* of $G$'s outputs, and a standard discriminator only ever sees one image at a time. So the generator can, and under pressure does, retreat to a small set of safe outputs. It is exploiting a genuine blind spot in the loss.

![Before-and-after diagram contrasting a mode-collapsed generator covering only two digit modes with a fully covering generator across all ten, plus the fixes that restore coverage](/imgs/blogs/generative-adversarial-networks-and-why-they-lost-2.png)

In the language of the generative trilemma, this is GANs spending their entire budget on **quality** and **speed** and going bankrupt on **coverage**. Diffusion makes the opposite trade by construction: its training objective is to denoise *every* noised real example, so it is *forced* to put probability mass everywhere the data has mass. A diffusion model literally cannot mode-collapse the way a GAN does, because its loss is a per-example reconstruction averaged over the whole dataset, not a game against a critic that only checks plausibility. This is the single most important sentence in the post: **diffusion's training objective has coverage built in; the GAN objective has to have coverage bolted on, and the bolts keep loosening.**

How do you even *diagnose* mode collapse, given that you cannot evaluate $p_g$? Several practical signals:

- **Visual inspection of a large batch.** Generate 256 samples from independent $z$ and look. If you see the same handful of images repeating, or suspiciously little variety, that is collapse. Crude but the first thing every practitioner does.
- **Recall (and precision-recall curves).** The improved-precision-and-recall metric (Kynkäänniemi et al., 2019) estimates, via $k$-nearest-neighbor manifolds, what fraction of the *real* manifold the generator covers (recall) versus what fraction of *generated* samples land on the real manifold (precision). Mode collapse shows up as high precision (samples look real) but low recall (most of the data is unrepresented). This is the metric that quantifies the coverage failure.
- **FID itself.** Fréchet Inception Distance compares the mean and covariance of Inception features between real and generated sets, $\mathrm{FID} = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$. Because it includes the *covariance* term, a collapsed generator whose outputs have artificially low variance gets penalized — FID will be high even if individual samples look sharp. FID is not a perfect coverage metric (it is summarizing everything in two moments of one feature space), but it does feel collapse.

And the fixes — none of which fully solve it — are a tour of GAN engineering folklore:

- **Minibatch discrimination / minibatch standard deviation** (used in ProGAN and StyleGAN). Append a statistic of the *batch* (e.g. the standard deviation of features across the batch) as an extra feature map to the discriminator, so it *can* see when the generator's outputs lack variety. This directly attacks the blind spot.
- **Wasserstein loss.** WGAN-GP's smoother objective empirically collapses less, because the critic provides a meaningful gradient that pulls toward *all* the data mass, not just toward fooling a saturated sigmoid.
- **Unrolled GANs, PacGAN, two-time-scale update rule (TTUR).** A grab-bag of tricks that stabilize the dynamics so the generator can't sprint to a collapsed solution faster than the discriminator can object.

The telling thing about that list is that it is a list of *patches.* Every one mitigates a symptom of a missing diversity term in the objective. Diffusion needed none of them. When a whole subfield's best practices are a pile of stabilizers for one structural flaw, that subfield is vulnerable to a method that doesn't have the flaw.

#### Worked example: precision-recall reads out the collapse

Make the coverage failure concrete with numbers. Suppose you train two GANs on a balanced 10-class dataset (say MNIST digits) and evaluate each with the improved precision-recall metric on 50,000 generated samples against 50,000 real ones. GAN-A is a vanilla DCGAN that has partially collapsed; GAN-B is the same architecture with a minibatch-stddev layer and a WGAN-GP loss. You might observe:

| Model | Precision | Recall | FID | Visual read |
| --- | --- | --- | --- | --- |
| GAN-A (collapsed) | 0.88 | 0.31 | 28.4 | Sharp digits, but mostly 1s, 3s, 8s; rarely a 4 or 7 |
| GAN-B (fixed) | 0.82 | 0.74 | 9.1 | All ten digits, slightly less crisp on average |

Read this carefully, because it is the entire GAN-vs-diffusion story in one table. GAN-A has *higher precision* — its individual samples look more real, because it only ever attempts the easy modes it has mastered. But its *recall is catastrophic* at 0.31: it is covering roughly a third of the real manifold. GAN-B trades a little precision (0.88 → 0.82, samples slightly softer on average) for a huge recall gain (0.31 → 0.74), and its FID more than halves (28.4 → 9.1) because FID's covariance term punishes the missing variety. The crucial insight: **a metric that only looked at sample quality would have ranked the collapsed model higher.** This is precisely why the field needed precision-recall and FID rather than Inception Score alone — and precisely the trap a GAN's objective walks into, optimizing the thing that looks good per-sample while quietly abandoning coverage. Diffusion does not face this trade because its loss never lets it drop a mode in the first place.

There is a deeper, almost philosophical way to see why GANs collapse and diffusion does not, and it is worth stating plainly because it is the load-bearing idea of the whole post. A GAN's generator is graded on a *pass/fail per image*: does this single output fool the critic? Nothing in that grade asks "and is your *ensemble* of outputs as varied as the data?" So the optimal strategy under a myopic critic is to find the few outputs that pass most reliably and produce those. A diffusion model's denoiser, by contrast, is graded on *reconstructing every real example* across every noise level — its loss is $\mathbb{E}_{x \sim p_{\text{data}}, t, \epsilon}\big[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\big]$, an expectation *over the real data distribution itself.* You cannot minimize an expectation over $p_{\text{data}}$ by ignoring most of $p_{\text{data}}$; every real example contributes to the loss and pulls the model toward representing it. Coverage is not a property you add to diffusion; it is the literal definition of its objective. That single structural difference, more than any benchmark number, is why the field switched.

## 6. A minimal DCGAN training loop in PyTorch

Theory is cheap; let's train one. DCGAN (Radford, Metz, Chintala, 2015) is the architecture that made GANs work on images — the recipe is "use a fully-convolutional generator and discriminator, strided convolutions instead of pooling, BatchNorm, and ReLU/LeakyReLU." It is still the right starting point for understanding the alternating optimization. Here is a compact but runnable training core for, say, 64×64 images.

First the two networks:

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64, nc=3):
        super().__init__()
        self.net = nn.Sequential(
            # z: [B, z_dim, 1, 1] -> upsample to [B, nc, 64, 64]
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),               # 4x4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),               # 8x8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),               # 16x16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),                   # 32x32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),                                            # 64x64, outputs in [-1, 1]
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),   # 32x32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),                    # 16x16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),                    # 8x8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),                    # 4x4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),                         # -> [B,1,1,1]
        )

    def forward(self, x):
        return self.net(x).view(-1)        # logits, [B]
```

Now the alternating optimization — the heart of the matter. We do one discriminator step then one generator step per batch, using the non-saturating loss via `BCEWithLogitsLoss`:

```python
import torch.optim as optim

device = "cuda"
z_dim = 100
G = Generator(z_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCEWithLogitsLoss()
# DCGAN's canonical hyperparameters: Adam, lr=2e-4, betas=(0.5, 0.999)
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(num_epochs):
    for real, _ in dataloader:               # real images in [-1, 1], shape [B, 3, 64, 64]
        real = real.to(device)
        b = real.size(0)

        # ---- (1) train D: maximize log D(x) + log(1 - D(G(z))) ----
        z = torch.randn(b, z_dim, 1, 1, device=device)
        fake = G(z)
        d_real = D(real)
        d_fake = D(fake.detach())            # detach: no generator grads in the D step
        loss_D = criterion(d_real, torch.ones_like(d_real)) \
               + criterion(d_fake, torch.zeros_like(d_fake))
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # ---- (2) train G: maximize log D(G(z))  (non-saturating) ----
        d_fake_for_g = D(fake)               # reuse fake, now WITH generator grads
        loss_G = criterion(d_fake_for_g, torch.ones_like(d_fake_for_g))  # label fakes "real"
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()
```

The three things that make or break this loop are all in those comments. **`fake.detach()` in the D step** stops generator gradients from flowing while you update the discriminator — forget it and the two optimizers fight over the same graph. **Labeling fakes as "real" in the G step** (`torch.ones_like`) is the non-saturating trick from Section 3: the generator is rewarded for the discriminator believing its fakes. And the **`betas=(0.5, 0.999)`** Adam setting — a lower first-moment decay than the default $0.9$ — is DCGAN folklore that genuinely matters; the standard $0.9$ makes GAN training noticeably less stable because it over-smooths a gradient that is already chasing a moving target.

To convert this to WGAN-GP you would: drop the sigmoid/BCE and have $D$ (now the critic) output a raw score; train the critic 5 steps per generator step; replace `loss_D` with `d_fake.mean() - d_real.mean() + gradient_penalty(...)` from Section 4; and replace `loss_G` with `-D(fake).mean()`. That is the whole difference, and it is worth internalizing how *small* the code delta is for how *large* the stability difference is.

**Stress-testing the loop.** Now let's reason like an engineer about what actually goes wrong when you run this, because GAN debugging is a distinct skill and the symptoms are diagnostic.

*What happens if the discriminator gets too strong?* If `loss_D` drops to near zero and stays there while `loss_G` climbs, your discriminator has won — it is rejecting every fake with high confidence, $D(G(z)) \to 0$, and (Section 3) the generator's gradient has vanished. The fix is to *handicap the discriminator*: lower its learning rate, train it fewer steps per generator step, add noise to its inputs (instance noise), or one-sided label smoothing (use $0.9$ instead of $1.0$ for the "real" target so $D$ never gets fully confident). The two-time-scale update rule (TTUR) — giving $D$ and $G$ *different* learning rates, often a higher one for $D$ — is the principled version of this balancing act.

*What happens if you see the losses oscillate forever?* That is the Nash-orbiting phenomenon from Section 2 — the game is cycling rather than converging. Lower the learning rates, increase the Adam $\beta_1$ slightly back toward stability, or add the gradient penalty (it damps the dynamics). If the *samples* are improving even while the losses oscillate, that is normal and fine; GAN losses are notoriously uninformative about sample quality, which is why you monitor FID on a held-out generated set rather than the loss curve. This is a genuinely disorienting property for someone coming from supervised learning: **the loss going down does not mean the model is getting better, and the loss oscillating does not mean it is getting worse.** You must look at FID and at the images.

*What happens if every sample looks identical?* Mode collapse (Section 5). Add a minibatch-stddev feature to the discriminator, switch to WGAN-GP, or check whether your generator's learning rate is so high it is sprinting to a collapsed solution. *What happens if you get checkerboard artifacts?* That is the `ConvTranspose2d` upsampling — strided transposed convolutions create periodic overlap patterns. Replace them with nearest-neighbor or bilinear upsampling followed by a regular convolution, the standard cure. Each of these failure-to-fix mappings is muscle memory for anyone who trained GANs in the 2018-2020 era, and the sheer length of the list is, again, the tell: this is a method held together by operator expertise.

## 7. The architecture lineage: DCGAN to StyleGAN to BigGAN

GANs did not lose because they stopped improving — they lost while at their absolute peak, which is part of what makes the post-mortem interesting. The architecture line from 2015 to 2019 is a sequence of sharp, well-motivated fixes, each targeting a concrete limitation of the last.

![Timeline of the GAN architecture lineage from the original GAN through DCGAN, WGAN-GP, ProGAN, StyleGAN2, and BigGAN with their headline results](/imgs/blogs/generative-adversarial-networks-and-why-they-lost-3.png)

**DCGAN (2015)** established the convolutional recipe above and proved GANs could do 64×64 natural images with interpretable latent arithmetic (the famous "smiling woman − neutral woman + neutral man = smiling man" vector algebra in $z$-space). Resolution and stability were the open problems.

**ProGAN / Progressive Growing (Karras et al., 2017)** cracked high resolution with a training-schedule idea: start by training $G$ and $D$ on 4×4 images, then *progressively fade in* new layers to reach 8×8, 16×16, … up to 1024×1024. Low resolutions establish coarse structure stably before fine detail is ever attempted. ProGAN was the first GAN to produce convincing megapixel faces, training on the FFHQ-precursor CelebA-HQ. It also introduced the **minibatch standard deviation** layer (the coverage patch from Section 5).

**StyleGAN (Karras et al., 2018) and StyleGAN2 (2019)** are the high-water mark and deserve real detail, because their design choices are clever and one of them — adaptive normalization — echoes directly into diffusion's AdaLN conditioning.

![Layered diagram of the StyleGAN generator showing the mapping network producing the disentangled W space, AdaIN style injection per resolution, progressive synthesis, and the truncation trick](/imgs/blogs/generative-adversarial-networks-and-why-they-lost-5.png)

StyleGAN's central insight is that feeding the latent $z$ in at the input of the generator is a mistake, because the input distribution is forced to be a simple Gaussian and the network has to *warp* that Gaussian into the curved manifold of real images, which entangles factors of variation. StyleGAN instead does three things:

1. **A mapping network.** An 8-layer MLP $f: z \mapsto w$ transforms the Gaussian $z \in \mathcal{Z}$ into an **intermediate latent** $w \in \mathcal{W}$. Crucially $\mathcal{W}$ is not forced to be Gaussian — the mapping network can learn to "unwarp" the data manifold so that $\mathcal{W}$ is more **disentangled**, meaning a single direction in $\mathcal{W}$ tends to change one semantic attribute (age, pose, hair) rather than several at once. This is the property that made StyleGAN editing so clean.

2. **AdaIN style injection.** Rather than entering at the input, $w$ controls the image at *every resolution* via **Adaptive Instance Normalization**. For a feature map, AdaIN normalizes each channel to zero mean and unit variance, then re-scales and shifts it using a style $(y_s, y_b)$ derived from $w$: $\mathrm{AdaIN}(x_i, y) = y_{s,i} \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$. Coarse-resolution styles control pose and face shape; fine-resolution styles control color and microtexture. This per-resolution control is what enables **style mixing** — run two different $w$ vectors for the coarse and fine layers and you get a face with one person's pose and another's coloring. (If "normalize then re-scale by a learned, conditioning-dependent shift and scale" sounds familiar, it should: it is exactly the AdaLN-Zero mechanism that conditions a Diffusion Transformer on the timestep and text. StyleGAN got there first.)

3. **The truncation trick.** At inference you can trade diversity for quality by pulling each $w$ toward the mean $\bar{w}$ of the $\mathcal{W}$ distribution: $w' = \bar{w} + \psi(w - \bar{w})$ with $\psi < 1$. Small $\psi$ gives extremely clean but more average-looking faces (low diversity, high quality); $\psi = 1$ gives full diversity with more artifacts in the tails of the distribution. This is a *sampling-time* quality/diversity knob — and notice it is the GAN analog of classifier-free guidance scale in diffusion, which trades the exact same two quantities. The trilemma's quality-versus-diversity axis shows up no matter which model family you pick.

To make AdaIN concrete, here is the operation in PyTorch — it is short, and seeing it next to the diffusion AdaLN it inspired is illuminating. The style vector is projected to a per-channel scale and bias, the feature map is instance-normalized, then rescaled:

```python
import torch
import torch.nn as nn

class AdaIN(nn.Module):
    """StyleGAN-style adaptive instance norm: w controls per-channel scale/bias."""
    def __init__(self, w_dim, n_channels):
        super().__init__()
        # project the style w into 2*C numbers: a scale and a bias per channel
        self.to_style = nn.Linear(w_dim, n_channels * 2)
        self.n_channels = n_channels

    def forward(self, x, w):                      # x: [B, C, H, W],  w: [B, w_dim]
        style = self.to_style(w)                  # [B, 2C]
        y_s, y_b = style.chunk(2, dim=1)          # scale, bias  -> each [B, C]
        y_s = y_s.view(-1, self.n_channels, 1, 1)
        y_b = y_b.view(-1, self.n_channels, 1, 1)
        # instance-normalize each channel to zero mean / unit variance
        mu = x.mean(dim=(2, 3), keepdim=True)
        sigma = x.std(dim=(2, 3), keepdim=True) + 1e-8
        x_norm = (x - mu) / sigma
        return y_s * x_norm + y_b                  # re-inject style
```

Compare this to AdaLN-Zero in a Diffusion Transformer, where a timestep-and-text embedding is projected to per-channel scale/shift/gate parameters applied to a LayerNorm'd token stream. It is the *same idea* — "normalize the features, then modulate them with a learned, conditioning-dependent affine transform" — transplanted from a convolutional GAN into a transformer denoiser. When you study DiT later in the series, you are partly studying StyleGAN's conditioning mechanism in a new home. Good ideas in this field are remarkably portable, and AdaIN is one of the most portable of all.

Style mixing falls right out of this design: run one $w_1$ for the coarse-resolution AdaIN layers and a different $w_2$ for the fine layers, and you blend the high-level structure of one sample with the texture of another — the basis of StyleGAN's celebrated control over generated faces.

StyleGAN2 fixed the characteristic "blob" artifacts of StyleGAN by replacing AdaIN with a **weight-demodulation** scheme (the normalization was creating droplet artifacts), removed progressive growing in favor of skip/residual connections, and added **path-length regularization** to keep the $\mathcal{W} \to$ image mapping smooth. The result on FFHQ (70,000 high-quality faces at 1024×1024) was an FID around **2.8**, which was state-of-the-art photorealism for years. StyleGAN3 (2021) then solved a subtler problem — texture "sticking" to pixel coordinates during animation — by making the generator equivariant to translation and rotation, important for video and interpolation.

**BigGAN (Brock, Donahue, Simonyan, 2018)** went the other direction: instead of one domain at high fidelity, it scaled *class-conditional* generation across all 1000 ImageNet classes. The recipe was brute force done carefully — very large batch sizes (up to 2048), large models, the **truncation trick** applied to $z$, and **orthogonal regularization** to keep training stable at that scale. BigGAN reached an Inception Score around 166 and an FID around **7.4** on 128×128 ImageNet, which was a stunning result for conditional generation and the clearest demonstration that GANs scaled with compute. It also made the trilemma trade explicit: BigGAN's own paper reports that turning the truncation knob trades FID against Inception Score (coverage against per-sample quality) along a smooth curve.

So by 2019, GANs could do photorealistic megapixel faces (StyleGAN2) *and* diverse 1000-class ImageNet (BigGAN). They were not a struggling technology. They were the best image generators in the world. And then they lost anyway.

## 8. The honest post-mortem: why diffusion overtook GANs

The turning point has a paper attached to it: "Diffusion Models Beat GANs on Image Synthesis" (Dhariwal & Nichol, 2021). The title was a deliberate flag in the ground, and the result backed it up — a diffusion model (ADM, with classifier guidance) beat BigGAN on ImageNet FID at multiple resolutions. Within a year, latent diffusion (Stable Diffusion) made the approach cheap enough to run on consumer hardware, and the field's center of gravity moved entirely. Here is the honest accounting of *why,* axis by axis.

![Matrix comparing GANs and diffusion across sample quality, mode coverage, training stability, steps to sample, and controllability](/imgs/blogs/generative-adversarial-networks-and-why-they-lost-4.png)

**Mode coverage — the decisive axis.** This is the one that ended it. Diffusion's per-example denoising objective forces coverage; the GAN objective fights against it forever (Sections 3 and 5). For *unconditional* or *open-domain text-to-image* generation, where you genuinely need the full diversity of the data — every species of dog, every art style, every composition — GANs' mode collapse was disqualifying. You cannot ship a text-to-image product that produces the same three compositions for "a castle." Diffusion produces a different valid castle every seed, with no special effort. This single property is why the open-source text-to-image ecosystem is built entirely on diffusion.

**Training stability.** Diffusion training is a regression: add known noise to a real image, predict the noise, take the $L_2$ error, repeat. There is no second network, no minimax, no equilibrium to balance, no learning-rate ratio to tune between two adversaries. You can scale the model and the data and it just keeps improving — the loss curve goes down monotonically and boringly. This is enormous for *engineering velocity.* A team can iterate on a diffusion model the way they iterate on a classifier. Iterating on a GAN means babysitting a fragile dynamical system, and that cost compounds across every experiment.

**Controllability.** Because diffusion generates iteratively over many steps, you have *many points of intervention.* Classifier-free guidance steers every step toward the text condition. ControlNet injects structural conditioning (depth, pose, edges) at every layer. SDEdit, inpainting, and image-to-image all exploit the fact that you can start the reverse process from a partially-noised real image. A GAN generates in one shot from a latent; your only handle is that latent, and editing means finding the right point in $\mathcal{W}$ — powerful for faces (where StyleGAN editing genuinely shines), but far less flexible than per-step conditioning for arbitrary prompts and structure. The entire conditioning-and-control track of modern image generation is native to diffusion and awkward for GANs.

**Likelihood-ish training and theory.** Diffusion has a clean variational interpretation (it optimizes a bound on the data likelihood) and an equivalent score-matching / SDE formulation. This is not just aesthetic — it means the objective is *principled and decomposable,* which is why innovations like v-prediction, min-SNR weighting, and flow matching could be derived and slotted in. GAN theory, by contrast, is a series of patches on an objective that is flat where it matters.

Where GANs were *not* beaten, and this matters for fairness: **per-image sharpness and single-image quality.** At the moment of the switch, a well-tuned StyleGAN2 face was arguably sharper than an early diffusion face, and GANs had no iterative sampling cost. The thing diffusion gave up to win everything else was **speed** — and that brings us to the comeback.

It is worth being precise about *why* the switch was so total and so fast, because "diffusion is better" undersells the dynamics. Three forces compounded. **First, the coverage advantage made diffusion the only viable base for open-domain text-to-image,** which is the application that drew in the entire research and product world — once the money and attention were in text-to-image, the model class that could actually do it diverse-and-controllable won by default. **Second, the stability advantage made diffusion radically easier to scale,** so the rate of improvement was faster: a team could pour data and compute into a diffusion model and watch FID drop predictably, whereas scaling a GAN meant re-tuning a fragile equilibrium at every size. Compounded over a year, that difference in iteration speed is enormous. **Third, the open-source release of Stable Diffusion in August 2022 put a high-quality, controllable, fine-tunable diffusion model on consumer GPUs,** and an explosion of community work — LoRA, ControlNet, ComfyUI, a thousand fine-tunes — accreted on top of it. None of that ecosystem could have formed around a model class that was this hard to fine-tune without collapse. The GAN didn't just lose a benchmark; it lost the *platform war*, because its structural flaws made it a poor foundation to build a community on top of. By the time GANs might have closed the quality gap, the entire tooling ecosystem had moved.

There is one more honest point to make, lest this read as triumphalism. Diffusion did not win because GANs were a bad idea — GANs were a *great* idea that ran into a hard structural wall (the JS-flatness / coverage problem) that no amount of engineering fully cleared. Diffusion won because its objective sidesteps that wall entirely, at the price of slow sampling. The field then spent 2023-2025 attacking diffusion's one weakness (speed), and the most effective weapon it found was — of all things — the discriminator. The wheel turns.

## 9. Where GANs still win: one-step speed and the distillation comeback

Here is the redemption arc, and it is the reason this post belongs in a diffusion series. The one axis of the generative trilemma where GANs are *unbeatable* is **sampling speed**: a GAN generates an image in a **single forward pass.** Diffusion, in its naive form, needs 25 to 1000 sequential network evaluations. That is a 25-to-1000× latency gap, and for real-time applications — interactive generation, video frame rates, on-device synthesis — it is decisive. So the question that defined diffusion-acceleration research became: *can we get diffusion's coverage and quality but a GAN's single step?*

The answer, beautifully, is to bring the discriminator back. Not to build a GAN, but to use the **adversarial loss as a distillation signal** that compresses a slow, high-quality diffusion *teacher* into a fast, few-step *student.* The discriminator that lost the war returns as the tool that makes the winner fast.

![Graph showing a diffusion teacher and a few-step student generator, with a discriminator providing an adversarial loss alongside a distillation loss to update the student toward one-step sampling](/imgs/blogs/generative-adversarial-networks-and-why-they-lost-7.png)

The intuition for *why you need the adversarial term* is sharp. If you distill a multi-step diffusion model into one step using only a regression loss (make the student's one-step output match the teacher's many-step output), you hit the exact same blur problem that plagued VAEs: a regression-to-the-mean over the teacher's plausible outputs produces a soft, averaged image. The adversarial loss fixes this precisely because a discriminator *rewards landing on the data manifold,* not matching a mean — it pushes the student's output to be a *sharp, specific real-looking image* rather than the blurry average of all the images the teacher might have produced. GANs are good at sharpness for the same structural reason they are bad at coverage; in distillation, you have already solved coverage (the teacher provides it) and you only need the sharpness. It is the perfect division of labor.

Three landmark methods, all shipping:

- **ADD — Adversarial Diffusion Distillation (SDXL-Turbo, Sauer et al., 2023).** Distills SDXL into a 1-to-4-step model using a combination of a score-distillation term (match the teacher) and an adversarial term (a discriminator built on a frozen DINOv2 feature backbone, judging real vs student). SDXL-Turbo generates a 512×512 image in a **single step** in roughly 0.1 seconds on an A100, at quality that is genuinely usable — a step-count reduction of 25-50× versus the SDXL teacher. The adversarial term is what keeps the single-step output sharp.
- **DMD and DMD2 — Distribution Matching Distillation (Yin et al., 2023/2024).** Matches the student's output *distribution* to the teacher's by using the gradient of the KL between them (estimated via two score networks), and DMD2 adds a GAN loss to close the remaining quality gap and remove the need for an expensive precomputed dataset. DMD2 reports one-step generation competitive with the multi-step teacher; on ImageNet 64×64 the one-step FID is in the low single digits, and on SDXL it produces a one-to-four-step model at near-teacher quality.
- **GigaGAN (Kang et al., 2023) and LADD (Latent ADD, used in SD3-Turbo).** GigaGAN scaled a *pure* GAN to a billion parameters for text-to-image and high-resolution upscaling, demonstrating that GANs can still scale and that their one-step upscaler is extremely fast and sharp — GAN-based super-resolution and inpainting remain genuinely competitive because those tasks are conditioning-rich (less reliant on the generator inventing diverse global structure). LADD moves ADD's adversarial distillation into latent space for efficiency.

You can run the end product of this research in a few lines today. Here is SDXL-Turbo (the ADD-distilled model) generating an image in a *single* step via 🤗 `diffusers` — note the `num_inference_steps=1` and `guidance_scale=0.0`, both of which would produce garbage from the un-distilled SDXL but work precisely because the adversarial distillation reshaped the model to land on the data manifold in one shot:

```python
import torch
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# ADD-distilled: 1 step, NO classifier-free guidance (guidance_scale=0.0).
# The discriminator-trained model is sharp in a single forward pass.
image = pipe(
    prompt="a photograph of a red fox in a snowy forest, golden hour",
    num_inference_steps=1,
    guidance_scale=0.0,
).images[0]
image.save("fox_one_step.png")
```

Two details encode the whole comeback. **`num_inference_steps=1`** is the single-step sampling that only a GAN — or a GAN-distilled diffusion model — can do at quality. **`guidance_scale=0.0`** turns off classifier-free guidance, which the distilled model does not need (and which would actually hurt it, since the adversarial training already baked in prompt adherence and over-saturation control). Try the same call on base `stabilityai/stable-diffusion-xl-base-1.0` with `num_inference_steps=1` and you get noise-flecked mush; the difference is entirely the adversarial distillation. That is the discriminator earning its keep, visible in two keyword arguments.

We go deep on exactly how the adversarial term is constructed, what it is distilling against, and the one-step FID numbers in the dedicated post on [distribution matching and adversarial distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation). The takeaway here is conceptual and a little poetic: the adversarial game did not die. It was the wrong *primary* objective for general generation — too unstable, too prone to collapse — but it is the *right auxiliary* objective for making a model that already has coverage produce sharp images in one step. GANs lost as a noun and won as an adjective.

## 10. Case studies: real numbers

Let me ground all of this in measured results from the literature, with sources, so the comparison is concrete rather than vibes. Be aware FID is benchmark- and resolution-specific — these are headline numbers on their stated datasets and not directly comparable across different reference sets.

**StyleGAN2 on FFHQ (1024×1024).** FID ≈ **2.84** on the 70k-image FFHQ face dataset (Karras et al., 2019). This was state-of-the-art photorealistic face generation and is still an excellent number; for *narrow-domain, high-resolution* generation, a well-trained StyleGAN remains hard to beat on per-image sharpness, and faces are the canonical case where mode collapse is least punishing (faces are a relatively unimodal, well-structured domain).

**BigGAN on ImageNet (128×128).** FID ≈ **7.4**, Inception Score ≈ **166** (Brock et al., 2018), the strongest conditional GAN result of its era. BigGAN's truncation sweep traces an explicit FID-vs-IS (coverage-vs-quality) frontier — turn the truncation up and IS rises while FID worsens, the trilemma in a single plot.

**Diffusion beats GANs (ADM, 2021).** Dhariwal & Nichol's guided diffusion reached FID ≈ **4.59** on ImageNet 256×256 (and lower at 128×128), beating BigGAN-deep, which was the empirical event that flipped the field. The cost was sampling steps — ADM used 250 steps — which is precisely the speed deficit that motivated the distillation work.

**The distillation comeback (one-step text-to-image).** SDXL-Turbo (ADD) produces a usable 512×512 image in a **single** U-Net evaluation versus SDXL's 25-50, roughly **0.1 s/image** on an A100 versus a few seconds — a 25-50× latency win, with the adversarial term carrying the sharpness. DMD2 reports one-step ImageNet 64×64 FID in the low single digits, approaching the multi-step teacher. These are the numbers that prove the discriminator earned its place back.

A word on **measuring any of these honestly**, because FID is more treacherous than its single-number simplicity suggests. FID is the Fréchet distance between two multivariate Gaussians fit to Inception-v3 features of the real and generated sets: $\mathrm{FID} = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$. To get a number you can trust and compare, fix several things. **Sample size:** FID is biased downward (looks better) with more samples and is unstable below ~10k; the convention is **50,000 generated samples** against the full reference set. **Reference set:** FID is only meaningful against a *specific* reference distribution — FFHQ FID and ImageNet FID are different universes and must never be compared. **Preprocessing:** resize, crop, and the exact Inception checkpoint all shift the number by points; mismatched preprocessing is the single most common way published FIDs fail to reproduce. **Seed and warm-up:** fix the random seed for reproducibility and discard any warm-up batches. And remember the deeper limitation: FID compresses everything into two moments of one frozen feature space (Inception, trained on 2015-era ImageNet), so it is blind to failure modes those features don't encode — which is exactly why the field added FID-DINOv2, CLIP-score, and human-preference metrics like HPSv2. A GAN that games Inception features could post a deceptively good FID; cross-checking with a second metric is not optional for a serious comparison.

#### Worked example: would I train a GAN for this product?

Put yourself in a real decision. You are building an avatar generator for a chat app: users type a short description, you generate a stylized portrait, and **latency must be under 200 ms** because it runs inline in a conversation, on a shared A10G-class GPU. Walk the decision tree. Open-domain text-to-image rules out a from-scratch GAN (coverage). A vanilla 30-step SDXL sample is ~2-4 s on an A10G — far over budget. So the move is *diffusion for coverage and control, distilled with an adversarial loss for speed*: take SDXL (or SD3) as the teacher, apply a LoRA fine-tune for your house style, then distill to a 2-4 step LCM/Turbo-style student with an ADD-style discriminator term. That lands you around 0.2-0.4 s/image — inside budget — with full prompt coverage and your style, and the adversarial term keeps the 2-step output from going soft. Notice you touched a GAN exactly once, as a *distillation loss*, and never as a standalone model. That is the canonical 2026 shape of "should I use a GAN": not as the generator, but as the term that makes a diffusion generator fast enough to ship. The standalone GAN you would only reach for if the domain were narrow (just faces, just textures) and the per-image sharpness ceiling genuinely mattered more than diversity and control.

#### Worked example: estimating the latency win from step count

Suppose your SDXL pipeline runs the U-Net at **0.06 s per step** on an RTX 4090 (a defensible order-of-magnitude for 1024×1024 latents with `torch.float16` and SDPA attention). A 30-step DPM-Solver++ sample costs $30 \times 0.06 \approx 1.8$ s of denoising, plus a fixed VAE-decode of maybe 0.1 s, so about **1.9 s/image.** Distill to a 4-step Turbo/LCM-style model and the denoising drops to $4 \times 0.06 = 0.24$ s, total **≈ 0.34 s/image** — a roughly **5.6× speedup.** Push to a true one-step ADD model and denoising is $1 \times 0.06 = 0.06$ s, total **≈ 0.16 s/image**, about **12× faster** than the 30-step baseline. The discriminator-trained student is what keeps that one-step image from collapsing into VAE-style mush — without the adversarial loss, the regression-distilled one-step output would be visibly soft. That is the entire commercial case for adversarial distillation in three lines of arithmetic. (We push this latency-vs-quality frontier much harder in the speed track posts.)

## 11. When to reach for a GAN (and when not to)

A decisive recommendation section, because every technique is a cost. The honest 2026 guidance:

![Decision tree for choosing a GAN versus diffusion, branching on whether single-step latency is critical or coverage and control dominate](/imgs/blogs/generative-adversarial-networks-and-why-they-lost-8.png)

**Reach for a GAN (or a GAN loss) when:**

- **Single-step latency is the dominant constraint.** Real-time interactive generation, on-device synthesis, video frame rates. A one-step GAN — or a one-step *adversarially-distilled* diffusion model — is the only thing that hits these budgets. This is the live, important use case.
- **You are doing few-step distillation.** If you are compressing a diffusion model to 1-4 steps, you almost certainly want an adversarial term (ADD/DMD2 style) to keep the output sharp. This is the most common reason a 2026 engineer touches a discriminator at all.
- **The task is conditioning-rich super-resolution or inpainting.** GAN losses remain competitive and fast where most of the structure is given and the generator only fills detail (GigaGAN upscaling, real-time SR). Less diversity is required, so mode collapse hurts less.
- **Narrow, well-structured domain with a quality ceiling that matters more than diversity.** A high-resolution face or texture generator where StyleGAN's per-image sharpness and clean latent editing are genuinely best-in-class, and the domain is unimodal enough that collapse is manageable.

**Do NOT reach for a GAN when:**

- **You want general, open-domain text-to-image.** Use diffusion (SDXL, SD3, FLUX). Mode collapse and training fragility make a from-scratch GAN the wrong tool; you would be fighting the objective for coverage you'd get for free from diffusion.
- **You need strong controllability** — ControlNet-style structural conditioning, per-step guidance, inpainting/editing workflows. These are native to diffusion's iterative process and awkward to bolt onto a one-shot generator.
- **You value engineering velocity over peak single-step latency.** If you can afford 4-30 steps, diffusion's stable training will let your team iterate far faster than babysitting a minimax game. For most products, that is the right trade.
- **You don't already have a strong teacher.** The adversarial-distillation magic depends on a high-quality diffusion teacher to provide coverage; a discriminator alone won't give you a good generator without one. Train (or download) the diffusion model first, distill second.

The meta-rule: **in 2026 you rarely train a GAN as a standalone generator, but you frequently use an adversarial loss as one term in a diffusion-distillation recipe.** That is the practical residue of everything in this post.

One caution about following that rule too literally: do not reach for the adversarial term *first* when distilling. The cleanest few-step results come from a *layered* recipe — start with a regression or score-distillation objective to get the student into the right neighborhood (cheap, stable), then add the adversarial loss to sharpen the last mile. Leading with a discriminator reintroduces all the instability we spent this post cataloguing, with none of the coverage cushion a teacher provides until the student is already close. Consistency distillation and LCM, which we cover in the speed track, get to 4-8 steps with *no* adversarial term at all; you only add the discriminator when you are pushing to 1-2 steps and the regression-mean blur becomes visible. So the honest hierarchy is: try step-reduction without a GAN loss first, and only spend the instability budget of a discriminator when the step count gets low enough that nothing else keeps the image sharp. The discriminator is a powerful, slightly dangerous tool — bring it in late, for the specific job it is uniquely good at, and keep it on a short leash with spectral normalization and a balanced update schedule.

## 12. Key takeaways

- **A GAN is a game, not a likelihood model.** The generator never sees data directly; its only signal is the gradient through the discriminator, which makes GAN samples sharp (no reconstruction-blur pressure) and GAN training unstable (a moving, equilibrium-seeking target).
- **The optimal discriminator turns the game into JS-divergence minimization:** $D^*(x) = \frac{p_{\text{data}}}{p_{\text{data}} + p_g}$ and $\min_G V(D^*, G) = -\log 4 + 2\,\mathrm{JS}(p_{\text{data}} \| p_g)$. This is the elegant theory — and the seed of the instability.
- **JS divergence is flat on disjoint supports** (constant $\log 2$, zero gradient), and image manifolds are always nearly disjoint, so the generator gets no useful direction. This is the analytic root of vanishing gradients.
- **The Wasserstein/Earth-Mover distance fixes it** by measuring transport cost (smooth everywhere), enforced via a 1-Lipschitz critic and, in practice, a gradient penalty with $\lambda = 10$ (WGAN-GP). Remember to drop BatchNorm in the critic.
- **Mode collapse is the structural flaw that lost the war:** the adversarial objective rewards plausibility, never diversity, so the generator can collapse onto a few modes. Diffusion's per-example denoising loss has coverage built in; GANs have to bolt it on, and the bolts keep loosening.
- **The architecture line peaked high** — DCGAN (conv recipe) → ProGAN (progressive growth) → StyleGAN2 (disentangled $\mathcal{W}$, AdaIN, truncation trick, FFHQ FID ≈ 2.8) → BigGAN (ImageNet FID ≈ 7.4) — and GANs lost while at their best, not in decline.
- **Diffusion overtook GANs on coverage, stability, and controllability,** conceding only one-step speed. "Diffusion Models Beat GANs" (2021) was the flag in the ground; latent diffusion made it cheap.
- **GANs came back as a loss, not a model:** the adversarial term in ADD (SDXL-Turbo), DMD2, and GigaGAN keeps few-step distilled diffusion *sharp* (a discriminator rewards landing on the data manifold, defeating regression-to-the-mean blur). The discriminator that lost makes the winner fast.
- **The practical 2026 rule:** rarely train a standalone GAN; frequently use an adversarial loss to distill a diffusion model to 1-4 steps, or for conditioning-rich super-resolution. Use diffusion for general, controllable, diverse generation.

## 13. Further reading

- **Goodfellow et al., "Generative Adversarial Nets" (2014)** — the original minimax formulation and the optimal-discriminator / JS-divergence proof.
- **Radford, Metz, Chintala, "Unsupervised Representation Learning with Deep Convolutional GANs" (DCGAN, 2015)** — the convolutional recipe and latent arithmetic.
- **Arjovsky, Chintala, Bottou, "Wasserstein GAN" (2017)** and **Gulrajani et al., "Improved Training of Wasserstein GANs" (WGAN-GP, 2017)** — the Earth-Mover distance and the gradient-penalty fix.
- **Karras et al., "A Style-Based Generator Architecture for GANs" (StyleGAN, 2018)** and **"Analyzing and Improving the Image Quality of StyleGAN" (StyleGAN2, 2019)** — the disentangled $\mathcal{W}$ space, AdaIN, truncation, and weight demodulation.
- **Brock, Donahue, Simonyan, "Large Scale GAN Training" (BigGAN, 2018)** — scaling class-conditional GANs on ImageNet.
- **Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (2021)** — the empirical turning point.
- **Sauer et al., "Adversarial Diffusion Distillation" (SDXL-Turbo, 2023)** and **Yin et al., "One-step Diffusion with Distribution Matching Distillation" (DMD / DMD2, 2023/2024)** — the adversarial comeback as a distillation loss.
- **Within this series:** [why generating images is hard](/blog/machine-learning/image-generation/why-generating-images-is-hard) (the manifold hypothesis and the four-family map), [the mathematics of image distributions](/blog/machine-learning/image-generation/the-mathematics-of-image-distributions) (JS/KL divergence and what FID measures), [distribution matching and adversarial distillation](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation) (GANs return as a distillation signal), and the capstone [building an image generation stack](/blog/machine-learning/image-generation/building-an-image-generation-stack).
