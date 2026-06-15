---
title: "Why do neural scaling laws exist? Manifolds, spectra, and Zipf tails"
date: "2026-06-15"
description: "Understand the three independent mechanisms that each derive a neural scaling power law, why that convergence makes scaling laws so robust, and why cleaner lower-dimensional data scales faster."
tags: ["scaling-laws", "power-laws", "data-manifold", "intrinsic-dimension", "spectral-decay", "random-features", "zipf", "kernel-methods", "sharma-kaplan", "bahri", "hutter", "deep-learning"]
category: "machine-learning"
subcategory: "Scaling Laws"
author: "Hiep Tran"
featured: true
readTime: 53
---

Every other post in this series has treated the scaling law as a measured fact. We fit $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$, we read off the exponents, we extrapolate, we spend the budget. The whole edifice — [Kaplan](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models), [Chinchilla](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling), [data-constrained scaling](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws), [inference-aware scaling](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) — rests on the empirical observation that loss falls as a power of scale, and that the resulting line on log-log axes is straight enough to bet money on. But we never answered the question a curious physicist asks first: **why a power law?** Why not an exponential, a logarithm, a sigmoid with a knee somewhere? Of all the functional forms a learning curve could take, why does the universe keep handing us $N^{-\alpha}$?

This is the capstone, and it answers that question three times over. The remarkable thing — the thing that should make you trust scaling laws more, not less — is that there is not one explanation but **three independent ones**, each starting from a different property of the data, each arriving at the same power law. A power law is what you get from the geometry of the data manifold. A power law is what you get from the decay of the data's covariance spectrum. A power law is what you get from the heavy Zipf tail of feature frequencies. Three lenses, three derivations, one answer. When three unrelated arguments converge on the same conclusion, the conclusion is not an artifact of any one argument — it is structural. The diagram below is the mental model for the whole post: three mechanisms, each fed by the same structured data, each producing the same $L = E + A\,N^{-\alpha}$.

![A layered graph showing structured training data feeding three independent mechanisms, manifold geometry, spectral decay, and Zipf feature frequency, each producing an exponent that converges on the same neural scaling power law](/imgs/blogs/why-power-laws-arise-1.png)

The three branches in that diagram are the three sections that form the spine of this post. On the left, the geometric mechanism of Sharma and Kaplan (2020): a network does piecewise regression over a $d$-dimensional data manifold, and the exponent comes out as $\alpha \approx 4/d$. In the middle, the spectral mechanism of Bahri et al. (2021) and Maloney et al. (2022): the eigenvalue spectrum of the data covariance decays as a power law, and the loss inherits that decay. On the right, the frequency mechanism of Hutter (2021): features arrive with Zipf-distributed frequencies, and that heavy tail alone forces a power-law learning curve with exponent $\beta = \alpha/(1+\alpha)$. They are not the same argument in three costumes; they make different assumptions and they can even disagree on the precise exponent. That they agree on the *form* is the deepest fact in the field.

> [!important]
> **The capstone takeaways**
> - **There are three independent derivations of the neural scaling power law**, from manifold geometry, from spectral decay, and from Zipf feature frequencies. Their convergence is *why* scaling laws are so robust across modalities — it is overdetermined, not a coincidence of any one model.
> - **Geometry (Sharma & Kaplan 2020):** an $N$-parameter network piecewise-fits a $d$-dimensional data manifold into $\sim N$ regions; anchor spacing is $\sim N^{-1/d}$, local error squares it, and loss curvature adds a factor giving $\boxed{\alpha \approx 4/d}$. The exponent depends on **intrinsic** dimension, not ambient.
> - **Spectrum (Bahri 2021, Maloney 2022):** in the wide-network limit the kernel eigenvalues decay as $\lambda_i \propto i^{-(1+\alpha_K)}$ with $\alpha_K \propto 1/d$, and the loss inherits that decay; the data-covariance spectrum's tail *is* the scaling exponent. A clean duality holds: $\alpha_P = \alpha_D = \alpha_K$.
> - **Four regimes (Bahri 2021):** variance-limited scaling gives the *generic* exponent 1 ($L - L_\infty \propto D^{-1}$ or $P^{-1}$) from a well-behaved infinite limit; resolution-limited scaling is the interesting, data-dependent regime where $\alpha \propto 1/d$.
> - **Frequency (Hutter 2021):** Zipf-distributed feature frequencies $P(i) \propto i^{-(1+\alpha)}$ produce a learning curve $\text{error} \propto n^{-\beta}$ with $\boxed{\beta = \alpha/(1+\alpha)}$, loss-independent. $\alpha=1$ gives $\beta = 1/2$; $\alpha \to \infty$ gives $\beta \to 1$.
> - **The one number to remember: $\alpha \approx 4/d$.** Better data has lower intrinsic dimension $d$, so it scales with a *larger* exponent — which is the mechanistic reason curation buys you compute (the [data-quality](/blog/machine-learning/scaling-laws/data-quality-scaling-laws) lever) and why the [broken laws](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws) are breaks *between* these clean power-law regimes.

A quick note on scope and honesty before we start. These are *theories*. They make idealizing assumptions — smooth targets, infinite width, clean power-law spectra — and real models violate those assumptions in ways that produce the kinks, plateaus, and double-descent bumps that the [broken neural scaling laws](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws) post is entirely about. The clean theories here explain *why each segment is a power law*; the broken-law machinery explains *where the segments join*. Read this post and that one as complements: this is the physics of a single regime, that is the engineering of stitching regimes together.

## Why "why" is the right question to ask

**Senior rule of thumb: a curve you can extrapolate is a curve you understand the mechanism of; everything else is interpolation dressed up as prophecy.**

Here is the uncomfortable truth about fitting curves. If all you do is fit $L = E + A\,N^{-\alpha}$ to a handful of points and extend the line, you are trusting that the *functional form* stays valid in the region you have not measured. That trust is only as good as your reason to believe the form. If the power law is a coincidence — if the data merely *looked* power-law over the range you tested — then your extrapolation is a guess. If the power law is a *consequence* of a structural property of the data and the architecture, then your extrapolation has teeth, because you can reason about whether that structural property still holds at the new scale.

This is not academic. The single most expensive mistake in scaling is extrapolating across a break you did not know was coming: you fit a shallow exponent on three small runs, you predict the big run will land at loss $X$, and instead it saturates early because the model exhausted the useful resolution of your data. The [Kaplan-versus-Chinchilla reconciliation](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation) is in large part a story about this: two labs measured exponents in different regimes and the lines disagreed. Understanding *why* the law has the exponent it has tells you which regime you are in and when the regime is about to change.

So we want a mechanism. A good mechanistic explanation of scaling should do four things. First, it should *derive* the power-law form $N^{-\alpha}$ rather than assume it. Second, it should *predict the exponent* from something measurable about the data — not just produce "a power law" but produce $\alpha = f(\text{data property})$. Third, it should *explain the modality-independence*: why the same form holds for language, vision, audio, and reinforcement learning. Fourth, it should *connect to practice*: it should tell you what makes one dataset scale better than another. The three mechanisms below each clear all four bars, by different routes.

| Property a good theory needs | Geometry (Sharma & Kaplan) | Spectrum (Bahri, Maloney) | Frequency (Hutter) |
|---|---|---|---|
| Derives the power-law form | Yes — from partition + Taylor remainder | Yes — from spectral sum convergence | Yes — from tail-sum of Zipf masses |
| Predicts the exponent | $\alpha \approx 4/d$ | $\alpha = \alpha_K \propto 1/d$ | $\beta = \alpha/(1+\alpha)$ |
| Measurable data property | intrinsic dimension $d$ | covariance eigenvalue decay | feature-frequency Zipf exponent |
| Explains modality independence | all natural data lives on low-$d$ manifolds | natural-data covariance is heavy-tailed everywhere | natural symbols are Zipfian everywhere |
| Practical lever | reduce $d$ by curation | steepen the spectrum by curation | put mass on high-value features |

Notice the last column of that table is the same lever three times: clean your data. Each mechanism gives a different formal name to "cleaner data scales better," and we will close the loop on that in the final section. First, the geometry.

### A short history of the "why" question

It helps to know the order in which the field figured this out, because the theory papers were a deliberate response to an empirical mystery, not abstract math done in a vacuum. The empirical scaling curve came first. Hestness and collaborators in 2017 ("Deep Learning Scaling is Predictable, Empirically") measured power-law generalization curves across machine translation, language modeling, image classification, and speech, and noted — almost in passing — that the *exponents* were small and oddly consistent, and that nobody could explain where they came from. Kaplan et al.'s 2020 "Scaling Laws for Neural Language Models" turned that observation into the engineering tool the rest of this series is built on, fitting clean power laws in $N$, $D$, and compute. But Kaplan's paper was explicitly phenomenological: it measured the exponents and used them, and stated plainly that the *origin* of the power-law form was an open question.

The theory papers we cover here all landed within about two years of that, and they are best read as three teams attacking the open question from three angles at once. Sharma and Kaplan (2020, the same Kaplan) went geometric, asking what property of the data could produce the observed exponents, and pinned it to manifold dimension. Bahri, Dyer, Kaplan, Lee, and Sharma (2021) went spectral and taxonomic, separating the trivial variance-limited exponent from the interesting resolution-limited one and proving the kernel duality. Maloney, Roberts, and Sully (2022) went for an exactly-solvable model so the result could not be blamed on approximations. And Hutter (2021), working in parallel, showed you did not even need any of that continuous machinery — a Zipf tail alone sufficed. The convergence we celebrate in this post was not planned; four groups chasing the same empirical exponents arrived at the same functional form from premises that barely overlap. That is the historical fact that should make you trust the result.

One more piece of history matters for honesty: none of these theories predicts *exact* exponents for real models, and they were never claimed to. They predict the *form* (a power law), the *dependence* (on intrinsic dimension / spectral decay / Zipf tail), and the *direction* of every lever. The exact constant out front depends on smoothness, on architecture efficiency, on how well the optimizer resolves the spectrum — confounds the theories deliberately abstract away. So treat them as explaining why the line is straight and which way the slope tilts, not as a substitute for measuring the slope on small runs. The two approaches are complementary, and the practical sections at the end of this post lean on both.

## 1. Geometry: a scaling law from the dimension of the data manifold

**Senior rule of thumb: the exponent is set by the intrinsic dimension of the data, not by how many raw pixels or tokens you fed in — which is why the same network scales differently on different datasets.**

The cleanest mechanistic story is Sharma and Kaplan's 2020 paper, "A Neural Scaling Law from the Dimension of the Data Manifold" (arXiv:2004.10802). Their claim is startlingly specific: the loss falls as $L \propto N^{-\alpha}$ with

$$
\alpha \approx \frac{4}{d},
$$

where $d$ is the **intrinsic dimension** of the data manifold — the number of degrees of freedom the data actually varies in, not the ambient dimension of the representation. A $1024 \times 1024$ image lives in a million-dimensional pixel space, but the set of *natural* images is a thin, curved sheet of far lower intrinsic dimension; that lower number is the $d$ that goes into the exponent. The result holds for both mean-squared-error and cross-entropy losses, which is the first hint that something structural rather than loss-specific is going on.

The argument is a partition argument, and the figure below walks through it in three steps. I find the cleanest way to hold it in your head is to forget deep learning entirely and think about piecewise approximation, the way you would think about approximating a smooth function with a step function or a piecewise-linear spline.

![A three-step intuition figure showing a one-dimensional data manifold partitioned into cells by anchor points with spacing proportional to N to the minus one over d, a flow from partition to spacing to error, and a zoomed cell where a linear fit leaves an error that grows as the square of the cell width](/imgs/blogs/why-power-laws-arise-2.png)

### 1.1 The partition argument, step by step

Start with the manifold. The data you train on is not spread uniformly through the ambient representation space; it concentrates on a $d$-dimensional surface. The target you are trying to learn — next-token logits, a denoised image, a value function — is some reasonably smooth function defined on that surface.

Now think about what an $N$-parameter network does to that surface. A trained network with $N$ parameters has roughly $N$ degrees of freedom with which to carve the input space into regions, within each of which it behaves like a simple (locally near-linear) function. This is literally true for piecewise-linear networks — a ReLU network with $N$ parameters partitions input space into a number of linear regions that scales with $N$ — and it is morally true for smooth activations as well. So the network tiles the $d$-dimensional manifold into on the order of $N$ cells, dropping one "anchor" per cell where its local fit is best.

Here is the geometric crux. If you scatter $\sim N$ anchors as uniformly as you can over a $d$-dimensional region, the typical spacing between neighboring anchors is

$$
s \sim N^{-1/d}.
$$

The exponent $1/d$ is the only place the dimension enters, and it is the reason high-dimensional data scales poorly: in a $d$-dimensional space, $N$ points are *sparse*, and doubling $N$ barely tightens the spacing once $d$ is large. (With $d = 2$, going from $N$ to $4N$ halves the spacing. With $d = 32$, you need $N$ to grow by a factor of $2^{32}$ to halve it. That is the curse of dimensionality wearing a scaling-law hat.)

Now the error. Within one cell, the network approximates the smooth target $f$ by a simple local model. If the local model is linear and the target is smooth, the leading approximation error over a cell of width $s$ is set by the *second-order* Taylor remainder — the curvature of $f$ that a linear fit cannot capture. The remainder of a smooth function over an interval of size $s$ scales as $s^2$:

$$
\text{pointwise error} \sim s^2 \sim \left(N^{-1/d}\right)^2 = N^{-2/d}.
$$

Finally the loss. For mean-squared error (and, with a bit more care, for cross-entropy near the optimum), the loss is the *square* of the pointwise error, integrated over the manifold. Squaring $N^{-2/d}$ gives

$$
L \sim \left(N^{-2/d}\right)^2 = N^{-4/d},
$$

so $\alpha = 4/d$. The factor of 4 is exactly "2 from the Taylor remainder of a smooth fit, times 2 from squaring in the loss." That is the whole derivation. Cells go as $N$, spacing as $N^{-1/d}$, local error as the square of spacing, loss as the square of local error.

A natural worry at this point: that derivation visibly used mean-squared error (the "square" in the last step is the MSE square). Does it survive for cross-entropy, which is what language models actually optimize? It does, and the reason is instructive. Near the optimum, cross-entropy is locally quadratic in the logit error — its second-order Taylor expansion around the correct distribution is a weighted squared error, the Fisher-information quadratic form. So to leading order, a small logit error $\epsilon$ contributes a loss $\sim \epsilon^2$ for cross-entropy just as for MSE, and the squaring step goes through with the same exponent. This is why Sharma and Kaplan report $\alpha \approx 4/d$ for *both* losses, and why it is not a coincidence: any smooth, locally-quadratic loss inherits the same partition-and-square argument, which is a strong hint that the exponent is a property of the data geometry and not of the particular loss you chose to write down. A loss that was locally *linear* in the error (like absolute error) would change the inner exponent and hence the constant — but the dominant losses in practice are locally quadratic, so $4/d$ is the relevant idealization.

### 1.2 The conservative bound and the smooth-target result

It is worth being precise about two numbers that float around the literature and are easy to conflate. The $4/d$ above assumes a *smooth* target: the second-order Taylor remainder is what gives the inner factor of 2. If you make the weaker assumption that the local fit is merely piecewise-constant (a nearest-neighbor-style estimate, no curvature exploited), the inner exponent drops and you get the *conservative bound*

$$
\alpha \gtrsim \frac{1}{d}.
$$

So $1/d$ is a floor that holds with very mild assumptions, and $4/d$ is the stronger result you earn when the target is smooth enough to exploit curvature. Real exponents for real datasets sit between these, depending on target smoothness. When you see "$\alpha \propto 1/d$" in Bahri (next section) and "$\alpha \approx 4/d$" in Sharma–Kaplan, they are not contradicting each other — $1/d$ is the conservative scaling and $4/d$ is the smooth-target instance of the same $1/d$ proportionality. The proportionality $\alpha \propto 1/d$ is the robust claim; the constant out front depends on smoothness.

### 1.3 The experiment that makes it believable

The reason this is more than a just-so story is that Sharma and Kaplan *dialed $d$ and watched $\alpha$ track it*. In teacher–student experiments, you can build synthetic data whose intrinsic dimension you control exactly — generate inputs on a known $d$-dimensional manifold, define a teacher network as the target, train students of varying size. They found that the measured scaling exponent $\alpha$ moved as $\sim 4/d$ as they swept $d$, independently of the ambient dimension and independently of the specific teacher. That is the kind of evidence that distinguishes a mechanism from a curve fit: you can intervene on the proposed cause and the effect moves the predicted amount.

There is a sanity-check table worth internalizing here, because it is the single most practically useful consequence of the whole geometric picture. It shows how punishing intrinsic dimension is, and why "lower-dimensional data" is not a vague nicety but a concrete multiplier on how fast you improve.

![A matrix relating intrinsic dimension d to the smooth-target exponent four over d, the conservative exponent one over d, the loss improvement per ten times parameters, and the data regime, showing that low dimension gives steep fast-improving curves and high dimension gives shallow slow ones](/imgs/blogs/why-power-laws-arise-3.png)

Read the loss-drop column. With $d = 4$ and the smooth-target exponent $\alpha = 1$, a $10\times$ increase in $N$ cuts the reducible loss by a factor of $10^{-1} = 0.1$ — a $90\%$ reduction per decade of parameters. With $d = 32$ and $\alpha = 0.125$, the same $10\times$ in $N$ buys you only $10^{-0.125} \approx 0.75$, a measly $25\%$ reduction. Same architecture, same training, same compute multiplier — and the well-conditioned, low-dimensional dataset improves more than three times as fast per decade. That gap *is* the value of data quality, expressed in the only currency that matters, the exponent.

### 1.4 Second-order consequence: why ambient dimension is a red herring

The non-obvious payoff is that nothing in this argument depends on the ambient dimension. A tokenizer that produces a 50,000-way vocabulary, an image at $512^2$ resolution, an audio stream at 24 kHz — these set the ambient dimension, and the partition argument does not care. What it cares about is how many *independent directions the data actually varies in*. This is why upscaling your inputs (more pixels, longer context, bigger vocab) does not by itself change your scaling exponent: you have added ambient dimensions, not intrinsic ones. And it is why a domain with genuinely fewer degrees of freedom — clean code with rigid syntax, structured tabular data, a narrow task distribution — scales with a steeper exponent than open-domain web text, which has enormous intrinsic dimension because it varies in so many semantic directions at once.

### 1.5 Why the partition count really scales with parameters

The one step in the partition argument that deserves a closer look is the claim that an $N$-parameter network carves the input space into $\sim N$ regions, because it is the load-bearing assumption and it is the one most likely to make a skeptical reader squint. For piecewise-linear networks the statement is precise. A ReLU network is a continuous piecewise-linear function: the input space is cut by the hyperplanes at which individual ReLUs switch from off to on, and within each resulting cell the network is exactly affine. The number of such linear regions grows with the number of units (and therefore with parameters); for a single hidden layer of $H$ units in input dimension $d$ it is polynomial in $H$, and depth can compound this. The exact counting is delicate — the worst-case bounds are enormous and the typical-case counts much smaller — but the relevant fact for scaling is just the *order of magnitude*: more parameters means proportionally more affine pieces with which to tile the data, and the tiling is what the partition argument needs.

The reason this matters is that it tells you the partition picture is not a metaphor. The network really is doing piecewise approximation, the pieces really do multiply with parameters, and the data manifold really is being tiled. Once you accept that, the rest of the argument — spacing $\sim N^{-1/d}$, error $\sim$ spacing-squared, loss $\sim$ error-squared — is just calculus on a tiling. The deep-learning-specific parts (which units switch where, how training places the cuts) affect the *constant* in front of the exponent, by deciding how efficiently the available pieces get spent on the parts of the manifold that matter. They do not change the *power* $4/d$, which is fixed by geometry. This is the recurring theme: architecture and optimization move the prefactor, data geometry moves the exponent.

### 1.6 A second worked example: how many parameters to halve the loss

Practitioners reason in "how much more do I need to spend," so let us put the geometry in those terms. Suppose you want to *halve* your reducible loss — drop it to $50\%$ of its current value — and you know your exponent $\alpha$. The required parameter multiplier $m$ satisfies $m^{-\alpha} = 0.5$, so

$$
m = 2^{1/\alpha} = 2^{d/4} \quad\text{(smooth target)}.
$$

For $d = 8$ ($\alpha = 0.5$): $m = 2^{2} = 4\times$ the parameters to halve the loss. For $d = 16$ ($\alpha = 0.25$): $m = 2^{4} = 16\times$. For $d = 40$ ($\alpha = 0.1$): $m = 2^{10} = 1024\times$. The cost to halve the loss is *exponential in intrinsic dimension*, which is the curse of dimensionality stated in budget terms. It also tells you, brutally, that there is a regime of $d$ beyond which scaling is simply not a viable strategy — once $d$ is large enough that halving the loss costs a thousand-fold or a million-fold more parameters, the only lever left is to lower $d$, and that means data work, not model work. Every team that has hit a "scaling wall" on a messy domain has, knowingly or not, hit this exponent.

## 2. Spectrum: the four regimes and the kernel duality

**Senior rule of thumb: in the wide-network limit, the scaling exponent is just the decay rate of the data covariance's eigenvalue spectrum — so if you can measure the spectrum, you can predict the exponent without training a single large model.**

The geometric argument is intuitive but informal. The 2021 paper "Explaining Neural Scaling Laws" by Bahri, Dyer, Kaplan, Lee, and Sharma (arXiv:2102.06701, published in PNAS in 2024) makes it rigorous, and in doing so reveals something the partition picture hides: there is not one scaling regime but *four*, and only some of them are the data-dependent power law we have been discussing. The other two are a generic, almost trivial exponent that shows up for boring statistical reasons. Knowing which regime you are measuring in is the difference between a meaningful exponent and a meaningless one.

### 2.1 The four regimes

Bahri et al. organize scaling along two binary axes. The first axis: are you scaling the **dataset size $D$** or the **parameter count $P$**? The second axis: is the bottleneck **variance** or **resolution**? Crossing those two axes gives four cells, shown in the grid below.

![A two-by-two grid of the four scaling regimes from Bahri 2021, with variance-limited regimes giving exponent one for both data and parameter scaling and resolution-limited regimes giving the data-dependent exponent proportional to one over d](/imgs/blogs/why-power-laws-arise-4.png)

**Variance-limited regimes (the top row, exponent 1).** When you have plenty of capacity relative to the structure you are trying to resolve, the remaining error is dominated by *statistical variance* — the noise from having a finite sample or a finite-width network rather than the infinite-data, infinite-width ideal. In this regime the law is almost embarrassingly simple:

$$
L - L_\infty \propto D^{-1} \quad\text{(scaling data)}, \qquad L - L_\infty \propto P^{-1} \quad\text{(scaling parameters)}.
$$

The exponent is **1**, full stop, and it does not depend on the data at all. This is the same $1/D$ you know from the central-limit-theorem behavior of any well-behaved estimator: average over $D$ independent samples and the variance of your estimate falls as $1/D$. Bahri et al. show this follows generically from the existence of a smooth infinite-data (or infinite-width) limit. It is real, it is a power law, and it is *not the interesting one* — it tells you nothing about your data, only that your estimator is well-behaved. If you measure $\alpha \approx 1$, you are probably variance-limited, which means you have not yet hit the structure of the problem.

**Resolution-limited regimes (the bottom row, the interesting power law).** When the model is large enough that variance is no longer the bottleneck, the error is dominated by how finely the model can *resolve* the structure of the data manifold — exactly the partition picture from Section 1. Here the law is

$$
L \propto D^{-\alpha_D}, \qquad L \propto P^{-\alpha_P}, \qquad \alpha \propto \frac{1}{d},
$$

with the conservative $\alpha = 1/d$, and the smoother the target, the larger $\alpha$ gets, recovering $4/d$ at the smooth end. This is the data-dependent regime, the one that carries information about the manifold, the one whose exponent you actually want to forecast. When people quote "the scaling exponent of language models is around $0.07$," they are quoting a resolution-limited $\alpha_D$, and its smallness is telling you that natural-language data has a large effective intrinsic dimension.

The practical reading of this grid is a diagnostic. Measure your exponent. If it is near 1, you are variance-limited — add structure-resolving capacity (more parameters, longer training) before you trust the slope as a statement about your data. If it is well below 1 and stable, you are resolution-limited and the exponent is meaningful. Mixing the two — fitting a single line across the transition from variance-limited to resolution-limited — is a classic way to get an exponent that predicts nothing, and it is one of the mechanisms behind the kinks the [broken laws](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws) post catalogs.

### 2.2 The kernel/manifold duality

The deepest result in Bahri et al. is what happens in the **large-width limit**, where a neural network's training dynamics are governed by a fixed kernel (the neural tangent kernel, or a related feature kernel). In that limit they prove a clean equality of exponents:

$$
\alpha_P = \alpha_D = \alpha_K,
$$

where $\alpha_K$ is a property of the **kernel's eigenvalue spectrum**. Specifically, if you diagonalize the kernel and sort its eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots$, they decay as a power law of their rank:

$$
\lambda_i \propto i^{-(1+\alpha_K)},
$$

and that single spectral exponent $\alpha_K$ controls *both* the data-scaling and parameter-scaling laws. The duality is the unification: parameter scaling and data scaling are not two phenomena that happen to both be power laws, they are the *same* power law viewed through two different resource constraints, and the common rate is the spectral decay rate. The figure below traces the chain — manifold to kernel to spectrum to the single exponent.

![A graph showing the data manifold inducing a kernel in the wide-network limit, the kernel having a power-law eigenvalue spectrum, and the spectrum decay exponent equaling both the data and parameter scaling exponents, so one number wears three faces](/imgs/blogs/why-power-laws-arise-6.png)

And the loop closes back to geometry: $\alpha_K \propto 1/d$. The kernel induced by data on a $d$-dimensional manifold has a spectrum that decays at a rate set by $d$ — low-dimensional, smooth manifolds induce *fast*-decaying spectra (large $\alpha_K$, large exponent, fast scaling); high-dimensional, rough manifolds induce *slow*-decaying spectra (small $\alpha_K$, small exponent, slow scaling). The spectral picture and the geometric picture are two descriptions of one object. Section 1 told you the exponent in terms of how anchors tile a manifold; Section 2 tells you the same exponent in terms of how fast the kernel's eigenvalues fall off. They must agree, and they do.

This is worth pausing on because it is *the* unifying insight of the post. We have a single scalar — call it the intrinsic dimension $d$ — that simultaneously sets the anchor-spacing exponent ($1/d$), the loss exponent ($4/d$ for smooth targets), and the kernel spectral decay ($\alpha_K \propto 1/d$). Three different mathematical objects, one underlying cause. That is what it feels like when you have actually understood a phenomenon rather than merely fit it.

### 2.3 Where the power law comes from in the spectral picture

It is worth seeing, even informally, *why* a power-law spectrum produces a power-law loss, because it is the same calculus the geometric argument used, dressed in eigenvalues instead of cells. In a kernel regression with $D$ training points, the modes you can resolve are roughly the top $D$ eigenvalues of the kernel; the modes you cannot resolve — the tail beyond rank $D$ — contribute their eigenvalue mass to the error. So the residual loss is the tail sum of the spectrum:

$$
L(D) \;\approx\; \sum_{i > D} \lambda_i \;\sim\; \sum_{i > D} i^{-(1+\alpha_K)} \;\sim\; D^{-\alpha_K}.
$$

The integral (or sum) of a $i^{-(1+\alpha_K)}$ tail beyond rank $D$ is $\propto D^{-\alpha_K}$ — that is elementary, the same reason the tail of any $p$-series converges at a power-law rate. So the loss falls as $D^{-\alpha_K}$, and the same argument with the number of *features* in place of samples gives $P^{-\alpha_K}$. This is why the "$+1$" in the spectral exponent $1+\alpha_K$ matters and why it is there: a spectrum that decayed only as $i^{-1}$ (i.e. $\alpha_K = 0$) would have a logarithmically divergent tail and no clean power-law loss; you need the decay to be *faster* than $1/i$ for the tail to behave, and the amount by which it is faster is exactly the scaling exponent. The structure of the data — how quickly its modes lose importance — is the structure of the learning curve. Resolve one more mode per sample, and you chase a power-law-shrinking tail.

This tail-sum view also makes the variance-versus-resolution distinction concrete. Variance-limited scaling is what you get when the bottleneck is statistical noise in estimating the *resolved* modes (giving the generic $1/D$); resolution-limited scaling is what you get when the bottleneck is the *unresolved* tail (giving the data-dependent $D^{-\alpha_K}$). A real run is variance-limited early — too few samples to nail even the top modes — and crosses into resolution-limited later, once the head is pinned down and the tail dominates. The crossover between those two is one of the breaks the broken-law form is built to fit.

## 3. Spectrum, made exactly solvable: the random-feature model

**Senior rule of thumb: when a phenomenon survives in a model you can solve in closed form with no training and no approximation, it is a property of the structure, not of the optimizer or the architecture.**

The Bahri results rely on the wide-network limit and some technical assumptions about the spectrum. Maloney, Roberts, and Sully's 2022 paper, "A Solvable Model of Neural Scaling Laws" (arXiv:2210.16859), removes the hand-waving by building a model simple enough to solve *exactly* — and the scaling law falls right out of the algebra, with no training loop anywhere in sight.

The model is a random-feature regression. You take inputs, pass them through a fixed random linear map to produce features, and fit a linear readout on top. There is no gradient descent on the features; the only learning is the closed-form linear regression of the readout. Because everything is linear and the randomness is structured, you can write down the expected test loss as an explicit function of the number of features (the analogue of model size $P$) and the number of training samples (the analogue of data $D$). It is, mechanically, a computation about the spectrum of a random matrix.

The result is the cleanest statement of the spectral mechanism you will find. The **data covariance has an eigenvalue spectrum** — call its decay rate $1 + \alpha$ — and the test loss inherits *exactly that power-law decay* as you scale features or samples:

$$
\text{data covariance } \lambda_i \propto i^{-(1+\alpha)} \;\Longrightarrow\; L \propto (\text{scale})^{-\alpha}.
$$

The spectrum's tail and the loss's scaling are the same exponent. If you flatten the spectrum (make $\alpha$ smaller, the eigenvalues fall off more slowly), the loss scales more slowly. If you steepen it (larger $\alpha$, eigenvalues fall off fast), the loss scales faster. Nothing about optimizers, nothing about depth, nothing about nonlinearity — just the spectrum of the data, passed through a linear model, reproducing the scaling law. The figure below is the picture you should carry: the spectrum is a straight line on log-log axes, and its slope is the scaling exponent.

![A log-log plot of the data covariance eigenvalue spectrum showing a fast-decaying straight line for clean low-dimensional data and a slow-decaying line for noisy high-dimensional data, with the slope of the spectrum equal to the scaling exponent the loss inherits](/imgs/blogs/why-power-laws-arise-5.png)

### 3.1 The finite-dimension plateau

Maloney et al. also give a clean account of *where the power law ends*, which is exactly the kind of "break" the [broken neural scaling laws](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws) post models phenomenologically. In the solvable model there is a **finite latent dimension** — the data really does live on a finite-dimensional structure, even if it is large. As long as you are resolving the head of the spectrum, you scale as a power law. But once you have effectively resolved all the latent directions — once your model is big enough to capture every eigenvalue above the noise floor — there is nothing left to resolve, and performance *plateaus*. The power law was the behavior in the regime where spectrum still had untapped structure; the plateau is what happens when you run out of structure.

This is the spectral version of "the irreducible-error floor" from the [foundations post](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations), and it is mechanistically satisfying: the power-law middle and the flat floor of a learning curve are not two unrelated phenomena, they are the resolved-head and exhausted-tail phases of resolving one spectrum. The exponent tells you how fast you climb the spectrum; the latent dimension tells you when you reach the top.

### 3.2 Why solvability matters

It is tempting to dismiss a random-feature model as a toy. The opposite is true: its toy-ness is the point. Deep networks have a thousand confounds — optimizer dynamics, normalization, depth, attention, learning-rate schedules — any of which a skeptic could nominate as "the real reason" for the power law. Maloney et al. strip all of them away and the power law *survives*. That isolates the cause. The scaling law is not an emergent property of SGD on transformers; it is a property of regressing onto data whose covariance has a power-law spectrum, and transformers exhibit it because natural data has that kind of spectrum and transformers are, in the relevant regime, fitting it. The optimizer and architecture decide *which* spectrum you effectively see and how efficiently you resolve it, but the *form* of the law is upstream of all of them.

## 4. Frequency: power laws from Zipf tails

**Senior rule of thumb: you do not need a manifold or a kernel to get a power law — heavy-tailed feature frequencies alone are sufficient, which is why even the simplest memorization-style model scales.**

The third mechanism is the most elementary and, in some ways, the most surprising. Hutter's 2021 "Learning Curve Theory" (arXiv:2102.04074) shows that you can get a power-law learning curve $\text{error} \propto n^{-\beta}$ for *any* $\beta > 0$ from a model so simple it barely qualifies as machine learning — and the only ingredient you need is that the features (or skills, or facts) the model must learn arrive with **Zipf-distributed frequencies**.

### 4.1 The toy model

Picture the simplest possible learner. There is a countable set of distinct "features" the model must master — think of them as facts, n-grams, skills, or tail entities. The model learns a feature the moment it has seen at least one example of it, and gets that feature right forever after; it gets a feature wrong only if it has never seen it. There is no generalization across features, no smoothing, no clever representation — pure per-feature memorization. The test error is just the total probability mass of the features the model has *not yet seen* after $n$ training examples.

Now impose the empirical fact about natural data: feature frequencies follow a Zipf law. The $i$-th most common feature appears with probability

$$
P(i) \propto i^{-(1+\alpha)},
$$

for some Zipf exponent $\alpha > 0$. This is not an assumption you have to argue for — word frequencies, n-gram frequencies, entity frequencies, API-call frequencies, essentially every "vocabulary of discrete things drawn from natural usage" obeys a Zipf law. It is one of the most robust empirical regularities in all of data.

### 4.2 The derivation and the punchline

After $n$ training examples, the features you are likely to have seen are the common ones; the error is the mass of the rare tail you have not yet hit. A feature with probability $p$ is unlikely to have appeared in $n$ draws once $n p \lesssim 1$, i.e. for ranks beyond roughly $i^* \sim n^{1/(1+\alpha)}$. The residual error is the tail mass beyond that rank:

$$
\text{error}(n) \approx \sum_{i > i^*} P(i) \;\sim\; (i^*)^{-\alpha} \;\sim\; \left(n^{1/(1+\alpha)}\right)^{-\alpha} = n^{-\alpha/(1+\alpha)}.
$$

So the learning curve is a power law with exponent

$$
\boxed{\beta = \frac{\alpha}{1+\alpha}}.
$$

That mapping is the whole result, and it is beautiful in its simplicity. The figure below shows both halves: the Zipf frequency tail on the left, and the $\alpha \mapsto \beta$ mapping on the right.

![A two-panel figure showing on the left a Zipf feature-frequency distribution as a straight line on log-log axes and on the right the mapping from Zipf exponent alpha to learning-curve exponent beta equal to alpha over one plus alpha, with the point alpha equals one mapping to beta equals one half](/imgs/blogs/why-power-laws-arise-7.png)

The mapping has two memorable endpoints. When $\alpha = 1$ (a fairly heavy tail, common in language), $\beta = 1/(1+1) = 1/2$ — the classic square-root learning curve, error falling as $n^{-1/2}$. As $\alpha \to \infty$ (a very light tail, where almost all mass is on a few features), $\beta \to 1$ — error falling as fast as $n^{-1}$, because there is barely any tail to chase. And crucially, the result is **loss-independent**: Hutter shows it holds whether you measure error as 0-1 loss, squared loss, or otherwise, because the mechanism is about *which features you have seen*, not about how you score them. The power law is manufactured purely by the heavy tail of the frequency distribution.

### 4.3 Why this matters more than its simplicity suggests

It would be easy to file Hutter's model under "cute toy" and move on. Do not. It carries three lessons the fancier theories obscure.

First, it explains why even pure memorization scales. A model that does nothing but store facts it has seen still exhibits a clean power-law learning curve, as long as the facts are Zipfian. This means a chunk of the scaling we observe in real LLMs is plausibly *tail-fact acquisition* — the steady, power-law-paced absorption of rarer and rarer entities, idioms, and facts — rather than anything to do with clever generalization. The geometry and spectrum stories are about generalizing over a manifold; the Zipf story is about memorizing a tail; real models do both, and both yield power laws, which is part of why the empirical law is so clean.

Second, it predicts a knob. If $\beta = \alpha/(1+\alpha)$, then anything that makes the effective feature distribution *lighter-tailed* (more mass on high-value features, less on a long junk tail) increases $\alpha$ and therefore increases $\beta$ — faster scaling. That is, again, the data-quality lever, now phrased in the language of frequency: deduplication and filtering reshape the feature-frequency distribution, and a better-shaped distribution scales faster.

Third, it is the cleanest illustration of the post's thesis. The geometric and spectral mechanisms share machinery — they are really two views of the same kernel-on-a-manifold object. Hutter's mechanism shares *none* of that machinery; it has no manifold, no kernel, no continuity, no Taylor expansion. It is a completely different argument from a completely different starting point, and it still lands on a power law. When an argument that shares no premises with the others reaches the same conclusion, you have crossed from "plausible" to "overdetermined."

## 5. Why three mechanisms is the actual answer

**Senior rule of thumb: a result with one proof is a theorem; a result with three independent proofs from different premises is a law of nature — and that is exactly the epistemic status of neural scaling.**

We can now answer the title question properly. Why do neural scaling laws exist? Because **three independent structural properties of natural data each force a power law**, and natural data has all three at once.

- Natural data lies on a **low-intrinsic-dimension manifold**, and piecewise approximation over a $d$-dimensional manifold gives $L \sim N^{-4/d}$.
- Natural data has a **heavy-tailed covariance spectrum**, $\lambda_i \propto i^{-(1+\alpha)}$, and regression inherits that decay as $L \sim (\text{scale})^{-\alpha}$.
- Natural data has **Zipf-distributed feature frequencies**, $P(i) \propto i^{-(1+\alpha)}$, and even pure memorization of a Zipf tail gives $\text{error} \sim n^{-\alpha/(1+\alpha)}$.

These are not three statements of one fact. The first is differential-geometric (it is about smoothness and curvature on a surface), the second is linear-algebraic (it is about matrix eigenvalues), the third is combinatorial (it is about counting how much tail mass you have not yet sampled). They make different assumptions; two of them (manifold and spectrum) are deeply related through the kernel duality, but the third (Zipf) is genuinely separate. The reason the empirical scaling law is so absurdly robust — holding across language, vision, audio, video, diffusion, reinforcement learning, and alignment, across more than a dozen orders of magnitude — is that you would have to break *all three* mechanisms simultaneously to destroy it, and natural data breaks none of them.

There is one more pattern worth flagging, because it is too suggestive to ignore. Notice that the spectral mechanism and the Zipf mechanism both feature the *same* exponent form, $i^{-(1+\alpha)}$ — eigenvalues by rank in one case, feature probabilities by rank in the other. That is not a typo or a coincidence of notation. A heavy-tailed frequency distribution over features induces a heavy-tailed covariance spectrum (the common, high-frequency features dominate the top eigenvalues; the rare tail features populate the small eigenvalues), so a Zipf tail of exponent $1+\alpha$ produces a covariance spectrum that also decays roughly as $i^{-(1+\alpha)}$. The frequency mechanism and the spectral mechanism are therefore *not quite* independent — they are two views of the same heavy tail, one in probability space and one in eigenvalue space. The genuinely independent axis is the geometric one (smoothness and curvature on a manifold), which is why I have been careful to say "two of them are related through the kernel duality." Even with that caveat, you have at least two structurally distinct routes to a power law, and arguably two-and-a-half, which is more than enough to make the conclusion overdetermined. The Zipf-to-spectrum bridge is, if anything, a bonus: it shows the same heavy tail wearing two of the three masks, which is exactly the kind of internal consistency a correct theory should exhibit.

This is also the precise sense in which this post complements the [broken neural scaling laws](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws) post. Each of these clean theories describes a *single regime*: a smooth manifold of fixed dimension, a spectrum with a single power-law tail, a stationary Zipf distribution. Real training trajectories cross between regimes — the data's effective intrinsic dimension changes as the model resolves coarse structure first and fine structure later; the spectrum has multiple tail segments; the feature distribution shifts as the model moves from common to rare facts. Each crossing is a *break*, and the broken-law functional form is the smooth stitch between the clean power-law segments that the theories in this post explain. The clean theories tell you why each piece is straight; the broken law tells you how the pieces join. You need both to forecast a real run across a wide range.

### 5.1 A worked numerical example

Let us make the whole chain concrete with numbers, because the geometric mechanism is the easiest to compute end to end. Suppose you measure your dataset's intrinsic dimension to be $d = 8$ (a plausible figure for a moderately clean, somewhat narrow text domain), and suppose the target is smooth enough that you sit near the $4/d$ end. Then the predicted parameter-scaling exponent is

$$
\alpha = \frac{4}{8} = 0.5.
$$

What does that buy you? Going from a $100\text{M}$-parameter model to a $1\text{B}$-parameter model is a $10\times$ increase in $N$, and it multiplies the reducible loss by

$$
10^{-0.5} \approx 0.316,
$$

so the reducible loss drops to about $32\%$ of its starting value — a $68\%$ reduction per decade. Now suppose instead your data is open-domain web text with a much larger effective intrinsic dimension, say $d = 40$, sitting nearer the conservative end at $\alpha = 1/d = 0.025$. The same $10\times$ in parameters multiplies the reducible loss by

$$
10^{-0.025} \approx 0.944,
$$

a paltry $5.6\%$ reduction per decade. To match the $68\%$ reduction the clean $d=8$ data got from a single decade of parameters, the messy $d=40$ data would need

$$
10^{\,0.68/0.025} \approx 10^{27}
$$

times more parameters — a physically absurd number. That is not a rhetorical flourish; it is the arithmetic consequence of $\alpha \propto 1/d$, and it is exactly why "just scale the model" stops working on high-intrinsic-dimension data and why curation that lowers $d$ is worth more than almost any architecture change. The exponent is destiny, and intrinsic dimension sets the exponent.

### 5.2 A worked frequency example

Run the Zipf mechanism end to end too, because it gives a different and complementary lever. Suppose your corpus has feature frequencies with Zipf exponent $\alpha = 1$ (heavy tail, typical of raw natural language). Then the learning-curve exponent is

$$
\beta = \frac{1}{1+1} = 0.5,
$$

the square-root curve: $10\times$ the data cuts error to $10^{-0.5} \approx 32\%$. Now suppose aggressive deduplication and quality filtering reshape the effective distribution so that mass concentrates on higher-value features and the tail lightens to $\alpha = 2$. Then

$$
\beta = \frac{2}{1+2} \approx 0.667,
$$

and $10\times$ the data now cuts error to $10^{-0.667} \approx 21.5\%$. The same data budget improves the model noticeably faster purely because the frequency distribution is better shaped. This is the same conclusion the manifold example reached — cleaner data has a steeper exponent — arrived at through an entirely different mechanism. Two unrelated derivations, one practical instruction: shape your data so its effective dimension is low and its frequency tail is light.

## 6. Closing the loop: why cleaner data scales better

**Senior rule of thumb: data curation is not preprocessing, it is exponent engineering — every filtering and deduplication decision is a bet about the intrinsic dimension and spectral decay you will scale against.**

The [data quality post](/blog/machine-learning/scaling-laws/data-quality-scaling-laws) showed empirically that curation behaves like a multiplicative factor on compute — the DataComp-LM result that good curation reaches a target loss with roughly $6.6\times$ less training compute than raw web text. That post argued the effect from measurements. This post explains *why* the effect exists, mechanistically, three times over. The figure below lines up the three mechanisms in before/after form: what noisy high-dimensional data does to each, and what curation does to each.

![A before-and-after figure contrasting noisy high-dimensional web data against curated low-dimensional data across the manifold, spectrum, and Zipf mechanisms, showing that curation lowers intrinsic dimension, steepens spectral decay, and lightens the frequency tail, all of which raise the scaling exponent](/imgs/blogs/why-power-laws-arise-8.png)

Walk the three rows. **Geometrically**, noise and redundancy *add intrinsic dimensions*: every spurious axis of variation — boilerplate templates, near-duplicate documents, formatting artifacts, low-quality scraped junk — is a direction the data varies in that carries no signal, inflating $d$. Curation prunes those directions, lowering $d$, raising $\alpha = 4/d$. **Spectrally**, junk directions show up as a flat, slowly-decaying tail of small-but-nonzero eigenvalues; deduplication and filtering remove that tail, steepening the spectral decay, raising $\alpha_K$. **In frequency terms**, raw web data has a long tail of low-value features (the $60{,}000$-times-repeated boilerplate string from the deduplication literature is a vivid example of mass wasted on a worthless feature); curation reallocates mass toward high-value features, lightening the tail, raising the Zipf $\alpha$ and hence $\beta = \alpha/(1+\alpha)$.

All three say the same thing in their own dialect: **curation raises the exponent.** And raising the exponent is the most valuable thing you can do, because the exponent compounds over every decade of scale. A constant-factor improvement (a better optimizer, a cleaner implementation) shifts the curve down once; an exponent improvement bends the *slope* of the curve, and bends it forever. That is the mechanistic reason the data-quality lever is so large, and why it dwarfs most architecture tweaks: architecture mostly moves the constant $A$, but curation moves $\alpha$.

This also resolves a question the empirical posts left implicit. Why is the *effective compute multiplier* of good data roughly constant across scale (the $\sim 6.6\times$ holding over a wide range), rather than vanishing or exploding? Because curation acts on the exponent, and an exponent difference produces a constant ratio of compute-to-reach-a-target-loss at every scale — exactly the scale-free behavior of a power law. The constancy of the multiplier is itself a fingerprint of the power-law mechanism.

## 7. How to apply this when you plan a run

**Senior rule of thumb: before you commit a large budget, estimate your exponent's *source*, not just its value — because the source tells you whether the exponent will hold where you are extrapolating to.**

The theory is not decoration; it changes what you measure and what you bet on. Here is the practical translation.

**Estimate intrinsic dimension, not just count.** Before treating your dataset as a scalar token count $D$, get a handle on its intrinsic dimension. There are cheap estimators (nearest-neighbor-based intrinsic-dimension estimates on a sample of embeddings, for example). A high estimate is a warning that your exponent will be shallow and "just scale it" will be expensive. A low estimate is permission to expect steep, cheap improvement. This is the single most actionable thing the geometric mechanism gives you.

**Measure the spectrum to forecast the exponent.** In the wide-network regime the loss exponent equals the kernel/covariance spectral decay rate. You can estimate the data-covariance spectrum (or an empirical kernel spectrum) from a sample *without training a large model at all*, fit the decay $\lambda_i \propto i^{-(1+\alpha)}$, and read off a predicted $\alpha$. This is a genuinely cheap forecast of the exponent, complementary to the [observational scaling](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations) idea of fitting small runs — here you forecast from a property of the data directly.

**Diagnose your regime before trusting a slope.** If you fit an exponent and it comes out near 1, suspect you are variance-limited and the slope is the trivial $1/D$, not a statement about your data. Push capacity until the exponent settles below 1 and stabilizes; only then are you measuring the resolution-limited, data-dependent law you actually want to extrapolate.

**Treat curation as exponent engineering.** Every dedup and filter decision is a bet on $d$, on the spectral tail, and on the frequency distribution. Budget for it accordingly: a curation step that lowers $d$ even modestly is worth more than a comparably-priced architecture change, because it compounds over scale. And resist the instinct to filter to a fixed top-$X\%$ regardless of compute — the [data-quality post](/blog/machine-learning/scaling-laws/data-quality-scaling-laws)'s quality-quantity tradeoff is the reminder that the *optimal* aggressiveness depends on your compute-to-data ratio, even though more aggressive filtering generally lowers $d$.

**Expect breaks at regime boundaries.** The clean theories hold within a regime. When your effective intrinsic dimension changes (the model finishes resolving coarse structure and starts on fine structure), when the spectrum transitions from one tail segment to another, or when you exhaust the latent dimension and hit Maloney's plateau — the exponent changes, and a single-power-law extrapolation will mis-predict. That is precisely when you reach for the [broken-neural-scaling-law](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws) functional form, which models the smooth stitching between these clean segments.

| Symptom you observe | Likely mechanism / regime | What to do |
|---|---|---|
| Exponent near 1, unstable | variance-limited (CLT $1/D$ or $1/P$) | add capacity; do not trust the slope yet |
| Shallow stable exponent ($\sim 0.05\text{–}0.1$) | resolution-limited, high intrinsic $d$ | curate to lower $d$; expect expensive scaling |
| Steep exponent ($\gtrsim 0.3$) | resolution-limited, low intrinsic $d$ | scale confidently; the data is well-conditioned |
| Curve flattens early, plateaus | exhausted latent dimension (Maloney) | you have resolved the structure; add data diversity, not just size |
| Sudden slope change / kink | regime boundary / break | switch to a broken-law fit; do not extrapolate across the kink |
| Curation buys a constant compute factor | power-law exponent shift | invest in curation; the multiplier holds at every scale |

## 7.5 Objections, nuances, and where the theories strain

**Senior rule of thumb: a theory you cannot poke holes in is a theory you do not understand — so here are the real objections, and the honest answers.**

A reader who has been paying attention will have accumulated objections, and the theories are stronger for confronting them. Let us take the serious ones in turn.

**"Intrinsic dimension is not a single number."** Correct, and this is the most important caveat. Real data does not lie on a clean manifold of one fixed dimension; the effective dimension varies across the data and across scale. Coarse structure (which language is this, roughly what topic) is low-dimensional and resolved early; fine structure (precise factual content, rare idioms, subtle style) is higher-dimensional and resolved late. So the "exponent" you measure is really a local slope of a curve whose effective $d$ is slowly changing, which is exactly why a single power law eventually breaks and why the [broken-law](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws) form, which stitches segments of different slope, fits real data better than any single line. The geometric mechanism is right *within a regime of roughly constant effective dimension*; it does not claim there is one global $d$.

**"The $4/d$ constant is too clean to be real."** Also fair. The factor of 4 comes from two clean steps (smooth-target Taylor remainder, then squaring in the loss), and real targets are not perfectly smooth, real losses are not perfectly quadratic near the optimum, and the optimizer does not place the partition cells perfectly. So $4/d$ is the smooth idealization, $1/d$ is the conservative floor, and reality sits between — the *proportionality* $\alpha \propto 1/d$ is the robust claim, the constant is soft. Treat $4/d$ as the optimistic end of a range, not a precise prediction. Every quantitative use in this post that needs precision (the worked examples) states the assumed end explicitly.

**"The kernel duality only holds in the infinite-width limit, which we are not in."** True. Real trained networks are not in the lazy, fixed-kernel regime; they do feature learning, which the duality does not capture. The honest position is that the duality explains why $\alpha_P$ and $\alpha_D$ are *related and both power-law* — a structural fact that survives outside the limit — even though the exact equality $\alpha_P = \alpha_D = \alpha_K$ is a limit statement. The Maloney solvable model is the antidote: it shows the spectral mechanism producing scaling without relying on the wide-width hand-waving, which is why it is in this post.

**"Zipf-frequency scaling is just memorization; surely real models generalize."** They do both, and that is the point. The Zipf mechanism does not claim models are pure memorizers; it claims that the memorization *component* of what a model does already scales as a power law because the facts are Zipfian, and that this component sums with the manifold-generalization component to produce the very clean empirical curve. If anything, the surprise is how much of observed scaling is plausibly tail-fact acquisition rather than generalization — which has implications for what "more scale" actually buys (more rare facts) versus what it does not (necessarily, more reasoning).

**"If the data is what matters, why do architecture and optimizer matter at all?"** They matter for the *prefactor*, not the *exponent*. A better architecture or optimizer resolves the same spectrum more efficiently — it spends its partition cells or its resolved modes on the parts of the data that matter, lowering the constant $A$ in $L = E + A\,N^{-\alpha}$. That is real and valuable; a 2x prefactor improvement is a 2x compute saving. But it shifts the curve down once; it does not bend the slope. Only the data's structure bends the slope. This is the precise sense in which "data is the exponent, architecture is the constant," and it is why the largest sustained advantages in the field have come from data, not architecture.

## 8. Case studies: the mechanisms in the wild

The theories above are vindicated less by their proofs than by the fact that, once you know them, a long list of otherwise-puzzling empirical observations snaps into focus. Here are eight, each a real pattern from the literature or from practice, each explained by one of the three mechanisms. Read them as "the theory predicted this, and here it is."

### 1. The shallow language-model exponent

The most-quoted number in the whole field is that the loss-versus-parameters exponent of large language models is small — around $0.05$ to $0.08$ depending on the fit. People sometimes read this as disappointing, as if models are bad at learning. The mechanistic reading is the opposite: a small $\alpha$ means a large intrinsic dimension, and natural language varies in an enormous number of semantic, syntactic, factual, and stylistic directions at once. With $\alpha \approx 0.07$ and the conservative $\alpha = 1/d$, that implies an effective $d \approx 14$; with the smooth-target $4/d$ it implies $d \approx 57$. Either way the number is large, and that is a *fact about language*, not a failure of transformers. The shallow slope is the price of modeling something genuinely high-dimensional, and it predicts — correctly — that brute-force parameter scaling on raw text yields diminishing returns long before you would like.

### 2. Code scales faster than prose

Teams repeatedly find that code domains scale with a steeper exponent than open-domain natural language: the same compute increment buys a larger loss reduction on code. The geometric mechanism explains it in one sentence — code has lower intrinsic dimension. Syntax is rigid, the vocabulary of valid continuations at any point is heavily constrained by grammar and scope, and the manifold of valid programs varies in far fewer effective directions than the manifold of valid English. Lower $d$, larger $\alpha = 4/d$, faster scaling. The same logic predicts that highly templated or structured data (logs, tabular records, formal proofs) should scale faster still, and it does.

### 3. Deduplication that helps more than it "should"

The deduplication literature reports a string repeated over $60{,}000$ times in a popular web corpus, and finds that removing such duplicates improves perplexity at fixed token count. From a pure "more data is better" frame this is paradoxical — you removed data and got a better model. From the frequency mechanism it is obvious: that $60{,}000\times$-repeated string is a single low-value feature hogging enormous probability mass, distorting the Zipf tail toward junk. Removing it reallocates mass toward useful features, lightens the effective tail, raises $\alpha$, and therefore raises $\beta = \alpha/(1+\alpha)$. Deduplication is not "less data," it is "a better-shaped frequency distribution," and the exponent improvement is the payoff.

### 4. The early plateau on a narrow domain

A team fine-tunes on a small, narrow, clean dataset and watches the loss improve beautifully as a power law — and then flatten abruptly, far sooner than the parameter count would suggest. The naive diagnosis is "overfitting" or "we need more parameters." The Maloney mechanism gives the real one: a narrow domain has a small *latent dimension*, so there are only so many spectral modes to resolve. Once the model has resolved them all, there is nothing left, and performance plateaus regardless of how much bigger you make the model. The fix is not more parameters, it is more *diversity* — adding latent dimensions by broadening the data — which is the opposite of what the overfitting diagnosis would suggest.

### 5. The variance-limited mirage

An engineer fits a scaling law on three small runs, measures an exponent suspiciously close to $1$, and excitedly predicts a huge loss reduction from the big run. The big run underdelivers badly. The Bahri taxonomy names the error: the small runs were *variance-limited*, where the generic $1/D$ exponent reigns and tells you nothing about the data. The true resolution-limited exponent — the one that governs the large-scale behavior — is much shallower, so the extrapolation across the variance-to-resolution crossover was wildly optimistic. The lesson encoded in the four-regime grid is to push capacity until the exponent settles well below $1$ before trusting it.

### 6. Two labs, two exponents, one reconciliation

The [Kaplan-versus-Chinchilla](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation) discrepancy is partly a regime story. When two careful teams measure different exponents for "the same thing," one strong possibility is that they sampled different parts of the curve — one nearer a variance-limited or transitional regime, one deeper into the resolution-limited regime — or fit across a break. The mechanisms in this post turn "whose exponent is right?" into the more productive "which regime was each measuring, and does the regime hold where I want to extrapolate?" That reframing is the whole practical value of having a mechanism.

### 7. The intrinsic-dimension estimate that predicted the slope

A more recent practice, enabled directly by the geometric mechanism, is to estimate the intrinsic dimension of a candidate dataset from a sample of embeddings *before* committing to a large run, and use $\alpha \approx 4/d$ (or the conservative $1/d$) as a prior on the exponent you will measure. Teams that do this report it catches "this dataset will scale poorly" early, when it is cheap to swap or curate the data, rather than after a large run disappoints. It is the geometric mechanism used as a forecasting tool, complementary to the small-run extrapolation of the [foundations](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations) post.

### 8. Pure memorization that still scales

A surprising-at-first observation: even models or model components that mostly *memorize* — storing facts they have seen rather than generalizing across them — exhibit clean power-law improvement with data. Hutter's mechanism is the explanation. Memorization of a Zipf-distributed set of facts gives $\text{error} \sim n^{-\alpha/(1+\alpha)}$ all by itself, no generalization required. This reframes a chunk of LLM scaling as steady tail-fact acquisition: the model keeps absorbing rarer and rarer entities and idioms at a power-law pace set by their Zipf tail, and that contributes to the overall scaling curve alongside the manifold-generalization contribution. Two mechanisms, both yielding power laws, summing to the very clean empirical curve we observe.

## 9. When to trust a mechanism, and when to just measure

**Senior rule of thumb: use the mechanisms to reason about direction and regime, and use small-run extrapolation to get the actual number — neither alone is enough.**

The mechanisms in this post are powerful for reasoning and dangerous if over-trusted for prediction. Here is the honest division of labor.

Reach for the mechanistic theory when you are asking *directional* and *structural* questions: Will cleaner data scale better? (Yes — lower $d$.) Should I expect this narrow domain to plateau? (Yes — small latent dimension.) Is my measured exponent meaningful or am I variance-limited? (Check whether it is near 1.) Will adding ambient resolution change my slope? (No — only intrinsic dimension matters.) Why is my code model scaling faster than my chat model? (Lower intrinsic dimension.) For all of these, the theory gives a confident, mechanistic answer, and that answer is robust to the messy details the theory abstracts away.

Do *not* reach for the theory to predict the exact exponent or exact loss of a specific run. The constant in $\alpha = c/d$ depends on target smoothness, the intrinsic-dimension estimate is noisy, the wide-network duality is exact only in a limit you are not in, and real spectra have multiple tail segments. For the actual number, fit small runs and extrapolate, exactly as the [foundations post](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations) prescribes — and use the [broken-law](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws) form if a break is in range. The right workflow uses both: the mechanism tells you which regime you are in and which way every lever points, and the empirical fit gives you the slope to bet on within that regime.

| Question type | Trust the mechanism? | Trust small-run extrapolation? |
|---|---|---|
| Will cleaner data scale better? | Yes (direction) | Only if you measure both datasets |
| What exact loss will the big run hit? | No | Yes (this is what it is for) |
| Am I variance-limited or resolution-limited? | Yes (the four-regime diagnostic) | Indirectly (slope near 1 is the tell) |
| Will this narrow domain plateau? | Yes (latent dimension) | Yes, if your runs reach the plateau |
| Is a break coming before my target scale? | Partially (regime boundaries) | Use the broken-law form |

## 10. The series, closed

This is the last post in the series, so let us close the loop deliberately. We opened in the [foundations post](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations) with a single empirical claim: loss is predictable before you train, because the learning curve is a power law and a power law is a straight line on log-log axes. Everything since has been an elaboration of that line — its slope ([Kaplan](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models)), its compute-optimal allocation ([Chinchilla](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling)), its behavior when data is scarce ([data-constrained](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws)) or when inference cost matters ([inference-aware](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws)), and what happens when one straight line is not enough ([broken laws](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws)).

The question this post answered is the one underneath all of those: *why is there a line at all?* And the answer turned out to be the most reassuring possible one. The line is not a coincidence, not an artifact of transformers, not a property of Adam. It is forced by the structure of natural data through three independent mechanisms — the geometry of the data manifold, the decay of its covariance spectrum, and the heavy tail of its feature frequencies — any one of which would produce a power law, all three of which operate at once. The exponent is set by a single underlying quantity, the intrinsic dimension $d$, which shows up as $4/d$ in the partition argument, as $\alpha_K \propto 1/d$ in the spectral argument, and as a reshapeable frequency tail in the Zipf argument.

If you remember one thing from the entire series, make it this: **the scaling exponent is destiny, and the intrinsic dimension of your data sets the exponent.** Compute moves you along the curve; curation bends the curve. The labs that win the next decade of scaling will be the ones who understood that the cheapest way to a steeper slope was never a bigger cluster — it was cleaner data, which is to say, lower intrinsic dimension, faster spectral decay, and a lighter Zipf tail. That is the whole game, mechanism and all.

## Further reading

- Sharma, Kaplan. "A Neural Scaling Law from the Dimension of the Data Manifold." arXiv:2004.10802 — https://arxiv.org/abs/2004.10802
- Bahri, Dyer, Kaplan, Lee, Sharma. "Explaining Neural Scaling Laws." arXiv:2102.06701 (PNAS 2024) — https://arxiv.org/abs/2102.06701
- Maloney, Roberts, Sully. "A Solvable Model of Neural Scaling Laws." arXiv:2210.16859 — https://arxiv.org/abs/2210.16859
- Hutter. "Learning Curve Theory." arXiv:2102.04074 — https://arxiv.org/abs/2102.04074
- Caballero, Gupta, Rish, Krueger. "Broken Neural Scaling Laws." arXiv:2210.14891 (ICLR 2023) — https://arxiv.org/abs/2210.14891
- Companion posts in this series: [scaling-laws foundations](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations), [broken neural scaling laws](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws), and [data quality as a scaling axis](/blog/machine-learning/scaling-laws/data-quality-scaling-laws).
