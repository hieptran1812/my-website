---
title: "How to Read the Math in an ML Paper: A Notation Decoder and a Three-Pass Method"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "Decode the notation every ML paper uses, read any equation in three passes, and take apart the DDPM training loss symbol by symbol."
tags: ["mathematics", "math-for-ai", "notation", "reading-papers", "diffusion-models", "ddpm", "expectation", "proofs"]
category: "machine-learning"
subcategory: "Mathematics"
author: "Hiep Tran"
featured: true
readTime: 17
---

> **Level:** Foundation · **Needs:** high-school algebra and a little patience · **Unlocks:** every later post in this series, and the method section of most ML papers

Here is the line that trains almost every image diffusion model you have used. It is Eq. 14 of *Denoising Diffusion Probabilistic Models* (Ho, Jain and Abbeel, 2020):

$$
L_{\mathrm{simple}}(\theta) := \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\left\lVert \boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\left(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\,t\right)\right\rVert^2\right]
$$

If you are a strong engineer who last did math at university, this line probably reads as a wall. You can ship the code that implements it, but the symbols do not speak. By the end of this post every symbol will, and you will have a method that works on equations you have never seen.

## The series in one picture

This post is the first rung of a long ladder. The series promise is simple: after it, you can follow the math and the proofs in LLM, diffusion, flow-matching, optimal-transport, MoE, RL and theory papers, and you can write a correct theory section of your own.

![A ladder of fourteen rungs from A at the bottom to N through T at the top; rung A, the language of papers, is lit blue, with arrows to the DDPM loss, the Track A papers and every later appendix](/imgs/blogs/how-to-read-the-math-in-an-ml-paper-1.webp)

The diagram above is the mental model. Rung A is the vocabulary; every rung above it (linear algebra, calculus, probability, optimization, stochastic processes, transport, geometry, learning theory) is written in that vocabulary. My thesis for the whole series: paper math is a small, closed vocabulary. Learn each idea once, with a picture, a derivation, a proof and a numerical check, and the appendix stops being a wall.

That thesis sets the shape of every post. Each one ends with the same four fixed blocks, so you always know where to look:

| Block | What it does |
| --- | --- |
| **Paper sentence, decoded** | One real equation from a named paper, taken apart symbol by symbol |
| **Proof, read line by line** | A real proof, every step justified, every assumption traced to where it is used |
| **Check it numerically** | A short NumPy script that tests the post's central claim, with its real output |
| **Where this shows up in practice** | Two or three named systems or recipes that use exactly this math |

Before any of that, though, we need the alphabet.

## Why notation, and why first

Notation is compression. A paper spends one symbol where prose would spend a sentence, because the authors assume you already hold the sentence in your head. Your job is to decompress.

A good analogy is a city's subway map. The map is unreadable until you know three conventions (colours are lines, dots are stations, a white ring is a transfer), and after that it is the fastest way to plan a trip. Math notation works the same way, with one honest difference: different papers sometimes use the same symbol for different things, as if two cities used red for different lines. So every decoder rule below comes with "usually", and the paper's own notation section always wins.

There are about a dozen pieces of notation that cover most of what you will meet in a method section. We take them one at a time, each with a hand-checkable example.

## The notation decoder

### Expectation: $\mathbb{E}_{x \sim p}[f(x)]$

**Expectation** answers one question: if I drew this random thing many times, what would the value be on average? It is a weighted average, where the weights are probabilities.

A dice game is the honest picture. If you win 1 dollar half the time, 4 dollars 30% of the time and 10 dollars 20% of the time, your long-run winnings per game are not the plain average of 1, 4 and 10. Rare outcomes count for less.

![Three panels: bars of probability 0.5, 0.3, 0.2; bars of value 1, 4, 10 with a dashed line at 3.7 below the plain mean of 5; and the products 0.5, 1.2, 2.0 stacked into one bar of height 3.7](/imgs/blogs/how-to-read-the-math-in-an-ml-paper-3.webp)

Work it by hand. Three outcomes $a, b, c$ have probabilities 0.5, 0.3 and 0.2 and payoffs 1, 4 and 10. Multiply and add: $0.5 \cdot 1 + 0.3 \cdot 4 + 0.2 \cdot 10 = 0.5 + 1.2 + 2.0 = 3.7$. The figure stacks those three products into one bar; the plain mean, 5.0, would overstate the game because it ignores that 10 is rare.

Formally, for a discrete distribution $p$,

$$
\mathbb{E}_{x \sim p}[f(x)] = \sum_{x} p(x)\, f(x).
$$

Here $x \sim p$ reads "$x$ is drawn from the distribution $p$", and $f$ is any function of the outcome. For a continuous distribution the sum becomes an integral, $\int p(x) f(x)\,dx$, with the same meaning. The subscript under $\mathbb{E}$ names what is random. When it lists several variables, as in $\mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}$, all of them are drawn and the average runs over all of them together.

In a paper, an expectation over data is almost always estimated by a minibatch average. That is the bridge from the math to your training loop: $\mathbb{E}$ on the page, `loss.mean()` in the code.

### Gradient: $\nabla_\theta$

The **gradient** tells you which way is uphill and how steep it is. A hiker in fog cannot see the summit, but can feel the slope under their boots; the gradient is that feeling, one number per direction.

Take $f(\theta) = \theta_1^2 + 3\theta_2$ at the point $\theta = (1, 2)$. Changing $\theta_1$ alone, the slope is $2\theta_1 = 2$. Changing $\theta_2$ alone, the slope is 3. So $\nabla_\theta f(1,2) = (2, 3)$, and gradient descent with step size $\eta = 0.1$ moves to $(1 - 0.2,\ 2 - 0.3) = (0.8, 1.7)$.

$$
\nabla_\theta f = \left(\frac{\partial f}{\partial \theta_1}, \dots, \frac{\partial f}{\partial \theta_d}\right).
$$

Each entry $\partial f / \partial \theta_i$ is a **partial derivative**: the slope when only $\theta_i$ moves. The subscript on $\nabla$ says which variables you differentiate with respect to, and it matters. In a loss $L(\theta)$ that also depends on data $x$, $\nabla_\theta L$ treats $x$ as fixed. In a 7-billion-parameter model the same picture holds; the vector simply has 7 billion entries, which is what backpropagation computes.

### argmin, min and the dropped constant

**min** asks how low a function goes. **argmin** asks where it gets that low. Papers almost always want the second: the best weights, not the best loss value.

![Two parabolas, f and h equal to half of f plus 2; both bottom out at theta equal to 2, at heights 1 and 2.5](/imgs/blogs/how-to-read-the-math-in-an-ml-paper-5.webp)

Take $f(\theta) = (\theta - 2)^2 + 1$. Its lowest value is 1, so $\min_\theta f = 1$, and it reaches it at $\theta = 2$, so $\arg\min_\theta f = 2$. Now build $h(\theta) = 0.5 f(\theta) + 2$. Its minimum moves to $0.5 \cdot 1 + 2 = 2.5$, but its argmin stays at 2. The figure shows both curves sharing one dashed vertical line.

That observation is a reading tool, not a curiosity. For any constant $c \gt 0$ and any constant $b$,

$$
\arg\min_\theta \big(c\, f(\theta) + b\big) = \arg\min_\theta f(\theta).
$$

In words: rescaling a loss by a positive number or adding a constant never changes the best weights. This is why papers silently drop constants, and why DDPM can throw away the weights in its longer loss and still train the same kind of model (we will see that in the decode).

### Proportional to: $\propto$

$a \propto b$ means "$a$ equals $b$ times some constant we have not bothered to write". It shows up wherever a normalising constant is expensive or irrelevant, most famously in Bayes' rule:

$$
p(\theta \mid D) \propto p(D \mid \theta)\, p(\theta).
$$

The posterior $p(\theta \mid D)$ is the likelihood $p(D \mid \theta)$ times the prior $p(\theta)$, up to a constant that makes it sum to one. With two hypotheses, prior 0.5 each and likelihoods 0.3 and 0.1, the unnormalised products are 0.15 and 0.05. Divide by their total 0.20 and you get 0.75 and 0.25. The $\propto$ let us postpone that division to the end.

### Defined as: $:=$

$a := b$ means "I am naming $b$ as $a$ right now". It is a definition, not a claim to verify. When you see $:=$, you have nothing to check; when you see $=$, the authors are asserting something that should follow from earlier lines. Some papers write $\triangleq$ or $\overset{\text{def}}{=}$ for the same thing.

### Norms: $\lVert \cdot \rVert$

A **norm** measures the length of a vector. The trouble is that "length" has several reasonable definitions, and the subscript tells you which one.

![A 2D plane with the unit circle, a diamond and a square around the origin, and the vector (0.6, 0.8) whose tip lies on the circle, outside the diamond and inside the square](/imgs/blogs/how-to-read-the-math-in-an-ml-paper-4.webp)

Take $x = (0.6, 0.8)$. Its Euclidean length is $\lVert x \rVert_2 = \sqrt{0.36 + 0.64} = 1.0$. Its taxicab length, the distance a car drives on a grid, is $\lVert x \rVert_1 = 0.6 + 0.8 = 1.4$. Its largest coordinate is $\lVert x \rVert_\infty = 0.8$. The figure draws every point of length one under each norm: a circle, a diamond and a square. The same arrow sits on the circle, outside the diamond and inside the square.

In general, for $p \ge 1$,

$$
\lVert x \rVert_p = \Big(\sum_{i=1}^{d} \lvert x_i \rvert^p\Big)^{1/p}, \qquad \lVert x \rVert_\infty = \max_i \lvert x_i \rvert.
$$

A bare $\lVert x \rVert$ with no subscript almost always means $\lVert x \rVert_2$. A squared norm $\lVert x \rVert^2$ is the sum of squared entries, which is exactly what a sum-reduced MSE loss computes. On a matrix, $\lVert W \rVert$ is ambiguous (Frobenius or spectral); the linear algebra posts will separate them.

### Subscripts, superscripts, hats, tildes and bars

These small marks carry a lot of meaning and are the most overloaded part of the vocabulary. The table lists the usual readings; your paper's notation section overrides it.

| Mark | Usual meaning | Example |
| --- | --- | --- |
| $x_t$ | time step or iteration | the noisy image after $t$ diffusion steps |
| $x_i$ | the $i$-th coordinate of a vector | pixel $i$ |
| $x^{(i)}$ | the $i$-th training example | the $i$-th image in the dataset |
| $W^{(\ell)}$ | layer $\ell$ | weights of layer $\ell$ |
| $x^\top$ | transpose, not a power | turns a column into a row |
| $\theta^\star$ | the optimum | $\theta^\star = \arg\min_\theta L(\theta)$ |
| $\hat{\theta}$ | an estimate of $\theta$ from data | a fitted parameter |
| $\tilde{x}$ | a perturbed or proposed version of $x$ | a noised input, a draft token |
| $\bar{x}$ | an average or a cumulative quantity | the mean, or $\bar{\alpha}_t$ in DDPM |
| $\epsilon_\theta$ | a function with learnable weights $\theta$ | a neural network |

Two conventions trip engineers repeatedly. First, papers index from 1 and code indexes from 0: DDPM's $t$ runs over $1, \dots, T$, so `abar[t-1]` is correct and `abar[t]` is an off-by-one bug. Second, a superscript in parentheses, $x^{(2)}$, is a label, while a bare superscript, $x^2$, is a power. When a paper writes both, the parentheses are doing real work.

## A three-pass method for any equation

Decoding symbols is not the same as understanding an equation. You need an order of attack, and the one that works for me has three passes. Each pass asks a different question, and you never try to answer the third before the first.

**Pass 1, shape.** Ignore every symbol you do not know. What is the outermost operation? An expectation, a sum, an argmin, a norm? What type of thing does the line produce: a scalar loss, a vector update, a probability? Most equations in a method section are one of four shapes: a loss to minimise, an update rule, a distribution, or a bound.

**Pass 2, symbols.** Now go inside, one layer at a time. For every symbol, find its definition (search the PDF for its first appearance) and write its type next to it: scalar, vector of dimension $d$, function, distribution. Mismatched types are where your misreadings hide.

**Pass 3, numbers.** Plug in the smallest numbers that make the line non-trivial: one dimension, two outcomes, a step size of 0.1. Compute the value by hand. If you cannot, you have not read the equation yet, only looked at it.

<figure class="blog-anim">
<svg viewBox="0 0 760 270" role="img" aria-label="The DDPM loss read in three passes: first its shape, then its symbols, then with small numbers; a highlight moves to the part each pass reads" style="width:100%;height:auto;max-width:860px">
<style>
.r1-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.r1-eq{font:500 18px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--text-primary,#1f2937);text-anchor:middle}
.r1-hl{fill:var(--accent,#6366f1);opacity:.22}
.r1-cap{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.r1-sub{font:400 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.r1-step{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.r1-dot{fill:var(--accent,#6366f1)}
@keyframes r1-p1{0%,30%{opacity:1}34%,96%{opacity:0}100%{opacity:1}}
@keyframes r1-p2{0%,30%{opacity:0}34%,63%{opacity:1}67%,100%{opacity:0}}
@keyframes r1-p3{0%,63%{opacity:0}67%,96%{opacity:1}100%{opacity:0}}
.r1-a{animation:r1-p1 12s ease-in-out infinite}
.r1-b{animation:r1-p2 12s ease-in-out infinite}
.r1-c{animation:r1-p3 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.r1-a{animation:none;opacity:1}.r1-b,.r1-c{animation:none;opacity:0}}
</style>
<rect class="r1-box" x="20" y="50" width="90" height="50" rx="8"/>
<rect class="r1-box" x="120" y="50" width="110" height="50" rx="8"/>
<rect class="r1-box" x="240" y="50" width="140" height="50" rx="8"/>
<rect class="r1-box" x="390" y="50" width="220" height="50" rx="8"/>
<rect class="r1-box" x="620" y="50" width="120" height="50" rx="8"/>
<g class="r1-a">
<rect class="r1-hl" x="120" y="50" width="110" height="50" rx="8"/>
<rect class="r1-hl" x="620" y="50" width="120" height="50" rx="8"/>
<text class="r1-cap" x="30" y="150">Pass 1, shape: an average (E) of a squared length (‖·‖²).</text>
<text class="r1-sub" x="30" y="176">So this is a loss: one number per choice of θ, to be minimised.</text>
<circle class="r1-dot" cx="180" cy="232" r="7"/>
</g>
<g class="r1-b">
<rect class="r1-hl" x="240" y="50" width="140" height="50" rx="8"/>
<rect class="r1-hl" x="390" y="50" width="220" height="50" rx="8"/>
<text class="r1-cap" x="30" y="150">Pass 2, symbols: the long argument is x_t, a noisy image.</text>
<text class="r1-sub" x="30" y="176">ε is the noise we added; ε_θ is the network guessing it back.</text>
<circle class="r1-dot" cx="380" cy="232" r="7"/>
</g>
<g class="r1-c">
<rect class="r1-hl" x="390" y="50" width="220" height="50" rx="8"/>
<text class="r1-cap" x="30" y="150">Pass 3, numbers: at ᾱ = 0.25, x_t = 0.5·x₀ + 0.866·ε.</text>
<text class="r1-sub" x="30" y="176">√0.25 = 0.5 and √0.75 = 0.866: this input is mostly noise.</text>
<circle class="r1-dot" cx="580" cy="232" r="7"/>
</g>
<text class="r1-eq" x="65" y="81">L(θ) :=</text>
<text class="r1-eq" x="170" y="81">E<tspan font-size="12" dy="6">t,x₀,ε</tspan><tspan dy="-6"> [</tspan></text>
<text class="r1-eq" x="310" y="81">‖ε − ε_θ(</text>
<text class="r1-eq" x="500" y="81">√ᾱₜ x₀ + √(1−ᾱₜ) ε</text>
<text class="r1-eq" x="680" y="81">, t)‖² ]</text>
<text class="r1-step" x="180" y="258">1 shape</text>
<text class="r1-step" x="380" y="258">2 symbols</text>
<text class="r1-step" x="580" y="258">3 numbers</text>
</svg>
<figcaption>The three-pass read of the DDPM loss: the highlight moves from the outer shape, to the symbols inside, to a small-number check of the noisy input.</figcaption>
</figure>

The animation runs this method on the DDPM loss. Notice that pass 1 already tells you the most important fact (this is a loss to minimise over $\theta$) before you know what $\bar{\alpha}_t$ is. That is the point of the order: the shape tells you what kind of answer to expect, so the symbols have somewhere to land.

## Decoding the DDPM loss

### Paper sentence, decoded

**Source:** Jonathan Ho, Ajay Jain, Pieter Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020, [arXiv:2006.11239](https://arxiv.org/abs/2006.11239), Eq. 14.

$$
L_{\mathrm{simple}}(\theta) := \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\left\lVert \boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\left(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\,t\right)\right\rVert^2\right]
$$

The paper says $t$ is uniform between 1 and $T$, and its Algorithm 1 draws $\mathbf{x}_0$ from the data and $\boldsymbol{\epsilon}$ from a standard Gaussian. Two definitions from earlier in the paper complete the picture: $\alpha_t := 1 - \beta_t$ and $\bar{\alpha}_t := \prod_{s=1}^{t} \alpha_s$, where $\beta_t$ is a small noise variance per step that the paper grows linearly from $10^{-4}$ to 0.02 over $T = 1000$ steps.

![The DDPM loss with each symbol called out: the definition sign, the expectation over t, x0 and epsilon, the true noise, the network, the shrunken clean image, the weighted noise, the noisy input they form, and the squared length of the error](/imgs/blogs/how-to-read-the-math-in-an-ml-paper-2.webp)

Here is the symbol-by-symbol decode, with where each piece was built in this post:

| Symbol | What it is | Built in |
| --- | --- | --- |
| $L_{\mathrm{simple}}(\theta)$ | a scalar loss, one number per weight setting $\theta$ | argmin section |
| $:=$ | "is defined as"; nothing to verify | defined-as section |
| $\mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}$ | average over a random step, a random image and random noise | expectation section |
| $\mathbf{x}_0$ | a clean training image (bold: a vector of pixels) | subscripts table |
| $\boldsymbol{\epsilon}$ | Gaussian noise of the same shape, drawn from $\mathcal{N}(\mathbf{0}, \mathbf{I})$ | |
| $\bar{\alpha}_t$ | how much signal survives after $t$ steps; a product, hence the bar | subscripts table |
| $\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}$ | the noisy image $\mathbf{x}_t$ | proof below |
| $\boldsymbol{\epsilon}_\theta(\cdot, t)$ | the network: it sees the noisy image and the step, and predicts the noise | subscripts table |
| $\lVert \cdot \rVert^2$ | squared Euclidean length of the prediction error | norms section |

![Five density panels at t equal to 100, 200, 300, 500 and 1000: a narrow Gaussian near 1.5 that slides toward zero and widens until it matches a standard normal](/imgs/blogs/how-to-read-the-math-in-an-ml-paper-6.webp)

Now pass 3, with numbers. With the paper's schedule, $\bar{\alpha}_{100} \approx 0.897$ and $\bar{\alpha}_{500} \approx 0.079$. For a one-pixel image $x_0 = 1.5$, the noisy pixel at $t = 500$ has mean $\sqrt{0.079} \cdot 1.5 \approx 0.42$ and variance ${1} - 0.079 = 0.921$: almost pure noise. The filmstrip shows that slide for five values of $t$. A network that always predicted zero noise would score $\mathbb{E}\lVert \boldsymbol{\epsilon} \rVert^2 = d$, the number of pixels, which gives you a free sanity baseline for any training curve.

**In words:** pick a random image, a random noise level and random noise; mix image and noise in the proportions that step $t$ prescribes; ask the network to name the noise; penalise the squared error; and average over everything. The word "simple" is the argmin lesson at work: the paper's longer derivation (its Eq. 12) puts a $t$-dependent weight in front of the same squared error, and Eq. 14 drops it. Dropping a weight that varies with $t$ is not the same as dropping a constant, so this does change the objective; the paper reports it as a deliberate choice that improved sample quality.

## Proof, read line by line

Track A posts teach proof reading itself, so instead of a long theorem we decode one short, real one: the closed form that the DDPM loss relies on. The paper states it as Eq. 4, $q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\ \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\ (1-\bar{\alpha}_t)\mathbf{I})$, and leaves the proof to the reader. The standard proof is a three-line induction.

**Statement.** Assume:

- (A1) Each forward step is $\mathbf{x}_t = \sqrt{\alpha_t}\,\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}_t$, which is DDPM's Eq. 2 written with $\alpha_t = 1 - \beta_t$.
- (A2) The noises $\boldsymbol{\epsilon}_1, \dots, \boldsymbol{\epsilon}_T$ are independent $\mathcal{N}(\mathbf{0}, \mathbf{I})$ draws, independent of $\mathbf{x}_0$.
- (A3) $0 \lt \alpha_t \lt 1$, so every square root is real.

Then for every $t \ge 1$, $\mathbf{x}_t$ has the same distribution as $\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\bar{\boldsymbol{\epsilon}}$ with $\bar{\boldsymbol{\epsilon}} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ independent of $\mathbf{x}_0$.

**The idea in one sentence.** Unroll one step, notice the signal coefficients multiply into $\bar{\alpha}_t$, and merge the two noise terms using the fact that independent Gaussian variances add.

**Steps.**

1. Base case $t = 1$: $\bar{\alpha}_1 = \alpha_1$, so the claim is (A1) itself.
2. Inductive hypothesis: suppose the claim holds at step $t-1$, so $\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\,\bar{\boldsymbol{\epsilon}}'$ in distribution.
3. Substitute into (A1):

$$
\mathbf{x}_t = \sqrt{\alpha_t \bar{\alpha}_{t-1}}\,\mathbf{x}_0 + \sqrt{\alpha_t (1-\bar{\alpha}_{t-1})}\,\bar{\boldsymbol{\epsilon}}' + \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}_t.
$$

4. Signal term: $\alpha_t \bar{\alpha}_{t-1} = \bar{\alpha}_t$ by the definition of $\bar{\alpha}_t$ as a running product, so the first term is $\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0$.
5. Noise terms: $\bar{\boldsymbol{\epsilon}}'$ depends only on $\boldsymbol{\epsilon}_1, \dots, \boldsymbol{\epsilon}_{t-1}$, so it is independent of $\boldsymbol{\epsilon}_t$ by (A2). A sum of independent zero-mean Gaussians is a zero-mean Gaussian whose variance is the sum of the variances.
6. Add the variances: $\alpha_t(1-\bar{\alpha}_{t-1}) + (1-\alpha_t) = 1 - \alpha_t \bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t$.
7. So the two noise terms together equal $\sqrt{1-\bar{\alpha}_t}\,\bar{\boldsymbol{\epsilon}}$ in distribution, with $\bar{\boldsymbol{\epsilon}}$ independent of $\mathbf{x}_0$ (both pieces were, by (A2)). By induction the claim holds for all $t$.

Check step 6 with small numbers before trusting it. With $\alpha_1 = \alpha_2 = 0.5$: $\bar{\alpha}_2 = 0.25$, and the variance is $0.5 \cdot 0.5 + 0.5 = 0.75 = 1 - 0.25$. That is the 0.866 in the animation, since $\sqrt{0.75} \approx 0.866$.

**Where each assumption was used:**

| Assumption | Used in |
| --- | --- |
| (A1) forward step | steps 1, 3 |
| (A2) independent Gaussian noise | steps 5, 7 |
| (A3) $0 \lt \alpha_t \lt 1$ | step 3 (real square roots) |

**What breaks without (A2).** Suppose a buggy data loader reused the same noise at every step, $\boldsymbol{\epsilon}_1 = \boldsymbol{\epsilon}_2$. Then the noise terms add as standard deviations, not variances. With $\alpha_1 = \alpha_2 = 0.5$, the noise coefficient becomes $\sqrt{0.5}\sqrt{0.5} + \sqrt{0.5} \approx 0.5 + 0.707 = 1.207$, a variance of about 1.457 instead of 0.75. The closed form, and every loss built on it, would be wrong.

**Proof moves used:** `induction (layers / steps)`. The one fact invoked, that independent Gaussian variances add, is proved in the probability track.

Notice how the proof reads: each line is either a definition being unfolded, an assumption being cited by name, or a fact being invoked. When you read a paper appendix, label every line with one of those three, and the gaps (lines that are none of the three) are exactly where you should slow down.

## Check it numerically

The script checks both halves of this post: the closed form from the proof, by running the actual 500-step chain, and the loss itself, by Monte Carlo.

```python
import numpy as np

rng = np.random.default_rng(0)
T = 1000
beta = np.linspace(1e-4, 0.02, T)        # DDPM linear schedule
alpha = 1.0 - beta
abar = np.cumprod(alpha)                  # alpha-bar_t = prod of alpha_s

# 1) Closed form q(x_t | x_0) vs iterating the forward chain 500 times
x0, n, t_check = 1.5, 200_000, 500
x = np.full(n, x0)
for t in range(t_check):
    x = np.sqrt(alpha[t]) * x + np.sqrt(beta[t]) * rng.standard_normal(n)
ab = abar[t_check - 1]
print(f"alpha_bar_500 = {ab:.5f}")
print(f"mean: chain {x.mean():.4f}   closed form {np.sqrt(ab) * x0:.4f}")
print(f"var:  chain {x.var():.4f}   closed form {1 - ab:.4f}")

# 2) Monte Carlo L_simple (Eq. 14) on a 4-pixel "image" for two predictors
d = 4
x0v = rng.uniform(-1, 1, size=(n, d))
t = rng.integers(1, T + 1, size=n)        # t ~ Uniform{1, ..., T}
eps = rng.standard_normal((n, d))         # eps ~ N(0, I)
a = abar[t - 1][:, None]
xt = np.sqrt(a) * x0v + np.sqrt(1 - a) * eps
oracle = (xt - np.sqrt(a) * x0v) / np.sqrt(1 - a)   # cheats: knows x_0
for name, pred in [("predict zero", np.zeros_like(eps)), ("oracle", oracle)]:
    loss = np.mean(np.sum((eps - pred) ** 2, axis=1))
    print(f"L_simple, {name:12s}: {loss:.4f}")
```

```console
alpha_bar_500 = 0.07859
mean: chain 0.4226   closed form 0.4205
var:  chain 0.9228   closed form 0.9214
L_simple, predict zero: 4.0093
L_simple, oracle      : 0.0000
```

The chain agrees with the closed form to within one Monte Carlo standard error: with 200,000 samples the standard error of the mean is about $\sqrt{0.92 / 200000} \approx 0.0021$, and the gap is 0.0021. The zero predictor lands on $d = 4$ as the decode predicted, and the oracle, which cheats by knowing $\mathbf{x}_0$, gets zero, which shows the loss is minimised exactly by recovering the noise. Note `abar[t - 1]`: the 1-based paper index meeting a 0-based array.

## Where this shows up in practice

**Hugging Face Diffusers.** The `DDPMScheduler.add_noise` method in the [Diffusers library](https://huggingface.co/docs/diffusers/api/schedulers/ddpm) computes exactly $\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}$, the closed form we proved, so training never runs the step-by-step chain. Its training examples then use a mean-reduced MSE, which divides our $\lVert \cdot \rVert^2$ by the number of pixels. By the argmin rule that rescaling changes the loss value (your zero-noise baseline reads 1.0, not $d$) but not the optimum.

**Latent diffusion and Stable Diffusion.** Rombach et al., [*High-Resolution Image Synthesis with Latent Diffusion Models*](https://arxiv.org/abs/2112.10752) (CVPR 2022), train with the same noise-prediction objective, with one substitution: $\mathbf{x}_0$ is replaced by the latent code an autoencoder produces for the image. Once you can read Eq. 14, their objective reads in one pass, because only the meaning of one symbol changed.

**Norm arguments in PyTorch.** The `ord` argument of [`torch.linalg.vector_norm`](https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html) is the subscript $p$ in $\lVert x \rVert_p$: `ord=2` is the default Euclidean norm, `ord=1` the taxicab norm, `ord=float('inf')` the max norm. Gradient clipping by norm, a staple of LLM training, clips $\lVert \nabla_\theta L \rVert_2$, which is two pieces of this post's decoder in one call.

## What to remember

> - Read every equation in three passes: shape, then symbols with their types, then the smallest numbers that make it non-trivial.
> - $\mathbb{E}_{x \sim p}$ is a probability-weighted average, and its subscript names what is random; in code it becomes a minibatch mean.
> - argmin ignores positive scaling and added constants, which is why papers drop them; a weight that varies with the input is not a constant.
> - Hats estimate, tildes perturb, bars average or accumulate; papers index from 1, code from 0.
> - Label every proof line as a definition, a cited assumption, or an invoked fact; the unlabeled lines are where to slow down.

## Where next

The other Track A posts build out the rest of the vocabulary. Read [Sets, functions and maps](/blog/machine-learning/mathematics/sets-functions-and-maps-the-grammar-of-ml-math) to see a network as a composed map, and [Sums, products, einsum and log-sum-exp](/blog/machine-learning/mathematics/sums-products-einsum-and-log-sum-exp) for the index gymnastics behind attention. [Proof techniques for ML researchers](/blog/machine-learning/mathematics/proof-techniques-for-ml-researchers) goes beyond induction to contradiction and construction, [The inequality toolkit every appendix uses](/blog/machine-learning/mathematics/the-inequality-toolkit-every-appendix-uses) covers the moves that fill appendices, and [Limits, big-O and asymptotics in ML papers](/blog/machine-learning/mathematics/limits-big-o-and-asymptotics-in-ml-papers) decodes rate statements.

If you want the model behind today's equation, [The math of DDPM](/blog/machine-learning/image-generation/the-math-of-ddpm) derives the full loss that Eq. 14 simplifies. Later in the series, the linear algebra track turns the norms above into geometry, and a post on the Fokker–Planck equation explains why the forward process in the filmstrip forgets its starting point and settles on a standard Gaussian.
