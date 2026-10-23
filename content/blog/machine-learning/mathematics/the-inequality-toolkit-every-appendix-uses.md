---
title: "The Inequality Toolkit Every Appendix Uses: Eight Bounds, One Picture Each"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "Learn the eight inequalities behind almost every ML proof, each with a picture and a hand-checkable example, then derive the VAE evidence lower bound with Jensen."
tags: ["mathematics", "math-for-ai", "inequalities", "jensen", "cauchy-schwarz", "elbo", "probability", "proofs", "variational-inference"]
category: "machine-learning"
subcategory: "Mathematics"
author: "Hiep Tran"
featured: true
readTime: 18
---

> **Level:** Foundation · **Needs:** [How to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper), [Proof techniques for ML researchers](/blog/machine-learning/mathematics/proof-techniques-for-ml-researchers) · **Unlocks:** VAE and diffusion ELBO derivations, SGD convergence proofs, generalization bounds, robustness bounds

Section 2.2 of Kingma and Welling's *Auto-Encoding Variational Bayes* contains one line that the whole VAE rests on: $\log p_\theta(x^{(i)}) \geq \mathcal{L}(\theta, \phi; x^{(i)})$. The left side is the likelihood you want and cannot compute. The right side is a quantity you can compute and differentiate. Everything between them is carried by a single symbol, the $\geq$, and the paper spends one sentence justifying it.

Every theory appendix works this way: a chain of steps, almost each one of about eight inequalities, applied without comment. Paper math is a small, closed vocabulary; learn each tool once, with a picture, an example and a proof, and the "$\leq$ by Cauchy–Schwarz" lines stop being walls. By the end you will derive the VAE bound yourself.

![Triangle inequality on a 2D plane: u then v tip to tail is longer than u plus v drawn directly](/imgs/blogs/the-inequality-toolkit-every-appendix-uses-1.webp)

The picture above is the mental model for all eight: two routes to the same point and a guarantee about which is shorter. An inequality never gives the exact value of something hard; it tells you which side of a computable number it sits on.

## 1. The triangle inequality

**What it is for:** it lets you split a hard distance into two easier ones, at the cost of a little slack.

Walking to a café via the bakery is never shorter than walking straight there. That holds in any dimension, for any genuine norm.

Take the vectors in the figure, $u = (2, 1)$ and $v = (-1, 1)$, so $u + v = (1, 2)$. The detour has length $\lVert u \rVert + \lVert v \rVert = \sqrt{5} + \sqrt{2} \approx 2.236 + 1.414 = 3.650$. The direct route has length $\lVert u + v \rVert = \sqrt{1 + 4} = \sqrt{5} \approx 2.236$. The inset shows the equality case, $v = \tfrac{1}{2} u$: legs pointing the same way, both routes about 3.35.

Formally, for any norm $\lVert \cdot \rVert$ on $\mathbb{R}^d$ and any vectors $u, v$:

$$
\lVert u + v \rVert \leq \lVert u \rVert + \lVert v \rVert .
$$

In words: the length of a sum is at most the sum of the lengths.

How papers use it: the **add-and-subtract** move. To bound the distance from a learned model $\hat{f}$ to the truth $f^\star$, insert the best model in your class, $f_{\text{best}}$, and split:

$$
\lVert \hat{f} - f^\star \rVert \leq \lVert \hat{f} - f_{\text{best}} \rVert + \lVert f_{\text{best}} - f^\star \rVert .
$$

The first term is estimation error (too little data), the second approximation error (too small a model). Every telescoping sum over training steps is the same inequality applied $T$ times.

## 2. Cauchy–Schwarz

**What it is for:** it caps how large an inner product can be using only the lengths of the two vectors, which is how a proof gets rid of an angle it cannot control.

A stick's shadow on the ground is never longer than the stick, and equals it only when the stick lies flat. The inner product $a \cdot b$ is exactly the length of $a$ times the length of $b$'s shadow on $a$, so it can never exceed $\lVert a \rVert \lVert b \rVert$.

![Cauchy-Schwarz as a shadow: b projected onto a gives a shadow of length 1.79, shorter than b's length 2.24](/imgs/blogs/the-inequality-toolkit-every-appendix-uses-2.webp)

The numbers in the figure: $a = (2, 1)$, $b = (1, 2)$. The inner product is $a \cdot b = 2 \cdot 1 + 1 \cdot 2 = 4$. The lengths are $\lVert a \rVert = \lVert b \rVert = \sqrt{5}$, so $\lVert a \rVert \lVert b \rVert = 5$, and indeed $4 \leq 5$. The shadow has length $4 / \sqrt{5} \approx 1.79$, shorter than the stick's 2.24, and the ratio ${4/5 = 0.8}$ is $\cos \theta$, the cosine of the angle between them.

The animation turns $b$ through a full circle at fixed length; the shadow bar never passes the reference box.

<figure class="blog-anim">
<svg viewBox="0 0 640 380" role="img" aria-label="Vector b of fixed length rotates around the origin; its shadow on the horizontal vector a grows and shrinks but never exceeds the length of b, matching it only when b is parallel to a" style="width:100%;height:auto;max-width:760px">
<style>
.c1-ax{stroke:var(--border,#d1d5db);stroke-width:1.5}
.c1-circ{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:5 5}
.c1-a{stroke:var(--text-primary,#1f2937);stroke-width:3}
.c1-b{stroke:var(--text-secondary,#6b7280);stroke-width:3}
.c1-sh{stroke:var(--accent,#6366f1);stroke-width:7;stroke-linecap:round}
.c1-drop{stroke:var(--text-secondary,#6b7280);stroke-width:1.5;stroke-dasharray:4 4}
.c1-ref{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.c1-box{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:1.5}
.c1-bar{fill:var(--accent,#6366f1)}
.c1-t{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.c1-s{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
@keyframes c1-rot{0.00%{transform:rotate(-0.0deg)}6.25%{transform:rotate(-22.5deg)}12.50%{transform:rotate(-45.0deg)}18.75%{transform:rotate(-67.5deg)}25.00%{transform:rotate(-90.0deg)}31.25%{transform:rotate(-112.5deg)}37.50%{transform:rotate(-135.0deg)}43.75%{transform:rotate(-157.5deg)}50.00%{transform:rotate(-180.0deg)}56.25%{transform:rotate(-202.5deg)}62.50%{transform:rotate(-225.0deg)}68.75%{transform:rotate(-247.5deg)}75.00%{transform:rotate(-270.0deg)}81.25%{transform:rotate(-292.5deg)}87.50%{transform:rotate(-315.0deg)}93.75%{transform:rotate(-337.5deg)}100.00%{transform:rotate(-360.0deg)}}
@keyframes c1-shk{0.00%{transform:scaleX(1.0000)}6.25%{transform:scaleX(0.9239)}12.50%{transform:scaleX(0.7071)}18.75%{transform:scaleX(0.3827)}25.00%{transform:scaleX(0.0000)}31.25%{transform:scaleX(-0.3827)}37.50%{transform:scaleX(-0.7071)}43.75%{transform:scaleX(-0.9239)}50.00%{transform:scaleX(-1.0000)}56.25%{transform:scaleX(-0.9239)}62.50%{transform:scaleX(-0.7071)}68.75%{transform:scaleX(-0.3827)}75.00%{transform:scaleX(-0.0000)}81.25%{transform:scaleX(0.3827)}87.50%{transform:scaleX(0.7071)}93.75%{transform:scaleX(0.9239)}100.00%{transform:scaleX(1.0000)}}
@keyframes c1-dropk{0.00%{transform:translateX(0.00px) scaleY(0.0000)}6.25%{transform:translateX(-9.13px) scaleY(0.3827)}12.50%{transform:translateX(-35.15px) scaleY(0.7071)}18.75%{transform:translateX(-74.08px) scaleY(0.9239)}25.00%{transform:translateX(-120.00px) scaleY(1.0000)}31.25%{transform:translateX(-165.92px) scaleY(0.9239)}37.50%{transform:translateX(-204.85px) scaleY(0.7071)}43.75%{transform:translateX(-230.87px) scaleY(0.3827)}50.00%{transform:translateX(-240.00px) scaleY(0.0000)}56.25%{transform:translateX(-230.87px) scaleY(-0.3827)}62.50%{transform:translateX(-204.85px) scaleY(-0.7071)}68.75%{transform:translateX(-165.92px) scaleY(-0.9239)}75.00%{transform:translateX(-120.00px) scaleY(-1.0000)}81.25%{transform:translateX(-74.08px) scaleY(-0.9239)}87.50%{transform:translateX(-35.15px) scaleY(-0.7071)}93.75%{transform:translateX(-9.13px) scaleY(-0.3827)}100.00%{transform:translateX(0.00px) scaleY(-0.0000)}}
@keyframes c1-bark{0.00%{transform:scaleX(1.0000)}6.25%{transform:scaleX(0.9239)}12.50%{transform:scaleX(0.7071)}18.75%{transform:scaleX(0.3827)}25.00%{transform:scaleX(0.0000)}31.25%{transform:scaleX(0.3827)}37.50%{transform:scaleX(0.7071)}43.75%{transform:scaleX(0.9239)}50.00%{transform:scaleX(1.0000)}56.25%{transform:scaleX(0.9239)}62.50%{transform:scaleX(0.7071)}68.75%{transform:scaleX(0.3827)}75.00%{transform:scaleX(0.0000)}81.25%{transform:scaleX(0.3827)}87.50%{transform:scaleX(0.7071)}93.75%{transform:scaleX(0.9239)}100.00%{transform:scaleX(1.0000)}}
.c1-rot{transform-box:view-box;transform-origin:170px 190px;animation:c1-rot 16s linear infinite}
.c1-shm{transform-box:view-box;transform-origin:170px 190px;animation:c1-shk 16s linear infinite}
.c1-dropm{transform-box:view-box;transform-origin:290px 190px;animation:c1-dropk 16s linear infinite}
.c1-barm{transform-box:view-box;transform-origin:400px 0;animation:c1-bark 16s linear infinite}
@media (prefers-reduced-motion:reduce){.c1-rot,.c1-shm,.c1-dropm,.c1-barm{animation:none}}
</style>
<line class="c1-ax" x1="20" y1="190" x2="360" y2="190"/>
<line class="c1-ax" x1="170" y1="40" x2="170" y2="340"/>
<circle class="c1-circ" cx="170" cy="190" r="120"/>
<line class="c1-sh c1-shm" x1="170" y1="190" x2="290" y2="190"/>
<line class="c1-a" x1="170" y1="190" x2="350" y2="190"/>
<text class="c1-t" x="356" y="195">a</text>
<line class="c1-drop c1-dropm" x1="290" y1="190" x2="290" y2="70"/>
<g class="c1-rot"><line class="c1-b" x1="170" y1="190" x2="290" y2="190"/><circle cx="290" cy="190" r="5" fill="var(--text-secondary,#6b7280)"/></g>
<text class="c1-s" x="110" y="360">dashed circle: every b with ‖b‖ fixed</text>
<text class="c1-t" x="400" y="120">‖b‖</text>
<rect class="c1-ref" x="400" y="130" width="200" height="22" rx="4"/>
<text class="c1-t" x="400" y="200">|shadow of b on a|</text>
<rect class="c1-bar c1-barm" x="400" y="210" width="200" height="22" rx="4"/>
<rect class="c1-box" x="400" y="210" width="200" height="22" rx="4"/>
<text class="c1-s" x="400" y="270">|a·b| = ‖a‖ × |shadow|</text>
<text class="c1-s" x="400" y="292">the bar never passes the box,</text>
<text class="c1-s" x="400" y="314">so |a·b| ≤ ‖a‖‖b‖; it fills the</text>
<text class="c1-s" x="400" y="336">box only when b lies along a</text>
</svg>
<figcaption>Cauchy–Schwarz in motion: b keeps its length and turns a full circle. Its shadow on a (accent) swells and shrinks, filling the reference box only at 0° and 180°, the two equality cases.</figcaption>
</figure>

Formally, for vectors $u, v$ in $\mathbb{R}^d$ (or any real inner product space):

$$
\lvert \langle u, v \rangle \rvert \leq \lVert u \rVert \, \lVert v \rVert ,
$$

where $\langle u, v \rangle = \sum_i u_i v_i$ and $\lVert u \rVert = \sqrt{\langle u, u \rangle}$. In words: the inner product is at most the product of the lengths, with equality exactly when $u$ and $v$ are parallel (proved below).

The same inequality holds for random variables, with $\langle X, Y \rangle = \mathbb{E}[XY]$:

$$
\bigl(\mathbb{E}[XY]\bigr)^2 \leq \mathbb{E}[X^2] \, \mathbb{E}[Y^2] .
$$

Applied to centred variables, it says correlation lies in $[-1, 1]$; the vector form says the same of cosine similarity.

How papers use it: convergence proofs for gradient methods are full of cross terms like $\langle \nabla f(\theta_t), g_t - \nabla f(\theta_t) \rangle$, the true gradient against the noise in a stochastic estimate. The proof cannot know their angle, so it writes "$\leq \lVert \nabla f(\theta_t) \rVert \, \lVert g_t - \nabla f(\theta_t) \rVert$ by Cauchy–Schwarz" and keeps only lengths, which the assumptions control.

## 3. Jensen's inequality

**What it is for:** it compares a curved function of an average with the average of the curved function. This is the inequality that produces the ELBO.

A convex function is a bowl. Stretch a rope (the chord) between two points on it: the rope sits above the surface. The average of the function values lies on the chord; the function at the average lies on the bowl, below. For a concave function like $\log$, the bowl flips and the chord sits below.

![Jensen in two panels: for x squared the chord midpoint 5 sits above the curve value 4; for log x the chord midpoint 0.693 sits below the curve value 0.916](/imgs/blogs/the-inequality-toolkit-every-appendix-uses-3.webp)

Left panel: let $X$ be 1 or 3 with probability one half each, and $f(x) = x^2$. Then $\mathbb{E}[X] = 2$ and $f(\mathbb{E}[X]) = 4$, while $\mathbb{E}[f(X)] = (1 + 9)/2 = 5$. The gap ${5 - 4 = 1}$ is exactly the variance of $X$, because $\mathbb{E}[X^2] - (\mathbb{E}[X])^2$ is the definition of variance.

Right panel: let $X$ be 1 or 4 with probability one half each, and use the concave $\log$. Then $\log \mathbb{E}[X] = \log 2.5 \approx 0.916$, while $\mathbb{E}[\log X] = (\log 1 + \log 4)/2 = (0 + 1.386)/2 = 0.693$. The average of the logs is smaller by about 0.223. In the VAE, this gap is the distance between the true log-likelihood and the bound.

Formally, let $\varphi$ be convex and $X$ a random variable with $\mathbb{E}\lvert X \rvert$ and $\mathbb{E}\lvert \varphi(X) \rvert$ finite. Then:

$$
\varphi\bigl(\mathbb{E}[X]\bigr) \leq \mathbb{E}\bigl[\varphi(X)\bigr] .
$$

In words: a convex function of the average is at most the average of the function. For concave $\varphi$ it flips; for strictly convex $\varphi$, equality only when $X$ is constant. The finite form, $\varphi(\sum_i \lambda_i x_i) \leq \sum_i \lambda_i \varphi(x_i)$ for weights $\lambda_i \geq 0$ summing to 1, follows by induction on $n$ from the two-point definition of convexity.

How papers use it: the ELBO (decoded below), KL $\geq 0$, and the bound $\mathbb{E}[\lVert g \rVert] \leq \sqrt{\mathbb{E}\lVert g \rVert^2}$ that turns a second-moment assumption into a first-moment one. For a finance-flavoured second angle on convexity, see [Convexity and Jensen in Math for Quants](/blog/trading/math-for-quants/convexity-jensen-math-for-quants).

## 4. Markov and Chebyshev

**What they are for:** they turn knowledge of an average into a guarantee about how often you see something extreme. This is where "with probability at least $1 - \delta$" comes from.

If a room's average salary is 50 thousand and nobody earns less than zero, at most half the room earns 100 thousand or more; otherwise they alone would push the average above 50. That is Markov's inequality. It knows only the mean, and it fails for quantities that can be negative.

![Markov and Chebyshev on a four-outcome distribution: true tail probabilities 0.2, 0.1, 0.1 next to bounds 0.5, 0.2, 0.1375](/imgs/blogs/the-inequality-toolkit-every-appendix-uses-4.webp)

The worked example in the figure: $X$ takes the values 0, 1, 2, 5 with probabilities 0.5, 0.3, 0.1, 0.1. Its mean is ${0.3 + 0.2 + 0.5 = 1}$. Markov says $P(X \geq 2) \leq 1/2 = 0.5$, while the true value is ${0.1 + 0.1 = 0.2}$. It says $P(X \geq 5) \leq 1/5 = 0.2$, while the truth is 0.1. Both hold; both are loose.

Formally, **Markov's inequality**: if $X \geq 0$ and $a \gt 0$, then

$$
P(X \geq a) \leq \frac{\mathbb{E}[X]}{a} .
$$

In words: the chance of reaching the level $a$ is at most the mean divided by that level. Proof: $X \geq a \cdot \mathbf{1}\{X \geq a\}$ pointwise; take expectations.

**Chebyshev's inequality** is Markov applied to the nonnegative quantity $(X - \mu)^2$, where $\mu = \mathbb{E}[X]$:

$$
P\bigl(\lvert X - \mu \rvert \geq k\bigr) \leq \frac{\operatorname{Var}(X)}{k^2} .
$$

In words: the probability of landing at least $k$ away from the mean is at most the variance divided by $k$ squared. In our example $\mathbb{E}[X^2] = 0.3 + 0.4 + 2.5 = 3.2$, so $\operatorname{Var}(X) = 3.2 - 1 = 2.2$. Chebyshev with $k = 4$ gives ${2.2/16 = 0.1375}$, against a true $P(\lvert X - 1 \rvert \geq 4) = P(X = 5) = 0.1$. Knowing the variance bought a much tighter ceiling.

How papers use them: Markov turns "expected excess loss at most $\varepsilon$" into "with probability at least $1 - \delta$, excess loss at most $\varepsilon / \delta$". Chebyshev gives the weak law of large numbers: an average of $n$ independent draws with variance $\sigma^2$ misses the mean by $\varepsilon$ or more with probability at most $\sigma^2 / (n \varepsilon^2)$. With our $\sigma^2 = 2.2$, $n = 100$ samples and $\varepsilon = 0.5$, that is ${2.2 / 25 = 0.088}$. Exponential versions (Hoeffding, Bernstein) get a later post; [the Math for Quants concentration post](/blog/trading/math-for-quants/concentration-inequalities-sample-complexity-math-for-quants) covers them now.

> An inequality never tells you the answer. It tells you which side of a number you can compute the answer is on, and in a proof that is all you need.

## 5. The compact three: Hölder, Young / AM–GM, and the union bound

### Hölder

**What it is for:** Cauchy–Schwarz when the two vectors are measured with different rulers. If $p, q \geq 1$ with ${1/p + 1/q = 1}$, then $\lvert \langle x, y \rangle \rvert \leq \lVert x \rVert_p \lVert y \rVert_q$. Setting $p = q = 2$ recovers Cauchy–Schwarz; the pair used most in ML is $p = 1$, $q = \infty$, where $\lVert x \rVert_1 = \sum_i \lvert x_i \rvert$ and $\lVert y \rVert_\infty = \max_i \lvert y_i \rvert$.

Example: $x = (1, -2, 3)$ and $y = (2, 1, -1)$ give $\langle x, y \rangle = 2 - 2 - 3 = -3$. Hölder with $(1, \infty)$ bounds $\lvert -3 \rvert$ by $\lVert x \rVert_1 \lVert y \rVert_\infty = 6 \cdot 2 = 12$. A proof picks the pair of norms its assumptions control. We state Hölder without proof; it follows from Young's inequality applied term by term (Steele, *The Cauchy–Schwarz Master Class*, Chapter 9).

### Young and AM–GM

**What it is for:** splitting a product into a sum, so each factor can be bounded on its own. For $a, b \geq 0$ and the same conjugate pair $p, q$: $ab \leq a^p/p + b^q/q$. The case $p = q = 2$ is $ab \leq (a^2 + b^2)/2$, which is just $(a - b)^2 \geq 0$ rearranged. Its cousin **AM–GM** says $\sqrt{ab} \leq (a + b)/2$: for $a = 2$, $b = 8$, the geometric mean 4 is below the arithmetic mean 5.

The form papers actually use has a free knob $\varepsilon \gt 0$:

$$
ab \leq \frac{\varepsilon}{2} a^2 + \frac{1}{2\varepsilon} b^2 .
$$

In words: shrink the $a^2$ term as much as you like by paying on the $b^2$ term. With $a = 3$, $b = 4$, so $ab = 12$: choosing $\varepsilon = 1$ gives ${4.5 + 8 = 12.5}$; choosing $\varepsilon = 1/2$ gives ${2.25 + 16 = 18.25}$; choosing $\varepsilon = b/a = 4/3$ gives ${6 + 6 = 12}$, which is equality. SGD convergence proofs use it to split the cross term $\langle g_t, \theta_t - \theta^\star \rangle$ into a gradient term and a distance term, then tune $\varepsilon$ so one cancels.

### The union bound

**What it is for:** making many guarantees hold at the same time. The probability that at least one of several bad events happens is at most the sum of their individual probabilities: $P(A_1 \cup \dots \cup A_k) \leq \sum_i P(A_i)$.

![Union bound on a 10 by 10 grid: events of 10, 8 and 6 cells with 4 overlapping cells give a union of 20 cells against a sum of 24](/imgs/blogs/the-inequality-toolkit-every-appendix-uses-5.webp)

In the grid, each cell is an equally likely outcome. Event $A$ covers 10 cells, $B$ covers 8, $C$ covers 6, so the sum is ${0.10 + 0.08 + 0.06 = 0.24}$. But four cells sit in two events at once and get counted twice, so the union covers only ${24 - 4 = 20}$ cells, probability 0.20. The bound overcounts exactly the overlaps, so it is nearly tight for rare, mostly disjoint failures.

How papers use it: "with probability at least $1 - \delta$, for all $k$ simultaneously". For 10 confidence intervals that must all hold with probability 0.95, give each failure probability ${0.05/10 = 0.005}$. Finite-class generalization bounds do the same over every hypothesis, which is where $\log \lvert \mathcal{H} \rvert$ comes from.

## Decoding the VAE bound

### Paper sentence, decoded

**Paper:** Diederik P. Kingma and Max Welling, *Auto-Encoding Variational Bayes*, 2013, [arXiv:1312.6114](https://arxiv.org/abs/1312.6114), Section 2.2 "The variational bound", Eq. (1) to (3).

$$
\log p_\theta(x^{(i)}) = D_{KL}\bigl(q_\phi(z \mid x^{(i)}) \,\Vert\, p_\theta(z \mid x^{(i)})\bigr) + \mathcal{L}(\theta, \phi; x^{(i)}) \qquad (1)
$$

$$
\log p_\theta(x^{(i)}) \geq \mathcal{L}(\theta, \phi; x^{(i)}) = \mathbb{E}_{q_\phi(z \mid x)}\bigl[-\log q_\phi(z \mid x) + \log p_\theta(x, z)\bigr] \qquad (2)
$$

$$
\mathcal{L}(\theta, \phi; x^{(i)}) = -D_{KL}\bigl(q_\phi(z \mid x^{(i)}) \,\Vert\, p_\theta(z)\bigr) + \mathbb{E}_{q_\phi(z \mid x^{(i)})}\bigl[\log p_\theta(x^{(i)} \mid z)\bigr] \qquad (3)
$$

Eq. (1) splits the log-likelihood into a KL term plus the bound; Eq. (2) is the bound; Eq. (3) is "reconstruction minus KL to the prior", the form a VAE trainer computes.

| Symbol | What it is | Built where |
| --- | --- | --- |
| $x^{(i)}$ | the $i$-th data point, say an image | notation, from the intro post |
| $z$ | the latent code that generated it | here |
| $p_\theta(x, z)$ | the decoder's joint model: prior on $z$ times likelihood of $x$ given $z$ | here |
| $p_\theta(x^{(i)})$ | the marginal likelihood $\int p_\theta(x^{(i)}, z) \, dz$, intractable | here |
| $q_\phi(z \mid x)$ | the encoder: a guess at which codes explain $x$ | here |
| $D_{KL}(q \,\Vert\, p)$ | KL divergence, $\mathbb{E}_q[\log q - \log p]$, always $\geq 0$ | Jensen, section 3 |
| $\mathcal{L}$ | the evidence lower bound (ELBO) | derived below |

The paper gets the $\geq$ in Eq. (2) from "KL is nonnegative". Most later papers take the direct route: multiply and divide by $q_\phi(z \mid x)$, then use concavity of $\log$:

$$
\begin{aligned}
\log p_\theta(x) &= \log \int p_\theta(x, z) \, dz = \log \mathbb{E}_{q_\phi(z \mid x)}\!\left[\frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right] \\
&\geq \mathbb{E}_{q_\phi(z \mid x)}\!\left[\log \frac{p_\theta(x, z)}{q_\phi(z \mid x)}\right] = \mathcal{L}(\theta, \phi; x) .
\end{aligned}
$$

In words: the log of an average ratio is at least the average log ratio, the right panel of the Jensen figure with $p_\theta(x, z)/q_\phi(z \mid x)$ as $X$. It assumes $q_\phi(z \mid x) \gt 0$ wherever $p_\theta(x, z) \gt 0$.

Both routes are the same inequality, since KL $\geq 0$ is itself Jensen: $-D_{KL}(q \,\Vert\, p) = \mathbb{E}_q[\log (p/q)] \leq \log \mathbb{E}_q[p/q] = \log 1 = 0$. The Jensen gap is exactly the KL term in Eq. (1): the bound is tight when $q_\phi$ equals the true posterior, making the ratio constant.

In one sentence: the uncomputable log-likelihood is at least an expectation you can estimate by sampling codes from the encoder, and the loss is how far the encoder is from the true posterior.

## Proof, read line by line

We prove Cauchy–Schwarz with the discriminant argument, simplified from the standard presentation in Steele, *The Cauchy–Schwarz Master Class* (2004), Chapter 1.

**Statement.** Let $u, v$ be vectors in a real vector space with an inner product $\langle \cdot, \cdot \rangle$ satisfying:

- (A1) bilinearity and symmetry: $\langle \cdot, \cdot \rangle$ is linear in each argument and $\langle u, v \rangle = \langle v, u \rangle$;
- (A2) positivity: $\langle w, w \rangle \geq 0$ for every $w$;
- (A3) definiteness: $\langle w, w \rangle = 0$ only when $w = 0$.

Then $\lvert \langle u, v \rangle \rvert \leq \lVert u \rVert \lVert v \rVert$, with equality if and only if $u$ and $v$ are parallel.

**The idea in one sentence:** the squared length of $u + t v$ is a parabola in $t$ that never dips below zero, and a parabola that never dips below zero has a nonpositive discriminant.

![Proof picture: q(t) for nonparallel vectors stays above zero with minimum 1.8; for parallel vectors it touches zero once at t equals 1](/imgs/blogs/the-inequality-toolkit-every-appendix-uses-6.webp)

1. If $v = 0$, both sides are 0. Assume $v \neq 0$. *Case split; uses (A1) to get $\langle u, 0 \rangle = 0$.*
2. Define, for every real number $t$, $q(t) = \langle u + t v, \, u + t v \rangle$. *A construction: the free parameter $t$ is the whole trick.*
3. $q(t) \geq 0$ for every $t$. *By (A2) applied to $w = u + t v$.*
4. Expanding, $q(t) = \lVert v \rVert^2 t^2 + 2 \langle u, v \rangle t + \lVert u \rVert^2$. *By (A1): bilinearity splits the product into four terms, and symmetry merges the two cross terms.*
5. The leading coefficient $\lVert v \rVert^2$ is strictly positive. *By (A3), since $v \neq 0$.* So $q$ is an upward-opening parabola.
6. An upward parabola $\alpha t^2 + \beta t + \gamma$ that is never negative has at most one real root, so $\beta^2 - 4 \alpha \gamma \leq 0$. *If the discriminant were positive, the parabola would cross zero twice and be negative in between, contradicting step 3.*
7. Substituting $\alpha = \lVert v \rVert^2$, $\beta = 2 \langle u, v \rangle$, $\gamma = \lVert u \rVert^2$ gives $4 \langle u, v \rangle^2 - 4 \lVert u \rVert^2 \lVert v \rVert^2 \leq 0$. *Arithmetic.*
8. Divide by 4 and take nonnegative square roots: $\lvert \langle u, v \rangle \rvert \leq \lVert u \rVert \lVert v \rVert$. *Square root is increasing on $[0, \infty)$.*
9. Equality case: a zero discriminant gives a root $t_0$ with $\langle u + t_0 v, u + t_0 v \rangle = 0$, so $u = -t_0 v$ and the vectors are parallel; parallel vectors give equality by substitution. *By (A3).*

The figure shows both cases. For $u = (2, 1)$, $v = (1, 2)$ we get $q(t) = 5t^2 + 8t + 5$; its discriminant is ${64 - 100 = -36}$ and its lowest value is ${1.8}$ at $t = -0.8$. For the parallel pair $u = (2, 1)$, $v = (-2, -1)$ we get $q(t) = 5t^2 - 10t + 5 = 5(t - 1)^2$, which touches zero once at $t = 1$.

**Where each assumption was used:**

| Assumption | Steps |
| --- | --- |
| (A1) bilinearity and symmetry | 1, 4 |
| (A2) positivity | 3 |
| (A3) definiteness | 5, 9 |

**What breaks without (A2).** Use the "Minkowski inner product" on $\mathbb{R}^2$, $\langle u, v \rangle = u_1 v_1 - u_2 v_2$, which satisfies (A1) but not (A2). Take $u = (1, 1)$ and $v = (1, 0)$. Then $\langle u, u \rangle = 1 - 1 = 0$ and $\langle u, v \rangle = 1$, so the inequality would claim $1 \leq 0 \cdot 1 = 0$. Step 3 is gone: $q(t) = t^2 + 2t$ is negative for $t$ between $-2$ and 0.

**Proof moves used:** a construction (the auxiliary quadratic) plus positivity of a square; this is the result every later `Cauchy–Schwarz` move cites.

## Check it numerically

The snippet stress-tests triangle and Cauchy–Schwarz, recomputes the Markov and Chebyshev numbers, and checks the ELBO where the log-likelihood is known exactly: prior $z \sim \mathcal{N}(0, 1)$, likelihood $x \mid z \sim \mathcal{N}(z, 1)$, so $x \sim \mathcal{N}(0, 2)$ and the true posterior at $x = 1$ is $\mathcal{N}(0.5, 0.5)$.

```python
import numpy as np
rng = np.random.default_rng(0)

# Triangle and Cauchy-Schwarz on 100,000 random pairs in 64 dimensions
u, v = rng.normal(size=(2, 100_000, 64))
nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
print("triangle violations:", np.sum(np.linalg.norm(u + v, axis=1) > nu + nv + 1e-12))
print("C-S violations:     ", np.sum(np.abs((u * v).sum(1)) > nu * nv + 1e-12))

# Markov and Chebyshev on the post's 4-outcome distribution
x, p = np.array([0, 1, 2, 5]), np.array([0.5, 0.3, 0.1, 0.1])
m, var = x @ p, (x**2) @ p - (x @ p) ** 2
print(f"E[X]={m:.2f} Var={var:.2f}  P(X>=5)={p[x>=5].sum():.3f} <= Markov {m/5:.3f}")
print(f"P(|X-1|>=4)={p[abs(x-m)>=4].sum():.3f} <= Chebyshev {var/16:.4f}")

# ELBO via Jensen: p(z)=N(0,1), p(x|z)=N(z,1), so p(x)=N(0,2); observe x=1
xo = 1.0
logN = lambda y, mu, s2: -0.5 * np.log(2 * np.pi * s2) - (y - mu) ** 2 / (2 * s2)
log_px = logN(xo, 0.0, 2.0)
for name, mu, s2 in [("q = prior N(0,1)", 0.0, 1.0), ("q = posterior N(.5,.5)", 0.5, 0.5)]:
    z = rng.normal(mu, np.sqrt(s2), size=1_000_000)
    elbo = np.mean(logN(xo, z, 1.0) + logN(z, 0, 1) - logN(z, mu, s2))
    print(f"{name:24s} ELBO={elbo:.4f}  log p(x)={log_px:.4f}  gap={log_px-elbo:.4f}")
kl = np.log(np.sqrt(0.5)) + (1 + 0.25) / (2 * 0.5) - 0.5
print(f"closed-form KL(prior || posterior) = {kl:.4f}")
```

Output:

```
triangle violations: 0
C-S violations:      0
E[X]=1.00 Var=2.20  P(X>=5)=0.100 <= Markov 0.200
P(|X-1|>=4)=0.100 <= Chebyshev 0.1375
q = prior N(0,1)         ELBO=-1.9184  log p(x)=-1.5155  gap=0.4029
q = posterior N(.5,.5)   ELBO=-1.5155  log p(x)=-1.5155  gap=0.0000
closed-form KL(prior || posterior) = 0.4034
```

With the prior as encoder the Monte Carlo gap is 0.4029 against the exact KL of 0.4034, a difference within the standard error of a million-sample average. With the true posterior the log ratio is constant, so the gap is zero: Jensen's equality case.

## Where this shows up in practice

**The diffusion training objective.** Ho, Jain and Abbeel's *Denoising Diffusion Probabilistic Models* (2020, [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)) trains by "optimizing the usual variational bound on negative log likelihood", their Eq. (3). It is the VAE bound with the latent $z$ replaced by the chain of noisy images $x_1, \dots, x_T$, via the same Jensen step. The [DDPM math post](/blog/machine-learning/image-generation/the-math-of-ddpm) shows how it becomes the noise-prediction loss; [VAEs from scratch](/blog/machine-learning/image-generation/variational-autoencoders-from-scratch) builds the encoder side.

**Gradient clipping in DP-SGD.** Abadi et al., *Deep Learning with Differential Privacy* (2016, [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)), clip each per-example gradient to $\ell_2$ norm at most $C$ before summing and adding Gaussian noise (their Algorithm 1). Privacy needs a bound on how far the sum moves when one example changes: adding or removing one shifts it by one clipped vector, at most $C$; replacing one shifts it by $g_i - g_i'$, which the triangle inequality caps at ${2C}$. The noise is calibrated to that number.

**Adversarial examples and Hölder.** Goodfellow, Shlens and Szegedy, *Explaining and Harnessing Adversarial Examples* (2014, [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)), Section 3, ask how much a linear score $w^\top x$ can change under a perturbation $\eta$ with $\lVert \eta \rVert_\infty \leq \varepsilon$. Hölder with $(1, \infty)$ answers exactly: $\lvert w^\top \eta \rvert \leq \varepsilon \lVert w \rVert_1$, attained by $\eta = \varepsilon \operatorname{sign}(w)$. That maximizer is the fast gradient sign method, and $\lVert w \rVert_1$ growing with dimension is the paper's linear explanation of adversarial fragility.

## What to remember

- Triangle and Cauchy–Schwarz remove what you cannot control (a middle point, an angle) and leave only lengths.
- Jensen is the ELBO: the log of an average is at least the average of the log, and the gap is exactly the KL from encoder to true posterior.
- Markov and Chebyshev turn means and variances into "with probability at least $1 - \delta$"; the union bound makes many such statements hold at once at the price of a sum.
- Read every bound with its equality case in mind: parallel vectors, a constant random variable, disjoint events. That is where the bound is tight, and it tells you what the proof is really assuming.

## Where next

If any notation here was unfamiliar, start with [how to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper). The induction proof of finite Jensen and the contradiction in step 6 above use the moves from [proof techniques for ML researchers](/blog/machine-learning/mathematics/proof-techniques-for-ml-researchers), and the rate statements that these bounds feed into, such as $O(1/\sqrt{T})$, are decoded in [limits, big-O and asymptotics in ML papers](/blog/machine-learning/mathematics/limits-big-o-and-asymptotics-in-ml-papers).

Later posts on concentration inequalities and on KL divergence build directly on Chebyshev and Jensen.
