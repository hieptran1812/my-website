---
title: "Limits, Big-O and Asymptotics in ML Papers: Reading Rates, Bounds and Regret"
date: "2026-10-23"
publishDate: "2026-10-23"
description: "Read O, Θ, Ω, o and Õ statements, tell asymptotic from non-asymptotic claims, decode 'with high probability', and see what O(1/√T) versus a linear rate costs in iterations."
tags: ["mathematics", "math-for-ai", "big-o", "asymptotics", "convergence-rates", "regret", "adam", "limits", "high-probability-bounds"]
category: "machine-learning"
subcategory: "Mathematics"
author: "Hiep Tran"
featured: true
readTime: 18
---

> **Level:** Foundation · **Needs:** [How to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper), [Sets, functions and maps](/blog/machine-learning/mathematics/sets-functions-and-maps-the-grammar-of-ml-math) · **Unlocks:** convergence theorems for SGD and Adam, regret bounds, complexity claims, generalization bounds

The Adam paper (Kingma and Ba, 2014) closes its theory section with a two-line corollary: the average regret of Adam satisfies $R(T)/T = O(1/\sqrt{T})$, "thus $\lim_{T\to\infty} R(T)/T = 0$". Almost every optimizer paper since has a sentence of that shape. If you have ever nodded at it without being sure what it promised, this post is for you. By the end you will be able to read that corollary, the theorem it rests on, and the 2018 paper that showed the theorem's proof has a gap.

Paper math is a small, closed vocabulary. Asymptotic notation is maybe ten symbols, and once you have each one as a picture, a worked example and a short proof, a whole category of theorem statements stops being a wall. The figure below shows where this vocabulary sits: at the bottom of the ladder, under everything else.

## The picture first: where asymptotic notation lives

![A ladder of the series tracks with rung A lit, arrows pointing to convergence theorems, regret bounds, complexity claims and generalization bounds](/imgs/blogs/limits-big-o-and-asymptotics-in-ml-papers-1.webp)

Every arrow on the right is a paper family whose headline results are written in this notation: how fast error shrinks, how fast mistakes accumulate, how cost grows with length, how error falls with sample size. Four families, one grammar.

## Limits: where a sequence is heading

A **limit** answers one question: if you keep going forever, what value do you settle near? It is the tool for talking about "eventually" without ever reaching the end.

Picture walking toward a wall, each step covering half the remaining distance. You never touch it, yet the wall is clearly where you are heading. Where the analogy breaks: real sequences can overshoot and oscillate and still converge, as long as the swings die out.

![A sequence oscillating around 1 with shrinking swings; from n equals 14 every point stays inside a band of plus or minus 0.15](/imgs/blogs/limits-big-o-and-asymptotics-in-ml-papers-2.webp)

The chart plots $a_n = 1 + (-1)^n \cdot 2/n$. The first terms jump around: ${a_1} = -1$, ${a_2} = 2$, ${a_3} \approx 0.333$, ${a_4} = 1.5$. Now draw a band of half-width 0.15 around 1. The term ${a_{13}} \approx 0.846$ is still outside, but ${a_{14}} \approx 1.143$ is inside, and from there on every term is inside because the distance to 1 is ${2}/{n}$, and $2/n \lt 0.15$ exactly when $n \gt 13.3$.

Here is the test that defines convergence, stated in words. A sequence $a_n$ converges to $L$, written

$$
\lim_{n \to \infty} a_n = L,
$$

when for every band you draw around $L$, no matter how thin, there is some index after which the sequence enters the band and never leaves. A thinner band just pushes the index further out: for half-width 0.01 you need $n \gt 200$. The formal epsilon-delta version, later in the series, is this picture in symbols.

In ML papers, limits appear as "the average regret goes to zero" or as a loss approaching an irreducible floor. A limit tells you where you end up, never how long it takes. That is the job of the next tool.

## The big-O family: growth classes, not numbers

Big-O notation sorts functions into **growth classes**, the way cars are sorted into classes by top speed. A sedan with a heavy trailer and the same sedan without one are in the same class; a bicycle is in a different one. The trailer is a constant factor. The class is about what happens as the road goes on forever.

### O: an upper bound that ignores constants

We write

$$
f(n) = O\big(g(n)\big) \quad \text{if there exist } C \gt 0 \text{ and } n_0 \text{ such that } |f(n)| \le C\, g(n) \text{ for all } n \ge n_0.
$$

In words: from some point on, $f$ is at most a constant multiple of $g$. Here $C$ is the trailer we agree to ignore and $n_0$ is the "from some point on".

Take $f(n) = 3n^2 + 100n$, a cost with a quadratic term and a linear overhead. Claim: $f(n) = O(n^2)$. Pick $C = 4$. We need $3n^2 + 100n \le 4n^2$, which simplifies to $100n \le n^2$, which holds exactly when $n \ge 100$. So $C = 4$ and $n_0 = 100$ work. At $n = 100$ the two sides are equal: $30000 + 10000 = 40000 = 4 \cdot {100}^{2}$.

Two traps. The equals sign really means "belongs to the class", so $n = O(n^2)$ is true and $n^2 = O(n)$ is false. And big-O is only an upper bound: an $O(n^3)$ algorithm may actually be linear.

### Ω and Θ: the floor and the sandwich

**Big-Omega** is the mirror image: $f(n) = \Omega(g(n))$ if $f(n) \ge c\, g(n)$ for some $c \gt 0$ and all large $n$. It is a floor. **Big-Theta** is both at once: $f(n) = \Theta(g(n))$ when $c\, g(n) \le f(n) \le C\, g(n)$ for all large $n$. For our example, $3n^2 \le 3n^2 + 100n$ always, so with $c = 3$, $C = 4$, $n_0 = 100$ we get $f(n) = \Theta(n^2)$.

![Left: log-log lines for square root of n, n, n log n and n squared with slopes one half, one and two. Right: 3n squared plus 100n trapped between 3n squared and 4n squared after n0 equals 100](/imgs/blogs/limits-big-o-and-asymptotics-in-ml-papers-3.webp)

Log-log axes are the right way to look at all of this, and the left panel shows why. If $f(n) = n^k$ then $\log f = k \log n$, so a power law becomes a straight line whose slope is the exponent. Constants only shift the line up or down. The right panel is the sandwich: on log-log axes $3n^2$ and $4n^2$ are parallel lines, and $f$ bends into the corridor between them at $n_0 = 100$ and stays there. Reading a scaling plot in a paper is reading slopes on exactly this kind of chart.

### Little o and Õ

**Little-o** is a strict version: $f(n) = o(g(n))$ means $f$ becomes negligible next to $g$,

$$
\lim_{n \to \infty} \frac{f(n)}{g(n)} = 0.
$$

So $n = o(n^2)$ because $n/n^2 = 1/n \to 0$, but $n^2 \ne o(n^2)$ because the ratio stays at 1. The most common ML use is the regret statement $R(T) = o(T)$, which says average regret $R(T)/T$ goes to zero. That is the definition of a "no-regret" algorithm, and it connects little-o straight back to limits.

**Soft-O**, written $\tilde{O}$, is big-O that also hides logarithmic factors: $f = \tilde{O}(g)$ means $f = O(g \cdot \log^k g)$ for some fixed $k$. Bandit and RL papers love it: a regret of $\sqrt{T}\log T$ is reported as $\tilde{O}(\sqrt{T})$. Unpack it when the hidden factor is $\log(1/\delta)$ or the dimension.

| Symbol | Read it as | Picture on log-log axes | Example |
| --- | --- | --- | --- |
| $f = O(g)$ | at most, up to a constant | below some shifted copy of $g$ | $3n^2 + 100n = O(n^2)$ |
| $f = \Omega(g)$ | at least, up to a constant | above some shifted copy of $g$ | $n^2 = \Omega(n)$ |
| $f = \Theta(g)$ | same growth class | in a corridor around $g$ | $3n^2 + 100n = \Theta(n^2)$ |
| $f = o(g)$ | negligible next to $g$ | slope strictly smaller | $\sqrt{T} = o(T)$ |
| $f = \tilde{O}(g)$ | $O$ up to log factors | slope of $g$, slight bend | $\sqrt{T}\log T = \tilde{O}(\sqrt{T})$ |

### Two directions: n grows, or ε shrinks

The same notation is used in two opposite directions, and papers rarely say which. For **cost**, the variable grows: attention costs $O(n^2 d)$ in sequence length $n$ and width $d$, and bigger is worse. For **error**, the variable is usually the number of steps $T$ or samples $n$, and the bounded quantity shrinks: $O(1/\sqrt{T})$ says the error is at most a constant times $1/\sqrt{T}$, which goes to zero. When a statement confuses you, first ask which variable is going where.

## Asymptotic versus non-asymptotic statements

An **asymptotic statement** describes behaviour in the limit and hides constants: "the error is $O(1/T)$". A **non-asymptotic statement** holds at every finite size with explicit constants: "for all $T \ge 1$, the error is at most ${1000}/T$". The first names the class; the second says what you can promise at ten thousand steps.

Constants can be enormous. Compare a method with error ${1000}/T$ to one with error $1/\sqrt{T}$. The first is in the better class, $O(1/T)$ against $O(1/\sqrt{T})$. Here is what each delivers:

| Steps $T$ | Method A: ${1000}/T$ | Method B: $1/\sqrt{T}$ | Better |
| --- | --- | --- | --- |
| 10 | 100 | 0.316 | B |
| 100 | 10 | 0.1 | B |
| 10,000 | 0.1 | 0.01 | B |
| 1,000,000 | 0.001 | 0.001 | tie |

The two curves cross where $1000/T = 1/\sqrt{T}$, that is $\sqrt{T} = 1000$, so ${T} = {10}^{6}$. Before a million steps, the asymptotically worse method wins every time.

So look for the quantifier. "For all $T \ge 1$" signals a non-asymptotic bound with inspectable constants; "as $T \to \infty$" or a bare $O(\cdot)$ signals an asymptotic one. Adam's Theorem 4.1, which we decode below, is non-asymptotic; its Corollary 4.2 throws the constants away to state the class.

## "With high probability"

Training is random: minibatches, initialization, sampled data. A bound that holds **with high probability** is a promise that can fail, but only on a small, stated fraction of runs.

The analogy is an insurance policy with a published failure rate: across all policyholders, the bad outcome happens at most a $\delta$ fraction of the time. Where it breaks: you never learn which runs were the bad ones.

The formal shape is: for every $\delta \in (0, 1)$, with probability at least $1 - \delta$, the error is at most $\varepsilon(n, \delta)$. More confidence costs a looser bound, usually through a factor $\sqrt{\log(1/\delta)}$. Some papers instead mean probability at least $1 - 1/n^c$; check the definitions section.

![Exact distribution of the fraction of heads in 1000 fair flips; tails beyond 0.457 and 0.543 shaded red holding 0.72 percent of the mass](/imgs/blogs/limits-big-o-and-asymptotics-in-ml-papers-5.webp)

The worked example uses Hoeffding's inequality (Hoeffding, 1963; for the finance-flavoured derivation see [concentration inequalities for quants](/blog/trading/math-for-quants/concentration-inequalities-sample-complexity-math-for-quants)). For $n$ independent coin flips with heads frequency $\hat{p}$ and true probability $p$,

$$
\Pr\big(|\hat{p} - p| \gt \varepsilon\big) \le 2 \exp(-2 n \varepsilon^2).
$$

The left side is the chance the empirical frequency misses by more than $\varepsilon$; the right side shrinks exponentially in the number of flips. Set the right side equal to $\delta$ and solve: $\varepsilon = \sqrt{\ln(2/\delta) / (2n)}$. With $n = 1000$ and $\delta = 0.05$: $\ln 40 \approx 3.689$, divided by 2000 gives about 0.001844, and the square root is about 0.0429.

So with probability at least 95%, the observed frequency is within 0.043 of the truth. In the chart, the red tails beyond 0.457 and 0.543 hold only 0.72% of the mass, well under the 5% allowed. That gap is normal: read a w.h.p. bound as "at most this often", not "about this often".

## Convergence rates: what each extra digit costs

A **convergence rate** says how fast the error falls with the iteration count. Its meaning shows up when you invert it: how many iterations to reach error $\varepsilon$?

Think of buying digits of accuracy. Some methods charge a fixed number of steps per digit; others charge ten or a hundred times more for each new digit than for the last.

![Grouped bars on a log scale: iterations to reach error 0.1, 0.01 and 0.001 for rates one over root T, one over T and 0.9 to the T](/imgs/blogs/limits-big-o-and-asymptotics-in-ml-papers-4.webp)

Here are the three rates you will meet constantly, inverted by hand:

- **Sublinear, $O(1/\sqrt{T})$.** Error $1/\sqrt{T} \le \varepsilon$ needs $T \ge 1/\varepsilon^2$. For $\varepsilon$ equal to 0.1, 0.01 and 0.001 that is 100, 10,000 and 1,000,000 steps. Each digit costs 100 times the last. This is the rate of subgradient and stochastic gradient methods on convex Lipschitz problems (Bubeck, [Convex Optimization: Algorithms and Complexity](https://arxiv.org/abs/1405.4980), 2015, Theorem 3.2 and Chapter 6), and the rate in Adam's corollary.
- **Sublinear, $O(1/T)$.** $T \ge 1/\varepsilon$: 10, 100, 1,000 steps. Each digit costs 10 times the last. Gradient descent on smooth convex functions lives here.
- **Linear, $O(\rho^T)$ with $\rho \lt 1$.** Error $\rho^T \le \varepsilon$ needs $T \ge \ln(1/\varepsilon) / \ln(1/\rho)$. With $\rho = 0.9$, $\ln(1/0.9) \approx 0.1054$, so $\varepsilon = 0.1$ needs $2.303 / 0.1054 \approx 21.9$, rounded up to 22 steps; then 44 and 66. Each digit costs the same 22 steps.

The name "linear" confuses everyone once: the error falls along a straight line on a log scale, because $\log(\rho^T) = T \log \rho$ is linear in $T$. The animation below draws all three for fifty iterations.

<figure class="blog-anim">
<svg viewBox="0 0 760 400" role="img" aria-label="Three error curves drawn over 50 iterations on a log-scale error axis: 0.9 to the t falls as a straight line to about 0.005, 1 over t bends to 0.02, 1 over square root of t bends to 0.14" style="width:100%;height:auto;max-width:820px">
<style>
.a6-grid{stroke:var(--border,#d1d5db);stroke-width:1;stroke-dasharray:4 4}
.a6-axis{stroke:var(--text-secondary,#6b7280);stroke-width:1.5}
.a6-ax{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.a6-lbl{font:600 14px ui-sans-serif,system-ui}
.a6-c{fill:none;stroke-width:3;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:1;stroke-dashoffset:1}
.a6-sq{stroke:#f08c00} .a6-in{stroke:var(--text-primary,#1f2937)} .a6-li{stroke:var(--accent,#6366f1)}
.a6-t-sq{fill:#f08c00} .a6-t-in{fill:var(--text-primary,#1f2937)} .a6-t-li{fill:var(--accent,#6366f1)}
@keyframes a6-draw{0%{stroke-dashoffset:1}70%,100%{stroke-dashoffset:0}}
@keyframes a6-show{0%,65%{opacity:0}75%,100%{opacity:1}}
.a6-anim{animation:a6-draw 9s ease-in-out infinite}
.a6-fade{animation:a6-show 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.a6-anim{animation:none;stroke-dashoffset:0}.a6-fade{animation:none;opacity:1}}
</style>
<line class="a6-grid" x1="80" y1="40" x2="620" y2="40"/>
<line class="a6-grid" x1="80" y1="140" x2="620" y2="140"/>
<line class="a6-grid" x1="80" y1="240" x2="620" y2="240"/>
<line class="a6-grid" x1="80" y1="340" x2="620" y2="340"/>
<line class="a6-axis" x1="80" y1="30" x2="80" y2="340"/>
<line class="a6-axis" x1="80" y1="340" x2="620" y2="340"/>
<text class="a6-ax" x="70" y="45" text-anchor="end">1</text>
<text class="a6-ax" x="70" y="145" text-anchor="end">0.1</text>
<text class="a6-ax" x="70" y="245" text-anchor="end">0.01</text>
<text class="a6-ax" x="70" y="345" text-anchor="end">0.001</text>
<text class="a6-ax" x="80" y="364" text-anchor="middle">0</text>
<text class="a6-ax" x="188" y="364" text-anchor="middle">10</text>
<text class="a6-ax" x="296" y="364" text-anchor="middle">20</text>
<text class="a6-ax" x="404" y="364" text-anchor="middle">30</text>
<text class="a6-ax" x="512" y="364" text-anchor="middle">40</text>
<text class="a6-ax" x="620" y="364" text-anchor="middle">50</text>
<text class="a6-ax" x="350" y="388" text-anchor="middle">iteration t</text>
<text class="a6-ax" x="20" y="24">error (log scale)</text>
<polyline class="a6-c a6-sq a6-anim" pathLength="1" points="80.0,40.0 90.8,55.1 101.6,63.9 112.4,70.1 123.2,74.9 134.0,78.9 144.8,82.3 155.6,85.2 166.4,87.7 177.2,90.0 188.0,92.1 198.8,94.0 209.6,95.7 220.4,97.3 231.2,98.8 242.0,100.2 252.8,101.5 263.6,102.8 274.4,103.9 285.2,105.1 296.0,106.1 306.8,107.1 317.6,108.1 328.4,109.0 339.2,109.9 350.0,110.7 360.8,111.6 371.6,112.4 382.4,113.1 393.2,113.9 404.0,114.6 414.8,115.3 425.6,115.9 436.4,116.6 447.2,117.2 458.0,117.8 468.8,118.4 479.6,119.0 490.4,119.6 501.2,120.1 512.0,120.6 522.8,121.2 533.6,121.7 544.4,122.2 555.2,122.7 566.0,123.1 576.8,123.6 587.6,124.1 598.4,124.5 609.2,124.9 620.0,125.4"/>
<polyline class="a6-c a6-in a6-anim" pathLength="1" points="80.0,40.0 90.8,70.1 101.6,87.7 112.4,100.2 123.2,109.9 134.0,117.8 144.8,124.5 155.6,130.3 166.4,135.4 177.2,140.0 188.0,144.1 198.8,147.9 209.6,151.4 220.4,154.6 231.2,157.6 242.0,160.4 252.8,163.0 263.6,165.5 274.4,167.9 285.2,170.1 296.0,172.2 306.8,174.2 317.6,176.2 328.4,178.0 339.2,179.8 350.0,181.5 360.8,183.1 371.6,184.7 382.4,186.2 393.2,187.7 404.0,189.1 414.8,190.5 425.6,191.9 436.4,193.1 447.2,194.4 458.0,195.6 468.8,196.8 479.6,198.0 490.4,199.1 501.2,200.2 512.0,201.3 522.8,202.3 533.6,203.3 544.4,204.3 555.2,205.3 566.0,206.3 576.8,207.2 587.6,208.1 598.4,209.0 609.2,209.9 620.0,210.8"/>
<polyline class="a6-c a6-li a6-anim" pathLength="1" points="80.0,40.0 90.8,44.6 101.6,49.2 112.4,53.7 123.2,58.3 134.0,62.9 144.8,67.5 155.6,72.0 166.4,76.6 177.2,81.2 188.0,85.8 198.8,90.3 209.6,94.9 220.4,99.5 231.2,104.1 242.0,108.6 252.8,113.2 263.6,117.8 274.4,122.4 285.2,126.9 296.0,131.5 306.8,136.1 317.6,140.7 328.4,145.2 339.2,149.8 350.0,154.4 360.8,159.0 371.6,163.5 382.4,168.1 393.2,172.7 404.0,177.3 414.8,181.8 425.6,186.4 436.4,191.0 447.2,195.6 458.0,200.2 468.8,204.7 479.6,209.3 490.4,213.9 501.2,218.5 512.0,223.0 522.8,227.6 533.6,232.2 544.4,236.8 555.2,241.3 566.0,245.9 576.8,250.5 587.6,255.1 598.4,259.6 609.2,264.2 620.0,268.8"/>
<text class="a6-lbl a6-t-sq a6-fade" x="628" y="130">1/√t: 0.14</text>
<text class="a6-lbl a6-t-in a6-fade" x="628" y="216">1/t: 0.020</text>
<text class="a6-lbl a6-t-li a6-fade" x="628" y="274">0.9ᵗ: 0.005</text>
</svg>
<figcaption>Fifty iterations, three rates, error on a log scale. The linear rate 0.9ᵗ is the only straight line: it gains a fixed number of digits per step, while 1/t and 1/√t keep bending flatter.</figcaption>
</figure>

Where does $\rho = 0.9$ come from in practice? Run gradient descent with step ${1}/{L}$ on the quadratic $f(x, y) = \tfrac{1}{2}(x^2 + 10y^2)$. The step is 0.1, the $y$-coordinate vanishes in one step, and the $x$-coordinate is multiplied by ${1} - {0.1} = {0.9}$ every step. In general the factor is $1 - \mu/L = 1 - 1/\kappa$ for a condition number $\kappa$, which is why badly conditioned problems need about $\kappa \ln(1/\varepsilon)$ steps.

## Paper sentence, decoded

**Paper:** Diederik P. Kingma and Jimmy Ba, "Adam: A Method for Stochastic Optimization", 2014 (ICLR 2015), [arXiv:1412.6980](https://arxiv.org/abs/1412.6980), Section 4, Equation 5, Theorem 4.1 and Corollary 4.2.

Section 4 uses the online convex learning setup: at each round $t$ the algorithm picks $\theta_t$, then a convex loss $f_t$ is revealed. Performance is measured by the regret (Eq. 5),

$$
R(T) = \sum_{t=1}^{T} \big[ f_t(\theta_t) - f_t(\theta^*) \big], \qquad \theta^* = \arg\min_{\theta \in \mathcal{X}} \sum_{t=1}^{T} f_t(\theta).
$$

Regret is the total extra loss you paid compared with the single best fixed parameter chosen in hindsight. Theorem 4.1 then states that, under its assumptions, for all $T \ge 1$:

$$
R(T) \le \frac{D^2}{2\alpha(1-\beta_1)} \sum_{i=1}^{d} \sqrt{T\, \hat{v}_{T,i}} + \frac{\alpha(1+\beta_1) G_\infty}{(1-\beta_1)\sqrt{1-\beta_2}\,(1-\gamma)^2} \sum_{i=1}^{d} \lVert g_{1:T,i} \rVert_2 + \sum_{i=1}^{d} \frac{D_\infty^2 G_\infty \sqrt{1-\beta_2}}{2\alpha(1-\beta_1)(1-\lambda)^2}.
$$

The decode:

| Symbol | Meaning | Status in the bound |
| --- | --- | --- |
| $R(T)$, $T$ | regret after $T$ rounds | the quantity being bounded |
| $\alpha$ | step size, schedule $\alpha_t = \alpha/\sqrt{t}$ | source of the $\sqrt{T}$ (proof below) |
| $\beta_1, \beta_2, \lambda, \gamma$ | moment decay rates; $\beta_{1,t} = \beta_1 \lambda^{t-1}$, $\gamma = \beta_1^2/\sqrt{\beta_2} \lt 1$ | constants |
| $d$; $G, G_\infty$; $D, D_\infty$ | dimension; gradient-norm bounds; iterate-distance bounds | constants |
| $\hat{v}_{T,i}$; $g_{1:T,i}$ | bias-corrected second moment; all coordinate-$i$ gradients up to $T$ | depend on $T$ |

![The Theorem 4.1 bound split into three boxed terms: two grow like the square root of T, the third has no T and is constant](/imgs/blogs/limits-big-o-and-asymptotics-in-ml-papers-6.webp)

Read it term by term, asking only how each grows with $T$. In the first term, $\hat{v}_{T,i}$ is a weighted average of squared gradient coordinates, each at most $G_\infty^2$, so $\sqrt{T \hat{v}_{T,i}} \le G_\infty \sqrt{T}$ and the sum is at most $d G_\infty \sqrt{T}$. In the second term, $\lVert g_{1:T,i} \rVert_2 = \sqrt{\sum_t g_{t,i}^2} \le \sqrt{T G_\infty^2} = G_\infty\sqrt{T}$, so again at most $d G_\infty \sqrt{T}$. The third term has no $T$ in it at all.

So $R(T) \le C_1 \sqrt{T} + C_2$ for constants that depend on everything except $T$, which is $R(T) = O(\sqrt{T})$. Divide by $T$ and you get Corollary 4.2, $R(T)/T = O(1/\sqrt{T})$, and since $1/\sqrt{T} \to 0$ the average regret has limit zero. In words: Adam's average excess loss per round shrinks like one over the square root of the number of rounds.

### What the 2018 paper found

Sashank Reddi, Satyen Kale and Sanjiv Kumar, "On the Convergence of Adam and Beyond" (ICLR 2018, [arXiv:1904.09237](https://arxiv.org/abs/1904.09237)), showed that the proof of Theorem 4.1 is incorrect. The analysis needs the quantity $\Gamma_{t+1} = \sqrt{V_{t+1}}/\alpha_{t+1} - \sqrt{V_t}/\alpha_t$ to be positive semi-definite. For SGD and Adagrad it is; for Adam the moving average can shrink $V_t$, and the original proof "erroneously assumes" otherwise.

They also gave a counterexample. On the interval $[-1, 1]$, take linear losses $f_t(x) = Cx$ when $t \bmod 3 = 1$ and $f_t(x) = -x$ otherwise, with $C \gt 2$. The best fixed point is $x = -1$. With $\beta_1 = 0$ and $\beta_2 = 1/(1 + C^2)$, Adam drifts to $x = +1$, so its regret grows linearly and $R(T)/T$ does not go to zero. Their fix, AMSGrad, keeps $\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$ so the effective step never grows.

The lesson: an $O(\sqrt{T})$ statement is exactly as trustworthy as its proof.

## Proof, read line by line

Behind every $O(\sqrt{T})$ regret bound with step size $\alpha/\sqrt{t}$ there is the same small fact: summing the step sizes gives about $2\sqrt{T}$, not $T$. Here it is with a complete proof.

**Statement.** For every integer $T \ge 1$,

$$
\sum_{t=1}^{T} \frac{1}{\sqrt{t}} \le 2\sqrt{T} - 1 \le 2\sqrt{T}.
$$

Assumptions:
- (A1) $T$ is a positive integer.
- (A2) The square root is increasing on $[0, \infty)$; equivalently, $1/\sqrt{x}$ is decreasing on $(0, \infty)$.

**The idea in one sentence.** Squeeze each term $1/\sqrt{t}$ under a difference of square roots, so the sum collapses to its two ends.

![Rectangles of height one over root t and width one sit under the curve one over root x; eight rectangles sum to 4.371 while the area under the curve is 5.657](/imgs/blogs/limits-big-o-and-asymptotics-in-ml-papers-7.webp)

The figure is the proof as a picture: each term is a width-one rectangle that fits under the curve, so the sum is at most the area, $\int_0^T x^{-1/2}\,dx = 2\sqrt{T}$. For $T = 8$ the rectangles total 4.371 against an area of $2\sqrt{8} \approx 5.657$. The algebra below makes the same argument without calculus.

**Steps.**

1. For $t \ge 1$: $\sqrt{t} - \sqrt{t-1} = \dfrac{t - (t-1)}{\sqrt{t} + \sqrt{t-1}} = \dfrac{1}{\sqrt{t} + \sqrt{t-1}}$. Multiply and divide by the conjugate $\sqrt{t} + \sqrt{t-1}$.
2. $\sqrt{t-1} \le \sqrt{t}$. By (A2).
3. So $\sqrt{t} + \sqrt{t-1} \le 2\sqrt{t}$. Add $\sqrt{t}$ to both sides of step 2.
4. Hence $\dfrac{1}{\sqrt{t} + \sqrt{t-1}} \ge \dfrac{1}{2\sqrt{t}}$. Taking reciprocals of positive numbers reverses the inequality.
5. Combine 1 and 4: $\dfrac{1}{\sqrt{t}} \le 2\big(\sqrt{t} - \sqrt{t-1}\big)$. Multiply both sides by 2.
6. Sum from $t = 2$ to $T$: $\sum_{t=2}^{T} \dfrac{1}{\sqrt{t}} \le 2\sum_{t=2}^{T} \big(\sqrt{t} - \sqrt{t-1}\big)$. Adding inequalities term by term preserves them.
7. The right side telescopes: every $\sqrt{k}$ for $2 \le k \le T-1$ appears once with a plus and once with a minus, leaving $2(\sqrt{T} - 1)$.
8. Add the $t = 1$ term, which equals 1: the total is at most $1 + 2\sqrt{T} - 2 = 2\sqrt{T} - 1 \le 2\sqrt{T}$. This uses (A1); for $T = 1$ the sum in step 6 is empty.

Check with $T = 4$: ${1} + {0.7071} + {0.5774} + {0.5} = {2.7845}$, and $2\sqrt{4} - 1 = 3$. The same conjugate trick in the other direction gives $1/\sqrt{t} \ge 2(\sqrt{t+1} - \sqrt{t})$, so the sum is at least $2\sqrt{T+1} - 2$, which is 2.472 for $T = 4$. Upper and lower bounds together say the sum is $\Theta(\sqrt{T})$: the $\sqrt{T}$ is not an artifact of a loose proof.

**Where each assumption was used.**

| Assumption | Step |
| --- | --- |
| (A1) $T$ a positive integer | 6, 8 |
| (A2) square root increasing | 2, and therefore 3 to 5 |

**What breaks without (A2).** The rectangle picture needs the function to decrease, so each rectangle on $[t-1, t]$ sits under the curve. Try the same move on the increasing function $\sqrt{x}$: the rectangle of height $\sqrt{2} \approx 1.414$ on $[1, 2]$ has more area than the curve beneath it, $\int_1^2 \sqrt{x}\,dx = \tfrac{2}{3}(2^{3/2} - 1) \approx 1.219$. For increasing summands the integral becomes a lower bound, not an upper bound.

**How this becomes a regret bound.** In online gradient descent with step $\alpha_t = \alpha/\sqrt{t}$, the gradient terms add up to at most $\tfrac{\alpha G^2}{2}\sum_t 1/\sqrt{t} \le \alpha G^2 \sqrt{T}$ by this lemma, and the distance term contributes $D^2\sqrt{T}/(2\alpha)$. Both are $O(\sqrt{T})$ (simplified from Bubeck 2015, Chapter 3).

**Proof moves used:** `telescoping`

## Check it numerically

The snippet checks three claims from this post: the lemma and its tightness, the w.h.p. bound by simulation, and the iteration counts behind the rates chart.

```python
import numpy as np
rng = np.random.default_rng(0)

# 1. The proof step: sum_{t<=T} 1/sqrt(t) <= 2 sqrt(T), and the ratio tends to 1
for T in [10, 1_000, 100_000, 10_000_000]:
    s = np.sum(1.0 / np.sqrt(np.arange(1, T + 1)))
    print(f"T={T:>10,}  sum={s:12.3f}  2*sqrt(T)={2*np.sqrt(T):12.3f}  ratio={s/(2*np.sqrt(T)):.4f}")

# 2. "With high probability": Hoeffding for n fair coin flips, delta = 0.05
n, delta, runs = 1000, 0.05, 200_000
eps = np.sqrt(np.log(2 / delta) / (2 * n))
p_hat = rng.binomial(n, 0.5, size=runs) / n
print(f"eps={eps:.4f}  fraction of runs with |p_hat-0.5|>eps: {np.mean(np.abs(p_hat-0.5) > eps):.4%}  (allowed: {delta:.0%})")

# 3. Rates: iterations until error <= 1e-3
target = 1e-3
rates = {"1/sqrt(T)": lambda T: 1/np.sqrt(T), "1/T": lambda T: 1/T, "0.9^T": lambda T: 0.9**T}
T = np.arange(1, 2_000_001)
for name, err in rates.items():
    print(f"{name:>10}: first T with error <= 1e-3 is {T[err(T) <= target][0]:,}")
```

```console
T=        10  sum=       5.021  2*sqrt(T)=       6.325  ratio=0.7939
T=     1,000  sum=      61.801  2*sqrt(T)=      63.246  ratio=0.9772
T=   100,000  sum=     630.997  2*sqrt(T)=     632.456  ratio=0.9977
T=10,000,000  sum=    6323.095  2*sqrt(T)=    6324.555  ratio=0.9998
eps=0.0429  fraction of runs with |p_hat-0.5|>eps: 0.7280%  (allowed: 5%)
 1/sqrt(T): first T with error <= 1e-3 is 1,000,000
       1/T: first T with error <= 1e-3 is 1,000
     0.9^T: first T with error <= 1e-3 is 66
```

The sum always stays below $2\sqrt{T}$ and the ratio climbs to 1, which is the $\Theta(\sqrt{T})$ claim in numbers. The gap itself settles near a constant of about 1.46, so the relative error shrinks like $1/\sqrt{T}$. The simulated failure rate, 0.728%, matches the exact binomial tail of 0.72% within Monte Carlo noise and sits far below the 5% the bound permits, as the histogram predicted.

## Where this shows up in practice

**AMSGrad in PyTorch.** The fix from Reddi et al. ships as a flag: `torch.optim.Adam(params, amsgrad=True)` keeps the running maximum of the second-moment estimate ([PyTorch Adam docs](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)). It exists because an $O(\sqrt{T})$ regret statement failed its proof; the repair restores the monotone step size the analysis needs. Most large runs still use plain Adam or AdamW, weighing the theorem against measured loss curves.

**Attention cost and FlashAttention.** Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135), 2022), starts from the fact that "the time and memory complexity of self-attention are quadratic in sequence length". FlashAttention computes exact attention, so its arithmetic is still quadratic in $n$; what it reduces is the number of reads and writes to GPU high-bandwidth memory. Same growth class in FLOPs, a much better constant in the resource that actually bottlenecks the kernel.

**Compute-optimal scaling.** Hoffmann et al., "Training Compute-Optimal Large Language Models" ([arXiv:2203.15556](https://arxiv.org/abs/2203.15556), 2022), fits loss as an irreducible floor plus two power-law terms in parameters and tokens, and concludes that model size and training tokens "should be scaled equally". Reading that fit uses this post twice: power laws are log-log straight lines whose slopes are the exponents, and the floor is a limit. See the [Chinchilla post](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) for the empirical side.

## What to remember

> - A limit is a band test: every band around $L$ is eventually entered for good. It says where, not how fast.
> - $O$ is a ceiling, $\Omega$ a floor, $\Theta$ a sandwich, $o$ "negligible", $\tilde{O}$ "up to logs".
> - Check the quantifier: "for all $T \ge 1$" with explicit constants is non-asymptotic; a bare $O(\cdot)$ hides constants that can be a million steps wide.
> - Invert every rate: $O(1/\sqrt{T})$ costs 100 times more per digit, $O(1/T)$ ten times, a linear rate a fixed number of steps.
> - $\sum_{t \le T} 1/\sqrt{t} \le 2\sqrt{T}$ is the step behind every $O(\sqrt{T})$ regret bound, and a big-O claim is only as good as its proof.

## Where next

If symbols like $\arg\min$ or $\mathbb{E}$ in the regret definition slowed you down, start with [how to read the math in an ML paper](/blog/machine-learning/mathematics/how-to-read-the-math-in-an-ml-paper). The telescoping move used here is one of a dozen catalogued in [proof techniques for ML researchers](/blog/machine-learning/mathematics/proof-techniques-for-ml-researchers), and the sums themselves get their own treatment in [sums, products, einsum and log-sum-exp](/blog/machine-learning/mathematics/sums-products-einsum-and-log-sum-exp). Hoeffding and its relatives live in [the inequality toolkit every appendix uses](/blog/machine-learning/mathematics/the-inequality-toolkit-every-appendix-uses).

The [bandits post](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) shows regret in a setting you can run. Later, a post on epsilon-delta arguments makes the band picture formal, and the optimization track proves the rates this post only stated.
