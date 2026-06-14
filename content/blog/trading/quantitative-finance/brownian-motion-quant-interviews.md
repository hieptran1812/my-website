---
title: "Brownian motion for quant interviews: the atom of stochastic finance"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch, interview-focused deep dive on Brownian motion: its three defining properties, the moments and covariance min(s,t), the martingale and quadratic-variation facts that power Ito calculus, why paths are continuous but nowhere differentiable, the reflection principle and hitting times, a preview of geometric Brownian motion for prices, and a full set of fully-solved interview problems."
tags:
  [
    "brownian-motion",
    "wiener-process",
    "stochastic-calculus",
    "martingales",
    "quadratic-variation",
    "reflection-principle",
    "hitting-times",
    "geometric-brownian-motion",
    "quant-interviews",
    "ito-calculus",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Brownian motion is the single object that every other piece of quant finance is built on, and interviewers test whether you know its defining properties cold and can compute with them on the spot.
>
> - Brownian motion $B_t$ is pinned down by three rules: it starts at zero ($B_0 = 0$), its increments are independent and Gaussian ($B_t - B_s \sim \text{Normal}(0,\, t-s)$), and its path is continuous.
> - The moments you must recite: $\mathbb{E}[B_t] = 0$, $\operatorname{Var}(B_t) = t$, and the covariance $\operatorname{Cov}(B_s, B_t) = \min(s, t)$. The scaling law is $B_{ct}$ has the same distribution as $\sqrt{c}\, B_t$.
> - $B_t$ is a martingale (a "fair game"), and so is $B_t^2 - t$. The drift you subtract, $t$, is exactly the **quadratic variation** — the rule $(\mathrm{d}B)^2 = \mathrm{d}t$ that powers Ito calculus.
> - Paths are continuous everywhere but differentiable nowhere, which is why prices need Ito calculus instead of ordinary calculus.
> - The reflection principle turns hard "did the path ever touch \$110?" questions into easy ones, and gives clean hitting-time and running-maximum answers.
> - The one number to remember: **variance equals time**. The standard deviation of $B_t$ is $\sqrt{t}$, so uncertainty grows with the *square root* of the horizon, not linearly.

Here is a question that has ended more quant interviews than any options-pricing brainteaser: *"What is Brownian motion, and what are its defining properties?"* It sounds like a warm-up. It is actually a filter. The candidate who can recite the three axioms, derive the covariance in two lines, and explain why the path is nowhere differentiable has internalized the object that the entire discipline rests on. The candidate who waves their hands about "random motion" has not.

This post builds Brownian motion from absolutely nothing — from a child's coin-flip game — all the way to the facts a derivatives desk uses every day. We will define every term, compute with real numbers, and frame each idea the way a researcher interview at Jane Street, Two Sigma, Citadel, DE Shaw, or SIG actually frames it. By the end you will be able to answer the warm-up *and* the follow-ups that separate offers from rejections.

![Four-stage flow showing a coin-flip random walk becoming Brownian motion in a scaling limit and then geometric Brownian motion for prices](/imgs/blogs/brownian-motion-quant-interviews-1.png)

The diagram above is the mental model for the whole article. On the left, a fair coin-flip game: each second your wealth goes up \$1 or down \$1 with equal probability. Take that jagged walk, shrink the time step and the jump size together in just the right way, and in the limit you get Brownian motion — a continuous random curve with three crisp properties. Exponentiate a drifting version of it and you get the standard model of a stock price. Every arrow in that figure is a section of this post.

## The building blocks: from a coin-flip game to a continuous curve

Before any formula, let us agree on the everyday picture. A *stochastic process* is just a quantity that changes randomly over time — a number that has a value at every instant, but whose path you cannot predict in advance. A stock price is a stochastic process. So is the temperature, or the position of a speck of pollen jostled by water molecules (which is the physical phenomenon Brownian motion is named after, observed by the botanist Robert Brown in 1827).

### The simple random walk: the toy we start from

Start with the simplest random process there is. You play a game: every step, a fair coin is flipped. Heads, you gain \$1; tails, you lose \$1. Let $W_n$ be your total winnings after $n$ steps. Then

$$W_n = X_1 + X_2 + \cdots + X_n,$$

where each $X_i$ is $+1$ or $-1$ with probability one half each. This is the *simple symmetric random walk*. The word *symmetric* means the up-move and the down-move are equally likely; *simple* means each step is the same fixed size. We cover the discrete version of this object — gambler's ruin, expected game length, absorbing states — in depth in [Markov chains and hitting times](/blog/trading/quantitative-finance/markov-chains-hitting-times-quant-interviews). Here we care about what happens when we zoom out.

Two facts about $W_n$ fall out immediately, and they are the seeds of everything later. First, the *expected* winnings are zero: each step has mean zero ($\tfrac12(+1) + \tfrac12(-1) = 0$), and the expectation of a sum is the sum of the expectations, so $\mathbb{E}[W_n] = 0$. The game is *fair* — on average you neither win nor lose. Second, the *variance* of the winnings is $n$. Each step has variance $\mathbb{E}[X_i^2] - (\mathbb{E}[X_i])^2 = 1 - 0 = 1$, and because the flips are independent, variances add: $\operatorname{Var}(W_n) = n$. (Variance is the average squared distance from the mean — a measure of spread. Independence is what lets us add variances instead of worrying about cross-terms.)

So after $n$ steps your typical distance from zero is the standard deviation $\sqrt{\operatorname{Var}(W_n)} = \sqrt{n}$. Not $n$. The square root. After 100 flips you are typically about \$10 from where you started, not \$100. **This square-root growth is the single most important quantitative fact about Brownian motion, and it is already visible in the coin game.**

It is worth pausing on *why* the square root appears, because it is the crux of the whole subject and interviewers probe it. The reason is *cancellation*. Your winnings are a sum of $n$ random $\pm 1$ steps; roughly half push up and half push down, and they partly cancel. If all $n$ steps went the same way you would be $n$ dollars from the start, but that is astronomically unlikely. If they cancelled perfectly you would be exactly at zero. The *typical* leftover after imperfect cancellation is the standard deviation, $\sqrt{n}$ — geometrically, it is the same Pythagorean reason that adding $n$ independent unit vectors in random directions leaves you about $\sqrt{n}$ from the origin, not $n$. Keep this picture: **independent random pushes accumulate in quadrature (squares add), so distances grow like $\sqrt{n}$ while the count grows like $n$.** Everything downstream — variance equals $t$, volatility scales with $\sqrt{t}$, the running maximum grows like $\sqrt{t}$ — is this one idea wearing different clothes.

#### Worked example: how far from zero after 100 and after 400 flips?

You play the \$1 coin game for $n = 100$ steps. Your expected winnings are $\mathbb{E}[W_{100}] = 0$. Your variance is $\operatorname{Var}(W_{100}) = 100$, so your standard deviation is $\sqrt{100} = 10$. A rough rule of thumb from the bell curve: about two thirds of the time you finish within one standard deviation, so you typically end somewhere between $-\$10$ and $+\$10$.

Now play for $n = 400$ steps — four times as long. Naive intuition says four times as far. But the standard deviation is $\sqrt{400} = 20$, only *twice* as far, not four times. Quadrupling the time only doubles the spread. The single sentence to remember: **to double your typical distance from the start, you need four times as much time.** That is the square-root law, and it is the reason a one-year option is not twice as uncertain as a six-month option — it is only about $\sqrt{2} \approx 1.41$ times as uncertain.

### The scaling limit: shrinking the steps until the walk goes smooth

Brownian motion is what the random walk *becomes* when you refine the steps infinitely. Here is the recipe. Instead of one step per second, take $n$ steps in a fixed time interval of length $t$, so each step lasts $\mathrm{d}t = t/n$. To keep the variance from exploding or collapsing, scale each jump down to size $\sqrt{\mathrm{d}t}$ rather than $\pm 1$. Then over the interval the total variance is (number of steps) $\times$ (variance per step) $= n \times \mathrm{d}t = n \times (t/n) = t$. The variance stays equal to $t$ no matter how finely you chop time.

Now let $n \to \infty$. The jagged staircase smooths into a continuous (but still infinitely wiggly) curve. By the Central Limit Theorem — the deep fact that a sum of many small independent shocks is approximately Gaussian — the value at time $t$ is normally distributed with mean 0 and variance $t$. That limit object is **Brownian motion**, also called the **Wiener process** after the mathematician Norbert Wiener who built its rigorous foundation. We write it $B_t$ (or sometimes $W_t$).

![A coarse twenty-step walk overlaid on a fine four-hundred-step walk tracing the same shape at different resolutions](/imgs/blogs/brownian-motion-quant-interviews-2.png)

The figure above makes the limit concrete. The thick line is a coarse 20-step walk; the thin line is a 400-step walk that is the *same* underlying path sampled 20 times more finely. They trace the same large-scale shape; the fine path just fills in the detail. Push the refinement to infinity and you get a genuine continuous curve. The crucial scaling — each jump scaling as $\sqrt{\mathrm{d}t}$ — is what keeps the variance equal to $t$ across every resolution. Get that scaling wrong and the limit either freezes (jumps too small) or blows up (jumps too big).

### The three defining properties

Mathematicians do not *define* Brownian motion by the limit; they define it by three properties and then prove the limit produces them. You should be able to recite these three lines in your sleep.

![Three cards listing the defining properties of Brownian motion: starts at zero, independent Gaussian increments, and continuous paths](/imgs/blogs/brownian-motion-quant-interviews-4.png)

1. **It starts at zero.** $B_0 = 0$. The clock and the price both begin at a known point. (You can shift this; a Brownian motion *started at* \$100 is just $100 + B_t$.)

2. **Independent, stationary, Gaussian increments.** For any times $s < t$, the increment $B_t - B_s$ is normally distributed with mean 0 and variance $t - s$:
   $$B_t - B_s \sim \text{Normal}(0,\, t-s).$$
   Two pieces of vocabulary here. *Stationary* means the distribution of an increment depends only on the *length* of the time interval $t - s$, not on where it sits — the increment from $t=3$ to $t=5$ has the same law as the increment from $t=10$ to $t=12$. *Independent* means increments over non-overlapping intervals are statistically independent of one another: what the path does between $t=0$ and $t=1$ tells you nothing about what it does between $t=1$ and $t=2$.

3. **Continuous paths.** The function $t \mapsto B_t$ has no jumps; you can draw it without lifting your pen. (This is why a stock modeled by Brownian motion cannot gap — to model gaps you add jumps, a different process.)

These three lines pin the object down completely. Everything else — the variance, the covariance, the martingale property, the quadratic variation — is a *theorem*, not an extra assumption. Interviewers love this because they can ask you to *derive* the famous facts from the three axioms in real time, and a candidate who memorized formulas without understanding the axioms will stumble.

#### Worked example: the distribution of $B_3 - B_1$ and of $B_3$ alone

Take a standard Brownian motion. What is the law of the increment from $t=1$ to $t=3$? By property 2, $B_3 - B_1 \sim \text{Normal}(0,\, 3 - 1) = \text{Normal}(0, 2)$. Its standard deviation is $\sqrt{2} \approx 1.41$.

What about $B_3$ on its own? Write $B_3 = (B_3 - B_0)$ and use $B_0 = 0$: $B_3 \sim \text{Normal}(0, 3)$, standard deviation $\sqrt{3} \approx 1.73$. Notice $B_3$ is more spread out than the increment $B_3 - B_1$, because it accumulates randomness over a longer interval ($3$ versus $2$ units of time). The intuition: **the variance is the clock**. Two units of elapsed time means two units of variance, full stop.

## The moments: expectation, variance, and the covariance min(s, t)

A *moment* is a summary number of a distribution — the mean is the first moment, the variance is built from the second. For Brownian motion these are short to state and you must be able to produce them instantly.

The mean is zero at every time: $\mathbb{E}[B_t] = 0$, because $B_t = B_t - B_0$ is a mean-zero Gaussian. The variance is the elapsed time: $\operatorname{Var}(B_t) = t$. The standard deviation — the typical magnitude of $B_t$ — is therefore $\sqrt{t}$.

![Several Brownian sample paths fanning out from zero inside a square-root-of-t envelope](/imgs/blogs/brownian-motion-quant-interviews-3.png)

The figure above shows six independent Brownian paths fanning out from the origin. The dashed curves are the $\pm\sqrt{t}$ envelope (one standard deviation); the dotted curves are $\pm 2\sqrt{t}$ (two standard deviations). About 68% of paths stay inside the inner band at any given time, about 95% inside the outer band — the usual Gaussian percentages. The shape of the envelope is the headline: it widens like $\sqrt{t}$, a sideways parabola, not a straight cone. **Uncertainty grows with the square root of the horizon.** Annualized volatility works exactly this way: if a stock has 20% annual volatility, its one-month volatility is not $20\%/12$ but $20\% \times \sqrt{1/12} \approx 5.8\%$.

### The covariance is the overlap of the two clocks

Now the fact interviewers reach for constantly: how do the values at two different times, $B_s$ and $B_t$, co-vary? *Covariance* measures whether two random quantities move together; a positive covariance means that when one is above its mean the other tends to be too. For Brownian motion the answer is beautiful:

$$\operatorname{Cov}(B_s, B_t) = \min(s, t).$$

The covariance of the values at times $s$ and $t$ is simply the *smaller* of the two times. Here is the one-line derivation you should be ready to give. Assume $s \le t$. Split $B_t = B_s + (B_t - B_s)$. Then

$$\operatorname{Cov}(B_s, B_t) = \operatorname{Cov}(B_s,\, B_s + (B_t - B_s)) = \operatorname{Cov}(B_s, B_s) + \operatorname{Cov}(B_s,\, B_t - B_s).$$

The first term is $\operatorname{Var}(B_s) = s$. The second term is zero, because $B_s$ (the increment over $[0,s]$) and $B_t - B_s$ (the increment over $[s,t]$) sit on *non-overlapping* intervals and are therefore independent — and independent things have zero covariance. So $\operatorname{Cov}(B_s, B_t) = s = \min(s, t)$. The deep idea: **only the shared, overlapping time carries common randomness.** Up to time $s$, the two values $B_s$ and $B_t$ ride the same noise; after $s$, the extra noise in $B_t$ is fresh and independent.

![A six-by-six grid of covariance values colored from red for small overlap to green for large overlap showing the min of s and t pattern](/imgs/blogs/brownian-motion-quant-interviews-5.png)

The grid above tabulates $\operatorname{Cov}(B_s, B_t) = \min(s, t)$ for integer times 1 through 6. Read off any cell: row $t=5$, column $s=2$ gives $\min(2,5) = 2$. The diagonal, where $s = t$, gives $\min(t,t) = t = \operatorname{Var}(B_t)$ — the covariance of a value with itself is just its variance. The colors ramp from red (small overlap, small covariance) to green (large overlap), so the staircase structure of $\min$ is visible at a glance.

#### Worked example: compute Cov(B₂, B₅) and Var(B₅ − B₂)

Two quick computations that interviewers chain together. First the covariance: $\operatorname{Cov}(B_2, B_5) = \min(2, 5) = 2$. Done.

Now the variance of the increment $B_5 - B_2$. The fast way uses property 2 directly: $B_5 - B_2 \sim \text{Normal}(0, 5 - 2)$, so $\operatorname{Var}(B_5 - B_2) = 3$. But suppose the interviewer wants you to do it the *long* way, treating $B_5$ and $B_2$ as correlated variables, to test whether you really understand the covariance. Use the identity $\operatorname{Var}(X - Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) - 2\operatorname{Cov}(X, Y)$:

$$\operatorname{Var}(B_5 - B_2) = \operatorname{Var}(B_5) + \operatorname{Var}(B_2) - 2\operatorname{Cov}(B_5, B_2) = 5 + 2 - 2(2) = 3.$$

Both routes give 3. The lesson: the covariance $\min(s,t)$ is exactly the cross-term that makes the long computation collapse back to the short one. If you get a different answer one way than the other, you have misremembered either the variance ($= t$) or the covariance ($= \min(s,t)$). Pitfalls in correlation and covariance arithmetic show up constantly on the desk; we collect a batch of them in [covariance and correlation pitfalls](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews).

### The scaling property: B(ct) behaves like √c times B(t)

One more structural fact, the *self-similarity* or scaling property. If you speed up the clock by a factor $c$, you can absorb the change by rescaling the height by $\sqrt{c}$:

$$B_{ct} \stackrel{d}{=} \sqrt{c}\, B_t,$$

where $\stackrel{d}{=}$ means "has the same distribution as." Check it through the variance: the left side has variance $ct$; the right side has variance $(\sqrt{c})^2 \operatorname{Var}(B_t) = c \cdot t$. They match. This is the precise sense in which Brownian motion looks the same at every zoom level — a fact we will see vividly when we look at non-differentiability. It is also the formal statement behind "volatility scales with the square root of time."

#### Worked example: a 4× longer horizon and the √c rescaling

A risk model quotes the daily move of a portfolio as $B_1$ with $\operatorname{Var}(B_1) = 1$ (in units of \$10,000, say). A colleague asks for the *four-day* move, $B_4$. By scaling, $B_4 \stackrel{d}{=} \sqrt{4}\, B_1 = 2 B_1$. So the four-day move has the same distribution as *twice* the one-day move: standard deviation \$20,000, not \$40,000. If the desk wants a 95%-confidence band (roughly $\pm 2$ standard deviations), the one-day band is about $\pm\$20{,}000$ and the four-day band is about $\pm\$40{,}000$ — wider, but only by the factor $\sqrt{4} = 2$, not by 4. The one sentence: **scaling time by $c$ scales the spread by $\sqrt{c}$.**

## Brownian motion is a martingale, and so is B(t)² − t

A *martingale* is the mathematical formalization of a *fair game*: a process whose expected future value, given everything you know today, equals its current value. Formally, $M_t$ is a martingale if $\mathbb{E}[M_t \mid \mathcal{F}_s] = M_s$ for $s < t$, where $\mathcal{F}_s$ is the information available up to time $s$ (the "filtration" — read it as "everything observed so far"). In plain money terms: if your wealth is a martingale, then no matter what has happened, your best guess for tomorrow's wealth is today's wealth. You expect to neither gain nor lose.

Brownian motion is the prototypical martingale. To see it, take $s < t$ and condition on the path up to time $s$:

$$\mathbb{E}[B_t \mid \mathcal{F}_s] = \mathbb{E}[B_s + (B_t - B_s) \mid \mathcal{F}_s] = B_s + \mathbb{E}[B_t - B_s \mid \mathcal{F}_s] = B_s + 0 = B_s.$$

The increment $B_t - B_s$ is independent of the past and has mean zero, so its conditional expectation is just its unconditional mean, zero. The current value $B_s$ is known given the past, so it passes through. Result: $\mathbb{E}[B_t \mid \mathcal{F}_s] = B_s$. Brownian motion is a fair game. This is why the *driftless* model of a price has no predictable trend — the best forecast of tomorrow's price is today's price, which is the efficient-market intuition in one equation.

### The famous follow-up: is B(t)² a martingale?

Here is the question that catches people. Is $B_t^2$ a martingale? Intuition says "it is a function of a martingale, so maybe?" — and intuition is wrong. Compute $\mathbb{E}[B_t^2 \mid \mathcal{F}_s]$ for $s < t$. Write $B_t = B_s + (B_t - B_s)$ and expand the square:

$$\mathbb{E}[B_t^2 \mid \mathcal{F}_s] = \mathbb{E}[B_s^2 + 2 B_s (B_t - B_s) + (B_t - B_s)^2 \mid \mathcal{F}_s].$$

Take the three terms. The first, $B_s^2$, is known given the past, so it stays. The middle term: $B_s$ is known and pulls out, leaving $2 B_s \, \mathbb{E}[B_t - B_s \mid \mathcal{F}_s] = 2 B_s \cdot 0 = 0$. The third term: $(B_t - B_s)^2$ is independent of the past, and its expectation is the variance of the increment, which is $t - s$. So

$$\mathbb{E}[B_t^2 \mid \mathcal{F}_s] = B_s^2 + (t - s).$$

This is *not* equal to $B_s^2$ — it is bigger by $t - s$. So $B_t^2$ is **not** a martingale; it has an upward drift of exactly $t - s$ over the interval. But look at what the algebra is begging you to do: move the $t$ to the left. Define $M_t = B_t^2 - t$. Then

$$\mathbb{E}[B_t^2 - t \mid \mathcal{F}_s] = B_s^2 + (t - s) - t = B_s^2 - s.$$

That *is* $M_s$. So **$B_t^2 - t$ is a martingale.** Subtracting off the deterministic drift $t$ turns the up-trending $B_t^2$ into a fair game.

![Three curves over time: B squared drifting up, its straight-line mean equal to t, and B squared minus t with its mean pinned at zero](/imgs/blogs/brownian-motion-quant-interviews-8.png)

The figure above shows one realization. The solid line is $B_t^2$, which wanders upward — its mean is $\mathbb{E}[B_t^2] = \operatorname{Var}(B_t) = t$, the dotted straight ramp. The dashed line is $B_t^2 - t$, and its mean is pinned flat at zero: a fair game. The vertical gap between the solid and dashed lines is exactly $t$, growing linearly. This $B_t^2 - t$ martingale is not a curiosity — it is the seed of the quadratic-variation fact and of Ito's formula, and it is one of the most-asked items in a stochastic-calculus interview. If a process built from Brownian motion is asked about, the move is always the same: compute the conditional expectation, find the drift, subtract it. The martingale machinery generalizes the discrete fair-game and stopping-time arguments we use in [the Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews).

#### Worked example: which of B(t), B(t)² , B(t)² − t , and B(t)³ are martingales?

Run the test on four candidates. **$B_t$**: martingale, shown above ($\mathbb{E}[B_t \mid \mathcal{F}_s] = B_s$). **$B_t^2$**: not a martingale, drift $t - s > 0$. **$B_t^2 - t$**: martingale, the drift cancels. **$B_t^3$**: compute $\mathbb{E}[B_t^3 \mid \mathcal{F}_s]$ by expanding $(B_s + I)^3$ where $I = B_t - B_s$ is the independent increment. The cross terms involve $\mathbb{E}[I] = 0$ and $\mathbb{E}[I^3] = 0$ (odd moments of a symmetric Gaussian vanish), but the term $3 B_s \, \mathbb{E}[I^2] = 3 B_s (t - s)$ survives. So $\mathbb{E}[B_t^3 \mid \mathcal{F}_s] = B_s^3 + 3 B_s (t - s) \ne B_s^3$. **$B_t^3$ is not a martingale.** (It *can* be fixed: $B_t^3 - 3 t B_t$ is a martingale — the same subtract-the-drift trick. We solve that one in full in the interview-room section.) The one-sentence takeaway: **the only thing that can break the martingale property is a leftover $\mathrm{d}t$ drift, and the way to find it is to compute the conditional expectation and read off what does not cancel.**

## Quadratic variation equals t: the (dB)² = dt rule that powers Ito

Now the single most useful computational fact in all of stochastic calculus, and the reason ordinary calculus fails for prices. Chop the interval $[0, t]$ into $n$ equal pieces and add up the *squared* increments:

$$Q_n = \sum_{i=1}^{n} \left( B_{t_i} - B_{t_{i-1}} \right)^2.$$

As $n \to \infty$, this sum does not blow up and does not vanish — it converges to $t$. We say the **quadratic variation** of Brownian motion over $[0, t]$ equals $t$. Why? Each squared increment $(B_{t_i} - B_{t_{i-1}})^2$ has expectation equal to the length of its sub-interval, $t/n$, so the sum has expectation $n \times (t/n) = t$. The remarkable part is that the *variance* of $Q_n$ shrinks to zero as $n \to \infty$, so $Q_n$ does not just average to $t$ — it converges to the constant $t$ with no randomness left. The squared increments are so numerous and so small that their fluctuations wash out.

![A Brownian path on the left and the running sum of its squared increments on the right climbing along the straight line y equals t](/imgs/blogs/brownian-motion-quant-interviews-6.png)

The figure above shows it. On the left, a single jagged Brownian path. On the right, the running sum of its squared increments, climbing steadily up the straight line $y = t$ — almost deterministically, despite the path itself being pure noise. This is the precise statement of the shorthand every quant uses:

$$(\mathrm{d}B_t)^2 = \mathrm{d}t.$$

The square of a Brownian increment behaves, when summed, like the time increment itself. This is the engine of **Ito's formula**, the chain rule for functions of Brownian motion. In ordinary calculus, if $y = f(x)$ then $\mathrm{d}y = f'(x)\,\mathrm{d}x$ and you stop. With Brownian motion you must keep the second-order term, because $(\mathrm{d}B)^2 = \mathrm{d}t$ is not negligible:

$$\mathrm{d}f(B_t) = f'(B_t)\, \mathrm{d}B_t + \tfrac{1}{2} f''(B_t)\, \mathrm{d}t.$$

That extra $\tfrac12 f''\,\mathrm{d}t$ term — the *Ito correction* — comes entirely from $(\mathrm{d}B)^2 = \mathrm{d}t$. Apply it to $f(x) = x^2$: $f' = 2x$, $f'' = 2$, so $\mathrm{d}(B_t^2) = 2 B_t\, \mathrm{d}B_t + \tfrac12 (2)\,\mathrm{d}t = 2 B_t\, \mathrm{d}B_t + \mathrm{d}t$. The $\mathrm{d}t$ term is exactly the drift we found by hand earlier, which is why $B_t^2 - t$ (drift removed) is a martingale. The two facts are the same fact. This connects directly to the machinery behind [the Black-Scholes equation](/blog/trading/quantitative-finance/black-scholes), where the Ito correction is what produces the famous $\tfrac12 \sigma^2$ term.

#### Worked example: the running sum of squared increments over a year

Suppose you sample a Brownian motion daily for one year — roughly 252 trading days, so 252 increments — and each daily increment has variance $1/252$ (since the whole year has variance 1). Add up the *squared* daily increments. Each squared increment averages $1/252$, and there are 252 of them, so the sum averages $252 \times (1/252) = 1$ — the full year's worth of "clock." Now compare with the *unsquared* total distance traveled: that sum behaves like (number of steps) $\times$ (typical step size) $\approx 252 \times \sqrt{1/252} = \sqrt{252} \approx 15.9$, and as you sample more finely it grows without bound. **The path's total up-and-down distance is infinite; the sum of its squared moves is finite and equals the elapsed time.** That contrast is the whole reason Brownian motion needs its own calculus.

## Why the path is continuous but nowhere differentiable

Here is the property that feels paradoxical and is a favorite "do you really understand this?" probe. Brownian motion is *continuous* — no jumps, you can draw it without lifting your pen. Yet at no single point does it have a well-defined slope: it is *nowhere differentiable*. There is no instant at which the path has a velocity.

The reason follows directly from the square-root scaling. The slope over a small interval $\Delta t$ is the *difference quotient*

$$\frac{\Delta B}{\Delta t} = \frac{B_{t + \Delta t} - B_t}{\Delta t}.$$

The numerator $\Delta B$ is a mean-zero Gaussian with standard deviation $\sqrt{\Delta t}$. So the typical size of the slope is

$$\frac{\sqrt{\Delta t}}{\Delta t} = \frac{1}{\sqrt{\Delta t}}.$$

As $\Delta t \to 0$, this *blows up* to infinity. The smaller the time window, the steeper the typical slope — there is no limiting value, so the derivative does not exist. The path is continuous because $\Delta B \to 0$ as $\Delta t \to 0$ (the numerator vanishes, no jumps). It is non-differentiable because $\Delta B / \Delta t \to \infty$ (the numerator vanishes *slower* than the denominator). Continuity needs $\Delta B \to 0$; differentiability needs $\Delta B / \Delta t$ to settle. Brownian motion clears the first bar and fails the second.

![Three panels zooming into the same Brownian path at increasing magnification, each panel still jagged](/imgs/blogs/brownian-motion-quant-interviews-7.png)

The figure above is the visual proof. The left panel is the full path over $[0, 1]$. The middle panel zooms in 10× on a tiny slice; the right panel zooms 100× further. At every magnification the path looks *equally* jagged — it never straightens into a line the way a smooth curve would under a microscope. This is the scaling property $B_{ct} \stackrel{d}{=} \sqrt{c}\, B_t$ made visible: zoom in on time and the wiggle persists, rescaled. The practical consequence is unavoidable: **$\mathrm{d}B_t$ is not a velocity times $\mathrm{d}t$; it is its own kind of increment, of size $\sqrt{\mathrm{d}t}$.** That is precisely why you cannot do ordinary calculus on price paths and must use Ito calculus instead. An interviewer who hears "the derivative blows up like one over root-delta-t" knows you understand the object; one who hears "it's just really wiggly" does not.

## The reflection principle, hitting times, and the running maximum

Now the most powerful *trick* in the elementary Brownian toolkit, and the source of a whole genre of interview questions: *"What is the chance the path ever reaches level $a$ before time $t$?"* The naive approach — integrate over all paths that touch $a$ — is a nightmare. The **reflection principle** makes it a one-liner.

Here is the idea. Fix a level $a > 0$. Look at the *running maximum* $M_t = \max_{0 \le u \le t} B_u$ — the highest the path has gotten by time $t$. We want $\mathbb{P}(M_t \ge a)$: the probability the path touched $a$ at some point. Consider any path that *does* reach $a$. Let $\tau$ be the first time it hits $a$ (the *hitting time*). After $\tau$, by the strong Markov property and symmetry, the continuation is just a fresh Brownian motion started at $a$ — and it is equally likely to go up as down. So for every path that reaches $a$ and ends *below* $a$, there is an equally likely *mirror-image* path (reflected across the level $a$ after time $\tau$) that ends *above* $a$.

![A Brownian path hitting a barrier and its mirror-image reflection after the first hitting time](/imgs/blogs/brownian-motion-quant-interviews-9.png)

The figure above shows the construction. The solid path rises, first touches the barrier $a = \$110$ at time $\tau$ (the dotted vertical), and continues. The dotted curve is its reflection: identical up to $\tau$, then mirrored across the barrier. Reflection is a *probability-preserving* pairing — the original and its mirror are equally likely. This bijection gives the reflection principle:

$$\mathbb{P}(M_t \ge a) = 2\, \mathbb{P}(B_t \ge a).$$

Read it carefully. The chance the path *ever* reached $a$ (the left side) is exactly *twice* the chance it *ends* at or above $a$ (the right side). The factor of 2 is the reflection: paths that touched $a$ but came back down get paired one-to-one with paths that ended above $a$. Since $B_t \sim \text{Normal}(0, t)$, the right side is a standard normal tail you can look up.

#### Worked example: the expected running maximum E[max over 0 to t]

A classic. What is the expected value of the running maximum $M_t = \max_{0 \le u \le t} B_u$? Use the reflection-principle fact that $M_t$ has the same distribution as $|B_t|$ — the absolute value of the endpoint. (That is a tidy corollary: $\mathbb{P}(M_t \ge a) = 2\mathbb{P}(B_t \ge a) = \mathbb{P}(|B_t| \ge a)$, so $M_t$ and $|B_t|$ share a distribution.) The expected absolute value of a $\text{Normal}(0, t)$ variable is a standard integral:

$$\mathbb{E}[\,|B_t|\,] = \sqrt{\frac{2t}{\pi}}.$$

So $\mathbb{E}[M_t] = \sqrt{2t/\pi}$. Plug in $t = 1$: $\mathbb{E}[M_1] = \sqrt{2/\pi} \approx 0.80$. The path's typical high-water mark over a unit of time is about $0.80$ standard deviations — comfortably less than one full standard deviation, which makes sense because the maximum is taken over a path that spends most of its time below its peak. The one sentence: **the expected high-water mark grows like $\sqrt{t}$, the same square-root law as everything else, with constant $\sqrt{2/\pi} \approx 0.80$.**

### Two barriers: which one does the price hit first?

The reflection principle handles one barrier. For *two* barriers — "does the price hit \$110 before it hits \$90?" — there is an even cleaner answer that needs no integral at all, just the martingale property. Model the price as a driftless Brownian motion started at \$100. Because $B_t$ is a martingale, the *probability* of hitting the upper barrier first is set entirely by *distances*, by a fair-game (no-free-lunch) argument identical to gambler's ruin.

![Three price paths starting at one hundred dollars between an up-barrier at one hundred ten dollars and a down-barrier at ninety dollars](/imgs/blogs/brownian-motion-quant-interviews-10.png)

The figure above sets the scene: a \$100 start, an up-barrier at \$110 (green zone) and a down-barrier at \$90 (red zone). Three sample price paths wander between them; the question is which barrier each touches first. The probability of reaching the upper barrier $U$ before the lower barrier $L$, starting from $x$, is

$$\mathbb{P}(\text{hit } U \text{ first}) = \frac{x - L}{U - L}.$$

It is purely the fraction of the distance you have already covered toward the top. This is the continuous version of the gambler's-ruin formula derived in [Markov chains and hitting times](/blog/trading/quantitative-finance/markov-chains-hitting-times-quant-interviews).

#### Worked example: does a \$100 stock touch \$110 before \$90?

A driftless stock sits at \$100. The up-barrier is \$110, the down-barrier is \$90. The probability it touches \$110 before \$90:

$$\mathbb{P}(\text{hit } \$110 \text{ first}) = \frac{100 - 90}{110 - 90} = \frac{10}{20} = \frac{1}{2}.$$

Exactly one half — because \$100 is exactly halfway between the barriers and the walk has no drift, so it is equally likely to be carried up or down first. Now move the start: if the stock were at \$104 instead, the probability of hitting \$110 first becomes $(104 - 90)/(110 - 90) = 14/20 = 0.70$. Being closer to the top barrier makes you more likely to hit it first, in exact proportion to distance. The intuition: **a fair game touches a barrier first with probability equal to the fraction of the gap you have already closed toward it.** The expected time to hit *either* barrier, by the way, is $\mathbb{E}[\tau] = (x - L)(U - x)$ — for the \$100 start, $(100-90)(110-100) = 10 \times 10 = 100$ in variance-time units. The martingale $B_t^2 - t$ is exactly what you use to derive that expected-time formula.

## A preview of geometric Brownian motion for prices

Brownian motion itself is a flawed model of a stock price for two reasons that an interviewer will press you on. First, it can go *negative* — $B_t$ is symmetric around zero, but a stock cannot be worth less than \$0. Second, it adds *absolute* dollar moves, whereas real prices move in *percentage* terms — a \$1 move means more to a \$10 stock than to a \$1,000 stock. The fix is **geometric Brownian motion (GBM)**, the model under Black-Scholes:

$$S_t = S_0 \exp\!\left( \left(\mu - \tfrac{1}{2}\sigma^2\right) t + \sigma B_t \right).$$

Read the pieces. $S_0$ is today's price. $\mu$ is the *drift* — the expected continuously-compounded growth rate. $\sigma$ is the *volatility* — how much the price wiggles, in annualized percentage terms. $B_t$ is our Brownian motion supplying the randomness. The exponential does two jobs at once: it keeps $S_t > 0$ always (an exponential is never negative), and it makes the *log-returns* — not the dollar moves — normally distributed, which matches how prices actually behave. The mysterious $-\tfrac12\sigma^2$ term is the Ito correction we met earlier; it is the adjustment that makes the *expected* price come out to the clean $\mathbb{E}[S_t] = S_0 e^{\mu t}$ rather than something larger.

![Five geometric Brownian motion price paths starting at one hundred dollars scattering around the dashed expected-value curve](/imgs/blogs/brownian-motion-quant-interviews-11.png)

The figure above shows five GBM paths from $S_0 = \$100$ with drift $\mu = 10\%$ and volatility $\sigma = 22\%$ per year. Every path stays positive; they scatter around the dashed mean curve $\mathbb{E}[S_t] = 100\, e^{0.10 t}$. The distribution of $S_t$ is *lognormal* — skewed, with a long right tail — because it is the exponential of a normal. This is the workhorse model behind option pricing; the full treatment lives in [the Black-Scholes deep dive](/blog/trading/quantitative-finance/black-scholes) and [derivatives pricing](/blog/trading/quantitative-finance/derivatives-pricing).

#### Worked example: the expected price and the median price diverge

You hold a stock at $S_0 = \$100$ with $\mu = 8\%$ drift and $\sigma = 30\%$ volatility, one-year horizon. The *expected* price is $\mathbb{E}[S_1] = 100\, e^{0.08} \approx \$108.33$. But the *median* price — the level the stock is equally likely to finish above or below — uses the drift *with* the Ito correction: $\text{median}(S_1) = 100\, e^{(\mu - \sigma^2/2) \cdot 1} = 100\, e^{0.08 - 0.045} = 100\, e^{0.035} \approx \$103.56$. The mean (\$108.33) sits well above the median (\$103.56). The gap is the lognormal skew: a handful of paths shoot far up and drag the *average* above the *typical* outcome. The lesson every options trader internalizes: **for a volatile asset, the expected price overstates the typical price, and the gap is driven by $\tfrac12\sigma^2$.** Ignore that correction and you will systematically misprice.

## In the interview room: seven fully-solved problems

These are the kinds of problems a quant researcher interview actually puts in front of you. Work each one cold before reading the solution. Notice how often the same three tools recur: the increment decomposition $B_t = B_s + (B_t - B_s)$, the martingale-and-optional-stopping argument, and the reflection principle. A good interviewer is not testing whether you have seen the exact question — they are testing whether you can reach for the right one of those three tools under time pressure and execute it cleanly. Say your reasoning out loud as you go; the *process* is being graded as much as the final number.

#### Worked example: problem 1 — is B(t)³ − 3tB(t) a martingale?

**Problem.** We showed $B_t^3$ is not a martingale. Show that $Y_t = B_t^3 - 3 t B_t$ *is* a martingale, and explain where the $3t$ comes from.

**Solution.** Compute $\mathbb{E}[B_t^3 \mid \mathcal{F}_s]$ with $B_t = B_s + I$, $I = B_t - B_s$ independent of the past, $\mathbb{E}[I] = 0$, $\mathbb{E}[I^2] = t - s$, $\mathbb{E}[I^3] = 0$:

$$\mathbb{E}[(B_s + I)^3 \mid \mathcal{F}_s] = B_s^3 + 3 B_s^2\,\mathbb{E}[I] + 3 B_s\,\mathbb{E}[I^2] + \mathbb{E}[I^3] = B_s^3 + 3 B_s (t - s).$$

Now the other term: $\mathbb{E}[3 t B_t \mid \mathcal{F}_s] = 3t\, \mathbb{E}[B_t \mid \mathcal{F}_s] = 3 t B_s$ (using that $B_t$ is a martingale). Subtract:

$$\mathbb{E}[B_t^3 - 3 t B_t \mid \mathcal{F}_s] = B_s^3 + 3 B_s (t - s) - 3 t B_s = B_s^3 - 3 s B_s = Y_s.$$

It checks. **$Y_t = B_t^3 - 3 t B_t$ is a martingale.** The $3t$ is precisely the drift that Ito's formula generates: applying $\mathrm{d}f = f'\,\mathrm{d}B + \tfrac12 f''\,\mathrm{d}t$ to $f(x) = x^3$ gives a $\tfrac12 (6 B_t)\,\mathrm{d}t = 3 B_t\,\mathrm{d}t$ drift, which integrates to $3 \int_0^t B_u\,\mathrm{d}u$ — and the clean fix $-3tB_t$ removes the corresponding deterministic part. The pattern (these are the *Hermite polynomials* of Brownian motion) is: $B_t$, $B_t^2 - t$, $B_t^3 - 3tB_t$, each is a martingale.

#### Worked example: problem 2 — probability a \$100 stock touches \$110 before \$90

**Problem.** A stock follows driftless Brownian motion from \$100. What is the probability it hits \$110 before \$90? And what if instead it starts at \$102?

**Solution.** Use the two-barrier formula $\mathbb{P}(\text{hit } U \text{ first}) = (x - L)/(U - L)$ with $U = 110$, $L = 90$. From $x = 100$: $(100 - 90)/(110 - 90) = 10/20 = 0.5$. From $x = 102$: $(102 - 90)/20 = 12/20 = 0.6$. Moving the start up \$2 — one tenth of the way across the \$20 gap — raises the up-hit probability by exactly $0.1$, from $0.5$ to $0.6$. **Driftless hitting probabilities are linear in the starting distance.** A common interviewer twist: "now add a drift." With drift the answer switches to an exponential (gambler's-ruin-with-bias) formula, and you should flag that the *linear* answer only holds in the driftless, fair-game case.

#### Worked example: problem 3 — the variance of an Ito integral

**Problem.** Compute the variance of the Wiener integral $\displaystyle X = \int_0^T t\, \mathrm{d}B_t$.

**Solution.** This is a *stochastic integral* — a weighted sum of Brownian increments, where at time $t$ the weight is $t$. The key tool is the **Ito isometry**, which says the variance of $\int_0^T f(t)\,\mathrm{d}B_t$ is $\int_0^T f(t)^2\,\mathrm{d}t$. The square-and-integrate rule is a direct consequence of $(\mathrm{d}B)^2 = \mathrm{d}t$: the increments are independent with variance $\mathrm{d}t$, so variances add up as $\int f^2\,\mathrm{d}t$. Here $f(t) = t$:

$$\operatorname{Var}(X) = \int_0^T t^2\, \mathrm{d}t = \frac{T^3}{3}.$$

The mean is zero (it is a sum of mean-zero increments), so $X \sim \text{Normal}(0,\, T^3/3)$. For $T = 3$: $\operatorname{Var}(X) = 27/3 = 9$, standard deviation $3$. **The Ito isometry turns the variance of a stochastic integral into an ordinary integral of the squared weight** — a computation you must be able to do without hesitation. A frequent follow-up: "what is $\operatorname{Cov}\!\big(\int_0^T \mathrm{d}B_t,\, \int_0^T t\,\mathrm{d}B_t\big)$?" Answer by the bilinear version of the isometry: $\int_0^T 1 \cdot t\, \mathrm{d}t = T^2/2$.

#### Worked example: problem 4 — expected running maximum and a barrier-touch probability

**Problem.** Over $[0, 4]$, what is the expected running maximum $\mathbb{E}[M_4]$ of a standard Brownian motion? And what is the probability the path ever reaches level $a = 3$ within that window?

**Solution.** For the expected maximum, use $\mathbb{E}[M_t] = \sqrt{2t/\pi}$ with $t = 4$: $\mathbb{E}[M_4] = \sqrt{8/\pi} \approx \sqrt{2.546} \approx 1.60$. For the barrier-touch probability, use the reflection principle $\mathbb{P}(M_4 \ge 3) = 2\,\mathbb{P}(B_4 \ge 3)$. Since $B_4 \sim \text{Normal}(0, 4)$, standardize: $\mathbb{P}(B_4 \ge 3) = \mathbb{P}(Z \ge 3/2) = \mathbb{P}(Z \ge 1.5)$ where $Z$ is standard normal. From the normal table, $\mathbb{P}(Z \ge 1.5) \approx 0.0668$. Double it: $\mathbb{P}(M_4 \ge 3) \approx 2 \times 0.0668 = 0.134$, about a 13% chance. **The reflection principle converts a path-property question into a single endpoint-tail lookup, with a factor of 2.** Note the expected maximum ($\approx 1.60$) is well below the level $a = 3$, consistent with only a 13% chance of ever touching $3$.

#### Worked example: problem 5 — correlation of B(2) and B(5)

**Problem.** Find the *correlation* (not covariance) between $B_2$ and $B_5$.

**Solution.** Correlation normalizes covariance by the two standard deviations: $\rho = \operatorname{Cov}(B_2, B_5) / (\sigma_{B_2} \sigma_{B_5})$. We know $\operatorname{Cov}(B_2, B_5) = \min(2, 5) = 2$, $\operatorname{Var}(B_2) = 2$, $\operatorname{Var}(B_5) = 5$. So

$$\rho = \frac{2}{\sqrt{2}\,\sqrt{5}} = \frac{2}{\sqrt{10}} \approx 0.632.$$

The general formula, for $s < t$: $\rho(B_s, B_t) = \min(s,t)/\sqrt{st} = s/\sqrt{st} = \sqrt{s/t}$. Here $\sqrt{2/5} = \sqrt{0.4} \approx 0.632$, matching. **The correlation between two Brownian values is $\sqrt{s/t}$ — it depends only on the *ratio* of the two times.** As $t$ pulls far ahead of $s$, the correlation decays toward zero (the future forgets the present); as $t \to s$, it climbs to 1. This $\sqrt{s/t}$ result is a favorite because it tests whether you keep covariance and correlation straight under pressure.

#### Worked example: problem 6 — first-passage time has infinite expectation

**Problem.** Let $\tau_a$ be the first time a standard Brownian motion reaches level $a > 0$. Is $\mathbb{E}[\tau_a]$ finite?

**Solution.** A surprising no. By the reflection principle the path reaches *any* finite level $a$ eventually with probability 1 — so $\tau_a < \infty$ almost surely. But its *expectation* is infinite: $\mathbb{E}[\tau_a] = +\infty$. The reason: the first-passage density has a heavy tail, $\mathbb{P}(\tau_a > t) \sim c/\sqrt{t}$ for large $t$, and $\int^\infty (1/\sqrt{t})\,\mathrm{d}t$ diverges. Intuitively, although the path *will* hit $a$, it occasionally takes enormously long excursions in the wrong direction first, and those rare-but-huge waiting times make the average blow up. **A driftless Brownian motion hits any level with certainty but takes, on average, forever.** This is the continuous cousin of the fact that a fair random walk is *recurrent* (it returns to every level) but *null-recurrent* (the expected return time is infinite) — a distinction interviewers use to separate candidates who memorized "it always hits" from those who understand the tail.

#### Worked example: problem 7 — expected exit time from a two-barrier band

**Problem.** A driftless price starts at \$100, with an upper barrier at \$110 and a lower barrier at \$90. In variance-time units, what is the expected time until the price first touches *either* barrier? Solve it from scratch using the $B_t^2 - t$ martingale.

**Solution.** Recenter so the start is $0$: the barriers sit at $+10$ and $-10$. Let $\tau$ be the first exit time. The trick is *optional stopping* applied to the martingale $M_t = B_t^2 - t$. Because $M_t$ is a martingale and $\tau$ is a well-behaved stopping time, $\mathbb{E}[M_\tau] = M_0 = 0$. That gives

$$\mathbb{E}[B_\tau^2 - \tau] = 0 \quad\Longrightarrow\quad \mathbb{E}[\tau] = \mathbb{E}[B_\tau^2].$$

At the exit time $\tau$, the price is sitting exactly on one of the two barriers, so $B_\tau = +10$ or $B_\tau = -10$, and in either case $B_\tau^2 = 100$. By symmetry (driftless, equidistant barriers) it lands on each with probability one half, but it does not even matter here because both give $B_\tau^2 = 100$. Therefore

$$\mathbb{E}[\tau] = \mathbb{E}[B_\tau^2] = 100.$$

The expected exit time is $100$ variance-time units, matching the general formula $\mathbb{E}[\tau] = (x - L)(U - x) = (100-90)(110-100) = 10 \times 10 = 100$. **The $B_t^2 - t$ martingale is not decorative — it is the exact tool that turns an expected-hitting-time question into one line of optional stopping.** A frequent follow-up asks for the exit time of an *asymmetric* band, say barriers at $+5$ and $-15$ around the start; the same argument gives $\mathbb{E}[\tau] = \mathbb{E}[B_\tau^2]$, and now you weight $25$ and $225$ by the gambler's-ruin hit probabilities ($15/20$ and $5/20$) to get $\mathbb{E}[\tau] = \tfrac{15}{20}(25) + \tfrac{5}{20}(225) = 18.75 + 56.25 = 75$.

## Common misconceptions

**"Brownian motion has a velocity, it just changes fast."** No. The path has *no* derivative at *any* point — the difference quotient $\Delta B / \Delta t \approx 1/\sqrt{\Delta t}$ diverges as $\Delta t \to 0$. There is no instantaneous speed to speak of. This is not a technicality; it is *why* prices need Ito calculus. If $\mathrm{d}B/\mathrm{d}t$ existed you could do ordinary calculus and there would be no $\tfrac12\sigma^2$ correction, no Black-Scholes as we know it.

**"$B_t^2$ is a martingale because $B_t$ is."** Functions of martingales are usually *not* martingales. $B_t^2$ drifts upward at rate $1$ (its mean is $t$); you must subtract $t$ to get the martingale $B_t^2 - t$. The general lesson: only *affine* (linear-plus-constant) functions automatically preserve the martingale property; anything nonlinear, like a square, generates an Ito drift you have to account for.

**"Variance grows linearly, so a 4-year bet is 4× as risky as a 1-year bet."** The *variance* grows linearly ($\operatorname{Var}(B_t) = t$), but *risk* in the sense of typical magnitude is the *standard deviation*, which grows like $\sqrt{t}$. A 4-year horizon has $\sqrt{4} = 2$ times the spread of a 1-year horizon, not 4 times. Confusing variance-scaling with standard-deviation-scaling is one of the most common and most costly errors in risk work.

**"The reflection principle gives the probability the path *ends* above $a$."** It gives the probability the path *ever reached* $a$ — the running maximum — which is *twice* the probability of ending above $a$. The whole content of the principle is that factor of 2. Mixing up "touched" with "ended above" will hand you exactly half the right answer.

**"A stock is a Brownian motion."** A stock is better modeled as a *geometric* Brownian motion — the exponential of a drifting Brownian motion — so that it stays positive and so that *returns*, not dollar moves, are normal. Modeling the price level itself as $B_t$ lets it go negative and mis-scales the moves across price levels. The Brownian motion lives in the *log* of the price, not the price.

**"Independent increments means the path has no memory of its level."** The *increments* are independent, but the *level* $B_t$ obviously depends on the whole past — $B_t = B_s + (B_t - B_s)$ literally contains $B_s$. The covariance $\min(s,t) > 0$ is exactly this dependence of levels. What is memoryless is the *future increment*, not the current position.

## How it shows up on a real desk

**Pricing every vanilla option.** The Black-Scholes price of a call or put is, at bottom, an expectation under geometric Brownian motion. A derivatives desk re-derives the GBM dynamics, the $(\mathrm{d}B)^2 = \mathrm{d}t$ rule, and the Ito correction every time it builds or sanity-checks a pricer. When a trader quotes a 3-month at-the-money option, the $\sqrt{t}$ scaling of volatility is what converts the annualized implied vol into the price. Get the square-root-of-time wrong and your quotes are systematically off by a factor that arbitrageurs will happily collect.

**Barrier and exotic options.** A *knock-out* option dies if the underlying ever touches a barrier; a *knock-in* springs to life only if it does. Pricing these is the reflection principle in industrial form — the probability of touching a barrier before expiry, computed path-wise, is precisely $\mathbb{P}(M_t \ge a)$ generalized to drifting GBM. Desks that trade [autocallables](/blog/trading/quantitative-finance/autocallables) and barrier structures live and die by hitting-time mathematics; a sign error in a reflection argument can misprice a structured note by millions.

**Value-at-Risk and the square-root-of-time rule.** Risk teams routinely scale a one-day VaR to a ten-day VaR by multiplying by $\sqrt{10}$. That rule *is* the Brownian scaling property $B_{ct} \stackrel{d}{=} \sqrt{c}\, B_t$, and it is exact only when daily moves are independent and identically Gaussian — which they are not, quite. Knowing *why* the rule is $\sqrt{10}$ and not $10$, and where it breaks (fat tails, autocorrelation, volatility clustering), is the difference between a risk manager who applies a formula and one who knows its failure modes. The 2008 crisis was, in part, a lesson in how badly $\sqrt{t}$ scaling underestimates tail risk when returns are not Brownian.

**Monte Carlo simulation.** When a payoff is too complex for a closed-form price, desks simulate thousands of GBM paths and average the discounted payoffs. Every simulated path is a discretized Brownian motion — increments drawn as $\sqrt{\mathrm{d}t} \cdot Z$ for standard normals $Z$, exactly the scaling we derived. The variance of the Monte Carlo estimate, and the convergence rate, are governed by the same moment facts. Researchers tune step sizes and variance-reduction tricks using precisely the quadratic-variation and isometry results above.

**Interest-rate and credit models.** Short-rate models (Vasicek, Hull-White, CIR) and credit-spread models are all stochastic differential equations driven by Brownian motion. The same toolkit — moments, martingales, Ito — prices bonds, swaptions, and credit derivatives. The yield-curve and fixed-income analytics in [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics) and [bond pricing](/blog/trading/quantitative-finance/bond-pricing) ultimately rest on a Brownian driver under the hood.

**Statistical arbitrage and signal research.** Even on the systematic-equity side, the *null hypothesis* against which an alpha signal is tested is usually "returns are a driftless random walk." Distinguishing genuine predictability from the wiggles a Brownian motion would produce by chance is the core statistical problem; it is why a Sharpe ratio is essentially a signal-to-noise ratio measured against Brownian benchmarks. The [decision-making-under-uncertainty](/blog/trading/quantitative-finance/decision-making-under-uncertainty-quant-interviews) and [classic probability problems](/blog/trading/quantitative-finance/classic-quant-probability-problems) toolkits build directly on the moment and martingale facts in this post.

## The decision procedure to carry into the room

When an interviewer hands you a process built from Brownian motion and asks "is it a martingale?", you do not guess. You run a fixed procedure.

![Decision flow: take a candidate process, apply Ito, read the dt drift coefficient, and branch on whether the drift is zero](/imgs/blogs/brownian-motion-quant-interviews-12.png)

The flow above is the algorithm. Take the candidate $X_t = f(B_t, t)$. Apply Ito's formula to get $\mathrm{d}f = f_t\,\mathrm{d}t + f_x\,\mathrm{d}B + \tfrac12 f_{xx}\,\mathrm{d}t$. Collect the coefficient of $\mathrm{d}t$ — the drift — which is $f_t + \tfrac12 f_{xx}$. If that drift is identically zero, $X_t$ is a martingale; if not, it is not (and the drift tells you exactly what deterministic piece to subtract to *make* it one). Run it on the examples: $f = x$ gives drift $0$ (martingale); $f = x^2$ gives $f_{xx} = 2$, drift $1$ (not a martingale, subtract $t$); $f = e^{x - t/2}$ gives $f_t = -\tfrac12 f$ and $\tfrac12 f_{xx} = \tfrac12 f$, drift $0$ (martingale — this is the *exponential martingale* at the heart of risk-neutral pricing). The drift term is the whole game; it is the only thing that can break the fair-game property, and Ito's $\tfrac12 f_{xx}\,\mathrm{d}t$ correction is exactly why a square or a cube picks up a drift you must cancel.

If you want to reproduce the figures and build your own intuition, here is a minimal simulator. It draws Gaussian increments, accumulates them, and checks the variance-equals-$t$ and quadratic-variation facts.

```python
import numpy as np

def brownian_path(n_steps, T=1.0, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n_steps                       # time per step
    incr = rng.normal(0.0, np.sqrt(dt), n_steps)  # each ~ Normal(0, dt)
    B = np.concatenate([[0.0], np.cumsum(incr)])  # B_0 = 0, then accumulate
    return B, dt

B, dt = brownian_path(100_000, T=2.0)
print("B_T variance check:", B[-1] ** 2, "vs T =", 2.0)   # one draw, noisy
incr = np.diff(B)
print("quadratic variation:", np.sum(incr ** 2), "-> T =", 2.0)  # converges to T
```

Run it: the sum of squared increments lands almost exactly on $T = 2$ every time (quadratic variation is deterministic), while the single endpoint $B_T^2$ scatters around $T$ (a single draw is noisy). That contrast — one quantity pinned, the other random — is the whole point of quadratic variation, reproduced in five lines.

## When this matters to you and where to go next

If you are preparing for quant researcher or derivatives interviews, treat the three defining properties, the moments ($\mathbb{E} = 0$, $\operatorname{Var} = t$, $\operatorname{Cov} = \min(s,t)$), the martingale facts ($B_t$ and $B_t^2 - t$), and the quadratic-variation rule ($(\mathrm{d}B)^2 = \mathrm{d}t$) as non-negotiable recall — the way a pianist knows scales. The reflection principle and the two-barrier formula are the highest-leverage *tricks*, because they turn intimidating path questions into one-line answers. Everything in derivatives pricing is downstream of these.

The natural next steps build directly on this foundation. [The Black-Scholes deep dive](/blog/trading/quantitative-finance/black-scholes) shows how the Ito correction and geometric Brownian motion combine into the most famous formula in finance. [Derivatives pricing](/blog/trading/quantitative-finance/derivatives-pricing) and [options theory](/blog/trading/quantitative-finance/options-theory) put the martingale and risk-neutral machinery to work. [Markov chains and hitting times](/blog/trading/quantitative-finance/markov-chains-hitting-times-quant-interviews) is the discrete sibling of everything here — gambler's ruin and first-passage in a setting where you can compute everything by hand first. And [the Kelly criterion](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) shows how the same fair-game and log-growth ideas drive optimal bet sizing.

*This is educational material, not investment advice. The models here are deliberate simplifications — real markets have fat tails, jumps, and changing volatility that Brownian motion does not capture, which is exactly why understanding the base model and its limits matters.*
