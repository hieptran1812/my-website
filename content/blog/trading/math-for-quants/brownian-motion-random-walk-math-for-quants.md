---
title: "Brownian motion from the random walk: the math of price noise"
date: "2026-06-15"
description: "How a fair coin-flip walk, shrunk in step size and sped up in time, becomes continuous Brownian motion -- the model behind why volatility scales with the square root of time, why ordinary calculus fails for prices, and how a first price model is built from drift plus noise, all from zero with worked dollar examples."
tags: ["brownian-motion", "random-walk", "wiener-process", "donsker", "quadratic-variation", "volatility-scaling", "diffusion", "stochastic-calculus", "arithmetic-brownian-motion", "square-root-of-time", "quantitative-finance"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** -- a stock price wiggling through the day is, to a first approximation, a fair coin-flip walk that has been shrunk until the steps are invisible; that limit is Brownian motion, and it controls almost everything about how risk grows with time.
>
> - **The random walk is the seed.** Flip a fair coin each instant, step up on heads and down on tails, and add the steps. Shrink the step size and speed up the clock together in the right ratio, and the jagged path converges to a smooth-looking but infinitely jagged continuous curve: *Brownian motion* $W_t$. This convergence is *Donsker's theorem*, and it is why the same model fits a thousand different markets.
> - **Four rules define it completely.** It starts at zero ($W_0 = 0$); its increments over non-overlapping intervals are *independent*; an increment over a window of length $t-s$ is *Normal with mean 0 and variance $t-s$*; and its path is *continuous*. Everything else in the post is a consequence of these four.
> - **Variance grows linearly, so the typical move grows with $\sqrt{t}$.** Over four times the horizon the variance is four times larger, but the typical move is only $\sqrt{4}=2$ times larger. This single fact -- the *square-root-of-time law* -- is how every desk annualizes volatility and sizes a move over any horizon.
> - **Ordinary calculus breaks.** A Brownian path is continuous but *nowhere differentiable* and of *infinite length*; the sum of its squared increments over $[0,t]$ converges not to zero but to exactly $t$. That number, the *quadratic variation* $[W]_t = t$, is the seed of Itô calculus and the reason prices need their own math.
> - **The one number to remember**: a 1%-per-day stock on a \$1,000,000 position has a typical daily move of about \$10,000, but a typical *one-year* (252-trading-day) move of about \$159,000 -- not \$2,520,000 -- because risk scales with $\sqrt{252}\approx 15.9$, not with 252.

## Why a stock chart looks like a drunk's walk home

Picture the most stripped-down gamble there is. You flip a fair coin once a second. Heads, you take one step east; tails, one step west. After an hour you have flipped 3,600 times and wandered some distance from where you started -- but which direction, and how far? You cannot know in advance, because each step is a fresh coin flip. This wandering path has a name that is almost too perfect: the *random walk*, sometimes called the *drunkard's walk*, after the staggering pedestrian who is as likely to lurch left as right.

Now open any intraday stock chart and squint. A share of some company opens at \$100. A second later it is \$100.02; a second after that, \$99.99; then \$100.01, \$100.04, \$100.00. Up a tick, down a tick, up two, down one. It is the drunkard again. Each tiny price change looks like a fresh coin flip: a little up or a little down, with no obvious memory of what just happened. The astonishing and slightly unsettling claim at the heart of quantitative finance is that this resemblance is not a coincidence and not merely poetic -- it is the *right first model*. A price, over short horizons, behaves remarkably like the running total of a very fast, very small coin-flip walk. And when you take that coin-flip walk and shrink the steps until they vanish, you get one specific, beautiful mathematical object that every quant must understand cold: *Brownian motion*.

![A pipeline from fair coin flips to a random walk to a rescaled limit to Brownian motion to a price noise model.](/imgs/blogs/brownian-motion-random-walk-math-for-quants-1.png)

The diagram above is the mental model for the entire post, read left to right. We start with the humblest ingredient -- fair coin flips. We add them up into a random walk. Then we do something clever: we *shrink the step size and speed up the clock together*, in a precisely tuned ratio, and watch what the walk converges to. The limit is Brownian motion, denoted $W_t$ (the $W$ honors Norbert Wiener, who built the rigorous version, which is why it is also called the *Wiener process*). That limit is then the engine of every continuous-time model of price *noise* -- the random, unpredictable part of how prices move.

This post builds the whole chain from zero. We assume you know what a fair coin is and are comfortable with a square root; we do not assume you have ever seen a normal distribution, a variance, or a derivative from calculus. By the end you will know exactly what Brownian motion is, why its variance grows linearly while its typical move grows with the square root of time, why ordinary calculus refuses to work on it, and how to assemble a first crude price model -- and see precisely where that first model breaks, which is the reason the next post in this series reaches for the *lognormal* fix. None of this is investment advice; it is the mathematical scaffolding that everything from option pricing to risk management is built on, and we will be honest throughout about where the model is a useful lie.

## Foundations: the random walk and its limit

Before any continuous-time machinery, we nail down the discrete object it comes from. A reader who already knows random walks cold can skim; a beginner cannot proceed without this section, because Brownian motion is *defined* as the limit of what we build here.

### What a random variable, mean, and variance are

A *random variable* is just a number whose value is decided by chance -- the outcome of a coin flip, a die roll, tomorrow's price change. We cannot say what it will be, but we can describe how it tends to behave with two summary numbers.

The first is the *mean*, also called the *expected value* and written $E[X]$: the long-run average value of the random variable $X$ if you could repeat the experiment forever. For a single coin-flip step that is $+1$ on heads and $-1$ on tails, each with probability one-half, the mean is $E[X] = (\tfrac12)(+1) + (\tfrac12)(-1) = 0$. On average you go nowhere -- the up and down cancel.

The second is the *variance*, written $\mathrm{Var}(X)$, which measures how *spread out* the outcomes are around the mean. It is defined as the average squared distance from the mean: $\mathrm{Var}(X) = E[(X - E[X])^2]$. We square the distance so that a step down counts just as much as a step up (a negative distance squared is positive) and so that big deviations are penalized more than small ones. For our coin step, the mean is 0, so the variance is the average of $(+1)^2$ and $(-1)^2$, which is $\tfrac12(1) + \tfrac12(1) = 1$. The *standard deviation*, written $\sigma$ (lowercase sigma), is just the square root of the variance, $\sigma = \sqrt{\mathrm{Var}(X)}$; it lives in the same units as $X$ itself and is the most natural single measure of "how big is a typical move." For our coin step, $\sigma = \sqrt{1} = 1$ -- a typical step is size one, which makes sense since every step is exactly size one.

Two facts about variance do all the heavy lifting in this post. First, **variance scales with the square of a constant**: if you multiply a random variable by a constant $c$, its variance multiplies by $c^2$, because $\mathrm{Var}(cX) = c^2\,\mathrm{Var}(X)$. (Standard deviation, being the square root, just scales by $|c|$.) Second, **variances of independent random variables add**: if $X$ and $Y$ are *independent* -- meaning knowing one tells you nothing about the other -- then $\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$. Crucially, the *means* add too, but for variance this additivity holds only when the pieces are independent. These two rules -- square scaling and additivity under independence -- are the entire mathematical reason that risk grows with the square root of time, as we will see.

### Building the symmetric random walk

Now we stack the steps. Let $\xi_1, \xi_2, \xi_3, \dots$ (lowercase xi) be a sequence of independent coin-flip steps, each $+1$ or $-1$ with probability one-half. The *symmetric random walk* after $n$ steps is the running total:

$$ S_n = \xi_1 + \xi_2 + \cdots + \xi_n. $$

Here $S_n$ is the position after $n$ steps, "symmetric" because up and down are equally likely. $S_0 = 0$: you start at the origin. Let us read off its two summary numbers using the rules above.

The mean is easy: each step has mean 0, and means add, so $E[S_n] = 0$ for every $n$. On average the walk goes nowhere -- it is exactly as likely to drift east as west, so the expectation is dead center.

The variance is where the magic is. The steps are independent, and variances of independent things add, so $\mathrm{Var}(S_n) = \mathrm{Var}(\xi_1) + \cdots + \mathrm{Var}(\xi_n) = 1 + 1 + \cdots + 1 = n$. The variance after $n$ steps is exactly $n$. Therefore the standard deviation -- the size of a typical displacement from the origin -- is $\sqrt{n}$. After 100 steps, the walk is typically about $\sqrt{100} = 10$ steps from home, not 100. After 10,000 steps, typically about 100 steps from home, not 10,000. **The walk spreads out, but slowly: distance from home grows like the square root of the number of steps.** Hold onto that. It is the discrete ancestor of the most important scaling law in the whole post.

#### Worked example: the spread of a 100-step walk

Suppose each step of our walk is one dollar -- you win or lose \$1 on each coin flip in a long game. After $n = 100$ flips, what is the typical size of your net position?

The mean is $E[S_{100}] = 0$: on average you break even, exactly as a fair game should leave you. The variance is $\mathrm{Var}(S_{100}) = 100$, so the standard deviation is $\sqrt{100} = \$10$. That is the key number. It does *not* mean you will be exactly \$10 up or \$10 down; it means a *typical* outcome is in the neighborhood of \$10 away from zero in either direction. You might be \$4 up, or \$22 down, but \$10 is the natural scale of the swing. Notice the asymmetry of intuition here: you played 100 rounds, risking \$1 each, for a total of \$100 of action, yet your typical net result is only \$10 -- one tenth of the gross. That tenfold gap is $\sqrt{100}/100 = 1/\sqrt{100} = 1/10$. The more you play a fair game, the more your *gross* exposure grows relative to your *net* result, because the net grows only with the square root. The intuition: in a fair game, wins and losses largely cancel, and what survives the cancellation grows with the square root of the number of bets, not with the number of bets.

### Why the square root, and not linear growth

It is worth pausing on *why* the standard deviation grows as $\sqrt{n}$ and not as $n$, because it confuses nearly everyone the first time and it underpins everything that follows. If the steps all went the same direction, the total distance would grow linearly -- 100 steps east is 100 units east. But the steps are random, and they *partially cancel*. Roughly half go east and half go west, and the cancellation is nearly complete; what is left over is the small imbalance between heads and tails. That imbalance, by the additivity of variance, has size $\sqrt{n}$. The walk is a tug-of-war between two nearly equal teams, and the rope ends up displaced not by the total number of pullers but by the square root of it.

It helps to compare two extremes side by side. A perfectly *trending* series -- every step in the same direction -- travels a distance proportional to the number of steps, $n$; double the steps and you double the distance. A perfectly *random* series travels a distance proportional to $\sqrt{n}$; double the steps and you multiply the distance by only $\sqrt{2}\approx 1.41$. Real markets sit somewhere between these poles, and the *whole question* of whether a strategy can work is the question of how far from the pure-random $\sqrt{n}$ baseline a particular series departs. A momentum strategy is a bet that the series leans toward the linear, trending pole; a mean-reversion strategy is a bet that it overshoots and snaps back, traveling *less* than $\sqrt{n}$. Brownian motion is the exact dead center -- the pure-random baseline against which both bets are measured.

This is also why you should be deeply suspicious any time someone reasons as if risk over a long horizon were the daily risk multiplied by the number of days. That would be the linear, all-steps-aligned answer, and it is wrong for anything that wanders randomly. The correct multiplier is the square root of the number of days, and the gap between the two grows enormous over long horizons -- a fact we will turn into dollars shortly.

### The scaling limit: shrinking the walk

Here is the leap from discrete to continuous. We want a model that lives in *continuous time* -- a price defined at every instant, not just on the tick of a clock. So we take our random walk and do two things at once, tuned to balance perfectly.

Suppose we want to describe the walk over a fixed stretch of real time, say from time 0 to time 1 (think "one trading day"). We chop that interval into $n$ tiny sub-steps. To keep the walk from either exploding or collapsing as $n$ grows, we must shrink each step's *size*. The right shrink, it turns out, is by a factor of $1/\sqrt{n}$: each step becomes size $1/\sqrt{n}$ instead of size 1, and we take $n$ of them in the unit interval. Why $1/\sqrt{n}$ and not, say, $1/n$? Because variances add and scale with the square of size: with $n$ steps each of variance $(1/\sqrt{n})^2 = 1/n$, the total variance over the unit interval is $n \times (1/n) = 1$ -- a fixed, finite number, no matter how fine we slice. If we shrank by $1/n$ instead, the total variance would be $n \times (1/n)^2 = 1/n \to 0$, and the walk would collapse to a flat line. The $1/\sqrt{n}$ shrink is the unique choice that keeps the limit alive and non-degenerate.

What do we get in the limit as $n \to \infty$? A continuous random curve, defined at every instant of $[0,1]$ (and then, by gluing such intervals, on all of $[0,\infty)$), whose value at time $t$ has mean 0 and variance exactly $t$. That curve is *standard Brownian motion*, written $W_t$. The fact that the rescaled random walk genuinely converges to this object -- not just in its endpoint but as a whole path -- is the content of *Donsker's theorem*, also called the *functional central limit theorem*, which we treat next.

## 1. The scaling limit: Donsker's theorem

The ordinary *central limit theorem* (CLT), which we cover in depth in [the law of large numbers and CLT post](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants), says that the sum of many independent steps, properly rescaled, becomes a *normal distribution* (the bell curve). Specifically, $S_n / \sqrt{n}$ converges to a normal random variable with mean 0 and variance 1. That is a statement about a single number -- the endpoint of the walk after $n$ steps.

Donsker's theorem is the CLT upgraded from a single number to an entire *path*. It says: if you take the whole rescaled random-walk path -- not just its endpoint, but its value at every fraction of the way through -- and let the step count go to infinity, the entire random curve converges (in a precise probabilistic sense) to the random curve called Brownian motion. The technical name is the *functional* central limit theorem, "functional" because the thing converging is a function (a path), not a number.

Why does a quant care about a convergence theorem? Two practical reasons. First, **universality**. The CLT and Donsker do not care about the fine print of the individual steps. The steps did not have to be $\pm 1$ coins; they could be any independent increments with finite variance -- daily returns of a stock, minute-by-minute moves of a currency, the jiggle of a futures contract. As long as the pieces are roughly independent and none of them dominates, their rescaled sum looks like the *same* Brownian motion. This is why one model -- continuous Brownian noise -- is the default first description across wildly different markets. It is not that prices "are" coin flips; it is that the *sum* of many small, roughly independent shocks always smooths into the same limiting shape, regardless of the shocks' details. The same reason the bell curve shows up everywhere makes Brownian motion show up everywhere.

Second, **it tells you what to keep and what to throw away**. Donsker says the limit depends on the steps only through their variance, not their exact distribution. So when modeling a price, the one number that survives into the continuous limit is the *volatility* -- the standard deviation of the increments. The skew, the precise tail shape, the exact spacing of trades -- all of that washes out in the idealized limit. That is both the model's great strength (you only need to estimate one number, the volatility) and its great weakness (real markets have fat tails and jumps that the Brownian limit erases, which is exactly the gap that more advanced models in this series are built to fill).

![A before-and-after contrast of a discrete random walk with fixed steps against a continuous Brownian path defined at every instant.](/imgs/blogs/brownian-motion-random-walk-math-for-quants-2.png)

The figure above contrasts the two worlds. On the left, the discrete random walk: a fixed step size, jumps only at clock ticks 1, 2, 3, a countable number of jumps, and -- importantly -- an object you can do ordinary calculus on, because between ticks it is a straight line with a perfectly well-defined slope. On the right, the Brownian limit: the step size has gone to zero, the path is defined at *every* instant, it is infinitely wiggly at every scale, and -- as we will prove -- ordinary calculus fails on it completely. The left is a staircase; the right is a coastline that stays jagged no matter how far you zoom in.

> The discrete walk and the continuous limit are *not* the same object with different resolution. The limit acquires genuinely new and strange properties -- infinite wiggle, infinite path length, non-differentiability -- that no finite walk has. The continuum is a different beast, and its strangeness is exactly what makes it useful.

## 2. The defining properties

Mathematicians do not usually define Brownian motion as "the limit of a rescaled random walk" -- that is the intuition for *where it comes from*. They define it by four properties that pin it down uniquely. Memorize these four; every formula in this post is a direct consequence of them.

![A stack of the four defining properties of Brownian motion, from starting at zero to continuous paths.](/imgs/blogs/brownian-motion-random-walk-math-for-quants-3.png)

The stack above lists them in the order they are usually stated. Let us define each from zero.

**Property 1: It starts at zero.** $W_0 = 0$. The process begins at the origin by convention. When we model a real price we will add the starting price back on top; the $W$ itself measures only the cumulative *noise* away from the start.

**Property 2: Independent increments.** An *increment* is the change in the process over a time window: $W_t - W_s$ is the increment from time $s$ to time $t$ (with $s < t$). "Independent increments" means that increments over *non-overlapping* windows are independent random variables -- the change from 9:30 to 10:00 tells you nothing about the change from 10:00 to 10:30. This is the continuous-time version of "each coin flip is fresh." It is also exactly the *efficient-market* intuition in mathematical form: the future move is independent of the past path, so there is no pattern in the noise to exploit. (Real markets violate this somewhat -- there is autocorrelation, there are regimes -- but it is the right baseline, and departures from it are precisely what statistical arbitrage hunts for.)

**Property 3: Increments are normally distributed with variance equal to the elapsed time.** The increment $W_t - W_s$ has a *normal distribution* with mean 0 and variance $t - s$. In symbols, $W_t - W_s \sim N(0, t-s)$, where $N(\mu, \sigma^2)$ denotes the normal (bell-curve) distribution with mean $\mu$ and variance $\sigma^2$. In particular, taking $s = 0$ and using $W_0 = 0$, we get $W_t \sim N(0, t)$: the value at time $t$ is normal with mean 0 and variance $t$. The variance equals the elapsed time -- this is the linear-variance-growth law, baked straight into the definition. The normality is the gift of Donsker: the sum of many small independent shocks is normal, so the increment of the limit is normal too.

**Property 4: Continuous paths.** As a function of time, $W_t$ is *continuous* -- it has no instantaneous jumps; you can draw it without lifting your pen. This rules out sudden gaps. (Real prices *do* gap -- on earnings, on overnight news -- which is why jump-diffusion models exist; but pure Brownian motion is continuous, and that continuity is what lets the calculus of the next post work.)

That is the entire definition. Notice what is *not* in it: nothing about differentiability, nothing about the path's length, nothing about the typical move over a horizon. Those are all *derived* facts, and deriving them is how we build real understanding.

### Reading the distribution of $W_t$

Because $W_t \sim N(0, t)$, everything we know about the normal distribution applies. The standard deviation of $W_t$ is $\sqrt{t}$ -- variance $t$, so standard deviation $\sqrt{t}$. The normal distribution has the famous *68-95-99.7 rule*: about 68% of outcomes fall within one standard deviation of the mean, about 95% within two, and about 99.7% within three. For $W_1$ (variance 1, standard deviation 1), that means about 95% of the time $W_1$ lands between $-2$ and $+2$, and only about 5% of the time does it land outside that band. For the full menu of distributions a quant uses, see [the distributions-for-markets post](/blog/trading/math-for-quants/probability-distributions-for-markets-math-for-quants); here the normal is all we need.

#### Worked example: the distribution of $W_1$ and a price move

A trader models the one-day noise in a futures contract as $\sigma \cdot W_1$ where the daily volatility is $\sigma = \$2{,}000$ of profit-and-loss per contract. So a one-standard-deviation day moves the position by \$2,000. What is the chance the day's noise exceeds \$4,000 in size, in *either* direction?

A \$4,000 move is two standard deviations ($\$4{,}000 / \$2{,}000 = 2$). We want $P(|W_1| > 2)$. From the 68-95-99.7 rule, about 95% of the probability sits inside two standard deviations, so about 5% sits outside -- split between the two tails. The precise figure: for a standard normal, $P(|Z| > 2) \approx 0.0455$, or about **4.55%**. So roughly one trading day in 22 (since $1/0.0455 \approx 22$), this single contract's noise moves the book by more than \$4,000.

Now scale it up. If the trader holds 50 such contracts and they all share the same daily noise (perfectly correlated, a simplification), the one-standard-deviation day is $50 \times \$2{,}000 = \$100{,}000$, and a two-standard-deviation day is \$200,000 -- which, again, happens about 4.55% of the time, or roughly once a month of trading days. The intuition: the normal distribution makes a two-standard-deviation move uncommon but far from rare -- it shows up about once every three weeks of trading, which is why a desk that is only prepared for one-sigma days will get hurt regularly.

A practical caution worth stating now: real return distributions have *fatter tails* than the normal. The true frequency of a "two-sigma" P&L day in equities is meaningfully higher than 4.55% -- often closer to 5-6% -- because markets have more extreme days than a bell curve predicts. The Brownian model understates tail risk, and the [tail-risk post](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants) is where that gap gets quantified. We use the normal here because it is the clean baseline; never mistake the baseline for the truth.

## 3. Non-differentiability and quadratic variation

Here is where Brownian motion stops being "a random walk with smaller steps" and becomes something genuinely alien -- and where we earn the right to claim that *ordinary calculus fails for prices*.

### A path with no slope

In ordinary calculus, the central object is the *derivative*: the slope of a curve at a point, defined as the limit of $(W_{t+h} - W_t)/h$ as the time-gap $h$ shrinks to zero. For a smooth curve like a parabola, that ratio settles down to a definite number -- the slope. For a Brownian path, it does not.

Here is the reason in one line. The numerator, $W_{t+h} - W_t$, is an increment over a window of length $h$, so by Property 3 it is normal with standard deviation $\sqrt{h}$ -- a *typical* numerator is of size $\sqrt{h}$. The denominator is $h$. So the ratio is of size $\sqrt{h}/h = 1/\sqrt{h}$. As $h \to 0$, that blows up to infinity. The slope does not settle to a number; it explodes. At every single point, in every shrinking window, the Brownian path is too jagged to have a well-defined slope. **Brownian motion is continuous everywhere but differentiable nowhere.** You can draw it without lifting your pen, yet at no point does it have a tangent line. This was a genuinely shocking fact when it was first proven -- such curves were once thought not to exist -- and it is the precise mathematical statement of "a price has no velocity."

The consequence for finance is profound. The whole apparatus of ordinary calculus -- the chain rule, integration, differential equations -- is built on derivatives. If a price path has no derivative anywhere, you cannot write $dP/dt$ and do calculus the usual way. You need a *new* calculus, one built for objects that wiggle infinitely. That new calculus is *Itô calculus*, the subject of the next post in this series, and the bridge to it is the single fact we build next.

### Unbounded variation: the path has infinite length

A related strangeness: a Brownian path traced over any time interval, however short, has *infinite length*. The *variation* of a path is the total distance it travels -- the sum of the absolute sizes of all its ups and downs. For a smooth curve over a finite interval, that total is finite (it has *bounded variation*). For Brownian motion, it is infinite. The reason rhymes with the non-differentiability: over $n$ sub-intervals each of length $t/n$, each up-or-down move has typical size $\sqrt{t/n}$, and there are $n$ of them, so the total distance traveled is about $n \times \sqrt{t/n} = \sqrt{n}\,\sqrt{t} = \sqrt{nt}$, which goes to infinity as $n \to \infty$. The finer you measure, the more wiggle you find, without bound. A Brownian path is a coastline: its measured length grows the more closely you look.

### Quadratic variation: the sum of squares that does not vanish

Now the punchline, and the most important single equation in the post after the definition itself. We just saw that the sum of the *absolute* increments blows up to infinity. What happens if instead we sum the *squared* increments?

Chop $[0,t]$ into $n$ equal sub-intervals of length $\Delta t = t/n$. The increment over the $k$-th sub-interval is $\Delta W_k = W_{t_k} - W_{t_{k-1}}$, which is normal with mean 0 and variance $\Delta t$. Now form two sums.

**Sum of the increments themselves:** $\sum_{k=1}^n \Delta W_k = W_t - W_0 = W_t$. The increments telescope -- each one's end is the next one's start -- so they collapse to a single random number, $W_t$, which has mean 0 and standard deviation $\sqrt{t}$. As you slice finer, this sum does not converge to a constant; it stays a random variable hovering around zero, because the plus and minus increments largely cancel. The expected value is 0.

**Sum of the squared increments:** $\sum_{k=1}^n (\Delta W_k)^2$. Each squared increment is positive (squaring kills the sign), so *nothing cancels* -- the terms pile up. What do they pile up to? The expected value of each $(\Delta W_k)^2$ is exactly its variance, which is $\Delta t = t/n$. There are $n$ of them, so the expected total is $n \times (t/n) = t$. And -- this is the deep part -- as the slicing gets finer, this sum does not just have *expected value* $t$; it *converges* to the constant $t$ itself, with the randomness washing out entirely. In symbols:

$$ [W]_t = \lim_{n\to\infty}\sum_{k=1}^n (\Delta W_k)^2 = t. $$

This quantity $[W]_t$ is the *quadratic variation* of Brownian motion, and the result $[W]_t = t$ is the single most consequential fact in stochastic calculus. Read it slowly: **the sum of the squared wiggles over $[0,t]$ is not random and is not zero -- it is exactly the elapsed time $t$.**

![A before-and-after contrast of the sum of signed increments cancelling toward zero against the sum of squared increments accumulating to the elapsed time t.](/imgs/blogs/brownian-motion-random-walk-math-for-quants-5.png)

The figure above makes the contrast concrete. On the left, the signed increments carry plus and minus signs, they cancel, the running total is just the wandering $W_t$, and its average is near zero. On the right, every squared increment is positive, nothing can cancel, the total locks onto $t$ no matter how finely you slice -- and that locked-on value is the quadratic variation. The reason this matters so much is that in ordinary calculus, squared increments are *negligible*: for a smooth function, $(\Delta f)^2$ shrinks like $(\Delta t)^2$, which vanishes faster than $\Delta t$ and can be dropped. For Brownian motion, $(\Delta W)^2$ is of size $\Delta t$ -- the *same order* as $\Delta t$ itself -- so it cannot be dropped. That single surviving second-order term is the entire reason Itô's formula has an extra piece that ordinary calculus does not, the famous "$\tfrac12 \sigma^2$" correction. Everything strange and powerful about pricing options traces back to this one line, $[W]_t = t$.

#### Worked example: sum of increments versus sum of squares

Let us watch both sums numerically over one unit of time, $t = 1$, sliced into $n = 4$ steps of length $\Delta t = 0.25$ each. Suppose a particular Brownian path produced these four increments (each should be roughly size $\sqrt{0.25} = 0.5$): $\Delta W_1 = +0.6$, $\Delta W_2 = -0.4$, $\Delta W_3 = +0.5$, $\Delta W_4 = -0.3$.

Sum of the increments: $0.6 - 0.4 + 0.5 - 0.3 = +0.4$. This equals $W_1$ for this path -- the net displacement after one unit of time, a modest number near zero because the ups and downs partly cancel. Run a different path and you would get a different value, hovering around zero with standard deviation $\sqrt{1} = 1$.

Sum of the squared increments: $(0.6)^2 + (-0.4)^2 + (0.5)^2 + (-0.3)^2 = 0.36 + 0.16 + 0.25 + 0.09 = 0.86$. With only 4 coarse steps this is a rough estimate of the true quadratic variation $t = 1$; slice into 100 or 10,000 steps and the sum tightens onto 1.00 with the scatter shrinking to nothing. To make the dollars concrete: if these were daily P&L moves on a position scaled so each unit is \$10,000, the *net* result for the period is $0.4 \times \$10{,}000 = \$4{,}000$ (small, near zero), while the *accumulated squared variation* -- the raw quantity of risk the position chewed through -- is locked near $1.00 \times (\$10{,}000)^2$, a fixed amount of realized variance independent of which way the path wandered. The intuition: the *direction* a price takes is random and averages out, but the *amount of variance it realizes* is deterministic and equal to the elapsed time scaled by volatility -- which is exactly why a volatility seller can be confident about how much variance they will collect even when they have no idea which way the market will go.

## 4. The $\sqrt{t}$ law of diffusion

We now cash the most useful consequence of the definition into the formula every desk uses daily. From Property 3, $W_t \sim N(0, t)$, so the standard deviation of $W_t$ is $\sqrt{t}$. The typical *size* of where Brownian motion has wandered after time $t$ is $\sqrt{t}$. This is the *square-root-of-time law*, also called the *diffusion scaling*, and it is the single most important practical fact in this entire post.

Stated in trading terms: if a price's noise over one unit of time has standard deviation $\sigma$ (its *volatility*), then its noise over a horizon of $t$ units has standard deviation $\sigma\sqrt{t}$ -- not $\sigma t$. Volatility scales with the *square root* of the horizon. This is the direct continuous-time descendant of the random walk's "displacement grows like $\sqrt{n}$" that we proved in the foundations.

![A matrix of horizons against the square-root-of-time factor, the implied move size, and the dollar band on a one-million-dollar position.](/imgs/blogs/brownian-motion-random-walk-math-for-quants-4.png)

The matrix above is the working trader's cheat sheet, and it repays study. Each row is a horizon. The first column is the $\sqrt{t}$ multiplier relative to one day. The second column converts a 1%-per-day volatility into the volatility over that horizon. The third column puts dollars on a \$1,000,000 position. Notice the headline: going from 1 day to 252 days (a trading year) multiplies the horizon by 252 but multiplies the move size only by $\sqrt{252} \approx 15.9$. The risk does not grow with calendar time; it grows with its square root. A reader who internalizes only this one table has captured most of the practical value of Brownian motion.

### Annualizing volatility

The $\sqrt{t}$ law is exactly how the industry converts between horizons. The standard convention uses about 252 trading days in a year (markets are closed on weekends and holidays). If you measure a stock's *daily* return volatility as $\sigma_{\text{daily}}$, the *annual* volatility is:

$$ \sigma_{\text{annual}} = \sigma_{\text{daily}} \times \sqrt{252}. $$

The $\sqrt{252} \approx 15.87$ factor is one of the most-used numbers on a trading floor. A stock with a 1% daily volatility has an annual volatility of about $1\% \times 15.87 \approx 16\%$ -- which happens to be a very typical figure for a large-cap stock. Going the other way, a stated annual volatility of 16% implies a daily volatility of $16\% / \sqrt{252} \approx 1\%$. The conversion runs both directions through the same square-root factor. (If you instead want to scale by calendar days for an instrument that trades every day, you would use $\sqrt{365}$; the principle is identical, only the day-count changes.)

#### Worked example: the $\sqrt{t}$ scaling of a move

You manage a \$1,000,000 position in a stock whose daily return volatility is $\sigma_{\text{daily}} = 1\%$. The typical *daily* move in dollars is $1\% \times \$1{,}000{,}000 = \$10{,}000$. Now size the typical move over three horizons.

Over **1 day**: the multiplier is $\sqrt{1} = 1$, so the typical move is $\$10{,}000 \times 1 = \$10{,}000$.

Over **10 days** (about two trading weeks): the multiplier is $\sqrt{10} \approx 3.16$, so the typical move is $\$10{,}000 \times 3.16 = \$31{,}600$. Note it is *not* $\$10{,}000 \times 10 = \$100{,}000$; that linear answer would overstate the 10-day risk by more than threefold.

Over **252 days** (one trading year): the multiplier is $\sqrt{252} \approx 15.87$, so the typical move is $\$10{,}000 \times 15.87 = \$158{,}700$. The naive linear answer of $\$10{,}000 \times 252 = \$2{,}520{,}000$ overstates the annual risk by a factor of about 16 -- it would have you holding capital against a swing that is fifteen-sixteenths imaginary.

To put a band on it: a *one-standard-deviation* annual move is about \$158,700, so a *two-standard-deviation* annual band -- inside which the position lands about 95% of the time under the normal model -- is roughly $\pm\$317{,}400$ on the million-dollar position. The intuition: risk does not add up day by day, it accumulates by the square root of time, so a year is only about 16 times as risky as a day even though it contains 252 of them -- and any risk system that scales linearly will demand far too much capital.

#### Worked example: comparing two horizons directly

A risk manager knows the 1-day value-at-risk band of a book is \$50,000 (a one-sigma daily move). A regulator asks for the 10-day figure. The wrong instinct is to multiply by 10 to get \$500,000. The right answer scales by $\sqrt{10} \approx 3.16$: the 10-day one-sigma band is $\$50{,}000 \times 3.16 = \$158{,}000$. This is not a rounding quibble -- the linear answer is more than three times too large, and over a quarter (about 63 trading days) the gap is starker still: $\sqrt{63} \approx 7.94$ versus a linear 63, an eightfold overstatement. Regulatory frameworks like Basel explicitly use the square-root-of-time rule to scale a short-horizon risk number to a longer one, precisely because the underlying model is Brownian. The intuition: whenever you stretch a risk number to a longer horizon, multiply by the square root of the horizon ratio, never the ratio itself -- the difference between the two is the difference between a sound risk limit and one that strangles the book.

## 5. Drift plus diffusion: arithmetic Brownian motion

So far our Brownian motion has mean 0 -- it wanders, but it does not *trend*. Real assets often do have a tendency to drift in one direction, whether from a risk premium, an interest rate, or a strategy's expected edge. We add that with a *drift* term, and the result is our first honest, if flawed, price model.

The model is *arithmetic Brownian motion*, sometimes called *Brownian motion with drift*:

$$ X_t = X_0 + \mu t + \sigma W_t. $$

Let us read every symbol. $X_t$ is the modeled price (or P&L) at time $t$. $X_0$ is the starting value. $\mu$ (mu) is the *drift*: the average rate of change per unit time -- the deterministic, predictable trend. $\sigma$ (sigma) is the *volatility*: the size of the random noise per unit time, scaling the Brownian part. And $W_t$ is our standard Brownian motion, contributing the randomness. The model has a clean split: $X_0 + \mu t$ is the *signal*, a straight line you would draw if there were no randomness, and $\sigma W_t$ is the *noise* wiggling around that line.

The distribution follows immediately. Since $W_t \sim N(0, t)$, scaling by $\sigma$ and shifting by $X_0 + \mu t$ gives:

$$ X_t \sim N\big(X_0 + \mu t,\ \sigma^2 t\big). $$

The mean grows *linearly* with time ($X_0 + \mu t$ -- the drift compounds steadily), while the standard deviation grows with the *square root* of time ($\sigma\sqrt{t}$ -- the diffusion). This is the crucial tension in all of finance: signal accumulates linearly, noise accumulates as the square root. Over short horizons the noise (size $\sqrt{t}$) dominates the signal (size $t$), because $\sqrt{t} > t$ when $t$ is small. Over long horizons the signal eventually wins, because $t$ eventually outgrows $\sqrt{t}$. The crossover -- when does the trend finally poke above the noise? -- is the mathematical heart of why short-term trading is mostly noise and long-term investing is mostly signal, and it is the reason a real edge needs either a large $\mu$ or a long $t$ to become statistically visible above the $\sqrt{t}$ fog.

![A timeline showing variance growing linearly with time while the standard deviation, its square root, grows more slowly.](/imgs/blogs/brownian-motion-random-walk-math-for-quants-6.png)

The timeline above tracks the noise term as time marches: variance climbs in a straight line (1, 2, 4, 9, 16) while the standard deviation -- the actual typical move -- climbs as the square root (1.00, 1.41, 2.00, 3.00, 4.00). Variance is the quantity that adds cleanly across independent periods; the standard deviation is the quantity you feel as a dollar move. Keeping the two straight -- variance adds, standard deviation grows by square root -- is the discipline that separates a correct risk calculation from a wrong one.

#### Worked example: one step of an arithmetic Brownian price path

Let us simulate a single time step and read off the P&L. Take a stock starting at $X_0 = \$100$, with an annual drift of $\mu = \$8$ per year (an 8% expected return, expressed in dollars on a \$100 stock) and an annual volatility of $\sigma = \$20$ per year (a 20% annual volatility). We step forward one trading day, $t = 1/252 \approx 0.003968$ years.

The deterministic drift over one day: $\mu t = \$8 \times (1/252) = \$0.0317$. Tiny -- about three cents of expected gain per day, because the annual edge is spread thinly across 252 days.

The volatility scaling over one day: $\sigma\sqrt{t} = \$20 \times \sqrt{1/252} = \$20 \times 0.0630 = \$1.26$. So one standard deviation of the day's noise is about \$1.26 -- already forty times larger than the drift. To take one concrete step, draw a single standard-normal random number; say the draw is $Z = +0.5$ (a half-standard-deviation up day). Then the Brownian increment is $\sigma\sqrt{t}\,Z = \$1.26 \times 0.5 = \$0.63$.

Putting it together, the new price is $X_t = \$100 + \$0.0317 + \$0.63 = \$100.66$. The day's P&L on one share is about $+\$0.66$, of which only about three cents was the "edge" and 63 cents was pure noise. On a 10,000-share position that is a \$6,600 gain, of which roughly \$317 is expected and the rest is luck. The intuition: over a single day the random noise utterly swamps the drift -- by more than an order of magnitude -- which is the precise reason you cannot tell a good trader from a lucky one over a day, a week, or even a quarter; you need enough time for the linearly-growing drift to climb out of the square-root-growing noise.

### The fatal flaw: prices can go negative

Arithmetic Brownian motion is a clean first model, but it has a defect that makes it unusable for actual stock prices: **it can go negative.** Because $X_t \sim N(X_0 + \mu t,\ \sigma^2 t)$ is a normal distribution, and the normal distribution stretches from $-\infty$ to $+\infty$, there is always a positive probability that $X_t$ lands below zero. A normal random variable can take any real value, including negative ones. But a stock cannot be worth less than \$0 -- a share of a company that goes bankrupt is worth zero, not minus fifty dollars; you cannot owe money for *owning* a share. Arithmetic Brownian motion does not know this, and over a long enough horizon or with a high enough volatility, it will happily forecast negative prices, which is nonsense.

There is a second, subtler flaw: arithmetic Brownian motion adds the *same dollar volatility* regardless of the price level. A \$1 noise on a \$100 stock (1%) is treated identically to a \$1 noise on a \$5 stock (20%), which does not match how real prices behave -- a \$500 stock moves in larger dollar increments than a \$5 stock, roughly in proportion to its price. Real prices move in *percentage* terms, not dollar terms.

Both flaws are fixed by the same move: instead of modeling the *price* as Brownian, model the *logarithm of the price* as arithmetic Brownian motion. Exponentiating a normal gives a *lognormal* distribution, which lives strictly above zero (the exponential of any real number is positive) and which moves in proportional terms. That model is *geometric Brownian motion* -- the engine of the Black-Scholes formula -- and it is exactly where the next steps in this series go: see [stochastic differential equations: GBM, OU, and CIR](/blog/trading/quantitative-finance/stochastic-differential-equations-gbm-ou-quant-interviews) for the full treatment, and [the Brownian motion interview deep-dive](/blog/trading/quantitative-finance/brownian-motion-quant-interviews) for how these properties show up in trading interviews. Arithmetic Brownian motion is not useless, though -- it is a perfectly good model for things that *can* go negative and move in absolute terms, such as the *spread* between two prices in a pairs trade, or a portfolio's P&L, or interest-rate differences. Use the right tool: arithmetic Brownian for additive quantities, geometric Brownian for prices.

![A tree of Brownian motion concepts rooted in the scaling limit and branching into its distribution, path behavior, and quant uses.](/imgs/blogs/brownian-motion-random-walk-math-for-quants-7.png)

The tree above is the concept map for the whole post. Everything descends from one root -- Brownian motion as the scaling limit of a random walk. From it branch the three families we have built: its *distribution* ($N(0,t)$, which gives the $\sqrt{t}$ law), its *path behavior* (rough, non-differentiable, with quadratic variation equal to $t$), and its *quant uses* (price models, starting with arithmetic Brownian motion and pointing toward geometric Brownian motion). If you can reconstruct this tree from memory, you understand Brownian motion.

## Common misconceptions

**"Brownian motion means prices are random and unbeatable."** Not quite. The *model* says the noise is unpredictable, but it explicitly includes a *drift* term $\mu$ for any predictable trend, and real strategies live in estimating that drift -- a risk premium, a factor exposure, a genuine alpha. The claim is narrower and more useful: the *noise* part has independent increments, so the *path of the noise* carries no exploitable pattern. Edge comes from the drift, from volatility itself, or from places where the Brownian assumptions fail (jumps, fat tails, autocorrelation). "It's a random walk" is a statement about the noise, not a verdict that nothing can be predicted.

**"Variance and standard deviation scale the same way with time."** This is the single most expensive beginner error in risk. *Variance* grows linearly with time ($\mathrm{Var} = \sigma^2 t$); *standard deviation* -- the thing you feel as a dollar move -- grows with the square root of time ($\sigma\sqrt{t}$). They are not interchangeable. If you scale a daily standard-deviation risk number to a year by multiplying by 252 instead of $\sqrt{252}$, you overstate the risk roughly sixteenfold. Variance adds; standard deviation does not.

**"A Brownian path is smooth if you just look closely enough."** The opposite is true. A Brownian path is *self-similar*: zoom into any tiny piece of it and, after rescaling, it looks statistically identical to the whole -- equally jagged at every magnification. It never resolves into a smooth curve, which is exactly why it has no derivative and infinite length. The discrete random walk *is* smooth between ticks (it is a straight line there), but its continuous limit is jagged all the way down.

**"Quadratic variation is just another name for variance."** They are cousins but not the same. *Variance* is the expected squared deviation -- a number describing the *distribution* of $W_t$, computed by averaging over many hypothetical paths. *Quadratic variation* $[W]_t = t$ is computed along a *single realized path* by summing its squared increments, and it equals $t$ with certainty, no averaging required. Two coins of the realized-risk concept: one (variance) is what you expect before the dice roll; the other (quadratic variation) is what you measure after, and remarkably it is the same value, $t$, for *every* path. That determinism is what makes variance trading possible.

**"Adding a drift makes the price trend, so I'll see it."** Over realistic horizons, no. The drift grows like $t$ and the noise like $\sqrt{t}$, but for any reasonable annual edge the noise dominates for a long time. A stock with an 8% drift and 20% volatility has a *signal-to-noise ratio* over one year of about $0.08 / 0.20 = 0.4$ -- the year's expected return is less than half the year's typical swing. You need many years (the *Sharpe ratio* times the square root of the number of years must get large) before the trend reliably pokes above the noise. This is why backtests need so much data, a point developed in [the law of large numbers and CLT post](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants).

## How it shows up in real markets

**Option pricing and Black-Scholes.** The most consequential use of Brownian motion in all of finance is the Black-Scholes-Merton option pricing model (1973), which assumes the *log* of the stock price follows Brownian motion with drift -- geometric Brownian motion. The model's central trick, the *Itô correction term*, exists entirely because of the quadratic variation fact $[W]_t = t$ we proved above: the squared increments do not vanish, so a second-order term survives into the price differential. When a trader quotes an option in terms of its *implied volatility* -- the $\sigma$ that makes the Black-Scholes price match the market -- they are reading off the market's estimate of the Brownian motion's diffusion rate. The entire \$10-trillion-plus listed options market runs on a formula whose engine is the random walk we started this post with.

**Value-at-risk and the square-root-of-time rule.** Regulators and risk desks routinely take a short-horizon risk estimate and scale it to a longer one using exactly the $\sqrt{t}$ law. The Basel market-risk framework historically scaled a 1-day value-at-risk to a 10-day figure by multiplying by $\sqrt{10} \approx 3.16$. This is a direct application of "standard deviation grows as $\sqrt{t}$." Its weakness is also a real-market lesson: the rule assumes independent, identically distributed daily moves, and in a crisis -- October 2008, March 2020 -- moves cluster and autocorrelate, so the true 10-day risk exceeds the $\sqrt{10}$-scaled figure. Risk models that took the $\sqrt{t}$ rule too literally underestimated drawdowns in exactly the months they mattered most.

**Volatility trading and variance swaps.** A *variance swap* is a contract that pays the difference between the *realized* variance of a stock over a period and a fixed strike. The reason such a contract can be priced and hedged at all is the quadratic-variation result: the realized variance is the sum of squared returns, which over a Brownian path converges to a definite quantity ($\sigma^2 t$) regardless of the path's direction. A volatility seller does not need to predict whether the market rises or falls; they need only a view on how much it will *wiggle*, because the squared increments accumulate deterministically. The worked example earlier -- net P&L random, accumulated squared variation locked -- is precisely the trade.

**Pairs trading and the spread.** When two related stocks (say two oil majors) are traded against each other, the quantity of interest is the *spread* -- the difference in their prices -- which *can* go negative and moves in absolute dollar terms. That makes arithmetic Brownian motion (not geometric) the natural baseline, often augmented with mean reversion (the Ornstein-Uhlenbeck process). The pairs trader is betting that the spread's drift pulls it back toward a mean faster than its Brownian noise pushes it away. The math of when the signal beats the noise is exactly the drift-versus-$\sqrt{t}$ tension from this post; see [stochastic differential equations: GBM, OU, and CIR](/blog/trading/quantitative-finance/stochastic-differential-equations-gbm-ou-quant-interviews).

**The 2020 volatility spike.** In March 2020, equity volatility exploded: the VIX index, a market estimate of one-month annualized volatility, jumped from around 15% to over 80% in weeks. Reading that through the $\sqrt{t}$ law: an 80% *annual* volatility implies a daily volatility of $80\% / \sqrt{252} \approx 5\%$, meaning the market was pricing *typical* daily swings of 5% -- and indeed the S&P 500 had several days that moved 7-12%. The square-root-of-time conversion is how a single quoted annual number translates into the daily moves a desk actually lives through, and it is why a spike in the VIX is an immediate, visceral signal of larger daily P&L swings to come.

**High-frequency and the limits of the model.** At the millisecond scale, prices are emphatically *not* Brownian -- they live on a discrete grid (the *tick size*), they have strong short-term mean reversion from order-book mechanics, and the "increments" are anything but independent. The Brownian model is an *idealization that emerges only when you zoom out* far enough that the discreteness and microstructure wash away, exactly as Donsker's theorem promises. A high-frequency desk that assumed Brownian motion at the tick level would mis-hedge constantly; the model is right at the scale of minutes-to-days and wrong at the scale of microseconds. Knowing *which scale* a model belongs to is half of using it well.

## When this matters to you / further reading

If you ever hold a portfolio overnight, the square-root-of-time law already touches your life: the reason a diversified stock fund can swing 1% on a quiet day but rarely 16% in a single session, yet routinely 16% over a year, is the $\sqrt{t}$ scaling of Brownian noise. The reason your retirement account's *expected* growth (the drift) is reliable over decades but invisible over weeks is the linear-signal-versus-square-root-noise tension. And the reason a single bad month does not, by itself, mean a strategy is broken is that one month of noise is large relative to one month of signal -- you cannot diagnose an edge over a horizon where the $\sqrt{t}$ fog dominates.

For the next mathematical steps, three directions build directly on this post. First, the calculus that the quadratic-variation fact forces into existence: the Itô integral and Itô's lemma, the subject of the next post in this series, where the surviving $\tfrac12\sigma^2$ term becomes the heart of every pricing formula. Second, the fix for the negative-price flaw: geometric Brownian motion and the family of stochastic differential equations in [SDEs: GBM, OU, and CIR](/blog/trading/quantitative-finance/stochastic-differential-equations-gbm-ou-quant-interviews). Third, for the interview-style mechanics and the classic puzzles built on Brownian motion -- hitting times, reflection, the maximum of a path -- see [the Brownian motion quant-interview deep-dive](/blog/trading/quantitative-finance/brownian-motion-quant-interviews). To shore up the probability underneath all of it, [the distributions markets actually use](/blog/trading/math-for-quants/probability-distributions-for-markets-math-for-quants) covers the normal and lognormal in the depth this post assumed, and [the law of large numbers and central limit theorem](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants) is the discrete engine that makes Brownian motion universal.

The one thing to carry away: a price is, to first order, a coin-flip walk shrunk until the steps are invisible, and that single model explains why risk grows with $\sqrt{t}$, why prices have no velocity, why their squared moves accumulate to a clock, and why every continuous-time pricing model on Earth begins with the letter $W$. Everything else in stochastic calculus is commentary on that one idea. This is educational material about how the models behave, not advice about what to trade -- but understanding the model is the first and most durable edge there is.
