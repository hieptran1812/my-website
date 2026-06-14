---
title: "Stochastic differential equations for quant interviews: geometric Brownian motion and Ornstein-Uhlenbeck"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch, interview-focused deep dive on the two stochastic differential equations every quant must know cold: geometric Brownian motion for prices and Ornstein-Uhlenbeck for mean-reverting rates and spreads, with closed-form solutions, moments, half-life, simulation, and a full set of solved interview problems."
tags:
  [
    "stochastic-differential-equations",
    "geometric-brownian-motion",
    "ornstein-uhlenbeck",
    "quant-interviews",
    "stochastic-calculus",
    "itos-lemma",
    "mean-reversion",
    "monte-carlo",
    "derivatives-pricing",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — A stochastic differential equation (SDE) describes how a price or a rate wiggles forward in time as a smooth *drift* plus a random *shake*, and the two SDEs you must know cold for a quant interview are geometric Brownian motion (GBM) for stock prices and the Ornstein-Uhlenbeck (OU) process for mean-reverting things like interest rates and pair spreads.
>
> - An SDE is written `dX = (drift) dt + (diffusion) dB`: the drift is where the path trends on average, the diffusion is the size of the random kick from Brownian motion `B`.
> - **GBM** `dS = mu S dt + sigma S dB` solves to `S_t = S_0 exp((mu - half sigma^2) t + sigma B_t)`. Prices stay positive, grow in percent, and end up *lognormal*. The expected price grows at `mu`, but the typical (median) path grows at the smaller rate `mu - half sigma^2`.
> - **OU** `dX = theta(m - X) dt + sigma dB` pulls the path back toward a long-run mean `m` with strength `theta`. A shock to the mean decays with **half-life `ln 2 / theta`**, and over long horizons `X` settles into a Normal centred at `m` with variance `sigma^2 / (2 theta)`.
> - The single most-tested trap: the average GBM return is `mu`, but the rate that actually compounds your wealth is `mu - half sigma^2`. Volatility is a tax on compounding.
> - Use GBM when the quantity compounds and can never go negative (stocks, FX, index levels). Use OU when it gets tugged back to a level (short rates, basis, pair spreads, volatility).

Here is a question a Two Sigma interviewer actually likes to open with: a stock trades at \$100, drifts up at 8% a year, and has 20% annual volatility. What is its *expected* price in one year — and is that the price you should bet on? Most candidates say \$108 and stop. The expected price is indeed about \$108.33. But the *most likely* price — the one the median path lands on — is only \$106.18. The gap between those two numbers, and why it exists, is the whole subject of this article. It comes straight out of a single equation that governs how prices move.

That equation is a **stochastic differential equation**, or SDE. If an ordinary differential equation tells you how a smooth quantity changes ("the population grows 3% a year"), an SDE tells you how a *random* quantity changes ("the stock drifts up 8% a year, give or take a random shock whose size is 20% a year"). Every model on a derivatives desk — Black-Scholes, the volatility surface, every short-rate model, every pairs-trading signal — is an SDE underneath. You do not need to be a measure theorist to use them. You need to know two SDEs deeply, know their closed-form solutions, and be able to compute their moments under interview pressure. This piece builds both from zero.

![SDE anatomy: a tiny change in the price equals a deterministic drift term plus a random diffusion term scaled by Brownian noise.](/imgs/blogs/stochastic-differential-equations-gbm-ou-quant-interviews-1.png)

The diagram above is the mental model. Every SDE we will meet has exactly this shape: the change in the thing we care about (left, in blue) is a **drift** term that pulls it smoothly in one direction (top, in green) plus a **diffusion** term that shakes it randomly each instant (bottom, in amber). Learn to read those two pieces off any SDE and you are most of the way there. Let us build up the vocabulary before we touch a single Greek letter in anger.

## Foundations: what an SDE actually says

We are going to assume you have seen `e^x`, a square root, and the idea of a bell-shaped Normal distribution, and nothing more. Everything else gets defined here.

### Brownian motion: the source of all the randomness

Start with the raw random ingredient. **Brownian motion** (also called a *Wiener process*, written `B_t` or `W_t`) is the mathematical idealization of a particle being jostled randomly — or a price being pushed around by a stream of tiny independent buy and sell orders. It has four properties you should memorize, because interviewers test each one:

1. **It starts at zero**: `B_0 = 0`.
2. **Independent increments**: the move from time `s` to time `t` does not depend on anything that happened before `s`. The market has no memory in this idealization.
3. **Gaussian increments**: the move over an interval of length `h` is Normal with mean 0 and variance `h`. In symbols, `B_{t+h} - B_t ~ Normal(0, h)`. The key fact buried here: variance grows *linearly* in time, so the standard deviation — the typical size of the move — grows like the *square root* of time. Over four times as long, the random move is only twice as big.
4. **Continuous paths**: it never jumps. The price wiggles, but it does not teleport.

That square-root-of-time scaling is the single most important fact about randomness in finance. A *basis point* is one hundredth of a percent (0.01%); when a desk says daily vol is 1% and annualizes it by multiplying by the square root of 252 trading days to get about 16%, that `sqrt(252)` is exactly property 3 in action. We will lean on it constantly.

The infinitesimal version of property 3 is written `dB`, read "a tiny increment of Brownian motion." It is a random number with mean 0 and variance `dt` (the tiny time step). The defining oddity — the thing that makes stochastic calculus different from ordinary calculus — is that `dB` is of size `sqrt(dt)`, not `dt`. When you square it, `(dB)^2` behaves like `dt`, not like the negligibly small `(dt)^2`. Hold onto that; it is the seed of Itô's lemma.

### Drift and diffusion: the two terms of every SDE

Now assemble the SDE. The simplest one, **arithmetic Brownian motion**, says the change in `X` over a tiny time step `dt` is

$$dX = \mu \, dt + \sigma \, dB$$

Read it left to right. `dX` is the change in `X`. The first term, `mu dt`, is the **drift**: a smooth, predictable push of size `mu` per unit time. If you turned off all the randomness (`sigma = 0`), `X` would just move in a straight line at slope `mu`. The second term, `sigma dB`, is the **diffusion**: a random kick whose size is controlled by `sigma`, the *volatility*. Bigger `sigma` means a wilder, noisier path.

- `mu` (the drift coefficient) has units of *X-units per unit time* — dollars per year, say. It is where the path goes on average.
- `sigma` (the diffusion coefficient, or volatility) controls the *spread* around that average. Its units are *X-units per square-root-of-time*, because of the `sqrt(dt)` scaling.

That is the entire grammar. An SDE is a drift term plus a diffusion term. Once you can stare at `dX = (something) dt + (something else) dB` and name the two pieces, you can read any model on the desk. The "something" multiplying `dt` and `dB` can depend on `X` and on `t` — and *how* they depend is what separates GBM from OU from everything else.

### Why we cannot just integrate like in calculus

In ordinary calculus, `dX = mu dt` integrates to `X_t = X_0 + mu t`. You might hope `dX = mu dt + sigma dB` integrates to `X_t = X_0 + mu t + sigma B_t`. For *arithmetic* Brownian motion that actually works, because the coefficients are constants. The complication arrives the moment `sigma` multiplies `X` itself, as it does for a stock — and it forces us to use a special chain rule called Itô's lemma. We will meet it in the GBM section. First, let us see why arithmetic Brownian motion is the wrong model for a stock at all.

A stock at \$100 with `dX = mu dt + sigma dB` and `sigma = \$20` per year could, over a year, take a random draw that pushes it *below zero*. A price of negative \$15 is nonsense. Worse, a \$20 move means something completely different on a \$10 stock than on a \$1,000 stock — but arithmetic Brownian motion treats them identically. Real prices move in *percentages*, not dollars: a stock is roughly as likely to go up 10% from \$100 as from \$1,000. That observation is exactly what geometric Brownian motion fixes.

## Geometric Brownian motion: the model for prices

The fix is to make the drift and diffusion *proportional to the price itself*. That is geometric Brownian motion:

$$dS = \mu S \, dt + \sigma S \, dB$$

Here `S` is the stock price, `mu` is the **expected return** (drift) expressed as a fraction per year — a percentage, not a dollar amount — and `sigma` is the **volatility**, also a fraction per year. Divide both sides by `S` and the meaning jumps out:

$$\frac{dS}{S} = \mu \, dt + \sigma \, dB$$

The *fractional* (percentage) change in the price is an arithmetic Brownian motion. The return drifts at `mu` per year with random kicks of size `sigma`. This is the right shape for a price: a \$100 stock and a \$1,000 stock with the same `mu` and `sigma` have the same *return* dynamics, and because the change is always a percentage of a positive number, the price can never cross zero. It can get arbitrarily small, but it stays positive — exactly like a real stock that can fall 99% but not to negative dollars.

![GBM sample paths fan out and drift up while their spread widens, producing a right-skewed lognormal terminal price cloud.](/imgs/blogs/stochastic-differential-equations-gbm-ou-quant-interviews-2.png)

The figure shows what GBM looks like when you simulate it many times from \$100. Each thin colored line is one possible future. They all start together, drift up along the dashed median line, and *fan out* — the spread grows with time because uncertainty compounds. Crucially, the fan is **asymmetric**: paths can double or triple to the upside, but the worst they can do is drift toward zero. That asymmetry is why the histogram of year-end prices on the right is *skewed to the right* — a long tail of high outcomes. This skewed distribution has a name: the **lognormal** distribution, meaning the *logarithm* of the price is Normal. Let us prove that.

### Solving GBM with Itô's lemma

We want a closed-form expression for `S_t`, not just the differential `dS`. The trick is to work with the *log* of the price, `Y = ln S`, because the log turns the multiplicative dynamics into additive ones. But we cannot use the ordinary chain rule. In ordinary calculus, if `Y = ln S` then `dY = dS / S`. In stochastic calculus there is a correction term, because `(dS)^2` is not negligible — remember `(dB)^2` behaves like `dt`.

**Itô's lemma** is the stochastic chain rule. For a function `f(S)` of a process driven by `dS = a \, dt + b \, dB`, it says

$$df = \left( f' a + \tfrac{1}{2} f'' b^2 \right) dt + f' b \, dB$$

The first two terms are the drift of `f`; the `½ f'' b²` piece is the **Itô correction** — the new thing that ordinary calculus does not have. It comes entirely from the `(dB)^2 = dt` fact. Apply it to `f(S) = ln S`, where `f' = 1/S` and `f'' = -1/S^2`, with `a = mu S` and `b = sigma S`:

$$d(\ln S) = \left( \frac{1}{S}\,\mu S + \tfrac{1}{2}\left(-\frac{1}{S^2}\right)\sigma^2 S^2 \right) dt + \frac{1}{S}\,\sigma S \, dB = \left( \mu - \tfrac{1}{2}\sigma^2 \right) dt + \sigma \, dB$$

Look what happened: the right-hand side is now an *arithmetic* Brownian motion with constant coefficients. The drift of the log price is `mu - ½ sigma²`, not `mu`. That subtracted `½ sigma²` is the Itô correction made concrete, and it is the most important quantity in this whole article. Because the coefficients are now constant, we can integrate from 0 to `t` exactly as in ordinary calculus:

$$\ln S_t - \ln S_0 = \left( \mu - \tfrac{1}{2}\sigma^2 \right) t + \sigma B_t$$

Exponentiate both sides and you have the closed-form solution:

$$\boxed{\,S_t = S_0 \exp\!\left( \left( \mu - \tfrac{1}{2}\sigma^2 \right) t + \sigma B_t \right)\,}$$

![Solving GBM with Ito's lemma: take the log, apply the lemma to get a constant-coefficient drift minus half the variance, integrate, and exponentiate.](/imgs/blogs/stochastic-differential-equations-gbm-ou-quant-interviews-3.png)

The pipeline above is the whole derivation in five steps, and it is worth being able to reproduce it on a whiteboard from memory: start with the multiplicative SDE, substitute `Y = ln S`, apply Itô to pick up the `- ½ sigma²`, integrate the now-constant drift, and exponentiate. Interviewers ask for this derivation directly, and they ask the trickier follow-up: *why* is there a `- ½ sigma²`? The honest one-line answer is that volatility drags down the compounded growth rate even though it does not change the average — and the next section makes that precise with numbers.

### The mean, variance, and lognormality of GBM

From the boxed solution we can read off everything. Since `B_t ~ Normal(0, t)`, the exponent `(mu - ½ sigma²) t + sigma B_t` is Normal with mean `(mu - ½ sigma²) t` and variance `sigma² t`. So `S_t` is `S_0` times the exponential of a Normal — that is the *definition* of a lognormal random variable. The standard moment formulas for a lognormal then give:

$$\mathbb{E}[S_t] = S_0 \, e^{\mu t}, \qquad \operatorname{Var}[S_t] = S_0^2 \, e^{2\mu t}\left( e^{\sigma^2 t} - 1 \right)$$

Notice the beautiful cancellation in the mean: the `- ½ sigma²` in the exponent of the solution is *exactly* undone when you take the expectation, because `E[e^{sigma B_t}] = e^{½ sigma² t}`. So the **expected price grows at the full drift `mu`**, even though every individual log-path only drifts at `mu - ½ sigma²`. The median, by contrast, follows the typical path: `median(S_t) = S_0 e^{(mu - ½ sigma²) t}`, which is *below* the mean. The mean is pulled up by the fat right tail — those few paths that triple drag the average above the median. This mean-versus-median gap is the lognormal skew, and it is the single most fertile source of GBM interview questions.

#### Worked example: the mean and variance of a \$100 GBM stock

You hold a stock at `S_0 = \$100` with expected return `mu = 8%` per year, volatility `sigma = 20%` per year, over horizon `T = 1` year. Compute the expected price, the variance, the standard deviation, and the median.

Expected price: `E[S_T] = 100 * e^{0.08 * 1} = 100 * 1.0833 = \$108.33`.

Variance: `Var[S_T] = 100² * e^{2 * 0.08} * (e^{0.20² * 1} - 1) = 10000 * e^{0.16} * (e^{0.04} - 1)`. Now `e^{0.16} = 1.1735` and `e^{0.04} - 1 = 0.0408`, so `Var = 10000 * 1.1735 * 0.0408 = \$479` (in squared dollars).

Standard deviation: `sqrt(479) = \$21.9`. So a one-standard-deviation band around the stock is roughly `\$108 ± \$22`.

Median: `100 * e^{(0.08 - 0.5 * 0.04) * 1} = 100 * e^{0.06} = \$106.18`.

![The moments of a $100 GBM stock at one year: the expected price grows at the full drift while the median grows at the smaller drift minus half the variance.](/imgs/blogs/stochastic-differential-equations-gbm-ou-quant-interviews-9.png)

There it is, the answer to the opening interview question, laid out in the figure: the mean is \$108.33, the median only \$106.18, and they differ by \$2.15 purely because of volatility. The intuition this teaches: **the average outcome is not the typical outcome.** When someone quotes you an "expected return," ask whether they mean the average (driven by `mu`) or the rate your money actually compounds at (driven by `mu - ½ sigma²`).

#### Worked example: a 95% range for next year's price

Same stock (`S_0 = \$100`, `mu = 8%`, `sigma = 20%`, `T = 1`). Where will the price be with 95% probability? Because `ln S_T` is Normal with mean `ln(100) + (0.08 - 0.02) = ln(100) + 0.06` and standard deviation `sigma sqrt(T) = 0.20`, a 95% interval for `ln S_T` is the mean plus or minus `1.96 * 0.20 = 0.392`. Exponentiate the endpoints:

Lower: `100 * e^{0.06 - 0.392} = 100 * e^{-0.332} = \$71.75`.
Upper: `100 * e^{0.06 + 0.392} = 100 * e^{0.452} = \$157.15`.

So with 95% confidence the stock lands in `[\$71.75, \$157.15]` after one year. Notice the asymmetry around the \$106 median: the upside (`+\$51`) is bigger than the downside (`-\$34`). That asymmetry is the lognormal skew showing up directly in a confidence interval, and it is why naive symmetric "value at risk" estimates that pretend prices are Normal understate the upside and overstate the crash probability.

### A quick simulation to make it concrete

If you ever doubt a moment formula in an interview, you can describe the brute-force check. Here is GBM simulated directly from its exact solution in `numpy`:

```python
import numpy as np

    # exact GBM terminal samples (no time-stepping needed)
rng = np.random.default_rng(0)
S0, mu, sigma, T, n = 100.0, 0.08, 0.20, 1.0, 2_000_000
Z = rng.standard_normal(n)                     # one Normal per path
ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

print(ST.mean())     # ~108.33  -> S0 * exp(mu*T)
print(ST.var())      # ~479     -> S0^2 e^{2 mu T}(e^{sigma^2 T} - 1)
print(np.median(ST)) # ~106.18  -> S0 * exp((mu - 0.5 sigma^2) T)
```

The point of showing this is not the code; it is that GBM has a *closed form*, so you can sample the terminal price with one Normal draw per path and never simulate the intermediate wiggles. That is what makes Monte Carlo option pricing under Black-Scholes so cheap.

### A deeper look at the lognormal moments

It is worth understanding where the GBM moment formulas come from, because the derivation is itself a common interview ask and the trick generalizes. The whole thing rests on one fact about the Normal distribution: if `Y ~ Normal(a, b²)`, then the expectation of `e^Y` is `e^{a + ½ b²}`. This is the **moment generating function** of the Normal, and it is the single formula that powers every lognormal computation. Memorize it.

Apply it to `S_t = S_0 e^Y` where `Y = (mu - ½ sigma²) t + sigma B_t` has mean `a = (mu - ½ sigma²) t` and variance `b² = sigma² t`. Then

$$\mathbb{E}[S_t] = S_0\,\mathbb{E}[e^Y] = S_0\,e^{a + \frac{1}{2}b^2} = S_0\,e^{(\mu - \frac{1}{2}\sigma^2)t + \frac{1}{2}\sigma^2 t} = S_0\,e^{\mu t}$$

Watch the cancellation in the exponent: the `- ½ sigma² t` from the solution and the `+ ½ sigma² t` from the moment generating function annihilate, leaving the clean `mu t`. That is *why* the expected price grows at the full drift even though the typical path does not — the upside skew of the lognormal exactly compensates for the volatility drag, in expectation. For the variance, you compute `E[S_t²]` the same way (it uses `2 sigma B_t`, so it picks up `e^{2 sigma² t}`) and subtract the squared mean:

$$\mathbb{E}[S_t^2] = S_0^2\,e^{(2\mu - \sigma^2)t + 2\sigma^2 t} = S_0^2\,e^{2\mu t + \sigma^2 t}, \qquad \operatorname{Var}[S_t] = S_0^2 e^{2\mu t}\!\left(e^{\sigma^2 t} - 1\right)$$

The factor `(e^{sigma² t} - 1)` is the entire contribution of volatility to the variance; when `sigma = 0` it is zero and the price is deterministic. For small `sigma² t`, that factor is approximately `sigma² t` (the first term of the exponential series), so the standard deviation of the price is roughly `S_0 e^{mu t} sigma sqrt(t)` — proportional to volatility and to the square root of time, exactly the scaling we keep meeting. This approximation is how a trader does a quick mental risk estimate without a calculator: "vol is 20%, horizon is a year, so the one-sigma move on a \$108 expected price is about `108 * 0.20 = \$21.7`," which lands within a few cents of the exact \$21.9. The approximation is excellent whenever `sigma sqrt(t)` is small and degrades only for very high vol or very long horizons, where the lognormal skew makes the standard deviation a poor summary of the risk anyway.

## Ornstein-Uhlenbeck: the model for things that revert

GBM is perfect for a stock, because a stock has no "home" — there is no level it gets pulled back toward. An interest rate is different. If the short rate is 8%, economic forces (central banks, the business cycle) tend to push it back down toward some normal level; if it is 1%, forces push it back up. The same is true of the *spread* between two closely-related assets in a pairs trade, or of volatility itself. These quantities **mean-revert**: they wander, but they get tugged home. GBM cannot capture that. The Ornstein-Uhlenbeck process can.

$$dX = \theta(m - X)\,dt + \sigma\,dB$$

Read the drift term `theta(m - X) dt` carefully, because it is the whole idea. `m` is the **long-run mean** — the home level. `theta` (theta, always positive) is the **speed of mean reversion**. When `X` is *above* `m`, the gap `(m - X)` is negative, so the drift is negative and pulls `X` *down*. When `X` is *below* `m`, the gap is positive, the drift is positive, and it pushes `X` *up*. The further `X` strays from home, the harder it is yanked back — the restoring force is proportional to the distance, exactly like a spring. The diffusion term `sigma dB` is the same random shaking as before, with constant `sigma` (it does not scale with `X`, unlike GBM).

![Ornstein-Uhlenbeck paths are tugged back toward the mean level by a restoring drift proportional to the gap, so excursions decay toward m equals fifty dollars.](/imgs/blogs/stochastic-differential-equations-gbm-ou-quant-interviews-4.png)

The figure shows four OU paths starting from different places — two above the \$50 mean, two below — all getting dragged into the green band around `m = \$50` and then jiggling around it. Unlike the GBM fan that spreads forever, the OU paths reach a *steady* spread: the mean reversion fights the diffusion to a draw, and the process settles into a stable band. That balance is what makes OU the workhorse for rates and spreads.

### Solving the OU SDE

The OU equation also has a closed-form solution, and the trick to find it is a classic: multiply through by an *integrating factor* `e^{theta t}` to make the left side a perfect derivative. Doing so (this is a standard exercise; you should be able to state the result even if you do not derive it under pressure) gives

$$X_t = m + (X_0 - m)\,e^{-\theta t} + \sigma \int_0^t e^{-\theta(t-s)}\,dB_s$$

Three pieces, each with a clean meaning. `m` is the home level. `(X_0 - m) e^{-theta t}` is the **deterministic decay of the starting gap**: whatever distance you began from the mean shrinks exponentially at rate `theta`. The integral term is the accumulated random noise, but weighted so that *recent* shocks count more than old ones (the `e^{-theta(t-s)}` factor down-weights shocks from the distant past). From this solution we read the conditional moments:

$$\mathbb{E}[X_t \mid X_0] = m + (X_0 - m)\,e^{-\theta t}, \qquad \operatorname{Var}[X_t \mid X_0] = \frac{\sigma^2}{2\theta}\left(1 - e^{-2\theta t}\right)$$

The expected value decays from `X_0` toward `m`; the variance grows from 0 and *saturates* at `sigma² / (2 theta)` rather than growing forever. That saturation is the mathematical signature of mean reversion, and it is the cleanest way to tell an OU process from a Brownian one on a chart: Brownian variance grows linearly without bound, OU variance flattens out.

### Half-life: how fast does it revert?

The number a trader actually quotes is not `theta` itself but the **half-life** of mean reversion: the time it takes for the expected gap to the mean to shrink by half. From the expected-value formula, the gap is `(X_0 - m) e^{-theta t}`; it halves when `e^{-theta t} = ½`, i.e. when

$$t_{1/2} = \frac{\ln 2}{\theta}$$

This is the single most-asked OU fact in interviews, because it converts an abstract `theta` into a tradeable time. A spread with `theta = 12` per year has a half-life of `ln 2 / 12 = 0.058` years, about 15 trading days — fast enough to trade actively. A spread with `theta = 0.5` per year has a half-life of `ln 2 / 0.5 = 1.39` years — too slow to be worth tying up capital, because you would wait years to collect.

![OU half-life: the expected gap to the mean decays exponentially, so the half-life equals the natural log of two over theta, here giving 0.35 years.](/imgs/blogs/stochastic-differential-equations-gbm-ou-quant-interviews-5.png)

The figure makes the exponential decay concrete: starting from a \$20 gap with `theta = 2` per year, the gap reaches \$10 after one half-life of `ln 2 / 2 = 0.35` years (the green dot), and \$5 after a second half-life (the amber dot). Each half-life chops the remaining gap in half, just like radioactive decay. The intuition this teaches: **half-life is the clock of a mean-reverting trade.** It tells you both how long to hold and, as we will see, how to size the expected profit.

#### Worked example: the half-life of an OU spread

A statistical-arbitrage desk fits an OU process to the price spread between two oil majors and estimates `theta = 2.0` per year. What is the half-life, and how much of a dislocation reverts in three months?

Half-life: `t_{1/2} = ln 2 / theta = 0.6931 / 2.0 = 0.3466 years`, about 4.2 months or roughly 87 trading days.

Fraction reverting in three months (`t = 0.25` years): the gap decays by the factor `e^{-theta t} = e^{-2.0 * 0.25} = e^{-0.5} = 0.6065`. So after three months `60.65%` of the original gap remains, meaning about `39.35%` has reverted. That is a useful sanity check: a quarter-year is a bit under one half-life, so a little under half the dislocation should be gone, and `39%` is indeed a little under `50%`. The intuition: a `theta` of 2 is a moderately fast reverter — quick enough to trade, slow enough that you will hold the position for weeks, not minutes.

### The stationary distribution

Run an OU process long enough and it *forgets where it started*: the `e^{-theta t}` decay wipes out the influence of `X_0`. What is left is a stable, unchanging distribution — the **stationary distribution**. Taking `t` to infinity in the conditional moments, the starting gap vanishes and the variance saturates, leaving

$$X_\infty \sim \operatorname{Normal}\!\left( m, \ \frac{\sigma^2}{2\theta} \right)$$

The long-run distribution is Normal — *not* lognormal like GBM — centred exactly at the mean `m`, with variance `sigma² / (2 theta)`. Read that variance: it grows with the noise `sigma²` and shrinks with the reversion speed `theta`. A faster reverter (bigger `theta`) holds the process tighter to its mean, so the stationary spread is narrower. This is the formula that tells a pairs trader how *wide* a normal dislocation is, which sets the entry threshold for the trade.

![The OU stationary distribution is a Normal bell centred on the mean m with standard deviation sigma over the square root of twice theta.](/imgs/blogs/stochastic-differential-equations-gbm-ou-quant-interviews-6.png)

The figure shows that long-run bell: symmetric (Normal, not skewed), centred at `m = \$50`, with a standard deviation of `sigma / sqrt(2 theta) = \$6` in this example. A spread sitting one standard deviation away — at \$44 or \$56 — is a routine, roughly-once-every-six-observations excursion, not a screaming opportunity. A spread three standard deviations away, at \$32 or \$68, is a genuine dislocation. The stationary standard deviation is the yardstick that turns "the spread looks wide" into "the spread is 2.5 sigma wide, which happens about 1% of the time."

### OU is just AR(1) in continuous time

Here is a connection that earns nods in interviews and unifies a lot of what you already know from statistics. If you sample an OU process at evenly-spaced times `dt` apart, the resulting sequence is *exactly* a first-order autoregressive process — the **AR(1)** model from time-series econometrics. Recall AR(1): `x_{n+1} = c + phi * x_n + epsilon_n`, where `phi` is the autoregressive coefficient (between -1 and 1 for stability) and `epsilon` is independent noise. Match it to the OU conditional mean `E[X_{t+dt} | X_t] = m + (X_t - m) e^{-theta dt}`, and you can read the dictionary straight off:

- The AR(1) persistence `phi = e^{-theta dt}`. Fast reversion (big `theta`) means small `phi` — each step forgets the last one quickly.
- The AR(1) intercept `c = m(1 - phi)`, so the unconditional mean is `c / (1 - phi) = m`, the OU long-run mean.
- The noise variance is the OU conditional variance `sigma² (1 - e^{-2 theta dt}) / (2 theta)`.

This matters in practice because it tells you *how to estimate OU parameters from data*: you do not need fancy continuous-time machinery, you just run an ordinary least-squares regression of `X_{t+dt}` on `X_t`. The regression slope is `phi`, from which `theta = -ln(phi) / dt`, and the half-life follows immediately as `ln 2 / theta`. A stat-arb researcher fitting a spread does exactly this — an OLS regression and three lines of arithmetic — to get the half-life and entry band. The deeper lesson the interviewer is checking: **continuous-time mean reversion and discrete-time autoregression are the same phenomenon viewed at two resolutions**, and a persistence coefficient `phi` near 1 is the same warning sign as a reversion speed `theta` near 0 — a process so slow to revert it is barely distinguishable from a random walk.

## Simulating SDEs with the Euler-Maruyama scheme

GBM and OU both have closed-form solutions, which is lucky. The moment you write down a slightly more realistic SDE — stochastic volatility, a rate model with a level-dependent diffusion — the closed form usually disappears, and you must *simulate*. The standard recipe is the **Euler-Maruyama scheme**, the stochastic cousin of the ordinary Euler method for differential equations.

The idea is to chop time into small steps of size `dt` and, at each step, add the drift over that step plus a random shock. For a general SDE `dX = a(X) dt + b(X) dB`, one step is

$$X_{t+dt} = X_t + a(X_t)\,dt + b(X_t)\,\sqrt{dt}\,Z, \qquad Z \sim \operatorname{Normal}(0,1)$$

The only subtlety — and it is the thing interviewers probe — is the `sqrt(dt)` on the diffusion term. The drift scales with `dt`, but the random kick scales with `sqrt(dt)`, because Brownian increments have *variance* `dt` and therefore *standard deviation* `sqrt(dt)`. Forgetting the square root is the single most common bug in a candidate's simulation code, and it produces results that look plausible but have the wrong volatility entirely.

![Euler-Maruyama marches an SDE forward in discrete steps, adding drift times dt plus a square-root-of-dt scaled Gaussian shock at each step.](/imgs/blogs/stochastic-differential-equations-gbm-ou-quant-interviews-7.png)

The timeline above shows one Euler-Maruyama step broken into its parts: start at a known `X_0`, add the deterministic drift `a(X) dt`, add the random shock `b(X) sqrt(dt) Z` with a fresh standard Normal `Z`, land at `X_1`, and repeat until you reach the horizon `T`. Each step needs exactly one new Normal draw. Here it is in `numpy` for the OU process:

```python
import numpy as np

def simulate_ou(theta, m, sigma, x0, T, n_steps, n_paths, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    x = np.full(n_paths, x0, dtype=float)
    for _ in range(n_steps):
        z = rng.standard_normal(n_paths)
            # drift pulls toward m; shock scales with sqrt(dt)
        x += theta * (m - x) * dt + sigma * np.sqrt(dt) * z
    return x

ending = simulate_ou(theta=2.0, m=50.0, sigma=12.0, x0=70.0,
                     T=5.0, n_steps=2000, n_paths=500_000)
print(ending.mean())  # ~50    -> reverts to m
print(ending.std())   # ~6     -> sigma / sqrt(2 theta) = 12/sqrt(4) = 6
```

The two printed checks are the stationary moments we derived: after a long run the mean is `m = 50` and the standard deviation is `sigma / sqrt(2 theta) = 12 / sqrt(4) = 6`. If your simulation disagrees with the closed form, your `dt` is too coarse or your `sqrt(dt)` is missing.

#### Worked example: one Euler-Maruyama step of a GBM stock

A stock is at `S = \$100` with `mu = 10%`, `sigma = 20%`, and you step forward by `dt = 0.01` years (about 2.5 trading days). Suppose the Normal draw comes out `Z = 0.5`. Compute the next price.

The GBM diffusion is `b(S) = sigma S` and the drift is `a(S) = mu S`. So:

Drift contribution: `mu * S * dt = 0.10 * 100 * 0.01 = \$0.10`.

Shock contribution: `sigma * S * sqrt(dt) * Z = 0.20 * 100 * sqrt(0.01) * 0.5 = 0.20 * 100 * 0.10 * 0.5 = \$1.00`.

Next price: `S_{new} = 100 + 0.10 + 1.00 = \$101.10`.

The intuition this teaches: notice how the random shock (\$1.00) dwarfs the drift (\$0.10) over a short step. Over tiny horizons, *volatility dominates* — the drift is invisible noise. It is only over long horizons that the steady drift accumulates enough to matter, because drift grows like `t` while the random spread grows only like `sqrt(t)`. This is why you cannot detect a stock's expected return from a few days of data, no matter how clever your statistics.

## Which model fits what: a decision map

You now have two SDEs. The interview skill is choosing the right one fast. The test is simple: does the quantity *compound and stay positive*, or does it *get pulled back to a level*?

![A decision map: prices that compound and stay positive want geometric Brownian motion, while rates and spreads that snap back to a level want Ornstein-Uhlenbeck.](/imgs/blogs/stochastic-differential-equations-gbm-ou-quant-interviews-8.png)

The map above routes the choice. If the thing compounds and can never go negative — a stock price, an FX rate, a stock index level — it wants **GBM**, because GBM's multiplicative structure keeps it positive and makes its growth proportional. If the thing gets tugged back toward a normal level — an interest rate, the spread in a pairs trade, the basis between a future and its underlying, an implied-volatility level — it wants **OU**, because OU has the restoring drift that GBM lacks. Here is the same logic as a comparison table, which is how you should be able to rattle it off:

| Quantity | Compounds? | Reverts? | Model | Why |
|---|---|---|---|---|
| Stock price, FX rate, index | Yes | No | GBM | Stays positive, grows in percent |
| Short interest rate | No | Yes | OU (Vasicek) | Pulled to a normal level by policy |
| Pair / stat-arb spread | No | Yes | OU | Cointegration forces it home |
| Future-vs-spot basis | No | Yes | OU | Arbitrage closes the gap |
| Implied volatility | No | Yes | OU-like | Vol clusters but reverts |

The boundary case worth knowing: when an OU process is applied to the *short rate*, it is called the **Vasicek model**, and it has one famous flaw — because the stationary distribution is Normal, it allows *negative* rates. For decades that was considered a bug; since 2014, with several central banks setting negative policy rates, it has occasionally looked like a feature. We will return to that in the real-desk section. For a deeper treatment of the rate-modeling family, see [short-rate models](/blog/trading/quantitative-finance/short-rate-models-vasicek-hull-white).

### Arithmetic versus geometric: do not confuse them

One more distinction that trips up candidates: *arithmetic* Brownian motion (`dX = mu dt + sigma dB`, constant coefficients) versus *geometric* Brownian motion (`dS = mu S dt + sigma S dB`, coefficients proportional to the level). They are genuinely different models with different terminal distributions, and choosing wrong gives nonsense.

![Arithmetic versus geometric Brownian motion compared on positivity, return units, and terminal distribution shape.](/imgs/blogs/stochastic-differential-equations-gbm-ou-quant-interviews-12.png)

The matrix lays out every difference. Arithmetic BM can go negative and grows in *dollars*; geometric BM stays positive and grows in *percent*. Arithmetic BM is Normal at the horizon; geometric BM is lognormal. Use arithmetic BM for small, mean-zero quantities where negativity is fine and the percentage framing is awkward — a spread that oscillates around zero, for instance, which is exactly why OU (an arithmetic-style process) suits spreads. Use geometric BM for prices. A surprising number of pricing bugs come from someone modeling a spread as lognormal (it can be negative, so the log blows up) or a price as arithmetic (it can go negative, which is nonsense).

## In the interview room

Now the part you came for. Below are six fully-solved problems in the style of quant-researcher and trading interviews at Jane Street, Two Sigma, Citadel, and DE Shaw. Work each one yourself before reading the solution.

#### Worked example: the missing half-variance

*"A stock follows GBM with `mu = 12%` and `sigma = 30%`. Your colleague says the expected continuously-compounded log return over a year is 12%. Is he right?"*

No. The expected log return is the drift of the *log* price, which Itô's lemma tells us is `mu - ½ sigma²`, not `mu`. Compute: `0.12 - 0.5 * 0.30² = 0.12 - 0.5 * 0.09 = 0.12 - 0.045 = 0.075`, or `7.5%`. The expected *log* return is 7.5%, even though the expected *simple* return `E[S_t]/S_0 - 1` grows at the full 12%. The colleague has confused the two. The gap is `½ sigma² = 4.5%` per year — a huge drag for a volatile stock. The lesson the interviewer is testing: **`mu` is the average return; `mu - ½ sigma²` is the rate your wealth actually compounds at.** With 30% vol, almost a third of the headline drift is eaten by the volatility drag. This is the same effect that makes leveraged ETFs decay over time, and it is the most important single idea in the GBM section.

#### Worked example: probability of finishing in the money

*"A stock at \$100 follows GBM with `mu = 10%` and `sigma = 20%`. What is the probability it finishes above \$120 in one year?"*

We need `P(S_T > 120)`. Take logs: `S_T > 120` exactly when `ln S_T > ln 120`. We know `ln S_T ~ Normal(ln 100 + (0.10 - 0.5 * 0.04) * 1, 0.20² * 1)`, i.e. mean `ln 100 + 0.08 = 4.6852` and standard deviation `0.20`. Standardize:

$$z = \frac{\ln 120 - (\ln 100 + 0.08)}{0.20} = \frac{4.7875 - 4.6852}{0.20} = \frac{0.1823 - 0.08}{0.20} = 0.5116$$

So `P(S_T > 120) = P(Normal > 0.5116) = 1 - Phi(0.5116) = Phi(-0.5116) = 0.3045`, about `30.5%`.

![The probability a $100 GBM stock finishes above a $120 strike is the shaded right tail of the lognormal terminal density, equal to 30.5 percent.](/imgs/blogs/stochastic-differential-equations-gbm-ou-quant-interviews-10.png)

The shaded green tail in the figure is exactly this probability: the chunk of the lognormal terminal density sitting to the right of the \$120 strike, which integrates to 30.5%. The interviewer's follow-up is almost always: *"Now do it under the risk-neutral measure for option pricing."* The answer is that you replace `mu` with the risk-free rate `r` and the quantity `Phi(z)` with the famous `N(d2)` from Black-Scholes — they are the *same calculation*, just with the drift swapped. That connection between a GBM tail probability and the Black-Scholes formula is the bridge interviewers love to test. The mechanics of that pricing formula live in [the Black-Scholes deep dive](/blog/trading/quantitative-finance/black-scholes).

#### Worked example: sizing a pair-spread trade

*"You trade a pair whose dollar spread is an OU process with mean `m = \$0`, reversion speed `theta = 2.5` per year, and `sigma = \$5` per year. Today the spread is `\$8` rich. If you put on the trade, how many dollars do you expect to make over the next year, and what is the half-life?"*

The expected spread at horizon `t` is `m + (X_0 - m) e^{-theta t} = 0 + 8 * e^{-2.5 t}`. Over one year, `t = 1`: `8 * e^{-2.5} = 8 * 0.0821 = \$0.66`. So you expect the spread to fall from \$8 to about \$0.66, an expected reversion of `8 - 0.66 = \$7.34` per unit. If you are short one unit of the spread (betting it narrows), your expected gain is about `\$7.34`. The half-life is `ln 2 / theta = 0.6931 / 2.5 = 0.277 years`, about 3.3 months — so half the \$8 dislocation should be gone in roughly a quarter.

![Sizing the expected reversion of a mean-reverting pair spread: an eight-dollar dislocation is expected to close most of the gap over one year, an expected gain near seven dollars.](/imgs/blogs/stochastic-differential-equations-gbm-ou-quant-interviews-11.png)

The figure tracks the expected spread decaying from `+\$8` today (red dot, the dislocation) through `+\$4` at one half-life (green dot) down toward the \$0 mean, with the green arrow marking the roughly `\$7` you expect to harvest. The lesson: **the expected profit of a mean-reverting trade is the current gap times one minus the decay factor** — here `8 * (1 - e^{-2.5}) = \$7.34`. The follow-up an interviewer adds is risk: the stationary standard deviation is `sigma / sqrt(2 theta) = 5 / sqrt(5) = \$2.24`, so an \$8 dislocation is `8 / 2.24 = 3.6` standard deviations wide — a genuine, rare opportunity, not noise. This connects directly to the [Kelly criterion](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) for sizing the bet given that edge and variance.

#### Worked example: the variance ratio that distinguishes the models

*"You are handed two time series. One is GBM, one is OU, but the labels are lost. You can only compute variances of changes over different horizons. How do you tell them apart?"*

Use the saturation property. For GBM (in log space) and for arithmetic Brownian motion, the variance of the change over a horizon `h` grows *linearly*: `Var(change over 2h) = 2 * Var(change over h)`. For OU, the variance of the change *saturates*: over long horizons the mean reversion caps it, so `Var(change over 2h) < 2 * Var(change over h)`. Concretely, compute the **variance ratio** `VR(h) = Var(change over h) / (h * Var(change over 1))`. For a Brownian process `VR` stays flat at 1 for all `h`. For an OU process `VR` *declines* as `h` grows, because the variance stops growing while the denominator keeps rising. The series whose variance ratio falls toward zero at long horizons is the OU process. This is the foundation of the **variance-ratio test** used to detect mean reversion in real data, and the interviewer is checking whether you understand *why* OU variance saturates — because the reversion fights the diffusion to a standstill, as we saw in the stationary-distribution formula.

#### Worked example: expected first passage and the reflection idea

*"A stock follows GBM. Without the drift, what is the probability that its log price ever rises by `a` before falling by `a` over an infinite horizon?"*

Strip out the drift and work in log space, where GBM becomes a symmetric Brownian motion `ln S_t = ln S_0 + sigma B_t` (taking `mu - ½ sigma² = 0` for the symmetric case the interviewer intends). A symmetric Brownian motion started at 0 is equally likely to hit `+a` first or `-a` first, by symmetry — the answer is `½`. The point of the question is to see whether you reduce GBM to Brownian motion via the log transform, and whether you know that *symmetric* Brownian motion treats up and down identically. The natural extension — *with* drift — uses the same machinery as the gambler's ruin problem: the probability of hitting the upper barrier first becomes a ratio of exponentials in the drift, exactly the formula from [Markov chains and hitting times](/blog/trading/quantitative-finance/markov-chains-hitting-times-quant-interviews). Interviewers chain these together: reduce GBM to Brownian motion, then reduce barrier-hitting to gambler's ruin.

#### Worked example: the Itô isometry sanity check

*"For arithmetic Brownian motion `dX = sigma dB` with `sigma = 0.4`, what is `Var(X_t)` at `t = 9`, and how does that compare to the standard deviation?"*

Because the drift is zero and `sigma` is constant, `X_t = X_0 + sigma B_t`, and `Var(X_t) = sigma² Var(B_t) = sigma² t` (using `Var(B_t) = t`). So `Var(X_9) = 0.4² * 9 = 0.16 * 9 = 1.44`, and the standard deviation is `sqrt(1.44) = 1.2`. The lesson the interviewer is checking: variance scales with `sigma²` and with `t` *linearly*, but the standard deviation — the thing you feel as risk — scales with `sigma` and with `sqrt(t)`. Going from `t = 3` to `t = 9` triples the time and so triples the variance, but multiplies the typical move by only `sqrt(3) ≈ 1.73`. The general fact, `Var(integral of sigma dB) = integral of sigma² dt`, is the **Itô isometry**, and it is the engine behind every variance calculation in stochastic calculus.

#### Worked example: two stocks, same drift, different volatility

*"Two stocks both start at \$100 and both have expected return `mu = 10%`. Stock A has `sigma = 10%`, stock B has `sigma = 40%`. After 10 years, which has the higher expected value, and which is more likely to have made you money?"*

This is the volatility-drag question dressed up over a long horizon, and it splits cleanly into two answers. The *expected* value depends only on `mu`: `E[S_10] = 100 * e^{0.10 * 10} = 100 * e^1 = \$271.83` for *both* stocks. They have identical expected values, because the mean grows at `mu` regardless of `sigma`. But the *median* — the outcome you are more likely to actually experience — grows at `mu - ½ sigma²`:

Stock A median: `100 * e^{(0.10 - 0.005) * 10} = 100 * e^{0.95} = \$258.57`.
Stock B median: `100 * e^{(0.10 - 0.08) * 10} = 100 * e^{0.20} = \$122.14`.

So while both stocks *average* \$271.83, the typical outcome for the calm stock A is \$259, but for the wild stock B it is only \$122 — barely above where it started. Stock B's enormous expected value is carried by a tiny number of explosive paths; the median investor in stock B does far worse than the median investor in stock A, despite identical expected returns. The lesson: **over long horizons, volatility silently transfers value from the median to the mean.** Two investments can have the same advertised expected return while one reliably builds wealth and the other mostly disappoints. This is the single most important fact for anyone choosing between a steady and a lottery-like strategy, and it is why the geometric (compounding) return, not the arithmetic mean, is the honest yardstick.

## Common misconceptions

These are the beliefs that get candidates rejected, each corrected with the *why*.

**"The expected return `mu` is the rate my money grows at."** No. Your money compounds at `mu - ½ sigma²`, the drift of the *log* price. The arithmetic mean of returns (`mu`) always exceeds the geometric mean (`mu - ½ sigma²`) by the volatility drag `½ sigma²`. A fund that returns +50% then -50% has an average return of 0% but has actually lost 25% of your money — that 25% loss is the volatility drag made painfully real. Confusing the two is the most common and most expensive error in the whole subject.

**"GBM lets the stock go negative if the random shock is big enough."** No, and this is the whole reason we use GBM instead of arithmetic Brownian motion. The solution `S_t = S_0 exp(...)` is an exponential, and an exponential of *any* real number is strictly positive. No matter how bad the Brownian draw, `S_t` stays above zero. It can become tiny, but it cannot cross into negative territory. *Arithmetic* Brownian motion, by contrast, genuinely can go negative — which is exactly why it is wrong for prices.

**"Mean reversion means the price always comes back, so it is a free lunch."** No. Mean reversion is a statement about the *expected* path, not a guarantee. The OU process still has a diffusion term, so any individual path can wander further from the mean before (or instead of) reverting, and it can stay dislocated for many half-lives. A trade that is "3 sigma cheap" can become "5 sigma cheap" and blow through your stop-loss before it reverts — that is exactly how the LTCM-style spread trades blew up. Mean reversion improves your *odds*; it does not remove the risk.

**"A bigger `theta` means a wider trading band."** Backwards. A bigger `theta` is a *faster* reverter, which holds the process *tighter* to its mean, so the stationary standard deviation `sigma / sqrt(2 theta)` is *smaller*. Faster reversion means a *narrower* band and a *shorter* half-life — both good for a trader, because you get in and out faster and the noise is smaller relative to the signal.

**"Itô's lemma is just the chain rule."** Almost, but the extra `½ f'' sigma²` term is not in the ordinary chain rule, and it is the whole point. That correction term is why `E[S_t]` grows at `mu` while the median grows at `mu - ½ sigma²`. Treating Itô's lemma as the plain chain rule will lose you the half-variance every time, and that is the term interviewers are checking you remember.

**"Volatility `sigma` and variance `sigma²` are interchangeable."** They are not, and mixing units is a classic blunder. Volatility has the units of the quantity per square-root-of-time; variance has units squared per unit time. You annualize *volatility* by `sqrt(252)`, but you annualize *variance* by `252`. Quote a vol when you mean a typical move, a variance when you are adding independent risks (variances add, volatilities do not).

## How it shows up on a real desk

The two SDEs are not academic. Here is where a working quant meets them.

**Vasicek and the rates desk.** The Vasicek model *is* the OU process applied to the short interest rate: `dr = theta(m - r) dt + sigma dB`. A rates desk calibrates `theta`, `m`, and `sigma` to the current yield curve and uses the closed-form bond price that OU dynamics imply to value and hedge interest-rate products. The model's Achilles heel is the Normal stationary distribution, which permits negative rates — historically dismissed as unrealistic, but after the European Central Bank and the Bank of Japan ran negative policy rates in the mid-2010s, the "flaw" briefly described reality. Modern desks mostly use richer descendants (Hull-White, which makes `m` time-dependent to fit the whole curve exactly, or shifted/lognormal variants that floor rates), but every one of them is a dressed-up OU process, and the [Vasicek and Hull-White machinery](/blog/trading/quantitative-finance/short-rate-models-vasicek-hull-white) starts from exactly the `dr = theta(m - r) dt + sigma dB` we wrote.

**Black-Scholes and the options desk.** The entire Black-Scholes framework assumes the underlying is a GBM. The famous formula's `N(d2)` term is precisely the risk-neutral probability that the stock finishes above the strike — the same lognormal tail calculation we did in the interview section, with `mu` swapped for the risk-free rate `r`. When traders talk about the [volatility surface](/blog/trading/quantitative-finance/volatility-surface), they are documenting all the ways real prices *deviate* from the single-`sigma` GBM assumption: real returns have fatter tails and jumps that GBM misses, so the market charges more for far-out-of-the-money options than flat GBM would imply. You cannot understand the surface until you understand the GBM baseline it departs from.

**Pairs trading and stat arb.** A statistical-arbitrage desk hunts for pairs (or baskets) of assets whose price *spread* is mean-reverting — formally, *cointegrated*. They fit an OU process to the spread, and three OU quantities become the entire trading rulebook: the long-run mean `m` is fair value, the stationary standard deviation `sigma / sqrt(2 theta)` sets the entry threshold (enter when the spread is, say, 2 sigma rich or cheap), and the half-life `ln 2 / theta` sets the holding period and the stop. A pair with a 2-week half-life is tradeable; a pair with a 2-year half-life ties up capital for too little reversion per year. The classic disaster — a spread that dislocates further and stays dislocated past every stop — is the diffusion term overwhelming the reversion drift, exactly the risk the OU model quantifies through its variance. The full workflow of finding and validating such a signal is its own discipline, covered in [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research).

**Volatility models.** Volatility itself mean-reverts — calm and stormy periods cluster, but vol always drifts back toward a long-run level. The Heston stochastic-volatility model uses a close relative of OU (a square-root, or CIR, process) for the variance, precisely so that variance stays positive *and* reverts. Every vol-arb trade — selling expensive vol, buying cheap vol — is a bet on mean reversion in an OU-like variance process. The reversion speed and long-run level of vol are quoted and traded directly in the variance-swap market.

**Monte Carlo pricing.** When a derivative is too complex for a closed form — a path-dependent exotic, a multi-asset basket — desks price it by Euler-Maruyama simulation of the underlying SDEs, exactly the scheme from the simulation section. The `sqrt(dt)` discipline is not pedantry there: a missing square root silently mis-prices the book by mis-stating the volatility, and the error compounds across thousands of paths. Convergence of the scheme as `dt` shrinks, and variance-reduction tricks like antithetic Normal draws, are standard production concerns. The broader practice of pricing by simulation is laid out in [derivatives pricing](/blog/trading/quantitative-finance/derivatives-pricing).

## When this matters and where to go next

If you are interviewing for a quant-researcher or derivatives seat, these two SDEs are table stakes. You will be asked to write down GBM, solve it with Itô's lemma, and recite its mean and variance from memory; you will be asked for the OU half-life and stationary distribution; and you will be asked, in some disguise, to compute a lognormal tail probability and connect it to Black-Scholes. The depth that separates a strong candidate is *fluency with the `- ½ sigma²` correction* — knowing not just that it is there but why, and being able to convert between the arithmetic mean of returns and the geometric (compounding) rate without fumbling.

Beyond the interview, this is the grammar of the whole derivatives world. Every pricing model you will ever calibrate is an SDE; every hedge you compute is a derivative of an SDE solution; every risk number is a moment of one. Master GBM and OU cold, and the rest of stochastic finance becomes variations on two themes you already own.

For the natural next steps: the [Black-Scholes deep dive](/blog/trading/quantitative-finance/black-scholes) shows how GBM plus a no-arbitrage argument produces an option price; [short-rate models](/blog/trading/quantitative-finance/short-rate-models-vasicek-hull-white) extends OU into the full machinery of fixed-income pricing; [Markov chains and hitting times](/blog/trading/quantitative-finance/markov-chains-hitting-times-quant-interviews) gives the discrete cousin of the first-passage problems; and the [probability distributions cheat sheet](/blog/trading/quantitative-finance/distributions-cheat-sheet-quant-interviews) catalogs the Normal and lognormal laws that these processes generate. This material is educational, not investment advice — every strategy that can profit from a model can also lose money when the model breaks, and knowing exactly where GBM and OU stop describing reality is as valuable as knowing where they start.
