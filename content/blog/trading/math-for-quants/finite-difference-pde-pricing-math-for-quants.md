---
title: "Finite differences: pricing on a grid, and the schemes that blow up"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "A pricing PDE solved on a grid gives you the whole price surface in one pass, and the Greeks come off it for free. The catch is that the obvious scheme is only conditionally stable, so above a step-size threshold it returns oscillation that looks exactly like a price."
tags: ["finite-difference", "pde-pricing", "crank-nicolson", "rannacher", "von-neumann-stability", "american-options", "greeks", "black-scholes", "monte-carlo", "numerical-methods", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 23
---

> [!important]
> **TL;DR:** A pricing PDE solved on a grid marches backward from the payoff and fills in the option value at every spot and every date in a single pass. Monte Carlo gives you one number at one spot. The price of that generality is that the obvious scheme is only conditionally stable, and it breaks without telling you.
>
> - The **explicit** scheme is stable only when $\lambda = \sigma^2 \Delta t / \Delta x^2 \le 1$. Refining the space grid forces a **quadratically** smaller time step, so halving $\Delta x$ costs eight times the work.
> - Violating it raises no error. At $\lambda = 1.010$ the grid returns \$4.688861 against a true \$4.614997: wrong by \$73,864 on a million options, and still shaped like an option price.
> - **Crank-Nicolson is unconditionally stable, which is not the same as unconditionally accurate.** On a plain vanilla call it can return a gamma of $-1.547$ when the true gamma is $+0.039288$. A long call cannot have negative gamma.
> - The fix is **Rannacher start-up**: replace the first two Crank-Nicolson steps with four implicit half-steps. That one change takes the gamma error from $-4038\%$ to $+0.039\%$.
> - The number to remember: grid gamma is accurate to 0.007% while a compute-matched Monte Carlo is uncertain by 1.21%. On a million-option book that is 3 shares of hedge error against 474.

## The number a simulation will not give you

You are short a million three-month call options and the stock has just moved. Before you can hedge you need to know three things: what the book is worth, how many shares to hold against it, and how fast that share count changes as the stock keeps moving. Price, delta, gamma.

Monte Carlo answers the first question well and the other two badly. It gives you a price at the spot you simulated from, with an error bar. Want delta? Bump the spot and simulate again. Want gamma? Bump twice more, difference three noisy numbers, and watch the noise get divided by the square of a small bump.

A partial differential equation solver has the opposite shape. It does not compute a price at a point. It computes the entire function: option value at every stock price on a range and every date between now and expiry. Delta and gamma are then differences between numbers already sitting next to each other in memory.

![Two panels. On the left, Monte Carlo fans a handful of simulated paths from one spot to a payoff at maturity and returns one price with a standard error. On the right, a grid over stock price and time to maturity is filled in from the payoff along the top edge, with price, delta and gamma available at every node from a single solve](/imgs/blogs/finite-difference-pde-pricing-math-for-quants-1.webp)

That figure is the trade in one image. Simulation answers one question at one point. A grid answers every question on the whole domain.

The catch is that the most natural way to fill that grid is only **conditionally stable**. Push the time step past a threshold and it does not warn you, throw, or return an infinity. It returns numbers. Plausible ones, for a while.

## Foundations: the PDE, and what a grid does to it

The pricing equation is the Black-Scholes PDE, derived from the delta-hedging argument in [Feynman-Kac and the Black-Scholes PDE](/blog/trading/math-for-quants/feynman-kac-black-scholes-pde-math-for-quants) and paired with [the Kolmogorov forward equation](/blog/trading/math-for-quants/fokker-planck-kolmogorov-forward-math-for-quants) for the density's own evolution. Taken as given:

$$\frac{\partial V}{\partial t} + \tfrac12 \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0$$

The payoff at expiry is not part of the equation. It is the **terminal condition**, the thing you march away from. Two changes of variable make it tractable: $\tau = T - t$ so the solve runs forward from the payoff, and $x = \ln S$, which turns the $S^2$ and $S$ coefficients into constants.

$$\frac{\partial V}{\partial \tau} = \tfrac12 \sigma^2 \frac{\partial^2 V}{\partial x^2} + \left(r - \tfrac12\sigma^2\right)\frac{\partial V}{\partial x} - rV$$

That is diffusion plus drift plus decay, and it is the whole model.

A **grid** replaces both continuous variables with a finite mesh: nodes spaced $\Delta x$ apart along the log-price axis, and $M$ steps of size $\Delta t = T/M$ along time. Write $V_j^n$ for the value at node $j$ and level $n$, then replace each derivative with a difference between neighbours:

$$\frac{\partial V}{\partial x} \approx \frac{V_{j+1} - V_{j-1}}{2\Delta x}, \qquad \frac{\partial^2 V}{\partial x^2} \approx \frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta x^2}$$

Both are exact to order $\Delta x^2$: add the Taylor expansions of $V_{j+1}$ and $V_{j-1}$ and the odd terms cancel. That turns one equation in calculus into a large system in arithmetic, marched from the payoff at $\tau = 0$ out to $\tau = T$.

Every worked number below uses the same instrument: a European call, spot and strike both at \$100, $r = 5\%$, $\sigma = 20\%$, three months to expiry, which the Black-Scholes closed form prices at \$4.614997. The log-price domain runs one unit either side of $\ln 100$, so $S$ spans \$36.788 to \$271.828, cut into 200 intervals of $\Delta x = 0.01$. All dollar figures are illustrative arithmetic on those assumed inputs.

## Three schemes, and the one thing that separates them

Everything in the difference formulas above concerns the space direction. The scheme is the choice about **time**: when you evaluate the right-hand side, do you use the level you already know, the level you are solving for, or both?

Write $\mathcal{L}V$ for the whole spatial right-hand side. The three answers, with what each costs:

| Scheme | Update rule | Accuracy | Cost per step | Stable when |
| --- | --- | --- | --- | --- |
| Explicit | $(V^{n+1} - V^n)/\Delta t = \mathcal{L}V^n$ | $O(\Delta t) + O(\Delta x^2)$ | one multiply per node | $\sigma^2 \Delta t/\Delta x^2 \le 1$ |
| Implicit | $(V^{n+1} - V^n)/\Delta t = \mathcal{L}V^{n+1}$ | $O(\Delta t) + O(\Delta x^2)$ | one tridiagonal solve | always |
| Crank-Nicolson | $(V^{n+1} - V^n)/\Delta t = \tfrac12(\mathcal{L}V^{n+1} + \mathcal{L}V^{n})$ | $O(\Delta t^2) + O(\Delta x^2)$ | one tridiagonal solve | always |

Explicit is a formula: every unknown sits alone on the left, so you evaluate and move on. The other two are systems, because $\mathcal{L}V^{n+1}$ couples each unknown to its neighbours. Since $\mathcal{L}$ reaches only one node either side that system is **tridiagonal**, and the Thomas algorithm solves it in one forward sweep and one back substitution, a handful of operations per node.

![Three finite difference stencils side by side. Explicit computes one unknown at the earlier time level from three known values at the later level. Implicit ties three unknowns at the earlier level to one known value. Crank-Nicolson connects three unknowns and three knowns, averaging the two. Each carries its accuracy order, its cost per step and its stability condition](/imgs/blogs/finite-difference-pde-pricing-math-for-quants-2.webp)

Cost per step is comparable across all three; accuracy is not. Crank-Nicolson averages the other two, the leading time errors cancel, and it is second order. If cost per step were the only consideration it would win outright and there would be nothing more to say. The last column is why there is.

## Stability is the load-bearing idea

Run the explicit scheme with too large a time step and the answer does not degrade gracefully. It oscillates, node to node, with an amplitude that doubles every few steps until the grid holds nothing but alternating signs.

The reason is visible in one line of analysis. Feed a single Fourier mode $e^{ikx}$ into the explicit update and ask what multiplies it each step. The answer, for the diffusion part, is the **amplification factor**

$$G = 1 - 2\lambda \sin^2(k\Delta x/2), \qquad \lambda = \frac{\sigma^2 \Delta t}{\Delta x^2}$$

Every mode in the payoff is multiplied by its own $G$ once per step. Modes with $|G| \lt 1$ shrink, which is what diffusion is supposed to do; a mode with $|G| \gt 1$ grows geometrically. The worst case is the highest frequency the grid can represent, the sawtooth alternating sign at every node, where $\sin^2$ hits 1 and $G = 1 - 2\lambda$. That stays inside the unit circle only when

$$\lambda = \frac{\sigma^2 \Delta t}{\Delta x^2} \le 1$$

This is the Courant-Friedrichs-Lewy condition for this equation, and its shape is what matters. **The time step is bounded by the square of the space step.** Halve $\Delta x$ and $\Delta t$ must fall by a factor of four, so you take four times as many steps over twice as many nodes: eight times the work for a four-fold gain in spatial accuracy. That is why nobody refines an explicit grid very far.

On the grid fixed above, $\Delta x = 0.01$ and $\sigma = 0.20$, so the threshold is $\Delta t \le \Delta x^2/\sigma^2 = 0.0001/0.04 = 0.0025$ years. With three months to expiry that is exactly 100 time steps, and the ratio reduces to $\lambda = 100/M$. Below 100 steps the scheme is unstable.

#### Worked example 1: a book that is fine at 110 steps and fiction at 99

Take the desk position: short 1,000,000 of these calls, which the closed form values at \$4,614,997. Solve the same PDE on the same grid with the explicit scheme, varying only the number of time steps.

| Steps $M$ | $\lambda = 100/M$ | Grid price | Error on 1,000,000 options |
| --- | --- | --- | --- |
| 125 | 0.800 | \$4.614137 | -\$860 |
| 110 | 0.909 | \$4.614686 | -\$311 |
| 101 | 0.990 | \$4.616417 | +\$1,420 |
| 100 | 1.000 | \$4.605073 | -\$9,924 |
| 99 | 1.010 | \$4.688861 | +\$73,864 |
| 97 | 1.031 | \$8.116760 | +\$3,501,763 |
| 95 | 1.053 | \$147.083217 | +\$142,468,220 |
| 90 | 1.111 | -\$775,462.215640 | nonsense |

Read the middle of that table slowly, because it is the point of the post. At 110 steps the grid is off by \$311 on a \$4.6m book, a rounding error. Take away eleven time steps, an 11% coarser march, and the answer becomes \$4.688861: still a perfectly plausible price for a three-month at-the-money call, nothing about it visibly wrong, and wrong by \$73,864. Two steps coarser again and the grid says \$8.116760, the right *kind* of number, roughly what this option would fetch at 35% volatility, and wrong by \$3.5m. Only at $M = 95$ is the output obviously absurd.

The growth rates are exactly what the amplification factor predicts. At $\lambda = 1.010$ the sawtooth is multiplied by 1.0202 each step, and 1.0202 raised to the 99th power is 7.2: a small ripple grows sevenfold, enough to shift the price, not enough to look insane. At $\lambda = 1.111$ the factor is 1.2222, and 1.2222 raised to the 90th is about $7 \times 10^7$. At $\lambda = 0.909$ it is 0.8182, which over 110 steps is $2.6 \times 10^{-10}$, so the ripple is annihilated.

![Two panels showing the same explicit grid at two time step counts. At 110 steps and a stability ratio of 0.909 the option value curve is smooth and monotone. At 90 steps and a ratio of 1.111 the same grid holds a violent sawtooth alternating between roughly plus and minus \$775,000 at adjacent nodes](/imgs/blogs/finite-difference-pde-pricing-math-for-quants-3.webp)

The instability is seeded by the payoff itself. A call payoff has a kink at the strike, and a kink on a grid contains energy at every frequency including the sawtooth. There is always something for an unstable scheme to amplify.

## Crank-Nicolson, and the failure mode nobody quotes

Both implicit schemes escape this. For the implicit update the amplification factor is $G = 1/(1 + 2\alpha \sin^2(k\Delta x/2))$ with $\alpha = \lambda/2$, positive and below 1 for every mode and every step size. For Crank-Nicolson it is

$$G = \frac{1 - 2\alpha \sin^2(k\Delta x/2)}{1 + 2\alpha \sin^2(k\Delta x/2)}$$

whose magnitude is at most 1 for every $\alpha$. Both are **unconditionally stable**: no step size makes them blow up. Crank-Nicolson is also second order in time. This is why it is the default.

Now read that expression again for a large $\alpha$. As $\alpha$ grows, $G$ at the sawtooth frequency tends to $-1$, not to zero. The highest mode is not damped at all: it is **sign-flipped and preserved**, step after step, and it rings. Unconditional stability says the oscillation will not grow. It says nothing about it decaying, and nothing whatsoever about accuracy. That gap is what the phrase hides.

It bites hardest exactly where a desk wants to be. You refine the space grid because you want clean Greeks. That shrinks $\Delta x$, which inflates $\alpha = \sigma^2 \Delta t / (2\Delta x^2)$, which pushes $G$ toward $-1$. **The better your spatial resolution, the worse Crank-Nicolson rings**, unless you also cut the time step, which is the cost you adopted it to avoid.

#### Worked example 2: a wrong-sign delta on a \$10m digital

Price a cash-or-nothing digital call paying \$1 per unit if the stock finishes above \$100, on 10,000,000 units, so \$10m of notional payout. The closed form is $e^{-rT}N(d_2) = 0.5233102$ per unit, worth \$5,233,102.

Refine the space grid to $\Delta x = 0.002$, five times finer than before, and take 25 time steps of $\Delta t = 0.01$. Then $\alpha = 50$ exactly, so the sawtooth amplification factor is $G = -99/101 = -0.980198$. Over 25 steps that is 0.606: three fifths of the initial ripple is still there at the end, flipping sign every step.

Here is the price profile Crank-Nicolson returns around the strike, against the truth:

| Spot | Crank-Nicolson | Rannacher | True |
| --- | --- | --- | --- |
| \$99.601 | 0.4915 | 0.5076 | 0.5076 |
| \$99.800 | 0.5908 | 0.5154 | 0.5154 |
| \$100.000 | 0.5235 | 0.5233 | 0.5233 |
| \$100.200 | 0.4559 | 0.5312 | 0.5312 |
| \$100.401 | 0.5549 | 0.5390 | 0.5390 |

At the money Crank-Nicolson gives 0.5235 against a true 0.5233, an error of \$2,000 on the \$10m ticket. You would ship that. But the node one tick below reads 0.5908 where the truth is 0.5154, an error of \$754,000, and the node one tick above reads 0.4559 where the truth is 0.5312. The price is not rising with spot. It is zig-zagging.

Now hedge it. Delta is the slope through the two neighbouring nodes:

$$\Delta = \frac{1}{S}\cdot\frac{V_{j+1} - V_{j-1}}{2 \Delta x} = \frac{1}{100}\cdot\frac{0.4559 - 0.5908}{0.004} = -0.337$$

The true delta of this digital is $+0.039288$, which at the money coincides with the vanilla call's gamma below because $S\varphi(d_1) = Ke^{-r\tau}\varphi(d_2)$. The grid says sell 3,370,000 shares. The truth says buy 392,880 shares. That is a hedge wrong by 3,762,880 shares, roughly \$376m of the wrong-way position, from a scheme whose at-the-money *price* was within \$2,000.

#### The fix: Rannacher start-up

Rannacher's remedy, from a 1984 paper on diffusion problems with irregular data, is almost embarrassingly small. Replace the first two Crank-Nicolson steps with **four half-steps of fully implicit Euler**, then carry on as normal.

It works because implicit Euler is the one scheme that crushes high frequencies: at $\alpha = 25$, which is what a half-step gives here, its sawtooth factor is 1/51, and four of them multiply to $1.5 \times 10^{-7}$. The ripple the discontinuity injects is destroyed before Crank-Nicolson ever sees it, so there is nothing left to ring on. You surrender second-order accuracy for four half-steps out of fifty and keep it everywhere else, so the global order survives.

The Rannacher column above is the same solver with that one change. Every entry matches the closed form to four decimals, and the delta comes back at $+0.039300$ against a true $+0.039288$.

![A chart of digital option value against spot from \$98.8 to \$101.2 with the strike at \$100. The Crank-Nicolson series zig-zags violently node to node, peaking at 0.5908 just below the strike and dropping to 0.4559 just above it, while the Rannacher series rises in a clean straight line through the true values](/imgs/blogs/finite-difference-pde-pricing-math-for-quants-4.webp)

This is not only a digital problem. A vanilla call's kink is milder than a jump but still non-smooth, and the same ringing shows up in its **gamma**, the second difference and so the most sensitive thing on the grid. On the vanilla call at $\Delta x = 0.002$:

| Steps $M$ | $\alpha$ | Price | Delta | Gamma | Gamma error |
| --- | --- | --- | --- | --- | --- |
| 25, plain | 50.0 | \$4.639187 | 0.569729 | -1.54703620 | -4038% |
| 25, Rannacher | 50.0 | \$4.614199 | 0.569450 | 0.03930337 | +0.039% |
| 50, plain | 25.0 | \$4.612154 | 0.569437 | 0.26666871 | +579% |
| 50, Rannacher | 25.0 | \$4.614651 | 0.569458 | 0.03929360 | +0.014% |
| 100, plain | 12.5 | \$4.614811 | 0.569461 | 0.03960445 | +0.805% |
| 100, Rannacher | 12.5 | \$4.614765 | 0.569460 | 0.03929085 | +0.007% |

The true values are \$4.614997, delta 0.569460, gamma 0.03928800.

Look at the 50-step rows. The plain Crank-Nicolson price is off by a quarter of a cent and its delta is right to four decimal places, so every sanity check a desk normally runs comes back clean. Its gamma is 0.26666871 against a true 0.03928800, nearly seven times too large. At 25 steps the gamma is -1.54703620: **negative gamma on a long call**, which is not a small error but an impossibility, and still the price and delta look fine.

That is the real danger of this scheme. It does not fail where you are looking.

## Boundary conditions, and where to cut the domain

A grid needs edges, and a real stock price has no upper bound, so you truncate. Both decisions are modelling, not bookkeeping.

At the edges you impose what you know. For a call, value goes to zero as $S \to 0$, and far above the strike the option is worth the discounted forward, $V \approx S - Ke^{-r\tau}$. Those are exact asymptotics, so the only error is applying them at a finite distance rather than at infinity.

How far is far enough? Hold $\Delta x$ at 0.01 and widen the domain, measuring against the closed form. The natural scale is $\sigma\sqrt{T} = 0.10$, one standard deviation of the log return.

| Half-width | In standard deviations | Spot range | Price | Truncation cost |
| --- | --- | --- | --- | --- |
| 0.08 | 0.8 | \$92.31 to \$108.33 | \$4.153483 | \$456,598 |
| 0.10 | 1.0 | \$90.48 to \$110.52 | \$4.441636 | \$168,445 |
| 0.12 | 1.2 | \$88.69 to \$112.75 | \$4.555737 | \$54,344 |
| 0.15 | 1.5 | \$86.07 to \$116.18 | \$4.602301 | \$7,780 |
| 0.20 | 2.0 | \$81.87 to \$122.14 | \$4.609924 | \$157 |
| 0.30 | 3.0 | \$74.08 to \$134.99 | \$4.610081 | \$0 |
| 1.00 | 10.0 | \$36.79 to \$271.83 | \$4.610081 | \$0 |

The truncation cost column is the price difference from the widest domain, on a million options. Past three standard deviations it is zero to six decimals: the extra 140 nodes between three and ten buy nothing at all. Below two it becomes the dominant error in the whole calculation, and at one standard deviation the boundary condition is effectively pricing the option for you.

The residual \$4,916 gap between \$4.610081 and the closed-form \$4.614997 is not truncation. It is the $\Delta x^2$ space error, and only a finer mesh shrinks it.

## Greeks for free, which is the actual argument

This is where the grid earns its keep. Once the march reaches $\tau = T$ you hold the value at every node, so delta and gamma are differences of numbers you already have:

$$\Delta = \frac{1}{S}\frac{\partial V}{\partial x}, \qquad \Gamma = \frac{1}{S^2}\left(\frac{\partial^2 V}{\partial x^2} - \frac{\partial V}{\partial x}\right)$$

using the same central differences as before, and the two extra factors coming from the chain rule for $x = \ln S$. Theta is free too: it is the difference between the last two time levels. No extra solve, no extra noise.

#### Worked example 3: grid Greeks against a compute-matched Monte Carlo

Give both methods the same arithmetic budget. The grid runs Crank-Nicolson with Rannacher start-up on 1,000 space nodes and 100 time steps, about 100,000 node updates. The Monte Carlo runs 100,000 terminal-value paths, revalued at the spot and at \$99 and \$101 under common random numbers, about 300,000 payoff evaluations. Repeating it 400 times with independent seeds gives its true dispersion rather than one lucky draw.

| Quantity | Truth | Grid | Grid error | Monte Carlo | MC standard deviation |
| --- | --- | --- | --- | --- | --- |
| Price | \$4.614997 | \$4.614765 | 0.005% | \$4.615663 | \$0.014091 (0.305%) |
| Delta | 0.569460 | 0.569460 | 0.000% | 0.569272 | 0.000329 (0.058%) |
| Gamma | 0.03928800 | 0.03929085 | 0.007% | 0.039260 | 0.000474 (1.21%) |

The price comparison is close to a draw. The gamma comparison is not: the grid is accurate to 0.007% and the Monte Carlo is uncertain by 1.21%, a factor of about 170, for the same compute.

Put that on the book. Short 1,000,000 calls, and total gamma is 0.03928800 times 1,000,000, which is 39,288 shares of delta per \$1 move. The grid's gamma error is 0.03929085 less 0.03928800, or 0.00000285 per option, so 2.85 shares per \$1. The Monte Carlo's one-sigma error is 0.000474 per option, so 474 shares. Over a \$5 move the delta you should have re-hedged changes by 196,440 shares; the grid mis-sizes that by 14 shares, worth about \$1,400 of unintended exposure, and a typical Monte Carlo run mis-sizes it by 2,370 shares, worth \$237,000. That is one standard deviation, so a bad seed is worse.

The reason is structural. Bump-and-revalue divides a difference of noisy numbers by the square of a small bump: shrink the bump to cut the bias and the variance explodes, widen it to control the variance and the bias grows. On a grid gamma is not estimated at all. It is read.

## American options: the constraint that a grid takes in its stride

An American option can be exercised at any time, so its value can never sit below its intrinsic value. On a grid this is one line. After each time step, before moving on, overwrite every node with

$$V_j^{n+1} \leftarrow \max\left(V_j^{n+1},\ \text{payoff}(S_j)\right)$$

Each node already carries the continuation value, so comparing it with the payoff *is* the exercise decision, taken everywhere at once, and the early-exercise boundary is where the two meet. The theory is in [optimal stopping](/blog/trading/math-for-quants/optimal-stopping-secretary-when-to-take-the-trade-math-for-quants) and the backward induction in [dynamic programming for optimal execution](/blog/trading/math-for-quants/dynamic-programming-optimal-execution-math-for-quants). Simulation finds this hard, because a forward path does not know its own continuation value; recovering it needs a regression across paths, which is Longstaff-Schwartz and a much larger piece of machinery.

#### Worked example 4: an American put, checked two ways

Same instrument, a put this time, strike \$100. Before trusting the American number, verify the solver on the problem that has a closed form: switch the exercise constraint off and the same grid returns \$3.372764 for the European put against the Black-Scholes value of \$3.372777, agreeing to five significant figures. **That check is against the analytic formula, not against a finer grid.** Checking one approximation against a better approximation only proves the two converge to the same place, never that the place is right.

Switch the constraint back on and the grid gives \$3.4796 per option, against \$3.4798 from a 40,000-step binomial tree. Two different discretisations of two different formulations agreeing to \$0.0002 is evidence in a way that refining either alone would not be. On 1,000,000 puts: European \$3,372,800, American \$3,479,600, so the right to exercise early is worth \$106,800. The grid also hands back the boundary: today it is optimal to exercise below about \$86.94 and to hold above it.

## The honest limit

The curse of dimensionality ends this method, and it ends it abruptly. With $N$ nodes per dimension, a $d$-factor model needs $N^d$ of them. At $N = 200$: one factor is 200 nodes, two is 40,000, three is 8,000,000, four is 1.6 billion before you have taken a single time step.

One or two factors is comfortable, which covers most equity and rate exotics. Three is painful and needs specialist splitting methods. Beyond that the grid is finished and Monte Carlo takes over, because simulation error depends on the number of paths and not the number of dimensions. The two methods are not rivals. They own different parts of the problem.

## Common misconceptions

**"Implicit is always better than explicit."** Implicit is unconditionally stable, which is worth a great deal, but both are first order in time and implicit is *more* diffusive. Plain implicit at 110 steps gives \$4.605542 against explicit's \$4.614686 at the same step count: the stable scheme was the less accurate one. Explicit is also trivial to code and to parallelise, and fine when $\Delta x$ is coarse. What you must never do is run it without checking $\lambda$.

**"Crank-Nicolson is unconditionally accurate."** It is unconditionally *stable*. The gap between those two claims is worked example 2 and the gamma table: a scheme that never blows up, returning a negative gamma on a long call. Stability says errors do not grow. It does not say they are small.

**"Finite differences are obsolete now that we have Monte Carlo."** Simulation is the only option past three factors and the wrong tool below that. For a single-factor American or barrier product a grid is faster, handles early exercise exactly, and returns the Greeks for free.

**"A finer grid is always a better grid."** Refining $\Delta x$ forces a quadratically smaller $\Delta t$ under the explicit scheme, and makes Crank-Nicolson ring harder under any time step. Refinement has to be done in both directions at once.

## Sources and further reading

- Paul Wilmott, Sam Howison and Jeff Dewynne, *The Mathematics of Financial Derivatives: A Student Introduction*, Cambridge University Press, 1995. The standard derivation and its finite-difference treatment.
- Daniel J. Duffy, *Finite Difference Methods in Financial Engineering*, Wiley, 2006. Book-length, including the case against naive Crank-Nicolson.
- Rolf Rannacher, "Finite element solution of diffusion problems with irregular data", *Numerische Mathematik* 43, 309-327, 1984. The start-up procedure.
- Michael Giles and Rebecca Carter, "Convergence analysis of Crank-Nicolson and Rannacher time-marching", *Journal of Computational Finance* 9(4), 2006.
- Michael Brennan and Eduardo Schwartz, "The Valuation of American Put Options", *Journal of Finance* 32(2), 449-462, 1977. The paper that put American options on a grid.
- Richard Courant, Kurt Friedrichs and Hans Lewy, "Über die partiellen Differenzengleichungen der mathematischen Physik", *Mathematische Annalen* 100, 32-74, 1928. The stability condition.
- Domingo Tavella and Curt Randall, *Pricing Financial Instruments: The Finite Difference Method*, Wiley, 2000.

All dollar figures here are illustrative arithmetic on the stated assumed inputs, not market observations.

## In the interview room and on the desk

The question arrives as "how would you price an American put?", and a grid is the strong answer. Say it in this order. Write the pricing PDE, state that American exercise adds the constraint that value never falls below intrinsic, and note that this is natural on a grid because each node already holds a continuation value, whereas simulation needs Longstaff-Schwartz regression to recover one. Then say you would use Crank-Nicolson with Rannacher start-up, and stop there, because you have just handed over the follow-up.

The follow-up is always stability or Greeks, and usually both.

On stability, the answer they want is not the formula but its shape: the explicit scheme needs $\sigma^2 \Delta t / \Delta x^2 \le 1$, the time step is bounded by the **square** of the space step, so refining space forces a quadratically smaller time step. Add that violating it produces oscillation rather than an exception, and that the oscillation can be small enough to look like a price, and you have said the thing that separates someone who has run a grid from someone who has read about one.

On Greeks, say that delta and gamma are central differences of adjacent nodes you already computed, so they cost nothing, and that this is the real argument against Monte Carlo for a low-factor book. Bump-and-revalue divides noise by the square of the bump, which is why gamma is the Greek that simulation handles worst.

The trap is quoting Crank-Nicolson as unconditionally safe. It is unconditionally stable, and the interviewer is waiting to ask what happens on a digital payoff. The answer is that its amplification factor tends to $-1$ rather than 0 at the highest grid frequency, so the discontinuity rings instead of decaying, and it rings worst in gamma, which is where you would notice last. Then name the fix: four implicit half-steps at the start, Rannacher, which damps the high modes without costing the second-order convergence anywhere else. Candidates who know the failure mode and the fix are rare, and it reads as desk experience because it usually is.

Jane Street and Citadel weight this heavily, as does any exotics, structured products or model-validation seat, where the grid is the production pricer and someone has to own the question of whether its Greeks can be hedged on.
