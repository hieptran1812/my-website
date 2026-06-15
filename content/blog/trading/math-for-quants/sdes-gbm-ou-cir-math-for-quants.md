---
title: "Stochastic differential equations: GBM, OU, and CIR"
date: "2026-06-15"
description: "A beginner-friendly, build-from-zero guide to stochastic differential equations, the Euler-Maruyama simulation scheme, and the three SDEs every quant uses: geometric Brownian motion for prices, Ornstein-Uhlenbeck for mean-reverting spreads, and Cox-Ingersoll-Ross for positive interest rates."
tags: ["stochastic-differential-equations", "gbm", "ornstein-uhlenbeck", "cox-ingersoll-ross", "euler-maruyama", "mean-reversion", "black-scholes", "short-rates", "monte-carlo", "quant-finance", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A stochastic differential equation is a one-line recipe that says how a quantity changes in the next instant: a predictable *drift* plus a random *diffusion* kick. Choosing the drift and the diffusion is how a quant decides whether a thing trends, reverts, or stays positive.
>
> - The master form is $dX = a(X,t)\,dt + b(X,t)\,dW$: the $dt$ term is the **drift** (the average direction) and the $dW$ term is the **diffusion** (the noise). Everything in this post is just two specific choices of $a$ and $b$.
> - **Geometric Brownian motion** $dS = \mu S\,dt + \sigma S\,dW$ is the price model behind Black-Scholes; its solution $S_t = S_0\,e^{(\mu - \sigma^2/2)t + \sigma W_t}$ guarantees the price can drift but never goes negative.
> - **Ornstein-Uhlenbeck** $dX = \theta(\mu - X)\,dt + \sigma\,dW$ is the math of **mean reversion** — spreads, rates, and volatility that get pulled back to a level. Its **half-life** is $\ln 2 / \theta$, the single number that tells you how long a trade takes to pay off.
> - **Cox-Ingersoll-Ross** $dr = \theta(\mu - r)\,dt + \sigma\sqrt{r}\,dW$ adds a $\sqrt{r}$ to the noise so an interest rate (or a variance) can never go negative; the **Feller condition** $2\theta\mu \ge \sigma^2$ says when zero is truly off-limits.
> - The one tool that makes all of this concrete is **Euler-Maruyama**: step time forward by a small $\Delta t$, add the drift, add a random shock of size proportional to $\sqrt{\Delta t}$, and repeat. That is how a one-line equation becomes a price chart you can put a dollar P&L on.

Here is a question that sits underneath almost every model on a trading desk, and almost nobody states it plainly. A stock price, an interest rate, the gap between two related stocks — each of these is a number that jiggles forward in time, partly with a sense of direction and partly at random. How do you write down a single, honest rule for *how it moves next*, one that you can both reason about with a pen and simulate ten million times on a computer? You cannot just say "it goes up on average" — that ignores the randomness. You cannot just say "it is random" — that ignores the direction, the floor at zero, the pull back to a level. You need a way to bottle *both at once*.

That bottle is the **stochastic differential equation**, or SDE. It is a deceptively short line of math that says: in the next sliver of time, this quantity moves by a predictable amount plus a random amount, and here is exactly how big each piece is. Master three specific SDEs — geometric Brownian motion, Ornstein-Uhlenbeck, and Cox-Ingersoll-Ross — and you have the engine behind option pricing, statistical arbitrage, interest-rate models, and the volatility models that run the derivatives world. The beautiful part is that all three are the *same equation* with two slots filled in differently. We are going to fill in those slots from zero.

![Matrix comparing GBM, OU, and CIR by equation shape, behavior, and market use](/imgs/blogs/sdes-gbm-ou-cir-math-for-quants-1.png)

The table above is the whole post in one picture, and we will earn every cell of it. Three rows, three SDEs. Each one is a choice of *drift* (the predictable part) and *diffusion* (the random part), and that choice alone decides the personality of the thing it models: a stock that wanders upward and never goes negative, a spread that keeps getting yanked back to its average, a rate that reverts but refuses to cross zero. By the end you will be able to look at any of these equations, say in plain words what it does, simulate a path with real dollar figures, and explain which market quantity it belongs to and why.

## Foundations: what an SDE actually says

Before any Greek letters, let us agree on a handful of words, building each from the simplest possible example and only then writing the formal version. If you have never seen a differential equation, you can still follow every line here.

### What is a differential equation, without the randomness?

Start with no randomness at all. Imagine money in a bank account that earns interest continuously. A **differential equation** is a rule that ties the *rate of change* of a quantity to the quantity itself. For the bank account: the rate at which your money grows is proportional to how much money you have. In symbols, if $X_t$ is your balance at time $t$ and $r$ is the interest rate, then

$$dX = r\,X\,dt.$$

Read that out loud: "the small change $dX$ in your balance over a small slice of time $dt$ equals $r$ times your balance times that slice of time." If you have \$1,000 and the rate is 5% per year, then over one year your balance changes by about $0.05 \times \$1{,}000 = \$50$. The notation $dt$ just means "an infinitesimally small step of time," and $dX$ is "the corresponding tiny change in $X$." Solve this equation — sum up all the tiny changes — and you get the familiar continuous-compounding formula $X_t = X_0\,e^{rt}$. The point for now is simpler than the algebra: a differential equation is a *local* rule (what happens in the next instant) that, when you stitch all the instants together, gives you a *global* path (the whole curve over time).

Everything in this post is that idea — a local rule for the next instant — with one addition: the world is not deterministic. The next instant has a random component. That is the "stochastic" in stochastic differential equation.

### What is the random piece? Brownian motion in one paragraph

The randomness comes from a single object called **Brownian motion**, written $W_t$ (the $W$ honours Norbert Wiener, who made it rigorous). Picture a tiny grain of pollen jittering on the surface of water, kicked around by molecules from every side. Brownian motion is the mathematical idealization of that jitter. Three facts are all we need. First, it starts at zero: $W_0 = 0$. Second, over a time interval of length $t$, the total displacement $W_t$ is a random draw from a normal (bell-curve) distribution with mean zero and variance $t$ — so its typical size is $\sqrt{t}$. Third, its increments over non-overlapping intervals are independent: what it does next is unrelated to what it did before. That last property is the mathematical heart of "the market has no memory in its noise."

The single most important consequence, and the one beginners trip on, is the **square-root scaling of randomness**. Over a small time step $\Delta t$, the random increment $\Delta W = W_{t+\Delta t} - W_t$ is a normal draw with mean zero and *standard deviation* $\sqrt{\Delta t}$ — not $\Delta t$. Halve the time step and the typical random kick shrinks only by $\sqrt{2} \approx 1.41$, not by 2. This is why noise dominates over short horizons and drift dominates over long ones, and it will show up in every simulation we run. If you want the full construction of $W_t$ from coin flips, the sibling post on [Brownian motion from the random walk](/blog/trading/math-for-quants/brownian-motion-random-walk-math-for-quants) builds it from the ground up; here we take it as our source of noise and move on.

### The master equation: drift plus diffusion

Now we can write the general stochastic differential equation. A quantity $X_t$ follows the SDE

$$dX = a(X,t)\,dt + b(X,t)\,dW.$$

There are exactly two pieces, and naming them correctly is half the battle.

- $a(X,t)\,dt$ is the **drift**. It is the predictable, average direction of motion over the next instant — the part you would keep if you erased all the randomness. The function $a(X,t)$ is the *drift coefficient*; it can depend on the current value $X$ and on time $t$.
- $b(X,t)\,dW$ is the **diffusion**. It is the random kick — the noise term — scaled by the *diffusion coefficient* $b(X,t)$. The $dW$ is the Brownian increment from the paragraph above; $b$ controls how violent the jitter is, and it too can depend on $X$ and $t$.

That is the entire grammar. An SDE is a sentence with a drift word and a diffusion word. Geometric Brownian motion, Ornstein-Uhlenbeck, and Cox-Ingersoll-Ross are three sentences in this grammar, differing only in which functions you plug into $a$ and $b$.

![Stack showing an SDE step as current value plus a drift nudge plus a diffusion kick](/imgs/blogs/sdes-gbm-ou-cir-math-for-quants-2.png)

The stack above is the mental model for a single step, and it is worth holding onto for the whole post. You start at the current value $X_t$. You add the drift — a predictable nudge in the average direction, proportional to the size of the time step. Then you add the diffusion — a random kick whose size is set by $b$ and by that square-root-of-time scaling. The sum is the next value. Drift is the steering wheel; diffusion is the bumpy road. A trending stock has a steering wheel pointed uphill; a mean-reverting spread has a steering wheel that always turns back toward home; a positive interest rate has a road that gets smoother the closer you get to the cliff edge at zero. Same machine, different settings.

### A note on existence and uniqueness (why these equations have answers at all)

A fair worry: we just wrote down an equation with randomness in it and declared it has a solution — a well-defined random path $X_t$. Does it? For the SDEs in this post, yes, and the reason is a clean theorem worth knowing in plain terms. If the drift and diffusion functions $a$ and $b$ are *not too wild* — specifically, if they do not change too fast as $X$ changes (a *Lipschitz* condition: the change in $a$ is bounded by a constant times the change in $X$) and do not blow up too quickly (a *linear growth* condition), then the SDE has a unique solution path for each realization of the noise. "Unique" here means: fix the random shocks, and the path is pinned down with no ambiguity.

Why care as a practitioner? Two reasons. First, it tells you the model is well-posed — your simulation is approximating something real, not chasing a phantom. Second, it warns you where models get dangerous. CIR's diffusion is $\sigma\sqrt{r}$, and $\sqrt{r}$ has an *infinite* slope at $r = 0$ — it violates the nice Lipschitz condition right at the boundary. That is not a bug; it is exactly the feature that lets the process kiss zero and bounce, and it is also why naive simulation of CIR can accidentally produce negative rates and crash. The theory flags the spot where you must be careful before your code does. We will come back to this in the CIR section with a fix.

## The Euler-Maruyama scheme: turning an equation into a path

An SDE is a statement about infinitesimal steps. A computer cannot take infinitesimal steps. So we approximate: chop time into many small-but-finite steps of length $\Delta t$, and at each step apply the drift-plus-diffusion rule with the infinitesimals replaced by finite quantities. This is the **Euler-Maruyama scheme**, the workhorse of every Monte Carlo simulation in quant finance, and it is just one line.

Take the master SDE $dX = a(X,t)\,dt + b(X,t)\,dW$. Pick a step size $\Delta t$ (say one trading day, $1/252$ of a year). At step $n$, with current value $X_n$, the next value is

$$X_{n+1} = X_n + a(X_n, t_n)\,\Delta t + b(X_n, t_n)\,\sqrt{\Delta t}\,Z_n,$$

where $Z_n$ is a fresh draw from the standard normal distribution (mean 0, variance 1) at each step. Stare at this and you will see exactly the stack from the last figure. The first term carries $X_n$ forward. The second term is the drift: coefficient times time step. The third term is the diffusion: coefficient times $\sqrt{\Delta t}$ times a random number. The $\sqrt{\Delta t}$, not $\Delta t$, is the whole subtlety — it is the square-root scaling of Brownian motion, and getting it wrong is the single most common simulation bug.

![Pipeline from an SDE through choosing a step, drawing a shock, stepping, and looping into a full path](/imgs/blogs/sdes-gbm-ou-cir-math-for-quants-3.png)

The pipeline above is the recipe in five moves. Start with the SDE. Discretize it by choosing a time step $\Delta t$. At each step, draw a standard normal shock $Z$. Apply the update equation to take one step. Loop that $N$ times, collecting the values as you go. The result is a full simulated path — and once you have a path of prices, you have a path of dollar P&L. Run the pipeline a thousand times with different random shocks and you have a thousand possible futures, which is exactly what a Monte Carlo option pricer or a risk system does.

Two practical truths about Euler-Maruyama before we use it. First, it is an *approximation*, and the error shrinks as $\Delta t$ shrinks — formally, the scheme has *strong order 1/2*, meaning the typical path error scales like $\sqrt{\Delta t}$. Halving the step roughly cuts path error by $\sqrt{2}$. For pricing where you only need the *distribution* at the end (not the exact path), the *weak* order is 1, which is better. Second, for some SDEs there is an exact solution you should use instead of stepping — geometric Brownian motion is one, and we will see that simulating GBM's *logarithm* sidesteps the approximation entirely. Euler-Maruyama is the universal tool; exact schemes are the sharp tool when you have one.

Here is the scheme in runnable Python, which we will reuse for all three processes:

```python
import numpy as np

def euler_maruyama(x0, drift, diffusion, T, n_steps, seed=0):
    """Simulate one path of dX = drift(x,t) dt + diffusion(x,t) dW."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    x = np.empty(n_steps + 1)
    x[0] = x0
    for n in range(n_steps):
        t = n * dt
        z = rng.standard_normal()              # fresh N(0,1) shock
        x[n + 1] = (x[n]
                    + drift(x[n], t) * dt        # drift term
                    + diffusion(x[n], t) * np.sqrt(dt) * z)  # diffusion term
    return x
```

Note the `np.sqrt(dt)` on the diffusion line and the plain `dt` on the drift line. That asymmetry *is* the mathematics. Everything that follows is choosing the `drift` and `diffusion` functions.

### How small must the step be? The bias-versus-cost trade-off

A natural question once you have the scheme: how fine should $\Delta t$ be? There is a real trade-off here, and it is worth feeling it concretely. Make $\Delta t$ smaller and each step hews closer to the true continuous dynamics, so the approximation bias falls — but you need more steps to cover the same horizon, so the simulation costs more. Make $\Delta t$ bigger and you finish faster but you accept more bias, and near a boundary (CIR's zero) a coarse step can overshoot into impossible territory. The art is choosing the coarsest step that is still accurate enough for the decision you are making.

A concrete rule of thumb helps. For a quantity that moves on a yearly clock — a stock, a rate — a daily step ($\Delta t = 1/252$) is plenty for pricing where you only care about the distribution at expiry, because Euler-Maruyama's *weak* order is 1: the error in expected payoffs falls roughly linearly in $\Delta t$, and a daily step already pins the mean and variance tightly. But if you are pricing a *barrier* option — one that knocks out if the path ever touches a level — the path between your sampled points matters, the true process can sneak across the barrier and back unseen between two daily points, and you may need an intraday step or a special barrier-correction to avoid systematically mispricing it. The same scheme, the same SDE; the *question you are asking of the path* sets how fine you must go. The everyday analogy: a coarse step is photographing a hummingbird once a second — you get the gist of where it went but miss every wingbeat; a fine step is filming at a thousand frames a second — exact, but you store a thousand times the data. You film at the rate the question demands and no finer.

A second practical guard: always seed your random number generator (the `seed=0` argument above) so a simulation is reproducible. A risk number you cannot reproduce is a risk number you cannot debug, and "the P&L distribution looks different every time I run it" is the first sign of an unseeded or mis-seeded engine. Production Monte Carlo engines run thousands of *independent* paths — each with its own shock sequence — but each individual run must be reproducible from its seed.

#### Worked example: simulating one GBM step on a \$100 stock

Let us do the very first step of a geometric Brownian motion by hand, with numbers. We model a stock at $S_0 = \$100$ with an expected return (drift) of $\mu = 10\%$ per year and volatility $\sigma = 20\%$ per year. We take a daily step, so $\Delta t = 1/252 \approx 0.003968$ years. The GBM drift coefficient is $a = \mu S$ and the diffusion coefficient is $b = \sigma S$, so the Euler-Maruyama update is

$$S_1 = S_0 + \mu S_0\,\Delta t + \sigma S_0 \sqrt{\Delta t}\,Z.$$

First the drift piece: $\mu S_0\,\Delta t = 0.10 \times \$100 \times 0.003968 = \$0.0397$. That is the average nudge for one day — about four cents on a hundred-dollar stock. Tiny, as it should be: a 10% annual return spread over 252 days is almost nothing per day.

Now the diffusion piece. Suppose today's random shock comes out to $Z = +0.5$ (a half-standard-deviation up day). The diffusion term is $\sigma S_0 \sqrt{\Delta t}\,Z = 0.20 \times \$100 \times \sqrt{0.003968} \times 0.5 = 0.20 \times \$100 \times 0.06300 \times 0.5 = \$0.630$. So the random kick is about 63 cents — *sixteen times larger* than the drift nudge. Add them up:

$$S_1 = \$100 + \$0.0397 + \$0.630 = \$100.67.$$

If you held one share, your one-day P&L is $+\$0.67$. Notice how thoroughly the randomness dominates the predictable drift over a single day — that 63-cent kick swamps the 4-cent drift. This is the square-root scaling in action: over one day, $\sqrt{\Delta t}$ is much larger relative to $\Delta t$ than it will be over a year. The lesson a quant internalizes here is that **over short horizons, noise is the story and drift is a rounding error; only as the horizon grows does drift catch up and dominate.** That single fact is why you cannot tell a good trader from a lucky one in a week, and can in a decade.

## Geometric Brownian motion: the price model

Now we commit to specific drift and diffusion functions. **Geometric Brownian motion** (GBM) is the SDE

$$dS = \mu S\,dt + \sigma S\,dW.$$

Compare it to the master form: the drift is $a(S) = \mu S$ and the diffusion is $b(S) = \sigma S$. The key word is *proportional* — both the average move and the random kick are proportional to the current price $S$. That single design choice is what makes GBM the natural model for an asset price, and it encodes a deep economic truth.

### Why proportional? Because returns, not dollars, are stable

A \$10 stock and a \$1,000 stock do not have the same dollar volatility — the \$1,000 stock might move \$20 in a day while the \$10 stock moves 20 cents — but they can easily have the same *percentage* volatility, say 2%. Volatility lives in returns, not in dollars. GBM bakes this in: divide the SDE by $S$ and you get $dS/S = \mu\,dt + \sigma\,dW$, which reads "the *return* over the next instant has average $\mu\,dt$ and random part $\sigma\,dW$." The percentage move is what is stable; the dollar move scales with the price. This is also why GBM prices can never go negative — a price near zero has a near-zero drift *and* a near-zero kick, so it can approach zero but never cross it. A model that let prices go negative would be embarrassing; GBM rules it out by construction.

### The exact solution and the famous $-\sigma^2/2$ correction

GBM is special: we can solve it exactly, no Euler-Maruyama needed. Applying Itô's lemma (the chain rule for stochastic calculus — see [Itô's lemma for quant interviews](/blog/trading/quantitative-finance/itos-lemma-quant-interviews) for the derivation) to $\ln S$ gives the closed-form solution

$$S_t = S_0\,\exp\!\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)t + \sigma W_t\right].$$

Two things deserve a beginner's full attention. First, the price is the exponential of something normal — so $S_t$ is **lognormal**. That is why option pricing assumes lognormal prices: it falls straight out of GBM. Second, and this trips up nearly everyone the first time: the growth rate inside the exponential is $\mu - \tfrac12\sigma^2$, *not* $\mu$. There is a subtraction of half the variance. This is the famous **volatility drag** or **Itô correction**, and it is not a typo — it is real money.

Where does the $-\sigma^2/2$ come from? Intuitively: a percentage gain and an equal percentage loss do not cancel. Lose 10% then gain 10% and you are at $0.90 \times 1.10 = 0.99$ — down 1%, not flat. Volatility eats compounded returns, and the bite is exactly $\tfrac12\sigma^2$ per unit time. So the *median* price grows at rate $\mu - \tfrac12\sigma^2$ even though the *average* price grows at $\mu$. A quant who confuses arithmetic average return $\mu$ with the compound growth rate $\mu - \tfrac12\sigma^2$ will overstate long-run wealth, sometimes badly. We have a deeper treatment of this exact effect in the post on [convexity and Jensen's inequality](/blog/trading/math-for-quants/convexity-jensen-math-for-quants); for GBM, just remember the median compounds slower than the mean, and the gap is half the variance.

### Simulating GBM step by step

Because GBM has an exact solution for $\ln S$, the *clean* way to simulate it is to step the logarithm, which is exact at any step size:

$$\ln S_{n+1} = \ln S_n + \left(\mu - \tfrac12\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\,Z_n.$$

Exponentiate to get $S_{n+1}$. In code:

```python
def gbm_path(s0, mu, sigma, T, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    log_s = np.empty(n_steps + 1)
    log_s[0] = np.log(s0)
    for n in range(n_steps):
        z = rng.standard_normal()
        log_s[n + 1] = (log_s[n]
                        + (mu - 0.5 * sigma**2) * dt   # drift with Ito correction
                        + sigma * np.sqrt(dt) * z)     # diffusion
    return np.exp(log_s)
```

The `- 0.5 * sigma**2` is the volatility drag, hard-coded into the drift of the log price. Use this exact-log scheme for GBM; reserve the raw Euler-Maruyama scheme for SDEs that have no closed form, which is most of them.

![Timeline of a single simulated GBM path from 100 dollars rising to 106 over 60 days](/imgs/blogs/sdes-gbm-ou-cir-math-for-quants-7.png)

The timeline above traces one realized GBM path day by day, the kind the code prints out. It starts at exactly \$100.00. Day 1 catches an up shock to \$100.83; day 2 a down shock pulls it to \$99.71; by day 20 the drift and a run of luck have it at \$102.40; a day-40 pullback to \$101.10 reminds you the noise never sleeps; and by day 60 it ends at \$106.20. That is a clean +6.2% over a quarter, but read the wiggle, not just the endpoints: on a daily basis the random kicks dwarf the drift, exactly as our hand calculation predicted, and the upward tilt only becomes visible because we let 60 of those days accumulate. Run a different random seed and the endpoint could just as easily be \$94 — the *drift* is a tendency, not a promise.

## The Ornstein-Uhlenbeck process: mean reversion

GBM wanders off. Some quantities do not — they keep getting pulled back to a level. The price *gap* between two near-identical stocks, the deviation of an interest rate from its long-run average, a volatility index hovering around a typical value: these mean-revert. The SDE that captures this is the **Ornstein-Uhlenbeck process** (OU):

$$dX = \theta(\mu - X)\,dt + \sigma\,dW.$$

The drift is $a(X) = \theta(\mu - X)$ and the diffusion is $b = \sigma$ (a constant). Read the drift slowly, because it is the whole idea. When $X$ is *above* the long-run mean $\mu$, the quantity $(\mu - X)$ is negative, so the drift is negative — the process is pushed *down*, back toward $\mu$. When $X$ is *below* $\mu$, the quantity $(\mu - X)$ is positive, so the drift pushes *up*. The drift always points home, and the strength of the pull is proportional to how far away you are, scaled by $\theta$ (theta), the **speed of mean reversion**. A big $\theta$ yanks the process back fast; a small $\theta$ lets it drift far before reeling it in.

This is the everyday analogy of a spring, or a thermostat. Pull a spring far from rest and it pulls back hard; nudge it a little and it pulls back gently. The diffusion term $\sigma\,dW$ is the constant random buffeting that keeps knocking it away from rest, and the whole process is the tug-of-war between the spring (drift) and the buffeting (diffusion).

### The half-life: the one number that runs a stat-arb book

Here is why the OU process is beloved by statistical-arbitrage traders. The strength of the pull, $\theta$, has units of "per unit time," which is hard to feel. So we convert it to a **half-life** — the expected time for a deviation from the mean to shrink to half its size, with the randomness averaged out. The math is clean: ignore the noise, and the deviation decays as $e^{-\theta t}$, so it halves when $e^{-\theta t} = 1/2$, i.e.

$$t_{1/2} = \frac{\ln 2}{\theta} \approx \frac{0.693}{\theta}.$$

The half-life is the single most useful number you can attach to a mean-reverting trade. It tells you, roughly, your holding period: if a spread is stretched and you bet it snaps back, the half-life says how many days until you have captured half the move. A two-day half-life is a fast intraday strategy; a forty-day half-life is a patient, capital-intensive one. It also sizes your risk: a longer half-life means more time exposed to the buffeting before you get paid, hence more variance in the outcome.

### The stationary distribution: where OU lives in the long run

Unlike GBM, which wanders off forever, OU settles into a **stationary distribution** — a fixed bell curve it bounces around once the initial condition is forgotten. It is normal, centered at the mean $\mu$, with variance

$$\text{Var}(X_\infty) = \frac{\sigma^2}{2\theta}.$$

Notice the trade-off baked in: more buffeting ($\sigma$ up) widens the long-run spread, but a stronger spring ($\theta$ up) tightens it. A spread with high noise and a weak pull rattles around in a wide band; a spread with a strong pull stays close to home. This stationary variance is what you standardize against to build a *z-score* — how many standard deviations the spread currently sits from its mean — which is the actual trading signal. Our deep-dive on [cointegration and pairs trading](/blog/trading/math-for-quants/cointegration-pairs-trading-math-for-quants) builds the full z-score trading rule on top of exactly this OU machinery.

It is worth pausing on what "stationary" buys you, because it is the deepest practical difference between OU and GBM. A stationary process has a fixed, knowable long-run distribution: ask "what is the chance this spread is more than \$1.50 from its mean?" and the answer is the same today, next month, and next year, because the process keeps forgetting where it started and settling back into the same bell curve. That is what makes the z-score meaningful — there is a stable yardstick to measure against. GBM has no such yardstick: its distribution keeps widening with time (the variance of $\ln S_t$ grows like $\sigma^2 t$), so "how far is the price from its mean" is a question whose answer drifts forever. You can build a *mean-reversion* trade on a stationary process because there is a fixed home to revert to; you cannot build the same trade on GBM because there is no home. This is why the very first thing a stat-arb researcher does with a candidate spread is *test whether it is stationary at all* — if it is not, there is nothing to revert to and the half-life is meaningless. The companion post on [stationarity and autocorrelation](/blog/trading/math-for-quants/stationarity-autocorrelation-math-for-quants) is the formal test; OU is the model you fit once the test passes.

### Estimating OU parameters from data

In practice you do not know $\theta$, $\mu$, and $\sigma$ — you estimate them from a history of the spread. The cleanest route exploits a fact we will use again: sampled at fixed intervals $\Delta t$, an OU process is *exactly* an AR(1) autoregression, $X_{n+1} = c + \phi X_n + \text{noise}$, with $\phi = e^{-\theta \Delta t}$. So you run a simple linear regression of each value on the previous value, read off the slope $\phi$, and invert: $\theta = -\ln(\phi)/\Delta t$. The intercept gives you the mean $\mu = c/(1-\phi)$, and the spread of the regression residuals gives you $\sigma$. A slope $\phi$ near 1 means slow reversion (long half-life); a slope near 0 means lightning-fast reversion. This is why an OU fit and an AR(1) fit are two faces of the same coin — the discrete sibling and the continuous one — and why the autoregression toolkit transfers directly to continuous-time mean reversion.

### Simulating OU with Euler-Maruyama

OU has a constant diffusion, so Euler-Maruyama is clean and the standard choice:

```python
def ou_path(x0, theta, mu, sigma, T, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    x = np.empty(n_steps + 1)
    x[0] = x0
    for n in range(n_steps):
        z = rng.standard_normal()
        x[n + 1] = (x[n]
                    + theta * (mu - x[n]) * dt        # mean-reverting drift
                    + sigma * np.sqrt(dt) * z)        # constant diffusion
    return x
```

The only change from the generic scheme is the drift line: `theta * (mu - x[n])` instead of GBM's `mu * x[n]`. That swap — from a drift that points *away* in proportion to the price, to a drift that points *home* in proportion to the distance from the mean — is the entire difference between a trending model and a reverting one.

#### Worked example: OU half-life and the holding period of a spread

You are a pairs trader watching the spread between two refiners. You fit an OU process to the spread's history and estimate a mean-reversion speed of $\theta = 0.0578$ per day. What is the half-life, and what does it mean for your trade?

Plug in: $t_{1/2} = \ln 2 / \theta = 0.693 / 0.0578 = 11.99 \approx 12$ days. So a deviation from the mean is expected to shrink by half in about 12 trading days. Suppose the spread is currently stretched to \$2.00 above its mean and you sell it, betting on reversion. After about 12 days you expect roughly half the gap — about \$1.00 — to have closed, and after another 12 days, half of *what remains* (down to about \$0.50). On a position sized to make \$10,000 per \$1.00 of convergence, your first half-life captures about \$10,000 of the \$20,000 total, and you are about two and a half half-lives — call it a month — from harvesting most of it.

Now sanity-check the risk. The stationary standard deviation tells you how wide the spread normally rattles; entering at \$2.00 when the typical deviation is, say, \$0.80 means you entered at about $+2.5$ standard deviations, a genuinely stretched level. The intuition to carry away: **the half-life converts the abstract reversion speed $\theta$ into a concrete holding period and capital plan — it is the difference between a trade you can run a hundred times a year and one you can run three times.** A faster $\theta$ (shorter half-life) means more turnover, more independent bets, and a higher Sharpe ratio for the same edge per trade.

![Before-after contrast of a trending GBM path against a mean-reverting OU path from the same start](/imgs/blogs/sdes-gbm-ou-cir-math-for-quants-4.png)

The before-after figure above puts the two personalities side by side, both starting at \$100. The GBM panel on the left drifts away: \$108 at day 60, \$119 at day 120, with no anchor pulling it back — its "home" is wherever it happens to be. The OU panel on the right wanders too, but it has a leash: \$103 at day 60, \$99 at day 120, always tugged back toward its mean of \$100. This is the contrast that decides which model you reach for. If the thing you are modeling has no natural level — a single stock's price, an exchange rate over years — you want GBM's wandering drift. If it has a level it keeps returning to — a spread, a rate's deviation, a volatility index — you want OU's homing drift. Picking the wrong one is not a small error; it is modeling a spring as a rocket.

#### Worked example: GBM versus OU from the same starting point

Let us make the contrast above quantitative. Two series both start at \$100. The GBM has $\mu = 8\%$, $\sigma = 18\%$. The OU has mean $\mu = \$100$, speed $\theta = 5$ per year (half-life $0.693/5 = 0.139$ years $\approx 35$ days), and $\sigma$ chosen so its stationary standard deviation is also meaningful. Feed both the *same* sequence of random shocks and watch the divergence.

Suppose the first few standard-normal shocks are $+1.0, +0.8, -0.3, +1.2$ on daily steps ($\Delta t = 1/252$). For the GBM, every up shock compounds onto a price that has no reason to come back: after a run of positive shocks it sits at, say, \$108 by day 60 and keeps climbing to \$119 by day 120, exactly as the figure shows. For the OU, the *same* up shocks push it above \$100, but the drift immediately fights back — at \$103 the drift is $\theta(\mu - X)\Delta t = 5 \times (100 - 103) \times (1/252) = -\$0.060$ per day, a steady downward pull that erases the excursion over a few half-lives, landing it back near \$99-\$100. Same noise, opposite destinies. The dollar lesson: with GBM your P&L from a long position is *path-dependent and unbounded* — you can ride a trend to the moon or watch it sink — while with OU your P&L from a reversion trade is *bounded and mean-seeking* — you are betting on the rubber band, and the half-life tells you when it snaps back. **Same equation grammar, opposite trading strategies, decided entirely by the sign and form of the drift.**

![Tree of the SDE model family branching from a general diffusion into trending and reverting drifts](/imgs/blogs/sdes-gbm-ou-cir-math-for-quants-5.png)

The tree above organizes the whole family so the relationships stop being a list and become a structure. At the root sits the general SDE — drift plus diffusion. The first branch is the single most important question you ask of any quantity: does its drift *trend* or *revert*? Down the trending branch you find GBM (drift proportional to the level, for stock prices) and plain Brownian motion with drift (constant drift, for quantities that can go negative, like a log-spread). Down the reverting branch you find OU (constant diffusion, for spreads and rate deviations) and CIR (square-root diffusion, for quantities that must stay positive). When you face a new modeling problem, you walk this tree: trend or revert, then can it go negative or not. Two questions, and you have picked your SDE.

## The Cox-Ingersoll-Ross process: mean reversion that stays positive

OU has one flaw for interest rates and variances: with its constant diffusion $\sigma\,dW$, a big enough down shock can push the process below zero. For a stock spread that is fine — spreads can be negative. But an interest rate going negative was long considered impossible (and even in the era of negative rates, a *variance* certainly cannot be negative). We need mean reversion *and* a hard floor at zero. The **Cox-Ingersoll-Ross process** (CIR) delivers it:

$$dr = \theta(\mu - r)\,dt + \sigma\sqrt{r}\,dW.$$

The drift is identical to OU: $\theta(\mu - r)$, the same homing spring. The only change is the diffusion: $b(r) = \sigma\sqrt{r}$ instead of a constant $\sigma$. That little square root is doing enormous work, and once you see how, you will never forget it.

### Why $\sqrt{r}$ keeps rates positive

Watch what happens to the random kick as the rate $r$ falls toward zero. The diffusion coefficient is $\sigma\sqrt{r}$. When $r = 0.04$ (a 4% rate), the kick scales with $\sqrt{0.04} = 0.20$. When $r$ drops to $0.01$ (1%), the kick scales with $\sqrt{0.01} = 0.10$ — half as violent. When $r$ approaches $0.0001$, the kick scales with $0.01$ — almost nothing. **The closer the rate gets to zero, the smaller the random shocks become, until at exactly zero there is no random kick at all.** Meanwhile the drift, $\theta(\mu - 0) = \theta\mu > 0$, is firmly positive and pushes the rate back up. So zero is a place the process can approach but where the noise switches off and the drift shoves it back. The square root is a self-tightening floor.

![Before-after showing how a constant kick can go negative while a square-root kick stays positive](/imgs/blogs/sdes-gbm-ou-cir-math-for-quants-6.png)

The before-after figure above shows the mechanism directly. On the left is the OU-style constant kick: at a rate of 0.5%, the shock size $\sigma$ is the same as it would be at any other level, so a big down shock can blow straight through zero into negative territory — a broken rate. On the right is the CIR square-root kick: at the same 0.5% rate, the shock size is $\sigma\sqrt{r}$, which has already shrunk because $r$ is small, and it keeps shrinking as the rate falls, so the rate stays positive. The two panels share a starting rate and differ only in how the noise scales with the level — yet that one difference is the line between a model that can embarrass you with a negative interest rate and one that cannot. This is exactly the spot the existence-and-uniqueness theory warned us about: $\sqrt{r}$ has infinite slope at zero, which is *why* it can pin the boundary.

### The Feller condition: when zero is truly unreachable

The square root keeps the rate from going negative, but can it touch zero exactly, even briefly? The answer is a crisp inequality called the **Feller condition**. Zero is unreachable — the rate stays strictly positive at all times — if and only if

$$2\theta\mu \ge \sigma^2.$$

Read it as a contest between the upward force and the noise. The left side, $2\theta\mu$, is the strength of the drift pushing away from zero (speed times mean, doubled). The right side, $\sigma^2$, is the intensity of the noise trying to drag the rate down to the floor. If the drift wins ($2\theta\mu \ge \sigma^2$), the process never reaches zero. If the noise wins ($2\theta\mu < \sigma^2$), the process can hit zero and instantaneously reflect back up (it never goes negative, but it can kiss the floor). For a quant calibrating a CIR model to market data — for short rates, or for the variance process inside the Heston stochastic-volatility model — checking the Feller condition is a standard sanity step. When it fails, your simulation will spend time near zero and naive Euler-Maruyama can produce a tiny negative $r$, whose $\sqrt{r}$ is then undefined and your code crashes. The standard fix is the *full truncation* scheme: replace $\sqrt{r}$ with $\sqrt{\max(r, 0)}$ at each step.

### Simulating CIR safely

```python
def cir_path(r0, theta, mu, sigma, T, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    r = np.empty(n_steps + 1)
    r[0] = r0
    for n in range(n_steps):
        z = rng.standard_normal()
        r_pos = max(r[n], 0.0)                 # full-truncation guard
        r[n + 1] = (r[n]
                    + theta * (mu - r[n]) * dt          # mean-reverting drift
                    + sigma * np.sqrt(r_pos) * np.sqrt(dt) * z)  # sqrt diffusion
    return r
```

The single line `r_pos = max(r[n], 0.0)` is the whole defense against the boundary. It is not just defensive coding — it is the discretized acknowledgement that the continuous process never goes negative but its finite-step approximation might overshoot, and you must catch it.

### The level-dependent volatility CIR gives you for free

There is a second, subtler gift in the $\sqrt{r}$ term that practitioners prize: it makes volatility *rise with the level* in a way that matches reality. Because the diffusion is $\sigma\sqrt{r}$, the *variance* of the random kick over a step is proportional to $r$ itself — high rates jiggle more in absolute terms, low rates jiggle less. This is empirically true of interest rates: a move from 5% to 5.2% (a 20-basis-point day, where a basis point is one hundredth of a percent) is unremarkable, while a 20-basis-point day on a 0.3% rate would be a once-a-decade event. OU, with its constant $\sigma$, says a 20-basis-point day is equally likely at 5% and at 0.3% — clearly wrong. CIR's square-root diffusion bakes the right behaviour in for free: the noise scales with the level, so the model breathes the way the market does. When you fit CIR to a rate history and OU to the same history, the CIR fit will usually leave cleaner residuals precisely because it captured this level-dependence; the OU fit will show "fatter" surprises at high rates and "thinner" ones at low rates that it cannot explain.

This is also the reason the same square-root process is the natural choice for *variance* in equity models. Realized volatility is famously higher when volatility is already high — a calm market stays calm, a panicked market swings violently — which is exactly variance-proportional-to-level behaviour. Plug a CIR process in for the variance and you have captured both mean reversion (volatility comes back to a long-run level) and level-dependent buffeting (high-vol regimes swing harder) in one tidy SDE. That is the structural reason CIR outlived its original rates application and became the engine of stochastic-volatility pricing.

#### Worked example: a CIR Feller-condition check with numbers

You are calibrating a CIR short-rate model. Your estimates are a long-run mean rate $\mu = 4\% = 0.04$, a mean-reversion speed $\theta = 0.3$ per year, and a volatility parameter $\sigma = 0.10$. Does the Feller condition hold — can the rate ever hit zero?

Compute the left side: $2\theta\mu = 2 \times 0.3 \times 0.04 = 0.024$. Compute the right side: $\sigma^2 = 0.10^2 = 0.01$. Compare: $0.024 \ge 0.01$ is **true**, so the Feller condition holds and the modeled rate stays strictly positive — it will never touch zero. Good: your simulated rates will behave, and you can trust paths near the floor.

Now stress it. Suppose your volatility estimate is actually higher, $\sigma = 0.18$. Recompute the right side: $\sigma^2 = 0.0324$. Now $2\theta\mu = 0.024 < 0.0324 = \sigma^2$ — the condition *fails*. With this noisier calibration the rate can reach zero and reflect, and a naive simulation could throw a `sqrt` error on a negative value; you would need the full-truncation guard above. Put it in dollar terms: if you are pricing a \$10 million floating-rate note whose coupon resets off this short rate, a model that occasionally drops the rate to zero versus one that floors it cleanly produces materially different coupon paths and present values. The intuition: **the Feller condition is a one-line test that tells you whether your positivity assumption is safe or whether your noise is strong enough to crash into the floor — check it before you trust a single simulated path.** This same machinery, applied to a *variance* rather than a rate, is the heart of the Heston model and connects directly to the [short-rate models like Vasicek and Hull-White](/blog/trading/quantitative-finance/short-rate-models-vasicek-hull-white) used across fixed-income desks.

## Putting the three together: one grammar, three personalities

Step back and admire how little actually changed across the three models. All three are $dX = a(X,t)\,dt + b(X,t)\,dW$. GBM chose drift $\mu S$ and diffusion $\sigma S$ — proportional to the level — and got a trending, always-positive price. OU chose drift $\theta(\mu - X)$ and diffusion $\sigma$ — homing pull, constant noise — and got a mean-reverting quantity that can go negative. CIR kept OU's homing drift but switched the diffusion to $\sigma\sqrt{r}$ and got mean reversion with a hard floor at zero. The drift decides *trend versus revert*; the diffusion decides *how the noise scales with the level*, which in turn decides *whether the process can go negative*.

That is the entire conceptual payload, and it generalizes. Want a process that reverts to a *time-varying* target (because the yield curve says rates should rise next year)? Make $\mu$ a function of $t$ — that is the Hull-White model. Want a stock whose volatility itself reverts and stays positive? Run a CIR process for the variance and feed it into the stock's GBM — that is Heston. Want jumps for crash risk? Add a third term to the SDE for sudden discontinuous moves — that is jump-diffusion. Every famous model in derivatives is a remix of drift and diffusion choices on the same master equation. Once you can read $a$ and $b$, you can read the whole zoo.

### A side-by-side comparison

| Property | GBM | OU | CIR |
|---|---|---|---|
| SDE | $dS = \mu S\,dt + \sigma S\,dW$ | $dX = \theta(\mu{-}X)dt + \sigma dW$ | $dr = \theta(\mu{-}r)dt + \sigma\sqrt{r}\,dW$ |
| Drift | proportional, trending | homing (mean-reverting) | homing (mean-reverting) |
| Diffusion | $\propto$ level | constant | $\propto \sqrt{\text{level}}$ |
| Can go negative? | No (stays $> 0$) | Yes | No (floor at 0) |
| Long-run behavior | wanders, lognormal | stationary normal | stationary, positive |
| Key number | drift $\mu - \tfrac12\sigma^2$ | half-life $\ln 2/\theta$ | Feller $2\theta\mu \ge \sigma^2$ |
| Classic use | stock and FX prices | spreads, rate deviations | short rates, variance |
| Exact solution? | Yes (lognormal) | Yes (normal) | Yes (noncentral $\chi^2$) |

Keep this table near your keyboard. When someone hands you a quantity to model, three columns and a couple of questions get you to the right SDE before you write a line of code.

## Common misconceptions

**"The drift $\mu$ is the rate at which my money grows in GBM."** No — the *median* price compounds at $\mu - \tfrac12\sigma^2$, not $\mu$. The arithmetic mean grows at $\mu$, but you do not experience the mean; you experience a single path, and a single path compounds at the lower, volatility-dragged rate. A fund quoting a 12% expected return with 30% volatility has a median compound growth nearer $12\% - \tfrac12(0.30)^2 = 12\% - 4.5\% = 7.5\%$. Ignoring the drag is how backtests promise riches that live paths never deliver.

**"Mean reversion means the price will definitely come back, so reversion trades are nearly riskless."** Mean reversion is a *drift*, a tendency, not a guarantee on any given path. The diffusion term keeps buffeting the process, and over your finite holding period the noise can push the spread *further* from the mean before (or instead of) reverting. The half-life tells you the expected time to revert with the noise averaged out — but you trade one realization, not the average. Reversion trades blow up precisely when the spread keeps widening; that is the LTCM lesson.

**"Euler-Maruyama just needs a small time step and it is exact."** It is never exact — it is an approximation with error that *shrinks* as the step shrinks (strong order $\tfrac12$), but for a fixed step it carries bias, and near boundaries (like CIR's zero) it can produce impossible values you must guard against. When an exact scheme exists — stepping $\ln S$ for GBM — use it instead; you get the right distribution at any step size for free.

**"The diffusion term is what makes the price go up over time."** The opposite, on net. The diffusion has mean zero — it adds no expected direction. It is the *drift* that supplies the average direction. Worse, in GBM the volatility (diffusion strength) actively *subtracts* from compound growth through the $-\tfrac12\sigma^2$ term. Noise is not your friend in the long run; it is a drag.

**"GBM and a random walk are the same thing."** Close but importantly different. A random walk (or arithmetic Brownian motion) has *additive* shocks and can go negative; GBM has *multiplicative* shocks (proportional to the price) and cannot. Model a \$5 stock as a random walk and you will get negative prices in your simulation; model it as GBM and you will not. The multiplicative structure is exactly what makes GBM the right model for a price.

**"A higher mean-reversion speed $\theta$ always means a better trade."** A higher $\theta$ shortens the half-life, which is good — faster payoff, more turnover, more independent bets. But it also *tightens the stationary band* ($\text{Var} = \sigma^2/2\theta$ shrinks), meaning the spread deviates less from its mean in the first place, so each opportunity is smaller. The edge per trade and the number of trades pull in opposite directions; the Sharpe ratio, not $\theta$ alone, is what you optimize.

## How it shows up in real markets

**Black-Scholes option pricing (1973–today).** The entire Black-Scholes-Merton framework assumes the underlying stock follows GBM, $dS = \mu S\,dt + \sigma S\,dW$. The lognormal price distribution that pops out of GBM is what lets you write a closed-form option price, and the single free parameter — $\sigma$, the diffusion strength — is what the market quotes as *implied volatility*. Every options screen in the world is, behind the scenes, displaying the $\sigma$ that makes a GBM-based formula match the traded price. When traders say the model is "wrong but useful," what they mean is that real prices are not exactly GBM (they jump, their volatility wanders), yet GBM is the common language. See the dedicated post on [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) for the full pricing derivation.

**LTCM and the limits of mean reversion (1998).** Long-Term Capital Management built a book of convergence trades — bets that stretched spreads would revert, exactly the OU logic. The math was sound: the spreads *were* mean-reverting, with estimable half-lives. What the OU model does not promise is that the spread will not stretch *further* during your holding period, and in the 1998 Russian default crisis spreads blew out far beyond any historical band. The diffusion term overwhelmed the drift for long enough to force liquidation before reversion arrived. The lesson lives in the model: mean reversion is a drift, the noise is real, and leverage on a reversion trade is leverage on the assumption that the noise stays small.

**The Heston stochastic-volatility model (1993–today).** The CIR process found its second life not in rates but in *volatility*. Heston models a stock's variance $v_t$ as a CIR process, $dv = \kappa(\theta - v)\,dt + \xi\sqrt{v}\,dW$, precisely because variance must stay positive and tends to revert to a long-run level. The $\sqrt{v}$ keeps variance off the floor; the Feller condition $2\kappa\theta \ge \xi^2$ decides whether variance can hit zero. This single model produces the *volatility smile* — the empirical fact that out-of-the-money options trade richer than flat-volatility GBM would predict — and it is the workhorse for pricing exotic equity derivatives.

**Vasicek, CIR, and Hull-White short-rate models (1977–today).** Fixed-income desks model the short interest rate as a mean-reverting SDE. Vasicek used OU (simple, but allows negative rates); CIR used the square-root process (no negative rates, but harder math); Hull-White made the mean time-varying to fit today's yield curve exactly. The choice between them is the choice between OU's constant diffusion and CIR's square-root diffusion — the same fork we drew in the tree. The [short-rate models post](/blog/trading/quantitative-finance/short-rate-models-vasicek-hull-white) walks through how each calibrates to market bond prices.

**Pairs trading and statistical arbitrage (1980s–today).** When two cointegrated stocks' spread is fitted to an OU process, the half-life $\ln 2/\theta$ sets the strategy's entire rhythm: holding period, capital turnover, and how many independent bets you get per year. Desks screen thousands of candidate pairs and keep the ones whose spread has a short, stable half-life and a wide stationary band — fast reversion plus big deviations equals a high Sharpe. The [cointegration and pairs-trading deep-dive](/blog/trading/math-for-quants/cointegration-pairs-trading-math-for-quants) builds the full pipeline from two price series to a tradeable z-score on top of this OU model.

**Monte Carlo risk and pricing engines (everywhere).** Any time a desk needs the *distribution* of a portfolio's value at some future date — value-at-risk, the price of a path-dependent option, a stress test — it runs Euler-Maruyama (or an exact scheme) thousands or millions of times, one path per random seed, and reads the histogram of outcomes. The drift-plus-diffusion structure is what makes this possible: each path is just the update equation looped forward. When you hear "we ran a million paths," that is the pipeline figure executed a million times.

**The negative-rate era and model breakage (2014–2022).** When European and Japanese policy rates went below zero, CIR (and any model with a hard zero floor) suddenly mis-specified reality — rates were doing the thing the model swore was impossible. Desks shifted to *shifted* CIR (model $r + c$ for a constant displacement $c$) or back to OU-style Gaussian models that allow negative values. It is a clean real-world case of choosing the diffusion structure to match the world: the $\sqrt{r}$ floor was a feature for decades and a bug for a few years.

## When this matters to you and further reading

If you ever simulate a price, value an option, build a mean-reversion strategy, or model an interest rate, you are using one of these three SDEs, whether the library names it or not. The practical skill is small and durable: look at any quantity, ask "does its drift trend or revert?" and "can it go negative or not?", and those two questions hand you GBM, OU, or CIR. Then attach the one number that governs it — the volatility-dragged growth $\mu - \tfrac12\sigma^2$ for a price, the half-life $\ln 2/\theta$ for a spread, the Feller check $2\theta\mu \ge \sigma^2$ for a rate — and you can size a trade or sanity-check a model before you write any code.

A note of honesty, since this is a place where explanation can be mistaken for advice: these models are *deliberate simplifications*. Real prices jump; real volatility is itself stochastic; real spreads can stay irrational longer than you can stay solvent. Every SDE here makes money *and* loses money, and the diffusion term — the part with no expected direction — is exactly where the losses live. Treat the equations as a disciplined way to *reason* about uncertainty, not as a promise about any single path. None of this is investment advice; it is the machinery underneath it.

For the prerequisites, start with [Brownian motion from the random walk](/blog/trading/math-for-quants/brownian-motion-random-walk-math-for-quants) to see where $dW$ comes from, then [Itô's lemma for quant interviews](/blog/trading/quantitative-finance/itos-lemma-quant-interviews) for the chain rule that solves GBM. For the interview-style companion to this exact material — the kind of derivations and gotchas that come up in a quant interview — see [stochastic differential equations: GBM and OU for quant interviews](/blog/trading/quantitative-finance/stochastic-differential-equations-gbm-ou-quant-interviews). For where these SDEs go next, the [short-rate models post](/blog/trading/quantitative-finance/short-rate-models-vasicek-hull-white) extends CIR and OU into the fixed-income world, and the [cointegration and pairs-trading deep-dive](/blog/trading/math-for-quants/cointegration-pairs-trading-math-for-quants) turns the OU half-life into a live trading rule. Three equations, two slots each, and most of quantitative finance falls out.
