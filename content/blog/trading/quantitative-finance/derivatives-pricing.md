---
title: "Derivatives Pricing: From Replication to Risk-Neutral Measures, the Engineering Way"
date: "2026-05-02"
publishDate: "2026-05-02"
description: "A senior-quant deep dive into derivatives pricing: replication, no-arbitrage, the risk-neutral measure, three production engines (closed-form, PDE, Monte Carlo), greeks, calibration, XVAs, and a long catalog of pricing failures that cost real money."
tags:
  [
    "derivatives-pricing",
    "no-arbitrage",
    "replication",
    "risk-neutral-measure",
    "martingale",
    "monte-carlo",
    "pde-pricing",
    "quantitative-finance",
    "hedging",
    "numeraire",
    "python",
    "quantlib",
  ]
category: "trading"
author: "Hiep Tran"
featured: true
readTime: 50
---

Most introductions to derivatives pricing open with a definition: "a derivative is a contract whose value depends on an underlying." That definition is technically correct and operationally useless. It tells you nothing about why the desk across the room is willing to quote a price, what number that price is, or what happens to the firm if that number is wrong. The reason desks can quote at all is that they have learned how to *manufacture* the payoff out of liquid instruments — the price they quote is the cost of the manufacturing process, not a forecast.

![Pricing as manufacturing the payoff](/imgs/blogs/derivatives-pricing-1.png)

The diagram above is the mental model the rest of this article is built on. A client wants the payoff on the left. The desk does not own that payoff anywhere; it builds the payoff out of stock and bonds (or futures, or other options) in the middle. The price the desk charges is the construction cost — plus a spread for risk, funding, capital, and the trader's bonus. Everything else in derivatives pricing — the partial differential equations, the change of measure, the Monte Carlo, the XVAs — is a refinement of this one idea: *price equals the cost of the replicating hedge*.

This article is the deep-dive I wish someone had walked me through when I started on a derivatives desk. It works through the binomial tree, generalises to continuous time, derives the risk-neutral measure as a *consequence* of replication (not an assumption), surveys the three production pricing engines that real desks actually run, walks through greeks, calibration, and the XVA stack that turns the textbook price into a live quote, then closes with eight named pricing failures that each cost institutions real money. The audience is a senior quant or staff-level engineer who already knows option payoffs and elementary stochastic calculus and wants the engineering picture: which engine to reach for, where the model lies, what the desk actually does at 8am.

## 1. The real problem: a price that survives a hedging desk

When a junior asks "what is this option worth?" a senior trader will almost always reply: *worth to whom, and under what hedge?* The question "what is it worth" treats price as a property of the instrument; the question "what does it cost me to neutralise" treats price as a property of the *manufacturing process*. The second framing is the only one that survives contact with a real book.

Suppose the desk sells a one-year European call on stock $S$ with strike $K = 100$, current spot $S_0 = 100$. If the desk simply collects the premium and waits, the payoff at expiry is $\max(S_T - K, 0)$, which can be anywhere between zero and unbounded. The desk has taken on unbounded short-tail risk for a fixed premium. No risk committee will allow this.

The way out is to *hedge*: the desk buys a quantity $\Delta$ of the stock, funded by a bond position $B$, such that the value of the portfolio $\Pi = \Delta S + B$ moves *in lock-step* with the option. If you can choose $\Delta$ and $B$ such that $\Pi$ matches the option's payoff in every future state of the world, then the desk is *immunised*: option moves up, hedge moves up by the same amount, net P&L is zero. The premium the desk charges is no longer a guess about the future stock price; it is the cost of the hedge it is about to put on.

This is not a metaphor. It is literally how the desk operates. Every morning the system computes a $\Delta$, the trader buys or sells stock to that $\Delta$, and the residual P&L is what the model attributes to gamma, theta, vega, and unmodelled noise. If the model is right and the market is liquid, the residual is small. The price that comes out of replication is therefore the price that survives the hedging process — which is the only price the firm can defend in a P&L review.

There are three immediate consequences of this framing, all of which structure the rest of the article.

First: **the real-world probability of the stock going up does not appear in the price**. This is the single most counter-intuitive fact in derivatives pricing for outsiders. If you replicate the payoff, the price is determined by *what the hedge costs to set up*, which depends on today's spot, today's interest rate, and the volatility of the path — but not on the drift, because drift is hedged out by holding $\Delta$ stock.

Second: **the price depends on what you can trade**. Replication requires liquid hedging instruments. If the underlying is illiquid, or if a risk factor (jumps, stochastic volatility, default) cannot be hedged with traded instruments, the model is *incomplete* and many prices are consistent with no-arbitrage. The price the desk quotes in an incomplete market is no longer unique; it depends on the desk's risk preferences, capital cost, and the spread it can charge.

Third: **the price changes if the hedging environment changes**. When the 2008 funding crisis hit, the cost of borrowing money spiked. Suddenly the bond-funding leg of the replicating portfolio was much more expensive than the textbook risk-free rate suggested. Prices on long-dated trades moved by tens of basis points overnight, not because the underlying changed but because the manufacturing cost did. This is exactly what FVA (funding valuation adjustment) is, and we will return to it in §9.

The mental model — *price equals manufacturing cost of the replicating hedge* — is the single most important sentence in this article. Every formal result that follows is a refinement of this idea.

## 2. The one-period binomial: where every pricing instinct comes from

The cleanest place to see replication work is the simplest possible model: one period, two states. The stock starts at $S_0$. After one period it is either $S_u = u S_0$ (up) or $S_d = d S_0$ (down). There is a risk-free bond paying $1+r$ per period. The option pays $C_u$ in the up state and $C_d$ in the down state. We want to price the option today.

![One-period binomial: where every pricing instinct comes from](/imgs/blogs/derivatives-pricing-2.png)

Build the replicating portfolio: $\Delta$ shares of stock plus $B$ bonds. After one period, the portfolio is worth $\Delta S_u + B(1+r)$ in the up state and $\Delta S_d + B(1+r)$ in the down state. We want the portfolio to match the option in both states:

$$
\Delta S_u + B(1+r) = C_u, \qquad \Delta S_d + B(1+r) = C_d.
$$

Two equations, two unknowns. Subtract the second from the first to isolate $\Delta$:

$$
\Delta = \frac{C_u - C_d}{S_u - S_d}.
$$

This is the *delta hedge ratio*: the number of shares the desk must hold per option sold to match the option's response to a one-step move in the stock. Substitute back to get $B$. The price today is $C_0 = \Delta S_0 + B$ — the cost of putting on the hedge.

Now do the algebra one more step. Define $q = \frac{(1+r) - d}{u - d}$. The expression for $C_0$ rearranges into

$$
C_0 = \frac{1}{1+r} \big[ q \, C_u + (1-q) \, C_d \big].
$$

This is the punchline. The price equals a *discounted expected payoff*, where the expectation is taken under a probability $q$ that has nothing to do with the real-world probability of an up move. The real probability $p$ — the thing the equity research desk argues about all day — *never enters the calculation*. The replication argument has eliminated it.

The probability $q$ is called the *risk-neutral probability* (or the *equivalent martingale measure* in the continuous-time generalisation). It is the probability under which the discounted stock price $S/(1+r)^t$ is a martingale, i.e., its expected value tomorrow equals its value today. Under $q$, every traded asset earns the risk-free rate in expectation. Hence the name "risk-neutral": it is the world a hypothetical risk-neutral investor would believe in, but more importantly, it is the world in which the hedging argument lives.

Three observations a senior quant should drill into a junior:

1. **$q$ is not a real probability**. It is a pricing artefact derived from the absence of arbitrage. Confusing $q$ with the desk's view on direction is a categorical error and leads juniors to "discover" that the market has mispriced an option whenever their forecast disagrees with the implied $q$. It hasn't.
2. **$q$ exists if and only if there is no arbitrage**. If $u > 1+r > d$ fails — say, $1+r > u$ — then the bond strictly dominates the stock and there is a free lunch. The formula gives $q$ outside $[0, 1]$, which signals the inconsistency. This is a baby version of the First Fundamental Theorem of Asset Pricing, which we'll meet next.
3. **The hedge depends on the model, not on any real probability**. $\Delta$ is computed from $C_u, C_d, S_u, S_d$ — all model quantities. If the model is wrong (real states are not just up/down by $u$ and $d$), the hedge will not perfectly replicate, and the residual P&L is the model error.

A few lines of Python to make it concrete:

```python
def binomial_one_period(S0, K, u, d, r, payoff):
    """One-period binomial pricer with explicit replication."""
    Su, Sd = u * S0, d * S0
    Cu = payoff(Su)
    Cd = payoff(Sd)
    # Replication: solve 2x2 system for (Delta, B)
    Delta = (Cu - Cd) / (Su - Sd)
    B = (Cd - Delta * Sd) / (1.0 + r)
    C0_replication = Delta * S0 + B
    # Risk-neutral form: same answer, derived from the same hedge
    q = ((1.0 + r) - d) / (u - d)
    C0_risk_neutral = (q * Cu + (1.0 - q) * Cd) / (1.0 + r)
    return {
        "price_replication": C0_replication,
        "price_risk_neutral": C0_risk_neutral,
        "delta": Delta,
        "bond": B,
        "q": q,
    }


payoff_call = lambda S: max(S - 100.0, 0.0)
print(binomial_one_period(S0=100, K=100, u=1.10, d=0.90, r=0.05,
                          payoff=payoff_call))
## {'price_replication': 7.142857..., 'price_risk_neutral': 7.142857...,
##  'delta': 0.5, 'bond': -42.857..., 'q': 0.75}
```

Two methods, identical answer. The replication path is interpretable (you can see the hedge); the risk-neutral path is generalisable (you can iterate it). Real production pricers use the risk-neutral path because it scales; the replication path is the *interpretation* of why that path is correct.

The binomial tree generalises by chaining one-period steps backward in time. With $N$ steps, the option is priced by computing the payoff at the leaves and rolling backward through the tree, applying the one-step formula at every node. This converges to the Black–Scholes price as $N \to \infty$ when $u, d, r$ are calibrated to the GBM volatility, which we'll see in §3 of [the Black-Scholes article](/blog/trading/quantitative-finance/derivatives/black-scholes) when it is published.

For path-dependent or American options the tree handles them with a small modification: at each backward step, take the maximum of the continuation value (discounted expected next-step value) and the immediate exercise value. This single line — `value = max(continuation, intrinsic)` — is the entire engine for American options under a binomial model. Most production short-dated American pricers are still binomial trees because the convergence is acceptable and the code is unbreakable.

## 3. No-arbitrage as an engineering invariant

Step out of the binomial model and into the abstract framework. The First Fundamental Theorem of Asset Pricing (FTAP I) says: *a market is arbitrage-free if and only if there exists at least one equivalent martingale measure $Q$ such that all discounted traded asset prices are $Q$-martingales*. The Second Fundamental Theorem (FTAP II) says: *the market is complete if and only if $Q$ is unique*.

![FTAP I and II as engineering invariants](/imgs/blogs/derivatives-pricing-3.png)

For the desk these are not abstractions; they are operational invariants. FTAP I says the firm cannot quote consistent prices unless the prices are computable as $E^Q[\text{discounted payoff}]$ for *some* $Q$. If the prices on the screen are not consistent with any $Q$, somebody is mispricing and a careful arbitrageur can extract free money. FTAP II says that if the market is *complete* (every payoff is replicable using traded assets), then $Q$ is unique and prices are determined; if the market is *incomplete*, many $Q$'s are consistent with the same liquid quotes and they disagree on exotics.

Real markets are incomplete. The dimensions of incompleteness are predictable:

- **Stochastic volatility.** Volatility itself moves randomly. Vega risk cannot be perfectly hedged with the underlying alone; you need other options. Heston, SABR, and rough-vol models all live in this regime.
- **Jumps.** If the price can move discontinuously (Merton, Kou, Lévy), gap risk between rebalances is unhedgeable in a continuous-trading framework. The desk needs portfolios of options to hedge wingrisk.
- **Credit.** Default is a Bernoulli event with no continuous tradable. CVA modeling lives here.
- **Liquidity.** Funding spreads, bid-ask, and capacity all break the "frictionless" assumption.

The engineering consequence is that every desk runs a *family* of $Q$'s — one calibrated daily, one calibrated weekly, one stress-shifted — and looks at the spread of exotic prices across the family as a model-risk reserve. We'll see in §11 (Calibration) that this spread can be *enormous* for path-dependent products.

How to spot arbitrage in practice. The "free lunch on the screen" rarely looks like a textbook example. It looks like:

- **Cash-and-carry mispricing.** If the futures price on a non-dividend stock is not equal to $S_0 (1+r)^T$ within bid-ask, the cash-and-carry trade prints money. This was a real phenomenon in the 1980s; today algorithmic arbitrageurs close it within milliseconds, but a stale OTC quote can still be off by a basis point or two.
- **Currency triangle.** EUR/USD × USD/JPY ≠ EUR/JPY within bid-ask is a riskless triangle trade. FX dealers' systems enforce this consistency to the basis point.
- **Put-call parity violation.** $C - P = S - K e^{-rT}$ for European options on non-dividend stocks. If a quote violates this, the synthetic forward via options is mispriced relative to the underlying. Common after dividend announcements before quotes refresh.
- **Calendar inversion in the volatility surface.** Implied variance must be increasing in time (under Black–Scholes, with some technicalities). If a 3-month vol prints higher than a 6-month vol on the same strike, a calendar spread harvests the arbitrage.

The First FTAP is the engineering invariant: *if your pricing system produces a price inconsistent with no-arbitrage, the system is broken*, full stop. Every pricing library worth deploying has a cross-product of consistency checks (parity, calendar, butterfly) and refuses to quote when any of them fail.

## 4. From discrete to continuous: the SDE step

The binomial tree is conceptually clean but combinatorially nasty for many-period or many-asset problems. The standard trick is to take the limit as the number of periods goes to infinity and the step size goes to zero. The tree converges to a *stochastic differential equation*, and the chain rule we use to differentiate functions of the path acquires a new term: the *Itô correction*.

The geometric Brownian motion (GBM) SDE for stock $S_t$ is

$$
dS_t = \mu \, S_t \, dt + \sigma \, S_t \, dW_t,
$$

where $W_t$ is a standard Brownian motion, $\mu$ is the real-world drift, $\sigma$ is the volatility, and $dt$ is an infinitesimal time step. This is the continuous-time analogue of the binomial tree's up/down move with $u = e^{\sigma \sqrt{dt}}$ and $d = e^{-\sigma \sqrt{dt}}$.

The Itô correction is the single most important fact in continuous-time finance after replication itself. It says: when you take the chain rule of a function of $W_t$, you must include a term proportional to $dW_t^2 = dt$.

![Itô's lemma: the dt term ordinary calculus misses](/imgs/blogs/derivatives-pricing-4.png)

In ordinary calculus, $df(x) = f'(x) \, dx$. The chain rule has only the first-order term because for a smooth path $x(t)$, $dx^2 = (x'(t) dt)^2 = O(dt^2)$ vanishes faster than $dt$. For Brownian paths, that is *false*. Brownian paths have *finite quadratic variation*: over a time interval $[0,t]$, the sum $\sum (W_{t_{i+1}} - W_{t_i})^2$ converges to $t$, not to zero. This is what "$dW^2 = dt$" means as a calculus rule.

Itô's lemma:

$$
df(W_t) = f'(W) \, dW_t + \tfrac{1}{2} f''(W) \, dt.
$$

For a function of both $t$ and $W_t$:

$$
df(t, W_t) = \partial_t f \, dt + \partial_W f \, dW_t + \tfrac{1}{2} \partial_{WW} f \, dt.
$$

For GBM, applying Itô to $\log S_t$:

$$
d(\log S_t) = (\mu - \tfrac{1}{2}\sigma^2) \, dt + \sigma \, dW_t.
$$

The $-\tfrac{1}{2}\sigma^2$ term is the Itô correction. It is *deterministic* and *negative*: high volatility drags down the log return. This is why arithmetic-mean and geometric-mean returns differ; the Itô correction is the bridge.

Why does any of this matter operationally? Because *every option price is a function $V(t, S_t)$*, and Itô's lemma is the way you compute its differential. Apply Itô to the option value, set up the replicating portfolio, demand that the resulting portfolio is risk-free, and out pops the Black–Scholes PDE. We'll do this derivation at length in [the dedicated Black-Scholes post](/blog/trading/quantitative-finance/derivatives/black-scholes); for the derivatives-pricing overview, it is enough to know that the Itô correction is what makes option pricing *non-trivial*: without it, options would price like forwards.

A small but important warning for code: Itô calculus is *not* the same as Stratonovich calculus, which is the alternative integration rule used in physics. Itô integrals are non-anticipating (the integrand at time $t$ uses information up to but not including $t$), Stratonovich integrals are symmetric. Stratonovich gives a chain rule that *looks* like ordinary calculus, but at the cost of integrals that are not martingales. In finance we always use Itô, because we want martingale properties. In a noise-driven physical system you might use Stratonovich, because you want the chain rule to be classical. Conflating them is a source of bugs in cross-disciplinary code.

## 5. The risk-neutral measure and Girsanov

The risk-neutral probability we derived from the binomial tree generalises in continuous time via *Girsanov's theorem*. Girsanov says: under suitable conditions, you can change the probability measure from $P$ (the real-world / "physical" measure) to $Q$ (the risk-neutral measure) by *tilting the drift* of the Brownian motion. The Brownian paths themselves are unchanged; only the probability weights assigned to them are reweighted.

![Girsanov: the drift tilt that turns P into Q](/imgs/blogs/derivatives-pricing-5.png)

The technical statement: let $W_t^P$ be a Brownian motion under $P$, and let $\theta_t$ be a process satisfying Novikov's condition. Define

$$
\frac{dQ}{dP}\bigg|_{\mathcal{F}_t} = \exp\!\left(-\int_0^t \theta_s \, dW_s^P - \tfrac{1}{2} \int_0^t \theta_s^2 \, ds\right).
$$

Then $W_t^Q := W_t^P + \int_0^t \theta_s \, ds$ is a Brownian motion under $Q$. In other words, under $Q$, the original process has its drift reduced by $\theta_t$.

For GBM with drift $\mu$ and volatility $\sigma$, choose $\theta = (\mu - r)/\sigma$, the *market price of risk*. Then under $Q$,

$$
dS_t = r \, S_t \, dt + \sigma \, S_t \, dW_t^Q.
$$

The drift becomes the risk-free rate $r$. This is exactly the binomial tree's risk-neutral world generalised to continuous time. The discounted price $S_t / B_t$ is a $Q$-martingale, where $B_t = e^{rt}$ is the money-market account.

The intuition that helps me most. The same set of paths is being weighted differently under $P$ and $Q$. A path that has the stock going up a lot has high probability under $P$ if $\mu$ is large (the real world expects equities to drift up) but only ordinary probability under $Q$ (where the drift is the much-smaller $r$). The Radon–Nikodym derivative $dQ/dP$ is the *re-weighting function*; pricing under $Q$ is a re-weighted average over the same paths.

Why use $Q$ at all? Because the *replication argument* says option prices must be discounted expectations under $Q$, not under $P$. The real-world drift hedges out (we saw this in the binomial case). What survives is the volatility — and the discount rate, which is the cost of funding the hedge.

This has profound and underappreciated consequences:

1. **You cannot estimate $Q$ from time-series data.** $Q$ is the measure under which prices are martingales, not the measure that generated the historical price path. Estimating $\sigma$ from history is fine; $\sigma$ is the same under $P$ and $Q$ (Girsanov shifts only the drift, not the diffusion). But estimating drift from history and using it for pricing is a categorical mistake: you would be using $\mu$ where $r$ belongs.
2. **$Q$ is implied from market quotes, not estimated.** This is *calibration*: you back out the parameters of the model (volatility surface, mean-reversion speeds, etc.) from the prices of liquid instruments such that the model reproduces those prices. We'll come back to this in §11.
3. **There is one $Q$ per numéraire.** The risk-neutral measure depends on what you are using as the unit of account. Using the money-market account gives the standard $Q$; using a stock as numéraire gives the *stock measure* $Q^S$; using a $T$-maturity zero-coupon bond gives the *$T$-forward measure* $Q^T$. Each is a martingale measure for *different* discounted processes.

The numéraire change is the most important practical trick in derivatives pricing.

![Numéraire zoo: pick the asset that makes the payoff trivial](/imgs/blogs/derivatives-pricing-6.png)

The strategy is: pick the numéraire that makes the payoff *trivial* in some way. For a vanilla European call on stock, the money-market account is fine. But for a quanto option (payoff in a different currency than the stock), the stock measure makes the payoff a martingale and the pricing collapses to a one-line formula. For a swaption (option on a swap rate), the *annuity* (the sum of discount factors at each fixing date) is the right numéraire — the swap rate is a martingale under that measure, and the Black formula falls out. For an interest-rate cap, the $T$-forward measure makes the LIBOR rate a martingale and again gives a Black-style formula.

The mantra: *one numéraire per payoff, the one that makes the dynamics simplest*. Production pricing libraries support multiple numéraires via an explicit measure-change step in the simulation engine. Hand-coded pricers that only use the money-market measure are forced into more complex SDEs and usually slower convergence.

A worked example helps make this concrete. Consider a quanto call: the holder receives $\max(S_T - K, 0)$ in USD, where $S$ is the EUR-denominated price of a European stock. The payoff in USD is the EUR call payoff converted at a *fixed* (not market) FX rate. Under the standard money-market measure $Q^{\text{USD}}$, the EUR stock has drift $r_{\text{USD}} - r_{\text{EUR}} - \sigma_S \sigma_{\text{FX}} \rho_{S, \text{FX}}$, where the last term is the *quanto correction* from the change of measure. Pricing under $Q^{\text{USD}}$ requires you to handle this drift correction explicitly, which is error-prone.

Switch numéraires. Use the EUR money-market account as numéraire on the EUR side; the EUR stock has drift $r_{\text{EUR}}$ under $Q^{\text{EUR}}$. Convert the payoff to EUR by dividing by the fixed FX, price the call as a vanilla EUR call, multiply by the fixed FX, discount with the *USD* curve. Done. The correction term that hurts you under $Q^{\text{USD}}$ disappears under $Q^{\text{EUR}}$ because the EUR stock dynamics under EUR-numéraire are textbook GBM. This is a 3-line price; the money-market-measure version is 30 lines and a calibration error budget.

The general lesson is that the change of numéraire is *the* trick for cross-currency, cross-rate, and any product where the payoff naturally lives in a non-default account. The Hull-White swaption pricer uses the annuity numéraire to get a Black-style formula; the LIBOR market model uses a sequence of forward measures, one per forward rate, to make each forward a martingale; the cross-currency swap model uses the foreign-currency money-market measure for the foreign leg. Every senior quant knows two or three numéraire tricks for the products they trade most. The cost is a one-page derivation each; the benefit is faster, simpler, more accurate pricing for the rest of the system's life.

## 6. Three pricing engines a desk actually runs

There are essentially three engines that price derivatives in production: closed-form, partial-differential-equation (PDE), and Monte Carlo. The choice between them is determined by the contract, the dimensionality, and the latency budget.

![Three pricing engines: closed-form, PDE, Monte Carlo](/imgs/blogs/derivatives-pricing-7.png)

### 6.1 Closed-form

When the contract is a vanilla payoff and the model is one of a small zoo of tractable SDEs (GBM, Black-76, Bachelier, Margrabe, displaced diffusion), there is an integral that can be evaluated in closed form (or in terms of the cumulative normal). Black–Scholes is the canonical example: a one-line formula returns the call price, and analytic differentiation returns all greeks for free.

Closed-form prices have two superpowers. First, they are *machine-precision* accurate; there is no numerical error to manage. Second, they are *fast*: a Black–Scholes call price is sub-microsecond on modern hardware. A market-making desk that prices 10,000 quotes per second on a vanilla equity option does so by running a vectorised closed-form pricer; you cannot do that with a PDE or a Monte Carlo.

The catch: the contract must be vanilla and the model must be one of the tractable ones. Real markets have smile (vol depends on strike) and term structure (vol depends on expiry). The standard production trick is to run a *local-volatility* model (Dupire) or *stochastic-volatility* model (Heston) that fits the surface and admits a semi-closed-form price for vanilla options via numerical integration of the characteristic function (Carr–Madan, Lewis). Latency is then in the milliseconds, but the price is consistent with the surface.

### 6.2 PDE / finite-difference

When the contract has *early exercise* or *path-dependence* in a low-dimensional state, PDE pricing is the right engine. For a Bermudan or American option on a single underlying, the value $V(t, S)$ satisfies a parabolic PDE on $[0,T] \times \mathbb{R}_+$. You discretise $S$ on a grid, $t$ on a time mesh, and roll backward from expiry to today using a finite-difference scheme.

![Crank-Nicolson PDE grid: backward induction across S × t](/imgs/blogs/derivatives-pricing-8.png)

The diagram shows the structure. The right-most column carries the terminal payoff $\max(S_i - K, 0)$ (the yellow column). The top and bottom rows carry boundary conditions that encode contract details — for a call, $V \to S - Ke^{-r(T-t)}$ as $S \to \infty$ and $V \to 0$ as $S \to 0$. Each backward step solves a tridiagonal linear system using the Crank–Nicolson scheme (an average of explicit and implicit Euler), which is second-order accurate and unconditionally stable.

```python
import numpy as np
from scipy.linalg import solve_banded


def crank_nicolson_european_put(S0, K, T, r, sigma, S_max, M, N):
    """
    Crank-Nicolson finite-difference for European put.
    M: spot grid points, N: time steps.
    """
    dS = S_max / M
    dt = T / N
    S = np.linspace(0, S_max, M + 1)
    V = np.maximum(K - S, 0.0)  # terminal payoff

    # Tridiagonal coefficients (implicit/explicit halves of CN)
    j = np.arange(1, M)
    alpha = 0.25 * dt * (sigma**2 * j**2 - r * j)
    beta = -0.5 * dt * (sigma**2 * j**2 + r)
    gamma = 0.25 * dt * (sigma**2 * j**2 + r * j)

    # Build LHS (1 - 0.5*L) and RHS (1 + 0.5*L) banded matrices
    A = np.zeros((3, M - 1))
    A[0, 1:] = -gamma[:-1]
    A[1, :] = 1 - beta
    A[2, :-1] = -alpha[1:]

    for n in range(N - 1, -1, -1):
        b = (alpha * V[:-2]) + ((1 + beta) * V[1:-1]) + (gamma * V[2:])
        # Boundary conditions: V(0) = K e^{-r(T-t)}, V(S_max) = 0
        b[0] += alpha[0] * K * np.exp(-r * (T - n * dt))
        V[1:-1] = solve_banded((1, 1), A, b)
        V[0] = K * np.exp(-r * (T - n * dt))
        V[-1] = 0.0

    return np.interp(S0, S, V)


print(crank_nicolson_european_put(S0=100, K=100, T=1, r=0.05,
                                   sigma=0.2, S_max=300, M=400, N=400))
## ~ 5.5735 (vs Black-Scholes closed-form 5.5735)
```

For an *American* put, replace the linear-system solve with a *projected SOR* or a penalty scheme that enforces $V(t, S) \geq \max(K - S, 0)$ at every node. The grid then naturally captures the early-exercise boundary as the locus where the constraint binds.

PDE methods scale poorly with dimension. For a 2D PDE (two underlyings), you can do alternating-direction implicit (ADI) schemes that are still tractable; for 3D, you are at the edge. For 4D and beyond, switch to Monte Carlo. This is the *curse of dimensionality* in PDE pricing: the grid grows as $M^d$ where $d$ is the dimension.

### 6.3 Monte Carlo

When the contract is *path-dependent* (Asian, lookback, barrier, autocallable, cliquet), or the dimension is high, Monte Carlo is the right engine. Simulate $N$ paths of the underlying under $Q$, evaluate the discounted payoff on each path, average. The error decays as $1/\sqrt{N}$, slowly but reliably regardless of dimension.

```python
import numpy as np


def mc_european_call(S0, K, T, r, sigma, n_paths=100_000, seed=42):
    """Vanilla MC with antithetic variates."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths // 2)
    Z = np.concatenate([Z, -Z])  # antithetic
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoff
    return discounted.mean(), discounted.std(ddof=1) / np.sqrt(n_paths)


price, stderr = mc_european_call(100, 100, 1.0, 0.05, 0.2)
print(f"price = {price:.4f}  +/-  {1.96 * stderr:.4f}  (95% CI)")
## price ≈ 10.45 +/- 0.05
```

For early-exercise contracts, Monte Carlo is harder because at each step you need to know the *continuation value* (the value of holding) to decide whether to exercise. The standard solution is *Longstaff–Schwartz* (LSM): regress the realised continuation values from the in-the-money paths onto a polynomial basis in $S$, use the regression as an estimate, and exercise when the immediate payoff exceeds it.

![Longstaff-Schwartz: regress the continuation value](/imgs/blogs/derivatives-pricing-9.png)

```python
import numpy as np
from numpy.polynomial.laguerre import lagvander


def lsm_american_put(S0, K, T, r, sigma, n_paths, n_steps, basis_degree=3, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for t in range(1, n_steps + 1):
        Z = rng.standard_normal(n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    cashflow = np.maximum(K - paths[:, -1], 0.0)
    exercise_time = np.full(n_paths, n_steps)

    for t in range(n_steps - 1, 0, -1):
        intrinsic = np.maximum(K - paths[:, t], 0.0)
        itm = intrinsic > 0
        if itm.sum() < 50:
            continue
        # Regress discounted future cashflow on basis(S_t)
        X = lagvander(paths[itm, t] / K, basis_degree)
        y = cashflow[itm] * np.exp(-r * dt * (exercise_time[itm] - t))
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        continuation = X @ coef
        exercise = intrinsic[itm] > continuation
        idx = np.where(itm)[0][exercise]
        cashflow[idx] = intrinsic[itm][exercise]
        exercise_time[idx] = t

    discounted = cashflow * np.exp(-r * dt * exercise_time)
    return discounted.mean(), discounted.std(ddof=1) / np.sqrt(n_paths)
```

Two engineering points worth nailing into a junior's head:

1. **LSM is biased low.** It uses an estimated, sub-optimal exercise rule, so the price you get is a lower bound on the true American price. The Andersen–Broadie *dual* method gives an upper bound. The true price is in $[$LSM, dual$]$, and the spread is your model error budget.
2. **Variance reduction is not optional**. Antithetic variates, control variates (price the corresponding European in closed form, use it as a control), stratified sampling, and quasi-Monte Carlo (Sobol' sequences) all matter. A naive MC with $10^5$ paths has a 95% CI around 5 cents on a $10 option; with antithetic + control variate it can be 0.5 cents at the same path count. Vega and gamma are even more sensitive — pathwise greeks plus AAD are essentially mandatory in production.

A practical menu of variance reduction tricks, in roughly the order I deploy them:

- **Antithetic variates**: pair each draw $Z$ with $-Z$. Cuts variance for symmetric payoffs (vanilla, lookback) by half or more. Essentially free in code complexity. Always-on in production.
- **Control variates**: simulate the exotic and a related vanilla in lockstep (same underlying paths). The vanilla has a closed-form price; subtract the simulated minus closed-form vanilla from the simulated exotic. This removes the *common* noise driven by the underlying. Variance reduction factor 5–50× for exotics on the same underlying as a tractable vanilla. Essentially mandatory for autocallables.
- **Stratified sampling**: divide the support of the driving randomness into strata, sample each in proportion. For 1D Brownian terminals, equivalent to inverse-CDF on uniform Latin hypercube draws. Variance reduction 2–10× for smooth payoffs, less for discontinuous (barriers).
- **Quasi-Monte Carlo (Sobol' / Halton)**: low-discrepancy sequences instead of pseudorandom. Convergence improves from $1/\sqrt{N}$ to roughly $\log(N)^d / N$ for low effective dimension. For 1D-equivalent payoffs, 10–50× speedup at the same accuracy. The catch is that QMC is "discrete" — you lose the standard-error estimator, must use randomised QMC (RQMC) to get one back.
- **Importance sampling**: tilt the distribution to over-sample the regions where the payoff is non-zero. Critical for rare-event payoffs (deep OTM options, far barriers). Implementation requires a Girsanov shift on the simulation paths and a likelihood-ratio reweighting. 100× variance reduction for far-OTM is routine.
- **Brownian bridge**: when only the terminal value matters, generate the terminal first then fill in the path conditional on terminal. Reduces effective dimension for QMC and improves convergence.
- **Multi-level Monte Carlo (MLMC, Giles 2008)**: simulate at multiple discretisation levels, telescoping the cost. For SDEs where the discretisation error matters (everything path-dependent), MLMC reduces total cost from $O(\epsilon^{-3})$ to $O(\epsilon^{-2})$.

Combine antithetic + control variate + Sobol' as the default and you typically get a 50–500× variance reduction over naive MC, depending on the payoff. The implementation effort is one afternoon per technique, repaid daily for the life of the system.

## 7. Greeks and the hedge that funds the price

The price is a single number. The *greeks* are the partial derivatives of the price with respect to its inputs. They are the operationally important quantities: they tell the desk how to hedge, what its risk is, and how its P&L will move.

The standard greeks:

- $\Delta = \partial V / \partial S$ — sensitivity to spot. The hedge ratio.
- $\Gamma = \partial^2 V / \partial S^2$ — convexity. Long gamma profits from large moves.
- $\Theta = \partial V / \partial t$ — time decay. Negative for long options.
- $\text{Vega} = \partial V / \partial \sigma$ — sensitivity to implied vol.
- $\rho = \partial V / \partial r$ — sensitivity to rates.
- Cross-greeks: vanna ($\partial^2 V / \partial S \partial \sigma$), volga ($\partial^2 V / \partial \sigma^2$).

Daily P&L decomposes into greek-times-move plus higher-order terms:

$$
\Delta V \approx \Delta \cdot \Delta S + \tfrac{1}{2} \Gamma \cdot (\Delta S)^2 + \text{Vega} \cdot \Delta\sigma + \Theta \cdot \Delta t + \rho \cdot \Delta r + \text{residual}.
$$

![Daily P&L attribution: where the price comes from](/imgs/blogs/derivatives-pricing-10.png)

The decomposition is the *desk's daily P&L attribution report*. Every morning the system posts what fraction of yesterday's P&L came from delta (typically zero for a delta-hedged book), from gamma (the convexity payoff), from theta (the time decay paid for being long gamma), from vega (the move in implied vols), and from a residual that captures everything the model didn't see — jumps, calibration drift, hedge slippage, basis risk.

The structural relationship for an option-buying book is that *gamma plus theta is the price*. You pay theta every day to be long gamma. If the realised volatility is exactly the implied volatility you paid for, on average $\Gamma \cdot (\Delta S)^2$ exactly offsets $\Theta \cdot \Delta t$ and the position breaks even. If realised exceeds implied, gamma harvests more than theta costs, and you make money. If realised is lower, you lose. The option price *is* the cost of the implied volatility; gamma–theta is the structural P&L line.

Three operational facts that take new traders months to internalise:

1. **Long gamma is long realised volatility, short implied**. Every long-gamma position is a bet that future realised vol will exceed today's implied. The price you pay is the implied vol; the payoff is the realised. Implied is *known* (it's the price); realised is *unknown* (it's what the market does). The bet is structurally clear.
2. **Vega risk is convex in vol**. Volga (second derivative of price with respect to vol) means that a vega-neutral book at one vol level becomes vega-positive (or negative) when vol moves. This is what makes vol-of-vol a real risk and why volatility hedging is rebalanced frequently.
3. **Rho matters more on long-dated trades than people think**. A 30-year structured note has rho that can swamp vega. The standard mistake is to focus on equity greeks and ignore the rates greeks; on long-dated equity-rate hybrids, the rates surface drives the price.

### Computing greeks: bumping vs analytic vs adjoint

Three methods for greeks:

| Method | Cost | Accuracy | When to use |
| --- | --- | --- | --- |
| Bumping (finite difference) | $O(n)$ revaluations | $O(\epsilon)$, cancellation noise | Quick, sanity-check only |
| Analytic (closed-form) | Free | Machine precision | Vanillas, when the formula exists |
| Pathwise + LRM (likelihood-ratio) | $O(1)$ per simulation | Machine precision pathwise; slow for some greeks | MC pricers |
| AAD (adjoint algorithmic differentiation) | $O(1)$ in the number of risk factors | Machine precision | Production for high-dim risk |

AAD is the modern engineering answer. It builds a tape of the forward computation, then runs the chain rule backward to compute *all* greeks in a single sweep — cost $\sim 3$–$5\times$ a single forward eval, regardless of the number of risk factors. We'll cover the architecture in §10. For now: a desk that prices a 50-risk-factor swap-book with AAD computes all 50 greeks for the cost of $3$–$5$ price evaluations; the desk that bumps does $51$ evaluations. The AAD desk wins by an order of magnitude on every overnight greek run.

## 8. Calibration: the price is only as good as the inputs

Up to now we have treated the model parameters ($\sigma$, mean-reversion, jump rates, etc.) as given. In practice they are *backed out* from market quotes via *calibration*: an inverse-problem optimisation that finds the parameters making the model reproduce the market.

![Calibration as an inverse problem with multiple minima](/imgs/blogs/derivatives-pricing-11.png)

Calibration is mathematically a non-linear least squares: minimise $\sum_i w_i (P_i^{\text{model}}(\theta) - P_i^{\text{market}})^2$ over the parameter vector $\theta$. The forward problem (parameters → prices) is generally easy; the inverse (prices → parameters) is hard for several reasons:

1. **Non-convexity.** The objective surface has multiple local minima. Levenberg–Marquardt converges to whichever basin you start in; basin-hopping or differential evolution can find others. Two basins can fit the vanilla market within a few basis points and disagree on exotics by tens of percent.
2. **Identifiability.** Some parameters are nearly redundant. In Heston, the parameter pair $(\sigma_v, \rho)$ is partially confounded — you can fit the same surface with high $\sigma_v$ and modest $\rho$ or modest $\sigma_v$ and steep $\rho$. Daily re-calibration jumps between these regimes if you don't constrain.
3. **Stability.** Small market moves should produce small parameter moves. A model whose calibrated parameters jump erratively day-to-day is producing unstable greeks and unhedgeable books. Regularisation (Tikhonov, L1-on-parameter-deltas) is the engineering fix.
4. **Speed.** A daily calibration of a Heston surface across 200 strike-expiry pairs takes seconds with closed-form vanilla pricing; a stochastic-volatility-with-jumps model with MC vanilla pricing takes minutes; a full LSV (local–stochastic vol) with PDE-vanilla pricing can take hours. The choice of vanilla pricing engine is dictated by the calibration latency.

The senior engineer's calibration playbook:

- **Use model-implied parameters as an initial guess from yesterday**, with a small perturbation to escape stale minima.
- **Run two or three different optimisers** (LM, BFGS-with-restart, basin-hopping) and reject any calibration whose vanilla RMSE differs across optimisers by more than a small tolerance.
- **Monitor parameter time series**. Plot $\sigma_v, \rho, \kappa, \theta, v_0$ over the past quarter. Spikes are red flags; smooth drift is a healthy sign.
- **Hold-out testing**. Calibrate to a subset of the surface, predict the held-out points, measure the prediction error. A model that fits 100% of the calibration set with 10 bp RMSE but predicts the held-out wings at 50 bp is over-fit.
- **Cross-validate exotics**. Price a benchmark exotic (e.g., a 1-year up-and-out call) under each calibration. If the price spread across calibrations is bigger than your bid-ask, your model is genuinely uncertain on this exotic and you must reserve for model risk.

A worked numerical example of the calibration trap. Suppose the desk runs Heston, fits the SPX 1Y vol surface across 9 strikes (80%–120% moneyness in 5% steps). On a quiet day, the optimiser converges to $(\kappa, \theta, \sigma_v, \rho, v_0) = (1.8, 0.04, 0.55, -0.72, 0.039)$ with a vanilla RMSE of 7 bp. The next morning, the surface barely moves — the index is up 30 bp, ATM vol is unchanged, the smile shifted by less than 5 bp on each strike. The calibration *should* produce roughly the same parameters. Instead the optimiser converges to $(2.4, 0.038, 0.71, -0.55, 0.041)$ — vol-of-vol jumped from 0.55 to 0.71, correlation moved from $-0.72$ to $-0.55$. The vanilla RMSE is 6 bp, marginally better. Same fit. Different parameters.

Now price a 1-year forward-starting cliquet on those two parameter sets. Day 1's calibration gives 318 bp; day 2's gives 387 bp. A 22% jump in the price of an exotic that the desk holds, driven not by the market but by the optimiser's basin choice. Bid-ask on a cliquet is typically 50–100 bp; this 70 bp jump is a margin call on every counterparty whose collateral is computed off mark-to-model. Senior quants who have seen this once never forget it. The defensive measures — warm-starting from yesterday's calibration, regularising the parameter delta, monitoring time series for jumps, holding multiple calibrations and averaging — are not bureaucracy; they are how you keep the book stable.

The mathematical view: calibration is an inverse problem on a non-convex objective. Multiple basins can have nearly identical RMSE on the calibration set yet wildly different values on the *test* set (exotics). This is over-fitting in a different basin. The fix is the same as in machine learning: regularisation. In Heston calibration the usual regularisers are L2-on-parameter-deltas-from-yesterday, prior penalties on $\rho$ near $-1$ (numerical instability) or $\kappa\theta < \frac{1}{2}\sigma_v^2$ (Feller condition for positivity), and (in advanced setups) Bayesian-style priors with Hessian-based posteriors. Production calibration that ignores regularisation is a daily P&L bomb waiting to detonate.

## 9. Where the textbook lies: market frictions

The clean theory we've assembled — replication, no-arbitrage, $Q$, greeks — assumes a frictionless market: continuous trading, no funding cost, no credit, no capital, no bid-ask. Real markets have all of these, and the cumulative effect on price can be larger than the model's stochastic-volatility correction.

![XVA stack: the real price is the model price plus adjustments](/imgs/blogs/derivatives-pricing-12.png)

The stack of *valuation adjustments* (collectively, *XVAs*) is what closes the gap. Each XVA is computed as an expectation of a specific cashflow under an appropriate measure, and each is layered onto the risk-neutral base price.

- **CVA** (credit valuation adjustment). The expected loss to the desk from counterparty default. $\text{CVA} = (1 - R) \int_0^T E^Q[D(0,t) \cdot \text{EE}(t) \cdot \lambda(t)] dt$, where $R$ is recovery, $\text{EE}$ is the expected exposure, and $\lambda$ is the default intensity. Typical impact: 5–100 bp on uncollateralised trades. After 2008, CVA desks became their own profit center, hedging the credit exposure of every uncollateralised counterparty.
- **DVA** (debit valuation adjustment). The mirror: expected gain to the desk from *its own* default. Symmetric in theory, controversial in practice (you cannot really realise this).
- **FVA** (funding valuation adjustment). The cost of funding the variation margin or the un-collateralised position at the desk's actual funding spread, not the textbook risk-free rate. This is what blew up in 2008–2010 when bank funding spreads exploded. Typical impact: 10–40 bp on long-dated trades.
- **KVA** (capital valuation adjustment). The cost of the regulatory capital the desk must hold against the trade's credit/market risk over its lifetime. Basel III made this large — typical impact 5–50 bp.
- **MVA** (margin valuation adjustment). The cost of funding the *initial margin* required by central clearing (post-2014). On a long-dated cleared swap, MVA can be 5–30 bp.

The "real" price the desk shows the client is

$$
V_{\text{client}} = V_{\text{RN}} - \text{CVA} + \text{DVA} - \text{FVA} - \text{KVA} - \text{MVA} + \text{spread}.
$$

A senior trader has to know which XVAs the firm charges and which it absorbs. Different banks have different policies. CVA is universally charged; FVA is widely but not universally charged; DVA is usually internal-only; KVA and MVA are charged on long-dated trades.

The XVA layer is also *where pricing models meet credit/liquidity models*. CVA needs a default intensity model (reduced-form like Jarrow-Lando-Turnbull, or structural like Merton-with-jumps); FVA needs a funding spread curve. These are not "derivatives" models in the textbook sense, but they are first-class inputs to the price the client sees.

Beyond XVA, the textbook also lies about:

- **Bid-ask.** Real prices come with a spread. Mid-market modeling is fine for valuation; for hedging cost, the desk needs to model the actual fills it can get, which depends on size, urgency, and venue.
- **Repo squeezes.** Borrowing the underlying for a short hedge can become expensive or impossible (a "repo squeeze"). When this happens, the cost of the hedge changes, and the price moves. The 2020 Tesla short squeeze and various 2008 squeezes are textbook cases.
- **Dividend uncertainty.** Discrete dividends are usually treated as deterministic. In stress, dividend cuts are real. A long-dated deep-OTM call is enormously sensitive to the dividend assumption.
- **Jump risk and gap risk.** Continuous-trading models cannot hedge a gap. Buy explicit OTM options as wing-protection if you are short a structure with large negative gamma in the wings.

## 10. Production architecture: a pricing library that doesn't lie

The pricing engine is one component of a *pricing library*, which is one component of a *risk system*. The architecture matters as much as the math.

![A pricing library that does not lie to traders](/imgs/blogs/derivatives-pricing-13.png)

The decomposition every serious library implements:

- **Model.** Pure dynamics. GBM, Heston, SABR, Hull–White 1F/2F, local-vol, LSV. No knowledge of the payoff. Should be testable by simulating paths and verifying martingale properties.
- **Method.** Pure numerics. Closed-form, PDE, Monte Carlo, lattice. No knowledge of market data. Should be testable on toy problems with known answers.
- **MarketData.** Snapshotted, versioned. Volatility surfaces, yield curves, dividend curves, repo curves, FX rates. Includes the calibration cache.
- **Instrument.** Declarative payoff specification. A payoff DSL is the engineering north star; the parser turns a contract description into a callable payoff function over the path. No math.
- **Engine.** Composes (Model, Method, MarketData, Instrument) into a price + greeks + diagnostics. The engine is where measurement (e.g., MC standard error, PDE convergence) lives.

Why orthogonal? Because every combination must be testable. *American put under Heston via PDE on yesterday's snapshot* must produce the same answer regardless of who calls the engine, in any environment. If American-ness is baked into the Heston model class, you cannot price European Heston options without changing the class, and the regression test surface explodes.

QuantLib (the open-source reference) is a good study in this architecture, with some warts. The key abstraction is the `PricingEngine` interface: `Instrument::setupArguments(args)`, `Engine::calculate()`, `Instrument::fetchResults(results)`. The Bridge pattern. A `VanillaOption` instrument can be priced by an `AnalyticEuropeanEngine`, an `MCAmericanEngine`, an `FdBlackScholesVanillaEngine` — all interchangeable.

```python
import QuantLib as ql

## Market data
today = ql.Date(2, 5, 2026)
ql.Settings.instance().evaluationDate = today

spot = ql.SimpleQuote(100.0)
spot_handle = ql.QuoteHandle(spot)
risk_free = ql.YieldTermStructureHandle(
    ql.FlatForward(today, 0.05, ql.Actual365Fixed())
)
dividend = ql.YieldTermStructureHandle(
    ql.FlatForward(today, 0.02, ql.Actual365Fixed())
)
vol = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(today, ql.NullCalendar(), 0.20, ql.Actual365Fixed())
)

## Process and engines
process = ql.BlackScholesMertonProcess(spot_handle, dividend, risk_free, vol)
engine_european = ql.AnalyticEuropeanEngine(process)
engine_american = ql.BinomialVanillaEngine(process, "crr", 1000)

## Instrument
expiry = ql.Date(2, 5, 2027)
payoff = ql.PlainVanillaPayoff(ql.Option.Call, 100.0)
exercise_eu = ql.EuropeanExercise(expiry)
exercise_am = ql.AmericanExercise(today, expiry)

eu_call = ql.VanillaOption(payoff, exercise_eu)
am_call = ql.VanillaOption(payoff, exercise_am)

eu_call.setPricingEngine(engine_european)
am_call.setPricingEngine(engine_american)

print(f"EU call: {eu_call.NPV():.4f}, delta = {eu_call.delta():.4f}")
print(f"AM call: {am_call.NPV():.4f}")
## Same instrument, two engines, two answers; the engine is interchangeable.
```

Two engineering rules I would die on a hill for:

1. **Determinism.** Every price must be reproducible from (instrument, model, method, market-snapshot, seed). Reproducibility is the discipline that lets you debug a bad mark in production months later.
2. **Greeks-by-bumping in unit tests, AAD/pathwise in production.** The bumped greek is the slow, sure ground truth; AAD is the fast, fragile production tool. If the AAD greek diverges from the bumped greek by more than tolerance, the AAD tape is broken — and you want to know that before a trader hedges off the broken number.

### AAD vs bumping

![AAD vs bumping: one backward sweep returns all greeks](/imgs/blogs/derivatives-pricing-14.png)

For a book with $n$ risk factors, bumping costs $n + 1$ revaluations (one for the base price, $n$ for the perturbed prices). AAD costs $\sim 3$–$5$ times one revaluation, *regardless* of $n$. For $n = 50$, AAD is roughly $10\times$ faster; for $n = 500$, $100\times$ faster. The cost of AAD is implementation complexity: you must build a *tape* during the forward pass and propagate adjoints backward. Modern tools (JAX, PyTorch, autograd, or proprietary AAD systems like CompatibL) automate this, but the model code must be written in a differentiable style.

```python
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


@jax.jit
def bs_call(S, K, T, r, sigma):
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)


## All five greeks in one backward pass via JAX grad
greeks = jax.grad(bs_call, argnums=(0, 1, 2, 3, 4))(100.0, 100.0, 1.0, 0.05, 0.2)
print(f"delta={greeks[0]:.4f}, dK={greeks[1]:.4f}, theta={greeks[2]:.4f}, "
      f"rho={greeks[3]:.4f}, vega={greeks[4]:.4f}")
```

For a Monte Carlo book, the AAD applies *per path*; the chain rule propagates from terminal payoff back through every step of the SDE simulation. Adjoint Monte Carlo (Giles 2007) is the production-grade implementation. The combination AAD + Monte Carlo + AAD-friendly variance reduction (control variates with adjoints) is the bread and butter of modern XVA desks.

## 11. Case studies: pricing failures that cost real money

The cleanest way to learn pricing is by studying its failures. Below are eight named incidents, each of which illustrates a structural issue we covered above.

### 11.1 The 1987 crash and the discovery of the smile

Before October 19, 1987, equity-options desks priced with a flat Black–Scholes volatility — the same $\sigma$ for every strike. The 1987 crash, where the S&P 500 dropped 20.5% in one day, broke that assumption. After the crash, deep out-of-the-money puts started trading at much higher implied vols than at-the-money options, and the volatility smile / skew was born. Operationally, every options desk that priced 1987-vintage OTM puts at flat-vol mispriced its inventory by 30–60% on the strike axis.

The mechanism: the market re-priced tail risk. A 20% one-day drop is a $\sim 25\sigma$ event under a Gaussian model with $\sigma = 1\%$ daily. After 1987, the market refused to price tails with a Gaussian and instead priced them with a fat-tailed distribution. The implied-vol smile is the market's way of pricing tails without changing the formula.

The fix: surface modeling. Dupire local volatility (1994) made the smile Markovian; Heston (1993) gave it stochastic-vol dynamics; SABR (2002) gave it tractable smile dynamics. We will cover this in [the Volatility Surface post](/blog/trading/quantitative-finance/derivatives/volatility-surface).

The pricing-architecture lesson: any pricing library that hard-codes flat volatility is fundamentally incompatible with markets after 1987. Build for surfaces from day one.

### 11.2 LTCM 1998: convergence trades and missing liquidity premium

Long-Term Capital Management ran a swap-spread / Treasury-spread "convergence" strategy: short the rich asset, long the cheap, wait for spreads to mean-revert. The pricing implicitly assumed continuous tradability and bounded VAR. In August 1998, the Russian default + LTCM's own size meant that as spreads widened, no one would take the other side. LTCM had to liquidate at progressively wider spreads, blowing up its capital.

The pricing model said the trades were nearly riskless under reasonable assumptions about rates. The lesson: *liquidity premium is part of the price* in stress. A convergence trade is short volatility and long liquidity; when liquidity dries up, the implicit option that has been written is exercised against you. Modern XVA frameworks treat this via FVA and stressed-funding scenarios. Pre-2008, FVA was not standard. LTCM was a $4.6B preview of the 2008 lesson.

The mathematical fix: do not use a flat-funding curve in pricing. Use a stressed-funding curve in scenario analysis. Reserve for funding-stress separately. Most banks now run a *funding-stress reserve* equal to a fraction of the FVA, sized for tail-funding events.

### 11.3 Bear Stearns 2007 super-senior CDOs and the Gaussian copula

The Gaussian copula (Li 2000) priced credit-portfolio derivatives by assuming default correlations were captured by a single Gaussian-copula correlation parameter. Banks calibrated the copula to liquid index tranches (5y CDX) and used it to price super-senior CDO tranches. The implicit assumption: the dependency structure is Gaussian, with one parameter.

The 2007 housing market deteriorated, and tail correlation in subprime mortgage defaults rose well beyond what the Gaussian copula could express. Super-senior tranches that had been priced as essentially riskless turned out to be exposed to dependent-default risk that the model couldn't see. Bear Stearns' two structured-credit hedge funds went to zero in June 2007, and the "$30/share for $0/share" liquidation was the first visible casualty of the crisis.

The pricing-architecture lesson is severe: *if the model cannot represent the risk, the price cannot reflect it*. The Gaussian copula is a single-parameter dependency model. Subprime defaults required a multi-state, contagious-default model (e.g., reduced-form with stochastic intensity). The product was invented faster than the model that could price it. A rule: never let a desk price a product whose risk dimensions exceed the model's representational capacity.

### 11.4 Lehman default and the OIS-LIBOR discount-curve war

Pre-2008, swap-pricing systems used LIBOR as both the funding rate and the discount rate. After Lehman's default in September 2008, the credit spread between LIBOR (unsecured interbank) and OIS (overnight-indexed, effectively risk-free) jumped from $\sim 10$ bp to $\sim 350$ bp. Suddenly the question "which curve do you discount with?" was a multi-billion-dollar question: a 10y collateralised swap discounted at LIBOR vs OIS differed by tens of basis points in NPV.

The industry consensus moved to OIS discounting for collateralised trades and a separate funding curve for uncollateralised ones. This is *multi-curve pricing*: a different curve per cashflow, depending on its collateralisation status. Production pricing systems had to be retrofitted in 2008–2010 to support this; it was a year-long project at every major bank.

The lesson: the discount curve is a *trade attribute*, not a global default. Pricing libraries built on a single global curve cannot represent the post-2008 reality. We will cover multi-curve construction in [the Yield Curve Modeling post](/blog/trading/quantitative-finance/fixed-income/yield-curve-modeling).

### 11.5 Negative oil futures, April 2020

On April 20, 2020, the May WTI crude oil futures contract settled at $-\$37.63$. Storage capacity at Cushing, OK was full; longs facing physical delivery were paying buyers to take the oil. Many oil-derivatives pricing systems had hard-coded the assumption $S \geq 0$ (consistent with GBM, where $\log S$ is the modeled quantity). A negative spot price is a $-\infty$ in log-space; the systems crashed or returned NaNs.

The lesson: the model's *state space* matters. GBM's positivity is an assumption baked into the SDE. The right model for oil is *Bachelier* (arithmetic Brownian motion: $dS = \mu \, dt + \sigma \, dW$) with $S$ allowed to be negative. CME quietly switched to Bachelier for short-dated WTI options in April 2020. Banks scrambled to retrofit their pricing systems within days.

A senior engineer's reflex: when designing a pricing system, ask *what is the support of the underlying* and design accordingly. Equities are positive (use GBM); rates can be negative since 2014 (use Hull–White, not log-normal LIBOR market model); commodities can be negative in principle (use Bachelier or shifted log-normal). Hard-coding $S \geq 0$ everywhere is a footgun.

Worth dwelling on one detail. Bachelier and Black-76 give numerically *different* prices for the same option even when the underlying is well above zero. Black-76 prices an option assuming log-normal returns (ATM straddle ≈ $S \cdot \sigma \sqrt{T} \cdot 0.8$); Bachelier assumes normal returns (ATM straddle ≈ $\sigma_N \sqrt{T} \cdot 0.8$ where $\sigma_N$ is in price units, not log units). For interest rates (small, low volatility, can go negative) the conversion factor between log-vol $\sigma$ and normal-vol $\sigma_N$ is roughly $\sigma_N \approx F \cdot \sigma$ where $F$ is the forward rate. The big trap: traders quote in log-vol, the model uses normal-vol, and the conversion is *non-linear* near zero forwards. In the post-2014 negative-rates regime, several European banks were caught with mispriced caps because their pricing system was internally consistent in log-vol but the calibration was given in shifted-Black quotes that did not round-trip.

The Bachelier-vs-Black choice is the single most common production-pricing question in fixed income. The senior engineer's habit is to record *both* the model and the quote convention with every market data point, and to round-trip ATM vols through the model on every calibration cycle to detect drift.

### 11.6 GameStop January 2021: gamma squeeze mechanics

GME's January 2021 rally was driven by a *gamma squeeze*: retail buyers bought OTM short-dated calls; market makers selling those calls were short gamma; to hedge, they had to buy stock as it rallied; that drove price further; calls got further in the money; market makers had to buy more. The feedback is a real, mechanical consequence of delta hedging.

For pricing, GME exposed two issues:

1. **Hedging cost is not zero in stress.** The Black–Scholes derivation assumes you can hedge at the model price. If you must buy stock into a thin order book, slippage is large. The realised P&L of a delta-hedged short option is no longer the implied–realised vol differential; it's the differential plus slippage.
2. **Implied vols re-priced wing-risk overnight.** GME 30-day implied vol went from $\sim 100$% in mid-January to $\sim 1000$% at the peak. Any system that quoted off a stale calibration was massively mispricing.

The architectural lesson: implied vol is a market quantity, not a model parameter. Re-calibrate continuously in stress. Have a *cooldown* and *circuit breaker* in pricing systems; refuse to quote when implied vol moves beyond a threshold without trader confirmation. GameStop hosed several mid-tier market-makers; the survivors were the ones whose systems noticed the regime change.

### 11.7 Archegos 2021: total return swaps, price ≠ exposure

Archegos was a family office running concentrated positions via *total return swaps* (TRS): the prime broker (e.g., Credit Suisse, Nomura) held the actual stock and Archegos took the synthetic exposure. Pricing the TRS itself is trivial — it's an equity forward. The problem is *exposure*: the prime broker has counterparty risk to Archegos for the difference between the stock price and the agreed forward.

The pricing systems priced the swaps correctly. The *risk* systems failed to aggregate exposure across counterparties (multiple primes financed Archegos in parallel without each knowing the others' exposure) and across positions (Archegos was concentrated in a few stocks; the marginal exposure when one of them fell was much larger than the netted-exposure metrics suggested). When the stocks fell in March 2021, Archegos couldn't post margin, and Credit Suisse and Nomura took $\sim \$10B$ of losses combined.

The pricing-architecture lesson: *price is not exposure*. A pricing system that computes V is necessary but not sufficient; you need a risk system that computes peak future exposure (PFE) under stress scenarios, aggregated across counterparties and netting sets. CVA capital is calibrated to PFE, not to V. Senior architects of pricing libraries should know exactly where the price-to-PFE handoff happens in their stack.

### 11.8 SVB March 2023: held-to-maturity bonds, mark-to-model

Silicon Valley Bank held a large portfolio of long-dated Treasuries and MBS in *held-to-maturity* (HTM) accounting, which lets the bank carry them at amortised cost rather than market value. As rates rose in 2022–2023, the *market* value of these bonds fell well below their accounting value. The pricing model was correct (the bonds had clear market quotes); the *accounting* policy chose not to mark to market. When depositors required liquidity and SVB had to sell the HTM portfolio, the unrealised loss became real, the bank was insolvent, and a deposit run forced regulatory takeover within 48 hours.

The pricing-architecture lesson is subtle: the price exists, the accounting choice is what hides it. Pricing systems must produce mark-to-market values *regardless* of accounting treatment, and risk systems must report on the gap between mark-to-market and book value as a primary risk metric. SVB's risk system surely produced this number; the question is whether anyone in management was forced to look at it. The engineering principle: *make the inconvenient truth easy to see* in your dashboards.

The corollary at a more granular level: every risk system needs a *liquidity-weighted price*. The mark-to-market price assumes you can sell at the mid; the real liquidation price for a $100B HTM portfolio in a stressed market is materially below mid. SVB's bond portfolio at the moment of the run was probably 10–15% below the held-to-maturity book value at *fair* market quotes, but 25–30% below at *forced-liquidation* quotes (because no buyer wanted $100B of duration in two days). Modern risk systems include a *days-to-liquidate* sensitivity and a *peak-stress haircut* applied to mark-to-market. SVB-style failures should be caught by these metrics if they're surfaced to the risk committee weekly.

A final, often-overlooked lesson from every case study above: **the pricing model and the operational reality are coupled**. The 1987 smile, LTCM's funding, Bear Stearns' tail correlation, Lehman's discount-curve change, the negative-oil contract spec, GameStop's order-book depth, Archegos's margin model, SVB's accounting policy — none of these are *purely* pricing-model issues. They are coupling issues between the model and the world it operates in. A senior quant's job is to know where those coupling points are. The model is right when the coupling points are stable; the model is dangerously wrong when the coupling points break. The discipline of *naming and monitoring* coupling points is what distinguishes a robust pricing system from a fragile one.

## 12. When to reach for which framework, and when not to

A summary table for the senior quant who has just inherited a derivatives book:

| Situation | First-line tool | Second-line tool | Avoid |
| --- | --- | --- | --- |
| Vanilla European, single underlying | Black–Scholes / Black-76 closed form | Local vol PDE | Monte Carlo (overkill) |
| American single-name | Binomial tree (CRR) for short-dated; PDE for long-dated | LSM if the model is non-Markovian | Bumped-greek MC (high noise) |
| Path-dependent, single underlying | PDE if state can be augmented; otherwise MC | LSM for early exercise | Closed-form (doesn't exist) |
| Basket / multi-asset (n ≥ 4) | Monte Carlo | Conditional MC, copula-based shortcut for tractable cases | PDE (curse of dimensionality) |
| Interest-rate swaption | Black formula on annuity numéraire; Hull–White swaption for term-structure curvature | LMM for many forwards | Equity-style GBM (wrong dynamics) |
| Cross-asset hybrid (equity × rates × FX) | Multi-factor MC with copula or full LSV | Hull–White × local-vol × Garman–Kohlhagen on FX | Anything that doesn't model joint dynamics |
| Credit-portfolio | Reduced-form with stochastic intensity; copula only for index-tranche calibration | Bottom-up name-by-name | Single-parameter Gaussian copula on bespoke tranches |
| Long-dated structured note | Hull–White × local-vol; full XVA stack | Stress scenarios for funding and credit | Flat-funding-curve discounting |

Three closing principles for the engineer-quant:

**Match the model to the risk dimensions of the product.** If your product has gamma, vega, vanna, volga, and theta exposures, your pricing model must represent vol of vol or you are systematically mispricing the moments you cannot see. The 2007 super-senior CDO mess is the canonical example.

**Treat XVAs as first-class price components.** The risk-neutral price is a *component*, not the whole price. The desk's quote includes funding, credit, and capital. Architect your library to accumulate XVAs as a stack from day one; retrofitting XVA after the fact is a year-long project, as 2008–2010 showed.

**Treat the calibration cache as load-bearing infrastructure.** Every price the library produces is conditional on a snapshot of market data and a calibration that fits that snapshot. The cache that stores these snapshots and calibrations is the *audit trail* of every quote the firm has ever made. Pricing libraries that overwrite calibrations in-place are unauditable; libraries that version every calibration with a hash and a timestamp can answer the question "what did we quote at 14:32:07.123 on March 14" deterministically. After the 2010 flash crash, regulators began asking exactly this question with subpoena power. A senior architect should make sure every price logged in production is reproducible from a versioned model + versioned calibration + versioned market-data snapshot, with the version IDs stamped on the trade ticket.

**Build the pricing library so that it can be wrong loudly.** The worst pricing failures are silent: a model gives a confident number that is wrong by 10%. The defence is a culture of *calibration diagnostics, model spread reports, hedge slippage attribution, and parameter time series*. A senior quant should be able to look at any price the system produces and see immediately *which calibration produced it, what the residual was, and how sensitive the price is to alternative calibrations*. If the answer to "how do we know this price is right" is "because the system says so", the system will eventually print a number that costs the firm tens of millions of dollars. Make the system honest about its uncertainty, and the desk will trust the prices that *are* correct because they are clearly distinguished from the ones that are not.

The remaining articles in this series — [Options Theory](/blog/trading/quantitative-finance/derivatives/options-theory), [Black–Scholes](/blog/trading/quantitative-finance/derivatives/black-scholes), [Volatility Surface](/blog/trading/quantitative-finance/derivatives/volatility-surface), [Bond Pricing](/blog/trading/quantitative-finance/fixed-income/bond-pricing), [Yield Curve Modeling](/blog/trading/quantitative-finance/fixed-income/yield-curve-modeling), [Fixed Income Analytics](/blog/trading/quantitative-finance/fixed-income/fixed-income-analytics), [Short-Rate Models](/blog/trading/quantitative-finance/rates-models/short-rate-models-vasicek-hull-white), [Exotic Derivatives](/blog/trading/quantitative-finance/exotics/exotic-derivatives), [Autocallables](/blog/trading/quantitative-finance/exotics/autocallables), and [Cliquets](/blog/trading/quantitative-finance/exotics/cliquets) — go deep on each component sketched above. This article is the *map*; the others are the *territory*.
