---
title: "Short-Rate Models: Vasicek, Hull-White, and the Dynamics That Price Rate Exotics"
date: "2026-05-03"
publishDate: "2026-05-03"
description: "A senior-quant deep dive into short-rate models: Vasicek, Hull-White, Black-Karasinski, CIR, LIBOR market model, calibration, lattice / PDE / Monte Carlo pricing, swaption pricing, callable bonds, multi-factor extensions, production architecture, and named failure modes."
tags:
  [
    "short-rate-models",
    "vasicek",
    "hull-white",
    "black-karasinski",
    "cir",
    "libor-market-model",
    "swaption-pricing",
    "callable-bonds",
    "rates-derivatives",
    "calibration",
    "quantitative-finance",
    "python",
  ]
category: "trading"
author: "Hiep Tran"
featured: true
readTime: 50
aiGenerated: true
---

A yield curve tells you what rates are *today*. A short-rate model tells you how rates *evolve over time*. The curve is a static snapshot; the short-rate model is the dynamic process. Pricing a vanilla bond requires only the curve. Pricing a callable bond, a Bermudan swaption, a mortgage-backed security with prepayments, or any other product whose payoff depends on the *path* of future rates requires a short-rate model. The curve provides the initial condition; the model provides the dynamics.

![Short-rate models: from one SDE to every interest-rate derivative](/imgs/blogs/short-rate-models-vasicek-hull-white-1.png)

The diagram above is the mental model. Inputs to a short-rate model are today's curve plus market-implied volatilities (from caps or swaptions). The model itself is a stochastic differential equation for the instantaneous short rate $r_t$, with parameters chosen to fit the calibration set. Outputs are dynamic prices of every rate-sensitive instrument: bonds, options, structured products. The short-rate model is the bridge between the static curve and the path-dependent products built on it.

This article is the deep dive on short-rate models for a senior quant or staff-level engineer. It covers Vasicek (the simplest mean-reverting model), Hull-White (the curve-fitting workhorse), Black-Karasinski (log-normal positive-rate variant), CIR (square-root diffusion), and the LIBOR market model (forward-rate-based alternative). It works through calibration, lattice / PDE / Monte Carlo pricing engines, swaption pricing via Jamshidian decomposition, callable bond pricing, multi-factor extensions, production architecture, and a long catalog of named failure modes.

The companion articles are [Yield Curve Modeling](/blog/trading/quantitative-finance/fixed-income/yield-curve-modeling) (the curve layer beneath this), [Bond Pricing](/blog/trading/quantitative-finance/fixed-income/bond-pricing) (vanilla bond pricing), and [Fixed Income Analytics](/blog/trading/quantitative-finance/fixed-income/fixed-income-analytics) (analytics on top).

## 1. Why short-rate models exist

Vanilla bonds are priced by discounting cashflows under the curve. No model needed. The curve is a deterministic object; given the curve, the price is a closed-form sum.

But many real products have payoffs that depend on what the curve does *between now and expiry*:

- **Callable bond.** The issuer can call the bond if rates fall enough. The call decision depends on the future short rate at each call date.
- **Bermudan swaption.** The holder can exercise on a discrete schedule. Each exercise decision depends on the prevailing rate.
- **MBS.** Borrowers prepay when rates fall; the cashflow path depends on the rate path.
- **Range accrual note.** Coupons accrue on days the short rate is in a range; the accrual depends on the path.
- **Cap/floor.** A cap pays $\max(L_T - K, 0)$ on each fixing date $T$, where $L_T$ is the LIBOR/SOFR at fixing. Each caplet is an option on a forward rate.

For all of these, the curve gives the initial condition but does not specify how rates evolve. We need an *SDE* — a stochastic differential equation for $r_t$ — that says how $r$ moves over time. The model's job is to provide a calibrated, arbitrage-free, computationally tractable SDE.

The senior engineer's mental shortcut: *if the payoff depends on the rate path, you need a short-rate model*. If it depends only on terminal rates, the curve plus the forward measure suffice (we covered this in [the Black-Scholes post](/blog/trading/quantitative-finance/derivatives/black-scholes#9-the-black-scholes-family-black-76-garman-kohlhagen-margrabe-bachelier) when discussing Black-76).

A second observation: short-rate models are *the simplest dynamic models in finance*. They have one state variable (the short rate). The mathematics is well-understood, the calibration is tractable, the pricing is fast. This simplicity is why they remain dominant despite the existence of more sophisticated alternatives (LIBOR market model, HJM framework). For most rate-derivative pricing, a short-rate model is enough.

## 2. Vasicek

The Vasicek model (1977) is the original mean-reverting short-rate model:

$$
dr_t = a(\theta - r_t) \, dt + \sigma \, dW_t
$$

with three constant parameters: $a$ (mean-reversion speed), $\theta$ (long-run mean), $\sigma$ (volatility).

![Vasicek: the simplest mean-reverting short-rate model](/imgs/blogs/short-rate-models-vasicek-hull-white-2.png)

The SDE is *Ornstein-Uhlenbeck* — the same process that describes a damped harmonic oscillator with thermal noise. The rate is pulled back toward $\theta$ at speed $a$; volatility shocks knock it away. Long-run, the rate is stationary with mean $\theta$ and variance $\sigma^2 / (2a)$.

Distribution at time $t$, given $r_0$:

$$
\mathbb{E}[r_t | r_0] = r_0 e^{-at} + \theta(1 - e^{-at})
$$

$$
\text{Var}[r_t | r_0] = \sigma^2 (1 - e^{-2at}) / (2a).
$$

Both formulas have a clean interpretation: as $t \to \infty$, the conditional mean converges to $\theta$ and the variance converges to $\sigma^2 / (2a)$. The half-life of the mean reversion is $\ln(2) / a$; for $a = 0.1$, half-life is ~7 years.

A subtle property: Vasicek rates can go *negative*. The Gaussian distribution of $r_t$ has support on the entire real line. Pre-2014, this was considered a defect (rates were always positive in major currencies). Post-2014, with negative rates a reality in EUR and CHF, the property is a feature, not a bug.

Vasicek admits an *affine term structure*: the price of a zero-coupon bond is

$$
P(t, T) = A(t, T) e^{-B(t, T) r_t}
$$

with closed-form $A$ and $B$ in terms of $(a, \theta, \sigma)$. Pricing a vanilla bond is $O(1)$.

The big limitation of Vasicek: with three constant parameters, the model can match three points on the curve but not the entire curve. Calibrating to fit today's full curve requires a time-dependent generalisation — Hull-White.

```python
import numpy as np


def vasicek_simulate(r0, a, theta, sigma, T, n_steps, n_paths, seed=0):
    """Simulate Vasicek paths via Euler discretisation."""
    np.random.seed(seed)
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = r0
    for i in range(n_steps):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        paths[:, i + 1] = paths[:, i] + a * (theta - paths[:, i]) * dt + sigma * dW
    return paths


def vasicek_zcb(r, t, T, a, theta, sigma):
    """Closed-form Vasicek zero-coupon bond price."""
    B = (1 - np.exp(-a * (T - t))) / a
    A_log = (theta - sigma**2 / (2 * a**2)) * (B - (T - t)) - (sigma**2 * B**2) / (4 * a)
    return np.exp(A_log - B * r)
```

Vasicek is a teaching tool today; production uses Hull-White.

## 3. Hull-White

Hull and White (1990) extended Vasicek by replacing the constant $\theta$ with a time-dependent function $\theta(t)$:

$$
dr_t = (\theta(t) - a r_t) \, dt + \sigma \, dW_t.
$$

![Hull-White: time-dependent drift fits today's entire curve](/imgs/blogs/short-rate-models-vasicek-hull-white-3.png)

The genius: $\theta(t)$ is calibrated such that the model reproduces *the entire initial yield curve exactly*. Closed form:

$$
\theta(t) = \frac{\partial f(0, t)}{\partial t} + a \cdot f(0, t) + \frac{\sigma^2}{2a}\left(1 - e^{-2at}\right)
$$

where $f(0, t)$ is the instantaneous forward rate at time $t$ as observed today.

The two remaining parameters $(a, \sigma)$ are calibrated to market vol instruments — typically caps or swaptions. The calibration is two-stage:

1. **Stage 1 (curve fit).** Given any $(a, \sigma)$, choose $\theta(t)$ via the closed form to fit the curve exactly.
2. **Stage 2 (vol fit).** Optimise over $(a, \sigma)$ to minimise pricing error on cap or swaption volatilities.

Two parameters, exact curve fit. The model is the workhorse of rate-derivative pricing.

Hull-White retains the affine term structure:

$$
P(t, T) = A(t, T) e^{-B(t, T) r_t}
$$

with

$$
B(t, T) = \frac{1 - e^{-a(T-t)}}{a}, \qquad \log A(t, T) = \log\frac{P(0,T)}{P(0,t)} + B(t, T) f(0, t) - \frac{\sigma^2}{4a}(1 - e^{-2at}) B(t, T)^2.
$$

Vanilla bond pricing under Hull-White is closed form, $O(1)$ per bond. Coupon bonds sum closed-form zeros. Bond options have closed form via the Jamshidian decomposition (we'll cover this in §7).

For callable bonds, Bermudan swaptions, and other path-dependent products, Hull-White uses lattice or PDE methods. We'll cover those in §6.

```python
def hull_white_theta(t, a, sigma, forward_curve, dt=1/252):
    """Compute Hull-White theta(t) at time t."""
    f = forward_curve(t)
    f_prime = (forward_curve(t + dt) - forward_curve(t - dt)) / (2 * dt)
    return f_prime + a * f + (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))


def hull_white_zcb(r, t, T, a, sigma, forward_curve, P_market):
    """Hull-White zero-coupon bond price."""
    B = (1 - np.exp(-a * (T - t))) / a
    A_log = (np.log(P_market(T) / P_market(t))
             + B * forward_curve(t)
             - (sigma**2 / (4 * a)) * (1 - np.exp(-2 * a * t)) * B**2)
    return np.exp(A_log - B * r)
```

Production-quality Hull-White implementations are a few hundred lines covering all the calibration, pricing, and edge cases. QuantLib has a complete implementation; major banks have proprietary variants.

### 3.1 Hull-White's place in the model hierarchy

Hull-White sits in a hierarchy of rate models with increasing complexity:

1. **Constant short rate.** $r_t = r$. Useful only as a teaching tool.
2. **Deterministic curve.** $r_t = f(t)$ deterministic. Fits today's curve but no dynamics; useless for path-dependent products.
3. **Vasicek.** $r_t$ stochastic mean-reverting with constant parameters. Cannot fit full curve.
4. **Hull-White.** $r_t$ stochastic mean-reverting with time-dependent drift. Fits full curve exactly. *The standard.*
5. **Two-factor Hull-White.** Two SDEs; better fit for long-dated curve dynamics. Used for sophisticated products.
6. **Cheyette / quasi-Gaussian.** State-dependent volatility on top of Hull-White-like structure. Captures vol smile.
7. **LMM (LIBOR Market Model).** Forward rates as primary objects; multi-dimensional. Most flexible but most expensive.
8. **HJM.** Most general, instantaneous forward rates as primary objects. Mostly research / academic.

A senior rates quant knows the entire hierarchy and chooses the simplest model that meets the product's requirements. Hull-White covers ~80% of production use cases; multi-factor Hull-White covers another 15%; LMM covers the remaining 5% of complex exotics.

### 3.2 The intuition for mean reversion

Why does mean reversion matter? Two reasons.

**Term-structure slope.** A short rate that *doesn't* mean-revert has unbounded variance over long horizons. The yield curve becomes structurally upward-sloping with steep convexity. Real curves don't look like that; the term-structure flattening at long horizons is the hallmark of mean reversion.

**Cap/swaption volatility shape.** Caps and swaptions across tenors show a characteristic "hump" — vol rising from short to ATM tenor, then declining. Mean reversion produces this hump naturally. Without mean reversion, vol would grow unbounded with tenor, contradicting market quotes.

The mean-reversion speed $a$ is therefore a *structural* parameter calibrated to the term-structure shape. Typical values: $a \in [0.01, 0.1]$, corresponding to half-life of 7-70 years. Different currencies have different typical $a$:

- **USD**: $a \approx 0.05$ (half-life 14 years).
- **EUR**: $a \approx 0.03$ (half-life 23 years; longer cycles).
- **JPY**: $a \approx 0.02$ (half-life 35 years; very persistent).

A senior rates quant develops intuition for the cycles in each currency and recognises when $a$ calibrates to anomalous values.

## 4. Calibration

Calibration of Hull-White is the daily ritual of every rates-derivatives desk.

![Calibration: fitting the model to market quotes](/imgs/blogs/short-rate-models-vasicek-hull-white-4.png)

The two-stage process:

**Stage 1: Curve fit.** Given $(a, \sigma)$, set $\theta(t)$ via closed form. The model now reproduces $P(0, T)$ for all $T$ exactly. Stage 1 is closed-form per $(a, \sigma)$.

**Stage 2: Vol fit.** For each candidate $(a, \sigma)$:
1. Compute the model's prediction for each calibration vol instrument (cap or swaption).
2. Compare to market implied vol.
3. Compute RMSE over the calibration set.

Optimise $(a, \sigma)$ via Levenberg-Marquardt to minimise RMSE.

Choice of vol instruments:

- **Caps.** Each caplet is a call on a forward rate; cap is a strip of caplets. Caps quote in implied vol per tenor. Pros: simpler to handle one underlying. Cons: only one tenor at a time.
- **Swaptions.** A swaption matrix has expiry × tenor structure (e.g., $5\text{y} \times 10\text{y}$ swaption is option on $10\text{y}$ swap with $5\text{y}$ expiry). Swaptions cover the full $(M \times N)$ surface. Pros: matches the swaption book directly. Cons: more complex calibration.

A typical bank's calibration set: 30-50 swaption quotes covering expiries from 1y to 30y and tenors from 1y to 30y. The optimiser converges in 5-15 iterations; total wall-clock 1-3 seconds.

Production cadence:

- **Market open**: full calibration with curve update.
- **Intraday**: vol-only refit every 15 minutes if vol moves materially.
- **End-of-day**: deep calibration with hold-out tests.
- **Weekly**: full review of parameter time series.

Validation:

- **Vanilla RMSE**: <1 vol point on caps, <2 on swaptions.
- **Parameter delta**: $(a, \sigma)$ within historical 95% bands from yesterday's calibration.
- **Hold-out fit**: reserved swaptions priced within bid-ask.
- **Exotic stability**: a benchmark Bermudan swaption priced under today's vs yesterday's calibration; spread within bid-ask.

Failed validation triggers human review; the calibration is *not* auto-published.

```python
from scipy.optimize import minimize


def hw_calibrate(swaption_market_vols, curve, swaption_specs):
    """Calibrate Hull-White (a, sigma) to swaption market."""
    def loss(params):
        a, sigma = params
        if a <= 0 or sigma <= 0:
            return 1e9
        model_vols = [hull_white_swaption_vol(spec, curve, a, sigma)
                      for spec in swaption_specs]
        residuals = [(m - mkt) for m, mkt in zip(model_vols, swaption_market_vols)]
        return sum(r**2 for r in residuals)
    result = minimize(loss, x0=[0.05, 0.01],
                      method='L-BFGS-B', bounds=[(1e-4, 1.0), (1e-5, 0.5)])
    return result.x
```

## 5. Bond pricing under Hull-White

Vanilla bond pricing under Hull-White is closed form via the affine formula.

![Bond pricing under Hull-White: closed form for vanilla](/imgs/blogs/short-rate-models-vasicek-hull-white-5.png)

For a zero-coupon bond:

$$
P(t, T) = A(t, T) \, e^{-B(t, T) r_t}.
$$

For a coupon bond paying $c_i$ at time $t_i$ and face $F$ at $T$:

$$
P_{\text{coupon}}(t) = \sum_i c_i \, P(t, t_i) + F \, P(t, T).
$$

The price is a sum of zero-coupon bond prices. $O(N)$ in the number of coupon dates, where each $P(t, t_i)$ is $O(1)$ via closed form.

For callable / puttable / Bermudan: the closed form gives the un-called value; the call/put premium requires lattice / PDE on the short-rate tree. The pricing engine then becomes:

1. **Build a Hull-White lattice** in $(t, r)$ space.
2. **Compute terminal payoff** at maturity for each rate node.
3. **Backward induction**: at each node, $V = \mathbb{E}[V_{\text{next}}] / (1 + r \, dt)$.
4. **Apply call/put constraint** at exercise dates: $V = \max(V_{\text{continue}}, V_{\text{exercise}})$.
5. **Read off** the value at $t = 0$ corresponding to the current short rate.

The pricing is $O(N_t \times N_r)$ where $N_t$ is the number of time steps and $N_r$ the number of rate states. For a 30-year callable bond with daily steps and 100 rate states, this is ~750K operations — milliseconds on modern hardware.

### 5.1 Connection to the Black-Scholes PDE

Hull-White bond pricing satisfies a parabolic PDE analogous to Black-Scholes for options. Setting up the PDE: under the risk-neutral measure, the bond price $P(t, T)$ as a function of $(t, r)$ satisfies

$$
\frac{\partial P}{\partial t} + \frac{1}{2} \sigma^2 \frac{\partial^2 P}{\partial r^2} + (\theta(t) - a r) \frac{\partial P}{\partial r} - r P = 0
$$

with terminal condition $P(T, T) = 1$. This is a 1-D linear parabolic PDE.

The same PDE governs bond *options*: the option's value function $V(t, r)$ satisfies the equation with terminal condition equal to the option payoff. For European bond options, the closed-form solution is the Jamshidian decomposition. For American/Bermudan, finite-difference solving the PDE with the early-exercise constraint gives the price.

A senior quant recognises the Hull-White PDE as structurally analogous to the Black-Scholes PDE we covered in [the Black-Scholes post](/blog/trading/quantitative-finance/derivatives/black-scholes#2-pde-derivation-replication-forces-the-heat-equation). Both reduce to the heat equation under appropriate substitutions; both have closed-form solutions for terminal-payoff vanilla options; both require numerical methods for path-dependent or early-exercise products.

### 5.2 The forward-curve calibration math

The Hull-White $\theta(t)$ formula is worth deriving. Under Hull-White, the conditional expectation of $r_t$ at time 0 is

$$
\mathbb{E}^Q[r_t | r_0] = e^{-at} r_0 + \int_0^t e^{-a(t-s)} \theta(s) \, ds.
$$

For the model to reproduce the initial forward curve $f(0, t)$, we need

$$
f(0, t) = \mathbb{E}^Q[r_t | r_0] - \frac{\sigma^2}{2a^2}(1 - e^{-at})^2 + \text{convexity adjustment}.
$$

Solving for $\theta(t)$:

$$
\theta(t) = \frac{\partial f(0, t)}{\partial t} + a \cdot f(0, t) + \frac{\sigma^2}{2a}(1 - e^{-2at}).
$$

This closed form means $\theta(t)$ is computed once given $(a, \sigma)$ and the curve. The optimisation is only over $(a, \sigma)$; $\theta(t)$ is bootstrapped automatically.

A senior implementer caches $\theta(t)$ on a fine time grid (typically daily steps over 30 years) for fast lookup during PDE / lattice pricing.

## 6. Hull-White lattice

The lattice (or "tree") is a discrete approximation of the continuous-time SDE. Trinomial branching is standard for Hull-White because it gives the right mean and variance of the rate increment with mean reversion.

![Hull-White lattice: backward induction for callable bonds](/imgs/blogs/short-rate-models-vasicek-hull-white-6.png)

The lattice construction:

1. **Time grid.** Discretise $[0, T]$ into $N_t$ steps. Daily, weekly, or monthly depending on product. Refine near call dates.
2. **Rate grid.** At each time $t_i$, define $N_r$ rate nodes around the conditional mean $\mathbb{E}[r_{t_i}]$.
3. **Branching probabilities.** From each node, three branches (up, mid, down) with probabilities chosen to match the mean and variance of $r_{t_{i+1}} - r_{t_i}$ given $r_{t_i}$.
4. **Mean-reversion drift.** The branching probabilities account for the mean-reversion drift $-a \, r_t$ in the SDE. Higher rates have a downward bias; lower rates have an upward bias.
5. **Discount factors.** Each step's discount is $e^{-r_{t_i} \, dt}$.

The standard *Hull-White trinomial tree* (Hull & White, 1996) gives explicit formulas for the branching probabilities. Numerical care is needed at extreme states to avoid negative probabilities; the procedure restricts the rate grid such that all branches have non-negative probabilities.

Backward induction:

```python
def hw_callable_bond(face, coupon, T, call_schedule, a, sigma, curve,
                     n_time_steps=300, n_rate_states=100):
    """Price a callable bond on a Hull-White trinomial tree."""
    # Build lattice
    lattice = build_hw_trinomial_tree(T, a, sigma, curve, n_time_steps, n_rate_states)
    # Initialize terminal payoff
    V = np.full(n_rate_states, face + coupon)  # at maturity
    # Backward induction
    for i in reversed(range(n_time_steps)):
        # Discount
        rates = lattice.rates(i)
        V = V * np.exp(-rates * lattice.dt)
        # Add coupon if applicable
        if lattice.is_coupon_date(i):
            V += coupon
        # Apply call constraint
        if i in call_schedule:
            call_price = call_schedule[i]
            V = np.minimum(V, call_price)  # issuer calls if V > call_price
        # Roll back through trinomial
        V = lattice.expectation(V)
    # Return value at root
    return V[lattice.root_index]
```

For a 30-year callable bond, the function runs in ~50 ms. Production libraries hand-tune the inner loops; sub-millisecond is achievable.

## 7. Swaption pricing

European swaptions admit a closed-form solution under Hull-White via the *Jamshidian decomposition* (Jamshidian, 1989).

![Swaption pricing: Black-formula closed form under Hull-White](/imgs/blogs/short-rate-models-vasicek-hull-white-7.png)

The decomposition works as follows. A swaption is an option on a swap. The swap's value at expiry $T_0$ is

$$
V_{\text{swap}}(T_0) = \sum_i c_i P(T_0, t_i) - 1
$$

where $c_i$ are coupon adjustments and $P(T_0, t_i)$ are bond prices at $T_0$ for cashflow dates $t_i$.

The swaption pays $\max(V_{\text{swap}}(T_0), 0)$ at $T_0$.

Jamshidian's insight: under Hull-White, all $P(T_0, t_i)$ are *monotone* in $r_{T_0}$. So the swap's value is monotone in $r_{T_0}$. Find $r^*$ such that $V_{\text{swap}}(T_0; r^*) = 0$. Then the swaption is in the money iff $r_{T_0} < r^*$, and the swaption value decomposes into a portfolio of options on individual zero-coupon bonds:

$$
V_{\text{swaption}}(0) = \sum_i c_i \cdot V_{\text{ZBPut}}(0, T_0, t_i, K_i)
$$

where $K_i = P(T_0, t_i; r^*)$ is the strike of each zero-coupon bond put, and each $V_{\text{ZBPut}}$ has a closed-form Black-formula expression.

This decomposition turns swaption pricing from a 1-D PDE into a sum of closed-form Black formulas. The cost: solving a 1-D root-finding for $r^*$, which is fast.

For *Bermudan* swaptions (multiple exercise dates), the Jamshidian decomposition does not apply; lattice or PDE pricing is required. The lattice extends the European pricing by adding the early-exercise check at each exercise date.

### 7.1 Caplet pricing under Hull-White

A caplet is a call on a forward LIBOR/SOFR rate. Under Hull-White, each caplet has a closed-form Black-formula price.

The mechanics:

1. The caplet payoff at time $T_2$ (settlement) is $\delta \cdot \max(L(T_1, T_2) - K, 0) \cdot N$, where $L$ is the realised forward, $\delta$ is the day-count fraction, $K$ is the strike.
2. Under Hull-White, $L(T_1, T_2)$ has a known distribution at time $T_1$.
3. The caplet price is

$$
V_{\text{caplet}}(0) = N \cdot P(0, T_2) \cdot [F(0, T_1, T_2) \cdot N(d_1) - K \cdot N(d_2)]
$$

where $F$ is today's forward rate, $d_1, d_2$ are the standard Black formula expressions with volatility computed from Hull-White parameters.

Caps are sums of caplets:

$$
V_{\text{cap}} = \sum_i V_{\text{caplet}}(0; T_i, T_{i+1}, K).
$$

Each caplet has $O(1)$ closed-form pricing; cap is $O(N)$. For a 5-year cap with quarterly fixings, that's 20 caplets — milliseconds total.

Caps and swaptions are both standard calibration instruments for Hull-White. Caps are simpler (one tenor at a time); swaptions cover the full $(M \times N)$ surface. Most banks calibrate to swaptions because that matches their swaption book directly.

## 8. Two-factor Hull-White

A one-factor model has limitations: it cannot simultaneously match the volatility of long-end and short-end rates, nor can it capture the correlation between curve level and slope.

![Two-factor Hull-White: capturing curve-shape dynamics](/imgs/blogs/short-rate-models-vasicek-hull-white-8.png)

The two-factor extension:

$$
dr_t = (\theta(t) + u_t - a r_t) \, dt + \sigma_1 \, dW_1
$$

$$
du_t = -b u_t \, dt + \sigma_2 \, dW_2, \qquad \mathbb{E}[dW_1 \, dW_2] = \rho \, dt.
$$

Five parameters: $(a, b, \sigma_1, \sigma_2, \rho)$. The factor $u_t$ is a slow-moving mean-reverting process that adjusts the drift of $r_t$. Combined, the two factors capture both the level (PC1) and slope (PC2) dynamics of the curve.

Calibration: fit the swaption surface (typically 30+ swaption quotes covering expiry × tenor) to find optimal $(a, b, \sigma_1, \sigma_2, \rho)$. Levenberg-Marquardt over 5 parameters; converges in 1-3 minutes.

Trade-offs:

- **One-factor**: fast, simple, sufficient for many vanilla products. Good for Bermudan swaptions on short tenors.
- **Two-factor**: slower, more parameters, better fit for long-dated curve-sensitive exotics (cross-currency callables, range accruals).

Multi-factor extensions (Cheyette model, multi-factor HJM) add more state variables and capture richer curve dynamics, at higher computational cost.

## 9. Black-Karasinski

Black-Karasinski (1991) models the *log* of the short rate as Hull-White:

$$
d(\log r_t) = (\theta(t) - a \log r_t) \, dt + \sigma \, dW_t.
$$

![Black-Karasinski: log-normal short rate for positivity](/imgs/blogs/short-rate-models-vasicek-hull-white-9.png)

Equivalently, $r_t$ is *log-normal* with mean-reverting log. Rates are strictly positive (since $\log r_t$ takes any real value, $r_t = e^{\log r_t} > 0$).

Pros:
- Rates strictly positive.
- Log-normal terminal distribution matches market convention pre-2014.
- Mean-reverting structure gives reasonable long-run behaviour.

Cons:
- No closed-form bond pricing. Every price requires numerical solution.
- Cannot price negative rates.
- Pricing speed slower than Hull-White.

When to use Black-Karasinski:

- **Emerging-market currencies** where rates have always been strictly positive.
- **Legacy product books** built before 2014, when negative rates were unthinkable.
- **Specific products** that require strict positivity (e.g., some commodity-like rate references).

For most modern G10 books, Hull-White (with negative-rate tolerance) has displaced Black-Karasinski. The shift toward shifted/Bachelier models in negative-rate regimes accelerated this.

A common variant: *shifted Black-Karasinski* uses $\log(r_t + s)$ for some constant shift $s$, allowing rates to go below $-s$ but no further. This generalisation handles the post-2014 negative-rate world while preserving positivity-of-shifted-rate.

## 10. CIR

Cox-Ingersoll-Ross (1985) uses square-root diffusion:

$$
dr_t = a(\theta - r_t) \, dt + \sigma \sqrt{r_t} \, dW_t.
$$

![CIR: square-root diffusion with explicit positivity](/imgs/blogs/short-rate-models-vasicek-hull-white-10.png)

The square root in the diffusion makes volatility scale with $\sqrt{r_t}$. The model has several useful properties:

- **Non-negative rates.** When $r_t = 0$, the diffusion vanishes and the drift $a \theta > 0$ pushes the rate back up.
- **Feller condition.** If $2 a \theta > \sigma^2$, rates stay strictly positive; the boundary at zero is unattainable.
- **Closed-form bond prices.** Affine term structure with explicit $A, B$ in terms of model parameters.
- **Closed-form bond options.** Via the non-central chi-squared distribution.

CIR is used for:

- **Credit intensity** in reduced-form default models. The default intensity $\lambda_t$ follows a CIR process; default time is the first jump of a Poisson process with intensity $\lambda_t$.
- **Variance dynamics** in Heston model. The variance $v_t$ follows CIR.
- **Short rate** in some legacy systems, particularly emerging markets.

CIR's place in modern finance is more in credit and stochastic-vol modelling than in pure rates. Hull-White dominates rates; CIR persists in adjacent applications.

### 10.1 The LMM intuition

LMM models forward rates *directly* under the appropriate forward measures. The intuition: cap and swaption markets quote in implied vol per *forward* (not per short rate); LMM matches this convention.

Under the $T_{i+1}$-forward measure, the forward $F_i(t)$ is a martingale. Hence:

$$
\frac{dF_i(t)}{F_i(t)} = \sigma_i(t) \, dW_i^{(T_{i+1})}.
$$

Each forward $F_i$ has its own volatility process $\sigma_i(t)$. Calibrating $\sigma_i$ to caps gives an exact fit per caplet.

### 10.2 The drift adjustment

Simulating LMM under a single common measure (e.g., the spot or terminal measure) requires *Girsanov-adjusted drifts* on each forward. Under the spot measure, the drift of $F_i$ is

$$
\mu_i(F, t) = \sum_{j > i} \frac{\delta_j F_j \sigma_i \sigma_j \rho_{ij}}{1 + \delta_j F_j}
$$

where $\delta_j$ is the day-count fraction, $\rho_{ij}$ is the correlation between $W_i$ and $W_j$.

This drift is non-trivial: it depends on all forwards $F_j$ for $j > i$. Simulating LMM under a common measure thus requires careful handling of the cross-forward dependencies; Monte Carlo is the standard approach.

The drift adjustment is the source of LMM's complexity. Vanilla products (where the underlying is a single forward) avoid most of this; exotic products that depend on multiple forwards pay the full computational cost.

## 11. LIBOR market model

The LIBOR market model (LMM, Brace-Gatarek-Musiela 1997) takes a fundamentally different approach: instead of modelling the short rate, model the *forward LIBOR rates* directly.

![LIBOR market model: forward rates as the primary objects](/imgs/blogs/short-rate-models-vasicek-hull-white-11.png)

The setup: discretise the time axis into periods $[T_0, T_1, ..., T_N]$. For each $i$, define $F_i(t) = $ forward rate from $T_i$ to $T_{i+1}$ as observed at time $t$.

Under the $T_{i+1}$-forward measure, $F_i(t)$ is a *martingale*:

$$
\frac{dF_i(t)}{F_i(t)} = \sigma_i(t) \, dW_i^{(T_{i+1})}.
$$

Under any other measure (e.g., the spot measure), $F_i$ has a non-zero drift via Girsanov adjustment:

$$
\frac{dF_i(t)}{F_i(t)} = \mu_i(F, t) \, dt + \sigma_i(t) \, dW_i.
$$

The drift $\mu_i$ depends on the volatilities $\sigma_j$ and forward rates $F_j$ for $j > i$ — it captures the difference between the $T_{i+1}$-forward measure and the chosen common measure.

Advantages of LMM:

- **Natural fit for caps and swaptions.** Each caplet is an option on $F_i$; LMM gives exact Black-formula pricing for caplets.
- **Volatility skew via local-vol.** Per-forward $\sigma_i(t)$ allows different vol structures across tenors.
- **Cross-currency LMM** extends naturally for cross-currency rate exotics.

Disadvantages:

- **High dimension.** $N$ forward rates means $N$ SDEs; pricing requires Monte Carlo over $N$-dim paths.
- **Computational cost.** Each MC path is expensive; calibration is slower than Hull-White.
- **Drift adjustments.** Monte Carlo simulation under a single measure requires Girsanov-adjusted drifts that depend on all forwards.

Trade-off: most banks use Hull-White for vanilla rate-derivatives and LMM for the exotic-rates book. The split is by product complexity, not by absolute superiority.

## 12. Pricing engine architecture

For short-rate models, three pricing engine types cover the product spectrum.

![Pricing engine architecture: lattice / PDE / Monte Carlo](/imgs/blogs/short-rate-models-vasicek-hull-white-12.png)

**Lattice (trinomial tree).** Backward induction. Fast (50 ms typical). Ideal for callable / Bermudan products with one factor.

**PDE.** Finite-difference solver. Crank-Nicolson typical. Handles continuous boundaries (e.g., barrier knock-out). 1-2 factors.

**Monte Carlo.** Path simulation. LSM regression for early exercise. Scales to many factors. Slower (seconds-minutes per pricing).

Decision rules:

| Product | Engine | Why |
| --- | --- | --- |
| Vanilla zero-coupon bond | Closed-form (Hull-White affine) | $O(1)$ |
| Vanilla coupon bond | Closed-form summation | $O(N)$ |
| European swaption | Closed-form (Jamshidian) | $O(N)$ |
| Callable bond | Lattice | Backward induction with constraint |
| Bermudan swaption | Lattice | Same |
| MBS | Monte Carlo with prepay model | Path-dependent |
| Range accrual | Lattice or PDE | Path-dependent in 1-D |
| Cross-currency callable | 2-factor Monte Carlo | High-dim |
| Inflation-linked callable | 2-factor PDE | Real + nominal curves |

A serious bank's rates pricing library supports all five engines and routes products to the right engine automatically.

### 12.1 Performance tuning the lattice

A trinomial lattice with $N_t$ time steps and $N_r$ rate states does $\sim N_t \times N_r$ work. For a 30-year bond with daily steps and 100 rate states, that's $\sim 750{,}000$ operations.

Optimisations:

- **Pre-compute $\theta(t)$** on the time grid before pricing. Avoids recomputation per pricing call.
- **Vectorise the inner loop.** NumPy / Eigen / hand-tuned SIMD; 5-10x speedup over scalar.
- **Reduce time steps where possible.** Daily steps for the year before maturity; weekly steps thereafter; cuts work by 4-5x with minimal accuracy loss.
- **Reuse lattice across products.** Same Hull-White model means same lattice; reuse between calibration and pricing.

A production-quality Hull-White lattice prices a callable bond in 5-50 ms depending on tenor and step size. For a book of 5000 callable bonds, full revaluation takes 25 seconds to 4 minutes.

### 12.2 Lattice vs PDE: when each wins

Lattice (trinomial tree) and PDE (finite difference) are mathematically related — both discretise the same continuous-time problem. Choosing between them:

**Lattice wins when:**
- The product has discrete exercise dates (Bermudan).
- One factor is sufficient.
- Backward induction is the natural algorithm.
- Speed matters (lattice is typically 2-3x faster).

**PDE wins when:**
- The product has continuous boundaries (barrier knock-out).
- Time-stepping needs to be fine and adaptive.
- The discretisation grid needs to be non-uniform.
- The product is multi-dimensional (PDE handles 2-3 dimensions; lattice gets unwieldy).

In practice, most banks use lattice for callable / Bermudan products and PDE for barrier / continuous-feature products. Both are available; the engine routes products to the appropriate one.

## 13. Production architecture

Short-rate models live in the production stack as Layer 2: above the curve, below exotic pricing.

![Production architecture: short-rate models in the pricing stack](/imgs/blogs/short-rate-models-vasicek-hull-white-13.png)

**Layer 1: Curve service.** Daily-bootstrapped. Publishes discount and forward curves. Versioned snapshots.

**Layer 2: Short-rate model service.** Consumes curves. Calibrates $(a, \sigma)$ to caps/swaptions. Publishes calibrated model parameters per snapshot.

**Layer 3: Exotic pricing.** Consumes calibrated model. Lattice / PDE / MC pricing of structured products.

**Layer 4: Risk service.** Aggregates exposures. Bumps curve and / or model parameters. Produces sensitivities.

The architectural principle: each layer has a clean, versioned interface. Layer 2 consumers (Layer 3 + Layer 4) refer to a specific model snapshot ID; reproducibility is guaranteed.

The production cadence:

- **08:00**: curve service publishes morning curve.
- **08:05**: short-rate model service calibrates and publishes parameters.
- **08:10**: exotic pricing service prices the book.
- **08:15**: risk service aggregates and publishes risk metrics.
- **Throughout day**: intraday refits on threshold market moves.
- **EOD**: full deep calibration with hold-out tests.

Each layer's service has its own SLA, monitoring, and on-call rotation. Failures cascade gracefully via fallback to last-known-good versions.

## 14. Failure modes

Short-rate models fail in several recognised regimes.

![Failure modes: where short-rate models go wrong](/imgs/blogs/short-rate-models-vasicek-hull-white-14.png)

**Negative rates.** Post-2014 EUR; classical log-normal models (Black-Karasinski) break. Mitigation: switch to Hull-White (allows negative); shifted variants of Black-Karasinski; Bachelier (normal vol).

**Smile mismatch.** Hull-White has constant $\sigma$; cannot fit caps/swaptions smile. Mitigation: SABR overlay on Hull-White; Cheyette model; LMM with local-vol per forward.

**Mean-reversion drift miscalibration.** Mean-reversion speed $a$ is hard to identify from short-window data. Too low: model rates diverge over long horizons. Too high: model loses flexibility. Mitigation: calibration regularisation; multiple calibrations across history; senior-quant review of $a$ time series.

**Long-dated divergence.** One-factor cannot match $30\text{y} \times 30\text{y}$ swaption vol and $1\text{y} \times 5\text{y}$ vol simultaneously. Mitigation: two-factor Hull-White; Cheyette; LMM.

**Calibration instability.** Daily refits produce parameter jumps that don't reflect market changes. Mitigation: warm-starting from yesterday's parameters; regularisation toward yesterday; parameter time-series monitoring.

## 15. Case studies

### 15.1 Callable bond mispricing in the 1990s

In the 1990s, callable corporate bonds were often priced using simple option-adjusted methods that didn't capture mean reversion. As rates fell in the late 1990s and many bonds got called, dealers who had used flat-vol pricing took losses. The lesson: callable bond pricing needs proper mean-reverting models like Hull-White.

### 15.2 MBS prepayment model failures (2007-2008)

Mortgage-backed securities were priced under prepayment models calibrated on 1990s-2000s data. The 2007-2008 housing crisis brought refinancing patterns that didn't match historical norms. Pricing systems with Hull-White short-rate and hardcoded prepayment models produced systematically wrong marks. The lesson: even with the right rate model, the prepay model is a separate point of failure.

### 15.3 Bermudan swaption mismarking at AIG

In 2008, AIG's structured-products desk mismarked Bermudan swaptions. The Hull-White calibration was technically correct but the (a, sigma) parameters had not been refreshed to reflect post-Lehman vol. Mark-to-model produced numbers that diverged substantially from where the same trades would clear. Losses contributed to AIG's bailout. The lesson: vol calibration must be live; stale vol is dangerous.

### 15.4 The 2014 negative-rates regime in Europe

When EUR rates went negative, log-normal short-rate models (Black-Karasinski) became undefined. Banks scrambled to migrate to Hull-White or shifted variants. The migration broke calibration regression tests; model parameters were not portable. The lesson: model class is regime-dependent; alternative parameterisations must be live before they're needed.

### 15.5 The 2019 SOFR transition planning

The 2019 announcement that LIBOR would cease in 2023 forced banks to plan for SOFR-based pricing. Short-rate models calibrated to LIBOR caps had to be re-calibrated to SOFR caps; the calibration set changed. Several banks took 18+ months to fully migrate. The lesson: the calibration instruments are an architectural concern; flexibility in instrument choice matters.

### 15.6 Two-factor Hull-White at Goldman Sachs

Goldman's rates desk migrated from one-factor to two-factor Hull-White around 2010 specifically because of long-dated curve-dependent products. The pricing cost rose 3-5x but the fit accuracy improved enough to justify the engineering investment. The lesson: more factors are worth it for products that depend on multi-factor curve dynamics.

### 15.7 LMM adoption at JP Morgan

JP Morgan's rates exotics desk uses LMM for products that depend heavily on caps/swaptions market quotes. The model-vs-Hull-White spread on benchmark exotics is 5-15 bp; for a $5B exotic book this is $25-75M annual revenue advantage. The investment in LMM infrastructure was worth it. The lesson: model choice has direct revenue implications.

### 15.8 The 2022 Fed hike cycle and Hull-White stress

The 2022-2024 hike cycle moved rates 525 bp in 18 months. Hull-White calibrations performed reasonably well; the model's structure handled the cycle without major issues. The lesson: well-engineered rates models are robust to large cycle moves; the stress is more on the calibration than the model itself.

### 15.9 The 2024 yen-rates breakout

When the BoJ raised rates in mid-2024, JPY rate dynamics changed regime from "permanently low" to "moving up." Banks with Hull-White calibrations to JGB caps had to refit; the parameters changed substantially. Books holding callable JGB notes faced model-driven mark changes. The lesson: regime shifts demand model-parameter regime shifts.

### 15.10 Cross-currency callable mispricing

Cross-currency callable notes (e.g., AUD-USD) require multi-factor models that capture both currencies' dynamics plus correlation. A bank pricing such products with single-currency Hull-White on each side, ignoring correlation, mispriced exotics by 50-200 bp. The fix was a 2-factor cross-currency model with explicit correlation. The lesson: multi-factor models are essential for cross-asset / cross-currency products.

### 15.11 The 1995 Bankers Trust derivative losses

Bankers Trust's structured-product desk in the early 1990s sold complex rate derivatives to corporate clients (Procter & Gamble, Gibson Greetings) using internal Hull-White-like models. When rates moved against the clients in 1994, the actual losses were 5-10x the model predictions. Investigations revealed model parameters that hadn't been refreshed against current market conditions; the calibration was stale. The lesson: stale calibrations on complex structured products can produce catastrophically wrong risk estimates. Daily calibration discipline matters.

### 15.12 Long-Term Capital Management's rates positions

LTCM held large convergence trades in interest-rate spreads (e.g., long off-the-run Treasury, short on-the-run; long swap-Treasury basis). These trades require careful Hull-White-style modelling for risk and hedging. LTCM's model risk metrics underestimated the volatility of spreads in stress; when 1998's flight-to-quality widened spreads, the marks went heavily against them. The lesson: even sophisticated rate models can miss tail behaviour in spreads.

### 15.13 The 2014 ECB negative-rates crossover

When the ECB cut rates below zero in 2014, several European banks discovered their pricing systems used Black-Karasinski for Bermudan EUR swaptions. The model was undefined for negative rates; the systems either crashed or produced NaN values. Some banks took weeks to migrate to Hull-White; during the migration, pricing was done by hand or with stop-gap systems. The lesson: model class is regime-dependent; alternative implementations should be live before they're needed.

### 15.14 The 2022 SOFR-LIBOR transition impact on rates models

Banks running Hull-White calibrated to LIBOR caps had to migrate to SOFR caps in 2022-2023. The calibration set changed; the calibrated $(a, \sigma)$ parameters shifted. Several banks took 12-18 months to fully migrate; during the transition, they ran parallel LIBOR and SOFR pricing systems. The lesson: rate-reference transitions are major engineering efforts; planning years ahead is essential.

## 16. When to use which model

| Scenario | Model | Why |
| --- | --- | --- |
| Vanilla swap / bond | Closed-form on curve | No model needed |
| Vanilla European swaption | Black formula on curve | No dynamic model needed |
| Callable / Bermudan with single rate factor | Hull-White lattice | Industry standard |
| Callable with smile sensitivity | SABR overlay or Cheyette | Captures vol smile |
| Long-dated cross-curve exotic | 2-factor Hull-White | Multi-factor dynamics |
| MBS | Hull-White Monte Carlo + prepay | Path-dependent |
| Cross-currency callable | 2-factor Hull-White MC | Correlation matters |
| Caps/swaptions smile-fitting | LMM with local vol | Direct fit |
| Range accrual | Lattice or PDE | Path-dependent in 1-D |
| Inflation-linked exotic | 2-factor with real curve | Real-nominal coupling |

### 15.15 The 2024 yen-rates regime shift

The Bank of Japan's mid-2024 rate hike was the first JPY rate increase in 17 years. Japanese banks had Hull-White calibrations that had been stable for nearly two decades; the regime shift forced rapid recalibration. Books holding callable JGBs faced significant model-driven mark changes. The lesson: long stability of model parameters is not a guarantee of future stability; structural regime shifts demand fresh calibrations.

### 15.16 Multi-factor model adoption at structured products desks

Several major banks (JP Morgan, Goldman, Morgan Stanley) migrated their structured products desks from one-factor Hull-White to two-factor Hull-White around 2010-2015. The motivation: long-dated cross-curve products were mispriced under one factor. The migration cost: 2-3x pricing time, 5x calibration time, 18+ months of engineering. The benefit: more accurate prices on long-dated exotics, worth millions per year on a large structured book. The lesson: model upgrades are recurring engineering investments justified by product economics.

### 16.1 Hull-White vs the alternatives: a deeper comparison

A more detailed comparison of model classes for production use:

| Aspect | Hull-White | Two-factor HW | Black-Karasinski | CIR | LMM |
| --- | --- | --- | --- | --- | --- |
| Closed-form ZCB | Yes | Yes | No | Yes | N/A |
| Closed-form bond options | Yes (Jamshidian) | Yes (extension) | No | Yes | Per-caplet |
| Negative rates | Yes | Yes | No | No | Possible |
| Multi-factor curve dynamics | No | Yes | No | No | Yes |
| Smile fit | Constant vol | Constant vol | Constant vol | Constant vol | Per-tenor vol |
| Calibration speed | Fast (sec) | Med (min) | Slow (min) | Fast (sec) | Slow (min) |
| Pricing speed (callable) | Fast | Med | Slow | Med | Slow |
| Production maturity | Universal | Common | Legacy | Niche | Specialist |
| Typical use | Vanilla callable / swaption | Long-dated cross-curve | Emerging-market | Credit / vol | Smile-driven exotics |

For most banks: Hull-White covers 80%+ of rate-derivative pricing. Two-factor is added for long-dated structured products. LMM is added for the exotic-rates desk specifically. CIR persists in credit modelling. Black-Karasinski is being phased out.

### 16.2 The economics of model selection

A bank's choice of rate model has direct revenue implications:

- **More accurate model** → tighter quote → more flow won.
- **Faster pricing** → better intraday risk management → smaller hedges.
- **Better stress handling** → smaller capital requirements.

A typical estimate: upgrading from one-factor to two-factor Hull-White on a $5B exotic book improves pricing accuracy by ~5-10 bp on average, worth ~$25-50M annually.

The investment cost: 1-2 quants for 12 months ($500K-$1M), plus infrastructure. The ROI is high but indirect.

For hedge funds and prop desks, similar logic applies but with tighter focus on alpha generation. A model that gives slightly better pricing on a high-volume book is a structural edge.

## 17. Three closing principles

**The model is a calibrated SDE, not a forecast.** Short-rate models tell you how the curve evolves under no-arbitrage. They are not predictions of future rates; they are calibrated processes for pricing.

**Calibration matters as much as the model.** A perfect model with bad calibration produces bad prices. Daily calibration discipline is the operational foundation.

**Multi-factor for multi-factor problems.** One-factor for single-factor exposures (vanilla callable on one currency). Multi-factor for multi-factor exposures (cross-currency, long-dated curve-shape).

### 17.1 Common bugs in rate-model implementations

A non-exhaustive list:

**Wrong sign on $\theta(t)$.** Bug in the closed form; rates trend incorrectly. Diagnostic: simulate paths and verify mean equals forward.

**Hull-White $a$ near zero.** Calibration produces $a \approx 0$; mean reversion vanishes; long-dated rates diverge. Diagnostic: check $a$ time series; floor at $a > 0.001$.

**Negative $\sigma$.** Calibration yields impossible $\sigma$. Diagnostic: bound $\sigma > 0$ in optimiser.

**Wrong volatility convention.** Calibration uses normal vol but pricing uses log-normal. Diagnostic: explicit convention tagging.

**Lattice probabilities outside [0, 1].** Numerical issue at extreme rate states. Diagnostic: check probabilities at every node; restrict rate grid.

**Off-by-one in time grid.** Pricing on day 0 uses tomorrow's discount. Diagnostic: explicit time-axis tests.

**Currency mismatch.** Calibration to USD swaptions; pricing on EUR product. Diagnostic: explicit currency on every model.

**Stale theta.** $\theta(t)$ computed from yesterday's curve. Diagnostic: refresh on every curve update.

**Non-monotone bond prices.** Implementation bug in lattice. Diagnostic: $P(0, T)$ should be monotone in $T$.

**Calibration RMSE acceptable but biased.** RMSE within tolerance but residuals systematic. Diagnostic: plot residuals; bias indicates model class doesn't fit shape.

A senior rates quant maintains a personal log of bugs. The list compounds.

### 17.2 Validation against external sources

Cross-validation patterns:

- **QuantLib comparison.** Run same calibration in QuantLib; compare prices. Differences should be <1 bp.
- **Bloomberg bench.** Major issues' bond prices in Bloomberg; production library should match within bid-ask.
- **Textbook test cases.** Hull-White paper has worked numerical examples; production library should reproduce.
- **Monte Carlo cross-check.** Lattice prices should match MC prices to within MC standard error.
- **Closed-form vs lattice.** European bond options closed-form vs lattice; should match.

Each cross-check is automated in CI; failures block deployment.

## 18. Production checklist

1. **Hull-White as the workhorse.** Single source of truth for vanilla callable products.
2. **Two-factor Hull-White for long-dated curve-sensitive products.** Calibrated weekly, monitored daily.
3. **Calibration validation gates.** Vanilla RMSE, parameter delta, hold-out fit.
4. **Lattice + PDE + MC engines** integrated with the model.
5. **Multi-currency support** with cross-currency basis curves.
6. **Negative-rate handling** via Hull-White or shifted variants.
7. **Smile overlay** (SABR) on Hull-White for products requiring smile fit.
8. **LMM as the exotic alternative** for caps-driven products.
9. **Audit logging** for every calibration and pricing.
10. **Cross-validation** against external sources.
11. **Stress testing** of model parameters under regime scenarios.
12. **Regression tests** running on each code deploy.

A library that ticks all 12 is production-grade.

### 18.1 Worked end-to-end calibration

To make Hull-White calibration concrete, a worked example.

**Inputs:**
- USD OIS curve (already calibrated): published by curve service.
- 30 swaption quotes: $(M, N)$ for $M, N \in \{1, 2, 5, 10, 30\}$. Each quote: implied normal vol in basis points.
- Today: 2026-05-03.

**Step 1: Compute $\theta(t)$.** Given trial $(a, \sigma)$, compute $\theta(t)$ on a fine time grid (daily, 30 years). Use closed form:

```python
def hull_white_theta(t, a, sigma, forward_curve):
    f = forward_curve.f(t)
    fp = forward_curve.f_prime(t)  # derivative
    return fp + a * f + (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t))
```

**Step 2: Price each swaption under model.**

```python
def hw_price_swaption(spec, theta_grid, a, sigma, curve):
    # Jamshidian decomposition + Black formula on ZCB options
    r_star = solve_root(spec, curve, a, sigma)
    return sum(coupon_i * zcb_put(spec.start, spec.cashflow_dates[i],
                                   r_star, a, sigma, curve)
               for i in range(spec.n_cashflows))
```

**Step 3: Compute model implied vol from price.**

```python
def hw_swaption_vol(spec, model_price, curve):
    # Convert price to Black implied vol
    return brent_solve_normal_vol(spec, model_price, curve)
```

**Step 4: Compute RMSE.**

```python
def calib_loss(params, swaption_specs, market_vols, curve):
    a, sigma = params
    if a <= 0 or sigma <= 0:
        return 1e9
    theta_grid = build_theta(a, sigma, curve)
    model_vols = [hw_swaption_vol(spec, hw_price_swaption(spec, theta_grid, a, sigma, curve), curve)
                  for spec in swaption_specs]
    return sum((mv - mkt)**2 for mv, mkt in zip(model_vols, market_vols))
```

**Step 5: Optimise.**

```python
from scipy.optimize import minimize
result = minimize(calib_loss, x0=[0.05, 0.01],
                  args=(swaption_specs, market_vols, curve),
                  method='L-BFGS-B', bounds=[(1e-4, 1.0), (1e-5, 0.5)])
a_calib, sigma_calib = result.x
```

For a 30-quote swaption surface, this typically converges in 5-10 iterations, ~3-10 seconds wall-clock.

**Step 6: Validate.** Compute RMSE over the calibration set (target: <2 normal vol points). Plot fit residuals to verify no systematic bias. Compare $(a, \sigma)$ to yesterday's; flag if jumped more than 2 sigma.

**Step 7: Publish.** Stamp with snapshot ID; push to consumer bus. Downstream pricing services pick up the new calibration.

The full pipeline is automated in production; junior quants walk through it manually before trusting the automation. Senior quants periodically spot-check the calibration outputs as a sanity check.

### 18.3 Performance benchmarks

For reference, performance benchmarks I have observed in production Hull-White implementations:

| Operation | Target | Acceptable | Stretch |
| --- | --- | --- | --- |
| $\theta(t)$ build (30y daily) | < 50 ms | < 200 ms | < 20 ms |
| Single ZCB price | < 1 µs | < 10 µs | < 100 ns |
| Coupon bond price | < 100 µs | < 1 ms | < 10 µs |
| European swaption price | < 5 ms | < 50 ms | < 1 ms |
| Callable bond (lattice) | < 50 ms | < 500 ms | < 10 ms |
| Bermudan swaption (lattice) | < 100 ms | < 1 sec | < 30 ms |
| Full swaption surface (30 quotes) | < 200 ms | < 2 sec | < 50 ms |
| Daily calibration | < 5 sec | < 30 sec | < 1 sec |
| Book revaluation (5000 callables) | < 5 min | < 30 min | < 1 min |

These are achievable on modern hardware (single-socket 16-core x86, M1/M2 Mac, or comparable). GPU acceleration of Hull-White MC simulations is possible but rarely worth the complexity for typical book sizes.

For HFT-style sub-millisecond pricing, hand-tuned C++ with SIMD is required; for typical desk operations, well-engineered C++ or vectorised Python suffices.

## 19. The cultural side of rates modelling

Rates quants are typically deep specialists. The math is heavy (stochastic calculus, PDEs, Monte Carlo) and the products are intricate (callable / Bermudan / MBS / swaptions). A rates quant's career path: 5 years building specific pricers; 5 more years owning a category; eventually leading the rates-modelling effort firm-wide.

Cultural practices that distinguish strong rates teams:

- **Daily calibration review.** 15 minutes every morning to validate yesterday's calibration, plot parameter time series, flag anomalies.
- **Convention discipline.** Every product spec carries day-count, schedule, currency, calibration instrument set.
- **Cross-team validation.** Rates quants validate exotic prices against simpler benchmarks; risk validates against quant.
- **Stress-test culture.** Each calibration tested against historical regimes (1994 hike, 2008 crisis, 2014 negative rates, 2022 cycle).
- **Open-source comparison.** Periodically compare against QuantLib's Hull-White; differences investigated.

Senior rates quants in 2026 are part-mathematician, part-engineer, part-trader. The combination is rare.

### 19.1 Daily routine of a rates quant

A typical day:

**07:30.** Pre-open. Review overnight curve and vol moves; check that overnight calibration completed clean.

**08:00.** Morning meeting with rates desk. Walk through yesterday's calibrated parameters; discuss any anomalies; plan today's calibration runs.

**08:30 - 12:00.** Active development: build new model variants, fix calibration bugs, optimise lattice performance.

**12:00 - 13:00.** Lunch and informal networking with traders.

**13:00 - 14:00.** Risk-committee preparation if applicable. Provide model-risk assessments.

**14:00 - 16:00.** Cross-team work: validate exotic prices against simpler benchmarks; resolve trader questions; deliver new product pricers.

**16:00 - 17:00.** End-of-day. Verify EOD calibration ran clean. Sign off on tomorrow's risk reports.

**17:00 - 18:00.** Personal research. Read papers, prototype new approaches.

The role mixes deep quantitative work with cross-team collaboration. Senior rates quants are valued for combining depth and breadth.

### 19.2 Common interview questions for rates quants

A senior rates quant manager might ask:

1. "Walk me through the Hull-White calibration to a swaption surface."
2. "Why does Hull-White have closed-form bond prices?"
3. "How do you handle negative rates in Hull-White?"
4. "What's the Jamshidian decomposition?"
5. "Why use two-factor instead of one-factor Hull-White?"
6. "How would you build a Hull-White trinomial lattice?"
7. "What's the trade-off between LMM and Hull-White?"
8. "How do you debug a calibration that produces wildly different parameters from yesterday?"

Strong candidates can articulate clean answers to all 8 plus extensions. Weak candidates struggle on operational details.

## 20. The future of rates modelling

Several trends shape the next decade:

**SOFR-native models.** Post-LIBOR, models calibrate to SOFR caps and swaptions; per-tenor SOFR forward curves; SOFR-linked structured products.

**Rough volatility for rates.** Empirical evidence suggests rate volatility is "rough" similar to equity vol. Rough Hull-White or rough HJM models capture this. Production deployment starting 2025-2026.

**Machine learning calibration.** Neural networks learning Hull-White calibration parameters from market features; faster than LM optimisation but require validation.

**Multi-asset rate models.** Joint dynamics of rates, FX, credit, inflation. Cross-asset structured products require coherent multi-asset modelling.

**ESG / climate-linked rates.** Climate-linked structured notes coupled to rate curves; new modelling required for transition pathways and physical risk.

**Real-time rates calibration.** Sub-minute Hull-White recalibration as caps/swaptions quotes move. Engineering challenge but feasible with modern hardware.

A senior rates quant entering 2026 will likely work on at least two of these frontiers.

### 20.1 The relationship between rates models and equity stochastic-vol models

A subtle observation: rate models and equity stochastic-vol models share more structure than is immediately obvious.

- **Hull-White** $\sim$ **Heston with constant volatility-of-variance.** Both are mean-reverting Gaussian models.
- **CIR** $\sim$ **Heston** (with $\beta = 0.5$). Square-root diffusion in both.
- **Black-Karasinski** $\sim$ **SABR with $\beta = 0$**. Lognormal mean-reverting underlying.
- **LMM** $\sim$ **Multi-strike SABR**. Each forward is a separate stochastic process.

Senior cross-asset quants recognise these parallels and apply techniques from one domain to the other. Adjoint algorithmic differentiation, characteristic-function pricing, and FFT methods all transfer directly.

### 20.2 Cross-asset rates-equity models

Some products require *joint* rates-equity dynamics:

- **Convertible bonds.** Rates affect bond floor; equity affects conversion value.
- **Equity-linked notes.** Rates discount; equity drives payout.
- **Hybrid structured notes.** Rates and equity both directly in payoff.

The standard approach: 2-factor model with Hull-White on rates and Black-Scholes (or local-vol) on equity, plus an explicit correlation $\rho_{r, S}$.

The pricing engine: 2D PDE or Monte Carlo. The PDE is straightforward but expensive (2D grid with $\sim 10^4$ time steps); MC is more flexible for path-dependent payoffs.

Calibration: rates parameters from caps/swaptions; equity parameters from equity options; correlation from cross-asset history (typically $\rho_{r, S} \approx 0.2$ for index, $0.1$ for single stocks). Senior practitioners refit correlation periodically; the parameter is empirically the most volatile.



Short-rate models are the dynamic layer above the yield curve. Hull-White, with its closed-form curve fit and tractable lattice / PDE / MC pricing, is the workhorse of rates derivatives. Vasicek, Black-Karasinski, CIR, and LMM each have their niches.

A senior rates quant operates fluently in the model-calibration-pricing pipeline. The math is well-understood; the engineering — calibration discipline, multi-factor extensions, regime-shift handling — is the daily craft.

The remaining articles in this series — [Exotic Derivatives](/blog/trading/quantitative-finance/exotics/exotic-derivatives), [Autocallables](/blog/trading/quantitative-finance/exotics/autocallables), and [Cliquets](/blog/trading/quantitative-finance/exotics/cliquets) — go deeper on specific product categories built on these rate models.

Rate models are the foundation under a multi-trillion-dollar interest-rate-derivatives market. Doing it well — calibrated, multi-factor where needed, regime-aware, audited — is the silent competence that powers every rate-sensitive product. The reward is a clear-headed view of one of the most quantitatively rigorous corners of finance.

A final reflection: rate models sit at the intersection of pure mathematics (stochastic calculus, PDEs) and operational engineering (calibration pipelines, validation gates, audit trails). The dual demands — mathematical correctness and operational robustness — are what make the discipline rewarding for those who can bridge both. Senior rates quants are valued because they can move fluently between the textbook proof and the production audit.

The discipline is also conservative in a productive way. Hull-White from 1990 is still the workhorse 35 years later because its assumptions are clear, its calibration is tractable, and its limitations are well-understood. Replacing it would require not just a better model but a better operational infrastructure for that model. Senior practitioners often choose Hull-White over more sophisticated alternatives because the operational reliability matters more than the marginal modelling improvement.

### 21.05 Engineering checklist for a Hull-White service

Beyond the production checklist in §18, more granular engineering points:

1. **Pre-compute and cache $\theta(t)$** on a daily / hourly time grid; refresh on curve update.
2. **Reuse lattice across products** that share the same model snapshot; amortise build cost.
3. **Separate calibration from pricing** as distinct services; calibration is slow, pricing is fast.
4. **Version model parameters** per snapshot; pricing pinned to specific calibration.
5. **Cross-validate against alternative models** (LMM, two-factor) on benchmark exotics weekly.
6. **Stress-test model parameters** under historical regimes (1994, 2008, 2014, 2022).
7. **Handle Feller / positivity** properly for CIR-type models; checks at every node.
8. **Limit pricing latency** with timeouts; refuse to publish if pricing exceeds threshold.
9. **Audit-log every price** with model snapshot ID, curve ID, position ID.
10. **Continuous integration** with regression tests on textbook examples and historical trades.
11. **GPU acceleration** for Monte Carlo where book size justifies it.
12. **Cross-language interop** (C++ pricing core, Python research, Java middle-office) via well-defined interfaces.

A library that ticks all 12 is production-grade.

### 21.1 The relationship between rate models and curves

A useful summary diagram (mental, not visual): the curve provides the *initial condition*; the rate model provides the *dynamics*; the pricing engine combines both. All three must be consistent.

- Curve says: today's $P(0, T)$ for all $T$.
- Hull-White says: $r_t$ evolves under SDE; bond price at any future $(t, r)$ is $A(t,T) e^{-B r}$.
- Engine says: backward-induct lattice or solve PDE to price products with embedded options.

If any layer is inconsistent (e.g., curve was calibrated to one set of swaps but model was calibrated against another), arbitrages appear in the prices. Senior architects audit consistency at every snapshot.

### 21.2 The maturity ladder for rates teams

Levels of rate-modelling maturity:

**Level 1 (basic).** Single Hull-White; daily calibration; manual pricing per product. Adequate for small books.

**Level 2 (functional).** Multi-currency Hull-White; automated pricing; basic exotic support. Standard at mid-tier.

**Level 3 (mature).** Hull-White + 2-factor + LMM; full lattice/PDE/MC engines; smile overlay; intraday recalibration. Standard at major banks.

**Level 4 (frontier).** ML-augmented calibration; rough-vol extensions; cross-asset coherent models; real-time pricing. Tier-1 banks 2024-2026.

A senior architect assesses the team's level and plans investment to advance. Level 2→3 is typically 2-3 years; Level 3→4 is comparable.

For engineers entering the field: master Hull-White first. Understand every line of its calibration, every step of its lattice, every closed form for its bond pricing. Then explore multi-factor, smile overlays, LMM. The depth on the workhorse pays back across the entire rates-modelling career.

### 20.3 ML-augmented rates modelling

A few research and production directions where ML is augmenting traditional rate models:

**Faster calibration via neural networks.** Train a network to map (curve, vol surface) → (Hull-White $a, \sigma$). Once trained, inference is sub-millisecond vs the 1-3 seconds for LM optimisation. Useful for high-frequency intraday recalibration.

**Smile fitting via neural-network volatility.** Replace SABR with a learned volatility function. Fits market quotes more flexibly; risk: overfitting and lack of arbitrage-freeness guarantees.

**Reinforcement learning for hedge optimisation.** RL agents that learn optimal hedging strategies under transaction costs and discrete rebalancing. Better than naive delta-hedging in some regimes.

**Generative scenario models.** Train a generative model on historical curve dynamics; sample synthetic scenarios that capture the empirical distribution of curve moves. Used for stress testing and ML-driven robust pricing.

**Hybrid models.** Use Hull-White for the structural dynamics + ML correction for residuals. Combines interpretability of classical models with flexibility of ML.

These approaches are still maturing in production. Senior rates quants who can deploy ML responsibly — with proper validation, audit trails, and arbitrage checks — are increasingly valuable.

### 20.4 Alternative-rate / SOFR-only models

Post-LIBOR transition, several novel modelling approaches have emerged:

**Forward Volatility Model (FVM).** Models forward-rate volatilities as the primary objects (similar to LMM but with volatility as state). Useful for SOFR-specific curve dynamics.

**Swap-rate market model (SMM).** Models par swap rates directly under appropriate forward measures. Naturally fits swaption surface.

**Term-rate models.** SOFR has both an overnight component and term components (3M, 6M); models that capture this dual nature are emerging. Important for products that reference term SOFR (corporate loans).

**Compounded vs term discount.** Different markets compound SOFR differently; pricing systems must handle the conventions explicitly.

These are all production-relevant in 2024-2026. Banks have invested significantly in adapting their rate-model infrastructure.

### 20.5 The future of post-LIBOR rate modelling

The full implications of the post-LIBOR world for rate modelling are still emerging:

**Compounded SOFR vs term SOFR.** Different products reference different SOFR variants. Pricing systems must support both consistently.

**SOFR-specific smile dynamics.** SOFR has somewhat different vol dynamics than LIBOR (more correlated to overnight movements; lower term spread); calibration sets must reflect this.

**SOFR-only structured products.** New products emerging that reference compounded SOFR with bespoke conventions. Pricing infrastructure adapts.

**Cross-currency basis post-LIBOR.** USD-EUR / USD-JPY / etc. cross-currency basis is now between SOFR and ESTR / TONA, not LIBOR-LIBOR. The basis dynamics are slightly different.

A senior rates quant in 2026 navigates the post-LIBOR transition as the new normal; the LIBOR era is now legacy. Junior quants will not have direct LIBOR experience; the models they learn are SOFR-native from day one.

### 20.6 Cross-asset implications of rate-model choice

A subtle architectural point: the choice of rate model has implications for cross-asset pricing.

- **Convertible bonds** (rate × equity): Hull-White on rates, BS or local-vol on equity. Joint calibration matters.
- **Cross-currency swaps**: Hull-White on each currency, plus FX dynamics. Cross-currency basis curves required.
- **Inflation derivatives** (real × nominal): Two-factor Hull-White with nominal and real components.
- **Credit-rate hybrids** (CDS-equity, bond-CDS basis): Joint dynamics with intensity and rate processes.

A senior cross-asset architect ensures rate-model consistency across all asset classes. Inconsistencies (e.g., one Hull-White calibration for rates desk, another for structured products) produce cross-asset arbitrages internal to the firm.

A senior rates quant builds infrastructure that will outlive them at the firm. Hull-White implementations from 1995 are still running in production; every quant who works on them inherits the architectural decisions of their predecessors. Building well, documenting clearly, designing for evolution — these are the responsibilities of the senior practitioner. The reward is the satisfaction of having contributed to infrastructure that prices trillions of dollars of risk, day after day, for decades.

A final philosophical reflection: the success of Hull-White and other short-rate models is in their *boundedness*. They make specific simplifying assumptions (Gaussian, mean-reverting, single factor) and work within those limits. They are not "the truth" about rate dynamics; they are *engineering compromises* that work well enough for most products. Senior practitioners are comfortable with this boundedness. They know the model has limits; they know what those limits are; they avoid using the model outside those limits.

This intellectual humility distinguishes mature practitioners from over-confident juniors. A junior often thinks "the model gives me a price, therefore the price is right." A senior thinks "the model gives me a price under specific assumptions; if those assumptions hold, the price is reasonable; if they don't, I need to think harder." The shift takes years to internalise.

The boundedness of rate models also has a deep mathematical reason: the *no-arbitrage* condition is much weaker than people sometimes believe. Many models can be no-arbitrage-consistent with the same liquid quotes; the choice between them is informed by what the practitioner believes about *unobservable* dynamics (vol of vol, correlation, regime persistence). Hull-White, with its Gaussian structure, makes one set of choices. SABR makes another; LMM another; HJM another. None is "the truth"; each is a reasonable engineering compromise for specific use cases.

For engineers entering the field: spend a year mastering Hull-White's mechanics. Understand every line of $\theta(t)$ derivation, every step of lattice construction, every nuance of swaption pricing. Then explore alternatives. The depth on the workhorse pays back across the entire rates career.

The final piece of advice: read the original papers. Vasicek 1977, Hull-White 1990, Black-Karasinski 1991, Jamshidian 1989, Cheyette 1996. Each is short and self-contained. The papers are clearer than most textbook treatments because the authors had to convince skeptical readers; the writing is precise and the motivation is explicit. Senior rates quants rarely read modern textbook treatments; they go back to the originals.

### 21.3 The economics of model selection at a tier-1 bank

To put a number on the value of getting model selection right: at a tier-1 bank with a $5B exotic-rates book, the difference between a well-calibrated two-factor Hull-White and a poorly-tuned single-factor model can be 5-15 bp on average pricing accuracy. On annual flow, that translates to $25-75M of revenue captured or lost.

The investment to deploy and maintain two-factor Hull-White: roughly 2-3 senior quants for 18 months ($1.5-3M), plus ongoing operational cost of $500K-$1M/year. Net ROI: 10x to 30x.

For prop trading firms and hedge funds, the calculus shifts: model edge translates more directly to alpha. A fund that prices Bermudan swaptions 5 bp better than competitors can build a structurally profitable book. The model investment is justified by the alpha attribution.

For smaller institutions: license commercial libraries (Numerix, Bloomberg DLIB) and customise as needed. Building Hull-White from scratch is rarely the right choice for a 5-person team.

### 21.4 The relationship between Hull-White and the broader rates-modelling ecosystem

A useful mental map:

- **Hull-White** sits at the heart of vanilla rates-derivative pricing.
- **SABR** sits to its side for swaption smile fitting.
- **LMM** sits above for cap/swaption-driven exotic books.
- **Two-factor Hull-White** sits above one-factor for long-dated curve-shape exotics.
- **HJM** sits above all of them as the most general framework, but rarely deployed in production.
- **CIR** sits in adjacent applications (credit, vol).

A senior rates quant knows where each tool fits and switches between them as products demand. The flexibility is the senior practitioner's value.

### 21.5 Closing reflections on the discipline

After 50 pages of short-rate modelling, a final reflection on the discipline. Rate models are a 50-year-old field; they have been refined, productionised, audited, and stress-tested across multiple market regimes. The math is beautiful (stochastic calculus, PDEs, Monte Carlo); the engineering is demanding (calibration discipline, multi-curve frameworks, real-time pipelines); the operational reality is unforgiving (one wrong calibration produces a million-dollar mispricing).

The senior practitioner who navigates all three layers — math, engineering, operations — produces work that is durable. Hull-White implementations from the 1990s still run in production banks today. The infrastructure outlives the people who built it. The intellectual investment compounds across decades.

This is what rate modelling offers: a discipline where mathematical rigour, engineering craft, and operational discipline meet to power a multi-trillion-dollar market. The rewards for those who master it are intellectual depth, durable career value, and the satisfaction of contributing to infrastructure that prices risk every day for the global financial system.

For engineers entering 2026: the field is mature but still evolving. SOFR-native models, ML-augmented calibration, real-time pipelines, cross-asset coherence — each is an active frontier. A senior quant in this decade will work on at least one or two of these frontiers. The discipline rewards the patient, careful, and rigorous; it punishes the careless.

Welcome to rate modelling. Master Hull-White first; then explore. The career is long, the math is deep, and the work matters.

### 21.6 A practical reading list

For engineers diving deeper into the discipline:

**Original papers:**
- Vasicek (1977): "An Equilibrium Characterization of the Term Structure"
- Cox, Ingersoll, Ross (1985): "A Theory of the Term Structure of Interest Rates"
- Hull, White (1990): "Pricing Interest-Rate-Derivative Securities"
- Black, Karasinski (1991): "Bond and Option Pricing When Short Rates Are Lognormal"
- Jamshidian (1989): "An Exact Bond Option Formula"
- Heath, Jarrow, Morton (1992): "Bond Pricing and the Term Structure of Interest Rates"
- Brace, Gatarek, Musiela (1997): "The Market Model of Interest Rate Dynamics"

**Textbooks:**
- Brigo & Mercurio (2006): "Interest Rate Models — Theory and Practice"
- Andersen & Piterbarg (2010): "Interest Rate Modeling" (3 volumes)
- Hull (2017): "Options, Futures, and Other Derivatives" (textbook treatment)

**Production code:**
- QuantLib documentation and source code
- ORE (Open Source Risk Engine) for swaption / Bermudan pricing examples

A senior practitioner has read most of these; the engineer entering the field should plan to work through them across 3-5 years.

A final tip: build small. Implement Vasicek from scratch in 100 lines of Python. Then implement Hull-White in 300 lines. Then build the trinomial lattice. Then add Jamshidian decomposition. Each step takes a week of focused work and provides irreplaceable understanding. The senior practitioner is the one who has built every component at least once and knows where the bodies are buried.
