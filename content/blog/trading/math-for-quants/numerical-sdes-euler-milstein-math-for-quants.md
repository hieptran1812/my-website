---
title: "Simulating an SDE properly: Euler, Milstein, and the two kinds of convergence"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "Writing down a stochastic differential equation is easy and simulating it correctly is not. Strong and weak convergence measure different things, discretisation bias is not Monte Carlo error, and the obvious discretisation of the Heston variance process goes negative on most paths."
tags: ["euler-maruyama", "milstein-scheme", "strong-convergence", "weak-convergence", "monte-carlo", "heston-model", "discretisation-bias", "stochastic-differential-equations", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 19
---

> [!important]
> **TL;DR:** A scheme that prices a vanilla correctly can be useless for anything path-dependent, because **strong** and **weak** convergence are two different rulers and Euler scores differently on each.
>
> - **Strong** convergence is pathwise: how far the simulated path lands from the true path on the *same* random draw. **Weak** convergence is distributional: how far the simulated *average* lands from the true average. Euler is order 0.5 strong and order 1.0 weak.
> - Measured on a \$100 stock at 30% vol with monthly steps: Euler's pathwise error is **\$1.994** a share, Milstein's is **\$0.1512**. Milstein adds one term and buys a full order.
> - **Discretisation bias is not Monte Carlo error.** At monthly steps the bias on a \$100 call is **\$0.01675**, and past **1.8 million paths** every additional path buys precision around the wrong number.
> - **Never Euler a geometric Brownian motion.** It has an exact solution, so the right scheme has zero discretisation error at any step size.
> - Naive Euler on the Heston variance process goes negative on **72.7%** of paths at monthly steps. On 250,000 options the repair you pick is worth **\$212,495** of price error, and reflection is worse than doing nothing clever.

Every quant can write down

$$
dS_t = \mu S_t\,dt + \sigma S_t\,dW_t
$$

and most can derive its solution. Far fewer can say what happens when you hand it to a computer, which cannot take infinitesimal steps and must approximate. That approximation introduces an error that is invisible in the output, does not shrink when you add paths, and changes size depending on what you are asking the simulation for.

That last clause is the whole post. The same scheme, run on the same model, is accurate for one question and inaccurate for another, and the vocabulary that separates the two cases is *strong* versus *weak* convergence.

![Two panels. Left, titled strong, shows a true path and a simulated path drifting apart with vertical connector segments marking the pathwise gap. Right, titled weak, shows two nearly overlapping bell curves with a small arrow between their peaks marking the gap in the average. A strip below reads: Euler is order 0.5 strong and order 1.0 weak, the same scheme with two different grades](/imgs/blogs/numerical-sdes-euler-milstein-math-for-quants-1.webp)

## Foundations: what discretising an SDE means

An [SDE](/blog/trading/math-for-quants/sdes-gbm-ou-cir-math-for-quants) is shorthand for an integral equation. Writing ${dX_t = a(X_t)\,dt + b(X_t)\,dW_t}$ is a compact way of saying

$$
X_T = X_0 + \int_0^T a(X_t)\,dt + \int_0^T b(X_t)\,dW_t,
$$

where the second integral is an [Ito integral](/blog/trading/math-for-quants/ito-integral-itos-lemma-math-for-quants) against Brownian motion. The function $a$ is the **drift**, the average direction of travel per unit time. The function $b$ is the **diffusion coefficient**, the size of the random kick per unit of square-root time.

A computer cannot evaluate those integrals. It can only chop ${[0,T]}$ into $n$ pieces of length ${\Delta t = T/n}$ and freeze the coefficients inside each piece. Do that and you get the **Euler-Maruyama scheme**:

$$
X_{k+1} = X_k + a(X_k)\,\Delta t + b(X_k)\,\Delta W_k, \qquad \Delta W_k \sim N(0, \Delta t).
$$

It is the obvious thing: evaluate the coefficients at the left endpoint, hold them constant, and step. The detail that surprises people arriving from ordinary differential equations is that the random increment scales like $\sqrt{\Delta t}$, not $\Delta t$. Halve the step and the drift contribution halves, while the noise only falls by a factor of 1.414. That mismatch is why [Brownian motion](/blog/trading/math-for-quants/brownian-motion-random-walk-math-for-quants) has infinite variation, and it is the root of everything below: the noise dominates at small steps, so the error is governed by how well you approximate the *stochastic* integral.

### Why there are two different errors

Suppose you simulate a path and compare it to the truth. There are two honest ways to ask "how wrong was that".

The first compares the paths themselves. Drive the scheme and the exact solution with the **same** Brownian increments, and measure how far apart they finish:

$$
e_{\text{strong}}(\Delta t) = \mathbb{E}\big|X_T^{\Delta t} - X_T\big|.
$$

The scheme has **strong order** $\gamma$ if this is bounded by ${C\,\Delta t^{\gamma}}$ for small $\Delta t$.

The second never compares paths at all. It compares the *answers* the two produce:

$$
e_{\text{weak}}(\Delta t) = \big|\mathbb{E}[f(X_T^{\Delta t})] - \mathbb{E}[f(X_T)]\big|
$$

for a payoff $f$. The scheme has **weak order** $\beta$ if this is bounded by ${C\,\Delta t^{\beta}}$.

These can differ, and for Euler they do: **Euler-Maruyama is order 0.5 strong and order 1.0 weak.** The reason is that the leading pathwise error term is a stochastic integral with mean zero. Pathwise it is a real error of size $\sqrt{\Delta t}$. In expectation it vanishes, so what survives in the average is the next term down, of size $\Delta t$.

The practical translation is short. **If the number you want is an expectation of a terminal payoff, you need only weak convergence.** A European call, a forward, a variance swap struck on a terminal quantity: the paths can be individually wrong as long as the distribution is right. **If the number depends on the path, you need strong convergence.** A barrier, a lookback, an American exercise boundary, a delta-hedging simulation, any pathwise Greek, anything where you compare a simulated path to something else on that same path.

## Milstein: one more term of the Ito expansion

The fix for Euler's weak pathwise accuracy is not a cleverer idea, it is simply the next term of the expansion. Inside one step,

$$
\int_{t_k}^{t_{k+1}} b(X_s)\,dW_s \approx b(X_k)\,\Delta W_k + b(X_k)b'(X_k)\int_{t_k}^{t_{k+1}}\!\!\int_{t_k}^{s} dW_u\,dW_s,
$$

because ${b(X_s) \approx b(X_k) + b'(X_k)(X_s - X_k)}$ and ${X_s - X_k \approx b(X_k)(W_s - W_k)}$ to leading order. Euler keeps the first term and throws the second away. But that double integral has a closed form: applying [Ito's lemma](/blog/trading/math-for-quants/ito-integral-itos-lemma-math-for-quants) to $W_t^2$ gives ${d(W^2) = 2W\,dW + dt}$, and rearranging over one step yields

$$
\int_{t_k}^{t_{k+1}}\!\!\int_{t_k}^{s} dW_u\,dW_s = \tfrac{1}{2}\Big[(\Delta W_k)^2 - \Delta t\Big].
$$

So the missing piece is computable from the increment you already drew. The **Milstein scheme** is Euler plus that term:

$$
X_{k+1} = X_k + a(X_k)\,\Delta t + b(X_k)\,\Delta W_k + \tfrac{1}{2}\,b(X_k)\,b'(X_k)\Big[(\Delta W_k)^2 - \Delta t\Big].
$$

It costs one extra multiplication per step, needs no extra random numbers, and lifts the scheme to **strong order 1.0**. The bracket has mean zero, since ${\mathbb{E}[(\Delta W)^2] = \Delta t}$, which is exactly why Milstein does nothing for weak convergence: the term it adds is invisible to an expectation. Milstein is order 1.0 strong and 1.0 weak, the same weak grade Euler already had. And if $b$ does not depend on the state, $b' = 0$ and Milstein *is* Euler: the correction only exists when volatility is state-dependent.

![A table comparing convergence, with rows for 12 monthly, 52 weekly and 252 daily steps, columns for Euler strong RMS error, Milstein strong RMS error and Euler weak bias on a one hundred dollar call, and a bottom row giving observed orders of 0.50, 1.00 and 0.95 measured from weekly to daily](/imgs/blogs/numerical-sdes-euler-milstein-math-for-quants-2.webp)

#### Worked example 1: what pathwise accuracy is worth on a hedging book

A risk desk runs a delta-hedging simulation on a **\$50m** equity position, 500,000 shares of a \$100 stock at 30% volatility, over one year. It is not pricing anything. It is asking what *this* path would have done to *this* hedge, so it is squarely a strong-convergence problem.

Simulate by Euler and by Milstein, both driven by the same Brownian increments as the exact solution, and measure the root-mean-square distance between the scheme's terminal price and the true one:

1. **Monthly steps.** Euler lands **\$1.994** away from the true terminal price per share, Milstein **\$0.1512**. On 500,000 shares that is ${\$1.994 \times 500{,}000 = \$997{,}000}$ of pathwise dispersion against ${\$0.1512 \times 500{,}000 = \$75{,}600}$. The scheme is injecting roughly \$921,400 of pure simulation noise into a hedging study.
2. **Daily steps.** Euler falls to **\$0.437** and Milstein to **\$0.00729**, so \$218,500 against \$3,645.
3. **The orders.** Between weekly and daily, Euler's error falls by a factor of ${0.960/0.437 = 2.197}$ while the step falls by ${252/52 = 4.846}$. Since ${\ln(2.197)/\ln(4.846) = 0.499}$, that is order **0.50**. Milstein falls by ${0.0353/0.00729 = 4.842}$ over the same refinement, an order of **1.00**.

The lesson: Euler needs four times the steps to halve its pathwise error, Milstein only two. Where you care about paths, that ratio is an hour of compute against a day of it.

## Discretisation bias is not Monte Carlo error

This is the distinction people get wrong. A Monte Carlo price carries **two** errors that behave completely differently.

**Monte Carlo error** is sampling noise. You drew ${M}$ paths instead of infinitely many, so your estimate wobbles around the expectation with standard deviation ${\sigma_f/\sqrt{M}}$. It is random, unbiased, shrinks as you add paths, and carries a confidence interval.

**Discretisation bias** is the gap between the expectation *you are actually computing* and the one you want. You are not sampling the model; you are sampling a slightly different model, the one your scheme defines. It is deterministic, it is signed, and **it does not move when you add paths**. More paths make you more confident about the wrong number.

![A log-log chart with paths simulated on the x-axis and error per option on the y-axis. A descending line labelled Monte Carlo standard error crosses three horizontal bias lines for monthly, weekly and daily steps, with crossing points marked at 1.8 million, 28 million and 570 million paths](/imgs/blogs/numerical-sdes-euler-milstein-math-for-quants-3.webp)

#### Worked example 2: the bias on a \$100 call, and the paths that cannot fix it

A desk prices a one-year at-the-money European call on a \$100 stock, 30% volatility, 4% rate. Black-Scholes gives the exact answer, **\$13.75326** a share, which is what makes this a clean test: the reference is a closed form, not a finer simulation.

Simulate with Euler and compare against the exact solution driven by the same Brownian increments, so the difference isolates the bias:

1. **Monthly steps (12 per year).** Euler overprices by **\$0.01675** a share. On a book of 250,000 options that is ${\$0.01675 \times 250{,}000 = \$4{,}187.50}$.
2. **Weekly steps (52).** The bias falls to **\$0.00418**, so \$1,045.00 on the same book.
3. **Daily steps (252).** It falls again to **\$0.00093**, so \$232.50.
4. **The observed order.** From weekly to daily, ${0.00418/0.00093 = 4.495}$ against a step ratio of 4.846, so ${\ln(4.495)/\ln(4.846) = 0.95}$. Theory says weak order 1.0; the shortfall is Monte Carlo noise on a bias this small.

Now the part that matters. The payoff's standard deviation here is **\$22.199**, so the Monte Carlo standard error on ${M}$ paths is \$22.199 divided by $\sqrt{M}$. Setting that equal to the monthly bias gives ${M = (22.199/0.01675)^2 = 1{,}756{,}000}$ paths. **Past about 1.8 million paths, a monthly-stepped Euler simulation stops getting better.** The weekly crossover is 28 million paths, the daily one 570 million.

A desk that answers a noisy price by throwing ten times the paths at it, without touching the step, buys a tighter confidence interval around a number still \$4,187.50 wrong.

#### Worked example 3: one compute budget, three ways to spend it

Because the two errors trade off, the step count and the path count have to be chosen together. Fix a budget of **2 billion simulated time steps**, so ${M \times n = 2{,}000{,}000{,}000}$, and spend it three ways. Total error is ${\sqrt{\text{bias}^2 + \text{standard error}^2}}$:

| steps/yr | paths | bias | std error | total | on 250,000 options |
| --- | --- | --- | --- | --- | --- |
| 12 | 167m | \$0.01675 | \$0.00172 | \$0.01684 | \$4,210 |
| 52 | 38m | \$0.00418 | \$0.00358 | \$0.00550 | \$1,375 |
| 252 | 7.9m | \$0.00093 | \$0.00788 | \$0.00793 | \$1,982.50 |

The monthly run is **all bias**: its confidence interval is ten times tighter than its error. The daily run is **all noise**: it nails the model, then cannot tell you where the answer is. The weekly split, where the two errors are roughly equal, is three times better than monthly for the same compute. That equality is the rule, and Duffie and Glynn (1995) make it precise: at the optimum the squared bias and the variance are the same order, which for a weak-order-1 scheme puts the step count near the cube root of the budget.

## Never Euler a geometric Brownian motion

All of the above assumes you have to discretise. Sometimes you do not.

Geometric Brownian motion has an exact solution. Applying Ito's lemma to $\ln S$ gives ${d\ln S = (\mu - \sigma^2/2)dt + \sigma\,dW}$, whose right-hand side has no state dependence at all, so it integrates exactly over any interval:

$$
S_{k+1} = S_k \exp\!\Big[\big(\mu - \tfrac{1}{2}\sigma^2\big)\Delta t + \sigma\,\Delta W_k\Big].
$$

This is not a scheme. It is the solution, sampled on a grid. It has **zero discretisation error at any step size**, strong and weak. If all you need is the terminal price, take one step of size ${T}$ and stop.

So every dollar of bias in worked example 2 was self-inflicted, and exists only because someone reached for Euler on a model that did not need it. The same holds for the [Ornstein-Uhlenbeck process](/blog/trading/math-for-quants/sdes-gbm-ou-cir-math-for-quants), which is Gaussian and steps exactly, and for CIR, whose transition law is a scaled non-central chi-square.

There is a second reason to care. Euler on GBM is not even **martingale-consistent**. Since ${\mathbb{E}[\Delta W] = 0}$, its expected terminal price telescopes to a closed form:

$$
\mathbb{E}\big[S_T^{\text{Euler}}\big] = S_0\,(1 + \mu\,\Delta t)^n \quad\text{against the exact}\quad S_0\,e^{\mu T}.
$$

With ${\mu = 4\%}$, ${T = 1}$ and 12 monthly steps, that is ${\$100 \times (1 + 1/300)^{12} = \$104.074154}$ against ${\$100 \times e^{0.04} = \$104.081077}$. The scheme misses the forward by **\$0.006923** a share before any option is priced, and no number of paths recovers it. Under the [risk-neutral measure](/blog/trading/math-for-quants/martingales-risk-neutral-measure-math-for-quants) the discounted stock is supposed to be a martingale, and this one is not. Notice too that the defect points *down* while the call bias points *up*, so the call is not overpriced because the forward drifted. It is overpriced because the Euler scheme's terminal distribution is the wrong shape, not because it sits in the wrong place.

## The Heston trap: a variance that goes negative

Now the case where you genuinely cannot avoid discretising. The [Heston model](/blog/trading/math-for-quants/jump-diffusion-stochastic-volatility-math-for-quants) is

$$
dS_t = r S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S, \qquad dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v,
$$

with the two Brownian motions correlated at $\rho$. The variance process is mean-reverting and, in continuous time, stays strictly positive whenever the **Feller condition** ${2\kappa\theta \ge \xi^2}$ holds.

Discretise it with Euler and that guarantee dies instantly:

$$
v_{k+1} = v_k + \kappa(\theta - v_k)\Delta t + \xi\sqrt{v_k}\,\Delta W_k.
$$

The increment $\Delta W_k$ is Gaussian and therefore unbounded below. Whatever $v_k$ is, some draw pushes $v_{k+1}$ below zero, and the next step needs $\sqrt{v_{k+1}}$. The scheme does not degrade gracefully. It throws. Nor is this a corner case: calibrated equity smiles routinely violate Feller, because fitting a steep short-dated skew wants a large vol-of-vol $\xi$.

![Top, a variance path starting at 0.04 that wanders down and dips below the zero line, with the dipping segment marked v below zero and the scheme needs the square root of v. Below, a table of four repairs with call prices of 8.56047 for absorption, 9.42046 for reflection, 8.02634 for full truncation and 7.71174 for Andersen QE, against a reference of 7.71049](/imgs/blogs/numerical-sdes-euler-milstein-math-for-quants-4.webp)

#### Worked example 4: which repair, and what it costs

Take ${v_0 = \theta = 0.04}$ (20% long-run volatility), ${\kappa = 1}$, ${\xi = 0.6}$, ${\rho = -0.7}$, ${r = 2\%}$, one year, a \$100 at-the-money call. Feller is violated badly: ${2\kappa\theta = 0.08}$ against ${\xi^2 = 0.36}$, a ratio of 0.22.

The reference price is **\$7.71049**, computed from the Heston characteristic function. I priced it twice, by the Gil-Pelaez two-probability integral and by the Carr-Madan damped-call transform, which are different derivations rather than the same algebra rewritten. They agree to ten decimal places, and both reproduce Black-Scholes to ten decimal places when fed a Black-Scholes characteristic function.

At monthly steps, **72.7%** of paths hit a negative variance at least once, and 12.8% of all individual steps do. Here is what each repair produces on a **250,000-option** book, with a Monte Carlo standard error of about \$0.00885 a share, \$2,212.50 on the book:

1. **Absorption**, ${v \leftarrow \max(v, 0)}$. Price **\$8.56047**, an error of **+\$0.84998** a share, which on 250,000 options is **\$212,495**.
2. **Reflection**, ${v \leftarrow |v|}$. Price **\$9.42046**, an error of **+\$1.70997**, or **\$427,492.50**. Reflection is *worse than absorption*: bouncing the variance back up injects volatility exactly where the model wanted none, and it is the intuitive fix that a candidate reaches for first.
3. **Full truncation** (Lord, Koekkoek and van Dijk, 2010). Keep $v$ as a signed number in the drift and use ${\max(v,0)}$ wherever the scheme actually evaluates it. Price **\$8.02634**, an error of **+\$0.31585**, or **\$78,962.50**. Against absorption it removes \$133,532.50 of error for one line of code.
4. **Andersen's Quadratic-Exponential scheme** (2008). Instead of patching a Gaussian step, moment-match the true non-central chi-square transition law: a squared-Gaussian draw when the variance is comfortably positive, an exponential-with-an-atom-at-zero draw when it is near zero. Price **\$7.71174**, an error of **+\$0.00125**, or **\$312.50**, which is well inside that \$2,212.50 noise band. It never produces a negative variance, by construction.

**What is standard:** QE is the production choice on derivatives desks and the benchmark in the literature. Full truncation is the standard cheap fallback and is comfortably the best of the plain Euler family, which is exactly what Lord and co-authors found. Absorption and reflection are what you get when nobody made a decision.

Note the direction of every error: all four repairs overprice. Clipping a variance process at zero can only add variance relative to a process that would have gone lower, and more variance means a more valuable option.

## The barrier the scheme never sees

The cleanest place where strong convergence turns into money is barrier monitoring.

A knock-out dies if the price *ever* touches the barrier. A simulated path is a finite list of points, and between two consecutive points the true path can dip below the barrier and come back. The scheme records two observations above the barrier and books a survivor. The market books a knock-out.

![A jagged price path with eight observation dots above a dashed ninety dollar barrier line. Between two dots observed at 92.10 and 91.40 the true continuous path dips to a low of 88.70 below the barrier and returns, with that excursion marked as the touch nobody recorded](/imgs/blogs/numerical-sdes-euler-milstein-math-for-quants-5.webp)

The bias is one-directional: discrete monitoring can only *miss* touches, never invent them, so a naively simulated knock-out is always too valuable. And the error shrinks like ${\sqrt{\Delta t}}$ rather than $\Delta t$, because what you are missing is the expected overshoot of a Brownian path between looks, a square-root quantity. That is strong convergence appearing as a booking error.

Two fixes exist, and neither is "use more steps". Simulate the **Brownian bridge** between each pair of points and sample whether the bridge crossed, which restores the continuous price at any step size. Or apply the **continuity correction** of Broadie, Glasserman and Kou (1997), which shifts the barrier by ${\exp(\pm 0.5826\,\sigma\sqrt{\Delta t})}$ and reuses the continuous formula. The [local time and reflection post](/blog/trading/math-for-quants/local-time-barriers-reflection-math-for-quants) derives that correction and prices it out in dollars.

## Common misconceptions

**"Halving the step halves the error."** Only for a weak-order-1 scheme measuring a weak quantity. Halving the step on Euler's *pathwise* error multiplies it by ${1/\sqrt{2} = 0.707}$, so you need four times the steps to halve it. The question "which error" has to be answered before the sentence means anything.

**"More paths will fix it."** More paths fix sampling noise and nothing else. Discretisation bias is a property of the scheme, not the sample, and sits at the same value whether you run a thousand paths or a billion. When a price stays off after a path-count increase, the step size is the suspect.

**"Milstein is always worth it."** Only when you need pathwise accuracy and volatility is state-dependent. For a vanilla European price you need only weak convergence, where Milstein and Euler are both order 1.0, so it buys nothing but arithmetic. And in multiple dimensions the correction needs Levy areas, iterated integrals with no closed form, which is why multi-asset Milstein is rare in production.

**"The Feller condition protects me."** It protects the continuous process. Every Euler discretisation can go negative regardless, because a Gaussian increment has no lower bound. Feller changes how often, not whether.

## Sources and further reading

The dollar figures in the worked examples are illustrative arithmetic on assumed inputs, computed for this post; the Black-Scholes and Heston reference prices are closed forms, and the simulation figures are measured against them rather than against a finer simulation.

- Kloeden, P. E. and Platen, E. (1992). *Numerical Solution of Stochastic Differential Equations*. Springer. The standard reference for strong and weak order, the Ito-Taylor expansion, and the Milstein scheme.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer. Chapter 6 covers discretisation bias, the bias-variance budget, and barrier simulation.
- Andersen, L. (2008). "Simple and efficient simulation of the Heston stochastic volatility model". *Journal of Computational Finance* 11(3), 1-42. The Quadratic-Exponential scheme.
- Lord, R., Koekkoek, R. and van Dijk, D. (2010). "A comparison of biased simulation schemes for stochastic volatility models". *Quantitative Finance* 10(2), 177-194. The full-truncation result.
- Broadie, M., Glasserman, P. and Kou, S. (1997). "A continuity correction for discrete barrier options". *Mathematical Finance* 7(4), 325-349.
- Duffie, D. and Glynn, P. (1995). "Efficient Monte Carlo simulation of security prices". *Annals of Applied Probability* 5(4), 897-905. The optimal bias-variance budget split.
- Heston, S. L. (1993). "A closed-form solution for options with stochastic volatility with applications to bond and currency options". *Review of Financial Studies* 6(2), 327-343.

## In the interview room and on the desk

The question arrives as **"how would you simulate this process?"**, usually with an SDE already on the whiteboard. It sounds like a coding question. It is not.

The strong answer starts by refusing to answer. **What is the number for?** A vanilla price is an expectation of a terminal payoff, so you need weak convergence and Euler at order 1.0 is fine. A barrier, a lookback, an American boundary, a hedging P&L study or any pathwise Greek needs strong convergence, and Euler at order 0.5 is not fine. Until the interviewer tells you which, every scheme choice is a guess. Candidates who open with "I would use Euler-Maruyama" have skipped the only part of the question that distinguishes them.

Then, in order: **does an exact scheme exist?** GBM and OU can be sampled exactly, so Euler on either is a self-inflicted wound. If not, **is the volatility state-dependent?** If yes and you need paths, Milstein costs one term and buys a full order. Then **quantify the two errors separately**, and note that they must be balanced, because a step count chosen without reference to the path count is chosen arbitrarily.

The trap is Heston. A candidate writes the Euler discretisation of the variance process on the board, and the interviewer waits. The scheme takes ${\sqrt{v_k}}$ of a quantity that will be negative on most paths at any realistic step size, and saying so unprompted is the single highest-value sentence available in this question. Naming full truncation as the cheap repair and Andersen's QE as the production one, and knowing that reflection is *worse* than absorption, separates someone who has run these simulations from someone who has read about them.

**Jane Street** and **Citadel** ask it in the "what could go wrong" register, where the negative variance is the point. Any **derivatives or exotics seat** weights it most heavily, because their products are path-dependent and the strong-versus-weak distinction decides whether their overnight risk run means anything. A pure alpha-research seat weights it least.
