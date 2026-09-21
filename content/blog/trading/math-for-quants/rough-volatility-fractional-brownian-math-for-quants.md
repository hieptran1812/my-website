---
title: "Rough volatility: the empirical fact that broke the standard models"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "For two decades every stochastic volatility model assumed volatility is a smooth diffusion. Then someone measured how rough realised volatility actually is, found a Hurst exponent near 0.1 instead of 0.5, and that one number explains the steep short-dated smile the old models could never fit."
tags: ["rough-volatility", "hurst-exponent", "fractional-brownian-motion", "rough-bergomi", "volatility-smile", "stochastic-volatility", "heston-model", "realised-volatility", "implied-volatility", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 22
---

> [!important]
> **TL;DR:** Every stochastic volatility model of the last thirty years assumed volatility is a smooth, mean-reverting diffusion. Then someone measured the regularity of realised volatility and found it far rougher than a diffusion permits, and that single measurement explains the steep short-dated smile those models could never fit.
>
> - The **Hurst exponent** $H$ measures how the size of a path's moves scales with the lag you look over. A diffusion forces $H = 0.5$. Gatheral, Jaisson and Rosenbaum (2018) measured $H = 0.142$ on S&P 500 realised volatility and $H$ "of order 0.1" across assets.
> - Rough means volatility reverses at **every** scale: it moves less than a diffusion demands over weeks and more over hours. In the illustrative arithmetic below, 0.175 against 0.313 at five days and 0.115 against 0.070 at two hours.
> - The payoff is the **short-dated smile**. A fractional driver gives an at-the-money skew scaling like $\tau^{H-1/2}$, which is $\tau^{-0.36}$ at $H = 0.14$: a blow-up. Any diffusion's skew saturates instead.
> - The cost is real: fractional Brownian motion is **not a semimartingale**, so no Ito calculus on the volatility driver, no Markov property, no closed forms, and slow simulation-based calibration.
> - **The number to remember:** on an illustrative book selling 2,000 weekly index puts, pricing the short end off a Heston fit gives away \$468,000 a week and \$6,084,000 a quarter, all of it the skew the model structurally cannot produce.
> - It is **not settled**. Cont and Das (2024) and Rogers (2019) argue the measured roughness is largely an artefact of estimating volatility from noisy prices.

## The measurement that broke the models

Two traders argue about a one-week index put. One prices it off a Heston model calibrated to the three-month surface, gets a number, and cannot understand why the market keeps paying more. The other stops modelling and looks at the data.

This is not a small discrepancy at the edge of a surface. It is the single most persistent failure of the entire stochastic volatility programme: models built by serious people over three decades, from [Heston's square-root variance process](/blog/trading/math-for-quants/jump-diffusion-stochastic-volatility-math-for-quants) onward, cannot produce a short-dated smile as steep as the one the market quotes. Add jumps and you can patch it, at the cost of a parameter that does nothing for you anywhere else on the surface.

The resolution came from measurement rather than modelling. Every one of those models writes volatility as a diffusion, and a diffusion is a particular kind of path: its moves scale with the square root of the lag. That is not a modelling choice anyone examined, it is what you get for free when you drive a process with Brownian motion. So Gatheral, Jaisson and Rosenbaum asked the obvious empirical question, which nobody had asked with high-frequency data: **does realised volatility actually scale that way?** It does not. The exponent that governs the scaling, the Hurst exponent, comes out near 0.1 instead of 0.5, on essentially every liquid asset they looked at.

That exponent is the whole post, so start with what it is.

![Three-column comparison of Hurst exponent 0.14 shown as rough, 0.5 as Brownian and 0.7 as trending, across increment scaling, the five-day versus one-day move ratio of 1.25 against 2.24 against 3.09, the two-hour ratio of 0.82 against 0.50 against 0.38, path behaviour, and where each is found, with a footer giving the measured S and P 500 value of 0.142](/imgs/blogs/rough-volatility-fractional-brownian-math-for-quants-1.webp)

One dial, three settings. The middle column is what every classical model assumes without saying so. The left column is what the data says. The gap between them is the rest of this article.

## Foundations: what you need first

**Realised volatility** is volatility you measure rather than imply. Take a day's worth of five-minute returns, square them, add them up, annualise the total: that is the realised variance for the day, and its square root is realised volatility. It is an estimate of how much the asset actually moved, computed from prices alone with no option and no model. A long history of daily values gives you a **time series of volatility itself**, which is the object whose path regularity we are about to measure.

**Brownian motion** $W_t$ ([built from scratch here](/blog/trading/math-for-quants/brownian-motion-random-walk-math-for-quants)) has independent Gaussian increments with $\mathrm{Var}(W_{t+\Delta} - W_t) = \Delta$. The standard deviation therefore scales as $\sqrt{\Delta}$. Double the lag and moves grow by a factor of 1.41; take a hundred times the lag and they grow tenfold. That square root is the signature of independence, and it is inherited by anything a Brownian motion drives.

**The Hurst exponent** generalises exactly that. Suppose a process has stationary increments whose typical size scales as a power of the lag:

$$\mathbb{E}\big|X_{t+\Delta} - X_t\big| \;=\; K\,\Delta^{H}.$$

The exponent $H$ is the Hurst exponent, and it is a statement about **path regularity**, nothing else. Three regimes, with no model attached to any of them:

- $H = 0.5$ is Brownian. Increments are uncorrelated, and the square-root law holds.
- $H \gt 0.5$ is **persistent**. Successive increments are positively correlated, so moves compound and the path looks trending.
- $H \lt 0.5$ is **rough** (or anti-persistent). Successive increments are negatively correlated, so a move up is more likely than not followed by a move down, at *every* time scale you examine. The path is visually jagged, and its moves grow more slowly with lag than a diffusion's.

The name is literal. The smaller $H$ is, the less regular the path is: a fractional path with $H = 0.14$ is continuous but violently more jagged than Brownian motion, which is itself already nowhere differentiable.

If you have met $H$ before it was probably applied to **prices**, in a rescaled-range or long-memory setting ([covered here](/blog/trading/math-for-quants/spectral-analysis-long-memory-math-for-quants)). Hold that separate. Everything below applies $H$ to the **volatility process**, and as you will see, that distinction does all the work.

## Measuring the exponent: the scaling of increments

Because $H$ is defined by how increment size scales with lag, you estimate it by measuring increment size at several lags and reading off the slope. Take logs of the relation above:

$$\log \mathbb{E}\big|X_{t+\Delta} - X_t\big| \;=\; \log K \;+\; H\log \Delta .$$

Plot the log mean absolute increment against the log lag and $H$ is the slope. For volatility you apply this to $\log \sigma_t$, because volatility is positive and roughly lognormal, and because the resulting model then has no trouble staying positive.

#### Worked example 1: backing out H from the scaling, and what it does to a \$3m vega book

The four increment sizes below are illustrative inputs chosen to be consistent with the published estimate, not a measurement of my own. Suppose the mean absolute change in log volatility comes out at 0.140 over one day and 0.175 over five days. Then between those two lags,

$$H \;=\; \frac{\log(0.175 / 0.140)}{\log 5} \;=\; \frac{\log 1.25}{1.6094} \;=\; 0.139 .$$

Add a 25-day reading of 0.220 and you get two more estimates for free: 0.142 from five days to 25 days, and 0.140 from one day to 25 days. Three overlapping windows agreeing to the third decimal is what "stable across scales" means in practice, and it is why the finding survived scrutiny.

![Table of mean absolute change in log volatility at lags of two hours, one day, five days and twenty-five days, showing measured values of 0.115, 0.140, 0.175 and 0.220 against the values a Hurst exponent of one half demands, 0.070, 0.140, 0.313 and 0.700, with implied exponents of 0.139 and 0.142, and a callout that five-day moves are 55.9 percent of what a diffusion needs while two-hour moves are 1.64 times as large](/imgs/blogs/rough-volatility-fractional-brownian-math-for-quants-2.webp)

Now force $H = 0.5$ and see what breaks. Anchored at 0.140 over one day, a diffusion demands $0.140 \times \sqrt{5} = 0.313$ over five days. The data shows 0.175, which is $0.175 / 0.313 = 55.9\%$ of it. Volatility moves barely half as much over a week as independence requires.

Run it the other way and the sign flips. Over two hours, a quarter of a trading day, a diffusion permits $0.140 \times 0.5 = 0.070$, while the measured exponent gives $0.140 \times 0.25^{0.14} = 0.115$. That is ${0.115 / 0.070 = 1.64}$ times as much movement as the diffusion allows. **Rough volatility is not "more volatile volatility". It is volatility that is too active intraday and too self-cancelling over weeks**, which is exactly what negatively correlated increments produce.

Both errors cost money in opposite directions. Take a desk short \$3m of vega per volatility point with the index at 20 volatility. Scaling the one-day number to five days the diffusion way gives an expected adverse move to $20 \times e^{0.313} = 27.35$, or 7.35 points, a \$22,050,000 risk budget. The measured scaling gives $20 \times e^{0.175} = 23.82$, or 3.82 points and \$11,460,000. The desk is carrying nearly twice the capital it needs for its weekly risk. Intraday the same desk is short: two hours of the diffusion's 0.070 implies a move to 21.45, or 1.45 points and \$4,350,000, while 0.115 implies 22.44, or 2.44 points and \$7,320,000. **One wrong exponent overstates the week and understates the afternoon.**

## Fractional Brownian motion, and what it costs

The process with exactly this property is **fractional Brownian motion**, written $W^H_t$. It is the Gaussian process with $W^H_0 = 0$, mean zero, and covariance

$$\mathbb{E}\big[W^H_t W^H_s\big] \;=\; \tfrac{1}{2}\Big(t^{2H} + s^{2H} - |t-s|^{2H}\Big).$$

Set $H = \tfrac{1}{2}$ and that collapses to $\min(t,s)$, the covariance of ordinary Brownian motion. Everything else about fBm follows from this one formula: increments are stationary, $\mathrm{Var}(W^H_{t+\Delta} - W^H_t) = \Delta^{2H}$, and the correlation between non-overlapping increments carries the sign of ${2H-1}$. Negative when $H \lt \tfrac{1}{2}$, which is the anti-persistence that makes the path rough.

Then comes the bill, and this is why the field is technically hard rather than merely new.

![Two-column comparison of a Brownian driver at Hurst one half against a fractional driver below one half, across semimartingale status, whether Ito's lemma applies, quadratic variation, the Markov property, closed-form option prices and calibration speed, with a footer noting the asset price itself remains a semimartingale so the model stays arbitrage free](/imgs/blogs/rough-volatility-fractional-brownian-math-for-quants-5.webp)

For $H \neq \tfrac{1}{2}$, fractional Brownian motion is **not a semimartingale** (Rogers 1997). A semimartingale is a process that decomposes into a local martingale plus a finite-variation part, and it is precisely the class for which the Ito integral is defined. Lose that and you lose the entire toolkit at once: no [Ito's lemma](/blog/trading/math-for-quants/ito-integral-itos-lemma-math-for-quants) for the volatility driver, no quadratic variation to work with (for $H \lt \tfrac{1}{2}$ it is infinite rather than equal to $t$), and no [Feynman-Kac route to a pricing PDE](/blog/trading/math-for-quants/feynman-kac-black-scholes-pde-math-for-quants).

You also lose the Markov property, which hurts more in practice than the missing calculus. In [Heston](/blog/trading/math-for-quants/jump-diffusion-stochastic-volatility-math-for-quants) today's variance is a sufficient statistic: hand the model the current level and it can price everything. Under a fractional driver the covariance formula depends on $t$ and $s$ separately, so the **entire history** of the driver conditions the future. There is no finite state to carry, which is why simulation is expensive and calibration slow.

The one thing you do **not** lose is arbitrage-freeness, and the reason matters. In these models the fractional process drives the *volatility*; the asset price is still $\mathrm{d}S_t = \sqrt{v_t}\,S_t\,\mathrm{d}Z_t$ with $Z$ an ordinary Brownian motion. The price remains a semimartingale, [risk-neutral pricing](/blog/trading/math-for-quants/martingales-risk-neutral-measure-math-for-quants) survives intact, and Rogers's arbitrage result simply does not bite.

## What the empirical result actually says

Gatheral, Jaisson and Rosenbaum applied this estimator to the Oxford-Man Institute realized library and concluded, in the paper's own words, that "log-volatility behaves essentially as a fractional Brownian motion with Hurst exponent $H$ of order 0.1, at any reasonable time scale". The headline single-asset numbers are $H = 0.142$ for the S&P 500 and $H = 0.139$ for the NASDAQ. They named the resulting model **RFSV**, rough fractional stochastic volatility, the "rough" flagging $H \lt \tfrac{1}{2}$ against the earlier fractional stochastic volatility literature, which had assumed $H \gt \tfrac{1}{2}$ to capture long memory.

The robustness is the striking part. Mouti (2026) extends the exercise across a much wider universe and reports class-median estimates of 0.07 to 0.10 for rates, FX, agriculture, energy and metals, 0.13 for single stocks and 0.20 for equity indices. Different asset classes, different microstructures, different decades, and the exponent stays far below 0.5 everywhere. Use these as the published numbers; do not substitute an estimate of your own without doing the work.

## Why the classical models miss the short end

Here is where the measurement pays for itself. Define the **at-the-money skew** as the sensitivity of implied volatility to log-strike at the money,

$$\psi(\tau) \;=\; \left|\frac{\partial \sigma_{\mathrm{BS}}(k,\tau)}{\partial k}\right|_{k=0},$$

where $k = \log(K/F)$. It is the steepness of the [volatility smile](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more) at a given maturity $\tau$, and its behaviour as $\tau \to 0$ is the cleanest test a volatility model faces.

For **any** diffusive stochastic volatility model, Heston included, the short-maturity at-the-money skew converges to a finite constant as $\tau \to 0$ (Alos, Leon and Vives 2007). The reason is structural: over a vanishing horizon a diffusive variance cannot move far enough to bend the smile, so the skew saturates. Market skew does not saturate, it keeps steepening all the way into the front expiry.

Fukasawa (2017) supplied the missing piece. A volatility driven by fractional Brownian motion with exponent $H$ generates

$$\psi(\tau) \;\propto\; \tau^{\,H - 1/2} \qquad \text{as } \tau \to 0 .$$

At $H = 0.5$ the exponent is zero and you recover the diffusion's flat short end, which is a good sign that the formula is right. At $H = 0.14$ the exponent is $-0.36$ and the skew blows up as maturity shrinks. Independently, the SPX at-the-money skew is known to obey a power law in maturity (Fukasawa 2017), with the fitted exponent close to $-0.4$, and Fukasawa (2021) proved the implication runs the other way too: in a viable market, a power-law short-dated skew *requires* rough volatility. The empirical roughness of realised volatility and the empirical shape of the front-month smile are the same fact seen twice.

#### Worked example 2: what the flat short end costs on 2,000 puts

Assume a three-month at-the-money skew of 0.40, calibrate both models there, and ask for one week, which is one thirteenth of three months. The power law multiplies the skew by $13^{0.36} = 2.518$, so

$$\psi(\text{1 week}) \;=\; 0.40 \times 2.518 \;=\; 1.01 .$$

The diffusion, having saturated, still offers about 0.40. Now price a one-week put 5% out of the money on a 6,000 index, so the strike is 5,700 and the log-moneyness is $\log 0.95 = -0.0513$. With at-the-money volatility assumed at 18.00%:

- **Measured skew:** $1.01 \times 0.0513 = 0.0518$, so 5.18 volatility points of lift, giving an implied volatility of 23.18%.
- **Diffusive skew:** $0.40 \times 0.0513 = 0.0205$, so 2.05 points of lift, giving 20.05%.

The gap is 3.13 volatility points on the strike a short-dated put seller trades most. Black-Scholes at those inputs prices the put at \$4.41 a share against \$2.07, a difference of \$2.34. On 2,000 contracts of 100 shares, that is 200,000 shares and a **\$468,000 shortfall on a single week's sale**. The model is not mispricing by a tick, it is selling the skew at less than half its value, because it cannot produce that skew at that maturity at any parameter setting.

![Two-line chart of at-the-money skew against maturity, showing a power-law curve with exponent minus 0.36 rising from 0.24 at one year through 0.40 at three months and 0.59 at one month to 1.01 at one week, against a flat diffusive line saturating at 0.40, with the two calibrated to agree at three months and a bracket marking the gap of 1.01 against 0.40 at one week](/imgs/blogs/rough-volatility-fractional-brownian-math-for-quants-3.webp)

Note where the two lines agree. Calibrate a diffusion at three months and it is fine at three months and fine at a year. The error is concentrated entirely in the front expiries, which is exactly where the volume is.

## The rough Bergomi model, in structure

Bayer, Friz and Gatheral (2016) turned the measurement into a pricing model. Rather than specifying a stochastic differential equation for variance, **rough Bergomi** specifies the forward variance curve directly:

$$v_u \;=\; \xi_0(u)\,\exp\!\left(\eta\,\widetilde{W}^H_u \;-\; \tfrac{1}{2}\eta^2 u^{2H}\right),$$

with the asset driven by $\mathrm{d}S_t = \sqrt{v_t}\,S_t\,\mathrm{d}Z_t$ and correlation $\rho$ between $\widetilde{W}^H$ and $Z$. Read the pieces: $\xi_0(u)$ is today's forward variance curve, taken as a market input rather than fitted; the exponential keeps variance positive; the $-\tfrac{1}{2}\eta^2 u^{2H}$ term is the compensator that makes the expectation match $\xi_0(u)$ exactly, so the model is calibrated to the variance term structure by construction.

That leaves **three parameters**: $H$ for roughness, $\eta$ for volatility of volatility, $\rho$ for the spot-volatility correlation. Their reported SPX fit was $H = 0.05$, $\eta = 2.3$, $\rho = -0.9$, and it fits the whole surface including the shortest-dated smile, with fewer parameters than the conventional models it beats. Note that the fitted $H$ is smaller than the 0.142 estimated from realised volatility. The two are not required to agree, which is a loose end the literature has not fully closed.

## The honest counter-argument

The case above is a strong one, but it is contested, and a candidate who presents it as settled fact is easy to catch.

The objection is about **measurement**. You never observe volatility. You estimate it from prices that carry microstructure noise: bid-ask bounce, discrete ticks, asynchronous trades. Cont and Das (2022, published 2024) show with a model-free roughness estimator that this estimation error alone produces apparent roughness. Their result is blunt: irrespective of the roughness of the true spot volatility process, realised volatility computed from discrete noisy prices always exhibits an apparent Hurst index below 0.5. In their simulations a spot volatility with $H = 0.5$ yields realised volatility measuring about 0.13, and on real S&P 500 data their estimates move over roughly 0.05 to 0.25 depending on the estimator's tuning. Their conclusion is that the roughness lives in the estimation error, not in the volatility.

Rogers (2019) presses a related point from a different direction. Roughness estimators built on regressing $p$-th variation against scale behave similarly over the relevant range of time scales even when the data comes from a plain Brownian Ornstein-Uhlenbeck volatility, which is not rough at all. He offers a two-factor Ornstein-Uhlenbeck alternative that reproduces the observations on the time scales option pricing cares about, without leaving the semimartingale world.

Neither author claims the short-dated smile is flat: that is an observed fact about quoted prices. What they dispute is the inference from *realised volatility statistics* to a fractional volatility process. Mouti (2026) argues in the other direction, that noise biases $H$ estimates downward and that correcting for it raises them without erasing the effect. **The correct summary is that the smile evidence is strong, the time-series evidence is contested, and the field is live.**

## What it costs a book that gets it wrong

#### Worked example 3: one quarter of selling weekly puts off a Heston calibration

Take the desk from example 2 and let it run: 2,000 of those one-week 5%-out-of-the-money puts sold every week for a quarter, 13 expiries, priced off a Heston model calibrated to the three-month surface. All inputs are assumed and the arithmetic is illustrative.

At the Heston implied volatility of 20.05% the desk collects $\$2.07 \times 200{,}000 = \$414{,}000$ a week, or \$5,382,000 over the quarter. Priced off the measured skew at 23.18% it would have collected \$882,000 a week, or \$11,466,000.

Twelve of the thirteen weeks are quiet and the puts expire worthless. In the thirteenth the index falls 6%, from 6,000 to 5,640. The strike is 5,700, so the puts finish 60 points in the money: $\$60 \times 200{,}000 = \$12{,}000{,}000$ paid out.

- **Heston-priced book:** \$5,382,000 collected minus \$12,000,000 paid equals **negative \$6,618,000**.
- **Skew-aware book:** \$11,466,000 collected minus \$12,000,000 paid equals **negative \$534,000**.

![Two-column money table comparing a quarter of selling weekly puts priced off a Heston fit against the measured skew, with implied volatility of 20.05 against 23.18 percent, put price of 2.07 against 4.41 dollars a share, weekly premium of 414,000 against 882,000 dollars, thirteen-week premium of 5,382,000 against 11,466,000 dollars, an identical crash-week payoff of 12,000,000 dollars, and quarter results of negative 6,618,000 against negative 534,000 dollars](/imgs/blogs/rough-volatility-fractional-brownian-math-for-quants-4.webp)

The same crash, the same position, the same hedging. The entire \$6,084,000 difference is 13 weeks of \$468,000 in skew the model could not see, and the lesson generalises past this arithmetic: **an understated short-end skew does not show up as a bad day, it shows up as a systematically underfunded premium account that is discovered only when the move arrives.** The book looked profitable for twelve weeks precisely because it was accumulating the error.

## Common misconceptions

**"Rough volatility means volatility is more volatile."** It means volatility is less *regular*, which is a different axis. The overall level of volatility of volatility is set by $\eta$, not by $H$. A rough process with small $\eta$ is calm and jagged; a smooth process with large $\eta$ is wild and fluid. As worked example 1 showed, roughness makes multi-week moves *smaller* than a diffusion demands, which is the opposite of "more volatile".

**"$H \lt 0.5$ means prices are predictable."** This is the most expensive confusion in the topic. The Hurst exponent here is measured on the **volatility** process, not on returns. Anti-persistence in volatility says a volatility spike tends to be partly reversed; it says nothing about the direction of the underlying. Returns in these models are driven by an ordinary Brownian motion and remain a martingale under the pricing measure. If you find $H \lt 0.5$ on *prices* you have found something else entirely, and it is usually microstructure.

**"This is settled, and the old models are wrong."** Neither half holds. Cont and Das (2024) and Rogers (2019) give serious reasons the time-series estimate may be an artefact of measuring volatility with noise. And Heston is not "wrong", it is a model whose short-maturity skew saturates, which is a known and quantified limitation you can price around. Its three-month and one-year surfaces remain perfectly serviceable.

**"You can just add jumps instead."** You can, and jumps do steepen the short-dated smile. But a jump component bolted on to fit the front month is a separate mechanism with its own parameters that then has to be prevented from distorting the long end. The rough driver produces the observed power law from a single exponent that is also measurable directly in the time series, which is a meaningfully stronger claim.

**"A smaller $H$ is just a faster mean reversion."** Fast mean reversion in a diffusion also flattens long-horizon scaling, so the two look similar over a limited range of lags. That similarity is precisely Rogers's argument, and it is why a single-scale estimate proves nothing. What distinguishes roughness is that the same exponent holds across scales from hours to months, which is what the three agreeing estimates in worked example 1 illustrate.

## Summary

- The **Hurst exponent** is a measure of path regularity read off the scaling of increments: moves grow like $\Delta^H$, and $H = 0.5$ is Brownian by construction, not by choice.
- Realised log-volatility measures far below 0.5. Gatheral, Jaisson and Rosenbaum (2018) report $H = 0.142$ for the S&P 500 and $H$ "of order 0.1" broadly, with the pattern repeating across asset classes.
- Rough volatility moves **less** than a diffusion demands over weeks and **more** over hours. Both errors cost money, in opposite directions, on the same book.
- The payoff is the short end: a fractional driver gives $\psi(\tau) \propto \tau^{H-1/2}$, matching the observed power law, where every diffusion's skew saturates.
- The cost is the loss of the semimartingale property for the driver, and with it Ito calculus, the Markov property, closed forms and fast calibration.
- **The single most important takeaway:** the argument for rough volatility rests on two independent legs, the time-series estimate and the shape of the short-dated smile, and only the second is uncontested. Argue from the smile and you are on solid ground; argue from the realised-volatility estimate alone and a good interviewer will ask you about microstructure noise.

## Sources and further reading

- J. Gatheral, T. Jaisson and M. Rosenbaum, "Volatility is rough", *Quantitative Finance* 18(6), 933-949, 2018 (arXiv:1410.3394, 2014). The founding empirical paper: $H$ of order 0.1, $H = 0.142$ for the S&P 500, the RFSV model.
- C. Bayer, P. K. Friz and J. Gatheral, "Pricing under rough volatility", *Quantitative Finance* 16(6), 887-904, 2016. The rough Bergomi model and the three-parameter SPX fit.
- M. Fukasawa, "Short-time at-the-money skew and rough fractional volatility", *Quantitative Finance* 17(2), 189-198, 2017 (arXiv:1501.06980), and "Volatility has to be rough", *Quantitative Finance* 21(1), 1-8, 2021. The $\tau^{H-1/2}$ skew law and its converse.
- E. Alos, J. A. Leon and J. Vives, "On the short-time behavior of the implied volatility for jump-diffusion models with stochastic volatility", *Finance and Stochastics* 11, 571-589, 2007. Why a diffusive skew saturates.
- L. C. G. Rogers, "Arbitrage with fractional Brownian motion", *Mathematical Finance* 7(1), 95-105, 1997. fBm is not a semimartingale for $H \neq \tfrac{1}{2}$.
- R. Cont and P. Das, "Rough volatility: fact or artefact?", *Sankhya B* 86(1), 191-223, 2024 (arXiv:2203.13820). The microstructure-artefact critique.
- L. C. G. Rogers, "Things we think we know", 2019. The estimator critique and the Ornstein-Uhlenbeck alternative.
- S. Mouti, "Rough volatility across assets", arXiv:2608.16749, 2026. Cross-asset estimates and a noise correction.

Every dollar figure in the worked examples is illustrative arithmetic on assumed inputs, not a quote from any market or a calibration result.

## In the interview room and on the desk

The question rarely arrives with "rough volatility" in it. It arrives as **"why does the short-dated smile look the way it does?"**, or as "your Heston model fits three months and misses one week, what is going on?". Both are the same question, and the interviewer is testing whether you reason from data or recite a model.

The strong answer runs in this order. First name the structural fact: for any diffusive stochastic volatility model the at-the-money skew converges to a finite constant as maturity goes to zero, because a diffusive variance simply cannot travel far enough over a vanishing horizon to bend the smile. Second, state what the market does instead: the observed at-the-money skew follows a power law in maturity, close to $\tau^{-0.4}$ for SPX, and keeps steepening into the front expiry. Third, connect them: Fukasawa showed that a volatility driven by fractional Brownian motion with exponent $H$ produces $\psi(\tau) \propto \tau^{H-1/2}$, so an exponent near $-0.4$ implies $H$ near 0.1, and that is roughly what Gatheral, Jaisson and Rosenbaum measure on realised volatility directly. Fourth, price it: an understated short-end skew is not a rounding error, it is selling the front-month wing at half its value, week after week, until the move arrives.

The trap is finishing there. Presenting rough volatility as established fact is the single most common way a candidate looks rigorous and is wrong, because the strongest objection is one line long and a volatility interviewer will be holding it: **you never observe volatility, you estimate it from noisy prices, and Cont and Das showed that the estimation error alone makes realised volatility look rough even when the true process is not.** Say it before they do. Note that the smile evidence and the time-series evidence are independent, that the smile evidence is the sturdier of the two, and that Rogers has a non-fractional model that fits the relevant time scales. Then you have demonstrated the thing the question is actually testing, which is whether you can distinguish a measurement from the quantity it measures.

Jane Street, Citadel Securities, Optiver and IMC weight this most, along with any index-volatility or dispersion seat. On a market-making desk the practical form is narrower and sharper: if your short-dated model skew is flatter than the market's, you will be the one filled on every front-month put the street wants to sell, and you will find out why in a single week.
