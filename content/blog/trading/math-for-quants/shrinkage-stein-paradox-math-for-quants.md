---
title: "Shrinkage and Stein's paradox: why the sample mean is the wrong estimate"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "In three or more dimensions the sample mean is provably beaten, everywhere, by an estimator that deliberately biases every number toward a common centre. Build Stein's paradox from zero, state the James-Stein estimator exactly, and watch shrinkage move fifty million dollars of position on a hundred million dollar book."
tags:
  [
    "shrinkage",
    "james-stein",
    "steins-paradox",
    "admissibility",
    "bias-variance",
    "ledoit-wolf",
    "expected-returns",
    "portfolio-optimization",
    "empirical-bayes",
    "estimation-error",
    "quantitative-finance",
    "math-for-quants"
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 19
---

> [!important]
> **TL;DR:** The sample mean is unbiased, maximum likelihood and minimum-variance among unbiased estimators, and in three or more dimensions it is still the wrong answer.
>
> - "Wrong" means **dominated**: another estimator has lower expected squared error at *every* possible true value, not on average and not if you are lucky.
> - The **James-Stein estimator** shrinks by ${(p-2)\sigma^2/\lVert X\rVert^2}$. At the target its total risk is $2\sigma^2$ however large $p$ is, so for 50 assets that is a **25x** cut.
> - The mechanism is geometric: ${\mathbb{E}\lVert X\rVert^2 = \lVert\theta\rVert^2 + p\sigma^2}$, so the sample mean vector is systematically **too long**.
> - The **target is a choice**. On five factor premia, shrinking to their cross-sectional mean cuts the error 3.7x; shrinking to zero makes it 2.5x worse, \$241k against \$732k a year per \$20m sleeve.
> - Chopra and Ziemba (1993) found errors in means cost roughly **11 times** what errors in variances cost, which is why means are where shrinkage pays.
> - The number to remember: shrinking the means moves **\$51.2m of position on a \$100m book** and turns one short into a long.

You have five years of monthly returns on five assets. Asset A averaged 2.0% a month, so you write down 24% a year and move on.

Now compute the standard error. A sample mean's is ${\sigma/\sqrt{T}}$, and asset A's monthly volatility is 9%, so on 60 observations it is ${9/\sqrt{60} = 1.16\%}$ a month, **13.9 percentage points** annualised. The honest version of "24% a year" is "somewhere between roughly -4% and +52%". The optimiser does not read the honest version. It reads 24.0, and sizes accordingly.

![Shrinking five sample means toward their common centre cuts the spread from 26.4 percentage points to 3.9, and reorders the top two](/imgs/blogs/shrinkage-stein-paradox-math-for-quants-1.webp)

That figure is the whole post. Pull each raw sample mean toward a common centre by an amount set by how badly it is measured, then hand the compressed set to the optimiser. That operation is **shrinkage**, and the startling part is not that it works in practice. It is that in three or more dimensions it is a *theorem*: the estimator you were taught to trust is beaten everywhere by one that is biased everywhere. None of this is investment advice; it describes how an estimator behaves and where it breaks.

## Foundations: the three reasons the sample mean feels unimprovable

You observe $p$ quantities at once and want their true means ${\theta = (\theta_1,\dots,\theta_p)}$. Write $X$ for the vector of sample means, each normal around its truth with the same known variance, so ${X \sim N(\theta, \sigma^2 I_p)}$. In finance $\sigma^2$ is the *sampling* variance of a mean, ${\sigma_r^2/T}$, not the variance of returns. Three arguments say to use $X$, and each is true.

**It is unbiased.** ${\mathbb{E}[X_i] = \theta_i}$ for every $i$: no systematic tilt in any direction.

**It is maximum likelihood.** The maximiser of the normal likelihood is exactly $X$, with the guarantees in [maximum likelihood and the method of moments](/blog/trading/math-for-quants/mle-method-of-moments-math-for-quants).

**It is minimum-variance among unbiased estimators.** None has smaller variance, in any dimension, and it attains the Cramér-Rao bound exactly. This is the one that does the damage, because it sounds like a proof of optimality and is not.

None of them says no estimator has smaller *error*. The third optimises within the unbiased class, a constraint you imposed rather than a goal anyone gave you, and unbiasedness is neither necessary nor sufficient for landing close to the truth. See [bias, variance and consistency](/blog/trading/math-for-quants/estimators-bias-variance-consistency-math-for-quants).

## What "dominated" means, and why admissibility is the right frame

Score an estimate with a **loss function**, here total squared error ${L(\theta,\delta) = \lVert \delta - \theta\rVert^2}$. You never observe it, so average over the data to get the **risk** ${R(\theta,\delta) = \mathbb{E}_\theta L(\theta,\delta)}$. For the sample mean each coordinate contributes its own variance:

$$R(\theta, X) \;=\; p\,\sigma^2 \qquad \text{for every } \theta .$$

A flat line: the risk does not depend on where the truth is, the mark of an estimator that treats all of parameter space alike.

Estimator $\delta_1$ **dominates** $\delta_2$ if ${R(\theta,\delta_1) \le R(\theta,\delta_2)}$ for every $\theta$, strictly somewhere, and an estimator is **admissible** if nothing dominates it. That is a low bar: not that an estimator is good, only that no single alternative beats it everywhere at once. Stein (1956) proved that for ${p \ge 3}$ the sample mean of a multivariate normal fails even that, and is **inadmissible** under squared error loss. For ${p = 1}$ and ${p = 2}$ it is admissible and everything you were taught holds. At ${p = 3}$ it stops, and no continuity argument warns you.

![The James-Stein risk curve sits strictly under the sample mean's flat risk line at every true mean, which is what it means to say the sample mean is dominated](/imgs/blogs/shrinkage-stein-paradox-math-for-quants-2.webp)

The flat line is the sample mean; the curve beneath it is James-Stein, lowest when the truth sits near the target, climbing toward the line as the truth moves away, never reaching it. No region of parameter space favours the sample mean, which is why admissibility rather than unbiasedness is the working frame.

## The James-Stein estimator, stated exactly

James and Stein (1961) wrote down the estimator that does it. For ${X \sim N(\theta, \sigma^2 I_p)}$ with ${p \ge 3}$,

$$\hat{\theta}^{\,\mathrm{JS}} \;=\; \left( 1 \;-\; \frac{(p-2)\,\sigma^2}{\lVert X \rVert^2} \right) X .$$

The numerator grows with dimension, so the more quantities you estimate jointly the harder you pull; the denominator is the squared length of the observed vector, so when the estimates are large relative to their noise you pull less. The risk is exactly

$$R(\theta, \hat\theta^{\,\mathrm{JS}}) \;=\; p\sigma^2 \;-\; (p-2)^2 \sigma^4 \,\mathbb{E}\!\left[ \frac{1}{\lVert X \rVert^2} \right] ,$$

and the subtracted term is strictly positive for every $\theta$, which *is* the domination result. At ${\theta = 0}$, ${\lVert X\rVert^2/\sigma^2}$ is chi-square with $p$ degrees of freedom and ${\mathbb{E}[1/\chi^2_p] = 1/(p-2)}$, so the risk collapses to $2\sigma^2$. Not per coordinate: $2\sigma^2$ in total, whatever $p$ is. On 50 means the sample mean carries $50\sigma^2$ and James-Stein carries $2\sigma^2$, a **25x** cut.

Two amendments. Shrinking toward the **grand mean** $\bar{X}$ rather than zero is usually what you want, and costs a degree of freedom:

$$\hat{\theta}^{\,\mathrm{JS}}_i \;=\; \bar{X} \;+\; \left( 1 - \frac{(p-3)\,\sigma^2}{\sum_j (X_j - \bar{X})^2} \right) (X_i - \bar{X}) ,$$

which needs ${p \ge 4}$, because you spent an observation locating the centre. And the factor goes negative when $\lVert X\rVert^2$ is small, flipping every sign; the **positive-part** version replaces it with ${\max(0,\cdot)}$ and dominates the original, so use that one.

### The geometry: your estimate is too long

Take the expected squared length of the observed vector:

$$\mathbb{E}\lVert X \rVert^2 \;=\; \lVert \theta \rVert^2 \;+\; p\,\sigma^2 .$$

The sample mean vector is **systematically too long**, by exactly $p\sigma^2$. Noise has no preferred direction, so it cannot push the vector reliably one way or another, but it always pushes it *outward*: every independent coordinate adds its own positive variance to the length. So one thing is knowable before you see any data, which is that your estimate overstates its own magnitude, and shortening it is the only correction available without knowing which coordinate is wrong.

In one dimension that is useless, because a single excess $\sigma^2$ is swamped by not knowing the sign of the error. With three or more coordinates the excess becomes a stable aggregate signal: **length is estimable even when direction is not.**

#### Worked example 1: shrinking five means, and what it does to a \$10m position

Five assets, 60 months of history. Sample means and volatilities, monthly:

| Asset | Sample mean | Volatility | Standard error | Annualised mean | Annualised s.e. |
| --- | --- | --- | --- | --- | --- |
| A | 2.00% | 9.0% | 1.16% | 24.0% | 13.9 pts |
| B | 1.40% | 4.0% | 0.52% | 16.8% | 6.2 pts |
| C | 0.90% | 6.0% | 0.77% | 10.8% | 9.3 pts |
| D | 0.40% | 3.0% | 0.39% | 4.8% | 4.6 pts |
| E | -0.20% | 7.0% | 0.90% | -2.4% | 10.8 pts |

**Step 1, the centre.** The equal-weighted mean of the five is 0.90% a month, 10.8% a year.

**Step 2, the classic factor.** Deviations from the centre are +1.10, +0.50, 0.00, -0.50 and -1.10, so ${\sum_j (X_j - \bar{X})^2 = 2.92}$ in percent-squared, and the sampling variances average ${\bar{\sigma}^2 = 191/300 = 0.637}$:

$$\frac{(p-3)\,\bar\sigma^2}{\sum_j (X_j - \bar X)^2} \;=\; \frac{2 \times 0.637}{2.92} \;=\; 0.436 ,$$

so you discard 43.6% of every deviation: A's 2.00% becomes 1.52%, E's -0.20% becomes 0.28%.

**Step 3, admit the estimates are not equally precise.** Standard errors range from 0.39% to 1.16%, so one common factor is wasteful. The empirical-Bayes form gives each asset its own weight ${w_i = \sigma_i^2/(\sigma_i^2 + \tau^2)}$, with $\tau^2$ the cross-sectional variance of the *true* means, estimated as observed dispersion minus average noise: ${2.92/4 - 0.637 = 0.093}$. The true means therefore spread about ${\sqrt{0.093} = 0.30\%}$ a month, 3.7 points a year, against 26.4 points in the raw estimates. Almost everything you are looking at is noise. The weights come out at 0.94, 0.74, 0.87, 0.62 and 0.90:

| Asset | Raw | Shrunk | Rank change |
| --- | --- | --- | --- |
| B | 16.8% | **12.4%** | 2nd to **1st** |
| A | 24.0% | **11.6%** | 1st to **2nd** |
| C | 10.8% | 10.8% | 3rd |
| E | -2.4% | **9.5%** | 5th to **4th** |
| D | 4.8% | **8.5%** | 4th to **5th** |

Both ends invert. A had the biggest number and the biggest standard error, so most of its lead was noise, while B's smaller number was measured four times as precisely and survives. At the bottom, D's mediocre result is real and E's terrible one is not. In money: a \$10m position in A carried an expected profit of \$2.40m a year on the raw estimate and \$1.16m on the shrunk one, and the second number is the defensible one.

*The lesson: shrinkage does not rank by size, it ranks by size per unit of measurement error.*

#### Worked example 2: the same five assets on a \$100m book

Estimates matter only through the positions they produce. Assume the assets are uncorrelated, so ${\Sigma^{-1}}$ is diagonal and the weights are proportional to ${\mu_i / \sigma_i^2}$, normalised to sum to one. [The efficient frontier](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants) derives that machinery; real correlations only make the effect larger.

On raw means, ${2.00/81 = 0.0247}$, ${1.40/16 = 0.0875}$, ${0.90/36 = 0.0250}$, ${0.40/9 = 0.0444}$ and ${-0.20/49 = -0.0041}$ sum to 0.1776. Divide through, repeat with the shrunk means, and scale to \$100m:

| Asset | Raw position | Shrunk position | Change |
| --- | --- | --- | --- |
| A | \$13.9m | \$6.1m | -\$7.8m |
| B | \$49.3m | \$32.8m | -\$16.5m |
| C | \$14.1m | \$12.7m | -\$1.3m |
| D | \$25.0m | \$40.1m | +\$15.1m |
| E | -\$2.3m | \$8.2m | +\$10.5m |

![Same five assets, same covariances, same 100 million dollars: replacing raw sample means with shrunk ones moves 51.2 million dollars of position and reverses one sign](/imgs/blogs/shrinkage-stein-paradox-math-for-quants-3.webp)

The absolute changes add to **\$51.2m**, so half the book moves. Asset E is the one to stare at: a \$2.3m short becomes an \$8.2m long, because its -2.4% was never distinguishable from the rest given a 10.8-point standard error. Gross exposure falls from \$104.6m to \$100.0m, the leveraged tilt having paid for a difference that was not there.

*The lesson: a change in the third decimal place of an expected return is a change in the seventh figure of a position.*

## The trade you are actually making

Risk splits into squared bias plus variance. The sample mean sets the bias to zero and accepts whatever variance follows, and at ${T = 60}$ that variance is enormous. Take a fixed intensity $w$ toward a fixed target $t$, so ${\hat\theta_i = (1-w)X_i + w\,t_i}$:

$$R \;=\; \underbrace{(1-w)^2 \sum_i \sigma_i^2}_{\text{variance}} \;+\; \underbrace{w^2 \sum_i (\theta_i - t_i)^2}_{\text{squared bias}} .$$

The variance term starts large and falls at a finite rate from ${w = 0}$ while the bias term leaves zero with zero slope, so the first slice of shrinkage is close to free. Minimising gives ${w^\ast = \sum_i \sigma_i^2 \,/\, (\sum_i \sigma_i^2 + \sum_i (\theta_i - t_i)^2)}$: **shrink in proportion to how much of your dispersion is noise.** James-Stein estimates that ratio from the same data.

## Shrinking toward what: the target is not a detail

The formula never said where to shrink *to*, and that choice does more work than the intensity.

- **Zero.** Right when the quantities have no common level, as with long-short factor returns or residual alphas. Wrong for asset returns, which share an equity risk premium nobody thinks is zero.
- **The grand mean.** The workhorse: "these assets are exchangeable until proved otherwise," letting the data pick the common level.
- **A factor model.** Each asset gets *its own* target, its CAPM or multi-factor expected return. This preserves genuine cross-sectional differences while killing noise, and is what a research desk usually means by shrinkage. Black-Litterman is the same idea with market-implied equilibrium returns.

A wrong target does not merely waste the benefit. It imports its own error at full strength, because ${w^2\sum_i(\theta_i - t_i)^2}$ cannot notice that $t$ is wrong.

#### Worked example 3: the wrong target, and its \$491k a year

Five long-short factor premia, 20 years of monthly data, factor volatility 3% a month. Each premium's sampling variance is ${9/240 = 0.0375}$ in percent-squared, so the raw total risk is 0.1875. Suppose the true premia are 0.40%, 0.55%, 0.35%, 0.30% and 0.20% a month, averaging 0.36%. Fix the intensity at ${w = 0.8}$ and change only the target.

**Target = the cross-sectional mean, 0.36%.** Squared distances sum to 0.0670:

$$R \;=\; 0.04 \times 0.1875 \;+\; 0.64 \times 0.0670 \;=\; 0.0504 ,$$

**3.7x better** than the raw 0.1875.

**Target = zero.** Squared distances now sum to 0.7150, because every premium is genuinely positive:

$$R \;=\; 0.04 \times 0.1875 \;+\; 0.64 \times 0.7150 \;=\; 0.4651 ,$$

**2.5x worse** than raw, and 9.2x worse than the right target.

![With the true premia clustered near 0.36 percent a month, shrinking toward their cross-sectional mean cuts the error 3.7x while shrinking toward zero makes it 2.5x worse](/imgs/blogs/shrinkage-stein-paradox-math-for-quants-4.webp)

In money: the root-mean-square error per premium is ${\sqrt{0.1875/5} = 0.194\%}$ a month raw, 0.100% shrunk to the mean and 0.305% shrunk to zero, annualising to 2.3%, 1.2% and 3.7%. On a \$100m book split into five \$20m sleeves, that is **\$464k, \$241k and \$732k** a year of misstated expected return per sleeve. The wrong centre costs **\$491k a year per sleeve**, on the same estimator, intensity and data.

It bites here and not in example 1 because with 20 years the estimates are good enough that bias dominates. Shrinkage's protection comes from your data being bad.

*The lesson: shrinkage intensity is a statistics question and shrinkage target is an economics question, and only one of them can be estimated from the returns.*

## Ledoit-Wolf: the same trade applied to a matrix

The covariance matrix has the same disease and the same cure. Ledoit and Wolf (2003, 2004) form

$$\hat{\Sigma} \;=\; \delta F \;+\; (1-\delta)\, S ,$$

where $S$ is the sample covariance matrix and $F$ a structured target: a scaled identity, a constant-correlation matrix, or a single-index model. $S$ is unbiased and desperately noisy; $F$ is biased and stable.

![Ledoit-Wolf blends the noisy unbiased sample covariance with a biased but stable structured target, at an intensity computed from the data rather than cross-validated](/imgs/blogs/shrinkage-stein-paradox-math-for-quants-5.webp)

What made it standard is that $\delta$ is **solved for**, not tuned. Minimising the expected Frobenius distance to the true covariance gives ${\delta^\ast = \kappa/T}$, clipped to ${[0,1]}$, where $\kappa$ trades the noise in $S$ against how wrong $F$ is and is itself estimable from the same data. No held-out set, no cross-validation loop, and the ${1/T}$ means shrinkage recedes as history accumulates.

Why the sample covariance needs help is [post 1](/blog/trading/math-for-quants/random-matrix-theory-covariance-cleaning-math-for-quants): most of its eigenvalue spectrum is statistically indistinguishable from noise. One difference is worth carrying: **linear shrinkage moves every eigenvalue toward their common mean, while eigenvalue clipping moves only the ones random matrix theory gives grounds to distrust.** Both are compared in [robust and regularised portfolios](/blog/trading/math-for-quants/robust-regularized-portfolios-math-for-quants).

## Why expected returns are the input that most needs this

Shrinkage helps the covariance matrix. It *rescues* the mean vector.

A mean's standard error over a calendar span of $Y$ years is ${\sigma_{\text{ann}}/\sqrt{Y}}$ whether you sample daily, monthly or annually, because per-observation volatility and observation count scale together and cancel. Merton (1980) made the point precisely: the expected return cannot be estimated more accurately by observing more frequently, whereas the variance can, since realised volatility improves with high-frequency data.

The arithmetic is brutal. At 20% annual volatility, five years gives a standard error of ${20/\sqrt{5} = 8.9}$ percentage points, and getting to one point takes **400 years**. Equivalently the $t$-statistic on a strategy's mean is ${\text{Sharpe} \times \sqrt{Y}}$, so a genuine Sharpe of 0.5 needs 16 years of live returns to clear ${t = 2}$. That sits behind [concentration inequalities and sample complexity](/blog/trading/math-for-quants/concentration-inequalities-sample-complexity-math-for-quants).

Meanwhile the optimiser cares about means more than anything. Chopra and Ziemba (1993), on ten Dow stocks at a risk tolerance of 50, found errors in the means cost roughly **11 times** as much certainty-equivalent wealth as errors in the variances, and errors in variances roughly twice as much as in covariances. The input you can least estimate is the one the machine is most sensitive to, which is why Michaud (1989) called an optimiser fed raw sample moments an "estimation-error maximizer".

## Common misconceptions

**"Biased estimators are always worse."** Squared error is squared bias plus variance, and only the sum shows up in performance. Unbiasedness forces the first term to zero whatever that costs the second, which is odd to insist on when the second is ten times the first. James-Stein is biased at every $\theta$ and has lower total risk at every $\theta$. Note what is guaranteed, though: the *total* across the $p$ coordinates, not each one.

**"Shrinkage is just regularisation with a nicer name."** They are relatives, and the arguments differ. Stein's result is from 1956 and James-Stein from 1961; ridge regression arrived with Hoerl and Kennard in 1970. Ridge is justified by ill-conditioning or a Gaussian prior on the coefficients, and its penalty is picked by cross-validation. James-Stein assumes no prior, holds out no data, delivers its intensity in closed form, and rests on a dominance theorem true for every $\theta$. See [regularised regression](/blog/trading/math-for-quants/regression-ols-gls-regularized-math-for-quants); "it shrinks" describes the operation, not the justification.

**"Shrink everything toward zero."** Worked example 3 is the counterexample with a number on it. Zero is legitimate only when it is a plausible common level: true of residual alphas and long-short spreads, false of asset returns.

## How it shows up in real markets

**Where it was first believed.** Efron and Morris took the batting averages of 18 major-league players over their first 45 at-bats of the 1970 season, shrank them toward the grand average of 0.265, and predicted the rest of the season. Total squared prediction error came in at less than a third of the raw averages', and the shrunk estimate was closer for 16 of the 18. The misses were the genuine outliers, Roberto Clemente at an observed 0.400 among them, exactly as the theory predicts: shrinkage under-calls the extremes and wins on the sum.

**Where it entered finance.** Jorion (1986) shrank sample means toward the global minimum-variance portfolio's expected return and beat raw historical means out of sample. Ledoit and Wolf (2003) ran the covariance version on NYSE and AMEX stocks from 1972 to 1995, producing portfolios with significantly lower out-of-sample variance than the sample covariance matrix or multi-factor alternatives.

**The limiting case people actually use.** The equal-weighted portfolio is shrinkage at ${w = 1}$, an infinitely strong prior that every asset has the same expected return, and DeMiguel, Garlappi and Uppal (2009) found naive ${1/N}$ beat sample-based mean-variance optimisation out of sample. That measures how bad raw sample means are, and is why partial shrinkage is the interesting region.

## In the interview room and on the desk

The question usually arrives as: *"You have five years of returns for 500 stocks. How do you estimate expected returns?"* It sounds like a request for a method. It is a test of whether you know how weak the input is. A strong answer runs in this order.

1. **Give the standard error before the estimate.** On 60 monthly observations at 20% annual volatility, each stock's mean carries a standard error of roughly 9 percentage points a year. You are ranking 500 numbers whose error bars are wider than any plausible spread between them.
2. **Say that frequency does not help.** Daily data multiplies observations by 21 and does nothing for the mean, because the standard error over a fixed calendar span is invariant to sampling frequency. It does sharpen the covariance matrix, and naming that asymmetry separates a candidate who has thought about this from one who has read about it.
3. **Reach for shrinkage, and name the target.** Pull each estimate toward a factor model or the cross-sectional mean, with intensity set by the ratio of sampling noise to genuine dispersion. Cite James-Stein for the theory and Ledoit-Wolf for the matrix, but lead with the target, because that is where the judgement is.
4. **Say what the optimiser does with it.** Chopra and Ziemba's 11-to-1 sensitivity of means over variances, and ${\Sigma^{-1}}$ amplifying the directions you measured worst.
5. **Offer the honest fallback.** Below some level of confidence, stop estimating means per stock: equal-weight within buckets, or forecast only factor premia. Knowing when not to use your own numbers is a senior answer.

The trap is optimising on raw sample means and presenting the resulting frontier as though it meant something. It will look beautiful, because the optimiser took every noise-inflated estimate at face value and built the portfolio that would have been superb in the sample you fitted it to. Produce that chart without a word about estimation error and you have demonstrated the exact failure the question was designed to find. The second trap is reciting "James-Stein" with no dimension condition and no target.

**Two Sigma** and **Citadel** weight this most heavily, in research and portfolio-construction interviews alike, and it comes up in any seat where someone owns the expected-return vector that goes into an optimiser.

## Sources and further reading

- C. Stein, "Inadmissibility of the usual estimator for the mean of a multivariate normal distribution," *Proc. Third Berkeley Symp.* 1, 197–206 (1956).
- W. James and C. Stein, "Estimation with quadratic loss," *Proc. Fourth Berkeley Symp.* 1, 361–379 (1961).
- B. Efron and C. Morris, ["Stein's Paradox in Statistics,"](https://efron.ckirby.su.domains/other/Article1977.pdf) *Scientific American* 236(5), 119–127 (1977), and ["Data Analysis Using Stein's Estimator,"](https://doi.org/10.1080/01621459.1975.10479864) *JASA* 70(350), 311–319 (1975). The baseball figures.
- O. Ledoit and M. Wolf, ["Improved estimation of the covariance matrix of stock returns,"](https://doi.org/10.1016/S0927-5398(03)00007-0) *J. Empirical Finance* 10(5), 603–621 (2003), and ["A well-conditioned estimator for large-dimensional covariance matrices,"](https://doi.org/10.1016/S0047-259X(03)00096-4) *JMVA* 88(2), 365–411 (2004).
- P. Jorion, ["Bayes-Stein Estimation for Portfolio Analysis,"](https://doi.org/10.2307/2331042) *JFQA* 21(3), 279–292 (1986).
- V. K. Chopra and W. T. Ziemba, ["The Effect of Errors in Means, Variances, and Covariances on Optimal Portfolio Choice,"](https://jpm.pm-research.com/content/19/2/6) *JPM* 19(2), 6–11 (1993). The 11-to-1 figure.
- R. C. Merton, "On estimating the expected return on the market," *JFE* 8(4), 323–361 (1980).
- R. Michaud, "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?," *FAJ* 45(1), 31–42 (1989).
- V. DeMiguel, L. Garlappi and R. Uppal, ["Optimal Versus Naive Diversification,"](https://doi.org/10.1093/rfs/hhm075) *RFS* 22(5), 1915–1953 (2009).

Every asset-level number here sits inside a labelled hypothetical worked example: illustrative arithmetic on assumed inputs, not measurements of any real portfolio.
