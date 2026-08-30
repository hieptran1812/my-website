---
title: "Random matrix theory and the noise in your covariance matrix"
date: "2026-08-30"
publishDate: "2026-08-30"
description: "When you estimate an N by N covariance matrix from T observations and N is comparable to T, most of what you measure is noise, and the noise has a known shape. Build the Marchenko-Pastur law from zero, read an eigenvalue spectrum the way a senior researcher does, and see what eigenvalue clipping does to a portfolio's weights."
tags:
  [
    "random-matrix-theory",
    "marchenko-pastur",
    "covariance-matrix",
    "eigenvalues",
    "covariance-cleaning",
    "eigenvalue-clipping",
    "portfolio-optimization",
    "estimation-error",
    "shrinkage",
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
> **TL;DR:** A sample covariance matrix estimated from a realistic amount of data is mostly measurement noise, and random matrix theory tells you exactly which part.
>
> - What decides how bad it is has a name: the **aspect ratio** $q = N/T$, assets over observations. Near zero your estimate is clean; near 1 it is almost entirely noise.
> - If the assets were genuinely uncorrelated every true eigenvalue would be 1, but the *measured* ones spread over the **Marchenko-Pastur bulk**, ${(1-\sqrt{q})^2}$ to ${(1+\sqrt{q})^2}$. For 100 stocks over 250 days that is 0.14 to 2.66, and nothing inside it is evidence of anything.
> - Eigenvalues **outside** the bulk are the real structure: the market factor, then a few sector factors. In the worked example the market sits at ${\lambda_1 = 26}$, ten times past the upper edge.
> - Optimisers make it worse, because weights depend on $\Sigma^{-1}$, which weights each direction by ${1/\lambda}$. The *smallest* and noisiest eigenvalues get the *largest* positions.
> - **Eigenvalue clipping** replaces the bulk with its own average, cutting leverage on the noisiest direction by 7.7x while leaving the market factor alone.
> - The number to remember: Laloux, Cizeau, Bouchaud and Potters (1999) found **94% of the eigenvalues** of an S&P 500 correlation matrix indistinguishable from noise.

Here is a question that sounds like arithmetic and is actually a trap. You want to build a risk model for 100 stocks. You have a year of daily returns, 250 trading days. Do you have enough data?

The instinct is to say yes. If you were estimating a single stock's volatility from 250 days you would be in fine shape, with a standard error around ${1/\sqrt{2T}}$, roughly 4.5%.

But you are not estimating one number. A covariance matrix for 100 assets has ${100 \times 101 / 2 = 5{,}050}$ distinct entries, and you have ${100 \times 250 = 25{,}000}$ return observations to pin them down. That is about five observations per parameter. No statistician would report a regression fit on five data points per coefficient without a warning label, and yet this matrix goes straight into a portfolio optimiser that treats it as truth.

![The cleaning pipeline: a 100 by 250 returns matrix becomes a sample covariance of 5,050 numbers, whose eigenvalue spectrum splits into 3 signal spikes and 97 noise eigenvalues, which are then clipped before the optimiser sees them](/imgs/blogs/random-matrix-theory-covariance-cleaning-math-for-quants-1.webp)

The diagram above is the whole post in one line. Random matrix theory gives you a *ruler* to hold against that matrix's eigenvalue spectrum: it says what the eigenvalues would look like if there were no structure at all, so anything matching the ruler is noise and anything sticking out of it is signal. Then you clean the noise part before the optimiser sees it. None of this is investment advice; it is a description of how an estimator behaves and where it breaks.

## Foundations: the matrix, and why an optimiser leans on it

For $N$ assets, the **covariance matrix** $\Sigma$ is an $N \times N$ grid whose diagonal holds each asset's variance and whose off-diagonal entry $\Sigma_{ij}$ is the **covariance** between assets $i$ and $j$, positive when they move together and negative when one zigs as the other zags. An **eigenvector** of $\Sigma$ is a direction in portfolio space, a recipe of long and short positions, and its **eigenvalue** $\lambda$ is the variance you experience holding that recipe. Big eigenvalue means a direction that swings a lot; small means a hedged spread whose legs cancel. Both are built from scratch in [the covariance matrix as linear algebra](/blog/trading/math-for-quants/covariance-matrix-linear-algebra-math-for-quants) and [eigendecomposition and PCA on returns](/blog/trading/math-for-quants/eigendecomposition-pca-returns-math-for-quants), and we take them as given.

One convention carries through: returns are **standardised**, each divided by its own volatility so every variance equals 1. That turns $\Sigma$ into a correlation matrix and gives us a conservation law, since the eigenvalues then sum to exactly $N$. That sum is the **trace**, and it matters later.

Now the part that does the damage. The minimum-variance portfolio, the least-risky combination of your assets, has weights

$$w \;=\; \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^{\top}\Sigma^{-1}\mathbf{1}},$$

where $\mathbf{1}$ is a vector of ones. Notice the **inverse**. Every mean-variance optimiser inverts the covariance matrix, and in the eigenbasis the inverse is

$$\Sigma^{-1} \;=\; \sum_{k=1}^{N} \frac{1}{\lambda_k}\, v_k v_k^{\top}.$$

Each direction $v_k$ enters with a multiplier of ${1/\lambda_k}$. The *smallest* eigenvalues, the ones the sample says are almost riskless, get the *largest* weights. That is the whole tragedy in one line: if the smallest eigenvalue is wrong, and it is about to turn out to be systematically and enormously wrong, the optimiser will find it and lever it up. The optimiser itself is covered in [mean variance and the efficient frontier](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants).

#### Worked example: how thin is your data, really?

| Setup | $N$ | $T$ | Parameters ${N(N{+}1)/2}$ | Observations $NT$ | Obs per parameter | $q = N/T$ |
| --- | --- | --- | --- | --- | --- | --- |
| 100 stocks, 1 year daily | 100 | 250 | 5,050 | 25,000 | 4.95 | 0.40 |
| 500 stocks, 2 years daily | 500 | 504 | 125,250 | 252,000 | 2.01 | 0.99 |

The second row is the standard interview setup and far worse than the first: two years of daily data on the S&P 500 gives about two observations per parameter. Put a book behind it. On a \$50M portfolio those 125,250 barely-estimated numbers decide where every one of those dollars goes. The intuition: **doubling the number of assets quadruples what you must estimate while only doubling the data you collect.** Breadth is expensive in a way depth is not.

## The aspect ratio q, and why q near 1 is the danger zone

The single number that governs everything is ${q = N/T}$. Classical statistics assumes $T$ runs to infinity with $N$ fixed, so $q \to 0$ and the sample covariance converges to the truth. That is the regime where the [law of large numbers](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants) saves you, and it is not the regime you are in: portfolio work lives where $N$ and $T$ grow together and $q$ sits at some uncomfortable value like 0.4 or 0.9.

Here is what $q$ does to a matrix with no structure at all. Imagine 100 assets whose true correlations are all exactly zero, so the true eigenvalues are a hundred copies of 1, then measure them from a finite sample.

![Four stacked noise bands showing how the Marchenko-Pastur bulk widens as q rises from 0.1 to 1.0, from 0.47 to 1.73 up to 0.00 to 4.00, around a reference line at 1](/imgs/blogs/random-matrix-theory-covariance-cleaning-math-for-quants-3.webp)

| Sample | $q$ | Measured eigenvalues run from | to |
| --- | --- | --- | --- |
| 100 stocks, 1,000 days | 0.1 | 0.47 | 1.73 |
| 100 stocks, 250 days | 0.4 | 0.14 | 2.66 |
| 100 stocks, 125 days | 0.8 | 0.01 | 3.59 |
| 100 stocks, 100 days | 1.0 | 0.00 | 4.00 |

Read that table slowly. Every one of those spreads is produced by data with **zero** true structure. At $q = 0.4$ the measured eigenvalues run from 0.14 to 2.66, a factor of nineteen between smallest and largest, purely from sampling noise. Without the theory you would conclude you had found a very risky direction and a nearly riskless one. Both are fictions.

Two features matter operationally. The band is **not** symmetric around 1: the low side compresses toward zero much faster than the high side stretches up, and since the optimiser cares about ${1/\lambda}$, the compressed side is the dangerous one. And at $q = 1$ the lower edge touches exactly zero, so the matrix is singular and the optimiser is dividing by zero. Above $q = 1$ it has ${N - T}$ exact zero eigenvalues by construction and cannot be inverted at all.

## The Marchenko-Pastur law: the shape of pure noise

That table is not simulation output. It is a theorem. Marchenko and Pastur proved in 1967 that as $N$ and $T$ both go to infinity with $q = N/T$ fixed, the eigenvalues of a sample covariance matrix built from independent, identically distributed data settle into a known density:

$$\rho(\lambda) \;=\; \frac{1}{2\pi q \lambda}\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}, \qquad \lambda_{\pm} = (1 \pm \sqrt{q})^{2},$$

with $\rho(\lambda) = 0$ everywhere outside ${[\lambda_-, \lambda_+]}$. Here $\rho(\lambda)$ is the density of eigenvalues at value $\lambda$, and $\lambda_-$ and $\lambda_+$ are the **bulk edges**. If the underlying variance is $\sigma^2$ rather than 1, every edge scales: $\lambda_{\pm} = \sigma^2(1 \pm \sqrt{q})^2$.

What makes this useful rather than merely elegant is the sharpness of the edges. The density does not tail off gently; it hits exactly zero at $\lambda_+$ and stays there, and at finite $N$ the edges blur by an amount that shrinks as $N$ grows. That gives you a **null hypothesis with teeth**: if the assets carried no shared structure essentially no eigenvalue would exceed $\lambda_+$, so one that does is a rejection, in the same sense as any [hypothesis test](/blog/trading/math-for-quants/hypothesis-testing-pvalues-math-for-quants), except the null covers an entire spectrum rather than one statistic.

Be precise about what this does *not* claim. Marchenko-Pastur does not prove the bulk is empty of information, only that it is consistent with being empty, so a weak factor whose true eigenvalue is 1.4 against an edge of 2.66 stays invisible. The confident half is asymmetric: an eigenvalue clearly outside the bulk is real, one inside it is unproven. That is enough to act on.

#### Worked example: the bulk edges for 100 stocks and 250 days

Take $N = 100$ and $T = 250$, so ${q = 0.4}$ and ${\sqrt{q} = 0.632}$.

$$\lambda_+ = (1 + 0.632)^2 = 2.66, \qquad \lambda_- = (1 - 0.632)^2 = 0.14.$$

Now suppose the measured spectrum of your 100 stocks looks like this: three eigenvalues at 26.0, 4.0 and 2.9, and ninety-seven scattered between 0.09 and 2.7.

![The Marchenko-Pastur density for q = 0.4 with the bulk shaded between 0.14 and 2.66, plus the market factor spike at lambda 1 = 26 and sector spikes at 4.0 and 2.9 past an axis break](/imgs/blogs/random-matrix-theory-covariance-cleaning-math-for-quants-2.webp)

You can now read it the way a senior researcher does, in about four seconds:

- ${\lambda_1 = 26.0}$ sits roughly ten times past the upper edge. Not a marginal call. It is the **market factor**, and its eigenvector will have roughly equal positive loadings on all 100 stocks. Since the trace is 100, it carries 26% of total variance.
- ${\lambda_2 = 4.0}$ and ${\lambda_3 = 2.9}$ are outside the bulk too, by less. These are typically **sector or style factors**, long one group and short another.
- The remaining 97 all live inside ${[0.14,\ 2.66]}$, exactly the shape a pure-noise matrix produces. **You have no evidence that any of those directions means anything.**

The intuition: the bulk is not "small structure that is hard to see", it is the fingerprint of having measured nothing at all.

## Eigenvalue clipping: the simplest thing that works

If the bulk is unproven, do not let the optimiser bet on it. **Eigenvalue clipping**, introduced in the same 1999 Laloux, Cizeau, Bouchaud and Potters paper, does exactly that: keep every eigenvalue above $\lambda_+$ untouched, replace every one below it with a single common value chosen so the trace still equals $N$, and rebuild from the original eigenvectors.

The trace constraint is what makes it work. You are not throwing the noisy directions away, you are declaring that you cannot tell them apart, so you assign them all the same risk. The eigenvectors are untouched, so the matrix stays positive definite and invertible.

![Before and after eigenvalue clipping: the three spikes at 26.0, 4.0 and 2.9 are untouched while the 97 bulk eigenvalues are replaced by their average 0.692, cutting the leverage on the noisiest direction from 11.1 to 1.45](/imgs/blogs/random-matrix-theory-covariance-cleaning-math-for-quants-4.webp)

#### Worked example: clipping the matrix, in dollars

Continue with that spectrum on a **\$50M book** of 100 stocks. Equal weight would be \$500,000 per name.

Three eigenvalues survive the cut and sum to ${26.0 + 4.0 + 2.9 = 32.9}$. The trace must stay at 100, so the 97 clipped eigenvalues share what is left:

$$\bar{\lambda} = \frac{100 - 32.9}{97} = \frac{67.1}{97} = 0.692.$$

Watch what that does to the leverage the optimiser applies, remembering that direction $k$ enters $\Sigma^{-1}$ with weight ${1/\lambda_k}$:

| Direction | Raw $\lambda$ | Raw ${1/\lambda}$ | Clipped $\lambda$ | Clipped ${1/\lambda}$ |
| --- | --- | --- | --- | --- |
| Market | 26.0 | 0.04 | 26.0 | 0.04 |
| Sector 1 | 4.0 | 0.25 | 4.0 | 0.25 |
| Sector 2 | 2.9 | 0.34 | 2.9 | 0.34 |
| Noisiest bulk direction | 0.09 | 11.1 | 0.692 | 1.45 |

Before cleaning, the noisiest direction, a hedged spread the sample happened to measure as almost riskless, gets ${26.0 / 0.09 = 289}$ times the weight of the market factor. After cleaning it gets 38 times: a **7.7x cut** in leverage on the most suspect direction in the matrix, with the market and sector factors untouched.

Now the dollars. Suppose the raw optimiser comes back wanting **\$18M in one name**, which on \$50M is 36% of the book and 36 times the equal-weight position. That size did not come from the market factor, whose ${1/\lambda}$ is 0.04. It came from the noise directions, and clipping divides their contribution by 7.7, so \$18M over 7.7 puts that name back at roughly \$2.3M: 4.6% of the book, 4.6 times equal weight, an active tilt but a survivable one.

The intuition: clipping does not shrink your portfolio, it stops the optimiser concentrating it in directions that were never actually observed.

#### Worked example: what the noise does to your risk forecast

A companion result, surveyed in [Bouchaud and Potters (2009)](https://arxiv.org/abs/0910.1205), makes the cost concrete. In the same large-$N$, fixed-$q$ limit, the in-sample risk a sample-covariance optimiser reports and the portfolio's true risk are related by

$$R_{\text{in}} = R_{\text{true}}\sqrt{1 - q}.$$

The optimiser does not just get the weights wrong. It also **understates the risk of the weights it picked**, because it fitted the noise and the noise looks like free diversification.

At $q = 0.4$, ${\sqrt{1 - 0.4} = 0.775}$, so true volatility is ${1/0.775 = 1.29}$ times what the optimiser reports. Ask for a 10% target and you are running about 12.9%. On the \$50M book that is a one-standard-deviation year of \$6.45M rather than \$5M: \$1.45M of risk nobody put in the report.

Now the 500-stock, two-year case, where ${q = 500/504 = 0.992}$, ${\sqrt{1 - q} = 0.089}$ and the ratio is ${1/0.089 = 11.2}$. The optimiser reports 4% annualised volatility and the truth is closer to 45%: on \$50M, a risk report saying \$2M against an actual \$22.5M. The formula diverges as $q \to 1$, so the exact multiplier that close to the boundary is not literal. The order of magnitude is the point, and it is a risk number wrong by a factor of ten.

## How this relates to shrinkage, and where it differs

Clipping is not the only answer, nor the best one. The other standard tool is **shrinkage**, most famously Ledoit-Wolf (2004), which pulls the sample matrix toward a structured target such as the identity with the mixing weight chosen analytically. It is derived in [robust and regularised portfolios](/blog/trading/math-for-quants/robust-regularized-portfolios-math-for-quants); what matters here is how the two differ in kind.

| Method | What it does to the eigenvalues | Uses the spectrum's shape? | Invertible after? |
| --- | --- | --- | --- |
| PCA truncation, keep top $k$ | Sets the rest to exactly 0 | No, $k$ is chosen by eye | No, it is singular |
| Eigenvalue clipping | Flattens everything below $\lambda_+$ to one value | Yes, $\lambda_+$ comes from $q$ | Yes |
| Linear shrinkage | Pulls all eigenvalues toward their common mean | No, one factor applied uniformly | Yes |
| Rotationally invariant estimator | Maps each $\lambda$ through a nonlinear function | Yes, the full density | Yes |

The structural difference: **shrinkage moves every eigenvalue, clipping moves only the ones it has grounds to distrust.** Linear shrinkage with intensity $\alpha$ replaces $\lambda_k$ with ${(1-\alpha)\lambda_k + \alpha}$, dragging the market factor at 26.0 toward 1 along with everything else, even though it is the one number in the matrix you are most confident about. In exchange, shrinkage needs no threshold and degrades gracefully, while clipping rests everything on a single estimated cut point.

Both approximate the same ideal, the **rotationally invariant estimator** of Bun, Bouchaud and Potters: keep the sample eigenvectors and map each eigenvalue to the best estimate of the true variance in that direction. Clipping is the crude two-level version of that map, shrinkage the linear one. Start with clipping because you can explain it in a sentence.

## Common misconceptions

**"More history always fixes it."** It helps, slowly. Since $q = N/T$, halving $q$ means doubling $T$: going from $q = 0.8$ to $q = 0.1$ on 100 stocks means 125 days to 1,000 days, six months to four years. And that is the optimistic reading, because covariance is **not stationary**. A four-year window spans regime changes, so the estimate averages over market states that no longer exist. You have traded estimation error for stale structure, and no window length makes both small. Worse, the *effective* $T$ is smaller than the nominal one because volatility clusters. Laloux and coauthors noted exactly this: their fit improved when they allowed a smaller effective sample, which they attributed to volatility correlations.

**"PCA already handles this."** PCA gives you the eigenvalues and eigenvectors. It does not tell you where to cut. A scree-plot elbow or an 80%-of-variance rule is an eyeball with no null distribution behind it, and it will hand you a different answer on the same data if you move the sample period. Marchenko-Pastur supplies the missing threshold. There is also a mechanical difference: PCA truncation *deletes* the small eigenvalues, which makes the matrix singular and useless to any optimiser that needs an inverse, while clipping *floors* them at a positive value and keeps it invertible.

**"Positive definite means good."** Whenever ${q \lt 1}$ the sample covariance is almost surely full rank, invertible and positive definite. It passes every numerical check you throw at it, and a [Cholesky factorisation](/blog/trading/math-for-quants/cholesky-positive-definite-math-for-quants) succeeds on it. It can still be, and at ${q = 0.9}$ it is, almost entirely noise. Positive definiteness is a statement about rank, not accuracy, and the condition number will not tell you which you have.

## How it shows up in real markets

The empirical result that started this field deserves its actual numbers. Laloux, Cizeau, Bouchaud and Potters analysed ${N = 406}$ S&P 500 stocks over ${T = 1{,}309}$ daily observations covering 1991–1996, giving ${T/N = 3.22}$ and therefore ${q = 0.31}$. From ["Noise Dressing of Financial Correlation Matrices"](https://arxiv.org/abs/cond-mat/9810255), *Physical Review Letters*, 1999:

- **94% of the eigenvalues** fell in the region where the Marchenko-Pastur formula applies. The bulk of a real S&P 500 correlation matrix is statistically indistinguishable from pure noise.
- The largest eigenvalue was about **25 times** the predicted upper edge, with roughly equal eigenvector components across all 406 stocks. That is the market. (Against the refitted band the authors preferred, their figure caption puts the gap at 30 times. Either way it is not close.)
- The under 6% of eigenvectors outside the band accounted for **26% of total volatility**. A few real factors, a lot of real risk, and a vast plain of nothing between.

Their closing sentence changed practice: Markowitz optimisation on a raw historical correlation matrix "is not adequate, since its lowest eigenvalues, corresponding to the smallest risk portfolios, are dominated by noise."

Three places this lives on a desk today. **Commercial risk models** exist substantially because of this problem: a factor model with 60 or 80 named factors is, among other things, a way of forcing $q$ down by construction rather than cleaning after the fact. **Statistical arbitrage** runs the diagnostic in reverse, since there the small eigenvalues are the product rather than a nuisance, and the noise floor says which spreads are worth trading. And any **portfolio construction** system that inverts a covariance matrix in production has a cleaning step in front of it, whether it is called clipping, shrinkage, a factor model or a minimum-eigenvalue floor. If you find one that does not, you have found the bug.

## In the interview room and on the desk

This gets asked in two shapes that are the same question. The first is diagnostic: *"You have 500 stocks and two years of daily data. What is wrong with your covariance matrix?"* The second is symptomatic: *"Why does the minimum-variance portfolio blow up out of sample?"* Either way the interviewer wants to know whether you think in terms of the aspect ratio or in terms of vague warnings about overfitting.

A strong answer runs in this order.

1. **Name $q$ immediately.** ${q = N/T = 500/504 \approx 0.99}$. Say the number out loud. Roughly 125,000 parameters against 252,000 observations, about two each, and the matrix is a hair from singular.
2. **Give the bulk.** At $q$ near 1 the Marchenko-Pastur band runs from nearly 0 to nearly 4 even when the true correlations are zero. Most of the measured spectrum is noise with a known shape, not a mystery.
3. **Say which eigenvalues are signal.** The handful above ${(1+\sqrt{q})^2}$: the market factor, then a few sector or style factors. Everything inside the band is unproven.
4. **Connect it to the weights.** The optimiser uses $\Sigma^{-1}$, which weights direction $k$ by ${1/\lambda_k}$, so the smallest and noisiest eigenvalues get the biggest positions. State the blow-up mechanically, not as a metaphor.
5. **Then say what you would do.** Clip the bulk to its trace-preserving average, or shrink, or impose a factor structure. Note that clipping keeps the eigenvectors and the trace, and that the reported in-sample risk is understated by roughly ${1/\sqrt{1-q}}$ regardless.

The trap is answering "use more data." It is not wrong, it is incomplete, and interviewers listen for the missing half. Covariance is non-stationary, so a longer window trades estimation error for **stale structure**: a cleaner estimate of a correlation regime that has already ended. The complete answer gives both halves, then notes that the effective sample is smaller than the nominal one anyway because volatility clusters.

Where this weighs most: **Two Sigma** and **Citadel** research interviews, and any portfolio-construction or risk-modelling seat where someone owns the matrix that goes into the optimiser.

Natural next steps are the estimator theory behind why the sample covariance is unbiased and still useless here, in [bias, variance and consistency](/blog/trading/math-for-quants/estimators-bias-variance-consistency-math-for-quants), and the shrinkage family clipping belongs to, in [robust and regularised portfolios](/blog/trading/math-for-quants/robust-regularized-portfolios-math-for-quants).

## Sources and further reading

- V. A. Marchenko and L. A. Pastur, "Distribution of eigenvalues for some sets of random matrices," *Matematicheskii Sbornik*, 1967. The original theorem.
- L. Laloux, P. Cizeau, J.-P. Bouchaud and M. Potters, ["Noise Dressing of Financial Correlation Matrices,"](https://arxiv.org/abs/cond-mat/9810255) *Physical Review Letters* 83, 1467 (1999). Source of every S&P 500 figure quoted above, and of eigenvalue clipping itself.
- J.-P. Bouchaud and M. Potters, ["Financial Applications of Random Matrix Theory: a short review,"](https://arxiv.org/abs/0910.1205) 2009. The accessible survey, including the in-sample versus true risk relation.
- J. Bun, J.-P. Bouchaud and M. Potters, ["Cleaning large correlation matrices: tools from random matrix theory,"](https://arxiv.org/abs/1610.08104) *Physics Reports*, 2017. The rotationally invariant estimator.
- O. Ledoit and M. Wolf, ["A well-conditioned estimator for large-dimensional covariance matrices,"](https://doi.org/10.1016/S0047-259X(03)00096-4) *Journal of Multivariate Analysis* 88(2), 365–411 (2004). The shrinkage counterpart.

Every other number here sits inside a clearly labelled hypothetical worked example. The 100-stock and 500-stock spectra, the \$50M book and the dollar positions derived from them are illustrative arithmetic, not measurements of any real portfolio.
