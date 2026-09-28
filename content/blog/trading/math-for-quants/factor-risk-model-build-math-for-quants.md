---
title: "Building a factor risk model: the decisions nobody writes down"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "Everyone can describe a factor model. Almost nobody can defend the twenty choices that turn one into a working risk system, and those choices move the risk number further than the estimator does. Factor selection, exposure windows, specific risk, half-life and the bias test, with the dollars attached."
tags:
  [
    "factor-model",
    "risk-model",
    "barra",
    "specific-risk",
    "bias-test",
    "covariance-matrix",
    "portfolio-risk",
    "factor-exposures",
    "collinearity",
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
> **TL;DR:** A factor risk model is twenty judgement calls wearing one equation. The estimator is the easy part, and the choices are what a senior owns.
>
> - The identity is ${V = BFB' + D}$: exposures, a factor covariance, and specific variance. It shrinks 125,250 free parameters onto 6,578 for a 500-name book. You are not avoiding shrinkage, you are **choosing its shape**.
> - The same book under two defensible models can report **6.32%** and **9.46%** annualised volatility. Against a 10% limit that is \$1,266m versus \$846m of allowed gross, a **\$420m** difference in position that no estimator argument resolves.
> - More factors is not better. Two factors correlated 0.92 moved the risk number by 0.04 points and the attribution from **100/0 to 32.3/67.7**. The risk number survived collinearity; the attribution did not, and the PM is paid on the attribution.
> - **Specific risk is where the concentrated book lives.** On an 18% position, 75.1% of the variance was specific, and averaging specific vols down to the cross-sectional mean hid **\$11.3m** of annualised volatility.
> - The one measurable thing is the **bias statistic**: the standard deviation of realised-over-forecast returns, which should be 1 inside a band of ${1 \pm \sqrt{2/T}}$. The same value of 1.27 is unfalsifiable at ${T = 12}$ and a clear failure at ${T = 60}$.

Ask a candidate to describe a factor risk model and almost all of them can. Returns decompose into a few common drivers plus a bit that belongs to the name alone; you estimate exposures, you estimate a factor covariance, you glue them together. It takes ninety seconds and it is correct.

Then ask them to build one for a specific book, and the ninety seconds run out. Which family? How many factors? Estimated over what window? What happens to the number when the answers change? That last question is the one that matters, because the answers change the risk number by more than any estimator choice ever will, and somebody has to sign their name under them.

![Three ways to build a factor model: statistical, fundamental and macro, compared by what you supply, what is estimated, and where each wins and fails](/imgs/blogs/factor-risk-model-build-math-for-quants-1.webp)

This post is the choices. All dollar figures below are illustrative arithmetic on assumed inputs, chosen so every line is reproducible from the numbers printed beside it.

## The building blocks, from zero

A **factor** is a source of return that many assets share. The market is one. Being a small company is another. Being an oil producer is a third. A **factor return** is how much that shared driver earned in a period. An **exposure** (or loading, or beta) is how much of that factor a given asset carries.

The claim of a factor model is that an asset's return in period ${t}$ splits into a part explained by the factors and a part that belongs to the asset alone:

$$r_{i,t} = \sum_{k=1}^{K} b_{ik} f_{k,t} + \varepsilon_{i,t}$$

The residual ${\varepsilon_{i,t}}$ is the **specific return**, sometimes called idiosyncratic. The model's central assumption is that specific returns are uncorrelated across names. Everything two assets have in common is supposed to be inside the factors.

Stack that across ${N}$ assets and the covariance matrix of returns becomes

$$V = B F B' + D$$

where ${B}$ is the ${N \times K}$ matrix of exposures, ${F}$ is the ${K \times K}$ covariance of factor returns, and ${D}$ is a diagonal matrix of specific variances. A portfolio with weights ${w}$ has variance ${w'Vw}$, and writing that out gives the layout of every risk report you will ever read:

$$\sigma_p^2 = \underbrace{(B'w)' F (B'w)}_{\text{factor risk}} + \underbrace{w'Dw}_{\text{specific risk}}$$

The vector ${B'w}$ is the portfolio's factor exposure. That is the whole model.

![The identity V equals B F B transpose plus D, shown as blocks, shrinking 125,250 free parameters onto 6,578](/imgs/blogs/factor-risk-model-build-math-for-quants-2.webp)

Look at what that structure buys. A 500-name book has ${500 \times 501 / 2 = 125{,}250}$ distinct covariance entries, typically estimated from two years of daily data, about 500 observations. [Random matrix theory](/blog/trading/math-for-quants/random-matrix-theory-covariance-cleaning-math-for-quants) says precisely how badly that goes: when the number of assets is comparable to the number of observations, most of what you measure is noise with a known shape. A 12-factor model replaces those 125,250 numbers with ${500 \times 12 = 6{,}000}$ exposures, ${12 \times 13 / 2 = 78}$ factor covariances and 500 specific variances, so 6,578 in total, 19.0 times fewer.

So a factor model **is** a shrinkage of the covariance matrix, in the sense of [the Stein paradox](/blog/trading/math-for-quants/shrinkage-stein-paradox-math-for-quants), except that the target is a structure rather than a scalar multiple of the identity. You have asserted that all cross-name correlation runs through 12 channels and none of it runs anywhere else. That assertion is the model, and every argument below lives inside it.

## The same book, two defensible models, \$420m apart

Before any of the decisions, here is why they matter.

#### Worked example 1: one book, two risk numbers

Take a \$500m market-neutral equity book running \$800m of gross exposure, against a firm limit of 10% annualised volatility.

**Model A** is statistical: three principal components. They are orthogonal by construction, so the factor covariance is diagonal and the arithmetic is a sum of squares. The book's exposures come out at 0.08 to PC1, 0.45 to PC2 and -0.30 to PC3, with factor volatilities of 18%, 11% and 7%:

- PC1: ${0.08^2 \times 0.18^2 = 0.00020736}$
- PC2: ${0.45^2 \times 0.11^2 = 0.00245025}$
- PC3: ${0.30^2 \times 0.07^2 = 0.000441}$

Factor variance 0.00309861, so factor volatility 5.57%. Specific volatility comes out at 3.00%, variance 0.0009. Total variance 0.00399861, total volatility **6.32%**, which is \$31.6m on \$500m.

**Model B** is fundamental: named style factors with characteristics as exposures. The same book shows market beta 0.05, value 0.60 and momentum -0.35, with factor volatilities of 16%, 10% and 12%. These factors are not orthogonal, so the cross terms are real. Write ${a_k}$ for exposure times volatility: ${a_{mkt} = 0.008}$, ${a_{val} = 0.06}$, ${a_{mom} = -0.042}$.

- Squares: ${0.000064 + 0.0036 + 0.001764 = 0.005428}$
- Market with value, ${\rho = -0.20}$: ${2(-0.20)(0.008)(0.06) = -0.000192}$
- Market with momentum, ${\rho = 0.10}$: ${2(0.10)(0.008)(-0.042) = -0.0000672}$
- Value with momentum, ${\rho = -0.40}$: ${2(-0.40)(0.06)(-0.042) = +0.002016}$

Factor variance 0.0071848, factor volatility 8.48%. Specific volatility 4.20%. Total variance 0.0089488, total volatility **9.46%**, which is \$47.3m.

Now price it. Against a 10% limit, Model A says the book can scale by ${10 / 6.32 = 1.5823}$, so gross grows to ${800 \times 1.5823 = \$1{,}266\text{m}}$. Model B says ${10 / 9.46 = 1.0571}$, so \$846m. The difference is **\$420m of gross exposure**, decided by which model the firm happens to run.

Both models are defensible and neither is an error. The largest single term in the gap is the value-momentum correlation of -0.40, which Model A never sees because its factors are orthogonal by construction and that correlation has been absorbed into the components. Not an estimation problem. A **modelling** choice, worth \$420m.

## Decision one: which family

The three families differ in which side of the equation you supply.

**Statistical models** hand the algorithm a returns panel and let it find both the factors and the exposures, so they need nothing but prices and they fit best on the data they were fitted to. The cost arrives in the risk meeting: the fourth component has no name, so when a PM asks why their risk rose 15% you have nothing beyond "the fourth eigenvector moved". [Eigendecomposition and PCA](/blog/trading/math-for-quants/eigendecomposition-pca-returns-math-for-quants) covers the extraction.

**Fundamental models**, the Barra lineage starting with Rosenberg (1974), reverse it. You supply exposures as observable characteristics, book-to-price, market cap, twelve-month return, industry membership, and a [cross-sectional regression](/blog/trading/math-for-quants/regression-ols-gls-regularized-math-for-quants) each period backs out the factor returns. Every factor has a name a PM recognises, which is most of why these dominate at multi-manager firms. The cost is that the exposures are your definitions, and a bad definition is invisible: if value is book-to-price and half the book is asset-light software, the reported value exposure is an accounting artefact.

**Macro models** supply observable factor series, the ten-year yield or the oil price, and estimate exposures by time-series regression. Easy to explain and easy to stress, but estimated macro betas are unstable, so the exposures move more than the book does.

Connor (1995) found the fundamental model ahead of the statistical one on explanatory power, and the statistical one ahead of the macro one. Worth knowing, and worth not over-reading, because explanatory power is not what you are buying.

## Decision two: how many factors, and is this one actually new

The instinct is that more factors explain more variance, so more factors is better. Explanatory power rises monotonically with factor count, so the instinct is arithmetically correct and practically backwards.

The test for whether a candidate factor is new is not whether it explains returns. It is whether it explains anything the existing set does not. Regress the candidate's return series on the factors you already have and read the residual:

$$\sigma_{\text{residual}} = \sigma_{\text{new}}\sqrt{1 - R^2}$$

A candidate quality factor with 9% annualised volatility and ${R^2 = 0.82}$ against the existing set leaves ${9\% \times \sqrt{0.18} = 3.82\%}$ of independent volatility. If the book's exposure to that residual is 0.20, the variance the new factor adds that nothing else could explain is ${0.20^2 \times 0.0382^2 = 0.0000584}$. On a book whose total variance is 0.0197, that is **0.30%** of the risk number.

Now the bill. The variance inflation factor for the affected exposures is ${1/(1 - R^2) = 5.56}$, so the standard error of every exposure that overlaps with the new factor rises by ${\sqrt{5.56} = 2.36}$ times. You bought 0.30% of variance explained and multiplied the attribution's standard errors by 2.36.

That trade is the whole of factor selection, and it produces the **null result** this post owes you: a case where the more sophisticated model is not better.

![Two fits of the same book with collinear factors: the volatility moves 0.04 points while the attribution moves 68 points](/imgs/blogs/factor-risk-model-build-math-for-quants-5.webp)

Take two factors that genuinely overlap, value and earnings yield, correlated 0.92, with volatilities of 10% and 11%. One fit puts the book at 0.60 value and 0.00 earnings yield. Factor variance ${0.60^2 \times 0.10^2 = 0.0036}$, volatility **6.00%**, \$30.0m on \$500m. A second fit, on data a month later or with a slightly different estimator, puts it at 0.20 value and 0.37 earnings yield:

$$0.02^2 + 0.0407^2 + 2(0.92)(0.02)(0.0407) = 0.00355425$$

Volatility **5.96%**, \$29.8m. The risk number moved \$0.2m, which is nothing.

The attribution did not. Each factor's contribution is ${f_k (Ff)_k}$. For value, ${0.20 \times 0.0057444 = 0.00114888}$, which is **32.3%** of the total. For earnings yield, ${0.37 \times 0.006501 = 0.00240537}$, or **67.7%**. The first fit said value was 100% of the style risk and earnings yield was zero.

So the richer model reported the same risk and a completely different reason for it. If the firm allocates capital by factor bucket, or charges a PM for crowding in value, the second fit moves real money on a distinction the data cannot support. Collinear factors leave the risk number alone and destroy the attribution, and the attribution is what people are paid on.

## Decision three: exposures, and why a bad beta is worse than no beta

A fundamental model reads exposures off characteristics, which are observable and stable. A statistical or macro model estimates them, which means a window, which means a trade-off nobody escapes.

#### Worked example 2: the 60-day beta

A stock with 2.2% daily volatility against a market with 1.0% daily volatility and a true beta of 1.0 has residual volatility ${\sqrt{0.022^2 - 0.010^2} = 1.96\%}$. The standard error of an OLS beta over ${T}$ days is

$$\text{SE}(\hat{\beta}) = \frac{\sigma_\varepsilon}{\sigma_m \sqrt{T}} = \frac{0.0196}{0.010\sqrt{60}} = 0.253$$

A 95% interval on that beta runs ${1.0 \pm 1.96 \times 0.253}$, which is 0.50 to 1.50. You measured the beta and you still do not know whether the stock is half as volatile as the market or half again as volatile.

Compare the lazy alternative: set every beta to 1.0 and skip the estimation. If the true cross-sectional dispersion of betas is ${\tau = 0.30}$, that estimator has zero variance and mean squared error ${0.30^2 = 0.09}$. The rolling beta has bias zero and mean squared error ${0.253^2 = 0.064}$. The measured beta wins, but only just, and the gap is far smaller than the effort suggests.

The answer is neither. Shrink the estimate toward the prior with weight ${w = \tau^2/(\tau^2 + \text{SE}^2) = 0.09/0.154 = 0.58}$, and the mean squared error becomes

$$(1 - 0.58)^2 (0.09) + 0.58^2 (0.064) = 0.0374$$

Root mean squared error 0.193, against 0.253 for the raw beta and 0.300 for the flat prior. Shrinking cuts the beta error by 23.7% relative to the rolling estimate, and it costs one line of code.

This is where "worse than no beta" becomes precise. A short window gives you an unbiased estimate with enough noise that the interval covers the decision either way, and the model then reports that noise as if it were exposure. Longer windows cut the noise and buy staleness instead: a 500-day beta on a company that did a debt-funded acquisition eighteen months ago is measuring a firm that no longer exists. The senior's version of this decision is not "pick a window" but "pick a window, then shrink, and say what you shrank toward".

## Decision four: specific risk, the part everyone under-funds

Specific risk is the diagonal. It gets an afternoon while the factor covariance gets a quarter, and on a concentrated book it is the larger half by a wide margin.

The reason is arithmetic. Specific variance is ${\sum_i w_i^2 s_i^2}$, quadratic in the weights, so concentration hits it much harder than it hits factor risk, which is linear in the weights through ${B'w}$. Double a position and its factor contribution doubles; its specific variance quadruples.

#### Worked example 3: an 18% position

A \$500m book holds eleven names: one at \$90m, which is 18%, and ten at \$41m each, which is 8.2%. The large position has 45% annualised specific volatility, the rest 35%. Factor volatility for the book is 7.00%.

$$0.18^2(0.45^2) + 10 \times 0.082^2(0.35^2) = 0.0065610 + 0.0082369 = 0.0147979$$

Specific volatility 12.16%. Add factor variance 0.0049 and the total is 0.0196979, so total volatility **14.03%**, or \$70.2m. The factor share of variance is ${0.0049/0.0196979 = 24.9\%}$. Three quarters of this book's variance sits in the part the factor model does not model.

![Two stacked bars comparing reported and actual risk on a concentrated book, showing 11.3 million dollars of hidden volatility](/imgs/blogs/factor-risk-model-build-math-for-quants-4.webp)

Now suppose the model does the usual thing and estimates specific volatility by a cross-sectional average of 30% rather than name by name:

$$(0.0324 + 0.06724) \times 0.09 = 0.0089676$$

Specific volatility 9.47%, total variance 0.0138676, total volatility **11.78%**, or \$58.9m. The model reports \$58.9m of annualised volatility where \$70.2m is present. The limit system never sees **\$11.3m** of it.

Then the large name gaps 35% on an earnings miss, which single names do. That is ${\$90\text{m} \times 0.35 = \$31.5\text{m}}$. The model's one-day 99% value at risk for the entire book was built on a daily volatility of ${11.78\%/\sqrt{252} = 0.742\%}$, so ${2.326 \times 0.742\% = 1.726\%}$, or \$8.63m. One position delivered **3.65 times the whole book's stated daily VaR**, and no factor moved.

Specific risk is also where the model's core assumption is most likely to be false. Two names in the same supply chain have correlated residuals whatever your factor set says, and a diagonal ${D}$ prices that correlation at zero.

## Decision five: half-life, and the direction of the error

Every covariance estimate weights history. An exponentially weighted estimator with decay ${\lambda}$ has an effective sample size of ${(1+\lambda)/(1-\lambda)}$, and the standard error of a volatility estimate is roughly ${\sigma/\sqrt{2 N_{\text{eff}}}}$.

A 21-day half-life gives ${\lambda = 0.9675}$ and ${N_{\text{eff}} = 60.6}$, so the estimate carries a 9.1% standard error. On a 10% volatility forecast that is 0.91 points of pure estimation noise, which is \$4.6m of volatility budget on a \$500m book. A 180-day half-life gives ${\lambda = 0.9962}$ and ${N_{\text{eff}} = 519.4}$, a 3.1% standard error, 0.31 points, \$1.6m.

So the short half-life is noisier, and the noise is not the interesting part. The **direction** is. A fast half-life tracks the recent past, and volatility clusters, so the model's forecast is lowest exactly after a long quiet stretch. A limit system driven by that model therefore grants the most capital at the moment when the conditional probability of a volatility spike is highest, and cuts it hardest after the spike has already happened, when forward volatility is mean-reverting downward. It is procyclical in both directions at once.

The specific, sayable version: **calibrate on a half-life much shorter than your holding period and the model will be too low going into stress and too high coming out of it.** The errors are autocorrelated, which is why they do not wash out over the period a PM is actually measured on. Match the half-life to the horizon at which you can actually change the book. A book that takes three weeks to unwind should not be risk-managed on a five-day half-life, however well that half-life backtests on daily returns.

## Validation: the bias test is the one thing you can measure

Everything above is judgement. This section is not.

If a risk model forecasts volatility ${\sigma_t}$ for period ${t}$ and the portfolio then realises return ${r_t}$, the **standardised return** is ${b_t = r_t/\sigma_t}$. When the model is right, these are draws from a distribution with standard deviation 1. The **bias statistic** is their sample standard deviation:

$$B = \sqrt{\frac{1}{T-1}\sum_{t=1}^{T}\left(b_t - \bar{b}\right)^2}$$

${B > 1}$ means the model under-forecast risk. ${B \lt 1}$ means it over-forecast. The confidence band, in the form the Barra literature uses, is

$$1 \pm \sqrt{2/T}$$

which comes from the sampling variance of a sample standard deviation, ${1/(2T)}$, taken to two standard deviations.

#### Worked example 4: twelve months of a \$500m book

| Month | Forecast vol ${\sigma_t}$ | Realised ${r_t}$ | P&L on \$500m | ${b_t = r_t/\sigma_t}$ |
| --- | --- | --- | --- | --- |
| 1 | 2.9% | +1.13% | +\$5.7m | 0.39 |
| 2 | 3.1% | -5.15% | -\$25.8m | -1.66 |
| 3 | 2.7% | +1.78% | +\$8.9m | 0.66 |
| 4 | 2.5% | +5.20% | +\$26.0m | 2.08 |
| 5 | 3.4% | -1.77% | -\$8.9m | -0.52 |
| 6 | 3.0% | -2.52% | -\$12.6m | -0.84 |
| 7 | 2.8% | +4.20% | +\$21.0m | 1.50 |
| 8 | 3.3% | -6.14% | -\$30.7m | -1.86 |
| 9 | 2.6% | +0.83% | +\$4.2m | 0.32 |
| 10 | 2.9% | +2.93% | +\$14.7m | 1.01 |
| 11 | 3.2% | -4.10% | -\$20.5m | -1.28 |
| 12 | 2.8% | +2.24% | +\$11.2m | 0.80 |

The twelve ${b_t}$ sum to 0.60, so ${\bar{b} = 0.05}$. Their squared deviations from 0.05 sum to 17.7262. Divide by ${T - 1 = 11}$ to get 1.6115, and the square root is

$$B = 1.27$$

The model under-forecast the book's volatility by 27%. Except that with ${T = 12}$ the band is ${1 \pm \sqrt{2/12} = 1 \pm 0.408}$, running from 0.592 to 1.408, and 1.27 sits comfortably inside it. **You cannot reject this model.** A year of monthly data is not evidence.

Run the same statistic over 60 months and suppose it still reads 1.27. Now the band is ${1 \pm \sqrt{2/60} = 1 \pm 0.183}$, from 0.817 to 1.183, and 1.27 is outside. Same number, opposite verdict, and the only thing that changed is ${T}$.

![The bias statistic against sample size, with the confidence funnel narrowing and the same value of 1.27 inside the band at T equals 12 and outside at T equals 60](/imgs/blogs/factor-risk-model-build-math-for-quants-3.webp)

Price the failure. The model says 10% annualised volatility, so \$50m on \$500m. A bias statistic of 1.27 says the truth is 12.7%, or \$63.5m. In daily VaR terms, ${12.7\%/\sqrt{252} = 0.800\%}$ against the model's ${10\%/\sqrt{252} = 0.630\%}$, so 99% one-day VaR is ${2.326 \times 0.800\% \times \$500\text{m} = \$9.30\text{m}}$ where the model reports ${2.326 \times 0.630\% \times \$500\text{m} = \$7.33\text{m}}$. Every day, the firm is carrying **\$1.97m** more tail exposure than its own system believes.

Two things make this statistic usable rather than academic. First, five years of monthly data is a long wait, which is why production risk models run the bias test **cross-sectionally**: take a thousand portfolios in a single month, standardise each, and the effective sample size arrives immediately. Menchero, Morozov and Shepard document exactly this in the GEM2 model notes. Second, the test is only as good as the portfolios you run it on. A model can be unbiased on random portfolios and badly biased on optimised ones, because an optimiser deliberately loads on whichever directions the model says are cheap, which are precisely the directions where the model is most likely wrong. Run the bias test on the portfolios you actually hold.

## Common misconceptions

**"More factors explain more variance, so the model is better."** More factors do explain more variance, in sample, always. What they also do is inflate the standard error of every exposure they overlap with, by ${\sqrt{1/(1-R^2)}}$. The earnings-yield example above bought 0.30% of variance and paid 2.36 times the attribution error. A risk model's job is stable decomposition, and past a point factors subtract from that.

**"${R^2}$ measures a risk model's quality."** ${R^2}$ measures how much of past return the factors absorbed. A risk model's output is a forecast of dispersion, so the correct scoreboard is whether realised returns divided by forecast risk have standard deviation 1. Those are different questions, and they can disagree: a model can have high ${R^2}$ and a bias statistic of 1.27, which is the model in the worked example above.

**"The risk model's job is to predict returns."** It is the opposite. The factor returns are what the model strips out. A risk model that quietly forecast returns would be a signal, and you would trade it rather than divide by it. Its job is the second moment, and confusing the two is how a shop ends up with a risk model that is short whatever its alpha model is long.

## The risk model decides who gets capital

This part is usually left out of the textbook, and it is why the job is senior.

The risk model is not a measuring instrument sitting beside the business. It is the arbiter. Limits are expressed in its units, capital is allocated against its numbers, and a PM's marginal contribution to risk is what decides whether they can add to a position. Change the half-life and you have moved money between desks. Add an industry factor and a trade that was idiosyncratic becomes crowded. Estimate specific risk name by name instead of by cross-sectional average and the concentrated book loses capacity while the diversified one gains.

None of that makes the choices political in a bad sense. It makes them **contested**, which is different and healthy. The PM who argues their sector's specific risk is over-estimated is often right. The failure mode is not disagreement, it is a risk team that cannot say why it chose what it chose, because then the argument is settled by seniority rather than by evidence.

What a senior owns, concretely: every decision above written down with its alternative and its reason, the bias statistic published on the portfolios that actually exist, and a standing answer to "what would change this number most". If you can produce those three things, the contested decisions stay technical. If you cannot, the risk model becomes whatever the loudest PM last negotiated.

## Sources and further reading

- Rosenberg, B. (1974). "Extra-Market Components of Covariance in Security Returns." *Journal of Financial and Quantitative Analysis*, 9(2), 263-274. The origin of the fundamental factor model.
- Connor, G. (1995). "The Three Types of Factor Models: A Comparison of Their Explanatory Power." *Financial Analysts Journal*, 51(3), 42-46.
- Grinold, R. C. and Kahn, R. N. (1999). *Active Portfolio Management*, 2nd ed. McGraw-Hill. Chapters 3 and 4 on risk models and factor structure.
- Menchero, J., Morozov, A. and Shepard, P. (2008). "The Barra Global Equity Model (GEM2)." MSCI Barra Research Notes. Contains the bias-test methodology and the ${1 \pm \sqrt{2/T}}$ band.
- Ledoit, O. and Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix." *Journal of Portfolio Management*, 30(4), 110-119.
- Fama, E. F. and French, K. R. (1993). "Common Risk Factors in the Returns on Stocks and Bonds." *Journal of Financial Economics*, 33(1), 3-56.

All dollar figures in the worked examples are illustrative arithmetic on assumed inputs, not measurements of any real book.

## In the interview room and on the desk

The question is usually open: "how would you build a risk model for this book?" Reciting statistical, fundamental and macro is the easy half, and every candidate gets there. The half that separates people is what comes next.

The strong answer has three moves, in order. **Pick one family and justify it from the book**, not from a ranking. A twenty-name concentrated book does not need forty factors; it needs specific risk estimated name by name, and you should say so. A thousand-name systematic book in forty countries needs the fundamental model because somebody has to explain the country and currency exposures to an allocator. **Then name the two or three decisions that would move the answer most**, which for most books are the factor set, the specific-risk estimator and the half-life, and give a direction for each: a short half-life is procyclical, an averaged specific risk under-states concentration, collinear factors leave the risk number alone and wreck the attribution. **Then say how you would validate**, which means the bias statistic, the band ${1 \pm \sqrt{2/T}}$, run cross-sectionally so you do not wait five years, and run on the portfolios you actually hold rather than random ones.

The trap is treating fit quality as model quality. A candidate who says "I would check the ${R^2}$ and add factors until it stops improving" sounds rigorous and has described a procedure that reliably produces a worse risk model, because in-sample explanatory power rises with every factor while the attribution's standard errors rise with it. The follow-up that catches this is gentle: "your model explains 85% of variance and the bias statistic is 1.3, what do you do?" The only correct answer is that the ${R^2}$ is irrelevant to that problem and the model is under-forecasting risk by 30%.

Citadel and Two Sigma weight this heavily in portfolio-construction and risk seats, where the model is the firm's allocation mechanism rather than a reporting layer. WorldQuant asks it from the alpha side, where the question becomes which risk factors to neutralise against. Any multi-strategy risk seat will ask some version of it, because that seat exists to defend these choices to people whose capital depends on them.
