---
title: "Combining Weak Alphas: What Actually Happens When You Stack 100 Signals"
date: "2026-08-30"
publishDate: "2026-08-30"
description: "The value of a signal book is not the sum of its signals. Correlation decides whether the hundredth alpha adds anything, and the arithmetic is more unforgiving than most researchers expect."
tags: ["alpha-research", "information-coefficient", "fundamental-law", "signal-combination", "portfolio-construction", "correlation", "quantitative-research", "breadth", "transaction-costs", "quant-interviews"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 19
---

> [!important]
> **TL;DR:** Stacking more alphas buys you less and less, and past a point it buys you nothing at all.
>
> - For n equally weighted signals with average pairwise correlation rho, the combined information coefficient scales like ${\sqrt{n}/\sqrt{1+(n-1)\rho}}$. That expression has a ceiling of ${1/\sqrt{\rho}}$, and it does not care how many signals you own.
> - At rho = 0.3, ten signals already capture 90% of what an infinite book would capture. Going from 50 signals to 100 adds 0.08 of an effective independent bet.
> - Breadth in the fundamental law counts *independent bets*, not signals and not tickers. Getting that wrong overstated the information ratio by a factor of 6.6 in the worked example below.
> - A new signal earns a positive weight only if its IC beats its correlation with the existing book times the book's IC. Below that hurdle it is a hedge, not an alpha.
> - On a \$200m book at 10% target volatility, net alpha peaks near 25 signals at \$24.5m a year. At 100 signals it is \$22.4m. The hundredth signal loses about \$37,000 a year.

## Introduction

There is a particular kind of meeting that happens at every systematic fund. A researcher has spent a quarter producing eleven new alphas. Each one is weak but real: an information coefficient somewhere around 0.02 to 0.03, statistically alive, economically sensible. The researcher wants them in the book. The portfolio manager asks the only question that matters, which is not "are these signals any good" but "what do these signals add to the signals we already have".

Those are wildly different questions, and the gap between them is the entire subject of this post.

The industry's public story encourages the confusion. The best known public artefact is *101 Formulaic Alphas* (Kakushadze, 2016), a catalogue of one-line expressions, most with ICs that would embarrass a single-signal researcher, and the implicit promise is that quantity compounds into quality. It does, but only along a curve that flattens fast. Firms that industrialise alpha production this way, WorldQuant most explicitly, would still tell you the same thing in an interview: the size of the library is not the edge. The independence inside it is.

The figure below is the mental model for everything that follows. It plots the multiplier that stacking n equally weighted signals applies to a single signal's IC, for four different levels of average pairwise correlation. Only the fantasy case, rho = 0, keeps climbing. Every real book lives on one of the curves that flattens.

![Line chart of the combined IC multiplier against the number of signals, for average pairwise correlations of 0, 0.1, 0.3 and 0.5. Each correlated curve flattens toward a ceiling of one over the square root of rho.](/imgs/blogs/combining-weak-alphas-math-for-quants-1.webp)

An earlier post, [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research), walks the pipeline from idea to position and spends one section on combination. This post does the combination properly: the arithmetic, the breadth accounting, the weighting schemes, the estimation error that makes the sophisticated scheme lose to the naive one, and the trading cost that turns a saturating gross curve into a declining net one.

## Foundations: three ideas you need first

### The information coefficient, briefly

An *alpha signal* is a number you compute for each stock at each rebalance that is supposed to predict which stocks will outperform. The *information coefficient*, or IC, is the correlation between your signal and the forward return you were trying to predict, measured across the cross-section of stocks at a point in time and then averaged over time. If you rank 500 stocks by your signal every week and correlate that ranking with the following week's returns, the average of those weekly correlations is your IC.

Scale calibration matters more than the definition. An IC of 0.03 means your signal explains about 0.09% of the cross-sectional variance in returns. That sounds like nothing, and per bet it is nothing. Run it across 500 names and 52 weeks a year and it becomes a business. A sustained IC above 0.05 on liquid equities is genuinely strong; anything above 0.10 usually means a bug, a look-ahead, or a backtest you have overfit. The full battery of evaluation metrics, including how to tell a real IC from a lucky one, is in [evaluating alpha signals](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research).

### The fundamental law of active management

Grinold and Kahn's *Active Portfolio Management* gives the bridge from a per-bet edge to a portfolio-level edge:

$$
\mathrm{IR} \approx \mathrm{IC} \cdot \sqrt{\mathrm{BR}}
$$

Here IR is the *information ratio*, the annualised active return divided by the annualised active risk, and BR is the *breadth*, the number of independent forecasts you make per year. The formula says two things. Skill per bet enters linearly. The number of bets enters as a square root, which is the same square root that governs the standard error of any average, because that is exactly what a portfolio of many small bets is.

The square root is the reason a 0.03 IC is a real business and also the reason the hundredth signal is worth so little. It giveth and it taketh away.

### The book we will keep returning to

Every dollar figure in this post refers to one hypothetical book, so that the numbers can be compared directly:

- **\$200m** of capital, market-neutral, run at a **10% annualised target volatility**. One standard deviation of annual P&L is therefore \$20m.
- **500 names**, rebalanced **weekly**.
- Effective breadth of **600 independent bets per year**, which we derive rather than assume in the breadth section below.
- Each signal has an IC of **0.03** unless stated otherwise.

One consequence of that setup is worth memorising, because it is the exchange rate that turns every piece of statistics in this post into money: on this book, **one point of information ratio is \$20m a year**. An IR of 1.0 is \$20m. An IR of 1.33 is \$26.6m. The whole argument that follows is about how many hundredths of an IR point a marginal signal is really worth.

## The arithmetic of stacking

Take n signals, each standardised to unit cross-sectional variance, each with the same IC of c, and each pair correlated at rho. Equal-weight them into one combined signal. The combined signal's covariance with forward returns is just the average of the individual covariances, so the numerator does not grow at all. What changes is the denominator: the combined signal's own volatility. Averaging n correlated variables gives a variance of ${[1 + (n-1)\rho]/n}$ rather than ${1/n}$, and the IC is a correlation, so it is the ratio of the two:

$$
\mathrm{IC}_n = c \cdot \frac{\sqrt{n}}{\sqrt{1 + (n-1)\rho}}
$$

Everything interesting about signal combination is a consequence of that one fraction.

When rho is zero it reduces to ${c\sqrt{n}}$, the textbook diversification result, and it never stops growing. When rho is positive, take the limit as n goes to infinity and the n's cancel: the multiplier converges to ${1/\sqrt{\rho}}$. An infinite library of signals correlated at 0.3 is worth exactly 1.83 signals' worth of IC. Not approximately, not eventually. That is the ceiling, and no amount of research headcount moves it.

It is more useful to think in *effective independent signals*, which is the multiplier squared:

$$
n_{\text{eff}} = \frac{n}{1 + (n-1)\rho}
$$

which has the memorable ceiling ${1/\rho}$. Three and a third independent signals is all a book of rho = 0.3 alphas can ever contain, however many rows the library has.

The saturation table is the part to keep:

| Signals | rho = 0.1 | rho = 0.3 | rho = 0.5 |
| --- | --- | --- | --- |
| 2 | 1.35 | 1.24 | 1.15 |
| 5 | 1.89 | 1.51 | 1.29 |
| 10 | 2.29 | 1.64 | 1.35 |
| 20 | 2.63 | 1.73 | 1.38 |
| 50 | 2.91 | 1.78 | 1.40 |
| 100 | 3.03 | 1.80 | 1.41 |
| ceiling | 3.16 | 1.83 | 1.41 |

Read the rho = 0.3 column and the honest answer to "how many signals are enough" falls out. Ten signals reach 90% of the ceiling. Twenty reach 95%. Fifty reach 98%. At rho = 0.5 the answer is five. At rho = 0.1, where signals are genuinely diverse, it is worth pushing to 30 or 50, and that is precisely why funds work so hard on decorrelating their research rather than on producing more of it.

#### Worked example 1: two signals, and the \$3.8m that correlation costs

Two signals, each with IC 0.03, on the standard book. Effective breadth is 600, so ${\sqrt{\mathrm{BR}} = 24.5}$.

Run one signal alone: IR = 0.03 × 24.5 = 0.73, and expected gross alpha is 0.73 × \$20m = **\$14.7m** a year.

Add a second signal, uncorrelated with the first. The multiplier is ${\sqrt{2} = 1.41}$, so the combined IC is 0.0424. IR = 0.0424 × 24.5 = 1.04, and gross alpha is **\$20.8m**. The second signal earned \$6.1m.

Now suppose the two are correlated at 0.5, which is entirely normal for two price-based signals built by the same team on the same data. The multiplier is ${\sqrt{2}/\sqrt{1.5} = 1.15}$, so the combined IC is 0.0346. IR = 0.85, and gross alpha is **\$17.0m**. The same second signal now earns \$2.3m.

The identical piece of research is worth \$6.1m or \$2.3m depending on nothing about its own quality. The intuition: what you pay a researcher for is not the IC of their signal, it is the IC of their signal conditional on everything the book already knows.

#### Worked example 2: scaling to 100 signals at rho = 0.3

Same book, all signals at IC 0.03, average pairwise correlation 0.3. Walk the ladder:

| Signals | Multiplier | Combined IC | IR | Gross alpha | Added by this step |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.00 | 0.0300 | 0.73 | \$14.7m | |
| 10 | 1.64 | 0.0493 | 1.21 | \$24.2m | \$9.5m |
| 50 | 1.78 | 0.0535 | 1.31 | \$26.2m | \$2.1m |
| 100 | 1.80 | 0.0541 | 1.33 | \$26.5m | \$0.3m |
| infinite | 1.83 | 0.0548 | 1.34 | \$26.8m | \$0.3m |

The first nine additional signals bought \$9.5m a year. The next forty bought \$2.1m. The next fifty bought \$298,000, which will not cover the salary of the person who built them. And the entire remaining infinity of signals beyond the hundredth is worth another \$300,000.

In effective-breadth terms: 100 signals at rho = 0.3 give ${n_{\text{eff}} = 100/30.7 = 3.26}$ independent bets, against a hard ceiling of 3.33. Fifty signals already gave 3.18. **Doubling the library from 50 to 100 bought 0.08 of one independent signal.** The one-sentence intuition: past the knee of the curve you are not building a book, you are decorating one.

## Breadth counts independent bets, not signals and not tickers

The fundamental law is exact only under assumptions nobody's book satisfies, and the assumption that breaks first is that the BR forecasts are independent. Two things destroy independence, and both of them are usually ignored.

**The cross-section is not independent.** Five hundred stocks do not give 500 independent bets, because a market factor, a handful of sector factors and a size factor drive most of their covariance. Once you have neutralised those exposures, which you must, the residual cross-section supports far fewer independent bets than it has tickers. The eigenvalue structure of the return covariance matrix is the honest accounting here, and it is covered in [eigendecomposition and PCA on returns](/blog/trading/math-for-quants/eigendecomposition-pca-returns-math-for-quants).

**Time is not independent either.** If your signal is 70% autocorrelated week to week, then this week's bet is mostly last week's bet placed again. For an AR(1) process with autocorrelation phi, the effective number of independent observations in T periods is approximately

$$
T_{\text{eff}} = T \cdot \frac{1 - \phi}{1 + \phi}
$$

At phi = 0.7 and T = 52 weeks, that is 9.2 effective periods a year, not 52. A slow value signal with phi = 0.95 monthly gives 12 × 0.05 / 1.95 = 0.31 independent bets a year, which is why value strategies have such punishing multi-year drawdowns: they place roughly one bet every three years and then wait to find out if it was right.

![Descending funnel showing 500 names times 52 weekly rebalances collapsing through factor neutralisation and signal autocorrelation down to an effective breadth of 600, with the naive information ratio of 4.8 next to the honest 0.73.](/imgs/blogs/combining-weak-alphas-math-for-quants-3.webp)

#### Worked example 3: where the \$20m book's breadth actually comes from

Naive count: 500 names × 52 weekly rebalances = 26,000 bets a year. ${\sqrt{26{,}000} = 161}$. Multiply by IC 0.03 and the fundamental law promises an IR of 4.8, worth \$97m a year on a \$200m book. No fund on earth runs a 4.8 IR, which should be the first clue.

Honest count. Strip the market and sector factors and the residual cross-section behaves like roughly 65 independent names, not 500. Apply weekly autocorrelation of 0.7 and 52 rebalances become 9.2 effective periods. Effective breadth is 65 × 9.2 = 600, and ${\sqrt{600} = 24.5}$.

IR = 0.03 × 24.5 = **0.73**, worth **\$14.7m**. The naive breadth overstated the information ratio by a factor of 6.6.

There is a further leak that Clarke, de Silva and Thorley documented in *Portfolio Constraints and the Fundamental Law of Active Management* (Financial Analysts Journal, 2002). Even with the right IC and the right breadth, you do not get to hold the portfolio the signal implies. Long-only constraints, position limits, sector caps and turnover budgets all stand between the forecast and the position. They added a *transfer coefficient*, the correlation between the positions you wanted and the positions you actually hold:

$$
\mathrm{IR} \approx \mathrm{TC} \cdot \mathrm{IC} \cdot \sqrt{\mathrm{BR}}
$$

An unconstrained long-short book can approach a transfer coefficient of 1.0; constrained long-only mandates are commonly reported in the 0.3 to 0.8 range, and the coefficient falls further as tracking-error targets rise, because the long-only constraint binds on more names. A TC of 0.5 halves everything computed above. The intuition: skill you cannot express is not skill.

## Choosing the weights

Three schemes account for almost everything done in practice.

**Equal weight.** Every signal gets ${1/n}$. It ignores that some signals are better than others, and it double-counts clusters of near-duplicates.

**IC weight.** Weight each signal by its measured IC. Better in principle: the stronger signal should count for more. Optimal only when the signals are uncorrelated.

**Covariance-aware.** With Σ the correlation matrix of the signals and **c** the vector of ICs, the weights that maximise combined IC are

$$
\mathbf{w}^{*} \propto \Sigma^{-1}\mathbf{c}
$$

This is the same object as a mean-variance portfolio, with signals in place of assets, and it inherits every one of that problem's pathologies. See [the mean-variance efficient frontier](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants) for the geometry and [robust and regularised portfolios](/blog/trading/math-for-quants/robust-regularized-portfolios-math-for-quants) for the fixes.

| Scheme | Weights | When it wins | How it fails |
| --- | --- | --- | --- |
| Equal weight | ${1/n}$ each | short history, similar ICs, correlations you cannot measure | ignores real quality differences, double-counts duplicate clusters |
| IC weight | proportional to each IC | signals close to uncorrelated | assumes away the correlation structure entirely |
| Covariance-aware | proportional to Σ inverse times the IC vector | long history, few signals, stable correlations | inverts a matrix that is mostly estimation noise |
| Cluster then equal weight | equal within theme, equal across themes | many signals in obvious families | needs the clustering to be right |
| Shrunk covariance | as above, with Σ pulled toward a target | the realistic middle | one more hyperparameter to overfit |

### The hurdle a new signal has to clear

Solve the two-signal case of ${\Sigma^{-1}\mathbf{c}}$ and something sharp falls out. Signal B gets a positive weight if and only if

$$
c_B > \rho \cdot c_A
$$

That is the whole rule, and it generalises: a candidate signal earns a positive weight in an existing book only if its IC exceeds its correlation with the book times the book's combined IC. Below the hurdle its optimal weight is *negative*. It still contributes, but as a hedge that strips a piece of noise out of the book, not as a source of return.

The hurdle rises as the book improves, which is the cruel part. Our 100-signal book has a combined IC of 0.0541. A candidate with IC 0.03 needs a correlation with the book below 0.03 / 0.0541 = 0.554 to earn a positive weight. A typical new signal that correlates 0.3 with each existing member correlates 0.54 with their average. It clears the hurdle by about two percent.

![Two bars comparing signal A at IC 0.05 with signal B at IC 0.03, showing 49% of B's variance already contained in A, above a rule box giving the hurdle test and the resulting negative optimal weight of minus 0.21.](/imgs/blogs/combining-weak-alphas-math-for-quants-2.webp)

#### Worked example 4: the \$3.2m you lose by blending

Signal A has IC 0.05. Signal B has IC 0.03. They correlate at 0.7. Same book, so combined IC × 490 gives gross alpha in millions of dollars.

Check the hurdle first: 0.70 × 0.05 = 0.035, and B's IC of 0.030 is below it. B fails.

Now compute all four options:

| Weighting | Weights (A, B) | Combined IC | Gross alpha |
| --- | --- | --- | --- |
| Signal A alone | (1.00, 0.00) | 0.0500 | \$24.5m |
| Equal weight | (0.50, 0.50) | 0.0434 | \$21.3m |
| IC weight | (0.63, 0.38) | 0.0458 | \$22.5m |
| Covariance-aware | (1.21, -0.21) | 0.0505 | \$24.7m |

Both naive blends are **worse than simply not using signal B at all**. Equal weighting costs \$3.2m a year against running A alone; IC weighting costs \$2.0m. Only the covariance-aware solution beats A, and it beats it by \$240,000, which is one percent, on the strength of a correlation estimate that is nowhere near one percent accurate.

Push on that. Suppose the true correlation is 0.6 rather than the 0.7 you estimated. At rho = 0.6 signal B is exactly redundant, since 0.6 × 0.05 = 0.03 is precisely B's IC, and the true optimal weight on B is zero. Your mis-estimated weights of (1.21, -0.21) then deliver an IC of 0.0494 against the 0.0500 available from A alone. A 0.1 error in a single correlation flipped the sophisticated answer from a small winner into a small loser. The intuition: at high correlation the blend is worth so little that estimation error swamps the prize.

### Why the covariance is mostly noise

The covariance-aware weights require inverting an ${n \times n}$ correlation matrix estimated from the same historical panel that produced the ICs. That matrix is far less trustworthy than it looks.

Random matrix theory gives the sharp version. With n signals and T observations, set ${q = n/T}$. Even if the true correlation matrix is the identity, meaning every signal is genuinely independent, the sample eigenvalues spread across the Marchenko-Pastur interval from ${(1-\sqrt{q})^2}$ to ${(1+\sqrt{q})^2}$. With 100 signals and 1,000 daily observations, q = 0.1 and that interval runs from 0.47 to 1.73. The smallest sample eigenvalue is 0.47 when the truth is 1.0, and matrix inversion divides by it. Pure sampling noise gets amplified by more than a factor of two and then handed a portfolio weight. [Random matrix theory and covariance cleaning](/blog/trading/math-for-quants/random-matrix-theory-covariance-cleaning-math-for-quants) is the whole story, and it is the reason the sophisticated scheme so often loses.

The empirical verdict is equally blunt. DeMiguel, Garlappi and Uppal (*Review of Financial Studies*, 2009) tested fourteen optimising strategies against naive equal weighting across seven datasets and found none consistently better on Sharpe ratio, certainty equivalent or turnover. Their calibration is the number to carry into an interview: for a 25-asset problem the sample-based mean-variance rule needs roughly **3,000 months** of data before it reliably beats equal weighting, and for 50 assets roughly **6,000 months**. Five hundred years of monthly data. You have twenty.

The practical resolution is not to choose a side but to know where you sit on the ${n/T}$ axis. With 5 signals and 10 years of daily data, optimise. With 100 signals and 4 years, shrink hard or cluster your signals into five or six themes, equal-weight within each theme and equal-weight across themes. The clustered version encodes the only part of the correlation structure you can actually estimate, which is the block structure, and throws away the part you cannot, which is the individual off-diagonal entries.

## Naive stacking double-counts, and there are only two fixes

Adding correlated signals does not add information, it adds emphasis. The names where two overlapping signals agree get double the weight, and the combined signal quietly becomes a proxy for whatever the two share rather than a balanced view.

**Fix one: orthogonalise.** Regress the new signal on the existing book and keep the residual. The residual is uncorrelated with the book by construction, so the combination adds only what is genuinely new. The mechanics are worked through in the [alpha signal pipeline](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) post. The catch is that orthogonalisation is not free. The regression coefficients are estimated, so the residual carries the regression's estimation error on top of the signal's own noise, and residualising a weak signal against a book of 100 correlated regressors can leave you with something that is almost entirely noise. Orthogonalise against a small number of themes, not against every signal individually.

**Fix two: accept the shrinkage.** Do nothing clever, equal-weight, and accept that the combined IC is ${\sqrt{n}/\sqrt{1+(n-1)\rho}}$ rather than ${\sqrt{n}}$. You have not solved the double-counting, you have priced it. Given what the section above says about estimation error, this is the right answer more often than the profession likes to admit, and it is what "1/N is hard to beat" means in this context.

What is never a fix is throwing the signals at a machine-learning model and hoping it sorts out the redundancy. A model fitted on the same panel inherits exactly the same estimation problem, with more parameters and a better story. The multiple-testing discipline in [overfitting, purged cross-validation and the deflated Sharpe ratio](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research) is the relevant defence.

## Turnover and cost: what actually kills a good combination

Everything so far has been gross. Now add the bill.

Gross alpha saturates along the curve we derived. Trading cost does not saturate, because each additional signal disagrees with the current book about something, and the disagreement gets traded. To a first approximation, cost grows linearly in the number of signals while gross alpha grows like a square root that is already flattening. A linear cost against a saturating benefit has one shape: it rises, rolls over, and declines.

![Chart of gross alpha, trading cost and net alpha in millions of dollars per year against the number of signals, with net alpha peaking near 25 signals and declining thereafter.](/imgs/blogs/combining-weak-alphas-math-for-quants-4.webp)

#### Worked example 5: net alpha peaks at 25 signals, not 100

Assume each signal past the first adds 20 percentage points of annual two-way turnover, and that all-in trading cost, meaning spread plus market impact plus fees, runs 10 basis points of every dollar traded. On the \$200m book that is \$40,000 of extra cost per signal per year. Combine that with the gross curve from worked example 2:

| Signals | Gross alpha | Trading cost | Net alpha |
| --- | --- | --- | --- |
| 1 | \$14.7m | \$0.2m | \$14.5m |
| 10 | \$24.2m | \$0.6m | \$23.6m |
| 25 | \$25.7m | \$1.2m | **\$24.5m** |
| 50 | \$26.2m | \$2.2m | \$24.1m |
| 100 | \$26.5m | \$4.2m | \$22.4m |

Net alpha peaks near 25 signals. The 100-signal book, which cost four times the research effort, delivers **\$2.1m a year less** than the 25-signal book.

The marginal view is even starker. The eleventh signal adds \$211,000 of gross alpha against \$40,000 of cost, so it is worth roughly \$171,000 a year and is an easy yes. The twenty-sixth adds \$42,000 of gross against the same \$40,000 of cost and is a coin flip. The hundredth adds about \$3,400 of gross against \$40,000 of cost and **loses roughly \$37,000 a year**. The one-sentence intuition: the question is never "is this signal good" but "does this signal clear its own trading bill on top of what the book already knows".

Two mitigations exist and both matter. Netting helps: signals that disagree partly cancel before any order is sent, so the linear cost assumption is pessimistic for a well-implemented book. And explicit cost-aware combination, where the optimiser is handed the trading cost and allowed to shade weights toward slower signals, pushes the peak to the right. Neither changes the shape.

## Common misconceptions

**"More signals is always better."** Only if they are independent and free to trade. Neither holds. The gross curve saturates at ${1/\sqrt{\rho}}$ and the net curve turns down. On the book above, going from 25 signals to 100 destroys \$2.1m a year.

**"IC weighting is optimal."** It is optimal exactly when the signal correlation matrix is the identity. The moment signals are correlated, the optimum is ${\Sigma^{-1}\mathbf{c}}$, not **c**, and the two can disagree about the *sign* of a weight. In worked example 4, IC weighting put 38% on a signal whose optimal weight was negative.

**"Breadth is the number of stocks."** Breadth is the number of independent forecasts per year. Five hundred names and 52 rebalances gave 26,000 raw bets and 600 real ones in worked example 3, a 6.6-times overstatement of the information ratio. Any candidate who computes breadth as names times periods without deducting factor structure and signal autocorrelation is telling you they have never had to defend an IR to a risk committee.

**"Full covariance optimisation is the rigorous answer."** It is the rigorous answer to a problem whose inputs you do not have. With 100 signals you are inverting a matrix whose smallest eigenvalues are pure sampling noise, estimated on the same data that produced the ICs. The rigorous answer is to shrink it, cluster it, or admit you cannot estimate it.

**"Orthogonalising removes the problem."** It removes the double-counting and adds estimation error in exchange. Residualising a 0.02-IC signal against 100 correlated regressors typically leaves noise with a plausible-looking name.

**"A signal with a positive IC always adds value."** Only above the hurdle. Below ${\rho \cdot c_{\text{book}}}$ its optimal weight is negative, and a researcher who ships it as a long alpha is adding an expensive way to be more of what the book already is.

## In the interview room and on the desk

The question arrives in one of two shapes. WorldQuant tends to ask it directly: *"You have 100 alphas. How do you combine them?"* Citadel tends to embed it: *"Your book runs 40 signals and the PM wants to add 15 more from a new team. How do you decide?"*

The weak answer is a list of weighting schemes. Equal weight, IC weight, inverse-variance, mean-variance, maybe a gradient-boosted model. It is a list of things you have heard of, and it answers a question nobody asked.

The strong answer starts one level earlier, with the correlation structure. In order:

1. **Ask what the correlations are**, before proposing any scheme. State the arithmetic: combined IC scales like ${\sqrt{n}/\sqrt{1+(n-1)\rho}}$ with a ceiling at ${1/\sqrt{\rho}}$, so the answer to "how do I combine 100 alphas" depends almost entirely on one number you have not been told yet.
2. **Give the saturation numbers.** At rho = 0.3, ten signals capture 90% of the ceiling and the 100 signals contain 3.3 effective independent bets. That single sentence does more than any weighting scheme you could name.
3. **Then talk weights**, and tie the choice to ${n/T}$. With 100 signals and a few years of data you cannot estimate a 100-by-100 correlation matrix, so cluster into themes, equal-weight within, and shrink across. Cite DeMiguel, Garlappi and Uppal if the interviewer pushes.
4. **Finish on cost.** Marginal gross alpha per signal against marginal trading cost per signal, and note that the net curve peaks well before the gross curve flattens.

The trap is proposing full covariance optimisation on 100 signals without mentioning that the covariance came out of the same data as the ICs and is mostly noise. It reads as rigour and it is the exact error that random matrix theory exists to name. The candidate who says "I would invert the signal covariance" and stops has failed the question; the one who says "I would want to, but with 100 signals and four years of data the smallest eigenvalues are sampling artefacts, so I would shrink toward a block structure" has passed it.

The second trap is the follow-up: *"Your new signal has an IC of 0.02 and correlates 0.6 with the book. Do you take it?"* Compute the hurdle out loud. If the book's combined IC is 0.05, the hurdle is 0.6 × 0.05 = 0.03, the signal is below it, and the honest answer is that it belongs in the book with a negative weight as a hedge, or not at all. Say that and the conversation changes.

WorldQuant weights this most heavily, since industrialised alpha production is the firm's entire operating model, and Citadel's multi-strategy portfolio-construction seats weight it nearly as much. Two Sigma will approach it from the estimation-error side and ask about the covariance before the weights.

## Where this sits and what to read next

The uncomfortable summary is that a signal library has a capacity, that capacity is roughly ${1/\rho}$ independent bets, and most research effort at most funds goes into filling a library whose capacity was reached years ago. The lever with real leverage is not the count of signals. It is rho: new data, new horizons, new universes, anything that lowers the average correlation of the next signal to the existing book. Halving rho from 0.3 to 0.15 raises the ceiling from 1.83 to 2.58, worth about \$11m a year on the standard book. No amount of stacking does that.

Three directions from here. For the covariance machinery underneath the weights, [the covariance matrix through linear algebra](/blog/trading/math-for-quants/covariance-matrix-linear-algebra-math-for-quants) and then [random matrix theory and covariance cleaning](/blog/trading/math-for-quants/random-matrix-theory-covariance-cleaning-math-for-quants). For whether the ICs you are combining are real at all, [overfitting, purged cross-validation and the deflated Sharpe ratio](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research). For the estimator theory that explains why a sample correlation from a short panel is biased in a direction that flatters your combination, [estimators, bias, variance and consistency](/blog/trading/math-for-quants/estimators-bias-variance-consistency-math-for-quants).

This is educational material about mechanism, not investment advice, and every dollar figure above comes from the stated hypothetical book rather than any live portfolio.

## Sources and further reading

- Richard Grinold and Ronald Kahn, *Active Portfolio Management*, 2nd edition, McGraw-Hill, 2000. The fundamental law and the breadth accounting.
- Roger Clarke, Harindra de Silva and Steven Thorley, ["Portfolio Constraints and the Fundamental Law of Active Management"](https://www.tandfonline.com/doi/abs/10.2469/faj.v58.n5.2468), *Financial Analysts Journal* 58(5), 2002. The transfer coefficient.
- Victor DeMiguel, Lorenzo Garlappi and Raman Uppal, ["Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy?"](https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901), *Review of Financial Studies* 22(5), 2009, pages 1915 to 1953. The 3,000-month and 6,000-month calibrations.
- Zura Kakushadze, ["101 Formulaic Alphas"](https://arxiv.org/abs/1601.00991), 2015. The WorldQuant-style alpha catalogue that makes the quantity-over-independence assumption so tempting.
