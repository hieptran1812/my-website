---
title: "Hierarchical Bayes: borrowing strength across strategies"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "Twenty strategies with short track records are neither twenty separate problems nor one. A hierarchical model puts them on a shared population and pulls each estimate toward it by an amount the data chooses rather than one you assume."
tags: ["hierarchical-bayes", "partial-pooling", "shrinkage", "empirical-bayes", "sharpe-ratio", "capital-allocation", "james-stein", "bayesian-statistics", "multi-strategy", "quant-interview", "two-sigma", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 18
---

> [!important]
> **TL;DR:** Judging twenty strategies in isolation throws away what they have in common. Judging them as identical throws away their differences. A hierarchical model sits between, and the data picks the spot.
>
> - Complete pooling, no pooling and partial pooling are not three rival methods. The first two are the limits of the third.
> - The shrinkage factor is the share of total variance that is sampling noise, $B_i = \sigma_i^2/(\sigma_i^2 + \tau^2)$. A noisy strategy shrinks hard, a long record barely moves.
> - This is James-Stein with the shrinkage **estimated from the data** instead of plugged in. Same object, one fewer assumption.
> - Across five strategies with 1 to 8 years of record, a raw 2.40 Sharpe on one year becomes 1.55, and the winner of the raw ranking loses it.
> - On a \$500m book that moves \$53.8m off the shortest record, worth about \$5.2m a year at matched risk.

## Twenty numbers and one meeting

It is January. Twenty portfolio managers at a multi-strategy fund each hand you a track record. Some have been running six years, some started last March. You have one column of Sharpe ratios and one afternoon to decide who gets capital.

The tempting move is to sort the column. The top name shows 2.4 and the bottom 0.5, which looks like a four-to-one difference in skill. It is almost never that. It is usually a difference in how long each has been running, because the shortest records produce both the best-looking and the worst-looking numbers. The opposite move, refusing to distinguish them at all because they are all at one firm hired by one process, is wrong for the obvious reason: some of these strategies really are better.

![Three panels on one Sharpe scale: complete pooling with five strategies at 1.34, no pooling with raw values 0.50 to 2.40, partial pooling with shrunk values 0.78 to 1.58 around a dashed mean](/imgs/blogs/hierarchical-bayes-pooling-math-for-quants-1.webp)

That figure is the whole post. The left panel is one answer, the middle the other, and the right is what a hierarchical model does: it keeps the differences but squeezes them, by an amount you estimate rather than choose.

## Foundations: the three positions, stated exactly

Write $\theta_i$ for the quantity you care about, the true long-run Sharpe of strategy $i$, the number it would converge to given infinite data. Write $y_i$ for what you observe, the Sharpe measured over the record you have. Conflating the two is the original sin of the sorted spreadsheet.

**Complete pooling** says every group shares one parameter, $\theta_1 = \dots = \theta_k = \mu$, estimated once from all the data and handed to everybody. Maximally stable, because every observation estimates one thing, and biased for any group that genuinely differs.

**No pooling** says each group gets its own parameter with no relationship to the others, $\hat\theta_i = y_i$. Unbiased, and for short records catastrophically noisy.

**Partial pooling** says the $\theta_i$ are different but *related*, specifically draws from a common population:

$$
\theta_i \sim N(\mu, \tau^2)
$$

That single line is the hierarchy. A strategy at this firm is not an arbitrary point on the real line, it is a sample from a distribution of strategies at this firm, centred at $\mu$ with spread $\tau$. Both are unknown and get estimated too, which is what makes the model hierarchical rather than merely Bayesian: the parameters have parameters.

Three definitions. **Sharpe ratio** is annualised excess return divided by annualised volatility, so a Sharpe of 1.0 means one unit of return per unit of risk per year. **Volatility** is the annualised standard deviation of returns. **Borrowing strength** is what the hierarchy does: group $i$'s estimate improves because groups $j \ne i$ exist, even though they trade different things.

## The model, and the formula that falls out of it

Two lines, the data given the truth and the truth given the population:

$$
y_i \mid \theta_i \sim N(\theta_i, \sigma_i^2), \qquad \theta_i \sim N(\mu, \tau^2)
$$

$\sigma_i$ is the standard error of strategy $i$'s measured Sharpe, known from the length of its record. $\tau$ is the standard deviation of true Sharpes across strategies, which you must estimate. These are the two sources of spread in the spreadsheet: $\sigma_i^2$ is **within-group** variation, the noise in one measurement, and $\tau^2$ is **between-group** variation, real differences in skill.

Because both lines are normal the posterior is closed form, and the posterior mean of $\theta_i$ is a precision-weighted average of what this group said and what the population said:

$$
\tilde\theta_i = y_i + B_i\,(\mu - y_i), \qquad B_i = \frac{\sigma_i^2}{\sigma_i^2 + \tau^2}
$$

$B_i$ is the **shrinkage factor**, the fraction of the distance from your raw number to the population mean that gets taken back. It is the share of total variance that is measurement noise.

Read the limits off that formula and the three positions collapse into one. If $\tau^2 = 0$, every $B_i = 1$ and every estimate becomes $\mu$: complete pooling. If $\tau^2 \to \infty$, every $B_i \to 0$ and every estimate becomes $y_i$: no pooling. They are the endpoints of one dial, and the hierarchy's job is to read where the data has set it.

## Where the shrinkage factor comes from

You need $\sigma_i$. For a Sharpe estimated over $T$ years the standard error is approximately ${1/\sqrt{T}}$. Lo (2002) gives the exact expression, which adds a correction depending on the Sharpe and the sampling frequency, small enough on daily data to ignore in a first pass. So $\sigma_i^2 = 1/T_i$: one year of record gives a standard error of a full 1.0 of Sharpe, four years gives 0.5, sixteen years gives 0.25. A strategy with one year of data and a measured Sharpe of 2.4 therefore carries a one-standard-error band from 1.4 to 3.4.

Substitute $\sigma_i^2 = 1/T_i$ into the shrinkage factor and something clean appears:

$$
B_i = \frac{1/T_i}{1/T_i + \tau^2} = \frac{n_0}{n_0 + T_i}, \qquad n_0 = \frac{1}{\tau^2}
$$

![Shrinkage factor B against years of track record, falling from 0.80 at one year to 0.29 at ten, five strategies marked, dashed guides meeting where four years gives one half](/imgs/blogs/hierarchical-bayes-pooling-math-for-quants-2.webp)

The population acts exactly like $n_0$ years of prior track record that every strategy already has in the bank. If the population spread is $\tau = 0.5$ then $n_0 = 4$, and the prior is worth four years of data. A strategy with four years of its own record is shrunk exactly halfway, because it has brought as much evidence as the population supplies; one year is shrunk 80% of the way, because the population's four years outweigh its one by four to one.

That is why "borrowing strength" is the right phrase. The other strategies are lending you sample size.

## This is James-Stein with the shrinkage estimated

If $\tilde\theta_i = y_i + B(\mu - y_i)$ looks familiar, it should: it is the [James-Stein estimator](/blog/trading/math-for-quants/shrinkage-stein-paradox-math-for-quants), which that post derives. Hierarchical pooling *is* that estimator with the shrinkage factor estimated from the data rather than plugged in.

That is the whole difference, and it buys two things. You stop having to assert how much to shrink, because the population you fit implies it. And you get a *different* factor per group, so eight years of record is not pulled as hard as eight months, which a single global constant cannot express and an allocation committee needs.

Estimating $\mu$ and $\tau$ from the data and then treating them as known is **empirical Bayes**: a short cut that understates uncertainty, and what Efron and Morris used on the example that made the idea famous.

## Worked example 1: five strategies and one population

#### Worked example: five Sharpe ratios on a \$500m book

Every number below is illustrative arithmetic on assumed inputs.

| Strategy | Record $T$ | Raw Sharpe $y_i$ | $\sigma_i^2 = 1/T_i$ |
| --- | --- | --- | --- |
| A, stat arb | 1 year | 2.40 | 1.000 |
| B, index vol | 3 years | 1.90 | 0.333 |
| C, macro | 5 years | 1.15 | 0.200 |
| D, credit relative value | 4 years | 0.75 | 0.250 |
| E, commodity trend | 8 years | 0.50 | 0.125 |

**Step 1, the population mean.** Average the five: ${(2.40 + 1.90 + 1.15 + 0.75 + 0.50)/5 = 6.70/5 = 1.34}$, so $\hat\mu = 1.34$.

**Step 2, the population spread.** This is the step people skip and the one doing the work. The spread you *see* is the real spread plus the measurement noise, so subtract the noise off. The sample variance of the five raw Sharpes is 0.632, and the average sampling variance is ${(1.000 + 0.333 + 0.200 + 0.250 + 0.125)/5 = 1.908/5 = 0.382}$. Therefore

$$
\hat\tau^2 = 0.632 - 0.382 = 0.250, \qquad \hat\tau = 0.50
$$

Of the total 0.632 of spread in the spreadsheet, 0.382 is noise, about 60%; only 0.250 is real differences between strategies. And $\hat\tau^2 = 0.25$ means $n_0 = 4$, which is where the four-year figure came from. Had the subtraction come out negative, which happens often with few groups and short records, the estimate is $\hat\tau^2 = 0$ and the honest answer is complete pooling: this data cannot tell these strategies apart.

**Step 3, shrink**, with $B_i = 4/(4 + T_i)$:

| Strategy | $B_i = 4/(4 + T_i)$ | $\tilde\theta_i = y_i + B_i(1.34 - y_i)$ |
| --- | --- | --- |
| A, 1 year | $\tfrac{4}{5} = 0.80$ | $2.40 - \tfrac{4}{5}(1.06) = 1.552$ |
| B, 3 years | $\tfrac{4}{7} = 0.571$ | $1.90 - \tfrac{4}{7}(0.56) = 1.580$ |
| C, 5 years | $\tfrac{4}{9} = 0.444$ | $1.15 + \tfrac{4}{9}(0.19) = 1.234$ |
| D, 4 years | $\tfrac{1}{2} = 0.500$ | $0.75 + \tfrac{1}{2}(0.59) = 1.045$ |
| E, 8 years | $\tfrac{1}{3} = 0.333$ | $0.50 + \tfrac{1}{3}(0.84) = 0.780$ |

Those shrunk values are exact, and the figures round them to two places.

The raw column spans 1.90 of Sharpe, the shrunk column 0.80. Most of the apparent dispersion was never there.

One refinement, so it is not a hidden fudge: the proper hierarchical estimate of $\mu$ weights each group by $1/(\sigma_i^2 + \tau^2)$ rather than equally, which pulls $\hat\mu$ to 1.12, drops every shrunk number by roughly 0.1 and leaves the ordering unchanged. The equal-weight version is the one you can do at a whiteboard.

## Worked example 2: the ranking flips, and why it usually does

#### Worked example: who wins the \$500m ranking

Sort the raw column and A wins with 2.40, half a Sharpe clear. Sort the shrunk column and B wins, 1.58 against A's 1.55.

![Slope chart from raw Sharpe to shrunk Sharpe, A falling steeply 2.40 to 1.55 and B gently 1.90 to 1.58 so B ends above A, dashed population mean reference](/imgs/blogs/hierarchical-bayes-pooling-math-for-quants-3.webp)

Nothing about B improved. B had three years of record against A's one, so B kept 43% of its distance from the mean while A kept 20%.

This is not a quirk of these five numbers, it is the normal case, and the reason is a sampling fact rather than a Bayesian one: the maximum of a set of noisy estimates is biased upward, and the noisiest estimate has the fattest upper tail, so a raw ranking is won disproportionately by whoever has the least data. Suppose all five truly had a Sharpe of 1.34 and differed only in record length. Posting 2.40 puts A just ${(2.40 - 1.34)/1.0 = 1.06}$ standard errors out, which happens 14.5% of the time, better than one year in seven. E would need to be 3.0 standard errors out, which happens 0.14% of the time, about one year in seven hundred. The short record is more than a hundred times likelier to produce that headline by luck alone.

A ranking that does not correct for this does not rank skill. It ranks skill times noise, and noise wins when the records are short.

## Worked example 3: what it costs in dollars

#### Worked example: allocating a \$500m book off the wrong column

Each sleeve runs to a 10% annualised volatility target on the capital it is given, and the sleeves are treated as uncorrelated, which is generous. Under those assumptions the allocation maximising the book's Sharpe is capital in proportion to each sleeve's Sharpe.

| Strategy | Raw allocation | Pooled allocation | Change |
| --- | --- | --- | --- |
| A, 1 year | \$179.1m | \$125.3m | **-\$53.8m** |
| B, 3 years | \$141.8m | \$127.6m | -\$14.2m |
| C, 5 years | \$85.8m | \$99.7m | +\$13.9m |
| D, 4 years | \$56.0m | \$84.4m | +\$28.4m |
| E, 8 years | \$37.3m | \$63.0m | +\$25.7m |

![Grouped bars of capital allocated in millions, raw Sharpe allocation against pooled, A and B losing capital while C, D and E gain, each pair labelled with its change](/imgs/blogs/hierarchical-bayes-pooling-math-for-quants-4.webp)

\$53.8m moves off the strategy nobody has watched through a full cycle, and \$68m in total moves from the two shortest records to the three longest.

Now price the error. Taking the shrunk numbers as the better forecast, the raw-column book has a portfolio Sharpe of 2.75 and the pooled-column book 2.85, a gap of 0.10. A 10% volatility target on \$500m is \$50m of annualised volatility, so 0.10 of Sharpe is \$5.2m a year, every year, for the cost of one extra spreadsheet column.

That prices the error *conditional on* the pooled estimates being the better forecast; the evidence for that is the Stein result and the out-of-sample record below, not this table. The table shows the size of the bet you make by allocating off raw numbers.

## Fitting it: closed form, and when you need a sampler

The normal-normal model above is **conjugate**: the posterior belongs to the same family as the prior, so you can write the answer down, which is why every number here came out of two lines of arithmetic. Conjugacy survives a few variations, such as a beta population over hit rates or a gamma population over event intensities.

It stops as soon as you want anything real: a $t$ population, a prior on $\tau$ instead of a point estimate, correlated strategies, or hyperparameters depending on covariates. The integral then has no closed form and you reach for [MCMC](/blog/trading/math-for-quants/mcmc-metropolis-gibbs-math-for-quants), which also carries the uncertainty in $\hat\tau$ that empirical Bayes throws away. One warning: near $\tau = 0$ the posterior is a funnel that Gibbs and vanilla Hamiltonian Monte Carlo traverse badly, and the fix is the **non-centred parameterisation**, $\theta_i = \mu + \tau z_i$ with $z_i \sim N(0,1)$ (Betancourt and Girolami, 2015).

## The failure mode: when the population does not fit

The hierarchy buys its variance reduction by asserting the groups are exchangeable draws from one distribution. When that is wrong the cost lands on the extremes, which is where a genuinely exceptional strategy lives.

Suppose one sleeve really does have a true Sharpe of 3.0, in a population estimated at $\mu = 1.34$ with $\tau = 0.50$. That is 3.3 population standard deviations out, which a normal says essentially cannot happen. With one year of record it is shrunk to $3.0 - 0.80 \times 1.66 = 1.67$, so the model takes a genuine outlier and reports it as slightly above average. No amount of data on the *other* strategies fixes this, because they created the problem.

**Stratify before you pool.** Exchangeability is a judgement, not a statistic. A crypto market maker and a bond relative-value desk are not plausibly draws from one population, so split into groups that belong together and run a hierarchy inside each.

**Give the population fat tails.** Replace $\theta_i \sim N(\mu, \tau^2)$ with a $t$ distribution on few degrees of freedom, or a two-component mixture. A heavy-tailed population assigns real prior mass to outliers, so an extreme group shrinks far less. This costs conjugacy and puts you on MCMC.

**Put the structure in the mean.** Instead of one $\mu$, model $\mu_i = \beta' x_i$ with covariates: strategy type, asset class, capacity, holding period. Each strategy is then shrunk toward what similar strategies do rather than toward the firm-wide average, which is more accurate and far easier to defend in a meeting. This is hierarchical regression, and it is where most of the practical value sits.

The diagnostic for all three is the same: compare the observed spread of the $y_i$ with what the fitted model predicts. If your tails are consistently fatter, the population is wrong, and your extreme groups are the estimates not to trust.

## Common misconceptions

**"A hierarchical model has more parameters, so it must overfit."** It has more parameters written down and fewer parameters spent. The effective number is $\sum_i (1 - B_i)$, the independent freedom the group means actually use, and for the five strategies above that is ${0.200 + 0.429 + 0.556 + 0.500 + 0.667 = 2.35}$, not 5. The hierarchy is a *regulariser*: complete pooling spends 1 parameter, no pooling spends $k$, partial pooling spends what the data licenses. Hodges and Sargent (2001) made the count precise.

**"Partial pooling is a compromise between two right answers."** It is the answer, and the other two are its boundary cases. If the data say the strategies are indistinguishable the model returns complete pooling on its own; if they say the strategies differ wildly it returns the raw numbers. You do not choose between three methods, you fit one and read off $\hat\tau$.

**"You need a lot of groups."** Efron and Morris used 18 batters, the canonical eight-schools example uses 8, the example above uses 5. What is true is that with few groups $\tau$ is poorly identified, and empirical Bayes is then fragile because it treats a badly estimated $\hat\tau$ as certain. The fix is a weakly informative prior on $\tau$ and a sampler, not abandoning the model; Gelman (2006) recommends a half-normal or half-Cauchy for exactly this case.

## How it shows up in real markets

The founding example is not from finance. Efron and Morris (1975, 1977) took the batting averages of 18 major-league players after their first 45 at-bats of the 1970 season and asked which estimator best predicted each player's average over the rest of that year. The raw averages ran from .400 down to .156. The shrunk estimates, pulled toward the group mean of .265, were closer to the truth for all but a handful of the eighteen and cut total squared error by roughly a factor of three.

The instructive case is the one it got wrong. Roberto Clemente hit .400 over those 45 at-bats and was shrunk to .290; he finished at .346, so the raw number was marginally closer. Clemente was a genuine outlier in a population modelled as normal, the failure mode above appearing in the paper that introduced the method.

In finance the direct descendant is Jorion (1986), which applies Bayes-Stein shrinkage to the expected-return vector in a mean-variance optimisation and shows out-of-sample improvement. Expected returns are the input a [mean-variance optimiser](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants) is most sensitive to and the one estimated worst, so an optimiser fed raw historical means reliably builds a portfolio of the luckiest assets. The same logic applied to the covariance matrix is [random matrix theory and covariance cleaning](/blog/trading/math-for-quants/random-matrix-theory-covariance-cleaning-math-for-quants).

On a desk the model fits anywhere you have many short records of the same kind of thing: PM Sharpes, per-sector alpha decay, per-venue fill quality, the information coefficient of each signal in a [library of weak alphas](/blog/trading/math-for-quants/combining-weak-alphas-math-for-quants). In each case the pooled estimate for a thinly observed group beats its own average and the global average alike.

## Sources and further reading

- Efron, B. and Morris, C. (1975). "Data Analysis Using Stein's Estimator and Its Generalizations." *JASA* 70(350), 311 to 319.
- Efron, B. and Morris, C. (1977). "Stein's Paradox in Statistics." *Scientific American* 236(5), 119 to 127.
- Gelman, A., Carlin, J., Stern, H., Dunson, D., Vehtari, A. and Rubin, D. (2013). *Bayesian Data Analysis*, 3rd edition, chapter 5.
- Gelman, A. and Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*.
- Gelman, A. (2006). "Prior Distributions for Variance Parameters in Hierarchical Models." *Bayesian Analysis* 1(3), 515 to 534.
- Jorion, P. (1986). "Bayes-Stein Estimation for Portfolio Analysis." *JFQA* 21(3), 279 to 292.
- Lo, A. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal* 58(4), 36 to 52.
- Hodges, J. and Sargent, D. (2001). "Counting Degrees of Freedom in Hierarchical and Other Richly-Parameterised Models." *Biometrika* 88(2), 367 to 379.
- Betancourt, M. and Girolami, M. (2015). "Hamiltonian Monte Carlo for Hierarchical Models." arXiv:1312.0906.
- Harvey, C., Liu, Y. and Zhu, H. (2016). "... and the Cross-Section of Expected Returns." *RFS* 29(1), 5 to 68. Why the top of a raw ranking is mostly selection.

The five-strategy book, its Sharpes, its record lengths and every dollar figure derived from them are illustrative arithmetic on assumed inputs. Empirical results attributed to published work are the ones those papers report.

## In the interview room and on the desk

The question is usually posed as a scenario rather than a maths problem. "You have twenty PMs and one year of returns each. Who gets capital?" Two Sigma and Citadel's multi-strategy side ask some version of it constantly, as does any allocator seat; it appears at Jane Street as a smaller question about which of several short-lived signals to size up.

The weak answer ranks by Sharpe and allocates down the list. It is not wrong about the data, it is wrong about what the data is. The strong answer takes the ranking apart in a fixed order.

First, say what you are estimating: each PM's true Sharpe, not their measured one. Second, put a standard error on it, roughly ${1/\sqrt{T}}$ in years, so one year carries a standard error of 1.0 and the whole ranking sits inside its own noise band. Third, decompose the observed spread: the variance of the twenty numbers is real skill dispersion plus measurement noise, and subtracting the average sampling variance leaves the real part. If that goes negative, say so, because it means the data cannot distinguish these PMs and the defensible allocation is near equal weight. Fourth, shrink each PM toward the pooled mean by $B_i = \sigma_i^2/(\sigma_i^2 + \tau^2)$, noting out loud that this is James-Stein with the constant estimated rather than assumed. Fifth, allocate off the shrunk numbers and say what changed: who lost the top slot, how much capital moved, and why the mover had the shortest record.

The trap is the answer that sounds most rigorous. A candidate who computes a t-statistic for every PM, ranks by it and allocates to the top decile has done real work and walked into the same hole, because ranking by a noisy statistic and then taking the maximum is precisely the operation that finds the luckiest rather than the best. The tell that you understand this is volunteering the direction of the error before you are asked: the top of a raw ranking is biased upward, the bias is largest for the shortest records, and the fix is to pool, not to demand a higher cutoff.

One more line lands well. Partial pooling is not caution. It is the estimate with the lower expected error, and refusing to shrink is the aggressive choice.
