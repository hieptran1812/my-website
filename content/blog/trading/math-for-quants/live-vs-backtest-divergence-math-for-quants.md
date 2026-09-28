---
title: "When live stops matching the backtest: telling decay from bad luck"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "A strategy is underperforming its backtest and you have three months of data. Decay, implementation leak, or the drawdown the backtest always said would happen? The honest answer is that three months cannot tell you, and the senior's job is to say so while still sizing the book today."
tags: ["backtest", "statistical-power", "sharpe-ratio", "information-coefficient", "implementation-shortfall", "bayesian-shrinkage", "alpha-decay", "model-validation", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 20
---

> [!important]
> **TL;DR:** When a live strategy misses its backtest, the question is not "what happened" but "what could I possibly detect with the data I have". Usually the answer is nothing, and you still have to size the book.
>
> - The standard error of an annualised Sharpe is about ${1/\sqrt{T}}$ with $T$ in years, almost regardless of the Sharpe. After three months it is **2.09**. Your live estimate is noise with a number attached.
> - Detecting that a Sharpe-1.5 strategy has **halved** takes **12.0 years** at conventional power. Detecting that the edge is **completely dead** takes **3.0 years**. At three months you catch a dead edge 17.7% of the time against a 5% false-alarm rate.
> - So stop testing the P&L. The **gross-net gap** is a 8.9-standard-error question because you have every fill, and the **signal's own IC** resolves a halving in 19 months rather than 12 years.
> - On a \$300m book, waiting for the P&L t-test to speak costs \$270m of forgone expected P&L. On the same book an implementation leak of 9 basis points costs \$16.2m a year and is fixable this quarter.
> - Size from a posterior, not a test. One year of live data earns a weight of **0.20**, moving a prior Sharpe of 0.50 to 0.44 and the risk budget from \$30m to \$8.8m of annual vol.
> - The real failure is organisational: the strategy gets retired after a drawdown that was always in the distribution, and nobody records the counterfactual.

## The meeting you will be in

A strategy went live fourteen months ago on a backtest with a Sharpe of 1.5. For the last three months it has been down. Someone asks whether the edge is gone, and the room fills with narrative: crowding got worse, the vol regime changed, the costs were always optimistic. All three stories fit the data, which is precisely the problem. Three months of returns is roughly sixty numbers, and sixty numbers cannot separate three hypotheses.

The junior answer is to pick the most plausible story. The senior answer is to work out first what the available data could possibly distinguish, and then to notice that most of the cheap information is not in the P&L at all.

![Matrix of three causes of live underperformance against four diagnostics, showing edge decay moves gross Sharpe and signal IC, implementation leak widens the gross-net gap, and bad luck moves only gross Sharpe and eventually reverts](/imgs/blogs/live-vs-backtest-divergence-math-for-quants-1.webp)

Three causes, three signatures. Edge decay shows up in the signal's own predictive power. Implementation leak shows up as a widening gap between gross and net. Bad luck shows up as a fall in realised return with nothing else moving, and it reverts. The diagnostics are not equally expensive, and the one everybody reaches for first is the worst of the three.

## Foundations: four things to have straight

**Sharpe ratio.** Annualised excess return divided by annualised volatility. A strategy earning 15% a year at 10% volatility has a Sharpe of 1.5. It is a signal-to-noise ratio, and its units are "standard deviations of annual return per year".

**Gross and net.** *Gross* return is what the signal earns at the prices your model assumed. *Net* is what landed in the account after commissions, spread, market impact and financing. The difference is the implementation shortfall, a term from [Perold (1988)](https://doi.org/10.3905/jpm.1988.409150), and on a fast book it is not small.

**Information coefficient (IC).** For a cross-sectional signal, the IC is the correlation, usually rank correlation, between the score you assigned each name today and that name's realised forward return. It is computed **each day across the whole universe**, so a year of data is 252 observations, not 12. This turns out to matter enormously.

**Statistical power.** The probability that your test rejects the null when the alternative is in fact true. A test with 10% power misses a real deterioration nine times in ten, so failing to reject it tells you essentially nothing. That is the single most common error in this whole discussion. Post 3, [concentration inequalities and sample complexity](/blog/trading/math-for-quants/concentration-inequalities-sample-complexity-math-for-quants), develops the general "how much data do I need" machinery; what follows points it at one question.

One more foundation, and it is the load-bearing one. For returns sampled $n$ times a year over $T$ years, the standard error of the estimated annualised Sharpe is, following [Lo (2002)](https://doi.org/10.2469/faj.v58.n4.2453),

$$\operatorname{SE}(\hat{S}) \approx \sqrt{\frac{1 + S^2/(2n)}{T}}.$$

Look at what the correction term does. With monthly data and a true Sharpe of 1.5, $S^2/(2n) = 2.25/24 = 0.094$, so the whole factor is 1.094 and its square root is 1.046. With daily data the correction is under half a percent. So to a very good approximation:

**The standard error of an annualised Sharpe is ${1/\sqrt{T}}$, with $T$ in years, and sampling more often does not help.** After one year it is 1.05. After three months it is 2.09.

That last number deserves a pause. Your three-month live Sharpe has a standard error of roughly two. A strategy whose true Sharpe is 1.5 will print a three-month annualised Sharpe below zero about 24% of the time, with nothing whatsoever having changed.

## The power problem, which is the spine

Now make the question precise. The backtest says the Sharpe is $S_0$. You suspect the truth is now $S_0 - \Delta$. You will run a one-sided test at significance $\alpha$ and you want power $1 - \beta$. Standard normal-theory sample sizing gives

$$T = \frac{(z_\alpha + z_\beta)^2 \left(1 + S^2/(2n)\right)}{\Delta^2},$$

with $T$ in years. At the conventional $\alpha = 0.05$ and 80% power, $(z_\alpha + z_\beta)^2 = (1.645 + 0.842)^2 = 6.18$. Take monthly returns and a backtest Sharpe of 1.5, so the correction factor is 1.094. Then:

| The deterioration you want to detect | $\Delta$ | Years of live data needed |
| --- | --- | --- |
| Sharpe 1.5 falls to 1.0 | 0.50 | **27.0** |
| Sharpe 1.5 halves to 0.75 | 0.75 | **12.0** |
| Edge completely dead, 1.5 to 0 | 1.50 | **3.0** |

Three years of live trading to establish, at ordinary confidence, that a strategy has **no edge at all**. Twelve years to establish that it halved. This is not a quirk of the assumptions. It follows from ${\operatorname{SE} \approx 1/\sqrt{T}}$, which is about as robust as results in this field get.

Run it the other way and ask what three months buys. Power at horizon $T$ is

$$\text{power}(T) = \Phi\!\left(\frac{\Delta\sqrt{T}}{\sqrt{1 + S^2/(2n)}} - z_\alpha\right).$$

![Line chart of statistical power against months of live data for three deterioration sizes, with all three curves under 18 percent at three months and only the fully-dead case reaching 80 percent power, at 36 months](/imgs/blogs/live-vs-backtest-divergence-math-for-quants-2.webp)

At $T = 0.25$ years the test detects a fully dead edge **17.7%** of the time, a halved Sharpe **9.9%** of the time, and a fall from 1.5 to 1.0 **8.0%** of the time. The false-alarm rate when nothing has happened is 5%. A test that fires 9.9% of the time when the edge has halved and 5% of the time when it has not is not a diagnostic. It is a coin with a slight lean, and the lean is smaller than the disagreement in the room.

#### Worked example 1: 12 years of waiting costs \$270m on a \$300m book

Take the \$300m book at a 10% volatility target. A Sharpe of 1.5 is an expected \$45m a year; a Sharpe of 0.75 is \$22.5m. So if the halving is real, it is costing \$22.5m a year of expected P&L, every year, starting now.

The test that would confirm it needs 12.0 years. Over that window the forgone expected P&L is $12.0 \times \$22.5\text{m} = \$270\text{m}$, which is ninety percent of the book. Waiting for statistical significance is not a conservative choice. It is the most expensive choice on the menu.

And in the other direction, look at why three months could never have worked. Three months of the halving costs \$5.6m of expected P&L. Three months of ordinary noise on a 10%-vol book has a standard deviation of $\$300\text{m} \times 10\% / 2 = \$15\text{m}$. You are trying to see a \$5.6m signal inside \$15m of noise. Of course you cannot.

*(All dollar figures in this post are illustrative arithmetic on assumed inputs, not measured results from any real book.)*

## The cheapest diagnostic: gross minus net

Here is the move that rescues the situation. The P&L is a terrible place to look because it is one noisy number per period. Costs are a wonderful place to look because you have **every fill**, and slippage per fill is measured directly against a decision price rather than inferred from a mean return.

#### Worked example 2: gross matches, net does not, and the gap is \$16.2m a year

A \$300m stat-arb book turning over \$18bn two-way in a year, about 24% of the book each day. The backtest assumed 5.0 basis points of all-in cost per dollar traded.

| | Backtest | Live, 12 months |
| --- | --- | --- |
| Gross return | +20.0% (\$60.0m) | +19.4% (\$58.2m) |
| Cost per dollar traded | 5.0 bps | 14.0 bps |
| Costs | -3.0% (\$9.0m) | -8.4% (\$25.2m) |
| Net return | 17.0% (\$51.0m) | 11.0% (\$33.0m) |
| Net Sharpe | 1.70 | 1.10 |

![Two stacked bars comparing backtest and live returns on a 300 million dollar book, with near-identical green gross segments and a live red cost segment almost three times the backtest assumption, annotated with a 16.2 million dollar implementation gap](/imgs/blogs/live-vs-backtest-divergence-math-for-quants-3.webp)

The net shortfall is 6.0 percentage points. The cost blowout alone is 5.4 points, so **costs explain 90% of the shortfall**. The implementation gap is \$16.2m a year, and halving the slippage by trading a slower schedule recovers \$8.1m of it.

Now the statistical part, which is what makes this diagnostic worth reaching for first. Slippage is correlated across names within a day, so cluster by day rather than by fill: 252 daily average-slippage numbers with a standard deviation of 16.0 bps give a standard error of ${16.0/\sqrt{252} = 1.01}$ bps. The 9 bp gap is therefore **8.9 standard errors**. (Treating the individual fills as independent would give 57 standard errors, which is a lie; the clustered number is the honest one, and it is still overwhelming.)

Be precise about what this does and does not establish. The gross side is *not* confirmed: 19.4% against 20.0% is a Sharpe difference of 0.06, which is 0.06 standard errors, indistinguishable from anything. What is established is that costs blew out by an amount accounting for nearly the whole shortfall, and that is enough to act on because it names a cheap fix. The cost question is an 8.9-sigma question and the Sharpe question is a 1-sigma question, asked of the same twelve months of data.

## Measure the signal, not the P&L

The second escape is to test the signal directly rather than the money it made. The IC is computed daily across the universe, and it is measured before position limits, risk-model neutralisation, capacity constraints and costs have thrown information away. That makes it a far higher signal-to-noise statistic than the P&L it produces.

Suppose the backtest showed a mean daily IC of 0.030 with a daily cross-sectional standard deviation of 0.12. That is an annualised t-statistic of ${0.030/0.12 \times \sqrt{252} = 3.97}$ on the signal, against a Sharpe of 1.5 on the strategy. The gap between those two numbers is the portfolio construction, and it is why testing the P&L throws away most of your statistical power.

#### Worked example 3: the null result today, and \$235m of waiting avoided

Over three months you have 63 daily ICs. The standard error of their mean is ${0.12/\sqrt{63} = 0.0151}$. If the live mean IC comes in at 0.015, an exact halving of the signal, that sits $(0.030 - 0.015)/0.0151 = 0.99$ standard errors below the backtest. **Not significant. The diagnostic tells you nothing.**

This is the honest null result, and it is worth stating plainly rather than burying: on three months, the IC test fails too. The difference is how fast it stops failing. Solving the same sample-size formula in daily observations:

- to detect the IC **halving** at 80% power: ${6.18 \times (0.12/0.015)^2 = 396}$ trading days, about **19 months**
- to detect the IC going to **zero**: ${6.18 \times (0.12/0.030)^2 = 99}$ trading days, about **5 months**

Nineteen months against the P&L test's 144 months. Testing the signal is **7.7 times faster** than testing the money, for the same question, on data you already have.

Put the \$22.5m-a-year bleed from worked example 1 against that gap. Resolving the halving at 19 months instead of 144 means you stop paying for it 125 months earlier, which is $10.5 \times \$22.5\text{m} = \$235\text{m}$ of expected P&L that the slower diagnostic would have thrown away. That is the practical case for instrumenting the signal separately from the book, and it is a thing a research platform either does from day one or retrofits painfully.

So the realistic answer at three months is not a verdict: the cost diagnostic can already speak, the IC diagnostic will speak in a few months, and the P&L diagnostic never will. Set the review date accordingly.

## Regime or decay, and why you often cannot tell

Suppose gross still matches, costs are where they should be, and the IC has fallen. Is the signal dead, or is it in a regime where it does not work and will come back?

Formally this is a [regime-switching problem](/blog/trading/math-for-quants/regime-switching-hidden-markov-math-for-quants), and formally it is estimable. Practically it usually is not, and the reason is the same power argument applied twice over. To estimate a regime-conditional Sharpe you need years *inside that regime*, and a three-month window contains at most one regime transition. If the unfavourable state has an expected duration of eight months, which corresponds to a monthly persistence of 0.875, then a quarter spent inside it contains no information at all about whether the favourable state still pays.

Worse, the two hypotheses are observationally equivalent until the regime actually reverts and the strategy reverts with it. "Decayed" and "in a bad regime that will end" make identical predictions about every observation you have. The distinction becomes falsifiable only after a full regime cycle, which is exactly the horizon you do not have.

The senior move is to stop pretending the distinction is resolvable and instead ask whether the regime story was **pre-registered**. If the research document said in advance that the signal depends on dispersion being above some level, and dispersion is now below it, that is a testable claim with a mechanism. If the regime story was invented this week to explain the drawdown, it is a narrative, and narratives always fit.

## Is this drawdown inside the backtest's own distribution?

There is one cheap calculation that turns "we are down and it feels bad" into a number. Model cumulative P&L as a Brownian motion with drift. The probability of ever being down $D$ annual volatility units is

$$P(\text{ever down } D \mid S) = e^{-2SD}.$$

On the \$300m book at 10% vol, a 6% drawdown is $D = 0.6$ vol units. Under the backtest Sharpe of 1.5 the probability of ever seeing it is $e^{-1.8} = 16.5\%$. Under a halved Sharpe of 0.75 it is $e^{-0.9} = 40.7\%$. The likelihood ratio is ${0.4066/0.1653 = 2.46}$.

So the drawdown **is** evidence for deterioration, and the evidence is 2.46 to 1. That moves even odds to about 71%, which is a real update and nowhere near a verdict; deepen the drawdown to a full vol unit and the ratio rises to 4.48. (This is an infinite-horizon result applied to a finite window, so treat it as an order of magnitude.) The discipline is that the number exists at all: it converts a feeling into a factor you can multiply a prior by, and the factor is usually smaller than the room assumes.

## The decision: a posterior, not a hypothesis test

You must size the strategy today whatever the tests say. A hypothesis test is the wrong tool for that, because it returns a verdict and you need a number. What you want is a posterior over the true Sharpe.

Set a prior. The crucial point is that **the prior is not centred on the backtest**. A backtest Sharpe is the maximum of many trials, so it is biased upward by selection; this is the whole content of [Bailey and Lopez de Prado's deflated Sharpe ratio](https://doi.org/10.3905/jpm.2014.40.5.094) and of [Harvey, Liu and Zhu's](https://doi.org/10.1093/rfs/hhv059) multiple-testing haircuts. If your group tested a hundred variants to find this one, a defensible prior mean is well below the headline. Take $\mu_0 = 0.50$ with prior standard deviation $\tau = 0.50$.

Twelve months of live data give $\hat{S} = 0.20$ with $\sigma = 1.0$. The conjugate normal update is

$$\mu_{\text{post}} = \frac{\mu_0/\tau^2 + \hat{S}/\sigma^2}{1/\tau^2 + 1/\sigma^2}, \qquad \sigma^2_{\text{post}} = \frac{1}{1/\tau^2 + 1/\sigma^2}.$$

With $1/\tau^2 = 4$ and $1/\sigma^2 = 1$: the weight on the live estimate is ${1/5 = 0.20}$, the posterior mean is $(2.0 + 0.2)/5 = 0.44$, and the posterior standard deviation is $1/\sqrt{5} = 0.447$. This is [hierarchical shrinkage](/blog/trading/math-for-quants/hierarchical-bayes-pooling-math-for-quants) applied to a single strategy, and it is the same trade the [James-Stein estimator](/blog/trading/math-for-quants/shrinkage-stein-paradox-math-for-quants) makes: accept a little bias to cut a lot of variance.

![Three overlaid density curves over true annualised Sharpe, showing a wide flat live estimate centred at 0.20, a narrower prior at 0.50, and a posterior at 0.44 close to the prior, with risk budgets of 4.0, 8.8 and 30.0 million dollars marked](/imgs/blogs/live-vs-backtest-divergence-math-for-quants-4.webp)

One year of live trading moved the belief from 0.50 to 0.44. That is what a year is worth, and it is the correct answer to "but the market has spoken".

#### Worked example 4: the sizing decision is worth \$11.2m a year

Give the strategy a risk budget $v$ in dollars of annual volatility and use the standard mean-variance objective $U(v) = Sv - \frac{\lambda}{2}v^2$, optimised at $v^\ast = S/\lambda$. Calibrate $\lambda$ so the original Sharpe-1.5 belief justified the \$30m of annual vol the strategy actually got: $\lambda = 1.5/30 = 0.05$ per \$m.

Evaluate three policies at the posterior mean of 0.44:

| Policy | Risk budget | Certainty equivalent |
| --- | --- | --- |
| Do nothing, stay anchored to the backtest | \$30.0m | **-\$9.30m** |
| Naive: resize on the raw live estimate of 0.20 | \$4.0m | +\$1.36m |
| Posterior: resize on 0.44 | \$8.8m | **+\$1.94m** |

Doing nothing is not neutral. It is negative \$9.3m a year of certainty equivalent, because the position is sized for an edge that is not there and the variance penalty dominates. Against the posterior, inertia costs \$11.2m a year and over-reacting to the live number costs \$0.58m. Both are errors; they are not remotely the same size, which is the practical argument for resizing promptly and moderately rather than debating and then flipping.

One refinement worth having. The posterior variance of 0.20 is itself risk: P&L variance becomes $v^2(1 + \sigma^2_{\text{post}})$, so the effective risk aversion rises to $0.05 \times 1.2 = 0.06$ and the optimal budget falls to $0.44/0.06 = \$7.33\text{m}$, a further 16.7% haircut. **Not knowing the Sharpe is itself a reason to run smaller**, independently of what the Sharpe turns out to be.

## The organisational failure mode

The statistics above have an institutional shadow, and it is where the money actually leaks.

A firm that retires any strategy after a drawdown of half a vol unit will, by the formula above, kill $e^{-2 \times 1.5 \times 0.5} = 22.3\%$ of its genuinely Sharpe-1.5 strategies. Move the trigger to a full vol unit and the false-retirement rate drops to 5.0%. Neither number is knowable inside the firm, because the retired strategy stops producing data the moment it is switched off. The counterfactual is never recorded, so the trigger is never recalibrated, and a policy that is quietly killing a fifth of the good ideas looks exactly like a policy that is working.

The fix is cheap and almost nobody does it: keep a **shadow book**. Every retired strategy keeps running on paper for twelve months, gross and net, with its IC logged. It costs a line in a config file, and it is the only way the organisation ever learns whether its retirement trigger is too tight.

## Common misconceptions

**"Three months of underperformance means the edge is gone."** Three months of live returns detects a fully dead edge 17.7% of the time and a halved Sharpe 9.9% of the time, against a 5% false-alarm rate. The drawdown is weak evidence, worth a likelihood ratio around 2.5, not a verdict. Most of what you are seeing is the variance the backtest itself predicted.

**"A t-test on live returns answers this."** It does not, for two reasons. First, at 10% power, failing to reject is uninformative in both directions: you have learned almost nothing about whether anything changed. Second, the null is wrong to begin with, because the backtest Sharpe is an estimate selected as the best of many trials. Testing against a number that is biased upward guarantees you reject too often when the strategy is fine. [Hypothesis testing and p-values](/blog/trading/math-for-quants/hypothesis-testing-pvalues-math-for-quants) covers the multiple-testing half of this properly.

**"If it still works gross, there is no problem."** Nine basis points of unexpected slippage was \$16.2m a year in the example above, which is a problem by any definition. And "works gross" is measured at your model's assumed fill prices, which are precisely the assumption under test. A gross number computed against a mid price your orders were pushing around is not independent evidence.

## Sources and further reading

- Lo, A. W. (2002). ["The Statistics of Sharpe Ratios."](https://doi.org/10.2469/faj.v58.n4.2453) *Financial Analysts Journal* 58(4), 36-52. The standard error formula used throughout.
- Bailey, D. H., and Lopez de Prado, M. (2014). ["The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."](https://doi.org/10.3905/jpm.2014.40.5.094) *Journal of Portfolio Management* 40(5), 94-107.
- Bailey, D. H., Borwein, J., Lopez de Prado, M., and Zhu, Q. J. (2014). ["Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance."](https://doi.org/10.1090/noti1105) *Notices of the AMS* 61(5), 458-471.
- Harvey, C. R., Liu, Y., and Zhu, H. (2016). ["... and the Cross-Section of Expected Returns."](https://doi.org/10.1093/rfs/hhv059) *Review of Financial Studies* 29(1), 5-68. The multiple-testing haircut on reported factor t-statistics.
- Harvey, C. R., and Liu, Y. (2015). ["Backtesting."](https://doi.org/10.3905/jpm.2015.42.1.013) *Journal of Portfolio Management* 42(1), 13-28.
- Grinold, R. C., and Kahn, R. N. (1999). *Active Portfolio Management*, 2nd ed. McGraw-Hill. The fundamental law and the IC machinery.
- Perold, A. F. (1988). ["The Implementation Shortfall: Paper versus Reality."](https://doi.org/10.3905/jpm.1988.409150) *Journal of Portfolio Management* 14(3), 4-9.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 11-14 on backtesting statistics.

Dollar figures in the worked examples are illustrative arithmetic on assumed inputs, chosen to make the magnitudes concrete; they are not measured results.

## In the interview room and on the desk

The question arrives as: *"Your strategy is down over three months and the backtest said it should be up. What do you do?"* It is asked at Citadel, Two Sigma, Millennium and every pod seat where you will one day have to defend a live book to a risk committee, and it is asked because it separates people who have run money from people who have only run research.

The weak answer is a narrative. Crowding, regime, costs got worse. Every one of those is plausible and none is a measurement, and an interviewer who has heard it three times that week will stop listening.

The strong answer is ordered by cost of information, and it starts before any hypothesis.

First, **gross minus net**. You have every fill, so a cost blowout lands at nearly nine standard errors where the Sharpe question lands at one. If realised cost per dollar traded has drifted from 5 bps to 14 bps, you have your answer in a week, it explains most of the shortfall, and it is an execution problem you can fix rather than an alpha problem you cannot.

Second, **the signal's own IC**, measured independently of the P&L. It is a daily cross-sectional statistic, so it accumulates evidence roughly eight times faster than the return series does. A dead signal and a live signal traded badly look identical in the P&L and completely different here.

Third, and only then, **state the power of the window before drawing any conclusion about the Sharpe**. Say the number out loud: with three months, detecting even a total collapse of the edge has under 20% power, so the P&L cannot settle this and I will not pretend it can. Then give the decision anyway, as a posterior rather than a verdict: prior mean 0.5, live estimate gets a weight of 0.2, posterior 0.44, so I cut the risk budget from \$30m to roughly \$9m and set a review at the horizon where the IC test actually has power.

The trap is running a significance test on three months, failing to reject, and reporting that as evidence nothing is wrong. It looks rigorous. It is the exact inverse of rigour, because a test with 10% power fails to reject almost regardless of the truth. The candidate who names that asymmetry, and who resizes without claiming to have diagnosed, is the one who gets the offer.
