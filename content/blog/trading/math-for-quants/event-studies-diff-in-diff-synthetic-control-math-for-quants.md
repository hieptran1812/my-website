---
title: "Event studies done properly: abnormal returns, parallel trends and synthetic controls"
date: "2026-09-21"
description: "A build-from-zero guide to event windows, normal-return models, cumulative abnormal returns, the clustering problem that inflates t-statistics, difference-in-differences, and synthetic controls."
tags: ["event-study", "abnormal-returns", "cumulative-abnormal-return", "difference-in-differences", "parallel-trends", "synthetic-control", "causal-inference", "clustered-standard-errors", "quant-research", "math-for-quants", "quant-finance"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 20
---

> [!important]
> **TL;DR:** The event study is the most-run and most-abused design in empirical finance. Done properly it is a genuine causal estimate. Done the usual way it measures the event plus everything else that happened that week.
>
> - It is four windows and one subtraction: fit a *normal return* on an estimation window, measure the *abnormal return* on an event window, sum those into a *cumulative abnormal return* (CAR).
> - The normal-return model is the design choice people treat as a formality. On one plausible day, four standard models give abnormal returns from ${+3.05\%}$ to ${+4.25\%}$: a 120 basis-point spread decided before any data was seen.
> - Two things break the t-statistic. **Event-induced variance**: the event day is louder than the estimation window, so the denominator is too small. **Cross-sectional correlation**: clustered events are not independent observations.
> - The number to remember: with 40 events and an average pairwise correlation of only ${0.10}$, the effective sample size is ${40 / (1 + 39 \times 0.10) = 8.2}$ events, and a t-statistic of 3.79 becomes 1.71.
> - Difference-in-differences and synthetic control are the ways out when a market model is not enough. Both rest on a counterfactual you never observe, and a clean pre-period is evidence for it, never proof.

## Introduction

A regulator changes a tick size. A company pre-announces guidance. An index provider adds a name. In every case someone on a desk is asked the same question: *what did that do to the price?*

The tempting answer is to look. The stock was up 4.05% over the three days around the announcement, so the announcement was worth 4.05%. On a \$20m position that is \$810,000, and the story writes itself.

It is the wrong answer, and not by a small amount. Some of that 4.05% was the market moving, some the stock's normal sensitivity to it, some a sector rotation with nothing to do with the company. The event study strips all of that out and leaves only the part the event can be held responsible for. The diagram below is the mental model: four windows on a calendar, one used to learn what normal looks like, one used to measure the departure from it.

![A trading-day timeline with four bands: an estimation window from day -280 to -31 where the normal-return model is fitted, a buffer gap to day -2, a highlighted event window from -1 to +1 where abnormal returns are measured, and a post-event window to day +20 for drift checks.](/imgs/blogs/event-studies-diff-in-diff-synthetic-control-math-for-quants-1.webp)

This post builds the design from zero, then shows the three places it fails: the model you picked for normal returns, the t-statistic you quoted, and the counterfactual you assumed. It is the design layer on top of [causal inference for alpha research](/blog/trading/math-for-quants/causal-inference-alpha-research-math-for-quants) and [instrumental variables and natural experiments](/blog/trading/math-for-quants/instrumental-variables-natural-experiments-math-for-quants). Every dollar figure below is illustrative arithmetic on assumed inputs, chosen to be checkable rather than to describe a particular stock.

## Foundations: four windows and four quantities

A **return** is the percentage change in price over a period: a stock closing at \$100 and then \$103.10 had a daily return of ${3.10\%}$. An **event** is a dated piece of news, and **day 0** is the day the market could first have traded on it, which is not always the day the press release is dated. A release at 6pm is a day-0 event for the following session.

The **estimation window** is a stretch of history used to learn how the stock normally behaves, commonly 250 trading days ending well before the event. The **event window** is the short span over which the effect is measured, often ${[-1, +1]}$ days. The **gap** between them keeps leakage and pre-announcement drift out of the fit, and a **post-event window** checks for persistence or reversal without entering the headline number.

The **normal return** is what the stock would have returned that day had the event not happened. You cannot observe it, so you model it: ${E[R_{i,t} \mid X_t]}$, the expected return of stock ${i}$ on day ${t}$ given the conditioning information ${X_t}$ the model uses, typically that day's market return. The **abnormal return** is the residual:

$$
AR_{i,t} = R_{i,t} - E[R_{i,t} \mid X_t]
$$

That single line is the whole idea. Everything else is a fight about the second term.

The **cumulative abnormal return**, or CAR, is the sum of abnormal returns across the event window:

$$
CAR_i(t_1, t_2) = \sum_{t=t_1}^{t_2} AR_{i,t}
$$

Sum, not compound: over three days the difference is a rounding error, and summing is what makes the variance arithmetic below tractable. Across many events, the **cumulative average abnormal return** (CAAR) averages those CARs. That average is the number a research note quotes, and what the statistics section below is about.

## Why the normal-return model decides the answer

Four models dominate, in increasing order of how much they remove. **Mean-adjusted returns** use the stock's own estimation-window average. **Market-adjusted returns** use the market return itself, the market model with the sensitivity forced to 1. **The market model** is an ordinary least squares regression fitted on the estimation window:

$$
R_{i,t} = \alpha_i + \beta_i R_{m,t} + \varepsilon_{i,t}
$$

Here ${\beta_i}$ is the stock's sensitivity to the market and ${\alpha_i}$ its average drift once the market is accounted for; the residual is the abnormal return. [OLS, GLS and regularized regression](/blog/trading/math-for-quants/regression-ols-gls-regularized-math-for-quants) covers the fitting machinery. **A factor model** adds more systematic exposures: size, value, momentum, sector. It removes more, a virtue when your event sample is tilted toward a factor and a hazard when the factor is itself downstream of the event.

These are not four routes to the same place.

![Four normal-return models applied to the same day-0 stock return of plus 3.10 percent against a market return of minus 0.30 percent. Mean-adjusted yields an abnormal return of plus 3.05 percent, market-adjusted plus 3.40, the market model plus 3.44, and a three-factor model plus 4.25, a spread of 1.20 percentage points.](/imgs/blogs/event-studies-diff-in-diff-synthetic-control-math-for-quants-2.webp)

Same observed day, four defensible models, answers spanning 120 basis points. If your headline effect is 3.5% and the model choice moves it by 1.2 percentage points, the model choice *is* the finding. Pick it before you look at the event window, write down why, and report the alternatives in a robustness table. Picking it afterwards, once you can see which choice gives the cleaner result, is how a specification search gets laundered into a result.

#### Worked example 1: one announcement, \$20m on the line

You hold \$20m of a single stock into a scheduled announcement.

**Step 1, fit the model.** Estimation window ${[-280, -31]}$, 250 trading days. OLS of the stock's daily return on the market's gives ${\hat\alpha = 0.02\%}$ per day, ${\hat\beta = 1.20}$, and a residual standard deviation of ${\hat\sigma = 1.10\%}$ per day. That last number is the one people forget to record, and the one that decides significance.

**Step 2, compute normal and abnormal returns on ${[-1, +1]}$.**

| Day | Stock return | Market return | Normal return | Abnormal return |
| --- | --- | --- | --- | --- |
| ${-1}$ | ${+0.70\%}$ | ${+0.40\%}$ | ${0.02 + 1.20 \times 0.40 = +0.50\%}$ | ${+0.20\%}$ |
| ${0}$ | ${+3.10\%}$ | ${-0.30\%}$ | ${0.02 + 1.20 \times (-0.30) = -0.34\%}$ | ${+3.44\%}$ |
| ${+1}$ | ${+0.25\%}$ | ${+0.10\%}$ | ${0.02 + 1.20 \times 0.10 = +0.14\%}$ | ${+0.11\%}$ |

**Step 3, cumulate.** ${CAR(-1, +1) = 0.20 + 3.44 + 0.11 = +3.75\%}$.

**Step 4, put it in money.** On \$20m, a 3.75% abnormal return is \$750,000. The raw three-day return was ${0.70 + 3.10 + 0.25 = 4.05\%}$, or \$810,000, so \$60,000 of the gain was the market and your beta exposure to it. Note the sign: the market was *down* over the window, so the model credits the event with more than the raw move, not less.

**Step 5, ask whether it is significant.** Under the model's own assumption of constant variance, the standard error of a three-day CAR is ${\hat\sigma \sqrt{3} = 1.10 \times 1.732 = 1.905\%}$, so the t-statistic is ${3.75 / 1.905 = 1.97}$. It clears the two-sided 5% critical value of ${1.96}$ by one hundredth. That should worry you.

![Left panel: raw and normal returns for the three event-window days, cumulating to an abnormal return of plus 3.75 percent, or 750,000 dollars on a 20 million dollar position. Right panel: the same CAR with a t-statistic of 1.97 under estimation-window variance and 1.03 once the day-zero sigma is tripled, against a dashed 1.96 critical value.](/imgs/blogs/event-studies-diff-in-diff-synthetic-control-math-for-quants-3.webp)

## The two reasons your t-statistic is too big

### Event-induced variance

The market model estimates the residual standard deviation on a window containing no event, then applies it to a day the world learned something. Announcement days are more volatile than ordinary days, which is the whole reason anyone trades them, so a quiet-period sigma puts too small a number in the denominator.

Suppose the day-0 residual standard deviation in the example above is three times the estimation-window value, ${3.30\%}$ rather than ${1.10\%}$. The three-day standard error becomes

$$
\sqrt{1.10^2 + 3.30^2 + 1.10^2} = \sqrt{13.31} = 3.65\%
$$

and the t-statistic falls from ${1.97}$ to ${3.75 / 3.65 = 1.03}$. Nothing about the measured effect changed. The claim of significance evaporated because the variance assumption was wrong.

The standard fix is the **standardized cross-sectional test** of Boehmer, Musumeci and Poulsen (1991): standardize each firm's abnormal return by its own estimation-window standard deviation, then take the cross-sectional variance of those standardized values, which lets the data set the event-window variance instead of the estimation window asserting it. Its ancestor, the Patell (1976) test, standardizes but still assumes the estimation-window variance carries over.

### Cross-sectional correlation when events cluster

This one is larger and less often handled. A naive standard error on a cross-sectional mean assumes independent observations. When 40 companies announce in the same week they share the same market, the same macro surprise, the same sector rotation, so their abnormal returns are correlated.

For ${N}$ events with equal variance ${\sigma^2}$ and average pairwise correlation ${\bar\rho}$, the variance of the mean is

$$
\operatorname{Var}(\overline{AR}) = \frac{\sigma^2}{N}\bigl[1 + (N-1)\bar\rho\bigr]
$$

The bracket is the damage. It rearranges into an effective sample size,

$$
N_{\text{eff}} = \frac{N}{1 + (N-1)\bar\rho}
$$

which is how many independent events your correlated sample is actually worth.

#### Worked example 2: forty events, one announcement week

You run a study on 40 merger announcements, holding \$50m spread equally across the names at \$1.25m each. The mean CAR over ${[-1, +1]}$ is ${1.80\%}$, worth \$900,000 across the book, and the cross-sectional standard deviation of those CARs is ${3.00\%}$.

**Treating the events as independent:** the standard error is ${3.00\% / \sqrt{40} = 0.474\%}$, giving a t-statistic of ${3.79}$. Comfortably significant. You would write it up.

**Accounting for clustering** with an average pairwise correlation of ${\bar\rho = 0.10}$: the inflation factor is ${\sqrt{1 + 39 \times 0.10} = \sqrt{4.9} = 2.21}$. The standard error becomes ${0.474\% \times 2.21 = 1.05\%}$ and the t-statistic is ${1.80 / 1.05 = 1.71}$. Below ${1.96}$. The same \$900,000 of measured abnormal profit, and it no longer clears the bar.

**The effective sample size** is ${40 / 4.9 = 8.2}$. You spent the effort of 40 events and bought the evidence of 8.

Try a correlation so small you would not bother mentioning it, ${\bar\rho = 0.02}$: the inflation factor is ${\sqrt{1.78} = 1.33}$ and the t-statistic falls from 3.79 to 2.84. Still significant, visibly weaker. ${\bar\rho}$ does not need to be large, because it is multiplied by ${N - 1}$.

![Left panel: 40 event markers scattered across two years, with a standard error of 0.47 percent and a t-statistic of 3.79. Right panel: the same 40 markers stacked on one announcement week, where an inflation factor of 2.21 raises the standard error to 1.05 percent and drops the t-statistic to 1.71, below the marked 1.96 threshold, for an effective sample size of 8.2 events.](/imgs/blogs/event-studies-diff-in-diff-synthetic-control-math-for-quants-4.webp)

Kolari and Pynnonen (2010) show that even low cross-correlation biases standard tests toward over-rejection under event-date clustering, and correct both the Patell and Boehmer-Musumeci-Poulsen statistics for it. The other routes are a **calendar-time portfolio**, which collapses the correlated names into one time series per day and sidesteps the cross-section entirely, or a **rank test** such as Corrado (1989), distribution-free and far less sensitive to fat tails ([hypothesis testing and p-values](/blog/trading/math-for-quants/hypothesis-testing-pvalues-math-for-quants) covers why the t-statistic is fragile to begin with).

## Difference-in-differences: subtracting what would have happened anyway

A market model works when the event hits one name and the market is a fair counterfactual. It fails when the event hits a *group* and the whole environment moved at once: a rule change, a tax, a new venue. For those you need a control group.

Difference-in-differences takes two differences: the treated group's change from before to after, and the control group's change over the same span. The estimate is the gap between them:

$$
\hat\delta = \bigl(\bar Y_{T,\text{post}} - \bar Y_{T,\text{pre}}\bigr) - \bigl(\bar Y_{C,\text{post}} - \bar Y_{C,\text{pre}}\bigr)
$$

The second difference is doing the work: it absorbs anything that hit both groups equally, which a simple before-and-after comparison cannot.

**Parallel trends**, stated precisely: *in the absence of treatment*, the treated group's average outcome would have changed by the same amount as the control group's. Writing ${Y(0)}$ for the untreated potential outcome,

$$
E\bigl[Y_{T,\text{post}}(0) - Y_{T,\text{pre}}(0)\bigr] = E\bigl[Y_{C,\text{post}}(0) - Y_{C,\text{pre}}(0)\bigr]
$$

Read the left-hand side carefully. It is about the treated group in the post period *had it not been treated*, which never happened and never will. That is the sense in which parallel trends is untestable.

#### Worked example 3: a tick-size change on fifty stocks

A venue widens the minimum price increment on 25 stocks, and you match 25 similar stocks left alone. The outcome is the average quoted spread in basis points, a **basis point** being one hundredth of a percent.

| Group | Pre | Post | Change |
| --- | --- | --- | --- |
| Treated (25 names) | 8.0 bps | 11.5 bps | ${+3.5}$ bps |
| Control (25 names) | 7.0 bps | 9.0 bps | ${+2.0}$ bps |

The naive before-and-after answer is ${+3.5}$ bps. The diff-in-diff estimate is ${3.5 - 2.0 = +1.5}$ bps. The other ${2.0}$ bps happened to everybody.

**In money.** Your desk trades \$400m of notional a day in these names and pays roughly half the quoted spread as execution cost. A ${1.5}$ bps widening is ${0.75}$ bps of extra cost, or ${0.000075 \times \$400\text{m} = \$30{,}000}$ a day, \$7.5m over 250 trading days. The naive estimate would have said \$70,000 a day, \$17.5m a year: more than double, with \$10m attributed to a rule that did not cause it.

**Now break parallel trends.** Suppose the treated names are smaller and more volatile, and volatility rose in the post period, so their spreads would have widened by ${3.0}$ bps on their own rather than the control's ${2.0}$. The true effect is ${3.5 - 3.0 = +0.5}$ bps, so your estimate of ${1.5}$ carries a ${+1.0}$ bps bias: \$2.5m a year of real cost reported as \$7.5m. That \$5m of misattribution came from an assumption, not from an error in the arithmetic.

![Average quoted spread in basis points against period. Left: treated rises 8.0 to 11.5, control 7.0 to 9.0, and a dashed counterfactual parallel to the control ends at 10.0, giving a difference-in-differences estimate of plus 1.5 basis points against a naive plus 3.5. Right: an honest counterfactual of 11.0 leaves a true effect of plus 0.5 and a bias of plus 1.0 basis points.](/imgs/blogs/event-studies-diff-in-diff-synthetic-control-math-for-quants-5.webp)

**What a pre-trend test can and cannot tell you.** Plotting several pre-periods and checking that the groups moved together is worth doing, and a visible divergence before treatment is strong evidence against the design. The reverse does not hold. Parallel pre-trends are consistent with a post-period divergence caused by anything that started when the treatment did, and the tests are often underpowered, so a flat pre-period may only mean you could not detect a slope.

Two further traps. Bertrand, Duflo and Mullainathan (2004) showed that serial correlation wrecks conventional diff-in-diff standard errors: with about 20 years of data, their placebo laws, which by construction did nothing, produced an effect significant at the 5% level in up to 45% of cases. Cluster at the unit level, or aggregate the pre and post periods into two points ([stationarity and autocorrelation](/blog/trading/math-for-quants/stationarity-autocorrelation-math-for-quants)). Second, under staggered timing the two-way fixed effects regression everyone reaches for averages comparisons that can use already-treated units as controls, with weights not guaranteed to be positive. Goodman-Bacon (2021) decomposes it; Callaway and Sant'Anna (2021) repair it.

## Synthetic control: when there is no clean control

Sometimes there is exactly one treated unit and no comparable untreated one: one country changes a capital-controls regime, one exchange changes its fee schedule, and no other unit looks like it.

The synthetic control method builds a control out of a weighted blend of the untreated units, the **donor pool**. Writing ${Y_{1t}}$ for the treated unit and ${Y_{jt}}$ for donor ${j}$, the counterfactual is

$$
\hat Y_{1t}(0) = \sum_{j=2}^{J+1} w_j Y_{jt}, \qquad w_j \ge 0, \qquad \sum_j w_j = 1
$$

The weights are chosen to track the treated unit as closely as possible over the *pre-treatment* period, on the outcome and on a set of predictors. The constraints are what make it honest: non-negativity and summing to one keep the counterfactual inside the convex hull of the donors, so you interpolate between real units rather than extrapolate to a fictional one. Sparse weights also make it legible, since you can name the three or four units that constitute it and argue about whether they belong.

![A treated unit and a dashed synthetic control track each other before a vertical treatment line and diverge after it, with the shaded gap labelled as the estimated effect. An inset shows donor weights of 0.42, 0.31, 0.18 and 0.09 summing to one; another shows the real treated path standing outside a bundle of grey placebo paths.](/imgs/blogs/event-studies-diff-in-diff-synthetic-control-math-for-quants-6.webp)

The canonical application is Abadie, Diamond and Hainmueller (2010): a synthetic California built from 38 donor states to estimate the effect of Proposition 99 on cigarette consumption, finding annual per-capita sales about 26 packs lower by 2000 than the counterfactual.

**The honest limits.** With one treated unit there is no cross-section to take a standard error over, so inference is by **permutation**: re-run the procedure pretending each donor was treated, collect the placebo gaps, and ask where the real gap sits among them. With 38 donors the smallest one-sided p-value available is ${1/39}$, a floor on how significant the design can ever be. A poor pre-treatment fit invalidates it entirely, since the method's only claim is that a unit matching for years beforehand would have kept matching. And any donor hit by the same event poisons the counterfactual. Abadie (2021) is the practitioner's guide.

## When the event is bundled

The design assumes the event is one thing. Often it is several. An earnings release arrives with forward guidance, sometimes a buyback in the same statement. An index addition is announced days before the rebalance trade, so the announcement effect and the mechanical demand effect live in different windows. A central bank changes a rate and publishes a statement whose tone moves the curve more than the rate.

There is no statistical trick that unbundles a bundle. What you can do:

- **Shrink the window to isolate the component.** If guidance is in the 8am call and the print was at 6am the previous evening, an intraday window around each timestamp separates them. High-frequency identification around policy announcements is this idea taken seriously.
- **Use cross-sectional variation in one component.** If all firms got the earnings surprise but only some changed guidance, the difference between those groups identifies the guidance effect: a diff-in-diff nested inside the event study. A comparison event carrying only one component, such as an index addition without a rebalance, does the same.
- **Report the bundle honestly.** A correctly labelled bundle is a useful number. A bundle labelled as one of its components is a false one.

## Common misconceptions

**"A significant CAR proves the event caused the move."** It proves the stock moved more than your normal-return model expected, over a window you chose. The causal content comes from the claim that nothing else systematic hit that name in that window, and that is an argument about the world, not an output of the regression. A confounded event produces a beautifully significant CAR.

**"Longer windows give more power."** They give more contamination. Extending from three days to sixty multiplies the noise by roughly ${\sqrt{20}}$ while adding no signal if the market is reasonably efficient, and sweeps in every unrelated thing that happened over three months. Long-horizon abnormal returns also become acutely sensitive to the normal-return model, because a small error in ${\hat\alpha}$ compounds. Short windows are the strength of the method, not a limitation of it.

**"Parallel trends is testable."** The pre-period is evidence and worth collecting, but the assumption is about the treated group's untreated post-period, unobservable by definition. Flat pre-trends make it more plausible. They cannot make it true. The same gap between in-sample fit and out-of-sample truth runs through [bootstrap and cross-validation](/blog/trading/math-for-quants/bootstrap-cross-validation-math-for-quants).

## The checklist before you present one

1. Is day 0 the first day the market could trade on it, and was the normal-return model chosen before the event window was examined?
2. Does the estimation window exclude the event and any leakage, and is the event window justified by information arrival rather than by which length gave a t-statistic above 2?
3. Do the events overlap in calendar time? If so, what is ${\bar\rho}$ and the effective sample size, and is the variance allowed to rise on the event day?
4. For a diff-in-diff: pre-trends plotted, errors clustered, timing checked for staggering. For a synthetic control: good pre-treatment fit, sparse weights, an uncontaminated donor pool.
5. Is the event bundled, and is it labelled as one?

## In the interview room and on the desk

The question arrives as a business problem, not a statistics problem: *how would you measure the impact of X on prices?* The X changes by seat. A short-sale ban, a tick-size pilot, an index reconstitution, a sanctions announcement. The question is the same.

The weak answer jumps to the number: a three-day CAR, a t-statistic, and stop. It has skipped every decision that determines the answer, and an interviewer who runs these for a living hears that immediately.

The strong answer goes in this order. State what day 0 is and why, including when the market could first trade on the information. Choose the normal-return model out loud and justify it: a market model if the event sample is not tilted, a factor model if it is concentrated in small caps or one sector, and say you will report both. Tie the event window to information arrival rather than to significance. Then, before being asked, raise the two statistical problems: event-induced variance, handled with a standardized cross-sectional test, and cross-sectional correlation if the events cluster, handled with a Kolari-Pynnonen correction, a calendar-time portfolio or a permutation test. Finish by naming the identification threat: what else happened in that window, and whether diff-in-diff or synthetic control takes care of it.

The follow-up is almost always about clustering, because it is the fastest way to find out whether you have run one of these yourself. Expect *"your 200 events are 200 banks in one quarter. What is your sample size?"* Being able to write ${N / (1 + (N-1)\bar\rho)}$ on the whiteboard and give the number for a plausible ${\bar\rho}$ ends the question.

The trap that makes a candidate look rigorous while being wrong is quoting a t-statistic on overlapping events that assumes independence. It arrives with full statistical formality, it has a p-value, and it is meaningless. Its cousin is extending the window until the result becomes significant, then reporting only that window.

Two Sigma and Citadel weight this heavily, since both run research organisations where an empirical claim has to survive a hostile read. So does any seat where you defend a number to somebody who loses money if it is wrong.

## Sources and further reading

- MacKinlay, A. Craig (1997). "Event Studies in Economics and Finance." *Journal of Economic Literature* 35(1), 13-39. The standard reference for the mechanics.
- Kothari, S.P. and Jerold Warner (2007). "Econometrics of Event Studies." In *Handbook of Empirical Corporate Finance*, Vol. 1.
- Fama, Eugene, Lawrence Fisher, Michael Jensen and Richard Roll (1969). "The Adjustment of Stock Prices to New Information." *International Economic Review* 10(1), 1-21.
- Brown, Stephen and Jerold Warner (1985). "Using Daily Stock Returns: The Case of Event Studies." *Journal of Financial Economics* 14(1), 3-31.
- Boehmer, Ekkehart, Jim Musumeci and Annette Poulsen (1991). "Event-Study Methodology under Conditions of Event-Induced Variance." *Journal of Financial Economics* 30(2), 253-272.
- Kolari, James and Seppo Pynnonen (2010). "Event Study Testing with Cross-Sectional Correlation of Abnormal Returns." *Review of Financial Studies* 23(11), 3996-4025.
- Corrado, Charles (1989). "A Nonparametric Test for Abnormal Security-Price Performance in Event Studies." *Journal of Financial Economics* 23(2), 385-395.
- Bertrand, Marianne, Esther Duflo and Sendhil Mullainathan (2004). "How Much Should We Trust Differences-in-Differences Estimates?" *Quarterly Journal of Economics* 119(1), 249-275. The placebo-law result quoted above.
- Abadie, Alberto, Alexis Diamond and Jens Hainmueller (2010). "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the American Statistical Association* 105(490), 493-505.
- Abadie, Alberto (2021). "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects." *Journal of Economic Literature* 59(2), 391-425. The practitioner's guide.
- Goodman-Bacon, Andrew (2021). "Difference-in-Differences with Variation in Treatment Timing." *Journal of Econometrics* 225(2), 254-277.

## Where to take this next

The natural companion is [instrumental variables and natural experiments](/blog/trading/math-for-quants/instrumental-variables-natural-experiments-math-for-quants), for when you cannot find a control group but can find variation that is as good as random. Behind both sits [causal inference for alpha research](/blog/trading/math-for-quants/causal-inference-alpha-research-math-for-quants), which turns "what else happened in that window" from a worry into a precise question.

Take an event study you already believe, recompute it with a different normal-return model, then recompute the standard error assuming ${\bar\rho = 0.05}$. If the conclusion survives both, it is worth defending.
