---
title: "Rolling-Correlation Regimes and Detecting the Break in Python"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How to detect a correlation regime change in near real time: compute rolling and EWMA correlations, build a rolling z-score, and apply a CUSUM-style change-point flag so you stop trusting a decayed correlation the moment it flips."
tags: ["macro", "correlation", "rolling-correlation", "ewma", "change-point", "cusum", "regime-detection", "z-score", "stock-bond-correlation", "python", "pandas", "quant"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A correlation does not break gently; it flips, and your job is to *detect the flip in near real time* rather than discover it months later in a drawdown. The toolkit is small: a rolling or EWMA correlation, a rolling z-score of that correlation against its own long-run normal, and a persistence filter (a CUSUM) that fires only when the deviation holds — so you stop trusting the old beta the day the regime changes.
>
> - A single full-sample correlation has no clock. Over 2007 to 2025 the correlation between gold and the 10-year real yield is about **−0.01** — apparently nothing — yet it was **−0.96** through 2021 and **+0.80** afterwards. The detector's whole purpose is to find the date the **−0.96** died.
> - The two estimators you will use are `df.rolling(window).corr()` (equal-weight, lags by roughly half the window) and `df.ewm(halflife=h).corr()` (exponential weight, reacts faster). Choosing the window or half-life is a **bias-variance** decision with no free lunch.
> - Detection turns the rolling correlation into a **z-score** against its trailing mean, then runs a **CUSUM** so a one-day blip never fires but a sustained shift does. The flag is binary: keep trusting the old number, or re-fit to the new regime.
> - The one number to remember: trading a hedge ratio that is one regime stale can cost you the full move. A 60/40 book that kept a **−0.3** stock-bond assumption into 2022 lost roughly **\$25** of stocks and **\$30** of bonds per **\$100** — both legs down, because the real correlation had already flipped to **+0.6**.

In October 2022 a risk officer I know ran the same report he ran every morning, and for the first time in his career the number at the bottom did not make sense. His firm's flagship balanced book was down hard on the year, and the *bonds* — the ballast, the part that was supposed to go *up* when stocks fell — were down even harder than the stocks. His value-at-risk model, fitted on twenty years of daily returns, still printed a stock-bond correlation of about −0.3. Comfortably negative. According to the model, the bonds were diversifying the equity risk that very morning. According to the profit-and-loss statement, they had been amplifying it for nine straight months.

The model was not wrong about history. It was wrong about *which* history was live. It had averaged the deeply negative correlation of the 2000-2021 disinflation era — when every scare was a growth scare and the Fed answered by cutting rates, lifting bonds while stocks fell — together with the violent positive correlation of an inflation shock. The blend came out negative, so the model slept. But the *live* correlation, measured over the trailing two years, had crossed zero in early 2022 and was sitting near **+0.6** by mid-year. The relationship his entire hedge depended on had reversed sign, and his single backward-looking number had no way to tell him, because a single number has no detector attached.

This post builds the detector. The previous post in this series, [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters), argued that a correlation is a moving target you must track over a window rather than measure once. This one is the operational sequel: given that the correlation moves, *how do you know — automatically, in near real time — that it has structurally flipped, so you stop trusting the old number before it costs you?* We will compute rolling and EWMA correlations in pandas, build a rolling z-score that measures how far today's correlation sits from its own normal, and apply a simple change-point method (a persistence-filtered z-score, which is the intuition behind a CUSUM) that flags the break. Every piece is a few lines of code you can run, and every figure is built from the documented macro correlations this series tracks.

![The change-point detection loop from rolling correlation to a break flag](/imgs/blogs/rolling-correlation-regimes-and-change-point-detection-1.png)

## Foundations: what "detecting a break" actually means

Before any code, fix the vocabulary, because the whole exercise turns on a few precise ideas.

A **correlation** between two return series is a number between −1 and +1 that measures how tightly they move together *linearly*. Plus one means they move in lockstep; minus one means they move exactly opposite; zero means no linear relationship. If you have never seen the formula, the companion post [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) builds it from covariance — here we treat it as a known quantity and focus on its *behavior over time*.

A **rolling correlation** is that same number re-estimated every period using only the most recent N observations. If you have daily returns and you set N to 63 (about three trading months), then on each day you compute the correlation of the last 63 days and throw the day-64-ago observation out. Plot that series and you get a line that wanders between −1 and +1 as the relationship strengthens, weakens, and flips. This wandering line is the raw material the detector watches.

A **regime** is a stretch of time over which the correlation is roughly stable around some level. The stock-bond correlation lived in a "negative regime" near −0.3 to −0.5 from roughly 2000 to 2021, then jumped into a "positive regime" near +0.5 in 2022. A **regime change** — equivalently a **structural break** or **change point** — is the date the series leaves one stable level and settles at a different one. The post [structural shifts: why today's correlations aren't yesterday's](/blog/trading/macro-correlations/structural-shifts-why-todays-correlations-arent-yesterdays) catalogs *why* these breaks happen (inflation-targeting regime change, the QE era, the 2022 inflation shock); this post is about *detecting* them mechanically.

**Change-point detection** is the statistical problem of finding the date — ideally as soon after it happens as possible — at which a series' generating process changed. There is a deep literature here (Bayesian online change-point detection, binary segmentation, the PELT algorithm), but you do not need any of it to get most of the value. The honest core is two questions: *is today's correlation far from what it has normally been?* and *has it stayed far for long enough that I should believe it rather than dismiss it as noise?* The first question is answered by a **z-score**; the second by a **persistence filter** such as a **CUSUM**. The rest of this post is those two ideas, dressed in just enough code to run.

One more term, because it is the trap the detector exists to avoid. A **full-sample correlation** is the single number you get from computing the correlation over the entire history. It is a weighted average of every regime the series ever passed through, and when the regimes have opposite signs, that average is worse than useless — it tells you "no relationship" precisely when there is a *strong* relationship that merely changed direction. The figure below is the canonical case: gold versus the real yield.

![Gold versus the ten-year real yield scatter showing a clean negative regime that broke to positive](/imgs/blogs/rolling-correlation-regimes-and-change-point-detection-3.png)

Through 2021 the relationship is one of the cleanest in macro: as the real yield (the inflation-adjusted return on a "safe" bond) rises, the opportunity cost of holding gold — which pays no interest — rises with it, so gold falls. The fitted line slopes down hard, r ≈ **−0.96**. Then in 2022 central banks (led by emerging-market buyers) began buying gold in size for reasons unrelated to real yields, and gold rose *even as* real yields rose. The post-2022 cloud slopes *up*, r ≈ **+0.80**. Pool the two and the regression line is nearly flat: full-sample r ≈ **−0.01**. A risk model fed that **−0.01** would conclude gold and real yields are unrelated and size positions accordingly — exactly backwards in both regimes. The detector's job is to notice, around 2022, that the **−0.96** had died.

One subtlety worth pinning down before we automate anything: the correlation we are tracking is the **Pearson** correlation, which measures *linear* co-movement and is sensitive to outliers. A single crash day — a −12% session that drags both series down together — can swing a short-window Pearson correlation by 0.2 all by itself, and your detector would dutifully flag a "break" that is really just one fat-tailed observation. For most regime detection this is a feature, not a bug: a crisis day genuinely *is* the start of the everything-correlates regime, so you want it to count. But when you are detecting a *slow structural* break (the kind that unfolds over months, like the gold decoupling), you can make the detector robust to single-day outliers by switching to the **Spearman** (rank) correlation, which correlates the *ranks* of the returns rather than their raw values and so cannot be yanked by one extreme print. In pandas it is a one-word change: `a.rolling(63).corr(b, method="spearman")` for the rolling version (the rank version is not available on `ewm`, so for an EWMA-style robust estimate you winsorize the returns first — clip them at, say, the 1st and 99th percentile — then run the ordinary EWMA correlation). The distinction between Pearson and Spearman is developed in [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta); the operational takeaway for the detector is: Pearson for fast crisis-regime monitoring where you *want* the tail to count, a rank or winsorized estimate for slow structural breaks where a single day should not trip the alarm.

## The two estimators: rolling and EWMA, in three lines each

Everything starts from a clean DataFrame of *returns* (not prices — correlations of price levels are a classic spurious-correlation trap, covered in [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data)). Assume you have already pulled and aligned two daily price series — say the S&P 500 and a long-Treasury total-return index — into a frame `px` with a `DatetimeIndex`. The dashboard post [building a macro asset correlation dashboard in Python](/blog/trading/macro-correlations/building-a-macro-asset-correlation-dashboard-in-python) covers the pull-and-align plumbing; here we start from returns.

```python
import numpy as np
import pandas as pd

rets = px.pct_change().dropna()        # daily simple returns of both columns
a, b = rets["SPX"], rets["BOND"]       # the two return series

simple_corr = a.rolling(63).corr(b)    # equal-weight 63-day (~3 month) rolling corr
print(simple_corr.tail())
```

`Series.rolling(window).corr(other)` does exactly what the name says: for each date it takes the trailing `window` observations of both series and returns their Pearson correlation, with `NaN` for the first `window - 1` rows where the window is not yet full. Equal-weight means the observation from 63 days ago counts exactly as much as yesterday's. That is the estimator's great virtue (simple, unbiased within a stable regime) and its great vice: when a break happens, the old-regime observations stay in the window, dragging the estimate, until they finally age out the far end. A 63-day window therefore lags a true break by roughly **half the window**, about 30 trading days — six weeks of trading on a stale number before the rolling estimate fully catches up.

The **EWMA** (exponentially weighted moving average) correlation fixes the lag by weighting recent observations more. Instead of a hard cutoff, every past observation contributes, but its weight decays geometrically the further back it sits. pandas exposes it through `ewm`:

```python
ewma_corr = a.ewm(halflife=21).corr(b)   # exponential weight, 21-day halflife
print(ewma_corr.tail())
```

The `halflife` parameter is the intuitive knob: it is the number of periods over which an observation's weight decays to half. A 21-day half-life means the return from 21 trading days ago counts half as much as today's, the return from 42 days ago counts a quarter as much, and so on. Because nothing is ever fully discarded but old data fades fast, the EWMA estimate pivots toward a new regime sooner than an equal-weight window of comparable smoothness. The trade is more day-to-day jitter: by leaning on recent data, EWMA also leans on recent *noise*.

The figure below makes the difference concrete. It applies both estimators to a path whose true correlation steps from the curated pre-2022 stock-bond level (−0.30) to the 2022 level (+0.60) on a known date, so you can see exactly how fast each estimator notices.

![Simulated correlation path comparing a simple rolling window against an EWMA after a true break](/imgs/blogs/rolling-correlation-regimes-and-change-point-detection-5.png)

The dotted black line is the truth: a clean step from −0.30 to +0.60 at the break. The blue line (simple 90-day rolling) crawls toward the new level, half its window still full of pre-break data; it does not fully arrive at +0.60 until roughly 45 trading days after the break. The red line (EWMA, 21-day half-life) is jumpier before the break but pivots toward the new regime within a couple of weeks. In a regime change, *that* head start is the entire game: it is the difference between re-hedging in mid-2022 and re-hedging in late 2022 after the damage.

#### Worked example: how far behind the lag puts you

Suppose the stock-bond correlation truly flips from −0.3 to +0.6 on day zero, and you are running a 90-day equal-weight rolling estimate. Equal weighting means the estimate is roughly the average of the true correlation over the trailing 90 days. Thirty trading days after the break, 30 of the 90 observations come from the new (+0.6) regime and 60 from the old (−0.3) regime, so the estimate reads about (30 × 0.6 + 60 × −0.3) / 90 = (18 − 18) / 90 = **0.0**. A month past a sign flip, your rolling number still says "uncorrelated" — it has not even crossed zero yet. The estimate does not reach a clean +0.5 until roughly day 75, when 75 of 90 observations are post-break: (75 × 0.6 + 15 × −0.3) / 90 ≈ **+0.45**. The intuition: an equal-weight window literally averages the old regime into the new one for a full window length, so a long window buys you smoothness at the price of being a month-and-a-half late to a flip that can cost you on every one of those days.

### Why the EWMA reacts faster: the recursion underneath

It is worth seeing exactly *why* the exponential weighting buys you a faster reaction, because the same recursion is the spine of the CUSUM you will build for detection. An EWMA is defined by a single decay parameter λ (lambda) between 0 and 1, and it updates with one line of arithmetic per step:

```
ewma_t = (1 − λ) × x_t + λ × ewma_{t−1}
```

Each new value is a blend: a fraction (1 − λ) of today's observation plus a fraction λ of yesterday's running average. Unroll that recursion and you find that the observation from k steps ago carries weight (1 − λ) × λ^k — a geometric decay. The half-life is just the k at which that weight halves, λ^k = 0.5, so `halflife = ln(0.5) / ln(λ)`, which is why pandas lets you specify the half-life directly instead of the opaque λ. The practical consequence is that an EWMA has *no hard window edge*: there is no day on which old data abruptly falls out (the discontinuity that makes equal-weight rolling correlations occasionally jump when one extreme observation ages off the back). Instead the influence of every past day fades smoothly to nothing. When a regime breaks, the EWMA does not have to wait for the old regime to "age out" — the old regime's weight was already decaying every single day, so by the time you are a half-life past the break, the new data already dominates the average.

Computing a *correlation* this way (not just a mean) means running three EWMAs — of the two return series and of their product — and combining them into the standardized covariance, which is precisely what `a.ewm(halflife=h).corr(b)` does internally. You never have to write that out, but knowing it explains a subtle pitfall: an EWMA correlation can occasionally print a value slightly outside [−1, +1] in tiny samples or right at the start of the series, because the weighted covariance and weighted variances are estimated separately and can be momentarily inconsistent. Clip the output to [−1, 1] and discard the first few half-lives of warm-up before you trust it.

#### Worked example: the EWMA weight on a single day

You run an EWMA correlation with a 21-day half-life. The half-life fixes λ from λ^21 = 0.5, so λ ≈ 0.5^(1/21) ≈ **0.968**, and the weight on each new day is (1 − λ) ≈ **0.032**, about 3.2%. So today's return gets a 3.2% vote, yesterday's gets 3.2% × 0.968 ≈ 3.1%, the day-21-ago return gets exactly half of today's vote (1.6%), and the day-63-ago return (three half-lives back) gets one-eighth (0.4%). Sum the geometric series and roughly 90% of the total weight sits in the most recent ~70 days, even though the formula technically uses all history. Compare that to a 90-day equal-weight window, where the day-89-ago return gets the *same* 1.1% vote as today's. The intuition: the EWMA quietly concentrates its attention on the recent past without a hard edge, which is exactly what lets it pivot through a regime break a couple of weeks faster than an equal-weight window of similar overall smoothness.

## Choosing the window and the half-life

There is no correct window. There is only a bias-variance trade you must own, and naming the two failure modes makes the choice concrete.

A **short window** (or short half-life) is **low bias, high variance**. It tracks a true break almost immediately, but it also jumps around on noise: a few coincidental days of co-movement can swing a 30-day correlation by 0.4 even when nothing structural changed. You will get many false alarms. A **long window** (or long half-life) is **high bias, low variance**. It is smooth and stable and ignores noise, but it lags real breaks badly — the very property that let the 2022 risk model sleep. Formally, the variance of a rolling correlation estimate falls roughly like 1/N in the window length N, so doubling the window cuts the noise by about 30%; but the *lag* to a break grows linearly with N. You cannot have both. This is the same bias-variance tension covered for the estimator itself in [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters) — here it becomes the lever that sets your detector's responsiveness.

A practical default that many desks land on: run *two* estimators, a fast one and a slow one. The slow one (say a 1-year window, or a 63-day half-life) is your "what regime are we in" baseline — stable enough to trust. The fast one (a 63-day window, or a 21-day half-life) is your *alarm* — responsive enough to catch a flip early. You only act when the fast estimator diverges *and stays diverged* from the slow one. That divergence is exactly what the z-score formalizes in the next section.

#### Worked example: picking an EWMA half-life from how fast you must react

You decide the most stale you can tolerate being is a correlation estimate that is "half-right" within four trading weeks (20 days) of a break — that is, you want the EWMA to have moved halfway from the old level to the new one within 20 days. By construction, an EWMA reaches half of a step change in exactly one half-life. So you set `halflife=20`. Now sanity-check the noise cost. The effective number of observations an EWMA with half-life h "remembers" is roughly h / ln(2) × 2 ≈ 2.9h, so a 20-day half-life behaves a bit like a ~58-observation equal-weight window for noise purposes. A correlation estimated on ~58 points has a standard error of roughly 1/√58 ≈ **0.13** when the true correlation is near zero. So your alarm estimator will routinely wiggle by ±0.13 on noise alone — which tells you the z-score threshold must be set well above that wiggle, or you will fire on nothing. The intuition: the half-life is not a free parameter, it jointly sets your reaction speed *and* your false-alarm rate, and you must check both.

## Step one of detection: the rolling z-score of the correlation

You now have a fast rolling correlation that moves. The raw question — "has it broken?" — is too vague to code. The sharp version is: *how far is today's correlation from what it has normally been, measured in its own units of normal variation?* That is a z-score.

A **z-score** standardizes a number by subtracting a baseline mean and dividing by a baseline standard deviation:

```
z_t = (r_t − mean(r over a long trailing window)) / std(r over that window)
```

A z of 0 means today's correlation sits exactly at its recent normal. A z of +2 means it is two standard deviations *above* normal — an unusually high reading. A z of −2 means unusually low. The beauty of standardizing is that it adapts: a correlation that normally wiggles by 0.05 and one that normally wiggles by 0.30 both produce a z of +2 at the same level of *surprise*, so a single threshold works across pairs with different noise levels.

In code, you compute the fast correlation, then take its rolling mean and rolling standard deviation over a *longer* window to define "normal", then standardize:

```python
fast = a.ewm(halflife=21).corr(b)          # the responsive correlation estimate
base_mean = fast.rolling(252).mean()        # ~1y trailing mean = "normal" level
base_std = fast.rolling(252).std()          # ~1y trailing std = "normal" wiggle

zscore = (fast - base_mean) / base_std
print(zscore.tail())
```

The choice of the long baseline window (252 days ≈ one year here) sets what counts as "normal". Too short and the baseline chases the very break you are trying to detect, flattening the z-score; too long and an old regime contaminates the baseline. One year is a reasonable default for a daily series — long enough to be stable, short enough to be relevant.

The next figure applies this idea to the documented stock-bond correlation. The blue bars are the rolling z-score of the correlation against its trailing long-run mean: near zero for most of the 1990-2018 history (the correlation is wandering within its normal range), then spiking sharply positive as the 2022 inflation shock drives the correlation far above anything in its recent history.

![Rolling z-score bars and a CUSUM line crossing the alarm threshold at the inflation-regime break](/imgs/blogs/rolling-correlation-regimes-and-change-point-detection-6.png)

#### Worked example: computing a rolling z-score by hand

Take a simplified slice. Suppose the trailing-year mean of your fast stock-bond correlation is **−0.30** with a standard deviation of **0.15** (it normally wiggles between about −0.45 and −0.15). One morning the fast estimate prints **+0.20**. The z-score is (0.20 − (−0.30)) / 0.15 = 0.50 / 0.15 ≈ **+3.3**. A reading of +3.3 means today's correlation is three-and-a-third standard deviations above its own recent normal — an event that, if the correlation were stationary and Gaussian, would happen well under once in a thousand days. That is not noise; that is the series telling you the ground moved. Contrast it with a print of −0.20, which gives (−0.20 − (−0.30)) / 0.15 ≈ **+0.7** — well inside the normal band, ignore it. The intuition: the z-score converts "the correlation looks high" into a calibrated surprise you can threshold, and it does so in units that travel across every pair you track.

## Step two: why a z-score alone is not enough, and the persistence filter

If you simply fired a "break detected" alarm every time the z-score crossed +2, you would be whipsawed to death. A noisy fast estimator crosses ±2 by chance all the time. A single day above the line is not a regime change; it is a coincidence. The fix is the same instinct any seasoned trader has: *don't act on the first tick, wait for confirmation.* In detection terms, you require the deviation to **persist**.

The crudest persistence filter is a counter: only flag a break once the z-score has stayed beyond the threshold for N consecutive periods. It works, and for many purposes it is enough:

```python
THRESH, N = 2.0, 20
beyond = (zscore.abs() > THRESH).astype(int)   # 1 on any day past the line
streak = beyond.groupby((beyond == 0).cumsum()).cumsum()   # length of current run
break_flag = streak >= N                        # True only after N straight days
print(break_flag[break_flag].head())            # the dates a break is confirmed
```

That `groupby((beyond == 0).cumsum())` trick is a standard pandas idiom for run-length: every time `beyond` drops to zero it starts a new group, and the cumulative sum within each group counts the length of the current streak of past-threshold days. The break flags only when that streak reaches N. A one-day or even ten-day blip never fires; a sustained shift does. The cost is obvious — you are deliberately late by N periods in exchange for almost never crying wolf.

The more elegant persistence filter, and the one worth knowing by name, is the **CUSUM** (cumulative sum). Instead of a hard streak counter, a CUSUM *accumulates* the standardized deviations, draining off a small "slack" each step so that pure noise cancels out but a sustained drift builds up and eventually trips an alarm. For detecting an upward shift:

```python
k, H = 0.5, 4.0          # slack (ignore drift < 0.5 std) and decision threshold
S = np.zeros(len(zscore))
for i in range(1, len(zscore)):
    zi = 0.0 if np.isnan(zscore.iloc[i]) else zscore.iloc[i]
    S[i] = max(0.0, S[i - 1] + zi - k)      # accumulate, but never below zero
    if S[i] > H:
        print("upward break detected at", zscore.index[i]); break
```

Read the recursion plainly. Each step you add today's z-score to the running sum, subtract the slack `k`, and floor the result at zero. When the correlation is at its normal level, the z-scores hover around zero, each step subtracts more slack than it adds, and the sum stays pinned at the floor — no alarm. When the correlation genuinely shifts up, the z-scores turn persistently positive, each step adds more than the slack drains, and the sum climbs. The moment it climbs past the decision threshold `H`, you declare a change point. The two knobs trade the same way as always: a larger `k` ignores more drift (fewer false alarms, slower detection); a larger `H` requires more accumulated evidence (later but surer).

In the z-score-and-CUSUM figure above, the lavender line is exactly this CUSUM running on the curated stock-bond correlation. It sits flat at zero through every wiggle of the 1990-2018 era — the bars cross neither alarm line in a way that accumulates — and then, as the 2022 z-scores spike past +2 and *stay* there, the CUSUM rockets through its H = 4 threshold and the break is flagged. The red marker is the change point: the date the detector says "stop trusting the negative correlation." A naive "z > 2 once" rule would have fired and un-fired a dozen times before this on noise; the CUSUM fired exactly once, at the real break.

#### Worked example: a change-point trigger firing on a sustained shift

Walk the CUSUM through a clean shift with k = 0.5 and H = 4.0. Before the break the z-scores are small, say a run of +0.3, −0.1, +0.2: each step adds the z and subtracts 0.5, so 0.3 − 0.5 = −0.2 → floored to 0; the sum never leaves zero. Now the regime breaks and the z-scores turn persistently positive — say +1.5 every day. Step by step the CUSUM climbs: 1.5 − 0.5 = 1.0, then 1.0 + 1.0 = 2.0, then 3.0, then 4.0, and on the *fourth* such day it crosses H = 4.0 and fires. So with a sustained +1.5-sigma shift the CUSUM needs four periods to confirm — fast enough to be useful, slow enough that a one- or two-day fluke (which would be floored back toward zero on the first sub-slack day) never trips it. The intuition: the CUSUM is a leaky bucket — noise leaks out through the slack, but a steady inflow eventually fills it, and the fill time *is* your detection lag.

## What the flag is worth: the dollar cost of a stale correlation

The reason any of this matters is that a correlation you trust past its break is not a harmless modeling error — it sizes real positions wrong, and the bill arrives in dollars. The cleanest way to feel it is the 60/40 portfolio, the most widely held correlation bet on earth, going into 2022.

The figure below shows the stock-bond rolling correlation with the two breaks the detector exists to catch: the 1998-2000 flip *down* into the diversifying era, and the 2022 flip *up* into the inflation regime. The whole comfortable premise of 60/40 — that bonds zig when stocks zag — lives entirely in the green (negative-correlation) band, and the detector's value is telling you the day the line crosses into the red band above zero.

![Stock-bond rolling correlation with the 1998 and 2022 regime breaks circled](/imgs/blogs/rolling-correlation-regimes-and-change-point-detection-2.png)

#### Worked example: the dollar cost of trading a decayed correlation

Take a simple \$100 balanced book: \$60 in stocks, \$40 in long Treasuries. The reason you hold the bonds is diversification — when stocks fall, you expect bonds to rise and cushion the blow. That expectation is correct *only while the correlation is negative.* Through 2022, stocks fell about 25% and long Treasuries fell about 30%. If your risk model still carried the old **−0.3** correlation, it told you to expect the bonds to be up while stocks were down — so on a 25% equity drop, a −0.3 correlation with the bonds' volatility would have you *expecting* the bond sleeve to gain a few percent and offset perhaps **\$3 to \$5** of the equity loss. Instead the correlation was **+0.6** and the bonds fell *with* stocks. Your \$60 equity sleeve lost about \$15 (25% of \$60) and your \$40 bond sleeve lost about \$12 (30% of \$40), for a total of about **\$27 per \$100** — and crucially the bond loss was not an unlucky tail, it was the *predictable* consequence of a correlation that had already flipped. A detector that flagged the +0.6 regime in early 2022 would have told you to cut bond duration or buy a different hedge *before* the \$12 bond loss, not after. The intuition: the dollar cost of a stale correlation is not abstract slippage — it is the entire hedge you thought you had, evaporating exactly when you reach for it.

This is the same mechanism that the cross-asset series covers from the portfolio-construction side in [stock-bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) and [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis). The detector is the operational layer underneath those posts: not "the correlation can flip" but "here is the date it flipped, here is the code that found it."

## What drives the break: conditioning the detector on the regime

A detector that only watches the correlation itself is reactive — it tells you the flip happened, but only after it accumulates enough evidence. You can often see the break coming, or at least confirm it faster, by watching the *macro variable that drives the correlation*. For the stock-bond pair, that driver is the inflation regime, and the relationship is sharp enough to chart.

![Stock-bond correlation conditioned on the inflation regime showing the sign flip above four percent](/imgs/blogs/rolling-correlation-regimes-and-change-point-detection-7.png)

When average core inflation sits below 2%, the stock-bond correlation is firmly negative (about −0.45): inflation is not the market's worry, so every scare is a growth scare, the Fed cuts, and bonds rally as stocks fall — bonds hedge. As inflation climbs through 3-4% the correlation crosses zero, and above 4% it is firmly *positive* (about +0.50): now inflation *is* the worry, a hot print sells off both stocks and bonds together (higher rates hurt both), and the hedge fails. This is why the 2022 break was, in hindsight, detectable from the inflation data alone: core inflation crossing ~4% is a leading tell that the stock-bond correlation is about to flip positive. The mechanism behind this — why the inflation regime sets the sign — is developed in [the stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) and, on the rates side, in [real vs nominal: inflation, real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).

The practical upgrade: run the correlation detector *and* a simple inflation-regime flag side by side. When the inflation flag flips (core CPI crosses your threshold) and the correlation z-score starts climbing, you have two independent confirmations and you can act on a shorter CUSUM streak. Conditioning on the driver is how you buy back some of the detection lag you paid for in the persistence filter.

```python
inflation_hot = core_cpi_yoy > 4.0                  # the driver flag
corr_breaking = (zscore > 2.0) & (cusum > 2.0)      # the correlation flag
high_confidence = inflation_hot & corr_breaking     # both agree -> act sooner
```

#### Worked example: a half-life choice that survives the inflation regime

You are tracking the BTC-Nasdaq correlation (next section) and you want a detector that would have caught its 2022 spike but not been whipsawed by crypto's day-to-day noise. Crypto returns are roughly twice as volatile as equities, so a correlation estimated on them is noisier — at a given window length the standard error is similar but the *underlying* correlation moves around more. Suppose you measure that BTC's true macro correlation can shift meaningfully (by ~0.3) over about two months when a regime turns. To catch a two-month true shift while smoothing daily noise, you want a half-life around half that horizon — say **21 days** (one trading month) — so the estimator is half-adjusted within a month and fully within ~10 weeks, fast enough to flag the 2022 spike by mid-year. A 5-day half-life would have caught it a week sooner but fired on a dozen noise spikes first; a 63-day half-life would have been so smooth it confirmed the spike only as the spike was already fading. The intuition: match the half-life to the *speed of the break you care about*, then verify the implied noise (≈ 1/√(2.9 × halflife)) is small enough that your z-score threshold clears it.

## A second case: the BTC-Nasdaq correlation that appeared and faded

The stock-bond flip is a *sign* change. The detector handles a different shape just as well: a correlation that switches on, peaks, and fades — a *strength* regime. Bitcoin versus the Nasdaq is the textbook case.

![Bitcoin versus Nasdaq rolling correlation spiking in 2022 then fading](/imgs/blogs/rolling-correlation-regimes-and-change-point-detection-4.png)

Before 2020, Bitcoin's correlation with the Nasdaq was essentially zero — crypto marched to its own drummer, an idiosyncratic asset with no macro anchor. Then, as institutional money arrived and the 2022 liquidity drain hit every risk asset at once, Bitcoin started trading as a high-beta macro-liquidity bet, and its 90-day correlation with the Nasdaq spiked to about **+0.65** in 2022. The detector's z-score on this series climbs through the period the amber circles mark — the "elevated-correlation regime" where Bitcoin is, for trading purposes, just a leveraged Nasdaq. Then through 2023-24 the correlation faded back toward **+0.2** to **+0.3** as crypto regained some idiosyncratic life. The deeper mechanism — why Bitcoin trades on liquidity in some regimes and not others — is in [crypto as a macro asset: the liquidity correlation](/blog/trading/macro-correlations/crypto-as-a-macro-asset-the-liquidity-correlation).

The lesson for the detector is that "break" is not only a sign flip. A correlation moving from 0.0 to 0.65 is just as much a regime change as one moving from −0.3 to +0.6, and the same z-score-plus-persistence machinery catches it: the z-score measures distance from the *recent normal*, which near zero was tight, so a climb to 0.65 reads as a large positive z exactly as a sign flip would. You do not need a different detector for "strength changed" versus "sign changed" — both are the correlation leaving its normal band and staying out.

## The third dimension: detecting a break in the lead/lag, not just the level

This series insists that every correlation has four properties — a sign, a strength, a *lead/lag*, and a regime — and so far the detector has only watched the first two. The lead/lag can break too, and it is the most insidious break because the *contemporaneous* correlation can look perfectly stable while the timing relationship quietly inverts. Two series can stay 0.6-correlated at lag zero while the one that used to *lead* becomes the one that *lags*, which destroys any trade that relied on the leading series as a signal.

The tool is the same rolling correlation, computed not at lag zero but at a *shift*. In pandas you simply shift one series before correlating:

```python
def lead_lag_profile(x, y, max_lag=10, window=126):
    out = {}
    for k in range(-max_lag, max_lag + 1):
        out[k] = x.rolling(window).corr(y.shift(k)).iloc[-1]
    return pd.Series(out)            # index = lag in periods, value = corr at that lag

profile = lead_lag_profile(ism_change, spx_eps_growth)
best_lag = profile.idxmax()         # the lag where the relationship is strongest
```

`y.shift(k)` slides the second series forward by k periods, so a positive `k` that maximizes the correlation means `x` *leads* `y` by k periods — `x` today best explains `y`'s value k periods from now. The series' lead/lag post, [lead/lag: leading, coincident, and lagging indicators](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators), catalogs the documented leads: ISM new orders lead S&P earnings growth by about six months, the yield-curve inversion leads recession by over a year, credit spreads lead equity drawdowns by a few months. A lead/lag *break* is when the `best_lag` itself moves — say a six-month lead compresses to one month as markets learn to front-run an indicator, which is exactly what happens to popular signals over time.

You detect a lead/lag break the same way you detect a level break: compute the `best_lag` (the argmax of the profile) on a rolling basis, and run a change-point flag on *that* series. When the argmax jumps and stays jumped, the timing relationship has restructured. The practical warning this gives you is sharp: a signal whose lead has decayed to zero is no longer a *signal*, it is a coincident reading with no predictive value, and you should retire it before it costs you. This is the timing analogue of the stale-beta problem — a lead that decayed is just as dangerous as a sign that flipped.

#### Worked example: a decaying lead and what it costs

Suppose you trade equity-cyclical exposure off ISM new orders, which historically led S&P earnings growth by six months — you buy cyclicals when ISM turns up, expecting earnings to follow half a year later, and you size the trade as if you have a six-month head start. Now run the rolling lead/lag profile and find the `best_lag` has slid from +6 months to +1 month over two years: the market learned to price the ISM read almost immediately. Your "six-month head start" is now a one-month head start, which means five of the six months of edge you were sizing for no longer exist — you are buying cyclicals that have already moved. If the original trade captured, say, a 4% relative outperformance over the six-month window, the compressed version captures a fraction of that, and after costs it may be negative. The detector that flagged the `best_lag` break would have told you to retire the ISM-lead trade — the relationship did not vanish, its *timing* collapsed, and only a lead/lag detector would have caught it because the contemporaneous correlation looked unchanged the whole time.

## A sketch with sklearn: segmenting the series without picking a window

Everything above is online (you process one new day at a time, never looking ahead). Sometimes you want the offline view: given a *completed* history, where are all the break points, found without you nominating a window or threshold? A clustering sketch with scikit-learn gives a quick, look-ahead-aware answer that is useful for labeling the past, not for live trading. (statsmodels is not used here — pure sklearn.)

```python
from sklearn.cluster import KMeans

roll = a.rolling(126).corr(b).dropna()   # cluster this level into "regimes"
X = roll.values.reshape(-1, 1)
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
regime = pd.Series(km.labels_, index=roll.index)

changes = regime[regime != regime.shift(1)].index   # dates the cluster label flips
print("offline change points:", list(changes))
```

This clusters the rolling-correlation *level* into k regimes (here three: a strongly negative, a near-zero, and a positive cluster) and reports every date the assigned cluster changes. It is crude — it ignores time ordering and the right k is a judgment call — but it cheaply confirms what the CUSUM found and is handy for *labeling* historical regimes when you build a regime-conditioned model. The crucial caveat: this is **look-ahead** detection. KMeans sees the whole series at once, including the future, so it can place a change point exactly at the break with perfect hindsight. You must never confuse this offline accuracy with what your live CUSUM can do, which is always somewhat late. Binary segmentation (recursively splitting the series at the single most likely break, then recursing on each half) is the same idea done properly with time order respected, and the PELT algorithm finds the optimal set of change points efficiently — both worth knowing if you go deeper, but the z-score-plus-CUSUM online detector is enough for the trading job.

## Putting it together: the whole detector in one function

Here is the complete online detector, every piece from the sections above assembled into one function you can drop on any pair of return series. It returns a tidy frame with the fast correlation, its z-score, the CUSUM, and the binary break flag — the four columns you would plot or alert on.

```python
import numpy as np
import pandas as pd

def detect_corr_break(a, b, halflife=21, base_win=252,
                      z_thresh=2.0, k=0.5, H=4.0):
    """Online correlation regime detector for two aligned return series."""
    fast = a.ewm(halflife=halflife).corr(b).clip(-1, 1)   # responsive estimate
    base_mean = fast.rolling(base_win).mean()             # trailing "normal" level
    base_std = fast.rolling(base_win).std()               # trailing "normal" wiggle
    z = (fast - base_mean) / base_std

    cusum = np.zeros(len(z))
    for i in range(1, len(z)):
        zi = 0.0 if np.isnan(z.iloc[i]) else z.iloc[i]
        cusum[i] = max(0.0, cusum[i - 1] + abs(zi) - k)   # accumulate |drift|
    cusum = pd.Series(cusum, index=z.index)

    flag = cusum > H                                       # break confirmed where True
    return pd.DataFrame({"corr": fast, "z": z, "cusum": cusum, "break": flag})

out = detect_corr_break(rets["SPX"], rets["BOND"])
breaks = out.index[out["break"] & ~out["break"].shift(1).fillna(False)]
print("regime breaks flagged on:", list(breaks))
```

Read what it does end to end. It builds the fast EWMA correlation and clips it to the valid range; it standardizes that into a z-score against a one-year trailing baseline; it runs the leaky-bucket CUSUM on the absolute z-score so that a sustained deviation in *either* direction (a sign flip or a strength surge) accumulates; and it flags the dates the CUSUM clears H. The final line extracts only the *first* day of each break (where the flag turns on but was off the day before), so you get one alert per regime change instead of a flag that stays lit. That is the whole toolkit — perhaps thirty lines — and it is enough to have told the 2022 risk officer, months early, to stop trusting his **−0.3**.

Two production notes the bare function omits. First, **warm-up**: the first `base_win` days have a `NaN` baseline and an unreliable z-score, so never alert during the warm-up period — slice it off. Second, **alert hygiene**: once a break fires and you re-fit to the new regime, reset the CUSUM to zero, otherwise it stays above H and re-fires on every subsequent day. The reset is the code equivalent of "you already acted on this break; start watching for the next one."

#### Worked example: tuning H from your tolerated false-alarm rate

You want the detector to throw at most one false alarm per year of calm trading (≈ 252 days). With slack k = 0.5 and a standardized input, a one-sided CUSUM's average run length to a false alarm grows roughly exponentially in H — a rule of thumb is ARL ≈ e^(2kH) / (2k²) for the standardized case. Set H = 4: 2kH = 2 × 0.5 × 4 = 4, so e^4 ≈ 55, divided by 2 × 0.25 = 0.5 gives ARL ≈ **110 days** between false alarms — a bit too trigger-happy, roughly two false alarms a year. Bump H to 5: 2kH = 5, e^5 ≈ 148, / 0.5 ≈ **296 days**, just over one false alarm per year — your target. The intuition: H is the dial that prices detection lag against false alarms on an exponential curve, so a small increase in H buys a large drop in false alarms at the cost of a few extra days of lag — and you set it from how much crying-wolf your process can tolerate, not by eyeballing a chart.

## Common misconceptions

**"A bigger window is always safer because it's smoother."** No — smoothness is bias in disguise. The 2022 risk model was smooth and stable and *wrong*, precisely because its long window averaged the dead regime into the live one. A 5-year correlation window will still be reporting "negative, bonds hedge" a year after the bonds stopped hedging. Smoothness reduces noise *and* reduces your ability to see the break; it is not free safety, it is a trade you must price against your reaction-speed need.

**"If the rolling correlation crosses zero, the regime has changed."** No — a single crossing is noise until it persists. An unfiltered fast correlation crosses zero constantly near a regime boundary. The whole point of the z-score and the CUSUM is to distinguish a crossing that *stays* (a break) from one that snaps back (a wiggle). Acting on the first crossing is how you get whipsawed into re-hedging four times for one real flip.

**"The full-sample correlation is the 'true' long-run value and the rolling one is just noise around it."** Exactly backwards when regimes flip. The full-sample number is a meaningless average of incompatible states — gold versus real yields at **−0.01** is not a "long-run truth," it is the arithmetic mean of −0.96 and +0.80, which describes no period that ever existed. The rolling number, regime by regime, is the real thing.

**"Change-point detection finds the break the day it happens."** No — every honest *online* detector is late by construction, because it needs to accumulate enough post-break evidence to rule out noise. The CUSUM's fill time *is* its lag. The offline KMeans/PELT view can pin the break exactly, but only because it cheats by seeing the future. If a method claims real-time detection with zero lag, it is either using look-ahead data or it is firing on noise.

**"A correlation break only matters for the sign."** No — a strength change (BTC-Nasdaq going from 0.0 to 0.65) breaks a diversification assumption just as badly as a sign flip. A position you sized as "uncorrelated, so it diversifies" becomes "0.65-correlated, so it concentrates" without the sign ever changing. The detector watches distance from normal, which catches both.

## How it shows up in real markets

**The 2022 stock-bond flip (the canonical case).** Through 2021 the trailing correlation sat near −0.3. By Q1 2022 the fast (EWMA) estimate had crossed zero; by mid-2022 it was near +0.6 and the CUSUM had long since fired. A desk running this detector would have flagged "stop trusting the −0.3" in the first or second quarter — months before the full-year carnage of stocks **−25%** and long bonds **−30%** together was complete. The full case study is [when correlations break: the 2022 stock-bond flip](/blog/trading/macro-correlations/when-correlations-break-the-2022-stock-bond-flip).

**Gold decoupling from real yields (2022-24).** The cleanest macro correlation in the book, **−0.96** for fifteen years, broke to **+0.80** as central-bank buying overwhelmed the real-yield channel. A detector watching the rolling correlation of gold returns against real-yield changes would have flagged the regime shift in 2022-23, well before a naive "gold is a real-yield play" model walked into two years of being wrong about the sign. The mechanism is in [inflation and gold: the real-yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story).

**Crisis correlation spikes (2008, 2020).** In a deleveraging crisis, *every* pairwise correlation across risk assets rushes toward +1 within days — far too fast for a slow window to catch, which is exactly when you most need to know. The average pairwise correlation across major risk assets sits near **0.25** in calm markets, rises to about **0.45** in an ordinary selloff, and jumps to roughly **0.80** in a full deleveraging crisis like 2008 or March 2020. A fast EWMA (short half-life) plus a low CUSUM threshold is the configuration for crisis monitoring: you accept more false alarms in calm markets in exchange for catching the everything-correlates regime while it still matters. Note the tension with the persistence filter — a crisis break is exactly the case where you *cannot* afford to wait twenty days for confirmation, so for crisis monitoring you deliberately run a short half-life, a low H, and accept the extra false alarms as the price of speed. The detector you run on a slow strategic book and the detector you run for crisis tripwires are the *same code with different knobs*, and choosing the knobs is the real skill. This crisis behavior is [correlation during crises: when diversification fails](/blog/trading/macro-correlations/correlation-during-crises-when-diversification-fails).

**The BTC macro-on / macro-off switch.** Bitcoin's correlation with the Nasdaq is itself regime-dependent — high when liquidity is the dominant macro force, low when crypto-native flows dominate. A standing detector on the BTC-Nasdaq pair tells you which regime you are in *for sizing purposes*: in the high-correlation regime you cannot count BTC as a diversifier against your tech book, because it is the same trade.

## How to read it and use it

Here is the operating procedure, distilled.

**Estimate fast and slow.** Compute a slow correlation (a 1-year window or a 63-day half-life) as your "what regime are we in" baseline, and a fast one (a 63-day window or a 21-day half-life) as your alarm. Always on returns, never price levels.

**Standardize.** Build the z-score of the fast correlation against the slow baseline's trailing mean and standard deviation. The z-score is the language your threshold speaks; it makes one rule work across every pair.

**Filter for persistence.** Run a CUSUM (or, if you want the simplest possible thing, a "stayed beyond ±2 for N days" counter) so a single-day cross never fires. Set the slack `k` and threshold `H` from how much false-alarm tolerance you have: tighter for crisis monitoring, looser for slow strategic re-hedging.

**Confirm with the driver.** Where you know the macro variable that drives the correlation — the inflation regime for stocks-bonds, real yields for gold, liquidity for crypto — watch it in parallel. Two independent flags let you act on a shorter streak and cut your lag.

**Then act on the flag, not the wiggle.** The flag is binary and it means one thing: *stop trusting the old beta, re-fit to the new regime.* Re-estimate your hedge ratio, resize the position, or step aside until the new regime is clear. Do not re-derive the trade on every noisy crossing — that is the whipsaw the CUSUM exists to prevent.

**What invalidates the whole exercise.** Look-ahead bias (any baseline, threshold, or window you tuned on data that includes the break is cheating, and it will make your detector look prescient in backtest and useless live). Non-stationarity in the *noise* (if the correlation's volatility itself changes regime, your z-score's denominator is wrong — re-estimate the baseline std on a trailing window, never a full-sample one). And the deepest one: the detector tells you *that* the correlation changed, never *why*. Pair it with the mechanism — the macro driver — before you bet, because a correlation that broke for a transient reason will revert, and one that broke for a structural reason will not, and only the *why* tells you which. For the stationarity machinery underneath all of this, see the math-for-quants treatment of [stationarity and why it matters for time series](/blog/trading/math-for-quants/stationarity-autocorrelation-math-for-quants). The next step — turning a detected regime into an actual position tilt without overfitting — is [from correlation to signal: building a macro overlay](/blog/trading/macro-correlations/from-correlation-to-signal-building-a-macro-overlay).

The single discipline that ties it together: a correlation has a sign, a strength, a lead/lag, *and a date* — and the detector's only job is to find the date the old number died, so you stop trading on a relationship that no longer exists.

## Further reading and cross-links

Within this series:
- [Rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters) — the estimator and the bias-variance window choice this post's detector sits on top of.
- [Structural shifts: why today's correlations aren't yesterday's](/blog/trading/macro-correlations/structural-shifts-why-todays-correlations-arent-yesterdays) — *why* the breaks this detector catches happen.
- [Spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data) — why you correlate returns, never price levels, and other ways the number lies.
- [Building a macro asset correlation dashboard in Python](/blog/trading/macro-correlations/building-a-macro-asset-correlation-dashboard-in-python) — the pull-and-align plumbing that feeds this detector.
- [From correlation to signal: building a macro overlay](/blog/trading/macro-correlations/from-correlation-to-signal-building-a-macro-overlay) — turning a detected regime into a sized position without overfitting.
- [When correlations break: the 2022 stock-bond flip](/blog/trading/macro-correlations/when-correlations-break-the-2022-stock-bond-flip) and [the stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) — the canonical break, in depth.

Cross-series:
- [Stock-bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) and [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — the portfolio-construction view of a broken correlation.
- [Real vs nominal: inflation, real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — the inflation/real-yield driver that conditions these breaks.
- [Stationarity and the tests that detect it](/blog/trading/math-for-quants/stationarity-autocorrelation-math-for-quants) — the time-series foundation under change-point detection.
