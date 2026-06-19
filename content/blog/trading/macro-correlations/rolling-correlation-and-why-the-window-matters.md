---
title: "Rolling Correlation: Why the Window Is the Whole Game"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why a single full-sample correlation lies, how a rolling or EWMA correlation reveals the regime you are actually in, and why choosing the window length is a bias-variance decision with no free lunch."
tags: ["macro", "correlation", "rolling-correlation", "ewma", "regime", "stock-bond-correlation", "gold", "real-yields", "bitcoin", "bias-variance", "window-length", "diversification"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A correlation is not a constant you measure once; it is a moving target you must track over a rolling window, and the window length you pick decides what you see. Measure it once over the whole sample and you average opposite regimes into a meaningless number.
>
> - The single most dangerous number in markets is a **full-sample correlation**. Over 2007 to 2025, the correlation between gold and the 10-year real yield is **−0.01** — apparently no relationship at all. But 2007 to 2021 it was **−0.96** (one of the cleanest relationships in macro), and 2022 to 2025 it was **+0.8**. The average of a near −1 and a strong +0.8 is roughly zero, and that zero hides everything.
> - The fix is a **rolling correlation**: re-estimate the correlation every period using only the last N observations. The plot of that number over time shows the **sign flip** that a single number erases.
> - The window length N is a **bias-variance choice**. A short window (say 30 to 60 days) is noisy and whippy but spots a regime change in weeks; a long window (3 to 5 years) is smooth and stable but lags the flip by a year or more. There is no "correct" N — only a tradeoff you must own.
> - The one fact to remember: a correlation has a **sign, a strength, a lead/lag, and a date**. Drop the date and you are trading on a number that may have flipped while you were averaging.

In October 2022 a portfolio manager I know was staring at a chart that, by every textbook, should not have existed. His stocks were down about 25% on the year. His long-dated Treasury bonds — the things that were supposed to *protect* him when stocks fell — were down about 30%. The two halves of his "balanced" portfolio had fallen together, hard, in the same year. The entire premise of the classic 60/40 portfolio is that bonds zig when stocks zag. His risk model, fitted on twenty years of data, had told him the stock-bond correlation was about −0.3. Comfortably negative. Bonds would cushion the fall.

The risk model was not wrong about the past. It was wrong about *which* past. It had averaged the deeply negative correlation of 2000 to 2021 — a period when every market scare was a growth scare, and the Fed responded by cutting rates, sending bond prices up while stocks fell — together with the brief, violent positive correlation of an inflation shock. Over the whole sample the number came out negative, and so the model slept soundly. But by mid-2022 the *live* correlation, measured over the trailing two years, had flipped to roughly **+0.6**. The relationship his portfolio depended on had reversed sign, and the single full-sample number he was trusting had no way to tell him, because a single number has no clock.

This post is about that clock. It is about why you must measure a correlation over a *moving window* rather than once, why the length of that window is the most consequential and least-discussed choice in the whole exercise, and how to read a rolling-correlation chart so that you are reacting to the regime you are in rather than the average of every regime you have ever been in.

![Static full-sample correlation versus a rolling-window correlation, contrasted side by side](/imgs/blogs/rolling-correlation-and-why-the-window-matters-1.png)

## Foundations: what a correlation actually is, and why it needs a window

Before we can talk about *rolling* a correlation, let us be painfully clear about what a correlation *is* — because the whole argument of this post hinges on a subtlety most introductions skip.

**Correlation measures whether two things tend to move in the same direction.** If stocks usually rise on the days bonds rise, and fall on the days bonds fall, they are *positively* correlated. If stocks usually rise when bonds fall, they are *negatively* correlated. If knowing what stocks did today tells you nothing about what bonds did, they are *uncorrelated*. The standard yardstick is the **Pearson correlation coefficient**, written *r* (or sometimes ρ, the Greek letter rho), and it always lands between −1 and +1.

- **r = +1**: the two move in perfect lockstep, same direction, every time.
- **r = 0**: no linear relationship — the moves are unrelated.
- **r = −1**: they move in perfect opposition — one up exactly when the other is down.

The formula, if you want to see it, is the *covariance* of the two series divided by the product of their standard deviations. We build that up carefully in the companion post on [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta); here I only need you to hold one idea: **to compute a single correlation number, you must feed it a fixed set of observations.** You hand it, say, the daily returns of stocks and bonds over some span of time, and it gives you back one number summarising *that span*.

One technical note that matters more than it looks. You almost always correlate *returns* (the percentage change from one period to the next), not *price levels*. Correlating raw price levels is one of the classic ways to manufacture a fake correlation — two assets that both happen to drift upward over a decade will show a high "correlation" of their levels even if their day-to-day moves are completely unrelated, simply because both lines slope up. That is a *spurious* correlation born of shared trend, and it is the subject of [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data). Throughout this post, when I say "correlation" I mean the correlation of returns or changes, the quantity that actually tells you whether two things move together rather than whether they both happened to grow.

And there is the rub. A correlation is not a property of two assets the way mass is a property of an object. It is a property of *two assets over a particular stretch of time*. Change the stretch and you can change the number entirely — not by a rounding error, but from −0.96 to +0.8. The number you get is an average over whatever window you fed it. If that window contains two regimes with opposite signs, the number you get back is a blend that describes *neither*.

#### Worked example: how a single number averages a flip into nonsense

Take the cleanest case in all of macro: gold versus the 10-year real yield (the yield on an inflation-protected Treasury, i.e. the interest rate *after* stripping out expected inflation). The intuition is simple and we cover the mechanism in [real yields, the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything): gold pays no interest, so when real yields rise, the opportunity cost of holding a lump of metal goes up, and gold should fall. Negative relationship.

Now do the arithmetic on the curated annual data. Over **2007 to 2021**, the correlation between the real yield and the gold price is **−0.96** — about as close to a deterministic law as macro ever gets. Over **2022 to 2025**, central banks (especially in emerging markets) bought gold aggressively for reasons that had nothing to do with the real yield, and the correlation flipped to **+0.8** — gold *rose* even as real yields climbed from −0.95% to +2.05%.

What does the full sample, 2007 to 2025, give you? Run the Pearson formula over all 19 years at once and you get **r = −0.01**. Essentially zero. A naive analyst glancing at that number would conclude gold and real yields have *no relationship* and move on. They would be wrong twice over: there was a ferociously strong relationship for fourteen years, and then a strong opposite one for four. The full-sample average is the one description of the data that is true for no single period in it.

![Full sample correlation lies gold versus real yield split into two periods](/imgs/blogs/rolling-correlation-and-why-the-window-matters-4.png)

The bar chart makes the point visceral. The two regime bars tower in opposite directions — a deep green −0.96 and a tall red +0.8 — while the full-sample bar in the middle is a barely-visible sliver at −0.01. The honest signal is the *two tall bars*; the dishonest summary is the sliver, and the sliver is the only number a one-shot estimate would ever report. If you measured the relationship once and walked away, you would walk away with the one bar that means nothing.

The intuition: averaging a −0.96 era with a +0.8 era gives you roughly zero, and that zero is not "no relationship" — it is "two relationships cancelling on paper." The number lies by omission; it drops the date.

This is the entire motivation for a **rolling correlation**. Instead of one number for the whole history, you compute a *separate* correlation for each point in time using only the most recent N observations, then plot that sequence. The result is a line — correlation as a function of time, r(t) — and that line can show you the −0.96, the +0.8, and the moment they swapped. The full-sample r erases all of it into a single dot.

### Two reasons a correlation moves: noise and regime

It pays to separate the two distinct reasons a measured correlation can change from one window to the next, because confusing them is the single most common error in reading these charts.

The first reason is **sampling noise**. Any correlation computed from a finite sample is an *estimate* of some underlying truth, and like every estimate it has error bars. Even if the true correlation were frozen forever at exactly 0, a sample of 30 returns would, just by chance, hand you a measured correlation that wanders — +0.2 here, −0.3 there — for no reason except the luck of the draw. This is pure statistical jitter. It is not telling you the world changed; it is telling you that 30 observations is not very many. The smaller the window, the larger this jitter.

The second reason is a **genuine regime change**: the underlying truth itself moved, the way the stock-bond relationship really did flip from −0.4 to +0.6 in 2022. This is signal, not noise, and it is the thing you actually want to detect.

The whole difficulty of rolling correlation is that *these two look identical on the chart*. A 30-day correlation jumping from −0.2 to +0.3 might be a real regime change you must act on, or it might be sampling noise you must ignore — and you cannot tell which from the jump alone. A longer window suppresses the noise so that what remains is more likely to be signal, but at the cost of the lag we will meet shortly. Holding this distinction — noise versus regime — in your head is the prerequisite for everything that follows.

#### Worked example: how noisy is a short-window correlation, really?

You can put a number on the jitter. The standard error of a Pearson correlation estimate is approximately (1 − r²) / √(N − 1), where N is the number of observations in the window. Plug in a window of N = 30 and a true correlation of r = 0: the standard error is about 1 / √29 ≈ **0.186**. That means a 30-day window measuring two genuinely *uncorrelated* assets will typically report correlations bouncing within roughly ±0.19 of zero, and a two-standard-error swing — well within normal — spans nearly **±0.37**. You will see "correlations" of +0.35 and −0.35 appear and vanish purely from randomness.

Now lengthen the window to N = 250 (one year). The standard error drops to 1 / √249 ≈ **0.063** — about a third as much. The same two uncorrelated assets now report correlations hugging zero within roughly ±0.06. The noise shrinks with the square root of the window length: to halve the jitter you must *quadruple* the window.

The intuition: a short window does not just respond faster — it lies more often, because its error bars are wide enough to manufacture correlations out of thin air. That ±0.19 of phantom correlation at N = 30 is exactly why a fast window "cries wolf," and exactly the noise you are buying when you buy responsiveness.

## How a rolling correlation works: a window that slides

The mechanic is almost embarrassingly simple, which is part of why it gets skipped. You pick a **window length** — a number of observations, N. To get the correlation "as of today," you take only the last N data points (today and the N−1 before it), compute their Pearson correlation, and plot that value at today's date. Tomorrow you slide the window forward one step: drop the oldest point, add the newest, recompute. You repeat for every date in your history. The window *rolls*.

If N = 60 trading days, then the value you plot for, say, March 31 is the correlation of the two assets' returns over roughly the previous three months. The value for April 1 is the correlation over the previous three months ending one day later. As you walk forward, old observations fall off the back of the window and new ones enter at the front. The plotted line is the correlation *as it would have appeared to someone who only had the trailing N days at each moment* — which is exactly the situation you are in when you trade.

That last point is worth pausing on, because it is the reason a rolling estimate is the *only honest* one for decision-making. When you actually make a trade, you do not have the future, and you do not have the convenience of a full-sample average computed with hindsight over data that includes years you have not lived through yet. You have the trailing window — the recent past — and nothing else. A rolling correlation reconstructs exactly that information set at every point in history, which is why it is the right tool for asking "what would I have known, and when?" A full-sample number, by contrast, quietly smuggles in future data: the −0.01 for gold and real yields uses 2022 to 2025 to compute the "correlation" you would supposedly have acted on in 2015, which is nonsense, because in 2015 the future had not happened. Rolling estimates respect the arrow of time; full-sample estimates do not.

In code, the whole thing is one line of pandas. You almost never compute the formula by hand:

```python
import pandas as pd

>>> rets = pd.DataFrame({"stocks": stock_returns, "bonds": bond_returns})
>>> rolling_corr = rets["stocks"].rolling(window=60).corr(rets["bonds"])
>>> rolling_corr.plot()   ;; one correlation per day, last 60 days each
```

That `window=60` is the only real decision in the line, and it is the decision this entire post is about. Everything else is plumbing. The deeper toolkit — change-point detection, formal regime models, CUSUM — is covered in [rolling-correlation regimes and change-point detection](/blog/trading/macro-correlations/rolling-correlation-regimes-and-change-point-detection); here we stay on the single most important question: *how long should the window be?*

### Why the window length is not a detail

Here is the trap. When people first meet rolling correlations they treat the window as a cosmetic smoothing parameter — "60-day, 90-day, doesn't much matter, I'll use whatever the default is." It matters enormously. The window length controls a genuine, unavoidable tradeoff between two things you both want and cannot have together:

- **Statistical precision (low noise).** A correlation estimated from only 30 observations is a *noisy* estimate. Even two genuinely uncorrelated assets will, over any 30-day window, show a correlation that bounces around — sometimes +0.4, sometimes −0.3 — purely by chance. More observations average out that sampling noise. So *longer windows give smoother, more reliable estimates.*
- **Responsiveness (low lag).** A correlation can genuinely *change* — that is the whole point. When it does, you want your estimate to notice quickly. But a long window keeps the old regime's data inside it for a long time, so it updates slowly and lags the change. *Shorter windows respond faster to a real regime shift.*

You cannot have both. A window short enough to catch a regime change in two weeks is, by construction, short enough to be whipped around by noise. A window long enough to be smooth and stable is, by construction, long enough to lag a real flip by months. This is the **bias-variance tradeoff** in its purest market form, and the window length *is* the knob.

![Matrix showing how short, medium, and long correlation windows trade noise against lag](/imgs/blogs/rolling-correlation-and-why-the-window-matters-3.png)

#### Worked example: the same data, two windows, two different stories

Imagine — concretely — that the *true* underlying correlation between two assets is exactly −0.5 for the first 200 days, then instantly jumps to +0.5 on day 201 and stays there. (Reality is never this clean, but it isolates the effect.) Now watch what two windows report.

With a **30-day window**, on day 201 the window still holds 29 days of the old −0.5 regime and just 1 day of the new +0.5. The estimate barely budges. By day 215, the window is roughly half old and half new, so it reports something near **0** — a number that was *never* the true correlation in either regime. By day 230, the old data has finally aged out and the estimate settles near the true **+0.5**. So the 30-day window takes about a month to fully register the flip, and along the way it passes through a misleading "zero" reading. Worse, even once settled, it jitters by ±0.2 around +0.5 from sampling noise alone.

With a **250-day (one-year) window**, the lag is far worse. On day 201 the window holds 249 days of −0.5 and 1 day of +0.5, so it reports roughly **−0.49**. Months later, on day 350, the window is still 60% old regime, so it reports about **−0.1** — still the *wrong sign* — five months after the world changed. The estimate does not cross zero until roughly day 325 and does not reach the true +0.5 until about day 450, a full year late. But — and this is the consolation — while it is settled, it barely jitters; it is a smooth, trustworthy −0.5 (then +0.5) with almost no noise.

The intuition: the short window screamed "something changed!" within a few weeks but cried wolf constantly from noise; the long window was calm and reliable but told you about the regime change nearly a year after it would have helped. Neither is "right" — you are buying responsiveness with precision and there is no discount.

## The lag is built in: why every window arrives late

It is worth dwelling on *why* a rolling window must lag, because once you internalise the mechanism you will never again mistake a calm long-window correlation for a current one.

A rolling correlation is, at heart, an **average over the window**. Every observation inside the window gets equal weight — the data point from N days ago counts exactly as much as today's. So the moment the world flips, the new regime's signal is a tiny minority inside a window still dominated by the old regime's data. The estimate is *diluted* by stale points. It can only move toward the new truth as fast as the old points age out of the back of the window. Mechanically, a flat-weighted window of length N lags a sudden regime change by roughly **N/2** — half the window — because that is when the window becomes evenly split between old and new and the estimate finally crosses through the midpoint.

![Pipeline showing why a rolling window lags a regime change by half its length](/imgs/blogs/rolling-correlation-and-why-the-window-matters-7.png)

That is the unavoidable cost of equal weighting. A 60-day window lags by about a month; a one-year window lags by about six months; a five-year window lags by *two and a half years*. When a five-year-windowed correlation finally turns, the regime it is reporting on may already be over.

There is a second, sneakier problem with flat windows: **end-point sensitivity**, sometimes called the "ghost" or "drop-off" effect. Because every point in the window has equal weight, a single extreme observation has a big influence on the estimate the day it *enters* the window — and an equally big, opposite influence on the estimate the day it *exits*, N days later. So a rolling correlation can lurch for no current reason at all: a violent day from N days ago suddenly drops out of the calculation, and the estimate jumps, even though *nothing happened today*. A reader who does not know about the ghost effect will misread that jump as a fresh signal when it is really an artefact of the window's tail. We will see this in the real data shortly.

## EWMA: a window with a soft edge

Both problems — the brutal lag and the drop-off ghost — come from the same source: a flat window treats the point from N days ago exactly like today's, and then abruptly throws it away. The fix is to stop weighting points equally. Instead of a hard window that says "the last N points count fully and everything older counts zero," use an **exponentially weighted moving average (EWMA)**, which says "today counts the most, yesterday a little less, the day before a little less still," fading smoothly into the past with no hard cutoff.

The recursion is one line. You keep a running estimate and nudge it each day toward the newest observation:

```python
ewma_today = lam * ewma_yesterday + (1 - lam) * newest_observation
```

Here `lam` (the decay factor lambda, λ, between 0 and 1) controls how fast the past fades. A λ close to 1 fades slowly — lots of memory, very smooth, very laggy. A λ closer to 0.9 fades fast — short memory, responsive, noisier. The weights decline *geometrically*: today gets weight (1 − λ), yesterday gets λ(1 − λ), the day before λ²(1 − λ), and so on. No point is ever fully dropped, it just shrinks toward zero, so there is **no drop-off ghost** — the influence of an old extreme observation fades away gradually instead of vanishing in one step.

The natural way to describe an EWMA is not by a window length but by its **half-life**: the number of days after which an observation's weight has halved. Half-life and λ are two faces of the same dial. The relationship is λ = 0.5^(1/half-life), and the "effective" amount of data the EWMA is averaging over is roughly (1 + λ) / (1 − λ).

#### Worked example: choosing an EWMA half-life

Suppose you want a correlation estimate that feels about as smooth as a 90-day flat window but without the lag and ghost problems. What half-life do you pick?

A flat 90-day window averages over 90 points with effective sample size around 90. To match that smoothness with an EWMA, you want an effective sample size near 90. Solving (1 + λ)/(1 − λ) ≈ 90 gives λ ≈ 0.978, which corresponds to a **half-life of about 30 days**. So a 30-day-half-life EWMA carries roughly the same statistical weight (effective N ≈ 87) as a 90-day flat window, while putting **6.7% of all its weight on the single most recent observation** versus a flat window's 1.1% — it leans far harder on what just happened. Shorten the half-life to 20 days and λ drops to about 0.966, effective N falls to about 58 (snappier, noisier); lengthen it to 60 days and λ rises to about 0.989, effective N climbs to about 173 (smoother, slower).

The intuition: half-life is the EWMA's version of window length, and the same bias-variance law applies — a short half-life is responsive and noisy, a long one is smooth and laggy. EWMA does not abolish the tradeoff; it just gives you a smoother, ghost-free way to *sit* on it.

EWMA is the industry default for exactly this reason. The RiskMetrics framework that banks built their value-at-risk systems on used a fixed λ of 0.94 for daily data (a half-life of about 11 days) precisely because it responds to volatility and correlation regime changes faster than a flat window of comparable smoothness, without the discontinuous jumps. When you see a "decay-weighted correlation" in a risk report, it is an EWMA, and somewhere there is a half-life that is doing all the work.

## Rolling beta: the same lesson for hedge ratios

Everything we have said about rolling *correlation* applies with equal force to rolling **beta**, and the two are close cousins worth distinguishing because traders often confuse them. Correlation tells you *whether* two assets move together and how reliably; beta tells you *how much* one moves per unit of the other. Formally, beta of asset Y on asset X is the slope of the regression line — the change in Y you expect for a one-unit change in X — and it equals the correlation times the ratio of the two assets' volatilities (β = r × σ_Y / σ_X). We build this distinction carefully in [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta); the point here is that beta, too, is a window-dependent estimate that drifts and flips.

Beta is the number you actually need to *hedge*. If you want to neutralise your portfolio's sensitivity to, say, the 10-year yield, you do not care about the correlation per se — you care about the beta, because that tells you how many bond futures to sell per dollar of exposure. And just like correlation, that beta is computed over a window and changes as the regime changes. Hedge with a stale beta and you are either over-hedged or under-hedged by exactly the amount the beta has drifted since you measured it.

#### Worked example: a hedge ratio that drifted with the regime

Return to gold and the real yield, but now in beta terms. Over **2007 to 2021**, the slope of gold (in dollars per ounce) against the 10-year real yield (in percentage points) was about **−354** — meaning each 1 percentage-point rise in the real yield was associated with gold *falling* about \$354 an ounce. A trader hedging a gold position against rate risk would have sold rate exposure sized to that −354 beta.

Over **2022 to 2025**, the same regression slope was about **+411** — gold *rose* about \$411 an ounce per 1-point rise in real yields, the opposite sign and a larger magnitude. A hedge built on the old −354 beta would not merely be the wrong size in 2022 to 2025; it would have been pointed the wrong way entirely, *adding* to the loss it was meant to offset. The beta did not shrink toward zero and quietly stop working — it flipped sign, just as the correlation did.

The intuition: a hedge ratio is a frozen correlation in disguise, and freezing it freezes a regime. Roll the beta on the same window you roll the correlation, or your hedge will fight the regime you are actually in rather than the one you measured.

## How to choose a window in practice

So what number do you actually type into `window=`? There is no universal answer, but there is a disciplined way to land on one. The decision flows from three questions, in order.

**First, what is your decision horizon?** The window should be roughly matched to how long you will hold the view it informs. A market-maker hedging overnight inventory cares about the correlation *over the next day*, so a short window (or short-half-life EWMA, on the order of 1 to 3 weeks) is appropriate — they will re-hedge constantly and a stale estimate is worse than a noisy one. A tactical asset allocator rebalancing quarterly is well served by something in the **3 to 12 month** range. A strategic allocator setting policy weights for a decade wants a multi-year window, because a one-month wobble is, for them, definitionally noise. The horizon sets the order of magnitude.

**Second, how noise-tolerant is the decision?** If acting on a false signal is cheap (you can reverse the trade tomorrow at low cost), you can afford a shorter, noisier window and catch regime changes early. If acting on a false signal is expensive (a large, illiquid reallocation, or a hedge that costs real money to put on and take off), you should lengthen the window to suppress the false alarms, accepting that you will be later to a real change. The cost of being *wrong-early* versus *right-late* is the second dial.

**Third, what does the regime driver say?** As we will see, many macro correlations have an identifiable driver — the inflation regime for stocks and bonds, central-bank buying for gold, global liquidity for Bitcoin. If you can name the driver, you can cross-check the windowed estimate against it and effectively *shorten your reaction time* without shortening the window: when the driver moves, you anticipate the correlation flip before the window confirms it.

For most practitioners watching macro correlations at a weekly-to-monthly cadence, the pragmatic default is a **medium window (roughly 6 to 12 months, or a 60- to 90-day EWMA)** plotted *alongside* a fast window — and the divergence between them, not either line alone, is the live signal. That two-window habit, which we will return to in the playbook, is the closest thing to a free lunch in this whole exercise: the slow line tells you the regime, the fast line tells you whether it is breaking, and the gap between them is the early warning.

## The signature chart: the stock-bond correlation through time

Now let us put the whole argument on one picture, with real data, on the single most important cross-asset relationship there is.

The chart below plots the rolling ~24-month correlation between US stocks and long-dated Treasuries from 1990 to 2025. Three things are marked: a dashed **zero line** (the sign boundary that decides whether bonds hedge your stocks), the shaded **regime bands**, and a dotted line for the full-sample average.

![Stock bond rolling correlation 1990 to 2025 with regime bands and zero line](/imgs/blogs/rolling-correlation-and-why-the-window-matters-2.png)

Read it left to right. Through the **1990s** the correlation is *positive* (+0.3 to +0.45): the residue of the high-inflation era, when both stocks and bonds keyed off the same inflation fear. Around **1998 to 2000** it crosses zero and goes firmly *negative*, and it stays negative — sometimes as low as −0.55 — for two decades. This is the celebrated **60/40 era**: every market scare in this stretch was a *growth* scare (the dot-com bust, the 2008 financial crisis, the European debt crisis, COVID), the Fed responded each time by cutting rates, bond prices rose as stocks fell, and a balanced portfolio worked beautifully. We tell the full mechanism story in [the stock-bond correlation, the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).

Then **2022**. Inflation, not growth, became the dominant fear. The Fed hiked rates at the fastest pace in forty years to fight it. Higher rates crush bond prices *and* stock valuations simultaneously, so the two fell *together*, and the rolling correlation rocketed to **+0.6** — the sharpest, fastest sign flip in the series. It has eased since (to +0.25 by 2025) but has not gone back negative.

#### Worked example: why a 35-year average was useless in 2022

Look at the dotted line. The full-sample average correlation over 1990 to 2025 is **−0.04** — almost exactly zero, leaning very slightly negative. A risk model that fed all 35 years into one correlation estimate would have carried roughly that number, or a 20-year version of it landing near −0.3.

Now suppose in early 2022 you hold a 60/40 portfolio (\$600,000 stocks, \$400,000 bonds) and your risk model uses that long-sample correlation of −0.3 to estimate how the two will offset. The model expects bonds to *rally* when stocks fall, cushioning you. What actually happened: the live correlation was already turning positive. Stocks fell about 18%, taking the equity sleeve down roughly \$108,000. Bonds — which the model promised would soften the blow — fell about 13%, *adding* a loss of roughly \$52,000 instead of offsetting anything. The combined 60/40 drawdown was about **−16%**, near \$160,000, the worst year for that portfolio since 1937. The full-sample correlation did not merely under-warn; it pointed the wrong way.

The intuition: the average of "two decades of −0.4" and "one year of +0.6" is still negative, so the static number kept telling the model bonds were a hedge for months *after* they had become a co-loss. A rolling 24-month correlation, by contrast, had already crossed into positive territory and was screaming the warning. Same data, two windows, opposite advice — and only one of them had a clock.

This is the thesis in one chart. The full-sample number (−0.04) is true of no period in the picture. The rolling line tells you which world you are in, and in late 2022 those two answers were as far apart as they can be.

## A second real example: Bitcoin's correlation came and went

The stock-bond flip is the most consequential time-varying correlation, but it is not a one-off curiosity. Time variation is the *norm*, not the exception. Bitcoin gives us a cleaner-cut example because its macro era is so short you can watch a correlation be *born*, peak, and fade inside six years.

![Bitcoin Nasdaq rolling correlation rose to 0.65 in 2022 then faded](/imgs/blogs/rolling-correlation-and-why-the-window-matters-5.png)

Before 2020, Bitcoin's rolling 90-day correlation with the Nasdaq 100 hovered near **zero** — it traded on its own crypto-native narratives (halvings, exchange drama, adoption) with little reference to macro. Then, through 2021 and especially the first half of **2022**, the correlation climbed and *peaked near +0.65*. For that stretch, Bitcoin had effectively become a high-beta technology stock: it rose and fell with the same global liquidity and risk-appetite tide that moved the Nasdaq, and when the Fed drained liquidity by hiking rates, both sold off together. People who had bought Bitcoin as an "uncorrelated" diversifier discovered, exactly when it mattered, that it had quietly become a leveraged bet on the same risk factor as their tech stocks.

Then it *faded*. By 2024 the rolling correlation had drifted back toward **+0.2**, as the post-ETF, post-halving cycle restored some idiosyncratic behaviour. Anyone who had hard-coded the 2022 peak of 0.65 into a risk model would, by 2024, be overstating the diversification cost of holding Bitcoin alongside tech.

#### Worked example: sizing a position when the correlation is moving

Suppose you run a \$1,000,000 portfolio that is 80% Nasdaq tech and you are considering a 10% (\$100,000) Bitcoin sleeve, justified as diversification. How much diversification you actually get depends entirely on *which* correlation you use.

If you use the **2022 peak (r ≈ 0.65)**, Bitcoin is barely a diversifier — it moves mostly with the tech you already own, so the sleeve adds risk without much offset, and a risk model would flag the combined book as dangerously concentrated in one factor. If you use the **2024 reading (r ≈ 0.2)**, the same sleeve looks genuinely diversifying — its moves are mostly independent of the tech book, so it lowers portfolio variance. The correct sizing differs by a large margin, and the *only* thing that changed is the date of the correlation you trusted.

The intuition: a position size is a bet on a correlation, and a correlation has a date. Sizing off a stale peak (or a stale trough) is sizing off a regime that may already be gone — you must re-estimate over a rolling window and let the *current* reading set the size.

## What the window is really measuring

There is a deeper point hiding in all of this, and it changes how you should *think* about a rolling correlation. The window is not measuring some abstract statistical property that wanders for no reason. It is measuring **the underlying regime** — and the regime has a real-world driver you can often name.

For the stock-bond correlation, that driver is the **inflation regime**. When inflation is low and stable, the market's dominant fear is recession, the Fed's reaction function favours bonds in a downturn, and the correlation is negative. When inflation is high, the dominant fear is the Fed itself, hikes hit both assets, and the correlation goes positive. The chart below conditions the stock-bond correlation on the average inflation rate during the window, and the pattern is stark.

![Stock bond correlation by inflation regime from negative to positive](/imgs/blogs/rolling-correlation-and-why-the-window-matters-6.png)

When core CPI ran below 2%, the stock-bond correlation averaged **−0.45** (bonds hedge). At 2 to 3% it was **−0.30**. At 3 to 4% it crept up to roughly **+0.05** — the hedge had essentially evaporated. And above 4%, it averaged **+0.50** (bonds do *not* hedge; they fall with stocks). The window, in other words, is a slow-moving proxy for "what was inflation doing over the last two years," and that is *why* a long window lags: it is averaging over a regime variable that itself takes time to shift. We unpack the inflation-as-driver mechanism in [real vs nominal: inflation, real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).

This is the payoff of thinking in regimes rather than constants. Once you know *what drives* a correlation, the rolling chart stops being a mysterious squiggle and becomes a readout of a regime you can independently check. If your rolling stock-bond correlation is negative but core inflation just printed 4.5%, you should suspect the window is lagging and the negative reading is stale — exactly the 2022 setup.

The same logic applies to every correlation in this series. Gold's correlation with the real yield is driven by *what is the marginal buyer of gold doing* — when private investors trading on real-rate opportunity cost are the marginal buyer, the −0.96 relationship holds; when price-insensitive central banks become the marginal buyer (as in 2022 to 2024), the relationship breaks and even flips. Bitcoin's correlation with the Nasdaq is driven by *whether it is trading as a macro-liquidity asset or on its own crypto narrative* — when global liquidity is the dominant force (the 2022 tightening), the correlation runs high; when crypto-specific catalysts dominate, it fades. In every case the rolling correlation is a thermometer, and naming the thing it is measuring the temperature of lets you read it like a clinician rather than squinting at a squiggle.

The practical upshot is that you should never read a rolling correlation in isolation. Read it *next to* its driver. A divergence between the two — the correlation says one regime, the driver says another — is the most reliable early warning you will get that the window is about to move, because the driver leads and the windowed correlation lags. This is the difference between reacting to a regime change after your laggy estimate finally confirms it, and anticipating it because you watched the thing that causes it.

## Common misconceptions

**"The correlation is −0.3, so bonds will hedge my stocks."** This is the 2022 mistake in one sentence. There is no such thing as "the" correlation; there is only the correlation *over some window ending on some date*. The honest statement is "the trailing 24-month stock-bond correlation is currently +0.25, up from −0.4 in 2020." Always attach the window and the date. A correlation without a date is a rumour.

**"A longer sample gives a more accurate correlation."** It gives a more *precise* estimate of the *average over that sample* — which is a different and often useless quantity. More data reduces sampling noise, but if the extra data spans a different regime, you are precisely estimating a blend of two things that never coexisted. The gold-vs-real-yields −0.01 is a *very precise* number (19 annual observations) and a *completely misleading* one. Precision about the wrong quantity is not accuracy.

**"Rolling correlation tells me the correlation right now."** No — it tells you the correlation *averaged over the trailing window*, which because of the built-in lag is the correlation as it was, on average, around N/2 ago. A one-year-windowed correlation is reporting on roughly six months ago. If you need to know about *now*, you need a short window or an EWMA, and you must accept the noise that comes with it. The map is not the territory; the trailing window is not the present.

**"A jump in the rolling correlation means something just happened."** Not necessarily. With a flat window, a single extreme observation jolts the estimate *twice* — once when it enters the window and once, oppositely, when it drops off the back N periods later. That second jump is the **ghost effect** and reflects nothing about today; it is an artefact of the window's tail. Before you act on a sudden move in a flat-windowed correlation, check whether an extreme day just aged out of the calculation. EWMA does not have this problem because nothing is ever abruptly dropped.

**"EWMA is just a fancier rolling window with no real difference."** EWMA changes two things that matter: it has no discontinuous drop-off (so no ghost jumps), and it weights recent data more heavily, so for a given smoothness it responds faster to a genuine regime change. It does not, however, escape the bias-variance tradeoff — a long-half-life EWMA lags just as a long flat window does. EWMA is a better *shape* of window, not an exemption from the central law.

## How it shows up in real markets

**The 2022 regime shift (stocks and bonds).** We have covered this as the headline case, but note the *timing* lesson. The rolling 24-month correlation did not flip positive on the day inflation arrived; it crossed zero gradually through 2021 into 2022 as old negative-correlation months aged out of the window and new positive-correlation months entered. A trader watching a *short* (60-day) correlation would have seen the warning in late 2021; one watching a *two-year* window saw it confirmed only in mid-2022; one trusting a *full-sample* number never saw it at all. Same event, three different "when did you know," set entirely by window length.

**Gold's 2022 to 2024 decoupling.** The −0.96 relationship between gold and real yields was so reliable for so long that an entire generation of macro traders used "real yields up, short gold" as a near-automatic trade. Then in 2022 to 2024, real yields rose from negative to over +2% and gold *rose anyway*, from about \$1,800 to \$2,650 an ounce, as central-bank and reserve-diversification buying overwhelmed the rate signal. A rolling correlation would have shown the −0.96 relationship weakening through 2022 and turning positive — a clear, dated warning that "the old trade is broken." Traders who kept shorting gold against rising real yields, trusting a full-sample relationship that no longer held, were on the wrong side of a roughly 47% rally.

**Crisis correlation convergence.** A different and brutal manifestation: in a panic, *all* risky-asset correlations rush toward +1, and they do it fast. During the March 2020 COVID crash and the 2008 crisis, assets that had spent years comfortably uncorrelated suddenly fell in unison as forced deleveraging swept everything down together. Consider March 2020 concretely. For years prior, gold and stocks, and even Bitcoin and stocks, had behaved largely independently — a 90-day window would have shown correlations near zero, and a portfolio diversified across them looked genuinely spread out. Then, over a few days in mid-March, margin calls forced leveraged players to sell *everything they could*, regardless of fundamentals. Stocks, gold, Bitcoin, and even Treasuries at the worst moment all fell together; short-window cross-asset correlations spiked toward +0.8 to +1.0 essentially overnight. The diversification that the long-window numbers had promised vanished in the span of a week — exactly when it was supposed to save you.

A long-window correlation, by construction, *cannot* show this in time — it is still averaging in years of calm, so it would have continued reporting "well diversified, correlations near zero" right through the crash. Only a short window or an EWMA with a short half-life registers the convergence while it is happening, which is precisely when diversification you were counting on is evaporating. This is the dark side of the bias-variance tradeoff: the moment you most need a responsive estimate is the moment a smooth one fails you most. It is also why crisis risk cannot be managed with the same window you use for strategic allocation — the two horizons demand two windows, and a desk that runs only the slow one is structurally blind to the fastest, most dangerous correlation move there is. We trace this dynamic in [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).

**The 2022 stock-bond flip as a standalone case.** Because it is so instructive, the full anatomy of that flip — the inflation trigger, the failure of 60/40, the timeline of how fast it happened — gets its own deep treatment in [when correlations break: the 2022 stock-bond flip](/blog/trading/macro-correlations/when-correlations-break-the-2022-stock-bond-flip). The lesson there reinforces this one: the people most surprised were those anchored to a long-sample correlation that had no way to register the regime change in time.

## How to read and use a rolling correlation

Here is the operational playbook — how to actually use this rather than just admire the charts.

**1. Never quote a correlation without its window and date.** "Stock-bond correlation is −0.3" is a statement with no information until you add "trailing 24 months, as of mid-2020." Train yourself to hear an undated correlation as incomplete. When a risk report hands you a number, your first question is always: *over what window, ending when?* If they cannot answer, the number is a full-sample average and should be treated as a historical curiosity, not a live input.

**2. Look at two windows at once.** The most informative single habit is to plot a *short* window (or short-half-life EWMA) and a *long* window on the same axes. The long line tells you the established regime; the short line tells you whether that regime is currently holding or starting to break. When the short line diverges sharply from the long line — say the long window still reads −0.4 but the 60-day window has popped to +0.3 — that divergence *is* the early-warning signal of a regime change. The 2022 flip announced itself exactly this way.

#### Worked example: reading the two-window divergence

Picture late 2021. Your slow gauge — a 24-month stock-bond correlation — still reads about **−0.3**, comfortably in 60/40 territory, and a risk model anchored to it sees nothing wrong. But your fast gauge — a 60-day correlation — has climbed through zero to about **+0.2** as the first inflation-driven joint sell-offs appear in the data. The two lines have diverged by roughly half a point, and the fast one has changed sign.

What do you do with that? You do not blindly trust the fast line — recall it has a ±0.19 noise band, so a single pop above zero could be sampling jitter. You *cross-check it against the driver*: core inflation is accelerating past 4% and the Fed has begun signalling hikes. Driver and fast window now agree, and they disagree with the slow window. That confluence — fast estimate flipped, driver confirming, slow estimate still lagging — is the highest-conviction setup for acting *ahead* of the regime: you would begin trimming the assumption that bonds will hedge, perhaps adding a different hedge (cash, commodities, or inflation-linked bonds), months before the slow correlation crossed zero in mid-2022.

The intuition: the slow line is the regime you are insured against, the fast line is the regime arriving, and the gap between them — confirmed by the driver — is your lead time. Trading the gap is how you turn the window's lag from a liability into a warning.

**3. Match the window to the decision's horizon.** A day trader hedging overnight risk wants a short, responsive window and accepts the noise. A pension fund setting a strategic asset allocation for the next decade wants a long, stable window and accepts the lag — a one-week wobble in correlation is noise to them, not signal. There is no universally correct window; there is only the window that matches *your* holding period and *your* tolerance for being whipsawed versus being late. Pick it consciously.

**4. Check the regime driver, not just the number.** Because the stock-bond correlation is driven by the inflation regime, you can sanity-check a rolling reading against the thing that drives it. If your rolling correlation is comfortably negative but inflation is running hot and the Fed is hiking, distrust the negative reading — the window is probably lagging, and the regime has likely already turned beneath it. The independent driver lets you anticipate a flip *before* even a short window confirms it.

**5. Watch for the ghost before you react.** When a flat-windowed correlation lurches, before you trade on it, ask whether an extreme observation just dropped off the back of the window. If the jump is a drop-off artefact rather than fresh information, ignore it. (Or switch to an EWMA and stop worrying about it — this is the strongest practical argument for EWMA over flat windows in a live monitoring dashboard.)

**6. Know what invalidates your view.** A rolling correlation is a *belief about the current regime*. State what would change your mind: a specified number of consecutive periods of the short window on the other side of zero, or the regime driver (inflation, in the stock-bond case) crossing a threshold. If you cannot say what reading would flip your view, you are not using the rolling correlation — you are decorating a decision you already made. The discipline of the rolling estimate is that it forces you to keep re-asking the question instead of locking in an answer.

The thread running through all six rules is the same one this post opened with: a correlation has a sign, a strength, a lead/lag, *and a date*. The full-sample number throws away the date, and the date is where the regime lives. Roll the window, and the date — and the regime, and the risk — comes back into view.

A final word on humility. Choosing a window does not solve the problem of time-varying correlation; it just makes the problem visible and forces you to take a position on it. Every window length is a different bet about how fast the world changes relative to how much it jitters. Short windows bet the world changes often and you must keep up; long windows bet the world changes rarely and you should not be fooled by noise. Most of the great correlation blow-ups — 2008, 2020, 2022 — were committed by people who had implicitly bet the world changes rarely, using a window so long it could not see the change until it was over. When in doubt, watch a fast window and a slow window together, name the regime driver, and never, ever trust a single number that has forgotten what year it is.

There is a tempting fantasy that with enough cleverness — a smarter weighting scheme, an adaptive window that lengthens in calm and shortens in stress, a formal regime-switching model — you could escape the tradeoff entirely and get an estimate that is both perfectly smooth and perfectly responsive. You cannot. The information simply is not there: at the moment a regime changes, the only evidence that it changed is a handful of recent observations, and a handful of observations is noisy. Every method, no matter how sophisticated, must ultimately choose how much to trust those few recent points versus the many older ones, and that choice *is* the bias-variance tradeoff wearing a different costume. The adaptive methods are genuinely useful — they let the tradeoff bend with conditions rather than staying fixed — but they relocate the judgement, they do not remove it. The change-point and regime-detection tools in [rolling-correlation regimes and change-point detection](/blog/trading/macro-correlations/rolling-correlation-regimes-and-change-point-detection) are exactly this: principled ways to decide *when* the recent data has accumulated enough weight to declare a regime change, rather than magic that knows before the data does.

So the discipline is not to find the perfect window. It is to *own* the window you chose, to know what it can and cannot see, to pair it with a faster or slower companion so its blind spot is covered, and to keep one eye on the real-world driver that the correlation is ultimately a measurement of. Do that, and a rolling correlation becomes one of the most honest instruments in macro — a number that, unlike its full-sample cousin, never forgets to tell you *when*. Forget it, average everything together, and you get a −0.01 that describes a relationship that was, in truth, never anything other than ferociously strong and then ferociously reversed. The window is not a detail. The window is the whole game.

## Further reading & cross-links

- [What correlation actually measures: Pearson, Spearman, beta](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) — the building block this post assumes: covariance, the −1 to +1 scale, and why beta is not the same as correlation.
- [Spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data) — the other great way correlation lies: non-stationarity, multiple testing, and the third-variable problem.
- [Rolling-correlation regimes and change-point detection](/blog/trading/macro-correlations/rolling-correlation-regimes-and-change-point-detection) — the Python toolkit follow-up: EWMA, rolling z-scores, and detecting the flip formally.
- [When correlations break: the 2022 stock-bond flip](/blog/trading/macro-correlations/when-correlations-break-the-2022-stock-bond-flip) — the full case study of the regime change used as this post's headline example.
- [The stock-bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) — the mechanism behind the sign and why inflation flips it.
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — why every window fails at the worst possible moment.
- [Real yields: the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything) — the driver behind the gold relationship that broke.
- [Real vs nominal: inflation, real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — the inflation regime that drives the stock-bond correlation's sign.
