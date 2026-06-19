---
title: "Unemployment, Claims, and the Recession Correlation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The unemployment rate lags the cycle, but its rate-of-change is a powerful recession signal. This post builds the Sahm rule and jobless claims from zero, then shows how rising recession odds drive the big cross-asset correlation: curve steepens, spreads widen, bonds beat stocks."
tags: ["macro", "correlation", "unemployment", "jobless-claims", "sahm-rule", "recession", "credit-spreads", "yield-curve", "business-cycle", "risk-off", "labor-market"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — The unemployment rate is a *lagging* indicator: it sits at its multi-year low right when the cycle is about to break, so its *level* is nearly useless as a warning. But its *rate of change* — the Sahm rule — is one of the most reliable recession signals ever found, and jobless claims lead even that. The real driver of the big cross-asset correlation is not the labor number itself but the *recession probability* it reveals: as recession odds rise, the yield curve steepens from the front, credit spreads widen, defensives beat cyclicals, and bonds beat stocks.
>
> - **The number to remember:** the **Sahm rule** — a **+0.5 percentage-point** rise in the 3-month-average unemployment rate above its lowest point of the prior 12 months — has flagged the start of every US recession since the 1970s.
> - **Claims lead, the rate lags.** Initial jobless claims (weekly, a flow) lead the unemployment rate (monthly, a stock) by roughly **2 months**. You watch the flow, not the stock.
> - **Recession risk is the master variable.** The labor signal matters because it moves the market's *recession probability*, and that probability is what reprices everything at once — the correlation goes risk-off.
> - **Be honest:** the Sahm rule rose **0.5pp in 2024** with no recession, the cleanest recent reminder that a real-time signal can give a false positive when the *composition* of the labor market shifts.

In the summer of 2024, a quiet line on an economist's spreadsheet set off alarms across Wall Street. The US unemployment rate had climbed from a cycle low of 3.4% in early 2023 to 4.3% by mid-2024 — and that climb, when run through a simple recession rule devised by a former Federal Reserve economist named Claudia Sahm, *triggered*. The Sahm rule had never given a false alarm in fifty years. Every single time the 3-month-average unemployment rate had risen half a percentage point above its recent low, a recession had either already begun or was about to. So when it fired in August 2024, a lot of serious people braced for a downturn. Equity volatility spiked, the market priced in emergency rate cuts, and the financial press ran "is the recession here?" on a loop.

It was a false alarm. The economy kept growing. By the standards of the rule's own inventor, the 2024 trigger was probably a *structural* artifact — the unemployment rate had risen partly because a wave of new workers (immigration, re-entrants) was joining the labor force faster than jobs were being created, not because people were being laid off en masse. The flow that the rule is really trying to detect — *firing* — was not happening at recession scale. The rule measured the right number and drew the wrong conclusion, because the number meant something different this time.

That episode is the perfect way into this post, because it contains the whole lesson in miniature. The unemployment rate is a *lagging* indicator: by the time it has risen enough to look alarming, the cycle has usually already turned. So traders learned long ago not to watch the *level* but the *change* — and the change is a genuinely powerful signal, except when the thing driving the change is not the thing the signal assumes. Underneath all of it sits the variable that actually moves markets: the *probability of recession*. The labor data matters not for its own sake but because it shifts that probability, and the probability is what flips the entire cross-asset correlation structure from risk-on to risk-off.

![The recession-risk cascade from the first layoff to a risk-off market](/imgs/blogs/unemployment-claims-and-the-recession-correlation-1.png)

If you have not yet read [lead, lag, or coincident](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators), it is the natural prerequisite: it builds the cross-correlation function and sorts indicators into leaders, coincident, and laggards. This post zooms in on the labor leg of that taxonomy — claims (a leader), the unemployment rate (a laggard), and the recession probability that ties them to every asset price.

## Foundations: what the labor numbers actually are

Let us build every term from absolute zero, because the words "unemployment," "claims," and "recession" all sound obvious and all hide a subtlety that will trip you up later.

### The unemployment rate (U-3) is a snapshot of a stock

The headline unemployment rate — the one in every newspaper, officially called **U-3** — is a fraction. The numerator is the number of people who do not have a job *and* are actively looking for one. The denominator is the **labor force**, which is everyone who is either employed or actively looking. So:

```
unemployment rate = unemployed / (employed + unemployed)
```

The crucial words are *actively looking*. A person who has given up looking — a "discouraged worker" — is not counted as unemployed; they drop out of the labor force entirely, which mechanically *lowers* the unemployment rate even though their situation got worse. This is the first reason the rate can mislead.

It is worth knowing that U-3 is only one of six official unemployment measures the Bureau of Labor Statistics publishes, labeled U-1 through U-6 in order of breadth. U-1 is the narrowest (people unemployed 15 weeks or longer); U-3 is the headline; U-6 is the broadest, adding "marginally attached" workers (those who want a job but have stopped actively looking) and people working part-time who want full-time work. U-6 is typically about double U-3 — when U-3 is 4%, U-6 is often near 8% — and it tells you about *labor-market slack* that the headline number hides. A trader watching for labor-market deterioration sometimes catches it in U-6 (more people stuck in part-time work, more marginally attached) before it shows in U-3, because the broader measure picks up the early softening that has not yet become outright joblessness. The point is not to memorize all six; it is to internalize that "the unemployment rate" is a *choice* of definition, and the headline definition is deliberately conservative about who counts.

There is a second, subtler measurement issue: the **participation rate** — the share of the working-age population that is in the labor force at all. When participation falls (people retire, go back to school, or simply stop looking), the unemployment rate can drop even though no new jobs were created, because the denominator shrank. When participation *rises* (immigration, re-entrants returning to look for work), the unemployment rate can climb even though no one was laid off, because the denominator and the numerator both swelled with job-seekers. Hold that fact in your head: it is the exact mechanism that produced the great false signal of 2024, and we will return to it.

In the language of [the lead-lag taxonomy](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators), the unemployment rate is a **stock**: a level you measure at a point in time, like the water in a bathtub. Stocks change slowly, because they are the accumulation of a flow. The water level only rises after the tap has been running for a while. And like any stock, it can move for two completely different reasons — more water coming in (layoffs) or the drain slowing down (the long-term unemployed not finding work) — which is why you can never read the rate in isolation.

### Jobless claims are a measure of a flow

Now meet the flow. When a person is laid off in the United States, they typically file for unemployment insurance within days. The Department of Labor reports two numbers every Thursday:

- **Initial claims** — the number of *new* filings in the past week. This is the rate at which the tap is opening: how many people *just* lost their jobs. It is a flow, measured weekly, and it is one of the most timely macro data points in existence.
- **Continuing claims** — the number of people *still* collecting benefits. This is closer to a stock, and it tells you whether the laid-off are finding new work (continuing claims fall) or staying jobless (continuing claims keep rising).

Here is the key relationship, and it is pure plumbing. The unemployment rate (the bathtub level) only rises after enough water has flowed in. Initial claims (the inflow) move *first*, because a layoff hits the claims data within a week but only nudges the rate after the layoffs accumulate. That is why **initial claims lead the unemployment rate** — in our curated cross-correlation data, by about **2 months**.

The two claims numbers tell a richer story together than either does alone. Initial claims measure the *rate of firing* — the speed at which the tap is opening. Continuing claims measure the *difficulty of re-hiring* — whether people who lost jobs are quickly finding new ones (the drain is open) or piling up jobless (the drain is clogged). In a healthy economy, initial claims can tick up briefly while continuing claims stay low, because the newly unemployed find work fast; that is churn, not deterioration. The dangerous pattern is when *both* rise together: more people losing jobs *and* fewer of them finding new ones. That combination is the signature of a labor market that is genuinely turning, and it is precisely the configuration the unemployment rate will reflect a couple of months later as the stock fills up. So the sophisticated read is not "initial claims rose" but "initial *and* continuing claims are both trending up" — the firing accelerated and the re-hiring slowed at the same time.

A practical warning about claims: they are extraordinarily *noisy*. The raw weekly number swings on factors that have nothing to do with the business cycle — auto plants shutting down for retooling, school-year start and end dates, holidays, hurricanes, even the timing of how many business days fell in a particular week. The government applies seasonal adjustments to smooth out the predictable parts, but residual noise remains. This is why practitioners almost never react to a single weekly print and instead watch the **4-week moving average** of initial claims, which damps the noise and reveals the trend. A single 270k week after months of 220k is probably noise; four straight weeks averaging 270k is a signal.

### A recession is officially a committee's verdict, dated in hindsight

In the United States, a recession is not "two negative quarters of GDP" — that is a rule of thumb, not the definition. The official arbiter is the **National Bureau of Economic Research (NBER)**, a private committee that declares a recession as "a significant decline in economic activity spread across the economy, lasting more than a few months." Critically, the NBER dates recessions *long after they begin* — often six to eighteen months later — because it waits for the data to be revised and confirmed. The recession that began in February 2020 was not officially dated until June 2020; the one that began in December 2007 was dated in December 2008, a full year later.

This timing matters enormously for our story. If the official body that *defines* a recession can only confirm one a year after the fact, then the unemployment rate — which is itself a laggard — is a *confirmation of a confirmation*. By the time U-3 has visibly risen and the NBER has officially called it, the asset prices that care about recessions moved months or years earlier. The whole game is reading the *leading* edge of recession risk, not the lagging confirmation.

### Correlation, quickly

Throughout this post I will say things like "the correlation flips to risk-off." A reminder of the vocabulary (built in full in [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta)): a **correlation** measures how two things move together, on a scale from −1 (perfect opposite) through 0 (unrelated) to +1 (perfect lockstep). A **beta** is the slope — how many units of asset Y move per unit of driver X. And the whole thesis of this series is that these numbers are **regime-dependent**: the correlation between, say, stocks and bonds is *not a constant* — it has one sign in calm expansions and the opposite sign in an inflation shock. (See [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant).) Recession is one of the most powerful regime switches there is, which is exactly why a labor signal that detects it early is worth so much.

## Why the level lags but the change leads

This is the heart of the post, so let us be precise about it.

Look at the actual path of the US unemployment rate over the last few years. It is one of the most instructive charts in macro.

![US unemployment rate path with the 2020 spike and slow normalization](/imgs/blogs/unemployment-claims-and-the-recession-correlation-2.png)

Read the chart left to right. In February 2020, U-3 was **3.5%** — a fifty-year low. The economy looked spectacular. Two months later, after the pandemic shutdowns, it hit **14.8%**, the highest since the Great Depression. Then it spent four years grinding back down toward 3.4%, before drifting *up* again to 4.3% by 2026.

Now ask the question that matters: at any point on this chart, could the *level* have warned you of the next downturn? In February 2020 the level was 3.5% and screaming "all is well" — one month before the sharpest recession in living memory. That is the lagging indicator's fatal flaw: **the level looks best right before the cycle breaks.** A low unemployment rate is not a sign of safety; it is, if anything, a sign of a late cycle, because unemployment is lowest at the *peak*, just before the contraction.

So the level lags. What leads? The *change*. The genius of the Sahm rule is that it ignores the level entirely and watches the rate of change off the bottom.

![Level versus change why the unemployment rate fools you and the Sahm rule does not](/imgs/blogs/unemployment-claims-and-the-recession-correlation-3.png)

The figure lays out the two readings side by side. The **level** measures the current stock of unemployment, sits at a multi-year low into the cycle peak, and only climbs after layoffs accumulate — so it *lags*. The **change** measures the 3-month-average rate minus its lowest 3-month-average of the prior 12 months; it reads near zero at the peak and crosses a threshold as the rate of change turns up — so it *signals*. Same underlying data, opposite usefulness.

### The Sahm rule, built from scratch

The **Sahm rule** is mechanical and you can compute it on a napkin. Here is the recipe:

1. Take the **3-month moving average** of the U-3 unemployment rate. (Averaging smooths out the monthly noise — a single hot month does not trigger it.)
2. Find the **minimum** of that 3-month average over the **trailing 12 months**.
3. If today's 3-month average is **0.5 percentage points or more** above that 12-month minimum, the rule triggers.

That is the entire rule. Why does it work? Because unemployment, once it starts rising, rises *fast and self-reinforcingly*. Layoffs reduce incomes, which reduces spending, which reduces revenues, which causes more layoffs — a feedback loop. So the unemployment rate almost never rises a *little*. Historically, once the 3-month average has climbed 0.5pp off its low, it has kept going, because the recession dynamic has taken hold. There is no "soft landing" in the historical record where U-3 rose exactly 0.5pp and then stopped — until, arguably, 2024.

It is worth dwelling on *why* the feedback loop makes unemployment a non-linear variable, because that non-linearity is the entire statistical reason the rule works. In a healthy economy, small shocks get absorbed: one firm lays off a hundred workers, they find jobs at other firms, and the rate barely moves. But near a cycle turn, the absorbers are gone — the other firms are also cutting, not hiring. Now the hundred laid-off workers stay jobless, their lost income shows up as lost demand at the businesses they used to patronize, those businesses cut their own staff, and the shock *amplifies* instead of damping. The system flips from a regime where shocks die out to a regime where they propagate. The Sahm threshold of +0.5pp is, in effect, a detector for *which regime you are in*: a 0.5pp rise off the low is large enough that it almost never happens in the absorbing regime and almost always signals you have entered the amplifying one. The rule is not arbitrary numerology; it is a calibrated tripwire for a phase transition in the labor market.

This is also why the *unemployment rate's own rate of change* is more informative than its level. The level tells you where the bathtub is. The rate of change tells you whether the inflow has overwhelmed the drain — whether the feedback loop has tipped from absorbing to amplifying. A rate of 4% reached by *falling* from 5% (the recovery side) means something totally opposite to a rate of 4% reached by *rising* from 3.4% (the deterioration side), even though the level is identical. Direction and acceleration carry the recession information; the level carries almost none.

#### Worked example: why the change beats the level

Make the idea concrete with two economies that have the *exact same* 4.1% unemployment rate today.

```
                Economy A (recovering)   Economy B (deteriorating)
12 months ago        5.5%                      3.4%
6 months ago         4.8%                      3.6%
3 months ago         4.4%                      3.9%
today                4.1%                      4.1%
Sahm gap         4.1% - 4.1% = 0.0pp       4.1% - 3.4% = 0.7pp
```

(In Economy A the trailing-12-month low *is* today's 4.1%, because the rate has been falling — so the Sahm gap is zero. In Economy B the trailing low is 3.4%, so the gap is 0.7pp.) Identical levels, opposite signals. Economy A is healing — the rate is *falling*, the feedback loop is running in reverse, and the Sahm rule reads a benign 0.0pp. Economy B is breaking — the rate is *rising* fast off its low, and the Sahm rule reads a recessionary 0.7pp. A trader who looked only at "4.1%, that's a healthy level" would treat the two economies identically and be catastrophically wrong about Economy B. The intuition: **a recession is a derivative, not a level** — it lives in how fast the labor market is changing, not in where it happens to sit.

#### Worked example: computing a Sahm-rule trigger

Suppose you are tracking a portfolio in late 2024 and want to know whether to shift toward defense. You pull the unemployment rate and compute the rule.

Assume the 3-month-average unemployment rate prints these values (illustrative, consistent with the cycle low near 3.4% and the drift to 4.3%):

```
month         3m-avg U-3
2023-06          3.5%      <- the trailing-12m low
2024-01          3.7%
2024-04          3.9%
2024-07          4.1%
2024-10          4.3%      <- today
```

Step 2: the lowest 3-month average in the trailing 12 months (from late 2023 through 2024) is **3.5%**. (We use the 12-month look-back window; the broader cycle low of 3.4% was earlier.) Step 3: today's reading is 4.3%, so the Sahm gap is:

```
4.3% - 3.5% = 0.8 percentage points
```

That is **well above the 0.5pp threshold**, so the rule has triggered. A trader following the rule mechanically would, in this moment, lean toward the recession playbook: lengthen bond duration, trim cyclicals, raise cash. The intuition: a 0.8pp rise off the low has, in every prior cycle, meant the labor-market feedback loop had started — so you position before the lagging confirmation arrives.

(The punchline, of course, is that 2024 was the exception — which is exactly why a worked example of the *trigger* must be paired, later, with a worked example of the *false positive*. A signal you cannot conceive of being wrong is a signal you do not really understand.)

### Claims lead the rate — the timing edge

If the unemployment rate lags and the Sahm rule fires off a 3-month average (so it is inherently a couple of months slow), the truly timely signal is initial claims. They arrive weekly, and they detect the *inflow* directly.

#### Worked example: claims leading the unemployment rate

Our curated cross-correlation data puts the peak lead of **initial claims → unemployment rate at +2 months**. Let us turn that into a trading timeline you could actually act on.

Suppose initial claims, which had been running at a calm ~220,000 per week, start climbing: 240k, 255k, 270k over three weeks, and the 4-week average breaks decisively above its prior range in, say, **March**. The +2-month lead says the unemployment *rate* will not visibly inflect until roughly **May**, and the Sahm rule (which needs a 3-month average to move) will not trigger until perhaps **June or July**.

So the claims watcher gets a two-to-four-month head start over the rate watcher, and an even larger head start over the Sahm-rule watcher. If you manage a \$10,000,000 equity book and you act on the claims break in March by rotating \$2,000,000 from cyclicals into long Treasuries, you are positioning while the unemployment rate is still printing a comforting low. By the time the rate confirms in May and the headlines turn fearful, your defensive tilt is already on — and, as we will see, bonds have already started outperforming. **The flow leads the stock; you trade the flow.**

The caution: claims are *noisy*. Weekly numbers jump around on seasonal quirks (auto-plant retooling shutdowns, holidays, hurricanes). That is why practitioners watch the **4-week moving average** of initial claims, not the single weekly print, and why a real break has to clear the noise band convincingly before you act on it.

## The real driver: recession probability moves every asset at once

Here is the conceptual leap that separates a beginner from someone who actually understands the labor-market correlations. The unemployment rate does not move stocks. Claims do not move bonds. What moves *everything* is the market's revised estimate of the **probability of recession**. The labor data matters only as an *input* to that estimate.

Think of recession probability as a single dial that the whole market reads. When the dial moves from "20% chance of recession in the next year" to "50% chance," a cascade fires across every asset class simultaneously — and because the *same* dial is driving them all, their correlations tighten and align. This is why the labor leg matters so much: it is one of the most direct inputs to that dial.

This "one dial drives them all" idea is also the deep statistical reason correlations *tighten* in a downturn. In a calm expansion, each asset is buffeted by its own idiosyncratic news — a tech earnings miss here, an oil-supply story there, a sector rotation somewhere else — and those independent shocks keep the assets only loosely correlated. But when recession probability becomes the dominant force, it overwhelms the idiosyncratic noise: every asset is suddenly being priced off the *same* variable, so they all move together. Mathematically, when one common factor explains most of the variance in everything, the pairwise correlations between everything rise toward the sign of that factor's loadings. Risk assets (stocks, high-yield credit, commodities, crypto) all load *negatively* on recession risk, so they all fall together; safe assets (Treasuries, cash) load *positively*, so they rally together; and the correlation *between* the two groups goes sharply negative. That is the entire risk-off correlation structure, and it is a direct consequence of a single dominant driver. (The same mechanism, taken to its extreme in a panic, is what makes [correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).)

What happens as the dial turns up?

- **The yield curve steepens from the front (a "bull steepener").** When recession odds rise, the market expects the central bank to *cut* rates to fight the downturn. Cuts are a front-end story: the 2-year yield falls hardest because it most directly prices the path of policy. The 10-year falls too but less, so the 2s10s spread *widens* (steepens) — but for the "good" reason of falling short rates, which is why it is a *bull* steepener. (The mechanism is built in [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession).) Importantly, the curve typically *inverts* well before the recession and *un-inverts by steepening* right around the onset — the un-inversion is often the final warning.
- **Credit spreads widen.** A recession means more corporate defaults, so lenders demand more compensation to hold risky (high-yield) debt. The extra yield over Treasuries — the spread — widens. Spreads are one of the most sensitive recession barometers there is. (See [credit spreads, the risk correlation](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary).)
- **Defensives beat cyclicals.** Within the stock market, sectors whose earnings hold up in a downturn (utilities, consumer staples, health care) outperform sectors whose earnings collapse (industrials, consumer discretionary, materials). The market rotates *toward* defense before the recession is confirmed.
- **Bonds beat stocks.** This is the big one. When recession odds rise, falling rates lift bond prices while falling earnings expectations hurt stocks. Government bonds become the flight-to-safety asset. The whole risk-on correlation inverts.
- **The dollar often firms.** In a global risk-off, capital flees to US Treasuries, which means buying dollars — so the USD tends to strengthen (with exceptions). (See [the dollar, cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity).)

### What actually wins in a recession

Let us put numbers on "bonds beat stocks." The investment clock — the documented rotation of asset-class returns across the business cycle — has a specific recession row, and it is unambiguous.

![Recession phase asset returns bonds and cash up stocks and commodities down](/imgs/blogs/unemployment-claims-and-the-recession-correlation-4.png)

In the recession phase, representative real returns are roughly **−10% for stocks**, **+10% for government bonds**, **−8% for commodities**, and **+1% for cash**. Bonds and cash are the winners; stocks and commodities are the losers. This is the entire risk-off correlation in one bar chart: the assets that thrive on growth get punished, and the assets that thrive on *the absence* of growth (long-duration government bonds, which rally as rates fall) get rewarded. (For the full clock across all four phases, see [the business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock) and the mechanism in [the business cycle's four phases](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders).)

#### Worked example: a defensive-rotation P&L into a recession

Now let us make the rotation pay, with explicit dollars. Suppose in March your labor signals line up — claims have broken higher, and you judge recession probability has jumped. You manage a **\$10,000,000** balanced book, currently allocated **70% stocks / 30% bonds**:

```
starting book          \$10,000,000
  stocks  (70%)        \$7,000,000
  bonds   (30%)        \$3,000,000
```

You execute the recession playbook and flip to **30% stocks / 70% bonds**:

```
rotated book
  stocks  (30%)        \$3,000,000
  bonds   (70%)        \$7,000,000
```

Over the following recessionary year, using the clock's representative returns (stocks −10%, government bonds +10%):

```
had you NOT rotated (70/30):
  stocks  \$7,000,000 x (1 - 0.10) = \$6,300,000
  bonds   \$3,000,000 x (1 + 0.10) = \$3,300,000
  total                              \$9,600,000   (-4.0%)

after rotating (30/70):
  stocks  \$3,000,000 x (1 - 0.10) = \$2,700,000
  bonds   \$7,000,000 x (1 + 0.10) = \$7,700,000
  total                              \$10,400,000  (+4.0%)
```

The rotation turned a **−\$400,000** loss into a **+\$400,000** gain — an **\$800,000 swing** on a \$10M book, worth **8 percentage points** of relative performance. The intuition: in a recession the sign of the stock-bond bet flips, so the single most valuable thing your labor signal bought you was the conviction to be *long bonds and light stocks* before the rate confirmed what the claims data already knew.

Note the honesty check baked into the numbers: this only works if the recession actually arrives. If you rotate on a *false* Sahm trigger and the economy keeps growing (stocks +12%, bonds +2% in a mid-expansion), the same flip *costs* you the upside. That asymmetry is the whole reason the signal's false-positive rate matters so much, and why we will dwell on 2024.

## The curve leads the recession (by a long, variable lag)

If recession probability is the master dial, the single most famous *leading* input to it is the shape of the yield curve. The 2s10s spread — the 10-year yield minus the 2-year — has historically *inverted* (gone negative) before recessions, because an inverted curve is the bond market pricing in future rate cuts, which is the bond market pricing in a future downturn.

But "the curve leads the recession" is a far softer statement than people think, and the data shows exactly why.

![Yield curve inversion to recession lead times across episodes](/imgs/blogs/unemployment-claims-and-the-recession-correlation-5.png)

Look at the lead times from the first 2s10s inversion to the recession's start across the modern episodes: **1989 inversion → 18 months**, **2000 → 13 months**, **2006 → 22 months**, **2019 → 6 months** (COVID arrived early and shortened it), and **2022 → 0 (and counting)** — no recession had been dated at the time of writing, which is the heart of the false-signal debate. The average lead of the four episodes that *did* precede a recession is roughly **15 months**. That is an enormous and *variable* lead. A signal that warns you somewhere between 6 and 22 months in advance is real — the curve has inverted before every modern US recession — but it is nearly useless for *timing*. You cannot run a portfolio that is defensive for two years waiting for a recession that may be eighteen months out.

This is precisely why the labor signals complement the curve. The curve tells you a recession is *possible* (a long, early warning). Claims and the Sahm rule tell you it is *starting* (a late, sharp confirmation). The curve is the smoke detector; the labor data is the smell of smoke in the room. A disciplined recession watcher reads them in sequence: the curve inverts and puts you on alert, then a year or so later claims break and the Sahm rule fires, and *that* is when you commit to the risk-off rotation. (The yield-curve growth signal and its asset correlation is built out fully in [the yield curve as a growth signal](/blog/trading/macro-correlations/the-yield-curve-as-a-growth-signal-and-its-asset-correlation).)

### Why "bonds beat stocks" — the duration mechanism

It is worth being precise about *why* bonds rally so hard in a recession, because the magnitude surprises people. The answer is **duration**. A bond's price moves opposite to its yield, and the size of that move scales with the bond's duration (roughly, its weighted-average time to repayment). A long-dated Treasury — a 20- or 30-year bond — has a duration of fifteen years or more, which means a 1-percentage-point fall in its yield produces roughly a *15%* rise in its price. In a recession, the central bank cuts rates aggressively, long yields fall a point or two, and long-duration bonds deliver equity-like *gains* with the safety of government credit. That is the mechanical engine behind the recession row of the investment clock: bonds do not merely "hold up," they actively *win*, because the same falling-rate environment that crushes growth-sensitive earnings is precisely what lifts bond prices the most.

#### Worked example: the bond leg of the rotation, in dollars

Suppose you put **\$2,000,000** into long Treasuries (duration ≈ 16 years) as your recession rotation, and over the downturn the long yield falls by **1.25 percentage points** as the central bank cuts. The price gain from duration alone is approximately:

```
price change  ≈  -duration x change in yield
              ≈  -16 x (-1.25%)
              ≈  +20%

dollar gain   ≈  \$2,000,000 x 0.20  =  \$400,000
```

So the long-bond leg earns roughly **\$400,000** on a 1.25pp rate decline — and that is *before* the coupons you collect along the way. Meanwhile the stocks you sold to fund it were falling. The intuition: in a recession the central bank's rate cuts are a transfer of value from growth assets to duration, and owning duration is how you stand on the receiving end of that transfer. The longer the duration, the larger your share — which is why "long the long end" is the classic recession trade, and why getting the labor signal right early enough to put it on matters so much.

## Credit spreads: the recession correlation you can actually trade

The yield curve leads by too much to time. The labor data confirms a little late. Credit spreads sit in a useful middle — they lead equity drawdowns by about **3 months** in our cross-correlation data, sharp enough to act on, and they are arguably the *cleanest* market-priced recession barometer because the people setting them (corporate bond investors) lose real money if a recession brings defaults, so they have skin in the game.

The relationship between spreads and forward equity returns is one of the most counterintuitive and most useful in all of macro.

![High yield spread versus S&P forward return scatter with regression line](/imgs/blogs/unemployment-claims-and-the-recession-correlation-6.png)

The scatter plots the high-yield credit spread (the extra yield on risky corporate debt over Treasuries) against the S&P 500's return over the *next* 12 months. The fitted line is *upward-sloping*, with a correlation of **r = +0.77** and a slope of about **+3.4% of forward return per 1 percentage point of spread**. Read that carefully: **wide spreads predict high forward returns.** When the spread is calm at ~3%, the next year delivers only ~+10%. When the spread blows out to ~10.8% in a panic, the next year delivers ~+35%.

Why? Because the spread is widest at the moment of *maximum fear* — deep in the recession or the crisis, when prices have already collapsed. The panic that widens the spread is the same panic that sets up the recovery. The spread is a *contemporaneous* fear gauge that, precisely because it overshoots, becomes a *forward* opportunity signal. This is the "be greedy when others are fearful" logic, made quantitative.

A short note on what a credit spread actually *is*, so the signal is not a black box. When a risky company borrows by issuing a bond, it must pay a higher yield than the US government does on a bond of the same maturity, because there is a chance the company defaults and the lender loses money. That extra yield — the "option-adjusted spread," or OAS — is the market's price of corporate default risk. In calm times, high-yield (junk-rated) bonds might yield only 3 percentage points more than Treasuries; in a recession scare, that gap can blow out past 8 or even 10 points as investors demand far more compensation to hold debt that might not be repaid. The spread is therefore a direct, real-money vote on recession risk by the people with the most at stake: if corporate-bond investors thought defaults were coming, they would sell, prices would fall, and the spread would widen *now*, ahead of the actual defaults. That is exactly why spreads lead equity drawdowns — the credit market prices the recession into the cost of borrowing before the stock market fully prices it into earnings.

There is a deep link back to the labor data here. Rising unemployment is the *cause* of rising corporate defaults: when people lose jobs, they spend less, company revenues fall, the weakest firms cannot service their debt, and they default. So the labor signal and the credit signal are two views of the same underlying process — the labor data shows the *input* (jobs being lost), the credit spread shows the *expected output* (defaults coming). When both turn together — claims breaking higher *and* spreads widening — you are seeing the recession from two independent angles at once, which is far more convincing than either alone. The 2024 false positive is instructive precisely because the two *diverged*: the unemployment rate rose (triggering Sahm) but credit spreads stayed tight, because corporate-bond investors — looking at the same economy — did not see a default wave coming. The market's own risk pricing contradicted the labor-rule trigger, and the market was right.

#### Worked example: sizing a position off the spread signal

Suppose the high-yield OAS has blown out to **8.0%** in a recession scare — well above its ~3–4% calm range. The fitted relationship (slope ≈ +3.4% forward return per 1pp of spread, intercept ≈ −9%) estimates the next-12-month S&P return as:

```
expected fwd return  =  3.4 x (spread) - 9
                     =  3.4 x 8.0 - 9
                     =  27.2 - 9
                     =  +18.2%
```

So with the spread at 8%, the model points to roughly **+18%** over the next year. Say you decide to add **\$500,000** of equity exposure on this signal. The point estimate implies an expected gain of:

```
\$500,000 x 0.182 = \$91,000 expected gain over 12 months
```

The intuition — and the giant caveat — is that this is an *average* over historical wide-spread episodes; the realized outcome has a huge spread of its own (sometimes the spread widens further before it recovers, and you are early and underwater for months). The signal tells you the *odds are in your favor when fear is extreme*, not that any single trade will work. You are paid for buying the fear, on average, because the fear was overdone — but "on average" is doing heavy lifting, and you must size for the path, not just the destination.

## How it shows up in real markets

Two cases tell the whole story: one where the labor signal worked and one where it didn't.

### 2020: the unemployment spike and the everything-to-one crisis

The COVID shock is the cleanest possible illustration of both the labor signal and the recession correlation — and of how diversification fails at the worst moment. In February 2020, U-3 was 3.5%, a fifty-year low. Within two months it was **14.8%**. Initial claims, normally ~220,000 per week, exploded to a peak of **over 6,000,000** in a single week — a number so far outside the historical range it looked like a data error. The recession was instant and total.

What did the correlation do? For about two weeks in March 2020, **everything fell together** — stocks, corporate bonds, gold, even Treasuries briefly, as the "dash for cash" forced investors to sell whatever they could. This is the recurring lesson that [correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis): the diversification you counted on evaporates exactly when you need it most. (We build this failure mode in [correlation during crises, when diversification fails](/blog/trading/macro-correlations/correlation-during-crises-when-diversification-fails).)

But then, once the Federal Reserve flooded the system with liquidity and cut rates to zero, the *normal* recession correlation reasserted itself with a vengeance: **bonds and duration won.** Long Treasuries rallied hard as yields collapsed toward record lows; the 10-year fell to **0.62%** by mid-2020. An investor who had been long duration through the spring — or who rotated into it as the labor catastrophe became clear — was richly rewarded. The everything-to-one phase was brief; the risk-off phase that followed was the textbook recession correlation, just compressed into weeks because the shock was so violent.

Watch the broader cross-asset map snap into the recession configuration as the panic resolved. The dollar spiked in the dash-for-cash (everyone needed dollars to meet margin calls and fund redemptions), then eased as the Fed's liquidity reached the system — a textbook risk-off-then-relief dollar path. Gold initially fell with everything else in the forced-selling phase, then rallied to record highs as real yields collapsed (gold tracks real yields, not the labor data directly — see [inflation and gold, the real-yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story)). High-yield credit spreads blew out toward 9% at the trough, then compressed violently — and by the scatter relationship above, that ~9% spread was a screaming forward-return signal: the S&P went on to nearly double off the March 2020 low over the following eighteen months. Every leg of the recession correlation fired, just on fast-forward.

The honest caveat is that almost no one *rotated into the 2020 panic in real time* — the speed and depth were paralyzing. The lesson is not "you would have nailed it" but "the recession correlation structure is reliable enough that, even in the most chaotic downturn in living memory, bonds beat stocks and wide spreads preceded high forward returns exactly as the framework predicts." The framework is robust; the discipline to act on it is the hard part.

The labor signal's role here was unusual: it did not *lead* anything, because the shock was exogenous (a virus, not a credit cycle). The 6,000,000-claims print was *coincident* with the crash, not ahead of it. The lesson is the limit of the labor signal: it leads *endogenous* recessions (the slow build of a credit/labor feedback loop) but not *exogenous* shocks (a pandemic, a war), which arrive faster than any leading indicator can warn.

### 2023–24: the Sahm rule that cried wolf

The opposite case is the 2024 false positive we opened with. Recall the path: the unemployment rate rose from a 3.4% cycle low (early 2023) to 4.3% (mid-2024), and the Sahm rule — fifty years without a miss — triggered in August 2024. Markets convulsed. And yet no recession came.

Why did the rule fail? Because the rule implicitly assumes that a *rising* unemployment rate means *rising layoffs* — the demand-side feedback loop. But in 2023–24, a large part of the rise came from the *supply* side: the labor force was growing rapidly (immigration, post-pandemic re-entrants), so more people were *looking* for work and being counted as unemployed before they found a job. The numerator rose for a benign reason. Layoffs — the thing the rule is really trying to detect — stayed low; initial claims never broke convincingly higher. The Sahm rule measured the unemployment rate correctly and inferred a recession incorrectly, because the *composition* of the change had shifted.

This is a profound and humbling lesson, and it ties directly to two other posts in this series. First, it is a [structural shift](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data): a historical relationship (rate-of-change → recession) broke because the data-generating process changed underneath it. Second, it is a reminder that **a signal is only as good as the mechanism behind it.** The Sahm rule worked for fifty years not because of numerology but because, for fifty years, a rising unemployment rate *was* a layoff story. When that stopped being true — when the rise was a labor-supply story — the rule's correlation with recession broke. The honest practitioner's response is not to throw out the Sahm rule but to *decompose* it: when it triggers, immediately check whether claims confirm. In 2024, claims did *not* confirm, which was the tell that this trigger was different.

#### Worked example: the cost of trusting a false signal

Let us price the 2024 mistake. Suppose in August 2024 you took the Sahm trigger at face value and rotated a **\$5,000,000** book from 80% stocks to 40% stocks, parking the difference in long bonds. Over the following year, instead of a recession, the economy kept growing — stocks returned, say, **+15%** and long bonds roughly **+2%**.

```
had you stayed 80/20 (no false rotation):
  stocks  \$4,000,000 x 1.15 = \$4,600,000
  bonds   \$1,000,000 x 1.02 = \$1,020,000
  total                        \$5,620,000   (+12.4%)

after the false rotation to 40/60:
  stocks  \$2,000,000 x 1.15 = \$2,300,000
  bonds   \$3,000,000 x 1.02 = \$3,060,000
  total                        \$5,360,000   (+7.2%)
```

The false rotation *cost* you **\$260,000** — about **5 percentage points** of return — versus simply holding. That is the price of a false positive, and it is the mirror image of the +\$800,000 swing the *true* recession rotation earned earlier. The asymmetry is the whole game: the recession signal is enormously valuable when right and meaningfully costly when wrong, so you do not act on it mechanically — you require *confirmation* (claims breaking, spreads widening) before you commit the rotation. The single tell that would have saved you \$260,000 in 2024 was that initial claims never broke higher.

## Common misconceptions

A handful of beliefs about the labor data and recessions are widespread, intuitive, and wrong. Each correction comes with a number.

**Myth 1: "A low unemployment rate means the economy is safe."** No — a low unemployment rate is, if anything, a *late-cycle* warning. U-3 was 3.5% in February 2020, one month before a 14.8% spike; it was near multi-decade lows before the 2001 and 2007 recessions too. The level is a laggard that looks best at the peak. What you watch is the *change* off the low (the Sahm rule's +0.5pp), not the comfortable-looking level.

**Myth 2: "Two negative quarters of GDP is the definition of a recession."** That is a rule of thumb, not the definition. The NBER defines a recession as a broad, sustained decline and weighs employment, income, and production — and it dates recessions *in hindsight*, often a year late. In 2022 the US had two negative GDP quarters and the NBER declared *no recession*, precisely because the labor market kept growing. GDP is a lagging, heavily-revised number; do not let it anchor your recession call.

**Myth 3: "The yield curve inverting means a recession is imminent."** The curve inverting means a recession is *probable within the next year or two* — the historical lead from first inversion to recession start ranges from **6 to 22 months**, averaging ~15. "Imminent" is wrong by up to two years. The curve is an early *alert*, not a *timing* tool; you pair it with the late-firing labor data to time the actual rotation.

**Myth 4: "Wide credit spreads mean you should sell stocks."** Counterintuitively, the opposite is closer to true *on a forward basis*. The correlation of the high-yield spread with the *next-12-month* equity return is **positive (r ≈ +0.77)**: the widest spreads (~10%+) have preceded the *highest* forward returns (~+35%), because the spread is widest at maximum fear, which is when prices have already overshot to the downside. Wide spreads are a contemporaneous danger sign and a forward opportunity sign at the same time.

**Myth 5: "The Sahm rule is infallible — it never gives false signals."** It went fifty years without a false positive, then gave one in 2024 when the unemployment rate rose for a labor-*supply* reason (more job-seekers) rather than a layoff reason. A rule with a perfect track record is not a law of nature; it is a correlation that held while its underlying mechanism held. When the mechanism changes — here, the composition of who is unemployed — the rule can break. Always decompose the signal: does the layoff data (claims) confirm what the rate is implying?

**Myth 6: "A strong jobs report is always good for stocks."** Not in every regime. The *sign* of the correlation between the jobs data and stocks flips with the dominant fear. In a normal expansion, a strong jobs print is good news (growth is healthy, earnings will be fine) and stocks rise. But in an inflation-fear regime like 2022–23, "good news was bad news": a strong jobs report meant the central bank would have to hike rates *more* to cool the economy, so strong jobs *hurt* stocks. The labor data's correlation with risk assets is itself regime-dependent — it keys on growth fear in a slowdown and on inflation/rate fear in an overheating. (This sign-flip is built out in [NFP and asset prices](/blog/trading/macro-correlations/nfp-and-asset-prices-the-king-of-data-correlation).) The lesson: always ask *what the market is currently afraid of* before you assume the sign of a labor surprise.

## How to read it and use it

Here is the practitioner's playbook for the labor-and-recession correlation, distilled.

![The order of operations when each recession signal actually fires](/imgs/blogs/unemployment-claims-and-the-recession-correlation-7.png)

The timeline figure shows the order of operations, and the order is the whole strategy. Read the signals in this sequence:

1. **The alert (T−14 months): the yield curve inverts.** When 2s10s goes negative, put recession on the watchlist — but do *not* yet de-risk, because the lead is 6–22 months and you cannot run defensive for two years. The inversion sets the prior.
2. **The leading edge (T−3 months): initial claims break higher.** Watch the 4-week average of initial claims. When it breaks convincingly above its recent range, the layoff feedback loop may be starting. This is the first *actionable* labor signal, because claims lead the rate by ~2 months. Begin tilting.
3. **The repricing (T−1 month): spreads widen, the curve bull-steepens.** Credit spreads (which lead equity drawdowns by ~3 months) widen, and the curve un-inverts by steepening from the front as the market prices cuts. This is the market confirming what claims suggested. Commit the risk-off rotation here.
4. **The confirmation (T+2 to +3 months): the Sahm rule fires, the unemployment rate visibly rises.** This is the *lagging* confirmation. If you waited for this, you are late — the rotation should already be on. Use it to *confirm*, not to *initiate*.
5. **The payoff (T+6 months): bonds and cash have already beaten stocks.** The recession correlation has played out: defensives over cyclicals, bonds over stocks. Your early rotation is paying.

**The signal, in one line:** watch claims (the leader) and the Sahm rule (the change, not the level), require spread confirmation, and trade the *recession probability* they imply — not the labor number itself.

**The regime check (the most important step):** before you act on any labor signal, ask *why* the unemployment rate is changing. Is it rising because of layoffs (claims confirm → real recession risk → rotate to defense) or because of a labor-supply surge (claims do *not* confirm → likely a false positive → do nothing)? The 2024 episode is the canonical example: the rate rose, the Sahm rule fired, but claims never broke — and the signal was wrong. **A labor signal without claims confirmation is a hypothesis, not a trade.**

**What invalidates the signal:** an *exogenous* shock (pandemic, war, financial accident) arrives faster than any leading labor indicator can warn — in 2020 the claims spike was coincident with the crash, not ahead of it. The labor leg leads slow-building *endogenous* recessions; it cannot front-run a meteor. And any time the *composition* of the labor data shifts (immigration waves, participation swings, definitional changes), the historical leads and thresholds should be treated as suspect until the mechanism is re-verified.

**How to size it, given the asymmetry:** the two worked examples above are the crux of the playbook. A *correct* recession rotation earned roughly an \$800,000 swing on a \$10M book; a *false* one cost \$260,000 on a \$5M book (about \$520,000 if scaled to \$10M). Because the upside of being right exceeds the downside of being wrong but the false-positive cost is still real, the rational response is to *scale into* the rotation as confirmations stack, rather than flipping the whole book on the first signal. Start tilting on the claims break (a partial shift), add as spreads widen (the credit market agreeing), and complete the rotation only when multiple independent signals — claims, spreads, the curve steepening, the Sahm trigger — all point the same way. Each additional confirmation raises the recession probability and justifies more of the rotation. This turns a binary, fragile bet ("recession: yes or no?") into a graded, robust one ("how high is recession probability, and how much defense does that justify?"), which is exactly how a professional treats a signal with a known false-positive rate.

The deepest reason this works is that the labor leg, the curve, and the credit market are *partially independent* windows onto the same recession variable. When they agree, the agreement is informative precisely because they could have disagreed — and in 2024 they *did* disagree (the Sahm rule fired but claims and spreads stayed calm), which was the tell. Demanding agreement across independent signals is the single most powerful protection against the structural-shift false positive, and it is the habit that separates a trader who survives a cycle from one who gets faked out by a single rule.

The deepest point is the one this whole series keeps returning to: the labor-recession correlation is not a fixed law. It is a regime relationship that holds while its mechanism holds — and the labor data is valuable precisely because it is one of the most direct, mechanism-rich windows onto the variable that flips the regime: the probability of recession. Get that probability right, early, and you get the biggest cross-asset correlation of all — risk-off — on your side.

## Further reading and cross-links

Within this series:

- [Lead, lag, or coincident: the time axis of every correlation](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators) — the cross-correlation function and where claims, U-3, and the curve sit on the time axis.
- [NFP and asset prices: the king of data](/blog/trading/macro-correlations/nfp-and-asset-prices-the-king-of-data-correlation) — the monthly jobs report and its same-day cross-asset betas.
- [The yield curve as a growth signal and its asset correlation](/blog/trading/macro-correlations/the-yield-curve-as-a-growth-signal-and-its-asset-correlation) — the curve's shape and what it implies for every asset.
- [Credit spreads: the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary) — why spreads are the cleanest market-priced recession barometer.
- [The business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock) — the full rotation of asset returns across all four phases.
- [Correlation during crises: when diversification fails](/blog/trading/macro-correlations/correlation-during-crises-when-diversification-fails) — the everything-to-one failure mode that defined March 2020.
- [Spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data) — the structural-shift lens that explains the 2024 Sahm false positive.

For the underlying mechanisms (the *why* behind the moves):

- [Reading the yield curve: slope, inversion, recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) — how an inverted curve forecasts cuts and a slowdown.
- [The business cycle: four phases for traders](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) — the cycle framework that organizes the rotation.
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — the cross-asset view of diversification failure.
