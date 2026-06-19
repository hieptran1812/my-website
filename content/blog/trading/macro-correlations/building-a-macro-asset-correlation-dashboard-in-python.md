---
title: "Building a Macro–Asset Correlation Dashboard in Python"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A reproducible pandas workflow: pull macro series and asset prices, align them, convert to changes, build a rolling correlation matrix, and plot a heatmap plus rolling-correlation lines you can re-run every week."
tags: ["macro", "correlation", "python", "pandas", "dashboard", "rolling-correlation", "heatmap", "fred", "quant-tools", "data-analysis"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A correlation dashboard is a six-stage pipeline you can run in 60 lines of pandas: pull macro series and asset prices, align them onto one calendar, convert to *changes* (never levels), then `.corr()` for the snapshot and `.rolling(window).corr()` for the time series. The output — a heatmap plus a few rolling-correlation lines — tells you which relationships are live *this week*, not which were true on average over twenty years.
>
> - Correlate **returns and changes**, not prices and levels. Two series that both trend up will show a correlation near +0.9 even when their week-to-week moves are unrelated — that is the single most common way a dashboard lies to you.
> - The **window is a choice, not a default**. A 63-day (one-quarter) window is twitchy and catches regime flips early; a 252-day (one-year) window is smooth and lags. The dashboard should show both.
> - `df.corr()` gives you the full-sample number; `.rolling(252).corr()` gives you the *movie*. The movie is where the money is, because correlation is a regime, not a constant.
> - The one number to remember: the stock–bond correlation went from about **−0.40** for two disinflation decades to **+0.60** in 2022. A dashboard that only printed the full-sample average would have hidden the most important risk event of the decade.

In October 2022, a portfolio manager I know pulled up the same 60/40 risk model his firm had run since 2009. The model assumed stocks and bonds were negatively correlated — when equities sold off, Treasuries would rally and cushion the blow. That assumption had been roughly true for two decades. It was the entire engine of the "balanced" portfolio. And in 2022 it was catastrophically wrong: stocks fell about 18% and long Treasuries fell more than 25% *in the same year*. The hedge had become a second bet on the same risk. The model didn't see it coming because nobody was looking at the correlation as a *live, moving number*. It was a constant baked into a spreadsheet from 2011.

The fix is embarrassingly simple, and it is the subject of this post. You build a small dashboard — a script you run every Monday morning — that pulls the macro series and asset prices you care about, computes their rolling correlation, and draws you a picture. When the stock–bond number crosses from negative to positive, you *see* it the week it happens, not the quarter after your portfolio has already absorbed the damage. The dashboard is maybe 60 lines of pandas. The discipline it enforces — looking at correlation as a film rather than a photograph — is worth more than any single trade.

This is a Track G post: it is a *toolkit*. We will walk every line of the workflow — imports, pulling series from FRED, aligning and resampling, converting to changes, the correlation matrix, the rolling correlation, the heatmap — and explain the *why* behind each step, because the failure modes are all in the "why." By the end you will have a dashboard you can run weekly and, more importantly, you will understand the three or four judgment calls inside it that separate a useful tool from a number generator that fools you.

![Correlation dashboard pipeline pull align changes roll correlate plot](/imgs/blogs/building-a-macro-asset-correlation-dashboard-in-python-1.png)

## Foundations: what a correlation dashboard actually computes

Before any code, let us be precise about the object we are building, because almost every mistake in macro correlation work is a confusion about *what is being correlated*.

**Correlation** is a single number between −1 and +1 that summarizes how two series move *together*. The specific number we use is the **Pearson correlation coefficient**, usually written *r*. If *r* = +1, the two series move in perfect lockstep; when one is above its average, the other is above its average by a proportional amount. If *r* = −1, they move in perfect opposition. If *r* = 0, knowing one tells you nothing about the other. The formula is the covariance of the two series divided by the product of their standard deviations:

```
r = cov(X, Y) / (std(X) * std(Y))
```

The covariance in the numerator measures whether the two series tend to be on the same side of their means at the same time; dividing by the standard deviations rescales that into the clean [−1, +1] range so you can compare a stock–bond correlation to a gold–dollar correlation on the same scale. We derive this properly in [the covariance matrix and linear algebra primer](/blog/trading/math-for-quants/covariance-matrix-linear-algebra-math-for-quants); here we just need to know what goes in and what comes out.

The critical question is: *correlation of what?* You have two choices for any macro series or asset, and they give wildly different answers:

- The **level**: the actual value. The gold price is \$2,650/oz. The 10-year yield is 4.48%. The CPI index is 312. The S&P 500 is at 6,100.
- The **change**: how the level moved over one period. Gold rose 1.2% this week. The 10-year yield rose 8 basis points. The S&P 500 returned −0.6% today.

A dashboard correlates **changes**, almost never levels. The reason is the single most important idea in this entire post, so we will give it its own section and its own figure below. For now, hold this rule: *correlate returns, correlate yield changes, correlate surprise; never correlate prices.*

The second piece of vocabulary is the **window**. A correlation is computed over some span of observations. "The full-sample correlation" uses every data point you have. A "**rolling** correlation" recomputes *r* over a sliding window — say, the last 252 trading days — and steps that window forward one day at a time, producing a *time series of correlations*. The full-sample number is a photograph; the rolling number is a film. Because macro correlations flip across regimes — a theme we develop in [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) — the film is what you actually want.

Put those together and the dashboard is two views of the same data:

1. A **correlation matrix** (a heatmap): the current-window *r* between every pair of series. This is your snapshot of the regime *right now*.
2. A set of **rolling-correlation lines**: how the most important pairwise correlations have evolved over time, so you can see flips coming.

Everything else — pulling data, aligning calendars, computing returns — is plumbing in service of those two pictures. Let us build the plumbing.

### Which correlation: Pearson, Spearman, and how much to trust the number

There is one more foundational choice the dashboard makes for you, and you should make it on purpose. `df.corr()` defaults to **Pearson** *r*, which measures *linear* co-movement — straight-line, proportional. It is the right default for most macro work and it is what every number in this post uses. But Pearson has two blind spots worth knowing.

First, Pearson is sensitive to **outliers**. One enormous day — a COVID crash, a flash event — can swing a Pearson correlation by 0.2 all by itself, because the squared deviations in the formula give a single huge move outsized weight. If you want a measure that asks only "did they move the same *direction*, ignoring magnitude," use **Spearman** rank correlation, `df.corr(method="spearman")`, which correlates the *ranks* of the observations rather than their values. Spearman is robust to outliers and to non-linear-but-monotonic relationships. A useful diagnostic: when Pearson and Spearman disagree sharply, a few extreme points are driving your Pearson number, and you should look at the scatter before trusting it. The full taxonomy of these measures lives in [what correlation actually measures: Pearson, Spearman, beta](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta).

Second, a correlation computed from *few* observations is **noisy** and can be large by pure chance. A rough rule: the standard error of a correlation is about 1/√n, so an *r* estimated from 25 weekly observations has a standard error near 0.20 — meaning a "correlation of 0.3" is statistically indistinguishable from zero. This is the deep reason the window choice matters: a 13-week window gives you fast reactions but each number carries a ±0.28 cloud of uncertainty, while a 104-week window pins the estimate down to roughly ±0.10. The dashboard never shows you that uncertainty band — so you must hold it in your head, especially for short windows and for any series with a short history. When a brand-new asset shows a striking 0.6 correlation off 20 data points, the honest read is "could easily be noise," not "found an edge."

#### Worked example: is a 0.3 correlation real?

Your dashboard's 26-week window shows the dollar correlating −0.30 with the S&P this quarter, and you are tempted to put on a \$50,000 dollar-hedge against the equity book on the strength of it. Check the significance first. With n = 26 observations the standard error is about 1/√26 ≈ 0.20. A −0.30 correlation is therefore only about 1.5 standard errors from zero — well short of the ~2.0 you'd want for "probably real." Translated to the trade: you would be risking \$50,000 of hedging cost and basis risk on a relationship that has roughly a one-in-seven chance of being pure noise. Widen the window to 104 weeks, where the standard error drops to ≈0.10; if the −0.30 *holds* at that window it is now three standard errors from zero and worth acting on. The intuition: a small correlation off a short window is a rumor, not a signal — confirm it survives a longer window before you spend a dollar on it.

## The pipeline, stage by stage

The whole dashboard is the six-stage pipeline in the figure above: **pull → align → changes → roll → correlate → plot**. We will write the code for each stage, run it in our heads, and call out the trap at each step. The full script at the end stitches them together.

### Stage 1: imports and the data sources

We use three libraries. `pandas` does all the data wrangling (alignment, resampling, returns, the correlation itself). `pandas-datareader` is a thin wrapper that pulls economic series straight from **FRED** — the Federal Reserve Bank of St. Louis's free database, which hosts essentially every macro series a trader cares about: CPI, the 10-year yield, real yields, the dollar index, oil, unemployment, credit spreads. `yfinance` (or any price source) pulls asset prices. `matplotlib` and `seaborn` draw the pictures.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import yfinance as yf
from datetime import datetime
```

FRED series are identified by short codes. The ones that matter most for a macro correlation dashboard:

```
fred_codes = {
    "DGS10": "UST10Y",      # 10-year Treasury nominal yield, %
    "DFII10": "UST10Y_real",  # 10-year TIPS real yield, %
    "T10YIE": "Breakeven10Y",  # 10-year breakeven inflation, %
    "DTWEXBGS": "DXY",      # broad trade-weighted US dollar index
    "DCOILWTICO": "WTI",    # WTI crude oil, USD/bbl
    "BAMLH0A0HYM2": "HY_OAS",  # high-yield credit spread, %
    "CPIAUCSL": "CPI",      # CPI index level (we will difference it)
}
```

A small but real point: FRED gives you the macro series; your broker or `yfinance` gives you the asset prices. They come from different places, on different calendars, with different update lags. Reconciling them is Stage 2 — and it is where most dashboards quietly break.

### Stage 2: pulling the series

`pandas-datareader` turns each FRED code into a pandas `Series` indexed by date. We loop over the codes and assemble a single DataFrame:

```
start = datetime(2010, 1, 1)
end = datetime.today()

macro = {}
for code, name in fred_codes.items():
    macro[name] = web.DataReader(code, "fred", start, end)[code]
macro = pd.DataFrame(macro)
```

For asset prices we pull adjusted closes, which already fold in dividends and splits so that a price change *is* a total return:

```
tickers = {"^GSPC": "SP500", "^NDX": "Nasdaq", "GLD": "Gold_ETF",
           "BTC-USD": "Bitcoin", "TLT": "LongBond_ETF"}
prices = yf.download(list(tickers), start=start, end=end)["Close"]
prices = prices.rename(columns=tickers)
```

> A note on reproducibility: FRED is occasionally unreachable from inside sandboxed or corporate networks, and the exact series codes drift over the years. Every number in this post's figures comes from a *curated, cited* dataset rather than a live pull, so the charts are stable. The code above is the real workflow you would run against a live FRED connection; treat the figures as what its output looks like.

At this point you have two DataFrames — `macro` and `prices` — each indexed by date, but on *different and misaligned* calendars. FRED's `DGS10` is published every business day; CPI is monthly; Bitcoin trades on weekends; the S&P does not. If you naively glued these together you would get a DataFrame riddled with `NaN`s, and `.corr()` would silently drop rows in ways you cannot see. Stage 2's output is two clean DataFrames; Stage 3 makes them speak the same calendar.

### Stage 3: align and resample onto one calendar

This is the stage everyone underestimates. You cannot correlate a daily series with a monthly series directly — they don't have the same number of observations or the same dates. You have to decide on **one frequency** and force every series onto it. For a macro dashboard, **weekly** is the sweet spot: daily is noisy and full of holiday gaps; monthly throws away too much of the regime detail you are hunting for.

We resample everything to weekly. For *prices and yields* (which are levels you can observe at any instant), the right resample is "last observation in the week":

```
macro_w = macro.resample("W-FRI").last()
prices_w = prices.resample("W-FRI").last()
```

`W-FRI` anchors each weekly bucket on Friday — the natural close of the trading week. For a *flow* series you might use `.mean()` or `.sum()` instead of `.last()`, but yields and prices are *stocks* (point-in-time values), so "last" is correct.

Now we join the two DataFrames on their shared weekly index and handle the holes. CPI is monthly, so on most weeks its cell is empty; we forward-fill it so each week carries the most recent known print:

```
df = macro_w.join(prices_w, how="outer")
df["CPI"] = df["CPI"].ffill()        # carry last monthly print forward
df = df.dropna(how="any")            # then drop weeks still missing anything
```

Two judgment calls live here, and both matter. **Forward-filling** a monthly series onto a weekly grid is the standard move, but be honest about what it means: between CPI releases, the value is *stale*, not new information, so a correlation that uses the forward-filled level can overstate how "live" the relationship is. (For CPI we will mostly use the *change at the release*, which sidesteps this.) And **`dropna`** silently deletes rows — if Bitcoin has a gap or a series starts late, you can lose years of data without noticing. Always print `df.shape` before and after and sanity-check that you kept what you expected.

There are three calendar gotchas specific to a *macro* dashboard that catch even experienced people, and they all live in this stage:

- **Weekend assets vs weekday assets.** Bitcoin and FX trade seven days a week; stocks, FRED yields, and most macro series do not. If you resample to weekly with `W-FRI`, Bitcoin's Friday close aligns cleanly with the equity Friday close — good. But if you ever work at *daily* frequency, you must decide whether to drop weekends (losing Bitcoin's Saturday/Sunday moves) or keep them (creating `NaN`s for every equity series on weekends). Weekly resampling sidesteps the whole problem, which is the main reason a macro dashboard should be weekly, not daily.
- **Release lag and timestamps.** FRED stamps a data point with the date it *refers to*, not the date it was *released*. CPI for June is stamped early June but isn't published until mid-July. If you align by the reference date, you are implicitly assuming you knew June's CPI in June — a one-month look-ahead that will inflate any correlation that uses CPI. For a live dashboard this rarely bites (you're looking at the past), but the moment you backtest with this data it becomes a fatal bug. Use the `ALFRED` vintage data or lag the release by its known publication delay.
- **Time zones.** A US equity close is 4pm New York; a "daily" crypto close from `yfinance` is midnight UTC. Correlating them day-on-day quietly offsets the two series by several hours, which on a volatile day can flip the sign of a daily correlation. Weekly resampling washes this out too — another vote for weekly.

#### Worked example: how a calendar mismatch silently halves your sample

Suppose you pull 10 years of weekly data — about 520 weeks. Your equity and yield series are complete, but your credit-spread series (`BAMLH0A0HYM2`) only goes back 7 years on your data plan, and Bitcoin has 30 scattered missing weekends that your weekly resample didn't fully clean. You write `df.dropna(how="any")` and move on. The result: every week before the credit series starts is dropped, taking you from 520 rows to about 364 — you have silently thrown away 30% of your history, including the entire 2015–2016 period. If you then compute a correlation and quote it as "10-year," you are lying by 3 years. The fix costs nothing: print `df.shape` before and after the `dropna`, and if the drop is large, correlate each pair on its own *maximal* overlapping window with `df[[a, b]].dropna()` rather than forcing every series onto the shortest one. The intuition: `dropna(how="any")` correlates everything on the *intersection* of all calendars, which is as short as your single worst series.

### Stage 4: convert levels to changes — the rule that prevents the biggest lie

Here is the heart of the post. **You must convert every series from a level to a change before you correlate it.** Skipping this step is the number-one way a macro correlation dashboard produces confident, professional-looking, completely fake numbers.

Why? Because almost every macro and asset series *trends*. The S&P 500 trends up. Gold trends up. The level of US federal debt trends up. CPI (the index, not the rate) trends up. When two series both trend in the same direction over a long sample, their *levels* will be strongly correlated — mechanically, trivially, meaninglessly — even if their period-to-period moves have nothing to do with each other. This is **spurious correlation**, and we devote a whole post to its traps in [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data). The cure on the dashboard is one line of pandas.

For *prices*, the change you want is the percent return:

```
returns = df[price_cols].pct_change()
```

For *yields, spreads, and rates* (already in percent), the change is the simple difference, because a "10% return on a yield" is nonsense — you want "the yield rose 8 basis points":

```
rate_changes = df[rate_cols].diff()
```

For an *index like CPI*, you usually want the period-over-period or year-over-year percent change to get the inflation *rate*:

```
df["CPI_yoy"] = df["CPI"].pct_change(52) * 100   # ~52 weeks = 1 year
```

Then you assemble one DataFrame of *changes* — returns for prices, diffs for rates — and that is the object you correlate. Never the levels.

The figure below shows exactly why, using two series that both relentlessly rise: US federal debt and the S&P 500. In **levels**, they correlate at +0.90 — a number that would make a junior analyst declare "the stock market runs on government borrowing." But that correlation is pure shared trend. Convert both to year-over-year **changes** and the correlation collapses to −0.09 — essentially zero. The week-to-week (or year-to-year) moves of debt and stocks are unrelated; the +0.90 was an artifact of two things going up over time.

![Changes versus levels US debt and S and P 500 correlation](/imgs/blogs/building-a-macro-asset-correlation-dashboard-in-python-5.png)

#### Worked example: a correlation from two return series, by hand

Let us compute one *r* the way the dashboard does, on real changes, so the formula stops being abstract. Take six years of S&P 500 returns and the matching year-over-year change in federal debt:

| Year | S&P return | Δ federal debt (\$T) |
|---|---|---|
| 2020 | +16.2% | +4.2 |
| 2021 | +26.9% | +1.5 |
| 2022 | −19.4% | +2.5 |
| 2023 | +24.2% | +2.3 |
| 2024 | +23.3% | +2.3 |
| 2025 | +3.7% | +1.9 |

The mean S&P return is about +12.5%; the mean debt change is about +2.45T. The covariance is the average product of the two columns' deviations from their means. Notice 2022: stocks were *down* 19.4% (a big negative deviation) while debt grew +2.5T (a near-average positive deviation) — that pairing pulls the covariance toward zero. Working it through, the covariance is slightly negative, the standard deviations are large (returns swing from −19% to +27%), and the resulting *r* is **−0.09**. In words: years of fast debt growth were *not* good years for stocks; the relationship is noise. The intuition: once you strip the shared upward trend by differencing, the spurious +0.90 evaporates and the honest near-zero appears — which is exactly the number a position-sizer should act on, because a \$1 trillion swing in the pace of borrowing told you nothing about next year's equity return.

### Stage 5: the correlation matrix — `df.corr()`

With a DataFrame of changes in hand, the snapshot view is a single method call:

```
corr_now = changes.tail(252).corr()    # last ~1 year, weekly or daily
```

`changes.corr()` returns a square matrix: every column correlated with every other column, +1.00 on the diagonal (each series with itself), symmetric off the diagonal. Slicing `.tail(252)` first restricts it to the most recent window so the matrix describes *the current regime* rather than a decade-long average that may be obsolete. That windowing choice is itself load-bearing — a full-sample `df.corr()` is the photograph that hid the 2022 stock–bond flip.

To make the matrix readable you draw it as a **heatmap** — a grid where each cell's color encodes the correlation, red for negative and green for positive, with the number printed in the cell:

```
plt.figure(figsize=(11, 9))
sns.heatmap(corr_now, annot=True, fmt="+.2f", cmap="RdYlGn",
            vmin=-1, vmax=1, center=0, square=True,
            cbar_kws={"label": "correlation of changes"})
plt.title("Cross-asset correlation (last 252 obs)")
plt.tight_layout()
```

`vmin=-1, vmax=1, center=0` pins the color scale so that white is always zero and the colors are comparable across runs — without it, seaborn auto-scales the palette and a matrix of all-positive correlations looks the same as one spanning −1 to +1, which is dangerously misleading. The result is the cross-asset heatmap below: the `df.corr()` output of a real dashboard, showing how the major assets co-move. Note the +0.92 between the S&P and Nasdaq (they are nearly the same bet), the −0.55 between gold and the dollar (gold is the dollar's rival as money), and the modest +0.40 between equities and Bitcoin (crypto trades partly as high-beta risk, a relationship we trace in [crypto as a macro asset](/blog/trading/macro-correlations/crypto-as-a-macro-asset-the-liquidity-correlation)).

![Cross asset correlation matrix heatmap S and P Nasdaq bond gold bitcoin dollar](/imgs/blogs/building-a-macro-asset-correlation-dashboard-in-python-4.png)

This heatmap is the dashboard's resting state — the thing you glance at first. But the more powerful view comes from putting the *macro drivers* on one axis and the *assets* on the other, so you can read down a column ("what moves gold?") or across a row ("what does a hot CPI print hit?"). That driver-by-asset matrix is the signature view of this whole series, and it is the next figure.

### Reading the signature view: the driver × asset matrix

The full dashboard's headline panel is a matrix whose **rows are macro drivers** (a rise in the 10-year yield, a hot CPI surprise, a stronger dollar, a wider credit spread) and whose **columns are assets**. Each cell is the correlation of that driver's *change or surprise* with that asset's *return*. This is the map you actually trade off: it tells you, for any incoming macro shock, which way each asset is wired to move.

![Macro driver versus asset correlation matrix heatmap](/imgs/blogs/building-a-macro-asset-correlation-dashboard-in-python-2.png)

Read it row by row. The "10Y yield (rise)" row is almost solidly red across risk assets — when yields jump, stocks (−0.45), Nasdaq (−0.60), gold (−0.55), and bonds themselves (−0.95) all fall, while the dollar (+0.45) rises. That is the master correlation we unpack in [bond yields, the master correlation with every asset](/blog/trading/macro-correlations/bond-yields-the-master-correlation-with-every-asset). The "real yield (rise)" row is gold's worst nightmare (−0.80) because real yields are the true price of holding a zero-coupon asset, which is the cleanest macro correlation there is and the subject of [real yields and the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation). The "stronger USD" row shows the dollar's cross-asset gravity. And the "credit spread (wider)" row is the risk-off signature — equities −0.70 while bonds catch a +0.30 bid.

#### Worked example: reading the heatmap into a position

Suppose your dashboard refreshes Monday and the incoming macro story is a hot core-CPI surprise plus a 12 bp jump in the 10-year real yield. Walk the relevant rows. The "CPI surprise (hot)" row says Nasdaq −0.65, the "real yield (rise)" row says gold −0.80. You run a \$500,000 book that is 60% long Nasdaq-heavy tech and holds \$100,000 in gold as a "diversifier." The heatmap is telling you those are not two independent bets in this regime — both are short the same macro factor (rising real rates). A −0.10 surprise (a moderately hot print) with the historical betas implies roughly a −1.0% to −1.3% hit to the Nasdaq sleeve and a −0.8% hit to gold, so the gold is *adding* to the loss, not offsetting it. Concretely, your \$300,000 tech sleeve might drop about \$3,500 and your \$100,000 gold position about \$800 — the "diversifier" lost money alongside the stocks. The position decision the dashboard informs: in a real-rates-up regime, gold is not a hedge for tech; if you want a true offset you need something on the green side of those rows (a stronger dollar, or simply less duration). The intuition: a diversifier is only a diversifier against the factor that is actually moving — and the heatmap tells you which factor that is *this week*.

### Stage 6: the rolling correlation — `.rolling(window).corr()`

The matrix is a snapshot. To see correlations *evolve* — to catch a flip the week it happens — you compute a **rolling pairwise correlation**. pandas makes this a one-liner: call `.rolling(window).corr()` on one column with another passed as the argument:

```
window = 252   # ~1 year of daily obs, or use 52 for weekly
roll = changes["SP500"].rolling(window).corr(changes["LongBond_ETF"])
```

`roll` is now a *time series*: for each date, the correlation of S&P and long-bond returns over the trailing `window` observations. Plot it with a zero line and you get the single most important chart in macro risk management — the stock–bond rolling correlation, which spent two decades negative (bonds diversified stocks) and snapped positive in 2022 (bonds and stocks fell together).

![Stock bond rolling correlation 1990 to 2025](/imgs/blogs/building-a-macro-asset-correlation-dashboard-in-python-3.png)

That single line *is* the warning system the 2022 portfolio manager lacked. From roughly 1998 to 2021 it sat between −0.30 and −0.55 — the green "bonds diversify" zone. In 2022 it rocketed to +0.60 — the red "both fall together" zone — as inflation became the dominant risk and repriced stocks and bonds simultaneously. A dashboard that drew this line would have flashed the regime change in early 2022. We dissect the mechanism in [the stock–bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) and the specific break in [when correlations break, the 2022 stock–bond flip](/blog/trading/macro-correlations/when-correlations-break-the-2022-stock-bond-flip).

To plot it with the regime shading shown above:

```
fig, ax = plt.subplots(figsize=(14, 7))
ax.axhline(0, color="0.4", lw=1.2)
ax.axhspan(0, 1, color="red", alpha=0.08)     # both fall together
ax.axhspan(-1, 0, color="green", alpha=0.08)  # bonds diversify
ax.plot(roll.index, roll, lw=2)
ax.set_title("Rolling stock-bond correlation")
ax.set_ylabel("rolling correlation")
```

## The window is a choice, not a default

Every `.rolling(window)` call hides a decision that changes the answer, and beginners almost always leave it on a thoughtless default. The window is the number of observations the correlation is computed over. A **short window** (say 63 trading days, one quarter) reacts fast — it catches a regime flip within a few weeks — but it is *twitchy*: it will show you a "−0.7 correlation" that is half noise and that whips around. A **long window** (252 days, one year) is *smooth* and statistically stable, but it *lags* — it averages the flip together with the old regime, so it confirms a change only months after it happened.

There is no single right window; there is a right window *for your question*. If you are managing tail risk and want early warning, you weight the short window. If you are setting a strategic allocation, you weight the long one. The dashboard's job is to show **both** so you can see the disagreement — when the 63-day line has flipped sign but the 252-day line hasn't yet, that *gap itself* is the signal that a regime is turning. We dedicate an entire post to this trade-off in [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters); here is the code to plot both at once:

```
for w, label in [(63, "63d (1 quarter)"), (252, "252d (1 year)")]:
    r = changes["SP500"].rolling(w).corr(changes["LongBond_ETF"])
    ax.plot(r.index, r, lw=2, label=label)
ax.legend()
```

#### Worked example: how the window changes the trade

It is March 2022. The stock–bond correlation has just turned. Your 63-day rolling window reads +0.35 — clearly positive, the regime has flipped. Your 252-day window still reads −0.20, because it is still averaging in the placid 2021 data. You run a \$10 million "risk-parity-lite" book that sizes its bond position assuming bonds hedge stocks. If you trust the *long* window (−0.20), you keep a large levered bond position as your "hedge," and when both legs fall through 2022 you take a double loss — on a \$4 million bond sleeve, the roughly −25% move is a \$1,000,000 hit *on the supposed hedge alone*. If you trust the *short* window (+0.35), you cut the bond leverage and add a genuine diversifier, and the same shock costs you a fraction of that. The window choice was a \$700,000–\$900,000 decision. The intuition: in a turning regime the short window is *right early and noisy*, the long window is *wrong late and smooth* — so you act on the short window for risk and confirm with the long, never the reverse.

There is one more window subtlety worth a sentence: the rolling correlation you plot today is computed only from *past* data inside each window, so it does not peek into the future — but if you later use these rolling correlations to drive a backtest, you must be religious about that, because a single look-ahead alignment can manufacture a beautiful, fake edge. That discipline is the whole subject of [backtesting a correlation without fooling yourself](/blog/trading/macro-correlations/backtesting-a-correlation-without-fooling-yourself).

## A second rolling line: when an asset *becomes* a macro asset

The stock–bond line shows a correlation flipping *sign*. The next one shows a correlation switching *on and off* — Bitcoin's correlation with the Nasdaq. For most of its life, Bitcoin lived in its own world, uncorrelated with anything macro (rolling correlation near 0). Then, in the 2022 liquidity drain, it suddenly traded as a high-beta risk asset and its 90-day correlation with the Nasdaq spiked to 0.60–0.65. By 2024–25 it faded back toward 0.2–0.3. A dashboard that tracks this line tells you *what Bitcoin currently is* — a diversifier, or just leveraged tech.

![Bitcoin Nasdaq rolling 90 day correlation 2019 to 2025](/imgs/blogs/building-a-macro-asset-correlation-dashboard-in-python-6.png)

The code is identical to the stock–bond line — that is the whole point of building a dashboard. Once the pipeline exists, adding a new pair is one line:

```
btc_nasdaq = changes["Bitcoin"].rolling(90).corr(changes["Nasdaq"])
ax.plot(btc_nasdaq.index, btc_nasdaq)
```

This is why the dashboard pays off over time: the marginal cost of monitoring one more relationship is nearly zero. You pull the series once, align once, difference once, and then every pairwise correlation — current and rolling — is a method call. The macro-mechanism *why* behind these moves (why liquidity drives risk assets together) lives in [global liquidity and the everything correlation](/blog/trading/macro-correlations/global-liquidity-and-the-everything-correlation); the dashboard's job is just to *measure* it cleanly and show you when it's live.

## Adding the lead/lag dimension with shifted correlations

So far the dashboard correlates series *contemporaneously* — same-week changes against same-week changes. But some of the most valuable macro relationships are *leading*: one series moves first and the other follows weeks or months later. The yield curve inverts roughly 14 months before a recession; ISM new orders lead S&P earnings growth by about 6 months; credit spreads widen about 3 months before an equity drawdown. A contemporaneous correlation *misses* these because the two series aren't moving in the same week — they're moving on a delay.

You uncover lead/lag by correlating one series against a *shifted* version of the other. pandas' `.shift(k)` slides a column forward by `k` periods, so correlating `A` against `B.shift(k)` for a range of `k` tells you at what lag the relationship is strongest — the **cross-correlation**:

```
def lead_lag(a, b, max_lag=12):
    out = {}
    for k in range(-max_lag, max_lag + 1):
        out[k] = a.corr(b.shift(k))
    return pd.Series(out)

cc = lead_lag(changes["SP500"], changes["ISM_orders"], max_lag=12)
best_lag = cc.abs().idxmax()    # the lag with the strongest relationship
```

The lag at which `cc` peaks is the lead time. A positive peak lag means the *second* series leads the first; a peak at zero means they are coincident. This is how the dashboard graduates from "what moves together" to "what moves *first*" — the difference between a confirming indicator and a forecasting one. The full catalog of which macro series lead which assets, and by how long, is in [lead/lag, leading, coincident, and lagging indicators](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators).

#### Worked example: trading the lead, not the level

Your cross-correlation panel shows credit spreads (high-yield OAS) leading S&P drawdowns by about 3 months, with the relationship peaking at lag −3 around −0.6. This week the dashboard flags high-yield spreads widening from 3.5% to 4.3% — a meaningful 0.8-point move — while the S&P is still grinding higher and the *contemporaneous* stock–spread correlation looks calm. The lead/lag panel says: don't trust the calm contemporaneous number; the spread move is the *early* signal, and equity weakness historically follows by a quarter. On a \$1,000,000 equity book, acting on the lead — trimming \$150,000 of beta now — costs you a little upside if the signal misfires but protects against the kind of 10–15% drawdown (\$100,000–\$150,000) that has historically arrived about three months after spreads gap wider. The intuition: the most valuable correlation on the dashboard is often the one that *isn't* contemporaneous — the series that moves first is the one that lets you act before the move, which is the whole reason credit spreads earn their nickname as the market's canary, covered in [credit spreads, the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary).

## A single dashboard row: the dollar's cross-asset gravity

One more view rounds out the dashboard: a horizontal bar of a single driver's correlations across assets — a "slice" of the matrix you can stare at when one macro factor is dominating. The dollar is the classic case. A stronger dollar tightens global financial conditions and pulls down nearly everything priced in or funded by dollars: gold (−0.55), copper (−0.50), EM equities (−0.55), oil (−0.45), Bitcoin (−0.35). The one positive is the 10-year yield (+0.40), because higher US yields are *why* the dollar is rising in the first place.

![Dollar DXY cross asset correlation bars gold oil copper EM equities](/imgs/blogs/building-a-macro-asset-correlation-dashboard-in-python-8.png)

```
dxy_row = changes.corrwith(changes["DXY"]).drop("DXY").sort_values()
dxy_row.plot.barh(color=["green" if v > 0 else "red" for v in dxy_row])
```

`corrwith` is the convenience method for "correlate one column against all the others" — perfect for a single dashboard row. The mechanism behind the dollar's gravity is in [the dollar (DXY) cross-asset correlation](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation) and the cross-asset-series companion [the dollar, cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity); the dashboard simply renders it.

## The full script, stitched together

Here is the whole dashboard in one runnable block. It is the six stages back-to-back. Run it weekly; the output is the heatmap and the rolling lines.

```
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import pandas_datareader.data as web, yfinance as yf
from datetime import datetime

start, end = datetime(2010, 1, 1), datetime.today()

fred = {"DGS10": "UST10Y", "DFII10": "RealYield", "DTWEXBGS": "DXY",
        "DCOILWTICO": "WTI", "BAMLH0A0HYM2": "HY_OAS"}
macro = pd.DataFrame({name: web.DataReader(c, "fred", start, end)[c]
                      for c, name in fred.items()})

tk = {"^GSPC": "SP500", "^NDX": "Nasdaq", "GLD": "Gold",
      "BTC-USD": "Bitcoin", "TLT": "LongBond"}
prices = yf.download(list(tk), start=start, end=end)["Close"].rename(columns=tk)

macro_w = macro.resample("W-FRI").last()
prices_w = prices.resample("W-FRI").last()
df = macro_w.join(prices_w, how="outer").ffill().dropna()

rate_cols = ["UST10Y", "RealYield", "DXY", "HY_OAS"]
price_cols = ["SP500", "Nasdaq", "Gold", "Bitcoin", "LongBond", "WTI"]
chg = pd.concat([df[price_cols].pct_change(),
                 df[rate_cols].diff()], axis=1).dropna()

corr_now = chg.tail(104).corr()      # last ~2 years of weekly data

plt.figure(figsize=(11, 9))
sns.heatmap(corr_now, annot=True, fmt="+.2f", cmap="RdYlGn",
            vmin=-1, vmax=1, center=0, square=True)
plt.title("Cross-asset correlation of weekly changes (last 2y)")
plt.tight_layout(); plt.savefig("heatmap.png", dpi=120)

fig, ax = plt.subplots(figsize=(14, 7))
ax.axhline(0, color="0.4"); ax.axhspan(0, 1, color="red", alpha=0.07)
ax.axhspan(-1, 0, color="green", alpha=0.07)
for w in (26, 104):                  # 26w and 104w windows on weekly data
    r = chg["SP500"].rolling(w).corr(chg["LongBond"])
    ax.plot(r.index, r, lw=2, label=f"{w}-week window")
ax.legend(); ax.set_title("Rolling stock-bond correlation")
plt.tight_layout(); plt.savefig("rolling.png", dpi=120)
```

Sixty lines, two pictures, run it Monday morning. That is the entire tool. Everything else in this post is about the judgment that keeps those sixty lines honest.

### Making it robust and fast

Two practical notes for when this script graduates from a notebook into something you rely on.

**Speed.** For the few dozen series a human can actually read, the naive version above runs in well under a second, so do not optimize prematurely. But `df.corr()` is an O(n × k²) operation — n observations, k series — and the *rolling* version recomputes that over every window, which is O(n × k²) again per window position. If you scale to hundreds or thousands of series (a full sector or single-stock universe), the rolling-correlation loop becomes the bottleneck and you want to vectorize it: compute rolling means, variances, and covariances with `.rolling().mean()` on the products rather than calling `.corr()` in a Python loop, or move the heavy matrix math to NumPy. The general principle — keep the work inside vectorized pandas/NumPy and out of Python `for` loops — is the entire subject of the [fast-Python playbook](/blog/software-development/python-performance/the-fast-python-playbook-a-decision-framework). One concrete win: `changes.rolling(w).corr()` on the *whole DataFrame* returns a multi-indexed panel of every pairwise rolling correlation in a single vectorized call, far faster than looping pair by pair.

**Robustness.** Three guardrails keep the dashboard from lying after a data hiccup. (1) Assert the shape: `assert df.shape[0] > expected_min` so a silently shrunken sample throws an error instead of producing a confident wrong number. (2) Check freshness: print `df.index[-1]` and alert if the latest date is more than a week stale, because a frozen data feed will happily keep drawing a heatmap from last month's regime. (3) Cache the raw pulls to a local parquet file (`df.to_parquet`) so a FRED or `yfinance` outage doesn't take your Monday routine down, and so you can reproduce exactly what the dashboard showed on any past date — which you will want the first time a colleague asks "what did the stock–bond line say in March?"

#### Worked example: the cost of an un-asserted shape

Imagine the dashboard normally runs on 520 weekly rows, but one Monday `yfinance` returns an empty frame for `TLT` (a transient API error), and your `dropna` quietly drops every row, leaving 40. Without a shape assertion, `.corr()` on 40 noisy rows still returns a clean-looking matrix — say, a stock–bond correlation of −0.55 computed from a tiny, unrepresentative window. You read "−0.55, bonds are diversifying," keep a \$2,000,000 levered bond hedge, and the real current regime (which the full sample would have shown as +0.4) bites you for a six-figure loss when both legs fall. A one-line `assert df.shape[0] > 400` would have halted the run and surfaced the data error instead. The intuition: the dangerous failure mode of a data tool is never a crash — it's a *plausible wrong answer*, so you spend assertions to convert silent lies into loud errors.

## Turning the dashboard into an alert: automating the regime check

Reading the heatmap by eye every Monday works, but the highest-value events — the flips — are rare and easy to miss in a busy week. The natural next step is to make the dashboard *tell you* when something material has changed, rather than waiting for you to notice. Two cheap automations do most of the work.

**The correlation-change panel.** Compute the current-window correlation matrix and the matrix from one window ago, and subtract them. The cells with the largest absolute change are exactly where the regime is shifting:

```
now = changes.tail(window).corr()
prev = changes.iloc[-2 * window:-window].corr()
delta = (now - prev)
flips = delta.abs().unstack().sort_values(ascending=False).head(10)
```

`delta` is a matrix of "how much did each correlation move." The pairs at the top of `flips` are your watch list this week — the relationships that drifted most. A stock–bond cell that jumped from −0.3 to +0.2 (a delta of +0.5) belongs at the top of your attention, and this panel surfaces it without you scanning every cell.

**Sign-flip alerts on the rolling lines.** For each key pair, check whether the short-window rolling correlation crossed zero since last week:

```
def crossed_zero(series):
    last, prev = series.iloc[-1], series.iloc[-2]
    return (last > 0) != (prev > 0)

if crossed_zero(changes["SP500"].rolling(26).corr(changes["LongBond"])):
    alert("Stock-bond correlation changed sign")
```

A zero-crossing on the stock–bond line is the single most important alert a macro risk dashboard can fire, because it means your portfolio's core diversification assumption just inverted. Wire that one alert to an email and the dashboard has earned its keep.

#### Worked example: the alert that pays for the whole project

Run the correlation-change panel through early 2022. In January, the stock–bond 26-week correlation sits at −0.20. By March it has crossed to +0.15 — `crossed_zero` returns `True` and fires the alert. You manage a \$5,000,000 balanced book at the conventional 60/40, with \$2,000,000 in long-duration Treasuries held explicitly as the equity hedge. The alert prompts you to cut that duration in half and rotate \$1,000,000 into short-dated bills and a commodity sleeve. Over the rest of 2022, long Treasuries fell roughly 25% while bills were flat and commodities rose; the half you cut would have lost about \$250,000, so acting on the alert saved on the order of \$125,000 on the bond sleeve alone, before counting the commodity gains. The alert cost one line of code and one email. The intuition: the entire value of the dashboard concentrates into a handful of regime-flip moments — automate the detection of those moments and the rest of the tool is almost free.

This same machinery generalizes beyond US assets. Swap the FRED codes and tickers for the dollar index, US 10-year, MSCI EM, and the VN-Index, and you have a dashboard for an emerging market whose risk asset has a meaningful beta to US conditions and an inverse link to a stronger dollar — the structure behind Vietnam's imported correlations. The pipeline is identical; only the inputs change. That portability is the deeper payoff of building the tool as six clean stages rather than a one-off script: the *workflow* is the asset, and any market with public data drops straight into it.

## Common misconceptions

**"More data is always better, so use the full sample."** No. The full-sample correlation is a *photograph that averages incompatible regimes*. The stock–bond full-sample number over 1990–2025 is roughly −0.10 — a meaningless blend of two decades at −0.40 and a 2022 spike to +0.60. The full-sample number describes a world that does not exist. Use a rolling window and *read the regimes separately*; that is the entire reason the dashboard plots a line, not a single number.

**"A high correlation means the dashboard found a real relationship."** Only if you correlated *changes*. A high correlation between *levels* is the default state of any two trending series and means almost nothing — recall debt and stocks at +0.90 in levels, −0.09 in changes. The first thing to check when a correlation looks suspiciously strong is: did you difference? If not, the number is probably spurious, per [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data).

**"`df.corr()` handles the missing data for me, so alignment doesn't matter."** It "handles" it by silently dropping rows, which is exactly the problem. pandas' pairwise correlation on a DataFrame with `NaN`s computes each pair on a *different subset* of dates, so your matrix is internally inconsistent — cell (A,B) might use 500 rows and cell (A,C) only 300. Align and `dropna` *explicitly*, print the shape, and know what sample each number rests on.

**"Correlation tells me what causes what."** It does not, ever. A +0.7 correlation between the copper/gold ratio and the 10-year yield does not mean copper *causes* yields — both respond to growth expectations (see [copper, gold, and the growth–inflation signal](/blog/trading/macro-correlations/copper-gold-and-the-growth-inflation-signal)). The dashboard measures *co-movement*. The *why* — the causal mechanism — comes from the macro-trading series, e.g. [how policy moves every asset](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map). Never read a heatmap as a causal diagram.

**"A correlation near zero means the two assets are unrelated and safe to combine."** Only on average, over your window, with a *linear* measure. Pearson *r* sees only straight-line co-movement; two series can have *r* ≈ 0 and still be tightly linked through a U-shape or a tail dependence that only shows up in a crisis. And the calmest pairwise correlations are exactly the ones that snap toward +1 when everyone deleverages at once — the subject of [correlation during crises, when diversification fails](/blog/trading/macro-correlations/correlation-during-crises-when-diversification-fails). A zero on the dashboard is a *current-regime* statement, not a guarantee.

## How it shows up in real markets

**2022, the stock–bond flip.** The cleanest case for owning this dashboard. For the entire 2009–2021 era, the stock–bond rolling correlation sat negative; the 60/40 portfolio worked because the 40 hedged the 60. In early 2022, as the Fed pivoted to fight 9% inflation, the rolling 6-month correlation crossed zero and kept climbing to +0.60. A manager watching the *line* cut bond duration and added real-asset or cash diversifiers in Q1. A manager watching the *full-sample average* (still showing a comfortable negative) held the old hedge and lost on both legs — a 60/40 portfolio fell about 16% on the year, its worst since 1937. The dashboard's only job was to show the line crossing zero, and that was enough.

**2020, everything-correlation in March.** When COVID hit, the average pairwise correlation across risk assets jumped from a calm ~0.25 to roughly 0.85 in three weeks. Stocks, credit, EM, even gold initially, all fell together as funds raised cash. A static correlation matrix from February said "well diversified." A *rolling* matrix updated weekly showed the diversification evaporating in real time. The lesson the dashboard teaches: the correlations you most rely on for safety are the ones most likely to fail precisely when you need them, so you watch them *move*, not assume them fixed.

**2022–2024, gold decouples from real yields.** The cleanest macro correlation on the board — gold versus the 10-year real yield, historically around −0.82 — *broke*. From 2007 to 2021 the relationship was almost mechanical: real yields up, gold down, an *r* near −0.96 in the data. Then in 2022–2024 gold *rose* even as real yields climbed to multi-year highs, as central-bank buying and geopolitical demand overwhelmed the rates channel; the correlation flipped to roughly +0.80. The figure below shows the break — the blue 2007–2021 cloud hugging the downward fit, the red 2022–2025 points floating far *above* it.

![Gold versus real yield scatter correlation that broke](/imgs/blogs/building-a-macro-asset-correlation-dashboard-in-python-7.png)

#### Worked example: the gold–real-yield break, sized

Use the 2007–2021 fit on the figure: slope ≈ −\$354/oz per 1% rise in the real yield, *r* = −0.96. In December 2021 the real yield was about −1.0% and gold was ~\$1,800. By 2024 the real yield had risen to about +2.0% — a 3-percentage-point move. The old relationship predicted gold should *fall* by roughly 3 × \$354 ≈ \$1,060, landing near \$740/oz. Instead, gold *rose* to about \$2,390. The model was wrong by over \$1,650 an ounce. If you had run a \$200,000 short-gold-versus-real-yields trade on the "airtight" −0.96 historical correlation, the break would have cost you on the order of \$180,000 as the relationship inverted. The intuition: a correlation, however clean and however long it held, is a *regime* statement — and a dashboard that plots the rolling number (not the frozen full-sample −0.96) would have shown the relationship decaying in real time, giving you the chance to cut before the loss compounded. This is why we track the *line*, the lead/lag, and the *flip*, as catalogued across [the macro–asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix).

## How to read it and use it: the weekly routine

The dashboard is only useful if you run it on a cadence and read it the same way each time. Here is the routine the whole post has been building toward.

![Four step weekly routine for reading the correlation dashboard](/imgs/blogs/building-a-macro-asset-correlation-dashboard-in-python-9.png)

**Step 1 — refresh and sanity-check the data.** Re-run the pull, print `df.shape`, and confirm you didn't silently lose rows to a `dropna`. A dashboard that quietly shrank its sample is worse than no dashboard, because it looks authoritative while lying. Confirm the last date is recent and that no series is stale.

**Step 2 — read the heatmap for the regime.** Glance at the driver × asset matrix. Which factor is reddest down its column — is everything keying off the 10-year yield (a rates regime), off the credit spread (a risk-off regime), off the dollar (a global-tightening regime)? The dominant red column tells you *which single macro variable is currently in charge*. That maps directly onto the four-quadrant framework in [correlation by regime, the four macro quadrants](/blog/trading/macro-correlations/correlation-by-regime-the-four-macro-quadrants).

**Step 3 — check the rolling lines for flips.** Look at your two or three key rolling correlations (stock–bond above all). Has the *short* window crossed a zero or a threshold the *long* window hasn't? That gap is your early-warning. A stock–bond correlation rising through zero says "bonds are no longer your hedge — find another."

**Step 4 — translate into a position check, not a position.** The dashboard does not generate trades; it audits your *risk*. The question it answers is: *are the positions I think are independent actually independent right now?* If your "diversified" book is three different ways to be short real yields (long tech, long gold, long long-duration bonds), the heatmap will show all three glowing the same color, and you will know your real exposure is one concentrated bet, not three.

**What invalidates the signal.** Three things. First, **a regime change you can see in the line itself** — once the rolling correlation has flipped, the old number is dead; do not anchor on it. Second, **a structural break in the data** — a series that changed definition, a new asset with too short a history for a 252-window, a forward-filled monthly series masquerading as live. Third, **a crisis**, when *every* off-diagonal correlation lurches toward the same sign and the matrix's information content collapses to "everything is one trade now." When the dashboard goes all-red, its message is not "here is the structure" but "there is no structure left — cut risk." The deep treatment of *why* these correlations are unstable, and what structurally shifts them over decades, is in [structural shifts, why today's correlations aren't yesterday's](/blog/trading/macro-correlations/structural-shifts-why-todays-correlations-arent-yesterdays).

Run it every Monday. The first month it feels like overhead. The first regime change it catches before your P&L does, it pays for itself for the rest of your career.

And keep the discipline that makes the tool honest: correlate changes, not levels; choose the window on purpose and show more than one; read the rolling line over time, not the full-sample number as one fixed value; and treat every striking correlation off a short window as a rumor until a longer window confirms it. The code is the easy 10% of this; the remaining 90% is the judgment baked into those four habits. A dashboard without them is a very fast way to generate confident, professional-looking, wrong numbers — and a dashboard with them is the cheapest risk-management edge you will ever build.

## Further reading and cross-links

**The measurement itself (this series):**
- [Correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) — the thesis behind the rolling line.
- [Rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters) — the window choice in depth.
- [Spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data) — why you difference levels.
- [What correlation actually measures: Pearson, Spearman, beta](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) — the statistic itself, defined from scratch.
- [The macro–asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix) — the full driver × asset map the dashboard renders.
- [The stock–bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) and [when correlations break, the 2022 stock–bond flip](/blog/trading/macro-correlations/when-correlations-break-the-2022-stock-bond-flip).
- [Measuring beta to data surprises: an event study in Python](/blog/trading/macro-correlations/measuring-beta-to-data-surprises-an-event-study-in-python) — the sibling toolkit post for surprise betas.
- [Backtesting a correlation without fooling yourself](/blog/trading/macro-correlations/backtesting-a-correlation-without-fooling-yourself) — the discipline before you trade any of this.

**The statistics (math-for-quants):**
- [The covariance matrix and linear algebra](/blog/trading/math-for-quants/covariance-matrix-linear-algebra-math-for-quants) — where the correlation matrix comes from.
- [OLS, GLS, and regularized regression](/blog/trading/math-for-quants/regression-ols-gls-regularized-math-for-quants) — the fitted line / beta behind every scatter.
- [Stationarity and autocorrelation](/blog/trading/math-for-quants/stationarity-autocorrelation-math-for-quants) — the formal reason you difference a trending series.

**The mechanisms (why the correlations exist):**
- [Interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and [real vs nominal: the real-yield master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).
- [How policy moves every asset: the cross-asset transmission map](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map).

**Performance, if you scale it up:**
- [The fast-Python playbook](/blog/software-development/python-performance/the-fast-python-playbook-a-decision-framework) — vectorizing pandas when the dashboard grows to thousands of series.
