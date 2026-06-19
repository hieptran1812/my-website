---
title: "Measuring Beta to Data Surprises: An Event Study in Python"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How to turn a vague macro correlation into a tradeable number: build a surprise dataset, align it to event-window returns, regress with OLS, and read the beta, its t-stat, and its regime split in Python."
tags: ["macro", "correlation", "event-study", "beta", "ols", "python", "data-surprise", "cpi", "nfp", "regression", "regime", "trading"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A correlation becomes tradeable only when you reduce it to a **beta**: the slope of an event-study regression of the asset's event-window return on the standardized data surprise. This post builds that regression end-to-end in Python and shows you how to read its four output numbers — beta, t-stat, confidence interval, R-squared — and how to split it by regime.
>
> - The recipe is mechanical: assemble release dates + actual + consensus, compute the surprise (actual − consensus), standardize it (divide by its historical standard deviation), align each release to the asset's return over a matched event window, then regress return on surprise with OLS and robust standard errors.
> - The **slope is the beta**: in the 2022-23 inflation-fear regime, an S&P event-study gives roughly −7% per +1pp core-CPI surprise, t-stat near −8, a 95% confidence interval of about [−9.3, −5.5], and an R-squared near 0.65 on event days — strong, by macro standards.
> - The **t-stat and confidence interval are the believe-it test**: a beta whose 95% CI straddles zero has an unestablished sign and is not tradeable, no matter how good the point estimate looks.
> - The **beta is regime-conditional**: split your sample (or add an interaction term) because a strong-jobs surprise had a *negative* S&P beta in 2022-23 and a *positive* one in a normal expansion. The one number to remember: beta ≈ slope, believe it only when |t| > 2.

## The morning a single regression would have paid for itself

On the morning of 10 August 2022, the US July CPI report printed cooler than expected — headline inflation came in flat month-over-month against a consensus of about +0.2%, and the year-over-year rate eased from 9.1% to 8.5%. In the ninety minutes after the release, the S&P 500 jumped more than 2% and the Nasdaq rose nearly 3%. Two-year Treasury yields fell, the dollar slid, and Bitcoin popped. A trader who had spent the previous week building a tiny spreadsheet — one column of past CPI surprises, one column of the S&P's reaction over the same window, and a single regression slope — would have known *in advance* the rough size of the move to expect per unit of surprise, and crucially which direction every asset would lean.

That trader would not have been forecasting inflation. Forecasting inflation is hard and mostly futile; the consensus already does it. What the regression gives you is something different and far more useful: a measured, standard-error-bounded estimate of *how much each asset moves per unit of surprise*, conditional on the regime you are in. That number is called the **beta to the surprise**, and producing it is a concrete, repeatable piece of empirical work — an *event study*. This post is the workshop where we build it.

![Pipeline from data release through surprise and aligned return to an OLS regression that outputs a beta with a confidence interval](/imgs/blogs/measuring-beta-to-data-surprises-an-event-study-in-python-1.png)

The companion post in this series, [correlate the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises), argues *why* you must use the surprise rather than the level. This post assumes you accept that and shows you *how to measure the resulting beta in code* — every step from raw release data to a regression you can read and trust. By the end you will be able to take any indicator, any asset, and any sample window, and produce a beta with a t-stat and a confidence interval. That is the unit of work that turns the rest of this series' qualitative correlations into numbers you can size a trade around.

## Foundations: what an event study actually is

Before any Python, build the idea from zero, because "event study" sounds more intimidating than it is. An **event study** is a research design that measures the effect of a discrete, dated event on an outcome by lining up the event dates and looking at the outcome in a tight window around each one. It was invented in academic finance to measure how, say, an earnings announcement or a stock split moves a share price. We are reusing the exact same design with the *event* being a scheduled macro data release (a CPI print, a jobs report) and the *outcome* being an asset's return over the window straddling that release.

The everyday analogy: suppose you want to know how much a particular chef's cooking improves a restaurant's nightly reviews. You would not compare the chef's restaurants to all restaurants on average — too many confounders. Instead you would find every night the chef personally cooked (the *events*), look at the review score for *just those nights* (the tight window), and compare it to what was expected. By isolating the events, you strip away the slow background drift (the restaurant's baseline reputation, the season, the neighborhood) and isolate the *effect of the event itself*. An event study does the same for data releases: by looking only at the short window around each release, you strip away the slow macro drift and isolate the market's reaction to the *new information* in that release.

The four ingredients of any event study are always the same:

1. **An event list with dates.** For us, the historical release calendar of an indicator — every CPI release day, every NFP release day.
2. **A measure of the "shock" at each event.** For us, the **surprise**: actual minus consensus, then standardized so different indicators are comparable.
3. **An event window over which to measure the outcome.** For us, the asset's return from just before the release to just after — could be a few minutes, the trading session, or close-to-close, depending on how fast the asset prices the news.
4. **A model linking shock to outcome.** For us, a linear regression of return on surprise. Its slope is the beta.

### Why a regression and not just an average

You could, in principle, skip the regression: take all the days CPI surprised hot and average the S&P's return, take all the days it surprised cold and average that, and compare. People do this, and it is not crazy. But it throws away most of the information. A regression uses the *magnitude* of each surprise, not just its sign. A +0.4pp surprise is twice the shock of a +0.2pp surprise, and the regression weights it accordingly; a sign-only average treats them identically. The regression also hands you, for free, the standard error of the slope — the basis for the t-stat and confidence interval that tell you whether to believe the number at all. So we regress.

The model is the simplest possible line:

```
return_i = alpha + beta * surprise_i + error_i
```

Here `i` indexes events (releases). `return_i` is the asset's event-window return for release `i`; `surprise_i` is the standardized surprise for that release; `beta` is the slope we want — the **beta to the surprise**; `alpha` is the intercept (the average drift on event days unrelated to the surprise, which should be near zero for a well-specified study); and `error_i` is everything else that moved the asset that day. Ordinary least squares (OLS) finds the `alpha` and `beta` that minimize the sum of squared errors. That `beta` is our deliverable.

For the deeper statistics behind what this slope *means* — the relationship between beta, the Pearson correlation r, and covariance — see [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta), and for the mechanics of OLS itself the regression posts under [math for quants](/blog/trading/math-for-quants/regression-ols-gls-regularized-math-for-quants). Here we focus on the applied pipeline.

### What the regression quietly assumes

OLS is the right default, but it carries assumptions you should know so you can tell when it is misleading you. Three matter for an event study.

First, OLS assumes the relationship is **linear** — that doubling the surprise doubles the reaction. For modest surprises this is a fine approximation, but markets are *convex* to large shocks: a four-sigma CPI surprise often moves prices more than four times a one-sigma surprise, because it forces a wholesale repricing of the policy path and can trigger forced de-leveraging. A linear beta will under-predict the move on the truly enormous prints. If your sample is dominated by a handful of giant surprises, consider a quadratic term or simply note that the beta is a *local* slope, accurate near the center of the surprise distribution and conservative in the tails.

Second, OLS assumes the errors are **homoskedastic** — that the unexplained scatter is the same size at every surprise. In event studies it almost never is: big-surprise days are also the noisiest days, so the error variance grows with the surprise. That is precisely why the code below specifies `cov_type="HC1"` — heteroskedasticity-consistent (robust) standard errors. Ordinary standard errors would be too small, your t-stats too big, and you would believe a beta more than you should. Robust SEs are not optional polish; they are the difference between an honest and a flattering t-stat.

Third, OLS assumes the events are **independent**. Scheduled releases a month apart are close enough to independent for our purposes, but if you widen the event window so that consecutive windows *overlap* (a five-day window on releases that are sometimes three days apart), you introduce autocorrelation that, again, shrinks your true standard errors. Keep windows shorter than the gap between events, or use Newey-West standard errors that account for the overlap.

### The expectation is a distribution, not a point

One more piece of foundation before the code. The "consensus" is a survey median, but the market's true expectation is a *distribution* of forecasts. When economists are tightly clustered — everyone says +0.3% — a +0.2pp surprise is a genuine shock and your regression will see a clean, large reaction. When forecasts are scattered — guesses span +0.1% to +0.5% — the same +0.2pp "surprise" relative to the median is less informative, because the market was already hedged across the range, and the reaction will be muted. This adds noise to your regression that no amount of clever modeling removes: it is a real feature of how the market was positioned. The practical consequence is that your cleanest betas come from indicators with tight forecast dispersion (CPI, NFP) and your muddiest from indicators forecasters disagree about. When a beta has a stubbornly wide confidence interval, forecast dispersion is often the culprit, not your code.

## Step 1: assemble the release data

Everything downstream depends on getting three aligned columns per indicator: the **release date**, the **actual** value, and the **consensus** value. In production you would pull these from a data vendor (Bloomberg's `ECO`, an economic-calendar API, or a maintained CSV from a provider like Trading Economics or Investing.com). For teaching, assume you have them in a small CSV with these columns — date, actual, consensus, all for the month-over-month core CPI:

```python
import pandas as pd

releases = pd.read_csv(
    "core_cpi_releases.csv",
    parse_dates=["release_date"],
)
releases = releases.sort_values("release_date").reset_index(drop=True)
print(releases.head())
    #   release_date  actual  consensus
    # 0   2021-01-13     0.10       0.20
    # 1   2021-02-10     0.10       0.20
    # 2   2021-03-10     0.10       0.20
    # 3   2021-04-13     0.30       0.20
    # 4   2021-05-12     0.90       0.30
```

The values above are month-over-month core CPI in percentage points; the May 2021 row (actual 0.9% vs consensus 0.3%) was the first of the great upside-surprise prints that kicked off the inflation regime. The key requirements at this stage are unglamorous but non-negotiable:

- **Use the consensus that was published *before* the release.** If you accidentally use a revised or back-filled consensus, you have leaked future information into your surprise and your beta will be inflated. This is the most common silent bug in event studies.
- **Use the *first-print* actual, not a later revision.** The market traded the number on the screen at release time. NFP in particular gets revised heavily; regressing on the revised figure measures a reaction that never happened.
- **Get the date *and the time* right.** US CPI and NFP release at 8:30 a.m. Eastern. If your event window is intraday, an hour's error in the timestamp ruins the alignment.

#### Worked example: turning a release into a surprise

Take the September 2022 core-CPI release: actual +0.6% month-over-month, consensus +0.3%. The raw surprise is

```
surprise = actual - consensus = 0.6 - 0.3 = +0.3 percentage points
```

To make this comparable across indicators with different scales (a 0.3pp CPI surprise is not the same "size" of shock as a 50k payrolls surprise), divide by the historical standard deviation of the surprise series. Suppose core-CPI surprises over the sample had a standard deviation of 0.15pp. Then the **standardized surprise** is `0.3 / 0.15 = +2.0` — a two-sigma upside surprise, a genuinely large shock. The intuition: this print didn't just beat expectations, it beat them by twice the typical month's miss, which is why \$2.5 trillion of S&P market value evaporated that session. Standardizing is what lets you put a CPI surprise and a jobs surprise on one axis and compare their betas.

## Step 2: compute the surprise and standardize it

In code the surprise and its standardized form are two lines:

```python
releases["surprise"] = releases["actual"] - releases["consensus"]

sd = releases["surprise"].std()      # z-score by its own historical std
releases["surprise_z"] = releases["surprise"] / sd

print(releases[["release_date", "surprise", "surprise_z"]].tail())
```

Three refinements separate a toy version from a defensible one:

**Standardize on a rolling, backward-looking window.** Using the full-sample standard deviation peeks at the future — at the start of the sample you would not have known the eventual `sd`. A more honest version uses an expanding or trailing standard deviation so the standardization at each date only uses data available up to that date:

```python
roll_sd = releases["surprise"].expanding(min_periods=12).std()
releases["surprise_z"] = releases["surprise"] / roll_sd
```

**Decide whether to demean.** If forecasters are systematically biased (consensus runs persistently low in a rising-inflation regime, so surprises are persistently positive), the surprise has a non-zero mean. Subtracting the rolling mean isolates the *unexpected* part more cleanly. Whether to do this is a judgment call; for inflation in 2021-22, consensus chronically under-predicted, so demeaning matters.

**Watch the units.** Keep the raw surprise in its natural unit (pp for CPI, thousands for NFP) *and* the standardized version. You will report the beta both ways: "−7% per +1pp" is intuitive for a CPI desk, while "−1.0% per one-sigma surprise" is what lets you compare CPI to NFP to ISM on one scale.

## Step 3: align the event-window return

Now the second column of the regression: the asset's return over the window around each release. The window choice is itself a modeling decision and depends on how fast the asset absorbs the news.

```python
prices = pd.read_csv(
    "spx_daily.csv", parse_dates=["date"], index_col="date"
).sort_index()

def event_return(release_date, window=("prev_close", "close")):
    #  close-to-close return spanning the release day
    rd = pd.Timestamp(release_date)
    prev = prices.loc[:rd - pd.Timedelta(days=1), "close"].iloc[-1]
    same = prices.loc[rd, "close"]
    return same / prev - 1.0

releases["ret"] = releases["release_date"].apply(event_return)
```

The function above takes a close-to-close return spanning the release day. For a liquid US index that prices an 8:30 a.m. release within minutes, the same-session move captures essentially all of the reaction; a close-to-close return is a clean, easy-to-build proxy. For slower-absorbing or globally-traded assets you might widen to a two-day window, and for an ultra-precise intraday study you would use the price 5 minutes before 8:30 versus 30 minutes after — but that requires tick data and careful timestamp hygiene. Start with close-to-close; it is robust and reproducible.

#### Worked example: the same release, two windows

For the 13 September 2022 hot CPI print, the S&P closed down 4.3% on the session — that is the close-to-close event return. But the actual *reaction* was front-loaded: futures gapped down roughly 2% in the first few minutes and the index was down about 3% within the hour. The remaining drift to −4.3% was partly continuation and partly unrelated afternoon selling. If your window is the full session you measure −4.3%; if it is the first 30 minutes you measure about −3%. The beta you estimate therefore depends on the window, and you must report which one you used. The intuition: a wider window catches more of the reaction but also more *noise* from unrelated flows, so it usually lowers your R-squared even as it captures the full move.

A subtle alignment trap: make sure each release date actually has a trading day. CPI sometimes releases on a day the bond market is closed but equities are open, or vice versa, and NFP always releases on a Friday. Defensive code drops events with missing price data rather than silently forward-filling, which would invent a return.

## Step 4: build the regression dataset and run OLS

With a surprise column and a return column, the regression is now a four-line block. This is the heart of the event study. The illustrative code below uses `statsmodels`, the standard Python library for this, with **heteroskedasticity-robust (HC) standard errors** — important because event-day volatility is uneven, and ordinary standard errors would understate the uncertainty on the beta.

```python
import statsmodels.api as sm

data = releases.dropna(subset=["surprise_z", "ret"]).copy()
X = sm.add_constant(data["surprise_z"])     # adds the alpha (intercept)
y = data["ret"] * 100                        # returns in percent

model = sm.OLS(y, X).fit(cov_type="HC1")     # robust standard errors
print(model.summary())
```

The `model.summary()` table is dense, but you only need four numbers from it, and you can pull them out directly:

```python
beta  = model.params["surprise_z"]           # the slope: % return per 1-sigma surprise
tstat = model.tvalues["surprise_z"]          # slope / standard error
ci_lo, ci_hi = model.conf_int().loc["surprise_z"]   # 95% confidence interval
r2    = model.rsquared                        # variance explained on event days

print(f"beta = {beta:.2f}%/sigma  t = {tstat:.1f}  "
      f"95% CI = [{ci_lo:.2f}, {ci_hi:.2f}]  R2 = {r2:.2f}")
```

That single print line is the deliverable of the whole exercise. Everything before it was data plumbing; everything after it is interpretation.

Because you will run this same study dozens of times — every indicator against every asset, in every regime — it pays to wrap the whole thing into one reusable function that takes a surprise series and a return series and hands back the four numbers in a tidy dictionary. That way the next study is a single call, and you can loop it across a whole grid of indicators and assets to build the cross-asset beta map at the end of this post:

```python
import statsmodels.api as sm

def event_study(surprise, ret, label=""):
    df = pd.DataFrame({"s": surprise, "r": ret}).dropna()
    X = sm.add_constant(df["s"])
    m = sm.OLS(df["r"] * 100, X).fit(cov_type="HC1")
    lo, hi = m.conf_int().loc["s"]
    return {
        "label":  label,
        "beta":   m.params["s"],
        "t":      m.tvalues["s"],
        "ci":     (lo, hi),
        "r2":     m.rsquared,
        "n":      int(m.nobs),
        "trust":  abs(m.tvalues["s"]) > 2 and lo * hi > 0,
    }

result = event_study(data["surprise_z"], data["ret"], label="SPX ~ CPI")
print(result)
```

The `trust` flag is the small but important touch: it encodes the believe-it test directly, returning `True` only when the absolute t-stat exceeds 2 *and* the confidence interval does not straddle zero (the product of its two endpoints is positive only when both share a sign). A study that comes back with `trust=False` is telling you, in one boolean, that the sign is not established and you should not size a trade around it. Building the discipline into the function means you never accidentally trade a pretty-looking but insignificant beta because you forgot to glance at the standard error.

![Matrix mapping the regression beta, t-stat, confidence interval, and R-squared each to the decision it informs](/imgs/blogs/measuring-beta-to-data-surprises-an-event-study-in-python-6.png)

The figure above is the cheat sheet for reading those four numbers, and the next four sections walk through each one. Notice the division of labor: the **slope** sizes the trade, the **t-stat and CI** tell you whether to believe the slope at all, and the **R-squared** sets your expectations about how much of the day's move the surprise explains. Beginners fixate on the slope and ignore the other three, which is exactly backwards — the slope is the easy part; knowing whether to trust it is the whole skill.

## Reading number one: the beta (slope)

The slope is the beta, and it is the number you actually trade around. Run the S&P core-CPI event study over the 2022-23 inflation-fear regime and you get something like the figure below: a tight, downward-sloping cloud of event dots with a steep fitted line through it.

![Scatter of CPI surprise versus S&P return with an OLS fit and shaded 95% confidence band, beta and t-stat annotated](/imgs/blogs/measuring-beta-to-data-surprises-an-event-study-in-python-3.png)

Reading the annotation box: the slope is about **−7.4% per +1 percentage point** of core-CPI surprise, equivalently roughly −0.74% per +0.1pp, the headline number you will see quoted for that regime. Stated in standardized terms — per one-sigma surprise — it would be the same line read off the z-scored x-axis. The intercept is near zero, as it should be: on event days where the surprise was zero, the S&P had no systematic drift from the surprise channel.

There is a discipline to *reading* this chart that beginners skip. Look first at the **dots**, not the line. Are they a genuine downward cloud, or a shapeless blob with one extreme point dragging a line through it? A real beta shows up as a coherent tilt across the bulk of the events; a fragile one is a flat scatter with a single outlier — usually a colliding event — defining the slope. Next, look at the **width of the confidence band** (the shaded region around the line). It pinches in the middle, where most of your data sits, and flares at the edges, where you have few large-surprise observations — which is exactly why extrapolating a beta to an enormous, never-before-seen surprise is dangerous: the band is widest precisely where you would most want certainty. Finally, check that the **line passes through the body of the dots** rather than floating above or below them; if it does not, your intercept is doing work it should not, often a sign of an event-day drift you have not accounted for. Only once the picture passes this eyeball test should you trust the number in the annotation box. A regression you have not plotted is a regression you have not checked.

#### Worked example: computing a beta by hand from four releases

You don't need a library to see where the slope comes from. Take four (surprise, return) pairs in raw units — core-CPI surprise in pp, S&P session return in %:

```
(+0.3, -4.3)   (-0.1, +2.1)   (+0.1, -1.0)   (-0.2, +2.6)
```

The OLS slope is covariance divided by variance. Mean surprise = (0.3 − 0.1 + 0.1 − 0.2)/4 = +0.025; mean return = (−4.3 + 2.1 − 1.0 + 2.6)/4 = −0.15. The covariance numerator (sum of `(x − x̄)(y − ȳ)`) works out to about −2.74, and the surprise variance numerator (sum of `(x − x̄)²`) to about 0.157. So the slope is

```
beta = -2.74 / 0.157 = -17.4  (this tiny 4-point sample over-steepens)
```

A real sample of 30-plus releases pulls the estimate to the ≈ −7%/pp range; four points over-fit wildly, which is itself the lesson. The intuition: the slope is just the average tilt of return against surprise, but with too few events the tilt is dominated by whichever single release was the biggest outlier — here the −4.3% September print — which is exactly why you need a t-stat and confidence interval to know how much to trust the slope. **More events, more trust.**

#### Worked example: the dollar move on a position from surprise × beta

Now use the beta to size a real trade. Suppose you hold a \$5,000,000 long S&P position into a CPI release. The published consensus for core CPI is +0.3% and you, like everyone, are positioned for it. The print comes in at +0.5% — a +0.2pp upside surprise. Using the regime beta of about −0.74% per +0.1pp:

```
expected move = beta_per_0.1pp * (surprise / 0.1pp)
              = -0.74% * (0.2 / 0.1)
              = -0.74% * 2 = -1.48%
```

On a \$5,000,000 position that is an expected loss of `0.0148 * \$5,000,000 = \$74,000` in the session, just from the surprise channel. If you had instead been positioned the *other* way — short \$5,000,000 into a hot print — the same beta implies a +\$74,000 gain. That is what a beta buys you: a dollar estimate of your exposure to the next release *before* it happens, so you can decide whether to cut, hedge, or hold the position through the event. The intuition: the beta converts an abstract "inflation is bad for stocks" into a concrete \$74,000 of P&L per two-tenths of surprise on this specific book.

## Reading number two: the t-stat (do you believe it?)

The slope alone is seductive and dangerous, because *every* regression produces a slope, even one fit to pure noise. The **t-statistic** is the antidote. It is the slope divided by its own standard error: how many standard errors the estimated beta sits away from zero.

```
t = beta / standard_error(beta)
```

The **standard error** deserves a plain-language definition because everything rests on it. Imagine you could re-run history many times — draw a fresh set of CPI releases from the same underlying process and re-estimate the beta each time. You would get a slightly different slope every run, and the standard error is the spread of those hypothetical slopes: how much your one estimated beta would jiggle if the dice were re-rolled. A small standard error means the data pin the slope down tightly; a large one means the slope is loosely determined and could easily have come out quite different. You only get one history, so you never see this spread directly — the regression *estimates* it from the scatter of the residuals and the spread of your surprises. That estimate is why robust standard errors matter so much: get the standard error wrong and every downstream judgment (t-stat, confidence interval, position size) is built on sand.

A `|t|` above roughly 2 means the beta is about two standard errors from zero, which corresponds to about 95% confidence that the *true* beta is not zero — that the relationship is real and not a fluke of the sample. Below 2, you cannot reject "the asset doesn't actually respond to this surprise." In the S&P CPI study above, the t-stat is around **−7.9** — wildly significant, because the inflation-regime reaction was strong and consistent across many releases. That is unusually clean for macro work; a t-stat of −7.9 means the chance the true beta is zero is vanishingly small. Be a little suspicious of t-stats that enormous, though: in genuine macro data they usually signal either a very strong regime (which 2022-23 was) or a hidden lookahead bias inflating the number, so a −7.9 should prompt you to double-check your data hygiene rather than simply celebrate.

#### Worked example: reading the t-stat and confidence interval together

Suppose your event study returns a gold beta to CPI surprises of **−0.80% per +0.1pp**, with a standard error of **0.20%**. Then:

```
t = -0.80 / 0.20 = -4.0
```

A t of −4.0 is comfortably past the |2| threshold, so the negative gold beta is established. The matching 95% confidence interval is the beta plus-or-minus roughly two standard errors:

```
95% CI = -0.80 ± 1.96 * 0.20 = [-1.19, -0.41]
```

The entire interval is below zero, so you can say with about 95% confidence that gold *falls* on a hot CPI print — consistent with the real-yields-up story (a hot CPI lifts real yields, which is bad for gold; see [inflation and gold, the real-yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story)). The intuition: the t-stat and CI are two views of the same thing — a |t| past 2 is exactly the condition under which the 95% CI clears zero — and together they convert "gold seems to drop on hot CPI" into a quantified, bounded claim you can stake \$ on.

Now contrast a *bad* case. Suppose a small-cap energy basket gives a CPI beta of +0.6% with a standard error of 0.5%. Then `t = +0.6 / 0.5 = 1.2`, and the 95% CI is `[-0.38, +1.58]` — it spans zero. The point estimate is positive, but you cannot rule out that the true beta is zero or even negative. **You do not trade that beta.** The point estimate is a mirage until the interval confirms its sign.

## Reading number three: the confidence interval (the believe-it test, visualized)

The confidence interval deserves its own figure because it is the single most underused diagnostic in retail macro work. People quote a point estimate ("Bitcoin drops 1.6% on hot CPI") as if it were exact. It is not; it is the center of a range. The figure below plots each CPI beta with its 95% interval as an error bar.

![Beta point estimates for five assets with 95% confidence interval error bars, the dollar's interval nearly touching zero](/imgs/blogs/measuring-beta-to-data-surprises-an-event-study-in-python-7.png)

Read the figure left to right. The S&P (−0.70 ± 0.31), Nasdaq (−1.00 ± 0.43), and gold (−0.80 ± 0.39) all have intervals comfortably below zero: established negative betas, tradeable. The dollar (+0.35 ± 0.29) is the cautionary case — its interval reaches from about +0.06 to +0.64, so it clears zero but *barely*; the positive dollar beta is real but estimated with much less precision, and in a slightly different sample it could straddle zero. Bitcoin (−1.60 ± 0.88) has the biggest point estimate *and* by far the widest interval, because crypto's session noise is enormous; the sign is established but the magnitude is mushy, anywhere from about −0.7% to −2.5% per +0.1pp.

The rule the figure teaches: **a beta whose 95% CI crosses the zero line has an unestablished sign and is not tradeable.** The width of the interval is your honesty check — a narrow interval (S&P) means you can size confidently; a wide one (Bitcoin) means hedge smaller or demand a bigger edge. This is the discipline that separates an event study you can trade from a number that merely sounds precise. For the related danger of a relationship that *looks* strong but is statistical noise, see [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data).

## Reading number four: R-squared (how much it explains)

The final number, **R-squared**, is the fraction of the variance in event-window returns that the surprise explains. In the S&P CPI study it was about **0.65** — meaning roughly two-thirds of the variation in S&P returns *on CPI days* is accounted for by the size and sign of the CPI surprise. That is extraordinarily high for financial data, and it tells you that in the inflation regime, CPI day *was* an inflation-surprise referendum and little else.

But beware two traps. First, that 0.65 is the R-squared *on event days only* — a tiny, hand-picked slice of all trading days. The surprise explains almost none of the asset's *unconditional* day-to-day variance, because most days have no major release. R-squared in an event study is a conditional statement, not a claim that CPI drives the market in general. Second, a *low* R-squared does not invalidate a beta. The beta can be highly significant (huge t-stat) even with a modest R-squared if the sample is large; the surprise reliably moves the asset in one direction, but a lot of *other* stuff also moves it that day. R-squared measures explanatory share, not whether the relationship is real — that is the t-stat's job. Don't conflate them.

#### Worked example: a significant beta with a low R-squared

Imagine an NFP study on the 10-year Treasury yield with 60 events: beta = +4 bp per +100k surprise, standard error = 1 bp, so `t = +4.0` (highly significant), but R-squared = 0.30. How can a rock-solid beta explain only 30% of the variance? Because on a typical NFP Friday, the surprise reliably pushes yields up by its beta, but *also* on that Friday the market digests wage data, the unemployment rate, revisions to prior months, and Fed-speak — all of which move yields and none of which are in the model. The beta is real and tradeable; the other 70% of variance is the rest of the report plus noise. The intuition: significance asks "is the slope real?" and R-squared asks "how much does it explain?" — a beta can be a confident *yes* on the first while scoring a humble 30% on the second, and that is a perfectly good, tradeable result.

## The full multi-asset picture

Run the same event study across every asset for a single indicator and you get a cross-asset beta map — the standardized response of each asset to one hot CPI surprise. This is just the per-asset slope from seven separate regressions, plotted together.

![Horizontal bar chart of seven assets' standardized betas to a hot CPI surprise, from US bonds most negative to the dollar most positive](/imgs/blogs/measuring-beta-to-data-surprises-an-event-study-in-python-5.png)

The ordering is the entire transmission story in one picture. Long-duration assets — the 10-year Treasury (−0.75) and the Nasdaq (−0.65) — have the most negative betas, because a hot print most directly reprices the rate path that discounts their distant cash flows. The broad S&P (−0.55), Bitcoin (−0.55), and EM equity (−0.45) cluster as high-beta risk assets. Gold (−0.10) is nearly flat, because its hot-CPI response runs through *real* yields and breakevens that partly offset on a single print. And the dollar (+0.45) is the lone positive: a hot print means a more hawkish Fed, higher US yields, and a stronger dollar. Every one of these is a separate OLS slope, and you can reproduce the whole bar with seven runs of the four-line regression block above. For the full driver-by-asset grid, see [the macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix).

The full beta numbers for the % assets, with native units, line up like this — the same numbers you would get from seven event studies on a +0.1pp core-CPI upside surprise in the 2022-23 regime:

![Two-panel bar chart of CPI surprise betas, percentage-return assets on the left and basis-point yields on the right](/imgs/blogs/measuring-beta-to-data-surprises-an-event-study-in-python-2.png)

Stocks, gold, and crypto fall; yields and the dollar rise. The Nasdaq (−1.0%) falls more than the S&P (−0.7%) because it is more rate-sensitive; Bitcoin (−1.6%) moves most because it is the highest-beta risk asset; the 2-year yield (+9bp) repriced more than the 10-year (+7bp) because the front end is where the Fed-path bet lives. None of these are guesses — each is the slope of a regression you now know how to run.

## The regime split: the most important step

Here is the step that separates a naive event study from a useful one. **A single beta estimated over your whole sample is usually a lie**, because the beta is regime-conditional. The cleanest illustration is the jobs report. In the 2022-23 "good-news-is-bad" regime, a strong NFP print meant a more hawkish Fed and *hurt* stocks: the S&P beta to an upside payrolls surprise was *negative*. In a normal expansion, a strong jobs print means good growth and *helps* stocks: the beta is *positive*. Pool both regimes into one regression and you get a mushy near-zero beta that describes neither.

![Grouped bar chart of NFP surprise betas by regime showing the S&P beta flips from negative in 2022-23 to positive in a normal expansion](/imgs/blogs/measuring-beta-to-data-surprises-an-event-study-in-python-4.png)

Look at the S&P bars: −0.5% per +100k surprise in the good-news-is-bad regime, +0.35% in a normal expansion. Same indicator, same asset, *opposite sign*. A pooled regression averages these to roughly zero and concludes, absurdly, that "jobs don't move stocks." The fix is to estimate the beta *conditionally*, and there are two standard ways to do it.

### Approach A: split the sample

The simplest fix is to cut the data into regime sub-samples and run a separate regression in each. You need a regime label per event — here, a flag for whether the market was in "good-news-is-bad" mode (high inflation, hawkish Fed) at the time:

```python
hawkish = data[data["regime"] == "good_news_is_bad"]
normal  = data[data["regime"] == "normal_expansion"]

for name, sub in [("hawkish", hawkish), ("normal", normal)]:
    Xs = sm.add_constant(sub["surprise_z"])
    m = sm.OLS(sub["ret"] * 100, Xs).fit(cov_type="HC1")
    b = m.params["surprise_z"]
    t = m.tvalues["surprise_z"]
    print(f"{name:8s}: beta = {b:+.2f}%/sigma   t = {t:+.1f}")
    #  hawkish : beta = -0.50%/sigma   t = -3.1
    #  normal  : beta = +0.35%/sigma   t = +2.4
```

Two clean, opposite, individually-significant betas. This is the honest answer. The cost is that each sub-sample is smaller, so the standard errors widen and the t-stats fall — you need enough events in each regime for the split to be meaningful.

### Approach B: the interaction term

The more elegant approach keeps the whole sample but adds an **interaction term**: a variable equal to the surprise multiplied by a regime indicator. The model becomes

```
return = alpha + beta1 * surprise + beta2 * (surprise * hawkish) + error
```

Now `beta1` is the surprise beta in the *baseline* regime (normal expansion), and `beta1 + beta2` is the beta in the hawkish regime. The coefficient `beta2` directly estimates *how much the beta changes between regimes*, with its own t-stat — so you can test whether the regime difference is itself statistically significant, not just eyeball it.

```python
data["hawkish_flag"] = (data["regime"] == "good_news_is_bad").astype(int)
data["surprise_x_hawkish"] = data["surprise_z"] * data["hawkish_flag"]

X = sm.add_constant(data[["surprise_z", "surprise_x_hawkish"]])
m = sm.OLS(data["ret"] * 100, X).fit(cov_type="HC1")
print(m.params)
    #  const                 0.02
    #  surprise_z            0.35    <- beta in normal regime
    #  surprise_x_hawkish   -0.85    <- the regime *shift* in the beta
```

#### Worked example: reading the interaction term

From the output above, the normal-regime S&P beta is `beta1 = +0.35%` per one-sigma surprise. The interaction coefficient is `beta2 = -0.85%`. So the hawkish-regime beta is

```
beta_hawkish = beta1 + beta2 = +0.35 + (-0.85) = -0.50% per sigma
```

The beta flips from +0.35% to −0.50% — a swing of 0.85 percentage points of return per one-sigma surprise, which is exactly `beta2`. If that `beta2` carries a t-stat past |2| (here it does, around −3), the regime difference is statistically established: you have *proven* the sign flip rather than just asserted it. To put a \$ on it, on a \$3,000,000 S&P position a one-sigma upside jobs surprise implies about `+0.35% * \$3,000,000 = +\$10,500` in a normal expansion but about `-0.50% * \$3,000,000 = -\$15,000` in the hawkish regime — a \$25,500 swing in your event P&L from the regime alone, on the identical print. The intuition: the interaction term is how you let one regression hold two betas at once, and its coefficient is the precise, testable measure of the regime flip that a single pooled beta would have erased.

For the deeper economics of why the jobs-report sign flips, see [NFP and asset prices](/blog/trading/macro-correlations/nfp-and-asset-prices-the-king-of-data-correlation) and the mechanism in [why news moves markets](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework).

## Practical pitfalls that quietly ruin a beta

The four-line regression is easy. Getting a *trustworthy* beta out of it is where the work hides. These are the failure modes that turn a clean-looking number into a trap, roughly in order of how often they bite.

### Lookahead bias: the silent inflater

The deadliest bug in any event study is using information you would not have had at the time. There are three places it sneaks in. The **consensus** must be the figure published before the release, not a value back-filled or revised by the data vendor after the fact; some providers overwrite the survey with a "smart" estimate that already leaned toward the actual, which mechanically tightens your surprise and inflates your beta. The **actual** must be the first print, because revisions arrive weeks later and the market never traded them. And the **standardization** must use a backward-looking standard deviation, as shown in Step 2, because the full-sample `sd` embeds the future. Any one of these makes your beta look stronger and your t-stat bigger than reality, and the error is invisible — the regression runs fine and prints a beautiful number. The only defense is paranoid data hygiene: every input to a row must have been knowable on that row's release date.

#### Worked example: how lookahead inflates a t-stat

Suppose the honest beta is −7% per +1pp with a standard error of 0.9%, giving `t = -7.8`. Now say your vendor's "consensus" was quietly nudged 0.05pp toward each actual, shrinking every surprise by about 0.05pp. Your x-variable is now compressed toward zero, so for the same returns the *slope* steepens — say to −8.5% — and because the contaminated surprises align suspiciously well with returns, the standard error falls to 0.7%, pushing the t-stat to about −12. You would report a far more "significant" beta than truly exists, size up accordingly, and be blindsided when the real-time relationship turns out weaker. The takeaway: lookahead bias does not just shift the beta, it shrinks the standard error too, so it attacks exactly the believe-it test you rely on — which is why a t-stat that looks too good to be true usually is.

### Too few events

Macro releases are monthly, so a few years of data is only a few dozen events. Split that by regime and each sub-sample might hold 15-20 points. With samples that small, one outlier release dominates the slope, the standard errors are wide, and the beta wobbles if you add or drop a single event. There is no magic minimum, but be deeply skeptical of any regime beta estimated on fewer than ~20 events, and always look at the confidence interval rather than the point estimate — a small sample announces its own unreliability through a wide CI. When you genuinely cannot get more events, pool related indicators (regress on a *combined* inflation surprise built from CPI and PPI) to borrow strength, or report the sign-only result and admit the magnitude is uncertain.

### Window mismatch across assets

A single event window does not fit every asset. US equities price an 8:30 a.m. release within minutes, so a same-session close-to-close window captures the reaction. But a globally-traded asset like gold or Bitcoin trades around the clock and may continue reacting into the next Asian session, so a one-day window clips part of the move; a Treasury future that gaps at 8:30 and then mean-reverts into the afternoon can show a *smaller* close-to-close move than the true initial reaction. If you run one window across a multi-asset comparison, you are measuring slightly different things for each asset. The honest fix is to choose the window per asset's absorption speed, or — better — use a tight intraday window (pre-release price versus 30-to-60 minutes after) for all of them if you have the tick data, which standardizes the comparison.

### The wrong event is in your window

Macro releases cluster. CPI sometimes lands the same week as a Fed meeting; NFP shares its Friday with the unemployment rate and wage data baked into the same report. If a second, unmodeled event sits inside your window, its move contaminates your beta. The classic case is an NFP study where the headline payrolls surprise is positive but the *wage* number simultaneously surprised low — the market may rally on the dovish wages even as your model attributes the move to the strong headline, corrupting the slope. Defenses: keep the window tight enough to exclude separate scheduled events, drop events that collide with an FOMC decision, and for composite reports consider regressing on the *component that actually drives the reaction* (average hourly earnings, not headline payrolls, in some regimes).

### Survivorship and selection in the regime label

When you split by regime you must label each event by what was *known at the time*, not by hindsight. It is tempting to define "hawkish regime" using the full inflation path you can now see, but at the moment of an early-2022 release, a trader did not yet know inflation would peak at 9%. A regime label built from future data is just another form of lookahead. Define the regime from a real-time, backward-looking signal — say, trailing core inflation above a threshold and the Fed in a hiking cycle — so the split you make in the regression is one you could actually have made in real time.

## Common misconceptions

**"A bigger beta is a better trade."** No — a bigger beta with a wide confidence interval can be worse than a smaller, tighter one. Bitcoin's −1.6% CPI beta is larger than the S&P's −0.7%, but its 95% interval (±0.88) is nearly three times as wide. Per dollar of conviction, the S&P beta is more reliable. The tradeable quantity is the beta *relative to its standard error* — which is just the t-stat again. Size by confidence, not by raw magnitude.

**"A high R-squared means the beta is significant."** These are different questions. R-squared measures how much variance the surprise explains; significance (t-stat) measures whether the slope is distinguishable from zero. A large sample can deliver a rock-solid, highly significant beta with a modest R-squared (the surprise reliably moves the asset, but lots of other things move it too), and a tiny sample can show a high R-squared by pure overfitting. Read both, conflate neither.

**"One beta describes the relationship."** The single most expensive error in this whole post. The beta is regime-conditional; the NFP-to-S&P beta literally flips sign across regimes. A full-sample beta that averages a +0.35% and a −0.50% regime into roughly zero is not a summary, it is a destruction of information. Always check whether your beta is stable across sub-samples; if it isn't, report it by regime. See [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant).

**"I should use the level of the indicator as my x-variable."** No — regress on the *surprise*, not the level. The level is priced; only the surprise is new information. Regressing returns on the level of CPI produces a weak, near-meaningless slope and is the classic beginner mistake the companion post [correlate the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) exists to correct.

**"My in-sample beta is what I'll get next month."** Beware. The beta you estimate is an in-sample average; the regime can shift the moment after your sample ends, and 2022's stock-bond and good-news-is-bad flips happened fast. A beta is a description of a past regime, useful only insofar as the current regime resembles it. Before you trade a beta, confirm you are still in the regime that produced it — which is the whole subject of [backtesting a correlation without fooling yourself](/blog/trading/macro-correlations/backtesting-a-correlation-without-fooling-yourself).

## How it shows up in real markets

**13 September 2022 — the textbook event.** August core CPI printed +0.6% month-over-month against +0.3% consensus, a +0.3pp (≈ two-sigma) upside surprise. The S&P fell 4.3%, the Nasdaq more than 5%, the 2-year yield jumped, the dollar surged, gold fell, Bitcoin dropped hard. Plug the +0.3pp surprise into the regime S&P beta of −0.74% per +0.1pp and you predict roughly `−0.74% × 3 = −2.2%` from the surprise channel; the realized −4.3% was that plus continuation and positioning unwinds. The *direction and rough magnitude* fell straight out of the beta — which is exactly what an event study is for.

**10 November 2022 — the cool print.** Two months later, October core CPI came in at +0.3% versus +0.5% expected — a −0.2pp *downside* surprise. The same beta predicts `−0.74% × (−2) = +1.5%` from the surprise channel; the S&P actually rose 5.5% in one of its best days of the year, again amplified by short-covering. The beta got the sign and ballpark right; the overshoot is the market's convexity to *positive* surprises in a fearful regime, a known feature the linear model doesn't capture.

**2019 versus 2022 jobs reports — the regime flip in the wild.** A blowout payrolls number in mid-2019 (a normal expansion) was greeted by a rising S&P: good growth, no Fed-tightening fear. The identical-looking blowout in late 2022 (the hawkish regime) was met with a falling S&P: strong jobs meant more hikes. The same indicator, the same kind of surprise, opposite market reactions — precisely the sign flip the interaction-term regression is built to capture, and precisely the reason a pooled beta would have told you "jobs don't matter."

**The 2022 stock-bond break as a meta-lesson.** Underlying all of this is that 2022 reset the market's entire reaction function: inflation, not growth, became the dominant risk, which is what made good news bad and flipped the stock-bond correlation positive at the same time. Any beta estimated before 2022 was estimating a *different regime's* reaction function. See [when correlations break, the 2022 stock-bond flip](/blog/trading/macro-correlations/when-correlations-break-the-2022-stock-bond-flip).

**2024 — the beta fades as the regime cools.** By 2024, with inflation drifting back toward target and the Fed near the end of its hiking cycle, the CPI-to-S&P beta visibly shrank. Prints that would have triggered a 2-3% session move in 2022 produced fractional reactions, and the R-squared on CPI days fell as the market's attention rotated back toward growth and earnings. This is the regime check playing out in real time: a desk that had hard-coded the 2022 beta would have over-hedged into every 2024 release, paying away premium for an exposure that had largely dissipated. The lesson is operational — re-estimate your betas on a rolling window so they decay with the regime that produced them, rather than freezing a number from the peak-fear era and treating it as permanent.

**The dollar's quieter, steadier beta.** Not every beta is a drama. The dollar's positive beta to a hot CPI surprise was smaller than the S&P's negative one and far less spectacular session-to-session, but it was *steadier* across the cycle, because the mechanism — hot inflation, more hawkish Fed, higher US yields, stronger dollar — held up even as the equity reaction function shifted. This is a useful reminder that the most *tradeable* beta is not always the biggest one; a modest beta with a stable sign and a tight confidence interval across regimes can be worth more to a systematic strategy than a huge beta that flips. See [the dollar, cross-asset gravity](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation) for why the dollar's macro linkages are unusually durable.

## How to read it and use it: the event-study playbook

Bringing the whole pipeline together, here is the working checklist for producing and using a data-surprise beta you can actually trade.

**The signal — how to build it.** (1) Assemble clean release dates, first-print actuals, and *pre-release* consensus. (2) Compute the surprise (actual − consensus) and standardize it with a backward-looking standard deviation. (3) Align each release to the asset's event-window return — start with close-to-close. (4) Regress return on the standardized surprise with OLS and robust (HC) standard errors. (5) Pull out the beta, t-stat, 95% CI, and R-squared.

**The believe-it test — before you size anything.** Demand `|t| > 2` and a 95% CI that clears zero. A point estimate without these is a number, not a signal. Prefer betas with tight intervals; size inversely to interval width. A wide CI (Bitcoin) means trade smaller or wait for a bigger edge.

**The regime check — the step everyone skips.** Never trust a single full-sample beta. Split the sample by regime (Approach A) or add an interaction term (Approach B) and confirm the beta's sign and size in the regime you are *currently* in. If the beta flips across regimes — as the NFP-to-S&P beta does — only the current-regime estimate is tradeable. Before each release, ask: am I still in the regime that produced this beta?

**What invalidates it.** A regime shift (the reaction function changed — 2022 is the cautionary tale); a sample too small for the standard errors to settle; a consensus or actual contaminated by revisions (lookahead bias); a window so wide it drowns the reaction in unrelated noise. Any of these and the beta is unreliable.

**Where it fits.** This beta is the atomic unit behind every quantitative claim in this series. Stack the betas across drivers and assets and you get [the macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix); roll the regression through time and you get the rolling-correlation pictures in [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters); automate the whole thing across indicators and you have the makings of a [macro-asset correlation dashboard](/blog/trading/macro-correlations/building-a-macro-asset-correlation-dashboard-in-python). The single four-line OLS block is the engine under all of it.

The deepest takeaway is a discipline, not a number. A correlation you can only describe ("inflation is bad for stocks") is a slogan. A correlation you can *measure* — beta −7% per +1pp, t-stat −8, 95% CI [−9.3, −5.5], R-squared 0.65 on event days, and it flips sign in a growth-led regime — is a tool. The distance between the two is one small Python script, and you have just written it.

## Further reading and cross-links

- [Correlate the surprise, not the level: betas to macro data surprises](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) — *why* the surprise, not the level, is the right x-variable; the conceptual companion to this how-to.
- [What correlation actually measures: Pearson, Spearman, beta](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) — the statistics behind the slope: covariance, r, and beta.
- [Building a macro-asset correlation dashboard in Python](/blog/trading/macro-correlations/building-a-macro-asset-correlation-dashboard-in-python) — automating these regressions across many indicators and assets.
- [Backtesting a correlation without fooling yourself](/blog/trading/macro-correlations/backtesting-a-correlation-without-fooling-yourself) — lookahead bias, regime drift, and out-of-sample honesty for a measured beta.
- [CPI and asset prices: the master inflation correlation](/blog/trading/macro-correlations/cpi-and-asset-prices-the-master-inflation-correlation) — the specific CPI relationships this post regresses.
- [The macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix) — the full grid of driver-to-asset betas this method produces.
- [Why news moves markets: the surprise framework](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework) — the intraday-reaction mechanism behind the betas.
- Ordinary least squares and regression mechanics under [math for quants](/blog/trading/math-for-quants/regression-ols-gls-regularized-math-for-quants).
