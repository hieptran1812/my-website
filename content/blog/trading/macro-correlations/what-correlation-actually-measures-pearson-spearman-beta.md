---
title: "What Correlation Actually Measures: Pearson, Spearman, and Beta for Markets"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A from-zero primer on the numbers that run this whole series: covariance, the correlation coefficient r, R-squared, Pearson versus Spearman, and the crucial gap between correlation (strength) and beta (slope in real units)."
tags: ["macro", "correlation", "pearson", "spearman", "beta", "r-squared", "regression", "scatter-plot", "statistics", "quant", "trading"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Correlation (r) measures *how tightly* two things move together on a fixed −1-to-+1 ruler; beta measures *how far* one moves per unit of the other in real units; R-squared measures *how much* of the variance is explained — and confusing the three is the single most common mistake in reading macro data.
>
> - **Correlation is unitless and bounded to −1..+1.** It answers strength and sign only. An r of 0.6 is "moderately positive", not "moves 60% as much".
> - **Beta is the slope of the best-fit line, in real units** — bp of yield per 0.1pp of CPI surprise, dollars of gold per 1pp of real yield. Correlation tells you *whether* to care; beta tells you *how much* to size.
> - **R-squared = r squared = the share of variance explained.** r = 0.6 sounds strong but explains only 36% of the moves; the other 64% is everything else.
> - **The one number to remember:** a full-sample correlation can read ≈ 0 even when the relationship is rock-solid — if the sign flipped halfway through. Gold versus the real yield is r = −0.96 in 2007–2021 and +0.80 in 2022–2025; pooled, it is r = −0.01. The number lies; the regime is real.

In the autumn of 2022, a hedge-fund risk manager pulled up a chart she had trusted for two decades. It showed the rolling correlation between US stocks and long-dated Treasury bonds — the relationship that the entire "60/40" portfolio (60% stocks, 40% bonds) is built on. For her whole career, that line had sat comfortably below zero: when stocks fell, bonds rose, and the bonds cushioned the blow. That negative correlation *was* the diversification. It was the free lunch.

That autumn the line had crossed above zero and kept climbing. Stocks were down roughly 25% for the year. Bonds, which were supposed to be the cushion, were down too — the worst year for US Treasuries in modern history. The two halves of the "diversified" portfolio had stopped offsetting each other and started falling together. The correlation she had relied on had not weakened; it had *changed sign*. And the number she had been quoting for years — a tidy, full-history average — had been quietly hiding the fact that this relationship is not a constant at all. It is a regime.

This series is about exactly that: the measurable statistical relationships between macro indicators (inflation, jobs, yields, the dollar, credit, liquidity) and asset prices (stocks, bonds, gold, crypto, commodities), and the honest truth that **every one of those relationships has a sign, a strength, a lead/lag, and a tendency to flip**. But before we can argue about *which* correlations matter and *when* they break, we have to agree on what a correlation even *is* — and on the three different numbers people sloppily call "correlation" when they mean three different things. This post is the measurement primer. Get these three numbers straight and the rest of the series is a victory lap.

![Matrix of three questions one scatter answers correlation beta and R squared](/imgs/blogs/what-correlation-actually-measures-pearson-spearman-beta-1.png)

## Foundations: covariance, correlation, and the scatter plot

Let's build from the ground floor, assuming nothing. The most important habit in this entire field — more important than any formula — is to **plot the data before you compute a single number**. So let's start with the picture.

### The scatter plot is the master tool

Suppose you suspect that two things move together. Maybe it is the 10-year real yield (a measure of how expensive borrowing is, after stripping out inflation) and the price of gold. Maybe it is a CPI surprise and the stock market. Whatever the pair, the first thing to do is draw a **scatter plot**: put one variable on the horizontal axis (call it X), the other on the vertical axis (call it Y), and drop one dot for each observation — each year, each month, each data release.

Each dot is one moment in history where you observed both X and Y at once. The cloud of dots they form tells you almost everything before you reach for a calculator:

- **Sign.** Does the cloud tilt *up* from left to right (high X tends to come with high Y — positive) or *down* (high X tends to come with low Y — negative)?
- **Strength.** Is the cloud a *tight, narrow line* (a strong relationship — knowing X tells you a lot about Y) or a *fat, fuzzy blob* (a weak one — knowing X barely narrows down Y)?
- **Slope.** *How steep* is the tilt? A steep cloud means a small change in X comes with a big change in Y.
- **Outliers.** Is there *one lonely dot* way off in a corner, far from the rest? A single extreme point can drag the whole story around.

![Anatomy of a scatter plot showing sign strength slope and outliers](/imgs/blogs/what-correlation-actually-measures-pearson-spearman-beta-4.png)

Hold onto these four readings. Every number we are about to define is just a way of *summarizing* one of them into a single value. Correlation summarizes sign-and-strength. Beta summarizes slope. R-squared summarizes how much of the cloud's vertical spread the tilt accounts for. The numbers exist to compress the picture — but the picture comes first, and when a number surprises you, the picture is where you go to find out why.

### Covariance: do they move together?

The raw ingredient behind correlation is **covariance**. The idea is intuitive once you slow down.

For any single variable, we measure how spread out it is with the **variance**: take each value, subtract the average, square that gap (so positives and negatives don't cancel), and average those squared gaps. The square root of the variance is the **standard deviation** — the typical distance of a value from its own average, in the variable's own units.

Covariance extends this to *two* variables at once. For each observation, ask: was X *above or below* its average, and was Y *above or below* its average, at the same time? Multiply those two gaps together:

```
For each observation i:
    gapX = X_i  - average(X)
    gapY = Y_i  - average(Y)
    product_i = gapX * gapY

Covariance = average of all the product_i
```

Walk through the sign of that product. If X was above its average *and* Y was above its average, both gaps are positive and the product is positive. If both were below average, both gaps are negative and the product is *still* positive (negative times negative). But if X was above average while Y was below, one gap is positive and one is negative, so the product is negative.

So a positive covariance means: the observations where X is high tend to be the same observations where Y is high (and X-low lines up with Y-low). The two variables *deviate from their averages in the same direction*. A negative covariance means they deviate in *opposite* directions — high X tends to come with low Y. A covariance near zero means there is no consistent pattern: sometimes they move together, sometimes opposite, and it washes out.

Covariance is the right idea. But it has a fatal flaw for comparison: **it is not unitless, and it has no fixed scale.** If X is a yield measured in percent and Y is gold measured in dollars, the covariance comes out in "percent times dollars" — a unit nobody can interpret. Worse, if you measured gold in cents instead of dollars, the covariance would be 100× bigger, even though the *relationship* did not change at all. You cannot look at a covariance of 7,000 and say whether that is a strong relationship or a weak one. You need a scale.

#### Worked example: variance, standard deviation, and why we square

Before we fix covariance, let's make sure the "spread" ingredient is concrete, because it reappears in every formula below. Take five monthly inflation readings, in percentage points: 2.0, 2.5, 3.0, 3.5, 4.0. The average is (2.0 + 2.5 + 3.0 + 3.5 + 4.0) / 5 = 15 / 5 = 3.0pp.

Now the deviations from that average: −1.0, −0.5, 0.0, +0.5, +1.0. If you simply *averaged* these deviations you would get zero — they always cancel, by construction, because the average sits in the middle. That is why we **square** them first: −1.0² = 1.0, −0.5² = 0.25, 0² = 0, 0.5² = 0.25, 1.0² = 1.0. Sum = 2.5; divide by 5 to get the **variance = 0.5** (in "percentage-points squared", an awkward unit). Take the square root to get the **standard deviation = 0.71pp** — back in plain percentage points, and readable as "a typical month sits about 0.71pp from the 3.0pp average."

Squaring is not arbitrary: it kills the cancellation problem *and* it makes far-away points count more (a deviation of 2 contributes 4, a deviation of 1 contributes 1). That same squaring is what makes both variance and Pearson's r sensitive to outliers — a feature when extremes are meaningful, a bug when one bad dot dominates. **Standard deviation is just "the typical distance from the average, in the variable's own units" — and it is the ruler we divide covariance by to turn it into a correlation.**

### Correlation: covariance scrubbed of units

The correlation coefficient — written **r** (or the Greek letter ρ, "rho") — fixes this with one elegant move: divide the covariance by *both* standard deviations.

```
r = covariance(X, Y) / ( stdev(X) * stdev(Y) )
```

Dividing by the standard deviations does two things at once. First, it **cancels the units**: the "percent" in the numerator cancels the "percent" from stdev(X), the "dollars" cancels the "dollars" from stdev(Y), and r comes out as a pure, unit-free number. Second — and this is the beautiful part — it **mathematically forces r into the range −1 to +1**. It is impossible to get an r of 1.5 or −3. There is a theorem (the Cauchy–Schwarz inequality, if you want to look it up; you do not need it) guaranteeing the covariance can never exceed the product of the two standard deviations in size. So r is *bounded*, and that bound is what makes it universally comparable: an r of −0.8 means the same strength of negative relationship whether you are looking at gold and yields, or jobs and stocks, or rainfall and umbrella sales.

![Pipeline from raw data through covariance and standard deviations to r and R squared](/imgs/blogs/what-correlation-actually-measures-pearson-spearman-beta-7.png)

Here is the scale, with what each value actually means:

- **r = +1**: a perfect positive line. Every dot sits exactly on an upward-sloping straight line. Knowing X tells you Y with no error.
- **r = +0.7 to +0.9**: strong positive. The cloud is a clear, fairly tight upward line, with some scatter.
- **r = +0.3 to +0.6**: moderate positive. There is a real upward tilt, but plenty of dots wander off it.
- **r ≈ 0**: no *linear* relationship. The cloud is a shapeless blob (or — careful — a strong but non-straight relationship; more on that below).
- **r = −0.3 to −0.9**: the same strengths, downward.
- **r = −1**: a perfect negative line.

A note on language, because it trips up almost everyone: **r is a measure of strength, not of magnitude of movement.** An r of 0.6 does *not* mean "Y moves 60% as much as X" or "they agree 60% of the time". It means the *tightness* of the linear relationship is moderate. How *much* Y moves per unit of X is a completely separate number — beta — that we will get to, and the two can be wildly different. You can have a near-perfect correlation (r = 0.99) with a tiny slope, and a weak correlation (r = 0.3) with an enormous slope.

#### Worked example: computing r by hand on a tiny dataset

Let's make it concrete with five observations, simple enough to do on paper. Suppose over five months we record a stylized inflation surprise X (in percentage points) and the same-month change in the 10-year yield Y (in basis points):

```
Month:     1     2     3     4     5
X (pp):   -0.2  -0.1   0.0   0.1   0.2
Y (bp):   -12    -5     1     8    13
```

Step 1 — the averages. average(X) = (−0.2 − 0.1 + 0 + 0.1 + 0.2) / 5 = 0.0. average(Y) = (−12 − 5 + 1 + 8 + 13) / 5 = 5/5 = 1.0.

Step 2 — the gaps and their products:

```
Month  gapX    gapY   gapX*gapY   gapX^2   gapY^2
  1   -0.2   -13.0      2.60       0.04     169
  2   -0.1    -6.0      0.60       0.01      36
  3    0.0     0.0      0.00       0.00       0
  4    0.1     7.0      0.70       0.01      49
  5    0.2    12.0      2.40       0.04     144
 sum                    6.30       0.10     398
```

Step 3 — assemble. Covariance ∝ sum of gapX·gapY = 6.30. The two "spread" sums are sum of gapX² = 0.10 and sum of gapY² = 398. (Because the same divide-by-N appears in the numerator and denominator of r, it cancels, so we can skip it and just use the sums.) Then:

```
r = 6.30 / sqrt(0.10 * 398)
  = 6.30 / sqrt(39.8)
  = 6.30 / 6.31
  = 0.999
```

The correlation is essentially +1: in this stylized sample, a bigger inflation surprise reliably comes with a bigger jump in yields, in lockstep. **The mechanics are just "do the deviations line up?" turned into one bounded number — and here they line up almost perfectly, so r ≈ +1.**

### R-squared: how much does the relationship actually explain?

Now the third number, and the one that quietly deflates a lot of confident-sounding claims. **R-squared is exactly what its name says: r, squared.** If r = 0.6, then R-squared = 0.36. If r = −0.8, then R-squared = 0.64.

What does squaring r *mean*? It is the **fraction of the variance in Y that is "explained" by the straight-line relationship with X.** Recall variance is the total vertical spread of the Y dots around their average. Fit the best straight line through the cloud. Some of Y's spread now sits *along* that line (explained by X moving), and some sits *as scatter around* the line (everything else — other drivers, noise, randomness). R-squared is the share that sits along the line:

```
R-squared = variance of Y explained by the line / total variance of Y
          = (a value from 0 to 1, often quoted as a percent)
```

This reframes correlation in a way that is much harder to oversell. A correlation of 0.6 *sounds* impressive — more than half! But squared, it is 0.36: the relationship accounts for only **36% of the ups and downs in Y**, leaving 64% to other forces. A correlation of 0.3 — which people often quote as "there's a relationship" — has R-squared 0.09: it explains *nine percent* of the variance and ignores 91% of it. Even a strong-sounding r = 0.7 explains only 49% — less than half.

#### Worked example: turning a quoted correlation into "how much it explains"

Suppose a research note tells you "the correlation between ISM new orders and S&P earnings growth is 0.65." Should you reorganize your portfolio around it?

Square it: R-squared = 0.65 × 0.65 = 0.42. So ISM new orders, in this stylized figure, accounts for about **42% of the variation** in earnings growth. That is genuinely useful — it is a real, leading signal worth tracking (we cover it in [ISM and PMI as leading signals](/blog/trading/event-trading/ism-pmi-the-business-surveys-that-lead)). But 58% of earnings-growth variation comes from elsewhere: margins, taxes, one-off charges, sector mix. So the honest read is "a meaningful tilt, not a crystal ball." **Always square the correlation before you decide how much to trust it; an r of 0.65 leaves the majority of the variance unexplained.**

A second, subtler use of R-squared: it lets you compare a relationship to itself over time. If gold's R-squared against real yields was 0.93 in one decade and 0.10 in the next, you do not need any other statistic to know the relationship fell apart. Hold that example — we return to it at full force below.

### Two things correlation does *not* capture (and you must check separately)

The single-number summary is powerful precisely because it throws information away — but you have to know *what* it threw away, or you will be surprised. Two omissions matter most in macro.

**Timing (lead/lag).** Pearson's r as we defined it compares X and Y *at the same instant*. But many macro relationships are about *timing*: building permits lead GDP by roughly nine months; the inverted yield curve leads recessions by over a year; credit spreads lead equity drawdowns by a few months. A *contemporaneous* correlation between a leading indicator and what it leads can look weak — because the indicator already moved and the asset has not caught up yet. To find these you compute correlations at *shifted* offsets (a "cross-correlation function") and look for the offset where the correlation peaks. That entire dimension is invisible to the plain r, and it is the subject of [leading, coincident, and lagging indicators](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders). For this primer, the lesson is: a low contemporaneous correlation does *not* prove "no relationship" — the relationship may simply be lagged.

**Tail behavior (what happens in the extremes).** Pearson's r is an *average* over the whole sample, which means it describes the *typical* day far better than the *extreme* day. Two assets can have a comfortable −0.3 correlation in normal times yet both crash together in a panic — their "tail correlation" is near +1 even though their average correlation is negative. Because the formula weights all observations into one mean, it can mask the fact that the relationship you care about (the one in a crisis) is the opposite of the average. We return to this in the diversification-failure case below; flag it now as the reason "the average correlation" can be dangerously reassuring.

## Pearson versus Spearman: linear versus monotonic

Everything above describes **Pearson's correlation** — the classic r, the covariance-over-standard-deviations formula. Pearson has one giant assumption baked in that almost nobody says out loud: **it only measures *straight-line* relationships.** It asks "how close is this cloud to a *straight* line?" If the true relationship is a strong but *curved* one, Pearson will understate it — sometimes badly — because the dots, though following a clear pattern, do not sit on a *straight* line.

This is where **Spearman's correlation** (the Greek ρ again, but "Spearman's rho" specifically) earns its keep. Spearman does something clever: it throws away the actual values and keeps only their **ranks** — 1st smallest, 2nd smallest, and so on — then runs Pearson's formula on the *ranks*. Because it only cares about *order*, Spearman measures whether the relationship is **monotonic**: does Y consistently go up as X goes up (even if the curve bends), or consistently down? It does not require the path to be straight.

The difference matters enormously in macro because real relationships are full of curves and saturation:

- A relationship can **saturate**: the first units of X move Y a lot, later units move it less. (Think of stimulus, or of a metric that hits a ceiling.) Pearson sees the bend as "imperfect" and marks it down; Spearman sees a clean staircase and reports near-perfect.
- A relationship can be **threshold-shaped**: flat until some level, then steep. Pearson dilutes the steep part with the flat part; Spearman, again, just checks the ordering.

![Two scatter panels Pearson understates a curve and is fooled by an outlier](/imgs/blogs/what-correlation-actually-measures-pearson-spearman-beta-3.png)

The left panel above shows a strictly **monotonic but curved** relationship — Y always rises with X, but it bends and flattens. Pearson reads **r = 0.77** ("strong-ish, with notable scatter"), while Spearman reads **ρ = 1.00** ("perfectly ordered — every increase in X comes with an increase in Y, no exceptions"). They disagree by a lot, and Spearman is telling the truer story: there is *no noise here at all*, just curvature that Pearson's straight-line ruler cannot see.

#### Worked example: when Pearson and Spearman disagree, and which to believe

Take a saturating series — say, ten readings where Y climbs fast then levels off: Y = 0, 55, 78, 89, 94, 96, 97, 98, 98, 99 as X goes 0 through 9. Eyeball it: Y *never once decreases* as X rises. That is a perfectly ordered relationship. Spearman, working on ranks, returns ρ = 1.00. Pearson, insisting on a straight line, returns only 0.77 — it "loses" 23 points of apparent strength purely to the curve.

Which do you believe? It depends on the question. If you only care *whether higher X reliably means higher Y* (a directional signal — "does a hotter print reliably push yields up?"), trust Spearman: the answer is an unambiguous yes. If you care *how much* Y moves per unit X — and you are willing to model the curve — Pearson on the raw data understates a real, usable relationship, and the fix is to look at the scatter and fit the curve, not a line. **When Pearson and Spearman diverge, it is a flag that the relationship is curved or has outliers — go back to the scatter; do not just quote the bigger number.**

There is a flip side worth stating: when the relationship genuinely *is* roughly linear and clean, Pearson and Spearman come out close, and Pearson is the right tool because it also feeds directly into beta and R-squared. Spearman is your *diagnostic* — run both, and when they disagree, the scatter will show you why.

#### Worked example: computing Spearman by ranking, on a curved pair

Spearman is easier to compute than it sounds — you just replace every value with its rank, then run the same correlation logic. Take four observations of a curved-but-monotonic pair: X = 1, 2, 3, 4 and Y = 2, 9, 28, 65 (a steep cubic-like climb).

Rank each variable from smallest (1) to largest:

```
Obs:    A    B    C    D
X:      1    2    3    4
rankX:  1    2    3    4
Y:      2    9   28   65
rankY:  1    2    3    4
```

The ranks line up *perfectly*: the smallest X goes with the smallest Y, the largest with the largest, with no exceptions. So Spearman's ρ = +1.00 — a perfect monotonic relationship. But run *Pearson* on the raw values and you get only about 0.92, because the cubic curve bends away from a straight line. Pearson docks 8 points for curvature that does not actually exist as *disorder*. The shortcut formula many textbooks teach — ρ = 1 − 6·Σd² / (n(n²−1)), where d is the rank difference per observation — gives the same answer instantly here: every d = 0, so ρ = 1 − 0 = 1. **Spearman asks only "do the orderings match?", which is exactly the right question when you care about direction rather than the shape of the curve.**

### Outliers and leverage points: how one dot rewrites the story

The right panel of the figure above shows Pearson's *other* great weakness, and it is even more dangerous in practice. Start with a cloud of ten dots that have **no relationship at all** — Pearson reads r = −0.18, basically zero, which is correct. Now add *one* extreme point far out in the top-right corner, an observation where both X and Y happen to be huge. Re-compute. Pearson now reads **r = +0.95** — a "very strong positive correlation" — conjured entirely out of a single dot.

This is called a **leverage point**: an observation with an extreme X value that sits far from the rest, and so exerts enormous pull on the best-fit line and on r, like a person standing at the far end of a see-saw. The squaring inside the formula is the culprit — far-away points contribute *squared* distances, so a point that is 10× farther out counts 100× more. One crisis month, one data-error, one once-in-a-decade print can manufacture or destroy a correlation that the other 99% of your data does not support.

Spearman is far more robust here, because ranks compress extremes: the outlier is still "the largest", rank 11 out of 11, no matter how astronomically large its raw value — its rank cannot be 1,000. So if Pearson says 0.95 and Spearman says 0.10, you have an outlier problem, full stop.

#### Worked example: a single point flipping the sign of a "correlation"

Imagine an analyst pooling daily data from 2015 through 2021 to estimate the correlation between some macro factor and an asset, and the sample happens to include March 2020 — the COVID crash, when *everything* moved violently in the same week. That one month is a giant leverage point. Strip it out and the correlation over the calm years might be near zero; leave it in and it can read 0.6 or more, driven almost entirely by the panic. The analyst publishes "0.6, a meaningful relationship" — and a year later it "mysteriously" stops working, because there was never a 0.6 relationship in normal times; there was a zero relationship plus one crisis. **Before you trust any correlation, ask which single observation is doing the work — drop your most extreme dot and re-compute; if the number collapses, you have a leverage artifact, not a signal.** We treat this trap in full in [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data).

## Correlation versus beta: strength versus sensitivity

We have now nailed down *correlation* (sign and strength, unit-free, bounded) and its square *R-squared* (variance explained). The third number — and the one that actually tells you how to *size a position* — is **beta**.

Beta is the **slope of the best-fit line.** When you draw the straight line that best fits the scatter (the line that minimizes the squared vertical distances — "ordinary least squares", OLS), beta is its steepness: *how many units of Y you get per one unit of X.* Critically, **beta carries units.** It is not a number between −1 and +1. It is "8 basis points of yield per 0.1 percentage point of CPI surprise", or "−354 dollars of gold per 1 percentage point of real yield", or "the stock moves 1.3% for every 1% the index moves."

The relationship between correlation and beta is exact and worth memorizing:

```
beta = r * ( stdev(Y) / stdev(X) )
```

Read that carefully, because it explains why the two numbers can diverge so wildly. Beta is correlation *times the ratio of the two spreads.* Correlation tells you the *tightness*; the ratio of standard deviations rescales that into Y's units. So:

- A relationship can have a **high correlation but a tiny beta** — if Y barely varies (small stdev(Y)), the slope is gentle even when the dots hug the line. (Two government bond yields that move almost identically but in small amounts.)
- A relationship can have a **modest correlation but a huge beta** — if Y is very volatile (large stdev(Y)), even a loose linear link translates into big moves. (A loose link to Bitcoin, which swings enormously.)

This is the heart of the matter: **correlation tells you *whether* X and Y are linked and how reliably; beta tells you *how much* Y moves when X does.** You need correlation to decide whether the relationship is worth trusting, and beta to decide how big to size around it. Quoting one without the other is half a sentence.

### The best-fit line: what "regression" actually does

Beta is the slope of "the best-fit line", so it is worth being precise about what *best* means. Draw any straight line through the scatter. For each dot, measure the *vertical* gap between the dot and the line — that is the line's error for that observation (the "residual"). Square each gap (the same squaring trick, to stop positives and negatives cancelling and to penalize big misses more) and add them all up. The **best-fit line is the one specific line that makes that total squared error as small as possible.** This procedure is called **ordinary least squares**, or OLS, and "regressing Y on X" is just shorthand for "find the OLS line."

Two outputs come from that line. The **slope** is beta — the rise in Y per unit run in X. The **intercept** is where the line crosses the Y-axis (the predicted Y when X = 0), which is often not economically meaningful but is needed to position the line. The residuals — the leftover scatter around the line — are the part of Y that X *failed* to explain, and their size relative to Y's total spread is exactly what R-squared measures: R-squared = 1 − (variance of the residuals / total variance of Y). So the three numbers are one machine seen from three angles: OLS draws the line, beta is its slope, r measures how tightly the dots hug it, and R-squared is the share of Y's wiggle the line captured.

#### Worked example: getting beta from correlation and the two volatilities

You do not always need the raw data to recover a beta — if you know the correlation and the two standard deviations, the formula beta = r × (stdev(Y) / stdev(X)) does it directly. Suppose a desk tells you the correlation between a CPI surprise and the 2-year yield is r = 0.6, that CPI surprises have a standard deviation of 0.12pp, and that the 2-year yield's daily change has a standard deviation of 5.5 bp on CPI days. Then:

```
beta = 0.6 * (5.5 bp / 0.12 pp)
     = 0.6 * 45.8 bp per pp
     = 27.5 bp per pp
     = 2.75 bp per 0.1pp surprise
```

So a +0.2pp surprise implies a point estimate of roughly 2.75 × 2 = +5.5 bp on the 2-year. Notice the *same correlation of 0.6* would produce a completely different beta if the 2-year were twice as volatile — the slope scales with the volatility ratio, not with r. **This is why "high beta" and "high correlation" are not the same thing: beta is correlation amplified (or muted) by how volatile the two series are relative to each other.**

![Beta to a CPI upside surprise in basis points and percent across assets](/imgs/blogs/what-correlation-actually-measures-pearson-spearman-beta-5.png)

The figure above makes the units point unavoidable. It shows the **beta of each asset to a +0.1 percentage-point upside surprise in core CPI** during the 2022–23 inflation-fear regime. Look at how the units *differ across the bars*: stocks and gold and Bitcoin move in *percent*; the 10-year and 2-year yields move in *basis points*. A single unitless correlation could never carry that information. The 2-year yield rises about +9 bp, the 10-year about +7 bp, while the Nasdaq falls about −1.0% and Bitcoin about −1.6%. (These are illustrative event-study magnitudes for that regime, synthesized from published reaction studies — the sign and rough size are well documented, the exact value depends on the sample.) Notice the **signs all agree with one mechanism**: a hot inflation print is hawkish, so yields up, the dollar up, and risk assets down — which we unpack as a chain in [cross-asset transmission of a single print](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market). What this *measurement* post adds is: those are *betas*, not correlations, and that is why they wear different units.

#### Worked example: computing a beta and using it to size a trade

Suppose you have estimated, for the current regime, that the 10-year yield's beta to a core-CPI surprise is **+7 bp per +0.1pp surprise**, with a correlation of r = 0.6 (so R-squared = 0.36 — the surprise explains 36% of the same-day yield move, the rest is positioning and other news).

Tomorrow's CPI comes in **+0.3pp above consensus** on core — a big hot surprise. Your point estimate for the yield move is beta × surprise = 7 bp × (0.3 / 0.1) = **+21 bp**. If you hold a bond position with a DV01 (dollar value of a 1 bp move) of \$10,000, the expected mark-to-market hit is roughly 21 × \$10,000 = **\$210,000** against you. That is the number you size against — not the correlation. The correlation (0.6) tells you to *attach only moderate confidence*: 64% of the move is *not* explained by the surprise, so you should not bet the farm. **Beta gives you the dollar magnitude to plan for; correlation tells you how much faith to put in that estimate; you need both to size honestly.** The mechanics of betting on the *surprise* rather than the *level* are the subject of [the surprise, not the level: betas to data surprises](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises).

### A worked beta from real data: gold and the real yield

Let's compute a real beta, not a stylized one, from this series' cleanest pair: the 10-year real yield versus the price of gold, over 2007–2021.

![Scatter of gold price against ten year real yield with regression line and r](/imgs/blogs/what-correlation-actually-measures-pearson-spearman-beta-2.png)

The scatter above is built from annual data: the 10-year TIPS real yield on the X-axis (in percent) and the average gold price on the Y-axis (in dollars per ounce), one dot per year from 2007 to 2021. The cloud tilts sharply *down* — high real yields come with cheap gold — and it is *tight*. The fitted line gives:

- **Correlation r = −0.96.** Extremely strong and negative — among the cleanest relationships in all of macro. Squared, **R-squared = 0.93**: the real yield alone explains 93% of the year-to-year variation in gold over this window. (That is freakishly high for macro; most relationships are far noisier.)
- **Beta (slope) = −354 dollars per ounce per +1.0 percentage point of real yield**, which is about **−35 dollars per ounce per +0.1pp.**

Now watch the two numbers do their separate jobs. The correlation (−0.96) tells you this relationship is *trustworthy* — it is not one outlier, the whole cloud lines up. The beta (−\$35/oz per 0.1pp) tells you the *magnitude*: if the real yield rises 0.5pp, your point estimate is gold falling about 5 × \$35 ≈ **−\$175/oz**. You cannot get that dollar figure from the correlation; you cannot get the trust from the beta. This is *why* gold is best understood through real yields rather than through inflation directly — a point we make in full in [real yields, the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything). Here, it is our showcase for "a strong, clean, usable correlation."

## Common misconceptions

A handful of confusions cause the great majority of correlation mistakes. Each one is correctable with a number.

**"Correlation means causation."** No. A correlation is a *description* of co-movement; it says nothing about *why*. Two series can correlate because A causes B, because B causes A, because a third thing C drives both, or by pure chance over a short sample. The classic macro version: gold and inflation expectations sometimes correlate — but the *cause* is that both respond to **real yields**, the genuine driver. Mistake the correlation for causation and you will be blindsided the moment the third variable behaves differently. Correlation tells you *what* moved together; you need a *mechanism* (the job of the [macro-trading series](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal)) to claim *why*.

**"An r of 0.6 is a strong relationship."** It is *moderate*, and — this is the part people forget — it explains only **36%** of the variance (R-squared = 0.36). Most of the movement is still unexplained. A correlation of 0.3, which sounds "real", explains a mere 9%. Always square it before you trust it.

**"Gold is an inflation hedge, so it correlates with CPI."** Over the long run, gold's correlation with CPI is *weak and unstable*; its correlation with *real yields* is strong and negative (we just measured −0.96 for one window). The "inflation hedge" story works only because high inflation *sometimes* coincides with low real yields — but when inflation rises *and real yields rise faster* (as in 2022), gold can fall even as CPI screams. Correlate gold with the right variable and the fog clears; this is [the real-yield story](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) the series keeps returning to.

**"A high correlation means a big move."** Correlation is *strength*, not *magnitude*. Two assets can be 0.99 correlated and one barely moves while the other swings 5% a day — because beta (which carries the magnitude) depends on the *ratio of their volatilities*, not on r alone. If you want to know how much something will move, you need beta, not correlation.

**"Correlate the levels."** A trap so common it deserves its own myth. If you compute the correlation between the *level* of two trending series — say the price level of two assets that both drifted upward for a decade — you will get a huge r that means almost nothing, because both were simply going up over time. Two unrelated things that both trend will correlate spuriously. The fix is almost always to correlate the *changes* (returns, or month-over-month differences), not the levels — which strips out the shared trend and asks the real question: "when one moves *more than usual*, does the other?" We dig into this non-stationarity trap in [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data); for now, remember: **default to correlating changes, not levels.**

**"The full-sample correlation is the correlation."** This is the most expensive myth in the series, so it gets its own section.

## How it shows up in real markets

### The number that lies: gold and the real yield, pooled

Take the *exact same* gold-versus-real-yield data, but instead of looking only at 2007–2021, pool **the whole 2007–2025 history** into one correlation. What do you get?

![Bar chart of one pair across three windows showing the full sample correlation hides the regime](/imgs/blogs/what-correlation-actually-measures-pearson-spearman-beta-6.png)

The full-sample correlation is **r = −0.01** — essentially *zero*. A naive analyst pooling all the years would conclude "gold and real yields are unrelated; ignore it." That conclusion is catastrophically wrong, and the bar chart shows why. Split the same data by regime:

- **2007–2021: r = −0.96.** A near-perfect negative relationship — the canonical "gold falls when real yields rise" story.
- **2022–2025: r = +0.80.** A *strong positive* relationship — gold rose *alongside* rising real yields, as central-bank buying and geopolitical demand overwhelmed the old mechanism.

Pool a strong negative and a strong positive together, and they *cancel to zero.* The full-sample number does not say "no relationship exists." It says "I have averaged two opposite regimes into a meaningless middle." This single example is the thesis of the whole series compressed into three bars: **correlation is a regime, not a constant.** A relationship can be one of the strongest in macro and still read zero over a window that straddles its sign flip.

#### Worked example: why the full-sample number reads zero

You do not need the formula to feel this; you need the picture. In 2007–2021 the dots form a tight downward line (top-left to bottom-right). In 2022–2025 they form a tight *upward* line (bottom-left to top-right). Overlay both on one scatter and you get an **X shape** — two crossing lines. A single best-fit straight line through an X is *flat*: it splits the difference and has slope ≈ 0, hence r ≈ 0. The math is doing exactly what you asked — fit *one* line to data that needs *two* — and the answer is honestly useless. **When a relationship you believe in reads zero over a long sample, suspect a sign flip and split the window; one flat line through two opposite regimes always reads zero.** This is precisely why the next post is about [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters) — a *rolling* correlation would have shown the −0.96 collapsing and re-emerging as +0.80, instead of hiding it in a single misleading average.

### The 2022 stock–bond flip, in correlation terms

Return to our risk manager from the opening. The stock–bond correlation she relied on had been negative for roughly two decades — call it around −0.3 to −0.5 across the 2000s and 2010s — and that negative sign is *what makes bonds a hedge for stocks*. In 2022 it flipped *positive*, to roughly +0.6, the highest in a generation. The driver was the **inflation regime**: when inflation is the market's dominant fear, a hot print sends *both* stocks and bonds down together (higher discount rates hurt both), so they correlate *positively* and the diversification vanishes exactly when it is needed.

In measurement terms, three things changed at once and you need all three numbers to describe it: the **sign** flipped (correlation went from negative to positive), the **strength** intensified (the magnitude of r rose), and — though we will not derive it here — the **beta** of bonds to the same shocks changed too. A single quoted "stock–bond correlation of −0.2" (the lazy full-sample average) would have told you *none* of this. We dedicate a full post to it in [the stock–bond correlation regime](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine); the measurement lesson here is that "the correlation" was never one number — it was a regime-conditional one.

### One driver, many assets: the dollar's cross-asset correlations

A correlation does not have to be between *two* assets — one of the most useful patterns in macro is *one driver* correlated against *many* assets at once, which sketches the entire transmission map in a single column of numbers. The US dollar (measured by the DXY index) is the classic case; traders call it "cross-asset gravity." Here are its longer-run correlations with each asset's returns, as researched approximations:

- **Gold: r = −0.55.** A stronger dollar pressures gold (gold is priced in dollars).
- **Oil (WTI): r = −0.45**, and **copper: r = −0.50.** Commodities are dollar-denominated, so a stronger dollar makes them pricier abroad and demand softens.
- **EM equities: r = −0.55.** A strong dollar tightens financial conditions for emerging markets that borrow in dollars.
- **Bitcoin: r = −0.35** — a looser link, but the same sign.
- **S&P 500: r = −0.20** — weak; US large-caps are far less dollar-sensitive than commodities or EM.
- **US 10-year yield: r = +0.40** — *positive*, and the lone one: higher US yields *attract* capital and *strengthen* the dollar.

Read this column the way this primer trained you. The *signs* tell a coherent mechanism story (a strong dollar pressures the dollar-denominated and dollar-funded, while higher yields pull the dollar up). The *strengths* rank the exposures: gold, copper, and EM are the most dollar-sensitive (around −0.5), while the S&P is barely linked (−0.20, R-squared just 0.04 — the dollar explains a trivial 4% of S&P variation). And the *one positive sign* (yields) is the causal driver hiding in plain sight — the dollar is strong *because* yields are high, not the reverse. None of this would survive being collapsed into "the dollar matters for risk"; the numbers are the whole point. We give the dollar its own deep-dive in [the dollar as cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity).

### A correlation that is *born* and then *fades*: crypto and the Nasdaq

Correlations do not only flip sign — they can *appear from nothing* and then dissolve, which is its own kind of regime story. Bitcoin's correlation with the Nasdaq 100 is the cleanest example. Track the rolling 90-day correlation of BTC daily returns with the Nasdaq:

- **2019: r ≈ 0.05.** Essentially uncorrelated — Bitcoin traded on its own crypto-native narrative.
- **2022 (first half): r ≈ 0.65.** A strong link emerged as Bitcoin started trading as a high-beta, macro-liquidity asset: when the Fed tightened, both tech and crypto fell together.
- **2024–2025: r ≈ 0.20–0.30.** The link faded again as crypto-specific drivers (ETF flows, halving cycles) reasserted themselves.

#### Worked example: a correlation that triples then halves

In 2019, an analyst computing BTC-versus-Nasdaq correlation would have found ≈ 0.05 and concluded, correctly for that era, "Bitcoin is an uncorrelated diversifier." By mid-2022 the same calculation returned ≈ 0.65 — a *thirteen-fold* jump in correlation — and "Bitcoin is just leveraged Nasdaq" became the consensus. By 2024–2025 it had eased to ≈ 0.25. A trader who sized a \$1,000,000 "crypto diversifier" allocation in 2021 on the 2019 correlation would have discovered in 2022 that the position was actually *amplifying* their tech-stock risk, not offsetting it — the diversification they paid for had quietly turned into concentration. **A correlation has a birthday and an expiry date; the number you measured last year may describe a regime that no longer exists, which is why this series insists on the rolling view rather than a single estimate.** We trace crypto's full macro link in [crypto as a macro asset: the liquidity correlation](/blog/trading/macro-correlations/crypto-as-a-macro-asset-the-liquidity-correlation), and the moving-window method that catches these shifts in [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters).

### Why diversification fails when you need it

One more real-world pattern that only makes sense once you separate strength from regime: in a crisis, *correlations across almost all risk assets rush toward +1.* In calm times, stocks, credit, EM, commodities, and crypto have moderate and varied correlations — that variety is the raw material of diversification. But in a panic (2008, March 2020), forced selling and deleveraging make *everything* sell off together, and the cross-asset correlations spike toward 1. The diversification that your *calm-period* correlation matrix promised evaporates in the *one regime* where you needed it. That is the brutal corollary of "correlation is a regime": the relationships you are counting on are conditional on the world staying ordinary, and crises are exactly when it does not. We cover this in [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).

The magnitudes make the point visceral. In the 2008 financial crisis the S&P 500 fell about 57% peak-to-trough; in the March 2020 COVID crash it fell about 34%; in the 2022 rate shock about 25%. In each, "diversifying" assets that had spent years lightly correlated suddenly tracked the sell-off — credit spreads blew out *with* equities (their normal strong-negative relationship tightening toward a perfect one), and even gold and Treasuries wobbled in the worst days as funds raised cash by selling whatever they could. The tail correlation — the correlation *conditional on a crash* — is the number that actually determines whether your hedge holds, and it is almost never the comfortable average correlation that a backtest reports. This is the deepest reason the series treats correlation as a conditional, regime-dependent quantity rather than a constant: the number you most need to know (how things move together when it matters) is exactly the number a full-sample average is worst at telling you.

#### Worked example: the hedge that vanishes in the tail

Suppose your calm-period numbers say credit (a high-yield bond fund) has a correlation of −0.3 with your equity book, and you hold \$2,000,000 of each, treating the credit as a partial offset. In a deleveraging crisis, that −0.3 can rush toward +0.8 as both sell off together. If equities drop 20% (a \$400,000 loss) and credit, now moving *with* them, drops 12% (a \$240,000 loss) instead of cushioning, your "diversified" book loses \$640,000 — *more* than either leg alone would suggest from the calm correlation, because the offset you paid for turned into amplification. **The correlation you hedge on must be the crisis-regime correlation, not the average one; a hedge sized on the calm number is no hedge in the storm.**

## How to read it and use it

Here is the practical checklist this primer leaves you with — the habits that turn a correlation number from a trap into a tool.

**1. Plot it first, always.** Before you quote any r, draw the scatter. Your eye catches the curve Pearson would understate, the outlier that is doing all the work, and the X-shape of a regime flip — none of which the single number reveals. A correlation you have not plotted is a correlation you do not understand.

**2. Run both Pearson and Spearman.** If they agree, the relationship is roughly linear and clean; use Pearson (it feeds beta and R-squared). If they disagree, you have curvature or outliers — go back to the scatter and find out which. The *gap* between them is itself a diagnostic.

**3. Square the correlation before you trust it.** r = 0.6 explains 36%; r = 0.3 explains 9%; r = 0.7 explains 49%. The squared number is the honest measure of "how much does this actually account for." Most macro relationships, squared, are humbling.

**4. Quote correlation *and* beta together.** Correlation = how reliable; beta = how much, in real units. Use the correlation (and its R-squared) to decide *whether and how confidently* to act; use the beta to decide *how big to size*. Either one alone is half the picture.

**5. Find the leverage point.** Drop your single most extreme observation and re-compute. If the correlation collapses, it was an artifact of one dot — a crisis month, a data error, a structural break — not a relationship you can trade.

**6. Never trust a full-sample number for a relationship that can flip.** This is the master rule of the series. A pooled correlation across regimes can read anything from its true value to *zero* to the *opposite sign*. Split by regime, or better, compute it on a *rolling* window so you can watch it move. Ask of every correlation: *which regime produced this number, and is that regime still live?*

#### Worked example: the full workflow on one relationship

Put the whole checklist together on a single decision. You are considering using gold as a hedge in a portfolio, and you want to know how it behaves versus real yields *right now*. Step 1, plot it — and the 2007–2025 scatter shows the tell-tale X shape, two crossing clouds. Step 2, do not quote the full sample (r = −0.01); split by regime. Step 3, in the *current* (2022–2025) regime the correlation is r = +0.80, R-squared = 0.64 — strong, and *positive*, the opposite of the textbook. Step 4, square it: 64% of gold's variation tracks real yields in this regime, a high and tradeable share. Step 5, find the beta for the current regime, not the old one. Step 6, size: if you hold \$500,000 of gold and your current-regime beta says gold gains roughly \$35/oz per +0.1pp of real yield, you can translate a forecast of a +0.3pp real-yield move into a point estimate of roughly +\$105/oz — and, because the sign is now *positive*, conclude that gold is *not* hedging your duration risk in this regime; it is *adding* to it. A trader who skipped to "gold and real yields are negatively correlated, so gold hedges rate risk" — the full-sample, wrong-regime, unsquared, beta-free version — would have built exactly the wrong position. **Plot, split, square, find the beta, check the sign, size — six steps, and every one of them needs a different number you now know how to read.**

**What invalidates the read.** A correlation estimate is invalidated when (a) the scatter shows a curve or a dominating outlier (Pearson is the wrong tool — use Spearman or a robust fit); (b) the sign is different in the recent window than the full sample (a flip is underway — the [rolling view](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters) will show it); (c) the relationship has no mechanism you can articulate (it may be [spurious](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data)); or (d) the R-squared is low enough that the "signal" is mostly noise. With those four checks, the three numbers — r, beta, R-squared — become exactly what they should be: a precise, honest summary of a picture you have already looked at.

Everything else in this series is built on these three numbers. When a later post says "CPI surprises and the dollar correlate +0.45 in the inflation-fear regime, with a beta of +0.35% per 0.1pp," you now know precisely what each clause means, how strong 0.45 really is (R-squared 0.20 — a fifth of the variance), why the beta wears a percent sign, and why the phrase "in the inflation-fear regime" is not a hedge but the whole point.

## Further reading and cross-links

Within this series:

- [Rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters) — the direct sequel: how a *moving* correlation reveals the regime flips that a full-sample number hides.
- [Spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data) — non-stationarity, multiple testing, the third-variable problem, and the leverage-point trap in full.
- [The surprise, not the level: betas to data surprises](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) — why you correlate the *surprise* (actual minus consensus), and how to estimate beta-to-surprise.

For the mechanisms behind these correlations (the *why*, not the *measurement*):

- [Real versus nominal: inflation and the real-yield master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — why real yields, not CPI, are gold's true driver.
- [How policy moves every asset: the cross-asset transmission map](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map) — the causal chain that *produces* the betas we measured.
- [Cross-asset transmission: how one print hits every market](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market) — the intraday version of the CPI-surprise betas.

For the cross-asset correlation structure these tools describe:

- [Real yields, the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything) — the gold-vs-real-yield relationship in depth.
- [Stock–bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) — the regime flip from the opening, fully measured.
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — why diversification fails exactly when you need it.
