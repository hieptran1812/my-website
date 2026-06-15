---
title: "Stationarity, autocorrelation, and ergodicity: when the past is allowed to predict the future"
date: "2026-06-15"
description: "A beginner-friendly, build-from-zero tour of stationarity, the autocorrelation function, unit roots and the Dickey-Fuller test, why prices are non-stationary but returns are not, and the ergodicity trap that quietly breaks single-sample reasoning."
tags: ["stationarity", "autocorrelation", "ergodicity", "time-series", "unit-root", "mean-reversion", "augmented-dickey-fuller", "quant-finance", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 45
---

> [!important]
> **TL;DR** — Before you can trust any model fit on market history, you have to ask whether the *rules of the series stayed the same* over that history. Stationarity, autocorrelation, and ergodicity are the three tests that answer it.
>
> - **Stationarity** means the statistical character of a series — its average level, its spread, its memory — does not drift over time. A model fit on a stationary series can hold; a model fit on a non-stationary one is fitting a moving target.
> - **Autocorrelation** measures how much today resembles yesterday. Positive autocorrelation is *momentum* (trends persist); negative autocorrelation is *mean reversion* (deviations snap back); near-zero is noise with no edge.
> - **Prices are non-stationary** (they wander with no fixed level — a "random walk" with a *unit root*), but **returns are roughly stationary**. The augmented Dickey-Fuller (ADF) test is the standard way to tell which you are holding.
> - **Ergodicity** is the deepest and most dangerous: it is the condition under which a single long history can stand in for the whole distribution. When a market switches regimes, that condition fails — and one historical sample lies to you about the future.
> - The one number to remember: a mean-reverting spread with a lag-1 autocorrelation of **0.87** has a **half-life of about 5 days**, which is exactly how long you should expect to wait to collect your edge.

Here is a question that quietly decides whether a quantitative strategy makes money or detonates: *is tomorrow's market going to behave like the market you just measured?*

Every backtest, every fitted model, every Sharpe ratio you ever compute rests on a silent assumption — that the slice of history you trained on is representative of the future you will trade in. Most of the pain in quantitative finance comes from that assumption being false in ways that are invisible until they cost you money. A trend-following model learns that prices go up, right up until they don't. A mean-reversion model learns that a spread always snaps back, right up until the two companies in the spread stop being related. A risk model learns that daily moves are small, right up until volatility triples overnight. In every case the math was fine; the assumption underneath it had quietly broken. This post is about the three precise mathematical conditions that tell you whether that assumption is safe: **stationarity** (did the rules stay the same?), **autocorrelation** (how does the series remember its own past?), and **ergodicity** (can one historical path stand in for the whole range of possible futures?). By the end you will be able to look at a price series, transform it into something modelable, test whether it is trending or mean-reverting, estimate how fast a spread reverts and how much that is worth in dollars, and — most importantly — recognize the regime-switch trap where one sample of history misleads you into a position that loses money.

![Before and after columns contrasting a wandering price series with stationary returns](/imgs/blogs/stationarity-autocorrelation-math-for-quants-1.png)

The diagram above is the mental model for the whole post. On the left is a *price* — it starts somewhere, wanders off, and never settles around a fixed level; its spread grows the longer you watch. You cannot build a reliable model on that, because the very thing you would predict (the level) has no stable target. On the right are the *returns* of that same series — the day-to-day percentage changes. Those hover around a fixed average near zero with a roughly stable spread, and *that* is something you can model. The single most important manual move in all of time-series finance is the transformation from the left column to the right column: turning a wandering, non-stationary price into a well-behaved, stationary return. Everything else in this post — autocorrelation, unit roots, the Dickey-Fuller test, ergodicity — is about deciding when that transformation worked, and when even it is not enough. Let us build all of it from absolute zero.

## Foundations: the building blocks

Before we can talk about stationarity, we need to agree on what every word means. We will define each term the first time it appears, build the simplest possible version of each idea, and only then climb toward the real machinery. If you already know what a time series and a correlation are, you can skim; if you do not, you will still be able to follow every step.

### What is a "time series"?

A **time series** is just a list of numbers recorded in order, one per time step. The closing price of a stock on each trading day, the temperature at noon each day, the number of trades in each minute — each is a time series. We write it $X_1, X_2, X_3, \dots, X_t, \dots$, where the subscript $t$ ("t" for time) is the time index. $X_t$ means "the value at time $t$." If $X$ is a stock price and $t$ counts trading days, then $X_5$ is the price on day five.

The defining feature of a time series — the thing that makes it harder than ordinary statistics — is that the numbers are *ordered* and usually *related to their neighbors*. In a normal statistics class, you draw a sample of, say, the heights of 100 people, and the order does not matter; person 7's height tells you nothing about person 8's. In a time series, the order is everything. Today's price is obviously related to yesterday's price — it is usually very close to it. That relationship between a series and its own past is called **autocorrelation**, and learning to measure and use it is half of this post.

### What is the "mean" and the "variance" of a series?

The **mean**, written $\mu$ (the Greek letter "mu"), is the long-run average level — the number the series hovers around. The **variance**, written $\sigma^2$ ("sigma squared"), measures how spread out the values are around that mean; its square root $\sigma$ is the **standard deviation**, and in finance the standard deviation of returns has its own name, **volatility**. If a stock's daily returns average $\mu = 0\%$ with a volatility of $\sigma = 1.2\%$, then most days the return lands within a percent or so of zero, and big days are rare.

The crucial twist for time series is a question ordinary statistics never has to ask: *do the mean and variance stay the same over time, or do they drift?* For the heights of 100 people there is one mean and one variance, full stop. For a 20-year price series, the average price in the first year might be \$30 and in the last year \$300 — the "mean" is not a single number; it is moving. A series whose mean and variance hold still over time is the well-behaved kind we can model. A series whose mean or variance drifts is the dangerous kind. That distinction is exactly what stationarity formalizes.

### What is "correlation"?

**Correlation** measures how two quantities move together, on a scale from $-1$ to $+1$. A correlation of $+1$ means they move in perfect lockstep (when one goes up, the other always goes up by a proportional amount); $-1$ means perfect opposition (one up, the other down); $0$ means no linear relationship at all. The formula for the correlation between two variables $A$ and $B$ is their *covariance* divided by the product of their standard deviations:

$$ \rho_{A,B} = \frac{\mathrm{Cov}(A,B)}{\sigma_A \, \sigma_B}, \qquad \mathrm{Cov}(A,B) = E\big[(A-\mu_A)(B-\mu_B)\big]. $$

Here $\rho$ ("rho") is the correlation, $\mathrm{Cov}$ is the covariance (the average product of how far each variable sits from its own mean), $E[\cdot]$ is the expected value (long-run average), and $\sigma_A, \sigma_B$ are the standard deviations. Dividing by the standard deviations is what rescales the raw covariance into the clean $-1$ to $+1$ range. If you want the full build-up of covariance and correlation from scratch, the [covariance matrix post](/blog/trading/math-for-quants/covariance-matrix-linear-algebra-math-for-quants) does exactly that; here we need just the one-number summary.

### What is "autocorrelation"?

**Autocorrelation** is correlation of a series *with a delayed copy of itself*. The "auto" means "self." Instead of correlating two different variables $A$ and $B$, we correlate the series today, $X_t$, with the series some number of steps ago, $X_{t-k}$. The number of steps $k$ is called the **lag**. The lag-1 autocorrelation answers "how much does today resemble yesterday?"; the lag-5 autocorrelation answers "how much does today resemble five days ago?" This single tool — correlation against a lagged self — is how we detect momentum and mean reversion, and we will spend a whole section on it.

> A time series is a story that keeps glancing back at its own earlier pages. Autocorrelation measures how hard it is glancing.

### What is "white noise"?

**White noise** is the simplest possible time series with no memory at all: a sequence of independent draws, each with the same mean (usually zero) and the same variance, where today's value tells you absolutely nothing about tomorrow's. The autocorrelation of white noise is zero at every lag except lag zero (a series is trivially perfectly correlated with itself at lag zero). White noise is the benchmark of "no edge" — if your trading signal is statistically indistinguishable from white noise, there is nothing to trade. We will contrast it against a *random walk*, its deceptively similar but profoundly different cousin, throughout the post.

## A map of the territory

![Tree diagram mapping stationarity, dependence over time, and ergodicity under one question](/imgs/blogs/stationarity-autocorrelation-math-for-quants-5.png)

The tree above lays out the whole post under a single root question: *will a model fit on the past hold in the future?* Three branches answer it. The first branch is **stationarity** — did the statistical rules (mean, variance, memory) stay constant? — and it splits into the *strict* and *weak* flavors we will define next. The second branch is **dependence over time** — how does the series remember its own past? — and it splits into the *autocorrelation function* (and its partner the partial autocorrelation function) and the *unit-root* machinery that distinguishes a stationary series from a wandering one. The third branch is **ergodicity** — can a single long history substitute for the full distribution of possible histories? — the subtlest and most expensive condition of the three. Keep this map in mind; every section below is one branch of it.

## 1. Stationarity: the core idea

### The intuition before the formula

Picture two rivers. The first is a canal: the water level is held constant by locks, the width is fixed, and if you measure the depth today or next month you get the same answer. The second is a tidal estuary: the level rises and falls, the width changes with the tide, and "the depth here" depends entirely on *when* you ask. If you wanted to build a bridge at a fixed height, the canal is trivial — measure once and you are done — while the estuary requires you to model the whole moving system, because any single measurement is only true for that instant.

Stationarity is the canal-versus-estuary distinction for time series. A **stationary** series is one whose statistical character does not depend on *when* you look at it. Its mean is the same in 2010 and in 2025; its spread is the same; the way it relates to its own recent past is the same. A **non-stationary** series is the estuary: its mean drifts, or its variance grows, or its memory changes, so that a model calibrated on one stretch of time is calibrated to conditions that no longer hold. The reason this is the first and most important concept in the post is blunt: *statistical learning assumes the thing you are learning about holds still long enough to learn it.* If the rules change while you measure, you are fitting yesterday's market to place tomorrow's trade.

### Strict stationarity

The strongest, purest version is **strict stationarity**. A series is strictly stationary if its *entire joint probability distribution* is unchanged when you shift it in time. Formally, for any set of time points $t_1, t_2, \dots, t_n$ and any shift $h$,

$$ (X_{t_1}, X_{t_2}, \dots, X_{t_n}) \;\stackrel{d}{=}\; (X_{t_1+h}, X_{t_2+h}, \dots, X_{t_n+h}). $$

The symbol $\stackrel{d}{=}$ means "has the same distribution as." In words: take any window of the series, slide it anywhere else in time, and the probabilities of what you see are identical. Nothing about the calendar matters — only the *relative* spacing of the points. This is a beautiful, complete definition, and it is almost useless in practice, because to verify it you would need to know the entire joint distribution at every set of points, which you never do. We need a weaker condition we can actually check.

### Weak (covariance) stationarity

![Stack of the three conditions required for weak stationarity](/imgs/blogs/stationarity-autocorrelation-math-for-quants-2.png)

The workhorse definition is **weak stationarity**, also called **covariance stationarity** or **second-order stationarity**. As the stack above shows, it requires only three things to be constant over time, and nothing more:

1. **Constant mean.** $E[X_t] = \mu$ for all $t$. The series hovers around the same level forever; there is no drift, no trend.
2. **Constant variance.** $\mathrm{Var}(X_t) = \sigma^2 < \infty$ for all $t$. The spread is the same at every point in time, and it is finite (this finiteness clause quietly excludes the heavy-tailed monsters from the [tail-risk post](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants), where variance can be infinite).
3. **Autocovariance depends only on the lag.** $\mathrm{Cov}(X_t, X_{t+k}) = \gamma(k)$ depends on the gap $k$ but *not* on the absolute time $t$. The relationship between today and five days ago is the same whether "today" is in 2010 or 2025.

Here $\gamma(k)$ ("gamma of k") is called the **autocovariance function** — the covariance of the series with itself $k$ steps apart. The third condition is the subtle one: it says the *memory structure* of the series is fixed. A weakly stationary series can still have rich, persistent memory (strong autocorrelation); what it cannot do is have memory that *changes character* over time.

Why "weak"? Because it only pins down the first two **moments** — the mean and the covariance (variance is the lag-0 covariance) — and says nothing about the higher moments like skewness or kurtosis. A series could be weakly stationary while its tail behavior quietly evolves. But for almost everything quants do — fitting an autoregressive model, estimating a covariance matrix, running a regression — weak stationarity is exactly the assumption being made, often without anyone saying so out loud. When this post says "stationary" without qualification, it means weakly stationary.

#### Worked example: is this series stationary?

Suppose you have two candidate series, each measured over 10 years, and you compute their statistics in the first half and the second half. Series P (a price) has first-half mean \$40 with volatility \$8, and second-half mean \$220 with volatility \$45. Series R (its returns) has first-half mean 0.02% with volatility 1.1%, and second-half mean 0.01% with volatility 1.3%.

Series P fails immediately. Its mean jumped from \$40 to \$220 — that is the constant-mean condition shattered — and its volatility grew from \$8 to \$45, breaking the constant-variance condition too. A model that learned "the price hovers near \$40" in the first half would have been catastrophically wrong in the second half. You cannot build on series P as-is.

Series R passes the eyeball test. Its mean is ~0.01–0.02% in both halves (constant mean, near zero), and its volatility is 1.1% versus 1.3% (close enough that you would test, not reject, the constant-variance condition). The day-to-day return process looks like the same process throughout. If you had \$1,000,000 to allocate and you sized your position on series R's measured volatility of ~1.2%, that estimate would still describe the second half — your risk forecast would hold. The intuition: *prices break every stationarity condition, but the returns derived from them usually pass — which is why quants almost never model prices directly and almost always model returns.*

### Why stationarity is the precondition for everything

Stationarity is not an academic nicety; it is the load-bearing assumption under the entire toolkit. When you fit a regression of returns on a signal (the subject of the [OLS regression post](/blog/trading/math-for-quants/regression-ols-gls-regularized-math-for-quants)), you assume the relationship between signal and return is the same in your training data and your live data — that is stationarity. When you estimate a covariance matrix to size a portfolio, you assume the covariances you measured will persist — stationarity again. When the [law of large numbers](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants) promises that your sample mean converges to the truth, it is implicitly assuming the "truth" is a fixed target — stationarity once more. The cost of ignoring it is silent: your math runs, your numbers look fine, and your model is confidently describing a world that no longer exists. The rest of this post is about detecting that failure before it bills you.

## 2. Autocorrelation and the correlogram

### The intuition before the formula

A series' relationship to its own past behaves like a kind of stubbornness. A very stubborn series keeps doing what it was just doing — if it went up today, it tends to go up tomorrow. That is **positive autocorrelation**, and in markets it is the signature of **momentum**: trends persist. A contrarian series does the opposite — if it shot up today, it tends to pull back tomorrow. That is **negative autocorrelation**, the signature of **mean reversion**: deviations snap back toward a center. And a series with no stubbornness and no contrarianism at all — where today says nothing about tomorrow — is white noise, with **zero autocorrelation**, and there is no edge to extract from it.

Autocorrelation is how we put a number on that stubbornness, at each time lag.

### The formal definition

The **autocorrelation function**, or **ACF**, at lag $k$ is the autocovariance at lag $k$ rescaled by the variance so that it lands on the clean $-1$ to $+1$ scale:

$$ \rho(k) = \frac{\gamma(k)}{\gamma(0)} = \frac{\mathrm{Cov}(X_t, X_{t-k})}{\mathrm{Var}(X_t)}. $$

Here $\rho(k)$ is the autocorrelation at lag $k$, $\gamma(k)$ is the autocovariance at lag $k$, and $\gamma(0)$ is the autocovariance at lag zero — which is just the variance. By construction $\rho(0) = 1$ (a series is perfectly correlated with itself at zero lag), and $\rho(k)$ ranges from $-1$ to $+1$ for every other lag. In practice we estimate it from data with the **sample autocorrelation**:

$$ \hat\rho(k) = \frac{\sum_{t=k+1}^{n} (X_t - \bar X)(X_{t-k} - \bar X)}{\sum_{t=1}^{n} (X_t - \bar X)^2}, $$

where $\bar X$ is the sample mean, $n$ is the number of observations, the numerator sums the products of paired deviations $k$ apart, and the denominator is the total sum of squared deviations. The hat on $\hat\rho$ means "estimated from the sample."

#### Worked example: the lag-1 autocorrelation of a return series by hand

Let us compute the lag-1 autocorrelation of a short daily return series with five observations (in percent): $+2, -1, +3, -2, +1$. We will do every step.

First, the mean: $\bar X = (2 - 1 + 3 - 2 + 1)/5 = 3/5 = +0.6\%$.

Next, the deviations from the mean: $2 - 0.6 = +1.4$; $-1 - 0.6 = -1.6$; $3 - 0.6 = +2.4$; $-2 - 0.6 = -2.6$; $1 - 0.6 = +0.4$.

The denominator (sum of squared deviations): $1.4^2 + 1.6^2 + 2.4^2 + 2.6^2 + 0.4^2 = 1.96 + 2.56 + 5.76 + 6.76 + 0.16 = 17.20$.

The numerator (sum of products of consecutive deviations, lag 1): pair each deviation with the next one. $(1.4)(-1.6) + (-1.6)(2.4) + (2.4)(-2.6) + (-2.6)(0.4) = -2.24 - 3.84 - 6.24 - 1.04 = -13.36$.

So $\hat\rho(1) = -13.36 / 17.20 = -0.78$.

A lag-1 autocorrelation of $-0.78$ is strongly negative. This series is *mean-reverting*: an up day tends to be followed by a down day and vice versa, which you can see in the raw data ($+2, -1, +3, -2, +1$ — it keeps flipping sign). If these were the daily returns of a tradable spread and the pattern were real (five points is far too few to trust, but bear with the arithmetic), you would *fade* each move: after an up day, lean short; after a down day, lean long. On a \$100,000 position with a typical 2% daily swing, reliably capturing even a third of the reversal would be worth roughly \$667 per signal. The intuition: *the sign of the lag-1 autocorrelation tells you whether to chase the last move (positive, momentum) or fade it (negative, mean reversion); the magnitude tells you how strong the tendency is.*

### The correlogram and the matrix of behaviors

![Matrix mapping autocorrelation sign to market behavior and trade idea](/imgs/blogs/stationarity-autocorrelation-math-for-quants-3.png)

A **correlogram** is simply a bar chart of $\hat\rho(k)$ against the lag $k$ — one bar per lag, showing the autocorrelation at each. It is the single most useful diagnostic plot in time-series analysis, because its *shape* tells you what kind of process you are holding. The matrix above summarizes the verdict the lag-1 bar delivers, and it is worth committing to memory:

- A **positive** ACF (say $+0.30$) means **momentum**: the next move tends to be in the same direction, so the trade idea is to follow the trend.
- A **near-zero** ACF means **white noise**: the next move is unpredictable from the last, so there is no edge in the lagged value alone.
- A **negative** ACF (say $-0.30$) means **mean reversion**: the next move tends to reverse, so the trade idea is to fade the deviation.

Real series rarely show a single clean lag; they show a *decay pattern*. A momentum series might show positive autocorrelation that fades slowly across many lags (memory that lingers for weeks). A mean-reverting series often shows a sharp negative spike at lag 1 that vanishes by lag 2. A trending price (non-stationary) shows autocorrelation that stays near $+1$ for dozens of lags and decays painfully slowly — that slow decay is itself a red flag for non-stationarity, a foreshadowing of the unit root we will meet next.

### Statistical significance: when is a bar real?

A sample autocorrelation is an estimate, so it has noise. For a true white-noise series of length $n$, each $\hat\rho(k)$ is roughly Normally distributed with standard error $1/\sqrt{n}$. The standard 95% significance band is therefore $\pm 1.96/\sqrt{n}$. With $n = 250$ trading days (one year), the band is $\pm 1.96/\sqrt{250} \approx \pm 0.124$. So a measured autocorrelation of $+0.05$ over one year is *inside* the band — indistinguishable from noise, no edge. A measured $-0.30$ is well outside it — a real, tradable signal. This is the same standard-error logic from the [law of large numbers post](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants): more data shrinks the band like $1/\sqrt{n}$, so a faint autocorrelation that is invisible in one year of data might become significant in five. Always draw the significance band on your correlogram; a bar that does not clear it is not a signal, it is a mirage.

### Partial autocorrelation (PACF)

The plain ACF has a blind spot. Suppose a series has strong momentum, so today correlates with yesterday, and yesterday correlated with the day before. Then today will *automatically* correlate with the day before too — not because of any direct two-day link, but purely *through* yesterday as a middleman. The lag-2 ACF picks up this indirect, laundered-through-lag-1 correlation and reports it as if it were real two-day memory.

The **partial autocorrelation function**, or **PACF**, fixes this. The partial autocorrelation at lag $k$ is the correlation between $X_t$ and $X_{t-k}$ *after statistically removing the influence of all the lags in between* ($X_{t-1}, \dots, X_{t-k+1}$). It is the *direct* link at lag $k$, with the middlemen subtracted out. Practitioners read the ACF and PACF together to identify a model: a process with one direct lag of memory (a first-order autoregressive process, "AR(1)") shows a PACF that spikes at lag 1 and is zero afterward, while its ACF decays geometrically. A moving-average process shows the mirror image. The deep point is that the ACF can be fooled by indirect chains of dependence, and the PACF is the tool that strips them away to reveal the true order of the memory. When you size how many lagged terms a forecasting model needs, the PACF is what tells you when to stop adding them.

## 3. Unit roots and the ADF test

### The intuition before the formula

![Before and after columns contrasting white noise with a random walk](/imgs/blogs/stationarity-autocorrelation-math-for-quants-6.png)

Two series can look almost identical on a chart and yet be opposites underneath. The before-and-after above shows the pair. On the left is **white noise**: each day is a fresh independent shock, the level is anchored at zero, and yesterday's shock is forgotten by today. On the right is a **random walk**: each day you take yesterday's level and *add* a fresh shock, so the level wanders off and — here is the key — *every past shock is permanent*, baked into the level forever. The random walk has no anchor; it can drift arbitrarily far from where it started, and its variance grows without bound the longer you watch.

The reason this matters is that a stock price behaves far more like the random walk than like white noise. A price is the running *sum* of all its past returns, and a running sum of shocks is exactly a random walk. That is the mathematical reason prices are non-stationary: they accumulate, so they have no fixed level and their variance grows. The technical name for the property that makes a random walk non-stationary is a **unit root**.

### What is a unit root?

Consider the simplest dynamic model, the first-order autoregression:

$$ X_t = \phi \, X_{t-1} + \varepsilon_t, $$

where $\phi$ ("phi") is the persistence coefficient — how much of yesterday carries into today — and $\varepsilon_t$ ("epsilon") is a fresh white-noise shock at time $t$. The behavior of the series hinges entirely on $\phi$:

- If $|\phi| < 1$ (say $\phi = 0.5$), each shock decays away over time — its influence halves, then halves again — so the series is **stationary** and mean-reverting toward zero. The smaller $\phi$, the faster the reversion.
- If $\phi = 1$ exactly, the model becomes $X_t = X_{t-1} + \varepsilon_t$ — a **random walk**. Shocks never decay; they are summed forever. This is the **unit root** case ("unit" because the coefficient equals one). The series is non-stationary.
- If $|\phi| > 1$, shocks *amplify* and the series explodes — rare in finance outside of bubbles.

So the entire question "is this series stationary?" reduces, in the AR(1) world, to "is $\phi$ less than 1, or is it equal to 1?" A unit root ($\phi = 1$) is the boundary between a mean-reverting, modelable series and a wandering, non-stationary one. The whole game is testing which side of that boundary your data is on.

### The augmented Dickey-Fuller test

The **augmented Dickey-Fuller (ADF) test** is the standard hypothesis test for a unit root. The idea is direct: rewrite the AR model so the thing being tested is the *deviation* of $\phi$ from 1. Subtract $X_{t-1}$ from both sides of $X_t = \phi X_{t-1} + \varepsilon_t$:

$$ \Delta X_t = (\phi - 1) X_{t-1} + \varepsilon_t = \delta \, X_{t-1} + \varepsilon_t, $$

where $\Delta X_t = X_t - X_{t-1}$ is the change from yesterday to today (the *first difference*), and $\delta = \phi - 1$. Now the unit-root case $\phi = 1$ is simply $\delta = 0$. The test sets up:

- **Null hypothesis** $H_0$: $\delta = 0$ (there *is* a unit root — the series is non-stationary, a random walk).
- **Alternative** $H_1$: $\delta < 0$ (no unit root — the series is stationary, mean-reverting).

The "augmented" part adds extra lagged difference terms ($\Delta X_{t-1}, \Delta X_{t-2}, \dots$) to the regression to soak up any short-term autocorrelation in the shocks, so the test isn't fooled by a series that has both a unit root *and* extra memory. You run the regression, compute the **ADF test statistic** (the coefficient $\hat\delta$ divided by its standard error), and compare it to special critical values. A subtlety that trips everyone up: because the series under the null is non-stationary, the test statistic does *not* follow the usual t-distribution — it follows the Dickey-Fuller distribution, whose critical values are *more negative* than the normal ones (roughly $-3.43$ at 5% for a typical setup, versus $-1.96$ for an ordinary t-test). The rule is:

- If the ADF statistic is **more negative** than the critical value (equivalently, p-value < 0.05), you **reject the null** and conclude the series is **stationary**.
- If the ADF statistic is **less negative** than the critical value (p-value > 0.05), you **fail to reject** and treat the series as **non-stationary** (unit root present).

A common mnemonic: *very negative ADF = stationary; near zero = unit root.* Note the asymmetry of failing to reject — you never "prove" a unit root, you just fail to rule it out, exactly as in any [hypothesis test](/blog/trading/math-for-quants/hypothesis-testing-pvalues-math-for-quants).

#### Worked example: ADF on a price versus its returns

You have three years of daily data (about 750 observations) on a stock and you want to know whether to model the price or the returns. You run the ADF test twice.

**On the price level.** The price drifted from \$50 to \$95 over the three years, wandering the whole way. The ADF regression gives $\hat\delta = -0.004$ with a standard error of $0.006$, so the test statistic is $-0.004 / 0.006 = -0.67$. The 5% critical value (with a constant and trend) is about $-3.43$. Since $-0.67$ is *much less negative* than $-3.43$, you **fail to reject** the null. The p-value comes out around $0.85$. **Verdict: the price has a unit root — it is non-stationary.** A mean-reversion model fit on this price would be fitting a random walk and would lose money chasing a "fair value" that does not exist.

**On the daily returns.** You difference the price into returns. The returns hover around 0% with no drift. The ADF regression now gives $\hat\delta = -0.95$ with a standard error of $0.04$, so the statistic is $-0.95 / 0.04 = -23.75$. That is *far more negative* than $-3.43$, with a p-value below $0.001$. **Verdict: the returns are emphatically stationary.** 

The decision is clear: model the returns, not the price. If you had \$500,000 to deploy and you built a forecasting model, doing it on the (stationary) returns means your fitted relationship has a chance of holding out of sample; doing it on the (non-stationary) price means you are almost guaranteed to find spurious patterns that evaporate. The intuition: *the ADF test is the formal referee that tells you whether you are holding a wandering random walk (don't model the level) or a stationary, mean-reverting series (safe to model).*

### The pipeline: from raw price to a stationarity verdict

![Pipeline from raw price through differencing and the ADF test to a safe model](/imgs/blogs/stationarity-autocorrelation-math-for-quants-4.png)

The pipeline above is the standard workflow, and it is worth running mechanically every time you start a new series. Start with the **raw price**, which you should assume is non-stationary until proven otherwise. **Difference it** (or take returns) to strip out the wandering level. Feed the result into the **ADF test**. Read off the **verdict** from the p-value. Only if the verdict is "stationary" do you proceed to **fit a model** — and even then you keep the rest of this post's warnings (especially ergodicity) in your back pocket. Skipping straight from raw price to model, without the difference-and-test steps, is the single most common rookie error in quantitative finance, and it produces backtests that look spectacular and live results that look like a coin flip.

## 4. Differencing and returns: making a series modelable

### Why prices are non-stationary

A price is a *cumulative* quantity. Today's price equals yesterday's price plus today's return: $P_t = P_{t-1}(1 + r_t)$, or in log terms $p_t = p_{t-1} + r_t$ where $p_t = \log P_t$. Iterating, today's log price is the *sum of every return that ever happened*: $p_t = p_0 + r_1 + r_2 + \dots + r_t$. A running sum of shocks is a random walk, and a random walk is non-stationary — its level wanders and its variance grows linearly with time. That is not a quirk of any particular stock; it is structural. *Every* price series built by accumulating returns inherits a unit root. This is why the left column of the very first figure — the wandering, fanning-out price — is the rule, not the exception.

### Differencing: the operation that removes a unit root

**Differencing** means replacing the series with its period-to-period changes: $\Delta X_t = X_t - X_{t-1}$. The magic is that differencing a random walk *removes* the unit root and hands you back stationary white noise. Watch: if $P_t = P_{t-1} + \varepsilon_t$ (a random walk), then $\Delta P_t = P_t - P_{t-1} = \varepsilon_t$ — which is exactly the stationary shock. The cumulative wandering cancels out, leaving only the fresh increment. In finance the difference of the log price *is* the return ($\Delta p_t = r_t$), so "take returns" and "first-difference the log price" are the same operation. A series that becomes stationary after one differencing is called **integrated of order one**, written $I(1)$; a series already stationary is $I(0)$. Most prices are $I(1)$; most returns are $I(0)$. Occasionally a series needs differencing twice (it is $I(2)$), but in equity-style finance one difference almost always does it.

#### Worked example: turning a wandering price into stationary returns

You have five days of closing prices: \$100, \$103, \$101, \$106, \$104. As a *level*, this is non-stationary — it has no fixed mean; it just wanders.

Compute the daily returns (the first differences, as percentages):
- Day 1→2: $(103 - 100)/100 = +3.00\%$
- Day 2→3: $(101 - 103)/103 = -1.94\%$
- Day 3→4: $(106 - 101)/101 = +4.95\%$
- Day 4→5: $(104 - 106)/106 = -1.89\%$

The return series is $+3.00, -1.94, +4.95, -1.89$. Notice it now hovers around a small mean (here $(3.00 - 1.94 + 4.95 - 1.89)/4 = +1.03\%$) rather than wandering ever upward, and — crucially — it is the same *kind* of object on every day, a percentage change with a stable spread. The prices climbed from \$100 to \$104, an upward drift; the returns reveal that drift as a small positive average per day, with most of the action being noise around it. If you held a \$100,000 position, your day-4 gain of $+4.95\%$ was worth \$4,950 and your day-5 loss of $-1.89\%$ cost \$1,890 — and these dollar swings are *comparable across days* in a way the raw prices are not. The intuition: *differencing converts an un-modelable wandering level into a stationary stream of changes you can actually forecast, size, and risk-manage.*

### The cost of over-differencing

Differencing is not free. If you difference a series that was *already* stationary, you introduce spurious negative autocorrelation and inflate the variance — you damage a perfectly good signal. The discipline is: difference exactly as many times as the ADF test says you need, and no more. Take returns of a price (one difference): good. Take "returns of returns" because you can: bad, you have just added noise. The ADF test is your stopping rule — once it says "stationary," stop differencing.

### Cointegration: the exception worth knowing

There is a beautiful exception where two *individually* non-stationary series combine into a stationary one. Two stocks, A and B, can each be wandering random walks (each $I(1)$, non-stationary), yet a particular linear combination of them — say $A - 1.5 B$ — can be *stationary*. When that happens, the two series are **cointegrated**, and that stationary combination is a **spread** that mean-reverts. This is the mathematical foundation of pairs trading and statistical arbitrage: you cannot model either stock's price (both have unit roots), but you *can* model the spread, because it is the stationary $I(0)$ object hiding inside two $I(1)$ ones. The next section's mean-reverting spread is exactly such a cointegrated combination. The lesson: non-stationary ingredients can sometimes be cooked into a stationary, tradable dish — but you must test for it (with an ADF test on the spread), never assume it.

## 5. Mean reversion, half-life, and the dollar edge

### The intuition before the formula

A mean-reverting series is like a spring. Stretch it away from its rest position and it pulls back; the further you stretch, the harder it pulls. In markets, a cointegrated spread is the spring: when it is unusually wide, forces (arbitrageurs, fundamentals) tend to pull it back toward its fair value; when it is unusually narrow, it tends to widen back out. The *edge* in trading mean reversion is fading the stretch — selling the spread when it is wide, buying it when it is narrow — and waiting for the spring to pull it home. The single most useful number for sizing that trade is the **half-life**: how long, on average, until the spread closes *half* of its current deviation from fair value.

### From the AR(1) coefficient to the half-life

Recall the stationary AR(1): $X_t = \phi X_{t-1} + \varepsilon_t$ with $|\phi| < 1$, where the series mean-reverts toward zero. After one step, a deviation of size $d$ shrinks to $\phi d$ (in expectation); after $h$ steps it shrinks to $\phi^h d$. The half-life is the number of steps $h$ at which the deviation has shrunk to half, i.e. $\phi^h = 0.5$. Solving with logarithms:

$$ h_{1/2} = \frac{\ln(0.5)}{\ln \phi} = \frac{-0.693}{\ln \phi}. $$

Here $h_{1/2}$ is the half-life in time steps, $\ln$ is the natural logarithm, and $\phi$ is the AR(1) persistence coefficient (which is also the lag-1 autocorrelation for an AR(1) process). The closer $\phi$ is to 1, the slower the reversion and the longer the half-life; the closer to 0, the faster. A half-life turns the abstract coefficient $\phi$ into a concrete planning number: *how many days do I expect to hold this trade before half my edge is realized?*

#### Worked example: the half-life and dollar edge of a mean-reverting spread

![Timeline of a mean-reverting spread decaying over ten days](/imgs/blogs/stationarity-autocorrelation-math-for-quants-7.png)

You are trading a cointegrated pair whose spread (currently \$2.00 above its fair value of \$0) has an estimated lag-1 autocorrelation of $\phi = 0.87$. The timeline above traces how the deviation decays.

**Step 1 — the half-life.** $h_{1/2} = -0.693 / \ln(0.87) = -0.693 / (-0.139) = 4.98 \approx 5$ days. So you expect the spread to close half its gap roughly every 5 days. That matches the timeline: \$2.00 at entry, about \$1.00 (half) by day 5.

**Step 2 — the trade.** The spread is \$2.00 wide, which is unusually large. You *fade* it: short the spread (sell the rich leg, buy the cheap leg) at \$2.00, betting it reverts toward \$0. You put on 10,000 units of the spread.

**Step 3 — the dollar edge.** If the spread reverts from \$2.00 to its fair value of \$0, you capture \$2.00 per unit × 10,000 units = \$20,000 gross. But you should not wait for the full reversion — the last cents take forever (it is exponential decay; it asymptotes, never quite arriving). Suppose you exit at \$0.25 (around day 10 on the timeline). You captured \$2.00 − \$0.25 = \$1.75 per unit × 10,000 = **\$17,500** over roughly 10 trading days.

**Step 4 — annualize the edge.** If you can recycle this capital into a fresh such trade every ~10 days, that is roughly 25 trades per year. At \$17,500 per trade (before costs), the strategy's gross annual edge is on the order of \$437,500 on the capital tied up. After transaction costs, slippage, and the trades that *don't* revert (the spring that breaks — see the next section), the realized number is much smaller, but the half-life is what told you the trade's natural holding period and let you compute the edge at all.

The intuition: *the half-life converts the autocorrelation coefficient into a holding period and a dollar figure — it tells you both how long to wait and how much the wait is worth.*

### What this costs and when it breaks

The half-life calculation assumes the AR(1) coefficient $\phi$ is *stable* — that the spring keeps its stiffness. That is a stationarity assumption on the spread itself. The expensive failure mode is a spread whose mean-reversion *stops*: the two cointegrated stocks decouple (one company is acquired, changes business, blows up), $\phi$ creeps toward 1, the spread becomes a random walk, and your "temporary" deviation keeps widening instead of reverting. Your model says "wait, it always comes back," and it never does. This is not a flaw in the half-life math; it is a regime change in the underlying series — which is exactly the bridge to the last and deepest concept in the post.

## 6. Ergodicity: when one path equals the whole distribution

### The intuition before the formula

Here is the question that ergodicity answers, and it is subtler than anything so far. When you estimate a strategy's average return from *one* historical path — the single sequence of prices that actually happened — you are implicitly claiming that this one path's *time average* (average over the days you observed) equals the *ensemble average* (the average over all the alternative histories that *could* have happened). **Ergodicity** is precisely the condition under which those two averages are equal. When a series is ergodic, one long sample is a faithful proxy for the whole distribution, and the law of large numbers applies along the single path you have. When it is *not* ergodic, your one historical sample can be systematically, dangerously misleading — and no amount of data along that single path will save you, because the path itself is unrepresentative.

The classic illustration is the difference between two ways of averaging. Take a casino game and 1,000 gamblers each playing it once: the *ensemble average* is the average outcome across those 1,000 people on that one day. Now take *one* gambler playing the same game 1,000 times in a row: the *time average* is that single person's average outcome over their long sequence. For a simple coin-flip bet these two are equal — the game is ergodic. But for many real processes, especially ones with *multiplicative* dynamics or *absorbing states* (like going bankrupt), the time average and the ensemble average diverge, sometimes spectacularly. The single gambler can go broke and stop, an outcome the ensemble average smooths over by averaging in all the lucky people who never went broke.

### The formal condition

A stationary process is **ergodic** (specifically, ergodic for the mean) if its time average converges to the ensemble mean as the sample grows:

$$ \frac{1}{T}\sum_{t=1}^{T} X_t \;\xrightarrow{\;T\to\infty\;}\; E[X_t] = \mu. $$

The left side is the time average over $T$ observations of one path; the right side is the ensemble expectation $\mu$. The arrow means "converges to as $T$ grows." A sufficient condition (for a weakly stationary series) is that the autocorrelation $\rho(k)$ dies away as the lag $k$ grows — that the series eventually *forgets* its distant past, so that far-apart observations behave almost independently and the single long path samples enough of the distribution. Notice the dependence on stationarity: ergodicity is a stronger property *built on top of* stationarity. A series can even be stationary and still fail to be ergodic if, for example, it has a persistent random level that never gets re-sampled along one path. For practical purposes, the headline is: **stationarity says the rules don't change; ergodicity says one long sample is enough to learn those rules.** You need both for a single backtest to be trustworthy.

#### Worked example: the regime-switch that makes the time average lie

Consider a strategy whose true return depends on an unobserved market **regime**. In a *calm* regime (which prevails 90% of the time over long horizons) the strategy earns a steady $+0.5\%$ per day. In a *crisis* regime (10% of the time over long horizons) it loses $-8\%$ per day, because the relationship it exploits inverts violently. The true *ensemble* expected daily return is:

$$ 0.90 \times 0.5\% + 0.10 \times (-8\%) = 0.45\% - 0.80\% = -0.35\% \text{ per day.} $$

The strategy is, on the full ensemble, a *loser*: $-0.35\%$ per day is roughly $-60\%$ over a year of 250 days. But now suppose your *historical sample* happens to be a three-year stretch that was **entirely calm** — no crisis ever showed up in your data. Your measured time average is $+0.5\%$ per day, a beautiful Sharpe ratio, a backtest that screams "deploy me." Your one path's time average ($+0.5\%$) does *not* equal the ensemble average ($-0.35\%$). The process is non-ergodic over your sample because your sample never visited the crisis regime that dominates the true expectation.

Now the dollar cost of the sizing decision. Fooled by the $+0.5\%$ backtest, you size the position to risk \$1,000,000 of capital and you lever it up, because the measured volatility (calm-only) looked tame. Six months in, the crisis regime arrives — exactly the regime your data never showed you. At $-8\%$ per day, a levered \$1,000,000 book can lose **\$80,000 in a single day**, and a multi-day crisis can take **half the book or more** before you cut it. The backtest was not "wrong" arithmetically — it faithfully averaged the days it saw. It was *non-ergodic*: the time average over the calm sample was a liar about the ensemble that includes crises. The intuition: *when a series switches regimes, one historical path can show you a strategy's average from a flattering angle that the full distribution does not support — and sizing on that average is how a "great backtest" becomes a blowup.*

### How to defend against non-ergodicity

You cannot test for ergodicity the way you test for a unit root — there is no clean p-value, because the failure lives in the regimes your data *didn't sample*. The defenses are structural rather than statistical. First, *demand long, regime-diverse samples*: a backtest that never saw a crash has not earned a crash-resistant Sharpe. Second, *stress-test against regimes absent from your data*: simulate the 2008 or 2020 conditions even if your sample missed them. Third, *size for the ensemble, not the sample*: assume the calm you measured is not the whole story and keep leverage modest. Fourth, *respect path-dependence*: a strategy that can go to zero (bankruptcy, a blown stop, a margin call) is non-ergodic by construction, because the time average of a path that hits zero and stops is nothing like the ensemble average that includes the survivors. The single most important habit is to never confuse "this is the average of what happened to me" with "this is the average of what could happen" — those are the time average and the ensemble average, and ergodicity is the fragile bridge between them.

## Common misconceptions

**"If the chart looks flat and ranges sideways, it's stationary."** Eyeballing a chart is not a test. A series can range sideways for years and still have a unit root (a random walk genuinely can wander sideways for a long stretch by chance), and a series with an obvious trend can still have stationary *deviations* around that trend. Stationarity is a statement about the *generating process* — mean, variance, and autocovariance over time — not about how a single realization happens to look. Run the ADF test; don't trust your eyes.

**"Stationary means constant or predictable."** A stationary series is not flat and not predictable in the sense of "I know the next value." White noise is perfectly stationary and *completely* unpredictable. Stationarity means the *statistical character* — the distribution from which each value is drawn — holds still, not that the values themselves do. A stationary series can be wild and jumpy; it just has to be wild and jumpy in the *same way* throughout.

**"A high R-squared means I found a real relationship between two price series."** This is the spurious-regression trap. Two *independent* non-stationary series (two unrelated random walks) will, alarmingly often, show a high R-squared and a "significant" t-statistic when you regress one on the other — purely because both happen to be trending. The relationship is an artifact of shared non-stationarity, not a real link. The fix is to test the series for unit roots first and either difference them or test for genuine cointegration. A regression on raw, non-stationary prices is one of the most reliable ways to fool yourself.

**"Negative autocorrelation in my equity curve means a great mean-reverting strategy."** Be careful what you measure. Negative autocorrelation in a tradable *spread* can be a real edge. But negative autocorrelation in your own *equity curve* often just reflects the mechanics of your sizing or rebalancing (you cut winners, add to losers), not a market edge — and it can be there even in a strategy with zero true alpha. Always ask whether the autocorrelation is in the market or in your bookkeeping.

**"More data always makes my estimate more reliable."** Only if the series is stationary *and* ergodic. Pile up ten years of data on a non-stationary series and you have ten years of a moving target — your "estimate" is an average across genuinely different regimes that may describe none of them. And on a non-ergodic series, more data *along the one path you have* never reveals the regimes that path never visited. Quantity of data does not cure non-stationarity or non-ergodicity; it can disguise them as confidence.

**"The ADF test rejecting the null proves the series is stationary forever."** The ADF test is a snapshot. It tells you the series looked stationary over the sample you tested. Stationarity can break in the future — a relationship that was mean-reverting for five years can decouple in the sixth. The test is a necessary check, not a permanent guarantee, and a stationarity that held in-sample is exactly what regime change destroys out-of-sample.

## How it shows up in real markets

### 1. The dot-com pairs that stopped reverting (2000–2001)

Statistical-arbitrage desks in the late 1990s built fortunes on cointegrated pairs — two similar stocks whose spread reliably mean-reverted. Many of these pairs were tech names that had traded together for years, with spreads that snapped back like clockwork and short half-lives that made the carrying cost trivial. When the dot-com bubble burst in 2000–2001, a great many of those cointegration relationships *broke*: companies that had been statistical twins diverged permanently as some survived and others went to zero. The spread that "always reverts" became a random walk, its ADF test (if anyone re-ran it) flipping from stationary to unit-root. Desks that had sized on the historical half-life — assuming the spread's $\phi$ was stable — found their deviations widening instead of closing, and the losses compounded because the strategy's whole logic was "add as it gets wider." The lesson is the bridge from Section 5 to Section 6: a half-life is only as trustworthy as the stationarity of the spread it is computed on.

### 2. LTCM and the convergence trades that diverged (1998)

Long-Term Capital Management ran enormous convergence trades — bets that spreads between related bonds (on-the-run versus off-the-run Treasuries, sovereign yield gaps) would mean-revert. The trades were grounded in beautifully estimated stationary relationships and modest historical volatilities. The fatal assumption was ergodicity: the fund sized as if its calm historical sample represented the full distribution of outcomes. When Russia defaulted in August 1998 and a global flight to liquidity hit, the world switched to a regime that LTCM's data had barely sampled — spreads that "always converged" *diverged* violently and in correlated fashion across every position at once. The time average over LTCM's calm history was a liar about the ensemble that included a liquidity crisis. The fund lost roughly \$4.6 billion in months. It is the canonical real-world cost of confusing the time average with the ensemble average.

### 3. The 2007 quant quake and crowded mean reversion

In early August 2007, a swarm of equity statistical-arbitrage funds — all running similar mean-reversion signals on similar stationary-looking factors — suffered enormous, simultaneous losses over a few days, even though the broad market barely moved. The mechanism was a crowded de-leveraging: when one large fund began liquidating, it pushed the very spreads everyone else was holding the wrong way, and the mean reversion that the backtests promised reversed into momentum precisely when capital was being pulled. The stationary relationships were real in normal times, but the *regime* of forced, correlated liquidation was outside the sample on which they were measured — another ergodicity failure dressed as a stationarity failure. Funds that survived had sized for the regime their data hadn't shown them.

### 4. Why central banks made inflation look stationary — until it wasn't

For roughly three decades into the early 2020s, developed-market inflation behaved like a stationary series anchored near a 2% target — its mean held, its variance was contained, and models that assumed reversion to target worked beautifully. Bond pricing, mortgage rates, and pension liabilities were all built on that implicit stationarity. When inflation spiked to multi-decade highs in 2021–2022, the anchored, stationary regime broke: the mean drifted up, the variance exploded, and models calibrated to the stationary era mispriced bonds badly — the U.S. 10-year yield roughly tripled off its 2020 lows and long-duration bonds fell sharply. A generation of models had quietly assumed a stationarity that was a policy *choice*, not a law of nature, and the assumption expired.

### 5. The slow-decay correlogram that flagged a trend (everyday use)

This one is mundane and constant rather than a famous blowup, but it is the most common practical use of the tools. Any quant onboarding a new price series runs the correlogram first. A series whose autocorrelation stays near $+1$ across dozens of lags and decays painfully slowly is showing the signature of a unit root before any formal test confirms it — that slow decay *is* non-stationarity, visible to the eye. The disciplined response is exactly the Section 3 pipeline: don't model the level, difference it, re-run the correlogram (now the autocorrelation should collapse toward zero, confirming the difference is stationary), then run the ADF test to be sure. This single workflow, run thousands of times a day across the industry, is the practical heart of everything in this post. It is described in detail in the [market-data exploration and biases post](/blog/trading/quantitative-finance/market-data-eda-biases-quant-research), which walks through the data-cleaning and diagnostic steps that precede any model.

### 6. Returns are stationary, but their *volatility* is not (the GARCH reality)

A clean caveat that catches beginners: while returns pass the ADF test for stationarity in their *level* (mean near zero), their *variance* is famously not constant — volatility comes in clusters, with calm stretches and turbulent stretches. This is **volatility clustering**, and it means returns are stationary in the weak (mean) sense but have a time-varying second moment that simple weak stationarity does not capture. The whole GARCH family of models exists to handle exactly this: they model the *conditional* variance as itself mean-reverting around a long-run level. So the honest statement is that returns are "stationary enough" to model their mean, but you must model their volatility separately — a reminder that weak stationarity only pins down the first two moments, and the higher-moment behavior can still evolve. This volatility-of-the-signal question feeds directly into how you build and size a tradable signal, the subject of the [alpha-signal construction post](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research).

### 7. The survivorship-biased backtest as a non-ergodicity machine

A backtest run on the current index constituents — the companies that *survived* to today — is a non-ergodicity trap by construction. The dead companies (the ones that went to zero, got delisted, or were acquired in distress) are the crisis-regime outcomes that your sample silently dropped. Their absence makes the time average over the survivors look far better than the ensemble average that would include the failures. This is why a strategy backtested on today's S&P 500 members back to 1990 can show a glittering Sharpe that collapses when run on the *historical* membership including the casualties. The fix is point-in-time data that includes the dead names — restoring the regime (corporate death) that survivorship bias erased. It is the cleanest everyday example of the time-average-versus-ensemble-average distinction costing real money, and it lurks in more backtests than practitioners like to admit.

## When this matters to you

If you ever build a trading model, forecast a financial series, or even just read a backtest someone is selling you, these three ideas are the questions you should ask first, before you look at a single performance number. *Did the rules stay the same over the sample?* — that is stationarity, and the ADF test is how you check it. *How does the series remember its past?* — that is autocorrelation, and the correlogram tells you momentum from mean reversion from noise. *Is this one historical path representative of the futures that could happen?* — that is ergodicity, and it is the question that no test can fully answer and that costs the most when ignored. A backtest with a beautiful Sharpe ratio that fails the stationarity check, or that ran on a single calm regime, is not an opportunity; it is a trap with good lighting.

The deeper habit these tools build is humility about history. The math of this entire field assumes the past resembles the future, and these three concepts are the precise inventory of the ways that assumption can be false: the rules drifted (non-stationary), the memory was misread (autocorrelation misdiagnosed), or the one sample you had was unrepresentative of the whole (non-ergodic). The quants who last are not the ones with the cleverest models; they are the ones who size as if their stationarity might break and their sample might be unrepresentative — because eventually, for everyone, it is.

This is educational material, not investment advice; nothing here is a recommendation to buy, sell, or trade anything. Markets change regimes precisely when models are most confident, which is the whole point of the ergodicity section.

For further reading, the natural next steps on this blog are the [probability distributions for markets post](/blog/trading/math-for-quants/probability-distributions-for-markets-math-for-quants), which gives you the distributional vocabulary underneath these moments; the [law of large numbers and central limit theorem post](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants), which formalizes when sample averages converge (and thus when ergodicity does its job); the [market-data exploration and biases post](/blog/trading/quantitative-finance/market-data-eda-biases-quant-research) for the practical diagnostic workflow; and the [building an alpha signal post](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) for turning a stationary, autocorrelated series into a sized, tradable bet. Beyond this blog, the primary sources worth knowing by name are Dickey and Fuller's original 1979 unit-root paper, Engle and Granger's 1987 work on cointegration (which won Granger a Nobel), and Ole Peters' modern work on ergodicity economics, which makes the time-average-versus-ensemble-average distinction the center of how it thinks about risk.
