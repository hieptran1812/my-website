---
title: "Regression deep-dive: from OLS to GLS to regularized"
date: "2026-06-15"
description: "How the workhorse formula of quant finance turns noisy returns into a beta, why its standard errors quietly lie when markets misbehave, and how generalized least squares and ridge or lasso penalties rebuild it on firmer ground -- built from zero with worked dollar examples."
tags: ["regression", "ols", "gls", "ridge-regression", "lasso", "elastic-net", "heteroskedasticity", "newey-west", "multicollinearity", "factor-models", "r-squared", "quantitative-finance"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** -- regression draws the best straight line through a cloud of returns, and almost every mistake a quant makes with it is about *trusting that line too much*.
>
> - **OLS** picks the coefficients $\hat\beta = (X^\top X)^{-1}X^\top y$ that minimize squared error. The slope on the market is a stock's *beta*, and it tells you exactly how much index to short to hedge a position.
> - **$R^2$ is a trap.** It measures fit on the data you already saw, not whether the relationship is real or will hold tomorrow. A regression on two trending series can show $R^2 = 0.95$ and predict nothing.
> - **Naive standard errors lie** when the noise clusters (heteroskedasticity) or persists in time (autocorrelation). Robust *Newey-West* errors widen the bars and can turn a t-stat of 3.0 into a 1.7 -- from "publish it" to "it was luck".
> - **When factors look alike** (multicollinearity), OLS coefficients explode to plus and minus hundreds and flip sign on a rounding error. *Ridge* and *lasso* penalties shrink them back to sane values that actually make money out of sample.
> - **The one number to remember**: a stock with $\beta = 1.3$ against the index means you short \$1,300,000 of the index to hedge a \$1,000,000 long -- a regression slope doing a real dollar job.

## Why a single straight line runs a trading desk

Here is a confession that surprises people outside finance: an enormous fraction of what a quantitative trading desk does, stripped of jargon, is *fitting straight lines through clouds of numbers*. How much does this stock move when the market moves? Straight line. How much of a bond's daily change comes from the level of interest rates versus the slope of the yield curve? Two straight lines fit at once. Does this brand-new signal actually predict tomorrow's returns once you remove the boring style exposures everyone already trades? Straight line, with the boring stuff subtracted first. The technical name for fitting that line is *regression*, and it is the single most-used piece of mathematics on a desk -- more than option pricing, more than optimization, more than anything exotic.

And yet regression is also where careers quietly go to die. The formula is so easy to run -- one line of code, one button in a spreadsheet -- that people stop respecting it. They read a coefficient off the output and trade on it. They see a big $R^2$ and declare victory. They look at a t-statistic of 3 and conclude the effect is real. Every one of those moves can be catastrophically wrong, and the failures are not exotic edge cases; they are the *normal behavior of financial data*. Returns are noisy in ways that change over time, they are correlated from one day to the next, and the factors we feed in are often near-copies of each other. Plain regression assumes none of that happens. When the assumptions break -- and in markets they always break -- the formula keeps producing confident-looking numbers that are wrong.

![Regression feeds a factor matrix and a returns column into a fitting step that produces a coefficient vector used as betas and hedge ratios.](/imgs/blogs/regression-ols-gls-regularized-math-for-quants-1.png)

The diagram above is the mental model for the whole post. You feed in a matrix of factors $X$ -- each column is one explanatory variable, each row is one day of data -- and a column $y$ of the thing you are trying to explain, usually a stream of returns. A fitting machine chews on it and produces a vector $\hat\beta$ of coefficients. Depending on what you fed in, the desk calls that output a *beta*, a *hedge ratio*, or a *factor loading*. Our job over the next several thousand words is to take this machine apart: build the simple version (OLS) from absolute zero, learn to read its output honestly, find every place it lies, and then climb a ladder of fixes -- generalized least squares for messy noise, ridge and lasso for too many look-alike factors -- that rebuild it on firmer ground.

We assume no finance and no statistics background. Every term is defined the first time it appears, every formula gets a plain-English analogy before the symbols, and every idea is anchored in a worked example with real dollars. By the end you should be able to run a regression, read its diagnostics without fooling yourself, spot the four classic failure modes, and know exactly which fix to reach for. This is educational material about how a standard tool behaves -- it is not investment advice.

## Foundations: the line, the error, and the fit

Before any matrices, let us nail the simplest possible case completely solid: fitting one straight line to a cloud of dots.

### What a regression is actually doing

Suppose that for each trading day you wrote down two numbers: how much the broad market index moved (call it $x$) and how much one particular stock moved (call it $y$). Plot each day as a dot, with $x$ running left-to-right and $y$ running up-and-down. The dots form a loose, tilted cloud. On days the market rose, the stock usually rose too -- but not by the same amount, and not every single time, because a stock's daily move is the market's pull *plus* a pile of company-specific news and noise that has nothing to do with the market.

You want one straight line through that cloud that captures the typical relationship. A line is two numbers: an *intercept* $a$ (where the line crosses the vertical axis when $x = 0$) and a *slope* $b$ (how many units $y$ moves for each one-unit move in $x$). The slope is the number the desk craves. It is the stock's *beta*: the answer to "if the market moves 1%, how much does this stock typically move?" A beta of 1 means the stock moves one-for-one with the market. A beta of 1.5 means it is 50% jumpier -- it amplifies both rallies and crashes. A beta of 0.5 means it is half as sensitive, a sleepy defensive name. A beta near 0 means it does its own thing regardless of the market.

The intercept $a$ has a name too. In finance it is the stock's *alpha* -- the average return the market move cannot explain. Alpha is what every active manager hunts: return you earned that was not just payment for taking market risk. For a pure hedge we care only about the slope, but the same single line hands you both the thing you want to hedge away (beta) and the thing you want to keep (alpha).

For any candidate line, each dot sits some vertical distance above or below it. That gap -- the difference between what actually happened and what the line predicted -- is the *residual*: $e_i = y_i - (a + b\,x_i)$. A residual is just the error the line makes on one data point. Good fits have small residuals; bad fits have big ones. We measure the gap *vertically*, in the direction of $y$, because we are treating $x$ (the market) as known and trying to predict $y$ (the stock). This asymmetry is why regressing the stock on the market gives a different slope than regressing the market on the stock -- a subtlety that trips up beginners and that we return to in the misconceptions section.

### Why we square the errors

We want the line whose residuals are "small overall". But small how? If we simply added up the raw residuals, the positives (dots above the line) and negatives (dots below) would cancel, and a terrible line could score zero. So we need a measure positives and negatives cannot cancel out of. Two candidates: sum the *absolute values*, or sum the *squares*. Both exist in practice -- absolute-value fitting is *least absolute deviations* and resists outliers better -- but squaring is the overwhelming default, for three reasons.

First, squaring makes every error positive, so they cannot cancel. Second, it punishes big misses far more than small ones (a residual of 4 contributes 16; a residual of 2 contributes only 4), which matches the trader's gut feeling that one catastrophic miss is worse than several small ones. Third, and decisively, squaring produces a *smooth* function with a single clean minimum you can find with calculus in closed form, whereas the absolute value has a kink that forces a numerical solver. The square gives you a formula; the absolute value gives you an algorithm. The total we minimize is the *sum of squared residuals*, also called the *squared error*:

$$\text{SSE} = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n}\bigl(y_i - a - b\,x_i\bigr)^2.$$

The line that makes this sum smallest is the *least-squares* line, and the method is *ordinary least squares* (OLS). The name is literal: of all possible lines, pick the one with the *least* sum of *squares*.

> The least-squares line is the one that, if you nudged it any direction, would make your total squared error go up. It sits at the bottom of a smooth bowl with exactly one floor.

That bowl image generalizes all the way to the matrix case, so hold onto it. Plot the squared error as a height above the plane of all possible $(a, b)$ choices. Because the error is a sum of squares of things linear in $a$ and $b$, this surface is a *paraboloid* -- a smooth, upward-opening bowl with one lowest point. No false bottoms, no local minima, no ambiguity. The answer always exists and is always unique (as long as the factors are not degenerate), and you find it by rolling downhill. Later, when collinearity strikes, the bowl flattens into a long shallow trough, and that flatness is the geometric face of an unstable answer -- but we are getting ahead of ourselves.

### The single-variable answer, with numbers

For one input variable, calculus hands you a clean closed-form answer. Find the bottom of the bowl by demanding that the surface be flat in both directions -- set the *partial derivatives* of the SSE with respect to $a$ and $b$ to zero. (A partial derivative is the rate of change of a function when you wiggle one input and hold the rest fixed; at the bottom of a bowl, wiggling any input does not change the height to first order, so every partial is zero.) Solving the two resulting equations gives:

$$b = \frac{\operatorname{Cov}(x,y)}{\operatorname{Var}(x)}, \qquad a = \bar y - b\,\bar x,$$

where $\operatorname{Cov}(x,y)$ is the *covariance* (how much $x$ and $y$ move together), $\operatorname{Var}(x)$ is the *variance* of $x$ (how much $x$ spreads out on its own), and $\bar x, \bar y$ are the averages. Read the slope as a ratio: the numerator picks up the part of $y$'s wiggle that lines up with $x$'s wiggle, and dividing by $x$'s own wiggle converts that into "units of $y$ per unit of $x$". This single fraction is the seed of everything that follows -- the matrix formula is literally this same ratio promoted to many factors at once.

#### Worked example: a one-factor hedge

You run a desk and you are long \$1,000,000 of a stock we will call NOVA. You want to hedge the part of NOVA that just rides the broad market, so you can isolate the company-specific bet. You pull five days of returns, in percent:

| Day | Market $x$ | NOVA $y$ |
|----|-----------|----------|
| 1  | +1.0%     | +1.4%    |
| 2  | -0.5%     | -0.4%    |
| 3  | +2.0%     | +2.9%    |
| 4  | 0.0%      | +0.2%    |
| 5  | -1.5%     | -2.1%    |

The averages: $\bar x = (1.0 - 0.5 + 2.0 + 0.0 - 1.5)/5 = 1.0/5 = 0.2\%$ and $\bar y = (1.4 - 0.4 + 2.9 + 0.2 - 2.1)/5 = 2.0/5 = 0.4\%$.

Now the covariance numerator, $\sum (x_i - \bar x)(y_i - \bar y)$, computing each deviation and product:

- Day 1: $(1.0-0.2)(1.4-0.4) = (0.8)(1.0) = 0.80$
- Day 2: $(-0.5-0.2)(-0.4-0.4) = (-0.7)(-0.8) = 0.56$
- Day 3: $(2.0-0.2)(2.9-0.4) = (1.8)(2.5) = 4.50$
- Day 4: $(0.0-0.2)(0.2-0.4) = (-0.2)(-0.2) = 0.04$
- Day 5: $(-1.5-0.2)(-2.1-0.4) = (-1.7)(-2.5) = 4.25$

Sum: $0.80 + 0.56 + 4.50 + 0.04 + 4.25 = 10.15$. The variance denominator, $\sum (x_i - \bar x)^2$: $0.64 + 0.49 + 3.24 + 0.04 + 2.89 = 7.30$. So the slope is $b = 10.15 / 7.30 = 1.39$. NOVA's beta is about **1.39** -- it moves roughly 1.4% for every 1% the market moves.

The hedge falls right out. To neutralize the market component of a \$1,000,000 long, you short beta times the position: $1.39 \times \$1{,}000{,}000 = \$1{,}390{,}000$ of the index. After that short, if the market drops 2% and drags NOVA down with it, your index short profits roughly the same dollars NOVA loses on its market component, and what is left is the pure stock-specific bet you wanted. The intuition: **a regression slope is not a number on a page; it is a dollar instruction for how much of one thing to trade against another.**

### Promoting one factor to many: the matrix form

Real desks rarely explain a stock with the market alone. They use several factors at once -- the market, a value factor, a momentum factor, a size factor, an industry, and so on. The line becomes a *plane* (with two factors) or a *hyperplane* (with more), and we need bookkeeping that scales. That bookkeeping is matrices.

Stack the data. Let $X$ be a matrix with one row per day and one column per factor (plus a leading column of all 1s, which carries the intercept). Let $y$ be the column of outcomes -- the returns we are explaining. Let $\beta$ be the column of coefficients we are solving for, one per factor. The model is

$$y = X\beta + \varepsilon,$$

where $\varepsilon$ is the column of residuals -- the part of $y$ no factor explains. We want the $\beta$ that makes the total squared residual $\lVert y - X\beta \rVert^2$ as small as possible (the double bars mean "length of the vector", i.e. the square root of the sum of squares, and minimizing the length is the same as minimizing the squared length). Setting the gradient to zero gives the *normal equations* $X^\top X \beta = X^\top y$, and solving for $\beta$ gives the formula at the heart of this post:

$$\boxed{\;\hat\beta = (X^\top X)^{-1}X^\top y\;}$$

Read it as the same single-variable ratio, generalized. $X^\top X$ is the matrix version of "variance" -- it captures how each factor spreads out *and* how the factors overlap with each other. $X^\top y$ is the matrix version of "covariance" -- how each factor moves with the outcome. The formula divides one by the other (matrix "division" is multiplying by an inverse), exactly like $b = \operatorname{Cov}/\operatorname{Var}$. Every coefficient $\hat\beta_j$ is the effect of factor $j$ *after holding all the other factors fixed* -- the marginal contribution, with the overlap with other factors stripped out. That "holding the others fixed" is what makes multiple regression powerful and also what makes it fragile when two factors are nearly the same, as we will see.

## 1. Reading the output honestly: R-squared and t-stats

Run a regression and the software hands back more than coefficients. It hands back diagnostics: an $R^2$, an *adjusted* $R^2$, a *standard error* and *t-statistic* for each coefficient, an overall *F-statistic*. These numbers are seductive because they look authoritative. Most of the damage regression does on a desk comes from misreading exactly these numbers, so let us define each one precisely and -- more importantly -- say what it does *not* tell you.

### What R-squared is

$R^2$ answers one narrow question: of all the up-and-down variation in $y$, what *fraction* did the line manage to explain? Formally,

$$R^2 = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2} = 1 - \frac{\text{SSE}}{\text{TSS}},$$

where $\hat y_i$ is the line's prediction, SSE is the sum of squared residuals (the variation the line missed), and TSS is the total sum of squares (the variation in $y$ around its own average, the variation a do-nothing model that just guesses the mean would leave on the table). If the line explains everything, SSE is 0 and $R^2 = 1$. If the line is no better than guessing the average, SSE equals TSS and $R^2 = 0$. So $R^2$ runs from 0 (useless) to 1 (perfect fit), and reads naturally as a percentage: an $R^2$ of 0.30 means the factors explained 30% of the wiggle.

Here is the analogy. Think of TSS as the total mess in the room before you cleaned, and SSE as the mess still left after the line did its tidying. $R^2$ is the fraction of mess you cleaned up. That sounds great -- bigger is better, right? -- and that instinct is precisely the trap.

### The R-squared trap

$R^2$ measures fit *on the data the model already saw*. It says nothing about whether the relationship is real, whether it is causal, or whether it will survive into tomorrow. Three things make a high $R^2$ worthless, and all three are everyday occurrences in markets.

First, **adding any factor can only push $R^2$ up, never down.** Even a column of pure random noise will, by luck, line up with a little of $y$'s wiggle, and OLS will happily use it. Throw enough junk factors at a regression and you can drive $R^2$ to nearly 1 on garbage. That is why we use *adjusted* $R^2$, which docks a penalty for each factor added:

$$R^2_{\text{adj}} = 1 - (1 - R^2)\frac{n-1}{n-k-1},$$

where $n$ is the number of data points and $k$ the number of factors. If a new factor does not earn its keep, adjusted $R^2$ falls even as raw $R^2$ rises. Always read the adjusted version.

Second, **a high $R^2$ on trending data is an illusion.** If both $y$ and $x$ are drifting upward over time -- as prices, GDP, or anything that grows tends to do -- a regression of one on the other will show a gorgeous $R^2$ even when the two have no real connection. They both went up; the line "explains" that; the $R^2$ glows. This is *spurious regression*, and it has fooled people into "discovering" that the stock market is driven by butter production in Bangladesh.

Third, **in-sample $R^2$ is silent about out-of-sample performance.** A model can fit the past beautifully and predict the future terribly. The gap between the two is *overfitting*, and the whole back half of this post is about controlling it.

![A grid of four classical regression assumptions mapped to the specific market behavior that violates each one.](/imgs/blogs/regression-ols-gls-regularized-math-for-quants-2.png)

The matrix above previews where the rest of the post is headed: the four classical OLS assumptions in the rows, and the concrete market behavior that breaks each one in the right column. Keep it nearby -- every section from here links back to one of these rows.

#### Worked example: the R-squared trap

You build a signal to predict the daily return of an exchange-traded fund. You regress the fund's return on three of your "factors" and the software reports $R^2 = 0.92$. You are thrilled -- 92% of the movement explained! You size up the bet.

Then a colleague asks what the three factors are. The first is the fund's own previous-day price level. The second is a calendar trend (1, 2, 3, ... counting days). The third is a 10-day moving average of the fund's price. None of these is a *return-predicting signal* -- they are all just restatements of where the price already is. Because the fund's price drifted steadily upward over your sample, all three trended up too, and OLS used the trend to "explain" the upward drift. The $R^2$ of 0.92 is measuring how well a rising line fits a rising line.

You test it out of sample on the next quarter, where the fund chops sideways instead of trending. The model, which only knew how to ride a trend, posts an out-of-sample $R^2$ of $-0.15$ -- *negative*, meaning it predicts worse than just guessing the average. Trading it would have lost money: a backtest on the new quarter shows a P&L of $-\$84{,}000$ on a \$1,000,000 book. The lesson: **$R^2$ tells you how well you fit the past, not whether you learned anything that generalizes -- a number near 1 is a reason to be suspicious, not satisfied.**

### What a t-statistic is

For each coefficient, the regression also reports a *standard error* and a *t-statistic*. The standard error is the estimated uncertainty in the coefficient -- roughly, "if you re-ran this on a different but similar sample, how much would this number jump around?" The t-statistic is the coefficient divided by its standard error:

$$t_j = \frac{\hat\beta_j}{\operatorname{SE}(\hat\beta_j)}.$$

It measures how many standard errors the coefficient sits away from zero. A common rule of thumb: $|t| > 2$ means the coefficient is "statistically significant" -- the data are unlikely to have produced a number this far from zero if the true effect were zero. The companion post on [hypothesis testing and p-values](/blog/trading/quantitative-finance/linear-regression-deep-quant-interviews) digs into exactly what that probability means and how badly it gets abused.

The t-statistic is only as honest as the standard error underneath it. And here is the heart of the matter: **the standard error OLS prints by default rests on assumptions that financial data routinely violate.** When those assumptions break, the standard error is too small, the t-statistic is too big, and a coefficient that is really just noise gets a significance stamp it does not deserve. The next two sections are entirely about that lie -- where it comes from and how to fix it.

## 2. When the noise misbehaves: heteroskedasticity and autocorrelation

OLS gives you the *right* coefficients (the point estimates $\hat\beta$ are still unbiased even when the noise is messy), but the *wrong* uncertainty around them when two assumptions about the residuals fail. These two failures are so common in finance that assuming they are absent is the single most expensive default mistake in applied regression.

### Homoskedasticity: the assumption that noise is one steady size

*Homoskedasticity* (homo = same, skedasis = scatter) is the assumption that the residuals have the *same* spread everywhere -- the cloud of dots is equally fat around the line whether $x$ is large or small, whether the market is calm or panicking. Picture a garden hose held at a fixed angle, spraying a steady cone of water: the scatter is the same near the nozzle as far away. That is homoskedastic noise.

Markets are not like that. Volatility *clusters*: quiet days follow quiet days, and wild days follow wild days. During a crash, daily moves are huge; during a sleepy summer, they are tiny. So the residuals of a return regression are *fat* in stressed periods and *thin* in calm ones. That is *heteroskedasticity* (hetero = different), and it is row three of our assumptions matrix. The point estimates survive it, but the standard formula for the standard error assumes constant scatter, so it computes a single average uncertainty that is too small in the wild periods and too large in the calm ones -- and on net usually too small, which inflates t-stats.

### Autocorrelation: the assumption that errors do not remember

The fourth assumption is *no autocorrelation*: each residual is independent of the ones before it. Today's surprise should not predict tomorrow's surprise. Picture flipping a fair coin -- the previous flip tells you nothing about the next. That is what OLS assumes about the errors.

Financial residuals violate this constantly. Returns exhibit momentum and reversal; positions stay on for days; and -- a subtle but huge source -- when you use *overlapping* windows (say, regressing 5-day forward returns measured every single day), consecutive observations literally share four of their five days, so their errors are mechanically correlated. When residuals are autocorrelated, the regression *thinks* it has $n$ independent data points when it really has far fewer independent pieces of information. It is like polling 1,000 people but accidentally interviewing the same family 1,000 times -- you have one opinion, not a thousand. The standard error, which divides by the (overstated) sample size, comes out far too small, and t-stats balloon.

> Autocorrelated, heteroskedastic noise does not bias your beta -- it bias your *confidence*. Your coefficient is fine; your certainty about it is a fantasy.

### The fix part one: robust and Newey-West standard errors

You do not have to re-fit the model to fix the standard errors -- you keep the same $\hat\beta$ and just recompute the uncertainty using a formula that does not assume constant, independent noise. Two standard tools:

- ***White* (heteroskedasticity-robust) standard errors** fix the constant-scatter assumption. They let each residual carry its own variance, so the wild days count their own fatness instead of an average. These are also called "robust" or "HC" standard errors.
- ***Newey-West* (HAC) standard errors** fix *both* heteroskedasticity *and* autocorrelation -- HAC stands for "heteroskedasticity- and autocorrelation-consistent". They add up not just each residual's own variance but also the correlations between nearby residuals, out to a chosen number of lags. The longer the autocorrelation, the more lags you include, and the more the standard error widens.

The mechanics: ordinary OLS computes the coefficient covariance as $\sigma^2 (X^\top X)^{-1}$, assuming one constant noise variance $\sigma^2$. The robust "sandwich" estimator replaces the meat of that expression, $\sigma^2 X^\top X$, with $X^\top \hat\Omega X$, where $\hat\Omega$ is built from the actual squared residuals (and, for Newey-West, their cross-products at nearby lags). You do not need to memorize the algebra; you need to know that this sandwich does not assume the noise is well-behaved, and that it almost always comes out *wider* than the naive version on financial data.

![Two side-by-side panels show naive OLS standard errors with a high t-stat next to wider Newey-West errors with a t-stat below the significance line.](/imgs/blogs/regression-ols-gls-regularized-math-for-quants-3.png)

The before-and-after above is the whole point in one image. Same coefficient, same data -- only the standard error changed -- and a result that looked publishable under naive errors evaporates under honest ones.

#### Worked example: Newey-West widens a t-stat

You have found what looks like a tradeable signal: a regression coefficient of **0.30** linking your factor to next-week returns. The naive OLS standard error is **0.10**, so the t-statistic is $0.30 / 0.10 = 3.0$. That clears the $|t| > 2$ bar comfortably; in fact it clears the stricter $|t| > 2.5$ bar people use after worrying about multiple testing. You are ready to allocate \$5,000,000 to it.

Before you do, you notice two things. First, you used *weekly* returns but sampled them *daily* (overlapping windows), so consecutive residuals share four days -- textbook autocorrelation. Second, your residuals are visibly fatter during the two stressed quarters in your sample -- heteroskedasticity. So you recompute with Newey-West standard errors using a lag of 5 (one trading week). The coefficient is unchanged at 0.30, but the standard error widens to **0.18**. The new t-statistic is $0.30 / 0.18 = 1.67$.

That is below 2. Under honest standard errors, you *cannot* reject the possibility that your signal is zero and you got lucky. The naive analysis would have had you put \$5,000,000 behind noise; the robust analysis sends you back to the drawing board. The intuition: **overlapping, volatility-clustered data carry far less independent information than the row count suggests, and robust standard errors are how you stop your software from lying to you about your own confidence.**

## 3. The proper fix for messy noise: generalized least squares

Robust standard errors patch the *uncertainty* while leaving the coefficients alone. *Generalized least squares* (GLS) goes further: it changes the coefficients themselves, by telling the regression to *trust the clean data points more than the noisy ones*. This is the second rung on our estimator ladder.

### The intuition: weight by reliability

Plain OLS treats every data point as equally trustworthy -- a wild crash day and a sleepy holiday day get the same vote in determining the line. But if the crash day's residual is naturally ten times fatter (heteroskedasticity), it contributes ten times the noise, and letting it vote equally drags the line around. The sensible fix is to *down-weight* the unreliable points and *up-weight* the reliable ones, exactly as you would trust a precise measurement more than a sloppy one when averaging.

The simplest version is *weighted least squares* (WLS): give each observation a weight inversely proportional to its noise variance, then minimize the *weighted* squared error $\sum w_i (y_i - x_i^\top\beta)^2$. A point with twice the noise variance gets half the weight. Full GLS is the general case: instead of a per-point weight, it uses the entire *covariance matrix of the residuals* $\Omega$, which captures both the differing variances (the diagonal) *and* the correlations between points (the off-diagonals, the autocorrelation). The GLS estimator is

$$\hat\beta_{\text{GLS}} = (X^\top \Omega^{-1} X)^{-1} X^\top \Omega^{-1} y.$$

Compare it to OLS, $\hat\beta = (X^\top X)^{-1}X^\top y$: GLS simply slips $\Omega^{-1}$ into both products. If $\Omega$ is the identity (all points equally noisy and uncorrelated), the $\Omega^{-1}$ disappears and GLS *is* OLS. The matrix $\Omega^{-1}$ is the "reliability weighting": it stretches the problem so that, in the stretched coordinates, the noise *is* well-behaved, OLS becomes valid again, and the answer it finds is the best linear unbiased estimate. (If you want to see exactly how positive-definite covariance matrices like $\Omega$ get factored and inverted in practice, the [Cholesky factorization post](/blog/trading/math-for-quants/svd-least-squares-regression-math-for-quants) covers the machinery -- GLS is one of its main customers.)

![A vertical stack climbs from plain OLS through weighted and generalized least squares to regularized fits, each rung adding one fix.](/imgs/blogs/regression-ols-gls-regularized-math-for-quants-4.png)

The stack above is the spine of the post. OLS sits at the bottom: equal weights, clean-noise assumption. WLS adds per-point weighting. GLS uses the full noise covariance to handle correlated errors. And the top rung -- ridge and lasso -- handles a different disease entirely, too many look-alike factors, which we reach next.

### The cost of GLS: you have to estimate the noise

GLS sounds strictly better, so why is OLS still the default? Because GLS needs $\Omega$, the residual covariance matrix -- and you do not know it. You have to *estimate* it from the same noisy data, and a badly estimated $\Omega$ can make GLS *worse* than OLS. This is the recurring tax in quant work: every fancier method needs more inputs, and each input you estimate adds its own error. In practice, desks often prefer plain OLS coefficients with robust (Newey-West) standard errors over full GLS, precisely because the robust route does not require estimating the entire noise structure -- it only fixes the error bars, where the damage is, and leaves the coefficients alone. The principle of *feasible GLS* -- estimate $\Omega$ in a first pass, then re-fit -- exists, but it is used carefully and with humility.

#### Worked example: WLS beats OLS when one period is wild

You regress a stock's daily return on the market over 100 days. The first 80 days were calm (residual standard deviation about 0.5%) and the last 20 were a crisis (residual standard deviation about 2.5%, five times fatter). Run plain OLS and the 20 crisis days, each contributing $5^2 = 25$ times the squared noise of a calm day, dominate the fit -- the beta you get is yanked around by the panic and comes out at, say, **1.55**, when the stock's calm-regime beta is really near **1.20**.

Now run WLS, weighting each crisis day by $1/2.5^2 = 0.16$ relative to a calm day's weight of $1/0.5^2 = 4$ (a 25-to-1 ratio). The crisis days still count, but they no longer drown out the 80 calm days. The WLS beta comes out at **1.23**, much closer to the stable relationship. On a \$1,000,000 long, the OLS hedge would short \$1,550,000 of the index, over-hedging by \$320,000 versus the WLS hedge of \$1,230,000 -- an over-hedge that itself loses money when the market rallies. The intuition: **when some data points are far noisier than others, counting them equally lets the loudest days, not the most informative ones, set your hedge.**

## 4. When factors look alike: multicollinearity

We now switch diseases. Heteroskedasticity and autocorrelation attacked the *standard errors*. *Multicollinearity* attacks the *coefficients themselves*, and it is the most insidious failure of all because the regression gives no error and prints confident numbers that are pure instability.

### The intuition: two factors carrying the same information

*Multicollinearity* means two or more of your factors are nearly the same thing -- highly correlated, carrying almost identical information. Suppose you try to explain a stock's return with two factors that happen to move in near-lockstep: say, a large-cap value index and a large-cap quality index that overlap 95% of the time. The regression's job is to assign credit to *each factor separately* -- the coefficient on factor A is "the effect of A holding B fixed". But if A and B always move together, "holding B fixed while A changes" almost never happens in the data. The regression is being asked a question the data cannot answer: how much is A versus B, when the two are never apart.

The geometric picture: recall the error bowl from the foundations. With independent factors, the bowl is round and steep, with a sharp, well-defined bottom. With collinear factors, the bowl flattens into a long, shallow *trough* -- a whole valley of $(\beta_A, \beta_B)$ combinations that all fit the data almost equally well. $\beta_A = +480, \beta_B = -470$ fits about as well as $\beta_A = 0.6, \beta_B = 0.6$, because A and B nearly cancel anyway. OLS, forced to pick one point in that valley, picks a wildly large, sign-flipped combination on a knife's edge -- and a tiny change in the data sends it skidding to a completely different point.

### The diagnostic: the variance inflation factor

The standard alarm for multicollinearity is the *variance inflation factor* (VIF). For each factor, regress it on all the *other* factors and read off that regression's $R^2$. The VIF is

$$\text{VIF}_j = \frac{1}{1 - R_j^2},$$

where $R_j^2$ is how well the other factors predict factor $j$. If factor $j$ is unrelated to the others, $R_j^2 = 0$ and $\text{VIF}_j = 1$ -- no inflation. If the others predict factor $j$ almost perfectly, $R_j^2 \to 1$ and VIF explodes toward infinity. The name is literal: the VIF is the multiple by which collinearity has *inflated the variance* (the squared standard error) of that coefficient. A VIF of 10 means the coefficient's standard error is $\sqrt{10} \approx 3.2$ times wider than it would be with independent factors. Rules of thumb: VIF above 5 is a yellow flag, above 10 a red one.

Underneath, multicollinearity is the same disease the [SVD and least-squares post](/blog/trading/math-for-quants/svd-least-squares-regression-math-for-quants) calls a high *condition number*: when factors are nearly parallel, $X^\top X$ becomes nearly impossible to invert, and $(X^\top X)^{-1}$ -- which sits at the center of the OLS formula -- amplifies tiny data wiggles into huge coefficient swings. VIF and condition number are two windows onto the same underlying problem.

![A tree branches regression failure modes into wrong coefficients, lying standard errors, and overfitting, each with its concrete causes.](/imgs/blogs/regression-ols-gls-regularized-math-for-quants-5.png)

The tree above organizes every failure we have met and every one still to come. Wrong-coefficient failures (multicollinearity, omitted-factor bias) attack the point estimate. Lying-standard-error failures (heteroskedasticity, autocorrelation) attack the confidence. Overfitting failures (the $R^2$ trap, too many factors) attack generalization. Different diseases, different cures -- the rest of the post is the cures for the right column.

#### Worked example: collinear factors blow up the betas

You want to explain a stock's return using two momentum factors: 12-month momentum (factor A) and 11-month momentum (factor B). These are nearly identical -- they share 11 of 12 months -- with a correlation of about 0.99. You regress the stock's return on both and OLS reports:

$$\hat\beta_A = +480, \qquad \hat\beta_B = -470.$$

These are absurd. A factor loading of 480 says a 1% move in factor A predicts a 480% move in the stock. What OLS has actually found is that A and B nearly cancel, so it can pile up enormous opposite coefficients and still fit the sample, because the +480 of A and the -470 of B net out to a sane small exposure *on the days A and B move together* -- which is almost every day. The VIF on each factor is roughly $1/(1 - 0.99^2) = 1/0.0199 \approx 50$ -- catastrophically high.

The danger is what happens out of sample. On the one day A and B *do* diverge -- the single month they do not share suddenly matters -- those two giant opposite coefficients no longer cancel, and the model predicts a move of hundreds of percent. Trading this would post a catastrophic loss; in a backtest, one divergence day alone cost $-\$210{,}000$ on a \$1,000,000 book. The intuition: **when two factors carry the same information, the regression cannot tell them apart, so it produces giant offsetting coefficients that are a time bomb waiting for the day the factors disagree.** The fix is to stop letting the coefficients grow without limit -- which is exactly what regularization does.

## 5. Regularization: paying the model to stay humble

Everything so far has been *unconstrained* least squares: find the coefficients that fit the data best, full stop. Regularization changes the objective. It says: fit the data well, *but also* keep the coefficients small -- and it charges a penalty for big coefficients. This single idea cures both the overfitting from too many factors and the explosion from collinearity, and it is the top rung of our estimator ladder.

### The intuition: a budget for boldness

Picture a regression that wants to be as bold as possible -- it will use any factor, however flimsy, and crank up coefficients however large, to squeeze out one more drop of in-sample fit. Regularization gives that regression a *budget*. Every unit of coefficient size costs something, so the model spends its boldness only where the data really earn it. A factor with strong, clean signal gets a large coefficient because the fit improvement outweighs the penalty; a factor that only helps a little, or only by exploiting collinearity, gets shrunk toward zero because it is not worth the cost. The result is a *humbler* model that fits the past slightly worse but the future much better.

Formally, we minimize the squared error *plus* a penalty on the coefficients, scaled by a knob $\lambda \ge 0$:

$$\hat\beta_{\text{reg}} = \arg\min_\beta \;\underbrace{\lVert y - X\beta\rVert^2}_{\text{fit the data}} \;+\; \lambda \underbrace{P(\beta)}_{\text{stay small}}.$$

The penalty $P(\beta)$ comes in three flavors, and $\lambda$ controls how harsh it is: $\lambda = 0$ recovers plain OLS (no penalty), and $\lambda \to \infty$ crushes every coefficient to zero.

### Ridge regression: the L2 penalty

*Ridge* uses the *sum of squared coefficients* as its penalty: $P(\beta) = \sum_j \beta_j^2 = \lVert\beta\rVert_2^2$ (the "L2 norm" squared). Because squaring punishes large coefficients harshly, ridge shrinks every coefficient smoothly toward zero -- but never *exactly* to zero. It keeps all factors, just dialed down. Ridge has a beautiful closed form:

$$\hat\beta_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y,$$

which is the OLS formula with $\lambda I$ added to $X^\top X$ before inverting. That little $+\lambda I$ is doing all the work: it adds a positive number down the diagonal, which fixes the near-singular, hard-to-invert $X^\top X$ that collinearity creates. Where OLS divides by a number near zero (and blows up), ridge divides by that number *plus $\lambda$* (and stays finite). This is exactly why ridge is the cure for multicollinearity: it props up the flat trough in the error bowl, restoring a single well-defined bottom. (The [SVD post](/blog/trading/math-for-quants/svd-least-squares-regression-math-for-quants) shows that ridge is mathematically identical to gently shrinking each singular value, the soft cousin of truncating the tiny ones.)

### Lasso regression: the L1 penalty

*Lasso* (Least Absolute Shrinkage and Selection Operator) uses the *sum of absolute coefficients*: $P(\beta) = \sum_j |\beta_j| = \lVert\beta\rVert_1$ (the "L1 norm"). The absolute value, unlike the square, has a sharp corner at zero, and that corner has a remarkable consequence: lasso drives weak coefficients *exactly* to zero, not just small. It performs *feature selection* -- it picks a subset of factors and discards the rest entirely. Where ridge keeps all 100 factors at reduced size, lasso might keep 12 and zero out 88. There is no closed form (the corner means you solve it numerically), but the payoff is a sparse, interpretable model.

The geometric reason lasso zeros things and ridge does not is worth one sentence: the L1 penalty's constraint region is a diamond with sharp corners on the axes, and the fit's contours tend to first touch that region *at a corner* -- where some coefficients are exactly zero -- whereas the L2 region is a smooth circle with no corners, so the contours touch it at a generic point where everything is small but nonzero.

### Elastic net: both at once

*Elastic net* simply uses both penalties together: $P(\beta) = \alpha\lVert\beta\rVert_1 + (1-\alpha)\lVert\beta\rVert_2^2$. It inherits lasso's feature selection *and* ridge's stability under collinearity. This matters because lasso alone behaves badly with correlated factors -- faced with two near-identical factors it arbitrarily picks one and zeros the other, and which one it picks can flip on a tiny data change. Elastic net's ridge component lets correlated factors *share* the load instead, keeping the selection stable. For the many-correlated-factor problems that dominate quant research, elastic net is often the workhorse.

![A grid contrasts ridge, lasso, and elastic-net penalties by their formula, their effect on coefficients, and when to use each.](/imgs/blogs/regression-ols-gls-regularized-math-for-quants-7.png)

The grid above is the decision table. Ridge when you believe all factors matter and just want stability; lasso when you believe only a few matter and want the model to find them; elastic net when you have many correlated factors and want both shrinkage and selection.

### The deep reason it works: the bias-variance trade-off

Why should making the fit *worse* on the training data make it *better* on new data? Because of the *bias-variance trade-off*, one of the most important ideas in all of statistics. The error a model makes on new data decomposes into two parts. *Bias* is error from the model being too simple to capture the true relationship -- it systematically misses. *Variance* is error from the model being too sensitive to the particular sample -- re-run it on fresh data and the coefficients jump around. (The companion post on [estimators, bias and variance](/blog/trading/quantitative-finance/linear-regression-deep-quant-interviews) develops this decomposition in full.)

OLS is *unbiased* -- on average it gets the true coefficients right -- but in the presence of collinearity or many factors it has *enormous variance*: any single sample produces wild coefficients. Regularization deliberately *adds a little bias* (by shrinking coefficients toward zero, it systematically underestimates the true effects) in exchange for a *large reduction in variance* (the shrunk coefficients are stable across samples). Total error is bias-squared plus variance, and when variance is huge, trading a little bias for a lot less variance is a winning deal. The penalty knob $\lambda$ is literally the bias-variance dial: turn it up and you buy more stability at the cost of more bias; turn it down toward zero and you recover unbiased-but-wild OLS. You pick $\lambda$ by cross-validation -- holding out part of the data, fitting on the rest, and choosing the $\lambda$ that predicts the held-out part best.

![Two panels contrast wild opposite-signed OLS coefficients on collinear factors with stable shrunken ridge coefficients that profit out of sample.](/imgs/blogs/regression-ols-gls-regularized-math-for-quants-6.png)

The before-and-after above shows the cure landing on the disease from section 4: the same two collinear momentum factors, with OLS's +480 / -470 time bomb on the left and ridge's tame, nearly equal coefficients on the right.

#### Worked example: ridge beats OLS out of sample

Return to the collinear momentum factors from section 4, where OLS gave $\hat\beta_A = +480$ and $\hat\beta_B = -470$. Now fit *ridge* with a penalty $\lambda$ chosen by cross-validation. The $+\lambda I$ term props up the near-singular $X^\top X$, and the wild coefficients collapse to something sane:

$$\hat\beta_{A,\text{ridge}} = +0.60, \qquad \hat\beta_{B,\text{ridge}} = +0.60.$$

Ridge refuses to assign A and B giant opposite roles; it splits the shared momentum signal evenly between them, giving each a modest positive loading. The model now says, sensibly, "these two factors together contribute a combined positive momentum exposure of about 1.2".

Now the out-of-sample test. Recall OLS lost $-\$210{,}000$ on the one day A and B diverged, because its giant offsetting coefficients stopped cancelling. Ridge, with its tame +0.6 / +0.6, barely flinches on that day -- a small +0.6 coefficient times a 1% factor divergence is a 0.6% prediction, not a 480% one. Over the full out-of-sample quarter, the OLS book posts a P&L of $-\$185{,}000$ while the ridge book posts $+\$92{,}000$ on the same \$1,000,000 -- a swing of \$277,000 from one change: capping how bold the coefficients are allowed to be. The intuition: **regularization trades a little accuracy on the data you have seen for a lot of robustness on the data you have not, and on collinear factors that trade is the difference between a blow-up and a profit.**

## 6. Putting the ladder together: which fix for which disease

We have built a complete toolkit. The discipline is matching the tool to the disease, because each fix solves a different problem and using the wrong one wastes effort or hides the real issue. Here is the map.

| Symptom you see | Disease | The fix | What it changes |
|---|---|---|---|
| t-stats look too good; overlapping windows | Autocorrelation | Newey-West standard errors | Widens error bars only |
| Residuals fatter in stressed periods | Heteroskedasticity | White / robust standard errors, or WLS | Error bars (and coefficients, if WLS) |
| Coefficients huge and sign-flipped; high VIF | Multicollinearity | Ridge, or drop a factor | Shrinks and stabilizes coefficients |
| Great in-sample fit, terrible out-of-sample | Overfitting / too many factors | Lasso or elastic net + cross-validation | Selects factors, shrinks coefficients |
| High $R^2$ on trending data | Spurious regression | Regress on returns/changes, not levels | The variables themselves |

Notice the division of labor. The two *standard-error* fixes (Newey-West, White) leave your coefficients untouched and only correct your confidence -- reach for them when your betas are fine but your t-stats are suspicious. The *GLS / WLS* family changes the coefficients by reweighting noisy data -- reach for it when you trust some observations far more than others. The *regularization* family (ridge, lasso, elastic net) changes the coefficients by penalizing their size -- reach for it when you have too many factors or factors that look alike. A real research workflow usually layers them: fit with regularization to tame the coefficients, then report Newey-West standard errors to tame the confidence. The two are not rivals; they fix orthogonal problems.

The art is also in *prevention*. The most reliable cure for the $R^2$ trap is to never regress price levels on price levels -- always work in returns or changes, which strips out the trends that manufacture spurious fit. The most reliable cure for multicollinearity is to think hard about your factors before you fit, dropping or combining the ones that obviously overlap, rather than waiting for the VIF alarm. And the most reliable cure for overfitting is honest out-of-sample testing -- the techniques in the [evaluating-alpha-signals post](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research) on information coefficient, Sharpe, and turnover exist precisely to catch the models that look brilliant in-sample and die in production.

## Common misconceptions

**"A high $R^2$ means the model is good."** No. $R^2$ measures fit on data already seen and can be driven arbitrarily high by adding junk factors or by trending variables. A model with $R^2 = 0.9$ in-sample can have negative $R^2$ out-of-sample. What matters is out-of-sample performance, and a suspiciously high in-sample $R^2$ is a reason to investigate for spurious fit, not to celebrate. Read adjusted $R^2$, and trust held-out data over fitted data.

**"A t-stat above 2 means the effect is real."** Only if the standard error underneath it is honest. On autocorrelated or heteroskedastic financial data -- which is to say, on almost all of it -- the naive standard error is too small and the t-stat too big. A naive t-stat of 3 can become a robust t-stat of 1.5. Always recompute with Newey-West before believing a t-stat, and remember that even an honest t-stat ignores the dozens of other signals you tested and discarded (the multiple-testing problem).

**"OLS gives the wrong answer when the noise is messy."** It gives the right *coefficients* -- the point estimates stay unbiased under heteroskedasticity and autocorrelation. What it gets wrong is the *uncertainty* around them. This distinction matters because it tells you which fix to use: if you only need a hedge ratio, plain OLS is fine; if you need to know whether a coefficient is significantly nonzero, you must fix the standard errors.

**"Regularization is a hack that makes the model worse."** It makes the model worse *on the training data* on purpose, and better on new data as a result. That is not a hack; it is the bias-variance trade-off working as designed. When factors are collinear or numerous, the variance of OLS is so large that accepting a little bias to crush the variance reduces total error. The shrinkage is the point.

**"More factors always means a better model."** Each factor you add can only raise in-sample $R^2$, which makes "more is better" feel true. But each factor also adds variance and degrees of freedom to overfit. Beyond a point, adding factors hurts out-of-sample performance even as it flatters the in-sample fit. This is why lasso and elastic net -- which actively *remove* factors -- often beat a kitchen-sink OLS.

**"Beta is a fixed property of a stock."** Beta is the slope of a regression over a specific window, and it drifts -- a stock's beta in a calm year differs from its beta in a crisis, and a beta estimated over five wild days (as in our first worked example) is a noisy snapshot, not a constant. Hedge ratios built on beta need refreshing, and a hedge that was perfect last quarter can be materially off this quarter.

## How it shows up in real markets

### 1. The CAPM beta hedge every equity desk runs

The single most common regression in equity finance is a stock's return on the market's return -- the *market model*, and its slope is the beta from our first worked example. Desks run it on thousands of stocks daily to build *beta-neutral* portfolios: long a basket they like, short index futures sized by the basket's aggregate beta, so the position makes money only if the stock picks beat the market rather than if the market simply rises. When a long book has an aggregate beta of 1.1 against \$50,000,000 of exposure, the desk shorts $1.1 \times \$50{,}000{,}000 = \$55{,}000{,}000$ of index futures. The lesson from this post: that beta is a regression slope with all of regression's frailties -- it drifts over time, it is noisier than the tidy number suggests, and during a crash (when correlations spike toward 1 and betas converge) the hedge that worked in calm markets can leave large residual exposure exactly when it is needed most.

### 2. Fama-French and the multi-factor regression that won a Nobel

In 1992 Eugene Fama and Kenneth French showed that stock returns are explained not by the market alone but by a handful of factors -- market, size (small minus big), and value (high book-to-market minus low), later expanded to five. The entire framework is a *multiple regression* of returns on those factors, and the coefficients are the *factor loadings*. The work earned Fama a share of the 2013 Nobel in economics. The practical catch, straight from this post: the factors are correlated with each other (value and size overlap; profitability and investment overlap in the five-factor model), so the loadings suffer multicollinearity, and a naive regression can hand you unstable, hard-to-interpret coefficients. Modern factor shops fit these with regularization and report robust standard errors precisely because the textbook OLS version is fragile.

### 3. The 1998 LTCM convergence trades and spurious stability

Long-Term Capital Management built enormous positions on relationships that *looked* stable in historical regressions -- spreads between similar bonds that had reliably converged for years. The regressions showed tight fits and small residuals. What the in-sample fit hid was that the residuals were heavily autocorrelated (the spreads moved slowly and persistently) and that the apparent stability was a calm-regime artifact. When the 1998 Russian default hit, the residuals that the models treated as small and independent all moved together and stayed moved, the leverage amplified the loss, and the fund lost roughly \$4.6 billion in months. The lesson is autocorrelation and regime change from sections 2 and 3: a regression that assumes independent, stationary residuals will badly understate the risk of a relationship that can dislocate and stay dislocated.

### 4. The pairs-trade cointegration regression

Statistical-arbitrage desks regress one stock's price on a related stock's price to find a *hedge ratio* for a pairs trade -- go long the cheap one, short the hedge-ratio's worth of the expensive one, and bet the spread mean-reverts. But regressing *price levels* on *price levels* is the textbook setup for spurious regression: both prices trend, the $R^2$ glows, and a relationship can look real that is not. The fix the field adopted -- *cointegration* testing -- is exactly the discipline from this post's misconceptions: only trust the relationship if the residual spread itself is stationary (trendless), not merely if the levels regression has a high $R^2$. Pairs that pass a high-$R^2$ levels regression but fail a cointegration test have blown up many trading books.

### 5. The factor zoo and the multiple-testing crisis

Academic finance has published hundreds of "factors" claimed to predict returns -- the so-called *factor zoo*. Campbell Harvey and co-authors argued in the 2010s that most are false positives: with hundreds of researchers each running thousands of regressions, plenty will produce a t-stat above 2 by pure chance. Harvey's prescription was to raise the significance bar to roughly $t > 3$ and to use *deflated* statistics that account for how many tests were run. This is the t-stat misconception from section 1 at industrial scale: a single t-stat of 2.5 means little when it is the best of 10,000 tried, and a desk that trades every "significant" factor it finds is mostly trading noise dressed up as signal.

### 6. The COVID volatility spike and heteroskedasticity

In March 2020, daily equity moves of 5-10% became routine for several weeks before markets calmed. Any beta or factor regression estimated through that window confronted extreme heteroskedasticity: a handful of crisis days with residuals an order of magnitude fatter than the surrounding calm. Desks that ran plain OLS over windows spanning the spike got betas yanked toward the crisis behavior, and naive standard errors that badly understated the true uncertainty. The fix from sections 2 and 3 -- robust standard errors, or WLS down-weighting the wild days -- was the difference between a hedge ratio that reflected the stock's normal behavior and one distorted by a few weeks of panic. It is the live version of the WLS worked example.

## When this matters to you

If you ever build a model that uses one stream of numbers to explain or predict another -- a trading signal, a risk model, a forecast, even a back-of-envelope "how much does X move with Y" -- you are doing regression, and every trap in this post is waiting for you. The discipline that separates a model that makes money from one that loses it is not running the regression; anyone can do that. It is *refusing to trust the output by default*: recomputing the standard errors before believing a t-stat, checking the VIF before believing a coefficient, working in returns instead of levels to dodge spurious fit, regularizing when factors are numerous or alike, and -- above all -- judging the model on data it has never seen rather than on the data it was fit to. A regression's confidence is not your confidence; the math will tell you whatever you ask it, and your job is to ask honestly.

Be clear-eyed about the risk. Every fix in this post reduces *one* failure mode while assuming the others away; a robust standard error does nothing for collinearity, and ridge does nothing for autocorrelation. Real markets serve up all of these at once, and no estimator is bulletproof -- the LTCM and factor-zoo episodes are what happens when smart people trust a fitted relationship past its breaking point. This is educational material about how a standard tool behaves, not a recommendation to trade anything; any strategy that can make money on a regression can lose it just as fast when the relationship the regression measured stops holding.

For where to go next on this blog: start with [linear regression from first principles](/blog/trading/quantitative-finance/linear-regression-deep-quant-interviews) for the single-variable derivation and the interview traps, then read [SVD, the pseudo-inverse, and least squares](/blog/trading/math-for-quants/svd-least-squares-regression-math-for-quants) for the linear-algebra machinery behind collinearity and the condition number. To turn a fitted coefficient into a tradeable signal, [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) walks the construction end to end, and [evaluating alpha signals with IC, Sharpe, and turnover](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research) is the out-of-sample discipline that catches the overfit models this post taught you to fear. Read those four together and you will have the full arc: fit the line, distrust it correctly, and test whether it survives contact with the future.
