---
title: "Linear regression from first principles for quant interviews"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch guide to the single most-used tool on a quant desk and a guaranteed interview topic: derive ordinary least squares by minimizing squared residuals, understand it as an orthogonal projection, state the Gauss-Markov assumptions and why OLS is BLUE, read R-squared and t-statistics correctly, and apply it to betas, hedge ratios, and factor models -- with fully worked dollar examples and seven solved interview problems."
tags:
  [
    "linear-regression",
    "ordinary-least-squares",
    "quant-interviews",
    "capm-beta",
    "factor-models",
    "hedge-ratio",
    "r-squared",
    "omitted-variable-bias",
    "gauss-markov",
    "statistics",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** -- linear regression is the workhorse of every quant desk and a guaranteed interview topic. If you can derive it, picture it, and read its diagnostics, you can answer most of the statistics questions Jane Street, Two Sigma, Citadel, DE Shaw, and AQR will throw at you.
>
> - **What it is**: the line (or hyperplane) that minimizes the sum of squared vertical gaps between your data and the fit. Those gaps are the *residuals*; squaring and summing them is *ordinary least squares* (OLS).
> - **The closed form**: in one variable the slope is $b = \text{Cov}(x,y)/\text{Var}(x)$; in matrix form $\hat{\beta} = (X^\top X)^{-1}X^\top y$. Both fall straight out of setting the derivative of the squared error to zero.
> - **The geometry**: OLS is the *orthogonal projection* of your outcome vector onto the column space of your regressors. The residual is perpendicular to everything you used to explain it.
> - **The diagnostics**: $R^2$ is the share of variance the line explains; the *t-statistic* tells you whether a coefficient is real or noise (rule of thumb: $|t| > 2$ is "significant").
> - **The traps**: $R^2$ worship, mistaking correlation for causation, multicollinearity inflating your standard errors, and omitted-variable bias flipping a coefficient's sign. Knowing these is what separates a hire from a reject.
> - **The one number to remember**: a stock with $\beta = 1.3$ needs you to short \$1,300,000 of the index to hedge a \$1,000,000 long. That is a regression slope doing real work.

## Why this one tool is everywhere

Walk onto any quantitative trading floor and eavesdrop, and within ten minutes you will hear someone say "the beta is one-three", "we regressed the signal on forward returns", "after we control for the sector factor the alpha disappears", or "the hedge ratio drifted, re-fit it". Every one of those sentences is a linear regression. It is, without much competition, the most-used statistical tool in finance -- not because it is fancy, but because it is the simplest honest answer to a question the desk asks a hundred times a day: *if this thing moves, how much does that thing move with it?*

That is also exactly why interviewers love it. A good linear-regression conversation reveals whether you actually understand statistics or just memorized formulas. The questions ladder beautifully: they can start with "fit a line to these five points" and end with "your factor model has two regressors that are 99% correlated -- what happens to your t-stats and why?" A candidate who can derive OLS, draw the projection picture, recite the assumptions, and read a regression table is a candidate who can be trusted with a model. One who can only recall "$y = mx + b$" is not.

![Linear regression is the single estimator behind betas, hedge ratios, factor models, and signal fitting on a quant desk; fit one line and read out all four.](/imgs/blogs/linear-regression-deep-quant-interviews-1.png)

The diagram above is the mental model for the whole post. You take paired data -- a column of $x$ values and a column of $y$ values -- run it through one estimator (OLS), and out the other side comes a number that the desk calls a different thing depending on what $x$ and $y$ were: a *beta* when $x$ is the market and $y$ is a stock, a *hedge ratio* when you want to know how much of one instrument offsets another, a *factor loading* when $x$ is a style like value or momentum, a *signal coefficient* when $x$ is your prediction and $y$ is what actually happened. One machine, four jobs. Master the machine and all four jobs come for free.

We are going to build this from absolute zero. No statistics background is assumed; every term gets defined the first time it appears. We will derive the formula two ways (calculus and geometry), state the assumptions that make it trustworthy, learn to read the diagnostics, walk through the classic traps, and solve seven interview problems with real numbers. By the end you should be able to whiteboard any of it.

This is educational material about how a standard tool works, not investment advice.

## Foundations: the line of best fit, residuals, and least squares

Let us start with the most concrete possible situation. You have a handful of paired measurements. Maybe $x$ is how many hours a student studied and $y$ is their exam score; maybe $x$ is yesterday's market return and $y$ is a stock's return. You suspect they are related, and you want to summarize that relationship with a straight line.

A straight line is described by two numbers: an **intercept** $a$ (where the line crosses the vertical axis, i.e. the predicted $y$ when $x = 0$) and a **slope** $b$ (how much $y$ changes when $x$ goes up by one). We write the line as

$$\hat{y} = a + b x.$$

The little hat on $\hat{y}$ means "predicted" -- it is the height of the line at a given $x$, *not* the actual data point. The actual data point is $y$, with no hat. The two will almost never be equal, because real data is noisy.

The gap between what actually happened and what the line predicted is the **residual**:

$$e_i = y_i - \hat{y}_i = y_i - (a + b x_i).$$

The subscript $i$ just indexes the data points: $e_1$ is the residual for the first point, $e_2$ for the second, and so on. A residual is *signed*: positive if the point sits above the line (the line under-predicted), negative if it sits below.

Now, what makes a line "best"? Intuitively, the best line is the one that sits as close as possible to all the points at once -- the one that makes the residuals small. But "small" is ambiguous. We could try to minimize the sum of the residuals, but positive and negative residuals would cancel, and a terrible line could have a sum of zero. We could minimize the sum of the *absolute values* of the residuals, which is a real method (it is called *least absolute deviations*), but absolute values are awkward to differentiate and the answer can jump around. The choice that wins, for reasons both mathematical and historical, is to minimize the sum of the *squared* residuals:

$$\text{SSR} = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} \big(y_i - a - b x_i\big)^2.$$

Squaring does two nice things: it makes every term positive (so they cannot cancel), and it punishes big misses much more than small ones (a residual of 2 contributes 4, a residual of 4 contributes 16). The line that minimizes this sum is the **ordinary least squares** line, and SSR stands for *sum of squared residuals*. "Ordinary" distinguishes it from weighted or generalized variants we will not need.

![Ordinary least squares picks the slope and intercept that minimize the sum of squared vertical residuals between the data and the fitted line.](/imgs/blogs/linear-regression-deep-quant-interviews-2.png)

The figure shows the whole idea on a tiny dataset (the one we will compute by hand in a moment). The blue squares are the data, the solid line is the best fit, and the dashed vertical segments are the residuals -- the squared lengths of which we are making as small as possible. Notice the residuals are vertical, not perpendicular to the line. That is deliberate: we are predicting $y$ from $x$, so we only care about errors in the $y$ direction. The lavender dot marks $(\bar{x}, \bar{y})$, the point of averages, through which the OLS line always passes -- a fact worth remembering for interviews.

#### Worked example: fitting a line by hand

Let us fit $\hat{y} = a + bx$ to five points: $(1, 2), (2, 4), (3, 5), (4, 4), (5, 5)$. We will do every step.

**Step 1 -- the means.** The mean of $x$ is $\bar{x} = (1+2+3+4+5)/5 = 15/5 = 3$. The mean of $y$ is $\bar{y} = (2+4+5+4+5)/5 = 20/5 = 4$.

**Step 2 -- the deviations and their products.** For each point compute $(x_i - \bar{x})$ and $(y_i - \bar{y})$, then multiply them:

| $x_i$ | $y_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | product | $(x_i-\bar{x})^2$ |
|---|---|---|---|---|---|
| 1 | 2 | $-2$ | $-2$ | $4$ | $4$ |
| 2 | 4 | $-1$ | $0$ | $0$ | $1$ |
| 3 | 5 | $0$ | $1$ | $0$ | $0$ |
| 4 | 4 | $1$ | $0$ | $0$ | $1$ |
| 5 | 5 | $2$ | $1$ | $2$ | $4$ |

**Step 3 -- the slope.** Sum the product column: $4 + 0 + 0 + 0 + 2 = 6$. Sum the last column: $4 + 1 + 0 + 1 + 4 = 10$. The slope is the ratio:

$$b = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{6}{10} = 0.6.$$

**Step 4 -- the intercept.** Because the line passes through $(\bar{x}, \bar{y})$, we have $\bar{y} = a + b\bar{x}$, so $a = \bar{y} - b\bar{x} = 4 - 0.6 \times 3 = 4 - 1.8 = 2.2$.

The best-fit line is $\hat{y} = 2.2 + 0.6x$. Every extra unit of $x$ buys you 0.6 more $y$, on average, in this dataset. The intuition this teaches: **the slope is a ratio of co-movement to spread -- how much $x$ and $y$ move together, divided by how much $x$ moves on its own.**

The numerator in step 3, $\sum (x_i - \bar{x})(y_i - \bar{y})$, is (up to a constant) the **covariance** of $x$ and $y$ -- a measure of how the two move together. The denominator, $\sum (x_i - \bar{x})^2$, is (up to the same constant) the **variance** of $x$ -- how much $x$ spreads out. So the simple-regression slope is exactly

$$b = \frac{\text{Cov}(x, y)}{\text{Var}(x)}.$$

This formula is worth memorizing cold; it shows up in interviews constantly, and it is the seed of the beta formula we will meet later.

## Deriving OLS by minimizing squared residuals

We pulled the slope formula out of a table, but where does it actually come from? It comes from calculus: we want the $(a, b)$ that make SSR as small as possible, so we take derivatives and set them to zero.

Think about SSR as a function of the slope $b$ (hold the intercept at its optimal value for a second). It is a sum of squared terms, each of which is a parabola in $b$, so the total is a parabola too -- a smooth U-shaped bowl with exactly one lowest point. There is no risk of getting stuck in a false minimum; the bottom of the bowl is *the* answer.

![The sum of squared residuals is a convex parabola in the slope with exactly one minimizing value, found by setting the derivative to zero.](/imgs/blogs/linear-regression-deep-quant-interviews-3.png)

The figure plots SSR against the slope $b$ for our five-point dataset (with the intercept fixed at its optimum, $a = 2.2$). It is a clean parabola bottoming out at $b = 0.6$, where SSR equals 2.0 -- the smallest squared error any slope can achieve here. Move the slope away from 0.6 in either direction and the error climbs. That single bottom is what "set the derivative to zero" finds.

#### Worked example: the normal equations

Let us actually differentiate. We minimize

$$\text{SSR}(a, b) = \sum_i \big(y_i - a - b x_i\big)^2.$$

Take the partial derivative with respect to the intercept $a$ and set it to zero. Using the chain rule, each term differentiates to $2(y_i - a - bx_i) \times (-1)$:

$$\frac{\partial \text{SSR}}{\partial a} = -2 \sum_i (y_i - a - b x_i) = 0 \;\Rightarrow\; \sum_i (y_i - a - b x_i) = 0.$$

This says the residuals sum to zero -- the line is balanced, with as much data above it as below in a squared sense. Now the partial with respect to the slope $b$; each term differentiates to $2(y_i - a - bx_i) \times (-x_i)$:

$$\frac{\partial \text{SSR}}{\partial b} = -2 \sum_i x_i (y_i - a - b x_i) = 0 \;\Rightarrow\; \sum_i x_i (y_i - a - b x_i) = 0.$$

This says the residuals are uncorrelated with $x$ -- there is no leftover linear relationship for the line to exploit. These two equations are the **normal equations**. Solving them (the first gives $a = \bar{y} - b\bar{x}$; substitute into the second and simplify) yields exactly the formulas from our table:

$$b = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2} = \frac{\text{Cov}(x,y)}{\text{Var}(x)}, \qquad a = \bar{y} - b\bar{x}.$$

The intuition: **the best line is the one whose residuals carry no remaining linear information -- they average to zero and they do not line up with the predictor.**

### The matrix form for many predictors

Real factor models have more than one predictor. A stock's return might depend on the market *and* a value factor *and* a momentum factor. To handle $k$ predictors at once we stack everything into matrices. Let $y$ be the column vector of outcomes ($n$ rows, one per observation), and let $X$ be the **design matrix**: $n$ rows, and one column per predictor -- plus a leading column of all 1s to carry the intercept. The coefficient vector $\beta$ collects the intercept and all the slopes.

The model is $y = X\beta + \varepsilon$, where $\varepsilon$ is the vector of residuals. We minimize the squared length of the residual vector, $\|y - X\beta\|^2$. Differentiating with respect to the vector $\beta$ and setting it to zero gives the matrix normal equation $X^\top X \beta = X^\top y$, and as long as $X^\top X$ can be inverted (more on when it cannot, later), the unique solution is

$$\hat{\beta} = (X^\top X)^{-1} X^\top y.$$

This one line is the entire machinery of multiple regression. It looks intimidating but it is just the multi-dimensional version of "covariance over variance": $X^\top y$ is a bundle of covariances between each predictor and the outcome, and $(X^\top X)^{-1}$ is the multi-dimensional analogue of "divide by the variance", which crucially also *corrects for the predictors overlapping with each other*. When the predictors are uncorrelated, $X^\top X$ is diagonal and each coefficient is just its own simple-regression slope. When they overlap, the inverse untangles them -- the source of both the power and the fragility of multiple regression.

To make the matrix form concrete, picture the smallest non-trivial case: three observations and one predictor plus an intercept. Then $X$ is a $3 \times 2$ matrix whose first column is $(1, 1, 1)^\top$ and whose second column holds your three $x$ values. $X^\top X$ is a $2 \times 2$ matrix: its top-left entry is $n = 3$ (the count), its off-diagonal entries are $\sum x_i$, and its bottom-right entry is $\sum x_i^2$. Inverting a $2 \times 2$ matrix is a one-liner (swap the diagonal, negate the off-diagonal, divide by the determinant), and when you carry the algebra through, the slope you get is *exactly* $\text{Cov}(x,y)/\text{Var}(x)$ again. The matrix form is not a different method -- it is the same least-squares idea written so it scales to any number of predictors without re-deriving anything. That is the payoff: one formula, $\hat{\beta} = (X^\top X)^{-1}X^\top y$, that you implement once and reuse for a one-factor beta or a fifty-factor risk model.

## The geometry: regression as orthogonal projection

Here is the single most elegant way to understand OLS, and the answer interviewers most want to hear when they ask "what is regression, geometrically?"

Picture your outcome data not as $n$ separate numbers but as one arrow -- a single vector $y$ living in an $n$-dimensional space, one dimension per observation. Your predictors are also vectors in that same space. The set of *all* the predictions you could possibly make -- every combination $X\beta$ as you dial the coefficients up and down -- forms a flat subspace, like a plane passing through the origin. This subspace is called the **column space** of $X$, because it is everything you can reach by mixing the columns of $X$.

Your actual data $y$ almost certainly does *not* lie in that plane -- if it did, the fit would be perfect with zero error. So the question becomes: of all the points *in* the plane, which one is closest to $y$? That closest point is your prediction $\hat{y}$, and the answer your geometric intuition already knows: you drop a perpendicular from $y$ straight down onto the plane. The foot of that perpendicular is the closest point.

![OLS projects the response vector orthogonally onto the column space of the regressors; the residual that remains is perpendicular to that subspace.](/imgs/blogs/linear-regression-deep-quant-interviews-4.png)

The figure makes this concrete. The gray plane is the column space -- every fitted vector $X\beta$ you could produce. The vector $y$ points up out of the plane (the actual returns, which your model cannot perfectly reproduce). The fitted vector $\hat{y} = X\beta$ lies *in* the plane. The dashed red arrow is the residual $e = y - \hat{y}$, and the little square at its foot marks the right angle: **the residual is perpendicular to the column space.** That perpendicularity is not an accident -- it is the geometric statement of the normal equations. Algebraically, "perpendicular to every column of $X$" means $X^\top e = 0$, which rearranges to $X^\top(y - X\beta) = 0$, which solves to $\hat{\beta} = (X^\top X)^{-1}X^\top y$. The calculus and the geometry are the same fact wearing different clothes.

This picture pays for itself in interviews. "Why are the residuals uncorrelated with the predictors?" Because the residual is *perpendicular* to the space the predictors span -- it is the part of $y$ your regressors literally cannot reach, so by construction they carry no information about it. "Why does adding a predictor never increase SSR?" Because enlarging the column space (adding a dimension to the plane) can only bring the closest point *closer* to $y$, never push it away. The geometry answers these instantly; the algebra makes you grind.

## The Gauss-Markov assumptions and why OLS is BLUE

OLS gives you an answer for any data you feed it. But is it a *good* answer? It depends on whether a handful of assumptions about the data hold. These are the **Gauss-Markov assumptions**, and when they hold, a famous theorem (the Gauss-Markov theorem) guarantees that OLS is **BLUE**: the **B**est **L**inear **U**nbiased **E**stimator. Let us unpack that acronym, because interviewers ask exactly what it means.

- **Unbiased** means that if you could re-run the experiment many times, the *average* of your estimated coefficients would equal the true coefficient. No systematic over- or under-shooting.
- **Linear** means we restrict attention to estimators that are linear functions of the outcome $y$ (OLS is one such -- $\hat{\beta} = (X^\top X)^{-1}X^\top y$ is linear in $y$).
- **Best** means *minimum variance*: among all linear unbiased estimators, OLS has the smallest sampling variance -- it is the most precise, the least jumpy from sample to sample.

So BLUE is a strong promise: not just "right on average" but "right on average *and* as tightly pinned down as any linear unbiased method can be." Here are the assumptions that buy you that promise.

![Five assumptions -- linearity, exogeneity, no autocorrelation, homoskedasticity, and full rank -- make ordinary least squares the best linear unbiased estimator.](/imgs/blogs/linear-regression-deep-quant-interviews-5.png)

The panel above is a cheat sheet you should be able to reproduce. Going row by row:

1. **Linearity.** The true relationship is linear in the coefficients: $y = X\beta + \varepsilon$. (It can be nonlinear in the *variables* -- you are allowed to use $x^2$ as a column -- but linear in the betas.) If the real link is curved and you fit a straight line, your line is systematically wrong.

2. **Exogeneity** (the big one): $E(\varepsilon \mid X) = 0$. In words, the errors are unrelated to the predictors -- whatever your model leaves out is not correlated with what you put in. When this fails you get *omitted-variable bias*, the trap we devote a whole section to below. This is the assumption that does the heavy lifting and the one that breaks most often in finance.

3. **No autocorrelation**: $\text{Cov}(\varepsilon_i, \varepsilon_j) = 0$ for different observations. One observation's error tells you nothing about the next one's. In financial *time series* this is routinely violated -- today's surprise is correlated with yesterday's -- which is why desks reach for Newey-West or other corrected standard errors.

4. **Homoskedasticity**: $\text{Var}(\varepsilon_i) = \sigma^2$, constant for every observation. The error spread does not widen or shrink across the data. Markets violate this too -- volatility clusters, so error variance balloons in turbulent periods (*heteroskedasticity*).

5. **Full rank**: $X^\top X$ is invertible, i.e. no predictor is an exact linear combination of the others. If two columns are identical (or one is twice another), the matrix cannot be inverted and the coefficients are not even defined. The near-violation of this -- predictors that are *almost* copies -- is *multicollinearity*, another section below.

The honest framing for an interview: assumptions 1, 2, and 5 are about getting the *right answer* (unbiasedness and existence); assumptions 3 and 4 are about getting the *right uncertainty* (correct standard errors). You can have unbiased coefficients with wrong standard errors if 3 or 4 fail -- which means your point estimates are fine but your t-stats lie to you. That distinction is a favorite interview probe.

## R-squared and adjusted R-squared

You have a fitted line. How well does it fit? The headline number is $R^2$ (pronounced "R-squared"), and it answers a precise question: **what fraction of the variation in $y$ does the line explain?**

To define it we split the total variation in $y$ into two pieces. Start with the **total sum of squares**, how much $y$ varies around its own mean:

$$\text{TSS} = \sum_i (y_i - \bar{y})^2.$$

This is what you would be stuck with if you had no model and just guessed $\bar{y}$ for everything. Now the regression carves TSS into two parts. The **explained sum of squares** is the variation the line captures -- how much the *predictions* vary around the mean:

$$\text{ESS} = \sum_i (\hat{y}_i - \bar{y})^2,$$

and the **residual sum of squares** is the variation left over, which is exactly our old friend SSR:

$$\text{RSS} = \sum_i (y_i - \hat{y}_i)^2.$$

A clean identity holds: $\text{TSS} = \text{ESS} + \text{RSS}$. Total variation equals explained plus unexplained. (This is the Pythagorean theorem in disguise -- recall the residual is perpendicular to the fit, so squared lengths add.) Then

$$R^2 = \frac{\text{ESS}}{\text{TSS}} = 1 - \frac{\text{RSS}}{\text{TSS}}.$$

It ranges from 0 (the line explains nothing -- you might as well have guessed the mean) to 1 (perfect fit, every point on the line).

![Total variation TSS splits into explained ESS and unexplained RSS, and R-squared is the explained share -- here 3.6 out of 6, so R-squared is 0.6.](/imgs/blogs/linear-regression-deep-quant-interviews-6.png)

#### Worked example: computing R-squared

Take our five-point dataset with fit $\hat{y} = 2.2 + 0.6x$. First the fitted values: at $x = 1,2,3,4,5$ they are $\hat{y} = 2.8, 3.4, 4.0, 4.6, 5.2$.

**Total sum of squares** (deviations of $y$ from $\bar{y} = 4$): the $y$ values are $2,4,5,4,5$, so deviations are $-2, 0, 1, 0, 1$, squared $4, 0, 1, 0, 1$, summing to $\text{TSS} = 6$.

**Residual sum of squares** (deviations of $y$ from $\hat{y}$): residuals are $2-2.8, 4-3.4, 5-4.0, 4-4.6, 5-5.2 = -0.8, 0.6, 1.0, -0.6, -0.2$. Squared: $0.64, 0.36, 1.00, 0.36, 0.04$, summing to $\text{RSS} = 2.4$. (Notice this matches the bottom of the parabola in the earlier figure -- the minimum SSR was 2.0... and here it is 2.4; the small difference is because the figure fixed the intercept while we now use the jointly optimal pair. The lesson stands: RSS is the leftover.)

**Explained sum of squares**: $\text{ESS} = \text{TSS} - \text{RSS} = 6 - 2.4 = 3.6$.

Therefore $R^2 = \text{ESS}/\text{TSS} = 3.6 / 6 = 0.60$. The line explains 60% of the variation in $y$; the remaining 40% is noise the line cannot reach. The intuition: **$R^2$ is the fraction of the wiggle in $y$ your predictors account for -- nothing more, nothing less.**

### Why you need adjusted R-squared

There is a catch that interviewers probe: **$R^2$ never decreases when you add a predictor**, even a useless one. Recall the geometry -- adding a column enlarges the column space, so the closest point can only get closer to $y$. Throw in enough random columns and $R^2$ creeps toward 1 while you have learned nothing. This is *overfitting*, and naive $R^2$ rewards it.

The fix is **adjusted $R^2$**, which penalizes each extra predictor:

$$\bar{R}^2 = 1 - \frac{\text{RSS}/(n - k - 1)}{\text{TSS}/(n - 1)},$$

where $n$ is the number of observations and $k$ the number of predictors (excluding the intercept). Each predictor you add costs a degree of freedom in the numerator; unless it earns its keep by cutting RSS enough, adjusted $R^2$ *falls*. Reporting adjusted $R^2$ on a multi-factor model, and explaining *why* over plain $R^2$, is a small thing that signals real understanding.

## Standard errors, t-statistics, and significance

A coefficient estimate is useless without a sense of how reliable it is. If a regression spits out $\hat{\beta} = 0.5$, is that a solid 0.5, or could the true value just as easily be $-0.3$? That is what the **standard error** answers.

The standard error of a coefficient, written $\text{SE}(\hat{\beta})$, is the standard deviation of the coefficient's *sampling distribution* -- how much your estimate would bounce around if you re-drew the data many times. A small standard error means a tightly pinned-down estimate; a large one means the data barely constrains it. Mechanically it shrinks with more data and with more spread-out predictors, and it grows with noisier residuals.

To judge whether a coefficient is *real* -- distinguishable from zero -- we form the **t-statistic**:

$$t = \frac{\hat{\beta}}{\text{SE}(\hat{\beta})}.$$

It measures how many standard errors the estimate sits away from zero. A $t$ of 0.4 means the estimate is well within the noise; a $t$ of 8 means the estimate is eight standard errors out, far too big to be a fluke. The **null hypothesis** here is $\beta = 0$ (the predictor has no effect); a large $|t|$ lets you *reject* that null and conclude the effect is real -- *statistically significant*.

![Under the null that beta equals zero the t-statistic follows a bell curve; an absolute value above 1.96 lands in the 5 percent tails and is called significant.](/imgs/blogs/linear-regression-deep-quant-interviews-7.png)

The figure shows the logic. If the true coefficient really were zero, the t-statistic would scatter around zero following the bell-shaped curve drawn (a *t-distribution*, which for large samples is essentially the standard normal). The green central band is where 95% of t-values land *if the null is true*. The red tails beyond $\pm 1.96$ together hold 5% of the probability. So the rule: **if $|t| > 1.96$, the result would be surprising under "no effect", and we call it significant at the 5% level.** Quants round this to the famous "$|t| > 2$" rule of thumb. A related summary is the **p-value** -- the probability of seeing a t-statistic at least this extreme if the null were true; $p < 0.05$ is the same threshold from the other direction.

#### Worked example: reading a regression line

Suppose you regress a momentum signal on next-day stock returns and the software reports a coefficient $\hat{\beta} = 0.018$ with a standard error of $0.004$. The t-statistic is $t = 0.018 / 0.004 = 4.5$. That is well past 2, so the signal has a statistically significant relationship with forward returns -- it is unlikely to be noise. Now suppose a second signal reports $\hat{\beta} = 0.011$ with standard error $0.009$. Its t-stat is $0.011/0.009 \approx 1.2$, comfortably inside the noise band; you cannot reject "this signal does nothing" at the 5% level, even though its point estimate is positive. The intuition: **a coefficient's size means little until you divide it by its standard error -- a big estimate with a big standard error is just a loud guess.**

A crucial caveat that earns points: statistical significance is *not* economic significance. A signal can be wildly significant ($t = 10$) and still predict only a microscopic, untradeable edge if its coefficient is tiny relative to costs. And with thousands of signals tested, some will clear $t > 2$ by pure luck -- the *multiple-testing* problem, which is why serious desks demand far higher t-thresholds (often $t > 3$ or more) for a signal to go live.

It also helps to know roughly where standard errors come from, because interviewers sometimes ask "what would shrink that standard error?" For the simple-regression slope, the standard error is approximately

$$\text{SE}(\hat{b}) = \frac{\sigma}{\sqrt{\sum_i (x_i - \bar{x})^2}},$$

where $\sigma$ is the standard deviation of the residuals. Read the formula and the levers fall out: the standard error *shrinks* when the residuals are smaller (a cleaner relationship, smaller $\sigma$), when you have more data (more terms in the sum), and when your predictor is more spread out (a bigger denominator). That last lever surprises people -- a predictor that barely varies gives you almost no leverage to estimate its slope, no matter how much data you have, because there is little $x$-variation to attach the $y$-variation to. This is why experiments deliberately spread their inputs and why a factor that was nearly constant over your sample will have a hopeless standard error even if it truly matters.

## Multicollinearity: when predictors step on each other

Recall full rank: $X^\top X$ must be invertible. When two predictors are *perfectly* correlated -- say you accidentally include both "return in percent" and "return in basis points", which are the same thing times 100 -- the matrix is singular and the regression simply cannot run; the coefficients are not identified, because there is no way to split credit between two columns that move identically.

The more common and more insidious case is **multicollinearity**: predictors that are *highly but not perfectly* correlated. The regression runs, but it becomes wildly unstable. The math still tries to give each predictor its own coefficient, but when two columns carry nearly the same information, tiny changes in the data swing the coefficients enormously -- the model cannot tell which of the twins deserves the credit.

The damage shows up in the **standard errors**, which inflate. The inflation is captured by the **variance inflation factor** (VIF). For a predictor with correlation $\rho$ to the others, $\text{VIF} = 1/(1 - R_j^2)$, where $R_j^2$ is the $R^2$ from regressing that predictor on all the others. The standard error scales with $\sqrt{\text{VIF}}$. If a predictor is 99% explained by the others, $R_j^2 = 0.98$ gives $\text{VIF} = 1/(1-0.98) = 50$, so the standard error is $\sqrt{50} \approx 7.1$ times larger than it would be with independent predictors.

![When two predictors carry nearly the same information the variance inflation factor explodes and the coefficient standard errors balloon, turning a sharp estimate into a useless one.](/imgs/blogs/linear-regression-deep-quant-interviews-8.png)

The figure contrasts the two worlds. On the left, two independent regressors (correlation 0, VIF 1): the coefficient is pinned down to a tight green interval, $\hat{\beta} = 1.00 \pm 0.10$, t-stat 10, clearly significant. On the right, the *same* coefficient with two regressors that are 99% correlated (VIF 50): the standard error blows up 7-fold to 0.71, the interval sprawls in red, and the t-stat collapses to $1.00/0.71 \approx 1.4$ -- no longer significant. **The point estimate did not move; the precision evaporated.** This is the signature of multicollinearity, and the interview-ready summary is: *it does not bias your coefficients, it just makes them so imprecise you cannot trust any individual one.* The cures are to drop one of the redundant predictors, combine them (e.g. into a single factor via principal components), or get more data.

## Omitted-variable bias: when a coefficient lies about its sign

Multicollinearity inflates your uncertainty but leaves your estimates unbiased. Omitted-variable bias is nastier: it makes your estimates flat *wrong*, sometimes wrong enough to flip a coefficient's sign. This is the exogeneity assumption ($E(\varepsilon \mid X) = 0$) failing, and it is the deepest trap in applied regression.

Here is the mechanism. Suppose the true world has two drivers of an outcome $Y$: a predictor $X$ you include, and a confounder $Z$ you leave out (maybe because you did not measure it, or did not think of it). If $Z$ affects $Y$ *and* $Z$ is correlated with $X$, then when you regress $Y$ on $X$ alone, the coefficient on $X$ absorbs not just $X$'s own effect but also a smuggled-in piece of $Z$'s effect. Your $\hat{\beta}$ is biased, and the size of the bias is

$$\text{bias} = (\text{true effect of } Z \text{ on } Y) \times (\text{coefficient of } Z \text{ regressed on } X).$$

![When an omitted driver Z affects both the included regressor X and the outcome Y, its true effect leaks into the estimated coefficient on X and biases it.](/imgs/blogs/linear-regression-deep-quant-interviews-9.png)

The causal triangle above is the picture to draw on the whiteboard. $Z$ (the omitted market regime, in amber) drives both $X$ (firm leverage, blue) and $Y$ (stock returns, green). When you leave $Z$ out, the only path you fit is the dashed $X \to Y$ arrow, and it silently carries the $Z \to X \to$ "credit for $Z \to Y$" contamination. The estimate on $X$ is biased by the leaked path.

#### Worked example: a sign flip

Let us make the bias flip a sign, which is the version that wins interviews because it is so counterintuitive. Suppose you want to know whether higher *leverage* (debt) makes a firm's stock return higher or lower. You regress returns on leverage across many firms and -- to your surprise -- get a coefficient of $\hat{\beta} = +0.80$: more leverage, higher returns. You are about to conclude "leverage is bullish."

But there is an omitted variable: the *market regime* $Z$. In booming markets, firms lever up aggressively (so $Z$ raises leverage, a positive $Z \to X$ link) *and* booming markets lift all returns (a strong positive $Z \to Y$ link). Leverage was just riding the boom. When you add the market-regime control to the regression -- so that the coefficient on leverage now means "holding the regime fixed" -- the leverage coefficient flips to $\hat{\beta} = -0.30$: within any given regime, more debt actually *drags* returns down (the genuine risk-of-distress effect). The naive +0.80 was the true −0.30 plus a +1.10 contamination smuggled in by the omitted boom.

![Adding the omitted market-regime control reverses the leverage coefficient from a spurious positive 0.8 to its true negative 0.3, a full swing of more than one unit.](/imgs/blogs/linear-regression-deep-quant-interviews-12.png)

The bars make the swing vivid: the naive regression (red, to the right of zero) says $+0.80$; the controlled regression (green, to the left) says $-0.30$. Same data, opposite conclusion, because one regression honored the confounder and the other did not. The intuition every quant must internalize: **a regression coefficient means "holding the other included variables fixed" -- and if the variable that matters is not included, the coefficient is answering a question you did not ask.**

## Interpreting coefficients: "holding others fixed"

That phrase deserves its own beat, because misreading it causes more bad trades than almost any other statistical error. In a multiple regression $\hat{y} = b_0 + b_1 x_1 + b_2 x_2$, the coefficient $b_1$ is *not* "the effect of $x_1$ on $y$". It is "the effect of $x_1$ on $y$ **holding $x_2$ fixed**" -- the *partial* effect, after $x_2$ has already accounted for whatever it can. Change which other variables are in the model and $b_1$ can change, even reverse, as we just saw. A coefficient is a statement about a *specific* model, not a law of nature.

This is also why "controlling for" a variable is so powerful and so abusable. Adding the right control (the genuine confounder) de-biases your estimate. Adding the *wrong* control -- a variable that sits on the causal path *from* $x$ *to* $y$, called a *mediator*, or a *collider* that opens a spurious path -- can *introduce* bias where there was none. The discipline of choosing controls is the discipline of drawing the causal diagram first and fitting second. Interviewers who ask "should we control for $W$?" are testing exactly this judgment, and the correct answer is always "it depends on the causal structure -- let me draw it."

## In the interview room

Now the part you came for: fully solved problems in the style desks actually use. Work through each with pencil before reading the solution.

#### Worked example: derive the simple-regression slope from scratch

*Problem.* "Without quoting a formula, derive the slope of the least-squares line for one predictor."

*Solution.* Write the objective: minimize $S(b) = \sum_i (y_i - a - b x_i)^2$ over $a$ and $b$. Set $\partial S / \partial a = 0$: this gives $\sum_i (y_i - a - bx_i) = 0$, i.e. $a = \bar{y} - b\bar{x}$ (the line passes through the means). Substitute that back and set $\partial S/\partial b = 0$: after replacing $a$, the condition becomes $\sum_i (x_i - \bar{x})(y_i - \bar{y} - b(x_i - \bar{x})) = 0$, which solves to

$$b = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2} = \frac{\text{Cov}(x,y)}{\text{Var}(x)}.$$

The thing to *say out loud*: "the slope is the covariance of $x$ and $y$ divided by the variance of $x$, and it falls straight out of forcing the residuals to be orthogonal to the predictor." That sentence is the whole interview.

#### Worked example: a stock's beta and the dollar hedge

*Problem.* "You hold \$1,000,000 of a stock. Its daily returns regressed on the market's daily returns give a slope of 1.3. How much of the index do you short to hedge the market risk, and what residual risk remains?"

*Solution.* The regression slope *is* the **beta**: $\beta = \text{Cov}(r_{\text{stock}}, r_{\text{mkt}}) / \text{Var}(r_{\text{mkt}}) = 1.3$. It says that for every 1% the market moves, the stock moves 1.3% on average. To neutralize the market exposure of a \$1,000,000 long, you short a dollar amount such that the two market exposures cancel:

$$\text{hedge} = \beta \times \text{position} = 1.3 \times \$1{,}000{,}000 = \$1{,}300{,}000.$$

So you short \$1,300,000 of the index. After hedging, a 1% market move costs the stock $1.3\% \times \$1{,}000{,}000 = \$13{,}000$ and earns the short $1\% \times \$1{,}300{,}000 = \$13{,}000$ -- they cancel. What is left is the **residual** (idiosyncratic) return -- the part of the stock's move the market does not explain, which is exactly the regression residual $e$. If the regression's $R^2$ was 0.6, then 60% of the stock's return variance was market-driven (now hedged) and 40% remains as stock-specific risk you are choosing to keep.

![The slope of a stock's returns regressed on the market gives its beta of 1.3, which sets the dollar hedge of 1.3 million dollars short against a 1 million dollar long.](/imgs/blogs/linear-regression-deep-quant-interviews-10.png)

The scatter shows the regression that produces the beta: each dot is one day, the x-axis is the market's return and the y-axis is the stock's, and the line's slope of 1.3 is the beta. The intuition: **beta is a regression slope, and a regression slope is a hedge ratio -- the same number tells you how the stock co-moves and how much index to short.**

#### Worked example: compute R-squared from sums of squares

*Problem.* "A regression has total sum of squares 200 and residual sum of squares 50. What is the $R^2$, and what does it mean? If you add a useless predictor, what happens?"

*Solution.* $R^2 = 1 - \text{RSS}/\text{TSS} = 1 - 50/200 = 1 - 0.25 = 0.75$. The model explains 75% of the variation in the outcome; 25% is unexplained noise. Adding a useless predictor cannot *raise* RSS (the geometry: a bigger column space, a closer projection), so $R^2$ stays the same or ticks up slightly even though the predictor is worthless -- which is precisely why you report *adjusted* $R^2$, which would *fall* here because the new predictor does not cut RSS enough to pay for its degree of freedom. The intuition: **$R^2$ measures fit, not truth, and it is rigged to reward complexity -- adjust it before you trust it.**

#### Worked example: is this coefficient significant?

*Problem.* "A factor's coefficient is 0.6 with a standard error of 0.15. Is it significant? What if the standard error were 0.40?"

*Solution.* The t-statistic is $t = 0.6 / 0.15 = 4.0$. Since $4.0 > 2$, the coefficient is significant at well beyond the 5% level -- you reject "this factor has no effect." With a standard error of 0.40, $t = 0.6/0.40 = 1.5 < 2$, so you *fail* to reject the null; the same point estimate of 0.6 is now indistinguishable from noise. The lesson worth voicing: "significance is the ratio of the estimate to its standard error, so a result becomes significant either by having a bigger effect *or* by measuring it more precisely -- more data shrinks the standard error and can rescue a borderline t-stat."

#### Worked example: the omitted-variable sign flip, quantified

*Problem.* "You regress $Y$ on $X$ and get $+0.8$. You suspect an omitted $Z$ with a true effect on $Y$ of $-1.0$, and $Z$ regresses on $X$ with a slope of $-1.1$. What is the bias, and what is the true coefficient?"

*Solution.* The omitted-variable bias formula: $\text{bias} = (\text{effect of } Z \text{ on } Y) \times (\text{slope of } Z \text{ on } X) = (-1.0) \times (-1.1) = +1.1$. The biased estimate equals the true coefficient plus the bias: $0.8 = \beta_{\text{true}} + 1.1$, so $\beta_{\text{true}} = 0.8 - 1.1 = -0.3$. The omitted confounder injected a spurious $+1.1$, masking a genuinely negative relationship. The takeaway: **two negatives in the bias formula multiply to a positive, which is exactly how an omitted variable can make a harmful factor look beneficial.**

#### Worked example: a two-factor P&L attribution

*Problem.* "Your strategy made \$12,000 today. You run a regression of its returns on two factors -- the market and a value factor (HML) -- with market beta 1.0 and value loading 0.5. Today the market contributed a move worth \$8,000 of P&L through your beta, and the value factor a move worth \$3,000 through your loading. Attribute the \$12,000."

*Solution.* A factor regression decomposes return (and hence dollar P&L) into factor pieces plus an intercept (alpha) plus a residual. Here the market exposure earned \$8,000, the value tilt earned \$3,000, and the leftover -- the part *no factor explains* -- is the intercept, your **alpha**: $\$12{,}000 - \$8{,}000 - \$3{,}000 = \$1{,}000$. So the day breaks down as \$8,000 market + \$3,000 value + \$1,000 alpha.

![A two-factor regression decomposes a 12,000 dollar day into 8,000 of market beta, 3,000 of value tilt, and 1,000 of unexplained alpha.](/imgs/blogs/linear-regression-deep-quant-interviews-11.png)

The waterfall shows the attribution stacking up to the total. This is precisely how a risk desk reports performance: the factor pieces are *beta returns* (you earned them by taking known, replicable risk), and only the residual intercept is *alpha* (genuine skill the factors cannot replicate). The intuition that makes this an interview-grade answer: **a factor regression separates the P&L you earned from taking risk anyone could take from the P&L you earned from being right -- and only the second kind is worth paying for.** The risk this hides: if your "alpha" is really an omitted factor you forgot to include, it will look like skill until that factor turns against you.

#### Worked example: why does adding a variable change my coefficient?

*Problem.* "I regress $y$ on $x_1$ and get a coefficient of $2$. I add $x_2$ and now $x_1$'s coefficient is $0.5$. Did I do something wrong?"

*Solution.* No -- you discovered that $x_1$ and $x_2$ share information. The single-variable coefficient of 2 was the *total* association of $x_1$ with $y$, including the part of $x_1$ that overlaps with $x_2$. The two-variable coefficient of 0.5 is the *partial* effect of $x_1$ *holding $x_2$ fixed* -- only $x_1$'s unique contribution. Both are correct answers to *different questions*. If $x_2$ is a genuine confounder, the 0.5 is the one you want (you de-biased); if $x_2$ is a downstream consequence of $x_1$, controlling for it may have *introduced* bias and the 2 was closer to the causal truth. The answer interviewers want: "coefficients are partial effects conditional on the model -- the change isn't a bug, it's the regression telling you $x_1$ and $x_2$ are correlated, and which number I trust depends on the causal diagram."

## Common misconceptions

**"A high $R^2$ means a good model."** No. $R^2$ measures how much variance the line explains *in your sample*, not whether the model is correct, causal, or useful out of sample. You can get $R^2 = 0.99$ by overfitting with junk predictors, by regressing two trending time series on each other (spurious regression), or by including the answer in disguise. And a *low* $R^2$ can accompany a perfectly real, tradeable, statistically rock-solid coefficient -- many genuine market signals explain under 1% of return variance yet print money at scale. Judge a coefficient by its t-stat and its economics, judge a model by its out-of-sample behavior, and treat $R^2$ as a rough fit gauge, never a verdict.

**"Correlation proves causation."** The single most repeated warning in statistics, and still the most violated. A regression slope is an *association*: $y$ tends to be higher when $x$ is higher. That is fully consistent with $x$ causing $y$, $y$ causing $x$, a third variable $z$ causing both, or pure coincidence. Regression cannot tell these apart from the numbers alone -- only the causal structure (which you bring from outside the data) can. The omitted-variable sign flip is this misconception made expensive.

**"OLS finds the line closest to the points."** Closest in the *vertical* sense, not the perpendicular sense. OLS minimizes squared errors in $y$ only, because it treats $x$ as known and $y$ as the thing being predicted. If you swap the roles and regress $x$ on $y$, you get a *different* line (unless the fit is perfect). The method that minimizes perpendicular distance is a different technique (total least squares / orthogonal regression), and confusing the two is a classic stumble.

**"A statistically significant signal is a profitable signal."** Significance ($|t| > 2$) says the relationship is unlikely to be zero; it says nothing about whether the edge survives transaction costs, whether it persists out of sample, or whether you found it by torturing the data through thousands of tests until one cleared the bar by luck. Desks demand higher t-thresholds, out-of-sample validation, and an economic story precisely because significance is necessary but nowhere near sufficient.

**"Outliers don't matter much."** They matter enormously, because squaring the residuals means a single wild point can dominate the entire fit -- a residual of 10 contributes 100 to the objective. One fat-fingered data point or one crisis day can swing your beta. This sensitivity is why robust regressions (which down-weight outliers) and careful data cleaning exist, and why "did you check for outliers?" is a fair interview follow-up to any fitting question.

## How it shows up on a real trading desk

**CAPM beta and equity hedging.** The most literal use. To run a market-neutral book, you regress each stock's returns on the market's, read off the beta, and short index futures or ETFs in proportion -- exactly the \$1,300,000 hedge from our worked example, scaled across hundreds of names. Betas drift (a stock's relationship to the market changes as its business changes), so desks re-estimate continuously, often with exponentially weighted windows that favor recent data. Get the beta wrong and your "hedged" book has hidden market exposure that shows up as P&L on the next big index move.

**Multi-factor models (Fama-French and descendants).** The academic and industrial backbone of equity risk. Instead of one regressor (the market) you regress returns on several style factors -- market, size (small minus big), value (high minus low, "HML"), profitability, momentum. Each coefficient is a *factor loading* telling you how exposed the stock or portfolio is to that style. Risk systems (Barra, Axioma) run thousands of these regressions to decompose a portfolio's risk into factor bets plus residual, so a manager can see they are unintentionally long momentum or short value. The two-factor P&L attribution we worked is this in miniature.

**Statistical arbitrage and pairs trading.** When two related instruments (two oil stocks, a stock and its ADR, an ETF and its basket) usually move together, you regress one on the other to get the **hedge ratio** -- how many units of B to trade against one unit of A so the combination is roughly market-neutral. The regression residual is the *spread*; stat-arb bets that the spread mean-reverts. The whole strategy lives or dies on the stability of a regression slope, which is why practitioners obsess over whether the relationship is genuine (cointegration) or a coincidence that will break.

**Fitting signals to forward returns.** Every alpha researcher's daily grind: regress a candidate signal (a sentiment score, an order-flow imbalance, an estimate revision) on the next period's returns. The slope is the signal's strength, the t-stat its reliability, the $R^2$ its (usually tiny) explanatory power. The discipline is brutal about multiple testing -- with thousands of signals tried, you trust only those that clear a high t-bar, survive out-of-sample, and have an economic rationale, because the alternative is trading noise that regressed significant by chance.

**Risk-model standard errors and the autocorrelation correction.** Because financial errors are autocorrelated and heteroskedastic (assumptions 3 and 4 fail), desks almost never use textbook standard errors. They use Newey-West or clustered standard errors that widen the error bars to reflect the real, messier structure -- otherwise every t-stat looks bigger than it should and you green-light signals that are noise. A candidate who knows that OLS *coefficients* survive these violations but OLS *standard errors* do not is a candidate who has actually fit models on real data.

**Rolling betas and the recency tradeoff.** Because a stock's beta drifts, no desk uses a single beta estimated over all history. They use *rolling* regressions -- re-fitting over a trailing window (say the last 252 trading days) that slides forward each day -- or exponentially weighted regressions that give recent observations more weight. This creates a tradeoff that interviewers like to probe: a short window tracks the current beta faithfully but is noisy (few observations, large standard errors); a long window is stable but stale (it averages in a relationship that may no longer hold). A bank hedging a \$50,000,000 equity book feels this directly -- pick a window that lags a regime change and the "hedged" book carries unintended exposure on exactly the days it matters most. The choice of window is itself a bet on how fast the world is changing, and there is no free lunch.

**The LTCM-style lesson on stable relationships.** Regression-based strategies assume the fitted relationship persists. In calm regimes a hedge ratio or a spread relationship looks rock solid -- high $R^2$, tight residuals -- and leverage piles onto it. Then a regime break (1998's Russian default, 2007's quant quake, 2020's March) snaps the relationship: residuals that "never" exceeded a band blow through it, and the bias and instability the diagnostics warned about arrive all at once. A spread that historically reverted within \$5 might gap to \$40 and keep going, turning a \$2,000,000 "low-risk" arbitrage position into a multiple of that in losses before any mean reversion arrives -- if it arrives at all. The math was never wrong; the assumption that the past relationship would hold was. Every regression on the desk carries that asterisk.

## When this matters and where to go next

If you are interviewing, linear regression is close to guaranteed to appear, and the questions reward depth over recall. Be able to (1) derive the slope as covariance over variance from the squared-error objective, (2) draw the projection diagram and explain why the residual is perpendicular to the regressors, (3) recite the Gauss-Markov assumptions and split them into "right answer" versus "right uncertainty," (4) read a regression table -- coefficient, standard error, t-stat, $R^2$ -- and say what each means, and (5) diagnose multicollinearity and omitted-variable bias on sight, including the sign flip. If you can whiteboard those five, you can hold a real conversation with anyone on a quant desk.

Beyond the interview, this is the foundation the rest of quantitative finance is built on. The natural next steps: how regressions price risk through the [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) framework and the Greeks; how the same fitting machinery reads the [volatility surface](/blog/trading/quantitative-finance/volatility-surface); how regression-style models calibrate the [yield curve](/blog/trading/quantitative-finance/yield-curve-modeling) and broader [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics). On the probability and decision side, the way coefficients carry uncertainty connects directly to [expected-value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) and to sizing those bets with the [Kelly criterion](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews). Linear regression is the door; those are the rooms it opens onto.
