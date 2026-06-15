---
title: "SVD, the pseudo-inverse, and least squares"
date: "2026-06-15"
description: "How the single most useful factorization in linear algebra turns a pile of noisy returns into a hedge ratio, why the textbook formula quietly breaks when your factors look alike, and how the singular value decomposition fixes it -- built from zero with worked dollar examples."
tags: ["svd", "least-squares", "pseudo-inverse", "linear-algebra", "regression", "hedge-ratio", "collinearity", "condition-number", "pca", "ridge-regression", "factor-models", "quantitative-finance"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 46
---

> [!important]
> **TL;DR** -- least squares finds the coefficient that best explains one stream of numbers using others, and the singular value decomposition (SVD) is the safest machine for computing it, especially when your inputs look alike.
>
> - **Least squares** minimizes $\lVert y - X\beta\rVert^2$, the total squared gap between what happened and what your factors predict. The answer is a *projection*: drop a perpendicular from your data onto the space your factors can reach.
> - **The textbook formula** is $\hat\beta = (X^\top X)^{-1}X^\top y$. It works until two factors are nearly identical, when $X^\top X$ becomes almost impossible to invert and your coefficients fly off to absurd values.
> - **SVD** writes any matrix as $A = U\Sigma V^\top$ -- a rotation, a set of stretches, and another rotation. The *pseudo-inverse* $X^+ = V\Sigma^+U^\top$ gives $\hat\beta = X^+y$ even when the textbook formula blows up.
> - **The condition number** $\sigma_{\max}/\sigma_{\min}$ is the single diagnostic for "are my factors too alike?" -- a ratio in the millions means your betas are noise amplifiers, and *truncating* the smallest singular value calms them down.
> - **The one number to remember**: a stock with $\beta = 1.3$ against the index means you short \$1,300,000 of the index to hedge a \$1,000,000 long -- a least-squares slope doing a real dollar job.

## Why a quant cares about one factorization

Here is a question every trading desk asks before lunch and after lunch: *if this thing moves, how much does that thing move with it, and how confident am I in the answer?* If a fund manager is long \$1,000,000 of a single stock and wants to neutralize the part of the position that just rides the broad market, she needs one number -- the *hedge ratio* -- that says how much of the index to short. If a researcher has built a signal and wants to know whether it actually predicts returns once you strip out the obvious style exposures, she needs a handful of numbers -- *factor loadings* -- that say how much of the return each factor explains. Both of those are the same calculation. Both are least squares.

The catch is that the formula every textbook hands you for least squares, $\hat\beta = (X^\top X)^{-1}X^\top y$, is a quiet liar. It is correct on paper and dangerous in practice. The moment two of your inputs are nearly the same thing -- two value factors that happen to overlap, two sector ETFs that move almost in lockstep, the level and a tiny tilt of the same yield curve -- the formula produces coefficients that swing from plus a thousand to minus a thousand on a rounding error, and a hedge built on those numbers will lose money in a way nobody can explain afterward. The fix is a different way of computing the exact same answer, one that degrades gracefully instead of exploding: the *singular value decomposition*.

![Least squares is one machine that turns a matrix of factors and a column of returns into a beta, a hedge notional, and factor loadings.](/imgs/blogs/svd-least-squares-regression-math-for-quants-1.png)

The diagram above is the mental model for the whole post. You feed in a matrix of factors $X$ -- each column is one explanatory variable, each row is one observation -- and a column $y$ of the thing you are trying to explain. One machine, least squares, chews on it and spits out a vector $\beta$ of coefficients. Depending on what you put in, the desk calls that output a different name: a *beta* when the factor is the market, a *hedge ratio* when you want to offset a position, a *factor loading* when the factor is a style like value or momentum. We are going to take that machine apart, see why the naive version is fragile, and rebuild it on the firmest foundation linear algebra offers.

We assume zero background. Every term gets defined the first time it appears, every formula gets an everyday analogy before the symbols, and every idea gets at least one worked example with real dollars. By the end you should be able to derive least squares, draw its geometry, factor a small matrix by hand, read a condition number, and know exactly when to reach for the pseudo-inverse instead of the textbook formula. This is educational material about how a standard tool works, not investment advice.

## Foundations: the line, the residual, and the squared error

Before any matrices, let us get the simplest possible version completely solid: fitting a single straight line to a cloud of dots.

### What we are actually trying to do

Suppose you have paired numbers. For each trading day you wrote down two things: how much the market index moved that day (call it $x$) and how much a particular stock moved (call it $y$). Plot each day as a dot, with $x$ left-to-right and $y$ up-and-down. The dots form a loose, tilted cloud -- on days the market rose, the stock usually rose too, but not by exactly the same amount and not every single day. The cloud has a *trend* -- it slopes up and to the right -- but it is fuzzy, because a stock's daily move is the market's pull *plus* a lot of stock-specific news, noise, and randomness that the market has nothing to do with.

You want one straight line through that cloud that captures the typical relationship. A line is described by two numbers: an *intercept* $a$ (where it crosses the vertical axis when $x = 0$) and a *slope* $b$ (how many units $y$ moves for each one-unit move in $x$). The slope is the number the desk cares about: it is the stock's *beta*, and it is the answer to "if the market moves 1%, how much does this stock typically move?" A beta of 1 means the stock moves one-for-one with the market; a beta of 1.5 means it is 50% more jumpy than the market (it amplifies both rallies and selloffs); a beta of 0.5 means it is half as sensitive (a defensive, sleepy name); a beta near 0 means the stock does its own thing regardless of the market.

The intercept $a$ has a name too: in finance it is the stock's *alpha* -- the average return it delivers that the market move cannot explain. Alpha is what every active manager is chasing: return you earned that was not just compensation for taking market risk. For a pure hedging calculation we usually care only about the slope, but it is worth knowing that the same single line hands you both the thing you want to hedge away (beta) and the thing you want to keep (alpha).

For any candidate line, each dot sits some vertical distance above or below it. That vertical gap -- the difference between the stock's actual move and what the line predicted -- is called the *residual*. A *residual* is just the error the line makes on one data point: $e_i = y_i - (a + b\,x_i)$. A line that fits well has small residuals; a line that fits badly has big ones. Notice we measure the gap *vertically*, in the direction of $y$, not perpendicular to the line. That choice is deliberate and it matters: we are treating $x$ (the market) as known and trying to predict $y$ (the stock), so the error we care about is the error in our prediction of $y$. This asymmetry -- predict $y$ from $x$, not the other way around -- is why regressing the stock on the market gives a different slope than regressing the market on the stock, a subtlety that trips up beginners and that we will return to.

### Why we square the errors

We want the line that makes the residuals "small overall". But small how? If we just added up the raw residuals, the positive ones (dots above the line) and the negative ones (dots below) would cancel, and a terrible line could score zero -- a line that overshoots half the points by a mile and undershoots the other half by a mile would have residuals summing to zero, which is absurd. So we need a measure that does not let positives and negatives cancel. Two candidates exist: sum the *absolute values* of the residuals, or sum the *squares*. Both are used in practice (absolute-value fitting is called *least absolute deviations* and is more robust to outliers), but squaring is the overwhelming default for three reasons.

First, squaring makes every error positive so they cannot cancel. Second, it punishes big misses far more than small ones (a residual of 4 contributes 16, a residual of 2 contributes only 4) -- which matches the trader's intuition that one catastrophic miss is worse than several small ones. Third, and decisively, squaring produces a *smooth* function with a single clean minimum you can find with calculus in closed form, where the absolute value has a kink that forces you to solve numerically. The square gives you a formula; the absolute value gives you an algorithm. The total we minimize is the *sum of squared residuals*, also called the *squared error*:

$$\text{SSE} = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n}\bigl(y_i - a - b\,x_i\bigr)^2.$$

The line that makes this sum as small as possible is the *least-squares* line, and the method is *ordinary least squares* (OLS). The name is literal: of all possible lines, we pick the one with the *least* sum of *squares*.

> The least-squares line is the one that, if you nudged it any direction, would make your total squared error go up. It sits at the bottom of a smooth bowl.

That bowl image is worth holding onto, because it generalizes all the way up to the matrix case. Plot the squared error as a height above the plane of all possible $(a, b)$ choices. Because the error is a sum of squares of things that are linear in $a$ and $b$, this surface is a *paraboloid* -- a perfectly smooth, upward-opening bowl with exactly one lowest point. There are no false bottoms, no local minima to get stuck in, no ambiguity. That single-bowl shape is the deep reason least squares is so well-behaved and so beloved: the answer always exists and is always unique (as long as your factors are not degenerate), and you can find it by rolling downhill from anywhere. Later, when collinearity strikes, the bowl will flatten into a long shallow trough -- still convex, but with a nearly-flat floor where many $(a,b)$ pairs score almost the same, and that flatness is the geometric face of an unstable answer.

### The single-variable answer, with numbers

For one input variable, calculus hands you a clean closed-form answer. We find the bottom of the bowl by demanding that the slope of the error surface be zero in both directions -- that is, we set the *partial derivatives* of the SSE with respect to $a$ and $b$ to zero. (A partial derivative is just the rate of change of a function when you wiggle one input and hold the rest fixed; at the bottom of a bowl, wiggling any input does not change the height to first order, so every partial derivative is zero.) Doing that and solving the two resulting equations gives:

$$b = \frac{\operatorname{Cov}(x,y)}{\operatorname{Var}(x)}, \qquad a = \bar y - b\,\bar x,$$

where $\operatorname{Cov}(x,y)$ is the *covariance* (how much $x$ and $y$ move together), $\operatorname{Var}(x)$ is the *variance* of $x$ (how much $x$ spreads out on its own), and $\bar x, \bar y$ are the averages. The slope is "how much they move together" divided by "how much $x$ moves on its own". Read it as a ratio: the numerator picks up the part of $y$'s wiggle that lines up with $x$'s wiggle, and dividing by $x$'s own wiggle converts that into "units of $y$ per unit of $x$". This single fraction is the seed of everything that follows; the matrix formula we build later is literally this same ratio promoted to many factors at once, with $X^\top X$ standing in for the variance and $X^\top y$ for the covariance. If you want a deeper tour of this single-variable case and its interview traps, the companion post on [linear regression from first principles](/blog/trading/quantitative-finance/linear-regression-deep-quant-interviews) walks the whole derivation; here we use it as the launch pad for the matrix story.

#### Worked example: a one-factor hedge

You run a desk and you are long \$1,000,000 of a stock we will call NOVA. You want to hedge the part of NOVA that just moves with the broad market index, so you can isolate the stock-specific bet. You pull five days of returns (in percent):

| Day | Market $x$ | NOVA $y$ |
|----|-----------|----------|
| 1  | +1.0%     | +1.4%    |
| 2  | -0.5%     | -0.4%    |
| 3  | +2.0%     | +2.9%    |
| 4  | 0.0%      | +0.2%    |
| 5  | -1.5%     | -2.1%    |

First the averages: $\bar x = (1.0 - 0.5 + 2.0 + 0.0 - 1.5)/5 = 1.0/5 = 0.2\%$ and $\bar y = (1.4 - 0.4 + 2.9 + 0.2 - 2.1)/5 = 2.0/5 = 0.4\%$.

Now the covariance numerator, $\sum (x_i - \bar x)(y_i - \bar y)$. Compute each deviation and product:

- Day 1: $(1.0-0.2)(1.4-0.4) = (0.8)(1.0) = 0.80$
- Day 2: $(-0.5-0.2)(-0.4-0.4) = (-0.7)(-0.8) = 0.56$
- Day 3: $(2.0-0.2)(2.9-0.4) = (1.8)(2.5) = 4.50$
- Day 4: $(0.0-0.2)(0.2-0.4) = (-0.2)(-0.2) = 0.04$
- Day 5: $(-1.5-0.2)(-2.1-0.4) = (-1.7)(-2.5) = 4.25$

Sum: $0.80 + 0.56 + 4.50 + 0.04 + 4.25 = 10.15$.

Now the variance numerator, $\sum (x_i - \bar x)^2$:

- $0.8^2 = 0.64$, $(-0.7)^2 = 0.49$, $1.8^2 = 3.24$, $(-0.2)^2 = 0.04$, $(-1.7)^2 = 2.89$.

Sum: $0.64 + 0.49 + 3.24 + 0.04 + 2.89 = 7.30$.

The slope -- NOVA's beta -- is $b = 10.15 / 7.30 = 1.39$. Round it to $\beta \approx 1.39$. That means when the market moves 1%, NOVA typically moves about 1.39%.

Now the dollar job. To neutralize the market component of a \$1,000,000 long, you short $\beta$ times the notional in the index:

$$\text{hedge notional} = \beta \times \$1{,}000{,}000 = 1.39 \times \$1{,}000{,}000 = \$1{,}390{,}000.$$

You short \$1,390,000 of the index. After that, if the market drops 1%, your NOVA long loses roughly 1.39% × \$1,000,000 = \$13,900, while your index short *gains* roughly 1% × \$1,390,000 = \$13,900. The market move cancels, and you are left holding only the stock-specific bet -- which is the part you actually have a view on.

Now the realistic wrinkle. That beta of 1.39 was estimated from only five days, and five days is nowhere near enough to trust. The *standard error* of a regression slope -- how much the estimate would jump around if you re-ran it on a fresh sample -- shrinks roughly like $1/\sqrt{n}$, so going from 5 days to 250 days (about a trading year) cuts the noise in the estimate by a factor of about $\sqrt{250/5} = \sqrt{50} \approx 7$. With five days your beta might really be anywhere from, say, 1.0 to 1.8; with a year of data it might tighten to 1.30 to 1.48. The dollar consequence is direct: if the true beta is 1.30 but your noisy five-day fit told you 1.39, you over-hedged by $0.09 \times \$1{,}000{,}000 = \$90{,}000$ of index short, and on a 1% market move that mis-sized hedge leaves a \$900 stub of unwanted exposure. Small here, but on a \$100,000,000 book the same 0.09 beta error is a \$9,000,000 hedging mistake. **The intuition: a regression slope is not an abstraction; it is the multiplier that turns a position you own into the size of the hedge you need to put on -- and the number of observations behind that slope is the difference between a hedge and a guess.**

## 1. The geometry: projection onto the column space

Now we leave the single line behind and meet the picture that makes everything else click. Least squares is, at heart, a *projection* -- the same operation as casting a shadow.

### The shadow analogy

Hold a pencil above a table and shine a light straight down. The pencil's shadow on the table is the *projection* of the pencil onto the table. The shadow is the closest point on the table to the pencil, and the line from the pencil tip to its shadow is perfectly vertical -- it meets the table at a right angle. That right angle is the entire secret of least squares. If you wanted to find the spot on the table nearest to the pencil tip, you would not search around -- you would just drop straight down. The perpendicular *is* the shortest path, always, and least squares is nothing more than that fact dressed up in many dimensions.

To make the leap to many dimensions we need one shift of viewpoint that feels strange at first but pays off enormously. Until now we pictured each observation as a *dot* on a 2-D scatter plot (market on one axis, stock on the other). Now we are going to picture each *variable* as a whole arrow in a high-dimensional space. Stack your $n$ observations of the outcome into a single tall vector $y$ living in $n$-dimensional space (five days means a vector with five numbers, a single point or arrow in 5-D). Each factor is also a tall vector of $n$ numbers -- one arrow per factor. So instead of $n$ dots in a 2-D picture, we have a handful of arrows in an $n$-D picture. The whole regression now lives in this one diagram.

The set of every point you can possibly reach by combining your factors -- every $X\beta$ as $\beta$ ranges over all values -- is a flat sheet through the origin called the *column space* of $X$ (because it is spanned by the columns of $X$). Think of it as everywhere your factors can "reach": every weighted blend of the factor arrows. It is usually a lower-dimensional slice of the full space: two factors span a 2-D plane sitting inside 5-D space, the way two pencils lying on a desk define a flat plane no matter how big the room is.

Your outcome $y$ almost never lies exactly on that sheet -- if it did, your factors would explain it perfectly with zero error, which never happens with noisy returns. So the best you can do is find the point *on* the sheet that is closest to $y$. That closest point is the shadow of $y$ on the sheet: the *projection*. Call it $\hat y = X\hat\beta$, the *fitted values*. The coefficients $\hat\beta$ are simply the recipe -- the blend of factors -- that lands you exactly on that shadow.

![Least squares drops a perpendicular from the outcome vector onto the flat plane the factors can reach, leaving a residual at a right angle.](/imgs/blogs/svd-least-squares-regression-math-for-quants-2.png)

The figure above is the geometry. The data vector $y$ floats off the plane; its shadow $\hat y$ is the fitted vector lying flat on the plane; and the gap between them -- the *residual vector* $e = y - \hat y$ -- is perpendicular to the plane. That perpendicularity is not a coincidence; it is what "closest" means. The shortest distance from a point to a flat surface is always along the perpendicular. Any other point on the plane is farther away, which is exactly the statement that any other $\beta$ has larger squared error.

### Why perpendicular means "uncorrelated with what you used"

The residual being perpendicular to the column space has a beautiful finance reading. The column space is everything your factors can explain. The residual is what is left over. Perpendicular means the leftover is *uncorrelated with every factor you used* -- you have squeezed out every drop of explanatory power your factors contained. If the residual still had some correlation with a factor, you could tilt your line to capture it and shrink the error further, contradicting that you found the minimum. This is not a metaphor: in this geometry, the *dot product* of two vectors is (up to scaling) their covariance, so "perpendicular" (dot product zero) is literally "zero covariance" is literally "uncorrelated". The geometry and the statistics are the same statement wearing different clothes.

This is why a quant trusts least squares: by construction, the alpha (the residual) is orthogonal to the factors (the explained part). The post on [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) leans on exactly this -- you regress your raw signal on the known factors and *keep the residual*, because the residual is the part of your idea that nobody else's factor model already captures. If your signal still has return left over *after* you have stripped out market, size, value, and momentum, that leftover is genuine alpha; if regressing it on those factors makes the leftover vanish, your "signal" was just a repackaging of factors everyone already harvests, and you have learned that the cheap way -- on a regression printout rather than in the market.

There is a second, money-saving consequence of orthogonality. Because the residual is perpendicular to the fit, the total variance splits cleanly into two non-overlapping pieces: variance explained by the factors plus variance left in the residual, with no cross-term. This is the Pythagorean theorem applied to your returns -- $\lVert y\rVert^2 = \lVert\hat y\rVert^2 + \lVert e\rVert^2$ -- and it is exactly what lets a risk manager say "62% of this book's variance is market risk and 38% is idiosyncratic" with the two numbers adding to 100%. Without orthogonality those buckets would double-count, and risk reports would not foot. The right angle is not a curiosity; it is what makes risk decomposition arithmetic honest.

## 2. The normal equations: the textbook solution

Now let us write the projection as a formula. We want $\hat\beta$ such that the residual $e = y - X\hat\beta$ is perpendicular to every column of $X$. "Perpendicular" between vectors means their dot product is zero, and stacking all those zero dot products into one statement gives:

$$X^\top (y - X\hat\beta) = 0.$$

Read it aloud: "$X$-transpose times the residual is zero," i.e. every factor is orthogonal to the leftover. Expand and rearrange:

$$X^\top X\,\hat\beta = X^\top y.$$

These are the *normal equations* ("normal" is old-fashioned mathematics for "perpendicular"). If the matrix $X^\top X$ can be inverted, multiply both sides by its inverse:

$$\boxed{\ \hat\beta = (X^\top X)^{-1}X^\top y\ }$$

This is the formula in every statistics textbook. Let us define every piece. $X$ is the $n \times k$ matrix with one row per observation and one column per factor (plus usually a column of all ones for the intercept). $X^\top$ is its *transpose* -- flip rows and columns, so a tall $n \times k$ matrix becomes a wide $k \times n$ one. $X^\top X$ is a small $k \times k$ matrix; for two factors it is just $2 \times 2$. $X^\top y$ is a short $k$-vector. So once you form these, you are solving a tiny system, no matter how many thousands of observations you started with. This is genuinely convenient: your data might be a million rows tall, but $X^\top X$ for a five-factor model is a humble $5\times5$ grid you could invert by hand. The dimensionality of the *problem* is the number of factors, not the number of observations -- which is why a desk can re-estimate a factor model in milliseconds even on years of tick data.

It also helps to see *why* the perpendicularity condition turns into $X^\top X\hat\beta = X^\top y$. The matrix $X^\top$ has one row per factor; multiplying it by any vector computes that vector's dot product with each factor in turn. So $X^\top(y - X\hat\beta) = 0$ is the single compact way of saying "the residual has zero dot product with factor 1, *and* zero dot product with factor 2, *and* so on for every factor". One matrix equation, $k$ perpendicularity conditions, one for each coefficient we are free to choose. The count matches: $k$ unknowns, $k$ equations, and (when the factors are independent) exactly one solution. Everything is in balance -- until the factors stop being independent, which is where the trouble starts.

![The normal-equations recipe stacks the factor matrix and outcome, forms two products, and solves a small system for the beta vector.](/imgs/blogs/svd-least-squares-regression-math-for-quants-3.png)

The pipeline above is the whole recipe: stack your data into $X$ and $y$, form $X^\top X$ and $X^\top y$, invert the first, multiply, and read off $\beta$. It is fast and, when your factors are well-behaved, perfectly fine. The trouble -- the entire reason this post exists -- is hiding in that one word, *invert*.

### Why $X^\top X$ is the danger zone

The matrix $X^\top X$ is, up to scaling, the *covariance matrix of your factors*. Its diagonal entries say how much each factor varies; its off-diagonal entries say how much two factors move together. When two factors are nearly identical -- highly correlated -- that off-diagonal correlation approaches 1, and the matrix becomes *nearly singular*: almost impossible to invert. "Singular" means a matrix that cannot be inverted at all, the matrix equivalent of the number zero, which you cannot divide by; "nearly singular" is the matrix equivalent of a number very close to zero, which you *can* divide by but which makes everything explode. Inverting a nearly-singular matrix is like dividing by 0.0000001: the operation is technically legal, but the answer is huge and any tiny error in the input gets magnified into a vast error in the output.

Here is the precise reason this is worse for the normal equations than for the underlying problem. Forming $X^\top X$ *squares* the matrix, and squaring a number near zero pushes it even closer to zero ($0.001$ becomes $0.000001$). In condition-number terms, which we develop fully later, $\kappa(X^\top X) = \kappa(X)^2$: the normal equations operate on a matrix whose conditioning is the *square* of the original problem's. If $X$ has a condition number of 1,000 -- borderline but workable -- then $X^\top X$ has a condition number of 1,000,000, and you have needlessly thrown away half your numerical precision before you even started solving. Tiny changes in the data produce enormous changes in the answer. We will make this concrete in the collinearity example, and the covariance-matrix machinery behind it is unpacked in [covariance and correlation pitfalls](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews).

For now hold this thought: the normal equations are correct but brittle. They square the matrix $X$ before inverting it, and squaring a matrix *squares its sensitivity to noise* too. There is a way to solve least squares that never forms $X^\top X$ at all, works directly on $X$, and so pays only the un-squared price of conditioning. It is the safest tool in the box, and meeting it is the whole point of the next several sections.

## 3. A two-factor regression by hand

Let us run the matrix machinery once on small numbers so the symbols turn concrete. We will regress a portfolio's return on two factors: the market and a value factor.

#### Worked example: two factors, the matrix way

You have four observations. Column 1 of $X$ is the market return, column 2 is a value factor; $y$ is your portfolio return (all in percent):

$$X = \begin{bmatrix} 1 & 2 \\ 2 & 1 \\ 3 & 4 \\ 4 & 3 \end{bmatrix}, \qquad y = \begin{bmatrix} 5 \\ 4 \\ 11 \\ 10 \end{bmatrix}.$$

(For clarity we drop the intercept column here.) Form $X^\top X$:

$$X^\top X = \begin{bmatrix} 1&2&3&4 \\ 2&1&4&3 \end{bmatrix}\begin{bmatrix} 1&2 \\ 2&1 \\ 3&4 \\ 4&3 \end{bmatrix} = \begin{bmatrix} 1+4+9+16 & 2+2+12+12 \\ 2+2+12+12 & 4+1+16+9 \end{bmatrix} = \begin{bmatrix} 30 & 28 \\ 28 & 30 \end{bmatrix}.$$

Form $X^\top y$:

$$X^\top y = \begin{bmatrix} 1\cdot5 + 2\cdot4 + 3\cdot11 + 4\cdot10 \\ 2\cdot5 + 1\cdot4 + 4\cdot11 + 3\cdot10 \end{bmatrix} = \begin{bmatrix} 5 + 8 + 33 + 40 \\ 10 + 4 + 44 + 30 \end{bmatrix} = \begin{bmatrix} 86 \\ 88 \end{bmatrix}.$$

Now invert the $2\times2$ matrix. For $\begin{bmatrix} a & b \\ c & d\end{bmatrix}$ the inverse is $\frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a\end{bmatrix}$. The determinant is $ad - bc = 30\cdot30 - 28\cdot28 = 900 - 784 = 116$. So:

$$(X^\top X)^{-1} = \frac{1}{116}\begin{bmatrix} 30 & -28 \\ -28 & 30 \end{bmatrix}.$$

Multiply by $X^\top y$:

$$\hat\beta = \frac{1}{116}\begin{bmatrix} 30 & -28 \\ -28 & 30 \end{bmatrix}\begin{bmatrix} 86 \\ 88 \end{bmatrix} = \frac{1}{116}\begin{bmatrix} 30\cdot86 - 28\cdot88 \\ -28\cdot86 + 30\cdot88 \end{bmatrix} = \frac{1}{116}\begin{bmatrix} 2580 - 2464 \\ -2408 + 2640 \end{bmatrix} = \frac{1}{116}\begin{bmatrix} 116 \\ 232 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}.$$

So $\hat\beta = (1, 2)$: one unit of market exposure and two units of value exposure. Sanity check on day 3: $X\hat\beta = 3\cdot1 + 4\cdot2 = 11$, exactly $y_3$. In this engineered example the factors explain $y$ perfectly. If you are long \$500,000 of this portfolio, the regression says \$500,000 of it behaves like one unit of market times two units of value -- and to hedge the market leg alone you would short $1 \times \$500{,}000 = \$500{,}000$ of the index. **The intuition: the matrix formula is just the single-variable slope generalized -- $X^\top X$ plays the role of variance and $X^\top y$ the role of covariance, and the answer is still "co-movement divided by self-movement", now in several dimensions at once.**

Notice the determinant was 116 -- comfortably away from zero -- because the two factors here are different enough. Hold onto that number. In the next collinearity example the determinant will crash toward zero and everything will fall apart.

## 4. The singular value decomposition: rotate, scale, rotate

We now meet the star of the post. The singular value decomposition (SVD) is a way to take *any* matrix -- square or rectangular, full-rank or degenerate -- and break it into three simple, well-understood pieces. It is the most useful factorization in applied linear algebra, and once you see it you start seeing it everywhere.

### The everyday picture

Any matrix, when it acts on vectors, does the same three things in sequence: it *rotates* the input, *stretches* it along certain axes, and *rotates* the result. That is it. Every matrix -- no matter how complicated it looks, no matter how many off-diagonal entries it has, no matter whether it is square or rectangular -- is secretly just turn, stretch, turn. This is a genuinely surprising claim the first time you hear it: a matrix can look like an arbitrary grid of numbers with no obvious structure, and yet underneath it is doing something as simple as spinning a globe, scaling it along three fixed axes, and spinning it again. The SVD is the statement that you can always pull those three moves apart and look at each one separately:

$$A = U\,\Sigma\,V^\top.$$

![Any matrix factors into U times Sigma times V-transpose: an input rotation, a set of stretches, and an output rotation.](/imgs/blogs/svd-least-squares-regression-math-for-quants-4.png)

The figure above names the three pieces. $V^\top$ is the first rotation (it reorients the input to point along the matrix's natural input axes). $\Sigma$ (capital sigma) is a diagonal matrix of non-negative *stretch factors* called the *singular values*, written $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ from largest to smallest. $U$ is the second rotation (it reorients the stretched result into the output space). The columns of $V$ are the *right singular vectors* (the natural input directions), and the columns of $U$ are the *left singular vectors* (the natural output directions). The singular values say how much the matrix stretches along each matched pair of directions.

A word on what "rotation" buys us, because it is the source of all the numerical stability. $U$ and $V$ are *orthogonal* matrices, which is the technical name for "pure rotations (and reflections)". An orthogonal matrix never changes the length of a vector or the angle between two vectors -- it just reorients the whole space rigidly, like turning a sheet of graph paper without stretching it. Crucially, an orthogonal matrix is perfectly conditioned: its own condition number is exactly 1, and its inverse is just its transpose (so inverting it is free and introduces no error). That means *all* the stretching, all the potential for trouble, all the conditioning of the matrix lives in one place: the diagonal $\Sigma$. The rotations are harmless. So when we ask "is this matrix dangerous to invert?", the SVD lets us answer by looking at a single list of numbers -- the singular values -- and ignoring the rotations entirely. No other factorization isolates the danger so cleanly.

Two more facts make the SVD the universal tool. First, *every* matrix has one -- square or rectangular, full-rank or rank-deficient, invertible or not. There are no exceptions and no preconditions, unlike the eigendecomposition, which only behaves for special matrices. Second, the singular values are *unique* (the rotations can have sign ambiguities, but the stretch factors are pinned down exactly), so they are a genuine fingerprint of the matrix. When two quants on two continents compute the SVD of the same return matrix, they get the same singular values to the last decimal, and those numbers mean the same thing: the strengths of the independent directions of variation in the data.

### Watching it work on a circle

The cleanest way to feel the SVD is to watch what a matrix does to a circle of unit-length input vectors. Take every arrow of length one pointing in every direction -- the tips trace a circle. Apply the matrix and the circle becomes an ellipse. The SVD reads that ellipse off directly: the *directions* of the ellipse's axes come from $U$, and the *lengths* of its half-axes are exactly the singular values $\sigma_1$ (the long one) and $\sigma_2$ (the short one).

![A unit circle is rotated, stretched along its axes by the singular values, then rotated again into an output ellipse.](/imgs/blogs/svd-least-squares-regression-math-for-quants-5.png)

The figure traces the three moves on the circle: $V^\top$ turns the circle (a circle looks the same when turned, but its hidden axes line up), $\Sigma$ stretches it into an ellipse whose half-widths are $\sigma_1$ and $\sigma_2$, and $U$ turns the ellipse to its final orientation. The largest singular value $\sigma_1$ is the most the matrix can stretch any unit vector; the smallest $\sigma_{\min}$ is the least. If the smallest singular value is nearly zero, the matrix nearly *flattens* the circle into a line segment -- it collapses a whole dimension -- and that collapse is precisely what makes the matrix hard to invert.

The flattening result is worth dwelling on, because it is the whole story of collinearity in one diagram. Inverting a matrix means *undoing* what it did -- taking the output ellipse and turning it back into the input circle. If the matrix squashed the circle nearly flat (because $\sigma_{\min} \approx 0$), then undoing it means stretching that flat sliver back out to full width, an enormous magnification by $1/\sigma_{\min}$. Any tiny error or noise riding along the squashed direction gets blown up by that same huge factor. This is the geometric reason a near-singular matrix amplifies noise: the inverse has to "un-squash" a nearly-collapsed dimension, and un-squashing by a factor of a thousand magnifies the noise a thousandfold. When a desk's factor matrix has a tiny singular value, it means two of the factors very nearly point the same way, the matrix nearly flattens the space they span, and any regression that tries to invert it is asking to un-squash that near-collapse -- which is exactly when coefficients explode.

#### Worked example: a 2x2 SVD by hand

Let us factor a small matrix and read off its singular values and condition number. Take:

$$A = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}.$$

This one is already diagonal, so the SVD is easy and instructive. A diagonal matrix with positive entries needs no rotation at all: $U = I$, $V = I$ (identity matrices), and $\Sigma = A$ itself. The singular values are the diagonal entries, but always sorted largest-first: $\sigma_1 = 2$, $\sigma_2 = 1$. Geometrically, $A$ stretches the horizontal axis by 2 and the vertical by 1, turning the unit circle into an ellipse twice as wide as it is tall.

To prove the recipe on something that *does* need rotation, consider the singular values of a general $A$: they are the square roots of the *eigenvalues* of $A^\top A$. Let us verify with our diagonal case. Compute $A^\top A = \begin{bmatrix} 4 & 0 \\ 0 & 1\end{bmatrix}$. Its eigenvalues (the diagonal, since it is diagonal) are 4 and 1. Their square roots are $\sqrt 4 = 2$ and $\sqrt 1 = 1$ -- exactly $\sigma_1$ and $\sigma_2$. The recipe works.

Now read off the *condition number*, the ratio of largest to smallest singular value:

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{2}{1} = 2.$$

A condition number of 2 is wonderful -- the matrix treats every direction within a factor of 2 of every other, so inverting it is stable. To see what a *bad* matrix looks like, change $A$ to $\begin{bmatrix} 2 & 0 \\ 0 & 0.001\end{bmatrix}$. Now $\sigma_1 = 2$, $\sigma_2 = 0.001$, and $\kappa = 2/0.001 = 2000$. This matrix nearly flattens the vertical direction to nothing, and inverting it means dividing by 0.001 -- amplifying any noise in that direction two-thousand-fold.

Let us do one matrix that genuinely needs rotation, so you trust the recipe on something non-diagonal. Take $A = \begin{bmatrix} 1 & 1 \\ 0 & 1\end{bmatrix}$, a *shear* (it slides the top of the square sideways). Form $A^\top A = \begin{bmatrix} 1 & 1 \\ 1 & 2\end{bmatrix}$. The eigenvalues of a $2\times2$ matrix $\begin{bmatrix} a & b \\ b & d\end{bmatrix}$ solve $\lambda^2 - (a+d)\lambda + (ad - b^2) = 0$; here that is $\lambda^2 - 3\lambda + 1 = 0$, giving $\lambda = (3 \pm \sqrt 5)/2 \approx 2.618$ and $0.382$. The singular values are the square roots: $\sigma_1 = \sqrt{2.618} \approx 1.618$ and $\sigma_2 = \sqrt{0.382} \approx 0.618$. (Those are the golden ratio and its reciprocal, a pleasant accident of this particular shear.) The condition number is $\sigma_1/\sigma_2 \approx 1.618/0.618 \approx 2.618$. So even this innocuous-looking shear, with all entries 0 or 1, stretches some directions about 2.6 times more than others -- the rotations $U$ and $V$ are non-trivial, but the recipe "singular values are square roots of the eigenvalues of $A^\top A$" reads them off mechanically. **The intuition: the singular values are the matrix's stretch factors, and the gap between the largest and smallest is the warning light -- a wide gap means the matrix is one step from collapsing a dimension, and any inverse built on it will amplify noise.**

### Solving least squares with the SVD

Here is the payoff. Substitute $X = U\Sigma V^\top$ into the least-squares problem and the algebra collapses beautifully. Because $U$ and $V$ are *rotations* (they preserve lengths and angles, and their transpose is their inverse), the solution becomes:

$$\hat\beta = V\,\Sigma^+ U^\top y,$$

where $\Sigma^+$ is the matrix you get by inverting each non-zero singular value ($\sigma_i \to 1/\sigma_i$) and leaving any zeros alone. Let us read that formula left to right as three physical steps, because it demystifies the whole thing. First, $U^\top y$ rotates your outcome vector into the matrix's natural output coordinates -- it asks "how much of $y$ points along each output direction?" Second, $\Sigma^+$ rescales each of those components by dividing by the corresponding singular value -- big, robust directions get divided by a big number (gentle), and small, fragile directions get divided by a small number (the dangerous amplification). Third, $V$ rotates the result back into the coordinate system of your actual factors so you can read the coefficients. The entire risk lives in that middle step, and you can *see* it: the troublesome $1/\sigma_i$ for a tiny $\sigma_i$ sits right there in plain view, ready to be tamed.

We never form $X^\top X$, so we never square the sensitivity to noise. The condition number of the problem is $\sigma_{\max}/\sigma_{\min}$ -- not its square. That single fact is why every serious numerical library solves least squares with the SVD (or a close cousin, the QR decomposition) and never with the literal normal-equations formula. When you call `numpy.linalg.lstsq` or the regression function in any statistics package, an SVD or QR is what runs internally; the textbook $(X^\top X)^{-1}X^\top y$ is taught for understanding and almost never executed verbatim by professionals. This same factorization, applied to the *data* matrix of returns, is exactly the engine behind principal component analysis; the [eigendecomposition and PCA](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews) view treats the columns of $V$ as risk factors and the singular values squared as the variance each factor carries.

## 5. The Moore-Penrose pseudo-inverse

The expression $V\Sigma^+ U^\top$ has a name: the *Moore-Penrose pseudo-inverse* of $X$, written $X^+$. With it, the least-squares solution is breathtakingly clean:

$$\boxed{\ \hat\beta = X^+ y\ }$$

### What "pseudo" buys you

A normal inverse $X^{-1}$ only exists for a square matrix that is not degenerate. But your data matrix $X$ is almost never square -- you have thousands of observations and a handful of factors, so $X$ is tall and skinny. A tall matrix has no ordinary inverse. The pseudo-inverse is the generalization that always exists, for any shape, any rank. It does the most sensible thing in every case:

- **More observations than factors** (the usual case): $X^+y$ returns the unique least-squares solution -- the $\beta$ that minimizes squared error. This is the over-determined system.
- **More factors than observations, or exactly collinear factors**: there are infinitely many $\beta$ that fit equally well, and $X^+y$ picks the one with the *smallest length* $\lVert\beta\rVert$ -- the most modest, least-extreme coefficients consistent with the data. That tie-breaking rule is itself a gentle form of regularization.
- **Perfectly invertible square $X$**: $X^+ = X^{-1}$, so it never disagrees with the ordinary inverse when the ordinary inverse exists.

The pseudo-inverse, in other words, is the inverse that refuses to panic. Where $(X^\top X)^{-1}$ throws up its hands at a singular matrix, $X^+$ quietly returns the most reasonable answer available.

It is worth being precise about what the pseudo-inverse *guarantees*, because "most reasonable answer available" sounds vague and it is not. The Moore-Penrose pseudo-inverse is defined by four algebraic conditions (the Penrose conditions) that, together, pin it down uniquely for every matrix. You do not need to memorize them, but the consequence is what matters: among all vectors $\beta$ that achieve the minimum possible squared error, $X^+y$ returns the one with the smallest length $\lVert\beta\rVert$. Two promises, in order of priority: first, fit the data as well as anything can (minimum residual); second, among all the tied best-fitters, be the most modest (minimum coefficients). That second promise is precisely the property a quant wants, because modest coefficients mean small, hold-able positions instead of giant offsetting bets. The pseudo-inverse does not just avoid crashing on a singular matrix; it actively chooses the *economically sanest* of the infinitely many answers a singular problem admits.

There is also a clean way to compute the pseudo-inverse for the two shapes you actually meet. When $X$ is tall and full column rank (more observations than factors, factors not collinear -- the everyday case), $X^+ = (X^\top X)^{-1}X^\top$, which recovers the normal equations exactly. When $X$ is wide and full row rank (more factors than observations), $X^+ = X^\top(XX^\top)^{-1}$. And in the dangerous in-between -- collinear factors, rank deficiency -- neither of those simple forms works, and you must go through the SVD: $X^+ = V\Sigma^+U^\top$. The SVD route is the one that always works, which is why it is the route the libraries take.

#### Worked example: the pseudo-inverse on an over-determined system

You manage a \$2,000,000 portfolio and want its exposure to a single factor (say, the market), but you have *three* observations and only *one* unknown slope -- an over-determined system, more equations than unknowns, with no exact solution. Your design matrix and outcomes (returns in percent) are:

$$X = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \qquad y = \begin{bmatrix} 2 \\ 5 \\ 7 \end{bmatrix}.$$

There is no single $\beta$ with $1\cdot\beta = 2$, $2\cdot\beta = 5$, and $3\cdot\beta = 7$ all at once (those want $\beta = 2, 2.5, 2.33$). Least squares finds the best compromise. The pseudo-inverse of a single-column matrix is $X^+ = \frac{X^\top}{X^\top X}$. Compute $X^\top X = 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14$. And $X^\top y = 1\cdot2 + 2\cdot5 + 3\cdot7 = 2 + 10 + 21 = 33$. So:

$$\hat\beta = X^+ y = \frac{X^\top y}{X^\top X} = \frac{33}{14} = 2.357.$$

The best-fit slope is $\beta \approx 2.36$. To translate into the dollar hedge on the \$2,000,000 book: you would short $\beta \times \$2{,}000{,}000 = 2.357 \times \$2{,}000{,}000 = \$4{,}714{,}000$ of the market factor to neutralize the estimated exposure. Let us sanity check the residuals: fitted values are $1\cdot2.357 = 2.36$, $2\cdot2.357 = 4.71$, $3\cdot2.357 = 7.07$, versus actuals 2, 5, 7. Residuals: $-0.36, +0.29, -0.07$. They are small and roughly balanced -- which is exactly what a good least-squares fit produces, and they sum to nearly zero because the fit is orthogonal to the factor. **The intuition: the pseudo-inverse handles "too many equations" by refusing to satisfy any one exactly and instead minimizing the total squared miss -- the same projection idea, now wearing a formula that works for any matrix shape.**

## 6. The condition number: why collinear factors blow up

Now we reach the practical heart of the matter -- the reason a working quant must understand this material rather than trusting a library blindly.

### The everyday version of collinearity

Imagine trying to figure out how much two ingredients contribute to a cake's sweetness, but in every recipe you have ever baked, you added *the same amount of sugar and honey together*. You can tell the combination is sweet, but you can never separate sugar's effect from honey's -- maybe sugar does all the work and honey none, or the reverse, or any split in between. The data simply cannot tell them apart. That is *collinearity*: two factors that move together so tightly the regression cannot decide how to allocate credit between them.

When two factors are collinear, the column space they span barely fills two dimensions -- it is nearly a single line. The matrix $X$ nearly collapses a dimension, which means its smallest singular value $\sigma_{\min}$ is nearly zero, which means the condition number $\kappa = \sigma_{\max}/\sigma_{\min}$ is enormous. And here is the killer fact: the *uncertainty in your coefficients scales with the condition number*. A condition number of a million means a one-part-in-a-million wiggle in your data can swing your coefficients by a factor of a million.

Some rough rules of thumb make the number actionable. A condition number under about 30 is healthy -- your factors are well separated and the coefficients are trustworthy. Between 30 and 100, mild collinearity is present; the coefficients are still usable but their standard errors are inflated, so be skeptical of any single one. Between 100 and 1,000 you are in real trouble: the individual coefficients are largely noise even if the overall fit looks fine. Above roughly $10^6$ the coefficients are essentially random, and any hedge built on them is a coin flip. A useful mental conversion: $\log_{10}\kappa$ is roughly the number of decimal digits of precision you lose in the answer. Double-precision arithmetic carries about 16 digits, so a condition number of $10^8$ leaves you only 8 trustworthy digits, and $10^{16}$ leaves you none -- the answer is pure rounding noise. This is why a working quant reads the condition number *before* believing a single coefficient, the way a pilot checks instruments before trusting the autopilot.

There is also a statistical face of the same number, called the *variance inflation factor* (VIF). For each factor, the VIF measures how much that factor's coefficient variance is inflated by its correlation with the other factors. A VIF of 1 means no inflation (the factor is uncorrelated with the rest); a VIF of 10 means the coefficient's standard error is $\sqrt{10} \approx 3.2$ times larger than it would be with clean factors. VIF and the condition number are two windows onto the same disease -- near-zero singular values -- and a careful risk report quotes at least one of them next to every regression coefficient, because a coefficient without a conditioning diagnostic is a number with no error bars.

![A descending ladder of singular values shows where real signal ends and where a tiny noise value should be truncated.](/imgs/blogs/svd-least-squares-regression-math-for-quants-6.png)

The figure above is the diagnostic you actually look at: a stack of the singular values from largest to smallest. The big ones at the top are real, robust directions of variation in your factors. The tiny one at the bottom -- here 0.02, while the top is 12.0, a condition number of 600 -- is a direction where your factors barely move, and dividing by it (which the inverse does) amplifies noise. The dotted cutoff is where a practitioner *truncates*: throw away singular values below a threshold and the instability disappears.

#### Worked example: two collinear factors and the cure

You want to explain a portfolio's return with two factors that are almost the same thing -- say, a large-cap value index and a large-cap value ETF that tracks it. Over four days (returns in percent):

$$X = \begin{bmatrix} 1.00 & 1.01 \\ 2.00 & 2.01 \\ 3.00 & 2.99 \\ 4.00 & 4.02 \end{bmatrix}, \qquad y = \begin{bmatrix} 2.0 \\ 4.0 \\ 6.0 \\ 8.0 \end{bmatrix}.$$

The two columns are nearly identical -- the second is the first plus a whisper of noise. The true relationship is simple: $y$ is just twice column one (or twice column two, or any blend). But watch what the normal equations do. Form $X^\top X$:

$$X^\top X \approx \begin{bmatrix} 30.00 & 30.07 \\ 30.07 & 30.15 \end{bmatrix}.$$

(Using $1^2+2^2+3^2+4^2 = 30$ for the top-left, and the near-identical cross and second-column sums.) The determinant is $30.00\times30.15 - 30.07\times30.07 \approx 904.5 - 904.2 = 0.3$ -- a hair above zero, compared to the comfortable 116 we saw with well-separated factors. Inverting a matrix with a determinant of 0.3 means dividing by 0.3 after multiplying large numbers, and any rounding in those large numbers gets blown up. The resulting coefficients come out wild -- something like $\hat\beta \approx (+480, -478)$ instead of a sensible split like $(1, 1)$. The two huge coefficients nearly cancel ($480 - 478 \approx 2$, which is the true total effect), but each one individually is nonsense, and tomorrow's data could flip them to $(-470, +472)$.

This is catastrophic for a hedge. If your model says "be long 480 units of the index and short 478 units of the ETF," you are putting on a colossal, leveraged spread bet that the two will diverge -- when all you actually wanted was a modest 2x exposure. On a \$1,000,000 book those coefficients would tell you to take on roughly \$480,000,000 of gross notional. One bad fill on the spread and you are wiped out.

Now the cure: *truncated SVD*. Take the SVD of $X$ and look at its singular values. You will find something like $\sigma_1 \approx 7.76$ and $\sigma_2 \approx 0.022$ -- a condition number of about 350, with the second value almost zero because the columns barely differ. The instability lives entirely in that tiny $\sigma_2$. So we *truncate*: set $\sigma_2$'s contribution to zero (do not invert it; treat $1/\sigma_2$ as 0) and rebuild $\hat\beta$ from only the robust $\sigma_1$ direction. The pseudo-inverse with $\sigma_2$ dropped returns the minimum-length solution, which splits the effect evenly: roughly $\hat\beta \approx (1.0, 1.0)$. Two modest, sensible coefficients that sum to the same total effect of 2, but with none of the explosive leverage.

![Dropping the tiny singular value converts an unstable plus-480 minus-470 fit into a stable fit near one and one.](/imgs/blogs/svd-least-squares-regression-math-for-quants-7.png)

The before-and-after figure makes the cure visceral. On the left, the full SVD keeps the tiny singular value and produces coefficients of $+480$ and $-470$ that nearly cancel -- a fragile, over-leveraged fit. On the right, truncating that singular value yields coefficients near $1$ and $1$ -- stable, modest, and just as accurate on the data that matters. The hedge built on the right is one you can actually hold overnight. **The intuition: collinearity does not destroy the information your factors carry jointly; it only destroys your ability to split that information between them -- and truncating the smallest singular value tells the model to stop pretending it can split what it cannot.**

## 7. Truncated SVD and ridge: regularization by throwing away noise

The truncation trick we just used is one of two close cousins that every quant should know. Both are forms of *regularization* -- deliberately accepting a tiny bit of bias to buy a large reduction in variance.

### Truncated SVD as a hard cut

Truncated SVD is the *hard* version: rank the singular values, pick a cutoff, and zero out everything below it. You keep only the top $r$ directions, which is why this is also called a *low-rank* approximation -- you are approximating $X$ with a simpler matrix of rank $r$. The kept directions are the robust, high-variance combinations of your factors; the discarded ones are the fragile, low-variance combinations that mostly carry noise. The choice of $r$ is the one judgment call: keep too many and you let noise back in; keep too few and you throw away real signal. A common rule is to drop any singular value smaller than, say, 1% of the largest.

### Ridge regression as a soft cut

Ridge regression is the *soft* version. Instead of zeroing small singular values, it *shrinks* them smoothly. The ridge estimate adds a penalty $\lambda\lVert\beta\rVert^2$ to the squared error, which works out to replacing each $1/\sigma_i$ in the pseudo-inverse with:

$$\frac{\sigma_i}{\sigma_i^2 + \lambda}.$$

Stare at that fraction. When $\sigma_i$ is large (a robust direction), $\sigma_i^2$ dominates $\lambda$ and the factor is essentially $1/\sigma_i$ -- ridge leaves big directions almost untouched. When $\sigma_i$ is tiny (the noisy collinear direction), $\sigma_i^2$ is negligible next to $\lambda$ and the factor becomes about $\sigma_i/\lambda$, which goes to *zero* as $\sigma_i$ shrinks -- ridge gently mutes the dangerous directions instead of guillotining them. That is the whole mechanism: ridge is a smooth dial where truncated SVD is an on-off switch, and both attack the exact same disease, those near-zero singular values.

| Approach | What it does to small $\sigma_i$ | When to reach for it |
|---|---|---|
| Normal equations | Inverts them anyway ($1/\sigma_i$ explodes) | Only when factors are clean and well-separated |
| Truncated SVD | Drops them entirely (sets $1/\sigma_i = 0$) | When you can name a clear noise floor / rank |
| Ridge regression | Shrinks them smoothly toward zero | When you want a tunable, continuous defense |

The practical upshot: if a desk hands you a factor model with twenty factors and several of them are obviously redundant, you do not delete factors by hand and you do not trust the raw normal equations. You either truncate the SVD at a sensible rank or add a small ridge penalty, and you tune the cutoff or $\lambda$ by checking out-of-sample performance.

#### Worked example: how much does regularization move the dollars?

Return to the collinear book from the last example, sized at \$1,000,000. The raw fit said long \$480,000,000 of the index and short \$478,000,000 of the ETF -- about \$958,000,000 of gross notional, an absurd 958x leverage on a \$1,000,000 book. After truncating the small singular value, the fit becomes long \$1,000,000 of the index and long \$1,000,000 of the ETF (coefficients near 1 and 1), for \$2,000,000 of gross notional -- a sane 2x. The economic exposure is essentially the same (both deliver roughly 2x value exposure), but the regularized version cut gross notional by a factor of nearly 500 and removed a divergence bet nobody wanted. **The intuition: regularization is not about getting a "more accurate" number on today's data -- the raw fit nails today's data too -- it is about getting a number that will not bankrupt you when tomorrow's data wiggles.**

## 8. The connection to PCA, and the full toolkit

We have been treating $X$ as a matrix of factors against which we regress. But the SVD of a *data* matrix -- rows are days, columns are assets, entries are returns -- is exactly *principal component analysis* (PCA). The right singular vectors (columns of $V$) are the principal components: the combinations of assets that vary together most. The singular values squared are proportional to the variance each component explains. The first component is usually "the market" (everything moving together); later ones are sector tilts, style tilts, and finally idiosyncratic noise.

That is why these topics are one family. PCA finds the robust directions of variation; truncated SVD keeps exactly those directions when solving a regression; ridge shrinks the fragile ones. A quant who internalizes the singular values of the return matrix can read, in one glance, how many real risk factors a basket has, how collinear the factors are, and how much a regression built on them can be trusted. The same numbers answer "how many bets am I really making?" and "will my hedge ratio be stable next week?"

### A compact recipe for a real regression

Putting it together, here is what computing a trustworthy hedge ratio actually looks like in practice, in order:

1. **Build $X$ and $y$.** Columns of $X$ are your factors (market, sectors, styles), each row an observation; $y$ is the thing you want to hedge or explain. Add a column of ones for the intercept.
2. **Take the SVD of $X$**, never the inverse of $X^\top X$. Read the singular values.
3. **Compute the condition number** $\sigma_{\max}/\sigma_{\min}$. Under ~30 is fine; over ~1000 is a red flag; over ~$10^6$ means your coefficients are essentially random.
4. **If the condition number is large, regularize** -- truncate the SVD at a sensible rank, or add a ridge penalty $\lambda$ tuned on held-out data.
5. **Read off $\hat\beta = X^+y$** (with the truncated or shrunk $\Sigma^+$), and translate the relevant coefficient into a dollar hedge: short $\beta \times \text{notional}$ of the factor.
6. **Re-fit periodically.** Betas drift; a hedge ratio computed last quarter is stale.

Every line of that recipe is something we built from zero in this post, and every line ties to a dollar decision a desk actually makes.

## Common misconceptions

**"The normal equations and the SVD give different answers."** No -- when $X$ is well-conditioned they give the *same* answer to many decimal places. The SVD is not a different estimator; it is a different, more numerically stable *route* to the identical least-squares solution. The disagreement appears only when the matrix is ill-conditioned, and there the normal equations are simply wrong (corrupted by amplified rounding error) while the SVD is right.

**"A high $R^2$ means my factors are good."** $R^2$ measures how much variance your fit explains, and it says nothing about whether your *coefficients* are stable. Two collinear factors can produce a beautiful $R^2$ of 0.99 while their individual betas are pure noise that flips sign on new data. The condition number, not $R^2$, tells you whether the coefficients can be trusted.

**"More factors always explain more, so add them."** Each redundant factor you add nudges the smallest singular value toward zero and inflates your condition number. Past a point, new factors do not add information; they add instability. A parsimonious model with three well-separated factors beats a kitchen-sink model with twenty collinear ones, every time.

**"The pseudo-inverse is an approximation."** When an ordinary inverse exists, the pseudo-inverse equals it exactly. When it does not, the pseudo-inverse is not approximating some "true" inverse that exists -- there is no true inverse -- it is giving the exact, uniquely-defined best answer (least squared error, and among ties, the smallest-length coefficients). It is exact about a precisely specified goal.

**"Regularization makes my model worse because it biases the coefficients."** It does add a little bias, deliberately. But the coefficients of an ill-conditioned regression have astronomical variance, and a tiny bias that slashes variance produces a coefficient that is far more accurate *out of sample* -- which is the only sample that pays. Trading today's in-sample fit for tomorrow's stability is the entire point.

**"Collinearity destroys information."** It does not destroy the *joint* information -- two collinear factors still pin down their combined effect perfectly (recall the wild $+480/-478$ summed to the correct total of 2). What collinearity destroys is your ability to *attribute* the effect to one factor versus the other. The fix is not more data in the same direction; it is to stop asking the model to split what cannot be split.

## How it shows up in real markets

### 1. The classic single-stock hedge

The most everyday use is exactly our first worked example. A portfolio manager long a single name regresses that stock's daily returns on the index over a rolling window, reads off the beta, and shorts beta-times-notional of an index future or ETF to strip out market risk. A \$10,000,000 long in a $\beta = 1.2$ name calls for a \$12,000,000 index short. The mechanism is pure least squares; the failure mode is using a stale beta -- betas drift through earnings, regime changes, and capital-structure shifts -- which is why desks re-fit weekly or daily, and why the [linear regression deep-dive](/blog/trading/quantitative-finance/linear-regression-deep-quant-interviews) hammers on the rolling-window discipline.

### 2. Factor models and the redundant-factor trap

Multi-factor risk models (the descendants of the academic Fama-French factors) regress each stock on factors like market, size, value, momentum, and quality. The trap is that value and quality, or size and a particular sector, are often highly correlated in a given universe -- and a naive normal-equations fit then reports unstable loadings that risk managers misread. Production risk systems use SVD-based or ridge-regularized fits precisely to keep loadings stable across rebalances, so that a stock's reported "value exposure" does not whipsaw from +0.8 to -0.5 between Monday and Tuesday on noise.

### 3. Yield-curve hedging and near-collinear tenors

A bond desk hedging a portfolio across the yield curve regresses against the curve's level, slope, and curvature. The problem: adjacent maturities (the 9-year and 10-year, say) move almost identically, so a regression on individual tenors is brutally collinear. The standard fix is to first run PCA (an SVD of the curve's daily changes), find that three components -- level, slope, curvature -- explain over 95% of curve variance, and hedge against those three robust components instead of the dozens of collinear individual tenors. This is truncated SVD doing risk management: keep three singular values, discard the noisy rest.

### 4. Statistical arbitrage and pairs

Stat-arb strategies look for combinations of assets whose joint movement is more predictable than any single name. Finding the right weights is a least-squares problem, and the assets are deliberately chosen to be similar (two refiners, two airlines), which means collinear -- exactly the regime where the normal equations misbehave. Practitioners use regularized regression and rank-constrained (truncated-SVD) fits so the hedge weights are stable enough to trade, because a pairs weight that flips sign on noise is a strategy that bleeds on transaction costs.

### 5. The 2007 quant quake and crowded factors

In August 2007, many quantitative equity funds were running similar factor models on similar collinear factors. When a few large funds deleveraged at once, the crowded factor exposures unwound violently and "market-neutral" books that looked perfectly hedged on paper took double-digit losses in days. Part of the lesson, in hindsight, was about *instability of exposures*: when your factor loadings sit on near-zero singular values, your "hedge" is a fragile spread bet in disguise, and it can detonate when everyone reaches for the same exit. Regularization would not have prevented the crowding, but it would have made the true gross exposure legible rather than hidden inside cancelling coefficients.

### 6. Index replication with a basket

A fund wanting to track an index cheaply with a small basket of stocks solves a least-squares problem: pick weights minimizing the tracking error between the basket's return and the index's. With hundreds of candidate stocks, many of them correlated, the design matrix is wildly collinear, and a raw fit produces a basket with huge offsetting long-short weights that costs a fortune to trade. Truncated SVD or an $\ell_2$ penalty produces a sparse, modest, tradeable basket that tracks nearly as well at a fraction of the turnover. The pseudo-inverse's "smallest-length coefficients" tie-break is doing real economic work here, choosing the cheapest basket among many that fit equally.

## When this matters to you

If you ever sit on a trading or risk desk, this is not optional background -- it is the daily mechanics of computing a hedge and reading whether you can trust it. The hedge ratio that protects a position, the factor loadings that tell you what bets you are really making, the basket that replicates an index: all of them are least squares, and all of them quietly fail the same way when your inputs look too much alike. Knowing to take an SVD instead of inverting $X^\top X$, to glance at the condition number before believing a coefficient, and to truncate or ridge when that number is large is the difference between a hedge you can hold and a leveraged spread bet you did not know you put on.

Even if you never trade, the same machinery runs inside most of applied data work: every regression in a spreadsheet, every "trend line", every machine-learning model with a linear layer is solving $\hat\beta = X^+y$ somewhere, and every one of them is vulnerable to collinearity. The condition number is a universal smoke detector. When a model's coefficients look insane, your first question should be "what does the SVD say?"

This has been educational material about how a standard tool works, not investment advice; any time real money rides on a hedge ratio, validate it out of sample and size it for the chance you are wrong.

**Further reading on this blog:**

- [Linear regression from first principles for quant interviews](/blog/trading/quantitative-finance/linear-regression-deep-quant-interviews) -- the single-variable foundation, the Gauss-Markov assumptions, and how to read a regression table.
- [Building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) -- why you regress your raw signal on known factors and keep the orthogonal residual.
- [Covariance and correlation pitfalls](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews) -- the covariance matrix that is hiding inside $X^\top X$, and where correlation estimates go wrong.

**Primary sources worth your time:** Gilbert Strang's *Introduction to Linear Algebra* for the projection-and-SVD geometry, Trefethen and Bau's *Numerical Linear Algebra* for why the SVD beats the normal equations numerically, and Golub and Van Loan's *Matrix Computations* for the canonical treatment of the pseudo-inverse and conditioning.
