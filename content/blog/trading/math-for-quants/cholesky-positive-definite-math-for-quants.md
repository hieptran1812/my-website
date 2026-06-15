---
title: "Positive-definiteness and the Cholesky factorization"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A from-scratch deep dive on why a valid risk model must be positive-definite, how the Cholesky factorization splits a covariance matrix into L times its transpose, and how that single factor turns independent random draws into correlated asset returns for Monte Carlo scenario generation, value at risk, the nearest-correlation-matrix repair, and fast Gaussian likelihoods."
tags:
  [
    "cholesky",
    "positive-definite",
    "covariance-matrix",
    "correlation",
    "monte-carlo",
    "value-at-risk",
    "risk-modeling",
    "linear-algebra",
    "nearest-correlation-matrix",
    "scenario-generation",
    "quantitative-finance",
    "math-for-quants",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A covariance matrix is only a *valid* risk model if it is **positive-definite** (no combination of holdings can have negative variance), and once it is, the **Cholesky factorization** $\Sigma = LL^\top$ hands you the single most useful tool in simulation: a recipe for turning independent random noise into correlated asset returns.
>
> - Positive-definite means $w^\top \Sigma w > 0$ for every non-zero portfolio $w$ — risk is always strictly positive. Positive-*semi*-definite allows $\ge 0$ (a perfectly hedged combination can have exactly zero risk). A matrix that fails this is not a risk model; it is a typo.
> - Cholesky splits $\Sigma$ into a lower-triangular matrix $L$ times its own transpose: $\Sigma = LL^\top$. Think of $L$ as the unique "square root" of your risk.
> - The workhorse identity: if $z$ is a vector of independent standard normals, then $x = \mu + Lz$ has mean $\mu$ and covariance exactly $\Sigma$. That one line is how every Monte Carlo engine generates correlated scenarios.
> - A sample correlation matrix built from messy data (missing prices, stale quotes, different history lengths) can fail to be positive-semi-definite. The fix is the **nearest-correlation-matrix** repair: compute the eigenvalues, clip the negative ones to zero, and rebuild.
> - We Monte Carlo the 1-day 99% value at risk of a **\$1,000,000** two-asset book and land near **\$28,800** — and we get there using nothing but a 2×2 Cholesky factor and a stream of independent normals.

You have a portfolio. Maybe it is two stocks, maybe it is two thousand. You want to answer a deceptively simple question: *how much could I lose tomorrow?* To answer it, you cannot just look at each holding in isolation, because the holdings move *together* — when tech sells off, your software stock and your semiconductor stock both fall, and the losses pile on top of each other instead of cancelling. The whole game of portfolio risk is the game of correlation, and the object that stores all of it is the **covariance matrix**.

But here is the catch that trips up almost everyone the first time. You cannot just write down any grid of numbers, call it a covariance matrix, and start simulating. Most grids of numbers are *impossible* — they describe a world where some combination of your assets would have negative risk, which is as nonsensical as a negative distance. The property that separates a real risk model from an impossible one is called **positive-definiteness**, and the algorithm that both *tests* for it and *exploits* it is the **Cholesky factorization**. The diagram below is the mental model for the entire post: independent random noise goes in on the left, passes through the Cholesky factor $L$, and comes out the right side as correlated asset returns that match your covariance matrix exactly.

![A pipeline showing independent draws z feeding into the factor Sigma equals L L transpose, then multiplying to give x equals mu plus L z, producing correlated returns x.](/imgs/blogs/cholesky-positive-definite-math-for-quants-1.png)

We will build this from zero. No assumed background in linear algebra beyond "a matrix is a grid of numbers", and no assumed finance background beyond "a return is a percentage gain or loss". By the end you will understand what positive-definiteness *means* geometrically, why a risk model must have it, how to compute a Cholesky factor by hand for a small matrix, how to use that factor to simulate correlated returns, how to repair a broken correlation matrix that data quality wrecked, and how to read off a Monte Carlo value-at-risk number from the whole machine. Every section is anchored by a worked example with real dollar figures. This is educational material, not investment advice — the goal is to make the machinery legible, not to tell you what to trade.

## Foundations: the building blocks of a risk model

Let us define every term carefully, because the rest of the post leans on these and nothing else.

A **return** is the percentage change in the price of an asset over some period. If a stock goes from \$100 to \$102 in a day, its daily return is $+2\%$, or $0.02$ in decimal form. Returns are the natural unit of risk because they are comparable across assets of different prices — a \$10 stock and a \$1,000 stock can both have a 2% day. Throughout this post, when we say "return" we mean a single-period return expressed as a decimal.

A **random variable** is a number whose value is determined by chance. Tomorrow's return on a stock is a random variable: you do not know it yet, but you can describe its likely range. The two numbers we care about most are its **mean** (the average value you expect, written $\mu$, the Greek letter "mu") and its **variance** (how spread out the possible values are around that mean, written $\sigma^2$, "sigma squared"). The square root of variance is the **standard deviation** $\sigma$, also called **volatility** in finance — it is the typical size of a daily move, measured in the same units as the return itself. A stock with $\sigma = 2\%$ per day typically moves about two percent up or down on a given day.

### Covariance and correlation: how two assets move together

The whole reason a portfolio is more than the sum of its parts is that assets move *together*. **Covariance** measures that co-movement. The covariance between two assets $A$ and $B$, written $\mathrm{Cov}(A, B)$ or $\sigma_{AB}$, is the average of the product of their deviations from their own means. In plain English: on days when $A$ is above its average, is $B$ also above its average? If yes, covariance is positive; if $B$ tends to be below its average on those days, covariance is negative; if there is no pattern, covariance is near zero.

Covariance has awkward units (it is a product of two returns, so it is in "squared return" units, and its size depends on how volatile the assets are). To get something more interpretable we divide it by the two volatilities to get the **correlation**:

$$\rho_{AB} = \frac{\sigma_{AB}}{\sigma_A \, \sigma_B}$$

Here $\rho$ (the Greek letter "rho") is the correlation, $\sigma_{AB}$ is the covariance, and $\sigma_A, \sigma_B$ are the two volatilities. Correlation is always between $-1$ and $+1$. A correlation of $+1$ means the two assets move in perfect lockstep; $-1$ means they move perfectly opposite; $0$ means no linear relationship. Correlation is the *clean*, unit-free version of covariance, which is why risk people talk in correlations even though the math runs on covariances. (If you want the full tour of the traps hiding in these two quantities, the [covariance and correlation pitfalls](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews) post is a companion to this one.)

### The covariance matrix: all the risk in one grid

Now stack every asset's variance and every pair's covariance into a single grid. For $n$ assets, the **covariance matrix** $\Sigma$ (capital sigma) is an $n \times n$ table where the entry in row $i$, column $j$ is the covariance between asset $i$ and asset $j$. The diagonal entries (row $i$, column $i$) are each asset's own variance, because the "covariance of an asset with itself" is just its variance. The off-diagonal entries are the pairwise covariances. The matrix is **symmetric** — the covariance of $A$ with $B$ equals the covariance of $B$ with $A$ — so the grid is a mirror image across its diagonal.

For a two-asset book the covariance matrix is just

$$\Sigma = \begin{pmatrix} \sigma_A^2 & \sigma_{AB} \\ \sigma_{AB} & \sigma_B^2 \end{pmatrix}.$$

The diagonal holds the two variances $\sigma_A^2$ and $\sigma_B^2$; the two off-diagonal cells both hold the single covariance $\sigma_{AB}$. That is the entire risk model for two assets: three distinct numbers. If you want the ground-up construction of $\Sigma$ from raw return data, that is the subject of the sibling post on the covariance matrix; here we take $\Sigma$ as given and ask what we can *do* with it.

### Portfolio variance: why we need the matrix at all

A **portfolio** is a set of holdings, summarized by a vector of **weights** $w = (w_1, w_2, \dots, w_n)$ giving the fraction of your money (or your dollar exposure) in each asset. The single most important formula in this post is the one that gives the variance of the whole portfolio:

$$\mathrm{Var}(\text{portfolio}) = w^\top \Sigma w.$$

Read $w^\top \Sigma w$ as "$w$-transpose times $\Sigma$ times $w$". It is a **quadratic form** — a way of squashing a whole matrix down to a single number using a vector on both sides. For two assets it expands to

$$w^\top \Sigma w = w_A^2 \sigma_A^2 + w_B^2 \sigma_B^2 + 2 w_A w_B \sigma_{AB},$$

which says: the portfolio's variance is each asset's own variance (weighted by the square of its weight), plus a cross term that depends on the covariance. When the cross term is negative — when the assets move opposite each other — the portfolio's variance is *less* than the sum of its parts. That is diversification, expressed in one line of algebra.

#### Worked example: portfolio variance of a two-asset book

Let us put numbers on it. You hold a \$600,000 position in a tech ETF (call it asset A) and a \$400,000 position in a government-bond ETF (asset B), for a \$1,000,000 book. The tech ETF has a daily volatility of $\sigma_A = 1.5\%$, so $\sigma_A^2 = 0.000225$. The bond ETF has $\sigma_B = 0.5\%$, so $\sigma_B^2 = 0.000025$. Their correlation is $\rho = -0.30$ (bonds tend to rally when stocks fall), so the covariance is $\sigma_{AB} = \rho \, \sigma_A \sigma_B = -0.30 \times 0.015 \times 0.005 = -0.0000225$.

Working in *dollar* exposures instead of fractions, the weights are $w_A = \$600{,}000$ and $w_B = \$400{,}000$. The portfolio's dollar variance is

$$w^\top \Sigma w = (600{,}000)^2 (0.000225) + (400{,}000)^2 (0.000025) + 2(600{,}000)(400{,}000)(-0.0000225).$$

Term by term: the first is $360{,}000{,}000{,}000 \times 0.000225 = 81{,}000{,}000$. The second is $160{,}000{,}000{,}000 \times 0.000025 = 4{,}000{,}000$. The third is $2 \times 240{,}000{,}000{,}000 \times (-0.0000225) = -10{,}800{,}000$. Summing: $81{,}000{,}000 + 4{,}000{,}000 - 10{,}800{,}000 = 74{,}200{,}000$ dollars-squared. The portfolio's daily dollar volatility is the square root: $\sqrt{74{,}200{,}000} \approx \$8{,}614$.

The intuition: because the bond is negatively correlated with the stock, the cross term *subtracts* \$10.8 million of variance, and the blended one-day risk of \$8,614 is meaningfully below what you would get by naively adding the two standalone risks together.

## Positive-definiteness: why risk can never go negative

Here is a property of that last formula that is so important it has its own name. Variance — of a single asset or of any portfolio you can build from your assets — is a measure of spread. Spread cannot be negative. There is no such thing as a portfolio whose returns are "less than perfectly certain by a negative amount." So whatever combination of weights $w$ you pick, the quantity $w^\top \Sigma w$ *must* come out non-negative, and for any portfolio that actually carries some risk it must come out strictly positive. That requirement is exactly the mathematical definition of **positive-definiteness**.

Formally, a symmetric matrix $\Sigma$ is **positive-definite** (often abbreviated PD) if

$$w^\top \Sigma w > 0 \quad \text{for every non-zero vector } w,$$

and it is **positive-semi-definite** (PSD) if the same expression is $\ge 0$ for every $w$ (it is allowed to hit exactly zero for some non-zero $w$). The distinction is small but it matters: positive-*semi*-definite permits a non-zero portfolio with *exactly* zero variance, which happens when one asset is a perfect linear combination of the others — a redundancy. Positive-*definite* rules even that out: every non-trivial portfolio has strictly positive risk.

The matrix below shows the simplest non-trivial example, a 2×2 correlation matrix with off-diagonal $0.60$, which is comfortably positive-definite. We will factor exactly this matrix by hand in a moment.

![A two by two matrix with ones on the diagonal and zero point six on both off-diagonal cells.](/imgs/blogs/cholesky-positive-definite-math-for-quants-2.png)

The diagram above is a valid correlation matrix: diagonal entries of $1.00$ (every asset is perfectly correlated with itself) and symmetric off-diagonal entries of $0.60$. Plug any non-zero $w$ into $w^\top \Sigma w$ and you will always get a positive number, which is what makes it a legitimate risk model.

### The three equivalent ways to read positive-definiteness

There are three lenses on the same property, and a quant should be fluent in all three because each one shows up in a different tool.

**The quadratic-form lens (risk view).** $w^\top \Sigma w > 0$ for all non-zero $w$. This is the definition we just gave, and it is the *meaning*: every portfolio has positive variance. It is conceptually clean but expensive to check directly — you cannot try every possible $w$.

**The eigenvalue lens (geometry view).** A symmetric matrix can be decomposed into a set of **eigenvalues** (numbers, written $\lambda$, "lambda") and **eigenvectors** (directions). Each eigenvalue is the variance of the portfolio that points along its eigenvector. The matrix is positive-definite if and only if *all* its eigenvalues are strictly positive, and positive-semi-definite if all are $\ge 0$. A *negative* eigenvalue means there is a direction — a specific portfolio — with negative variance, which is impossible, so a negative eigenvalue is the unambiguous fingerprint of a broken matrix. This is the lens behind the repair we do later. (Eigenvalues and the related decomposition are the subject of the next post in this series, on eigendecomposition and PCA.)

**The Cholesky lens (computational view).** A symmetric matrix is positive-definite if and only if it can be written as $\Sigma = LL^\top$ for a real lower-triangular matrix $L$ with strictly positive diagonal entries. This is the most *useful* characterization, because computing $L$ is fast and the attempt itself doubles as a test: if the algorithm ever asks for the square root of a negative number, the matrix was not positive-definite, and you have your answer for free.

These three are not three different facts; they are three faces of one fact. Positive-definiteness is the single condition "this matrix describes a possible world of risk", and the eigenvalue test, the quadratic-form test, and the Cholesky test all detect it.

> A covariance matrix that fails positive-definiteness is not a conservative risk model or an aggressive one. It is not a risk model at all — it is a contradiction wearing the costume of one.

## The Cholesky factorization: the square root of risk

Now to the star of the post. The **Cholesky factorization** takes a positive-definite matrix $\Sigma$ and writes it as the product of a **lower-triangular** matrix $L$ and that same matrix's transpose:

$$\Sigma = L L^\top.$$

"Lower-triangular" means every entry *above* the diagonal is zero — the non-zero numbers form a triangle in the bottom-left. The transpose $L^\top$ flips $L$ across its diagonal, so it is upper-triangular. The matrix below shows the target $\Sigma$ we factor by hand next; remember it, because $L$ for this matrix is what we compute first.

![A two by two matrix with ones on the diagonal and zero point six on both off-diagonal cells, the same correlation matrix to be factored.](/imgs/blogs/cholesky-positive-definite-math-for-quants-2.png)

Why call $L$ the "square root" of $\Sigma$? Because for an ordinary positive number $a$, its square root $\ell$ satisfies $a = \ell \cdot \ell$. The Cholesky factor does the matrix version of exactly this: $\Sigma = L \cdot L^\top$. It is *a* square root, not *the* square root (there are others, like the symmetric matrix square root from the eigenvalue decomposition), but it is the one that is cheapest to compute and the most convenient for simulation, because triangular matrices are easy to multiply by and easy to solve with.

### How the algorithm works, one entry at a time

You do not need to memorize a formula to do a small Cholesky by hand; you just enforce $\Sigma = LL^\top$ entry by entry. For a 2×2 matrix, write

$$L = \begin{pmatrix} \ell_{11} & 0 \\ \ell_{21} & \ell_{22} \end{pmatrix}, \qquad L L^\top = \begin{pmatrix} \ell_{11}^2 & \ell_{11}\ell_{21} \\ \ell_{11}\ell_{21} & \ell_{21}^2 + \ell_{22}^2 \end{pmatrix}.$$

Now match this against $\Sigma = \begin{pmatrix} \sigma_{11} & \sigma_{12} \\ \sigma_{12} & \sigma_{22} \end{pmatrix}$ cell by cell, top-left first and working down:

1. Top-left: $\ell_{11}^2 = \sigma_{11}$, so $\ell_{11} = \sqrt{\sigma_{11}}$.
2. Bottom-left: $\ell_{11}\ell_{21} = \sigma_{12}$, so $\ell_{21} = \sigma_{12} / \ell_{11}$.
3. Bottom-right: $\ell_{21}^2 + \ell_{22}^2 = \sigma_{22}$, so $\ell_{22} = \sqrt{\sigma_{22} - \ell_{21}^2}$.

Notice step 1 and step 3 both take a square root. *That* is where positive-definiteness shows up computationally: if $\sigma_{11}$ or the leftover $\sigma_{22} - \ell_{21}^2$ ever comes out negative, the square root is not a real number and the algorithm halts. A clean run to the end is a certificate that the matrix was positive-definite. The general $n \times n$ algorithm is the same idea — march down the diagonal, take a square root for each diagonal entry and a division for each below-diagonal entry — and it costs about $n^3/3$ arithmetic operations, roughly half the work of a general LU decomposition, which is one reason it is the default for symmetric positive-definite systems everywhere.

#### Worked example: Cholesky of a 2×2 correlation matrix by hand

Take the correlation matrix from the figures, with off-diagonal $0.60$:

$$\Sigma = \begin{pmatrix} 1.00 & 0.60 \\ 0.60 & 1.00 \end{pmatrix}.$$

Apply the three steps. First, $\ell_{11} = \sqrt{1.00} = 1.00$. Second, $\ell_{21} = 0.60 / 1.00 = 0.60$. Third, $\ell_{22} = \sqrt{1.00 - 0.60^2} = \sqrt{1.00 - 0.36} = \sqrt{0.64} = 0.80$. So

$$L = \begin{pmatrix} 1.00 & 0 \\ 0.60 & 0.80 \end{pmatrix}.$$

Let us verify by multiplying $LL^\top$ back out. The top-left is $1.00^2 = 1.00$. The off-diagonal is $1.00 \times 0.60 = 0.60$. The bottom-right is $0.60^2 + 0.80^2 = 0.36 + 0.64 = 1.00$. We recover $\Sigma$ exactly, which confirms $L$ is correct.

To make this concrete in dollars: if you had two assets each with \$10,000 of daily volatility and a $0.60$ correlation, this same $L$ is the gadget that will let you draw thousands of joint scenarios for the pair where roughly 60% of their day-to-day variation moves in common. The intuition: the second row of $L$, $(0.60, 0.80)$, says that asset B's return is built as $0.60$ parts of asset A's shock plus $0.80$ parts of its own independent shock — the factorization literally decomposes B's risk into "shared with A" and "unique to B."

### Scaling the hand-method to three assets

The same march-down-the-diagonal procedure handles any size; it just gains more steps. Take a 3-asset correlation matrix where every pair is tied at $0.50$:

$$C = \begin{pmatrix} 1.0 & 0.5 & 0.5 \\ 0.5 & 1.0 & 0.5 \\ 0.5 & 0.5 & 1.0 \end{pmatrix}.$$

Work the entries of $L$ left to right, top to bottom. The first column comes straight from the first row of $C$: $\ell_{11} = \sqrt{1.0} = 1.0$, then $\ell_{21} = 0.5 / 1.0 = 0.5$ and $\ell_{31} = 0.5 / 1.0 = 0.5$. For the second diagonal entry, subtract what asset 2 already shares with asset 1: $\ell_{22} = \sqrt{1.0 - \ell_{21}^2} = \sqrt{1.0 - 0.25} = \sqrt{0.75} \approx 0.866$. The third-row middle entry uses the matched cross term: $\ell_{32} = (0.5 - \ell_{31}\ell_{21}) / \ell_{22} = (0.5 - 0.25)/0.866 = 0.25/0.866 \approx 0.289$. Finally the last diagonal entry subtracts everything asset 3 shares with the first two: $\ell_{33} = \sqrt{1.0 - \ell_{31}^2 - \ell_{32}^2} = \sqrt{1.0 - 0.25 - 0.0833} = \sqrt{0.6667} \approx 0.816$. So

$$L = \begin{pmatrix} 1.0 & 0 & 0 \\ 0.5 & 0.866 & 0 \\ 0.5 & 0.289 & 0.816 \end{pmatrix}.$$

The triangular structure tells the same story it did for two assets, just longer: asset 1 is the pure driver; asset 2 is half of asset 1's shock plus $0.866$ of its own; asset 3 is half of asset 1's shock, a little of asset 2's residual, and $0.816$ of a shock unique to itself. Each diagonal entry shrinks as you go down because more of each later asset's variance has already been "claimed" by the assets above it.

In code, you never do this by hand past 2×2 — you call a library — but it is the *same* arithmetic. In Python:

```python
import numpy as np

C = np.array([[1.0, 0.5, 0.5],
              [0.5, 1.0, 0.5],
              [0.5, 0.5, 1.0]])
L = np.linalg.cholesky(C)   # lower-triangular factor, L @ L.T == C
z = np.random.standard_normal(size=(3, 100_000))  # independent draws
x = L @ z                   # correlated draws: cov(x) ≈ C
```

`np.linalg.cholesky` raises `LinAlgError` the instant the input is not positive-definite — that exception *is* the positive-definiteness test, surfaced as a crash you must handle. The intuition to carry forward: a Cholesky of a 3×3 or a 3000×3000 matrix is the same idea as the 2×2 you did by hand — take a square root down the diagonal, divide below it, and let each asset inherit the shocks of the ones before it.

## How Cholesky actually makes correlated returns

This is the payoff section, the reason every Monte Carlo engine on a trading floor has a Cholesky factor buried inside it. We have a covariance matrix $\Sigma$, we have factored it into $L$, and now we want to *generate* random asset returns that have exactly that covariance. The trick is one of the most elegant in all of applied math, and it rests on a single property of how covariance behaves under a linear transformation.

Start with the raw material: a vector $z = (z_1, z_2, \dots, z_n)$ of **independent standard normals**. "Standard normal" means each $z_i$ is drawn from the bell curve with mean 0 and variance 1; "independent" means knowing one tells you nothing about the others. These are the easiest random numbers in the world to produce — every language's math library makes them — and crucially they have *no* correlation structure at all. Their covariance matrix is the **identity matrix** $I$ (ones on the diagonal, zeros everywhere else): each has variance 1, and every pair has covariance 0.

Now apply the linear map $x = \mu + Lz$, where $\mu$ is the vector of mean returns. What is the covariance of $x$? Here is the key fact about linear transformations of random vectors: if you multiply a random vector by a matrix $L$, its covariance matrix gets sandwiched, transforming from $\Sigma_z$ to $L \Sigma_z L^\top$. Since our $z$ has covariance $\Sigma_z = I$, the covariance of $x$ is

$$L \, I \, L^\top = L L^\top = \Sigma.$$

That is the whole magic. Because we *chose* $L$ to satisfy $LL^\top = \Sigma$, the transformed vector $x$ comes out with covariance exactly $\Sigma$ and mean exactly $\mu$. The figure below traces it end to end: independent draws $z$ on the left, the factorization in the middle, the multiply, and correlated returns $x$ on the right — it is the same mental-model pipeline from the top of the post, now earned.

![A pipeline showing independent draws z feeding the factorization Sigma equals L L transpose, then x equals mu plus L z, producing correlated returns x.](/imgs/blogs/cholesky-positive-definite-math-for-quants-1.png)

The reason it has to be a *triangular* factor and not just any square root is partly convenience and partly tradition: the lower-triangular form means $x_1$ depends only on $z_1$, $x_2$ depends on $z_1$ and $z_2$, and so on — a clean recursive structure that is fast to evaluate and that gives a tidy "first asset is the driver, later assets add their own twists" interpretation. But any matrix $M$ with $MM^\top = \Sigma$ would correlate the draws correctly; Cholesky is simply the cheapest such $M$.

### Reading the recipe in plain English

Strip away the symbols and the procedure is four steps. One: build your covariance matrix $\Sigma$ from your volatilities and correlations. Two: Cholesky-factor it into $L$. Three: draw a vector of independent standard normals $z$. Four: compute $x = \mu + Lz$, and you have one scenario of correlated returns. Repeat steps three and four a hundred thousand times and you have a hundred thousand plausible tomorrows, each respecting the volatilities and correlations you fed in. This is **scenario generation**, and it is the foundation of simulation-based risk. If you want the broader picture of how simulation answers questions you cannot solve in closed form, the [Monte Carlo and simulation](/blog/trading/quantitative-finance/monte-carlo-simulation-coding-quant-interviews) deep dive is the natural companion.

#### Worked example: simulate two correlated returns

Let us run the machine once, by hand, so there is no mystery left. We will use the two assets from earlier with one simplification for clarity: assume both have a daily volatility of exactly $\sigma = 1.0\%$ (so $\sigma^2 = 0.0001$), a correlation of $\rho = 0.60$, and a mean daily return of $\mu = 0$. The covariance matrix is

$$\Sigma = \begin{pmatrix} 0.0001 & 0.00006 \\ 0.00006 & 0.0001 \end{pmatrix},$$

where the off-diagonal is $\rho \sigma_A \sigma_B = 0.60 \times 0.01 \times 0.01 = 0.00006$. Its Cholesky factor scales the correlation-matrix factor we already found by the volatility $0.01$:

$$L = \begin{pmatrix} 0.01 & 0 \\ 0.006 & 0.008 \end{pmatrix}.$$

(Check: top-left $0.01^2 = 0.0001$; off-diagonal $0.01 \times 0.006 = 0.00006$; bottom-right $0.006^2 + 0.008^2 = 0.000036 + 0.000064 = 0.0001$. Correct.)

Now draw two independent standard normals. Say the random number generator hands us $z_1 = +1.20$ and $z_2 = -0.50$. Apply $x = Lz$ row by row:

- Asset A return: $x_A = 0.01 \times 1.20 + 0 \times (-0.50) = 0.012$, i.e. $+1.2\%$.
- Asset B return: $x_B = 0.006 \times 1.20 + 0.008 \times (-0.50) = 0.0072 - 0.004 = 0.0032$, i.e. $+0.32\%$.

Notice that even though asset B's *own* shock $z_2$ was negative, its return came out *positive*, because it inherited $0.0072$ of upward push from asset A's shock through the off-diagonal term. That shared push is the correlation, made mechanical.

Now the dollar P&L. Suppose you hold \$500,000 of each asset. Asset A makes $0.012 \times \$500{,}000 = +\$6{,}000$. Asset B makes $0.0032 \times \$500{,}000 = +\$1{,}600$. The book's P&L on this single simulated day is $\$6{,}000 + \$1{,}600 = +\$7{,}600$. The figure below shows this exact path: two seeds in, correlated returns through $L$, a dollar result out.

![A graph showing two random draws z1 and z2 entering an apply-L step that splits into a return for asset A and a return for asset B, which combine into a book profit and loss.](/imgs/blogs/cholesky-positive-definite-math-for-quants-4.png)

The intuition: a single Monte Carlo scenario is nothing more than this — two coin-flips of independent noise, run through the Cholesky factor to glue in the correlation, scaled by your dollar holdings, and summed into one day's profit or loss. Do it a hundred thousand times and the distribution of those P&L numbers *is* your risk picture.

### What the sign of correlation does to the simulation

Before we scale up, it is worth pinning down what changing the correlation does to the scenarios, because this is where intuition often slips. The 2×2 grid below lays out the four cases for the *direction* the two assets move on a given simulated day.

![A two by two matrix labeling the four cases both rise together, A rises B falls, A falls B rises, and both fall together.](/imgs/blogs/cholesky-positive-definite-math-for-quants-6.png)

When correlation is *positive*, the simulation produces more days in the two diagonal cells — both assets rising together or both falling together. Those are the dangerous days for a long book, because the losses stack. When correlation is *negative*, the off-diagonal cells dominate — one asset up while the other is down — and the book's swings are damped because the moves partly cancel. The Cholesky factor encodes exactly this: a larger off-diagonal entry $\ell_{21}$ means asset B inherits more of asset A's shock, pushing more scenarios onto the "move together" diagonal. The whole reason correlation matters for risk is that it controls how often the bad-for-you cells light up, and the simulation makes that frequency concrete.

## When the matrix is broken: the nearest-correlation fix

Everything so far assumed $\Sigma$ was positive-definite. In the real world, you build correlation matrices from *data*, and data is filthy. The single most common production failure in a risk system is feeding it a correlation matrix that is *not* positive-semi-definite — at which point the Cholesky factorization fails (a square root of a negative number), the simulation cannot run, and a junior quant spends an afternoon wondering why the code crashed. So we need to understand why this happens and how to repair it.

### Why a sample correlation matrix can fail to be PSD

A correlation matrix estimated cleanly — every asset measured over the *same* days, with the *same* full history — is mathematically guaranteed to be at least positive-semi-definite. The failures come from the ways real data violates that ideal:

- **Missing data and different history lengths.** Asset C only started trading two years ago, while assets A and B have ten years of history. If you estimate $\rho_{AB}$ over ten years but $\rho_{AC}$ and $\rho_{BC}$ over only the two years they overlap, the three correlations are computed over *different* samples and need not be mutually consistent. The result can be a matrix that is internally contradictory.
- **Pairwise deletion.** When some prices are missing, a common shortcut is to compute each pairwise correlation using only the days where *both* of that pair are present. Every entry is then estimated on a different subset of days, and the consistency guarantee evaporates.
- **Stale prices.** An illiquid bond or a thinly traded name may not print a fresh price every day; its quote lags. Stale prices distort correlations in ways that can push the overall matrix off the PSD cone.
- **Hand-edited or blended matrices.** A risk manager who overrides a single correlation by hand ("set the stock-bond correlation to $-0.5$ for the stress scenario") can easily produce an internally inconsistent matrix, because correlations are not free to vary independently — they constrain each other.

The figure below contrasts the broken state and the repaired one: a matrix with a negative eigenvalue, the failed Cholesky, the clipping step, and the valid result.

![A before and after showing an eigenvalue of minus zero point one zero causing Cholesky to fail, then clipping the eigenvalue to zero producing a valid PSD matrix.](/imgs/blogs/cholesky-positive-definite-math-for-quants-3.png)

The before-state on the left is what the data hands you; the after-state on the right is what you must build before you can simulate. The bridge between them is the eigenvalue clip.

### The eigenvalue-clipping repair, intuitively

Recall the eigenvalue lens: a symmetric matrix is positive-semi-definite exactly when all its eigenvalues are $\ge 0$, and a negative eigenvalue is the fingerprint of a "direction with negative variance." The repair follows directly from that picture. Under the hood, the algorithm decomposes the broken matrix into its eigenvalues and eigenvectors, finds the offending negative eigenvalues, and simply *clips them up to zero* (or to a tiny positive floor), leaving the eigenvectors and the non-negative eigenvalues alone. Then it reassembles the matrix from the repaired eigenvalues. The figure above is the visual of this exact move.

The reassembled matrix is positive-semi-definite by construction, and it is, in a precise least-squares sense, *close* to the original — you have changed the matrix as little as possible while making it valid. This is the eigenvalue-clipping flavor of the **nearest-correlation-matrix** problem. (A subtlety: after clipping, the diagonal may drift slightly away from exactly $1$, so a production-grade routine re-normalizes the diagonal back to ones and iterates; the gold-standard method, due to Nick Higham, alternates the eigenvalue projection with a diagonal projection until both hold. The simple one-shot clip is the intuition, and it is usually 95% of the way there.)

#### Worked example: repair a non-PSD 3-asset correlation matrix

Here is a correlation matrix that *looks* innocent but is impossible. Three assets, each pair strongly tied, but with one of the three ties flipped negative:

$$C = \begin{pmatrix} 1.0 & 0.9 & 0.9 \\ 0.9 & 1.0 & -0.9 \\ 0.9 & -0.9 & 1.0 \end{pmatrix}.$$

In words, this claims A is highly correlated with B ($0.9$), A is highly correlated with C ($0.9$), but B and C are highly *anti*-correlated ($-0.9$). That is a contradiction: if A moves almost identically with B and almost identically with C, then B and C must move almost identically with each other — they cannot be near-opposite. The matrix is lying, and the eigenvalues prove it.

The eigenvalues of $C$ work out to approximately $\lambda_1 \approx 1.9$, $\lambda_2 \approx 1.9$, and $\lambda_3 \approx -0.8$. That last negative eigenvalue is the impossible direction. If you tried to Cholesky-factor $C$ to drive a simulation, the algorithm would hit a negative number under a square root and abort.

The clip: set the negative eigenvalue $\lambda_3 = -0.8$ up to $0$, keep $\lambda_1$ and $\lambda_2$, and reassemble the matrix from the same eigenvectors with the repaired eigenvalue list $(1.9, 1.9, 0)$. After re-normalizing the diagonal back to ones, you get a *valid* correlation matrix that is the closest legitimate matrix to the broken one — the impossible $-0.9$ between B and C gets pulled toward something the other two ties can actually support (a markedly less negative number), and the matrix is now positive-semi-definite and safe to factor.

To make the stakes dollar-concrete: suppose this broken matrix slipped into the risk engine for a \$1,000,000 book and you computed portfolio variance with the contradictory $C$. For some weight vectors $w^\top \Sigma w$ would come out *negative*, and your risk system would happily report a portfolio with negative variance — an impossible "risk-free with a discount" position that, if a naive optimizer chased it, would load up on a phantom arbitrage and blow up the moment the real correlations reasserted themselves. The intuition: clipping the negative eigenvalue is not cosmetic — it is the difference between a risk model that describes a possible world and one that invites the optimizer to exploit a hole that does not exist.

## Worked example: Monte Carlo 99% VaR

Now we assemble everything into the single most common production use of Cholesky: estimating **value at risk** by simulation. Value at risk (VaR) at the 99% level over a one-day horizon is the loss that you expect to *exceed* on only 1 day out of 100 — the threshold such that 99% of days are better than it. It is a single dollar number that summarizes the bad tail of your P&L distribution. The figure below stacks the five steps from covariance to that number.

![A stack of five steps factor Sigma into L, draw many z vectors, correlate x equals L z, compute profit and loss per path, then take the first percentile loss.](/imgs/blogs/cholesky-positive-definite-math-for-quants-5.png)

The steps, top to bottom, are: factor $\Sigma$ into $L$ once; draw many vectors of independent normals $z$; correlate each one into a return vector $x = \mu + Lz$; turn each return vector into a dollar P&L for your specific holdings; and finally read the 1st-percentile (the 1%-worst) loss off the resulting distribution. That bottom number is your 99% VaR.

#### Worked example: Monte Carlo 99% VaR of a \$1,000,000 book

Take a clean two-asset book: \$500,000 in asset A and \$500,000 in asset B. Asset A has daily volatility $\sigma_A = 2.0\%$, asset B has $\sigma_B = 1.5\%$, their correlation is $\rho = 0.50$, and we assume zero mean daily return (over one day the mean is negligible next to the volatility). The covariance matrix is

$$\Sigma = \begin{pmatrix} 0.0004 & 0.00015 \\ 0.00015 & 0.000225 \end{pmatrix},$$

with off-diagonal $\rho \sigma_A \sigma_B = 0.50 \times 0.02 \times 0.015 = 0.00015$. Cholesky-factor it: $\ell_{11} = \sqrt{0.0004} = 0.02$; $\ell_{21} = 0.00015 / 0.02 = 0.0075$; $\ell_{22} = \sqrt{0.000225 - 0.0075^2} = \sqrt{0.000225 - 0.00005625} = \sqrt{0.00016875} \approx 0.012990$. So

$$L = \begin{pmatrix} 0.02 & 0 \\ 0.0075 & 0.012990 \end{pmatrix}.$$

Now the simulation. For each of, say, 100,000 scenarios, draw $z = (z_1, z_2)$ independent standard normals, compute the correlated returns $x_A = 0.02 \, z_1$ and $x_B = 0.0075 \, z_1 + 0.012990 \, z_2$, and turn them into a dollar P&L for the book: $\text{P\&L} = \$500{,}000 \cdot x_A + \$500{,}000 \cdot x_B$. Sort the 100,000 P&L numbers from worst to best, and the 1st-percentile value — the 1,000th-worst — is the 99% VaR.

We can sanity-check what the simulation will land on, because for jointly normal returns the portfolio P&L is itself normal and we can compute its standard deviation in closed form. The portfolio dollar variance is $w^\top \Sigma w$ with $w_A = w_B = \$500{,}000$:

$$w^\top \Sigma w = (500{,}000)^2(0.0004) + (500{,}000)^2(0.000225) + 2(500{,}000)^2(0.00015).$$

Each $(500{,}000)^2 = 250{,}000{,}000{,}000$. The three terms are $250{,}000{,}000{,}000 \times 0.0004 = 100{,}000{,}000$; $250{,}000{,}000{,}000 \times 0.000225 = 56{,}250{,}000$; and $2 \times 250{,}000{,}000{,}000 \times 0.00015 = 75{,}000{,}000$. Summing: $100{,}000{,}000 + 56{,}250{,}000 + 75{,}000{,}000 = 231{,}250{,}000$ dollars-squared. The portfolio daily dollar volatility is $\sqrt{231{,}250{,}000} \approx \$12{,}409$.

For a normal distribution, the 1%-worst outcome sits $2.326$ standard deviations below the mean (that $2.326$ is the 1st-percentile z-score of the standard normal — a number worth memorizing, alongside $1.645$ for 95%; the [distributions cheat sheet](/blog/trading/quantitative-finance/distributions-cheat-sheet-quant-interviews) tabulates these). So the 99% one-day VaR is approximately

$$\text{VaR}_{99\%} \approx 2.326 \times \$12{,}409 \approx \$28{,}864.$$

A 100,000-path Monte Carlo run using the Cholesky factor above lands right on top of this — typically reporting something like **\$28,800 ± \$200**, where the wiggle is just sampling noise that shrinks as you add paths. The simulation and the formula agree, which is exactly the confidence check you want: the closed-form is the answer when returns are normal, and the Monte Carlo is the answer when they are *not* (fat tails, options, path-dependent payoffs), but on this simple case they must coincide, and they do.

The intuition: on a \$1,000,000 book of two moderately correlated assets, you would expect to lose more than roughly \$28,800 on about one day in a hundred — and every piece of that number flowed through the Cholesky factor that glued the two assets' randomness together.

### Why simulate VaR instead of using the formula

If the closed-form gives the same answer, why bother with the simulation? Because the closed-form *only* gives the same answer when the P&L is normal — which requires returns to be normal *and* the portfolio to be a linear function of those returns. The moment your book holds options, the P&L is a curved (nonlinear) function of the underlying returns, and the normal formula breaks. The moment returns have fat tails (and they always do), the $2.326$ multiplier understates the true tail. Monte Carlo with a Cholesky factor handles both: you can draw the underlying returns from a fat-tailed joint distribution, run them through your option-pricing function, and read the empirical 1st percentile off the simulated P&L — no formula required. The Cholesky machinery is the part that stays the same; only the draws and the P&L function change.

## Beyond simulation: solving and the log-determinant

Generating correlated draws is the famous use of Cholesky, but the same factor pays for itself two more ways that quants lean on constantly. The tree below lays out the family of uses around the one factor $L$.

![A tree with Cholesky factor L at the root branching to correlate Monte Carlo draws, solve Sigma x equals b, log-determinant likelihood, and test positive definiteness.](/imgs/blogs/cholesky-positive-definite-math-for-quants-7.png)

The root of the tree is the single object $L$ you compute once; the branches are everything you get for that one cost. We have spent the post on the "correlate draws" branch and the "test positive-definiteness" branch (the factorization either succeeds or it tells you the matrix is broken). The other two branches are worth a paragraph each.

### Solving linear systems with Σ

Many quant problems reduce to solving $\Sigma x = b$ for the unknown vector $x$ — for example, the optimal mean-variance portfolio weights are proportional to $\Sigma^{-1} \mu$, which is exactly the solution of $\Sigma x = \mu$. You should almost never compute the inverse $\Sigma^{-1}$ explicitly; it is slower and numerically worse. Instead, factor $\Sigma = LL^\top$ once, then solve in two cheap triangular sweeps: first solve $Ly = b$ for $y$ by **forward substitution** (top to bottom, since $L$ is lower-triangular each row has only one new unknown), then solve $L^\top x = y$ for $x$ by **back substitution** (bottom to top). Each sweep is fast — about $n^2$ operations — and triangular solves are numerically stable. Once you have $L$, solving for any new right-hand side $b$ is nearly free, which is why risk and optimization libraries cache the factor and reuse it.

### The log-determinant and the Gaussian likelihood

When you *calibrate* a model — fitting a covariance matrix to data by maximum likelihood, or evaluating how probable some observed returns are under a multivariate normal — the formula needs the **determinant** of $\Sigma$, specifically its logarithm, $\log \det \Sigma$. Computing a determinant directly is unstable and can overflow for large matrices. But with Cholesky it is trivial: because $\Sigma = LL^\top$ and $L$ is triangular, $\det \Sigma = \left(\prod_i \ell_{ii}\right)^2$, the square of the product of $L$'s diagonal entries, so

$$\log \det \Sigma = 2 \sum_i \log \ell_{ii}.$$

A sum of logarithms of the diagonal — cheap, stable, and impossible to overflow. The multivariate-normal log-likelihood that powers Gaussian model fitting (think calibrating a factor risk model, or scoring a regime under a Gaussian assumption) needs exactly two ingredients beyond the data: a solve against $\Sigma$ (which Cholesky gives you) and this log-determinant (which Cholesky also gives you). So the *same* factor you computed to simulate returns is the factor that lets you evaluate the likelihood — one decomposition, the whole Gaussian toolkit.

#### Worked example: the log-determinant of our 2×2 factor

Reuse the VaR example's factor $L = \begin{pmatrix} 0.02 & 0 \\ 0.0075 & 0.012990 \end{pmatrix}$. Its diagonal entries are $\ell_{11} = 0.02$ and $\ell_{22} = 0.012990$. Then $\log \det \Sigma = 2(\log 0.02 + \log 0.012990) = 2(-3.912 - 4.343) = 2(-8.255) = -16.510$. As a check, $\det \Sigma$ should equal $(0.02 \times 0.012990)^2 = (0.0002598)^2 \approx 6.75 \times 10^{-8}$, and indeed $e^{-16.510} \approx 6.75 \times 10^{-8}$. The two agree.

In a likelihood that you maximize over thousands of iterations to calibrate a risk model, this $-16.510$ (and its many-asset analog) is recomputed every step. Doing it as a sum of two logs instead of a raw determinant is the difference between a calibration that runs and one that silently produces garbage. The intuition: Cholesky is not just a simulation trick — it is the numerical backbone that makes the entire Gaussian risk toolkit, from sampling to solving to scoring, both fast and trustworthy. If you ever wonder how much money rides on getting a \$1,000,000 book's risk number right, the answer is that the same five lines of code value it, optimize it, and stress it.

## Common misconceptions

**"Any symmetric matrix of correlations is a valid correlation matrix."** No. Symmetry and a diagonal of ones are necessary but nowhere near sufficient. The matrix must also be positive-semi-definite — every eigenvalue $\ge 0$ — and that constraint ties the off-diagonal entries together. You cannot set three pairwise correlations to whatever you like; once two of them are fixed, the third is boxed into a range. The 3-asset $(0.9, 0.9, -0.9)$ matrix is symmetric with a unit diagonal and is *still* impossible.

**"Cholesky is just one way to take a matrix square root, so any square root would do for simulation."** True that any $M$ with $MM^\top = \Sigma$ correlates draws correctly, but Cholesky is the *cheapest* such factor (about $n^3/3$ operations versus the much costlier eigenvalue route), it is numerically stable for positive-definite inputs, and its triangular structure gives you the bonus solve and log-determinant for free. There is rarely a reason to use a different square root for simulation unless your matrix is only positive-*semi*-definite, where the eigenvalue square root degrades more gracefully.

**"If my correlation matrix has a tiny negative eigenvalue, I can just ignore it."** You cannot run Cholesky on it at all — the factorization will fail outright at the first negative square root, not return an approximate answer. Even a $-0.0001$ eigenvalue halts the algorithm. You must repair the matrix (clip the eigenvalue) before any Cholesky-based code will touch it. A tiny negative eigenvalue is usually a harmless rounding artifact, but the *fix* is still mandatory, not optional.

**"Positive-definite and positive-semi-definite are the same thing in practice."** They differ on a real boundary case. PSD-but-not-PD means some non-zero portfolio has *exactly* zero variance — there is a redundant asset (a perfect linear combination of others), so $\Sigma$ is singular and has no inverse. Mean-variance optimization, which needs $\Sigma^{-1}$, breaks on a merely-PSD matrix even though simulation can limp along. If you want a clean inverse and a stable optimizer, you want strictly positive-definite, which in practice means adding a small ridge to the diagonal or shrinking toward the identity.

**"More assets always make the matrix better-conditioned and safer to factor."** The opposite, usually. The more assets you add relative to your number of observation days, the more nearly-redundant directions appear, the closer the smallest eigenvalue creeps to zero, and the more fragile the Cholesky becomes. A 500-asset matrix estimated from 250 days of returns is guaranteed to have zero eigenvalues (it is rank-deficient) and *cannot* be Cholesky-factored without regularization. High dimensionality is a reason to regularize, not a free pass.

**"The Cholesky factor L is the volatilities and correlations, just rearranged."** Not quite — $L$ mixes them in a way that is not a simple lookup. The first row/column of $L$ does carry the first asset's volatility directly, but every later diagonal entry $\ell_{ii}$ is a *residual* volatility — how much of asset $i$'s risk is left after removing what it shares with the earlier assets. That is why $\ell_{22} = 0.80$ in our unit-correlation example even though both volatilities were $1$: the $0.80$ is the part of asset B's risk that is *independent* of asset A.

## How it shows up in real markets

### 1. Front-office Monte Carlo VaR

Large banks and asset managers compute VaR for books with thousands of risk factors every night. The dominant approach for the linear part of the book is exactly the pipeline in this post: estimate a covariance matrix over the risk factors, Cholesky-factor it (often after shrinkage or a nearest-correlation repair), draw tens of thousands of correlated factor scenarios, revalue the book under each, and read the loss percentile. The 1996 Basel market-risk amendment effectively institutionalized 99% 10-day VaR as a regulatory capital input, and Cholesky-driven simulation became standard plumbing across the industry. When a risk run "fails to start" overnight, a non-PSD correlation matrix from a data outage is one of the first suspects an on-call quant checks.

### 2. The data-outage non-PSD crash

A concrete recurring failure: an exchange has a holiday or a feed drops, so for a block of assets the day's returns are missing. The naive pipeline computes pairwise correlations over whatever data each pair has, producing a matrix that is internally inconsistent and has a small negative eigenvalue. The nightly Cholesky aborts, and the VaR report is late. The durable fix that desks adopt is to *always* pass the estimated correlation matrix through a nearest-correlation-matrix projection (eigenvalue clip plus diagonal renormalization, or full Higham iteration) before factoring — turning a hard crash into a silent, well-defined repair. Higham's 2002 paper on computing the nearest correlation matrix exists precisely because this problem is so common in practice.

### 3. Stress testing with hand-set correlations

Risk committees often want to stress a book under a hypothetical regime: "what if the stock-bond correlation flips from $-0.3$ to $+0.7$ in a crisis?" When an analyst hand-edits one correlation in a large matrix to build that scenario, the edited matrix is frequently no longer PSD, because that one number was constrained by all the others. The stress engine then either crashes or, worse, silently produces nonsense. Mature stress frameworks edit the matrix and *then* project it to the nearest valid correlation matrix, so the stressed scenario is both severe and mathematically possible. This is the eigenvalue-clipping idea doing quiet daily work.

### 4. Pricing multi-asset derivatives

A basket option, a rainbow option, or any payoff on several underlyings must be priced by simulating the *joint* paths of the underlyings, which means correlated Brownian increments. The standard recipe is to Cholesky-factor the correlation matrix of the underlyings' returns and apply it to independent normal increments at each time step — exactly $x = Lz$ inside the time loop. A typical equity-basket option on, say, five names with a correlation matrix from the dealer's marks is priced this way millions of times a day. If the correlation marks are inconsistent (dealers mark correlations somewhat independently), the desk repairs the matrix to PSD before the pricer will run.

### 5. The 2008 correlation breakdown

The deeper lesson from the financial crisis is not about the Cholesky algorithm itself but about the input it consumes. Structured-credit models (notably the Gaussian copula behind CDO pricing) plugged a *single* correlation number into a multivariate-normal dependence structure and simulated correlated default times — Cholesky machinery underneath. The math worked; the *correlations* were the lie. In the calm pre-crisis years, estimated default correlations were low, so the models said senior tranches were nearly safe. When the housing market turned, correlations spiked toward one — everything defaulted together — and the realized joint distribution looked nothing like the simulated one. The episode is a permanent reminder that a clean Cholesky of a *wrong* covariance matrix gives you confident, precise, wrong answers. The factorization is only ever as good as the $\Sigma$ you feed it.

### 6. Scenario generation for capital and liquidity planning

Beyond a single VaR number, treasury and risk teams generate full sets of correlated market scenarios to plan capital and funding. A bank stress-testing its trading book under, say, a thousand correlated shocks to rates, credit spreads, equities, and FX is running exactly the $x = \mu + Lz$ engine at scale, with $\Sigma$ estimated across hundreds of risk factors. The same scenarios feed expected shortfall (the average loss *beyond* the VaR threshold), which the post-2016 Basel "Fundamental Review of the Trading Book" made the headline risk measure in place of VaR. Each scenario is a correlated draw; the Cholesky factor is what keeps a "stocks crash" scenario from absurdly pairing with "credit spreads tighten." When a planning run produces nonsensical scenarios — equities and their own index futures drifting apart — the first thing a quant checks is whether the correlation matrix was repaired to PSD before factoring, because an un-repaired matrix quietly distorts the joint shocks even when it does not outright crash.

### 7. Risk-parity and minimum-variance portfolios

Strategies that solve for portfolio weights — minimum-variance, risk parity, mean-variance — all need to either invert $\Sigma$ or solve a system against it, and they do it via Cholesky, not by forming $\Sigma^{-1}$. When these strategies misbehave, the culprit is often a near-singular $\Sigma$ (a smallest eigenvalue close to zero from too many correlated assets), which makes the solve numerically explosive and the resulting weights wild and unstable. Practitioners regularize — shrink $\Sigma$ toward a simpler target, or add a ridge to the diagonal — precisely to push the matrix safely into the positive-definite interior where the Cholesky solve is well-behaved. The smallest eigenvalue of your covariance matrix is, quietly, one of the most important numbers in a systematic strategy's stability.

## When this matters to you

If you are learning quantitative finance, this post is one of the highest-leverage pieces of linear algebra you can internalize, because it sits underneath almost everything downstream. The day you write your first Monte Carlo risk engine, the line $x = \mu + Lz$ is the heart of it. The day a correlation matrix crashes your code, the eigenvalue clip is the fix. The day you fit a Gaussian model, the Cholesky log-determinant is in your likelihood. Three different problems — simulation, repair, calibration — and one factorization solves all of them.

The honest caveats, stated plainly. First, Cholesky is flawless arithmetic on top of a possibly-wrong input: a precise factor of a bad covariance matrix is precisely wrong, and 2008 is the monument to that fact. Second, correlations are not stable — they drift in calm markets and spike toward one in crises, so any VaR or pricing number built on a static $\Sigma$ understates exactly the tail risk you most want measured. Third, the normal-distribution assumption that makes the closed-form VaR check work is itself an approximation; real returns have fatter tails, so simulation (which can use fat-tailed draws) beats the formula precisely when it matters most. None of this is investment advice; it is a description of how the machinery behaves and where it lies to you.

For the next steps, the natural companions on this blog are the [Monte Carlo and simulation deep dive](/blog/trading/quantitative-finance/monte-carlo-simulation-coding-quant-interviews) (how simulation answers questions no formula can), the [covariance and correlation pitfalls](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews) post (everything that goes wrong before the matrix even reaches Cholesky), and the [distributions cheat sheet](/blog/trading/quantitative-finance/distributions-cheat-sheet-quant-interviews) (the z-scores and tail facts you need to read a VaR number). Beyond this blog, the canonical references are Nick Higham's work on the nearest correlation matrix and any solid numerical-linear-algebra text's chapter on the Cholesky decomposition. Master the four-line pipeline — build $\Sigma$, factor to $L$, draw $z$, compute $\mu + Lz$ — and you hold the key that unlocks correlated simulation, fast solving, and stable likelihoods all at once.
