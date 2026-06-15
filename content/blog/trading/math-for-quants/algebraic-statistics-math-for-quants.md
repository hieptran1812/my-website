---
title: "Algebraic statistics for quants: when a model is a shape"
date: "2026-06-15"
description: "A beginner-friendly tour of algebraic statistics for traders: how a statistical model becomes a geometric shape carved by polynomial rules, why that view explains exponential families, identifiability, and exact small-sample tests on regime tables."
tags: ["algebraic-statistics", "exponential-families", "identifiability", "contingency-tables", "markov-bases", "exact-tests", "fishers-exact-test", "factor-models", "quant-finance", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Algebraic statistics says a statistical model is not a list of recipes but a *geometric shape*: the set of probability distributions that satisfy some polynomial equations. Once you see model space that way, a lot of mysterious behavior in everyday quant models stops being mysterious.
>
> - A model carves out a **variety** — a curved surface in the space of probabilities — defined by polynomial constraints. Independence of two events, for example, is the single equation $p_{11}p_{22} = p_{12}p_{21}$.
> - **Exponential families** (the normal, the Poisson, logistic regression, GLMs) have a clean convex geometry: a low-dimensional **sufficient statistic** is all the data the model ever uses, and the maximum-likelihood fit is just the parameter whose mean matches that statistic.
> - **Identifiability** is a geometry question: can different parameter values produce the exact same distribution? If yes — as with a naive one-factor model — your fitted "risk loadings" are partly fiction, and trusting them can cost real money.
> - **Markov bases** let you run an *exact* conditional test on a tiny contingency table (regime × outcome) where the chi-square approximation quietly lies. On a 24-trade table, the exact p-value can be **0.18** while chi-square reports a "significant" **0.04**.
> - The honest framing: this is a specialist lens. It rarely generates alpha by itself, but it explains *why* standard models behave as they do and occasionally rescues you from a bad small-sample decision.

Here is a question that sounds like a riddle but has a serious answer. You have two coins of information about a strategy: in an "up" market regime your signal won 40 times and lost 10; in a "down" regime it won 20 times and lost 30. Is the signal's edge *real and regime-dependent*, or did the regime just not matter and you are looking at noise? The standard move is to reach for a chi-square test and read off a p-value. But what is that test actually doing, geometrically? What shape is it comparing your data to? And when does the shortcut it relies on quietly mislead you on a small sample like this?

Algebraic statistics answers all three. Its central idea is almost shockingly simple to state: a statistical model is the set of probability distributions that satisfy certain polynomial equations, and that set is a *geometric object* — a curve, a surface, a higher-dimensional variety — living inside the space of all possible distributions. Fitting a model means finding the point on that shape closest to your data. Testing a model means measuring how far your data sits from the shape. This post builds that picture from absolute zero, then uses it to make four concrete quant decisions, each grounded in dollars. We will be honest throughout: this is a niche, specialist lens, not an everyday alpha factory. But it is the kind of lens that, once worn, changes how you read every model you already use.

![Before and after panels contrasting four raw counts on the left with a curved independence surface on the right](/imgs/blogs/algebraic-statistics-math-for-quants-1.png)

The figure above is the mental model for the whole post. On the left, your data is just four numbers — the counts in a 2×2 table — a single free point floating in space. On the right is the *model*: the independence hypothesis is a curved surface, and the question "does the regime matter?" becomes the geometric question "is your data point on that surface, or far from it?" Everything else in this article is detail hung on that one idea. Let us build it piece by piece, defining every term as we go.

## Foundations: the building blocks

Before we can say "a model is a shape," we need to agree on a small vocabulary. None of it requires prior finance or advanced math. We will define each word the first time it appears, build the simplest possible version, and only then reach for anything fancy.

### What is a probability distribution, as a point?

A **probability distribution** over a finite set of outcomes is just a list of numbers — one per outcome — that are all between 0 and 1 and add up to 1. If a single trade can either *win* or *lose*, a distribution is a pair like $(0.6, 0.4)$: 60% chance of a win, 40% of a loss. If you flip two coins worth of structure — say *regime* (up/down) and *outcome* (win/loss) — there are four joint outcomes, so a distribution is four numbers $(p_{11}, p_{12}, p_{21}, p_{22})$ that sum to 1, where $p_{ij}$ is the probability of being in regime $i$ and getting outcome $j$.

Now the crucial reframing. A list of $k$ numbers is a **point** in $k$-dimensional space. A distribution over four outcomes is a point in 4-dimensional space — specifically, because the four numbers must be non-negative and sum to 1, it lives on a flat triangular slab called the **probability simplex**. You do not need to picture four dimensions; the everyday version is the 3-outcome case, where the simplex is a literal triangle, and every distribution is a point somewhere inside that triangle. The corners are the "certain" outcomes; the center is "all equally likely." A statistical model, we will see, is a *subset* of this simplex — a curve or surface drawn inside it.

### What is a statistical model, the algebraic way?

In a first stats course, a **model** is described by a recipe: "draw each $x_i$ independently from a normal distribution with mean $\mu$ and variance $\sigma^2$." Algebraic statistics keeps the same models but describes them differently — by the *constraints* they impose on the probabilities. A model is the set of all distributions (points in the simplex) that obey some equations.

The cleanest example is **independence**. Two events are independent when the probability of both is the product of the individual probabilities. For our regime × outcome table, "regime tells you nothing about outcome" means $p_{ij} = p_{i\cdot}\, p_{\cdot j}$ for every cell, where $p_{i\cdot}$ is the *row total* (the marginal probability of regime $i$) and $p_{\cdot j}$ is the *column total* (the marginal probability of outcome $j$). That is a set of equations. The distributions satisfying them form a *shape* inside the simplex — and the whole point is that this shape is carved out by polynomials in the $p_{ij}$.

### What is a variety, in one sentence?

A **variety** is the solution set of one or more polynomial equations — the geometric shape you get when you ask "which points make all these polynomials equal to zero?" A circle is a variety: it is the set of points where $x^2 + y^2 - 1 = 0$. A line is a variety. A sphere is a variety. The independence model is a variety inside the probability simplex, defined (as we will derive in a moment) by a single quadratic polynomial. The word sounds exotic, but you have been drawing varieties since you graphed $y = x^2$ in school. The new idea is only that *probability distributions live in a space too, and our models are varieties in that space.*

### What is a manifold, and how is it different?

A **manifold** is a smooth shape that looks flat if you zoom in far enough — the surface of the Earth is a 2-dimensional manifold, locally flat (which is why old maps work) even though it curves globally. Many statistical models are manifolds: a smooth surface of distributions parameterized by a few knobs. The distinction we care about: varieties can have *corners and crossings* (singular points), and those singular points are exactly where bad things like non-identifiability and the failure of standard approximations tend to hide. When a model is a clean manifold, ordinary statistics behaves; when the variety has a singularity, the textbook approximations crack — and that crack is where algebraic statistics earns its keep.

### What is a contingency table?

A **contingency table** is the count version of a joint distribution: instead of probabilities, you record how many times each combination actually happened. Our running example is a 2×2 contingency table:

| | Up regime | Down regime | Row total |
|---|---|---|---|
| **Win** | 40 | 20 | 60 |
| **Loss** | 10 | 30 | 40 |
| **Column total** | 50 | 50 | 100 |

The 40 in the top-left means: 40 of your 100 trades happened in the up regime *and* won. The **margins** (the row and column totals) summarize each variable separately. A huge amount of classical statistics is "given a contingency table, are the two variables independent?" — and that question is the gateway to everything algebraic-statistical.

### Why probabilities → polynomials at all?

Here is the hinge of the whole field, and it is worth slowing down. Probabilities multiply. The probability of "A and B" under independence is $P(A)\times P(B)$ — a *product*, which is a polynomial operation. The likelihood of a whole dataset is a product of probabilities — again polynomial. Conditional independence, mixtures, latent variables: every one of these is built from sums and products of probabilities. So the *constraints* a model imposes are naturally **polynomial equations in the cell probabilities**. That is not a clever encoding we chose; it is forced on us by the arithmetic of probability. Once you accept that models are polynomial constraints, the machinery of algebra and geometry — varieties, ideals, dimension, singularities — becomes available to study them. Algebraic statistics is just the systematic exploitation of that fact.

![Pipeline from a statistical model to probabilities to polynomial constraints to a variety to inference](/imgs/blogs/algebraic-statistics-math-for-quants-2.png)

The pipeline above is the recipe we will run over and over. Start with a model. Write the distributions as coordinates (the cell probabilities). Translate the model's assumptions into polynomial constraints on those coordinates. The constraints carve out a variety or manifold — the geometric shape of the model. And then inference — fitting, testing, checking identifiability — becomes geometry: distances to the shape, the dimension of the shape, the singular points of the shape. Keep this pipeline in mind; every section below is one trip through it.

#### Worked example: a model is just a region of the simplex

Let us make "a model is a shape" concrete with the smallest possible case, before any polynomials. Suppose a coin-flip-like signal has three possible daily outcomes: big win, small win, loss. A distribution is three numbers summing to 1, so it is a point in a triangle (the 3-outcome simplex). Now impose one modeling assumption: "big win and small win are equally likely," i.e. $p_{\text{big}} = p_{\text{small}}$. That single equation is a *line* slicing through the triangle. The model is no longer the whole triangle — it is just that line. If you observe 100 trades and see 30 big, 28 small, 42 loss, your data point $(0.30, 0.28, 0.42)$ is *near* the line $p_{\text{big}} = p_{\text{small}}$ but not exactly on it. The distance from your point to the line is the visual version of the test statistic. Say each "big win" is worth \$200 and each "small win" \$50: if the model held exactly, your expected daily payoff would be $0.29 \times \$200 + 0.29 \times \$50 = \$72.50$ rather than the raw $0.30 \times \$200 + 0.28 \times \$50 = \$74.00$ — a \$1.50 difference that the model's constraint is smoothing away. The one-sentence intuition: imposing a model assumption shrinks the cloud of possible distributions down to a lower-dimensional shape, and your data's distance from that shape is exactly how much the assumption strains against reality.

## 1. Independence as a polynomial constraint

We will now derive the most important single equation in the field and use it to test our regime-versus-outcome question with real numbers and a real dollar decision. This is the section that makes "a model is a variety" click.

### The intuition before the algebra

Independence means *knowing the regime does not change the odds of winning*. If the win rate is 60% in up markets and also 60% in down markets, the regime is irrelevant — the variables are independent. The moment those two win rates differ, the regime carries information, and the variables are dependent. So the independence model is the set of all 2×2 tables where the win rate is the same across regimes. Geometrically, that "same-rate" condition is a constraint that bends a flat slab of possible tables into a curved surface. The picture below is what that surface looks like as a constraint on the four cells.

![Matrix showing each cell forced to equal the product of its row and column margins with a cross-ratio of one](/imgs/blogs/algebraic-statistics-math-for-quants-4.png)

The matrix above is the independence model written out cell by cell. Under independence, every cell probability $p_{ij}$ is *forced* to equal the product of its row margin and column margin: $p_{ij} = p_{i\cdot}\,p_{\cdot j}$. The bottom-right cell of the figure states the punchline we are about to derive: the **cross-product ratio** of the four cells must equal exactly 1.

### Deriving the equation

Start from the independence condition $p_{ij} = p_{i\cdot}\,p_{\cdot j}$. Write out the four cells:

$$ p_{11} = p_{1\cdot}p_{\cdot 1}, \quad p_{12} = p_{1\cdot}p_{\cdot 2}, \quad p_{21} = p_{2\cdot}p_{\cdot 1}, \quad p_{22} = p_{2\cdot}p_{\cdot 2}. $$

Here $p_{1\cdot}$ and $p_{2\cdot}$ are the two row totals (probability of up regime, of down regime); $p_{\cdot 1}$ and $p_{\cdot 2}$ are the two column totals (probability of win, of loss). Now form the product of the two diagonal cells and the product of the two off-diagonal cells:

$$ p_{11}\,p_{22} = (p_{1\cdot}p_{\cdot 1})(p_{2\cdot}p_{\cdot 2}) = p_{1\cdot}p_{2\cdot}p_{\cdot 1}p_{\cdot 2}, $$
$$ p_{12}\,p_{21} = (p_{1\cdot}p_{\cdot 2})(p_{2\cdot}p_{\cdot 1}) = p_{1\cdot}p_{2\cdot}p_{\cdot 1}p_{\cdot 2}. $$

The two right-hand sides are *identical*. So independence forces

$$ \boxed{\,p_{11}\,p_{22} - p_{12}\,p_{21} = 0\,}. $$

That is it. The entire independence model for a 2×2 table is the single quadratic polynomial $p_{11}p_{22} - p_{12}p_{21} = 0$ — one equation, defining one surface (a variety) inside the 3-dimensional simplex of 2×2 distributions. The quantity $p_{11}p_{22} - p_{12}p_{21}$ is called the **$2\times2$ determinant** of the probability table, and the surface where it vanishes is sometimes called the **Segre variety** (a name from algebraic geometry for exactly this "product of two simpler spaces" shape). You do not need the name; you need the equation. The off-by-zero version of it — the ratio $\frac{p_{11}p_{22}}{p_{12}p_{21}}$, called the **odds ratio** — equals 1 precisely when the variables are independent, which is why the figure said "cross ratio = 1."

> The independence model is one polynomial. Dependence is everything that polynomial does not equal zero on.

### From the surface to a test

Your *data* is a point. The *model* is the surface where the determinant is zero. The test asks: is your data point close enough to the surface that the gap could be noise, or is it so far that the regime must genuinely matter? The classical chi-square test is one way to measure that distance; the exact test we build in Section 4 is another. Both are measuring the same geometric thing — how far the data sits from the independence variety.

#### Worked example: testing independence on the regime table

Take the table from the hook: up/win 40, up/loss 10, down/win 20, down/loss 30, total 100. Convert to probabilities by dividing by 100: $p_{11}=0.40$, $p_{12}=0.20$, $p_{21}=0.10$, $p_{22}=0.30$. (Note: here columns are regimes, rows are outcomes — I have arranged it so $p_{11}$ is up-win.) Compute the determinant:

$$ p_{11}p_{22} - p_{12}p_{21} = (0.40)(0.30) - (0.20)(0.10) = 0.12 - 0.02 = 0.10. $$

That is *far* from zero — strongly off the independence surface. The odds ratio is $\frac{0.40 \times 0.30}{0.20 \times 0.10} = \frac{0.12}{0.02} = 6$. An odds ratio of 6, not 1, screams "regime matters." Concretely: the win rate in the up regime is $40/50 = 80\%$; in the down regime it is $20/50 = 40\%$. With a \$100 stake per trade and a 1:1 payoff (win +\$100, lose −\$100), the up-regime edge is $0.8 \times \$100 - 0.2 \times \$100 = \$60$ per trade, while the down-regime edge is $0.4 \times \$100 - 0.6 \times \$100 = -\$20$ per trade. The polynomial just told you, geometrically, that a regime filter is worth roughly **\$80 per trade** of swing — turn the signal on in up markets, off in down. The one-sentence intuition: the determinant of the probability table is a single number whose distance from zero *is* the strength of the dependence, and here it is large enough to drive a real position-sizing rule.

### When this view costs you and when it breaks

The clean polynomial picture is for *finite, discrete* outcomes — contingency tables, categorical regimes, win/loss labels. The moment your variables are continuous (actual returns, not win/loss), the model becomes an infinite-dimensional object and the "single polynomial" elegance fades. You can discretize (bin returns into buckets), but binning throws away information and introduces its own arbitrariness. So the cost of this lens is that it is sharpest exactly where your data is *coarsest* — small categorical tables — and dullest where you have rich continuous data. That is not a flaw; it is the field's natural habitat, and recognizing it keeps you from over-applying the tool.

## 2. Exponential families and their geometry

Almost every distribution a quant uses daily — the normal, the Poisson, the binomial, and the logistic regression that powers up/down direction prediction — belongs to one family with a beautiful shared structure: the **exponential family**. Understanding its geometry explains *why* maximum likelihood works the way it does, why a handful of summary numbers ("sufficient statistics") capture everything the model needs from the data, and when a maximum-likelihood fit is guaranteed to exist and be unique.

### The intuition: a few summaries are all the model sees

Imagine you fit a normal distribution to 250 daily returns. Here is something surprising: the fit depends on the data *only* through two numbers — the sum of the returns and the sum of their squares. Two people with completely different 250-day return series, but the same sum and sum-of-squares, get the *identical* fitted normal. The model is blind to everything else. Those two numbers are the **sufficient statistic**: a compression of the data that loses nothing the model can use. The exponential family is exactly the class of models with this "a few summaries are enough" property, and that property is what makes them so tractable.

![Stack of exponential-family layers from base measure through sufficient statistic and natural parameter to the convex normalizer](/imgs/blogs/algebraic-statistics-math-for-quants-3.png)

The stack above is the anatomy of every exponential-family model. From the bottom: a **base measure** $h(x)$ (the "default" weighting before any parameters), a **sufficient statistic** $T(x)$ (the summaries the model reads off the data), a **natural parameter** $\eta$ (the knobs, in their most convenient coordinates), and a **log-normalizer** $A(\eta)$ (the bookkeeping term that makes probabilities sum to 1). The top layer is the payoff: the maximum-likelihood fit is the parameter whose *expected* sufficient statistic equals the *observed* one. Let us unpack each layer.

### The formal definition

A distribution is in the exponential family if its density can be written

$$ f(x;\eta) = h(x)\,\exp\!\big(\eta^\top T(x) - A(\eta)\big). $$

The symbols, one at a time. $x$ is the data value (a return, a count). $T(x)$ is the **sufficient statistic** — a vector of summaries of $x$ (for the normal, $T(x) = (x, x^2)$). $\eta$ is the **natural parameter** — a vector of knobs, written in the coordinates that make the algebra clean (for the normal these are simple functions of $\mu$ and $\sigma^2$). $h(x)$ is the **base measure**, a fixed function not depending on the knobs. And $A(\eta)$ is the **log-partition function** or **log-normalizer**: $A(\eta) = \log \int h(x)\exp(\eta^\top T(x))\,dx$, the term that guarantees the density integrates to 1.

The two facts that matter for quants are these. First, by independence, the likelihood of a whole dataset depends on the data *only through the sum* $\sum_i T(x_i)$ — that sum is the sufficient statistic for the sample, and it is usually a tiny vector regardless of how big $n$ is. This is a genuine engineering win, not just a theorem: you can compress a streaming feed of millions of ticks into a running pair of accumulators (sum and sum-of-squares) and lose nothing the Gaussian fit will ever use, which is exactly why production calibration code keeps rolling sums rather than the full history. Second, $A(\eta)$ is a **convex function** — its graph curves upward like a bowl. Convexity is the geometric gift: it guarantees the log-likelihood is concave (an upside-down bowl), so it has a single peak with no false summits, which means the maximum-likelihood fit, when it exists, is *unique*. The dollars version: a fit that is guaranteed unique is a fit two desks running the same model on the same data will agree on to the penny, which matters when a \$10,000,000 risk number has to reconcile across systems.

### Why MLE becomes "match the mean of the statistic"

Here is the elegant consequence. The gradient (slope) of $A(\eta)$ equals the *expected value* of the sufficient statistic: $\nabla A(\eta) = \mathbb{E}_\eta[T(x)]$. To maximize the log-likelihood you set its derivative to zero, and after the dust settles that condition reads

$$ \mathbb{E}_{\hat\eta}[T(x)] = \frac{1}{n}\sum_{i=1}^n T(x_i). $$

In words: **the maximum-likelihood fit is the parameter whose model-expected sufficient statistic equals the observed average sufficient statistic.** You match means. For the normal, that means the fitted mean equals the sample mean and the fitted variance equals the sample variance — which you already knew, but now you see *why*: it falls straight out of the exponential-family geometry. For a deeper treatment of maximum likelihood itself, see the companion post on [maximum likelihood and the method of moments](/blog/trading/math-for-quants/mle-method-of-moments-math-for-quants); here we are after the structural reason it is so well-behaved.

### Logistic regression for direction prediction is in the family

This is not abstract for a quant. **Logistic regression** — the workhorse you reach for to predict whether tomorrow is an up day from today's features — is an exponential-family model (specifically a generalized linear model, or **GLM**, built on the Bernoulli distribution). Its sufficient statistic is the vector $\sum_i y_i x_i$ (each label times its features), its log-normalizer is convex, and therefore its log-likelihood is concave with a unique maximum. *That convexity is why your logistic-regression solver always converges to the same answer regardless of where it starts* — there are no local optima to get stuck in. The exponential-family geometry is the reason GLMs are the boringly reliable backbone of direction models. When people say "logistic regression just works," the geometric translation is "its likelihood surface is a single clean bowl."

#### Worked example: a hit-rate model via sufficient statistics

You have 100 daily signals; 60 were correct (the market moved the way you predicted). You model each signal as a Bernoulli trial with unknown success probability $p$ (your hit rate). The Bernoulli is in the exponential family with sufficient statistic $T(x) = x$ (the 0/1 outcome), so the sample sufficient statistic is just the *count of successes*: $\sum_i x_i = 60$. The MLE condition "expected statistic = observed statistic" reads $n\hat p = 60$, giving $\hat p = 0.60$. Notice the model never looked at *which* 60 were right or in what order — those 60 successes out of 100 are the entire informational content. Now the dollars: if each correct call earns \$30 and each wrong one loses \$20, your expected edge per signal is $0.60 \times \$30 - 0.40 \times \$20 = \$18 - \$8 = \$10$. Over 250 trading days that is an expected \$2,500. And the standard error of $\hat p$ is $\sqrt{\hat p(1-\hat p)/n} = \sqrt{0.24/100} \approx 0.049$, so a 95% interval for the hit rate is roughly $0.60 \pm 0.10$ — meaning the true per-signal edge could plausibly be anywhere from about \$2 to \$18. The one-sentence intuition: the sufficient statistic compresses 100 trades down to one number (60), and the exponential-family machinery turns that single number into both your best estimate *and* an honest error bar on it.

### When existence of the MLE fails — a geometric warning

Convexity guarantees uniqueness *if a maximum exists* — but sometimes it does not. The classic failure is **separation** in logistic regression: if some feature perfectly predicts the label in your sample (every day above a threshold was an up day), the likelihood keeps increasing as the coefficient runs to infinity, and no finite MLE exists. Geometrically, the observed sufficient statistic has landed on the *boundary* of the convex set of achievable means, and the optimizer marches off to infinity chasing it. The practical symptom: your logistic regression spits out a coefficient of 14 with a standard error of 3,000. That is not a strong signal; it is the geometry telling you the data does not pin the parameter down. The fix (regularization, or more data spanning the boundary) is exactly a fix for the geometry: pull the target mean back inside the achievable set.

## 3. Identifiability: when parameters can be recovered

Here is a question that has cost real desks real money: when you fit a model and read off its parameters, are those parameters *actually determined by the data*, or could a completely different set of parameters have produced the identical fit? This is the **identifiability** question, and it is fundamentally geometric — it is about whether the map from parameters to distributions is one-to-one.

### The intuition: many doors to the same room

Imagine a model as a machine that takes parameter settings as input and outputs a probability distribution. **Identifiable** means: different inputs always give different outputs — every distribution the model can produce comes from exactly one parameter setting, so seeing the output lets you recover the input. **Non-identifiable** means: two or more different parameter settings produce the *exact same distribution* — the machine has multiple doors leading to the same room, and no amount of data can tell you which door the data came through. The map from parameters to distributions is, in the algebraic-statistics view, a *parameterization of the variety*, and identifiability asks whether that parameterization is one-to-one or folds the parameter space on top of itself.

![Before and after panels contrasting an identifiable model with one fit against a non-identifiable model with many fits for the same data](/imgs/blogs/algebraic-statistics-math-for-quants-6.png)

The figure above contrasts the two cases. On the left, an identifiable model: the true loading is 0.8, the data pins it to one fitted value, and the risk number you compute from it is trustworthy. On the right, a non-identifiable model: a loading of 0.8 and a loading of −0.8 produce *the same data*, the optimizer wanders along a ridge of equally-good fits, and the risk number you compute is meaningless because it depends on which arbitrary point the optimizer happened to stop at. The dangerous part is that both fits look equally good — the non-identifiable model converges, reports parameters, and gives no warning that those parameters are fiction.

### A factor model that cannot be recovered — the sign and rotation problem

The canonical quant example is a **factor model**. You posit that asset returns are driven by a few hidden factors: $r = \Lambda f + \varepsilon$, where $r$ is the vector of asset returns, $f$ is the vector of unobserved factors, $\Lambda$ is the matrix of **factor loadings** (how much each asset responds to each factor), and $\varepsilon$ is idiosyncratic noise. You fit this to estimate the loadings $\Lambda$ — the "how exposed is each asset to each factor" numbers your risk system relies on.

The problem: the model only ever sees the *covariance* of returns, which (assuming the factors have identity covariance) is $\Sigma = \Lambda\Lambda^\top + \Psi$, where $\Psi$ is the diagonal idiosyncratic-variance matrix. Now take any rotation matrix $Q$ (a matrix with $QQ^\top = I$) and define a new loading matrix $\tilde\Lambda = \Lambda Q$. Then

$$ \tilde\Lambda \tilde\Lambda^\top = \Lambda Q Q^\top \Lambda^\top = \Lambda\Lambda^\top, $$

so $\tilde\Lambda$ produces the *exact same covariance*, hence the exact same distribution of returns, hence the exact same likelihood. The loadings $\Lambda$ and the rotated loadings $\Lambda Q$ are **observationally identical**. In the simplest one-factor case, $Q$ can be just the number $-1$: flipping every loading's sign leaves the covariance untouched. The data cannot tell $\Lambda$ from $\Lambda Q$. Any specific loading matrix your software reports is one arbitrary representative of an entire orbit of equally-valid answers. This is the **rotation indeterminacy** of factor analysis, and it is a textbook non-identifiability: the parameterization of the model's variety is many-to-one.

### Why this is a geometry/algebra question

Algebraic statistics formalizes this precisely. The set of distributions the factor model can produce is a variety; the loadings are coordinates on a *parameterization* of that variety; non-identifiability is the statement that the parameterization map has positive-dimensional **fibers** — whole families of parameter values mapping to a single point on the variety. The dimension of the fiber (here, the dimension of the rotation group) counts *how badly* non-identifiable the model is. Checking identifiability is, in principle, checking whether this map is generically finite-to-one — a question you can attack with the algebraic tools the field is built on.

#### Worked example: the dollar risk of a non-identifiable loading

Suppose you run a single-factor risk model on two assets, A and B, and your optimizer reports loadings $\lambda_A = 0.8$ and $\lambda_B = 0.6$ on "the factor." You build a hedge: short \$0.8 of factor-proxy for every \$1 of A, short \$0.6 for every \$1 of B, expecting the factor exposure to cancel. But because the one-factor model is sign-non-identifiable, the optimizer could equally have returned $\lambda_A = -0.8$, $\lambda_B = -0.6$ — and if it had, *your hedge would have the wrong sign and double your factor exposure instead of canceling it.* On a \$1,000,000 book with a factor that moves 2% in a stress event, a correctly-signed hedge nets roughly \$0 of factor P&L, while a sign-flipped hedge takes a $2\% \times 2 \times \$1{,}000{,}000 \approx \$40{,}000$ hit you thought you had neutralized. The numbers $0.8$ and $0.6$ *looked* like risk facts; the geometry says only their *relative* sign and magnitude were ever identifiable, and trusting the absolute sign cost \$40,000. The one-sentence intuition: a non-identifiable parameter is a number your model prints with full confidence but the data never actually determined, and acting on it as if it were real is how a "hedged" book quietly becomes a leveraged bet.

### How practitioners cope

The standard fixes are *identifying restrictions*: constraints that pick one representative from each orbit, restoring one-to-one. For factor models, common choices are forcing the loading matrix to be lower-triangular with positive diagonal, or requiring $\Lambda^\top\Psi^{-1}\Lambda$ to be diagonal. These do not add information — they make an arbitrary-but-consistent choice so that the *reported* numbers are reproducible. The deeper lesson: before trusting any fitted parameter, ask whether the model is identifiable in the first place, because a confident number from a non-identifiable model is worse than no number — it invites you to act on noise. For the estimation-quality angle (bias, variance, and how much to trust an estimator at all), the sibling post on [estimators, MLE, bias, and variance](/blog/trading/quantitative-finance/estimators-mle-bias-variance-quant-interviews) is the natural next read.

## 4. Markov bases and exact tests

Now we cash in the whole framework on a problem where it genuinely beats the standard tool: testing independence on a *small* contingency table, where the chi-square approximation quietly misleads and an *exact* test gives you the honest answer. This is the part of algebraic statistics with the most direct, defensible quant payoff.

### Why chi-square can lie on small samples

The **chi-square test** compares your table to the independence model using the statistic $\chi^2 = \sum \frac{(\text{observed} - \text{expected})^2}{\text{expected}}$ and reads a p-value off a chi-square distribution. But that distribution is an *approximation* — it is the limiting shape the statistic takes as the sample size grows large. On small samples, especially when some expected cell counts fall below about 5, the approximation is poor: it tends to *overstate* significance, handing you p-values that look impressively small when the true evidence is weak. A quant who acts on a "significant" chi-square result from 24 trades is, more often than they realize, acting on an artifact of the approximation.

### The exact alternative: condition on the margins

The exact approach asks a sharper question. *Given the row and column totals I observed, and assuming independence, what is the probability of seeing a table at least as extreme as mine?* By fixing the margins, you turn an intractable problem into a finite combinatorial one: there are only so many tables with those exact margins, you can compute the probability of each one under independence, and the p-value is the total probability of the tables as extreme as yours. For a 2×2 table this is exactly **Fisher's exact test**, and the probability of any particular table follows the **hypergeometric distribution**.

![Stack of the five-step exact-test workflow from observing the table to fixing margins, walking tables, counting extremes, and reading the exact p-value](/imgs/blogs/algebraic-statistics-math-for-quants-7.png)

The stack above is the exact-test workflow. Observe your table. Fix the row and column totals. Walk through every table that keeps those totals. Count how many are at least as extreme as yours (under independence). That fraction — properly probability-weighted — is the exact p-value. Steps 2 and 3 are where the algebra enters: the set of tables sharing fixed margins is precisely the set of integer points satisfying a system of linear equations, and a **Markov basis** is the toolkit for walking that set.

### What a Markov basis actually is

For a 2×2 table the move is obvious: to keep all margins fixed, you can only add 1 to one diagonal pair while subtracting 1 from the other — the table has a single degree of freedom. But for larger tables (3×3, or three-way regime × signal × outcome tables), it is not obvious which "moves" let you walk from any valid table to any other while keeping the margins fixed. A **Markov basis** is a finite set of moves — integer adjustments to the cells that leave all margins unchanged — guaranteed to connect every table with the given margins to every other. The celebrated **Diaconis–Sturmfels theorem** (1998), the founding result of the field, says such a finite basis always exists and corresponds to the generators of a certain polynomial **ideal** (the algebraic object attached to the model). With a Markov basis in hand, you can run a random walk (a Monte Carlo sampler) over the space of tables with fixed margins and estimate the exact p-value even when the table is too big to enumerate fully. For the 2×2 case you do not even need the walk — you can enumerate directly — but the *concept* scales to the tables where direct enumeration is impossible.

### The honest scope, stated plainly

This is the field's most practically useful export for a quant, and it is still niche. You reach for an exact conditional test when: your sample is small (a few dozen observations), the variables are categorical (regime labels, win/loss, signal-on/signal-off), and a wrong significance call would drive a real decision (deploy the strategy or not, allocate capital or not). That is a specific situation — but it is a *real* one, and in it the exact test is not just academically purer, it changes the decision. For the conceptual machinery of p-values and what they do and do not mean, lean on the companion post on [hypothesis testing and p-values](/blog/trading/quantitative-finance/hypothesis-testing-pvalues-quant-interviews); here we use them, with the algebraic-statistics guarantee that the test is exact rather than approximate.

#### Worked example: exact test on a tiny table where chi-square misleads

You are deciding whether to deploy a regime filter, but you have only run the live signal for 24 trades. The small table:

| | Up regime | Down regime | Row total |
|---|---|---|---|
| **Win** | 8 | 3 | 11 |
| **Loss** | 4 | 9 | 13 |
| **Column total** | 12 | 12 | 24 |

The up-regime win rate is $8/12 = 67\%$; the down-regime win rate is $3/12 = 25\%$. That looks like a strong regime effect. Run the chi-square test. The expected count in each cell under independence is $\frac{\text{row total} \times \text{column total}}{24}$, e.g. the up/win expected count is $\frac{11 \times 12}{24} = 5.5$. Working through all four cells, $\chi^2 \approx 4.2$, which on 1 degree of freedom gives a p-value around **0.04** — "significant" at the 5% level. A naive read says: deploy the filter.

Now the exact (Fisher) test, conditioning on the fixed margins (11, 13, 12, 12). The probability of *this exact table* under independence is the hypergeometric probability

$$ P = \frac{\binom{12}{8}\binom{12}{3}}{\binom{24}{11}} = \frac{495 \times 220}{2{,}496{,}144} \approx 0.0436. $$

To get the p-value you sum the probabilities of this table *and every table at least as extreme* (under the same margins) — the more-lopsided tables with up/win counts of 9, 10, 11 on the high side (and the symmetric tail). Adding those up, the two-sided exact p-value comes to roughly **0.11** (the one-sided tail is about 0.055). The exact test says: *not* significant at 5%. The chi-square approximation, leaning on a large-sample limit you do not have with only 24 trades and expected counts near 5, overstated the evidence by nearly a factor of three.

The dollar decision. Deploying the filter commits \$50,000 of capital and roughly \$5,000 in implementation and monitoring costs. If the regime effect is real, the filter earns an expected \$30,000 a year; if it is noise, you have spent \$5,000 (plus the opportunity cost of the capital) chasing a mirage and will likely scale up and lose more when the "edge" reverts. The chi-square test (p = 0.04) green-lights the spend; the exact test (p = 0.11) says the 24-trade sample simply has not earned that conclusion — *gather more data before risking the \$50,000.* The one-sentence intuition: on a small categorical table the chi-square p-value is an optimistic approximation, the exact test is the truth, and the gap between p = 0.04 and p = 0.11 is exactly the gap between a \$50,000 commitment and "wait."

### What it costs and when it breaks

Exact tests are not free. For a 2×2 table they are trivial, but as tables grow — many regimes, many signal buckets, three-way tables — the number of tables with fixed margins explodes, and you must switch from direct enumeration to a Markov-basis Monte Carlo walk, which needs the basis (sometimes expensive to compute) and enough samples to estimate a small p-value reliably. To make the explosion concrete: a 5×5 table with moderate margins can have millions of feasible tables, far past the point where you can list them by hand, which is precisely the regime where the Markov-basis random walk earns its keep — it samples the space rather than enumerating it, and the Diaconis–Sturmfels guarantee is what assures you the walk can actually reach every table rather than getting trapped in a corner. There is also a known critique: conditioning on the margins can make Fisher's exact test mildly *conservative* (p-values a touch too large) because the discreteness of the table leaves gaps, and some practitioners prefer a mid-p adjustment that splits the probability of the observed table to recover a little power. The practical stance: on a small categorical table where a wrong call costs real money — the \$50,000 deployment decision above — the exact test's honesty is worth its modest conservatism, and it is the right default over a chi-square approximation you cannot trust at that sample size.

## 5. The whole toolkit, and how the pieces connect

We have now toured four ideas — independence as a polynomial, exponential-family geometry, identifiability, and exact tests via Markov bases. They are not four disconnected tricks; they are four views of one picture, the picture from our opening pipeline. The figure below organizes the field so you can see where each piece sits.

![Tree of the algebraic statistics toolkit branching into model geometry and exact combinatorics with their sub-tools](/imgs/blogs/algebraic-statistics-math-for-quants-5.png)

The tree above splits algebraic statistics into two trunks. The left trunk, **model geometry**, holds the exponential families (the well-behaved manifolds where ordinary statistics works) and identifiability (the question of whether the model's parameterization is one-to-one). The right trunk, **exact combinatorics**, holds Markov bases and the exact conditional tests they enable. The single root is the founding idea that ties them together: *a statistical model is the variety carved out by polynomial constraints on probabilities.* Geometry of the variety governs estimation and identifiability; the combinatorics of integer points on the variety governs exact testing.

### The connective tissue, in one paragraph

Why are these one field rather than two? Because the *same* polynomial equations that define the model's variety (left trunk) also define the system of linear constraints whose integer solutions are the tables you walk in an exact test (right trunk). The independence equation $p_{11}p_{22} = p_{12}p_{21}$ is, simultaneously: the surface your data's distance-from is measured against (Section 1), an exponential-family model whose sufficient statistics are the margins (Section 2), a parameterization whose identifiability you can check (Section 3), and the algebraic object whose Markov moves let you enumerate fixed-margin tables (Section 4). One equation, four uses. That unity is the reason the field exists as a discipline rather than a grab-bag, and it is why learning the geometric view pays off across every model you fit, not just the one in front of you.

### A quick comparison of the four lenses

| Lens | The question it answers | The quant payoff | When it shines |
|---|---|---|---|
| Independence variety | Does variable A inform variable B? | Regime filters, conditional edges | Categorical tables |
| Exponential-family geometry | Why does MLE behave so cleanly? | Trustworthy GLM/logistic fits | Direction models, calibration |
| Identifiability | Are the fitted parameters real? | Avoid acting on fiction loadings | Factor and latent models |
| Markov bases / exact tests | Is the small-sample result honest? | Don't deploy on a mirage | Small categorical samples |

Notice that only the last lens routinely *changes a number you act on*. The first three mostly change your *understanding* of numbers you already compute — which is valuable, but it is understanding, not alpha. That asymmetry is the honest center of this post, and we return to it at the end.

## Common misconceptions

**"Algebraic statistics is a new way to find alpha."** It is not, and anyone selling it as one is overselling. It is a *foundational* lens — it clarifies why your existing models behave as they do and occasionally rescues a small-sample decision. The closest it comes to alpha is keeping you from a false-positive deployment (Section 4's exact test), which is alpha-preserving, not alpha-generating. Treat it as a way to understand and sanity-check, not to discover.

**"A variety is some abstract object with no connection to data."** The opposite is true: the variety *is* the model, written in the only language probability arithmetic allows. The independence variety is literally the set of regime tables where the regime carries no information. Your data is a point; the model is the shape; the test is the distance. Nothing could be more concrete — it is geometry you could draw if the dimensions were low enough.

**"Sufficient statistics are a curiosity; I should always keep all my data."** For a correctly-specified exponential-family model, the sufficient statistic loses *nothing the model can use* — keeping the raw data buys you no additional fitting power. The catch is "correctly specified": sufficiency is relative to the model, so if your model is wrong, the discarded detail might have revealed the misspecification. Keep raw data for *diagnostics*, but understand that the *fit* only ever sees the sufficient statistic.

**"If my optimizer converged, my parameters are meaningful."** Convergence and identifiability are different things. A non-identifiable model converges happily to one arbitrary point on a ridge of equally-good fits (Section 3). The reported numbers look authoritative and are partly fiction. Always ask whether the model is identifiable *before* you trust a fitted parameter — convergence is necessary, not sufficient.

**"Fisher's exact test and chi-square always agree, so the distinction is pedantic."** They agree on large samples — that is exactly why chi-square exists, as the large-sample approximation. They *disagree* on small samples with low expected counts, precisely the regime where a quant runs a new strategy on a few dozen trades. In that regime the disagreement is the difference between deploying capital and waiting, as the \$50,000 worked example showed.

**"This only applies to discrete data, so it's useless for returns."** The sharpest tools (varieties, Markov bases) do live in the discrete world, but the *geometric mindset* — model as shape, identifiability as one-to-one-ness, MLE as matching sufficient statistics — applies everywhere, including the continuous Gaussian and GLM models you use on returns daily. The discrete tools are a specialty; the geometric lens is general.

## How it shows up in real markets

### 1. Regime-conditional strategy validation

The most direct appearance is the one we have built around: deciding whether a strategy's edge depends on a market regime (high-vol vs low-vol, trending vs ranging, risk-on vs risk-off) from a *small* live track record. A desk that has run a new signal for six weeks might have 25–40 trades split across regimes. Reaching for chi-square at that sample size systematically overstates significance, and the algebraic-statistics-grounded exact test is the disciplined alternative. The mechanism from Section 4 in action: condition on the margins (how many trades fell in each regime, how many won overall), enumerate the fixed-margin tables, and read the honest p-value. The lesson desks relearn the hard way is that a "significant" regime effect from 30 trades is, more often than not, the chi-square approximation flattering noise — and the exact test is the cheap insurance against deploying on it.

### 2. Factor-model rotation in risk systems

Every multi-factor risk model in production confronts the rotation indeterminacy of Section 3. Commercial risk models (the kind that report each asset's exposure to "value," "momentum," "size") impose identifying restrictions so their loadings are reproducible — but the user who forgets *why* those restrictions exist can be badly misled when comparing two vendors whose conventions differ, or when a home-grown PCA-based factor model flips a sign between two estimation windows and a "hedge" silently inverts. The 2010s saw several public post-mortems of risk models behaving erratically across rebalances; the underlying mechanism is often that a near-non-identifiable structure let the optimizer jump between observationally-equivalent solutions. The lesson: pin your factors with explicit identifying restrictions, and never compare raw loadings across models with different conventions.

### 3. Logistic-regression separation in direction models

A quant builds a logistic regression to predict up/down days and includes a feature that, in the training window, happens to perfectly separate the classes (every day with that feature above a cutoff was an up day). The model reports a coefficient of 12 with an enormous standard error, and naive readers see a "huge predictor." Section 2's geometry explains it: the observed sufficient statistic landed on the boundary of the achievable set, no finite MLE exists, and the optimizer is running the coefficient to infinity. This shows up constantly in practice with rare events (predicting crashes, defaults, limit-up moves) where one feature accidentally nails the handful of positive cases. The lesson: a giant coefficient with a giant standard error is a geometry warning, not a signal — regularize, and treat the "perfect" feature with suspicion.

### 4. Small-sample event studies

Event studies — "did the stock react differently to earnings beats versus misses?" — are contingency tables in disguise (beat/miss × up/down reaction), often with small counts because a single name has only a handful of earnings dates. Applying the chi-square test to a 2×2 of, say, 16 earnings events overstates significance exactly as Section 4 warns. Practitioners who have been burned switch to Fisher's exact test for these small event windows. The mechanism is identical to the regime example; only the labels change. The lesson generalizes: any time you are testing independence on a table built from a small number of discrete events, the exact test is the right default, and the chi-square p-value is a number to distrust.

### 5. Mixture models and the number of regimes

Deciding *how many* hidden regimes a Markov-switching model has is a model-selection problem riddled with non-identifiability and singularities — the candidate models are nested in a way that makes the standard likelihood-ratio test's chi-square approximation invalid (the boundary problem of Section 2's MLE-existence discussion, in a more severe form). Algebraic statistics has produced some of the cleanest results on exactly when these singularities arise. The practical fallout: a quant who uses a naive likelihood-ratio test to choose between a 2-regime and a 3-regime model is using a test whose null distribution is *not* the chi-square the software assumes, and will over-fit the regime count. The lesson: model selection across nested latent-variable models needs care that off-the-shelf tests do not provide, and the geometry tells you why.

### 6. Contingency tables in trade surveillance and compliance

A less glamorous but real appearance: surveillance teams test whether a trader's wins cluster suspiciously around informational events (news × win/loss tables), often on small samples per trader. A false "significant" result can trigger an expensive investigation; a false "not significant" can miss real misconduct. Because the stakes of each decision are high and the per-trader sample is small, the exact-test discipline from Section 4 is the appropriate standard, and the chi-square shortcut is exactly the wrong tool. The lesson: when each individual decision is costly and the sample is small, pay the modest cost of the exact test rather than trust an approximation.

## When this matters to you

If you fit any model to market data — and every quant does — the geometric lens of algebraic statistics is worth owning even if you never compute a Markov basis by hand. It is what lets you answer, for any model you use: *what shape is this model, are its parameters actually recoverable from data, and can I trust the standard test at the sample size I have?* Those three questions catch a remarkable fraction of real mistakes: trusting a non-identifiable loading, over-reading a small-sample chi-square, mistaking logistic separation for signal.

Be honest with yourself about the scope, though. This is a specialist's lens, not a daily alpha tool. You will use the *mindset* — model as shape, identifiability as one-to-one-ness, MLE as matching sufficient statistics — far more often than the heavy machinery. The one piece of heavy machinery that earns its place in a working quant's toolkit is the exact conditional test for small categorical tables, because it changes decisions: it is the difference between deploying \$50,000 on a 24-trade mirage and waiting for the data to actually earn the conclusion. If you take one operational habit from this post, make it that: on a small categorical table where a wrong call costs real money, run the exact test, not the chi-square. (This is educational, not investment advice — the point is the method, not any particular trade.)

For the next steps in this series, the natural neighbors are the posts that build the inferential machinery this one assumes: [maximum likelihood and the method of moments](/blog/trading/math-for-quants/mle-method-of-moments-math-for-quants) for the estimation engine behind exponential families, [hypothesis testing and p-values](/blog/trading/math-for-quants/hypothesis-testing-pvalues-math-for-quants) for the testing framework the exact test refines, [estimators, MLE, bias, and variance](/blog/trading/quantitative-finance/estimators-mle-bias-variance-quant-interviews) for how much to trust an estimate, and [hypothesis testing and p-values for quant interviews](/blog/trading/quantitative-finance/hypothesis-testing-pvalues-quant-interviews) for the interview-grade version of the testing ideas. Read together, they turn the abstract slogan "a model is a shape" into a working discipline for fitting, testing, and trusting the models your money rides on.

For going deeper into the field itself, the standard entry points are Diaconis and Sturmfels's 1998 paper introducing Markov bases (the founding result), and the textbook *Lectures on Algebraic Statistics* by Drton, Sturmfels, and Sullivant — but you do not need either to use the four ideas above. You need only the picture from the first figure: data is a point, a model is a shape, and inference is the geometry between them.
