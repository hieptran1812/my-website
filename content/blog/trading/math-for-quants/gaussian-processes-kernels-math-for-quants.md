---
title: "Gaussian processes: regression that tells you when it does not know"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "A Gaussian process is a distribution over functions. You never choose a functional form, you only state how much two nearby points should resemble each other, and what comes back is a curve with honest error bars that widen exactly where you have no data."
tags: ["gaussian-processes", "kernel-methods", "bayesian-inference", "yield-curve", "volatility-surface", "bayesian-optimisation", "interpolation", "uncertainty-quantification", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 18
---

> [!important]
> **TL;DR:** A Gaussian process does not fit a curve, it conditions a normal distribution whose index set happens to be continuous. The output is a mean **and** a width, and the width is the part worth paying for.
>
> - The only modelling choice is the **kernel**, a function saying how correlated $f(x)$ and $f(x')$ should be. Choose it and you have chosen smoothness, curvature and periodicity. There is no second choice.
> - The posterior is closed form. Two swap quotes at 2y and 10y, an RBF kernel with a 4-year length-scale, and the 5y comes out at **4.04% with a standard deviation of 22.02 bp**.
> - **The predictive variance never touches the observed $y$ values.** Invert the data completely and the error bars do not move a basis point.
> - On a \$250m five-year swap position at roughly \$111k of DV01, that 22 bp of model width is **\$2.45m of mark uncertainty**, and one real 5y quote removes \$2.23m of it.
> - Cost is $O(n^3)$ in observations, so GPs live on hundreds or thousands of points: curves, surfaces, backtest hyperparameters. They are not a return-prediction workhorse, and pretending otherwise is how people get hurt.

## The number you are about to quote, and the number you actually know

A client asks you to price a five-year interest rate swap. Your screen has a two-year at 3.80% and a ten-year at 4.50% and nothing in between, because those are the tenors anyone trades in size today.

Almost everyone reaches for a spline, reads off 4.06%, and quotes it. The number goes into a pricing sheet, then a risk system, then a P&L attribution, and by the third system nobody remembers it was never observed. It looks exactly like the 3.80% next to it.

The honest answer differs in kind, not in precision. It is **4.04%, plus or minus 22 basis points**, where a basis point is one hundredth of a percent. That second number is not decoration: it says the model does not know the five-year rate to better than about a fifth of a percent, which on a large book is millions of dollars of mark. A Gaussian process is the machinery that produces it, built here from the multivariate normal upward.

![Posterior mean and two-standard-deviation band for a swap curve fitted to quotes at two years and ten years, showing the band pinching to plus or minus 4 basis points at each quoted tenor, flaring to plus or minus 44 basis points at five years, and opening to plus or minus 71 basis points at fifteen years as the mean falls back toward the four percent prior](/imgs/blogs/gaussian-processes-kernels-math-for-quants-1.webp)

That figure is the whole idea in one frame. The band is tight at the two tenors someone actually quoted, opens between them, and past the last quote opens wider still while the mean drifts back toward what you believed before seeing data. None of that shape was designed. It fell out of one conditioning step.

## Foundations: from the multivariate normal to a distribution over functions

Start with something you already know. Two random variables $u$ and $v$ are jointly normal with means $m_u, m_v$, variances $\sigma_u^2, \sigma_v^2$ and covariance $c$. Observe $u$, and $v$ is still normal, with

$$
\mathbb{E}[v \mid u] = m_v + \frac{c}{\sigma_u^2}(u - m_u), \qquad \mathrm{Var}[v \mid u] = \sigma_v^2 - \frac{c^2}{\sigma_u^2}.
$$

Two things there do all the work for the rest of this post. The conditional mean moves away from the prior mean in proportion to the covariance, and the conditional variance **shrinks by an amount that has no $u$ in it**. The data moves the centre. It does not move the width.

Now generalise twice. First, take $n+1$ variables instead of two, stack the observed ones into a vector $\mathbf{y}$ and keep one unknown $f_{\ast}$ aside: the same identity holds with matrices in place of scalars, which is bookkeeping rather than new mathematics. Second, and this is the step people find strange even though it is the smaller one, index those variables not by ${1, 2, \ldots, n}$ but by a continuous $x$: a tenor, a strike, a lookback window in days. A **Gaussian process** is a collection of random variables, one per index, such that *any finite subset of them is jointly multivariate normal*. That is the entire definition:

$$
f(\cdot) \sim \mathcal{GP}\big(m(\cdot),\, k(\cdot,\cdot)\big)
$$

where $m(x)$ is the prior mean at index $x$ and $k(x, x')$ is the covariance between $f(x)$ and $f(x')$. Because every finite subset is a multivariate normal, and because you only ever need finitely many points, that two-variable conditioning formula is the exact and complete inference rule. There is no approximation anywhere in this post until the section on cost.

![Left panel showing a tilted bivariate normal ellipse over axes f of x one and f of x two with a vertical observation line and a narrowed conditional density on the second axis, right panel showing three sampled function paths through one observed point fanning apart with distance, connected by the statement that the same conditioning formula applies once the index set is continuous](/imgs/blogs/gaussian-processes-kernels-math-for-quants-2.webp)

Two terms before we go further. **Par swap rate** is the fixed rate that makes a new swap worth zero at inception, the natural quantity to interpolate along a curve. **DV01** is the dollar change in a position's value per basis point move in rates, which is how a desk turns a rate uncertainty into a money uncertainty.

This is [Bayesian inference](/blog/trading/math-for-quants/bayesian-inference-traders-math-for-quants) with an infinite-dimensional parameter, and it is [the Kalman filter](/blog/trading/math-for-quants/kalman-filter-state-space-math-for-quants) run all at once over an index set rather than sequentially in time.

## The kernel is the model

In ordinary regression you pick a functional form, then estimate coefficients. A GP has no functional form to pick. You supply $k(x, x')$ and that one function fixes everything the model believes. The workhorse is the **squared exponential**, usually called RBF:

$$
k(x, x') = \sigma_f^2 \exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right)
$$

Two parameters, each with a plain reading. $\sigma_f$ is the prior standard deviation of the function around its mean: how far from 4.00% a swap rate can plausibly sit before you see data. $\ell$ is the **length-scale**, the distance in $x$ over which the function moves by roughly one $\sigma_f$. Set $\ell = 4$ years and you assert that a two-year gap in tenor leaves rates strongly linked while a twelve-year gap leaves them nearly independent.

The RBF's hidden assumption is the severity of its smoothness. Its sample paths are infinitely differentiable: analytic, no kinks, no jumps, ever. That is a strong belief and often a wrong one. The **Matern** family relaxes it with a roughness parameter. The 3/2 case,

$$
k(r) = \sigma_f^2\left(1 + \frac{\sqrt{3}\,r}{\ell}\right)\exp\!\left(-\frac{\sqrt{3}\,r}{\ell}\right), \qquad r = |x - x'|,
$$

gives paths that are continuous and once differentiable but visibly crumpled. Implied volatility term structures look like that, not like an analytic curve, which is why the Matern is the default on vol surfaces and the RBF is the default on discount curves.

A **linear** kernel $k(x, x') = \sigma_b^2 + \sigma_v^2 x x'$ gives back Bayesian linear regression exactly: sample paths that are straight lines and nothing else. A GP is therefore not a rival to regression, it contains regression as the special case where the covariance assumption is that severe. A **periodic** kernel $k(x,x') = \sigma_f^2 \exp(-2\sin^2(\pi|x-x'|/p)/\ell^2)$ asserts the function repeats with period $p$, which expresses intraday volume seasonality or a monthly settlement effect without a single dummy variable.

Kernels also add and multiply, the algebra staying closed because sums and products of positive semi-definite functions remain positive semi-definite. A sum models "a trend plus a wiggle", a product "a seasonal effect whose amplitude drifts".

![Comparison matrix of four kernels, RBF, Matern three halves, linear and periodic, against what each one assumes, what its sample paths look like, what it is right for, and when it is wrong, with a note that the length-scale is the distance over which the function moves about one prior standard deviation](/imgs/blogs/gaussian-processes-kernels-math-for-quants-3.webp)

The practical consequence is worth stating bluntly. When a GP gives you an answer you dislike, the kernel is the thing to argue with. Nothing else in the model carries an opinion.

## The posterior in closed form

Write $K$ for the $n \times n$ matrix of kernel values between observed inputs, $\mathbf{k}_{\ast}$ for the vector of kernel values between the query point and each observation, and $k_{\ast\ast} = k(x_{\ast}, x_{\ast})$. Let $\sigma_n^2$ be the observation noise, which on a quoted rate is the bid-ask, not measurement error. Conditioning the joint normal gives

$$
\mu_{\ast} = m(x_{\ast}) + \mathbf{k}_{\ast}^\top \left(K + \sigma_n^2 I\right)^{-1} (\mathbf{y} - \mathbf{m})
$$

$$
\sigma_{\ast}^2 = k_{\ast\ast} - \mathbf{k}_{\ast}^\top \left(K + \sigma_n^2 I\right)^{-1} \mathbf{k}_{\ast}
$$

Look hard at the second line. The vector $\mathbf{y}$ appears in the mean and **does not appear in the variance at all**. Your uncertainty at an unobserved tenor is determined entirely by where you took observations, how noisy they are, and what the kernel says about distance. It is fixed before anyone tells you the rates.

#### Worked example 1: a two-point posterior computed by hand on a \$250m swap book

Illustrative arithmetic on assumed inputs. RBF kernel, $\sigma_f = 40$ bp, $\ell = 4$ years, quote noise $\sigma_n = 2$ bp, prior mean 4.00%. Observations: 2y at 3.80%, 10y at 4.50%. Query: 5y. Work in basis points of deviation from the prior, so $\mathbf{y} - \mathbf{m} = (-20, +50)$.

1. **Build $K$.** The tenors are 8 years apart, so ${8^2/(2 \times 4^2)} = 2$ and $k = 1600 e^{-2} = 216.54$. Adding $\sigma_n^2 = 4$ to the diagonal,

   $$
   K + \sigma_n^2 I = \begin{bmatrix} 1604 & 216.54 \\ 216.54 & 1604 \end{bmatrix}, \qquad \det = 2{,}525{,}928.
   $$

2. **Build $\mathbf{k}_{\ast}$.** The 5y sits 3 years from the 2y and 5 from the 10y: $1600e^{-9/32} = 1207.74$ and $1600e^{-25/32} = 732.53$.

3. **Solve for the weights.** ${(K + \sigma_n^2 I)^{-1}(\mathbf{y} - \mathbf{m})} = (-0.016985,\, 0.033466)$.

4. **Predictive mean.** $1207.74 \times (-0.016985) + 732.53 \times 0.033466 = -20.51 + 24.51 = 4.00$ bp above the prior, so **4.0400%**.

5. **Predictive variance.** $\mathbf{k}_{\ast}^\top (K + \sigma_n^2 I)^{-1}\mathbf{k}_{\ast} = 1115.29$, so $\sigma_{\ast}^2 = 1600 - 1115.29 = 484.71$ bp² and $\sigma_{\ast} = 22.02$ bp.

A straight line between the two quotes gives 4.0625%. The GP gives 4.0400%, pulled 2.25 bp toward the prior mean because a 4-year length-scale across an 8-year gap means neither quote has much to say about the middle. That pull is the same [shrinkage](/blog/trading/math-for-quants/shrinkage-stein-paradox-math-for-quants) logic you meet in covariance estimation, arriving here through the kernel rather than through a Stein argument.

Now the money. A five-year par swap carries roughly \$445 of DV01 per \$1m of notional, so a \$250m position has about **\$111,250 per basis point**. One standard deviation of 22.02 bp is therefore **\$2,449,000 of mark uncertainty** on a position whose price looked like a fact. Get one real 5y quote with 2 bp of bid-ask and the posterior standard deviation there collapses to 1.99 bp, worth **\$222,000**. That single phone call is worth **\$2.23m of removed uncertainty**, and the GP told you to make it before anyone answered.

### The variance ignores your data, and that is the useful part

People find step 5 unsettling, so it is worth dwelling on. Take the same two tenors and replace the rates with an inverted curve: 2y at 4.60%, 10y at 3.20%. The posterior mean at 5y moves to 4.13%. The posterior standard deviation is 22.02 bp, unchanged to four figures.

![Two panels with identical tenors and kernel, the left fitted to an upward sloping pair of quotes giving a five-year mean of four point zero four percent, the right fitted to an inverted pair giving four point one three percent, both carrying an identical twenty two point zero two basis point standard deviation at five years](/imgs/blogs/gaussian-processes-kernels-math-for-quants-4.webp)

This is not a quirk, it is a design tool. Because the width depends only on the *design* of your observations, you can compute tomorrow's uncertainty today and decide where to spend a market maker's patience. Which tenor should you go and get quoted? The one that shrinks the band most where your book actually has risk. That calculation needs no data at all.

The caveat, and interviewers reach for it: this holds for a **fixed** kernel. In practice you fit $\sigma_f$, $\ell$ and $\sigma_n$ by maximising the marginal likelihood, and that fit does depend on $\mathbf{y}$. So the variance is data-independent conditional on the hyperparameters and not unconditionally. Say that sentence out loud before someone says it to you.

## Why the width is the product

Every model extrapolates. Almost none of them tell you they are doing it.

Fit a cubic spline through six quoted tenors, ask for the 30-year, and you get a number as crisp as the 10-year someone actually traded. Fit a Nelson-Siegel or Svensson curve and you get the same crispness plus the hazard that a three-factor form will bend the front end to accommodate the back. Neither object can say "I am guessing here".

A GP does. Past the last observation the posterior mean decays toward the prior at the rate the kernel dictates and the variance climbs back toward $\sigma_f^2$. In that first figure the band at 15 years is plus or minus 71 bp at two standard deviations against plus or minus 4 bp at the 10-year quote: the model telling you, in your own units, that it has left the region where it knows anything.

That is worth more than a better fit. A wrong number with a wide band gets sized correctly. A wrong number with no band gets sized as though it were true, and the loss arrives as a surprise rather than a budgeted cost.

## The cost: why this lives on thousands of points, not millions

Both formulas need $(K + \sigma_n^2 I)^{-1}$. In practice you never invert it, you take a [Cholesky factorisation](/blog/trading/math-for-quants/cholesky-positive-definite-math-for-quants) and solve two triangular systems, but the leading term is the same: $O(n^3)$ time and $O(n^2)$ memory. At $n = 1{,}000$ that is instant. At $n = 10{,}000$ it is a few seconds and 800 MB. At $n = 10^6$ it is not a computation, it is a research program.

The standard responses trade exactness for scale. **Inducing-point** methods summarise $n$ observations with $m \ll n$ pseudo-inputs at a cost of $O(nm^2)$, the variational treatment giving a bound you can optimise rather than a heuristic. **Structured kernels** exploit grid or Kronecker structure for near-linear cost when inputs are laid out regularly, which a strike-by-expiry vol grid nearly is. **Local GPs** fit a separate small process per neighbourhood. All three cost you something, usually some of the calibrated variance that was the reason you came. For curve and surface work you rarely need them, because the number of genuinely liquid quotes on any curve is in the dozens.

## Where this earns its place in a quant workflow

Be clear about what a GP is not. It is not a return-prediction engine. Cross-sectional equity alpha lives in tens of thousands of names and hundreds of features at a signal-to-noise ratio near zero, the exact regime where an $O(n^3)$ method with a smoothness prior beats nothing that a penalised linear model does not already do. Two places do suit it, and in both the *width* is the deliverable.

#### Worked example 2: quoting a \$50m off-cycle option from a GP vol surface

Illustrative arithmetic on assumed inputs. Listed expiries give 95%-strike implied vols of 18.10% at three months and 17.40% at six months. A client wants a four-month, 95%-strike option on \$50m of notional. Matern 3/2 kernel, $\sigma_f = 1.5$ vol points, $\ell = 6$ months, quote noise $\sigma_n = 0.15$ vol points, prior mean 18.0%.

Running the same two steps as before, the posterior at four months is **17.87% with a standard deviation of 0.265 vol points**. The mean is almost exactly what linear interpolation gives, 17.867%, which is the point: the mean was never the contribution.

Convert the width to money. With a notional $S$ of \$50m, $T = 1/3$ and $\sigma = 17.87\%$, the 95% strike puts $d_1 = 0.549$, so $\varphi(d_1) = 0.343$ and vega is $S\sqrt{T}\varphi(d_1)/100$, or **\$99,060 per vol point**. One standard deviation of interpolation uncertainty is therefore **\$26,280**.

Now compare that to the desk's normal quoting spread of 0.25 vol points, which is **\$24,770**. The model's own uncertainty about the four-month vol is *larger than the entire spread you were planning to charge*. Quote the off-cycle date at your listed-tenor spread and you are not earning a spread, you are taking an unpriced position in your own ignorance. The GP's answer is either widen to around 0.5 vol points, or go and get a four-month quote, which would cut the standard deviation to 0.13 and the exposure to about \$12,900.

#### Worked example 3: Bayesian optimisation of a backtest parameter on a \$200m book

Illustrative arithmetic on assumed inputs. A mean-reversion strategy has one parameter, a lookback window $L$ between 5 and 60 days. Each walk-forward evaluation costs 40 minutes of cluster time, so the 56-point grid is 37 hours and you have budget for six. The objective is annual net P&L on a \$200m allocation. RBF kernel with $\sigma_f$ of \$2.0m, $\ell = 12$ days, backtest noise $\sigma_n$ of \$0.6m, prior mean \$5.0m.

Three space-filling evaluations come back: $L = 10$ gives \$4.2m, $L = 30$ gives \$6.8m, $L = 55$ gives \$5.1m, so the incumbent best $f^{+}$ is \$6.8m. Score every candidate by **expected improvement**, the expected amount by which a new evaluation beats that incumbent:

$$
\mathrm{EI}(x) = (\mu_{\ast} - f^{+})\,\Phi(z) + \sigma_{\ast}\,\varphi(z), \qquad z = \frac{\mu_{\ast} - f^{+}}{\sigma_{\ast}}
$$

Two candidates make the trade-off visible:

- $L = 32$: posterior mean \$6.669m, standard deviation 0.643, so $z = -0.204$, $\Phi(z) = 0.419$, $\varphi(z) = 0.391$, and $\mathrm{EI} = (-0.131)(0.419) + (0.643)(0.391) = 0.196$.
- $L = 38$: posterior mean \$6.429m, standard deviation 1.141, so $z = -0.325$, $\Phi(z) = 0.373$, $\varphi(z) = 0.378$, and $\mathrm{EI} = (-0.371)(0.373) + (1.141)(0.378) = 0.293$.

The rule picks $L = 38$, the candidate with the **lower** expected P&L, because its wider band gives it more room above \$6.8m. That is exploration paid for out of the error bars, and it is why the acquisition function needs a GP rather than a fitted curve.

![Top panel showing the Gaussian process posterior over a backtest lookback parameter with three evaluated points and a one standard deviation band, bottom panel showing the expected improvement curve on the same axis peaking at a lookback of thirty eight days where the posterior mean is lower but the band is wider](/imgs/blogs/gaussian-processes-kernels-math-for-quants-5.webp)

Evaluating $L = 38$ returns \$7.6m. Refit, and the acquisition collapses almost everywhere: its maximum falls from 0.29 to 0.11 and moves to $L = 39$, the signal that little is left to buy. Six evaluations and four hours later the posterior peaks at $L = 38$ with a mean of \$7.45m. Stopping instead at the best of the initial three, $L = 30$, earns \$6.8m against \$7.6m: a gap of **\$0.8m a year on \$200m, or 40 basis points of return**, for the price of three more backtests.

Then the honesty check the same machine hands you free. The posterior difference between $L = 38$ and $L = 30$ is \$0.64m with a standard deviation of \$0.56m on that difference, a ratio of 1.1, so the bands overlap heavily. The conclusion is not "38 is the optimum" but "38 is mildly preferred, everything from 33 to 43 sits within one standard deviation of the peak, so take the middle of that plateau." A grid search hands you an argmax with no such warning, which is exactly how a backtest gets overfitted to a parameter.

## Common misconceptions

**"A GP is a neural network competitor."** They occupy different regimes. A GP is the right tool at $n$ in the hundreds or low thousands, a few well-understood inputs, and a genuine need for calibrated uncertainty. A network is the right tool at $n$ in the millions with raw high-dimensional inputs. The regimes barely overlap, so choosing between them is a data-size question, not a taste question.

**"The kernel is a hyperparameter detail."** The kernel *is* the model, in the sense that the likelihood is the model in a parametric setting. It fixes smoothness, differentiability, periodicity and how fast information decays with distance. Swapping an RBF for a Matern 3/2 on the same data changes the answer more than swapping optimisers ever will. Defend your kernel choice; do not inherit it as a default.

**"Wide error bars mean the model is bad."** Usually they mean it is honest. A band that widens where you have no observations correctly reports a real state of ignorance. The bad case is the opposite: narrow bands everywhere, which normally means a length-scale fitted too long or an observation noise fitted too small, and a model that will be confidently wrong on its first extrapolation. If a GP shows a 71 bp band at the 30-year point, the fault is your quote coverage, not the mathematics.

**"Bayesian methods need a lot of data."** The opposite: small data is where the prior earns its keep, which is also the argument behind [hierarchical pooling](/blog/trading/math-for-quants/hierarchical-bayes-pooling-math-for-quants) and, where no closed form exists, behind [MCMC](/blog/trading/math-for-quants/mcmc-metropolis-gibbs-math-for-quants). A GP is the lucky case needing no sampler at all.

## Sources and further reading

- Carl Rasmussen and Christopher Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006. Free at [gaussianprocess.org/gpml](https://gaussianprocess.org/gpml/). Chapter 2 derives the predictive equations above, chapter 4 catalogues kernels, chapter 5 covers marginal-likelihood fitting, chapter 8 the large-$n$ approximations.
- David MacKay, "Introduction to Gaussian Processes" (1998), and chapter 45 of *Information Theory, Inference, and Learning Algorithms*, Cambridge University Press, 2003, free at [inference.org.uk/mackay/itila](https://www.inference.org.uk/mackay/itila/). The clearest short route from linear models to GPs.
- Jasper Snoek, Hugo Larochelle and Ryan Adams, "Practical Bayesian Optimization of Machine Learning Algorithms", NeurIPS 2012, [arXiv:1206.2944](https://arxiv.org/abs/1206.2944), and Donald Jones, Matthias Schonlau and William Welch, "Efficient Global Optimization of Expensive Black-Box Functions", *Journal of Global Optimization* 13, 1998, for the original expected-improvement treatment.
- Joaquin Quiñonero-Candela and Carl Rasmussen, "A Unifying View of Sparse Approximate Gaussian Process Regression", *JMLR* 6, 2005, and Michalis Titsias, "Variational Learning of Inducing Variables in Sparse Gaussian Processes", AISTATS 2009, for the inducing-point methods above.
- Charles Nelson and Andrew Siegel, "Parsimonious Modeling of Yield Curves", *Journal of Business* 60, 1987, and Lars Svensson, "Estimating and Interpreting Forward Interest Rates", NBER working paper 4871, 1994, the parametric curve-fitting incumbents a GP is an alternative to.

The dollar walkthroughs above are illustrative arithmetic on assumed inputs, not quoted market levels.

## In the interview room and on the desk

The question rarely arrives with the words "Gaussian process" in it. It arrives as: *you have liquid quotes at three tenors and you need a price at a fourth, what do you do?* Or, on a vol desk: *the client wants a four-month expiry and you have three and six, how do you quote it?*

A strong answer reaches uncertainty within two sentences, in this order. First, name the interpolation you would actually run, because refusing to give a point estimate reads as evasion. Second, say the point estimate is not the deliverable: an interpolated rate without a width cannot be risk-managed, since a risk system cannot tell it from an observed one. Third, describe how the width should behave, the part that separates candidates. It should be near zero at a quoted point, grow with distance from the nearest quote, and grow faster once you leave the quoted range. Fourth, say a Gaussian process gives exactly that for free, that the kernel encodes how fast rates decorrelate along the curve, and that the posterior variance is available before the quotes are, so you can choose which quote to go and buy. If you can convert a basis point of width into a DV01 number, do it. That is the answer they remember.

The follow-up is almost always the kernel: *which one, and why?* An RBF assumes an analytic curve, defensible for discount factors and indefensible for a vol term structure with a known event in it, where a Matern 3/2 is safer. "RBF, it is the standard one" is weak even when RBF is right, because the kernel is the only place your assumptions live.

The trap catches rigorous people. A candidate fits a beautiful cubic spline, quotes the interpolated tenor to four decimals, and never says it is a model output, so every downstream system treats it as observed. The second version is fitting the hyperparameters on the quotes and then presenting the posterior variance as unconditional. It is not, and saying so unprompted beats a cleaner derivation.

Two Sigma and Jane Street weight this most, the first for the Bayesian modelling and the second for fast reasoning about what you do and do not know. It appears in any vol-surface or curve-marking seat, and in quant research wherever hyperparameters are tuned against an expensive backtest.
