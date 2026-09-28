---
title: "Optimal transport: a distance between distributions that respects geometry"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "KL divergence goes to infinity the moment two distributions stop overlapping, which is exactly when a quant most needs to know how far apart they are. Wasserstein distance measures the cheapest way to move mass from one to the other, so it sees the line the returns live on and reports an answer in the units of the returns themselves."
tags: ["optimal-transport", "wasserstein-distance", "kantorovich-duality", "distributionally-robust-optimization", "sinkhorn", "kl-divergence", "model-validation", "backtest-overfitting", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 19
---

> [!important]
> **TL;DR:** Wasserstein distance asks the cheapest cost of moving one distribution onto another, so unlike KL divergence it knows the difference between a near miss and a far one, and it answers in the units of the variable you are measuring.
>
> - KL divergence is infinite whenever one distribution puts mass where the other puts none. Two return distributions sitting 10 bps apart and two sitting 200 bps apart both score infinity, which is the least useful possible answer.
> - In one dimension the whole computation collapses to arithmetic: sort both samples, average the gaps between matched quantiles. Two four-outcome distributions with **identical means** come out 25 bps apart, and the same pair breaks KL entirely.
> - Because Wasserstein-1 equals the largest expected-value gap over all 1-Lipschitz payoffs, a 25 bp distance is a hard **\$1.25m** bound on a \$500m book. A divergence cannot be converted into money; a metric with units can.
> - A live book running 0.78% per month from its backtest in Wasserstein terms is giving up \$3.9m a month in the worst case, of which \$3.5m is the mean shortfall and **\$400k a month is pure change of shape**.
> - The honest limit: Wasserstein charges a rare event only its probability. One day in 250 arriving at -12% moves the 99% expected shortfall by \$20.4m and moves Wasserstein-1 by **\$204k**. For tails it is the wrong instrument, and there KL's infinity is the correct alarm.

## The risk report that says infinity

A model validation team is asked a simple question: does the live return distribution of a strategy match the one the backtest produced? They compute a Kullback-Leibler divergence between the two and the number comes back infinite. They compute it again on a different strategy, whose live returns are visibly miles from the backtest, and get infinity again. Both answers are correct and neither is any use.

This is not a coding error. It is the defining property of *divergences* built out of likelihood ratios. They can only read a distribution where both densities are positive, so the instant one distribution puts probability somewhere the other does not, the ratio divides by zero and the answer saturates. Two samples of real returns almost never share a single value, so on raw empirical data this failure is not an edge case. It is the normal case.

![Two panels on a shared return axis: a near miss where the two distributions sit 0.10 percentage points apart and a far miss where they sit 2.00 percentage points apart, with KL reading infinity in both panels while Wasserstein reads 10 basis points and 200 basis points](/imgs/blogs/optimal-transport-wasserstein-math-for-quants-1.webp)

That figure is the whole thesis. **Optimal transport** fixes it by asking a different question. Instead of comparing the two distributions pointwise, it asks what it would *cost* to physically move the mass of one until it sits exactly on top of the other, where cost is mass multiplied by the distance it travels. That question is always answerable, it is finite whenever the distributions have finite means, and its answer carries the units of the axis the returns live on.

## Foundations: divergence, metric, and why the difference is money

A **distribution** here just means the pattern of outcomes: for a daily return series, how often each size of move happens. A **divergence** and a **metric** are both ways of putting a number on how different two of them are, but they obey different rules.

The **Kullback-Leibler divergence** of $Q$ from $P$, for discrete outcomes, is

$$D_{KL}(P \parallel Q) = \sum_x p(x)\,\ln\frac{p(x)}{q(x)}.$$

Here $p(x)$ and $q(x)$ are the probabilities the two distributions assign to outcome $x$. Read it as the average number of nats of surprise you suffer by believing $Q$ when the truth is $P$. It is a genuinely deep quantity, and the [information geometry post](/blog/trading/math-for-quants/information-geometry-math-for-quants) works through the curved-manifold picture that sits behind it. But look at the formula when $q(x) = 0$ and $p(x) \gt 0$: the logarithm diverges and the whole sum is $+\infty$.

KL also fails two of the three axioms a **metric** must satisfy. A metric $d$ needs $d(P,Q) = 0$ only when $P = Q$, symmetry $d(P,Q) = d(Q,P)$, and the triangle inequality. KL is asymmetric and violates the triangle inequality, which is why nobody calls it a distance. The [metric spaces post](/blog/trading/math-for-quants/metric-spaces-convergence-math-for-quants) works through why those axioms are worth insisting on.

The practical consequence is about **units**. KL is measured in nats, an information unit with no relationship to the quantity being modelled. Wasserstein distance is measured in whatever the axis is measured in, so on a return axis it comes out in return, and return multiplied by notional is dollars. That single fact is why a desk can act on one number and cannot act on the other.

A note on notation used throughout: a **basis point**, or bp, is one hundredth of a percent, so 25 bps is 0.25%. **Notional** is the dollar size the returns apply to.

## The Monge problem, and why Kantorovich relaxed it

Gaspard Monge posed the original question in 1781 in terms of moving earth. You have a pile of soil shaped like $P$ and a hole shaped like $Q$, and you want the cheapest scheme for filling the hole. Formally you look for a **map** $T$ that sends every point $x$ to a destination $T(x)$, such that pushing $P$ through $T$ produces exactly $Q$, and you minimise

$$\int c\big(x, T(x)\big)\,dP(x),$$

where $c(x,y)$ is the cost of moving one unit of mass from $x$ to $y$.

The trouble is that such a map need not exist. Suppose $P$ has three equally likely outcomes, each carrying mass ${1/3}$, and $Q$ has two, each carrying ${1/2}$. A map must send each source point somewhere *whole*. Three chunks of ${1/3}$ cannot be reassembled into two chunks of ${1/2}$ without splitting one of them, so the feasible set is empty and the minimisation is meaningless.

![Left panel shows the Monge problem failing because three source atoms of mass one third cannot be mapped whole onto two target atoms of mass one half, right panel shows the Kantorovich coupling splitting the middle atom into two halves of one sixth so a feasible plan always exists](/imgs/blogs/optimal-transport-wasserstein-math-for-quants-2.webp)

Leonid Kantorovich's 1942 relaxation is the move that made the field usable. Stop looking for a map and look for a **coupling**: a joint distribution $\pi(x,y)$ over pairs, whose marginals are $P$ and $Q$. A coupling is a *plan*, a full table saying how much mass goes from each source to each destination, and it may split a source across several destinations. Now minimise

$$\inf_{\pi \in \Pi(P,Q)} \int c(x,y)\,d\pi(x,y).$$

Two things change. First, the feasible set is never empty, because the independent coupling $\pi = P \otimes Q$ always qualifies, so a minimum always exists. Second, the objective is *linear* in $\pi$ and the constraints are *linear* equalities, so this is a linear program. An intractable search over functions became an object every optimiser on earth can solve. The relaxation is the whole trick.

## Wasserstein-p, and what the exponent does

Take the cost to be a power of the distance moved, $c(x,y) = |x-y|^p$. The **Wasserstein-p distance** is

$$W_p(P,Q) = \left(\inf_{\pi \in \Pi(P,Q)} \int |x-y|^p \, d\pi(x,y)\right)^{1/p}.$$

The outer root restores the units, so $W_p$ is measured in the same units as $x$ for every $p$. It is a genuine metric on distributions with finite $p$-th moment.

The exponent decides how a long move is priced against many short ones. $W_1$ charges mass times distance, so moving one unit five steps costs the same as moving five units one step. $W_2$ charges the square, so it is the choice when a big relocation should be penalised disproportionately, and it is the exponent with the richest geometry. As $p$ rises the distance rises too:

$$W_1 \le W_2 \le \dots \le W_\infty$$

and at the limit $W_\infty$ reports only the single largest move the plan is forced to make.

For the four-outcome pair in the next section the three read 25 bps, 27.4 bps and 40 bps. Most desk work uses $W_1$, because mass times distance is what a P&L gap actually is.

## The one-dimensional case, where it all becomes arithmetic

In one dimension there is a closed form, and it is the reason a quant can use this by hand.

Write $F_P^{-1}$ for the **quantile function** of $P$: feed it 0.25 and it returns the level below which a quarter of the outcomes fall. Then

$$W_p(P,Q)^p = \int_0^1 \left|F_P^{-1}(u) - F_Q^{-1}(u)\right|^p du.$$

For two equally weighted samples of the same size $n$, the quantile functions are step functions on the sorted data, and the integral becomes a plain average:

$$W_1(P,Q) = \frac{1}{n}\sum_{i=1}^{n}\left|x_{(i)} - y_{(i)}\right|,$$

with $x_{(i)}$ and $y_{(i)}$ the $i$-th smallest values of each sample. **Sort both, match them in order, average the gaps.** That is the entire computation.

Why is matching in sorted order optimal? Take any plan that *crosses*: it sends $x_1$ to $y_2$ and $x_2$ to $y_1$, where $x_1 \lt x_2$ and $y_1 \lt y_2$. Swap the two destinations. The cost changes from $|x_1-y_2| + |x_2-y_1|$ to $|x_1-y_1| + |x_2-y_2|$, and for an absolute-value cost the second is never larger. With $x_1 = -0.8$, $x_2 = +1.0$, $y_1 = -0.5$ and $y_2 = +0.6$, the crossed pairing costs ${1.4 + 1.5 = 2.9}$ and the sorted pairing costs ${0.3 + 0.4 = 0.7}$. Any crossing can be uncrossed without raising the cost, so an optimal plan has no crossings, and the only crossing-free plan is the sorted one.

![A matched quantile ladder showing four backtest outcomes at minus 0.8, minus 0.1, plus 0.3 and plus 1.0 percent connected in sorted order to four live outcomes at minus 0.5, plus 0.1, plus 0.2 and plus 0.6 percent, with gaps of 0.30, 0.20, 0.10 and 0.40 summing to 1.00 and dividing by four to give 25 basis points](/imgs/blogs/optimal-transport-wasserstein-math-for-quants-3.webp)

#### Worked example 1: two four-day return distributions and a \$500m book

A strategy's backtest produced four equally likely daily outcomes and the live book produced four more. In percent:

| | outcome 1 | outcome 2 | outcome 3 | outcome 4 | mean |
| --- | --- | --- | --- | --- | --- |
| Backtest $P$ (sorted) | -0.80 | -0.10 | +0.30 | +1.00 | **+0.10** |
| Live $Q$ (sorted) | -0.50 | +0.10 | +0.20 | +0.60 | **+0.10** |
| gap | 0.30 | 0.20 | 0.10 | 0.40 | |

The means are identical, so every mean-based comparison reports a perfect match. Now the transport calculation, step by step:

1. Sort each sample. Done in the table.
2. Take the gap at each matched quantile: ${0.30, 0.20, 0.10, 0.40}$.
3. Sum them: ${0.30 + 0.20 + 0.10 + 0.40 = 1.00}$.
4. Divide by the four outcomes: $1.00 / 4 = 0.25\%$, which is **25 bps**.

The higher exponents come from the same four gaps. Squaring gives ${0.09 + 0.04 + 0.01 + 0.16 = 0.30}$, and ${0.30/4} = 0.075$, so $W_2 = \sqrt{0.075} = 0.2739\%$, or 27.4 bps. The largest single gap is 0.40, so $W_\infty = 40$ bps.

Now KL on the same pair. The two samples share no common outcome at all, so for every value $Q$ puts mass on, $P$ puts none. Every term of the sum divides by zero and $D_{KL}(Q \parallel P) = +\infty$, as does the reverse direction. The distributions differ in a perfectly ordinary way, the spread narrowed from a standard deviation of 0.652% to 0.394%, and the divergence has nothing to say about it.

*The one-sentence intuition: two distributions can have exactly the same mean and still be 25 bps apart, and it takes a distance that reads the axis to notice.*

**Verifying it a second way.** The sorted-average formula is a theorem about the optimal plan, and a theorem applied wrongly still returns a number. So the same distance was recomputed by solving the Kantorovich linear program directly, with sixteen decision variables $\pi_{ij}$, eight equality constraints forcing each row and each column to sum to ${1/4}$, and the cost matrix $|x_i - y_j|$. The LP optimum is 0.25, matching to machine precision, and the plan it returns puts ${1/4}$ on each diagonal entry and zero everywhere else. That is the sorted matching, recovered by an optimiser that was never told about sorting.

## The dual, and why the answer converts into dollars

The Kantorovich problem is a linear program, so it has a dual, and for $p = 1$ the dual is startlingly clean. **Kantorovich-Rubinstein duality** says

$$W_1(P,Q) = \sup_{\;\|f\|_L \le 1\;} \left(\mathbb{E}_P[f] - \mathbb{E}_Q[f]\right),$$

where the supremum runs over all **1-Lipschitz** functions, meaning functions that never change by more than one unit of output per unit of input. Constructing the dual and reading the multipliers as prices is the subject of the [convex duality post](/blog/trading/math-for-quants/convex-duality-shadow-prices-math-for-quants); here we only need the statement.

Read it as a trading result and it is the most useful sentence in this post. A payoff function on the daily return, whether linear P&L, an option overlay, a stop-loss rule, or a risk charge, has some sensitivity $L$ to that return. Duality then says

$$\left|\mathbb{E}_P[f] - \mathbb{E}_Q[f]\right| \le L \cdot W_1(P,Q),$$

and the supremum is attained, so the bound is tight. **The Wasserstein-1 distance is exactly the worst expected P&L gap per unit of exposure, across every payoff of unit sensitivity.** For worked example 1 on a \$500m book, $L$ is \$500m and the bound is $\$500\text{m} \times 0.0025 = \$1.25\text{m}$, with \$1.37m for the $W_2$ reading. A divergence in nats offers nothing comparable.

#### Worked example 2: backtest versus live on a \$500m book

A systematic book runs \$500m of capital. The backtest produced ten monthly returns and the live account has now produced ten more. Sorted, in percent:

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Backtest | -1.9 | -0.8 | -0.4 | +0.6 | +1.1 | +1.4 | +2.1 | +2.3 | +2.8 | +3.2 | **+1.04** |
| Live | -1.5 | -1.3 | -0.9 | -0.2 | +0.3 | +0.8 | +1.1 | +1.4 | +1.7 | +2.0 | **+0.34** |
| gap | 0.4 | 0.5 | 0.5 | 0.8 | 0.8 | 0.6 | 1.0 | 0.9 | 1.1 | 1.2 | |

The gaps sum to 7.8, so $W_1 = {7.8/10} = 0.78\%$ per month. The mean shortfall is $1.04 - 0.34 = 0.70\%$ per month. Converting on \$500m:

- Mean shortfall: $\$500\text{m} \times 0.0070 = \$3.5\text{m}$ per month, or \$42.0m a year at twelve months, ignoring compounding.
- Wasserstein bound: $\$500\text{m} \times 0.0078 = \$3.9\text{m}$ per month, or \$46.8m a year.
- The difference, **\$400k per month or \$4.8m a year, is the part of the degradation that is not a level shift at all.**

That residual is the reason to compute the distance rather than the mean. The live book is not simply a worse version of the backtest: its standard deviation fell from 1.578% to 1.201%, and its worst month, -1.5%, is actually *better* than the backtest's worst of -1.9%. The distribution changed shape. Any overlay whose payoff is nonlinear in the monthly return, an options hedge, a volatility-target sizing rule, a drawdown stop, sees up to \$400k a month of gap that the mean comparison prices at zero. KL, once again, returns infinity, because no two of these twenty numbers coincide.

*The one-sentence intuition: the mean tells you how much worse live is, and the Wasserstein distance tells you how much worse it could be for someone whose payoff is not the mean.*

## Entropic regularisation and Sinkhorn

Exact optimal transport between two samples of size $n$ is an assignment problem costing roughly $O(n^3 \log n)$. Beyond one dimension that becomes the binding constraint, and the standard fix is **entropic regularisation**: add a penalty on the entropy of the plan,

$$\text{OT}_\eta(P,Q) = \inf_{\pi \in \Pi(P,Q)} \left[\int c\,d\pi + \eta \sum_{ij}\pi_{ij}\big(\ln \pi_{ij} - 1\big)\right].$$

The regularised problem has a solution of the form $\pi = \text{diag}(u)\,K\,\text{diag}(v)$ with $K_{ij} = e^{-c_{ij}/\eta}$, and Cuturi's 2013 **Sinkhorn** algorithm finds $u$ and $v$ by alternately rescaling rows and columns. Each iteration is a matrix-vector product, so the cost per pass is $O(n^2)$ and it runs on a GPU. This is what made optimal transport practical at scale.

The honesty requirement is that **what Sinkhorn returns is not the Wasserstein distance.** It is a regularised surrogate, and it is biased upward. Running the algorithm to convergence on worked example 1, whose true $W_1$ is 25 bps:

| regularisation $\eta$ | $\text{OT}_\eta(P,Q)$ | bias | $\text{OT}_\eta(P,P)$ | Sinkhorn divergence |
| --- | --- | --- | --- | --- |
| 1.00 | 53.8 bps | +28.8 bps | 45.1 bps | 15.2 bps |
| 0.50 | 45.4 bps | +20.4 bps | 25.3 bps | 21.6 bps |
| 0.20 | 32.9 bps | +7.9 bps | 4.7 bps | 27.1 bps |
| 0.10 | 27.5 bps | +2.5 bps | 0.4 bps | 26.3 bps |
| 0.05 | 25.6 bps | +0.6 bps | 0.0 bps | 25.3 bps |
| 0.02 | 25.0 bps | +0.0 bps | 0.0 bps | 25.0 bps |

Two readings matter. At $\eta = 0.2$ the reported number is 32.9 bps against a truth of 25 bps, an overstatement of 32%. And the fourth column is worse than a bias: $\text{OT}_\eta(P,P) = 45.1$ bps at $\eta = 1$ means the regularised object reports a distribution as 45 bps away *from itself*, so it is not a metric at all. The **Sinkhorn divergence**, $S_\eta(P,Q) = \text{OT}_\eta(P,Q) - \tfrac12\text{OT}_\eta(P,P) - \tfrac12\text{OT}_\eta(Q,Q)$, subtracts the self-terms and behaves far better, reading 27.1 bps at $\eta = 0.2$ where the raw quantity read 32.9. If you report a Sinkhorn output, say which of the three objects it is and at what $\eta$.

## Where it earns its place: a distributionally robust book

The application that has actually changed practice is **distributionally robust optimisation**. Rather than optimising against your estimated return distribution $\hat{P}$, you optimise against the worst distribution within a Wasserstein ball of radius $\varepsilon$ around it. The ball is the honest statement that $\hat{P}$ came from a finite sample and the truth is somewhere nearby.

Mohajerin Esfahani and Kuhn (2018) proved the result that makes this tractable: for a loss that is Lipschitz in the returns, the worst case over the ball equals the empirical expectation plus $\varepsilon$ times the Lipschitz constant. For a linear loss $-w^\top R$ with ground metric $\|\cdot\|_\infty$ on returns, the Lipschitz constant is $\|w\|_1$, so

$$\sup_{Q \in \mathcal{B}_\varepsilon(\hat{P})} \mathbb{E}_Q\!\left[-w^\top R\right] = -w^\top \bar{R} + \varepsilon \|w\|_1.$$

The robust problem is the ordinary problem with a penalty on gross exposure. Ambiguity about the distribution becomes a regulariser, which is the same lesson the [robust and regularized portfolios post](/blog/trading/math-for-quants/robust-regularized-portfolios-math-for-quants) reaches from the estimation-error side.

#### Worked example 3: the cost of robustness on a \$500m book

A fund runs \$500m across three sleeves, with risk aversion $\lambda = 10$ and, for arithmetic that a reader can check, a diagonal covariance. Only the first moment is robustified; the risk model is held fixed, because a quadratic is not Lipschitz on unbounded support and the theorem above does not cover it.

| sleeve | expected return $\mu$ | volatility $\sigma$ |
| --- | --- | --- |
| Momentum | 6.0% | 12% |
| Carry | 4.0% | 10% |
| Value | 2.0% | 8% |

Maximising

$$w^\top\mu - \varepsilon\|w\|_1 - \tfrac{\lambda}{2}w^\top\Sigma w$$

over long positions gives a closed form that is worth memorising:

$$w_i = \frac{\max(\mu_i - \varepsilon,\; 0)}{\lambda \sigma_i^2}.$$

**The Wasserstein radius is a haircut of $\varepsilon$ on every expected return**, and any sleeve whose edge does not clear $\varepsilon$ is dropped outright. At $\varepsilon = 0$ the weights are $0.06/(10 \times 0.0144) = 0.4167$, then 0.4000 and 0.3125, for 112.9% gross, an expected return of 4.725% and a certainty equivalent of 2.3625%, which is \$11.81m a year on \$500m.

![Two cost curves against Wasserstein radius from zero to six percent on a five hundred million dollar book, one showing expected return given up rising almost linearly to twenty three million dollars and one showing certainty equivalent cost rising quadratically to twelve million, with sleeve drop-out lines marked at two and four percent](/imgs/blogs/optimal-transport-wasserstein-math-for-quants-4.webp)

Now walk the radius out. At $\varepsilon = 0.50\%$ the weights fall to 0.3819, 0.3500 and 0.2344, and expected return falls to 4.1604%. That is \$2.82m a year of expected return surrendered, which sounds punitive. It is not, because most of what was surrendered was compensation for risk that also went away. Substituting the closed form back into the objective gives, for every sleeve still in the book,

$$U_i(\varepsilon) = \frac{\mu_i^2 - \varepsilon^2}{2\lambda\sigma_i^2}, \qquad \text{so the loss is} \qquad \frac{\varepsilon^2}{2\lambda\sigma_i^2}.$$

Summed over the three sleeves, $\sum_i 1/(\lambda\sigma_i^2) = 32.5694$, so the certainty-equivalent cost of robustness is $\tfrac{\varepsilon^2}{2}\times 32.5694$ while all three sleeves remain. At $\varepsilon = 0.005$ that is $0.005^2 / 2 \times 32.5694 = 0.00040712$, or **\$203,559 a year**, 4.07 bps of capital. Every dollar of expected return given up costs only 7.2 cents of genuine risk-adjusted value.

| radius $\varepsilon$ | gross | expected return | vol | return given up | certainty-equivalent cost |
| --- | --- | --- | --- | --- | --- |
| 0.00% | 112.9% | 4.725% | 6.87% | \$0 | \$0 |
| 0.25% | 104.8% | 4.443% | 6.47% | \$1.41m | \$50,890 |
| 0.50% | 96.6% | 4.160% | 6.06% | \$2.82m | \$203,559 |
| 1.00% | 80.3% | 3.596% | 5.28% | \$5.65m | \$814,236 |
| 2.00% | 47.8% | 2.467% | 3.89% | \$11.29m | \$3.26m |
| 4.00% | 13.9% | 0.833% | 1.67% | \$19.46m | \$8.34m |

Because the cost is quadratic, doubling the radius quadruples the bill: \$203,559 becomes \$814,236 exactly. The kinks are the interesting part. At $\varepsilon = 2.00\%$ the Value sleeve's entire 2.0% edge is consumed and it leaves the book; at $\varepsilon = 4.00\%$ Carry follows.

*The one-sentence intuition: the first slice of robustness is close to free, so the argument for buying some of it is strong, and the argument for buying a lot of it has to clear a quadratic bill.*

## Where Wasserstein is the wrong tool

Every technique deserves the case where it fails, computed rather than asserted.

**It under-weights tails by exactly their probability.** Wasserstein cost is mass multiplied by ground distance, and it is linear in both. Take a 250-day sample and let one live day arrive at -12.0% where the backtest's worst day was -1.8%. The move is 10.2 percentage points on one observation out of 250, so its contribution to $W_1$ is ${10.2/250} = 0.0408\%$, which is 4.08 bps. On a \$500m book that is **\$204,000**, and a monitoring system with a 20 bp alert threshold sees nothing.

![Two bars on a five hundred million dollar book comparing what Wasserstein one reports for a single minus twelve percent day, two hundred and four thousand dollars, against what the ninety nine percent expected shortfall reports, twenty point four million dollars, with the actual one day loss of fifty one million noted alongside](/imgs/blogs/optimal-transport-wasserstein-math-for-quants-5.webp)

The actual loss on that day is $\$500\text{m} \times 0.102 = \$51\text{m}$. The 99% expected shortfall averages the worst 1% of observations, which is the worst 2.5 of 250, so moving one observation down by 10.2 points shifts it by ${10.2/2.5} = 4.08$ percentage points, or **\$20.4m**. The ratio between the two readings is exactly $1/\alpha = 100$: a distance that is linear in mass divides a catastrophe by the sample size, while a tail risk measure divides it by the tail probability. When the question is tail risk, [extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants) is the right instrument. And note the reversal: KL is infinite here, and that infinity is the *correct* alarm, because the backtest genuinely assigned zero probability to a -12% day.

**It degrades badly with dimension.** The empirical Wasserstein distance converges to the truth at rate $n^{-1/2}$ in one dimension but only $n^{-1/d}$ for $d \ge 3$ (Fournier and Guillin, 2015). The practical symptom is a noise floor: two independent samples drawn from the *same* distribution are measured as a long way apart. In a simulation with $n = 250$ draws per sample, 60 repetitions and seed 20260921, two independent standard Gaussian samples measure 0.115 apart in one dimension and 2.330 apart in ten. Introducing a genuine mean shift of 0.5 standard deviations in one coordinate lifts the measured distance by **347% in one dimension and by 2.1% in ten**. To get a comparable signal in ten dimensions the shift has to reach about 3 standard deviations. With a year of daily data and a ten-asset book, a raw multivariate Wasserstein distance is mostly estimation noise, and the defensible move is to compute it on a one-dimensional projection such as the portfolio's own return series.

*(The dollar figures throughout this post are illustrative arithmetic on assumed inputs, not observed market data.)*

## Common misconceptions

**"Wasserstein is just another divergence, like KL with different weights."** It is a metric, and that is a stronger property, not a cosmetic one. Symmetry means the answer does not depend on which distribution you call the model. The triangle inequality means distances compose, so if live is 25 bps from the backtest and the backtest is 40 bps from the paper strategy, live is at most 65 bps from the paper strategy, and you can chain model validation results. KL supports neither statement.

**"A small Wasserstein distance means the distributions have the same shape."** It means the *total* mass-times-distance of the rearrangement is small, which is a budget, not a shape statement. Worked example 1 has identical means and a 25 bp distance; the tail example has a completely different shape and a 4.08 bp distance. A small number says no large amount of mass moved far, and nothing more.

**"Sinkhorn gives you the Wasserstein distance."** It gives a regularised surrogate biased upward by an amount that depends on $\eta$, the ground cost scale and the sample size. At $\eta = 0.2$ on the worked example, that bias is 32%. The raw entropic cost is not even zero between a distribution and itself. Report the Sinkhorn divergence if you want something that behaves like a distance, and always state $\eta$.

**"Use $W_2$, it is the better one."** $W_2$ has the richer geometry and the elegant Brenier map, but it charges the square of the distance moved, which makes it *more* sensitive to outliers, not less. On a return series with a fat left tail, $W_2$ gives a single terrible day disproportionate weight. If you want the P&L interpretation from the dual, $W_1$ is the one that has it.

**"A Wasserstein ball makes the portfolio safe."** It makes the portfolio robust to distributions within $\varepsilon$ in transport distance of your empirical one. It says nothing about a regime that arrives further away than $\varepsilon$, and choosing $\varepsilon$ is itself a judgment call that the theory only partially resolves.

## Sources and further reading

- Cédric Villani, *Optimal Transport: Old and New*, Springer, 2009. The standard reference; chapters 1 to 6 cover Monge, Kantorovich, duality and the one-dimensional case.
- Gabriel Peyré and Marco Cuturi, *Computational Optimal Transport*, Foundations and Trends in Machine Learning 11(5-6), 2019. Free at [optimaltransport.github.io](https://optimaltransport.github.io/). The practical companion, and the source for the entropic and Sinkhorn material here.
- Peyman Mohajerin Esfahani and Daniel Kuhn, "Data-driven distributionally robust optimization using the Wasserstein metric: performance guarantees and tractable reformulations", *Mathematical Programming* 171, 115-166, 2018. The tractable reformulation used in worked example 3.
- Marco Cuturi, "Sinkhorn distances: lightspeed computation of optimal transport", NeurIPS, 2013.
- Jean Feydy et al., "Interpolating between optimal transport and MMD using Sinkhorn divergences", AISTATS, 2019. Where the debiased divergence comes from.
- Nicolas Fournier and Arnaud Guillin, "On the rate of convergence in Wasserstein distance of the empirical measure", *Probability Theory and Related Fields* 162, 707-738, 2015. The $n^{-1/d}$ rate behind the dimension result.
- Leonid Kantorovich, "On the translocation of masses", Doklady Akademii Nauk SSSR 37, 1942.

## In the interview room and on the desk

The question arrives as a practical one, usually from someone who has just described a strategy whose live Sharpe came in below its backtest: *"How would you tell whether the live returns match the backtest?"* It is a model validation question wearing plain clothes, and most candidates answer with a two-sample Kolmogorov-Smirnov test or a KL divergence. Both are defensible and neither is strong.

A strong answer moves in this order. Start by asking what decision the answer feeds, because "are they the same distribution" is a hypothesis test and "how far apart are they, and what does that cost" is an estimation problem, and a desk almost always wants the second. Then note that KS returns a probability with no units and saturates on disjoint support, while KL returns infinity on any two real return samples because they share no atoms. Then propose Wasserstein-1, and say why in one sentence: it is a metric, it carries the units of the returns, and by Kantorovich-Rubinstein duality it *is* the worst expected P&L gap per unit of exposure over all payoffs of unit sensitivity. Show that you can compute it, because in one dimension it is the average gap between sorted samples and takes thirty seconds on a whiteboard. Finish by naming the limit before they ask: it is linear in mass, so it charges a one-in-250 catastrophe only 1/250 of its size, and for tail work you would reach for expected shortfall or extreme value theory instead.

The follow-up is usually about computation, and that is where the trap is. A candidate who says "we use Sinkhorn, it is fast" and then quotes the output as the Wasserstein distance has just reported a number that is biased upward and is not zero between a distribution and itself. Naming the bias, and naming the Sinkhorn divergence as the debiased fix, is the difference between having read the library documentation and having understood the object. The second trap is dimension: quoting a multivariate Wasserstein distance from a few hundred observations without mentioning the $n^{-1/d}$ rate.

Two Sigma weights this most, both in model validation and in the distribution-comparison questions their statistical learning rounds favour. Citadel's multi-strategy side cares about the distributionally robust formulation, since a Wasserstein ball is how a risk officer's discomfort becomes a gross-exposure penalty. Any model validation or model risk seat at a bank will ask some version of it.
