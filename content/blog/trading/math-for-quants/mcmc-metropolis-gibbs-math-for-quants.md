---
title: "MCMC: sampling from a distribution you cannot write down"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "Most posteriors worth having are impossible to integrate. MCMC sidesteps the integral by building a Markov chain whose stationary distribution is the posterior and then walking it. Detailed balance is why it works, and the diagnostics are the only reason to trust the answer."
tags: ["mcmc", "metropolis-hastings", "gibbs-sampling", "detailed-balance", "bayesian-inference", "convergence-diagnostics", "effective-sample-size", "r-hat", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 23
---

> [!important]
> **TL;DR:** the posterior you want is almost always one you cannot normalise, and MCMC gets you its answers without ever computing the normalising constant.
>
> - The constant is an integral over the whole parameter space. Ten parameters at a hundred grid points per axis is ${10^{20}}$ evaluations, about 3,170 years at a billion per second.
> - MCMC builds a Markov chain whose **stationary distribution is the posterior**, then averages along the path. The time the walk spends in a region is that region's probability.
> - **Detailed balance** is the load-bearing idea: balance the flow between every pair of states and the chain holds the target. It is a *local* condition, which is why an algorithm can enforce it.
> - **Metropolis-Hastings** accepts with probability $\min(1, r)$, where $r$ is a ratio of two unnormalised densities, so the constant cancels. **Gibbs** is the case where every conditional can be drawn exactly, and then $r = 1$ always.
> - The number to remember: a chain stuck in one mode of a bimodal posterior reported a Sharpe of 1.50 instead of 0.60, which sized \$250m instead of \$100m and put \$16.2m of expected P&L into an annual plan that will not arrive.

Write down a model, any model, and Bayes' theorem hands you the posterior in one line: prior times likelihood, divided by a constant that makes the whole thing integrate to one. The line is short, it is correct, and for most models worth building it is useless, because that constant is an integral over every value every parameter could take, and nobody can do it.

That is not a fussy theoretical complaint. It is why a working quant reaches for a sampler the moment a model stops being a textbook example: a hierarchical model of alpha decay across fifty tickers, a regime-switching model of a spread, a fat-tailed likelihood on daily P&L. Every one has a posterior you can write and cannot integrate.

Markov chain Monte Carlo is the way out, and the idea is simple enough to derive at a whiteboard. Instead of computing the posterior, build a random walk that wanders the parameter space, arrange it so the walk spends time in each region in proportion to that region's posterior probability, and read the answer off the walk's own history. The integral never appears. What is hard is not *using* a sampler, it is *trusting* one: knowing whether what came back is the posterior, or a confident-looking summary of a walk that got stuck. On a desk that gap is priced in dollars.

![Two panels contrasting the Bayes normalising constant as an integral costing 10 to the 20 evaluations or 3,170 years, against an MCMC chain of six states wandering a posterior density with a histogram of visit times beneath it](/imgs/blogs/mcmc-metropolis-gibbs-math-for-quants-1.webp)

That figure is the whole post in one image: on the left, what Bayes asks for and what it costs; on the right, a chain of states that visits high-density regions often and low-density regions rarely, with a histogram of where it spent its time converging to the posterior itself. The rest is the machinery that makes the right panel true rather than hopeful.

## Foundations: the constant you cannot compute, and the chain that dodges it

### The posterior and its awkward denominator

This post assumes you know what a prior, a likelihood and a posterior are. If any of those is new, [Bayesian inference for traders](/blog/trading/math-for-quants/bayesian-inference-traders-math-for-quants) builds all three from zero with conjugate updating and credible intervals, and this post picks up where its conjugate shortcuts run out.

For a parameter vector $\theta$ and data $y$:

$$ p(\theta \mid y) = \frac{p(y \mid \theta)\, p(\theta)}{p(y)}, \qquad p(y) = \int p(y \mid \theta)\, p(\theta)\, d\theta. $$

The numerator is easy: given a specific $\theta$ you evaluate the prior density and the likelihood and multiply, usually in microseconds. The denominator $p(y)$, the marginal likelihood, is a single number that does not depend on $\theta$ at all. Its only job is to rescale the numerator into a proper distribution. And it is an integral over the entire parameter space.

Put a number on the difficulty. Lay a grid over the space: ${m}$ points along each of ${d}$ axes, evaluate the numerator at every point, add them up. That costs $m^d$ evaluations. Ten parameters at a hundred points per axis is ${10^{20}}$ evaluations, and at a billion per second that is ${10^{11}}$ seconds, about 3,170 years. Ten parameters is a small model: a hierarchical model with one alpha per name across fifty names has more than five times that. No cleverer quadrature rescues it, because anything that tries to *cover* the space costs exponentially in the dimension. What is needed is a method that does not try to cover.

### What a Markov chain is, and what "stationary" means

A Markov chain is a sequence $\theta_0, \theta_1, \theta_2, \ldots$ in which the distribution of the next state depends only on the current one, encoded in a transition kernel $P(x \to y)$: the density of landing at $y$ given that you are at $x$. A distribution $\pi$ is **stationary** for $P$ if drawing the current state from $\pi$ means the next state is also distributed as $\pi$:

$$ \int \pi(x)\, P(x \to y)\, dx \;=\; \pi(y) \qquad \text{for every } y. $$

Concretely: imagine a million walkers scattered across the space according to $\pi$, and let each take one step under $P$. Every walker has moved, but the *map* of where they are is unchanged. That is what it means for $\pi$ to be the shape the chain holds.

The entire design goal of MCMC is a $P$ whose stationary distribution is the posterior you could not normalise. Given one, run a chain until it forgets its start and the path average converges to the posterior average:

$$ \frac{1}{N}\sum_{t=1}^{N} f(\theta_t) \;\longrightarrow\; \mathbb{E}_{\pi}\!\left[f(\theta)\right]. $$

That is a [law of large numbers](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants) for dependent draws. Two conditions make it true: the chain must reach every region (irreducible) and must not cycle deterministically (aperiodic). It is also asymptotic, so the finite run you did may not be there yet. Every diagnostic below is about one of those two.

## Detailed balance: the reason any of this works

You cannot build $P$ from the stationarity equation directly. It is an integral equation in the unknown $P$, and it involves $\pi$, the thing you cannot normalise. Detailed balance is the move that makes the problem constructive. A chain satisfies it with respect to $\pi$ if, for every pair of states,

$$ \pi(x)\, P(x \to y) \;=\; \pi(y)\, P(y \to x). $$

Read it as an accounting identity for probability flow: in the long run the mass flowing from $x$ to $y$ each step exactly equals the mass flowing back. Such a chain is called reversible, because a film of it run backwards has the same statistics.

Detailed balance is **sufficient** for stationarity, and that is the whole point. The proof is three lines:

$$ \int \pi(x) P(x \to y)\, dx \;=\; \int \pi(y) P(y \to x)\, dx \;=\; \pi(y) \int P(y \to x)\, dx \;=\; \pi(y). $$

The first equality substitutes detailed balance inside the integral, the second pulls $\pi(y)$ out because it does not depend on $x$, and the third uses the fact that $P(y \to \cdot)$ integrates to one.

![Two states A and B with opposing arcs labelled with the probability flows pi of A times P of A to B and pi of B times P of B to A joined by an equals sign, above a three line proof that detailed balance implies stationarity](/imgs/blogs/mcmc-metropolis-gibbs-math-for-quants-2.webp)

The difference in kind that figure makes visible is why this is load-bearing rather than an algebraic convenience. Stationarity is a **global** condition, a statement about the whole space balancing at once. Detailed balance is **local**, a statement about one pair of states at a time, and local conditions are things an algorithm can enforce one proposed move at a time, with no knowledge of the rest of the space and no normalising constant. It is sufficient, not necessary: chains can hold a distribution without being reversible, and some modern samplers give up reversibility to explore faster. But Metropolis-Hastings and Gibbs, the two you will be asked about, are both built on it.

## Metropolis-Hastings: propose, compare, accept or reject

The algorithm is four lines. From the current state $\theta_t$:

1. Draw a proposal $\theta'$ from any distribution $q(\theta' \mid \theta_t)$ you like.
2. Compute

   $$ r = \frac{\pi(\theta')\, q(\theta_t \mid \theta')}{\pi(\theta_t)\, q(\theta' \mid \theta_t)}. $$

3. Draw $u$ from a uniform on the unit interval.
4. If $u \lt \min(1, r)$, set $\theta_{t+1} = \theta'$. Otherwise set $\theta_{t+1} = \theta_t$ and move on.

Here is the trick the whole method rests on. The target $\pi$ appears in $r$ **only as a ratio**. Write $\pi(\theta) = \tilde\pi(\theta)/Z$, where $\tilde\pi$ is the unnormalised density, prior times likelihood, and $Z$ is the constant nobody can compute. Then

$$ \frac{\pi(\theta')}{\pi(\theta_t)} = \frac{\tilde\pi(\theta')/Z}{\tilde\pi(\theta_t)/Z} = \frac{\tilde\pi(\theta')}{\tilde\pi(\theta_t)}. $$

The $Z$ cancels. Every decision the sampler makes needs only quantities you can evaluate in microseconds. The integral that made the problem impossible was never required, because the algorithm only ever compares two points and never needs the scale of the whole thing.

Why does that acceptance rule give detailed balance? Suppose $\pi(x) q(y \mid x) \geq \pi(y) q(x \mid y)$. Then $x$ to $y$ is accepted with probability $\pi(y)q(x \mid y) / (\pi(x)q(y \mid x))$, at most one, while $y$ to $x$ is accepted with probability one. Multiply each direction's proposal density by its acceptance probability and both sides come to the same number, $\pi(y)q(x \mid y)$, which is detailed balance. The $\min$ is not a hack, it is exactly what forces the two directions to agree.

With a symmetric proposal, meaning $q(y \mid x) = q(x \mid y)$ as for a random walk, the proposal terms cancel too and $r$ becomes a bare ratio of target densities. That is the original method of Metropolis and co-authors (1953); Hastings (1970) generalised it to asymmetric proposals.

#### Worked example 1: four Metropolis-Hastings steps by hand

A systematic strategy runs inside a \$500m book. Let $\theta$ be its true mean daily return in basis points of deployed capital. The prior is Normal with mean 0 and standard deviation 1 bp, centred on "no edge", because edges are rare. You have 400 trading days: sample mean 1.5 bps, daily volatility 20 bps, so the standard error of the mean is ${20/\sqrt{400}} = 1.0$ bp, and the likelihood for the observed sample mean is Normal centred at $\theta$ with standard deviation 1. Dropping every constant, the unnormalised log posterior is

$$ \log \tilde\pi(\theta) = -\frac{\theta^2}{2} - \frac{(1.5 - \theta)^2}{2}. $$

Use a symmetric random-walk proposal, so $r = \tilde\pi(\theta')/\tilde\pi(\theta)$. Start at $\theta_0 = 0.00$, where $\log \tilde\pi = -1.125$.

1. **Propose 0.50.** $\log \tilde\pi(0.50) = -0.125 - 0.500 = -0.625$, so $\log r = -0.625 + 1.125 = +0.500$ and $r = 1.649$. Since $r \gt 1$ the move is accepted without drawing $u$ at all. Now $\theta_1 = 0.50$.
2. **Propose 1.30.** $\log \tilde\pi(1.30) = -0.845 - 0.020 = -0.865$, so $\log r = -0.865 + 0.625 = -0.240$ and $r = 0.787$. Draw $u = 0.42$. Since $u \lt r$, accept. Now $\theta_2 = 1.30$.
3. **Propose 2.10.** $\log \tilde\pi(2.10) = -2.205 - 0.180 = -2.385$, so $\log r = -2.385 + 0.865 = -1.520$ and $r = 0.219$. Draw $u = 0.55$. Since $u \gt r$, **reject**. $\theta_3 = 1.30$, the same value again.
4. **Propose 0.85.** $\log \tilde\pi(0.85) = -0.36125 - 0.21125 = -0.5725$, so $\log r = -0.5725 + 0.865 = +0.293$ and $r = 1.340$. Accept. $\theta_4 = 0.85$.

![Table of four Metropolis-Hastings steps showing current theta, proposal, log ratio, ratio, the uniform draw and the accept or reject decision, with the accepted rows green and the rejected row red](/imgs/blogs/mcmc-metropolis-gibbs-math-for-quants-3.webp)

Two things in that figure deserve a second look. The rejected step is **still a draw**: the chain records 1.30 twice, and that repetition is not waste, it is how a high-density region earns extra weight in the time average. And nowhere in those eight numbers did a normalising constant appear.

This model is conjugate, deliberately, so the sampler can be checked. The posterior precision is ${1 + 1 = 2}$, giving a Normal posterior with mean ${(0 \times 1 + 1.5 \times 1)/2 = 0.75}$ and standard deviation ${1/\sqrt{2} = 0.707}$. A long run returns 0.75. Four steps return a running mean of ${(0.50 + 1.30 + 1.30 + 0.85)/4 = 0.99}$, which is the honest lesson: four steps demonstrate mechanics, they are not a posterior.

The money: 0.75 bps a day on \$500m is \$37,500 a day, and 189 bps over 252 trading days, so \$9.45m a year. The raw sample mean of 1.5 bps would have booked \$18.9m. The prior halved the number before a sampler was involved at all, and the sampler's job is to deliver that same halving in models where you cannot do it in your head. These dollar figures are illustrative arithmetic on assumed inputs.

## Gibbs sampling: when every conditional is one you can draw from

Sometimes the joint posterior is hopeless while every *conditional* is a distribution you already know how to sample. Gibbs sampling exploits that: cycle through the coordinates and replace each one, in turn, by an exact draw from its full conditional given the current value of all the others. For $\theta = (\theta_1, \ldots, \theta_d)$, one sweep draws $\theta_1$ from $p(\theta_1 \mid \theta_2, \ldots, \theta_d, y)$, then $\theta_2$ from $p(\theta_2 \mid \theta_1, \theta_3, \ldots, y)$ using the $\theta_1$ just drawn, and so on. Always condition on the most recent value of everything else.

There is no accept-or-reject step, and the reason makes Gibbs a special case rather than a separate algorithm. Treat the update of coordinate $i$ as a Metropolis-Hastings proposal: propose $y$ with $y_{-i} = x_{-i}$ and $y_i$ drawn from $\pi(\cdot \mid x_{-i})$. Then $q(y \mid x) = \pi(y_i \mid x_{-i})$, and the reverse proposal is $q(x \mid y) = \pi(x_i \mid x_{-i})$ because the conditioning set is unchanged. Factor the joint as conditional times marginal and everything cancels:

$$ r = \frac{\pi(y)\,\pi(x_i \mid x_{-i})}{\pi(x)\,\pi(y_i \mid x_{-i})} = \frac{\pi(y_i \mid x_{-i})\,\pi(x_{-i})\,\pi(x_i \mid x_{-i})}{\pi(x_i \mid x_{-i})\,\pi(x_{-i})\,\pi(y_i \mid x_{-i})} = 1. $$

Every proposal is accepted, because it was drawn from exactly the distribution the target wants for that coordinate. Geman and Geman (1984) introduced the method for image restoration, and statistics adopted it wholesale in the early 1990s.

It is not a free lunch. Gibbs moves one axis at a time, so two strongly correlated parameters make it crawl up a narrow diagonal ridge in tiny axis-aligned steps: a posterior correlation of 0.99 can mean thousands of sweeps to cross what one diagonal move would cover, with a perfect acceptance rate of 1 and dreadful mixing. It is the same geometry that slows first-order methods in [stochastic gradient optimisation](/blog/trading/math-for-quants/stochastic-gradient-optimizers-math-for-quants), and the same fix applies: reparameterise so the ridge is not diagonal.

#### Worked example 2: two Gibbs sweeps on a two-parameter P&L model

A sleeve of the same \$500m book has 100 days of daily P&L in thousands of dollars, and two unknowns: the true mean daily P&L $\mu$ and the true daily variance $\sigma^2$. The sample mean is ${\bar y = 12.0}$, so \$12k a day, and the sum of squared deviations about it is ${S = 2{,}227{,}500}$, which is ${99 \times 150^2}$, a sample standard deviation of \$150k a day. With a flat prior on $\mu$ and a prior on $\sigma^2$ proportional to ${1/\sigma^2}$, both conditionals are standard, which is the whole reason Gibbs applies:

$$ \mu \mid \sigma^2, y \;\sim\; \text{Normal}\!\left(\bar y,\; \frac{\sigma^2}{n}\right), \qquad \sigma^2 \mid \mu, y \;\sim\; \text{Inv-Gamma}\!\left(\frac{n}{2},\; \frac{S + n(\bar y - \mu)^2}{2}\right). $$

Start at ${\sigma^2 = 22{,}500}$, that is $\sigma = 150$.

**Sweep 1.** Draw $\mu$ from Normal with mean 12.0 and variance ${22{,}500/100 = 225}$, a standard deviation of 15.0; the standard normal draw comes back at ${-0.40}$, giving ${\mu = 12.0 - 6.0 = 6.0}$. Now draw $\sigma^2$ conditional on that $\mu$: the scale is ${[2{,}227{,}500 + 100 \times (12.0 - 6.0)^2]/2 = 1{,}115{,}550}$ with shape 50, whose mean is ${1{,}115{,}550/49 = 22{,}766}$, and the draw returns ${21{,}904}$, that is $\sigma = 148.0$.

**Sweep 2** repeats with the new $\sigma^2$: $\mu$ comes from Normal with mean 12.0 and variance ${21{,}904/100 = 219.04}$, and a draw of ${+0.20}$ standard deviations gives ${\mu = 12.0 + 2.96 = 14.96}$; the scale for $\sigma^2$ becomes ${[2{,}227{,}500 + 876]/2 = 1{,}114{,}188}$ and the draw returns ${23{,}104}$, that is $\sigma = 152.0$. Every draw was kept: no ratio, no uniform, no rejection.

Run 4,000 sweeps and the $\mu$ draws centre on \$12.0k a day with a standard deviation near \$15k, so a 95% credible interval runs from ${12.0 - 29.4 = -17.4}$ to ${12.0 + 29.4 = 41.4}$ thousand dollars a day. Annualised over 252 days: a central estimate of \$3.02m, with the interval running from a loss of \$4.38m to a gain of \$10.43m. A hundred days on a strategy this volatile cannot tell you whether it makes money, and the posterior says so in a way a point estimate never does.

## Diagnostics: the four questions you must be able to answer

![Matrix of four MCMC diagnostics, trace plot, burn-in, effective sample size and R-hat, each with a healthy column in green, a broken column in red, and what the break costs](/imgs/blogs/mcmc-metropolis-gibbs-math-for-quants-4.webp)

A sampler always returns a pile of numbers, and that pile always has a mean and a standard deviation whether or not it is the posterior. The diagnostics in that figure are the only thing standing between "the sampler returned 1.50" and "the posterior mean is 1.50". Each answers a different question, and a chain can pass three and fail the fourth.

**Burn-in.** The chain starts wherever you put it, and theory only promises convergence eventually. The draws made on the way in come from the wrong distribution and drag the average toward the starting value, so discard them: run 20,000 and throw away the first 5,000. *Broken looks like* a trace still trending where you cut. A chain launched at 1.5 that is passing 0.9 and still falling when burn-in ends has contaminated everything you kept.

**Autocorrelation and effective sample size.** A random-walk sampler moves a small step at a time, so $\theta_{t+1}$ looks a lot like $\theta_t$. The effective sample size is how many independent draws the correlated chain is actually worth:

$$ \mathrm{ESS} = \frac{N}{1 + 2\sum_{k=1}^{\infty} \rho_k}, $$

where $\rho_k$ is the lag-$k$ autocorrelation, the same quantity used on [a return series](/blog/trading/math-for-quants/stationarity-autocorrelation-math-for-quants). For a chain whose autocorrelation decays geometrically from a lag-1 value of 0.95, the sum is ${0.95/(1 - 0.95) = 19}$, the denominator is 39, and ${\mathrm{ESS} = 20{,}000/39 \approx 513}$: you paid for 20,000 draws and you own 513. *Broken looks like* autocorrelation still above 0.5 at lag 50. The cost is precision, through the Monte Carlo standard error, the posterior standard deviation divided by $\sqrt{\mathrm{ESS}}$. With a posterior standard deviation of 0.18 on an annualised Sharpe, an ESS of 513 gives ${0.18/22.65 = 0.008}$, negligible; an ESS of 12 gives ${0.18/3.464 = 0.052}$, and under a sizing rule of \$20m of volatility budget per unit of Sharpe that is over \$1m of risk allocation swinging on Monte Carlo noise alone.

**R-hat.** Run several chains from deliberately dispersed starts, then compare the variance *between* chains to the variance *within* each chain. If every chain found the same distribution the two agree and the statistic sits near 1. Gelman and Rubin (1992) introduced it; modern practice follows Vehtari and co-authors (2021) in using a rank-normalised version and a threshold of 1.01, much tighter than the 1.1 in older textbooks. *Broken looks like* 1.4, or 2.41. A single chain has no R-hat at all, which is why "I ran one very long chain" is not a convergence argument.

**Trace plots.** Plot each chain against iteration number. Healthy is a fuzzy horizontal band, every chain on top of every other, no trend: the shape people call a caterpillar. Three failures are visible instantly and invisible in a summary table. A trend that has not flattened means burn-in was too short. Long flat plateaus mean the proposal steps are too wide and nearly everything is being rejected. And chains sitting at different levels that never cross means the posterior has more than one mode.

## Why a chain that converged can still be wrong

Every diagnostic above answers some version of "have these chains settled down". None answers "have these chains seen everything". A posterior with two well-separated modes, sampled by a chain whose steps are much smaller than the valley between them, will settle down beautifully inside one mode and stay there for the age of the universe. Its trace is a perfect caterpillar, its autocorrelation decays fine, its ESS is large. It is describing a third of the posterior and reporting it as the whole thing.

Multimodal posteriors are not exotic in finance. Any model with a discrete latent state produces them: a regime-switching model has one mode for "we are in the high-edge regime" and another for "we are not". So does any model in which a parameter can explain the data two genuinely different ways. R-hat is the one diagnostic with a chance of catching it, and only if the starts are dispersed enough that different chains land in different modes. Start four chains from the same optimiser output and they will agree with each other, in the same wrong place, and R-hat will read 1.00.

![Bimodal posterior over annualised Sharpe with a mode at 0.12 weighted 65 percent and a mode at 1.50 weighted 35 percent, a dashed line at the true mean of 0.60, one chain confined to the upper mode, and sizing callouts of 250 million dollars versus 100 million dollars deployed](/imgs/blogs/mcmc-metropolis-gibbs-math-for-quants-5.webp)

#### Worked example 3: the cost of the mode you never visited

A systematic equity stat-arb sleeve sits inside a \$500m book, and run at full allocation it carries 12% annualised volatility. Research fits a two-regime model of its edge, and as that figure shows, the posterior over annualised Sharpe is bimodal: mode A, "the edge survived the crowding", centred at 1.50 with weight 35%, and mode B, "the edge has decayed", centred at 0.12 with weight 65%. The posterior mean is ${0.525 + 0.078 = 0.603}$, which the desk rounds to 0.60 for sizing.

*Run 1.* One chain, 20,000 draws, started at the backtest point estimate. It lands in mode A and never leaves, reporting a posterior mean Sharpe of 1.50 with a posterior standard deviation of 0.18. Every within-chain diagnostic is clean.

*Run 2.* Four chains from dispersed starts. Two land in A and two in B, and R-hat comes back at 2.41. Re-run with a sampler that can move between modes, the pooled posterior mean is 0.60.

The desk sizes by risk rather than capital, at \$20m of annualised volatility budget per unit of posterior-mean Sharpe.

- Off run 1: 1.50 times \$20m is \$30m of annual volatility. At 12% strategy volatility that is \$30m divided by 0.12, so \$250m deployed, half the book.
- Off run 2: 0.60 times \$20m is \$12m of annual volatility, so \$100m deployed, a fifth of the book.

Same data, same model, same afternoon, and the \$150m difference is entirely an artefact of whether one chain crossed one valley. Price it two ways.

**The plan.** The extra \$18m of volatility budget was approved on a believed Sharpe of 1.50, so it was expected to earn 1.50 times \$18m, which is \$27m. Its true expected return is 0.60 times \$18m, which is \$10.8m. That is **\$16.2m of expected P&L sitting in the annual plan that will not arrive**, and nobody finds out until the year is over.

**The limit.** Say the drawdown limit on the sleeve is \$25m, 5% of the book. Treating annual P&L as roughly normal around the true posterior mean, the oversized book expects 0.60 times \$30m, which is \$18m, with a standard deviation of \$30m, so the chance of losing more than \$25m is $\Phi(-1.43) \approx 7.6\%$. The correctly sized book expects \$7.2m with a standard deviation of \$12m, so the same chance is $\Phi(-2.68) \approx 0.4\%$. The probability of breaching the limit rose roughly twentyfold, and the only thing that changed was a diagnostic nobody ran. The sizing rule, the volatility figure and the normal approximation are stated assumptions rather than measurements, and the dollar amounts are illustrative arithmetic on them.

## Common misconceptions

**"More samples always fixes it."** More samples fixes Monte Carlo error, the noise from a finite run of a chain that is exploring properly. It does nothing for a chain that is not exploring. Run 1 above, at 20 million draws, returns 1.50 with a tighter interval around it: more confident, equally wrong. The remedy for a stuck chain is never a longer run. It is dispersed starts, a tempered or mode-jumping sampler, or a reparameterisation that flattens the valley.

**"A high acceptance rate means the sampler is working."** Backwards. A 95% acceptance rate means almost every proposal is taken, which means the proposals are barely moving, which means consecutive draws are nearly identical and the ESS is tiny. A rate near zero is the opposite failure, steps so large that almost everything lands somewhere implausible. For a random-walk Metropolis sampler the optimum is about 0.234 in high dimensions and about 0.44 in one dimension (Roberts, Gelman and Gilks, 1997). Both 0.95 and 0.02 are broken, and the 0.95 chain gives the smoother-looking trace, which makes it the more dangerous.

**"MCMC gives you independent draws from the posterior."** It gives you a *dependent* sequence whose long-run time average matches posterior expectations. Quoting percentiles of the draws as a credible interval is fine. Computing a standard error on the posterior mean by dividing by $\sqrt{20{,}000}$ instead of $\sqrt{\mathrm{ESS}}$ understates it by a factor of ${\sqrt{39} \approx 6.2}$ in the example above. For the same reason, thinning the chain does not buy independence: it reduces storage and throws away information, so keep every draw and quote the ESS.

## Sources and further reading

- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H. and Teller, E. (1953), "Equation of State Calculations by Fast Computing Machines", *Journal of Chemical Physics* 21(6), 1087-1092.
- Hastings, W. K. (1970), "Monte Carlo sampling methods using Markov chains and their applications", *Biometrika* 57(1), 97-109.
- Geman, S. and Geman, D. (1984), "Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images", *IEEE Transactions on Pattern Analysis and Machine Intelligence* 6(6), 721-741.
- Gelman, A. and Rubin, D. B. (1992), "Inference from Iterative Simulation Using Multiple Sequences", *Statistical Science* 7(4), 457-472.
- Roberts, G. O., Gelman, A. and Gilks, W. R. (1997), "Weak convergence and optimal scaling of random walk Metropolis algorithms", *Annals of Applied Probability* 7(1), 110-120.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. and Rubin, D. B. (2013), *Bayesian Data Analysis*, 3rd edition, CRC Press. Chapters 11 and 12 are the reference treatment of everything here.
- Hoffman, M. D. and Gelman, A. (2014), "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo", *Journal of Machine Learning Research* 15, 1593-1623.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and Bürkner, P.-C. (2021), "Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC", *Bayesian Analysis* 16(2), 667-718.

Every dollar figure in the worked examples is illustrative arithmetic on assumed inputs, not a measured result.

## In the interview room and on the desk

The question is usually "how would you fit this model?", and it arrives attached to something deliberately non-conjugate: a hierarchical alpha model across a sector, a regime-switching spread, a likelihood with fat tails. Saying "MCMC" is the easy half, and everyone says it.

The strong answer goes in this order. Start with why the posterior is intractable and name the culprit: the normalising constant is an integral over the whole parameter space, and any method that tries to cover that space costs exponentially in the dimension. Then say what MCMC substitutes for it, a Markov chain whose stationary distribution is the target and a time average along its path. If pushed on why that works, give detailed balance and the one-line reason Metropolis-Hastings needs only an unnormalised density, that the constant cancels in the ratio. Then, without waiting to be asked, say how you would know it worked: several chains from dispersed starts, R-hat under 1.01, effective sample size in the thousands rather than the hundreds, overlapping trace plots, an acceptance rate nowhere near 0 or 1. Finish with what you would do if it had not worked, and make the first item a reparameterisation rather than more iterations, because a funnel or a near-perfect posterior correlation is a geometry problem that compute does not solve.

The trap is quoting a posterior mean with nothing behind it. It looks rigorous, it has decimal places, it came out of a real library. A candidate who says "the posterior mean Sharpe is 1.5" and cannot say how many chains were run, what R-hat was, or what the effective sample size was has told the interviewer they treat a sampler as an oracle. The follow-up that exposes it is always some version of "and what if the posterior were bimodal?", because the honest answer is that every within-chain diagnostic would still have looked clean and the position would still have been more than twice the size it should have been.

Two Sigma and WorldQuant weight this most heavily, since both run hierarchical and Bayesian models in production research and both prefer computational-statistics questions to closed-form probability puzzles. Jane Street and Citadel are likelier to probe the layer underneath, the Markov chain and its stationary distribution, than the sampler sitting on top of it.
