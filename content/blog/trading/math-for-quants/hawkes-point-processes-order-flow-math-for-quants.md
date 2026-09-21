---
title: "Hawkes processes: why trades arrive in clusters"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "A Poisson process says the next trade is independent of the last one, and one look at a tape shows that is false. A Hawkes process is the smallest honest fix, and it changes the size of a market maker's inventory tail by a factor of four."
tags: ["hawkes-process", "point-processes", "order-flow", "market-microstructure", "self-excitation", "branching-ratio", "market-making", "reflexivity", "quantitative-finance", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 18
---

> [!important]
> **TL;DR:** Trades beget trades. A Hawkes process is the smallest honest change to a Poisson process that admits this: each event lifts the intensity of future events, and the lift decays.
>
> - The Poisson process says the next trade is independent of the last one. Real tape clusters, so that assumption is not an approximation, it is the wrong shape.
> - The Hawkes intensity is a baseline plus a sum of decaying kernels over past events. With an exponential kernel the whole thing is computable in one pass.
> - The **branching ratio** ${n}$ is the fraction of activity the market generated for itself. At ${n = 0.75}$, three of every four trades on the tape are echo, and the average cluster runs to four trades.
> - Clustering is a variance story, not a mean story. The variance-to-mean ratio is ${1/(1-n)^2}$, so at ${n = 0.75}$ a market maker's one-sigma inventory is \$2,000,000 where a Poisson model says \$500,000.
> - Fitted branching ratios near 1 are as much a statement about your kernel as about the market. Say so before someone else does.

## Trades do not arrive like raindrops

Sit and watch the tape of a liquid stock for ten seconds. You will not see trades falling at a steady drizzle. You will see a burst of fifteen prints inside half a second, then two seconds of nothing, then another burst. Volume arrives in gusts.

The standard model of "things happening at random times" is the **Poisson process**, and it says the opposite. Under Poisson, the fact that a trade just happened tells you exactly nothing about when the next one comes. Gusts are impossible except by luck. Yet gusts are what every tape shows, in every asset class, on every venue, in every decade.

![Two stacked event rasters over ten seconds carrying the same eighty trades, the Poisson one spread evenly and the Hawkes one clustered into bursts, with the conditional intensity curve below spiking at each event and decaying between them](/imgs/blogs/hawkes-point-processes-order-flow-math-for-quants-1.webp)

The fix is a **Hawkes process**, named for Alan Hawkes, who wrote it down in 1971 for earthquakes. Its one idea: an event raises the rate of future events, and that raise fades away. Everything interesting in this post, including a factor of four on a market maker's risk, follows from that single sentence.

## Foundations: the Poisson process and what independence buys you

Start from zero.

### Counting processes

A **counting process** ${N(t)}$ is just a running tally: the number of trades that have printed by time ${t}$. It starts at 0, it only goes up, and it jumps by 1 at each trade. All the modelling lives in *when* it jumps.

The object that controls the jumps is the **intensity** ${\lambda(t)}$, measured in events per unit time. Informally, over a tiny window of length ${dt}$, the chance of exactly one event is ${\lambda(t)\,dt}$, and the chance of two or more is negligible. If trades come at 8 per second, then in a 10-millisecond window you expect ${8 \times 0.01 = 0.08}$ trades.

### The Poisson process

A **Poisson process** is the counting process whose intensity is a constant ${\lambda}$, no matter what has happened. That single assumption buys you a lot:

- The count in any window of length ${T}$ is Poisson distributed with mean ${\lambda T}$.
- The gaps between consecutive events are exponential with mean ${1/\lambda}$, and they are independent of each other.
- The process is **memoryless**. Having waited two seconds for the next trade tells you nothing about how much longer you will wait.
- Variance equals mean. If you expect 4,800 trades in ten minutes, the standard deviation is ${\sqrt{4800} = 69.3}$ trades.

That last line is the one that breaks. The ratio of variance to mean is called the **Fano factor**, and for a Poisson process it is exactly 1, always. This post is the story of what happens when it is 16 instead.

### The empirical fact Poisson gets wrong

Take a day of trades in a liquid name and split it into one-second bins. Poisson predicts a Poisson histogram of counts. What you actually get is far more zero-bins and far more very-large bins than Poisson allows, with a thin middle. The gaps are also wrong: the distribution of inter-trade times has a much heavier tail than the exponential, because the quiet stretches between clusters are far longer than a constant-rate model can produce.

None of this is subtle. It is the first thing you see, and it is why nobody trading at the second scale models arrivals as Poisson. It is the same empirical world [order book imbalance](/blog/trading/math-for-quants/order-book-imbalance-short-horizon-prediction-math-for-quants) lives in: microstructure is full of short-horizon structure, and assuming independence throws all of it away. For the wider family of distributions markets need, see [the probability distributions that markets actually use](/blog/trading/math-for-quants/probability-distributions-for-markets-math-for-quants).

## Conditional intensity: the object that actually matters

Here is the pivot. Instead of a constant ${\lambda}$, write

$$
\lambda(t \mid \mathcal{F}_{t-}) = \lim_{dt \to 0} \frac{\mathbb{P}\big(\text{an event in } [t, t+dt) \mid \mathcal{F}_{t-}\big)}{dt}
$$

where ${\mathcal{F}_{t-}}$ is everything observable strictly before ${t}$: every past trade time, and whatever else you condition on. This is the **conditional intensity**, and it is a random process, not a number. It is the rate *given what you have seen*, which is the only rate a trader ever has access to. The bookkeeping of "what you have seen by time ${t}$" is exactly the filtration machinery in [filtrations and no look-ahead](/blog/trading/math-for-quants/filtrations-no-lookahead-math-for-quants), and the discipline there is the same discipline here: ${\lambda(t)}$ may use the past, never the present or the future.

Poisson is the special case where the conditioning does nothing. Every other point process is a choice about how it should matter.

## The Hawkes intensity

Hawkes made the simplest non-trivial choice: each past event adds a decaying bump.

$$
\lambda(t) = \mu + \sum_{t_i \lt t} \phi(t - t_i)
$$

${\mu > 0}$ is the **baseline** or immigration rate, the rate of events that arrive for reasons outside the model. ${\phi}$ is the **kernel**, a non-negative function that says how much a past event lifts the present rate, and how that lift fades. The sum runs over every trade that has already printed.

Take the **exponential kernel**, ${\phi(s) = \alpha e^{-\beta s}}$. It is the one you can actually compute with, for a reason we will get to when we fit the thing. Two parameters: ${\alpha}$ is the size of the jump in intensity that a trade causes, and ${\beta}$ is how fast it decays. The half-life of one trade's influence is ${\ln 2 / \beta}$.

For the whole of this post use ${\mu = 2.0}$ trades per second, ${\alpha = 3.0}$ per second, and ${\beta = 4.0}$ per second. These are assumed inputs chosen to keep the arithmetic clean, not fitted values from any dataset.

#### Worked example 1: the intensity path after one print, and what it does to \$50,000 of expected tape

The book has been quiet long enough that the intensity has settled back to its baseline ${\mu = 2.00}$ trades per second. At ${t = 0}$ a trade prints. Track ${\lambda}$ by hand:

$$
\lambda(t) = 2 + 3 e^{-4t}
$$

| time after the trade | ${\lambda(t)}$ |
| --- | --- |
| just before, ${t = 0^-}$ | 2.00 /s |
| just after, ${t = 0^+}$ | 5.00 /s |
| ${t = 0.25}$ s | 3.10 /s |
| ${t = 0.50}$ s | 2.41 /s |
| ${t = 1.00}$ s | 2.05 /s |

The jump is instant and the decay is fast: the half-life is ${\ln 2 / 4 = 0.17}$ seconds. Now integrate to get the expected number of trades in the first second, counting only this one trade's direct influence:

$$
\int_0^1 \big(\mu + \alpha e^{-\beta s}\big)\, ds = \mu + \frac{\alpha}{\beta}\big(1 - e^{-4}\big) = 2 + 0.736 = 2.736
$$

Now put money on it. Suppose the average print is 500 shares of a \$50 stock, so each trade is \$25,000 of notional. A quiet second carries two trades, which is \$50,000 of tape. The second immediately after one print carries 2.736 trades, which at \$25,000 each is \$68,400. One trade pulled \$18,400 of extra volume into the next second, and it did so without a single new piece of information arriving.

![Two line-chart panels, the left showing the intensity stepping from 2.00 to 5.00 at the trade and decaying through 3.10, 2.41 and 2.05 with a 0.17 second half-life, the right showing the measured excess above the 8.0 per second stationary mean decaying four times more slowly](/imgs/blogs/hawkes-point-processes-order-flow-math-for-quants-2.webp)

That figure's right panel is a number we have not derived yet. It comes next.

## The branching interpretation

That equation is honest but opaque. The branching picture makes everything obvious.

Read the Hawkes process as a population. **Immigrants** arrive at the constant rate ${\mu}$ for exogenous reasons: a news print, a fund rebalancing, an index event. Every event, immigrant or not, then independently produces **offspring** by a Poisson process with rate ${\phi(s)}$ at lag ${s}$. Offspring produce offspring. The observed tape is every generation piled together, with no label saying which is which.

The expected number of direct offspring per event is the integral of the kernel:

$$
n = \int_0^\infty \phi(s)\, ds = \int_0^\infty \alpha e^{-\beta s}\, ds = \frac{\alpha}{\beta}
$$

This is the **branching ratio**. With ${\alpha = 3.0}$ and ${\beta = 4.0}$, ${n = 0.75}$.

Now the whole cluster. One immigrant has ${n}$ children on average, ${n^2}$ grandchildren, ${n^3}$ great-grandchildren. Sum the geometric series and the expected total cluster size, immigrant included, is

$$
\mathbb{E}[S] = 1 + n + n^2 + \cdots = \frac{1}{1-n} = \frac{1}{0.25} = 4
$$

So one exogenous event drags three more trades onto the tape behind it, on average. Equivalently, ${n}$ is the fraction of all trades that are endogenous: at ${n = 0.75}$, three of every four prints you see were generated by the market reacting to itself.

![A branching tree with one immigrant trade producing two offspring and one grandchild, beside an accounting panel showing the branching ratio of 0.75, an expected cluster size of 4, and one minute of tape split into 120 immigrants and 360 offspring](/imgs/blogs/hawkes-point-processes-order-flow-math-for-quants-3.webp)

#### Worked example 2: what the branching ratio does to a \$50,000-per-signal sizing rule

The stationary rate follows immediately. Trades arrive at ${\mu}$ per second exogenously, and each drags a cluster of ${1/(1-n)}$ behind it:

$$
\Lambda = \frac{\mu}{1 - n} = \frac{2}{0.25} = 8 \text{ trades per second}
$$

Note what that means for anyone fitting a Poisson process to this tape. They will measure 8 trades per second and call it ${\lambda}$. The true exogenous rate is 2. The baseline ${\mu}$ is not the number you observe, and it is off by a factor of ${1/(1-n)}$.

Over one minute you see ${8 \times 60 = 480}$ trades, of which ${2 \times 60 = 120}$ are immigrants and 360 are echo. Now suppose a desk runs a sizing rule of the form "each informative trade justifies \$50,000 of position", and counts trades as informative events. Over that minute the rule fires 480 times and takes 480 lots of \$50,000, which is \$24,000,000 of exposure. The information content justifies 120 lots, which is \$6,000,000. The rule has taken \$18,000,000 more risk than its own logic supports, purely by double-counting the echo.

The same arithmetic on a single large print: a block of 40,000 shares at \$50.00 is \$2,000,000 of notional, and it will pull an expected ${n/(1-n) = 3}$ further trades behind it, worth three prints at \$25,000 each, so \$75,000 of follow-on tape. Small in size, and that is the point. The echo is not about volume. It is about the fact that the next three prints are not news.

## Stationarity, and what a fitted ratio of 0.9 means

The branching picture hands you the stability condition for free. If each event has more than one child on average, the population explodes. So:

$$
n \lt 1
$$

is exactly the condition for a stationary Hawkes process. Cross it and ${\Lambda = \mu/(1-n)}$ goes negative, which is the algebra telling you the expectation does not exist. This is the same stationarity that [stationarity and autocorrelation](/blog/trading/math-for-quants/stationarity-autocorrelation-math-for-quants) treats for time series, arriving here through a completely different door.

There is a second consequence of branching that almost everyone meets late. Conditional on a trade at time 0, the expected intensity afterwards is **not** ${\mu + \alpha e^{-\beta t}}$. That is only the first generation. Summing every generation gives

$$
\mathbb{E}[\lambda(t) \mid \text{event at } 0] = \Lambda + \alpha e^{-(\beta - \alpha) t}
$$

With our numbers, ${\beta - \alpha = 1.0}$ per second. The bare kernel has a half-life of 0.17 seconds; the clustering you can actually measure has a half-life of ${\ln 2 / 1.0 = 0.69}$ seconds, four times longer. That second figure's right panel shows it: the excess starts at ${8 + 3 = 11.00}$ per second and is still at 9.10 a full second later. Integrating the excess over the first second gives 1.90 extra trades, so Poisson says 8.00 trades in the second after a print and Hawkes says 9.90.

This gap matters when you read the literature. Filimonov and Sornette (2012) fitted a Hawkes process to mid-price changes in E-mini S&P 500 futures and reported branching ratios rising to roughly 0.7 to 0.8 by 2010, arguing that markets had become more reflexive as algorithmic trading grew. Hardiman, Bercot and Bouchaud (2013) refitted the same contract with a **power-law** kernel whose short-time exponent is close to ${-1.15}$, and found that the kernel integrates to essentially 1 in every year from 1998 to 2011. Same data, different kernel, a different story about the world: on their reading markets have always sat near criticality, and what actually changed over the decade is the timescale over which events stay correlated, which shortened.

So when your fit comes back at 0.9, the honest reading is narrow. It says: *given this kernel family and this estimation window*, 90% of activity is attributable to the model's own feedback. It does not say the market is unstable, and it does not, on its own, say anything about reflexivity. A kernel that is too short-ranged pushes unexplained long-range correlation into the branching ratio, and a baseline ${\mu}$ held constant when the true baseline varies over the day does the same thing. A real rise in endogeneity and a misspecified kernel produce the same number.

## Variance: why clustering is an inventory problem

Clustering does not hurt in the mean. A Hawkes process and a Poisson process with ${\lambda = \Lambda}$ have identical expected counts by construction. The difference is all in the second moment.

Use the branching decomposition. Over a long window ${T}$ the count is a compound Poisson sum: ${\text{Poisson}(\mu T)}$ immigrants, each carrying an independent cluster of random size ${S}$. For Poisson offspring of mean ${n}$, the total progeny has ${\mathbb{E}[S] = 1/(1-n)}$ and ${\mathbb{E}[S^2] = 1/(1-n)^3}$. Compound Poisson gives ${\mathrm{Var}(N) = \mu T\, \mathbb{E}[S^2]}$, so

$$
\frac{\mathrm{Var}\big(N(T)\big)}{\mathbb{E}\big[N(T)\big]} \;\longrightarrow\; \frac{1}{(1-n)^2}
$$

At ${n = 0.75}$ the Fano factor is ${1/0.0625 = 16}$. The variance of the trade count is sixteen times Poisson, and the standard deviation is four times Poisson. That factor of four is the entire practical content of this post.

#### Worked example 3: the market maker's \$2,000,000 inventory tail

You make a market in a \$50 stock. Trades arrive at ${\Lambda = 8}$ per second and you are the passive counterparty to one in twelve of them, quoting 500 shares a side. Over a ten-minute window:

1. Trades on the tape: ${8 \times 600 = 4{,}800}$.
2. Your fills: ${4{,}800 / 12 = 400}$, split 200 where you bought and 200 where you sold.
3. Net inventory in shares is ${500 \times (N_{\text{buy}} - N_{\text{sell}})}$.

Assume for now that all the excitation is same-side: buys beget buys, sells beget sells, no cross terms. The next section is about what happens when that assumption is wrong, and it is the load-bearing one.

**Under Poisson.** Both counts are Poisson(200) and independent, so the difference has variance ${200 + 200 = 400}$ and standard deviation ${\sqrt{400} = 20}$ trades. One sigma of inventory is ${20 \times 500 = 10{,}000}$ shares, which at \$50 a share is \$500,000 of notional. A three-sigma window is \$1,500,000, comfortably inside a \$2,000,000 risk limit.

**Under Hawkes at ${n = 0.75}$.** Multiply the variance by the Fano factor: ${16 \times 400 = 6{,}400}$, so the standard deviation is ${\sqrt{6400} = 80}$ trades. One sigma of inventory is ${80 \times 500 = 40{,}000}$ shares, or \$2,000,000. Your risk limit is now a one-sigma event, and three sigma is \$6,000,000, three times the limit.

![Two overlaid bell curves of net inventory after ten minutes, a narrow one at one sigma of \$500,000 for Poisson and a wide one at \$2,000,000 for Hawkes, with the \$2m risk limit marked and the wide curve's tails beyond it shaded](/imgs/blogs/hawkes-point-processes-order-flow-math-for-quants-4.webp)

Price it. To restore the same three-sigma headroom you must cut clip size by the same factor of four, from 500 shares to 125. Holding the fill count fixed at 400 (a simplification, since smaller quotes attract less flow), and taking half-spread capture of 1 cent per share on a 2-cent spread, the window's gross capture falls from 400 fills of 500 shares at 1 cent, which is \$2,000, to 400 fills of 125 shares, which is \$500. Across a 6.5-hour session, that is 39 windows: \$78,000 a day becomes \$19,500 a day. The Poisson assumption is a \$58,500-per-day decision, and you pay it either as forgone capture or as an inventory limit you breach far more often than your model says.

## Multivariate Hawkes: the cross terms are the whole story

Buys and sells are not one process. Write them as two:

$$
\lambda_b(t) = \mu_b + \sum_{t_i^b \lt t} \phi_{bb}(t - t_i^b) + \sum_{t_i^s \lt t} \phi_{bs}(t - t_i^s)
$$

and symmetrically for ${\lambda_s}$. Collect the kernel integrals into a matrix ${\mathbf{N}}$ with entries ${n_{bb}, n_{bs}, n_{sb}, n_{ss}}$. Stationarity now needs the **spectral radius** of ${\mathbf{N}}$ below 1, not any single entry.

Take the symmetric case, ${n_{bb} = n_{ss} = a}$ and ${n_{bs} = n_{sb} = c}$. The eigenvectors are ${(1,1)}$, the total flow, and ${(1,-1)}$, the buy-minus-sell imbalance. Their eigenvalues are ${a + c}$ and ${a - c}$. In each direction the process behaves like a scalar Hawkes with that branching ratio, so the standard deviation multiplier over Poisson is ${1/(1 - a - c)}$ for total flow and

$$
\frac{1}{1 - a + c}
$$

for the imbalance, which is the quantity that becomes your inventory. Now compare two calibrations with the **same** aggregate branching ratio ${a + c = 0.75}$:

- **Self-excitation only**, ${a = 0.75}$, ${c = 0}$. Imbalance multiplier ${1/(1 - 0.75 + 0) = 4}$. One-sigma inventory \$2,000,000, exactly worked example 3.
- **Cross-excitation dominant**, ${a = 0.35}$, ${c = 0.40}$. Imbalance multiplier ${1/(1 - 0.35 + 0.40) = 1/1.05 = 0.95}$. One-sigma inventory is \$500,000 divided by 1.05, which is \$476,190, slightly *below* the Poisson number.

![A two-by-two excitation matrix over two regime cards, one with self-excitation of 0.75 giving a multiplier of 4 and one-sigma inventory of \$2,000,000, the other with cross-excitation of 0.40 giving a multiplier of 0.95 and \$476,190, both with aggregate branching ratio 0.75](/imgs/blogs/hawkes-point-processes-order-flow-math-for-quants-5.webp)

Two calibrations, one aggregate branching ratio, and a 4.2x difference in the number that decides whether you survive the day. The aggregate ratio is the number everyone quotes and it is the number that tells you least.

Which regime is real depends on what you count. For **trade signs**, same-side excitation usually dominates, because institutions split large parent orders into many child orders that all arrive on the same side. That is the long-memory order flow of Lillo and Farmer (2004), and it is why the frightening calibration is the relevant one for a market maker. For **signed price moves**, the off-diagonal usually dominates instead: an up-tick makes a down-tick more likely, which is how a bivariate Hawkes reproduces the mean-reverting bid-ask bounce (Bacry, Delattre, Hoffmann and Muzy, 2013). Same mathematics, opposite sign, opposite conclusion.

## Fitting it

The log-likelihood of a point process observed on ${[0, T]}$ is

$$
\ell(\theta) = \sum_{i} \log \lambda(t_i) - \int_0^T \lambda(u)\, du
$$

The first term rewards high intensity where events happened; the second penalises high intensity everywhere else. For the exponential kernel the compensator has a closed form,

$$
\int_0^T \lambda(u)\, du = \mu T + \frac{\alpha}{\beta} \sum_i \Big(1 - e^{-\beta (T - t_i)}\Big)
$$

and the sum inside ${\lambda(t_i)}$ collapses into a recursion. Define ${R_i = \sum_{j \lt i} e^{-\beta(t_i - t_j)}}$. Then

$$
R_i = e^{-\beta(t_i - t_{i-1})}\big(1 + R_{i-1}\big), \qquad R_1 = 0, \qquad \lambda(t_i) = \mu + \alpha R_i
$$

which turns an ${O(m^2)}$ double sum into one ${O(m)}$ pass. On four trades at ${t = 0.00, 0.10, 0.35, 0.40}$ seconds with our parameters:

| ${i}$ | ${t_i}$ | ${R_i}$ | ${\lambda(t_i)}$ |
| --- | --- | --- | --- |
| 1 | 0.00 | 0.0000 | 2.000 |
| 2 | 0.10 | 0.6703 | 4.011 |
| 3 | 0.35 | 0.6145 | 3.843 |
| 4 | 0.40 | 1.3218 | 5.965 |

That recursion is the only reason the exponential kernel is the default, and it is why a power-law kernel, which has no such collapse, is fitted either by approximating it as a sum of exponentials or by paying the quadratic cost. Ogata (1978, 1981) established the maximum-likelihood framework and the residual diagnostics; the general apparatus is the one in [maximum likelihood and the method of moments](/blog/trading/math-for-quants/mle-method-of-moments-math-for-quants).

Three traps, in order of how often they bite:

1. **The kernel choice dominates the fit.** This is the Filimonov-Sornette versus Hardiman result in miniature. Report the branching ratio under at least two kernel families, or report it as a property of the specification rather than of the market.
2. **Edge effects.** Events before ${t = 0}$ are unobserved but still exciting the process inside your window. A model that ignores them attributes their excitation to ${\mu}$, biasing ${n}$ down. Discard a burn-in period longer than several kernel half-lives.
3. **A non-constant baseline looks like excitation.** Intraday volume has a U-shape. Fit a constant ${\mu}$ across a session and the model explains the open-and-close hump as self-excitation, inflating ${n}$. Fit ${\mu(t)}$, or fit within a window short enough that the baseline is flat.

Diagnostics are the part people skip. The **random time change** says that if the model is right, the transformed times ${\Lambda(t_i) = \int_0^{t_i} \lambda(u)\, du}$ form a unit-rate Poisson process. Plot those residual gaps against an exponential; a bad fit shows up there long before the likelihood ratio notices.

## Common misconceptions

**"Hawkes is just a fancy Poisson."** Only in the mean. Every interval statistic differs: the Fano factor is 16 rather than 1 at ${n = 0.75}$, the inter-arrival distribution has a fat tail rather than an exponential one, and the count autocovariance is positive at every lag rather than zero. If you only ever look at average volume, the two models are indistinguishable and you have thrown away the reason to use either.

**"A high branching ratio means the market is unstable."** No. Stationarity holds for every ${n \lt 1}$, and ${n = 0.9}$ describes a perfectly well-behaved process, just a very clustered one. What it does mean is that the *conditional* tail is much fatter than the unconditional one, so your risk numbers are wrong in a specific direction. The thing that feels like instability is the cascade decay rate ${\beta(1 - n)}$: as ${n \to 1}$ clusters get longer and you need far more data to measure anything. That is an estimation problem, not a stability one. Related tail machinery lives in [tail risk and extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants).

**"Clustering is the same thing as autocorrelation in returns."** These are different objects and they can be independent. Clustering lives in the *arrival times*; return autocorrelation lives in the *signed price changes*. Symmetric cross-excitation, ${a = 0.35}$ and ${c = 0.40}$, gives a heavily clustered tape with an imbalance that is slightly less variable than Poisson. Trades gust, direction does not persist, and returns can be near-uncorrelated the whole time. Confusing the two is how a clustering result gets mis-sold as a forecasting result.

## Honest limits

Two things are unresolved, and saying so is part of knowing the material. First, the kernel debate: exponential kernels are tractable and power-law kernels fit the long-range correlation better, and the branching ratio you report depends on which you picked more than it depends on the year of data. Second, the linear Hawkes process rules out inhibition by construction, since ${\phi \geq 0}$. A trade cannot make the next trade *less* likely, which is a real mechanism in a thin book that has just been cleared. Non-linear variants exist; none is standard.

All dollar figures in this post are illustrative arithmetic on assumed inputs, not measurements of any market.

## Sources and further reading

- Hawkes, A. G. (1971). "Spectra of some self-exciting and mutually exciting point processes." *Biometrika* 58(1), 83-90. The original.
- Bacry, E., Mastromatteo, I., and Muzy, J.-F. (2015). "Hawkes processes in finance." *Market Microstructure and Liquidity* 1(1). [arxiv.org/abs/1502.04592](https://arxiv.org/abs/1502.04592). The survey to read first; covers the multivariate covariance structure used above.
- Filimonov, V., and Sornette, D. (2012). "Quantifying reflexivity in financial markets: toward a prediction of flash crashes." *Physical Review E* 85, 056108.
- Hardiman, S. J., Bercot, N., and Bouchaud, J.-P. (2013). "Critical reflexivity in financial markets: a Hawkes process analysis." *European Physical Journal B* 86, 442. [arxiv.org/abs/1302.1405](https://arxiv.org/abs/1302.1405).
- Ogata, Y. (1981). "On Lewis' simulation method for point processes." *IEEE Transactions on Information Theory* 27(1), 23-31, and Ogata (1988) on residual analysis. The estimation and diagnostic toolkit.
- Bacry, E., Delattre, S., Hoffmann, M., and Muzy, J.-F. (2013). "Modelling microstructure noise with mutually exciting point processes." *Quantitative Finance* 13(1), 65-77.
- Lillo, F., and Farmer, J. D. (2004). "The long memory of the efficient market." *Studies in Nonlinear Dynamics and Econometrics* 8(3).

## In the interview room and on the desk

The question arrives as "how would you model trade arrivals?", and naming the Poisson process is the easy half. Everyone gets there. The interviewer is waiting to see whether you know why it fails.

Answer in this order. One: Poisson assumes the conditional intensity is constant, so inter-arrival times are independent and exponential. Two: real tape clusters, which shows up as too many empty seconds, too many very busy seconds, and an inter-arrival distribution with a far heavier tail than exponential. Three: the smallest honest fix is to let each event lift the intensity and let the lift decay, which is a Hawkes process, ${\lambda(t) = \mu + \sum \alpha e^{-\beta(t - t_i)}}$. Four, and this is where candidates separate: give the branching interpretation. Immigrants at rate ${\mu}$, offspring with mean ${n = \alpha/\beta}$, stationary only for ${n \lt 1}$, average cluster size ${1/(1-n)}$, observed rate ${\mu/(1-n)}$. Five: say what changes in practice, which is variance, not mean. The Fano factor goes from 1 to ${1/(1-n)^2}$, so at ${n = 0.75}$ your inventory standard deviation is four times what a Poisson model told you.

If there is time, take it to the multivariate case unprompted, because that is where the actual desk question lives. The aggregate branching ratio does not determine a market maker's inventory risk; the split between ${n_{bb}}$ and ${n_{bs}}$ does, through the multiplier ${1/(1 - n_{bb} + n_{bs})}$, and the two calibrations above differ by 4.2x on identical aggregate numbers.

The trap is quoting a fitted branching ratio near 1 as a finding about market reflexivity. It sounds sophisticated and it is the exact claim Hardiman, Bercot and Bouchaud dismantled: an exponential kernel fitted over a short window pushes unmodelled long-range correlation into ${n}$, and a constant baseline fitted across a U-shaped session does it again. A candidate who says "my fit gave 0.92, and before I call that reflexivity I would refit with a power-law kernel and a time-varying baseline to see how much of it survives" has just demonstrated more than the candidate with the higher number.

Jump Trading, Citadel Securities and any market-making seat weight this most, because clustered arrivals are a direct input to quote sizing. Execution desks care for the same reason at a different horizon: the cluster that follows your child order is partly your own, which is where this connects to [dynamic programming and optimal execution](/blog/trading/math-for-quants/dynamic-programming-optimal-execution-math-for-quants).
