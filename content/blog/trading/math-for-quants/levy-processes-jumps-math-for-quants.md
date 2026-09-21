---
title: "Levy processes and jumps: what Brownian motion cannot do"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "Brownian motion has continuous paths, so in principle a hedge can be rebalanced through any move. Real prices gap. Admit jumps and three things break together: the market stops being complete, the hedge stops being unique, and the volatility smile stops being a puzzle."
tags: ["levy-processes", "jump-diffusion", "merton-model", "variance-gamma", "cgmy", "volatility-smile", "incomplete-markets", "gap-risk", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 19
---

> [!important]
> **TL;DR:** Continuity is the assumption that makes a hedge work. Prices gap, and once you write the gap into the model the unique price, the unique hedge and the flat volatility surface all go at once.
>
> - The **Levy-Khintchine** theorem says a process with stationary independent increments is a drift, a Brownian part and a jump measure, and nothing else. The modelling choice is entirely a choice of jump measure.
> - **Finite activity** (Merton) says jumps are rare events on top of a diffusion. **Infinite activity** (variance gamma, CGMY) says there is no smallest jump: at a ${0.1\%}$ cutoff the variance gamma example counts ${35.86}$ jumps a year, at ${0.01\%}$ it counts ${58.74}$, and the variance stays finite at ${0.0445}$.
> - Merton's price is a Poisson mixture of Black-Scholes prices. On a one-month \$85 put with spot \$100, the no-jump term is **${0.81\%}$** of the value and the single-jump term is **${88.34\%}$**.
> - Match the *total variance* and Black-Scholes still misprices that put by **\$96,200** on a \$50m book, because variance is a number and the smile is a shape.
> - Jumps cannot be hedged by rebalancing faster. A ${12\%}$ overnight gap costs the delta-hedged short book **\$316,250**, which is ${2.70}$ times the entire premium it collected.

## The assumption that is doing all the work

Sell an option, hedge it, and the textbook tells you what to do: hold $\partial V/\partial S$ shares, rebalance as the stock moves, and the payoff replicates. What makes that a theorem rather than a heuristic is a property of the path so quiet it is easy to miss. A Brownian path is **continuous**. To get from \$100 to \$88 it must pass through \$97, \$94, \$91 and every level in between, in order, and a hedger who is paying attention can in principle trade at each of them.

Real prices do not do this. A stock closes at \$100 and opens at \$88. A central bank moves at 14:00 and the level that existed at 13:59:59 is never seen again. Between those two prints there is no market, so there is no rehedge, and the hedge you put on at \$100 is the hedge you carry into \$88.

![Two panels. Left: a continuous price path from one hundred dollars down to eighty-eight with rehedge marks at ninety-seven, ninety-four and ninety-one, labelled every level is visited and every level is tradable. Right: the same path breaking vertically from one hundred to eighty-eight between a close and an open, with the gap span labelled no trade possible in here](/imgs/blogs/levy-processes-jumps-math-for-quants-1.webp)

That figure is the whole article in two panels. Everything Black-Scholes delivers, the unique price, the unique hedge, the single volatility number, is bought with the left panel. Take the right panel seriously and the bill is high: the market becomes incomplete, the hedge becomes a choice, and the smile stops being an embarrassment and becomes the obvious output of the model.

## Foundations: what you need first

**Brownian motion** $W_t$ ([built from scratch here](/blog/trading/math-for-quants/brownian-motion-random-walk-math-for-quants)) has continuous paths and independent Gaussian increments. Under geometric Brownian motion $\log S$ is Brownian motion with drift, and under the risk-neutral measure that drift is $r - \sigma^2/2$ ([derived here](/blog/trading/math-for-quants/sdes-gbm-ou-cir-math-for-quants)). A **European put** with strike $K$ pays $\max(K - S_T, 0)$, so it is insurance against a fall, and **implied volatility** is the single $\sigma$ you must feed [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) to reproduce a quoted price. A **Poisson process** $N_t$ with intensity $\lambda$ counts events arriving independently at a constant rate, so the count over a window of length $T$ is Poisson with mean $\lambda T$.

A **Levy process** is the general object: $X_0 = 0$, increments over disjoint intervals are independent, the law of $X_{t+s} - X_t$ depends only on $s$, and the paths are right-continuous with left limits. That is it. Brownian motion is one, the Poisson process is one, and so is any sum of the two. The definition says nothing about continuity, and that omission is the entire subject.

Throughout, one illustrative scenario: spot \$100, risk-free rate ${4\%}$, diffusion volatility $\sigma = {20\%}$, jumps arriving at $\lambda = 1$ per year with log-size drawn from $\mathcal{N}(-0.10,\, 0.15^2)$, a one-month option, and a position of 500,000 shares, which is \$50m of notional. Every dollar figure below is arithmetic on those assumed inputs, not a market quote.

## Levy-Khintchine: three ingredients and nothing else

For any Levy process $X_t$ there exist a number $b$, a number $\sigma^2 \ge 0$ and a measure $\nu$ such that

$$
\mathbb{E}\!\left[e^{iuX_t}\right] = \exp\left( t\left[\, iub - \tfrac{1}{2}\sigma^2u^2 + \int_{\mathbb{R}}\left(e^{iux} - 1 - iux\mathbf{1}_{\{|x| \le 1\}}\right)\nu(\mathrm{d}x) \right] \right).
$$

Read it as three terms, because that is all it is. The $iub$ term is a **deterministic drift**. The $-\tfrac{1}{2}\sigma^2u^2$ term is a **Brownian motion**, the same continuous wiggle Black-Scholes uses. The integral is the **jump part**, and $\nu$, the **Levy measure**, says how often jumps of each size arrive: $\nu(A)$ is the expected number of jumps per unit time with size in $A$. The only condition is $\int \min(1, x^2)\,\nu(\mathrm{d}x) \lt \infty$, which permits infinitely many tiny jumps but not infinitely many large ones, and subtracting $iux\mathbf{1}_{\{|x|\le1\}}$ is the bookkeeping that keeps it convergent when small jumps are infinitely numerous.

![Three stacked blocks labelled drift b times t, Brownian sigma times W of t, and jumps nu of dx, each pointing at its component of the characteristic triplet, with a closing bar reading no fourth ingredient exists](/imgs/blogs/levy-processes-jumps-math-for-quants-2.webp)

The force of the theorem is the word *only*. The triplet $(b, \sigma^2, \nu)$ determines the process completely, and no fourth ingredient is waiting to be discovered. Write down any model of returns with stationary independent increments and you have written down a drift, a Brownian part and a jump measure, whether or not you meant to. So the modelling question is never "should I add jumps", it is "what is my $\nu$", and setting $\nu = 0$ is an answer rather than an abstention.

## The jump measure is where the modelling lives

Two families split on one question: how many jumps are there?

**Finite activity.** If $\nu(\mathbb{R}) = \lambda \lt \infty$, jumps arrive as a Poisson process at rate $\lambda$ and the process is a diffusion with occasional gaps. Merton's model takes lognormal jump sizes; Kou's takes a double-exponential distribution to get asymmetric tails. The assumption is explicit: **small moves are diffusion, and only rare large moves are jumps.**

**Infinite activity.** If $\nu(\mathbb{R}) = \infty$, every interval of time contains infinitely many jumps, almost all microscopic. The **variance gamma** process (Madan, Carr and Chang) has Levy density

$$
\nu(x) = \frac{1}{\kappa|x|}\exp\!\left(\frac{\theta x}{\sigma^2}\right)\exp\!\left(-\frac{\sqrt{2/\kappa + \theta^2/\sigma^2}}{\sigma}|x|\right),
$$

which blows up like ${1/|x|}$ at the origin, so the integral diverges. **CGMY** generalises it with

$$
\nu(x) = C\,\frac{e^{-Gx}}{x^{1+Y}}\mathbf{1}_{\{x \gt 0\}} + C\,\frac{e^{-M|x|}}{|x|^{1+Y}}\mathbf{1}_{\{x \lt 0\}},
$$

where $Y$ is the dial: $Y \lt 0$ gives finite activity, ${0 \le Y \lt 1}$ infinite activity with finite variation, and ${1 \le Y \lt 2}$ infinite variation, at which point the jumps alone do the work a Brownian motion used to do.

![Table comparing expected jumps per year above a cutoff for Merton with lambda one, constant at 1.00 in every row, against variance gamma rising from 3.26 above five percent to 58.74 above one hundredth of a percent, with a footer noting the integral of x squared against nu stays finite at 0.0445](/imgs/blogs/levy-processes-jumps-math-for-quants-3.webp)

The table makes the distinction concrete for $\sigma = 0.20$, $\kappa = 0.20$, $\theta = -0.15$. Merton counts exactly ${1.00}$ jump a year no matter how fine you cut, because below the smallest drawn jump there is nothing. Variance gamma counts ${3.26}$ a year above ${5\%}$, ${14.23}$ above ${1\%}$, ${35.86}$ above ${0.1\%}$ and ${58.74}$ above ${0.01\%}$, growing without bound, roughly logarithmically in the cutoff. Yet $\int x^2 \nu(\mathrm{d}x) = 0.0445$ per year, finite, matching the closed form $\sigma^2 + \theta^2\kappa = 0.04 + 0.0225 \times 0.20$ to fifteen decimals. Infinite activity does not mean infinite risk. It means the model has no level below which it stops calling a move a jump, a claim about the *microstructure* of the path rather than its variance. The choice bites at short horizons: finite activity says a five-minute return is essentially Gaussian, and tick data disagrees.

## Merton's model: a Poisson mixture of Black-Scholes

Merton's 1976 construction works by conditioning. Write the risk-neutral dynamics as geometric Brownian motion multiplied by jumps whose log-sizes are drawn from $\mathcal{N}(a, b^2)$, then condition on exactly $n$ jumps having occurred by expiry. Now $\log S_T$ is a sum of a Gaussian and $n$ independent Gaussians, so it is *still Gaussian*, and the conditional price is an ordinary Black-Scholes price with adjusted inputs. Uncondition and you get a Poisson-weighted average:

$$
V = \sum_{n=0}^{\infty} \frac{e^{-\lambda' T}(\lambda' T)^n}{n!}\; \mathrm{BS}\big(S, K, T, r_n, \sigma_n\big),
$$

$$
\sigma_n^2 = \sigma^2 + \frac{nb^2}{T}, \qquad r_n = r - \lambda\bar{k} + \frac{n\ln(1+\bar{k})}{T}, \qquad \bar{k} = e^{a + b^2/2} - 1, \qquad \lambda' = \lambda(1 + \bar{k}).
$$

One honest note on the weights. They are Poisson probabilities, but with intensity $\lambda' = \lambda(1+\bar{k})$, not $\lambda$: the compensator that keeps the stock's expected return equal to $r$ has been absorbed into them. With $a = -0.10$ and $b = 0.15$, $a + b^2/2 = -0.08875$, so $\bar{k} = e^{-0.08875} - 1 = -0.084926$ and $\lambda' = 0.915074$. The *physical* probability of at least one jump in a month is a different number, $1 - e^{-1/12} = {7.996\%}$.

#### Worked example 1: a one-month \$85 put, term by term

Spot \$100, strike \$85, $T = {1/12}$, $\sigma = {20\%}$, $r = {4\%}$, $\lambda = 1$, jump log-size $\mathcal{N}(-0.10, 0.15^2)$. Then $\lambda' T = 0.915074/12 = 0.076256$, so $w_0 = e^{-0.076256} = 0.926579$, $w_1 = 0.076256 \times 0.926579 = 0.070657$, and each later weight is the previous one times $\lambda' T/n$.

The adjusted inputs for one jump: $\sigma_1 = \sqrt{0.04 + 0.0225 \times 12} = \sqrt{0.31} = {55.68\%}$, and $r_1 = 0.124926 + (-0.08875 \times 12) = 0.124926 - 1.065 = -0.940074$. That $r_1$ is not an interest rate, it is the conditional drift once you know a jump landed, and a single ${-8.875\%}$ log move annualised over one month is exactly a ${-94\%}$ drift.

![Table of the Merton price as a sum over jump counts, showing weights, adjusted volatilities, Black-Scholes put values and contributions for zero through four jumps, totalling twenty-three and a half cents, with the no-jump term at 0.81 percent and the one-jump term at 88.34 percent of the price](/imgs/blogs/levy-processes-jumps-math-for-quants-4.webp)

The table above is the computation. The no-jump term holds ${92.66\%}$ of the weight and a Black-Scholes value of \$0.002057, because a ${15\%}$ fall in one month at ${20\%}$ volatility is a 2.81-sigma event. It contributes \$0.00191, which is ${0.81\%}$ of the price. The single-jump term holds only ${7.07\%}$ of the weight but a Black-Scholes value of \$2.931854, and contributes \$0.20716, or ${88.34\%}$. The series then collapses: \$0.02426, \$0.00114, \$0.00003. The total is **\$0.23450 a share**, or \$117,250 on 500,000 shares.

The lesson is where that number came from. An out-of-the-money put under this model is almost entirely a bet on the jump, and the diffusion, which is all Black-Scholes has, contributes less than a cent on the dollar.

**Two checks that the formula is right, not just the arithmetic.** Re-deriving my own series would only reproduce my own mistake, so both checks come from outside it. First, set $\lambda = 0$. Every term but $n = 0$ vanishes, $r_0$ collapses to $r$ and $\sigma_0$ to $\sigma$, and the series must return Black-Scholes exactly. At 30 significant figures it does, giving ${1.36543070816778566250258557135}$ for the \$90-strike six-month put by both routes, with a difference of exactly zero. Second, and more searching, price the same contract by **Fourier inversion of the Levy-Khintchine exponent**: build $\psi(u)$ from the triplet, form $\mathbb{E}[e^{iu\log S_T}] = \exp(iu\log S_0 + T\psi(u))$, and recover the price by Gil-Pelaez inversion. That derivation never conditions on a jump count and never sees a Black-Scholes formula. It returns ${0.23449590406742}$ against the series' ${0.23449590406742}$, agreeing to better than $10^{-39}$. Two routes with nothing in common but the model, agreeing to forty digits, is evidence the formula is right. One route checked against itself is not.

## Same variance, different price

The natural objection is that jumps just add variance, and Black-Scholes has a knob for variance. So take the knob away: give Black-Scholes the *same total variance* and see whether it reproduces the price. Under the jump model the variance rate of log returns is $\sigma^2 + \lambda(a^2 + b^2) = 0.04 + (0.01 + 0.0225) = 0.0725$, so the matched flat volatility is $\sqrt{0.0725} = {26.93\%}$.

#### Worked example 2: the jump premium on a \$50m book

Price the same \$85 put with Black-Scholes at ${26.93\%}$ and it comes to **\$0.0421 a share**, against **\$0.2345** under the jump model, a ratio of ${5.57}$. On 500,000 shares the two valuations are \$21,050 and \$117,250. The gap is \$0.1924 a share, or **\$96,200**.

Same variance, different price by a factor of five and a half. Variance is a single number and the payoff is not linear in the outcome: a put ${15\%}$ out of the money pays only when the return lands in a specific region, and the jump model puts far more mass there than a Gaussian of equal variance does. Moving mass from the middle of a distribution to its tail leaves the variance untouched and reprices every wing contract.

Which is why the comparison runs the other way at the money. At the ${\$100}$ strike the flat-variance Black-Scholes price is *higher* than the jump price, because a Gaussian at ${26.93\%}$ has fatter shoulders than a jump distribution that concentrates its excess in the tails. Jumps do not raise all prices. They redistribute them.

## Why the smile stops being a puzzle

Now do the thing every desk does: take the jump model's prices and ask what single Black-Scholes volatility reproduces each one.

![Line chart of implied volatility against strike, showing the Merton curve falling from 41.61 percent at the eighty strike through 23.61 at the hundred strike and turning up to 23.08 at one hundred and ten, against a flat line at 26.93 percent labelled same total variance](/imgs/blogs/levy-processes-jumps-math-for-quants-5.webp)

#### Worked example 3: the smile, in vol points and in dollars

Inverting the jump prices strike by strike gives ${41.61\%}$ at the \$80 strike, ${36.34\%}$ at \$85, ${30.52\%}$ at \$90, ${25.91\%}$ at \$95, ${23.61\%}$ at the money, ${22.83\%}$ at \$105 and ${23.08\%}$ at \$110. The flat matched-variance volatility is ${26.93\%}$, a horizontal line through the middle of that.

That shape is a skew with an upturned right wing, and it is not fitted to anything. Nobody supplied a smile. A jump measure with a negative mean was assumed, a pricing integral was evaluated, and the smile fell out.

In dollars: at the \$80 strike the jump model prices the put at \$0.1229 a share against \$0.0036 from flat-variance Black-Scholes. On 500,000 shares that is \$61,450 against \$1,800, a difference of **\$59,650** on a contract Black-Scholes says is worth almost nothing. The ${14.68}$ volatility points between curve and line at that strike are the dollar price of crash insurance.

The asymmetry has a cause you can point at. The jump mean is negative, so downside jumps are both likelier and larger, and the contracts that pay on downside jumps are out-of-the-money puts. Symmetric jumps give a symmetric smile, which is roughly what [FX markets quote](/blog/trading/forex/fx-options-and-the-volatility-smile) and roughly not what equity markets do. The [equity skew](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more) is the jump measure's asymmetry, read back through a formula with no slot for it. The model also predicts the smile's own shape across maturities: jump effects scale with the *number* of jumps, so one jump dominates a short horizon while many average toward a Gaussian over a long one, which is why every [volatility surface](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) is steepest at the front.

## Incompleteness: the hedge stops being unique

The pricing above quietly did something the Black-Scholes derivation never has to. It *chose* a risk-neutral measure.

In a diffusion world you do not choose. [The martingale representation theorem](/blog/trading/math-for-quants/martingale-representation-hedging-math-for-quants) says that when the traded assets span the filtration, every claim is a stochastic integral against them, so every claim is replicable, so the risk-neutral measure is unique and the price is whatever the replicating portfolio costs. Completeness is not a regularity condition there, it is the load-bearing assumption.

Jumps break the span. The stock gives you one instrument, and the uncertainty now has two distinguishable sources: where the diffusion goes, and whether a jump lands and how big it is. If jump sizes are drawn from a continuous distribution there are infinitely many states to hedge and one instrument to hedge them with. Formally, [Girsanov's theorem](/blog/trading/math-for-quants/girsanov-change-of-measure-math-for-quants) for jump processes lets you rescale the jump intensity and reshape the jump-size distribution as well as shift the drift, and a whole family of those changes leaves the discounted stock a martingale. Every member of that family is an arbitrage-free price.

Three consequences follow, and they are really one. The **price becomes a band**, not a point, bounded by super-replication and sub-replication costs that for jump models are famously wide, so picking a number inside it is a judgement about risk premia rather than a derivation. The **hedge becomes an objective**, not an identity: you cannot replicate, so you minimise something instead, and quadratic hedging and utility indifference pricing give different ratios, all defensible. And **calibration takes over from derivation**, because a desk pins the measure by fitting liquid quoted options, which lets the market choose the risk premium for you.

That last move is honest, but notice the cost. The smile becomes an *input*, and the model's job is interpolating to the exotics nobody quotes. A barrier priced off a calibrated jump model and off a calibrated stochastic-volatility model that fits the same vanilla surface will disagree, and neither is wrong. That is why [barrier pricing](/blog/trading/math-for-quants/local-time-barriers-reflection-math-for-quants) is a model-risk conversation rather than a formula lookup.

## Gap risk is what kills the book

The theory has a blunt operational form. You are short options, you are delta hedged, and you are fine until the night you are not.

![Line chart of net profit and loss after the delta hedge against the size of an overnight move, rising slightly for a one and three percent fall then turning down sharply through minus twenty-three thousand at six percent to minus three hundred and sixteen thousand at twelve percent](/imgs/blogs/levy-processes-jumps-math-for-quants-6.webp)

#### Worked example 4: one night, one gap, \$316,250

Sell 500,000 of the one-month \$85 puts at \$0.2345 and collect \$117,250. The model delta is $-0.0217$ a share, so the book is long ${10{,}850}$ shares of delta and the hedge is a short of ${10{,}850}$ shares.

A ${3\%}$ down day, the ordinary kind: the put marks at \$0.2957, so the option leg loses \$147,850 minus \$117,250, which is \$30,600, and the short stock makes ${10{,}850 \times \$3 = \$32{,}550}$. Net **plus \$1,950**. The hedge did its job.

Now the gap. The stock closes at \$100 and opens at \$88. The put marks at \$1.1274, so the option leg loses \$563,700 minus \$117,250, which is \$446,450, and the short stock makes only ${10{,}850 \times \$12 = \$130{,}200}$. Net **minus \$316,250**, which is ${2.70}$ times the entire premium the book collected. At roughly \$5,985 a day of decay the position needs about 53 days of theta to earn that back, on a contract with 21 trading days left to live.

The shape of the curve is the point. At ${-1\%}$ and ${-3\%}$ the hedge slightly overshoots and the book is up. At ${-6\%}$ it is down \$23,150, at ${-12\%}$ down \$316,250, and at ${-20\%}$ down \$2,297,900. The loss grows far faster than the move, because the delta hedge is a straight line and the option is curved, and a gap collects the whole curvature in one print.

Here is the part with no fix inside the hedging framework. Had that ${12\%}$ arrived as a continuous slide over a week, you would have rehedged at \$97, \$94, \$91 and \$89, each rehedge shaving the error and the accumulated cost showing up as the gamma bleed that theta exists to pay for. Against a gap, rehedging hourly or every minute changes nothing at all: there is no print between \$100 and \$88 at which to trade. Hedging frequency buys accuracy inside a continuous path and nothing across a discontinuity, which is not a limitation of the technique but the definition of the thing.

## Common misconceptions

**"Jumps are just fat tails."** Fat tails are a property of the distribution at one horizon; jumps are a property of the path. A continuous-path model with fat marginal tails is still *complete*, still uniquely hedgeable, and still lets you rebalance through every level. Jumps remove that. The distinction is invisible in a histogram of returns and decides whether replication exists.

**"You can hedge jumps by rebalancing more often."** Continuous rebalancing is what makes delta hedging exact in a diffusion, so the instinct is right about the mechanism and wrong about the case. Halving the interval halves the error against a continuous path and does nothing against a discontinuity: worked example 4 puts the un-hedged remainder at \$316,250 at any rebalancing frequency you like. The real defences are other convex instruments, a further-out-of-the-money put or a calendar spread.

**"The smile is a Black-Scholes bug."** The smile is the market quoting prices in the only unit the screen has. Prices are not wrong; the *translation* into a single volatility is lossy, and the smile is the residue. Worked example 3 generated a full smile from a model that was never shown one.

**"Infinite activity means infinite risk."** The variance gamma example has infinitely many jumps a year and a variance of ${0.0445}$, below a ${22\%}$ annual volatility. Activity counts jumps; risk integrates their squares.

## Sources and further reading

- Merton, R. C. (1976). "Option pricing when underlying stock returns are discontinuous." *Journal of Financial Economics* 3(1-2), 125-144. The conditioning argument and the Poisson-mixture formula used above.
- Cont, R. and Tankov, P. (2004). *Financial Modelling with Jump Processes*. Chapman and Hall/CRC. Levy-Khintchine and the activity classification in Chapter 3, incompleteness and quadratic hedging in Chapters 10 and 11.
- Carr, P., Geman, H., Madan, D. and Yor, M. (2002). "The fine structure of asset returns: an empirical investigation." *Journal of Business* 75(2), 305-332. The CGMY model and the empirical case for infinite activity with finite variation.
- Madan, D., Carr, P. and Chang, E. (1998). "The variance gamma process and option pricing." *European Finance Review* 2(1), 79-105.
- Kou, S. G. (2002). "A jump-diffusion model for option pricing." *Management Science* 48(8), 1086-1101. The double-exponential alternative to lognormal jump sizes.
- For the completeness half of the argument, this series' post on [martingale representation and hedging](/blog/trading/math-for-quants/martingale-representation-hedging-math-for-quants); for calibration and the stochastic-volatility half, [jump-diffusion and stochastic volatility](/blog/trading/math-for-quants/jump-diffusion-stochastic-volatility-math-for-quants).

All dollar figures in this post are illustrative arithmetic on the assumed inputs stated in the foundations section. They are not quotes and not calibrated to any market.

## In the interview room and on the desk

The question usually arrives in its innocent form: *why is there a volatility smile?* Sometimes as "why do out-of-the-money puts trade at a higher implied vol than at-the-money", which is the same question with the answer half given away.

The weak answer is supply and demand: institutions buy crash protection, dealers charge for it. That is true and it is not an answer, because it explains a price without explaining why the *model* could not have produced it.

The strong answer is structural and runs in four steps. Black-Scholes assumes a continuous path, which is what makes delta hedging a replication rather than an approximation. Real prices gap, and Levy-Khintchine says any process with stationary independent increments is a drift, a diffusion and a jump measure, so admitting gaps means specifying a jump measure. A jump measure with a negative mean puts extra mass in the left tail, exactly where out-of-the-money puts pay, so their prices and their implied volatilities rise: the smile is the model's output, not a correction applied to it. And the same jumps break replication, so the price depends on a risk premium the market chooses rather than falling out of a hedging argument. Say that last step unprompted and the conversation changes, because it shows you know the smile and the incompleteness are one fact rather than two.

The trap is the confident claim that jumps can be hedged away by rebalancing more often. It sounds rigorous, it uses the right vocabulary, and it is exactly backwards: continuity is what fine rebalancing buys, and a jump is the removal of continuity. A candidate who says it has memorised the delta-hedging recipe without noticing the assumption underneath. The recovery is to name the un-hedgeable remainder and size it, as worked example 4 does at \$316,250 on a \$50m book, then name the real defences: another convex instrument, a calendar spread, or a position limit.

Jane Street, Citadel Securities, Optiver and IMC weight this heavily for options seats, and any desk quoting exotics will push on the incompleteness follow-up, because that is where model risk becomes a trading decision.
