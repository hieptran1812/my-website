---
title: "Strategy Capacity: The Size at Which a Good Strategy Stops Being One"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Every strategy has a size where its own trading eats its edge. Capacity is where the gross alpha line and the impact cost curve cross, and because impact grows faster than linearly, the crossing is sharper than almost anyone expects."
tags: ["market-impact", "strategy-capacity", "square-root-law", "almgren-chriss", "transaction-costs", "turnover", "optimal-execution", "portfolio-construction", "quantitative-research", "quant-interviews"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 20
---

> [!important]
> **TL;DR:** Gross alpha grows in a straight line with capital while impact cost grows as capital to the power ${3/2}$, so net dollars are concave and have a maximum. That maximum is capacity, and it is a number you compute rather than a feeling you have.
>
> - On the strategy worked below, net dollars peak at \$1.23bn and reach zero at \$2.78bn. The optimum is exactly ${4/9}$ of break-even, always, and at the optimum you keep exactly one third of gross alpha. Both fractions depend only on the impact exponent.
> - The curve is asymmetric. Running 19% under the optimum costs \$0.46m a year. Running 62% over it costs \$4.34m.
> - Capacity is assets times turnover. Same 4% gross alpha, same gross Sharpe: monthly rebalancing gives \$1.23bn, weekly gives \$65.7m.
> - Two strategies sharing half their names pay 20.7% more impact than the sum of their standalone costs, so the firm number is \$1.69bn rather than \$2.47bn.
> - The number you know least enters squared. Over an impact coefficient band of 0.30 to 1.00, the same strategy's capacity is anywhere from \$309m to \$3.43bn.

## Introduction

There is a question that gets asked in every allocator meeting, every risk committee, and most senior quant interviews, and it sounds simple enough to answer in a sentence: *how much money can this strategy run?*

The weak answer is a multiple of average daily volume. Ten times ADV, say, or some house rule about never being more than 5% of a name. That answer is not exactly wrong, but it is not an estimate. It is a convention with no error bar, and the moment somebody asks where the multiple came from it falls apart.

The strong answer is a calculation. Your strategy has an edge, and that edge produces gross dollars in proportion to the capital behind it. Your strategy also has to trade, and trading moves prices against you by an amount that grows *faster* than in proportion to the capital behind it. Those two facts alone guarantee that net dollars rise, peak, and fall. The size at the peak is the capacity, and everything else in this post is about finding it and putting an honest error bar on it.

![Chart of gross alpha, impact cost and net alpha against AUM in \$bn. Gross alpha is a straight line, impact cost is convex, and net alpha peaks at \$1.23bn with \$16.5m before the two cross at \$2.78bn.](/imgs/blogs/strategy-capacity-market-impact-math-for-quants-1.webp)

That figure is the whole post in one picture. Note the thing that surprises people the first time they see it: the peak of the net curve is nowhere near the crossing point. The fund that maximises its dollar profit is running at less than half the size at which it would make nothing at all. Most of the argument inside a firm happens in the gap between those two numbers.

All the dollar figures in this post are illustrative arithmetic on clearly stated assumed inputs. The empirical claims about how impact behaves are cited.

## Foundations: why trading costs money at all

Start from the mechanics, because the whole thing rests on them.

When you want to buy, somebody has to be willing to sell to you right now. That somebody is a liquidity provider, and they are not doing it for free. They quote a **bid** (the price at which they will buy from you) and an **ask** (the price at which they will sell to you), and the gap between them is the **spread**. Crossing the spread is the first and most visible cost of trading, and for small orders it is most of the cost.

For large orders it is nearly irrelevant, because of what happens next. The liquidity sitting at the best ask is finite. Buy more than that and you take the next price level, and the next. Meanwhile every other participant watches a stream of buy orders arrive and draws the obvious conclusion: somebody knows something, or somebody has to buy, and either way the fair price is higher than they thought a minute ago. They move their quotes up. This is **market impact**: the price moves against you *because you traded*.

A few terms that the rest of the post uses constantly:

- **ADV** is average daily volume, the typical number of shares or dollars that trade in a name in a day. It is the natural yardstick for "is this order big".
- A **metaorder** is one investment decision executed as many small child orders over minutes or hours. When practitioners say "impact", they almost always mean the impact of a metaorder, not of a single fill.
- **Participation rate** is the metaorder's size divided by the ADV over the same window. A \$10m order in a name that trades \$200m a day is 5% participation.
- A **basis point**, or bp, is one hundredth of a percent. Impact costs are quoted in bps because they scale with price rather than with the dollar amount.
- **Turnover** is how many times a year the book trades its own value, one way. A strategy holding positions for a month has turnover around 12.
- **Gross alpha** is the return before trading costs. **Net alpha** is what the investor actually receives.

### Temporary versus permanent, and why the difference matters

Impact has two components that behave completely differently, and conflating them is the single most common modelling error in this area.

![Price path in basis points above the arrival price over time. Impact rises to 33 bps while the metaorder executes, then decays to a permanent level of 20 bps and stays there.](/imgs/blogs/strategy-capacity-market-impact-math-for-quants-2.webp)

**Temporary impact** is the part caused by your demand for immediacy. You emptied the book faster than market makers could refill it, and they charged you for the inconvenience. When you stop, they refill, and the price comes back. Temporary impact is a toll: you pay it, it is gone, and the next metaorder pays it again from scratch.

**Permanent impact** is the part the market keeps. It is the revision in what everyone believes the asset is worth, given that they have just watched somebody buy a lot of it. That revision does not unwind when you stop trading, because it was never a liquidity artefact in the first place. Empirical work on metaorder impact finds that a substantial fraction of the peak survives after execution ends, commonly reported in the region of a half to two thirds ([Bouchaud et al., 2018](https://doi.org/10.1017/9781316659335)), though the exact fraction depends on the market and on how long you wait before measuring it.

Here is why the distinction earns its own section. Temporary impact does not accumulate: it resets between orders. Permanent impact does, because it changes the reference price that every one of your *subsequent* trades in that name starts from. A strategy that buys the same hundred names every month is buying at a level it raised itself last month, and raised again the month before. That is the component which compounds over the life of a position and which shows up as a slow, unglamorous widening of the gap between the backtest and the live book.

One honest caveat, because it is a favourite interview follow-up. If your permanent footprint is still in the price when you eventually exit, you sell into a level you helped create and you get much of it back. The cost is real when the footprint has decayed by the time you close, or when you never close because the strategy keeps rolling the same exposure. Which of those you are in is a modelling choice you should state rather than assume.

## The square-root law

Now the empirical anchor. Across markets, asset classes, decades and half a dozen independent datasets, the impact of a metaorder scales roughly with the **square root** of participation:

$$\Delta P / P \;=\; Y \, \sigma \, \sqrt{\frac{Q}{V}},$$

where $Q$ is the metaorder size, $V$ is the volume over the execution window, $\sigma$ is the asset's volatility over that same window, and $Y$ is a dimensionless coefficient of order one.

Three things to say about that formula immediately.

**It is an empirical regularity, not a theorem.** Nothing in no-arbitrage forces it. [Tóth et al. (2011)](https://doi.org/10.1103/PhysRevX.1.021006) documented it on a large database of proprietary metaorders and argued it reflects a market that sits permanently close to a critical point, where the revealed order book is a vanishingly small fraction of latent supply and demand. [Almgren et al. (2005)](https://www.cfm.com/wp-content/uploads/2022/12/2005-direct-estimation-of-equity-market-impact.pdf) fitted temporary impact on a large brokerage dataset and got an exponent near 0.6 rather than 0.5, with permanent impact close to linear in size. [Kyle (1985)](https://doi.org/10.2307/1913210) derived a strictly linear impact from an equilibrium with a single informed trader, which is the theoretical foundation everyone still teaches and which the data does not reproduce at metaorder scale. So the exponent is somewhere around a half, defensibly, and you should never call it exact.

**The volatility scaling is the part people drop and should not.** Impact is quoted relative to how much the asset moves anyway. A 5% participation order in a quiet utility and in a biotech cost very different numbers of basis points, and the formula already knows that.

**$Y$ is where all the uncertainty lives.** It absorbs the market, the venue mix, the execution style, and one convention question that matters enormously: whether you are measuring the *peak* impact at the end of the metaorder or the *average* price you actually paid across it, which for a uniform schedule is roughly two thirds of the peak. Two desks quoting "Y equals 0.5" may not be quoting the same quantity.

![Impact in basis points against participation rate as a percent of average daily volume. The square-root law curve passes through 22.4 bps at 5 percent and 31.6 bps at 10 percent, far below the dashed linear line.](/imgs/blogs/strategy-capacity-market-impact-math-for-quants-3.webp)

#### Worked example 1: what 5% of ADV actually costs

A stock trades 5,000,000 shares a day at \$40, so its dollar ADV is \$200m. Its daily volatility is 2.0%. Take $Y = 0.5$, in line with the Tóth et al. estimate.

You want to buy \$10m of it in a day. Participation is ${10/200} = 5\%$, so

$$\Delta P / P \;=\; 0.5 \times 0.02 \times \sqrt{0.05} \;=\; 0.5 \times 0.02 \times 0.22361 \;=\; 0.0022361,$$

which is **22.36 bps**, or ${0.0022361 \times \$10{,}000{,}000 = \$22{,}361}$ on the order.

Now double the order to \$20m. Participation goes to 10%, and

$$\Delta P / P \;=\; 0.5 \times 0.02 \times \sqrt{0.10} \;=\; 0.0031623,$$

which is **31.62 bps** and ${0.0031623 \times \$20{,}000{,}000 = \$63{,}246}$.

The cost *per share* rose by only $\sqrt{2} = 1.41$, which sounds forgiving. The **total dollars** rose by ${2^{3/2} = 2.83}$, from \$22,361 to \$63,246. That asymmetry between the per-share number a trader quotes and the total-dollar number the P&L feels is the mechanical root of everything that follows.

*The intuition: size hurts you twice, once through the worse price and once through having more shares priced at it.*

### Where the square-root law breaks

Say this unprompted and you sound like someone who has used the model rather than read about it.

- **Very small orders.** Below roughly a tenth of a percent of ADV, spread and fees dominate and the square root is not the binding term.
- **Very large orders.** The law is fitted mostly on metaorders between about 0.1% and 10% of ADV. Extrapolating it to 50% of ADV is unsupported, and the real answer there is usually that you cannot trade it at any price you would like.
- **Illiquid names.** The formula needs a stable $V$ and $\sigma$. In a name whose ADV triples on news, both inputs are estimates with wide bands and the output inherits them.
- **Crowded and stressed markets.** The law is an average over normal conditions. In a deleveraging, everyone's impact is correlated and realised costs blow well past the model, which is exactly when your capacity number is being tested.

## Almgren-Chriss: the trade-off inside a single order

Before capacity, one level down. Given that you have decided to trade $Q$ shares, how fast should you do it? Trade quickly and you pay heavy temporary impact. Trade slowly and you pay less impact but you are exposed to the price wandering away from you while you wait.

[Almgren and Chriss (2000)](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf) wrote this as one objective: minimise expected cost plus a risk-aversion parameter $\lambda$ times the variance of cost. Expected cost is increasing in speed. Variance of cost is increasing in *duration*, because the unexecuted inventory is exposed to volatility for longer. Those two pull in opposite directions and the trade-off has an interior solution.

The structural result is worth carrying even if you never derive it. The optimal inventory path decays exponentially toward zero with a characteristic time set by $\sqrt{\eta / (\lambda \sigma^2)}$, where $\eta$ is the temporary impact coefficient. A risk-neutral trader ($\lambda = 0$) trades at a constant rate, which is TWAP. A risk-averse trader front-loads, and the more the price can move, the more front-loaded the schedule becomes. Every execution algorithm you will meet is a version of this dial.

The series covers the derivation and the backward induction that produces it in [dynamic programming and optimal execution](/blog/trading/math-for-quants/dynamic-programming-optimal-execution-math-for-quants), and the algorithms that implement it in [execution algorithms: VWAP, TWAP and POV](/blog/trading/quantitative-finance/execution-algorithms-vwap-twap-pov-quant-research). What matters for capacity is only this: execution optimisation changes the *constant* in front of the cost, sometimes by a lot. It does not change the exponent, and the exponent is what makes capacity finite.

## From impact to capacity

Now the central calculation.

Let $A$ be the capital in the strategy and $g$ its gross alpha per year, assumed constant in $A$. Gross dollars are ${gA}$: a straight line.

Suppose the strategy trades $N$ names and turns its book over $T$ times a year, rebalancing $T$ times, so each rebalance trades the whole book once across $N$ names. Each name therefore takes a trade of size ${q = A/N}$, into a name of dollar ADV $V_\$$. The number of trades a year is ${M = NT}$ and the annual dollar volume traded is ${D = TA}$.

Cost per dollar traded is the square-root law, so annual impact cost in dollars is

$$C(A) \;=\; T A \cdot Y \sigma \sqrt{\frac{A}{N V_\$}} \;=\; k\,A^{3/2}, \qquad k \;=\; \frac{T Y \sigma}{\sqrt{N V_\$}}.$$

That single line is the post. **Gross scales as $A$, cost scales as $A^{3/2}$.** Net dollars are

$$\Pi(A) \;=\; g A \;-\; k A^{3/2},$$

which is concave, zero at the origin, and zero again at a positive $A$. So it has an interior maximum.

Take the first-order condition:

$$\Pi'(A) \;=\; g - \tfrac{3}{2} k A^{1/2} \;=\; 0 \quad\Longrightarrow\quad A^{\ast} \;=\; \frac{4g^2}{9k^2} \;=\; \frac{4 g^2 N V_\$}{9 T^2 Y^2 \sigma^2}.$$

And the break-even size, where net alpha reaches zero, is where ${gA = kA^{3/2}}$:

$$A_0 \;=\; \frac{g^2}{k^2}.$$

Two consequences fall out immediately and neither depends on a single estimated parameter.

**The optimum sits at exactly ${4/9}$ of break-even.** ${A^{\ast}/A_0 = 4/9 = 0.4444}$, for any strategy, any market, any impact coefficient. It follows purely from the exponent ${3/2}$. For a general cost exponent ${1+\delta}$ the ratio is ${(1/(1+\delta))^{1/\delta}}$, which gives ${4/9}$ at $\delta = 1/2$ and ${1/2}$ under Kyle's linear impact.

**At the optimum you keep exactly one third of gross alpha.** Substituting back, ${\Pi(A^\ast) = \tfrac{1}{3} g A^\ast}$: impact takes two thirds. In general you keep ${\delta/(1+\delta)}$, so a fund at its dollar-maximising size under the square-root law is paying twice as much to trade as it keeps.

#### Worked example 2: the capacity of a \$4%-alpha equity strategy

Assumed inputs: gross alpha ${g = 4.0\%}$ a year, gross volatility 4.0% so gross Sharpe is 1.0, a universe of ${N = 500}$ names with median dollar ADV ${V_\$ = \$50\text{m}}$, daily volatility ${\sigma = 2.0\%}$, monthly rebalancing so ${T = 12}$, and ${Y = 0.5}$.

$$k \;=\; \frac{12 \times 0.5 \times 0.02}{\sqrt{500 \times 50{,}000{,}000}} \;=\; \frac{0.12}{158{,}113.883} \;=\; 7.58947 \times 10^{-7}.$$

Then

$$A_0 \;=\; \frac{0.04^2}{k^2} \;=\; \frac{0.0016}{5.76 \times 10^{-13}} \;=\; \$2{,}777{,}777{,}778, \qquad A^{\ast} \;=\; \tfrac{4}{9} A_0 \;=\; \$1{,}234{,}567{,}901.$$

So this strategy makes the most money at about **\$1.23bn** and makes nothing at all at **\$2.78bn**.

Check the mechanics at the optimum. Each rebalance trades ${\$1{,}234{,}567{,}901 / 500 = \$2{,}469{,}136}$ per name, which is 4.94% of a \$50m ADV. That participation is exactly ${(2/9)^2}$, so slippage is exactly ${0.01 \times 2/9 = 1/450}$, or 22.222 bps. Annual cost is ${12 \times \$1{,}234{,}567{,}901 / 450 = \$32{,}921{,}811}$ against gross of ${0.04 \times \$1{,}234{,}567{,}901 = \$49{,}382{,}716}$, leaving **\$16,460,905** net, which is one third of gross exactly as the algebra promised. Net alpha is 1.333%, not 4%.

*The intuition: the first dollar of capital keeps almost all its alpha and the last dollar keeps none, and capacity is where the marginal dollar stops paying for itself.*

### Verifying the optimum two ways

An optimum found by differentiating your own expression and an optimum found by walking your own curve are worth having both, because the first catches an arithmetic slip and the second catches a wrong derivative.

Analytically, ${A^{\ast} = 4g^2/(9k^2) = \$1{,}234{,}567{,}901.23}$.

Numerically, running a golden-section search on ${\Pi(A) = gA - kA^{3/2}}$ over ${[0, A_0]}$ with no knowledge of the closed form returns ${\$1{,}234{,}567{,}932.27}$, a relative difference of ${2.5 \times 10^{-8}}$. A brute grid of two million points over the same interval returns ${\$1{,}234{,}568{,}056}$, a relative difference of ${1.3 \times 10^{-7}}$, which is the grid spacing. Three routes, one answer.

### The curve is flat on the left and a cliff on the right

![Net dollars per year in \$m against AUM in \$bn, zoomed. The curve is flat near the \$1.23bn optimum at \$16.5m and falls away steeply above it, reaching \$12.1m at \$2.0bn.](/imgs/blogs/strategy-capacity-market-impact-math-for-quants-4.webp)

At a maximum the first derivative vanishes, so near the peak the curve is locally flat and errors are second order. Below the optimum you are on that flat shoulder. Above it you are not for long, because the cost term is accelerating.

Net dollars are \$16.00m at \$1.0bn, \$16.46m at the \$1.23bn optimum, \$15.91m at \$1.5bn and \$12.12m at \$2.0bn. So running 19% **under** the optimum costs \$0.46m a year, about 2.8% of the maximum. Running 62% **over** it costs \$4.34m, about 26.4%.

That asymmetry is the practical argument for erring small, and it is the sentence that wins the interview. The cost of being conservative is second order. The cost of being greedy is first order and compounding.

## Turnover is the multiplier everyone forgets

Look again at the closed form:

$$A^{\ast} \;=\; \frac{4 g^2 N V_\$}{9 T^2 Y^2 \sigma^2}.$$

Capacity is **quadratic in gross alpha**, **linear in how much liquidity the universe carries**, and **inverse quadratic in turnover**. That last one is the term people leave out of the answer, and it is the largest one in the expression.

The reason is not mysterious. A strategy's edge is an annual number. If it turns the book over twelve times to earn 4%, it is earning 33 bps of edge per unit of dollar volume traded. If it turns the book over fifty-two times to earn the same 4%, it is earning 7.7 bps per unit traded, and 7.7 bps does not buy much impact. The cost per trade falls as you trade smaller, but only as a square root, while the edge per trade falls linearly. The square root loses.

![Capacity in \$bn against annual turnover. Capacity falls as one over turnover squared, from \$1.23bn at monthly turnover to \$66m at weekly turnover.](/imgs/blogs/strategy-capacity-market-impact-math-for-quants-5.webp)

#### Worked example 3: the same Sharpe at two speeds

Take the strategy from example 2 and change nothing except how often it rebalances. Gross alpha stays 4.0%, gross volatility stays 4.0%, gross Sharpe stays 1.0, and the universe is identical. Only $T$ moves, from 12 to 52.

$$A^{\ast}(12) \;=\; \$1{,}234{,}567{,}901, \qquad A^{\ast}(52) \;=\; \frac{4 \times 0.0016 \times 500 \times 5\times10^{7}}{9 \times 52^2 \times 0.25 \times 0.0004} \;=\; \$65{,}746{,}220.$$

The ratio is ${(52/12)^2 = 18.78}$, and the check at the weekly speed reproduces: each rebalance trades ${\$65{,}746{,}220/500 = \$131{,}492}$ per name, which is 0.263% of ADV, costing ${0.5 \times 0.02 \times \sqrt{0.002630} = 5.13}$ bps per trade. Fifty-two of those a year is 2.667% of assets, the same two thirds of gross alpha as before, as it must be at the optimum.

Net dollars at the optimum: **\$16,460,905** a year at monthly turnover, **\$876,616** a year at weekly turnover. Same Sharpe, same alpha, same universe. Nineteen times less money.

*The intuition: capacity is not a statement about assets, it is a statement about dollars traded at a given edge per dollar traded, and turnover is the exchange rate between the two.*

One caveat a good interviewer will push on. Holding gross alpha fixed while raising turnover is an artificial comparison, because the fundamental law of active management says more independent bets should buy more information ratio. A genuinely faster strategy usually *does* have more gross alpha. The point survives anyway: it has to have quadratically more, or its capacity collapses regardless of how good its Sharpe looks.

## Capacity is a portfolio property, not a strategy property

Two strategies trading the same name at the same time in the same direction do not each pay their own impact. The market sees the sum of their flow.

If each would trade $q$ in a shared name, the combined order is ${2q}$, costing ${2q \cdot Y\sigma\sqrt{2q/V_\$}}$, which is ${2^{3/2} = 2.83}$ times what one of them would pay alone, against ${2\times}$ if they were independent. Let $\phi$ be the fraction of each strategy's names the other also trades. The pair's impact cost relative to the sum of standalone costs is

$$R(\phi) \;=\; \frac{\phi \cdot 2^{3/2} + 2(1-\phi)}{2} \;=\; 1 + \phi(\sqrt{2}-1).$$

#### Worked example 4: the firm number is not the sum

Two copies of the example-2 strategy, standalone capacity \$1,234,567,901 each, so the naive firm number is \$2,469,135,802.

At ${\phi = 0.5}$, half the names shared, ${R = 1 + 0.5 \times 0.414214 = 1.2071}$. The pair pays **20.7% more** impact than the sum of the two standalone bills. Feeding that into the optimum, the firm's combined capacity is ${2 A^{\ast}/R^2 = \$1{,}694{,}546{,}916}$, which is **\$1.69bn** against a naive \$2.47bn, a shortfall of 31.4%.

At ${\phi = 1}$, complete overlap, ${R = \sqrt{2}}$ and the pair's combined capacity collapses to ${2A^\ast/2 = \$1{,}234{,}567{,}901}$: exactly the capacity of one strategy, because two identical strategies *are* one strategy with a bigger book and an internal argument about credit.

This is why capacity lives in portfolio construction rather than in the research notebook, and why a multi-strategy firm's allocation meeting is really a capacity auction. It is also the reason a strategy's P&L attribution and its capacity have to be read together: see [P&L attribution](/blog/trading/math-for-quants/pnl-attribution-math-for-quants) for the machinery that says whose flow caused what.

## How wrong can this number be

Here is the part most capacity presentations skip, and it is the part that makes the rest credible.

$A^{\ast}$ goes as ${1/Y^2}$. And $Y$ is the parameter you know least well, because it is not observable from public data, it moves with market regime, it differs by venue and execution style, and it depends on whether the estimate is peak or average impact.

![Estimated capacity in \$bn against the impact coefficient Y. Over the reported range of 0.30 to 1.00 the estimate falls from \$3.43bn to \$309m, an eleven fold spread.](/imgs/blogs/strategy-capacity-market-impact-math-for-quants-6.webp)

Take a band from ${Y = 0.30}$ to ${Y = 1.00}$. This is not a confidence interval from any single study; it is a fair reflection of how far published estimates and desk recalibrations move across markets, sample periods and the peak-versus-average convention.

| $Y$ | capacity $A^{\ast}$ |
| --- | --- |
| 0.30 | \$3,429,355,281 |
| 0.50 | \$1,234,567,901 |
| 1.00 | \$308,641,975 |

A factor of 3.3 of doubt about the impact coefficient becomes a factor of **11.1** in the capacity estimate, because the coefficient enters squared. The point estimate \$1.23bn, quoted alone, conveys precision it does not have.

The exponent is the second source. If the true exponent is 0.6, as [Almgren et al. (2005)](https://www.cfm.com/wp-content/uploads/2022/12/2005-direct-estimation-of-equity-market-impact.pdf) estimated, the optimum sits at ${0.4569}$ of break-even rather than ${0.4444}$ and you keep 37.5% of gross rather than 33.3%. That is a small change to the shape, which is reassuring, but it changes the *level* too, and the level is what gets allocated against.

**The null result, stated plainly.** The capacity calculation does not produce a number. It produces a curve, and the curve's location is uncertain by an order of magnitude from the impact coefficient alone. What the calculation genuinely gives you, and what a multiple of ADV never will, is three things that are robust: the *shape* (net dollars peak and then fall), the *ratios* (${4/9}$ of break-even, one third of gross kept), and the *scalings* (quadratic in alpha, inverse quadratic in turnover). Those survive any plausible $Y$. Use them, and quote the level as a range.

## Common misconceptions

**"Capacity is a fraction of ADV."** ADV enters the formula, linearly, but so do alpha squared, turnover squared and volatility squared. Two strategies in the same universe with the same ADV rule can have capacities that differ by two orders of magnitude.

**"Better execution raises capacity a lot."** Better execution lowers $Y$. Capacity goes as ${1/Y^2}$, so cutting the effective coefficient by 20% raises capacity by 56%, which is real money and worth having. It does not change the exponent, so it buys you a multiple, never a reprieve.

**"If we are under break-even we are fine."** Break-even is where you make *zero*. Between ${4/9}$ of it and all of it you are running a fund that generates progressively less money on progressively more capital, and charging fees on the capital. That region is where most capacity disputes actually sit.

**"Gross alpha is constant in size."** It is the friendliest assumption in this post and it is optimistic. A strategy scaling up usually has to reach into smaller, less liquid or less attractive names, so $g$ itself declines with $A$. Every such effect makes real capacity *lower* than this calculation says.

**"We would see it in the backtest."** Only if the backtest charged the right costs, which means charging ${A^{3/2}}$ rather than a flat bps assumption. A constant cost-per-trade assumption cannot produce a capacity limit at all. That failure mode is one rung of the diagnostic ladder in [live versus backtest divergence](/blog/trading/math-for-quants/live-vs-backtest-divergence-math-for-quants).

## The organisational reality

Capacity estimates are contested for a reason that has nothing to do with the mathematics: the number caps assets, assets cap fees, and fees cap compensation. A portfolio manager arguing for a higher capacity is arguing for their own book. A risk officer arguing for a lower one is arguing for the firm's. Both are being rational and neither is being dishonest.

This is exactly why a senior owns the number rather than delegating it. The defensible position is never "capacity is \$1.23bn". It is: here is the impact model, here is the coefficient it assumes, here is where that coefficient came from, here is the band it moves over, here is what the capacity estimate does across that band, and here is the realised-cost series we are monitoring to update it. A capacity number without its impact coefficient attached is not a forecast, it is a negotiating position.

The monitoring half matters as much as the estimate. You recalibrate $Y$ from your own fills, comparing realised slippage against the model at the participation rates you actually traded, and you watch it drift. A strategy's short-horizon alpha and its execution costs are measured on the same data, which is why the signal work in [order book imbalance](/blog/trading/math-for-quants/order-book-imbalance-short-horizon-prediction-math-for-quants) and the cost work here belong to the same person.

## Sources and further reading

- Kyle, A. S. (1985), "Continuous Auctions and Insider Trading", *Econometrica* 53(6), 1315-1335. The linear-impact equilibrium foundation.
- Almgren, R. and Chriss, N. (2000), "Optimal Execution of Portfolio Transactions", *Journal of Risk* 3(2), 5-39. The impact-versus-timing-risk trade-off and the exponential trajectory.
- Almgren, R., Thum, C., Hauptmann, E. and Li, H. (2005), "Direct Estimation of Equity Market Impact", *Risk* 18(7), 58-62. Empirical impact estimates with a temporary exponent near 0.6.
- Tóth, B., Lempérière, Y., Deremble, C., de Lataillade, J., Kockelkoren, J. and Bouchaud, J.-P. (2011), "Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets", *Physical Review X* 1, 021006. The square-root law and the latent-liquidity argument behind it.
- Gatheral, J. (2010), "No-Dynamic-Arbitrage and Market Impact", *Quantitative Finance* 10(7), 749-759. Why impact and its decay cannot be chosen independently.
- Bouchaud, J.-P., Bonart, J., Donier, J. and Gould, M. (2018), *Trades, Quotes and Prices: Financial Markets Under the Microscope*, Cambridge University Press. Book-length treatment of impact, including metaorder decay.
- Grinold, R. C. and Kahn, R. N. (1999), *Active Portfolio Management*, 2nd edition, McGraw-Hill. The fundamental law that ties breadth to information ratio.

All dollar figures in the worked examples are illustrative arithmetic on the assumed inputs stated alongside them.

## In the interview room and on the desk

The question arrives as *"how much money do you think this strategy can run?"*, and it is almost never a question about your strategy. It is a question about whether you can reason about cost at scale without hand-waving.

The weak answer is a multiple of ADV, or a number with no derivation behind it. It sounds like a house rule because it is one, and the immediate follow-up, "where does the multiple come from?", has nowhere to go.

The strong answer has four moves and takes about ninety seconds. **Name the impact model**: the square-root law, impact proportional to ${Y\sigma\sqrt{Q/V}}$, empirically anchored by Tóth et al. and Almgren et al., an empirical regularity rather than a theorem. **Set up the crossing**: gross alpha is linear in capital, impact cost goes as capital to the ${3/2}$, so net dollars are concave with an interior maximum at ${4/9}$ of the break-even size, where you keep one third of gross. **Say that capacity is assets times turnover**, and that at fixed gross alpha capacity falls as one over turnover squared, so a weekly strategy and a monthly strategy with identical Sharpes are not remotely the same business. **Give an error bar**: capacity goes as ${1/Y^2}$, so a factor of three of doubt in the impact coefficient is a factor of ten in the answer, and the honest output is a range plus the recalibration process that will tighten it.

If there is time, add the portfolio point: two strategies overlapping in the same names share one capacity budget, and the firm-level number is materially below the sum of the standalone numbers. That is the answer that sounds like someone who has sat in the allocation meeting.

The trap is quoting a capacity number without saying what impact coefficient it assumes. It is a trap precisely because it looks rigorous: you have done a real calculation and produced a specific figure, and the figure is the most fragile object in the room. The second trap is the mirror image, refusing to give a number at all on the grounds that it is uncertain. Both fail. What passes is a range, the assumption that generates it, and the sentence about which direction you would rather be wrong in, which is small, because the capacity curve is flat below the optimum and a cliff above it.

Citadel, Two Sigma and WorldQuant all weight this heavily, as does any seat that faces allocators, since capacity is the first question a large investor asks and the last one a fund wants to answer precisely. On the desk it shows up less as a set-piece question and more as a standing obligation: somebody has to own the number, defend it when it caps the book, and revise it when the fills say it was wrong.
