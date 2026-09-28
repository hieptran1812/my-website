---
title: "Risk budgeting and drawdown control: deciding who gets to lose money"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "A multi-strategy firm does not allocate capital, it allocates risk, and risk does not add up the way capital does. Here is the arithmetic of risk contributions, why equal risk is an allocation rule rather than an optimality claim, and why the drawdown stop sitting on top is a completely different problem."
tags: ["risk-budgeting", "risk-parity", "risk-contribution", "drawdown-control", "volatility-targeting", "expected-shortfall", "portfolio-construction", "multi-strategy", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 23
---

> [!important]
> **TL;DR:** Allocating capital is the wrong frame. A multi-strategy firm allocates risk, and risk combines through a covariance matrix rather than by addition, so the two arithmetics give different answers.
>
> - Split \$500m equally across three books and the risk split is **11.34% / 29.38% / 59.28%**. One book holding a third of the capital carries nearly sixty percent of the firm's risk.
> - Risk contributions sum **exactly** to portfolio volatility, by Euler's theorem. On that portfolio, 1.15% + 2.99% + 6.03% = 10.17%. That additivity is the only reason the phrase "risk budget" means anything.
> - Equal risk contribution is an **allocation rule, not an optimum**. It equals the maximum-Sharpe portfolio only when every book has the same Sharpe and every pair the same correlation. When Sharpe rises with volatility across your books, risk parity is **\$3.64m a year worse** than the naive equal-capital split it claims to improve on.
> - A volatility target controls the second moment and says nothing about the path. On a \$200m book at Sharpe 1.0, a 10% trailing drawdown stop fires in **42.9%** of years even with no deterioration at all, costs **\$4.76m a year** of a \$23.98m expectation, and when it fires the book would have finished higher without it **73.5%** of the time.
> - The stop costs you most on exactly the books you least want to stop: at Sharpe 2.0 it costs \$4.04m, at Sharpe 0.0 it costs nothing.

You are in the annual risk meeting. Five portfolio managers, one number to divide. The CIO says "let's keep it simple, everyone gets the same allocation," someone writes five equal numbers on the whiteboard, and the meeting moves on feeling fair.

It is not fair, and it is not simple. The thing being divided is not what anyone thinks it is. A dollar handed to a credit book and a dollar handed to a stat arb book do not buy the same amount of anything. They buy wildly different amounts of risk, and risk is the scarce resource, the one the firm's mandate is written in and the one that runs out first.

That figure is the mental model for the whole post: the same three books, the same \$500m, and two different pictures depending on whether you draw the capital or the risk.

![Two stacked bars comparing an equal capital split of 500 million dollars in thirds against the resulting risk split of 11.34, 29.38 and 59.28 percent, with the credit carry book taking nearly sixty percent of the risk](/imgs/blogs/risk-budgeting-drawdown-control-math-for-quants-1.webp)

Everything below is illustrative arithmetic on assumed inputs. The covariance matrix, the Sharpes and the book sizes are stipulated so the mechanics are checkable, not drawn from any firm's actual book.

## Foundations: what a risk contribution actually is

Three books, carried through the whole post. A **stat arb** book with 8% annualised volatility, a **macro** book at 12%, and a **credit carry** book at 20%. *Volatility* here means the standard deviation of the book's annual return: a 12% vol book has returns that typically land within about 12% either side of their average. Correlations: stat arb to macro 0.10, stat arb to credit 0.20, macro to credit 0.50. Firm capital \$500m.

Write $w_i$ for the fraction of firm capital in book $i$ and $\Sigma$ for the covariance matrix of the books' returns. Portfolio volatility is

$$\sigma_p(w) = \sqrt{w^\top \Sigma w}.$$

The first thing to notice is that this is not a sum. You cannot ask "how much of the 10% is the credit book" the way you can ask how much of the \$500m it holds, because the square root has already blended everything.

The trick that rescues the question is to ask a *marginal* one instead. If I give book $i$ one more dollar, how much does firm volatility go up? That is a derivative, and it has a name, the **marginal contribution to risk**:

$$\mathrm{MCR}_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}.$$

Multiply the marginal rate by how much of the book you actually hold and you get the **risk contribution**:

$$\mathrm{RC}_i = w_i \cdot \mathrm{MCR}_i = \frac{w_i (\Sigma w)_i}{\sigma_p}.$$

So far this is a definition, and definitions are cheap. What makes this one worth having is that $\sigma_p$ is *homogeneous of degree one* in $w$: double every weight and you double the volatility. Euler's theorem on homogeneous functions then says the parts add up:

$$\sum_{i=1}^{n} w_i \frac{\partial \sigma_p}{\partial w_i} = \sigma_p.$$

That is the whole engine. The contributions sum exactly to the portfolio's own volatility, nothing left over and nothing double-counted. A quantity that adds up is a quantity you can divide into shares, hand out, and hold people to. Without Euler's theorem "risk budget" would be a metaphor. With it, it is arithmetic.

Note what the contribution is *not*. It is not the book's standalone risk. A book can be volatile alone and contribute almost nothing if it hedges the rest, or be quiet alone and contribute a lot if it leans the same way as everything else. The contribution is a property of the book **inside this portfolio**, and it moves when any other book moves.

## Worked example 1: the risk inside a three-book firm

Take the equal-capital split, $w = (1/3,\ 1/3,\ 1/3)$, and do the arithmetic.

The covariance diagonal is the squared vols, ${0.0064}$, ${0.0144}$ and ${0.0400}$. The off-diagonals are correlation times the two vols: $0.10 \times 0.08 \times 0.12 = 0.00096$ for stat arb and macro, $0.20 \times 0.08 \times 0.20 = 0.0032$ for stat arb and credit, and $0.50 \times 0.12 \times 0.20 = 0.0120$ for macro and credit.

Multiplying through by $w$ gives $\Sigma w = (0.00352,\ 0.00912,\ 0.01840)$, and then

$$w^\top \Sigma w = \tfrac{1}{3}(0.00352 + 0.00912 + 0.01840) = 0.0103467, \qquad \sigma_p = \sqrt{0.0103467} = 0.101719.$$

So the firm runs at **10.17%** annualised volatility, which on \$500m is **\$50.86m** of annual risk. Now the marginal contributions, each $(\Sigma w)_i / \sigma_p$:

| Book | Weight $w_i$ | Marginal contribution | Risk contribution | Share | Dollars of risk |
| --- | --- | --- | --- | --- | --- |
| Stat arb | 0.3333 | 3.46% | **1.15%** | 11.34% | \$5.77m |
| Macro | 0.3333 | 8.97% | **2.99%** | 29.38% | \$14.94m |
| Credit carry | 0.3333 | 18.09% | **6.03%** | 59.28% | \$30.15m |
| **Total** | 1.0000 | | **10.17%** | 100.00% | **\$50.86m** |

The identity is on the page, so check it rather than trusting it. In volatility terms:

$$1.15\% + 2.99\% + 6.03\% = 10.17\% = \sigma_p.$$

And in dollars, \$5.77m + \$14.94m + \$30.15m = \$50.86m, which is the firm's total risk. Carry more decimals and it stays exact rather than nearly exact: 1.15351% + 2.98864% + 6.02971% = 10.17186%, against a portfolio volatility of 10.17186%. That is Euler's theorem, not a coincidence of rounding.

![A three row computation ladder showing weight times marginal contribution equals risk contribution for each book, with the risk contribution column summing to 10.17 percent, the portfolio volatility](/imgs/blogs/risk-budgeting-drawdown-control-math-for-quants-2.webp)

The intuition to take away: an equal capital split handed the credit book **59.28%** of the firm's risk, \$30.15m out of \$50.86m, in exchange for a third of the money. Nobody in the meeting voted for that. It is what the whiteboard said anyway.

## Equal risk contribution, and what it quietly assumes

The obvious repair is to choose weights so the contributions come out equal. That is **equal risk contribution**, usually sold as *risk parity*. The objective is to solve

$$w_i (\Sigma w)_i = w_j (\Sigma w)_j \quad \text{for all } i, j, \qquad \sum_i w_i = 1, \quad w_i > 0.$$

On our three books the solution is $w = (51.78\%,\ 30.74\%,\ 17.49\%)$, which on \$500m means \$258.88m to stat arb, \$153.68m to macro and \$87.44m to credit. Portfolio volatility falls to **8.05%**, and each book now contributes **\$13.41m** of the \$40.23m total.

Two things are worth saying plainly, because they are where candidates and committees get it wrong.

**First, this is an allocation rule, not an optimality claim.** Nothing in the construction mentions expected returns, so nothing in it can promise the best portfolio. Maillard, Roncalli and Teiletche (2010) pinned down exactly when it happens to be optimal: ERC equals the maximum-Sharpe portfolio **only when every asset has the same Sharpe ratio and every pair the same correlation**. Both conditions, not one. Forcing equal Sharpes on our books but leaving the real correlation structure gives an ERC Sharpe of 1.689 against the tangency portfolio's 1.697, so even half the condition is not enough. The same paper gives the sandwich that makes risk parity a reasonable default anyway: its volatility always sits between minimum variance and equal weight, here 6.95% ≤ 8.05% ≤ 10.17%.

**Second, the risk reduction is not free.** Risk parity bought lower volatility by moving money into the quiet book. If the mandate is a 10% volatility target rather than a fixed \$500m, you have to lever back up: ERC needs 124.28% of capital gross to reach 10% volatility, against 98.31% for equal capital. Diversification that arrives with leverage attached is a different trade from diversification that arrives free.

#### Worked example 2: equal capital against equal risk, in tail dollars

Compare the two allocations on the same \$500m at the same gross. Under a normal assumption the 97.5% **expected shortfall**, meaning the average loss in the worst 2.5% of years, is portfolio volatility times 2.3378.

- Equal capital: 10.17% volatility, an expected shortfall of 23.78%, which is **\$118.90m**.
- Equal risk: 8.05% volatility, an expected shortfall of 18.81%, which is **\$94.06m**.

The difference is **\$24.84m** in the worst 2.5% of years. The honest caveat sits right beside it: most of that gap is simply that the ERC portfolio is smaller, and levered back to 10% volatility its Gaussian expected shortfall matches the equal-capital one exactly. What ERC buys at matched volatility is not a thinner tail under this model, it is a less concentrated one, which only pays off in a world where one book can break on its own.

## When risk parity is the wrong answer

The null result is worth more than the sales pitch. Risk parity systematically underweights the volatile book, so it is a bet that Sharpe ratios do **not** rise with volatility across your books. Suppose they do: give the three books true Sharpes of 0.80, 1.20 and 1.60, the pattern where the risky book is risky because it is being paid to be. Scale every allocation to a 10% volatility target on \$500m and compare expected P&L:

| Allocation | Expected P&L at a 10% volatility target |
| --- | --- |
| Maximum Sharpe (tangency) | \$86.795m |
| Equal capital | \$86.513m |
| Equal risk contribution | \$82.869m |

Risk parity gives up \$3.926m a year against the optimum, and \$3.644m against the naive equal-capital rule it was introduced to fix. It is worse than doing nothing.

Reverse the pattern, with Sharpes of 1.60, 1.20 and 0.80 so the quiet book is the good one, and ERC earns \$86.072m against equal capital's \$70.784m, a gain of \$15.288m. Make the Sharpes flat and ERC wins by \$5.822m. So risk parity is not right or wrong in general. It is right whenever Sharpe is flat or falling in volatility and wrong when Sharpe rises with volatility, and the size of the error is the size of that slope. Knowing which world you are in is a research question, not an allocation rule, and the next section is about how little you actually know it.

## Budgeting against a view, and why the view gets shrunk

Once you have Sharpe estimates, the budget should tilt. The natural generalisation is **risk budgeting**: pick shares $b_i$ summing to one and solve for weights with $\mathrm{RC}_i = b_i\,\sigma_p$, then set $b_i$ proportional to your Sharpe estimate.

The trouble is that the Sharpe estimate is the least reliable number in the building. Its standard error, from Lo (2002), is roughly

$$\sigma_e = \sqrt{\frac{1 + S^2/2}{T}}$$

with $T$ in years. Three years of live data on a Sharpe-1.2 book gives $\sigma_e = \sqrt{1.72/3} = 0.757$. The estimate and its own error bar are the same size. Tilting a real risk budget on that is not a view, it is a coin flip with a spreadsheet attached.

So shrink it, exactly as in [the shrinkage post](/blog/trading/math-for-quants/shrinkage-stein-paradox-math-for-quants):

$$\hat{S}_i^{\text{shrunk}} = \bar{S} + \kappa\,(\hat{S}_i - \bar{S}), \qquad \kappa = \frac{\tau^2}{\tau^2 + \sigma_e^2},$$

where $\tau$ is how far apart you believe book Sharpes genuinely are across the firm. Take $\tau = 0.30$, a reasonable prior at a house where every book cleared a research bar. Then $\kappa = 0.09 / (0.09 + 0.757^2) = 0.136$. You keep **13.6 cents of every dollar of observed dispersion**.

#### Worked example 3: the tilt that survives three years of data

Observed Sharpes after three years: 0.80, 1.20, 1.60, mean 1.20. Shrunk, they become $1.20 - 0.136 \times 0.40 = 1.146$, then 1.200, then 1.254. The budget shares go from a raw 22.2% / 33.3% / 44.4% to a shrunk 31.8% / 33.3% / 34.8%.

Solve for the weights behind each and the money moves as follows. The raw tilt puts \$220.61m in stat arb and \$116.88m in credit. The shrunk tilt puts \$254.30m in stat arb and \$91.16m in credit. Believing your three-year Sharpes literally rather than shrinking them moves \$33.69m out of the stat arb book and \$25.72m into the credit book, on the strength of a difference you cannot measure.

![Grouped horizontal bars comparing risk budget shares under equal risk contribution, a raw Sharpe tilt of 22.2, 33.3 and 44.4 percent, and a shrunk Sharpe tilt of 31.8, 33.3 and 34.8 percent](/imgs/blogs/risk-budgeting-drawdown-control-math-for-quants-3.webp)

The intuition: shrinkage does not stop you having a view, it makes the size of the tilt match the strength of the evidence. After three years the evidence supports a tilt of a few percentage points, not twenty. The same diagnostic discipline applies to deciding whether a book has decayed at all, which is the subject of [the live-versus-backtest post](/blog/trading/math-for-quants/live-vs-backtest-divergence-math-for-quants) and [the alpha lifecycle post](/blog/trading/math-for-quants/alpha-lifecycle-decay-retirement-math-for-quants). The covariance matrix that all of this runs on is built in [the factor risk model post](/blog/trading/math-for-quants/factor-risk-model-build-math-for-quants).

## Drawdown is a different problem

Everything so far controls a **second moment**. A volatility target says the dispersion of outcomes will be such and such. It says nothing whatsoever about the order those outcomes arrive in, and drawdown is a statement about order. Two return streams with identical means and identical annual volatilities can have very different drawdown profiles, because drawdown is a path property and volatility is not.

That is a claim you can check rather than assert. Take the same 12% annual volatility and the same 12% expected return and change only the serial correlation of the daily returns, rescaling so the annual volatility stays pinned at 12.0%:

| Daily returns | P(10% trailing drawdown in a year) | Cost of the same stop |
| --- | --- | --- |
| Mean-reverting, $\phi = -0.20$ | 45.93% | \$5.39m |
| Independent | 42.89% | \$4.76m |
| Trending, $\phi = +0.20$ | 40.02% | \$4.19m |

Same mean, same volatility, and the identical stop rule costs \$1.20m a year more on one than the other. Note that the sign is not the one most people guess, which is the point: you cannot read a drawdown profile off a volatility number even approximately.

There is one clean closed form worth carrying. For a book with drift $\mu$ and volatility $\sigma$, the long-run distribution of how far below its high-water mark it currently sits is exponential, and

$$P(\text{depth} > D) = \exp\!\left(-\frac{2\mu D}{\sigma^2}\right) = \exp\!\left(-\frac{2 S D}{\sigma}\right),$$

with average depth $\sigma / (2S)$. At Sharpe 1.0 and 12% volatility, the book spends $\exp(-0.20/0.12) = 18.9\%$ of its life more than 10% below its high-water mark, and averages 6.0% below. Two things follow. The drawdown profile depends on Sharpe and volatility only through the ratio $S/\sigma$, so at a fixed volatility target it is set entirely by the Sharpe, which is the number you cannot measure. And the *maximum* drawdown is a different animal again: it grows without bound as the horizon lengthens, so "a stop at our worst historical drawdown" is a stop you are guaranteed to hit eventually.

## Worked example 4: what a drawdown stop actually costs

State the rule precisely enough to price. A \$200m book, 12% volatility, Sharpe 1.0, so expected P&L of \$24.0m a year. If it falls 10% below its high-water mark, measured at the daily close, it is flattened for the rest of the year. No deterioration, no regime change: the book is exactly as good as advertised throughout.

Getting this number right requires care, because there are two very different questions hiding here. The stationary formula above is about how *often* a book sits in a drawdown, not about whether the stop *ever fires* in a given year, and using one for the other is wrong by a factor of two. The trigger probability needs simulation, so it needs validation: the engine below was first checked against two cases with known answers. For a fixed loss limit rather than a trailing one, the exact first-passage probability is

$$P(\tau_{-L} \le T) = \Phi\!\left(\frac{-L-\mu T}{\sigma\sqrt{T}}\right) + e^{-2\mu L/\sigma^2}\,\Phi\!\left(\frac{-L+\mu T}{\sigma\sqrt{T}}\right),$$

which gives 14.03% for a 10% annual loss limit on this book. And for a driftless book, Lévy's theorem makes the trailing drawdown exactly distributed as the running maximum of an absolute Brownian motion, with a known series giving 78.45%. Simulated on a daily grid the engine returns 12.78% and 71.70%; refined to 16,128 steps it returns 13.90% and 77.58%, converging monotonically on both. The discretisation bias is understood and the engine is trusted.

With that established, on 2,000,000 simulated years:

- **The stop fires in 42.9% of years** (95% interval ±0.07 percentage points), against 14.03% for a fixed 10% loss limit. A trailing stop is three times more trigger-happy than a loss limit at the same number, because the high-water mark keeps moving up under it.
- Expected P&L falls from **\$23.98m** without the stop to **\$19.22m** with it. **The stop costs \$4.76m a year**, which is 19.9% of the book's entire expected return.
- **When it fires, 73.5% of the time the book would have finished the year higher without it.** That is the brief's temporary loss made permanent, with a number on it.
- What it buys: the 97.5% expected shortfall falls from 16.01% (\$32.02m) to 10.47% (\$20.94m), a saving of **\$11.08m**. The worst path in two million years goes from -49.87% to -12.74%.

![Line chart of a 200 million dollar book's cumulative P&L over twelve months, showing a high-water mark of plus 4 percent, a stop triggering at minus 6 percent in month 7, and a dotted counterfactual recovering to plus 8 percent by year end](/imgs/blogs/risk-budgeting-drawdown-control-math-for-quants-4.webp)

So the stop is insurance with a \$4.76m annual premium and an \$11.08m tail benefit. Whether that trade is worth it depends on how convex your funding is, which is a real question with a real answer, and the point is that you can now argue it in dollars rather than in adjectives.

Two second-order results matter more than they look. Monitoring frequency is a free parameter nobody debates: check the drawdown continuously rather than at the daily close and the trigger probability rises to 49.4% and the cost to \$5.74m. And the premium is upside down in the book's quality:

| True Sharpe | P(stop fires) | Annual cost | P(would have finished higher) |
| --- | --- | --- | --- |
| 0.0 | 71.7% | \$0.00m | 49.9% |
| 0.5 | 57.6% | \$3.34m | 62.8% |
| 1.0 | 42.9% | \$4.76m | 73.5% |
| 1.5 | 29.7% | \$4.80m | 81.6% |
| 2.0 | 19.1% | \$4.04m | 87.2% |

A stop on a book with no edge is free, because there is no expected return to interrupt. A stop on a good book is expensive, and the better the book the more often the stop is simply wrong. **The drawdown stop charges you most on exactly the books you least want to stop.** That is not an argument against stops, which exist to protect the firm's funding rather than the book's expectation. It is an argument for pricing them.

## The correlation problem, and what you do about it beforehand

Every number above rests on one covariance matrix, and a crisis is precisely the event that replaces it. Take the ERC portfolio and reprice it under a stress where macro and credit volatilities double and their correlation goes to 0.90, leaving the stat arb book alone.

![Two stacked bars comparing the equal risk portfolio under a normal covariance at 8.05 percent volatility against a stress covariance at 15.20 percent, where the risk split moves from thirds to 11.3, 45.0 and 43.8 percent](/imgs/blogs/risk-budgeting-drawdown-control-math-for-quants-5.webp)

Volatility goes from **8.05% to 15.20%**, and the risk split goes from thirds to **11.3% / 45.0% / 43.8%**. Two books that were budgeted 66.7% of the risk between them now carry 88.8%. The budget you spent a quarter negotiating has evaporated, and the level has nearly doubled, so the firm is running \$76.0m of risk against a \$40.23m plan.

The honest reading is that the level matters more than the split. Under a milder, uniform stress with every correlation at 0.75 the split barely moves at all, from thirds to 36.9% / 32.5% / 30.6%, while volatility still climbs from 8.05% to 15.52%. Crisis correlation is mostly a *scaling* problem, not a *shares* problem, which is a more useful thing to know than the cliché.

What a senior does about it is decided in advance, not during, and there are only two real options. The first is to size so that the stressed number, not the normal one, fits the mandate: de-lever to 0.53 times the normal-regime allocation and the stressed portfolio sits back at target. That is expensive and visible, and it is the only thing that works in the middle of the event. The second is to solve the budget on the stress matrix in the first place, which here gives weights of 70.85% / 18.62% / 10.53%. Look at what that costs: in normal times those weights put 66.8% of the risk in the stat arb book, so you have bought crisis robustness with a badly concentrated everyday portfolio. That is the trade, and the senior's contribution is to make the firm choose it consciously in March rather than discover it in October.

## The most contested number in the firm

Strip away the mathematics and the risk budget is an organisational document. It decides who gets to trade, at what size, and therefore whose bonus has a ceiling. It is the most contested number in the building, and every PM in the room understands the arithmetic well enough to argue that their book's correlation estimate is too high.

Which is why the two defences that matter are procedural rather than mathematical. Fix the shrinkage parameter before you see the Sharpes, so that the tilt is not negotiated after the fact. And publish the stressed budget alongside the normal one, so that "my book is uncorrelated" is a claim about the stress matrix rather than a claim about last year.

## Common misconceptions

**"Volatility targeting is drawdown control."** It is not, and this is the single most common error. A volatility target fixes the second moment. Drawdown is a path property, and the table above shows two streams with identical mean and volatility whose stop costs differ by \$1.20m a year.

**"Risk parity is optimal."** It is optimal under equal Sharpes and equal pairwise correlations, and not otherwise. When Sharpe rises with volatility across books it is \$3.644m a year worse than an equal-capital split.

**"Risk contributions are what each book loses."** They are marginal quantities, valid for small changes around the current weights. Double a book and its contribution more than doubles, because it is now correlated with a bigger version of itself.

**"A stop protects the P&L."** A stop truncates the left tail and pays for it out of the expectation. On a Sharpe-1.0 book that is \$11.08m of tail protection for \$4.76m a year, and 73.5% of the time it fires it was the wrong call in hindsight.

**"We are diversified, we have twelve books."** Twelve books running the same factor exposure are one book with twelve reporting lines. Risk contributions computed on a covariance matrix estimated from a calm sample will not tell you this, which is why the stress matrix is not optional.

## What this track was for

Thirty-three posts of mathematics, and the thing they were all circling is narrower than it looks. Every one of them was about the gap between a technique that works and a decision you can defend.

A model runs on inputs. Owning one means knowing which of those inputs carries the weight, how badly it is measured, what the answer does when it is wrong, and what the honest version of the number looks like when you say it out loud to people whose bonuses depend on it. Risk budgeting is the clearest case in the series because all four questions have answers you can compute: the covariance matters more than the Sharpe, the Sharpe is measured so badly that three years of data supports a thirteen-cent tilt, the answer roughly doubles under a stress nobody disputes is plausible, and the stop everyone assumes is prudent costs a fifth of the book's expected return.

Someone who can run a model produces the allocation. Someone who can own it produces the allocation, the number it becomes in the stress case, the input that would have to be wrong to change the decision, and a straight answer about what the insurance costs. That is the whole difference, and it is mostly a willingness to compute the uncomfortable half.

## Sources and further reading

- Maillard, S., Roncalli, T. and Teiletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." *Journal of Portfolio Management* 36(4), 60-70. The ERC definition, the tangency-equivalence condition, and the volatility sandwich.
- Roncalli, T. (2013). *Introduction to Risk Parity and Budgeting.* Chapman & Hall/CRC. The standard reference for risk budgeting with general budgets.
- Grossman, S. J. and Zhou, Z. (1993). "Optimal Investment Strategies for Controlling Drawdowns." *Mathematical Finance* 3(3), 241-276. The drawdown constraint as a control problem rather than a rule of thumb.
- Magdon-Ismail, M., Atiya, A., Pratap, A. and Abu-Mostafa, Y. (2004). "On the Maximum Drawdown of a Brownian Motion." *Journal of Applied Probability* 41(1), 147-161. Why maximum drawdown grows with the horizon.
- Lo, A. W. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal* 58(4), 36-52. The standard error used above.
- Qian, E. (2006). "On the Financial Interpretation of Risk Contribution." *Journal of Investment Management* 4(4). Risk contributions as loss contributions.
- Broadie, M., Glasserman, P. and Kou, S. (1997). "A Continuity Correction for Discrete Barrier Options." *Mathematical Finance* 7(4), 325-349. Why monitoring frequency moves a barrier probability.

## In the interview room and on the desk

The question is usually posed as a scenario: *you have five PMs and a risk budget, how do you split it?* It is a portfolio-construction question wearing a management costume, and the weak answer is "equally," sometimes dressed up as "equal capital, then adjust."

The strong answer moves in a fixed order. Start by refusing the frame: capital is not the thing being allocated, risk is, and risk contributions are marginal quantities that sum to portfolio volatility by Euler's theorem. Give the arithmetic, because an interviewer wants to see that you can actually compute $w_i(\Sigma w)_i / \sigma_p$ and not merely name it. Then state equal risk contribution as the default, and immediately name what it assumes: equal Sharpes and equal pairwise correlations, which is a condition and not a theorem. Then say that if you have Sharpe estimates the budget should tilt, and that the tilt must be shrunk, because three years of data gives a Sharpe standard error near 0.76 and an unshrunk tilt is mostly estimation noise being paid a salary. Then raise crisis correlation without being asked, and say what you would do about it in advance rather than during: size so the stressed number fits the mandate, or budget on the stress matrix and accept a concentrated normal-times book.

The trap, and it catches strong candidates, is treating a volatility target as drawdown control. If you say "we control drawdown with a 10% vol target" you have answered a different question, because volatility is a second moment and drawdown is a path property. The follow-up is then usually about stops, and the answer that lands is a priced one: a 10% trailing stop on a Sharpe-1.0 book fires in 42.9% of years with no deterioration at all, costs about a fifth of the book's expected return, and is wrong in hindsight nearly three times in four. Stops exist to protect the firm's funding, not the book's expectation, and knowing the difference is the whole job.

Citadel, Millennium, Two Sigma and every multi-strategy seat weight this more heavily than any single pricing question, because the risk budget is the product. Get it wrong and every good model in the building is sized incorrectly.
