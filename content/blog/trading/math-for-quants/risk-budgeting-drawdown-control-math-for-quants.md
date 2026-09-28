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
readTime: 21
---

> [!important]
> **TL;DR:** Allocating capital is the wrong frame. A multi-strategy firm allocates risk, and risk combines through a covariance matrix rather than by addition, so the two arithmetics give different answers.
>
> - Split \$500m equally across three books and the risk split is **11.34% / 29.38% / 59.28%**. A third of the capital carries nearly sixty percent of the risk.
> - Risk contributions sum **exactly** to portfolio volatility, by Euler's theorem: 1.15% + 2.99% + 6.03% = 10.17%. That additivity is the only reason "risk budget" means anything.
> - Equal risk contribution is an **allocation rule, not an optimum**. When Sharpe rises with volatility across your books it is **\$3.64m a year worse** than the naive equal-capital split it claims to improve on.
> - A volatility target controls the second moment and says nothing about the path. On a \$200m book at Sharpe 1.0, a 10% trailing drawdown stop fires in **42.9%** of years with no deterioration at all, costs **\$4.76m** of a \$23.98m expectation, and when it fires the book would have finished higher without it **73.5%** of the time.
> - The stop costs most on exactly the books you least want to stop: \$4.04m at Sharpe 2.0, nothing at Sharpe 0.0.

You are in the annual risk meeting. Five portfolio managers, one number to divide. The CIO says "let's keep it simple, everyone gets the same allocation," someone writes five equal numbers on the whiteboard, and the meeting moves on feeling fair.

It is not fair. A dollar handed to a credit book and a dollar handed to a stat arb book do not buy the same amount of anything. They buy wildly different amounts of risk, and risk is the scarce resource, the one the mandate is written in and the one that runs out first.

That figure is the mental model: the same three books, the same \$500m, two different pictures depending on whether you draw the capital or the risk.

![Two stacked bars comparing an equal capital split of 500 million dollars in thirds against the resulting risk split of 11.34, 29.38 and 59.28 percent, with the credit carry book taking nearly sixty percent of the risk](/imgs/blogs/risk-budgeting-drawdown-control-math-for-quants-1.webp)

Everything below is illustrative arithmetic on assumed inputs. The covariance matrix, the Sharpes and the book sizes are stipulated so the mechanics are checkable, not drawn from any firm's actual book.

## Foundations: what a risk contribution actually is

Three books, carried through the whole post. A **stat arb** book with 8% annualised volatility, a **macro** book at 12%, and a **credit carry** book at 20%. *Volatility* here means the standard deviation of the book's annual return: a 12% vol book has returns that typically land within about 12% either side of their average. Correlations: stat arb to macro 0.10, stat arb to credit 0.20, macro to credit 0.50. Firm capital \$500m.

Write $w_i$ for the fraction of firm capital in book $i$ and $\Sigma$ for the covariance matrix of the books' returns. Portfolio volatility is

$$\sigma_p(w) = \sqrt{w^\top \Sigma w}.$$

This is not a sum. You cannot ask "how much of the 10% is the credit book" the way you can ask how much of the \$500m it holds, because the square root has already blended everything.

The question is rescued by asking a *marginal* one instead. If I give book $i$ one more dollar, how much does firm volatility rise? That is a derivative with a name, the **marginal contribution to risk**:

$$\mathrm{MCR}_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}.$$

Multiply the marginal rate by how much of the book you actually hold and you get the **risk contribution**:

$$\mathrm{RC}_i = w_i \cdot \mathrm{MCR}_i = \frac{w_i (\Sigma w)_i}{\sigma_p}.$$

So far this is a definition, and definitions are cheap. What makes this one worth having is that $\sigma_p$ is *homogeneous of degree one* in $w$: double every weight and you double the volatility. Euler's theorem on homogeneous functions then says the parts add up:

$$\sum_{i=1}^{n} w_i \frac{\partial \sigma_p}{\partial w_i} = \sigma_p.$$

That is the whole engine. The contributions sum exactly to the portfolio's own volatility, nothing left over and nothing double-counted, and a quantity that adds up is one you can divide into shares and hold people to. Without Euler's theorem "risk budget" would be a metaphor. With it, it is arithmetic.

Note what a contribution is *not*: it is not the book's standalone risk. A book can be volatile alone and contribute almost nothing if it hedges the rest, or be quiet alone and contribute a lot if it leans the same way as everything else. The contribution is a property of the book **inside this portfolio**, and it moves when any other book moves.

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

In dollars, \$5.77m + \$14.94m + \$30.15m = \$50.86m, the firm's total risk. Carry more decimals and it stays exact rather than nearly exact: 1.15351% + 2.98864% + 6.02971% = 10.17186%, against a portfolio volatility of 10.17186%. That is Euler's theorem, not a coincidence of rounding.

![A three row computation ladder showing weight times marginal contribution equals risk contribution for each book, with the risk contribution column summing to 10.17 percent, the portfolio volatility](/imgs/blogs/risk-budgeting-drawdown-control-math-for-quants-2.webp)

The intuition: an equal capital split handed the credit book **59.28%** of the firm's risk, \$30.15m out of \$50.86m, in exchange for \$166.7m, a third of the money. Nobody in the meeting voted for that. It is what the whiteboard said anyway.

## Equal risk contribution, and what it quietly assumes

The obvious repair is to choose weights so the contributions come out equal. That is **equal risk contribution**, usually sold as *risk parity*:

$$w_i (\Sigma w)_i = w_j (\Sigma w)_j \quad \text{for all } i, j, \qquad \sum_i w_i = 1, \quad w_i > 0.$$

On our three books the solution is $w = (51.78\%,\ 30.74\%,\ 17.49\%)$, which on \$500m means \$258.88m to stat arb, \$153.68m to macro and \$87.44m to credit. Portfolio volatility falls to **8.05%**, or 8.0464% before rounding, and each book now contributes **\$13.41m** of the \$40.23m total.

Two things are worth saying plainly, because they are where committees get it wrong.

**First, this is an allocation rule, not an optimality claim.** Nothing in the construction mentions expected returns, so nothing in it can promise the best portfolio. Maillard, Roncalli and Teiletche (2010) pinned down when it happens to be optimal: ERC equals the maximum-Sharpe portfolio **only when every asset has the same Sharpe ratio and every pair the same correlation**. Both conditions, not one. Forcing equal Sharpes on our books but keeping the real correlation structure gives an ERC Sharpe of 1.689 against the tangency portfolio's 1.697, so even half the condition is not enough. The same paper gives the sandwich that makes risk parity a reasonable default anyway: its volatility always sits between minimum variance and equal weight, here 6.95% ≤ 8.05% ≤ 10.17%.

**Second, the risk reduction is not free.** If the mandate is a 10% volatility target rather than a fixed \$500m, you have to lever back up: ERC needs 124.28% of capital gross to reach 10% volatility, against 98.31% for equal capital. Diversification that arrives with leverage attached is a different trade from diversification that arrives free.

#### Worked example 2: equal capital against equal risk, in tail dollars

Under a normal assumption the 97.5% **expected shortfall**, meaning the average loss in the worst 2.5% of years, is portfolio volatility times 2.3378. Equal capital runs at 10.1719% volatility, an expected shortfall of 23.780% or **\$118.90m**. Equal risk runs at 8.0464%, an expected shortfall of 18.811% or **\$94.06m**. The difference is **\$24.84m** in the worst 2.5% of years.

The honest caveat sits right beside it: most of that gap is simply that the ERC portfolio is smaller, and levered back to 10% volatility its Gaussian expected shortfall matches the equal-capital one exactly. What ERC buys at matched volatility is not a thinner tail under this model, it is a less concentrated one, which only pays off in a world where one book can break on its own.

## When risk parity is the wrong answer

Risk parity systematically underweights the volatile book, so it is a bet that Sharpe ratios do **not** rise with volatility across your books. Suppose they do: give the three books true Sharpes of 0.80, 1.20 and 1.60, the pattern where the risky book is risky because it is being paid to be. Scale every allocation to a 10% volatility target on \$500m and compare expected P&L:

| Allocation | Expected P&L at a 10% volatility target |
| --- | --- |
| Maximum Sharpe (tangency) | \$86.795m |
| Equal capital | \$86.513m |
| Equal risk contribution | \$82.869m |

Risk parity gives up \$3.926m a year against the optimum, and \$3.644m against the naive equal-capital rule it was introduced to fix. It is worse than doing nothing.

Reverse the pattern, with Sharpes of 1.60, 1.20 and 0.80 so the quiet book is the good one, and ERC earns \$86.072m against equal capital's \$70.784m, a gain of \$15.288m. Make the Sharpes flat and ERC wins by \$5.822m. So risk parity is right whenever Sharpe is flat or falling in volatility, wrong when Sharpe rises with it, and the size of the error is the size of that slope. Which world you are in is a research question, and the next section is about how little you know the answer.

## Budgeting against a view, and why the view gets shrunk

Once you have Sharpe estimates the budget should tilt. The natural generalisation is **risk budgeting**: pick shares $b_i$ summing to one, solve for weights with $\mathrm{RC}_i = b_i\,\sigma_p$, and set $b_i$ proportional to your Sharpe estimate.

The trouble is that the Sharpe estimate is the least reliable number in the building. Its standard error, from Lo (2002), is roughly

$$\sigma_e = \sqrt{\frac{1 + S^2/2}{T}}$$

with $T$ in years. Three years of live data on a Sharpe-1.2 book gives $\sigma_e = \sqrt{1.72/3} = 0.757$, call it 0.76. The estimate and its own error bar are the same size. Tilting a real risk budget on that is not a view, it is a coin flip with a spreadsheet attached. So shrink it, exactly as in [the shrinkage post](/blog/trading/math-for-quants/shrinkage-stein-paradox-math-for-quants):

$$\hat{S}_i^{\text{shrunk}} = \bar{S} + \kappa\,(\hat{S}_i - \bar{S}), \qquad \kappa = \frac{\tau^2}{\tau^2 + \sigma_e^2},$$

where $\tau$ is how far apart you believe book Sharpes genuinely are across the firm. Take $\tau = 0.30$, a reasonable prior at a house where every book cleared a research bar. Then $\kappa = 0.09 / (0.09 + 0.757^2) = 0.136$. You keep **13.6 cents of every dollar of observed dispersion**.

#### Worked example 3: the tilt that survives three years of data

Observed Sharpes after three years: 0.80, 1.20, 1.60, mean 1.20. Shrunk, they become $1.20 - 0.136 \times 0.40 = 1.146$, then 1.200, then 1.254. The budget shares go from a raw 22.2% / 33.3% / 44.4% to a shrunk 31.8% / 33.3% / 34.8%.

Solve for the weights behind each and the money moves. The raw tilt puts \$220.61m in stat arb and \$116.88m in credit; the shrunk tilt puts \$254.30m and \$91.16m. Believing your three-year Sharpes literally moves \$33.69m out of the stat arb book and \$25.72m into the credit book, on the strength of a difference you cannot measure.

![Grouped horizontal bars comparing risk budget shares under equal risk contribution, a raw Sharpe tilt of 22.2, 33.3 and 44.4 percent, and a shrunk Sharpe tilt of 31.8, 33.3 and 34.8 percent](/imgs/blogs/risk-budgeting-drawdown-control-math-for-quants-3.webp)

Shrinkage does not stop you having a view, it makes the size of the tilt match the strength of the evidence: after three years that is a few percentage points, not twenty. The same discipline decides whether a book has decayed at all, in [the live-versus-backtest post](/blog/trading/math-for-quants/live-vs-backtest-divergence-math-for-quants) and [the alpha lifecycle post](/blog/trading/math-for-quants/alpha-lifecycle-decay-retirement-math-for-quants). The covariance matrix all of this runs on is built in [the factor risk model post](/blog/trading/math-for-quants/factor-risk-model-build-math-for-quants).

## Drawdown is a different problem

Everything so far controls a **second moment**. A volatility target fixes the dispersion of outcomes and says nothing about the order they arrive in, and drawdown is a statement about order. That is checkable rather than assertable. Hold the 12% annual volatility and the 12% expected return fixed and change only the serial correlation of the daily returns, rescaling so annual volatility stays pinned at 12.0%:

| Daily returns | P(10% trailing drawdown in a year) | Cost of the same stop |
| --- | --- | --- |
| Mean-reverting, $\phi = -0.20$ | 45.93% | \$5.39m |
| Independent | 42.89% | \$4.76m |
| Trending, $\phi = +0.20$ | 40.02% | \$4.19m |

Same mean, same volatility, and the identical stop rule costs \$1.20m a year more on one than the other. The sign is not the one most people guess, which is the point: you cannot read a drawdown profile off a volatility number even approximately.

One clean closed form is worth carrying. For a book with drift $\mu$ and volatility $\sigma$, the long-run distribution of how far below its high-water mark it currently sits is exponential:

$$P(\text{depth} > D) = \exp\!\left(-\frac{2\mu D}{\sigma^2}\right) = \exp\!\left(-\frac{2 S D}{\sigma}\right),$$

with average depth $\sigma / (2S)$. At Sharpe 1.0 and 12% volatility the book spends $\exp(-0.20/0.12) = 18.9\%$ of its life more than 10% below its high-water mark, averaging 6.0% below. Two things follow. The profile depends on Sharpe and volatility only through $S/\sigma$, so at a fixed volatility target it is set entirely by the Sharpe, the number you cannot measure. And *maximum* drawdown grows without bound as the horizon lengthens, so "a stop at our worst historical drawdown" is a stop you are guaranteed to hit eventually.

## Worked example 4: what a drawdown stop actually costs

State the rule precisely enough to price. A \$200m book, 12% volatility, Sharpe 1.0, so expected P&L of \$24.0m a year. If it falls 10% below its high-water mark, measured at the daily close, it is flattened for the rest of the year. No deterioration, no regime change: the book is exactly as good as advertised throughout.

Two different questions hide here, and conflating them is wrong by a factor of two. The formula above is about how *often* a book sits in a drawdown, not whether the stop *ever fires* in a given year. The second needs simulation, so it needs validation against cases whose answers are already known. For a fixed loss limit rather than a trailing one the exact first-passage probability is

$$P(\tau_{-L} \le T) = \Phi\!\left(\frac{-L-\mu T}{\sigma\sqrt{T}}\right) + e^{-2\mu L/\sigma^2}\,\Phi\!\left(\frac{-L+\mu T}{\sigma\sqrt{T}}\right),$$

giving 14.03% for a 10% annual loss limit on this book. And for a driftless book, Lévy's theorem makes the trailing drawdown exactly distributed as the running maximum of an absolute Brownian motion, with a known series giving 78.45%. On a daily grid the simulator returns 12.78% and 71.70%; refined to 16,128 steps it returns 13.90% and 77.58%, converging monotonically on both. The discretisation bias is understood, so the engine is trusted. On 2,000,000 simulated years:

- **The stop fires in 42.9% of years** (95% interval ±0.07 percentage points), against 14.03% for a fixed 10% loss limit. A trailing stop is three times more trigger-happy at the same number, because the high-water mark keeps moving up under it.
- Expected P&L falls from **\$23.98m** without the stop to **\$19.22m** with it. **The stop costs \$4.76m a year**, 19.85% of the book's entire expected return.
- **When it fires, 73.5% of the time the book would have finished the year higher without it.** That is a temporary loss made permanent, with a number on it.
- What it buys: the 97.5% expected shortfall falls from 16.01% (\$32.02m) to 10.47% (\$20.94m), a saving of **\$11.08m**, and the worst path in two million years goes from -49.87% to -12.74%.

![Line chart of a 200 million dollar book's cumulative P&L over twelve months, showing a high-water mark of plus 4 percent, a stop triggering at minus 6 percent in month 7, and a dotted counterfactual recovering to plus 8 percent by year end](/imgs/blogs/risk-budgeting-drawdown-control-math-for-quants-4.webp)

That figure traces one such year. The book peaks at +4% in month 5, falls to -6% by month 7, which is 10% below the high-water mark, and is flattened there. The dotted line is the year it would have had: +8% by December.

So the stop is insurance with a \$4.76m annual premium and an \$11.08m tail benefit. Whether it is worth taking depends on how convex your funding is, and the point is that you can argue it in dollars rather than adjectives.

Two second-order results matter more than they look. Monitoring frequency is a free parameter nobody debates: check the drawdown continuously rather than at the daily close and the trigger probability rises to 49.4% and the cost to \$5.74m. And the premium runs upside down in the book's quality:

| True Sharpe | P(stop fires) | Annual cost | P(would have finished higher) |
| --- | --- | --- | --- |
| 0.0 | 71.7% | \$0.00m | 49.9% |
| 0.5 | 57.6% | \$3.34m | 62.8% |
| 1.0 | 42.9% | \$4.76m | 73.5% |
| 1.5 | 29.7% | \$4.80m | 81.6% |
| 2.0 | 19.1% | \$4.04m | 87.2% |

A stop on a book with no edge is free, because there is no expected return to interrupt. A stop on a good book is expensive, and the better the book the more often the stop is simply wrong. **The drawdown stop charges you most on exactly the books you least want to stop.** That is not an argument against stops, which exist to protect the firm's funding rather than the book's expectation. It is an argument for pricing them.

## The correlation problem, and what you do about it beforehand

Every number above rests on one covariance matrix, and a crisis is the event that replaces it. Reprice the ERC portfolio under a stress where macro and credit volatilities double and their correlation goes to 0.90, leaving stat arb alone.

![Two stacked bars comparing the equal risk portfolio under a normal covariance at 8.05 percent volatility against a stress covariance at 15.20 percent, where the risk split moves from thirds to 11.3, 45.0 and 43.8 percent](/imgs/blogs/risk-budgeting-drawdown-control-math-for-quants-5.webp)

Volatility goes from **8.05% to 15.20%** and the risk split from thirds to **11.3% / 45.0% / 43.8%**. Two books budgeted two thirds of the risk between them now carry 88.8%, and the firm is running \$76.0m of risk against a \$40.23m plan.

The honest reading is that the level matters more than the split. Under a milder, uniform stress with every correlation at 0.75 the split barely moves, from thirds to 36.9% / 32.5% / 30.6%, while volatility still climbs from 8.05% to 15.52%. Crisis correlation is mostly a *scaling* problem rather than a *shares* problem, which is more useful than the cliché.

What a senior does about it is decided in advance, and there are two real options. Size so the stressed number fits the mandate: de-lever to 0.53 times the normal-regime allocation and the stressed portfolio sits back at target. Or solve the budget on the stress matrix in the first place, which here gives weights of 70.85% / 18.62% / 10.53%. Look at what the second costs: in normal times those weights put 66.8% of the risk in the stat arb book, so crisis robustness has been bought with a badly concentrated everyday portfolio. The senior's job is to make the firm choose that consciously in March rather than discover it in October.

## The most contested number in the firm

Strip away the mathematics and the risk budget is an organisational document. It decides who gets to trade, at what size, and therefore whose bonus has a ceiling. Every PM in the room understands the arithmetic well enough to argue that their book's correlation estimate is too high.

Which is why the two defences that matter are procedural. Fix the shrinkage parameter before you see the Sharpes, so the tilt is not negotiated after the fact. And publish the stressed budget alongside the normal one, so "my book is uncorrelated" becomes a claim about the stress matrix rather than a claim about last year.

## Common misconceptions

**"Volatility targeting is drawdown control."** The most common error here. A volatility target fixes the second moment; drawdown is a path property, and the table above shows two streams with identical mean and volatility whose stop costs differ by \$1.20m a year.

**"Risk parity is optimal."** It is optimal under equal Sharpes and equal pairwise correlations, not otherwise. When Sharpe rises with volatility across books it is \$3.644m a year worse than an equal-capital split.

**"Risk contributions are what each book loses."** They are marginal quantities, valid for small changes around the current weights. Double a book and its contribution more than doubles, because it is now correlated with a bigger version of itself.

**"A stop protects the P&L."** A stop truncates the left tail and pays for it out of the expectation: \$11.08m of tail protection for \$4.76m a year, and in 73.5% of the years it fires it was the wrong call in hindsight.

**"We are diversified, we have twelve books."** Twelve books running the same factor exposure are one book with twelve reporting lines. Contributions computed on a covariance matrix estimated from a calm sample will not tell you this, which is why the stress matrix is not optional.

## What this track was for

Thirty-three posts of mathematics, all circling something narrower than it looks: the gap between a technique that works and a decision you can defend.

A model runs on inputs. Owning one means knowing which input carries the weight, how badly it is measured, what the answer does when it is wrong, and how the honest version of the number sounds said out loud to people whose bonuses depend on it. Risk budgeting is the clearest case in the series because all four questions have computable answers: the covariance matters more than the Sharpe, the Sharpe is measured so badly that three years of data supports a thirteen-cent tilt, the answer nearly doubles under a stress nobody disputes is plausible, and the stop everyone assumes is prudent costs a fifth of the book's expected return.

Someone who can run a model produces the allocation. Someone who can own it produces the allocation, the number it becomes in the stress case, the input that would have to be wrong to change the decision, and a straight answer about what the insurance costs. That is the whole difference, and it is mostly a willingness to compute the uncomfortable half.

## Sources and further reading

- Maillard, S., Roncalli, T. and Teiletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios." *Journal of Portfolio Management* 36(4), 60-70. The tangency-equivalence condition and the volatility sandwich.
- Roncalli, T. (2013). *Introduction to Risk Parity and Budgeting.* Chapman & Hall/CRC.
- Grossman, S. J. and Zhou, Z. (1993). "Optimal Investment Strategies for Controlling Drawdowns." *Mathematical Finance* 3(3), 241-276.
- Magdon-Ismail, M., Atiya, A., Pratap, A. and Abu-Mostafa, Y. (2004). "On the Maximum Drawdown of a Brownian Motion." *Journal of Applied Probability* 41(1), 147-161.
- Lo, A. W. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal* 58(4), 36-52. The standard error used above.
- Qian, E. (2006). "On the Financial Interpretation of Risk Contribution." *Journal of Investment Management* 4(4).
- Broadie, M., Glasserman, P. and Kou, S. (1997). "A Continuity Correction for Discrete Barrier Options." *Mathematical Finance* 7(4), 325-349.

## In the interview room and on the desk

The question is usually posed as a scenario: *you have five PMs and a risk budget, how do you split it?* It is a portfolio-construction question wearing a management costume, and the weak answer is "equally," sometimes dressed up as "equal capital, then adjust."

The strong answer moves in a fixed order. Refuse the frame first: capital is not the thing being allocated, risk is, and risk contributions are marginal quantities that sum to portfolio volatility by Euler's theorem. Give the arithmetic, because the interviewer wants to see you compute $w_i(\Sigma w)_i / \sigma_p$ rather than name it. Then offer equal risk contribution as the default and immediately name what it assumes: equal Sharpes and equal pairwise correlations, a condition and not a theorem. Then say that Sharpe estimates should tilt the budget and that the tilt must be shrunk, because three years of data gives a Sharpe standard error near 0.757 and an unshrunk tilt is estimation noise drawing a salary. Then raise crisis correlation unprompted, and say what you would do in advance rather than during: size so the stressed number fits the mandate, or budget on the stress matrix and accept a concentrated normal-times book.

The trap that catches strong candidates is treating a volatility target as drawdown control. "We control drawdown with a 10% vol target" answers a different question, because volatility is a second moment and drawdown is a path property. The follow-up is usually about stops, and the answer that lands is a priced one: a 10% trailing stop on a Sharpe-1.0 book fires in 42.9% of years with no deterioration at all, costs about a fifth of the book's expected return, and is wrong in hindsight nearly three times in four. Stops exist to protect the firm's funding, not the book's expectation.

Citadel, Millennium, Two Sigma and every multi-strategy seat weight this more heavily than any single pricing question, because the risk budget is the product. Get it wrong and every good model in the building is sized incorrectly.
