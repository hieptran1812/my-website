---
title: "P&L Attribution: Making the Pieces Add Up to the Number"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "A strategy made money and nobody can say why. Attribution decomposes realised P&L into causes that sum exactly to the total, and the reason it is hard is that the causes interact, so the decomposition is not unique."
tags: ["performance-attribution", "factor-models", "brinson", "implementation-shortfall", "portfolio-management", "risk-models", "multi-period-linking", "quantitative-research", "pnl", "quant-interviews"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 20
---

> [!important]
> **TL;DR:** Attribution is the discipline of splitting realised P&L into causes that sum exactly to the total. The hard part is that the causes interact, so the split is a choice you defend rather than a formula you apply.
>
> - Run the book through the risk model first. On the \$500m example below, \$34.75m of a \$40.0m year was factor exposure and only \$5.25m was specific. The manager owns the second number.
> - The cross term between allocation and selection is a real rectangle of money. Brinson gives it its own bucket; other schemes assign it. On \$11.4m of active P&L, the stock picker earns \$4.0m or \$7.0m depending on which convention is running.
> - Costs belong inside the decomposition, not beside it. A \$250m market-neutral book earning \$14.0m gross paid \$12.3m of implementation shortfall and kept \$1.7m.
> - That \$5.25m specific return is a t of 0.26 against 4.0% specific volatility. About 79% of zero-skill years land further from zero than that, and you would need 58 years of it to reach t = 2.
> - Contributions do not add across periods. Three quarters summing to 6.00% arithmetically are worth 6.1798% compounded, a \$719,200 gap on \$400m that Carino linking closes exactly.

## Introduction

A strategy made 8% last year. The investment committee wants to know where it came from, and three people in the room already have an answer they like. The portfolio manager says stock selection. The risk officer says the book was long momentum into a momentum year. The head of trading says nobody has subtracted what the execution desk paid to get in and out.

All three can be simultaneously right, and attribution is the machinery that says how much each of them is right by. Its promise is narrow and absolute: whatever you decompose, the pieces must reach the total. Not approximately, not up to a rounding plug. Exactly. A decomposition that does not close is not an explanation, it is a guess with arithmetic attached.

The figure below is the mental model, and it is the first thing a senior reaches for.

![Waterfall of a \$500m book's \$40.0m realised P&L into four factor contributions and a specific residual, the five bars reaching the \$40.00m total exactly.](/imgs/blogs/pnl-attribution-math-for-quants-1.webp)

What makes this genuinely difficult is not the arithmetic. It is that the causes overlap. Being overweight a sector *and* picking well inside it produces a lump of money that belongs to both decisions, and no amount of care makes that lump split itself. Someone has to decide where it goes, and that decision moves millions of dollars of credit between two people who are both in the room. The rest of this post is about doing it honestly.

This is the companion to [building the risk model itself](/blog/trading/math-for-quants/factor-risk-model-build-math-for-quants). That post produces the exposures and factor returns; this one spends them.

## Foundations: what attribution is, and the three things it is not

**P&L** is money: the change in the book's value over a period. **Return** is that money divided by the capital that produced it. **Attribution** takes one of those numbers and writes it as a sum of named causes.

**Exposure** is how much of a thing you own, measured in units the model understands. A **factor** is a common driver of returns that many positions share: the market, company size, value, momentum, an industry. A book's exposure to momentum is a single number saying how much of the book's return moves when momentum moves. The **specific** or **residual** return is what is left after every factor has been paid its due. It is the part of the return the model cannot explain with anything shared, which is why people call it alpha, and why the argument about it is never over.

Three things attribution is not.

**It is not a prediction.** Attribution is a statement about a period that has already happened. That the book made 4.20% from market exposure last year is a fact about last year. It says nothing about next year beyond what you separately believe about the market.

**It is not a test.** A decomposition always closes if you build it correctly, because closing is an algebraic property of how you defined the terms, not an empirical finding. The identity holds for a skilled manager and for a coin flip. Passing the sum check is a hygiene condition, not evidence.

**It is not unique.** This is the one people are most surprised by, and it gets its own section.

## Factor versus specific: the first cut

Every serious attribution starts by running the realised returns through the risk model. Write the portfolio's realised return in period $t$ as

$$r_{p,t} \;=\; \sum_{k=1}^{K} x_{k,t}\, f_{k,t} \;+\; u_t,$$

where $x_{k,t}$ is the book's exposure to factor $k$ at the start of the period, $f_{k,t}$ is that factor's realised return over the period, and $u_t$ is the specific return. This is an identity by construction: $u_t$ is *defined* as the leftover. The modelling choice sits entirely in which factors you put in the sum, which is exactly why the residual is contested.

#### Worked example 1: a \$500m book's \$40.0m year, split into factor and specific

A \$500m long-biased equity book returned 8.00%, so \$40.0m. Its average exposures over the year, and the factors' realised returns:

| Factor | Exposure | Factor return | Contribution | Dollars |
| --- | --- | --- | --- | --- |
| Market | 0.35 | +12.0% | +4.20% | +\$21.00m |
| Size | -0.20 | -4.0% | +0.80% | +\$4.00m |
| Value | 0.45 | +3.0% | +1.35% | +\$6.75m |
| Momentum | 0.10 | +6.0% | +0.60% | +\$3.00m |
| **Factor total** | | | **+6.95%** | **+\$34.75m** |
| **Specific** | | | **+1.05%** | **+\$5.25m** |
| **Total** | | | **+8.00%** | **+\$40.00m** |

Read the second row carefully, because it is the one that trips people. The size factor *lost* 4.0% and the book still made \$4.00m from it, because the book was short size. A negative exposure to a negative return is a positive contribution. Sign errors here are the single most common attribution bug, and they are invisible until someone asks why a losing factor is in the profit column.

Now check the close: 21.00 + 4.00 + 6.75 + 3.00 + 5.25 = 40.00. Exactly \$40.0m, no plug.

The headline is the ratio. \$34.75m of \$40.0m, or 86.9%, came from exposures a client could have bought for a few basis points in an index fund. \$5.25m, or 13.1%, is the part that required this manager. That reframing is the whole reason the first cut is factor versus specific: it converts "we made 8%" into "we made 1.05% that is arguably ours", and every subsequent conversation is about that smaller number.

(The dollar figures in this post are illustrative arithmetic on assumed inputs, chosen so every identity closes exactly. They are not drawn from a live book.)

## The interaction problem

Here is the heart of it. Suppose you are measured against a benchmark, and you differ from it in two ways at once: you weight sectors differently, and inside each sector you hold different names. Two decisions, two people, often two bonuses.

Write $w_i$ for your weight in sector $i$, $W_i$ for the benchmark's, $R_i$ for your return inside that sector and $B_i$ for the benchmark's. Your active return is $\sum_i w_i R_i - \sum_i W_i B_i$. Now try to split it.

The natural pieces are an **allocation** effect, for weighting sectors differently, and a **selection** effect, for picking better inside them. But when you write them down, a third term falls out that belongs to neither:

$$w_i R_i - W_i B_i \;=\; \underbrace{(w_i - W_i)B_i}_{\text{allocation}} \;+\; \underbrace{W_i(R_i - B_i)}_{\text{selection}} \;+\; \underbrace{(w_i - W_i)(R_i - B_i)}_{\text{interaction}}.$$

That third term is not a modelling artefact. It is a real rectangle of money, and it exists because you were overweight a sector *in which you also picked well*. It is the product of two decisions, so it is genuinely the joint property of both.

![Area decomposition of a sector's active contribution into an allocation rectangle, a selection rectangle and the interaction rectangle in the corner that belongs to neither.](/imgs/blogs/pnl-attribution-math-for-quants-2.webp)

Brinson, Hood and Beebower (1986) leave it in its own bucket and report three numbers. That is the honest answer, and it is also the one nobody likes, because a committee cannot pay a bonus to "interaction". So most shops fold it into one of the other two. Both foldings are algebraically valid. Both close exactly. They tell different stories about who earned what.

#### Worked example 2: \$11.4m of active P&L, three defensible answers

A \$400m long-only equity book against a three-sector benchmark:

| Sector | Portfolio weight | Benchmark weight | Portfolio return | Benchmark return |
| --- | --- | --- | --- | --- |
| Tech | 45% | 30% | 18% | 14% |
| Financials | 20% | 30% | 4% | 6% |
| Industrials | 35% | 40% | 9% | 8% |

The portfolio returned 12.05%, the benchmark 9.20%, so the active return is 2.85%, which on \$400m is \$11.40m. Every scheme below must reach that number.

**Scheme A, Brinson-Hood-Beebower with three terms.** Allocation +1.10%, selection +1.00%, interaction +0.75%. In dollars: \$4.40m, \$4.00m, \$3.00m. Sum: \$11.40m.

**Scheme B, interaction folded into selection.** Use the portfolio's own weight for the selection term, $w_i(R_i - B_i)$, which absorbs the cross term, and use the Brinson-Fachler allocation $(w_i - W_i)(B_i - B_p)$ measured against the benchmark's total return. Allocation +1.10%, selection +1.75%. In dollars: \$4.40m and \$7.00m. Sum: \$11.40m.

**Scheme C, interaction folded into allocation.** Use $(w_i - W_i)R_i$ for allocation and keep the benchmark-weighted selection term. Allocation +1.85%, selection +1.00%. In dollars: \$7.40m and \$4.00m. Sum: \$11.40m.

![Three stacked bars, each totalling \$11.40m, showing the \$3.00m interaction block sitting alone, then folded into selection, then folded into allocation.](/imgs/blogs/pnl-attribution-math-for-quants-3.webp)

Three arithmetically correct decompositions of one P&L. The stock picker earned \$4.00m or \$7.00m. The sector allocator earned \$4.40m or \$7.40m. The \$3.00m does not vanish and it does not duplicate, it just changes hands. That is 26% of the active P&L whose ownership is settled by a convention, and in most firms nobody in the meeting knows which convention the report is using.

Two things follow. First, the allocation *total* happens to be the same under Brinson-Hood-Beebower and Brinson-Fachler, at +1.10%, because the weight differences sum to zero and subtracting the benchmark's total return from each $B_i$ therefore cancels in aggregate. What Brinson-Fachler changes is the *per-sector* split: it says an overweight in a sector that beat the overall benchmark was a good call, rather than merely a sector with a positive return. Financials returned +6%, so Brinson-Hood-Beebower charges the underweight -0.60%, while Brinson-Fachler credits it +0.32%, because financials lagged the 9.20% benchmark. Same total, opposite verdict on that one decision.

Second, and this is the senior's actual job: a decomposition is not reportable until you say which convention produced it. Presenting scheme B's \$7.00m selection number without that sentence is not a lie, but it is an argument dressed as a measurement.

## Costs are a term, not a footnote

Attribution done on gross P&L answers a question nobody is paying for. The money that reaches the investor is net, so net is what has to be decomposed, and the costs of getting in and out belong inside the sum with the factors.

The right frame is Perold's (1988) **implementation shortfall**: the gap between the paper portfolio you decided to hold at the moment of the decision and the real portfolio you ended up with. It splits into delay (the price moved between decision and order), spread (you crossed), impact (your own trading moved the price), and opportunity cost (the part you never filled), plus commissions and fees.

#### Worked example 3: \$14.0m gross, \$1.7m net on a \$250m book

A \$250m market-neutral book turned over \$6.0b of two-way notional over the year and earned \$14.0m gross, a 5.60% gross return.

| Cost term | bps of notional | Dollars |
| --- | --- | --- |
| Commissions and fees | 1.0 | \$0.60m |
| Spread paid | 5.0 | \$3.00m |
| Market impact | 11.0 | \$6.60m |
| Delay | 3.5 | \$2.10m |
| **Total shortfall** | **20.5** | **\$12.30m** |

Net P&L is 14.00 - 12.30 = \$1.70m, a 0.68% return. Market impact alone consumed 47.1% of the gross P&L, and all costs together consumed 87.9%.

A gross attribution on this book reports a 5.60% specific return and a research team that looks excellent. A net attribution reports 0.68% and a strategy whose binding constraint is its own footprint. The brief version of the senior's point: a strategy that is profitable gross and flat net does not have a cost problem, it has an attribution problem, because the decomposition it has been showing excludes the largest term.

The scaling makes it worse in a specific way. Gross P&L is roughly linear in book size, and so are commissions, spread and delay. Impact is not: under a square-root law it scales with the 1.5 power of size at constant turnover. Double this book to \$500m and gross goes to \$28.00m, the linear costs go to \$11.40m, and impact goes to \$18.67m, for total costs of \$30.07m and a net of **negative \$2.07m**. The same strategy, the same signal, twice the money, and the attribution flips sign. [Optimal execution](/blog/trading/math-for-quants/dynamic-programming-optimal-execution-math-for-quants) is the discipline that attacks the impact term directly.

## The unexplained residual, and how big it is allowed to be

The residual is the number a senior is actually asked about, and there are two separate questions hiding in it.

**Is the residual large relative to its own noise?** This is the one that gets skipped. Take worked example 1: the specific return was +1.05% on a book whose specific volatility is 4.0% a year. The realised information ratio of the specific part is ${1.05/4.0 = 0.26}$, and over one year the t-statistic is the same 0.26, because $t = \mathrm{IR}\sqrt{T}$ and $T = 1$.

![Number line of annual specific return with a shaded one-standard-deviation band from -4.0% to +4.0% and the realised +1.05% marked inside it.](/imgs/blogs/pnl-attribution-math-for-quants-4.webp)

Two ways to feel how weak that is. If the true specific return were zero, a normal draw with 4.0% volatility lands further from zero than 1.05% about 79% of the time: four years in five, a manager with literally no stock-picking skill reports a specific return at least this big in absolute value. And reaching t = 2 at this information ratio takes ${(2/0.26)^2}$ years, which is about 58 years of identical performance. The \$5.25m is real money that really arrived. It is not evidence of skill, and saying so in the meeting is the single highest-value sentence in this post.

This is the honest null result, and it is worth stating as a general rule: a clean attribution that closes to the penny tells you nothing whatsoever about whether the residual is signal. The identity and the inference are separate machines, and only one of them is running.

**Is the residual the size the model expected?** This is a different and more answerable question, and it is about the risk model rather than the manager. Compare the realised specific volatility against the volatility the model predicted, period by period. The ratio of the two is the **bias statistic**, and a well-calibrated model produces one near 1. With $T$ observations the approximate 95% band around 1 is ${1 \pm \sqrt{2/T}}$, so on 250 daily observations anything between roughly 0.91 and 1.09 is unremarkable.

A bias statistic well above 1 says the residual is bigger than the model thinks it can be, which usually means a factor is missing: the book has a real, systematic exposure that the model has no column for, and it is being booked as alpha. A bias statistic well below 1 means the model is over-fitting the residual, often because exposures were estimated on the same window they are being applied to. Both are risk-model problems presenting as attribution problems, and [the covariance cleaning post](/blog/trading/math-for-quants/random-matrix-theory-covariance-cleaning-math-for-quants) covers why estimated covariances misbehave in exactly this direction.

## Multi-period linking: why the contributions stop adding

Everything so far was one period. Over several periods it breaks, and the reason is simple: **returns compound, contributions do not add.**

If the portfolio returns $r_{p,1}, r_{p,2}, \ldots$ and the benchmark $r_{b,1}, r_{b,2}, \ldots$, the total active return is

$$\prod_t (1 + r_{p,t}) \;-\; \prod_t (1 + r_{b,t}),$$

which is not the sum of the single-period active returns $r_{p,t} - r_{b,t}$. The reason is economic, not algebraic: money made in the first quarter is invested in the second, so an early contribution earns a return on itself and a late one does not. The naive sum ignores that, and its error is not a rounding artefact you can shrug at.

#### Worked example 4: three quarters, a \$719,200 gap on \$400m

| Quarter | Portfolio | Benchmark | Arithmetic active |
| --- | --- | --- | --- |
| Q1 | +6% | +3% | +3.00% |
| Q2 | -4% | -5% | +1.00% |
| Q3 | +8% | +6% | +2.00% |

The naive sum is +6.0000%, which on \$400m is \$24,000,000. The truth: the portfolio compounded to $1.06 \times 0.96 \times 1.08 - 1 = 9.9008\%$ and the benchmark to $1.03 \times 0.95 \times 1.06 - 1 = 3.7210\%$, an active return of **+6.1798%**, or \$24,719,200. The naive sum is short by \$719,200, and it is short for a structural reason, not a random one.

Carino (1999) fixes this with a logarithmic scaling coefficient. Define, for each period and for the whole period,

$$k_t = \frac{\ln(1+r_{p,t}) - \ln(1+r_{b,t})}{r_{p,t} - r_{b,t}}, \qquad k = \frac{\ln(1+R_p) - \ln(1+R_b)}{R_p - R_b},$$

and scale each period's contribution by $k_t/k$. The adjusted contributions then sum to the total active return exactly, because $\sum_t k_t A_t$ telescopes into $\ln(1+R_p) - \ln(1+R_b) = k(R_p - R_b)$. Applied here:

| Quarter | Arithmetic | $k_t/k$ | Linked | Dollars |
| --- | --- | --- | --- | --- |
| Q1 | +3.0000% | 1.021899 | +3.0657% | \$12.2628m |
| Q2 | +1.0000% | 1.118137 | +1.1181% | \$4.4725m |
| Q3 | +2.0000% | 0.997983 | +1.9960% | \$7.9839m |
| **Total** | **+6.0000%** | | **+6.1798%** | **\$24.7192m** |

![Grouped columns comparing three quarterly contributions summed naively to 6.0000% against Carino-linked contributions summing to 6.1798%, with the \$719,200 gap marked.](/imgs/blogs/pnl-attribution-math-for-quants-5.webp)

Note which quarter gained most. Q2's contribution was scaled up 11.8%, because it was earned when the book was smaller after a losing quarter, so each point of it did more work. Menchero (2000) solves the same closure problem with an optimised linking coefficient chosen to minimise the residual distortion across periods rather than to follow a logarithm, and it is the scheme most commercial systems ship. Both close exactly. They allocate the smoothing differently, which is the interaction problem again, arriving from a different direction.

The same machinery links factor contributions, not just allocation and selection. Apply the coefficients to each factor's per-period contribution and the multi-period factor attribution closes too.

## Attribution is an argument, not a report

Every choice above moves money between people. The interaction convention decides whether the stock picker earned \$4.00m or \$7.00m. Gross versus net decides whether the research team produced 5.60% or 0.68%. The factor list decides how much of the year is called alpha at all: add a momentum factor to the model and a momentum-tilted book's residual shrinks, without one position changing.

So the attribution scheme is chosen once, in advance, by someone who does not benefit from the choice, and it is written down. The failure mode is a shop that re-picks the convention after seeing the numbers, which is not fraud and is not detectable from any single report, and which reliably produces attributions where everyone did well.

## Common misconceptions

**"The attribution closed, so the alpha is real."** Closing is algebra. You defined the residual as the leftover, so it is the leftover, and it would be the leftover for a random portfolio. Worked example 1 closes to the penny and its residual carries a t of 0.26. The sum check and the significance test share no machinery.

**"The residual is the alpha."** The residual is everything the model does not span. That includes skill, but also every factor you left out, every exposure measured with error, and the gap between the exposures you had at the start of the period and the ones you actually carried through it. A book with a genuine industry tilt and no industry factor in the model reports that tilt as alpha every time. This is why [choosing what goes in the regression](/blog/trading/math-for-quants/regression-ols-gls-regularized-math-for-quants) is a substantive decision and not plumbing.

**"Attribution is arithmetic, not a judgement call."** The interaction rectangle disproves this in one figure. Three schemes, three stories, one P&L, all correct. Anyone who says their attribution is objective has simply not noticed which convention their system defaults to.

**"A bigger residual is better."** A large unexplained residual is as likely to mean a broken risk model as an unusual manager, and the bias statistic distinguishes the two in a way that staring at the residual cannot.

## Sources and further reading

- Brinson, G. P., Hood, L. R. and Beebower, G. L. (1986). "Determinants of Portfolio Performance." *Financial Analysts Journal* 42(4), 39-44. The origin of the allocation / selection / interaction decomposition.
- Brinson, G. P. and Fachler, N. (1985). "Measuring Non-US Equity Portfolio Performance." *Journal of Portfolio Management* 11(3), 73-76. The variant that measures allocation against the benchmark's own total return.
- Carino, D. R. (1999). "Combining Attribution Effects Over Time." *Journal of Performance Measurement* 3(4), 5-14. The logarithmic linking coefficient used in worked example 4.
- Menchero, J. (2000). "An Optimized Approach to Linking Attribution Effects over Time." *Journal of Performance Measurement* 5(1), 36-42. The optimised alternative shipped in most commercial systems.
- Perold, A. F. (1988). "The Implementation Shortfall: Paper Versus Reality." *Journal of Portfolio Management* 14(3), 4-9. Where the cost decomposition comes from.
- Grinold, R. C. and Kahn, R. N. (2000). *Active Portfolio Management*, 2nd ed. McGraw-Hill. Chapters on performance analysis and the information ratio; the source for treating the residual as a statistical object rather than a number.
- Bacon, C. R. (2008). *Practical Portfolio Performance Measurement and Attribution*, 2nd ed. Wiley. The reference implementation-level treatment of every scheme above.

## In the interview room and on the desk

The question arrives as a statement about your own work: *"your strategy made 8% last year, where did it come from?"* It is not a trick, and it is not really about the strategy. It is a test of whether you think in decompositions.

The strong answer moves in a fixed order. **Separate factor from specific first**, before anything else, and give both numbers: "8.00% total, 6.95% from factor exposures, 1.05% specific." That single sentence sorts candidates, because most people start describing the signal and never mention that most of the return was beta they did not intend to own. **Then volunteer the residual's noise before being asked**: "1.05% against 4.0% specific vol is a t of 0.26 over one year, so I cannot call it skill on this sample." Volunteering it is the whole move. An interviewer who has to extract that from you has learned something different about you than one who hears it unprompted. **Then name the costs** as a line in the decomposition rather than a haircut applied afterwards, and **then say which convention** you used for anything that involved a benchmark: "this is Brinson-Fachler with interaction in selection, so the selection number is flattered by the cross term."

The trap is presenting a decomposition without naming the convention. It is not a small omission, because it is exactly the omission that makes a report into an argument. A candidate who says "selection contributed \$7.0m" sounds precise and is making an unexamined choice worth \$3.0m of that number. The follow-up is always some form of "where did the interaction term go?", and there is no recovering from not knowing that there was one. The second trap is the naive multi-period sum: if you are asked to link four quarters and you add them, you have said you have never reconciled an attribution to an audited return.

Citadel, Two Sigma and Point72 weight this heavily, as does any multi-strategy platform, because attribution there is not reporting, it is how capital is allocated between pods and how the pods are paid. Allocator-facing seats at fund-of-funds and pensions weight it for the mirror-image reason: they are the ones reading the attribution, and they are paid to notice which convention produced it.
