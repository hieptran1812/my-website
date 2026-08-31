---
title: "Optimal stopping: when to take the trade you have"
date: "2026-08-31"
publishDate: "2026-08-31"
description: "A whole class of trading decisions is not 'what is this worth' but 'is this offer good enough, or do I wait?'. That is optimal stopping, it has exact answers you can derive at a whiteboard, and the answer changes completely depending on whether you can see the prices or only their ranks."
tags: ["optimal-stopping", "secretary-problem", "reservation-price", "execution", "dynamic-programming", "optional-stopping-theorem", "quant-interview", "jane-street", "probability", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 21
---

> [!important]
> **TL;DR:** Some decisions are not valuation problems, they are stopping problems. The question is not what an offer is worth but whether it beats the option to keep looking, and that option has a price you can compute.
>
> - In the classic secretary problem you see only ranks, cannot go back, and want the single best. The answer is: reject the first 1/e of the sequence, then take the first record. It wins **37.1%** of the time on 100 candidates, and that barely moves with the length of the sequence.
> - Change one assumption, that you can see the actual values and know the distribution they come from, and the rule stops being a proportion and becomes a **reservation price that falls as the deadline approaches**. The best-choice win rate jumps from 36.8% to about **58.0%**.
> - On a \$5m fill with 20 quotes to come, the reservation price starts at 2.84 bps below mid and decays to 7.00 bps by the second-last quote. Following it costs \$1,401 against \$3,500 for taking the first quote you see.
> - Stopping too early is usually the more expensive mistake. On that book it costs \$599 a trade against \$244 for stopping too late, roughly two and a half times as much.
> - None of this creates edge in a fair game. That is a different theorem, and it goes the other way.

## The offer in front of you, or the one you hope for

A trader has \$50m of a corporate bond to move before the close. A counterparty comes back at 4 basis points below mid. A *basis point*, or bp, is one hundredth of a percent, so 4 bps on \$50m is \$20,000 of cost. Is that a good print?

You cannot answer that by valuing the bond. The bond is worth what it is worth. The real question is whether 4 bps beats the distribution of what will arrive over the next four hours, given that shopping the block harder leaks information and that at 4pm you have to be done. That is not a valuation problem. It is a **stopping problem**: offers arrive one at a time, you can take one or pass, passing is usually irreversible, and there is a deadline.

Stopping problems are everywhere on a desk and almost nowhere in an introductory finance course. They also have exact answers, which is why they show up in quant interviews so often. The rest of this post derives two of those answers and shows what the gap between them is worth in money.

![Timeline of 100 arriving quotes split into a look phase covering quotes 1 to 37 where nothing is accepted and a leap phase covering quotes 38 to 100 where the first record is taken, with the two outcomes labelled 37.1 percent win and 37 percent forced](/imgs/blogs/optimal-stopping-secretary-when-to-take-the-trade-math-for-quants-1.webp)

The figure above is the mental model for the simplest version. Every stopping rule of this family splits the sequence into a phase where you only learn and a phase where you can commit, and the entire art is deciding where to put the line.

## Foundations: what makes a decision a stopping problem

You need four ingredients, and every one of them changes the answer.

**Sequential arrival.** Offers arrive one at a time, and you never see them all laid out. That is what makes this hard.

**A decision at each arrival.** You accept and the game ends, or you reject and move on. There is no "let me think about it".

**No recall.** A rejected offer is gone. In markets that is close to literally true: a dealer's quote is live for seconds, and coming back an hour later to take that 4 bps gets you a worse price, because the dealer now knows you have been shopping.

**A deadline.** The sequence has a known length or a known end time. Without one there is no urgency and the problem dissolves.

The thing most people skip is **what you can observe**. In one version you learn only *ranks*: whether this offer is the best you have seen so far, and nothing else. In another you see the actual number and know the distribution it came from. Those are different problems with different answers, and confusing them is the single most common mistake.

Two terms before we start. **Mid** is the midpoint between the best bid and the best offer, the closest thing to a fair price at that instant, and execution cost is quoted as distance from it. A **reservation price** is the worst price at which you are willing to deal right now. Everything below is about computing that number instead of guessing it.

### The everyday version: house hunting

Strip out the finance and this is apartment hunting in a tight market. You view flats one at a time, good ones go the same day, and you have six weeks. Early viewings are worth almost nothing as decisions and almost everything as education, because you do not yet know what the distribution looks like. Everyone who has done this has felt both failure modes: taking the third flat out of fatigue and then seeing two better ones in week five, or holding out for perfect and signing for whatever is left. The maths below says exactly how long to look, and the answer is roughly the same whether you have twenty viewings or two hundred.

## The secretary problem, from zero

Here is the canonical statement. There are $n$ candidates arriving in a uniformly random order, so no position is more likely than any other to hold the best one. After each interview you can rank that candidate against everyone you have already seen, but you learn nothing about absolute quality and nothing about those still to come. You hire or reject on the spot, with no recall, and you win only if you hire the single best of all $n$.

The name is an accident of history. Ferguson's 1989 survey traces the problem to Martin Gardner's Mathematical Games column of February 1960, where it was posed as the "game of googol": slips of paper with arbitrary numbers written on them, turned over one at a time, stop when you think you have the largest.

The rank-only restriction is doing enormous work. If you could see values, you could say "that is a great number, I'll take it". Seeing only ranks means the sole signal you ever get is binary: this one is a record, or it is not. That forces the shape of the answer. Any sensible rule must be **reject the first ${r-1}$ candidates no matter what, then accept the first candidate who beats all of them.** The first phase is pure sampling, used only to build a benchmark, and the only free parameter is where to cut.

### Deriving the 1/e rule

Fix the cutoff at $r$, meaning you reject candidates 1 through ${r-1}$ and then accept the first record. You win exactly when two things happen together for some position $k$ at or after $r$: candidate $k$ is the best of all $n$, and the best of the first ${k-1}$ candidates fell inside the reject window. The second condition is what stops the rule from firing early on some lesser record.

Those two probabilities are ${1/n}$ and $(r-1)/(k-1)$ respectively, and they are independent, so

$$
P(r) = \frac{r-1}{n} \sum_{k=r}^{n} \frac{1}{k-1}
$$

Every symbol: $n$ is the number of candidates, $r$ is the first position at which you are willing to accept, and $k$ indexes the position where the overall best sits.

To find the best $r$, write $c = (r-1)/n$ for the fraction you reject and let $n$ grow. The sum becomes an integral:

$$
P(c) = c \int_{c}^{1} \frac{dx}{x} = -c \ln c
$$

Differentiate: the derivative of $-c \ln c$ is $-\ln c - 1$, which is zero when $\ln c = -1$, so the optimal fraction is $c = e^{-1}$, about 0.368. Substituting back, the win probability at that cutoff is also $e^{-1}$.

That coincidence is the memorable part. **You reject 1/e of the sequence and you win 1/e of the time.** Nothing else in the problem produces the same constant twice, and saying it out loud is what tells an interviewer you derived it rather than recalled it.

![Curve of the probability of landing the single best of 100 quotes against how many quotes are rejected first, peaking at 37.1 percent when 37 are rejected, with a broad flat region from 30 to 45 and a horizontal line marking the 36.8 percent large-n limit](/imgs/blogs/optimal-stopping-secretary-when-to-take-the-trade-math-for-quants-2.webp)

Two things in that curve matter more than the peak. First, the peak is **broad**: reject anywhere between 30 and 45 of 100 and you stay above 36%, so the rule is forgiving of a bad estimate of the sequence length. Second, it is steep on the left. Rejecting only 10 first drops you to 23.5%, and taking the first quote outright gives you 1%. Under-sampling is punished much harder than over-sampling.

The win rate is also remarkably stable in $n$. With 10 candidates the optimum is 39.9%, with 20 it is 38.4%, with 100 it is 37.1%, and with 1,000 it is 36.8%. It approaches 1/e from above and it gets there fast, which is why the rule is usually quoted as a single number.

#### Worked example 1: the 1/e rule on 100 counterparties for a \$50m block

You have \$50m of a bond to sell, so one bp of price is \$5,000. Suppose 100 counterparties will show you a bid over the session, each somewhere between 2 and 12 bps below mid, and you can only tell whether a bid is the best you have seen, not how good it is absolutely.

The prize is real. The expected best of 100 draws spread evenly over that band sits at about 2.1 bps, roughly \$10,500 of cost, against \$35,000 for an average bid at 7.0 bps. The gap between the best bid of the day and a typical one is about \$24,500 on this single block.

The rule: reject the first 37 bids, remember the best of them, then hit the first bid that beats it. Evaluating the formula at ${r = 38}$:

$$
P(38) = \frac{37}{100}\left(\frac{1}{37} + \frac{1}{38} + \cdots + \frac{1}{99}\right) = 0.37 \times 1.0028 = 0.3710
$$

So 37.1%, against 1% for closing your eyes and hitting the first bid. That is a 37-fold improvement from a rule you can state in one sentence.

Now the honest part, which is the part interviews probe. The rule fires only if some bid after position 37 beats the best of the first 37, and that fails exactly when the overall best bid sat inside the reject window, with probability 37/100. **So 37% of the time this rule accepts nothing at all and dumps you on bid number 100**, a random draw worth \$35,000 in expectation. You win 37.1% of the time and get forced 37% of the time. Those two numbers being nearly equal is not a coincidence, it is the same cutoff seen from both sides.

*The intuition: the 1/e rule maximises the probability of the very best outcome and pays for that with a large chance of the worst one. If you care about expected money rather than a trophy, it is the wrong objective function.*

## The variant that matches trading

On a desk you almost never face the rank-only problem. You see the actual price, and you have traded this bond enough to know roughly how wide the quotes come. That extra information changes the structure of the answer, not just its parameters.

![Comparison table of rank-only versus full-information stopping showing what you observe, the optimal rule, the chance of the best at large n of 36.8 percent versus 58.0 percent, and what each objective optimises](/imgs/blogs/optimal-stopping-secretary-when-to-take-the-trade-math-for-quants-3.webp)

With values visible and the distribution known, the optimal rule is no longer "reject a fixed proportion". It is a **threshold**: accept anything above a cutoff, where the cutoff depends on how many chances remain. Gilbert and Mosteller worked this case out in 1966. Even keeping the demanding best-or-nothing objective, the win rate rises from 36.8% to a limit of about 58.0%, because you can now recognise an excellent draw immediately instead of waiting for something to prove itself by being a record.

The deeper change is that once values are visible you can switch to the objective a trader actually has, which is the expected quality of the fill rather than the probability of the single best. That objective has a clean recursive answer.

## The trading version: a reservation price that decays

Let $V_t$ be the value of playing optimally with $t$ chances still remaining, measured in the same units as the offers. When the last offer arrives you have no choice, so $V_1 = E[X]$, the mean of the distribution. With $t$ chances left you face a draw $X$ and can either take it or fall back on optimal play with $t-1$ remaining, which is worth $V_{t-1}$. So

$$
V_t = E\left[\max\left(X,\, V_{t-1}\right)\right]
$$

That single line is the whole method. It is the same backward induction that drives [optimal execution schedules](/blog/trading/math-for-quants/dynamic-programming-optimal-execution-math-for-quants), applied to a stop-or-continue decision instead of a how-much-to-trade decision.

Read the recursion and the trading rule falls out: **accept the current offer if it beats $V_{t-1}$, refuse otherwise.** So $V_{t-1}$ *is* the reservation price. It is not a preference or a limit somebody set, it is the market value of the option to keep looking, and it is computable.

Because $V_t$ increases with $t$ (more chances can never be worth less), the reservation price falls as the deadline approaches. Early in the day you can afford to be fussy, because refusing costs you almost nothing. At the last quote you take whatever arrives.

![Line chart of the reservation price on a five million dollar sale, rising from 2.84 basis points with 20 quotes to come to 7.00 basis points with 2 to come and 12 basis points at the last quote, annotated with dollar equivalents](/imgs/blogs/optimal-stopping-secretary-when-to-take-the-trade-math-for-quants-4.webp)

#### Worked example 2: the reservation price on a \$5m sale with 20 quotes

Take a concrete, deliberately simple book. You need to sell \$5m of something. Over the day 20 quotes will arrive, each independently spread evenly between 2 and 12 bps below mid. No recall, and you must be flat by the close. One bp on \$5m is \$500, so the band runs from \$1,000 to \$6,000 of cost.

Rescale to a quality score $X$ between 0 and 1, where the cost in bps is ${12 - 10X}$. For a uniform draw and a threshold $c$, a short integral gives $E[\max(X, c)] = c^2 + (1 - c^2)/2$, so the recursion collapses to

$$
V_t = \frac{1 + V_{t-1}^{2}}{2}, \qquad V_1 = \frac{1}{2}
$$

Turn the handle. $V_1 = 0.5000$, then $V_2 = 0.6250$, $V_3 = 0.6953$, and so on up to $V_{19} = 0.9164$ and $V_{20} = 0.9199$.

Now read off the reservation price, remembering that with $t$ quotes to come you compare against $V_{t-1}$:

| Quotes still to come | Reservation price | Cost in bps | Cost on \$5m |
| --- | --- | --- | --- |
| 20 | $V_{19} = 0.9164$ | 2.84 | \$1,418 |
| 10 | $V_{9} = 0.8498$ | 3.50 | \$1,751 |
| 2 | $V_{1} = 0.5000$ | 7.00 | \$3,500 |
| 1 | none | up to 12.00 | up to \$6,000 |

With 20 quotes ahead you refuse anything worse than 2.84 bps, and refusing is cheap because 19 more chances are coming. Halfway through the day the same book should be hitting 3.50 bps without hesitation. With two quotes left the bar has collapsed to the unconditional mean, because the alternative to accepting is a coin flip on the last quote.

The value of the whole day is $V_{20} = 0.9199$, which is 2.80 bps, or **\$1,401 of expected cost**. Compare that against accepting the first quote that shows up, which costs the mean 7.00 bps, or \$3,500.

*The intuition: the reservation price is the value of the option to keep looking, and options with less time left are worth less. Any execution rule with a fixed limit price is implicitly claiming that option value does not decay, which is false by construction.*

## What stopping too early and too late actually costs

Traders rarely follow a decaying threshold. They follow a fixed limit, and the interesting question is what that habit costs. Both directions of error are expensive, and they are not symmetric.

#### Worked example 3: four policies on the same \$5m book

Same book: \$5m, 20 quotes, uniform between 2 and 12 bps, forced fill at the close. Four rules.

| Policy | What it does | Expected cost | On \$5m | Extra versus optimal |
| --- | --- | --- | --- | --- |
| Optimal | Reservation price decays from 2.84 to 7.00 bps | 2.80 bps | \$1,401 | baseline |
| Stop too late | Hold the opening 2.84 bps bar all day, forced fill at the close | 3.29 bps | \$1,645 | \$244 |
| Stop too early | Flat limit: hit anything at 6.00 bps or better | 4.00 bps | \$2,000 | \$599 |
| No rule at all | Hit the first quote that arrives | 7.00 bps | \$3,500 | \$2,099 |

The arithmetic behind the two middle rows is worth seeing. Holding out for 2.84 bps all day, a quote clears that bar 8.36% of the time, so over the 19 quotes before the last one you fail to fill 19.0% of the time and get dumped on a random last quote at 7.00 bps. The other 81.0% of the time you fill at an average of 2.42 bps. That blends to 3.29 bps, or \$1,645. Subtract the \$1,401 baseline and holding out costs \$244 a trade.

The flat 6.00 bps limit fills almost immediately, at an average of 4.00 bps once you condition on clearing the bar. That is \$2,000, and \$2,000 minus \$1,401 is \$599 a trade.

So **stopping too early costs about two and a half times what stopping too late costs on this book**, and both are dwarfed by having no rule at all. Scale it up: a desk doing 250 of these blocks a year gives up roughly \$149,750 to the flat 6 bps habit and roughly \$61,000 to stubbornness. Neither number appears in any P&L line, because the counterfactual is invisible. That is precisely why it persists.

*The intuition: the cost of a wrong stopping rule is a slow leak, not a blowup, and slow leaks survive because nothing on a risk report is shaped like them.*

### Where the model breaks on a real desk

The uniform, independent, known-distribution setup above is a teaching device, and three of its assumptions fail in ways that matter.

**Quotes are not independent of your own behaviour.** Shopping a block to 20 counterparties leaks the direction and size of your interest. The twentieth quote is worse than the first partly because the market now knows you are a seller. In the model, waiting is free; in reality, waiting has a cost that grows with how many people you have asked. That pushes the whole reservation-price schedule down.

**The distribution moves.** The band you calibrated on last month's quotes is not this afternoon's, especially around a data release. A threshold rule built on a stale distribution is a confident, precise, wrong number, which is the topic of [estimation error and why it bites](/blog/trading/math-for-quants/estimators-bias-variance-consistency-math-for-quants).

**The deadline is not the clock.** The threshold depends on *chances remaining*, not on time remaining. If liquidity dries up at 2pm and no more quotes are coming, your deadline effectively arrived early and your reservation price should have collapsed with it. Desks that hard-code the schedule to the clock get this backwards on exactly the days it costs the most.

## Why "quit while you are ahead" is not a stopping rule

Everything above extracts value because the offers genuinely differ and you are selecting among them. A **martingale**, a process whose expected next value equals its current value, offers nothing to select: every future price is, in expectation, today's price. Doob's optional stopping theorem makes this precise, and under its conditions the expected value at any stopping time equals the starting value. No rule built only on information available at the time can turn a fair game into a profitable one. The reason "quit while you are ahead" feels like it works is the classic counterexample: in a symmetric random walk, "stop the first time you are up \$1" succeeds with probability 1, which looks like a violation. It is not, because that stopping time has infinite expected duration and unbounded interim loss, so the theorem's hypotheses simply do not hold. On a desk the unbounded interim loss has a name, which is your risk limit, and it binds long before the strategy pays. The measure-theoretic machinery behind this, filtrations and adapted processes and stopping times, is developed in [martingales and the risk-neutral measure](/blog/trading/math-for-quants/martingales-risk-neutral-measure-math-for-quants) and [filtrations and no look-ahead](/blog/trading/math-for-quants/filtrations-no-lookahead-math-for-quants). The short version to carry around: optimal stopping is about choosing well among unequal offers, optional stopping says you cannot choose your way out of a fair game, and they are near-opposite statements with confusingly similar names.

## Common misconceptions

**"The 1/e rule is the answer to every stopping problem."** It answers exactly one: ranks only, no recall, no information about the distribution, and the objective of landing the single best. Relax any one of those four and the answer changes shape. Values visible turns a proportion rule into a threshold rule. Recall allowed makes the problem trivial. A different objective, like expected value, changes it again. Quoting 37% for a problem where you can see the numbers is the most common way to look rigorous while being wrong.

**"Waiting longer is always better."** The option to keep looking is worth $V_{t-1}$, and that value falls to the unconditional mean as the deadline closes. Holding the opening threshold all day cost \$244 per \$5m block above, or about \$61,000 a year over 250 blocks. Patience is a position, and it decays.

**"This is the optional stopping theorem."** No. Optimal stopping is about selection among genuinely different offers, and it creates value. Optional stopping says that in a fair game there is nothing to select, and no stopping rule can manufacture an edge. Same word, opposite direction.

**"The secretary rule maximises the price you get."** It maximises the probability of the single best and is indifferent to how bad the failures are. In worked example 1 it left you on the last bid 37% of the time, at an expected \$35,000 on a \$50m block. Objectives are not interchangeable, and choosing one is a modelling decision you should make out loud.

**"No recall is a technicality."** It is the entire problem. If you could go back to any earlier quote, the optimal rule would be to watch all 100 and then take the best, winning 100% of the time. Everything interesting here comes from irreversibility.

**"Threshold rules are the same as limit orders."** A limit order is a fixed threshold. The optimal rule is a threshold that moves with the number of chances remaining. If you use the first as an implementation of the second, you are systematically too fussy early and too generous late, which is a specific and measurable cost, not a rounding error.

## In the interview room and on the desk

**How the question is actually asked.** Nobody says "solve the secretary problem". They say: *"I show you 100 numbers one at a time. You must stop on one and you cannot go back. Maximise the probability you stop on the largest. What do you do?"* The interviewer is watching whether you set up a recursion or reach for a remembered constant.

**What a strong answer contains, in order.** First, pin down the information structure before proposing anything: do I see the values or only whether each is a record, is there recall, and am I maximising the chance of the best or the expected value? Asking this is half the signal, because it is the fork the whole problem turns on. Second, argue the shape of the optimal policy: with ranks only, the sole signal is "record or not", so any rule reduces to reject the first ${r-1}$, then take the first record. Third, write the win probability and say where each factor comes from, ${1/n}$ that the overall best sits at position $k$, and $(r-1)/(k-1)$ that the best of the first ${k-1}$ landed in the reject window. Fourth, take the continuum limit to $-c \ln c$, differentiate, and land on $c = e^{-1}$, noting that the win probability equals the same constant. Fifth, sanity-check out loud: 37 of 100, 37.1%, and the peak is flat enough that 35 or 40 barely differ.

**The follow-up and the trap.** The push is always the same: *"now suppose you can see the numbers, and they are uniform on the unit interval."* Answering 1/e here is the trap, and it is the most common failure on this question. That is a different and easier problem: set $V_1 = 1/2$ and iterate $V_t = (1 + V_{t-1}^2)/2$, accept whenever the draw beats $V_{t-1}$, and note the threshold falls as the deadline nears. Say that the best-choice win rate rises to about 58.0% with full information. The desk version of the same trap is a flat limit price held all day, which is the rank-only rule in disguise.

**Who weights this most.** Jane Street and SIG lean on it hardest, because it tests sequential decisions under a deadline rather than a formula you can memorise. It also shows up wherever the interview is really about execution, since a reservation price that decays into the close is the same object.

## Sources and further reading

- Thomas S. Ferguson, ["Who Solved the Secretary Problem?"](https://projecteuclid.org/journals/statistical-science/volume-4/issue-3/Who-Solved-the-Secretary-Problem/10.1214/ss/1177012493.full), *Statistical Science*, Vol. 4, No. 3 (August 1989), pp. 282-289. The definitive history, including the trail back to Martin Gardner's February 1960 *Scientific American* column and the game of googol.
- John P. Gilbert and Frederick Mosteller, ["Recognizing the Maximum of a Sequence"](https://www.tandfonline.com/doi/abs/10.1080/01621459.1966.10502008), *Journal of the American Statistical Association*, Vol. 61, No. 313 (1966), pp. 35-73. The full-information variant. The asymptotic win rate of 0.580164 quoted above was derived explicitly later, by [Gnedin and Miretskiy](https://arxiv.org/abs/math/0510568).
- Y. S. Chow, Herbert Robbins and David Siegmund, *Great Expectations: The Theory of Optimal Stopping* (Houghton Mifflin, 1971). The standard reference for the general theory in discrete time.
- Goran Peskir and Albert Shiryaev, *Optimal Stopping and Free-Boundary Problems* (Birkhäuser, 2006). The continuous-time treatment, and the bridge to the free-boundary problems that price American options.

Every other number in this post sits inside a labelled hypothetical example: the \$50m block, the \$5m sale, the 20 quotes and the 2 to 12 bps band are chosen for arithmetic that a reader can check by hand, not drawn from any live market.
