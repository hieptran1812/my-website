---
title: "CVaR, Expected Shortfall, and Asking How Bad Is Bad"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Value-at-Risk tells you how often you breach a loss line but goes silent on how deep the fall is; expected shortfall answers the question VaR refuses to by averaging the worst cases, and it is the coherent tail measure that rewards diversification."
tags: ["risk-management", "expected-shortfall", "cvar", "value-at-risk", "tail-risk", "coherent-risk-measure", "sub-additivity", "position-sizing"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **The one idea:** Value-at-Risk tells you *how often* your loss crosses a line; Conditional VaR — expected shortfall — tells you *how bad it is on average once you cross it*, which is the number that actually decides whether you survive.
> - VaR is a **single point on the loss tail**: the loss you breach 1% (or 5%) of the time. It says nothing about the days that are worse than that point — and those are the days that end careers.
> - **Expected shortfall (ES) is the average of every loss past the VaR cutoff.** Build it from scratch: sort your losses, find the worst (1−X)% of them, average them. That average is the ES.
> - Two books can have the **identical VaR and wildly different ES** — same cliff edge, very different drop. ES sees the depth of the tail; VaR is blind to it.
> - ES is a **coherent** risk measure: it is *sub-additive*, so combining two books never reports more risk than the two stood alone. VaR can violate this and *punish* diversification — a mathematical scandal that pushed regulators to replace it.
> - The practical upshot is the whole series in one sentence: **size to the loss you must survive, not the loss you usually take.** That loss is the ES, and budgeting to it keeps you in the game when the tail finally arrives.

A risk manager at a large bank once told me about the morning his VaR model passed every test and his desk still lost a year of profit before lunch. The 99% one-day VaR on the book was about \$8 million. That is the number the firm reported to regulators, the number on the daily risk sheet, the number everyone trusted. It said, in plain English, "on 99 days out of 100, you will not lose more than \$8 million." And it was *true*. The model was well-calibrated; the breaches happened roughly as often as advertised. The problem was never the frequency. The problem was that on the one-in-a-hundred day the model went politely silent about, the book did not lose \$8 million. It lost \$60 million. VaR had told him exactly how often he would fall off the cliff. It had told him *nothing* about how far down the cliff went.

That gap — between *how often* and *how far* — is the single most dangerous blind spot in the most widely used risk number on the planet. [Value-at-Risk, and exactly how it lies](/blog/trading/risk-management/value-at-risk-and-exactly-how-var-lies), the post just before this one, took VaR apart: what it claims, how it is computed three ways, and the precise ways it misleads. The deepest of those failures, the one this post is built to fix, is that VaR is a *threshold*, not a *magnitude*. It marks the edge of the bad region and then refuses to look inside it. Everything that lives past the cutoff — the moderate breaches, the severe breaches, the catastrophic once-a-decade breach that takes the firm down — collapses into a single sentence: "worse than \$8 million." A \$9 million day and a \$90 million day are, to VaR, the same event.

Expected shortfall is the answer. It asks the question VaR will not: *given* that you breach the VaR, *how bad is it, on average?* Instead of reporting the edge of the tail, ES reaches past the edge, gathers up every outcome in the bad region, and averages them. It tells you not the loss you barely avoid but the loss you should *expect* on the days you do not. This single change — from "the threshold" to "the average past the threshold" — fixes nearly everything wrong with VaR, and it does so with math elegant enough that the world's bank regulators rewrote their rulebooks around it.

That contrast is Figure 1, and it is the picture to keep in your head for the rest of the post. It shows one loss distribution for a \$10,000,000 book. The amber dashed line is the VaR cutoff — here a 95% VaR of \$122,700 — which tells you *how often*: you breach it on 5 days in 100. The solid red line, further out into the loss at \$187,226, is the ES — the average of the whole red tail beyond the cutoff, *how bad*. The VaR marks where the cliff begins; the ES tells you the average height of the fall, and on this book that fall averages over 50% deeper than the cutoff itself.

![Profit and loss distribution for a ten million dollar book with the amber VaR cutoff line marking the worst five percent and a solid red expected shortfall line at the average of the shaded loss tail beyond it](/imgs/blogs/cvar-expected-shortfall-and-asking-how-bad-is-bad-1.png)

This post owns that second number. We will build expected shortfall from absolutely nothing — no statistics background assumed — define it as the plain average of your worst cases, compute it by hand from a real loss sample, and then follow it everywhere it leads: why two books with the same VaR can carry completely different real risk, why ES deepens as you look further into the tail, why it is a *coherent* risk measure where VaR is not, and why a desk that sizes to its ES survives the day a desk sizing to its VaR does not. By the end you will see VaR for what it is — a useful but dangerously incomplete number — and you will reach for ES whenever the question that actually matters is not "how often" but "how bad."

## Foundations: the building blocks of expected shortfall

Before we touch ES, let's nail down every term from zero. If you trade for a living, skim this; if you don't, this section is the floor everything else stands on.

**Profit and loss (P&L).** The P&L of a position over some period is simply how much money you made or lost. On a \$10,000,000 book, a +\$50,000 day is a 0.5% gain; a −\$200,000 day is a 2% loss. Throughout this post we will work in *dollars of P&L*, because the whole point of a risk measure is to put a dollar figure on how bad a bad day can get. A *loss* is just a negative P&L, and when it is convenient we will flip the sign and talk about the *loss* as a positive number — a \$200,000 loss rather than a −\$200,000 P&L — so that "bigger is worse" reads naturally.

**Distribution.** If you record your daily P&L for a few years and tally how many days landed in each bucket, you get a *distribution* — a shape showing which outcomes are common and which are rare. Most days cluster near the middle (small gains and small losses); a few land far out in the tails (big gains, big losses). The left tail — the big losses — is the only part a risk manager truly cares about. The right tail (the windfalls) is somebody else's job to celebrate.

**Percentile (quantile).** A percentile cuts the distribution at a point and tells you what fraction lands below it. The 5th percentile of your P&L is the value such that 5% of your days are *worse* than it and 95% are better. The 1st percentile is the value 1% of your days fall below. Percentiles are how we turn "the bad tail" into a precise number: the 5% worst days are everything below the 5th percentile, the 1% worst days everything below the 1st. Quantile is the same idea stated as a fraction (the 0.05 quantile = the 5th percentile).

**Confidence level.** When someone says "95% VaR" or "99% ES," the percentage is the *confidence level* — the fraction of days the measure is meant to cover *normally*. A 95% measure looks at the worst 5% of days; a 99% measure at the worst 1%; a 99.9% measure at the worst one-in-a-thousand. Higher confidence means looking further out into the tail, at rarer and deeper losses. Always check the level: a "VaR of \$8 million" is meaningless until you know whether it's the 95% or 99.9% figure, and a one-day or ten-day horizon.

**Value-at-Risk (VaR).** Here's the one we're improving on. The X% VaR over some horizon is the loss you will *not* exceed with probability X. The 99% one-day VaR is the loss the model says you breach on only 1 day in 100 — concretely, it is the negative of the 1st percentile of your daily P&L. If your 99% VaR is \$8 million, then 99% of days you lose less than \$8 million, and 1% of days you lose more. The key word is *more*: VaR tells you the cutoff and stops. It never quantifies the "more." That is the entire reason this post exists. We won't re-derive VaR's three computation methods here — the [VaR post](/blog/trading/risk-management/value-at-risk-and-exactly-how-var-lies) does that in full — we just need its definition as the tail cutoff.

**Expected value (the average).** The expected value of a set of numbers is the plain average — add them up, divide by how many. The word "expected" trips people up: it does not mean the most likely outcome, it means the long-run average if you repeated the situation many times. Expected shortfall is, as the name says, an *expected value* — but not of all your days, only of your *worst* ones.

**The tail.** From here on, "the tail" means the loss region past the VaR cutoff: the worst (1−X)% of outcomes. At 95% confidence, the tail is the worst 5% of days. At 99%, the worst 1%. ES lives entirely inside the tail — it is a statement about that region and nothing else. VaR points at the *boundary* of the tail; ES describes its *interior*.

With those terms — P&L, distribution, percentile, confidence level, VaR, expected value, and the tail — you have everything you need. Now let's build the number.

## Expected shortfall, defined from one plain sentence

Here is the entire definition, in one sentence you could explain to a ten-year-old: **expected shortfall is the average of your worst losses.** Specifically, the X% expected shortfall is the average loss across the worst (1−X)% of outcomes. The 95% ES is the average of your worst 5% of days. The 99% ES is the average of your worst 1%. That's it. Everything else in this post is a consequence of that one sentence.

It goes by several names, and they all mean the same thing. **Conditional Value-at-Risk (CVaR)** is the most common name in trading and academia — "conditional" because it is the expected loss *conditional on* (given that) you are in the tail. **Expected shortfall (ES)** is the name regulators use. **Average Value-at-Risk (AVaR)** and **expected tail loss (ETL)** show up in textbooks. CVaR, ES, AVaR, ETL — for our purposes they are interchangeable. We'll mostly say ES, occasionally CVaR.

The recipe to compute it from real data is a five-step procedure with no calculus, no normal distribution, no assumptions:

1. Collect your P&L outcomes (one number per day, say).
2. Sort them from best to worst.
3. Pick your confidence level X; the tail is the worst (1−X)% of the sorted list.
4. The VaR is the *boundary* of that tail — the outcome where the tail begins.
5. The ES is the *average* of every outcome in the tail.

Two numbers fall out of the same sorted list. VaR is step 4: where does the bad region start? ES is step 5: how bad is it on average once you're in there? VaR is one point on the curve; ES is the mean of the whole bad end of the curve. This is why ES is *always at least as large as VaR* — the average of a set of losses, every one of which is at least as big as the cutoff, can never be smaller than the cutoff. ES ≥ VaR, always, at the same confidence level. The gap between them is the depth of your tail.

It is worth pausing on the word *conditional* in "Conditional Value-at-Risk," because it is the whole idea in one word. A *conditional* expectation is an average computed only over a sub-population, not the whole. The unconditional average of your daily P&L might be a small positive number — you make money on most days. That number is useless for risk: it tells you what an ordinary day looks like, not a disaster. ES throws away the ordinary days entirely and averages *only* the days inside the tail — the population *conditional on* having breached the VaR. "Given that today is one of your worst 1% of days, how much do you lose on average?" That conditioning is what makes ES a risk number and the plain average a return number. The same dataset yields two completely different statistics depending on which population you average over, and risk lives in the conditional one.

A small but important consequence: because ES conditions on being in the tail, it is *robust to how often* the tail happens in a way VaR is not. If a position's bad days got twice as frequent but stayed the same size, its VaR cutoff might barely move (the cutoff is a quantile of the whole distribution), but the *frequency* of breaches doubles — VaR's frequency claim breaks while its magnitude is unchanged. ES, by contrast, is a statement purely about *severity given a breach*; it reports the same average-disaster number regardless of how the breach frequency drifts, and you read frequency off VaR and severity off ES as two clean, separate dials. This separation is exactly why a complete risk sheet shows both: VaR for "how often," ES for "how bad."

There is also a clean continuous formula for readers who like one. If your loss has a probability density, the X% expected shortfall is the average of the VaR over all confidence levels *beyond* X:

$$\text{ES}_X = \frac{1}{1-X}\int_X^1 \text{VaR}_u \, du$$

Read it in English: instead of reporting the single VaR at level X, you sweep the VaR across every level from X to 100% and average them. That is exactly "average all the losses past the cutoff," written in the language of integrals. You do not need this formula to use ES — the five-step sort-and-average recipe gives the same answer on real data — but it shows why ES is sometimes called *average* VaR: it is literally the average of all the VaRs deeper than yours. If you want the measure-theoretic version and the subtleties when the loss distribution has atoms, the [tail-risk and extreme value theory post](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants) carries the heavy machinery; here the sort-and-average intuition is all we need.

#### Worked example: computing VaR and ES from the same sample by hand

Let's make this concrete on the recurring \$10,000,000 book. Suppose we have 1,000 daily P&L observations. We sort them worst to best and look at the very worst 1% — that's the worst **10 days** out of 1,000. Say those ten worst losses came out (in dollars of loss) as:

\$473,478 · \$476,325 · \$480,075 · \$501,674 · \$505,828 · \$527,025 · \$557,503 · \$631,481 · \$633,667 · \$751,566

The **99% VaR** is the *boundary* of the worst 1% — the smallest loss that still counts as being in the tail, which is the 10th-worst day: **\$473,478**. That is the VaR: "on 99 days out of 100, you lose less than \$473,478." It is the cliff edge.

The **99% ES** is the *average* of all ten of those worst losses. Add them: 473,478 + 476,325 + 480,075 + 501,674 + 505,828 + 527,025 + 557,503 + 631,481 + 633,667 + 751,566 = \$5,538,622. Divide by 10:

$$\text{ES}_{99\%} = \frac{\$5{,}538{,}622}{10} = \$553{,}862$$

So on this book, the 99% VaR is \$473,478 and the 99% ES is \$553,862. The ES is about **17% larger** than the VaR — and notice it is dragged up by the single worst day, \$751,566, which VaR completely ignores. If you size your book so you can survive a \$473,478 day, you are *under-prepared*: on a genuinely bad day you should expect to lose \$553,862 on average, and the worst day in the sample took \$751,566 — over 50% more than the VaR cutoff and 1.6 times what VaR told you to brace for.

*VaR tells you where the bad days begin; ES tells you how bad they are on average once they start, which is the number you actually have to survive.*

Figure 6 is that exact sample drawn out: a thousand sorted losses, the worst 1% lit up in red, the amber VaR cutoff where the tail begins, and the solid red ES line at the average of those ten worst bars. It is the five-step recipe made visible — count to the cutoff, then average everything past it.

![A sorted sample of one thousand daily losses on a ten million dollar book with the worst one percent highlighted in red the amber ninety-nine percent VaR cutoff and the red expected shortfall line at the average of the worst ten days](/imgs/blogs/cvar-expected-shortfall-and-asking-how-bad-is-bad-6.png)

## Why VaR is silent about the tail — and ES is not

The single most important property of ES is that it *sees depth*, and the single most dangerous property of VaR is that it does not. Let's prove this with the cleanest possible example: two books that share the *exact same VaR* and carry *completely different* real risk.

Take two desks, A and B. Both report a 95% VaR of \$200,000 on a \$10,000,000 book. On the daily risk sheet, they look identical — same cutoff, same number, same regulatory capital under a VaR regime. A risk manager scanning the report sees two desks with matching risk and moves on.

But the shape of their loss tails could not be more different. Desk A is a market-making book: its losses cluster. When A has a bad day, it loses a bit more than \$200,000 — \$210,000, \$230,000, occasionally \$280,000 — but its losses have a hard practical ceiling because A runs tight stops and small inventory. Desk B is a short-volatility book: most of the time it collects small premiums, but when it's wrong it is *catastrophically* wrong. B's bad days mostly cluster near \$200,000 too — but a few of them are \$600,000, \$900,000, even \$1,500,000. B has a *fat tail*; A does not.

Now compute the ES. For Desk A, the average loss across its worst 5% of days is about **\$238,628** — just a bit past the \$200,000 cutoff, because A's tail is thin and bunched. For Desk B, the average loss across its worst 5% is about **\$346,597** — dramatically deeper, because B's tail is studded with catastrophes that drag the average up. Same VaR of \$200,000. ES of \$238,628 versus \$346,597. Desk B carries roughly **45% more** expected loss in the tail than Desk A, and VaR reports them as identical.

Figure 2 is this comparison drawn on a log scale so the tails are visible. Both distributions cross the same amber VaR line at \$200,000. But Desk B's red distribution stretches far out to the right — a long, fat tail of catastrophic days — while Desk A's blue distribution falls off a cliff just past the cutoff. The two ES lines tell the true story that the single shared VaR conceals.

![Two loss distributions for a ten million dollar book on a log scale sharing the same two hundred thousand dollar VaR cutoff with the thin-tailed book ending sharply past it and the fat-tailed book stretching far to the right at a much deeper expected shortfall](/imgs/blogs/cvar-expected-shortfall-and-asking-how-bad-is-bad-2.png)

This is not a contrived edge case — it is the *normal* situation whenever real markets are involved, because market returns are fat-tailed almost everywhere you look. Any strategy that sells insurance, harvests carry, shorts volatility, or earns a small premium most of the time in exchange for a rare large loss has exactly Desk B's profile: a thin body and a monstrous tail. VaR rates these strategies as *safe* precisely because their bad days are rare and their typical day is calm — and then the tail arrives and the VaR-blessed book is the one that blows up. The full anatomy of why market losses bunch and fatten — negative skew, excess kurtosis, the higher moments VaR throws away — is the subject of [skew, kurtosis, and the shape of your losses](/blog/trading/risk-management/skew-kurtosis-and-the-higher-moments-the-shape-of-your-losses); the point here is simply that ES *measures* that shape and VaR discards it.

#### Worked example: same VaR, very different survival

Take the two desks above and ask the survival question directly. Both desks run on the same \$10,000,000 book and both report a 95% VaR of \$200,000. Suppose both have a string of three bad days in a row — a once-a-year stretch.

For Desk A, three tail days at roughly the ES of \$238,628 each costs about \$715,884 — a 7.2% drawdown on the book. Painful but survivable; the book is still standing, the desk trades tomorrow.

For Desk B, three tail days at its ES of \$346,597 each costs about \$1,039,791 — a 10.4% drawdown. And that's the *average* tail day. If B's three bad days happen to include one of its true catastrophes — a \$1,500,000 day — the three-day loss balloons toward \$2,000,000, a 20% drawdown that triggers redemptions, margin calls, and the forced-selling spiral. Recall the recovery math from [the asymmetry of losses](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain): a 20% drawdown needs a +25% gain to recover, and it arrives at the worst possible moment for raising it.

Two desks. Identical VaR. One survives the bad stretch with a 7% dent; the other risks a 20% hole that can end it. *The number that told them apart was the ES, and a firm that only watched VaR would have sized both desks the same and been blindsided by the one that mattered.*

## How deep is the tail? ES at rising confidence

Because ES averages *everything* past the cutoff, where you put the cutoff changes the answer — and changes it in a revealing way. Push the confidence level higher and you are looking at a smaller, deeper slice of the tail, so the ES grows. The 99.9% ES averages a far worse set of days than the 95% ES, because you have walked further out into the disaster zone before you start averaging.

This is worth dwelling on, because it is where ES becomes a genuine *tail microscope*. At 95%, you average the worst 5% of days — a fairly broad slice that includes a lot of merely-bad days. At 99%, you average the worst 1% — a narrower, more severe slice. At 99.9%, you average the worst one-in-a-thousand days — the genuine catastrophes, and nothing else. Each step deeper throws away the milder breaches and keeps only the worse ones, so the average climbs.

Figure 4 shows this on a fat-tailed \$10,000,000 book. At each confidence level you see the VaR (the cutoff) and the ES (the average past it), and both climb steeply as you push from 95% to 99% to 99.9%. Crucially, the *gap* between VaR and ES is wide at every level — ES is always meaningfully larger — and the whole ES bar grows from a few hundred thousand to over a million and a half dollars as you walk into the deep tail.

![Grouped bars showing VaR and expected shortfall both rising as the confidence level increases from ninety-five to ninety-nine to ninety-nine point nine percent on a fat-tailed ten million dollar book with ES always larger than VaR at every level](/imgs/blogs/cvar-expected-shortfall-and-asking-how-bad-is-bad-4.png)

#### Worked example: choosing the confidence level on the book

Concretely, on the fat-tailed \$10,000,000 book in Figure 4:

- **95% level** — the worst 5% of days. VaR = \$254,255; ES = **\$402,876**. The average bad day, broadly defined.
- **99% level** — the worst 1% of days. VaR = \$466,634; ES = **\$712,828**. The average *severe* day.
- **99.9% level** — the worst 0.1% of days. VaR = \$1,032,323; ES = **\$1,563,687**. The average *catastrophe*.

Read across and the lesson is stark. A desk that budgets to its 95% ES is braced for a \$402,876 day. But roughly once every two trading years (a 1-in-500 day sits between the 99% and 99.9% level), this book delivers something near its 99.9% ES of \$1,563,687 — nearly **four times** the 95% figure. The choice of confidence level is the choice of *which disaster you are preparing for*. Budget to 95% and you are ready for an ordinary bad month and blindsided by a true crisis; budget to 99% or 99.9% and you hold more capital in reserve but you survive the day that ends the 95% desk.

*The confidence level is not a statistical nicety — it is the dial that sets how deep a disaster your capital is sized to survive, and on a fat-tailed book the deep tail is multiples worse than the shallow one.*

There's a second-order point hiding here that separates ES from VaR even at a single level. Because ES *integrates* the whole tail, it is far less sensitive to the exact cutoff than VaR is. Nudge the confidence level on VaR and you can jump discontinuously from "the loss is a small profit" to "the loss is enormous" if a catastrophe sits right at the boundary (we'll see exactly this in the coherence example). ES moves smoothly, because shifting the boundary just adds or removes one outcome from a large average. This smoothness is not cosmetic — it is one of the formal properties that makes ES *coherent* and VaR not, which is the next, and deepest, part of the story.

## Coherence: why ES is a "real" risk measure and VaR is not

In 1999, four mathematicians — Artzner, Delbaen, Eber, and Heath — asked a deceptively simple question: what properties *should* any sensible risk measure have? They wrote down four axioms, called a risk measure that satisfied all four **coherent**, and then proved something that shook the industry: **VaR is not coherent.** Expected shortfall is. This is not academic hair-splitting. The one axiom VaR violates is the one that says *diversification should never increase your measured risk* — and a risk measure that can punish you for diversifying is, in a precise mathematical sense, broken.

Here are the four coherence axioms in plain English, each a property you would *want* any honest risk number to have:

1. **Monotonicity.** If portfolio A always loses at least as much as portfolio B in every scenario, then A's risk should be at least as large as B's. (A strictly worse book is not allowed to look safer.)
2. **Translation invariance.** Add \$1 of risk-free cash to the book and its risk should drop by exactly \$1. (Holding more cash makes you safer by exactly the amount of cash.)
3. **Positive homogeneity.** Double the size of every position and the risk should exactly double. (Scaling the book scales the risk proportionally.)
4. **Sub-additivity.** The risk of two books combined should never exceed the sum of their separate risks: *risk(A + B) ≤ risk(A) + risk(B)*. (Merging two books can only *reduce* total risk through diversification, never increase it.)

VaR satisfies the first three. It fails the fourth — and the fourth is the soul of risk management. Sub-additivity is the mathematical statement of "don't put all your eggs in one basket": if combining two positions could *raise* your total risk, then diversification would be penalized and the whole logic of portfolio construction would collapse. Diversification, as the series keeps insisting, is the [only free lunch](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine) — and a coherent risk measure must, as a matter of axiom, recognize that lunch. VaR does not.

How can VaR possibly punish diversification? The trick is that VaR is a *quantile*, and quantiles do not add up. A loss that hides *just past* the cutoff in two separate books can *combine* to push the joint loss back *across* the cutoff. The cleanest illustration is two defaultable bonds.

#### Worked example: VaR explodes when you diversify; ES does not

Hold two corporate bonds, A and B, on the \$10,000,000 book. Each bond, independently:

- with probability **4%**, the issuer defaults and you lose **\$1,000,000**;
- with probability **96%**, you collect the coupon and gain **\$20,000**.

Take the **95% VaR of bond A alone.** The worst 5% of outcomes — that's the question. But the *only* bad outcome (default) happens just **4%** of the time, which is *less* than 5%. So the 5th-percentile outcome is *not* the default — it's the coupon. The 95% VaR of bond A is therefore the *negative* of a \$20,000 gain: **−\$20,000.** A negative VaR means "no loss at this confidence level" — at the 5% quantile, bond A is actually making money. Same for bond B: 95% VaR = **−\$20,000.** Sum the two stand-alone VaRs:

$$\text{VaR}(A) + \text{VaR}(B) = -\$20{,}000 + (-\$20{,}000) = -\$40{,}000$$

VaR is telling you these two bonds together are basically riskless — it reports a \$40,000 *profit* at the 95% level. Now actually **combine them** and compute the 95% VaR of the portfolio. Because defaults are independent, the joint outcomes are:

- both pay (prob 0.96 × 0.96 = **92.16%**): P&L = +\$40,000;
- exactly one defaults (prob 2 × 0.04 × 0.96 = **7.68%**): loss = \$1,000,000 − \$20,000 = **\$980,000**;
- both default (prob 0.04 × 0.04 = **0.16%**): loss = **\$2,000,000**.

The probability of a loss of \$980,000 *or worse* is 7.68% + 0.16% = **7.84%** — which is *more* than 5%. So now the 5th-percentile outcome lands squarely inside the disaster region, and the 95% VaR of the combined book is **\$980,000.** Compare:

$$\text{VaR}(A+B) = \$980{,}000 \quad\text{but}\quad \text{VaR}(A)+\text{VaR}(B) = -\$40{,}000$$

Diversifying — holding two *independent* bonds instead of relying on one — made the *reported* VaR jump from −\$40,000 (a profit) to +\$980,000, an increase of **\$1,020,000.** VaR says combining two independent bonds is over a million dollars *riskier* than the sum of the two alone. That is sub-additivity violated, in the open. A trader optimizing against a VaR limit would be told to *concentrate* in a single bond to keep VaR low — the exact opposite of sound risk management.

Now redo it with **ES.** The 95% ES of bond A alone averages over its worst 5% of mass: the worst 4% is the default (\$1,000,000 loss), and the next 1% is the coupon (a \$20,000 gain, i.e. a −\$20,000 loss). The average loss over that 5% slice is:

$$\text{ES}_{95\%}(A) = \frac{0.04 \times \$1{,}000{,}000 + 0.01 \times (-\$20{,}000)}{0.05} = \$796{,}000$$

So ES *sees* the default that VaR hid behind the cutoff — \$796,000, a real and serious number. Same for B: ES = \$796,000. Sum: **\$1,592,000.** Now the combined book's 95% ES averages over its worst 5% of mass: the worst 0.16% is the double-default (\$2,000,000), then the next 4.84% is the single-default (\$980,000):

$$\text{ES}_{95\%}(A+B) = \frac{0.0016 \times \$2{,}000{,}000 + 0.0484 \times \$980{,}000}{0.05} = \$1{,}012{,}640$$

And the verdict:

$$\text{ES}(A+B) = \$1{,}012{,}640 \;\le\; \text{ES}(A)+\text{ES}(B) = \$1{,}592{,}000 \;\checkmark$$

ES *rewards* the diversification — the combined book's tail risk (\$1,012,640) is well below the sum of the two stand-alone tail risks (\$1,592,000), because spreading across two independent issuers makes the worst case (both defaulting at once) genuinely rare. This is sub-additivity *holding*, exactly as a coherent measure guarantees it always will.

*VaR can tell you to put all your eggs in one basket; ES never will — and that single difference is why ES is a real risk measure and VaR is a quantile wearing a risk measure's clothes.*

Figure 5 lays the two side by side: the same two bonds, the VaR side reporting a risk explosion when you diversify, the ES side reporting the risk *falling* the way it must.

![A before and after comparison of two independent defaultable bonds showing value at risk violating sub-additivity by reporting nine hundred eighty thousand dollars of combined risk versus negative forty thousand dollars summed while expected shortfall stays coherent with combined risk below the sum](/imgs/blogs/cvar-expected-shortfall-and-asking-how-bad-is-bad-5.png)

The deeper reason VaR misbehaves and ES does not comes back to *what each one is*. VaR reads a single point off the loss distribution — one quantile — and points are fragile: a catastrophe sitting one inch on the wrong side of the cutoff is either fully counted or fully ignored, with nothing in between, so combining books can flip the count. ES averages the *entire* region past the cutoff, so no single catastrophe gets ignored just for being rare, and the average of a sum is well-behaved in a way the quantile of a sum is not. ES is coherent because averaging is coherent; VaR is incoherent because thresholding is not. If you want the full proof that the four axioms force ES-like measures (the representation theorem), it lives in the [tail-risk and extreme value theory post](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants); the working trader needs only the conclusion — *use the measure that can't be gamed into concentrating.*

Coherence is not just a certificate of mathematical good behavior — it buys you something concrete: you can *decompose* ES cleanly into per-position contributions that sum to the total. Because ES is sub-additive and positively homogeneous, the marginal ES contribution of each position — how much portfolio ES would change if you trimmed that position slightly — adds up exactly to the portfolio ES. That means you can look at a book and say "this single position is responsible for 40% of my tail risk" and *trust the number*, because the contributions are mathematically guaranteed to be consistent. With VaR you cannot do this reliably: because VaR is not sub-additive, the "contributions" don't have to add up sensibly, and a position can appear to *reduce* portfolio VaR while genuinely increasing tail danger. This is why the desks that decompose risk to decide what to trim — where the risk actually lives — increasingly do it on ES, not VaR. A coherent measure is one you can do arithmetic with; an incoherent one will betray the arithmetic exactly when the book is most dangerous.

## The question each one answers, side by side

Step back from the math and the two measures resolve into two different *questions*, and it is worth being crisp about which is which, because using the wrong one is how desks get surprised.

VaR answers: **"How often will my loss cross this line?"** It is a *frequency* statement. It is genuinely useful for that — for setting a daily loss limit, for calibrating how often a book should breach, for backtesting whether a model's breach rate matches its claim. If your question is "how many bad days should I expect," VaR is the right tool and ES is overkill.

ES answers: **"Given that I cross the line, how bad is it on average?"** It is a *magnitude* statement. It is the right tool for the question that actually decides survival — how much capital do I need to hold, how large can this position be, what is the loss I must be able to absorb without being forced to sell. If your question is "how bad is a bad day," VaR is silent and ES is the answer.

Figure 3 sets the two questions next to each other. The left column is VaR's question — *how often?* — and what it sees (the cliff edge) and what it's blind to (the depth). The right column is ES's question — *how bad when it happens?* — and what it sees (the whole tail) and what it buys you (a number that sizes the real disaster). Same tail, two questions; you need the right one for the decision in front of you.

![A before and after diagram contrasting value at risk answering how often a loss is breached against expected shortfall answering how bad the loss is on average once breached with VaR blind to the tail depth and ES sizing the actual disaster](/imgs/blogs/cvar-expected-shortfall-and-asking-how-bad-is-bad-3.png)

Here is the failure mode this distinction guards against, and it is everywhere. A trader is given a VaR limit. The trader's job is to maximize return *subject to* keeping VaR under the limit. So the trader finds positions that produce return while keeping the *cutoff* low — and the most efficient way to do that is to sell things that almost never lose but lose enormously when they do. Selling deep out-of-the-money options. Shorting volatility. Picking up nickels in front of a steamroller. Each of these keeps VaR pristine, because the loss is *rare* enough to hide past the cutoff, while loading the book with exactly the catastrophic tail VaR can't see. A VaR limit, optimized against, *manufactures* tail risk. An ES limit cannot be gamed the same way, because ES *averages the catastrophe in* the moment it exists at all — there is nowhere past the cutoff for the disaster to hide. This is the practitioner's reason, beyond the axioms, that sophisticated desks run ES limits: not because ES is mathematically prettier, but because it is *harder to game into a blowup*. The strategic dimension of traders routing around the limits set for them is the whole subject of [the variance risk premium and why selling vol pays until it doesn't](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt), which is the canonical "low VaR, monstrous ES" trade.

#### Worked example: ranking two strategies that VaR and Sharpe call a tie

Here is the cleanest demonstration of ES earning its keep. Suppose your risk committee is choosing between two strategies to fund, each running on the \$10,000,000 book. You compute the two numbers every committee looks at first:

- **Strategy 1 (symmetric):** an ordinary, roughly bell-shaped P&L. Annualized **Sharpe ratio ≈ 0.78.** 99% **VaR ≈ \$60,276.**
- **Strategy 2 (short-vol):** small steady carry most days, a rare violent loss. Annualized **Sharpe ratio ≈ 0.78.** 99% **VaR ≈ \$60,276.**

By the two headline numbers — risk-adjusted return and Value-at-Risk — these strategies are a *dead heat*. Identical Sharpe, identical VaR. A committee using only those would flip a coin, or fund both equally, or pick on personality. Now compute the **99% ES:**

- **Strategy 1:** 99% ES ≈ **\$69,269.** Its tail is shallow — the average bad day is only a touch worse than the \$60,276 cutoff, because its losses are well-behaved.
- **Strategy 2:** 99% ES ≈ **\$343,283.** Its tail is a chasm — the average bad day is **five times** the cutoff, because the short-vol book's rare losses are enormous.

Same Sharpe. Same VaR. ES of \$69,269 versus \$343,283 — a **5x** difference in the loss you must actually survive. Strategy 2 is *vastly* riskier, and *only* ES revealed it. A committee that funded both equally on the strength of matching Sharpe and VaR would be loading up on a hidden catastrophe; a committee that looked at ES would size Strategy 2 to a fraction of Strategy 1, or demand it be hedged, or pass on it entirely.

*Sharpe and VaR can call two strategies a tie when one of them is five times more likely to end you — ES is the number that breaks the tie in the direction of survival.*

Figure 7 shows this exactly: on the left, Sharpe says "dead heat"; on the right, the VaR bars match while the ES bars diverge by a factor of five. The short-vol book is the fatter-tailed one, and ES is what flags it.

![A two panel comparison where the left panel shows two strategies with equal annualized Sharpe ratios and the right panel shows them with equal ninety-nine percent VaR but the short-vol strategy carrying a five times larger expected shortfall that flags its hidden fat tail](/imgs/blogs/cvar-expected-shortfall-and-asking-how-bad-is-bad-7.png)

## Common misconceptions

**"ES is just VaR plus a fudge factor — they move together, so why bother?"** They move together only when the tail is well-behaved. The whole point of ES is the cases where they *don't* — the fat-tailed books where ES is multiples of VaR. In the Figure 7 example, two strategies with identical 99% VaR of \$60,276 had ES of \$69,269 and \$343,283 — the ES/VaR ratio was 1.15 for one and 5.7 for the other. ES is not a multiple of VaR; the *gap between them is itself the information*, and it is exactly the information VaR throws away. A constant fudge factor would tell you nothing; ES tells you the shape of your tail.

**"Higher confidence is always safer, so use 99.9% ES and be done."** Higher confidence looks deeper into the tail, but it also relies on *fewer and rarer* data points, so the estimate gets noisier and more model-dependent precisely where you can least afford error. A 99.9% ES is computed from the worst 1-in-1,000 days — if you have five years of data (about 1,250 days), you are averaging *one or two observations*, which is statistically almost meaningless. The 99.9% ES is the most important number and the *least* reliably estimated; that tension is real. The fix is not to pick a single magic level but to look at ES across levels (as in Figure 4) and to supplement deep-tail ES with [extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants), which fits the tail's *shape* rather than counting the handful of points that landed there.

**"If my ES is \$553,862, that's the most I can lose."** No — ES is an *average* of the tail, not the *worst* of it. In the Figure 6 sample, the 99% ES was \$553,862 but the single worst day was **\$751,566** — over 35% worse than the ES. By construction, roughly half the tail days are *worse* than the ES (it's their average). ES sizes the *typical* disaster, not the *maximum* one. For the maximum, you stress-test specific scenarios; ES is the day-to-day budgeting number, not the absolute floor. Anyone who treats ES as a hard ceiling is making a smaller version of the exact mistake VaR-users make with the cutoff.

**"ES requires assuming a normal distribution, just like VaR."** It requires *no* distributional assumption at all when computed the way we did it — sort the historical losses, average the worst tail. That's the *historical* (non-parametric) ES, and it is the most honest version precisely because it makes no claim about the shape of the tail; it just reports what actually happened. You *can* compute a parametric ES assuming normality, and people do, but then you inherit the normal distribution's catastrophic under-counting of tail events — the same flaw that sinks parametric VaR. The normal-assumption problem belongs to the *method*, not to ES; historical ES sidesteps it entirely.

**"Regulators use VaR, so VaR must be the standard."** They *used* to. The Basel Committee's Fundamental Review of the Trading Book (FRTB), finalized in 2019, formally **replaced 99% VaR with 97.5% expected shortfall** as the regulatory market-risk measure for banks, with phased adoption through the early 2020s. The explicit reasons in the rule text: VaR is not sub-additive (it can penalize diversification) and it ignores the severity of losses in the tail. The world's bank regulators, after the 2008 crisis exposed exactly how VaR's tail-blindness fed the catastrophe, rewrote the rulebook around the number this post is about. VaR is the legacy standard; ES is the current one.

**"ES tells me how to hedge."** ES tells you how *much* tail risk you carry, not how to remove it. It is a *measurement*, not a *prescription*. Knowing your ES is \$553,862 tells you the disaster to budget for; it does not tell you whether to cut the position, buy protective puts, or diversify the book. The mechanics of *reducing* a tail — protective puts, collars, tail hedges and their cost — are the domain of the options-volatility series; ES is the dashboard that tells you the tail is there and how deep, so you know a hedge is worth paying for.

## How it shows up in real markets

The history of modern finance is, in large part, a history of VaR-blessed books that ES would have flagged. The case studies all share one fingerprint: a strategy with a *low VaR* — calm, profitable, well-inside-its-limit on the daily risk sheet — and a *monstrous ES* that the firm either didn't compute or didn't believe.

**Long-Term Capital Management, August–September 1998.** LTCM ran convergence trades that lost a small amount very rarely — exactly the low-VaR, fat-ES profile. Its models, calibrated on calm-period data, reported tail risk that was a fraction of the real thing. When Russia defaulted and correlations went to 1, the book lost about **\$4.6 billion** of capital in roughly four months on roughly **25:1** balance-sheet leverage and **\$1.25 trillion** of gross notional. VaR, fitted to a placid recent past, had no idea the tail was that deep; an honest ES on a stress distribution would have screamed. The full anatomy is in the [game-theory case study on LTCM](/blog/trading/game-theory/case-study-ltcm-1998-the-crowded-genius-trade), but the risk-measurement lesson is precise: the trade had a tiny VaR and a catastrophic expected shortfall, and the firm capitalized to the first.

**The 2008 crisis and the failure of VaR.** Through 2006–2007, bank trading desks reported comfortable VaR numbers on books stuffed with mortgage credit and structured products — assets whose loss distribution was the textbook fat tail: a thin body of carry income and a monstrous left tail of correlated default. VaR, calibrated on years of rising house prices, reported the tail as negligible. The ES — the average loss *given* a crisis — was enormous, and unmeasured. There is a darkly instructive detail in the post-mortems: several banks' VaR models were so calmed by the placid recent past that their reported 99% VaR was *exceeded* far more than 1% of the time once the crisis hit — the model wasn't just silent about the tail's *depth*, its frequency claim broke too. But the depth was the killer. A book whose VaR said "you lose \$50 million on a bad day" was, when the tail finally arrived, losing many multiples of that, because the *conditional* loss — the ES — had always been a chasm that nobody had measured or capitalized against. When the tail arrived, the VaR-comforted numbers were revealed as fiction, and the regulatory response a decade later was to replace VaR with ES outright. 2008 is the single largest piece of evidence that "how often" is the wrong question and "how bad when it happens" is the right one.

**Volmageddon, February 5, 2018.** The cleanest modern example of the low-VaR-monstrous-ES trade. Short-volatility products like XIV collected a small premium most days — a beautiful low-VaR P&L stream that had paid out for years. Their ES was a cliff: a single large VIX spike could wipe them out. On February 5, 2018, the VIX jumped about **20 points** (roughly 17.3 to 37.3, a ~116% one-day rise) and XIV's NAV fell about **96%** after the close, terminating the product. Every day before that, VaR called XIV safe. Its expected shortfall — the average loss *given* a vol spike — was always near-total. This is the Desk B of Figure 2 made real, and the [Volmageddon case study](/blog/trading/options-volatility/case-study-volmageddon-2018-and-the-short-vol-blowup) walks the reflexive feedback loop in full.

**Archegos, March 2021.** Concentrated, swap-financed single-stock exposure at roughly **5x+** leverage, hidden from each prime broker because no single counterparty saw the total. The daily VaR each bank computed on its *slice* looked manageable; the *true* tail — the loss given the concentrated positions unwinding all at once — was the real risk, and it cost the banks over **\$10 billion** in aggregate, with Credit Suisse alone losing about **\$5.5 billion.** When the position broke, the loss was not a VaR-sized wobble; it was an ES-sized chasm that several risk systems had been blind to.

**The yen-carry unwind, August 5, 2024.** A crowded funding-carry trade — borrow cheap yen, buy higher-yielding assets — paid a small steady spread for years, the canonical low-VaR carry profile. When it unwound, the Nikkei fell about **12.4%** in a single day (its worst since 1987) and the VIX spiked to an intraday peak near **65.7.** The trade that "never lost" lost catastrophically in days. Same fingerprint: thin body, fat tail, low VaR, deep ES — and a deleveraging cascade once the tail arrived.

The pattern is so consistent it is almost a law: the trades that blow up are the ones with a *flattering VaR and a terrifying ES*, and the firms that survive are the ones that sized to the second number. VaR told all of them how often. ES would have told them how bad. They learned the second number the hard way.

## The risk playbook: sizing to an ES budget

Here is how to put expected shortfall to work, concretely, from the trader's seat.

**1. Watch ES, not VaR, as your headline tail number.** VaR is fine as a frequency check — how often should I breach? — but the number that governs survival is the average loss *given* a breach. On any book that sells premium, harvests carry, shorts volatility, or earns a small edge most of the time, VaR will flatter you and ES will tell the truth. Compute both; *decide* on ES.

**2. Size positions to an ES budget, not a VaR limit.** Set the maximum you can lose on a genuinely bad day as a fraction of capital — say you decide your 97.5% one-day ES must not exceed 2% of the book. On a \$10,000,000 book that's a \$200,000 ES budget. Size every position so the *portfolio* ES stays under it. Because ES is sub-additive, diversifying *helps* you stay under budget — the measure rewards exactly the behavior that keeps you alive. A VaR budget, by contrast, can be satisfied by concentrating into a low-frequency catastrophe, which is the opposite of what you want.

**3. Use historical (non-parametric) ES as your default.** Sort your actual losses and average the worst tail — no normal-distribution assumption, no false precision. Supplement the deep tail (99% and beyond), where you have too few data points to trust the raw average, with [extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants) to model the tail's *shape* rather than counting the handful of points that landed in it.

**4. Look at ES across confidence levels, not one magic number.** The 95%, 99%, and 99.9% ES answer different survival questions (Figure 4). The 95% is your ordinary-bad-month number; the 99.9% is your once-in-a-few-years catastrophe number. Hold capital sized to the level of disaster you actually intend to survive — and know that on a fat-tailed book the deep-tail ES is *multiples* of the shallow one, so "I survive the average bad day" is a much weaker claim than it sounds.

**5. Treat a low VaR and a high ES as a red flag, not a green light.** The widest VaR-to-ES gap on your sheet is the position most likely to end you: it loses rarely (so VaR is calm) and enormously (so ES is deep). That gap is the signature of the short-vol/carry/sell-insurance trade. When you see it, the position is either a deliberate, hedged, carefully-sized bet — or a future blowup. Decide which on purpose.

**6. Backtest the magnitude, not just the count.** A VaR backtest checks one thing: did breaches happen about as often as claimed? That's a frequency test, and it can pass on a book that is quietly catastrophic — the breaches arrive on schedule, they're just far deeper than budgeted. An ES backtest is harder (you're testing a conditional average, which needs more tail observations) but it asks the right question: when breaches happened, were they as bad as the ES predicted, or worse? If your realized tail losses keep coming in above your ES estimate, your tail model is too thin and your capital is too low — fix it before the market fixes it for you.

**7. Remember the spine.** Everything here serves the one rule the series never stops repeating: *you can only compound if you're still in the game.* VaR tells you how often you'll have a bad day; ES tells you how bad, on average, the bad day is — and it is the *bad day*, not the average day, that determines whether you survive to compound tomorrow. The asymmetry of losses makes this concrete: a tail loss is not just larger than a typical loss, it is disproportionately harder to recover from, so the day you under-budget for the tail is the day the recovery math turns terminal. Size to the loss you must survive, not the loss you usually take. That loss is the ES.

The handoff from VaR is now complete. VaR asked *how often* and went silent on the rest. Expected shortfall asks the question that actually decides survival — *how bad is it, on average, when it happens?* — and answers it with a single, coherent, ungameable number. It is the tail measure regulators moved to, the number sophisticated desks budget against, and the one that would have flagged nearly every blowup in this post before it happened. Watch it. Size to it. Survive.

### Further reading

- [Value-at-Risk, and exactly how it lies](/blog/trading/risk-management/value-at-risk-and-exactly-how-var-lies) — the measure this post hands off from: what VaR claims, how it's computed, and the precise ways it fails.
- [Skew, kurtosis, and the higher moments: the shape of your losses](/blog/trading/risk-management/skew-kurtosis-and-the-higher-moments-the-shape-of-your-losses) — why market losses are negatively skewed and fat-tailed, the shape that ES measures and VaR discards.
- [The asymmetry of losses: why a 50% loss needs a 100% gain](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) — the recovery math that makes a deep tail loss so much more dangerous than its size suggests.
- [Tail risk and extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants) — the formal machinery behind ES, the coherence representation theorem, and how to estimate the deep tail when data is scarce.
- [The variance risk premium: why selling vol pays until it doesn't](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) — the canonical low-VaR, monstrous-ES trade, and the economics of getting paid to carry a fat tail.
