---
title: "Ergodicity: Time-Average vs Ensemble-Average and the Coin Flip That Ruins You"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A coin flip with positive expected value can still bankrupt almost everyone who plays it through time, because multiplicative wealth is non-ergodic, and that is the deepest reason risk management exists."
tags: ["risk-management", "ergodicity", "time-average", "ensemble-average", "geometric-growth", "expected-value", "ruin", "position-sizing", "compounding"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **One-sentence thesis:** A gamble can have a positive expected value across many people and still drive almost every single person who plays it through time to ruin, because wealth multiplies rather than adds, and multiplication is *non-ergodic*.
> - The canonical coin flip: heads pays +50%, tails costs −40%, on a fair coin. Each round the *average across players* grows by +5%. Yet a single player's wealth compounds at about **−5.1% per round** and decays toward zero.
> - The **ensemble average** (average across many people at one instant) and the **time average** (one person's growth across many rounds) are different numbers — and you live in the time average.
> - The number you actually compound at is the **geometric (time) growth rate**, not the arithmetic expected value. Risk management is the discipline of maximising the geometric rate, not the arithmetic one.
> - The mean of terminal wealth is huge only because a *handful* of lucky paths drag it up; the **median** (the typical player) is near zero — strong right skew.
> - This is the deepest justification for position sizing: you size for the path you have to live, not the average you'll never experience.

In 1738 Daniel Bernoulli looked at a gamble that, by the rules of expected value, was worth an infinite amount of money — the St. Petersburg lottery — and noticed that no sane person would pay more than a few coins to play it. For nearly three centuries the standard escape was to say people are *risk-averse*: they dislike uncertainty, so they discount the upside. That answer is half right and deeply misleading. The cleaner answer, sharpened by the physicist Ole Peters in the 2010s, is that the person who plays a multiplicative gamble does not experience the expected value at all. They experience something else — the **time-average growth rate** — and for the gambles that matter, that number is brutally lower, often negative when the average looks positive.

This is the single most important idea in this entire series, and most traders have never heard its name. It is called **ergodicity**, and it is the mathematical reason a strategy can have a real, positive edge and still blow up the person running it. It is why "positive expected value" is a trap if you stop reading there. It is why a −50% drawdown needs a +100% gain to recover, and why that asymmetry is not a quirk but a law. Everything else in risk management — Kelly sizing, drawdown limits, the survival mandate — is downstream of this one fact: **you only get one path through time, and that path is not the average.**

We will build the whole thing from a single coin. Heads and your money grows 50%. Tails and it shrinks 40%. The coin is fair. The expected value per flip is positive. And almost everyone who plays it goes broke. Figure 1 shows the entire paradox in two panels: the crowd's average wealth rising on the left, your one life sinking on the right, both from the *same coin*.

![Two panel chart showing the plus fifty minus forty coin flip with the ensemble average wealth rising on the left and a single individual path declining toward zero on the right](/imgs/blogs/ergodicity-time-average-vs-ensemble-average-and-the-coin-flip-that-ruins-you-1.png)

## Foundations: the building blocks, defined from zero

Before we can take the coin apart, we need four ideas with no jargon left in them. If you already know expected value and the difference between an arithmetic and a geometric mean, skim this; if you don't, this section is the whole foundation.

**Expected value.** The expected value of a gamble is the average payoff you'd get if you could play it an enormous number of times *side by side* — one dollar on each of a million independent tables at once — and then averaged the results. For our coin: half the time you multiply your money by 1.5, half the time by 0.6. The expected *multiplier* is `0.5 × 1.5 + 0.5 × 0.6 = 1.05`. So the expected value of one round is +5% of your stake. If you have \$100,000, the expected wealth after one round is \$105,000. This number is real, it is correct, and — this is the trap — it describes a situation you are not in.

The subtlety hidden in the phrase "side by side" is the whole ballgame. Expected value is, by construction, an average over a *population of parallel outcomes that all happen at the same instant*. It answers the question "if I cloned this bet across a million simultaneous worlds, what's the average across those worlds?" That is a perfectly good question, and for a casino with a million tables running tonight, it's *the* question. But it is not the question a single trader asks. The trader asks: "I have one account, and I'm going to make this bet over and over across time — what happens to me?" Those are different questions, and the breathtaking part is that they have different answers whenever wealth multiplies. Expected value silently assumes they're the same. They are not.

**Additive vs multiplicative dynamics.** When outcomes *add* — you win \$50 or lose \$40 in fixed dollars, regardless of how much you already have — the gamble is additive. When outcomes *multiply* — you win 50% or lose 40% *of whatever you currently hold* — the gamble is multiplicative. Almost all wealth is multiplicative: a 40% loss on a \$1,000,000 book costs \$400,000, while a 40% loss on a \$10,000 account costs \$4,000. The percentage is fixed; the dollars scale with what you have. This distinction looks pedantic. It is the entire post.

Why is wealth multiplicative and not additive? Because returns are *rates*, and rates compound. A trader who makes 2% on a position makes 2% of whatever capital is allocated to it; double the capital, double the dollars. Reinvested profits become next period's base; realized losses shrink that base. Interest compounds, dividends reinvest, a percentage drawdown removes a percentage of everything. There is essentially no large pool of money in finance that grows additively — additive growth would mean you make the same fixed dollar amount whether you're managing \$10,000 or \$10,000,000, which describes a salary, not an investment. The moment money is *at risk as a percentage*, you are in the multiplicative world, and the multiplicative world is non-ergodic. That's why this isn't a niche puzzle for probabilists; it's the default physics of every account you'll ever run.

**Arithmetic mean vs geometric mean.** The arithmetic mean of a set of growth factors is what you get by adding them and dividing. The geometric mean is what you get by *multiplying* them and taking the root. For our two factors, 1.5 and 0.6, the arithmetic mean is `(1.5 + 0.6) / 2 = 1.05`, but the geometric mean is `√(1.5 × 0.6) = √0.9 ≈ 0.9487`. One is above 1 (growth); the other is below 1 (decay). When your wealth multiplies round after round, **the geometric mean is the factor your money actually compounds at** — because compounding *is* repeated multiplication, and the right "average" for repeated multiplication is the geometric one. The +5% is a fact about the crowd; the −5.1% is a fact about you.

**Ensemble vs time average.** This is the pair the whole post turns on. The **ensemble average** is the average computed *across many systems at one moment* — a million gamblers each flipping once, averaged. The **time average** is the average computed *along one system across many moments* — you, flipping a million times, and the long-run growth rate of your single bankroll. A process is **ergodic** when these two are equal: it doesn't matter whether you average across the crowd or along one life, you get the same number. A process is **non-ergodic** when they differ. Additive coin flips are ergodic. Multiplicative wealth is *not*, and that single failure of ergodicity is where ruin lives.

#### Worked example: one round of the coin on a \$100,000 account

You start with \$100,000 and flip the coin once.

- **Heads (+50%):** `$100,000 × 1.5 = $150,000`. You're up \$50,000.
- **Tails (−40%):** `$100,000 × 0.6 = $60,000`. You're down \$40,000.

The expected wealth is `0.5 × $150,000 + 0.5 × $60,000 = $75,000 + $30,000 = $105,000`. So on average you're up \$5,000 — a +5% expected gain, exactly the arithmetic mean of the multipliers. If you only ever played one round, or if you played one round on each of ten thousand separate accounts and added them up, this number would be the truth you experience.

*The expected value is honest about the crowd of one-shot bets; it says nothing yet about what happens when the same account flips again and again.*

## The trap: positive expectation, near-certain ruin

Now we let the same account keep flipping. This is the move that breaks the expected-value intuition, because the second flip multiplies whatever the first flip *left you*, not your original stake.

#### Worked example: two rounds, and why order stops mattering to the crowd but not to you

Start again with \$100,000. Suppose you flip a head then a tail.

- After heads: `$100,000 × 1.5 = $150,000`.
- After tails: `$150,000 × 0.6 = $90,000`.

You are down \$10,000 — even though you won once and lost once, one of each. Now reverse the order, tails then heads:

- After tails: `$100,000 × 0.6 = $60,000`.
- After heads: `$60,000 × 1.5 = $90,000`.

Same \$90,000. The order didn't change *your* outcome (multiplication commutes), but notice what it did to your wealth: **one head and one tail leaves you below where you started.** A +50% and a −40% don't cancel. They compound to `1.5 × 0.6 = 0.9`, a 10% loss over two rounds, because the 40% was taken off a *larger* number than the 50% was added to. This is the asymmetry of losses showing up in its native habitat, and it's covered in depth in [the asymmetry of losses](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain).

Meanwhile the *expected value over two rounds* is still climbing: `$100,000 × 1.05 × 1.05 = $110,250`. The crowd's average is up 10.25%. Your single most-likely two-round outcome — exactly one head and one tail — is down 10%. The crowd and the individual have already split, after just two flips.

Push it to many rounds and the split becomes a chasm. The probability of getting roughly half heads grows with the number of flips (that's the law of large numbers), and *roughly half heads is exactly the path that decays*, because each balanced head-and-tail pair multiplies your money by 0.9. The longer you play, the more certain you are to converge to the geometric outcome — which is decay.

#### Worked example: a hundred rounds, the typical path

A player who flips the coin 100 times and gets the typical split — 50 heads, 50 tails — ends with:

```
W = $100,000 × 1.5^50 × 0.6^50
  = $100,000 × (1.5 × 0.6)^50
  = $100,000 × 0.9^50
  = $100,000 × 0.00515
  ≈ $515
```

A \$100,000 account, on a coin with positive expected value, played for 100 rounds, ends at about **\$515** for the typical player. That is a 99.5% loss. And 50/50 is not a tail outcome — it is the single *most likely* number of heads. The time-average trajectory, `$100,000 × 0.9487^100`, lands at the same \$515.38, because 0.9487 is just `0.9^(1/2)` per round. Your most-likely path and your long-run growth rate agree, and both say: ruin.

*Positive expected value tells you the crowd prospers; it is silent on the fact that the crowd's prosperity is being financed by your near-certain decay.*

This is not a contrived edge case. It is the generic behaviour of any multiplicative gamble where the geometric mean is below 1 while the arithmetic mean is above it. And because *any* gamble with both upside and downside has a geometric mean strictly below its arithmetic mean (that's a mathematical certainty — see the box below), there is *always* a gap. The only question is whether you've sized the bet small enough to keep the geometric mean above 1. That sentence is the whole of position sizing.

It's worth lingering on *why* the typical path equals the decaying path, because it's the load-bearing claim and it's counterintuitive the first time. Over 100 flips, the number of heads follows a binomial distribution centered on 50. The most likely single outcome is exactly 50 heads, and the overwhelming bulk of the probability sits within a few flips of 50 — getting, say, 40 or fewer heads has a probability under 3%, and getting 60 or more is equally rare. So when we ask "what does a randomly chosen player most likely experience?", the answer is "something very close to 50 heads and 50 tails." And 50 heads with 50 tails is, exactly, `0.9^50 ≈ $515`. The decaying outcome isn't a pessimistic scenario you're warning about; it is the *modal* outcome, the peak of the probability distribution. The handful of paths that get 60+ heads and end rich are the genuine rarities. Expected value, by averaging in those rarities at full weight, paints a picture of an experience almost no actual player has.

There's a second, sharper way to see it that connects directly to the law of large numbers. Define your *per-flip log-return* as the logarithm of the multiplier — `ln(1.5) ≈ +0.405` on heads, `ln(0.6) ≈ −0.511` on tails. Your wealth after `N` flips is `W₀ × exp(sum of the N log-returns)`. By the law of large numbers, the *average* of those log-returns converges, as `N` grows, to its expectation: `0.5 × 0.405 + 0.5 × (−0.511) = −0.053`. So with near-certainty, for large `N`, your wealth behaves like `W₀ × exp(−0.053 × N)` — exponential *decay*. The law of large numbers, the very theorem people invoke to argue "it'll work out in the long run," is here *guaranteeing* your ruin, because what it makes certain is the average *log*-return, and the average log-return is negative. This is the precise sense in which the time-average growth rate is the expectation of the log-return, not the log of the expected return — and those two quantities are different numbers whenever there's any spread in outcomes.

> **Why the geometric mean is always below the arithmetic mean.** For any set of positive numbers that aren't all identical, the geometric mean is strictly less than the arithmetic mean (the AM–GM inequality). For wealth, the gap is well-approximated by half the variance of returns: `geometric growth ≈ arithmetic growth − ½ × variance`. The more your outcomes spread out — the more *volatile* you are — the bigger the penalty subtracted from your long-run growth. Volatility isn't just discomfort; it is a direct, quantifiable tax on the rate at which you compound. We derive this properly via the heavy distribution math in [probability distributions for markets](/blog/trading/math-for-quants/probability-distributions-for-markets-math-for-quants).

## The cloud of lives: mean rising, median falling

The cleanest way to *see* non-ergodicity is to simulate ten thousand people playing the same coin and watch what happens to the average versus the typical player. Figure 2 plots a sample of those ten thousand paths as a faint grey cloud on a logarithmic wealth axis, then overlays the two summary lines that matter: the **mean** (the ensemble average) climbing, and the **median** (the typical player) sinking.

![Spaghetti chart of ten thousand simulated coin flip wealth paths on a log scale with the rising mean line and the falling median line diverging over time](/imgs/blogs/ergodicity-time-average-vs-ensemble-average-and-the-coin-flip-that-ruins-you-2.png)

The picture is unambiguous and, the first time you see it, genuinely disorienting. The green mean line rises, exactly as expected value promised. The red median line falls, exactly as the geometric mean warned. They are computed from the *same ten thousand paths*. Nothing is contradictory; the mean and the median of a strongly right-skewed distribution are simply different numbers, and for multiplicative wealth they diverge without limit as time goes on.

Here is the mechanism, stated plainly. Most paths decay, because most paths get roughly half heads and roughly-half-heads compounds at 0.9 per pair. A *few* paths get a lucky streak of extra heads early, when the base is still large, and those few paths explode to enormous values. The mean is an *additive* summary — it adds up every path and divides — so a single path worth a billion dollars can single-handedly hold the mean up while ten thousand other players sit near zero. The median couldn't care less about the billionaire: it just asks "what does the person in the middle of the pack have?" and the answer is "almost nothing."

The casino, it turns out, *is* the mean. The gambler *is* the median. Same coin, opposite fate. We'll come back to that.

#### Worked example: where the mean's money actually is

Simulate 200,000 players for 100 rounds (seeded, so it's reproducible). The realized sample mean comes out around **\$6,400,000** per player and the exact mathematical expectation is `$100,000 × 1.05^100 ≈ $13,150,000`. The *median* player ends with about **\$515**. Ask where the mean's millions live:

- **86.5%** of all 200,000 players end with *less* than the \$100,000 they started with.
- **54%** of players end with *less than 1%* of their stake — under \$1,000 left from \$100,000.
- The few thousand players in the extreme right tail — the ones who hit long early head-streaks — hold essentially all of the mean's wealth.

The sample mean (\$6.4M) even *undershoots* the true expectation (\$13.15M), and that gap is itself a symptom: the distribution is so heavy-tailed that 200,000 samples aren't enough to reliably catch the rare giant winners that the mean is made of. The expected value is concentrated in events so rare your simulation barely sees them — and that you, with one life, will never see at all.

*If the only way to collect the average is to be one of the handful of players the average is built from, then "the average is positive" is cold comfort to the 86.5% who are underwater.*

## The two growth rates, side by side

We've now met the two numbers — the arithmetic +5% and the geometric −5.1% — as facts about a fixed bet (stake everything every round). But the more useful version asks: what if you only bet a *fraction* of your wealth each round? That single knob, the fraction staked, is where risk management lives, and it makes the two growth rates into two *curves*. Figure 3 plots them against the fraction of wealth you put at risk on each flip.

![Line chart comparing arithmetic ensemble growth rising with bet fraction against geometric time growth that peaks then turns negative as the bet fraction increases](/imgs/blogs/ergodicity-time-average-vs-ensemble-average-and-the-coin-flip-that-ruins-you-3.png)

Read the green line first. The **arithmetic / ensemble** growth is `+0.05 × f` — it just rises linearly with the fraction `f` you stake. The crowd's average always benefits from betting more; double the stake, double the expected gain. If expected value were what you experienced, the optimal bet size would be *everything*, every time. This is precisely the advice that bankrupts people.

Now the amber line — the **geometric / time** growth, `√((1 + 0.5f)(1 − 0.4f)) − 1`. It starts at zero (bet nothing, grow nothing), rises to a peak, and then turns *negative*. Past a certain stake, betting more *lowers* the rate at which your single path compounds, even though it raises the crowd's average. At `f = 1` — bet your whole bankroll every round — the time-growth is the −5.1% we already met. The red shaded region is the zone where the average is positive but *your path loses money*: the entire gap between optimism and ruin.

#### Worked example: finding the stake that maximises *your* growth

The geometric growth `g(f) = √((1 + 0.5f)(1 − 0.4f)) − 1` is maximised at `f* = 0.25` — stake one quarter of your wealth per flip. At that fraction:

- Heads: wealth `× (1 + 0.5 × 0.25) = × 1.125` (+12.5%).
- Tails: wealth `× (1 − 0.4 × 0.25) = × 0.90` (−10%).
- Per-round time-growth: `√(1.125 × 0.90) − 1 = √1.0125 − 1 ≈ +0.62%`.

A positive, sustainable +0.62% per round — versus the −5.1% you'd compound at by betting everything. Same coin, same edge; the *only* thing that changed is the fraction staked, and it flipped your personal fate from ruin to slow growth. This optimal fraction is the **Kelly criterion**, and it falls straight out of maximising geometric (time) growth — derived in full in [the Kelly criterion](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews).

*Position sizing is not caution applied to a good bet; it is the act of choosing where on the amber curve you live, and the wrong choice turns a positive edge into a negative life.*

The amber curve is the most important object in risk management, and it explains a paradox practitioners feel but rarely name: a strategy can have *too much* edge to bet fully. Over-betting a winning system is not aggressive, it is suicidal, because it pushes you off the right-hand side of the amber curve into the red zone where the ensemble still rises but your equity curve rolls over. The number you are paid to maximise is the geometric one. The arithmetic one is a siren.

## Why multiplying breaks the average over time

We keep saying multiplication is the culprit. Figure 4 makes the *why* concrete by putting the additive world and the multiplicative world side by side.

![Before and after comparison showing additive dynamics where order does not matter and ensemble equals time average versus multiplicative dynamics where order is fate and the averages decouple](/imgs/blogs/ergodicity-time-average-vs-ensemble-average-and-the-coin-flip-that-ruins-you-4.png)

In the additive world on the left, the coin pays a *fixed dollar amount*: win \$50 or lose \$40, no matter your current wealth. Here the order of outcomes is irrelevant in the deepest sense — not just because addition commutes, but because each outcome is independent of your history. A loss doesn't shrink the base that future wins are computed on, because there is no "base"; you just add. In this world the average across people and the average over time are *the same number*. The game is **ergodic**. One gambler over a thousand rounds and a thousand gamblers over one round each see the same +\$10-per-round drift. Additive gambles are safe to reason about with expected value, because expected value *is* the time average.

In the multiplicative world on the right, the coin pays a *percentage* — and now your history is load-bearing. A 40% loss removes 40% of *everything you have*, permanently shrinking the base that every future percentage gain will be applied to. A head-then-tail and a tail-then-head both leave you at \$90,000, below your \$100,000 start. The ensemble average keeps rising at +5% because it's computed across fresh, full-sized stakes; your path compounds at −5% because it's computed along a *single shrinking* stake. The two averages have **decoupled**. The game is **non-ergodic**, and expected value is no longer the time average — it has come unmoored from anything you'll experience.

This is the whole secret. Ergodicity isn't an exotic property; it's just the question *"does my history matter?"* When outcomes add, history doesn't matter and the crowd average is your average. When outcomes multiply — as wealth, leverage, and compounding all do — history matters intensely, and the crowd average becomes a number that exists for the crowd alone.

#### Worked example: the \$10,000,000 book and the percentage that doesn't forgive

Take a \$10,000,000 trading book that runs the multiplicative coin once a quarter. Suppose it gets one head and one tail over two quarters — a perfectly average year by count of wins:

- Q1 heads: `$10,000,000 × 1.5 = $15,000,000`.
- Q2 tails: `$15,000,000 × 0.6 = $9,000,000`.

The book is down \$1,000,000 on a 1-and-1 record. Now compare what additive accounting would have said: a fixed +\$5,000,000 / −\$4,000,000 pair would have left the book at `$10M + $5M − $4M = $11,000,000`, *up* a million. Same record, opposite sign, and the only difference is whether the win and loss were dollars (additive) or percentages (multiplicative). A real book is multiplicative — your P&L is a percentage of capital that scales with the capital — so the \$9,000,000 outcome is the real one. The −10% has to be earned back from a \$9,000,000 base, requiring a +11.1% gain, not the +10% you "lost".

*On a percentage book a 1-and-1 record is a losing record, and no amount of being right half the time changes that arithmetic — only sizing the percentages down does.*

## Where ergodicity breaks: the loss that compounds against your future

We can now name the exact mechanism that decouples your path from the crowd. Figure 6 traces it as a causal chain: each round multiplies whatever the last round left you, so a loss doesn't just hurt today — it shrinks the base that *every future bet* will work on, which is what makes a drawdown path-dependent and permanent.

![Causal flow diagram showing how a loss shrinks the compounding base creating path dependence which decouples the time average from the ensemble average and drives the path toward ruin](/imgs/blogs/ergodicity-time-average-vs-ensemble-average-and-the-coin-flip-that-ruins-you-6.png)

Follow the chain. Your wealth this round is the *full base* that all future gains will multiply against. A head multiplies that base by 1.5 — but if a previous tail already cut the base by 40%, the 1.5 is acting on a smaller number, so the recovery is worth less in dollars than the loss cost. A tail multiplies by 0.6, permanently removing 40% of the compounding base. That smaller base means every later bet now works on less capital; the ceiling on your future wealth has dropped and it does not come back just because you win the next flip. This is **path dependence**: the order and history of outcomes determine where you end up, because a deep loss caps all later compounding. Path dependence is exactly what it means for the **time average to decouple** from the ensemble average — and once decoupled, the typical path drifts down at the geometric rate toward zero.

The deepest consequence is **absorbing ruin**. In the multiplicative world, zero is a wall you can't bounce off. If a sequence of percentage losses takes you to zero (or to the margin call that forces liquidation), no future percentage gain helps — 50% of nothing is nothing. The ensemble average doesn't have this problem, because for the crowd there is always *another* player with a full stake to pull the average back up. You have no other player. You have one path, and on that path zero is forever. This is why "survive first, optimise second" is not a platitude but a mathematical necessity, developed as the core engine in [survival as a compounding engine](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine).

## The casino and the gambler: same odds, opposite fate

Everything above crystallises into one comparison that you already half-know from real life. The casino and the gambler face *identical odds* on every bet. One of them gets rich with mathematical certainty; the other goes home broke with near-certainty. The difference is not the odds. It is that the casino lives the ensemble average and the gambler lives the time average. Figure 7 lays them side by side.

![Before and after comparison contrasting the casino running thousands of simultaneous independent bets as an ergodic ensemble against a single gambler running sequential bets through time as a non ergodic path](/imgs/blogs/ergodicity-time-average-vs-ensemble-average-and-the-coin-flip-that-ruins-you-7.png)

The casino, on the left, runs thousands of bets *simultaneously* — many tables, many players, all at once. Its outcome on any given night is the *average across* all those independent bets, and the law of large numbers locks that average tight around the house edge. The casino's risk is spread across a parallel ensemble; no single loss can ruin it, because the losers are diversified against the winners *in the same instant*. The house edge compounds steadily, and the casino survives essentially forever. The casino *is* the ensemble.

The gambler, on the right, runs the same bets *in sequence*, one bankroll, one wager riding on whatever the last wager left behind. There is no parallel ensemble to average over — there is only this one path through time. The compounding is multiplicative: a bad run shrinks the base every later bet uses, losses are path-dependent, and the time average sits below break-even. Worst of all, ruin is absorbing: hit zero and the gambler is out for good, with no future bet to recover from. The gambler *is* the time average.

This is the punchline of the whole post. The casino and the gambler aren't separated by skill or by odds. They're separated by *whether they get to be an ensemble or are forced to be a path*. And here is the uncomfortable mirror: as a trader running a single account through time, **you are the gambler, not the casino** — even when your edge is real. Your job in risk management is to borrow as much of the casino's structure as you can: diversify across many *simultaneous* uncorrelated bets so your single path behaves a little more like an ensemble, and size each bet small enough that no sequence of losses can take you to the absorbing wall. That's it. That's the discipline. The connection to [risk of ruin and why positive expectancy is not enough](/blog/trading/risk-management/risk-of-ruin-why-positive-expectancy-is-not-enough) is direct: ruin probability is just the chance your one path hits the wall before the time average can rescue you.

## Stealing the casino's edge: how to make your path more ergodic

If the problem is that you're a single non-ergodic path and the casino is a happy ergodic ensemble, the practical question becomes obvious: *how do I make my path behave more like an ensemble?* This is exactly what good risk management does, and naming it this way turns a grab-bag of received wisdom — diversify, size down, buy insurance, rebalance — into a single, coherent program. Every one of those moves is a technique for pulling your time-average growth rate up toward your ensemble-average growth rate. Here are the three that matter most.

**Size the bet down — the cheapest cure.** We already saw on the growth-rate curve that staking a quarter of your wealth turns a −5.1% path into a +0.62% path. Sizing down is the most direct way to raise your geometric growth, because geometric growth is `arithmetic − ½ × variance`, and shrinking the bet shrinks the *variance* term quadratically while shrinking the *edge* term only linearly. Cut your bet in half and you keep half the edge but only a quarter of the variance penalty. That asymmetry is why small bets compound and large bets ruin: the variance drag, the thing that separates the time-average from the ensemble-average, falls away fastest exactly when you most need it to.

#### Worked example: the same edge, two bet sizes, two fates on a \$100,000 account

Run the coin 100 times on \$100,000, once at full stake and once at quarter stake (the median 50/50 path):

- **Full stake (f = 1):** each pair multiplies by `1.5 × 0.6 = 0.9`. After 50 pairs: `$100,000 × 0.9^50 ≈ $515`. A 99.5% loss.
- **Quarter stake (f = 0.25):** heads `× 1.125`, tails `× 0.90`, each pair multiplies by `1.125 × 0.90 = 1.0125`. After 50 pairs: `$100,000 × 1.0125^50 ≈ $185,000`. An 85% *gain*.

Identical coin, identical edge, identical 50/50 luck — and the only difference is bet size. Full stake takes \$100,000 to \$515; quarter stake takes the same \$100,000 to roughly \$185,000. *Position size is not a detail layered on top of a good strategy; it is the variable that decides whether a positive-edge strategy makes you rich or broke.*

**Diversify across simultaneous, uncorrelated bets — manufacture a private ensemble.** A casino's edge comes from running many bets *at the same time* so the law of large numbers averages the variance away within each instant. You can imitate this. Instead of putting your whole stake on one flip, split it across ten *independent* coins flipped at once. Your one-period outcome is now the average of ten independent coins, which has the same +5% expectation but one-tenth the variance — and lower variance means higher geometric growth. You've built a small ensemble *inside* your single path. This is the real, ergodicity-grounded reason diversification raises return rather than merely smoothing it: each uncorrelated position is a parallel "player," and parallel players are what turn a gambler into a casino. The catch is *uncorrelated* — and in a crisis correlations rush to 1, your ten coins become one coin, your private ensemble collapses, and you're a single path again at the worst possible moment. That collapse is a risk failure mode, not an allocation choice, and it's covered in [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).

**Cap the downside — change the dynamics so ruin is unreachable.** Insurance and stop-losses look like they cost you expected value (you pay a premium, you accept a small certain loss), and from the *ensemble* point of view they do. But from the *time-average* point of view they can raise your growth rate, because they truncate the left tail that drags the geometric mean down and, crucially, push the absorbing wall out of reach. A bet that can lose 40% has a worse geometric mean than the same bet capped to lose at most 15%, even though the cap costs something in expectation. This is the ergodic resolution of the old puzzle "why does anyone buy insurance if it has negative expected value?" — insurance has negative *ensemble* value and positive *time-average* value for the buyer, because the buyer is a single non-ergodic path for whom a large multiplicative loss is permanently destructive. The hedging mechanics — protective puts, collars, tail hedges — are developed in the options series; here the point is only *why* paying for them can raise the rate at which your one path compounds. The detailed mechanics live in [hedging a portfolio with options](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk).

All three cures do the same underlying thing: they shrink the gap between your time-average and the ensemble-average by attacking variance and by walling off the absorbing zero. That is the entire technical content of risk management, stated in one sentence — and it falls directly out of taking ergodicity seriously.

## The shape of the end: a few winners, a broke majority

We've leaned on the phrase "strong right skew" several times; Figure 5 shows it directly. After 100 rounds, the distribution of terminal wealth (plotted in log dollars, because the spread is enormous) is wildly lopsided. The bulk of players pile up at the low end, near zero. A thin tail stretches far to the right — the lucky few who caught early head-streaks. The mean sits *way* out to the right of the median, dragged there by that tail.

![Histogram of terminal wealth after one hundred rounds on a log dollar axis showing a strong right skew with the median near zero and the mean far to the right pulled up by a few large winners](/imgs/blogs/ergodicity-time-average-vs-ensemble-average-and-the-coin-flip-that-ruins-you-5.png)

This shape — **log-normal**, the natural distribution of anything that grows by repeated multiplication — is why mean and median diverge so violently and why the divergence *grows* over time. Take logarithms of the terminal wealth and the picture becomes a tidy bell curve, because the log of a product is a sum of independent terms and sums of independent terms go normal (the central limit theorem). But the *center* of that log-bell sits to the *left* of zero growth, since the average log-return per round is `0.5 × ln(1.5) + 0.5 × ln(0.6) = 0.5 × 0.405 + 0.5 × (−0.511) = −0.053`, a negative log-drift. Exponentiate that bell back to dollars and the long left side of "log space" squashes into the near-zero pile, while the short right side stretches into the explosive tail. The mean lives in the tail; the median lives in the pile; you live wherever your one draw lands, and the odds say the pile.

#### Worked example: the median is the time-average, exactly

For this multiplicative game the median terminal wealth and the time-average trajectory are the *same number*, and that's not a coincidence. The median player gets the median number of heads (50 of 100), and:

```
Median wealth = $100,000 × 1.5^50 × 0.6^50 = $100,000 × 0.9^50 ≈ $515
Time-average  = $100,000 × (√0.9)^100      = $100,000 × 0.9487^100 ≈ $515
```

Both land at \$515.38. The median *is* the geometric-mean path because the log-distribution is symmetric, so its center (the mean of the logs) maps to the median of the dollars. The arithmetic mean, meanwhile, has run off to \$13,150,000 in the tail. **The median is what you should plan around, and the median is the time-average.** When someone quotes you an expected return, ask for the median; if they don't know the difference, they're quoting you the casino's number while you're holding the gambler's ticket.

*On a multiplicative game, "what does the typical player end with" and "what is my long-run growth rate" are the same question with the same brutal answer — and it is not the mean.*

## The 300-year-old version of the same mistake

The coin flip isn't a modern curiosity; it's a sharpened version of a puzzle that has confused brilliant people for three centuries. In 1713 Nicolas Bernoulli posed it; in 1738 his cousin Daniel tried to resolve it. The setup, the **St. Petersburg paradox**, is a game where a fair coin is flipped until it lands heads, and the pot doubles with every preceding tail: \$2 if heads comes on flip one, \$4 if on flip two, \$8 on flip three, and so on. The expected payoff is `½ × $2 + ¼ × $4 + ⅛ × $8 + …`, which is `$1 + $1 + $1 + …` — an *infinite* sum. By the rules of expected value you should be willing to pay any finite amount — your entire fortune, a billion dollars — to play once. Yet nobody would pay more than a handful of dollars, and they are right not to.

For 300 years the textbook resolution was Daniel Bernoulli's: people value money according to a *concave utility function* (a logarithm, he suggested), so the marginal dollar of a huge payoff is worth less than the first dollar, which tames the infinite sum. This works numerically but it smuggles in a psychological assumption — that you're *risk-averse*, that your brain discounts large sums — to fix what is actually a *dynamics* problem. The ergodicity reframing is cleaner and assumes nothing about your psychology: the reason you won't pay a fortune is that **paying a fortune to play once is a multiplicative bet with a terrible time-average.** If you wagered your whole \$100,000 net worth on a single St. Petersburg ticket, the overwhelmingly likely outcome is that the coin lands heads early, you collect \$2 or \$4, and you've vaporised your wealth — the time-average growth of "bet everything on this" is catastrophic, even though the ensemble-average payoff is infinite. You decline not because you're squeamish about uncertainty but because, as a single path through time, the bet would ruin you with near-certainty.

This is the same split we've been tracing: a positive (here, infinite) *ensemble* expectation sitting on top of a *time-average* that's disastrous for the individual. Bernoulli reached for the log utility and got the right answer for what turns out to be a slightly wrong reason — the logarithm shows up not because human happiness is logarithmic but because **the logarithm is the function that converts multiplicative dynamics into additive ones**, and the time-average of a multiplicative process is exactly the exponential of the average log-return. Three centuries of economists treated a property of the *dynamics* as a property of *preferences*. Once you see that, you stop asking "how risk-averse should I be?" and start asking "what is the time-average growth of this bet?" — and the second question has an objective answer that doesn't depend on how you feel about risk.

## Common misconceptions

**"Positive expected value means I'll make money in the long run."** No — for a multiplicative bet, positive *arithmetic* expected value is compatible with *negative* long-run (geometric) growth. The +50%/−40% coin has +5% expected value per round and a −5.1% time-growth rate at full stake. The number that governs your long run is the geometric one, and it can be negative while the arithmetic one is positive. Long-run wealth tracks the median, and the median here goes to \$515 from \$100,000.

**"The average and the typical outcome are basically the same."** Only for additive, ergodic processes. For multiplicative wealth they diverge without bound: in our simulation the mean was about \$13,150,000 (exact) while the median was \$515 — a ratio of roughly 25,000-to-1. Quoting the mean of a right-skewed wealth distribution as "what you can expect" is one of the most expensive errors in finance, because the mean is a number almost nobody in the distribution actually has.

**"Volatility is just discomfort; it doesn't change my returns."** Volatility is a direct subtraction from your compound growth: `geometric ≈ arithmetic − ½ × variance`. Two strategies with identical +5% arithmetic edge but different volatilities compound at different rates; the more volatile one ends with less money, with certainty, over a long enough horizon. On the coin, the entire −5.1% vs +5% gap *is* this variance drag. Reducing volatility (by sizing down) literally raises your realized return.

**"Diversification is about smoothing the ride."** Diversification across *simultaneous, uncorrelated* bets is how a single trader steals a piece of the casino's ensemble structure — it raises your geometric growth rate by cutting the variance term, not just the emotional volatility. Ten uncorrelated quarter-Kelly bets compound faster than one full-Kelly bet with the same total edge. When correlations go to 1 in a crisis, that stolen ensemble evaporates and you're back to being one path — which is why a correlation breakdown is a *risk* failure, covered in [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).

**"Ruin is just a really bad outcome, like any other tail."** Ruin is qualitatively different because it is *absorbing*. A −90% drawdown is recoverable in principle (it needs +900%, but the path exists); zero is not, because every future percentage gain on zero is zero. The ensemble average never hits this wall because it always has other full-stake players; your single path can, and once it does, your edge is worth nothing.

**"Backtests prove my strategy works, so I should run it at full size."** A backtest is a single historical path — one draw from the ensemble of paths the strategy *could* have produced. It tells you the strategy's edge existed, not the size you can safely run it at. A strategy with a great backtested Sharpe ratio can still have a left tail that, at full leverage, takes your *live* path to the absorbing wall on a sequence the backtest happened not to contain. The backtest reports something close to the ensemble average (it's smoothed by hindsight and survivorship); the live account lives the time average, with one shot and a wall. Size for the path you'll live, not the path you already know turned out fine.

**"If the bet is positive-expectancy, holding longer only helps."** For a multiplicative bet sized too large, holding *longer* makes you *more* certain of ruin, not less, because the longer you play the more tightly your realized growth converges to the (negative) geometric rate. Time is the gambler's enemy when the per-round geometric growth is negative, and the gambler's friend only once you've sized the bet down into the region where it's positive. The law of large numbers, the very theorem invoked to argue "it'll work out if I'm patient," applies *across the ensemble*, not *along your single path* — along your path it guarantees convergence to the time-average growth rate, which here is decay. The LLN is the casino's friend, not the gambler's.

## How it shows up in real markets

Ergodicity is abstract until you watch a real book confuse the ensemble for the time average and walk into the wall. These are the canonical cases, with the dates and numbers.

**LTCM, August–September 1998.** Long-Term Capital Management ran convergence trades at roughly 25-to-1 balance-sheet leverage on about \$4.7 billion of equity, with around \$1.25 trillion of gross derivative notional. Their models said the *ensemble* of their thousands of small, diversified spread bets had a positive, low-variance expected return — the casino's math. But they were one fund, one path through time, and when a flight to quality in 1998 drove their supposedly-uncorrelated trades to correlation 1, the multiplicative losses compounded against a leveraged base and removed about \$4.6 billion of capital in roughly four months, forcing a \$3.6 billion Fed-organised rescue. They had the casino's diversification on paper and the gambler's single path in reality. (The strategic dimension — being the crowded trade everyone had to exit at once — is in [the LTCM case study](/blog/trading/game-theory/case-study-ltcm-1998-the-crowded-genius-trade).)

**Amaranth, September 2006.** Amaranth Advisors lost roughly \$6.6 billion — most of it in a single week — on concentrated, levered natural-gas calendar spreads. One bet, one illiquid book, one path. The expected value of the position may have been positive in their model's ensemble sense, but a single adverse sequence in an illiquid market is a multiplicative loss with no parallel bets to average against. The path hit the wall.

**Archegos, March 2021.** Archegos held concentrated single-stock exposure financed via total-return swaps at around 5×+ leverage, with each prime broker blind to the *total* size. When the underlying stocks fell, margin calls forced liquidation, and the multiplicative, leveraged losses cascaded — over \$10 billion in aggregate losses to the banks, about \$5.5 billion at Credit Suisse alone. A leveraged single path met a sequence of losses it could not survive; the ensemble math (each swap "diversified") was an illusion once the concentration showed up all at once.

**Volmageddon, 5 February 2018.** The VIX jumped from 17.3 to 37.3 in a day (its largest one-day percentage rise) and the XIV short-volatility note lost about 96% of its value after the close and was terminated. Short-vol carry has *positive expected value most of the time* — the ensemble of quiet days is profitable — which is exactly the ergodicity trap: a strategy whose arithmetic edge is positive but whose time-path carries an absorbing tail. One bad day was a multiplicative loss large enough to take the single path to zero. (The mechanics are in [the Volmageddon case study](/blog/trading/options-volatility/case-study-volmageddon-2018-and-the-short-vol-blowup).)

**COVID, February–March 2020, and the yen-carry unwind, 5 August 2024.** Both were episodes where correlations went to 1 (VIX peaked at 82.69 on 16 March 2020 as the S&P 500 fell about 34% peak-to-trough in roughly a month; the Nikkei fell 12.4% on 5 August 2024, its worst day since 1987, with the VIX spiking to an intraday 65.7) and crowded, levered single paths faced multiplicative losses with no ensemble to hide in. The yen-carry trade is the purest illustration: borrow cheaply in yen, lend in higher-yielding currencies, and collect the spread — a strategy with a *positive ensemble expectation* that pays a little every quiet day, exactly the shape that hides an absorbing tail. When the yen reversed sharply in days, the leverage turned a modest adverse move into a near-fatal multiplicative loss, and the unwinding became reflexive: forced sellers drove the move that forced more selling. In each case the trades that "diversified" in calm regimes turned into one correlated path the moment it mattered — the recurring way a portfolio that thought it was the casino discovers it was the gambler. The crowded-exit dynamics behind these unwinds, where everyone's single path becomes the *same* single path at once, are the strategic mirror image of the statistical story told here.

The thread through every case: a positive *ensemble* expectation, a leveraged or concentrated *single* path, and a sequence of multiplicative losses that the ensemble math said couldn't ruin them — but ruined them, because they were never the ensemble.

## The risk playbook

Ergodicity converts vague prudence into hard rules. Here is how to trade as the gambler you are while stealing what you can from the casino.

- **Size for the time-average, never the ensemble.** Maximise geometric growth, not arithmetic expected value. Practically: never stake so much that a plausible losing streak pushes your per-round geometric growth negative. On the coin, full stake gives −5.1%/round; quarter stake gives +0.62%/round. The cap on bet size is set by *your* growth curve (the amber line), not the crowd's.
- **Bet a fraction, and bet less than full Kelly.** The growth-optimal fraction (Kelly) maximises geometric growth, but it's brutally volatile and assumes you know your edge exactly — which you don't. Half-Kelly keeps ~75% of the growth for ~half the volatility, and volatility is a direct tax on geometric growth. When uncertain about your edge, *underbet*; the cost of underbetting is linear and small, the cost of overbetting is ruin.
- **Treat any percentage drawdown as a permanent cut to your compounding base.** A −40% loss isn't a −40% event you'll average out; it's a 40% reduction in the capital every future bet works on, requiring +66.7% just to recover. Set a hard drawdown limit (e.g. cut size by half at −15%, stop at −25%) *before* the path gets near the absorbing wall, because near the wall the arithmetic is merciless.
- **Manufacture an ensemble out of your single path.** Run many *simultaneous, genuinely uncorrelated* bets rather than one big one. Each one is a parallel "player," and the diversification cuts the variance term, raising your geometric growth. But audit the correlations honestly — in a crisis they go to 1 and your manufactured ensemble collapses back into one path, which is when blow-ups happen.
- **When someone quotes you a return, ask whether it's the mean or the median.** The mean is the casino's number; the median is yours. For any multiplicative, skewed payoff they diverge — sometimes by thousands of times. Plan around the median (the time-average) and treat the mean as the property of a tail you probably won't draw.
- **Above all: survive. Ruin is absorbing.** No edge, no expected value, no model survives contact with zero. The first and last job of risk management is to keep your single path away from the wall, because the time-average can only compound a path that's still in the game.

### Further reading

- [Risk of ruin: why positive expectancy is not enough](/blog/trading/risk-management/risk-of-ruin-why-positive-expectancy-is-not-enough) — the probability your one path hits the wall before the time-average can rescue it.
- [The asymmetry of losses: why a 50% loss needs a 100% gain](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) — the recovery math that lives at the heart of multiplicative ruin.
- [Risk management as the only free lunch: survival as a compounding engine](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine) — why staying in the game is the highest-return decision you make.
- [Probability distributions for markets](/blog/trading/math-for-quants/probability-distributions-for-markets-math-for-quants) — the log-normal, the variance drag, and the heavy-tail math underneath this whole post.
