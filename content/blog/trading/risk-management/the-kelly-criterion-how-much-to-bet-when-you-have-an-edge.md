---
title: "The Kelly Criterion: How Much to Bet When You Have an Edge"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Finding an edge and sizing it are two separate decisions, and the growth-optimal answer to the second is the Kelly criterion: bet the fraction that maximizes the long-run growth rate, because both under-betting and over-betting are mistakes."
tags: ["risk-management", "kelly-criterion", "position-sizing", "geometric-growth", "leverage", "log-wealth", "compounding", "over-betting", "edge"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **One-sentence thesis:** Finding an edge and deciding how much to bet on it are two separate problems, and the growth-optimal answer to the second is the Kelly criterion — bet the fraction that maximizes the long-run growth rate of your wealth, which is the same as maximizing the expected *logarithm* of wealth.
> - For a simple bet that wins `b`-to-1 with probability `p`, the Kelly fraction is `f* = (p(b+1) − 1) / b` — equivalently, **edge divided by odds**. A coin that pays even money and wins 55% of the time gives `f* = 10%`.
> - The continuous version, for a strategy with arithmetic drift `μ` and volatility `σ`, is the **Kelly leverage** `L* = μ/σ²`.
> - **Both errors are real.** Under-betting leaves growth on the table; over-betting destroys it. Betting *twice* the Kelly fraction gives you **exactly zero** long-run growth — all that risk for nothing — and more than that compounds you toward ruin.
> - The penalty is **asymmetric**: half-Kelly keeps about three-quarters of the growth at a fraction of the risk, while double-Kelly throws all the growth away. That asymmetry is why almost nobody bets full Kelly in practice.
> - Kelly is the *sizing* answer that sits downstream of the ruin and ergodicity results: it is what you do once you accept that you live on one path through time, not the average across many.

In 1956 a researcher at Bell Labs named John Larry Kelly Jr. published a paper with the dry title "A New Interpretation of Information Rate." It was nominally about how much an edge in a noisy communication channel was worth. Buried inside it was the answer to a question gamblers and investors had been getting wrong for centuries: *given that you have an edge, exactly how much of your money should you put at risk on each bet?* Not "should you bet" — you've already decided that, you have an edge — but *how much*. Kelly's answer was a single clean formula, and it turned out to be the unique bet size that makes your money grow as fast as it possibly can over the long run without driving you to ruin.

Here is the thing almost everyone gets wrong, and it's the whole reason this post exists. **Having an edge and sizing the bet are two completely different decisions.** You can have a genuine, real, repeatable edge — a coin that lands heads 55% of the time, a strategy with positive expected return — and still go broke, not because the edge was fake, but because you bet the wrong *amount*. Bet too little and you crawl when you could compound. Bet too much and you hand the edge back, then keep paying until there's nothing left. The edge tells you *which way* to bet. Kelly tells you *how big*. Confusing the two is one of the most expensive mistakes in all of finance.

This is the post that opens Track C of this series — position sizing and leverage — and it is the practical sizing answer to the survival mathematics we built earlier. If you've read [why positive expectancy is not enough](/blog/trading/risk-management/risk-of-ruin-why-positive-expectancy-is-not-enough), you already know that a positive expected value does not save you from ruin, because ruin is an absorbing barrier and you only get one path through time. Kelly is the constructive other half of that warning: it is the *exact* bet size that maximizes your growth along that single path while keeping you off the barrier. Figure 1 is the entire idea in one curve — your long-run growth rate plotted against how much of your bankroll you bet each round. It rises to a single peak at the Kelly fraction, then falls back through zero and into negative territory. There is one right answer, and standing on either side of it costs you.

![Line chart of long-run growth rate per bet against the fraction of bankroll bet each round, rising to a single peak at the Kelly fraction near ten percent, crossing zero at twenty percent which is twice Kelly, and turning steeply negative beyond that](/imgs/blogs/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge-1.png)

## Foundations: the building blocks, defined from zero

Before we can derive anything, we need five ideas defined with no jargon left in them. If you've read the ergodicity and ruin posts you can skim this; if not, this section is the whole foundation, and every later derivation leans on it.

**An edge.** An edge is any situation where the odds are in your favor — where, if you could play the same bet a huge number of times, you'd come out ahead on average. A coin that lands heads 55% of the time, when heads pays you even money, is an edge: over many flips you win more than you lose. A trading strategy whose average return is positive after costs is an edge. The edge is a property of the *bet*, not of how much you stake on it. This post takes the edge as given — finding edges is a different discipline — and asks only how to size it.

**Odds, and the payout `b`.** When you make a bet, you risk some stake to win some amount. The payout `b` is how much you win *per unit staked* when you win. Even money is `b = 1`: risk \$1 to win \$1. A 2-to-1 payout is `b = 2`: risk \$1 to win \$2. The general bet we'll size wins `b` times your stake with probability `p`, and loses your whole stake with probability `1 − p`. A trade is the same shape with fuzzier numbers: you risk some capital, and the "payout" is the distribution of returns.

**Bankroll and bet fraction.** Your **bankroll** is the total capital you're sizing against — a \$100,000 retail account, a \$10,000,000 book. The **bet fraction** `f` is the share of that bankroll you put at risk on a single bet. Bet `f = 0.10` of a \$100,000 account and you're risking \$10,000. The crucial move in everything that follows is that you re-bet the same *fraction* every round, so your stake scales with your bankroll: after a win you bet more dollars, after a loss you bet fewer. This is **fixed-fractional** betting, and it's what makes wealth multiply rather than add.

**Multiplicative wealth and compounding.** When you bet a fraction of your bankroll, your wealth gets *multiplied* by a factor each round, not increased by a fixed dollar amount. Win an even-money bet at `f = 10%` and your bankroll multiplies by `1 + 0.10 = 1.10`; lose and it multiplies by `1 − 0.10 = 0.90`. Two rounds later your wealth is `W₀ × (first factor) × (second factor)`. This is compounding, and compounding is *repeated multiplication*. The deep consequence — the engine of this entire series — is that multiplied factors behave nothing like added ones. A +10% and a −10% don't cancel: `1.10 × 0.90 = 0.99`, a net loss. This is the [asymmetry of losses](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) showing up the moment money is at risk as a percentage.

**Arithmetic mean vs geometric mean, and the growth rate.** The **arithmetic mean** of your per-round multipliers is what you get by adding them and dividing — it describes the *average across a crowd* of one-shot bettors. The **geometric mean** is what you get by multiplying them and taking the root — and *that* is the factor your single bankroll actually compounds at over time. The long-run **growth rate** is the logarithm of the geometric mean: if your wealth multiplies by an average geometric factor `G` per round, then after `N` rounds it's `W₀ × Gᴺ`, and the per-round growth rate is `g = ln(G)`. Maximizing `g` is the same as maximizing the geometric mean is the same as making your money grow as fast as possible along the one path you actually live. Hold onto that sentence — it is the objective Kelly maximizes, and the reason the whole derivation works.

A quick way to feel the gap between the two means: take a multiplier of `1.5` (a 50% win) and `0.6` (a 40% loss). The arithmetic mean is `(1.5 + 0.6)/2 = 1.05`, suggesting a 5% gain. The geometric mean is `√(1.5 × 0.6) = √0.9 ≈ 0.949`, a 5% *loss*. One number says you grow, the other says you shrink, from the *same two outcomes* — and the geometric one is the truth your account experiences, because wealth multiplies. The whole job of position sizing is to keep that geometric mean as high as possible, and Kelly is the bet size that does it. This same arithmetic-versus-geometric split is the engine of [the ergodicity result](/blog/trading/risk-management/ergodicity-time-average-vs-ensemble-average-and-the-coin-flip-that-ruins-you); Kelly is what you *do* about it.

#### Worked example: one even-money bet on a \$100,000 account

You have a \$100,000 account and a coin that pays even money (`b = 1`) and wins with probability `p = 0.55`. You decide to bet a fraction `f = 0.10` — that's \$10,000 at risk.

- **Win (55% chance):** you gain `b × f = 1 × $10,000 = $10,000`. Your bankroll becomes `$100,000 × (1 + 0.10) = $110,000`.
- **Lose (45% chance):** you lose your \$10,000 stake. Your bankroll becomes `$100,000 × (1 − 0.10) = $90,000`.

The expected dollar change is `0.55 × (+$10,000) + 0.45 × (−$10,000) = $5,500 − $4,500 = +$1,000`, a +1% expected gain on the account. The per-unit edge here is `p×b − (1−p) = 0.55 − 0.45 = 0.10`, ten cents of expected profit per dollar staked.

*The edge is real and positive — but nothing in this single round tells you whether 10% was the right amount to bet; that's the separate question the rest of the post answers.*

## The two-objective trap: maximize wealth, or maximize log-wealth?

Here's where almost everyone goes wrong before they even start. Ask a smart beginner "what should you maximize?" and they'll say, reasonably, "my wealth — I want to end up with as much money as possible." That sounds obviously correct. It is a trap, and it's worth seeing exactly why, because the trap is the doorway to Kelly.

Suppose you'll play the `p = 0.55`, even-money coin many times, always re-betting the same fraction `f`. Consider two different things you might try to maximize.

**Objective A — maximize expected wealth.** Your expected wealth after one bet is `W₀ × (1 + f × edge)` where the edge is 0.10. After `N` bets it's `W₀ × (1 + 0.10f)ᴺ`. Look at that expression: it is *strictly increasing in `f`*. The bigger you bet, the bigger your expected final wealth, with no peak and no limit — it is maximized by betting `f = 100%`, your entire bankroll, every single round. Expected wealth says: bet everything.

**Objective B — maximize the growth rate (expected log-wealth).** Your growth rate per bet is `g(f) = p × ln(1 + bf) + (1 − p) × ln(1 − f)` — the probability-weighted average of the *log* of each outcome multiplier. This function does have a peak, at a finite, sensible `f`, and falls off on both sides.

These two objectives give wildly different answers, and Figure 2 puts them side by side. The left panel is objective A: expected wealth climbing forever as you bet more, maximized at the cliff edge of betting it all. The right panel is objective B: the growth rate, peaking at a modest fraction and turning down.

![Two panel chart, left panel shows expected wealth after many bets rising without limit and maximized by betting one hundred percent of the bankroll, right panel shows the long-run growth rate per bet peaking at the Kelly fraction near ten percent](/imgs/blogs/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge-2.png)

So which objective is right? Objective A — maximize expected wealth — is a disaster in practice, and the reason is everything we built in the [ergodicity post](/blog/trading/risk-management/ergodicity-time-average-vs-ensemble-average-and-the-coin-flip-that-ruins-you). Expected wealth is an average *across a crowd of parallel bettors*. Betting 100% maximizes it only because, in that crowd, a tiny number of impossibly lucky players who never lose end up with astronomical wealth and drag the average to the sky — while every single one of the other players, the moment they hit one loss, goes to *zero* and is finished. The mean is huge; the median is dead. You are not the crowd. You get one path. And on one path, betting 100% means you go broke the first time you lose, which at 55% win odds happens, on average, by your third bet.

Objective B — maximize the growth rate — is the right one *because you live on one path through time*, and the growth rate is the number that path compounds at. Maximizing expected log-wealth is identical to maximizing the geometric mean is identical to making your single bankroll grow as fast as it can. This is not a preference or a risk taste; it is the mathematics of what actually happens to a fixed-fractional bettor over many rounds. **Kelly is what you get when you ask the right question.** And the reason we take the logarithm is precise, not arbitrary: the log turns the *product* of round-by-round multipliers (which is what your final wealth is) into a *sum*, and the average of that sum is the growth rate. Logs convert compounding into adding, and adding is the thing the law of large numbers stabilizes.

#### Worked example: betting it all on a \$10,000,000 book

Take the \$10,000,000 book and bet the whole thing — `f = 100%` — on the same 55%-edge even-money coin, re-betting everything each round.

- **Win (55%):** `$10,000,000 × 2 = $20,000,000`.
- **Lose (45%):** `$10,000,000 × 0 = $0`. Game over, permanently.

Your *expected* wealth after one round is `0.55 × $20,000,000 + 0.45 × $0 = $11,000,000` — a healthy +10%. And if you keep winning, it doubles every round: after ten straight wins you'd have over \$10 billion. That fat expected value is entirely real. But the probability you string together even ten wins is `0.55¹⁰ ≈ 0.25%`, and the probability you're already at zero by round ten is over 99%. The expected value lives almost entirely in a sliver of impossibly lucky paths you will essentially never be on.

*Betting to maximize expected wealth optimizes for a crowd of clones you are not a member of; on your one actual path it guarantees ruin.*

## The whole decision in one picture: which objective do you serve?

Step back and notice what just happened, because it's the conceptual hinge of the entire post. We did not change the bet, the edge, the odds, or the coin. We changed only the *objective* — the thing we declared we were trying to maximize — and the recommended bet size swung from a sane 10% all the way to a suicidal 100%. The objective is not a detail you bolt on at the end; it *is* the sizing decision. Choose to maximize expected wealth and the math hands you ruin with a straight face. Choose to maximize the growth rate and the same math hands you Kelly. Figure 5 lays the two paths side by side so you can see that they diverge from the very first choice you make, long before any number is computed.

![Side by side comparison of two sizing objectives, the left column betting to maximize expected wealth leading through a one hundred percent bet and an absorbing zero to ruin, the right column betting to maximize the growth rate leading through the Kelly fraction and never risking the whole bankroll to compounding and survival](/imgs/blogs/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge-5.png)

The left column is the seductive, wrong path. Its goal — the biggest expected wealth after many bets — sounds like exactly what an investor should want. Follow it honestly and it tells you to bet as much as possible, which drives the optimal fraction to 100% of the bankroll. One loss then takes you to zero, an absorbing barrier you never leave. The expected value still looks magnificent, because it's being held aloft by a handful of lucky paths that never lost — but the *typical* path, the one you'll almost certainly walk, is ruin. The right column is the correct path. Its goal — the biggest expected *log*-wealth, the growth rate per bet — leads to betting the Kelly fraction, which on the 55% coin is 10%, and which by construction never stakes the whole bankroll on a single outcome. The typical path here compounds; you stay in the game. Same edge, same coin, two objectives, two destinies.

This is why I keep insisting that finding an edge and sizing it are different jobs. The edge is fixed across both columns — it's the 55% coin in both. What differs is the *question you ask about the edge*, and the answer to "how much" depends entirely on whether you're optimizing for a crowd you'll never join or for the one life you'll actually live. Everything that follows — the formula, the curve, the leverage, the playbook — is just the right column made precise.

## Deriving Kelly from first principles

Now we'll derive the formula. It's three lines of calculus, and seeing it built makes the result unforgettable — and shows you exactly where it comes from, so you can adapt it.

We want the fraction `f` that maximizes the growth rate

$$
g(f) = p \cdot \ln(1 + bf) + (1-p)\cdot \ln(1 - f).
$$

The first term is the log of your wealth multiplier when you win (you had `1`, you gained `bf`), weighted by the win probability `p`. The second term is the log of your multiplier when you lose (you had `1`, you lost `f`), weighted by `1 − p`. To find the peak, take the derivative with respect to `f` and set it to zero:

$$
g'(f) = \frac{p \cdot b}{1 + bf} - \frac{1-p}{1 - f} = 0.
$$

Cross-multiply: `p·b·(1 − f) = (1 − p)·(1 + bf)`. Expand both sides — `pb − pbf = (1−p) + (1−p)bf` — collect the `f` terms, and after a few lines of algebra the `b` denominators tidy up into a remarkably clean result:

$$
\boxed{f^* = \frac{p(b+1) - 1}{b}}
$$

That is the Kelly criterion. The fraction of your bankroll that maximizes long-run growth is `f* = (p(b+1) − 1) / b`. There's an even more memorable way to read it. Rewrite the numerator: `p(b+1) − 1 = pb − (1 − p)`. The quantity `pb − (1 − p)` is your **edge** — your expected profit per unit staked (what you win times how often, minus what you lose times how often). So

$$
f^* = \frac{\text{edge}}{\text{odds}} = \frac{pb - (1-p)}{b}.
$$

**Kelly is edge divided by odds.** That one phrase is worth memorizing. The more edge you have, the more you bet. The longer the odds you're paid, the *smaller* the fraction you need to bet to capture the same growth, because a long-shot win moves your bankroll more per dollar risked. Everything about how much to bet flows from those two quantities and nothing else.

#### Worked example: the canonical coin, p = 0.55, b = 1

Take the even-money coin (`b = 1`) that wins 55% of the time (`p = 0.55`). Plug into the formula:

```
f* = (p(b+1) − 1) / b
   = (0.55 × (1 + 1) − 1) / 1
   = (0.55 × 2 − 1) / 1
   = (1.10 − 1) / 1
   = 0.10
```

So `f* = 10%`. On the \$100,000 account, full Kelly says risk \$10,000 per bet. Reading it as edge-over-odds gives the same thing: the edge is `pb − (1−p) = 0.55 − 0.45 = 0.10`, the odds `b = 1`, so `f* = 0.10 / 1 = 10%`. For an even-money bet this collapses to the cleanest possible rule: **`f* = 2p − 1`, exactly your win-rate edge.** A 55% coin → bet 10%. A 60% coin → bet 20%. A 50% coin → bet *nothing*, because you have no edge.

*The growth-optimal bet on a 55% even-money coin is 10% of your bankroll — not 5%, not 50%, a specific number you can compute before you ever place the bet.*

## Why the formula is right, beyond the algebra

The calculus is airtight, but it's worth understanding *why* this particular fraction is special, in a way that survives even when the numbers change. The Kelly fraction is the point where one more dollar of bet size adds exactly as much expected log-gain on your wins as it subtracts on your losses — the marginal growth from betting a little more has fallen to zero. Below Kelly, betting a touch more still adds net growth (your wins' contribution outweighs your losses'), so you're leaving growth uncollected. Above Kelly, betting a touch more *subtracts* net growth (your losses' log-penalty now outweighs your wins' log-gain), so you're actively destroying it. Kelly is simply the balance point of that marginal trade, which is exactly what "set the derivative to zero" found.

There's a second deep property worth naming: a Kelly bettor's wealth grows *faster, almost surely, in the long run* than a bettor using any other fixed fraction. This isn't a statement about averages — it's a statement about the actual path. Given enough bets, the Kelly bettor will, with probability approaching one, end up ahead of anyone who systematically bets more or less. That "almost surely" is the strongest possible guarantee in this setting, and it's why Kelly is called *growth-optimal* rather than merely "high expected growth." It's also why the result is robust: it doesn't depend on your risk preferences, your utility function, or your mood. It falls straight out of the multiplicative nature of wealth and the law of large numbers acting on logarithms.

One caution to plant now, though, because it motivates the entire next post: that "in the long run, almost surely" carries fine print. The long run can be *very* long, the drawdowns along the way can be savage, and the whole edifice assumes you knew the true `p` and `b` exactly when you computed `f*`. Loosen any of those — a shorter horizon, a lower drawdown tolerance, a mis-estimated edge — and full Kelly stops being the bet you actually want. We'll honor the theorem here and then, in the playbook, explain precisely why practitioners deliberately bet below it.

## How much does it actually matter? The growth curve

It's one thing to compute `f* = 10%`; it's another to feel why missing it costs you. The growth rate at the Kelly fraction is `g(0.10) ≈ 0.50%` per bet. That sounds tiny. It is not. Over 100 bets it compounds to a multiplier of `e^(0.005 × 100) ≈ 1.65` — a 65% gain. Over 1,000 bets it's `e^(0.005 × 1000) ≈ 150×` — your bankroll grows 150-fold. The growth rate is small per bet and enormous in aggregate, which is exactly why getting it right matters: a small per-bet shortfall, compounded over thousands of bets, is the difference between 150× and a rounding error.

Now look back at Figure 1, the growth curve. The peak is at `f* = 10%`, where growth is about 0.50% per bet. To the *left* of the peak — under-betting — growth is positive but lower than it could be. At `f = 5%`, half of Kelly, growth is about 0.375% per bet, which is **75% of the maximum**: you gave up a quarter of your growth by betting half as much. To the *right* of the peak — over-betting — growth falls faster, crosses zero at `f = 20%` (exactly twice Kelly), and goes negative beyond that. At `f = 40%`, four times Kelly, the growth rate is about *−4.5%* per bet: a positive-edge bet, sized this badly, *loses* money on average over time.

Worth dwelling on the shape itself, because it explains the entire practitioner instinct toward caution. Near the peak the curve is *flat* — the derivative is zero there, by definition — so small sizing errors in either direction cost almost nothing in growth. That flatness is a gift: you don't need to hit Kelly exactly. But the flatness is *not symmetric* once you walk away from the top. Move left and the curve descends gently and stays positive all the way down to zero bet. Move right and it descends gently at first but then accelerates, plunging through zero growth at 2× Kelly and diving into deep negatives beyond. The geometry encodes a hard asymmetry between the two kinds of mistake, and we'll quantify it precisely in a moment — but the upshot is already visible: if you're going to miss, miss low.

That last fact deserves to stop you cold. The coin has a real edge. You are betting the right direction. And by betting too much, you have converted a winning bet into a losing strategy. The edge didn't disappear — your sizing ate it and then some. This is the single most important practical lesson in position sizing, and it is why the rest of this series treats over-betting as a survival threat, not a performance issue.

#### Worked example: the same edge, three bet sizes, over a year of trading

Suppose you place this 55%-edge coin-flip-equivalent bet 300 times in a year (a little more than once per trading day), starting from \$100,000, and you take the *typical* path — close to 55% wins. Compare three fractions:

- **f = 5% (half-Kelly):** growth ≈ 0.375%/bet. After 300 bets: `$100,000 × e^(0.00375 × 300) ≈ $100,000 × 3.08 ≈ $308,000`. You tripled the account.
- **f = 10% (full Kelly):** growth ≈ 0.50%/bet. After 300 bets: `$100,000 × e^(0.005 × 300) ≈ $100,000 × 4.48 ≈ $448,000`. The fastest survivable result.
- **f = 40% (four times Kelly):** growth ≈ −4.5%/bet. After 300 bets: `$100,000 × e^(−0.0448 × 300) ≈ $100,000 × 0.00000145 ≈ $0.15`. The account is gone — fifteen cents left — on a *winning* bet.

*Same edge, same 300 bets, same direction — and the only difference between quadrupling your money and losing all of it is the size of each bet.*

## Kelly as edge divided by odds

Let's sit with the edge-over-odds reading, because it's the form you'll actually use at a desk, and it generalizes far beyond coins. Figure 4 plots the Kelly fraction against the win probability `p` for three different payout structures — even money (`b = 1`), 2-to-1 (`b = 2`), and 5-to-1 (`b = 5`). Two things jump out.

![Line chart of the Kelly fraction against win probability for three payout odds, even money two to one and five to one, each line rising from the break even win rate and marked at the point where a fifty five percent win rate on even money gives a ten percent Kelly fraction](/imgs/blogs/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge-4.png)

First, **for a fixed payout, the Kelly fraction rises with the win probability** — more edge, bigger bet. Each line starts at zero exactly at the win rate where the bet becomes break-even (where `p = 1/(b+1)`, the point at which your edge is zero), then climbs. Below that win rate the formula returns a *negative* number, which is Kelly's way of telling you the bet has no edge and you shouldn't take it at all — or, if you could, you'd bet the *other* side.

Second, **the odds reshape the whole picture.** For longer odds (the 5-to-1 line) the curve climbs much faster, because each rare win moves your bankroll a lot. But this is also where the naive intuition fails: a high payout does *not* mean bet big. A lottery-style bet with `b = 100` but a true win probability of 1% has edge `0.01 × 100 − 0.99 = 0.01`, divided by odds 100, giving `f* = 0.0001` — bet *one hundredth of one percent*. Long odds with a thin edge demand tiny fractions, precisely because the variance is brutal and a string of losses between wins would gut a large bet. The edge-over-odds formula gets this right automatically; gut feel gets it spectacularly wrong, which is why so many traders blow up on "asymmetric" lottery trades that were correctly directional and catastrophically oversized.

#### Worked example: a long-shot trade with a thin edge

You have a trade that pays off 4-to-1 (`b = 4`) — risk \$1 to make \$4 — and you estimate it works 25% of the time (`p = 0.25`). Should you load up because the payoff is fat?

```
edge = p×b − (1−p) = 0.25 × 4 − 0.75 = 1.00 − 0.75 = 0.25
f*   = edge / odds = 0.25 / 4 = 0.0625 = 6.25%
```

The edge is genuinely large (25 cents per dollar), but the Kelly fraction is only 6.25%, *smaller* than the 10% you'd bet on the boring even-money coin whose edge was just 10 cents. On the \$10,000,000 book that's \$625,000 at risk, not the \$2,500,000 the fat payoff might tempt you toward. Why so much smaller? Because you lose three times out of four. A run of five or six losses between wins is routine, and a bet sized for the payoff rather than the odds would be cut to ribbons before the win arrives.

*A fat payoff is not permission to bet big; the win rate and the variance set the size, and Kelly bakes both in through the odds in the denominator.*

## The over-betting penalty is asymmetric

We've established that there's one optimal fraction and that both sides of it cost you. But the two sides do *not* cost you symmetrically, and the asymmetry is the single most important thing to internalize before you ever size a real position. Figure 6 makes it vivid: it plots growth as a percentage of the maximum, against bet size measured in *multiples* of Kelly. For a small even-money edge this curve is almost exactly the universal parabola `g/g* = 2k − k²`, where `k = f/f*` is how many Kelly units you're betting.

![Chart of growth as a percentage of the peak against bet size in multiples of Kelly, an asymmetric inverted parabola where half Kelly keeps about seventy five percent of growth, the peak is at one times Kelly, and two times Kelly gives zero growth before turning negative](/imgs/blogs/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge-6.png)

Read the curve from the peak outward. At full Kelly (`k = 1`) you have 100% of the available growth. Walk *left* toward under-betting and the curve descends gently: at `k = 0.5`, half-Kelly, you still have `2(0.5) − 0.5² = 1.00 − 0.25 = 0.75`, **75% of the maximum growth**. You cut your bet in half and only gave up a quarter of your growth. Walk *right* toward over-betting and the curve falls off a cliff: at `k = 2`, double-Kelly, you have `2(2) − 2² = 4 − 4 = 0` — **zero growth** — and past that it's negative, accelerating downward. Triple-Kelly is deeply negative; the bottom drops out.

This asymmetry has a clean intuition you can carry forever. **Under-betting only costs you upside; over-betting costs you the bankroll.** When you under-bet, the worst that happens is you grow more slowly — you're always still in the game. When you over-bet, you grow more slowly *and* the volatility of your bankroll explodes, and past 2× Kelly the volatility drag overwhelms the edge entirely. The downside of being too cautious is bounded and gentle; the downside of being too aggressive is unbounded and catastrophic. Given that you can never know your true edge precisely — more on that shortly — this asymmetry is the entire argument for erring *below* Kelly, never above. The cost of a 50% sizing error in the safe direction is a 25% growth haircut; the cost of a 100% sizing error in the dangerous direction is *everything you would have earned*.

#### Worked example: why double-Kelly gives exactly zero growth

Take the 55% coin and bet `f = 20%`, which is exactly twice the Kelly fraction of 10%. Compute the growth rate directly:

```
g(0.20) = 0.55 × ln(1 + 0.20) + 0.45 × ln(1 − 0.20)
        = 0.55 × ln(1.20)       + 0.45 × ln(0.80)
        = 0.55 × 0.18232        + 0.45 × (−0.22314)
        = 0.10028               − 0.10041
        ≈ −0.00013   (essentially zero)
```

The two terms — the gain from winning and the loss from losing — almost perfectly cancel. Over 1,000 such bets your bankroll multiplies by `e^(−0.00013 × 1000) ≈ 0.88`: after a thousand bets with a real edge, betting double-Kelly, your \$100,000 account is worth about \$88,000. You took on enormous volatility for *less than nothing*. Meanwhile full Kelly turned the same edge into a 150× gain. The chasm between those outcomes is pure sizing.

*Twice the Kelly bet wipes out exactly the growth that one times the Kelly bet earned — the over-betting penalty isn't a discount, it's a complete confiscation.*

## Watching it happen: four bet sizes on the same coin

Formulas and growth curves are one thing; watching real bankrolls diverge is another. Figure 3 simulates the same 55%-edge coin, played 300 times from \$100,000, at four fixed fractions — 5%, 10% (Kelly), 20% (twice Kelly), and 40% (four times Kelly) — where every fraction sees the *exact same sequence of coin flips*. The only thing that differs between the four curves is the bet size, so any difference in outcome is sizing and nothing else.

![Chart of four simulated bankroll paths on the same coin flips played three hundred times from one hundred thousand dollars, the ten percent Kelly path compounding highest, five percent slightly below, twenty percent roughly flat, and forty percent collapsing toward zero on a log scale](/imgs/blogs/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge-3.png)

The picture confirms every claim of the math. The **10% Kelly path** (green) compounds the highest — it's the steepest climber. The **5% half-Kelly path** (amber) tracks just below it, growing nicely but visibly slower: that's the 25% growth haircut you pay for halving your bet, on display. The **20% double-Kelly path** (lavender) is roughly *flat* — it wobbles around its starting point and goes essentially nowhere, exactly as the zero growth rate predicted. And the **40% path** (red) collapses, marching down the log axis from \$100,000 toward single dollars, a winning edge sized into a wealth-destruction machine. Same coin, same luck, four destinies, set entirely by the fraction.

Notice the log scale matters here. On a linear axis the 40% collapse would just hug the floor invisibly while the Kelly path shot off the top; the log axis lets you see that the 40% path is dropping by a roughly constant *percentage* each stretch — it's compounding *downward* at about −4.5% per bet, the negative growth rate made visible as a straight downward line.

## The continuous version: Kelly leverage for a strategy

Real strategies don't pay a clean `b`-to-1 on a coin. A trading strategy has a *distribution* of returns: some arithmetic average return per period (call it the drift `μ`) and some volatility (the standard deviation `σ`). We need a Kelly for that world, and it's just as clean. Model your strategy's return over a period as roughly Gaussian with mean `μ` and volatility `σ`, and apply leverage `L` (so `L = 1` is fully invested with no borrowing, `L = 2` is twice your capital deployed). The geometric growth rate of the levered strategy is approximately

$$
g(L) \approx \mu L - \tfrac{1}{2}\sigma^2 L^2.
$$

That second term, `−½σ²L²`, is the famous **volatility drag** — the penalty that compounding extracts from a volatile path, growing with the *square* of how much you lever up. The first term is the reward, growing only linearly. Maximize `g(L)` by setting the derivative `μ − σ²L = 0`, and you get the **continuous Kelly leverage**:

$$
\boxed{L^* = \frac{\mu}{\sigma^2}}
$$

and the maximum growth it delivers, found by substituting `L*` back in, is `g* = μ²/(2σ²)`. Notice that `μ/σ` is essentially the strategy's Sharpe ratio, so `L* = (μ/σ)/σ = \text{Sharpe}/σ` and the optimal growth `g* = ½ × \text{Sharpe}²`. **Kelly leverage scales with the edge and inversely with the *square* of volatility** — double your volatility at the same drift and you should cut your leverage to a quarter. That square is the whole reason volatility is so dangerous: it hits your optimal size twice, once in the denominator of the leverage and once in the drag.

Figure 7 computes `L*` for a handful of realistic-looking edges. The pattern is exactly what the formula says: lower-volatility edges support far more leverage at the same growth, and every edge with the same `μ/σ` (the same Sharpe) reaches the same optimal growth rate even though its optimal leverage differs.

![Horizontal bar chart of the growth optimal Kelly leverage for five trading edges, each labeled with its drift and volatility and Sharpe ratio, showing lower volatility edges supporting much higher optimal leverage while edges of equal Sharpe reach the same optimal growth](/imgs/blogs/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge-7.png)

#### Worked example: Kelly leverage on an index-like edge

Take an equity-index-like edge: an expected excess return of `μ = 8%` per year with a volatility of `σ = 16%` per year. The Kelly leverage is

```
L* = μ / σ²
   = 0.08 / (0.16)²
   = 0.08 / 0.0256
   = 3.125
```

So full Kelly says deploy about **3.1× your capital** in this strategy. On the \$10,000,000 book that's roughly \$31,000,000 of gross exposure — borrowing \$21,000,000 on top of your equity. The growth this delivers is `g* = μ²/(2σ²) = 0.08²/(2 × 0.0256) = 0.0064/0.0512 = 0.125`, a 12.5% geometric growth rate, versus only `μL − ½σ²L² = 0.08 − ½(0.0256) = 6.7%` if you ran the same edge fully invested but unlevered. Kelly nearly doubled the long-run growth — *and* it's exactly the leverage past which more borrowing starts subtracting. Push to `L = 6.25` (twice Kelly) and the growth formula gives `0.08 × 6.25 − ½ × 0.0256 × 6.25² = 0.50 − 0.50 = 0`, the same zero-growth-at-double-Kelly result we saw on the coin. The asymmetry is universal.

*The growth-optimal leverage for an 8%/16% edge is about 3.1 times capital — and that same factor of "double it and growth vanishes" reappears, because the continuous Kelly is the same idea as the discrete one wearing different clothes.*

But hold that 3.1× lightly. A 3-to-1 levered book that *feels* like a Kelly-optimal genius play is also a book where a single bad period can take a deep, ego-bruising bite — and where your estimate of `μ` is far shakier than your estimate of `σ`. That gap between the clean formula and the messy real edge is the whole subject of the next section, and it's why nobody sane runs the full 3.1×.

### Kelly, Sharpe, and the brutal arithmetic of volatility drag

The continuous formula carries a lesson that the coin version hides, and it's worth drawing out because it governs every leveraged strategy you'll ever run. Rewrite the optimal growth as `g* = ½ × (μ/σ)² = ½ × Sharpe²`. Your best-possible long-run growth rate is *half the square of your Sharpe ratio*, full stop. A strategy with a Sharpe of 0.5 caps out at `½ × 0.25 = 12.5%` geometric growth no matter how you lever it; a Sharpe of 1.0 caps at `½ × 1 = 50%`; a Sharpe of 2.0 at a colossal 200%. Growth scales with the *square* of the risk-adjusted edge, which is why a small improvement in Sharpe is worth so much more than a small improvement in raw return — and why chasing Sharpe (the *quality* of the edge) beats chasing leverage (the *quantity* of the bet) every time. You cannot lever a mediocre Sharpe into a great growth rate; the volatility drag eats you before you get there.

That volatility drag, the `−½σ²L²` term, is the silent tax on every levered position, and its square dependence is the killer. Suppose you find an edge with `μ = 10%` and `σ = 20%`. The unlevered geometric growth is `μ − ½σ² = 0.10 − ½(0.04) = 8%`, already below the 10% arithmetic mean because volatility alone drags it down by two points. Lever it to `L = 2` and the reward term doubles to 20% but the drag *quadruples* to `½ × 0.04 × 4 = 8%`, netting 12%. Lever to `L = 4` and the reward is 40% but the drag is `½ × 0.04 × 16 = 32%`, netting just 8% — you're back where you started, having taken four times the risk for nothing. The Kelly leverage here is `L* = 0.10/0.04 = 2.5`, the exact crest before the quadratic drag overtakes the linear reward. This is the *same* inverted-parabola shape as the coin's growth curve, because it is the same idea: a linear gain from sizing up fighting a quadratic loss from variance, with one optimum in between.

#### Worked example: a per-trade edge sized by continuous Kelly

Suppose a systematic strategy on the \$10,000,000 book makes, per trade, an average return of `μ = 0.6%` with a volatility of `σ = 8%` (these are per-trade figures, not annual). The continuous Kelly says deploy

```
L* = μ / σ²
   = 0.006 / (0.08)²
   = 0.006 / 0.0064
   ≈ 0.94
```

So you should run this edge at about **0.94× capital — slightly *under* fully invested**, deploying roughly \$9,400,000 of the \$10,000,000 rather than borrowing to lever up. The Sharpe per trade is `μ/σ = 0.006/0.08 = 0.075`, a thin edge, so Kelly correctly refuses to lever it: `g* = ½ × 0.075² ≈ 0.28%` of growth per trade is all that's there to capture, and reaching for it with leverage would surrender it to the drag. Contrast this with the index edge whose Kelly was 3.1×: same formula, but the higher Sharpe and the way the numbers fall produce a completely different size.

*Kelly leverage isn't a license to borrow — for a thin per-trade edge it tells you to hold back below fully invested, because the volatility drag would devour any leverage you added.*

## Common misconceptions

**"Kelly is aggressive — it's for gamblers, real investors size conservatively."** Backwards. Kelly is the *most aggressive bet you can make without sacrificing long-run growth* — it sits exactly at the peak, the boundary past which more aggression *reduces* growth. Everything to the right of Kelly is the gambler's zone where you take more risk for *less* return. Conservative sizing means betting a fraction *of* Kelly (half-Kelly, quarter-Kelly), which is still defined entirely by the Kelly number. You can't size conservatively against Kelly without first knowing where Kelly is. The 55% coin's Kelly is 10%; a "conservative" 5% is half-Kelly, and a reckless 25% is 2.5× Kelly with negative growth.

**"A bigger edge means I should bet a bigger fraction."** Only if the odds stay fixed. Kelly is edge *divided by odds*, so a fat payoff with a thin win rate can demand a *tiny* fraction. The 4-to-1 trade with a 25% win rate had a large 25-cent edge but a Kelly of only 6.25% — smaller than the boring coin's 10%. Sizing on the edge alone, ignoring the odds in the denominator, is exactly how lottery-style trades blow up accounts.

**"If betting Kelly is good, betting a bit more must be a bit better."** No — this is the asymmetry. The growth curve is *flat* at the very top (the derivative is zero at the peak, by construction), so a small step left or right barely changes growth near Kelly. But the curvature is dangerous on the right: by the time you've doubled to 2× Kelly, growth has fallen all the way to zero, and beyond that it plunges. A "bit more" near the peak is nearly free; a "bit more" repeated until you're well past it is ruinous. Half-Kelly keeps 75% of growth; double-Kelly keeps 0%.

**"Kelly maximizes my expected wealth."** It does *not*, and this is the deepest confusion. Expected wealth is maximized by betting 100% (and going broke on one path). Kelly maximizes expected *log*-wealth — the growth rate of your single path through time. The two objectives genuinely disagree, and Kelly deliberately leaves expected wealth on the table in exchange for not dying. On the coin, full-Kelly's expected one-round wealth (+1%) is far below all-in's (+10%) — but full-Kelly compounds 150× over 1,000 bets while all-in is at zero by bet three.

**"Kelly tells me whether to take the bet."** No — Kelly assumes you already have an edge and only sizes it. If the formula returns zero or negative, that's not Kelly evaluating the bet; it's Kelly reporting that there's no edge to size (`p ≤ 1/(b+1)`). Edge-finding is a separate discipline upstream of everything here. Kelly is purely the *how much*, never the *whether*.

**"The continuous Kelly leverage `L* = μ/σ²` is some different formula I have to memorize separately."** It's the same idea. The discrete `f* = edge/odds` and the continuous `L* = μ/σ²` are two faces of "bet in proportion to your edge and inversely to your risk," and both share the identical signature behaviors: a single optimum, gentle penalty for under-betting, and *exactly zero growth at twice the optimum*.

## How it shows up in real markets

Kelly is a theorem about coins, but over-betting is a body count, and the case files all read the same way: a real edge, sized far past Kelly, converted into ruin by a single adverse run.

**Long-Term Capital Management, 1998.** LTCM ran convergence trades with a genuine, well-researched edge — small mispricings that historically closed. The problem was size. By early 1998 the fund carried roughly **\$125 billion of assets on about \$4.7 billion of equity — balance-sheet leverage near 25-to-1** — with gross derivative notional around **\$1.25 trillion**. That leverage was wildly past any defensible Kelly fraction for trades whose volatility spiked and whose correlations went to one in the August–September flight to quality. The edge was arguably real; the sizing was multiples of Kelly, and **about \$4.6 billion of capital evaporated in roughly four months**, forcing a Fed-organized \$3.6 billion rescue. A smaller, sub-Kelly book survives the same shock with a drawdown instead of a death. The strategic dimension — why so many smart funds crowded the *same* trade — is dissected in [the LTCM case study](/blog/trading/game-theory/case-study-ltcm-1998-the-crowded-genius-trade).

**Archegos Capital Management, 2021.** Archegos held concentrated single-stock bets financed through total-return swaps, levering perhaps **5× or more** — and crucially, each prime broker could see only its own slice, so no one priced the *total* size. When a few positions fell, the leverage forced selling that fed on itself, and the fund vaporized while handing its brokers **over \$10 billion in losses, including about \$5.5 billion at Credit Suisse alone**. Whatever edge the underlying stock views had, the position was sized for a world that doesn't have bad weeks. Kelly on a concentrated, illiquid single-name book is a *small* fraction; Archegos ran a large multiple of it.

**Volmageddon, February 5, 2018.** Short-volatility carry was a real edge for years — selling insurance that mostly expires worthless pays steadily. But the strategy was sized as if the steady payments were the whole distribution, ignoring the fat left tail. When the **VIX roughly doubled from about 17 to 37 in a single day**, the inverse-VIX product XIV lost about **96% of its value after the close and was terminated**. The edge (a positive variance risk premium) was genuine; the sizing treated a left-tailed bet like an even-money coin, which is over-betting in disguise. The mechanics of why selling vol "pays until it doesn't" are in [the variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) and [the Volmageddon case study](/blog/trading/options-volatility/case-study-volmageddon-2018-and-the-short-vol-blowup).

**The yen-carry unwind, August 5, 2024.** A crowded funding-carry trade — borrow cheap yen, buy higher-yielding assets — carried a real positive carry for years. When it unwound, the **Nikkei fell about 12.4% in a single day, its worst since 1987**, and the **VIX spiked intraday toward 65**. Same story: a genuine carry edge, sized and levered as if the funding leg would never reprice violently, deleveraging reflexively in days. The carry was the edge; the leverage was the over-bet.

The thread is unmistakable. In none of these cases was the edge obviously fake at the outset — that's what makes them instructive. The fatal error was *sizing*, betting a multiple of any sane Kelly fraction and meeting the one bad run that turns over-betting from a slow drag into an instant absorbing barrier. Kelly is the discipline that would have capped each of them below the cliff. This is the [risk-of-ruin lesson](/blog/trading/risk-management/risk-of-ruin-why-positive-expectancy-is-not-enough) wearing a position-sizing hat.

## The risk playbook: applying Kelly when you don't actually know your edge

Here's the uncomfortable truth that every honest practitioner runs into: **the Kelly formula needs your true edge as an input, and you never know your true edge.** You estimate `p`, `b`, `μ`, `σ` from limited, noisy history, and your estimates are wrong — usually optimistically wrong, because the strategies you're looking at are the ones that *happened* to do well in-sample. Plug an inflated edge into Kelly and you get an inflated, over-sized bet. And we just spent a whole post establishing that over-betting is the catastrophic direction. So the practical version of Kelly is not "compute `f*` and bet it." It's this:

- **Bet a *fraction* of Kelly, never full Kelly.** Half-Kelly is the standard practitioner default, and the asymmetry is exactly why: at half-Kelly you keep about **75% of the growth** while roughly *halving* the volatility of your bankroll and dramatically shrinking your drawdowns. You give up a quarter of your speed to buy a much smoother, much safer ride — and, critically, to buy a margin of safety against having over-estimated your edge. Many run quarter-Kelly. Almost nobody sane runs full Kelly with real money. The full reasoning — why fractional Kelly is the rational response to estimation error, and how it connects to optimal-`f` — is the subject of the next post, [fractional Kelly and optimal-f](/blog/trading/risk-management/fractional-kelly-and-optimal-f-betting-less-to-sleep-at-night).
- **Estimate conservatively, then discount again.** Use a deliberately pessimistic edge (shade `p` down, shade `σ` up), *then* take a fraction of the resulting Kelly. You are stacking two margins of safety because the cost of over-estimating is so much steeper than the cost of under-estimating. When in doubt, bet less — the growth curve forgives it.
- **Size against the bet you actually have, not the average bet.** Kelly is derived for one repeated bet. Real books hold many positions at once, with correlations that tighten in a crisis. The relevant `σ` is the *portfolio's* volatility under stress, not each position's in calm times — and [in a crisis correlations go to one](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis), which inflates the effective `σ` and *shrinks* your true Kelly exactly when you can least afford to find out. Size for the correlated, stressed book.
- **Re-bet on current bankroll, and cap the drawdown anyway.** Kelly's growth-optimality assumes you continuously re-bet the same fraction of your *current* (shrinking, in a drawdown) bankroll — which automatically de-risks you as you lose, a built-in survival feature. But pair it with a hard drawdown limit regardless, because Kelly optimizes growth, not your tolerance for the deep, prolonged drawdowns that even half-Kelly produces.
- **Never let sizing turn a winning edge into a losing strategy.** The one-line summary of the whole post: a real edge sized past 2× Kelly has *negative* growth. If a sizing choice could put you there — through leverage, concentration, or a fat-tailed payoff you've mismodeled as a coin — it doesn't matter how good the edge is. Cut the size first, admire the edge second.

The deepest takeaway is the one we opened with: **finding an edge and sizing it are two separate decisions, and the second is where fortunes are actually kept or lost.** Kelly is the bridge between a positive expectancy and a surviving, compounding bankroll — the constructive answer to the ruin and ergodicity warnings of Track A. It tells you that there is a *right* amount to bet, that it's computable from edge and odds, that both timidity and aggression cost you, and that the cost of aggression is the kind you don't come back from. Once you've internalized that the growth-optimal bet is a specific, finite, knowable fraction — and that you should bet *less* than it because you can't trust your own edge estimate — you've made the single most important upgrade a sizer can make. The next post takes the obvious follow-up seriously: if full Kelly is already the most you should ever bet, why does almost everyone bet a *fraction* of it, and how much less should you really bet to sleep at night?

### Further reading

- [Risk of ruin: why positive expectancy is not enough](/blog/trading/risk-management/risk-of-ruin-why-positive-expectancy-is-not-enough) — the survival result Kelly is the sizing answer to: a positive edge does not save you from an absorbing barrier.
- [Fractional Kelly and optimal-f: betting less to sleep at night](/blog/trading/risk-management/fractional-kelly-and-optimal-f-betting-less-to-sleep-at-night) — the very next post: why nobody bets full Kelly, and how much to shade down for estimation error and drawdown tolerance.
- [Leverage and the arithmetic of ruin](/blog/trading/risk-management/leverage-and-the-arithmetic-of-ruin) — how the continuous Kelly leverage `L* = μ/σ²` behaves when borrowing, margin, and volatility drag enter the picture.
- [Kelly criterion and sequential betting (quant interviews)](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) — the heavier probability derivations, sequential-betting variants, and the interview-grade treatment.
- [Position sizing and risk of ruin in options trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) — applying these sizing ideas to fat-tailed, asymmetric option payoffs where the coin model breaks.
