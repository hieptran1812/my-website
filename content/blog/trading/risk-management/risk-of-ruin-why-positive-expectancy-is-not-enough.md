---
title: "Risk of Ruin: Why Positive Expectancy Is Not Enough"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A winning edge can still take your account to zero. Here is how the gambler's-ruin math works, why bet size decides your fate as much as your edge, and how to size so ruin stays vanishingly small."
tags: ["risk-management", "risk-of-ruin", "gamblers-ruin", "position-sizing", "expectancy", "kelly-criterion", "bankroll-management", "survival"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **One sentence:** A positive edge tells you a bet is worth taking, but it says nothing about how *much* to bet — and the wrong size turns a winning strategy into a near-certain blow-up.
> - **Expectancy is not survival.** A strategy can have a genuinely positive average outcome per bet and still drive your account to zero with high probability if you bet too large.
> - **Zero is an absorbing barrier.** The instant your stake hits zero you are out of the game forever; no edge, however real, can compound on a balance of nothing.
> - **Ruin depends on three things:** your edge (how favourable each bet is), your bankroll (how many losses you can absorb), and your bet fraction (how much you stake each time). Bet fraction is the one you control most directly and abuse most often.
> - **Two flavours of ruin.** *Fixed-amount* ruin (you bet a constant dollar amount and can literally reach zero) and *fixed-fractional* ruin (you bet a percentage, never technically hit zero, but get dragged so close that you are practically finished).
> - **The fix is sizing, not edge-hunting.** Size every position so that your worst plausible losing streak is survivable, keep the bet fraction well below the growth-optimal point, and ruin probability collapses toward zero while the edge still pays.

A profitable strategy that blows up is not a profitable strategy. It is a losing strategy with good marketing.

This is the single hardest idea for a new trader to accept, because it contradicts the thing everyone is taught to chase: *find an edge*. Find a setup that wins more than it loses, or wins bigger than it loses, and the money takes care of itself. That belief is half true and dangerously incomplete. You can build a strategy with a real, measurable, positive edge — one that, averaged over thousands of trades, makes money on every single bet — and still take your account to zero with frightening regularity. Not because the edge failed. Because you bet too big.

The discipline that separates traders who survive from traders who flame out spectacularly is the gap between two questions that beginners treat as the same: *"Is this bet worth taking?"* and *"How much should I stake?"* The first is about expectancy. The second is about ruin. They are different questions with different math, and confusing them is the most common way a winning trader destroys a winning account.

The casino is the cleanest illustration of the asymmetry. A casino's edge on a roulette wheel is small — a couple of percent. Yet casinos never go broke at the tables, and the reason is not that they win every spin. They lose individual spins constantly. They survive because their bankroll is gigantic relative to any single bet, and they cap how much one player can wager, so no run of bad luck can reach their absorbing barrier before the law of large numbers grinds their small edge into certain profit. The casino has internalized the lesson this post is about: a small edge plus disciplined sizing plus a deep bankroll is unbeatable, while a *large* edge plus reckless sizing is a coin flip on your survival. The gambler across the table has the worse side of the same bet not because the game is rigged spin by spin, but because the gambler is *under-capitalized relative to their bet size* — and that, not the house edge, is what ruins them. Most blown-up traders are the gambler, not the house, and they got there by sizing like the gambler while believing they had the house's edge.

![Many simulated equity paths from a positive-edge strategy sized too large, with survivors compounding upward in green and a meaningful fraction collapsing to a practical-ruin floor in red](/imgs/blogs/risk-of-ruin-why-positive-expectancy-is-not-enough-1.png)

Look at the chart above before reading another word. Every one of those 220 wiggly lines is the same strategy — a coin that lands in your favour 55% of the time, paying you one dollar for every dollar you risk. That is a strong, real edge. The strategy is positive-expectancy on every single bet. The only thing that differs across the paths is luck: which way each individual coin landed. Most of the lines drift up and to the right, exactly as the edge promises — the survivors compound their money handsomely. But notice the red lines. Roughly one path in five gets dragged down to the floor and stays there. Same edge. Same strategy. Same positive expectancy. *Killed anyway* — purely because the bet was sized at 20% of the account each time, large enough that an ordinary run of bad luck was fatal.

This post is about why that happens, the math behind it, and how to size so it does not happen to you. We will build the idea from absolute zero — the gambler's-ruin problem from probability — and connect it all the way to a concrete sizing playbook you can apply to a real account. This is the third leg of the series' survival thesis: the [recovery asymmetry](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) tells you why deep losses are catastrophically hard to recover, and the [ergodicity argument](/blog/trading/risk-management/ergodicity-time-average-vs-ensemble-average-and-the-coin-flip-that-ruins-you) tells you why your *personal* outcome differs from the average outcome. Risk of ruin is where those two truths meet a single number you control: how much you bet.

## Foundations: expectancy, bankrolls, and the barrier at zero

Before we can talk about ruin we need four ideas defined from scratch. Skip nothing here — the whole argument rests on these.

### What "positive expectancy" actually means

**Expectancy** (also called *expected value*, or EV) is the average amount you make per bet if you could repeat the same bet infinitely many times. You compute it by weighting each possible outcome by its probability and adding them up.

Concretely: suppose you have a bet that wins 55% of the time and loses 45% of the time, and when you win you gain exactly as much as you risk (a "1:1" or "even-money" payoff). For every dollar you stake:

- 55% of the time you gain \$1.
- 45% of the time you lose \$1.

The expectancy per dollar staked is:

(0.55 × +\$1) + (0.45 × −\$1) = +\$0.55 − \$0.45 = **+\$0.10**

So on average, every dollar you put at risk earns you ten cents. That is a *huge* edge — a real-world casino would weep with envy, and a real-world trading strategy with that edge would be worth a fortune. The key word is **positive**: the expectancy is greater than zero, so over enough repetitions, you make money. This is what people mean when they say a strategy "has an edge."

Here is the trap, stated as plainly as possible: **expectancy is computed over infinite repetitions at a fixed bet size, assuming you can always afford the next bet.** That last clause is doing enormous, invisible work. Expectancy quietly assumes you never run out of money. The moment that assumption breaks — the moment a losing streak takes your balance to zero — the infinite future of profitable bets that the expectancy calculation depends on simply never happens.

### What a bankroll is and why it is finite

Your **bankroll** is the total pool of money you are willing to risk on a strategy. In trading, it is your trading capital; at a poker table, it is the chips in front of you; in a coin-flip game, it is the cash in your pocket. The defining feature of a real bankroll is that it is **finite** — there is a bottom, and the bottom is zero.

This is the gap between the math of expectancy and the reality of a trading account. Expectancy lives in a world of infinite repetitions. Your bankroll lives in a world where a long-enough losing streak ends the game. The size of your bankroll, relative to the size of your bets, decides how long a losing streak you can survive before you hit that bottom.

### The absorbing barrier — the most important concept in this post

Lay out a number line for your account balance, running from zero on the left up to whatever your account is worth. Each bet nudges your position right (a win) or left (a loss). Now here is the crucial property of the point at zero: **it is absorbing.** Once your balance touches zero, you are stuck there permanently. You have no stake left to make the next bet, so the game ends. You can never bounce back, because there is nothing left to bet.

![The absorbing barrier at zero showing a bankroll that bounces around and either recovers from a reflecting boundary or terminates permanently at the absorbing barrier](/imgs/blogs/risk-of-ruin-why-positive-expectancy-is-not-enough-4.png)

The figure above contrasts an **absorbing** barrier with a **reflecting** one. A reflecting barrier is what you would have if, every time you dropped near zero, someone handed you fresh capital and you bounced back up. With a reflecting boundary, an edge eventually wins — you keep getting more chances and the favourable odds reassert themselves. But markets do not give you a reflecting barrier. Markets give you an absorbing one. When a leveraged account hits a margin call and gets liquidated, when a fund's investors redeem after a deep drawdown, when your capital is simply gone — the path terminates. The edge becomes irrelevant the instant you touch zero, because the edge can only pay you if you are still holding a stake.

This is why "the strategy is +EV" is not a defense. Expectancy assumes the path goes on forever. The absorbing barrier guarantees that, for some paths, it does not.

### Risk of ruin, defined

**Risk of ruin** is the probability that your bankroll reaches the absorbing barrier (zero, or some practical "I'm finished" level) before you reach some goal — or, in the strict version, ever. It is a probability between 0 and 1. A risk of ruin of 0.88 means: run this strategy, with this bet size, from this bankroll, and 88% of the time you go broke. A risk of ruin of 0.0001 means you go broke once in ten thousand lifetimes.

The entire game of risk management is keeping that number small *while still collecting your edge*. And the punchline of this whole post is that risk of ruin depends only loosely on your edge and very heavily on your **bet size** — the one variable you fully control.

### Two kinds of average — and why your account follows the wrong one

There is one more foundation to lay, and it is the conceptual hinge of the entire post: the difference between the **arithmetic** average and the **geometric** average of your returns. Get this and everything else falls into place.

When you compute expectancy — the +\$0.10 per dollar from earlier — you are computing an *arithmetic* average. You add up the outcomes, weighted by probability, and divide. Arithmetic averages describe what happens when outcomes *add together*: bet a fixed \$1,000 each time and the dollars stack additively, win or lose. The arithmetic average is the right tool for fixed-amount betting.

But your trading account does not add returns — it *multiplies* them. A +10% month followed by a −10% month does not leave you flat. It leaves you at `1.10 × 0.90 = 0.99`, down 1%. Returns that multiply are governed by the **geometric** average, and the geometric average is *always less than or equal to* the arithmetic average. The gap between them is exactly the volatility drag we will meet later, and it widens with the size of your swings.

This is the deep reason expectancy lies to you. Expectancy reports the *arithmetic* average — the average of the individual bets. But your *wealth* compounds along the *geometric* average — the average of the multiplied path. A strategy can have a positive arithmetic average (every bet is a winner on average) while having a *negative* geometric average (your compounded wealth shrinks). When that happens, you are watching a +EV strategy march steadily toward ruin, and the arithmetic that told you "this is profitable" was answering the wrong question. Expectancy answers "is the average bet good?" Your survival answers "does the multiplied path grow?" — and only the second one pays your bills.

## The gambler's ruin: the cleanest version of the problem

To see the mechanics with perfect clarity, strip away the messiness of markets and reduce the problem to its purest form. This is a classic of probability theory, two centuries old, called the **gambler's ruin**.

Here is the setup. You start with `i` dollars (or chips, or units). You play a game where each round you bet exactly **one unit**. You win the round — gaining one unit — with probability `p`, and lose the round — losing one unit — with probability `q = 1 − p`. You keep playing until one of two things happens: you go broke (reach zero), or you reach some target wealth `N` and walk away. The question is: **what is the probability you go broke before you reach the target?**

This is *fixed-amount* betting — every bet is the same size in dollars (one unit) regardless of how rich or poor you currently are. It is the version of ruin where you can literally reach zero, because a constant dollar bet, repeated, can subtract your whole bankroll away.

The math has an exact, closed-form answer. With `r = q / p`, the probability of ruin starting from `i` units with target `N` is:

P(ruin) = (r^i − r^N) / (1 − r^N)

You do not need to memorize it. You need to feel what it does. The series' shared `data_risk.py` module computes it exactly via `dr.ruin_prob_unit_bet(p, i, N)`, so every number below is reproducible and identical across the series. Let us pump some numbers through it.

![Gambler's ruin probability versus starting bankroll for a slight edge and a slight disadvantage, showing how a one-percent per-bet edge swings ruin from near-certain to near-zero](/imgs/blogs/risk-of-ruin-why-positive-expectancy-is-not-enough-3.png)

The chart traces P(ruin) against your starting bankroll, for a target of `N = 100` units, under three cases: a fair coin (`p = 0.50`), a slight disadvantage (`p = 0.49`), and a slight edge (`p = 0.51`). Three features jump out, and each one is a lesson.

First, the fair-coin line is a straight diagonal: with no edge, your probability of ruin is exactly `1 − i/N`. Start halfway (`i = 50`, `N = 100`) and you have a 50% chance of going broke before doubling. Pure symmetry — no surprise.

Second — and this is the headline — look at the red line, the *slight disadvantage*. A one-percent disadvantage per bet (`p = 0.49`) is almost nothing on any single round. But starting from `i = 50` with `N = 100`, the probability of ruin is not 51% or 55%. It is **88%**. A trivial per-bet disadvantage, compounded over a long sequence, becomes a near-certainty of ruin. The slight edge case (`p = 0.51`, green line) is the mirror image — ruin drops to about 12%. A two-percentage-point swing in the per-bet odds flips your fate from almost-certain ruin to probably-fine.

Third, notice the *shape*. These are not straight lines once there is an edge; they are steep S-curves. A small edge, compounded over many bets, gets *amplified* — the more bets you play, the more decisively the edge (or the disadvantage) asserts itself. This is the double-edged blade of compounding: a real edge becomes near-certain profit, but a real disadvantage becomes near-certain ruin.

#### Worked example: a slight disadvantage that ruins you

You sit down with a **\$50,000** stake, betting **\$1,000** a hand (so `i = 50` units), aiming to reach **\$100,000** (`N = 100`). The game is *almost* fair — you win 49% of hands and lose 51%. How likely are you to bust before you double?

Plug into the formula with `p = 0.49`, `q = 0.51`, so `r = 0.51 / 0.49 ≈ 1.0408`:

- `r^50 ≈ 7.21`
- `r^100 ≈ 51.98`
- P(ruin) = (7.21 − 51.98) / (1 − 51.98) = (−44.77) / (−50.98) ≈ **0.88**

So **88% of the time you go broke before you reach your goal**, despite the game being only barely tilted against you. And if you came in undercapitalized — say `i = 10` units (a \$10,000 stake at the same \$1,000 bet) — the probability of ruin climbs to **99.1%**. Thinner cushion, near-certain death.

*A microscopic per-bet disadvantage, repeated, is not a small problem — compounding turns it into an almost-guaranteed wipeout.*

#### Worked example: even a real edge can ruin a thin bankroll

Now flip it. Suppose you genuinely have an edge: you win **55%** of unit bets (`p = 0.55`), the same 1:1 game. That is a strong, real, money-making edge. Surely you are safe?

It depends entirely on your cushion. With a healthy `i = 50` of `N = 100`, `dr.ruin_prob_unit_bet(0.55, 50, 100)` returns essentially **0.00004** — you are practically immortal. But come in thin, with only `i = 5` units before your \$100-unit target, and the same edge gives:

P(ruin) = `dr.ruin_prob_unit_bet(0.55, 5, 100)` ≈ **0.37**

A 37% chance of ruin — *with a 55% win rate.* The edge is real; the bankroll is too thin to ride out an early bad streak before the edge has time to assert itself. Bump the cushion to `i = 20` units and ruin falls to about **1.8%**. Same edge, very different survival, purely from how many losses the bankroll can absorb.

*Edge protects you only if you have enough bankroll to survive the variance until the edge shows up; a thin stack can be killed by ordinary bad luck before the edge ever pays.*

### The simpler formula you can carry in your head

The full gambler's-ruin formula above handles a finite target `N`. But there is a stripped-down version for the case traders care about most — playing indefinitely, with no fixed cash-out target — and it is simple enough to keep in your head. If you have an edge and you measure your bankroll in **units of your bet size**, the probability of *ever* being ruined, starting from `i` units, is approximately:

R ≈ (q / p)^i

That is it: the ratio of your loss-probability to your win-probability, raised to the power of how many bets your bankroll can absorb. Two forces fight inside that one expression. The base `q/p` is your **edge**: with `p = 0.55`, `q/p = 0.45/0.55 ≈ 0.818`, a number less than one. The exponent `i` is your **cushion**: how many unit-bets deep your bankroll is. Because the base is below one, raising it to a higher power drives the whole thing toward zero — *every extra unit of cushion multiplies your ruin probability by the edge ratio again.* This is why bankroll depth is so powerful: ruin falls *geometrically*, not linearly, as you thicken your cushion.

Run the numbers and the lesson is stark. With the 55% edge:

- 1 unit of cushion → R ≈ 0.818^1 ≈ **82%** ruin. One bet deep, you are almost certain to bust.
- 5 units → R ≈ 0.818^5 ≈ **37%** ruin. Still dangerous.
- 10 units → R ≈ 0.818^10 ≈ **13%** ruin.
- 20 units → R ≈ 0.818^20 ≈ **1.8%** ruin. Now you are reasonably safe.
- 40 units → R ≈ 0.818^40 ≈ **0.03%** ruin. Effectively immortal.

Notice that these match the finite-`N` numbers from the worked examples (a 5-unit cushion gave 37% ruin both ways) — because for a real edge and a distant target, the two formulas converge. The single most actionable consequence: **how deep your bankroll is, measured in bets, is the master dial of survival.** Want to cut your ruin probability by a factor of forty? Hold twenty units of cushion instead of five. The edge sets the base; you set the exponent — and the exponent is just another name for *bet small relative to your bankroll.*

The gambler's ruin gives us the first half of the answer: **ruin depends on edge and on bankroll-relative-to-bet-size.** Now we turn to the version that matters most for traders, where bet size is not fixed in dollars but as a fraction of a moving account — and where the math gets stranger and, in some ways, scarier.

## Fixed-fractional betting: you never hit zero, and that is the problem

Real traders almost never bet a constant dollar amount. They bet a *fraction* of their current account. "Risk 1% of capital per trade." "Put 5% of the book into this position." This is **fixed-fractional** betting, and it changes the ruin problem in a profound way.

Here is the mechanical difference. If you bet a fixed fraction `f` of your current bankroll and lose, your bankroll shrinks — so your *next* bet, being `f` of a smaller number, is smaller in dollars too. Lose again, and the bet shrinks again. Mathematically, you can never reach exactly zero, because you are always betting a fraction of whatever is left, and a fraction of a positive number is still positive. The account is like a candle that keeps halving — it never quite goes out.

This sounds like a feature. It is actually a trap with a different shape. You do not hit *literal* zero, but you can be dragged so close to it that you are **practically ruined** — down 90%, 95%, 99% — a level from which, given the [brutal recovery asymmetry](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain), you will never realistically come back. A 95% drawdown needs a 1,900% gain to recover. Technically alive; functionally dead. This is the distinction the chart at the top of this post was built to show — those "ruined" paths never hit exactly \$0, they were dragged to a floor 98% below where they started and could not climb out.

### The hidden tax: volatility drag

Fixed-fractional betting introduces a force that fixed-amount betting does not have, and it is the secret villain of this whole topic: **volatility drag** (also called *variance drain* or the *geometric-vs-arithmetic gap*).

The mechanism is brutal arithmetic. When you bet a fraction of your account, gains and losses *multiply* rather than add. And multiplication is asymmetric in a way that always works against you. Watch what one win-then-loss pair does to a fraction `f`:

- Win: multiply your bankroll by `(1 + f)`.
- Loss: multiply your bankroll by `(1 − f)`.
- Net of one win and one loss: `(1 + f) × (1 − f) = 1 − f²`.

That `1 − f²` is *less than one* for any non-zero `f`. A win followed by an equal-percentage loss does **not** leave you where you started — it leaves you slightly poorer. The larger the fraction `f`, the bigger the leak, and it grows with the *square* of the bet size. Bet 10% and a win-loss pair costs you 1% (`f² = 0.01`). Bet 50% and a win-loss pair costs you 25% (`f² = 0.25`). Double the bet size, *quadruple* the drag.

This is why a positive arithmetic edge can produce a negative *geometric* growth rate. Your average bet makes money (arithmetic expectancy is positive), but your *compounded* wealth shrinks, because volatility drag is eating you faster than the edge is feeding you. Over-bet enough, and you are guaranteed to grind toward ruin even though every individual bet is, on average, a winner.

#### Worked example: the same edge, sized two ways, on a \$100,000 account

Take our 55% / 1:1 edge again — arithmetic expectancy +10 cents per dollar risked — and run it on a **\$100,000** account two ways, over the *same* sequence of coin flips.

**Disciplined: bet 5% of the account each time.** A win multiplies by 1.05, a loss by 0.95. A win-loss pair leaves you at `1.05 × 0.95 = 0.9975` — a 0.25% leak per pair, tiny. With a 55% win rate the edge swamps that small drag and your account compounds upward. In the seeded simulation, this disciplined account grows from \$100,000 to roughly **\$206,000** over 300 bets.

**Reckless: bet 40% of the account each time.** Now a win multiplies by 1.40 and a loss by 0.60. A win-loss pair leaves you at `1.40 × 0.60 = 0.84` — a **16% leak per pair.** That volatility drag is enormous, far bigger than the 10-cent edge can repair. The same account, on the *identical* sequence of wins and losses, gets ground down to the practical-ruin floor — about **\$1,000**, a 99% loss.

![A single positive-edge strategy run at two bet sizes on the same outcome sequence, with the disciplined five-percent path compounding upward and the over-sized forty-percent path collapsing to the floor](/imgs/blogs/risk-of-ruin-why-positive-expectancy-is-not-enough-5.png)

Read that chart slowly, because it is the whole post in one picture. The two lines share *the same edge* (55% wins, 1:1), *the same luck* (an identical, seeded sequence of which bets won and lost), and *the same starting capital* (\$100,000). They differ in exactly one input: bet size. The green line, sized at 5%, ends near \$206,000. The red line, sized at 40%, is dragged to the \$1,000 floor. **Same edge, same luck, opposite destiny — purely because of size.**

*A positive edge is necessary but nowhere near sufficient; bet size alone can convert a money-making strategy into a guaranteed blow-up.*

### Why fixed-amount and fixed-fractional ruin are different beasts

It is worth pinning down precisely how the two betting styles differ, because they fail in different ways and the difference matters for how you think about your account.

**Fixed-amount betting** (the gambler's-ruin model) stakes a constant dollar amount regardless of your balance. Its danger is *literal* zero: a long enough losing streak subtracts your whole bankroll, one fixed chunk at a time, and you hit the absorbing barrier exactly. The protection is your cushion in *units* — how many fixed bets your bankroll holds. The good news: if you survive the early variance, a fixed bet becomes a *smaller and smaller fraction* of a growing account, so the edge gets safer over time. The bad news: in a drawdown, a fixed bet becomes a *larger* fraction of a shrinking account, accelerating you toward zero exactly when you can least afford it. Fixed-amount betting is most dangerous when you are already losing.

**Fixed-fractional betting** stakes a percentage of your *current* balance. It can never reach literal zero — but it pays for that with volatility drag, the geometric tax that fixed-amount betting avoids. Its danger is *practical* ruin: being dragged to a balance so small that recovery is hopeless. The protection is keeping the fraction `f` small enough that the drag stays negligible relative to the edge. The good news: your bets shrink automatically in a drawdown, which is a built-in brake. The bad news: that same auto-shrinking means after a deep drawdown your dollar bets are tiny, so climbing back takes forever even when the edge reasserts — the recovery asymmetry in slow motion.

Most real traders are fixed-fractional, often without realizing it ("I always risk 1% of capital"). So the volatility-drag math, not the literal-zero math, is the one that governs their survival. The practical rule is the same in both worlds — **bet small relative to your bankroll** — but the failure you are guarding against differs: literal zero for the fixed-amount bettor, a hopeless 95% hole for the fixed-fractional one.

#### Worked example: the fractional account that grinds down on a positive edge

Suppose you have a thin edge — you win **51%** of even-money bets, so arithmetic expectancy is a positive +\$0.02 per dollar — and you bet a hefty **30%** of a **\$100,000** account each time. Every single bet is +EV. Watch the geometric growth rate, which is what actually governs your account.

The expected log-growth per bet is `p × ln(1 + f) + (1 − p) × ln(1 − f)`:

- Win leg: `0.51 × ln(1.30) = 0.51 × 0.2624 = +0.1338`
- Loss leg: `0.49 × ln(0.70) = 0.49 × (−0.3567) = −0.1748`
- Sum: `0.1338 − 0.1748 = −0.0410` per bet

The geometric growth rate is **negative** — about −4.1% per bet — even though the arithmetic expectancy is positive. Your wealth multiplies by roughly `e^(−0.041) ≈ 0.96` per bet *on average*, so over 100 bets your \$100,000 account is expected to compound down to about `100,000 × e^(−0.041 × 100) ≈ 100,000 × 0.0166 ≈` **\$1,660** — a 98% loss — *while winning the majority of bets and being +EV on every one.* The over-sized fraction turned a real edge into a wealth-destruction machine, purely through volatility drag.

*A positive arithmetic edge can hide a negative geometric growth rate; when the bet fraction is too large, you can win most of your bets, profit on every one in expectation, and still watch your account melt to nothing.*

### Risk of ruin climbs steeply with bet size

Because volatility drag grows with the square of the bet fraction, the probability of practical ruin does not rise gently as you bet bigger — it rises *steeply*, and then slams into certainty. We can see this directly by simulation: take the same +EV edge, vary only the bet fraction, and count what share of paths get dragged to a practical-ruin level (say, an 80% drawdown) over a fixed run of bets.

![Probability of practical ruin versus bet fraction for a fixed positive edge, showing ruin probability climbing steeply from near zero in a half-Kelly survival zone to near certainty as bets grow](/imgs/blogs/risk-of-ruin-why-positive-expectancy-is-not-enough-2.png)

The curve is the shape of the danger. Down in the **half-Kelly survival zone** — betting 2% to 5% of the account — practical ruin is essentially nil; the edge compounds and the drag is negligible. As the bet fraction climbs, ruin probability lifts off, turns steeply upward through the middle, and by the time you are betting 35%–40% of the account *every single trade*, ruin is a near-certainty — even though the edge is unchanged and still positive on every bet.

The blue dashed line marks `f* = 10%`, the **full-Kelly** fraction for this edge (we will get to what that means). The important thing for now: full-Kelly is the *growth-optimal* fraction — the bet size that maximizes your long-run compounding. It is *not* the *ruin-optimal* fraction. Even at full-Kelly, the path is wild, drawdowns are savage, and the practical risk of ruin is uncomfortable. The safe place to live is to the *left* of full-Kelly — typically half of it or less.

#### Worked example: scaling the same logic to a \$10,000,000 book

The math is identical at institutional scale; only the zeros change, and the lesson gets sharper because the dollar drawdowns are vivid. Suppose you run a **\$10,000,000** book with that same 55% / 1:1 edge.

At a disciplined **5% per position** (\$500,000 at risk per trade), volatility drag costs about 0.25% per win-loss pair, the edge dominates, and the book compounds. A nasty run of, say, eight consecutive losing positions takes you from \$10,000,000 to `10,000,000 × (0.95)^8 ≈` **\$6,634,000** — a painful 34% drawdown, but a survivable one. You are still very much in business.

At a reckless **40% per position** (\$4,000,000 at risk per trade), the *same* eight-loss streak takes you to `10,000,000 × (0.60)^8 ≈` **\$167,000** — a **98.3% drawdown.** From \$10 million to \$167,000 on an eight-trade losing streak that any real strategy will hit eventually. To climb back to \$10 million from \$167,000 you would need to make about **5,900%** — you are, for all practical purposes, finished. The edge did not fail. The size did.

*The same losing streak that costs a 5%-sized book a recoverable third costs a 40%-sized book everything; bet fraction is the dial that decides whether a normal bad run is a setback or a death.*

## Putting it together: the ruin map of edge versus bet size

We now have the two inputs that govern ruin under fractional betting: how big your edge is, and how big you bet. It helps enormously to see them on the same picture — a map of where ruin lives.

![Heatmap of ruin probability as a function of per-bet edge on the horizontal axis and bet fraction on the vertical axis, with a green safe corner of real edge and small bets and a red ruin zone of thin edge or oversized bets](/imgs/blogs/risk-of-ruin-why-positive-expectancy-is-not-enough-6.png)

The heatmap puts edge (win probability, at 1:1 odds) on the horizontal axis and bet fraction on the vertical axis; each cell's colour is the simulated probability of practical ruin. The geography is the lesson:

- **The green safe corner — bottom right.** A real edge (high win probability) combined with small bets (low fraction) gives near-zero ruin. This is where you want to live, always. Real edge, small bets.
- **The red ruin zone — top, and the whole left.** Two separate ways to land here. Bet too large (high up the chart) and ruin is near-certain *even with a strong edge* — that is the volatility-drag mechanism doing its work. Or have a thin edge (far left, near a coin flip) and ruin creeps up even at modest bet sizes, because there is almost nothing to lean against. With *no* edge or a negative one, no bet size is safe — every fraction eventually ruins you.
- **The 10% ruin contour** is the boundary worth memorizing. It slopes upward to the right: the stronger your edge, the larger a fraction you can *get away with* before ruin probability rises past 10%. But notice how shallow the slope is. Even a strong edge buys you only a little extra room to bet bigger. The dominant axis is bet fraction, not edge.

The map encodes the single most important practical takeaway of risk management: **you cannot out-edge bad sizing.** Improving your strategy from a 54% to a 58% win rate moves you only modestly to the right. Cutting your bet fraction from 40% to 5% moves you from the deep-red top of the chart to the deep-green bottom. The lever that matters is the one in your hands every single time you click the button: *how much.*

![Before-and-after comparison of expectancy thinking that bets big on a positive edge versus survival thinking that sizes so the probability of ruin stays tiny and only then collects the edge](/imgs/blogs/risk-of-ruin-why-positive-expectancy-is-not-enough-7.png)

This is the mental shift the whole post is arguing for, drawn as before-and-after. **Expectancy thinking** sees a positive edge and reasons "it's +EV, so bet as much as I can to maximize the win" — and lands on the absorbing barrier. **Survival thinking** treats the edge as the reason the bet is *worth taking* but treats *size* as a completely separate decision, governed by one rule: keep the probability of ruin tiny. The edge tells you *whether* to bet. Survival math tells you *how much*. They are different questions, and conflating them is how good traders blow up.

## Where Kelly comes in (and why you should bet less than it says)

Once you accept that bet size is a free variable you must choose, the obvious question is: *what size is best?* There is a famous, mathematically precise answer called the **Kelly criterion**, and it is worth knowing exactly what it does and does not promise.

The Kelly fraction is the bet size that maximizes your **long-run geometric growth rate** — the rate at which your compounded wealth grows. For a bet that wins with probability `p` at odds `b`-to-1, the full-Kelly fraction is `f* = (p(b+1) − 1) / b`. The series' `dr.kelly_fraction(p, b)` computes it. For our running 55% / 1:1 edge:

f* = `dr.kelly_fraction(0.55, 1.0)` = (0.55 × 2 − 1) / 1 = **0.10**, i.e. 10% of the bankroll.

So full-Kelly says: with this edge, bet 10% of your account each time to grow your money as fast as possible in the long run. Crucially, Kelly *already* prices in ruin avoidance to a degree — betting more than full-Kelly is strictly worse on *every* axis (lower growth *and* higher risk), so the Kelly fraction is a hard ceiling no rational sizer should exceed. We covered the full derivation in a [dedicated Kelly post](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews), so we will not re-derive it here.

But here is the part beginners miss: **full-Kelly is far too aggressive for real trading, for three concrete reasons.**

First, **Kelly assumes you know your edge exactly.** You do not. Your "55% win rate" is an *estimate* from a finite, noisy sample, and if your true edge is lower than you think, you are over-betting — and over-betting past Kelly is catastrophic, not merely suboptimal. Betting full-Kelly on an over-estimated edge is a fast road to the red zone.

Second, **full-Kelly produces drawdowns most humans cannot stomach.** At full-Kelly you should expect to see your account cut in half from time to time *as a normal, healthy outcome of the optimal strategy.* A 50%+ drawdown that is "supposed to happen" is one that triggers panic, redemptions, margin calls, and abandonment of the strategy at the worst moment — which converts a paper drawdown into a real, permanent ruin.

Third, real strategies have **fat tails** — losses larger and more frequent than a clean coin-flip model predicts. Kelly's clean math underestimates the true risk, so the practical safe fraction is lower still. (The heavy math on tail behaviour lives in the [extreme-value theory post](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants); the working takeaway is: real losses are worse than your model thinks.)

The practitioner's resolution is to bet a **fraction of Kelly** — typically half-Kelly or less. Half-Kelly gives up only about 25% of the long-run growth rate while roughly *halving* the volatility and dramatically shrinking the drawdowns and ruin probability. It is one of the best risk-reward trades in all of finance: you sacrifice a quarter of your growth to buy a huge reduction in the chance of being wiped out. For our 55% edge, full-Kelly is 10%; **half-Kelly is 5%** — which is exactly the disciplined sizing in the worked examples above. That is not a coincidence. That is where survivors live.

The reason half-Kelly is such a good deal traces straight back to the volatility-drag arithmetic. Growth rises roughly *linearly* as you increase the bet fraction toward Kelly, but drag rises with the *square* of the fraction. Near the Kelly peak the growth curve is flat — you are at the top, so moving a little either way barely changes growth — while the risk is still climbing steeply. Backing off from full-Kelly to half-Kelly slides you down the gentle far side of the growth hill (losing only a sliver of growth) while sliding a long way down the steep risk slope (shedding a large chunk of volatility and drawdown). You are trading on a wildly favourable exchange rate: a little growth for a lot of safety. And because the growth curve is *symmetric* around Kelly — betting *twice* Kelly gives you the *same* growth as betting *zero*, namely none — over-betting past Kelly is the worst of all worlds: you take maximum risk for growth that collapses back toward zero. There is never a reason to be at or above full-Kelly, and every reason to be well below it.

#### Worked example: why half-Kelly is the survivor's bet

You have the 55% / 1:1 edge and a \$100,000 account. Full-Kelly says bet \$10,000 (10%) per trade. Should you?

Consider what an eight-loss streak — uncommon but entirely possible — does at each sizing:

- **Full-Kelly, 10% per trade.** After eight straight losses: `100,000 × (0.90)^8 ≈` **\$43,000.** A 57% drawdown. To recover you need a +132% gain. Brutal, and right at the edge of what is psychologically survivable.
- **Half-Kelly, 5% per trade.** After the same eight straight losses: `100,000 × (0.95)^8 ≈` **\$66,300.** A 34% drawdown, needing a +51% gain to recover. Painful but clearly survivable — and the long-run growth is only about a quarter lower.

You gave up a quarter of your growth rate. In exchange, your worst-streak drawdown shrank from "career-threatening" to "rough quarter." That trade — a little growth for a lot of survival — is the entire discipline of position sizing in one decision.

*Half-Kelly keeps almost all of the upside while removing most of the chance that an ordinary losing streak ends your career; the growth you give up is cheap insurance against the absorbing barrier.*

## Common misconceptions

**"If my strategy is +EV, I can't go broke — I'll make money eventually."** False, and it is the central error this entire post exists to correct. Expectancy assumes infinite repetitions at a fixed size with no bottom. The absorbing barrier at zero ends the repetitions early for some paths. We *showed* this: a 55% / 1:1 edge — strongly positive — sized at 40% of a \$100,000 account gets dragged to a \$1,000 floor on perfectly ordinary luck. +EV makes the *average* path profitable; it says nothing about whether *your* path survives long enough to collect.

**"A small per-bet disadvantage is no big deal."** False. A 1% per-bet disadvantage (`p = 0.49`) starting from `i = 50` of `N = 100` gives an **88%** probability of ruin. Compounding amplifies tiny edges *and* tiny disadvantages into near-certainties. In trading, fees, slippage, and the bid-ask spread are exactly this kind of small, relentless per-bet disadvantage — which is why a strategy that looks marginally profitable before costs can be a reliable ruin machine after them.

**"With fixed-fractional betting I can never go to zero, so I'm safe."** False — this is the most dangerous half-truth in money management. You never hit *literal* zero, true. But you can be dragged to a 95% or 99% drawdown, a level from which the [recovery asymmetry](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) (a 95% loss needs a +1,900% gain) means you never come back. "Never technically zero" and "practically finished" are the same outcome for your career.

**"More edge solves over-betting."** False. The ruin map shows it: improving your win rate from 54% to 58% nudges you a little to the right; cutting your bet fraction from 40% to 5% moves you from the red zone to the green corner. The dominant lever is bet size, not edge. You cannot out-edge reckless sizing.

**"Bet full-Kelly — it's mathematically optimal."** Misleading. Full-Kelly maximizes long-run *growth*, but it assumes you know your edge exactly (you do not), it produces 50%+ drawdowns as a normal outcome (most people abandon the strategy mid-drawdown, converting a paper loss into a real one), and it ignores fat tails (real losses are worse than the model). Half-Kelly gives up ~25% of growth for ~50% less volatility — a far better real-world trade.

**"Risk of ruin is a gambling concept, not a trading one."** False. Every leveraged position, every margin account, every fund with redeemable investors faces an absorbing barrier — a liquidation level, a margin call, a redemption threshold. The casinos just make the barrier vivid. In markets the barrier is often *closer* than at the table, because leverage moves it up from zero to wherever your maintenance margin sits.

## How it shows up in real markets

The gambler's-ruin math is not a toy. The most spectacular blow-ups in modern finance are, underneath, the same story: a real edge, sized far too large, meeting an absorbing barrier.

**Long-Term Capital Management (Aug–Sep 1998).** LTCM's convergence trades had a genuine, well-researched statistical edge — small, reliable mispricings that should pay off as prices converged. The problem was size: roughly **25-to-1** balance-sheet leverage on about \$4.7 billion of equity, with around \$1.25 trillion in gross derivatives notional. At that leverage, the "small" adverse moves of the 1998 Russia crisis were fatal. The fund lost about **\$4.6 billion** — most of its capital — in roughly four months and required a Fed-organized \$3.6 billion rescue. The edge was real; the position size, relative to the bankroll, guaranteed that an ordinary tail event would touch the absorbing barrier. (The strategic-crowding angle is dissected in the [LTCM game-theory case study](/blog/trading/game-theory/case-study-ltcm-1998-the-crowded-genius-trade).)

**Amaranth Advisors (Sep 2006).** A concentrated, leveraged bet on natural-gas calendar spreads. The edge thesis may have had merit, but the *size* — a position so large it dominated the market it traded — meant there was no liquid exit when the spreads moved against it. Amaranth lost about **\$6.6 billion**, most of it in a single week. Over-sizing in an illiquid instrument is the absorbing barrier wearing a different costume: you cannot bet your way out because you cannot get out at all.

**Archegos Capital Management (Mar 2021).** Concentrated single-stock exposure financed through total-return swaps at roughly **5x+ leverage**, with each prime broker blind to the total size. When the underlying stocks fell, the margin calls came, the forced unwind cascaded, and the absorbing barrier did its work: Archegos was wiped out and its banks lost over **\$10 billion** (Credit Suisse alone about \$5.5 billion). The "edge" was high conviction in a few names; the size, hidden and enormous, converted a normal drawdown into total ruin.

**Volmageddon / the XIV blow-up (5 Feb 2018).** The short-volatility carry trade had paid off for years — a real, persistent edge (selling insurance collects premium most of the time). But the product was effectively sized at extreme implicit leverage to that edge. When the VIX spiked about 20 points in a day (from 17.3 to 37.3, its largest one-day percentage jump on record), the XIV note's value fell roughly **96%** after the close and the product was terminated. Years of edge erased in hours, because the size left no room for the tail. (The mechanism is dissected in the [Volmageddon case study](/blog/trading/options-volatility/case-study-volmageddon-2018-and-the-short-vol-blowup).)

**The COVID crash (Feb–Mar 2020).** Not a single firm but a market-wide demonstration of how an absorbing barrier reaches up to meet leverage. The S&P 500 fell about **34%** from its 19 February peak to its 23 March trough — the fastest bear market on record — and the VIX closed at a record **82.69** on 16 March. Strategies that were "diversified" and "+EV" on paper saw their correlations snap to one and their drawdowns blow through risk limits in days, not months. Levered players who had sized for normal volatility hit margin calls and were forced to sell into the collapse, the classic doom loop where the very act of de-risking pushes prices down and tightens the noose. The funds that survived were the ones that had sized small enough that a once-in-a-decade move was painful rather than terminal.

**The yen-carry unwind (5 Aug 2024).** A crowded funding-carry trade — borrow cheap yen, buy higher-yielding assets — that had paid a steady edge for years, unwinding in a matter of days. The Nikkei fell about **12.4%** in a single session, its worst day since 1987, and the VIX spiked intraday to around **65.7**. Leverage built quietly on a reliable carry edge met a sudden repricing, and the deleveraging fed on itself. Same skeleton as every other entry on this list: real edge, oversized and levered, meeting a barrier that the leverage had quietly moved up to within striking distance.

The pattern is monotonous because the math is universal. In every case the edge was real and the size was the killer. Each firm could have run the *exact same strategy* at a fraction of the size and survived — poorer, slower, but alive to compound for another decade. The graveyard is full of correct trades sized to fail.

## How long a losing streak should you expect?

The whole playbook below hinges on sizing against your worst plausible losing streak. So it is worth quantifying: how long a streak will a winning strategy actually hit? People dramatically underestimate this, and the underestimate kills them.

If you lose any single bet with probability `q`, the chance of losing `k` in a row, starting at a given bet, is `q^k`. With our 55% edge, `q = 0.45`. So:

- A 5-loss streak from a given starting bet: `0.45^5 ≈ 1.8%`.
- An 8-loss streak: `0.45^8 ≈ 0.17%`.
- A 10-loss streak: `0.45^10 ≈ 0.034%`.

Those look reassuringly tiny — *per starting point.* But you do not get one starting point. You get *thousands*, because a streak can begin on any bet in your career. Over a long sequence of `n` bets, the expected number of times you start a run that reaches length `k` or more is roughly `n × q^k`, and the probability you see *at least one* such streak climbs fast. Over **1,000 bets** with `q = 0.45`:

- Expected 5-loss streaks: `1,000 × 0.45^5 ≈ 18`. You will see roughly *eighteen* five-loss streaks — they are routine.
- Expected 8-loss streaks: `1,000 × 0.45^8 ≈ 1.7`. You should *expect* to hit an eight-loss streak, probably more than once.
- The probability of seeing at least one 10-loss streak over 1,000 bets is roughly `1 − (1 − 0.45^10)^1000 ≈ 29%` — better than a one-in-four chance.

This is the streak you must survive *in a recoverable state.* An eight-loss streak is not a freak event you can dismiss; over a real trading career of thousands of trades, it is a near-certainty. That is exactly why the worked examples kept stress-testing against eight straight losses: it is the routine bad run, not the apocalypse. If your bet fraction does not survive an eight-loss streak with most of your capital intact, your sizing is wrong — not because you are unlucky, but because you are *normal.*

#### Worked example: the streak that decides your sizing

You run the 55% edge on a **\$100,000** account and want to guarantee that your near-certain eight-loss streak leaves you with at least **70%** of your capital (a recoverable 30% drawdown). What is the largest fixed fraction `f` you can bet?

You need `(1 − f)^8 ≥ 0.70`. Solving: `1 − f ≥ 0.70^(1/8) = 0.70^0.125 ≈ 0.9563`, so `f ≤ 0.0437` — about **4.4%** of capital per bet. At that fraction, an eight-loss streak takes \$100,000 down to `100,000 × 0.9563^8 ≈` **\$70,000**, a survivable 30% dent. Push to `f = 10%` (full-Kelly) and the same streak leaves `100,000 × 0.90^8 ≈` **\$43,000** — a 57% hole. Push to `f = 20%` and you are at `100,000 × 0.80^8 ≈` **\$16,800** — an 83% loss, practical ruin from an *ordinary* bad run.

*Size by your worst routine streak, not your average outcome: the fraction that survives the eight-loss run you will certainly hit is far smaller than the fraction your edge alone seems to justify.*

## The risk-of-ruin survival playbook

Concrete rules, ordered by how much they protect you. None of these requires improving your edge — they all work on the one variable you control directly: size.

1. **Separate the two decisions, always.** First ask "does this bet have positive expectancy?" — that decides *whether* to take it. Then, as a completely independent question, ask "how much can I lose here without threatening my survival?" — that decides *how much*. Never let a strong edge talk you into a large size. Edge is the entry ticket; size is a separate, ruin-governed choice.

2. **Cap risk per position at a small fraction.** A widely used rule is **risk no more than 1–2% of capital on any single position** (where "risk" is your loss if your stop is hit, not your notional). On a \$100,000 account that is \$1,000–\$2,000 of risk per trade. On a \$10,000,000 book, \$100,000–\$200,000. At that fraction, even a long losing streak produces a recoverable drawdown, never a fatal one — you stay in the green corner of the ruin map.

3. **Bet a fraction of Kelly, never more, usually half or less.** If you size by edge, compute the Kelly fraction as a *ceiling* and then bet *half of it or less*. You sacrifice about a quarter of your growth to roughly halve your volatility and slash ruin probability. Going over full-Kelly is strictly dominated — more risk *and* less growth — so it is never correct.

4. **Size against your worst plausible losing streak, not your average.** Before sizing, ask: "If I hit eight (or ten, or fifteen) losses in a row — which I will, eventually — does my account survive in a recoverable state?" If `(1 − f)^k` for your realistic worst streak `k` leaves you below ~70% of capital, your fraction `f` is too big. Size so the streak you *will* eventually hit is survivable.

5. **Account for fat tails and edge uncertainty by sizing down further.** Your measured edge is an over-estimate more often than you would like, and real losses cluster and exceed the model. Both errors point the same way: bet *smaller* than the clean math suggests. When in doubt, the survivor's bias is to under-bet.

6. **Watch leverage — it moves the absorbing barrier up toward you.** Unlevered, your barrier sits at zero, far below. Every turn of leverage raises the liquidation/margin barrier closer to your current equity, shrinking the losing streak you can survive. The 1998, 2006, 2021, and 2018 blow-ups were all, at root, leverage moving the barrier up until an ordinary move touched it.

7. **Treat a deep drawdown as a sizing signal, not a doubling-down opportunity.** After a large loss your bankroll is smaller, so your fixed-fractional dollar bets *should* shrink automatically — let them. The instinct to "bet bigger to win it back" is precisely the move that converts a survivable drawdown into the absorbing barrier. Smaller account, smaller bets.

The thread running through all seven rules is the series' spine: **you can only compound if you are still in the game.** A positive expectancy is the reason to play. Surviving long enough to collect it is the reason to size small. Get the size right and a real edge, given enough time, is nearly unstoppable. Get the size wrong and the same edge will, with mathematical reliability, take you to zero. The edge is necessary. Survival is everything.

### Further reading

- [The asymmetry of losses: why a 50% loss needs a 100% gain](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) — why the drawdowns ruin produces are so hard to climb out of.
- [Why risk management is the real edge: surviving to trade tomorrow](/blog/trading/risk-management/why-risk-management-is-the-real-edge-surviving-to-trade-tomorrow) — the survival thesis this whole series is built on.
- [Ergodicity: time-average vs ensemble-average and the coin flip that ruins you](/blog/trading/risk-management/ergodicity-time-average-vs-ensemble-average-and-the-coin-flip-that-ruins-you) — why *your* path differs from the average path, the deep reason ruin matters.
- [The Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) — the full derivation of the growth-optimal bet fraction.
- [Position sizing and risk of ruin in options trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) — applying these ideas to the asymmetric payoffs of options.
