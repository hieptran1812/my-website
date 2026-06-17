---
title: "Expected Value, Edge, and Variance: Thinking Like the House"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Why win rate is not edge, how variance hides a real edge in the short run, and how the house survives the swings to let a small positive expectation pay out."
tags: ["game-theory", "expected-value", "edge", "variance", "risk-of-ruin", "kelly-criterion", "position-sizing", "trading", "probability"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The casino does not predict the next spin; it owns a tiny positive edge on every bet and a bankroll deep enough to survive the swings, so the law of large numbers turns that edge into near-certain profit. Your trading edge works the same way: it only exists against a specific counterparty, and you must size it to survive.
>
> - **Win rate is not edge.** A strategy that wins 90% of the time can have negative expected value (EV); one that wins 35% of the time can be hugely positive. EV, not win rate, is what pays you.
> - **Variance hides edge in the short run.** A real edge can lose money over hundreds of trades by pure chance, and a real loser can look like a genius for months. You cannot read your edge off a short streak.
> - **Risk of ruin is the silent killer.** Even with a positive edge, betting too big relative to your bankroll makes eventual bankruptcy near-certain. Survival comes first; profit comes second.
> - **The one rule to remember:** be the house, not the gambler. Find a small measured edge, bet a fraction the Kelly criterion sizes for you, and stay in the game long enough for the law of large numbers to pay out.

In March 2000, a poker pro named Chris Ferguson started with \$1 in an online account and turned it into \$20,000 over about a year and a half — not by winning every hand, but by never once risking enough of his stack to go broke. Around the same time, thousands of players with bigger bankrolls and better cards busted out. The difference was almost never who held the best hand on any given night. It was who understood that a hand of poker, like a trade, is not a forecast you either get right or wrong. It is a *bet with an edge and a variance*, and the player who survives the variance long enough to collect the edge is the one who walks away rich.

This is the single most important idea in this whole series, and it is the one that separates people who think they understand markets from people who actually make money in them. The casino on the Las Vegas strip is not psychic. The roulette croupier has no idea where the ball will land, and does not care. The house wins for one reason: on every \$1 you wager, it expects to keep about a nickel, and it has enough money in the vault to survive any streak of bad luck you can throw at it before that nickel-per-bet edge grinds you down. It is not a forecaster. It is a survivor of variance with a small, relentless, *measured* edge.

The reason this matters so much for trading is that almost every losing trader is, without realizing it, playing the gambler's game while believing they are playing the house's. They hunt for the trade that will be *right* — the perfect call, the big win — and they bet too much of their account on it because being right feels certain. The house never tries to be right on a single bet; it arranges to be right *on average*, across enough bets that the average is all that matters, and it sizes every bet small enough that being wrong a dozen times in a row changes nothing. By the end of this post you will be able to look at any strategy — yours or someone else's — and answer the only three questions that decide whether it makes money: is the edge positive, can the bankroll survive the variance, and is the bet sized to let the law of large numbers pay out?

The diagram below is the mental model for the entire post: two trading strategies, side by side. The one on the left wins nine times out of ten — and loses money. The one on the right loses almost two times out of three — and makes money. If your gut says the left one is better, this post is for you, because that gut feeling is exactly what the market takes your money for.

![Two bar charts comparing a high win rate negative EV strategy with a low win rate positive EV one](/imgs/blogs/expected-value-edge-and-variance-thinking-like-the-house-1.png)

## Foundations: expected value, edge, variance, ruin, and Kelly from zero

Before we can talk about who is on the other side of your trade, we need five words to mean exactly one thing each. Most traders use these words loosely, and the looseness is where the money leaks out. Let us build them from nothing.

### Expected value: the average over a thousand parallel universes

**Expected value** (EV) is the average outcome you would get if you could run the exact same bet a thousand times and average the results. It is not the most likely outcome, and it is not what will happen this time. It is the long-run average. The recipe never changes: list every possible outcome, multiply each payoff by its probability, and add them all up.

A *probability* is just a number between 0 and 1 (or 0% and 100%) describing how often something happens. A *payoff* is what you win or lose if it happens — a positive number for money in, a negative number for money out. The formula is:

$$EV = \sum_i p_i \cdot x_i$$

where $p_i$ is the probability of outcome $i$ and $x_i$ is its payoff in dollars. The Greek $\sum$ ("sigma") just means "add up all of these." That is the whole of it.

#### Worked example: the EV of a fair coin and a casino spin

Start with the simplest game in the world. You flip a fair coin: heads you win \$1, tails you lose \$1. There are two outcomes, each with probability 0.5:

$$EV = (0.5)(+\$1) + (0.5)(-\$1) = +\$0.50 - \$0.50 = \$0$$

The EV is exactly zero. This is a *fair game* — over many flips you neither gain nor lose. No casino would ever offer it, because a casino that breaks even pays no rent.

Now the real game. On a European roulette wheel there are 37 pockets: numbers 1 through 36 plus a single green zero. You put \$1 on a single number. If it hits (probability 1/37) you are paid 35-to-1, so you collect \$35 of profit. If it misses (probability 36/37) you lose your \$1. The expected value is:

$$EV = \left(\tfrac{1}{37}\right)(+\$35) + \left(\tfrac{36}{37}\right)(-\$1) = +\$0.946 - \$0.973 = -\$0.027$$

You expect to lose about 2.7 cents on every \$1 you wager. That 2.7% is the **house edge**, and it is the entire business model of every casino on Earth. The croupier does not predict the ball. The croupier collects 2.7 cents per dollar, forever, from people who think the next spin is "due." Note the sign: your EV is negative, which means the house's EV is positive, because every dollar you lose is a dollar it keeps. Hold onto that — it is the spine of this series. *Your loss is a specific counterparty's gain.* One sentence of intuition: expected value is the only number that tells you whether a game is worth playing, and at a casino it is always tilted against you by design.

Two subtleties are worth nailing down right now, because they trip up almost everyone. First, **the EV is almost never one of the actual outcomes.** You will never lose 2.7 cents on a single roulette spin — you either lose \$1 or win \$35. The EV of $-\$0.027$ is a fiction that only becomes real across many spins. People dismiss EV because "that's not what actually happens," and they are confusing the average with the outcome. The average is exactly the point: it is what *accumulates*. Second, **the probabilities have to be honest.** At a roulette table the probabilities are fixed by physics and you can compute them exactly. In a trade, you have to *estimate* them, and a wrong probability poisons the whole calculation. Much of the skill in trading is not the arithmetic of EV — that part is easy — but the honesty of the probabilities you feed into it. A trader who quietly rounds a 40% win probability up to 55% because the setup "feels right" has fooled the only person the math cannot protect: themselves.

Here is the same idea laid out as a table, so the difference between the four numbers people constantly confuse is unmistakable:

| Quantity | What it measures | Roulette single-number bet | Why it matters |
|---|---|---|---|
| Win rate | Fraction of bets that pay off | 1 in 37 ≈ 2.7% | Feels like skill; says nothing about size |
| Average win | Size of a winning outcome | +\$35 | Half of the edge story |
| Average loss | Size of a losing outcome | −\$1 | The other half |
| Expected value (edge) | Probability-weighted average payoff | −\$0.027 per \$1 | The only number that decides if you should play |

Read across the roulette row: a *terrible* win rate (2.7%) combined with a *huge* payoff (35-to-1) still nets a negative edge, because the payoff is not quite big enough to cover how rarely you win. Change any one of those three numbers and the edge moves. That is the whole game — and notice that win rate, on its own, told you nothing.

### Win rate versus edge: the trap that empties accounts

Here is where most traders get fleeced. The **win rate** is the fraction of your bets that come out positive. The **edge** is your expected value per bet. They are *different numbers*, and confusing them is the most expensive mistake a beginner makes.

A high win rate feels wonderful. Winning nine trades out of ten feels like skill. But the win rate says nothing about *how much you win when you win* versus *how much you lose when you lose*. The edge — the EV — combines both the probability and the size of each outcome. You can win almost always and still bleed to death, if the rare loss is large enough.

#### Worked example: the 90%-win-rate loser and the 35%-win-rate winner

Look back at the two strategies in the cover figure. They are computed, not invented; the numbers come straight from the model.

**Strategy A** wins 90% of the time, pocketing \$10 each win, but the other 10% of the time it loses \$100:

$$EV_A = (0.90)(+\$10) + (0.10)(-\$100) = +\$9 - \$10 = -\$1$$

Nine wins out of ten, and the strategy *loses* \$1 per trade on average. This is the shape of selling far out-of-the-money options, or "picking up pennies in front of a steamroller." You win small, constantly, and feel brilliant — right up until the one trade that gives back everything and more.

**Strategy B** wins only 35% of the time, but each win is \$300 against a \$100 loss:

$$EV_B = (0.35)(+\$300) + (0.65)(-\$100) = +\$105 - \$65 = +\$40$$

A 35% win rate, and the strategy *makes* \$40 per trade. This is the shape of trend-following: most trades are small losers, a few are huge winners, and the math is wildly positive. The trader running Strategy B loses on almost two trades out of three and goes home richer than the one running Strategy A.

The lesson, stamped into your skull: **win rate is a feeling; edge is a fact.** One sentence of intuition: you can be wrong most of the time and still win, or right most of the time and still lose — only the expected value decides which.

Why does the high-win-rate trap fool so many people? Because human psychology is wired to count *frequency*, not *magnitude*. Nine wins in a row releases nine little hits of being right, and the brain files the strategy under "works." The one big loss arrives as a single event, and the brain files it under "bad luck, an exception." But the math does not care how the experience *feels* — it weights by dollars, and the nine \$10 wins (\$90 total) do not cover the one \$100 loss. The market is, in a very real sense, a machine that converts your psychological miscounting into its profit. The casino does the same thing: the slot machine that pays out small amounts frequently and a jackpot almost never is engineered to maximize the *feeling* of winning while keeping the EV firmly negative. Whenever a strategy makes you feel like you almost never lose, that is precisely the moment to compute the EV, because pleasant frequency is exactly what a negative edge feels like from the inside.

There is a deeper structural reason the two payoff shapes recur. Strategy A — many small wins, rare large losses — is the payoff of *selling insurance*: you collect a steady premium and pay out big in the rare disaster. Strategy B — many small losses, rare large wins — is the payoff of *buying insurance* or *buying a lottery ticket on a real trend*: you bleed small premiums and cash in occasionally for a fortune. Neither shape is good or bad on its own; what makes them win or lose is whether the premium is priced correctly relative to the tail. The insurer wins when the premium overpays for the risk, and loses when it underprices the disaster. So the question is never "do I want a high win rate or a low one?" It is "is the price right for the risk I am taking?" — and the only way to answer that is EV.

For the deep version of this exact point with a trading dataset behind it, see [Expectancy: why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies). This post owns the *strategic* framing — your edge versus a counterparty — and links out for the trade-log mechanics.

### Variance: the noise that hides the signal

So edge is what matters. But here is the cruel part: **you cannot see your edge directly.** Edge is a long-run average, and the long run is long. In the short run, your results are dominated by **variance** — the random scatter of outcomes around their average.

Variance is why a coin can land heads eight times in a row even though it is fair. It is why a casino has losing nights, a great poker player has losing weeks, and a real trading edge can lose money for months. Variance does not contradict the edge; it *hides* it. The edge is the slow, quiet drift; variance is the loud, fast noise on top.

The technical measure is the **standard deviation** of your per-bet outcome — roughly, the typical size of a single swing. (Standard deviation is just a number that says how spread out your outcomes are; a bigger one means wilder swings.) What matters for our purposes is the relationship between the two as you make more bets. The edge accumulates *linearly* with the number of bets $N$: ten times the bets, ten times the expected profit. But the random swing of your *total* grows only with the *square root* of $N$. Ten times the bets, only about three times the swing ($\sqrt{10} \approx 3.16$). So as $N$ grows, the edge pulls ahead of the noise. This is the **law of large numbers**, and it is the engine that makes the house rich.

The square-root rule is worth pausing on because it is the mathematical heart of "be the house." Your expected profit is $N \times \text{edge}$, and your typical swing is $\sqrt{N} \times \text{(per-bet standard deviation)}$. The ratio of edge to swing — call it your *signal-to-noise* — therefore grows like $N / \sqrt{N} = \sqrt{N}$. Four times as many bets, twice as much signal-to-noise. A hundred times as many bets, ten times as clean a result. This is why the house wants *volume* above all else: every extra bet does not just add a little edge, it makes the whole operation more certain. It is also why a trader who takes 2,000 small, positive-EV trades a year has a fundamentally more reliable business than one who takes 20 big ones, even at the same edge per trade — the high-volume trader is sitting much further out the $\sqrt{N}$ curve where variance has surrendered.

### Risk of ruin: the wall you can hit before the edge pays out

The law of large numbers has a fatal catch: it only helps you if you are *still playing*. If a bad streak wipes out your entire bankroll before the long run arrives, you are out of the game, and your beautiful positive edge becomes worthless. This is the **risk of ruin** — the probability that you go broke before your edge can rescue you.

Ruin depends on three things: your edge (bigger edge, less ruin), the size of each bet relative to your bankroll (smaller bets, less ruin), and how deep your bankroll is in the first place (deeper, less ruin). For a simple even-money game where you bet one unit at a time, the probability of eventually losing a bankroll of $u$ units is:

$$P_{\text{ruin}} = \left(\frac{q}{p}\right)^{u}$$

where $p$ is your win probability and $q = 1-p$ is your loss probability. If $p \le 0.5$ you have no edge and ruin is certain ($P_{\text{ruin}} = 1$). With an edge ($p > 0.5$), ruin shrinks fast as your bankroll deepens — but it is never zero, and it can be uncomfortably large if your bankroll is thin. We will compute real numbers from this in a moment.

### Kelly: the bet size that maximizes long-run growth

The last piece. Once you have a positive edge, the question is not *whether* to bet but *how much*. Bet too little and you leave growth on the table. Bet too much and variance ruins you even though your edge is real. The **Kelly criterion** is the bet size that maximizes the long-run growth rate of your bankroll — the mathematically optimal compromise. For a bet that pays net odds $b$ (you win $b$ dollars per \$1 risked) with win probability $p$:

$$f^* = \frac{bp - q}{b}, \qquad q = 1 - p$$

$f^*$ is the fraction of your bankroll to wager. If $f^*$ comes out negative, you have no edge and should not bet at all. We will not re-derive Kelly here — the full derivation lives in [Position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion). What we will do is build the intuition and compute a worked number, because Kelly is the bridge from "I have an edge" to "I will survive long enough to collect it."

Those are the five words. Now let us watch them fight.

## Why the house wins: a small edge plus survival

Step back and look at what a casino actually is. It is not a fortune-teller. It is a machine for harvesting a tiny, known edge across an enormous number of independent bets, backed by a bankroll so large that no realistic streak can break it. Every design choice — the table minimums, the betting limits, the comped drinks that keep you playing — exists to maximize the *number* of bets and minimize the *size* of any single swing relative to the vault. The house is the purest expression of "small edge, many repetitions, deep bankroll, survive the variance."

The figure below makes the law of large numbers visible. It plots a player's *running average* result per \$1 bet at American roulette, where the house edge is a steeper 5.26% (two green pockets instead of one). Four different players, four different luck draws. Early on, the lines swing wildly — a player who hits a hot streak in the first hundred spins might genuinely be up, and the edge is invisible in the noise. But watch what happens as the bets pile up.

![Four running average lines converging onto the negative house edge as bets accumulate](/imgs/blogs/expected-value-edge-and-variance-thinking-like-the-house-5.png)

Every line gets dragged down onto the same fixed edge. The amber band — one standard deviation of the running average — pinches in like a funnel, narrowing as $1/\sqrt{N}$. By a few thousand spins, every player, lucky or unlucky, is pinned to the edge of losing 5.26 cents per dollar. The house did not get luckier over time. It simply waited for variance to surrender to the edge. That is the law of large numbers doing the casino's collection work.

#### Worked example: the house's nickel becomes a fortune

A single \$1 bet at American roulette has a house edge of about \$0.0526. That is a rounding error — nobody gets rich a nickel at a time, right? Watch it compound across volume:

- Over 100 bets, the house expects to keep $0.0526 \times 100 = \$5.26$.
- Over 1,000 bets, $0.0526 \times 1{,}000 = \$52.60$.
- Over 1,000,000 bets, $0.0526 \times 1{,}000{,}000 = \$52{,}600$.

Now the crucial second half. The *swing* on those million bets — one standard deviation of the total — is only about $\sqrt{1{,}000{,}000} \times \$1 = \$1{,}000$. So the house expects \$52,600 in profit with a swing of roughly \$1,000 around it. The profit is *fifty times* the noise. The casino's monthly take is, for all practical purposes, a certainty. One sentence of intuition: a microscopic edge, repeated enough times against a deep enough bankroll, stops being a gamble and becomes an annuity.

#### Worked example: a market-maker's edge per trade is tinier than a casino's

A casino keeps about 5 cents per dollar. An electronic market-maker keeps far less — but it makes vastly more bets. Suppose a market-maker captures an average of just **\$0.002** of edge per share traded (a fifth of a penny — a sliver of a one-cent spread), and it trades **50 million shares** in a day across thousands of names. Its expected daily profit is:

$$EV_{\text{day}} = \$0.002 \times 50{,}000{,}000 = \$100{,}000$$

The per-trade edge is a rounding error — a hundredth the size of the casino's. But the volume is enormous, so the $\sqrt{N}$ machine grinds the variance down to almost nothing relative to the \$100,000 expected take. The firm's profit, like the casino's, becomes a near-certainty *not because any single trade is reliable* — most are essentially coin flips — but because there are tens of millions of them and the bankroll dwarfs any single swing. One sentence of intuition: you do not need a big edge to be the house, you need a small honest edge and the volume and bankroll to let the law of large numbers collect it.

This is the entire reason market-makers, high-frequency firms, and quantitative funds look like casinos and not like fortune-tellers. They are not predicting where any single stock goes. They are collecting a fraction of a penny of edge — a sliver of the bid-ask spread, a tiny statistical mispricing — across millions of trades, with risk controls that cap the size of any single swing. We pick up that thread in [Zero-sum, positive-sum, and the house](/blog/trading/game-theory/zero-sum-positive-sum-and-the-house-where-trading-profits-come-from), where "be the house" is the whole point.

### Where does a trading edge actually come from?

There is one disanalogy between the casino and the market that you must respect. The casino's edge is *structural and permanent* — it is baked into the geometry of the wheel, and no amount of clever play by the gambler can erase it. A trading edge is *contested and temporary*. The moment a mispricing is widely known, traders pile in, the price corrects, and the edge evaporates. So unlike the casino, you cannot just "own the wheel"; you have to keep finding a counterparty who, for some real reason, is willing to take the worse side of your bet. There are only a handful of such reasons, and naming yours is the difference between an edge and a delusion:

- **A speed edge.** You see and act on information microseconds before others. This is the high-frequency market-maker's edge — they update their quotes faster than slower traders can pick them off, so the slow trader is the predictable counterparty paying the spread.
- **An information edge.** You know something the price has not yet absorbed — a better forecast, a fundamental insight, a dataset others lack. The counterparty is trading on stale or incomplete information.
- **A structural edge.** You are paid to provide a service the market needs — liquidity, insurance, the willingness to hold a risk others want off their books. The counterparty pays you a premium for taking a risk they do not want. This is the market-maker's spread and the insurer's premium.
- **A behavioral edge.** The counterparty is acting on emotion — panic-selling into a crash, chasing a bubble, refusing to take a loss. You are simply more disciplined than a predictable crowd. This is the most durable edge for an individual, because human psychology does not get arbitraged away.

If you cannot place your edge in one of those four buckets, you do not have one — you are the counterparty, the predictable player funding someone else's annuity, and the house math is running *against* you. That is the uncomfortable question every honest trader asks before sizing a position: *who is on the other side of this, and why are they wrong?*

## Variance, the great deceiver

If the law of large numbers is the casino's friend, variance is the trap that springs on everyone with a small bankroll and a short attention span. Here is the brutal truth that most retail traders never internalize: **a real, positive edge can lose money for a long time, and a real, negative edge can win money for a long time.** Variance is large enough, relative to a typical edge, that you simply cannot tell them apart from a short track record.

The figure below makes this visceral. It shows 60 different bankroll paths for the *same* genuinely positive-edge game: a 55% even-money bet that pays \$1 per win, starting from a thin \$30 bankroll, played 300 times. The straight slate line is the expected drift, computed from the per-bet expected value. Every path should, on average, climb that line.

![Sixty bankroll paths scattering around an upward expected value drift, several hitting zero](/imgs/blogs/expected-value-edge-and-variance-thinking-like-the-house-4.png)

Look at the chaos. The edge is real — the average path climbs — but individual paths fan out enormously. Some sprint ahead of the drift; some lag for the whole 300 bets; and several (the red ones) hit zero and stop, because a thin \$30 bankroll cannot absorb an early bad streak even when the edge is positive. Those players went broke holding a *winning* hand. They were right about the game and wrong about survival.

This is exactly what destroys traders. You find a genuine edge, you size it too aggressively, you hit a normal-and-expected losing streak early, and you blow up before the edge can pay you. Or the mirror image: you have *no* edge, but variance hands you a hot first month, you conclude you are a genius, you scale up, and the edge — which was negative all along — collects.

#### Worked example: how long a winning edge can look like a loser

Suppose you have a genuine edge: 55% win rate on even-money bets, so an EV of $+\$0.10$ per \$1 risked. Over 100 trades your expected profit is $100 \times \$0.10 = +\$10$ per dollar of unit size. Sounds safe. But the standard deviation of your *total* over 100 bets is about $\sqrt{100} \times \$0.995 \approx \$9.95$. So a one-standard-deviation bad run leaves you at $+\$10 - \$9.95 \approx \$0.05$ — essentially flat — and a two-standard-deviation bad run (which happens roughly 1 time in 40) leaves you *down* about \$9.85 after 100 trades, despite a real edge. You could trade a true 55% edge for a hundred trades and end up in the red, purely by chance, and conclude your strategy is broken when it is fine. One sentence of intuition: a hundred trades is a small sample, and a small sample is mostly noise — never kill a measured edge on a short losing streak, and never trust an unmeasured one on a short winning streak.

There is a brutal asymmetry hiding in that mirror image. The trader with a *real* edge who gets unlucky early and busts is gone — the edge dies with the bankroll. But the trader with *no* edge who gets lucky early does not bust; they get *promoted*. They raise more capital, increase size, and trade longer — which means they trade long enough for their negative edge to finally collect. Variance does not just hide edges; it actively *selects* for over-confidence, because the lucky survive to bet bigger and the unlucky-but-correct disappear. This is why the trading world is full of people with three great years and a catastrophic fourth, and why a track record short enough to be mostly luck is worse than no track record at all — it manufactures false confidence in exactly the people about to blow up.

The defense against the great deceiver is not to predict the swings. It is to (1) measure your edge honestly over a large sample, (2) size small enough that no realistic streak ruins you, and (3) judge results over enough trades that the law of large numbers has something to work with. Notice that none of those three is "get better at forecasting." The house does not win by forecasting; it wins by *process discipline* in the face of noise it has stopped trying to predict. Your edge is a structural fact about a counterparty; your survival is a structural fact about your sizing; and your results, given both, are a structural fact about how many bets you live to make. Which brings us to ruin.

## Risk of ruin: the only number that can end the game

You can have the best edge in the world and it means nothing if you go broke first. Ruin is the absorbing wall: once you hit zero, there is no bouncing back, no "the edge will pay out eventually," because *eventually* requires you to still be at the table. This is why professionals obsess over risk of ruin and amateurs obsess over win rate.

The figure below plots the probability of eventually going broke against how many betting units of bankroll you carry, for three different edges. The single most important thing in it: *the edge is held fixed within each curve.* The only thing changing along each line is how deep your bankroll is. A thin bankroll and a deep one can run the identical winning strategy and meet opposite fates.

![Three risk of ruin curves falling toward zero as bankroll units increase](/imgs/blogs/expected-value-edge-and-variance-thinking-like-the-house-2.png)

Follow the blue curve — a 55% edge ($p = 0.55$), a solid, realistic, professional-grade edge. With a bankroll of just **1 unit** (you bet your whole stack on one even-money flip), your probability of eventual ruin is **81.8%**. With **10 units**, it falls to **13.4%**. With **20 units**, **1.8%**. With **40 units**, about **0.03%**. Same edge, every time. The only thing that moved is how many bets deep your bankroll lets you go. Survival is bought with bankroll depth, not with a better forecast.

#### Worked example: the same edge, two bankrolls, two fates

You and a friend both discover the identical edge: a coin-flip-style bet you win 55% of the time, even money. Your friend, impatient, funds an account with exactly 5 betting units and bets one unit a time. You fund yours with 20 units, same bet size.

Plug into $P_{\text{ruin}} = (q/p)^{u}$ with $p = 0.55$, $q = 0.45$, so $q/p = 0.818$:

- Your friend, 5 units: $0.818^{5} = 0.367$, a **36.7% chance of going broke**.
- You, 20 units: $0.818^{20} = 0.018$, a **1.8% chance of going broke**.

Identical edge. Identical bet. Your friend has more than a one-in-three chance of busting; you have less than one in fifty. The edge did not protect your friend, because the edge cannot pay out to a player who has already hit zero. One sentence of intuition: ruin is decided by how many losing bets in a row your bankroll can swallow, so the cheapest insurance you can buy is a deeper bankroll and a smaller bet.

This is the most under-appreciated idea in all of trading. People spend years hunting for a better signal and zero minutes asking how big their position should be. But position size is what decides whether a real edge ever reaches your account. The next section is about getting that size right.

## Kelly: the optimal compromise between greed and survival

So you have a measured, positive edge, and you understand that betting too big invites ruin while betting too small wastes the edge. How much, exactly, should you bet? The Kelly criterion answers this precisely: it gives the fraction of your bankroll that maximizes the long-run *growth rate* of your money — the rate at which your bankroll compounds over many bets.

The intuition is the tension between two forces. The bigger you bet, the faster you grow *when you win* — but the deeper you fall *when you lose*, and because losses compound just like gains, a big loss takes a bigger win to recover from. (Lose 50% and you need a 100% gain to get back to even.) Kelly finds the bet size where the growth from winning exactly balances the drag from losing, maximizing the compounding rate. Bet more than Kelly and you actually grow *slower* while taking on more ruin risk — the worst of both worlds.

The figure below plots the Kelly fraction against your win probability for an even-money bet. Two features matter. First, at exactly $p = 0.5$ — a fair coin, no edge — Kelly says bet **zero**. No edge, no bet, full stop. Second, the fraction rises smoothly as your edge grows: the better your edge, the more of your bankroll Kelly is willing to commit.

![Kelly fraction rising from zero as win probability increases, with a half Kelly band](/imgs/blogs/expected-value-edge-and-variance-thinking-like-the-house-3.png)

For an even-money bet the formula simplifies beautifully: $f^* = 2p - 1$, which is exactly *twice your edge*. The green dashed line is **half-Kelly**, which is what almost every professional actually bets — and we will see why in a moment.

#### Worked example: Kelly-sizing a 55% edge and a 35% lottery edge

You have the 55% even-money edge ($p = 0.55$, $b = 1$). Kelly says:

$$f^* = \frac{(1)(0.55) - 0.45}{1} = 0.10$$

Bet **10% of your bankroll** per trade. On a \$10,000 account, that is \$1,000 risked per bet. Now suppose your edge is stronger, $p = 0.60$:

$$f^* = \frac{(1)(0.60) - 0.40}{1} = 0.20$$

Now Kelly wants **20%** — double the bet for the stronger edge. But here is the surprise that ties back to our win-rate lesson. Take a *low* win-rate strategy that pays well: you win only 35% of the time, but each win pays 3-to-1 ($b = 3$):

$$f^* = \frac{(3)(0.35) - 0.65}{3} = \frac{1.05 - 0.65}{3} = \frac{0.40}{3} = 0.133$$

A strategy that loses almost two-thirds of the time still gets a healthy **13.3%** Kelly allocation, *larger* than the 55%-win-rate even-money bet — because the big 3-to-1 payoff makes the edge fat. Kelly does not care about your win rate either; it cares about your edge. One sentence of intuition: Kelly sizes your bet to your edge, not your hit rate, and it tells a 35%-winner with a fat payoff to bet *more* than a 55%-winner with a thin one.

Why do pros bet *half* of what Kelly says? Because full Kelly is brutally volatile — it is optimized for growth, not comfort, and it routinely produces 50% drawdowns. Half-Kelly captures about three-quarters of the growth with roughly *half* the volatility and far less ruin risk. And critically, full Kelly assumes you know your edge *exactly*. You never do. If you overestimate your edge — easy to do given variance — full Kelly tips you into the over-betting zone where you grow slower and risk ruin. Half-Kelly is the humility discount for not really knowing your edge. The full treatment, including fractional Kelly and the geometric-growth derivation, is in [Position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion).

The reason over-betting is so dangerous deserves its own moment, because it is the single most common way a person with a genuine edge still goes broke. The trap is that **growth compounds geometrically, not additively.** If you lose 50% of your bankroll, you do not need a 50% gain to recover — you need a 100% gain, because the gain is computed on the smaller base. Lose 75% and you need a 300% gain. This asymmetry means a single large drawdown does disproportionate, lasting damage to your long-run growth rate, and large drawdowns are exactly what over-sized bets produce. Kelly is, precisely, the bet size where the *median* long-run growth is maximized once this asymmetry is accounted for. Past Kelly, the extra size you commit buys you bigger losses that compound against you faster than the bigger wins compound for you, so your long-run growth *falls* even as your risk of ruin *rises*. That is the worst trade in finance: paying more risk to earn less growth. The table makes the regimes concrete for our 55%-edge example, where full Kelly is 10%:

| Bet size | Fraction of bankroll | Long-run growth | Volatility / ruin risk |
|---|---|---|---|
| Under-betting | 2% (one-fifth Kelly) | Low but safe | Very low |
| Half Kelly | 5% | ~75% of the maximum | Moderate, comfortable |
| Full Kelly | 10% | Maximum growth | High; deep drawdowns |
| Over-betting | 20% (double Kelly) | Lower than under-betting | Severe; ruin likely |

Read the bottom two rows together: doubling your bet from full Kelly to 20% does not double your growth — it drives growth *below* the timid 2% better, while sending ruin risk through the roof. There is no upside to over-betting, only a feeling of action. The professional sits at half Kelly not because they are timid but because they are honest about not knowing their exact edge, and half Kelly is the size that survives that uncertainty.

## Putting it together: the EV of one real trade

We have the pieces. Now let us value an actual trade the way a desk does — branch by branch, payoff times probability, summed. This is the bridge from casino math to a live position, and it is where your *edge versus a counterparty* turns into a single number.

A trade is not a coin flip with two outcomes. It is a small tree of possibilities, each with a payoff and a probability. The figure below lays out a realistic one: you risk \$100 to make \$300 (a 3-to-1 reward-to-risk setup), but the position has a tail — sometimes the market gaps through your stop and you lose more than planned.

![Decision tree of a trade with three weighted branches summing to a positive expected value](/imgs/blogs/expected-value-edge-and-variance-thinking-like-the-house-7.png)

#### Worked example: the expected value of a 3-to-1 setup with gap risk

You enter a trade with three outcomes:

- **Target hit (55%):** the trade works, you make $+\$300$.
- **Stop hit cleanly (30%):** it fails as planned, you lose $-\$100$.
- **Gap through the stop (15%):** bad news hits overnight, the price blows past your stop, and you lose $-\$400$ before you can get out.

Weight each branch by its probability and sum:

$$EV = (0.55)(+\$300) + (0.30)(-\$100) + (0.15)(-\$400)$$
$$EV = +\$165 - \$30 - \$60 = +\$75$$

The trade is worth **+\$75 in expectation**, even though it loses outright on 45% of paths and carries an ugly gap tail. The 3-to-1 payoff on the winning branch carries the whole thing. Notice the discipline this imposes: if your win probability were only 40% instead of 55%, the EV would be $(0.40)(300) + (0.45)(-100) + (0.15)(-400) = \$120 - \$45 - \$60 = +\$15$ — still positive, but thin enough that costs and slippage could erase it. And if the gap branch were 30% instead of 15%, the trade would flip negative. The tree forces you to price the tail you would rather ignore. One sentence of intuition: a trade is a probability-weighted tree, and the only honest way to know if it is worth taking is to put a number and a probability on every branch — including the one that hurts.

This is precisely how a market-maker or a poker pro thinks about *every* decision: not "will this win?" but "what is the EV, and can I size it to survive the variance?" The reward-to-risk side of this — why a 3-to-1 payoff lets you be wrong most of the time — is worked in depth in [Risk, reward, and expectancy in practice](/blog/trading/technical-analysis/risk-reward-and-expectancy-in-practice).

## Common misconceptions

**"A high win rate means a good strategy."** No. Win rate ignores the size of wins and losses. A 90%-win-rate strategy that loses \$100 on the 10% and makes \$10 on the 90% has an EV of $-\$1$ per trade — it is a slow-motion bankruptcy with a great-looking track record. Always compute EV, never judge by win rate alone. The cover figure is the entire rebuttal in one chart.

**"I'm up over my last 30 trades, so my edge is proven."** Thirty trades is statistical noise. We saw that a real 55% edge can be *down* after 100 trades roughly 1 time in 40, and a true coin flip can run 8 heads in a row. A short winning streak is just as consistent with zero edge plus good luck as with a real edge. You need a large sample and an honest accounting before you believe any edge — and you should believe a *losing* streak even less, since variance cuts both ways.

**"With a positive edge I can't go broke."** You absolutely can, and people do it constantly. The risk-of-ruin curve showed a real 55%-edge player going broke 81.8% of the time on a 1-unit bankroll and 36.7% on a 5-unit bankroll. A positive edge plus an oversized bet is a recipe for ruin. Edge is necessary for long-run profit but not sufficient for survival — survival requires *sizing*.

**"Bigger bets mean faster growth."** Only up to Kelly. Past the Kelly fraction, bigger bets make you grow *slower* while piling on ruin risk, because compounding punishes big losses asymmetrically — lose 50% and you need a 100% gain just to recover. Over-betting is the single most common way a person with a real edge still blows up. When in doubt, bet *less* than you think you should — half-Kelly exists for exactly this reason.

**"The market is random, so it's all just luck."** The market is *noisy*, which is not the same as random-with-no-edge. A casino game is genuinely random and the house still wins, because it owns a structural edge and survives the variance. Your job is to find a structural edge — an informational, behavioral, or speed advantage over a specific counterparty — and then treat the noise exactly the way the casino does: size small, repeat often, survive. The edge is real even when any single outcome is unpredictable.

**"A losing trade was a bad decision."** This is the single most expensive cognitive error in trading, and it is why firms like SIG teach poker. A *decision* is good or bad based on its EV at the time you made it; an *outcome* is good or bad based on luck. A positive-EV trade that loses was still a good decision — make it again. A negative-EV trade that happened to win was a bad decision that got bailed out by variance — and if you "learn" from it and repeat it, the edge will eventually collect. Judging your decisions by their outcomes, in a world this noisy, trains you to abandon good strategies after unlucky losses and double down on bad ones after lucky wins. Judge the decision by its EV; judge the outcome separately and forgive it. Over a large sample of well-made decisions, the outcomes take care of themselves.

## How it shows up in real markets

**The high-frequency market-maker (the purest house).** Firms like Citadel Securities and Virtu are the clearest real-world casinos. Virtu famously disclosed in its 2014 IPO filing that it had exactly *one* losing trading day in roughly 1,238 days — a streak that is mathematically impossible without a tiny, relentless edge spread across millions of trades and ruthless control over the size of any single swing. They do not predict where Apple stock goes. They capture a sliver of the bid-ask spread on each trade and let the law of large numbers do the rest. They are the house, and your retail order is occasionally a bet at their table. The mechanics of how a market-maker prices the other side of your trade are in [How an options market-maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade).

**Trend-following funds (the 35%-win-rate winners).** Managed-futures funds like those run by Winton or AHL famously win on only 30–40% of their trades. They sit through long strings of small losses — each capped tight by a stop — waiting for the rare, enormous trend that pays for everything. In 2008, while the equity market fell about 38%, many trend-followers had one of their best years ever (some indices up 15–20%) because a few huge moves (crude oil collapsing, government bonds rallying) paid 10-to-1 or more on positions held for months. This is Strategy B from our cover, lived out at fund scale: most trades are small losers, a handful are gigantic winners, and the EV is excellent precisely *because* the rare win is so much larger than the typical loss. Their win rate looks terrible on paper, and an investor who judged them by win rate — or who fired them during a long flat stretch of small losses — would have bailed out right before the payoff. The discipline that lets them survive those flat stretches is exactly the house's discipline: keep making the same positive-EV bet, sized small, and wait for the law of large numbers.

**Long-Term Capital Management (the ruin lesson, written in blood).** LTCM had Nobel laureates, a genuine statistical edge in bond arbitrage, and they nearly took down the global financial system in 1998. Their edge was real but tiny per trade, so they used enormous leverage — reportedly around 25-to-1 on the balance sheet and far more in notional terms — to make it meaningful. That leverage was a bet sized far past Kelly: they had effectively committed many multiples of their bankroll to a strategy whose true variance they had underestimated. When a rare event (Russia's default in August 1998) hit, correlations they had assumed were independent all moved together, the variance they had ignored arrived at once, and a portfolio with a positive long-run edge lost roughly \$4.6 billion in under four months — wiped out before the edge could ever pay. They were the gambler who knew the odds and still over-bet the bankroll. Risk of ruin does not care how smart you are; it cares how much of your stack a single bad streak can take, and theirs could take all of it.

**The 2021 meme-stock option sellers (picking up pennies).** During the 2021 retail-options boom, an army of traders sold far out-of-the-money options for small, frequent premiums — a textbook high-win-rate, negative-tail strategy, just like Strategy A in our cover. They won on the vast majority of trades and felt like geniuses; the steady stream of small premiums looked like free money, and the rare large loss had simply not shown up *yet*. Then in late January 2021 GameStop ran from around \$20 to a peak near \$480 in days, and AMC and others gapped hundreds of percent alongside it. The rare \$100 loss landed against all those \$10 wins at once — and for sellers of naked calls, the loss was not even capped, because a stock that triples has no ceiling on how much the call seller owes. Accounts that had grown steadily for months were wiped out in a single session, some posting losses many multiples of the entire account. Win rate near 95%; EV negative; ruin total. This is the cleanest modern illustration that a beautiful win rate and a deadly negative tail can coexist, and that the tail does not announce itself before it arrives. The gamma-squeeze mechanics that powered those gaps are in [Dealer gamma, charm, and vanna](/blog/trading/options-volatility/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot).

**Susquehanna's poker pipeline (training the house mindset).** Susquehanna International Group (SIG), one of the largest options trading firms in the world, famously teaches new hires *poker* before it lets them trade. The reason is exactly this post: poker forces you to separate the quality of a decision (its EV) from the quality of an outcome (which is mostly variance), to size your bets to your edge, and to fold the hands where you are the sucker. A trader who has internalized that a correct, positive-EV decision can lose — and that the answer is to make it again, sized correctly, not to abandon it — is a trader who thinks like the house. The full story is in [The SIG playbook: poker, game theory, and EV](/blog/trading/quant-careers/sig-susquehanna-playbook-poker-game-theory-and-ev).

## The playbook: how to play it

Here is the strategic frame, in the spine of this series. Every concept above collapses into one operating system.

**Who is on the other side, and what is your edge?** Your edge is never absolute — it exists only relative to a counterparty who is making the opposite bet for a worse reason. Maybe you are faster (the HFT edge), better-informed (the fundamental edge), better-modeled (the quant edge), or simply more disciplined than a panicked crowd (the behavioral edge). If you cannot name *who* is on the other side and *why* they are taking the worse end, you probably do not have an edge — you are the counterparty, and you are the one funding someone else's annuity. The figure below contrasts the two roles you can play at the table.

![Two column comparison of the gambler chasing a big win versus the house grinding a small edge](/imgs/blogs/expected-value-edge-and-variance-thinking-like-the-house-6.png)

**Measure the edge before you trust it.** Do not eyeball your win rate. Compute your expected value over a large sample — payoff times probability, summed, including the ugly tail. If you cannot estimate the probabilities and payoffs honestly, you do not yet have a strategy; you have a hope. And remember the great deceiver: a hundred trades is not enough to confirm an edge, and it is certainly not enough to confirm its absence.

**Size to survive, using Kelly as a ceiling.** Once the edge is measured, compute the Kelly fraction — and then bet *less* than that, typically half. Full Kelly assumes perfect knowledge you do not have; half-Kelly is the discount for your uncertainty about your own edge. Concretely: with a measured 55% even-money edge, full Kelly is 10% of bankroll, so risk about 5% per trade. Keep enough betting units of bankroll behind you that your risk of ruin is negligible — 20-plus units of edge-adjusted bankroll is a reasonable floor, where a unit is your typical per-trade risk.

**Define the invalidation.** Your edge is a hypothesis about a counterparty's behavior. It can stop being true — the counterparty wises up, the inefficiency gets arbitraged away, the regime changes. The invalidation is not a losing streak (variance does that to real edges); it is *evidence the structural reason for your edge has disappeared*. The HFT firm's edge dies when a faster firm arrives, not when it has a red day. Know what would actually break your thesis, separate from normal variance.

**Survive the variance; let the law of large numbers pay you.** This is the whole game. You will have losing weeks and months with a real edge — the bankroll-path figure guarantees it. Your job in those stretches is not to predict the turn or to abandon a sound strategy. It is to stay sized so that no realistic streak ruins you, keep making the same positive-EV decisions, and let the slow drift of the edge grind down the loud noise of variance. Be the house: small edge, many repetitions, deep bankroll, ironclad survival. The gambler asks "will this one win?" The house asks "is my edge positive and am I sized to be here for the thousandth bet?" Only one of them is still at the table when the law of large numbers pays out.

## Further reading and cross-links

- [Expectancy: why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) — the win-rate-versus-edge point worked through a real trading log.
- [Position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) — the full Kelly derivation, fractional Kelly, and sizing in practice.
- [Risk, reward, and expectancy in practice](/blog/trading/technical-analysis/risk-reward-and-expectancy-in-practice) — how a fat reward-to-risk ratio lets you be wrong most of the time and still win.
- [The SIG playbook: poker, game theory, and EV](/blog/trading/quant-careers/sig-susquehanna-playbook-poker-game-theory-and-ev) — how a top trading firm trains the decision-versus-outcome, EV-first mindset through poker.
- [The trade is a game: why markets are strategic, not random](/blog/trading/game-theory/the-trade-is-a-game-why-markets-are-strategic-not-random) — the series spine: your edge exists only relative to a counterparty.
- [Zero-sum, positive-sum, and the house](/blog/trading/game-theory/zero-sum-positive-sum-and-the-house-where-trading-profits-come-from) — where trading profits actually come from, and what it means to be the house rather than the gambler.

*This is educational material about how expected value, variance, and sizing work — not financial advice. Every strategy that can make money can lose it, and position sizing decides which.*
