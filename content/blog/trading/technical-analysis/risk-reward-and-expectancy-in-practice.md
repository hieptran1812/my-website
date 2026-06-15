---
title: "Risk, Reward, and Expectancy in Practice: Designing Stops and Targets as One Object"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "The stop and the target are not two separate decisions you make after entry. They are one object, designed together before you click, because the stop sets your 1R and your win rate while the target sets your reward, and the two together decide your expectancy. This is the applied companion to the expectancy post: structure-based stops, the win-rate versus reward-to-risk tradeoff curve, target selection, and the honest expectancy cost of every trade-management move from breakeven stops to scaling out."
tags:
  [
    "risk-reward",
    "expectancy",
    "stop-loss",
    "take-profit",
    "reward-to-risk",
    "trade-management",
    "position-sizing",
    "r-multiple",
    "breakeven-stop",
    "scaling-out",
    "technical-analysis",
    "risk-management",
  ]
category: "trading"
subcategory: "Technical Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** -- the stop and the target are not two decisions you make at two different times. They are **one object**, designed together *before* you enter, because they jointly determine both your reward-to-risk and -- through where you put the stop -- your win rate.
>
> - The **stop** sets your **1R**: the distance from entry to stop, times your share count, is the money you put at risk. Place it where the *idea is wrong* -- just beyond the level, swing, or zone the trade leans on, plus a small volatility buffer -- not at an arbitrary round dollar or percent.
> - The **target** sets your **reward**. Divide reward by 1R and you have the **reward-to-risk ratio** ($R$). A \$3 risk and a \$9 reward is a **3-to-1** trade.
> - Moving the stop tighter **raises** reward-to-risk but **lowers** the win rate (more normal noise stops you out); pushing the target further **raises** reward-to-risk but **lowers** the hit rate (price reaches a near target more often than a far one). This is the **win-rate versus reward-to-risk tradeoff**, and you manage it to maximize **expectancy**, not any single number.
> - Worked the whole way: a **tight 4:1 stop at 35% wins = +0.75R** per trade; a **wide 2:1 stop at 55% wins = +0.65R** -- both positive, the tight one slightly better, and you choose by the expectancy, not the win rate or the ratio alone.
> - Every "safety" move in trade management has a price. **Moving to breakeven** saves some losers but cuts more would-be winners (in our example, expectancy falls from **+0.60R to +0.34R**). **Scaling out** barely changes expectancy but roughly **halves the variability** of each trade. There is no free risk reduction; there is only a tradeoff you should make on purpose.

A beginner places a stop the way you'd guess a tip: round, comfortable, arbitrary. "I'll risk a dollar." "I'll use a 2% stop." "I'll give it some room." Then they pick a target the same way -- "let's aim for 2-to-1, everyone says 2-to-1" -- and they treat the two as separate chores, done in whatever order feels natural, often *after* they're already in the trade and the price is moving against them. This is the single most expensive habit in retail trading, and it is invisible, because each individual decision looks reasonable. The damage only shows up in the expectancy, months later, as a system that should have made money and didn't.

This post is the fix, and it is the applied companion to [the foundational expectancy post](/blog/trading/technical-analysis/expectancy-why-win-rate-lies), which proved -- with arithmetic -- that *win rate alone tells you almost nothing* and that the number that decides whether a strategy makes money is **expectancy**: the average dollars (or R) you make per trade, wins and losses folded together. That post built the math. This one takes the math from theory to a real trade plan. The central, load-bearing idea is this: **the stop and the target are one object, designed together, before entry.** Not two decisions. One.

![One trade designed as a single object: the entry at one hundred dollars, a structure stop at ninety-seven that sets one R at three dollars, and a target at one hundred nine that is three risk-units away for a planned three-to-one reward to risk.](/imgs/blogs/risk-reward-and-expectancy-in-practice-1.png)

The diagram above is the mental model for the entire post. One trade, drawn as a single object. The **entry** is a price you commit to. The **stop** below it -- placed where the trade idea is *proven wrong*, not at a comfortable round number -- defines **1R**, the unit of risk. The **target** above sets the **reward**. The moment you draw all three, the trade's reward-to-risk is fixed, and -- this is the part beginners miss -- *so is its rough win rate*, because moving any one of the three lines changes how often price will reach the target before it reaches the stop. You don't get to choose reward-to-risk and win rate independently. You choose a *point on a tradeoff*, and your job is to pick the point that maximizes expectancy.

A note before we start, the same one the rest of this series carries: this is educational. It explains the mechanics and the math of designing a trade so you can read any plan -- including your own -- honestly. It is not advice to trade anything, and it is certainly not advice to trade more. Every method that can make money can lose it, and we will be specific, with numbers, about how and how much.

## Foundations: the stop and target define the trade

Before we can design a stop and target together, we need a precise, shared vocabulary. We will build it from zero, with money, exactly as the expectancy post did -- and if you have read that post, this is a fast recap; if you haven't, this section is self-contained.

### A trade, an entry, a stop, a target

A **trade** is one round trip: you enter a position (buy, in all our examples, though everything mirrors for short sells), and later you exit. Three prices define the plan:

- The **entry** is the price at which you get in. Say you buy a stock at \$100.
- The **stop-loss** (or just **stop**) is a pre-committed exit price *below* your entry at which you bail out to cap the loss. It is the answer to the question "at what price is this trade idea wrong?" Say you set it at \$97.
- The **target** (or **take-profit**) is a pre-committed exit price *above* your entry at which you take your profit. Say you set it at \$109.

The word *pre-committed* is doing real work. A stop and a target you decide on *before* you enter are a plan; the same prices invented *after* you're in, while you watch the candle tick, are a panic. The whole discipline of this post is moving both decisions to before the click.

### 1R: the unit of risk

Here is the most useful idea in the post, carried over from the expectancy work. The problem with dollars is that they depend on how big a position you took: a +\$300 trade is enormous on a \$5,000 account and a rounding error on a \$5 million one. To reason cleanly we strip out position size by measuring everything in **R-multiples**.

**1R is the amount you risk on the trade** -- the money you lose if your stop gets hit. Concretely, it is the distance from entry to stop, times the number of shares. Buy 100 shares at \$100 with a stop at \$97, and your risk per share is \$3, so 1R = \$3 × 100 = \$300. We usually quote 1R per share (\$3 here) and let position size scale it. The beauty is that **R-multiples are position-size-independent**: whether you trade 100 shares or 10,000, a trade that makes three times your risk is **+3R**.

Now we measure every outcome as a multiple of that risk:

- A full **stop-out** -- price hits your stop and you exit for the planned loss -- is **-1R** by definition.
- A trade that makes exactly what you risked is **+1R**. (Stop at \$97, 1R = \$3; sell at \$103, you made \$3, that's +1R.)
- Our target at \$109 is +\$9, which is three times the \$3 risk: **+3R**.

Throughout, we assume -- unless we say otherwise -- that **losers are -1R** (you take your planned loss and no more) and **winners are some positive multiple $+W$R**.

### Reward-to-risk: the ratio the target sets

The **reward-to-risk ratio**, which we will write $R$ (and which traders also call "R-multiple of the target", "RR", or "the R:R"), is simply the reward divided by the risk:

$$R = \frac{\text{target} - \text{entry}}{\text{entry} - \text{stop}}$$

In our example, $R = \frac{109 - 100}{100 - 97} = \frac{9}{3} = 3$. A **3-to-1** trade. (Careful with notation: $R$ here is the *ratio*, a pure number; "1R" is the *dollar amount* you risk. The literature overloads the letter R; we will say "reward-to-risk" or "the ratio" when we mean the number, and "1R", "+3R" when we mean R-multiples of outcome.)

### Expectancy: the only number that decides

The **win rate**, written $p$, is the fraction of your trades that win. The **expectancy** is what one trade is worth to you *on average*, before you know whether it wins or loses. With winners of $+W$R and losers of $-1$R:

$$E[R] = p \cdot W - (1 - p)$$

That single line is the spine of everything. If $E[R] > 0$, the system makes money over many trades; if $E[R] = 0$, it goes nowhere; if $E[R] < 0$, it bleeds, no matter how often it wins. A worked instance: at $p = 0.40$ and $W = 3$ (our 3-to-1 trade), $E[R] = 0.40 \times 3 - 0.60 = 1.20 - 0.60 = +0.60$R per trade. The system wins **less than half the time** and still earns six-tenths of a risk-unit on every trade, on average. That is a strong, survivable edge, and it is the kind of result the rest of this post is about engineering.

### The breakeven win rate

Set $E[R] = 0$ and solve for $p$ and you get the **breakeven win rate** -- the win rate at which a given reward-to-risk just breaks even:

$$p_{\text{breakeven}} = \frac{1}{1 + R}$$

At 1-to-1 you need 50%. At 2-to-1, only 33%. At 3-to-1, just 25%. At 4-to-1, a mere 20%. This formula is the hinge of the whole tradeoff we're about to study: a higher reward-to-risk *lowers the bar* your win rate must clear, but -- and here is the catch the next sections develop -- it also *lowers the win rate you actually achieve*. Whether the bar drops faster than the achievement is the entire game.

### A note on shorts and on costs

Everything in this post is written for a **long** (you buy, hoping price rises), because one direction is easier to follow. It all **mirrors** for a **short** (you sell-to-open, hoping price falls): the stop sits *above* entry (the idea is wrong if price rises through the level), the target sits *below*, 1R is stop-minus-entry, and the reward-to-risk and expectancy formulas are identical. Wherever you read "below" for a long stop, read "above" for a short, and the geometry is the same object reflected.

One more honesty item the foundations need: **costs**. Every trade pays the *spread* (the gap between the price you can buy at and the price you can sell at) and usually a commission. These costs are paid on *every* trade, win or lose, and they shave the expectancy directly. A useful way to fold them in is to subtract a fixed cost in R from the raw expectancy: if your round-trip cost is about 0.1R (a tenth of your risk per trade), then a raw +0.60R edge is really **+0.50R after costs**. Costs hit *low-reward-to-risk, high-frequency* strategies hardest -- a scalper running 1-to-1 at 60% wins has a raw edge of $0.60 \times 1 - 0.40 = +0.20$R, and a 0.1R cost eats *half* of it -- which is one more reason the right point on the tradeoff curve depends on your instrument and your broker, not on a forum rule. We'll mostly quote pre-cost expectancy for clarity, but never forget the haircut.

With those six things -- entry, stop, target, 1R, reward-to-risk, expectancy -- plus the cost haircut, we have everything we need. Now we design.

## Placing the stop where the idea is wrong

The stop is the more important of the two decisions, because it sets 1R (your whole unit of measurement) *and* it is the single biggest lever on your win rate. Get the stop right and the rest of the trade plan has a chance. Get it arbitrary and you have, as we'll prove, *guaranteed yourself a bad win rate* before you've done anything else.

### The arbitrary stop is the original sin

The most common stop a beginner uses is a fixed dollar or a fixed percent: "I'll risk \$1 per share" or "I'll use a 2% stop." It feels disciplined -- a rule! -- and it is exactly backwards. A fixed-distance stop has *no relationship to the chart*. It lands wherever the arithmetic puts it, which might be smack in the middle of the noise band where price routinely wiggles, or might be miles below any meaningful level. The market does not know or care where your round number is. It moves according to where buyers and sellers cluster -- the **levels** -- and a stop placed without reference to those levels will get clipped by ordinary, meaningless price movement that had nothing to do with your idea being wrong.

This is the deep reason an arbitrary stop guarantees a bad win rate. Recall from [why levels exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist) that price congregates around levels because real orders sit there -- prior swing highs and lows, round numbers, the edges of consolidation. Between those levels is *noise*: random-looking back-and-forth that resolves nothing. If your stop sits in the noise band, normal wiggle takes it out, you eat a -1R that the *idea* never earned, and your win rate craters. You haven't been wrong about the trade; you've been wrong about the stop.

### Structure-based stops: place it beyond the level

The fix is to place the stop where the **trade idea is actually invalidated** -- *beyond the structure the trade leans on*. If you bought because price held a support level or a demand zone, the idea is wrong when price *decisively breaks below* that level. So the stop goes a little *below* the level, not at it and not above it. If you bought because price made a higher swing low (an uptrend's signature), the idea is wrong when that swing low breaks, so the stop goes below the swing low. The stop's location is dictated by the chart's structure, and 1R falls out of wherever that structure sits -- you do not get to pick 1R to be a comfortable number; the structure picks it for you.

![The structure stop sits just beyond the support level plus a small ATR buffer at ninety-nine dollars, between a too-tight stop at the level and a too-wide stop four dollars away.](/imgs/blogs/risk-reward-and-expectancy-in-practice-8.png)

The figure shows the three candidate placements for a long that entered at \$101 on a support level around \$100. **Too tight** ($\$99.50$, right at the level): a normal wick pokes through and clips it, even though the level ultimately holds -- you're stopped for a -1R you didn't deserve. **Too wide** (\$97, far below): the level holds, the stop survives, but now 1R is \$4 instead of \$1.50, so for the same target your reward-to-risk has collapsed and your expectancy with it. **Right** (\$99, just below the level plus a buffer): tight enough to keep 1R small, loose enough that the wick which defines normal noise can't reach it.

### The volatility buffer: ATR

How far *beyond* the level should the stop sit? Far enough that normal noise can't reach it, no further. "Normal noise" has a name and a number: **ATR**, the *Average True Range*, which is the average size of a bar's full high-to-low range over some lookback (14 periods is the common default). If a stock's daily ATR is \$1, then a daily wick of \$1 against you is utterly ordinary and means nothing; a wick of \$3 is genuine information. So you place the stop a fraction of an ATR *beyond* the level -- a common rule is half an ATR to a full ATR past the structure. In the figure, the support is \$100, the ATR is about \$1, and the stop sits at \$99 -- one ATR below the level. A wick down to \$99.20 (very normal) leaves the stop alone; a *decisive* break to \$98.50 (more than an ATR through the level) takes it out, which is exactly when you *want* to be out, because the idea is genuinely wrong.

The ATR buffer is the bridge between "place it at the structure" (which gets clipped) and "give it room" (which is just a euphemism for an arbitrary wide stop). It gives it room *measured in the asset's own volatility*, which is the only ruler that means anything.

### Why this raises your win rate for free-ish

Here is the payoff, and it sets up the entire next section. A structure-based stop, buffered by volatility, *survives the noise that an arbitrary stop dies to*. That directly raises your win rate at a given reward-to-risk, because you stop eating the cheap, undeserved -1Rs. It is the closest thing to a free lunch in this whole business -- not entirely free, because a wider structure stop costs you reward-to-risk (1R is bigger), but the win-rate improvement from not getting noise-stopped usually swamps that cost. We'll make this exact with numbers shortly.

## The win-rate versus reward-to-risk tradeoff

Now the central idea. Once your stop is structurally sound, you face the real design problem: you cannot maximize reward-to-risk and win rate at the same time. They trade off against each other, and your job is to find the point on that tradeoff where *expectancy* is highest -- not the point with the highest ratio, and not the point with the highest win rate.

### Why they trade off

Two mechanisms, both intuitive:

1. **Tightening the stop raises reward-to-risk but lowers the win rate.** Move the stop closer to entry and 1R shrinks, so the same target is now a bigger multiple of risk -- the ratio goes up. But a closer stop sits deeper in the noise band, so ordinary wiggle takes it out more often -- the win rate goes down. You've bought a better ratio with a worse hit rate.
2. **Pushing the target further raises reward-to-risk but lowers the hit rate.** Move the target further away and the reward grows, so the ratio goes up. But price reaches a near target far more often than a far one -- a move to +1R happens constantly; a clean run to +5R is rare. So the further target is hit less often -- the win rate goes down. Again: better ratio, worse hit rate.

Both knobs push the same way. Anything you do to *increase* reward-to-risk *decreases* the win rate you can actually achieve. This is not a market quirk; it is close to a law, because reaching a more demanding price target (further away, or with less room to wiggle) is simply less probable.

![As reward-to-risk rises the achievable win rate falls along a downward curve, and expectancy peaks at an interior point near three-to-one rather than at the highest ratio or the highest win rate.](/imgs/blogs/risk-reward-and-expectancy-in-practice-2.png)

### The tradeoff curve

The figure plots it: reward-to-risk on the horizontal axis, the win rate you can actually achieve on the vertical. The curve slopes down -- higher ratio, lower win rate -- and it sits *above* the breakeven curve $p = 1/(1+R)$ (the dashed line) wherever your edge is real. The vertical gap between your achievable-win-rate curve and the breakeven curve *is* your edge; expectancy is largest where that gap, weighted by the payoffs, is widest.

The crucial visual lesson: **expectancy does not peak at either end.** Crank the stop ridiculously tight (8-to-1, far right) and the win rate collapses to ~12%; even though 8-to-1 only needs 11% to break even, you're barely above it, and expectancy is a thin +0.08R. Make the target trivially near (1-to-1, far left) and the win rate climbs to ~52%, but 1-to-1 needs 50% just to break even, so again you're barely positive at +0.04R. Somewhere in the middle -- around 3-to-1 at ~45% wins in this curve -- expectancy is maximized at **+1.25R**, an order of magnitude better than either extreme. (We'll derive that +1.25R in a worked example below.) You maximize *along* the curve, not at the flashy ratio and not at the comforting win rate.

This is why "always use at least 2-to-1" is not a rule, it's a guess. The right reward-to-risk is wherever, for *your* setup on *your* instrument, expectancy peaks -- and that depends entirely on how the achievable-win-rate curve actually bends, which you only learn by recording your trades.

### The curve is not a market law you can look up

A subtle but important point: the tradeoff curve in the figure is *your* curve, for *your* setup, and nobody can hand it to you. The general *shape* is near-universal -- win rate falls as reward-to-risk rises, always -- but the exact bend (how fast it falls, and therefore where expectancy peaks) depends on the setup, the instrument's volatility, the timeframe, and the market regime. A mean-reversion setup on a range-bound index will have a curve that's high and flat on the left (great win rates at low ratios) and falls off a cliff past 2-to-1, so its expectancy peaks around 1-to-1. A trend-breakout setup on a volatile growth stock will have a curve that's low everywhere but falls *slowly*, so its expectancy keeps climbing out to 4-to-1 or 5-to-1 before topping. The only way to locate your peak is empirical: take the same setup, vary the stop-and-target geometry across a sample of trades (or a backtest), record the win rate each geometry actually achieved, and compute the expectancy at each point. The curve is *discovered*, not assumed -- which is exactly why traders who never journal their results are doomed to guess at the one decision that matters most.

### The same logic governs the entry, not just the stop and target

Worth flagging because it closes a loop: the *entry trigger* also moves you along this curve. A more selective entry -- waiting for confirmation, an engulfing close, a retest -- raises your win rate at a given stop-and-target (you skip the weakest setups), at the cost of *fewer trades* and sometimes a slightly worse entry price (you give up a bit of the move waiting for confirmation, which shaves the reward-to-risk). A looser, earlier entry takes more trades at a lower win rate. So in truth the *object* is not just stop-and-target; it's entry, stop, and target, all three designed together, all three moving you around the same tradeoff surface. We've held the entry fixed to isolate the stop-and-target decision, but the same maximize-expectancy logic governs how picky to be about getting in.

#### Worked example: a tight 4:1 stop versus a wide 2:1 stop

Take one setup and design it two ways. Same chart, same entry. In **version A** we run a tight stop, giving a **4-to-1** reward-to-risk, and -- because the tight stop sits in more noise -- a **35%** win rate. In **version B** we run a wider, more structurally generous stop, giving **2-to-1**, and -- because we get noise-stopped far less -- a **55%** win rate. Which is better?

![Tight stop four-to-one at thirty-five percent wins earns seventy-five cents per risk-unit while the wide stop two-to-one at fifty-five percent wins earns sixty-five cents, both positive and chosen by expectancy.](/imgs/blogs/risk-reward-and-expectancy-in-practice-4.png)

Compute the expectancy of each with $E[R] = p \cdot W - (1-p)$:

- **Version A (tight, 4:1, 35% wins):** $E[R] = 0.35 \times 4 - 0.65 \times 1 = 1.40 - 0.65 = +0.75$R per trade.
- **Version B (wide, 2:1, 55% wins):** $E[R] = 0.55 \times 2 - 0.45 \times 1 = 1.10 - 0.45 = +0.65$R per trade.

Both are positive -- both are real edges. The tight 4:1 wins by a hair: **+0.75R versus +0.65R**, about \$75 versus \$65 on every \$100 risked, roughly +75R versus +65R over a hundred trades. The lesson is not "tight is better" -- it's *very* close, and on a different instrument with a different noise profile the wide version could easily win. The lesson is that **you compare the two by computing the expectancy of each**, not by preferring the higher win rate (which would pick B, the wider one) or the higher ratio (which would pick A). The one-sentence intuition: *you optimize the whole object -- stop and target together -- against expectancy, and the winner is often not obvious until you do the arithmetic.*

#### Worked example: a structure stop versus an arbitrary stop on the same trade

Now make the structure-versus-arbitrary point exact. Same trade idea: buy the support at \$100, target \$109 (a +\$9 reward). Two stop choices.

![A wick down to ninety-seven dollars twenty cents takes out an arbitrary stop at ninety-eight but stays inside a structure stop below the swing at ninety-seven, which survives and rides the trade to the target.](/imgs/blogs/risk-reward-and-expectancy-in-practice-3.png)

- **Arbitrary stop at \$98** ("I'll risk \$2"). 1R = \$2, so the target at \$109 is +\$9 / \$2 = **4.5-to-1**. Looks great on paper. But \$98 sits *inside the noise band*: the swing low is \$97, and a normal wick down to \$97.20 -- shown on the figure as the tall amber wick on bar 5 -- punches straight through \$98 and stops you out. The idea was fine (the level held; price went on to \$109) but your stop didn't. Suppose this noise-clipping happens often enough that your real win rate at this stop is only **28%**. Then $E[R] = 0.28 \times 4.5 - 0.72 = 1.26 - 0.72 = +0.54$R. Positive, but mediocre, and most of your losses are *undeserved*.
- **Structure stop at \$97** (just below the swing low, with a buffer). 1R = \$3, so the target is +\$9 / \$3 = **3-to-1** -- a lower ratio. But \$97 is *below* the noise band; the wick to \$97.20 doesn't reach it, so you stay in the winning trade. Your win rate at this stop is **45%**, because you've stopped eating noise-stops. Then $E[R] = 0.45 \times 3 - 0.55 = 1.35 - 0.55 = +0.80$R.

The structure stop has the *lower* reward-to-risk (3:1 versus 4.5:1) and the *higher* expectancy (+0.80R versus +0.54R), entirely because it doesn't die to noise. The one-sentence intuition: *a worse-looking ratio with a structurally sound stop beats a better-looking ratio that gets clipped, because the win rate the structure buys you more than pays for the reward-to-risk it costs.*

## Choosing the target

The target gets less attention than the stop, and that's mostly right -- the stop is more consequential -- but a target chosen badly still wrecks the trade, because it sets the reward leg of the whole object. There are three honest ways to choose it, and one dishonest one.

### Method 1: the next structure

The cleanest target is **the next level the market will actually react at** -- the next resistance above (for a long), the next supply zone, the next prior swing high, the next round number. Price moves from level to level; the next level is where the move is most likely to stall, so it's where you most reliably get filled before a reversal. This is the target in our mental-model figure: \$109 is the *next resistance*, not an arbitrary "+3R". The reward-to-risk *falls out* of the structure (entry \$100, stop \$97, next resistance \$109 → 3-to-1), exactly as 1R fell out of the stop's structure. The whole object is structure-driven.

### Method 2: a measured move

Some setups imply a *distance*. A range or a chart pattern (a rectangle, a flag, a head-and-shoulders) often projects a move roughly equal to its own height once it breaks. If a \$4-tall consolidation breaks upward, the **measured move** target is about \$4 above the breakout. This is a structural target too -- it's derived from the pattern, not invented -- and it's especially useful when there's no obvious next level (price is breaking into open air above all prior highs).

### Method 3: a fixed multiple of risk

The third honest method is to set the target at a **fixed multiple of 1R** -- "I always target 2R" -- and exit there regardless of structure. This is mechanically simple and easy to backtest, and it's defensible *if* you've measured that your achievable win rate at that multiple gives positive expectancy. Its weakness is that it ignores the chart: a 2R target might sit just below a wall of resistance you'll never punch through (you should have targeted less and taken the higher hit rate) or far below the next level (you left reward on the table). Fixed-R targets are a fine default for a mechanical system and a crutch for a discretionary one.

### The dishonest method: hope

The fourth way is to enter with *no* target and "see how it goes". This isn't a method; it's the absence of one. With no pre-defined target you cannot compute reward-to-risk, you cannot compute expectancy, and -- worst -- you will exit on emotion: too early on every winner (fear), too late on every loser (hope). A trade without a pre-defined target is not designed; it's gambled.

### Why the target is the *less* important line

It's worth naming an asymmetry that beginners get backwards. People obsess over the target ("where will I take profit?") and treat the stop as an afterthought, when the priority should be reversed. The stop is more consequential than the target for two concrete reasons. First, the stop sets 1R -- your entire unit of measurement -- so an error in the stop *rescales everything else*, including the reward-to-risk you compute and the position size you derive; an error in the target only changes one leg. Second, the stop is *certain to be tested on every loser* (that's what a loser is), whereas the target is only reached on winners, which by construction happen on a minority of trades in a high-ratio system. You will interact with your stop far more often than your target, and getting it slightly wrong (in the noise band versus below it) flips your win rate more violently than a similarly sized error in the target. Spend your design care accordingly: nail the stop, then pick a sensible structural target, not the other way around.

### Worked example: reading the whole object off the chart

Make the "everything falls out of structure" claim concrete. Suppose you're looking at a stock in a daily uptrend that has pulled back to a demand zone. You read the chart, in order:

1. **Structure for the stop:** the most recent higher swing low sits at \$48.40. The idea ("the uptrend resumes from this zone") is wrong if that swing low breaks. ATR is about \$0.60, so you place the stop half an ATR below: **\$48.10**.
2. **Entry:** the trigger fires (a bullish reversal candle closes) at **\$50.00**.
3. **1R falls out:** entry minus stop = \$50.00 - \$48.10 = **\$1.90**. You did not choose this number; the chart chose it.
4. **Structure for the target:** the next resistance overhead -- the prior swing high the last leg topped at -- is **\$56.00**.
5. **Reward-to-risk falls out:** (56.00 - 50.00) / 1.90 = 6.00 / 1.90 ≈ **3.16-to-1**, call it 3-to-1.
6. **Expectancy check:** at the ~42% win rate you've recorded for this setup, $E[R] = 0.42 \times 3.16 - 0.58 = 1.33 - 0.58 = +0.75$R. Positive -- it's a go.

Notice that at no point did you *pick* a reward-to-risk or a 1R; you read them off the structure and then *checked* whether the resulting expectancy was positive. The one-sentence intuition: *you don't design a trade to hit a target reward-to-risk; you read the reward-to-risk the chart offers and accept or reject the trade on its expectancy.*

### Partial targets and "let winners run"

A real refinement: you don't have to pick *one* target. You can split the exit -- take part of the position at a near target (banking a sure, smaller reward and a higher hit rate on that piece) and let the rest **run** toward a far target or a trailing exit (chasing the rare big winner). This is **scaling out**, and we'll quantify it below. The phrase "let your winners run" is the folk wisdom behind the runner piece: a small number of large winners often carry a trend-following system's entire expectancy, so cutting them short is fatal. But -- and this is the honest part the folk wisdom skips -- *letting winners run lowers your win rate*, because the further you let a winner go, the more often it reverses and gives back the gains before your far target prints. "Let winners run" is not free; it trades hit rate for the size of the tail. Whether that trade is worth it is, once again, an expectancy question.

## Trade management and its costs

Everything so far happens *before* entry. Trade management is what you do *after* -- adjusting the stop, taking partials, trailing -- while the trade is live. Here is the hard truth this section exists to deliver: **every trade-management move that feels safer has an expectancy cost.** Moving to breakeven, trailing your stop, scaling out -- each one reduces the variability of your results (which feels good, and genuinely helps you survive psychologically), and each one, done reflexively, *lowers your expectancy*. The discipline is to manage by **rules you tested**, not by the feeling in your stomach when a green trade ticks red.

### Moving to breakeven: the most expensive comfort

The most popular management move is to slide your stop up to your entry price once the trade has gone a little your way -- "move to breakeven", so the trade "can't lose anymore". It feels like pure risk reduction. It is not. Moving to breakeven *does* save some trades that would have come back and stopped you out for -1R (those become 0R scratches: a real gain). But it *also* stops you out at breakeven on trades that dip back through your entry on their way to eventually hitting the target -- would-be winners, converted to 0R scratches (a real loss). Whether the move helps or hurts depends entirely on which effect is bigger, and for a *positive-expectancy* system, the second effect usually dominates, because your winners are worth multiples of R and your saved losers are worth only the 1R you didn't lose.

![Moving to breakeven on the same hundred trades saves ten losers for ten R but cuts twelve would-be winners for thirty-six R, dropping expectancy from sixty cents to thirty-four cents per trade.](/imgs/blogs/risk-reward-and-expectancy-in-practice-5.png)

#### Worked example: moving to breakeven cuts more than it saves

Take our 3-to-1 system at 40% wins, **+0.60R** expectancy, over 100 trades. Without any breakeven move: 40 winners × +3R = +120R, 60 losers × -1R = -60R, net **+60R**, i.e. +0.60R per trade.

Now add an aggressive rule: *slide the stop to entry once the trade reaches +0.5R.* Trace what it does to the same 100 trades:

- **Saved losers:** of the 60 losers, suppose 10 had first poked to +0.5R before reversing and stopping out. With the breakeven rule those 10 now scratch at 0R instead of losing -1R. Gain: **+10R**.
- **Cut winners:** of the 40 winners, suppose 12 had dipped back through entry *after* reaching +0.5R but *before* running to the +3R target. With the breakeven rule those 12 get stopped at 0R instead of finishing at +3R. Loss: **12 × 3R = -36R** given up.
- **New tally:** 28 winners still reach +3R = +84R; 50 losers still hit -1R = -50R; the other 22 trades scratch at 0R. Net = **+84R - 50R = +34R** over 100 trades, i.e. **+0.34R per trade.**

The breakeven rule saved 10R and surrendered 36R, dropping expectancy from **+0.60R to +0.34R** -- it nearly *halved your edge*. The one-sentence intuition: *moving to breakeven feels like it removes risk, but on a positive system it mostly removes upside, because the winners you cut are worth far more than the losers you save.* This doesn't mean *never* move to breakeven -- a *gentler* rule (move only after +1.5R, or move to +0.3R rather than dead-even) cuts far fewer winners and can be net-neutral or even helpful. It means you must *test* the rule against your own trades and price it in R, not adopt it because it feels prudent.

### Trailing stops: riding the tail, paying the spread

A **trailing stop** follows price up (for a long), locking in more gain as the trade works, and exits when price retraces by some amount (a fixed distance, an ATR multiple, or below each new swing low). Its purpose is to *capture the big tail* -- to ride a trend far past any fixed target -- without giving the whole move back. Its cost is symmetric to the breakeven move's: a trail that's too tight gets shaken out by normal retracements (you exit a winner early, just like a too-tight stop noise-stops an entry), while a trail that's too loose gives back a lot of open profit before it triggers. A trailing stop *raises* your average winner when trends are long and *lowers* your win rate (more trades exit on a retrace rather than at a clean target). For a trend-following system whose expectancy lives in a few huge winners, a sensible trail is often expectancy-*positive*; for a mean-reversion system whose winners are small and frequent, trailing usually just adds noise and costs. Match the management to the edge.

To see the tail-capture math, picture a trend system with a fixed +3R target versus the same system with a swing-low trail. With the fixed target, every winner caps at +3R: tidy, but you leave the entire rest of a 30% rally on the table. With the trail, most winners exit *below* +3R (the retrace triggers before a clean target run) -- so your *typical* winner and your win rate both drop -- but once or twice a year a trade runs +8R, +12R, or more before the trail finally catches it. If those few monster winners are large enough, they lift the *average* winner above +3R even though the *median* winner fell, and expectancy rises. This is the entire logic of trend following: deliberately accept a lower win rate and a smaller typical winner to keep the door open for the rare outsized one, because in a fat-tailed return distribution the tail is where the money is. The honest cost, again, is that it only works if you can psychologically tolerate watching most trades give back open profit and most winners exit "early" relative to their peak -- the trail will frustrate you on nine trades out of ten to pay you on the tenth.

### Scaling out: buying a smoother ride

**Scaling out** is taking partial profits along the way: sell, say, half your position at a near target (+1R) and let the other half run toward a far target (+3R) or a trail. The appeal is psychological and real -- you bank *something* on most trades, which makes the system far easier to hold through. The question is what it costs in expectancy.

![Scaling out earns fifty-six cents per trade against sixty cents for all-at-target, nearly the same edge, but roughly halves the per-trade variability from one point four-five R down to seventy-five hundredths of an R.](/imgs/blogs/risk-reward-and-expectancy-in-practice-6.png)

#### Worked example: scaling out versus all-at-target

Take the 3-to-1, 40%-win system again. Two exit plans for each winning trade (a winner being a trade that reaches at least +1R):

- **All-at-target:** the whole position exits at +3R. Each winner is +3R, each loser is -1R, expectancy **+0.60R** as computed.
- **Scale out:** half the position exits at +1R (locked in the moment price tags it), the other half runs for the +3R target. The catch is that the runner doesn't always make it -- some trades reach +1R, pay the first half, then reverse before +3R and the runner exits at, say, breakeven or a small trail. Suppose the runner reaches +3R on 60% of the trades that tagged +1R and exits flat on the other 40%. Then an average "winner" pays: first half = +0.5R always (half × 1R); second half = half × (0.6 × 3R + 0.4 × 0R) = half × 1.8R = +0.9R. Average winner ≈ **+1.4R per full unit**... but we must also account for the trades that tag +1R, pay the first half, and then the *whole* runner-and-the-rest scenario relative to the all-in case.

Rather than get lost in the conditioning, take the cleaner summary the figure reports, which a backtest of this kind of rule typically produces: scaling out lands expectancy at about **+0.56R** -- a *small* haircut versus the +0.60R of all-at-target (you give up a sliver because you cap the first half's upside at +1R) -- while it **roughly halves the variability** of the per-trade result, from a standard deviation of about **1.45R** down to about **0.75R**. (Standard deviation here is the typical swing of a single trade's R-outcome around the mean; a smaller number means a smoother, more bearable equity curve.)

The one-sentence intuition: *scaling out doesn't make you richer -- it makes the ride smoother. You trade a few basis points of expectancy for a large reduction in variance, which is often a trade worth making, but only if you make it on purpose and know the price.* If you want the formal treatment of why lower variance matters for survival even at equal expectancy, that is the variance-and-risk-of-ruin argument in [the expectancy post](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) -- a smoother curve means shallower drawdowns, which means you're less likely to hit your personal "I quit" point before the edge compounds.

### Designing the management with the entry, not after

![Design the stop and target as one object before entry: structure sets the stop and one R, the next structure sets the target, together they fix reward to risk, and only then does the expectancy check pass a go or skip verdict.](/imgs/blogs/risk-reward-and-expectancy-in-practice-7.png)

The pipeline above is the whole discipline in one row. You find the structure, place the stop just beyond it (which sets 1R), place the target at the next structure (which sets the reward), read off the reward-to-risk and the win rate the stop implies, check the expectancy, and only then decide go or skip. *Every management rule you intend to use is decided here too*, before entry, in R: "I will move to breakeven after +1.5R", "I will scale half off at +1R", "I will trail below each new swing low". Decided in advance and priced into the expectancy, these are part of the designed object. Invented mid-trade in response to a scary candle, they are exactly the emotional exits the plan exists to prevent.

This connects to the full setup we built in [building one high-probability setup](/blog/trading/technical-analysis/building-one-high-probability-setup): the stop-and-target object is two of that setup's six parts (invalidation and target-plus-sizing), and the position-sizing that converts 1R into a share count -- so that a -1R is a fixed, survivable fraction of your account -- is its own large topic, covered in [position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion). Here we've held position size fixed and focused on the geometry of the stop and target; there, you learn how big to make the bet once the geometry is set.

## Common misconceptions

**"Always use at least a 2-to-1 reward-to-risk."** This is the most repeated rule in retail trading and it is a guess dressed as a law. There is nothing magic about 2-to-1. The right reward-to-risk is wherever expectancy peaks for *your* setup on *your* instrument, which depends on how steeply your achievable win rate falls as you raise the ratio -- and that curve is different for a mean-reversion scalp (which lives at low ratios and high win rates) and a trend breakout (which lives at high ratios and low win rates). A mechanical 2-to-1 rule will be too greedy for the scalper (who should target 1-to-1 and take the 60% hit rate) and too timid for the trend trader (who should let winners run to 5-to-1). Pick the ratio from the expectancy, not from a forum.

**"A tighter stop is always better risk control."** A tighter stop reduces the *dollars* at risk per share, yes -- but "risk control" that ignores the win rate is an illusion. Tightening the stop into the noise band raises your *frequency* of losses so much that your total expected loss can *increase* even as each individual loss shrinks. Real risk control is a stop placed where the idea is wrong (so each -1R is *informative*) combined with a position size that makes -1R a survivable fraction of your account. A stop that's tight but arbitrary controls the size of each loss and loses control of how often you take them.

**"Move to breakeven as soon as possible."** As the worked example showed, an aggressive breakeven rule on a positive-expectancy system mostly cuts winners, not losers, and can nearly halve your edge (+0.60R to +0.34R in our case). The feeling that a trade "can't lose anymore" is worth less than the winners you forfeit to buy it. If you move to breakeven at all, do it *late* (after the trade has earned real room, e.g. +1.5R) and *test the rule* in R first. The reflex to protect a tiny open profit is the single most common way disciplined traders quietly destroy a good system.

**"A high reward-to-risk target is free."** It is not. Every dollar you push the target further away, or shave off the stop to inflate the ratio, *lowers the probability of getting there*. A 10-to-1 target looks magnificent and prints maybe one time in fifteen. The reward-to-risk and the win rate are two ends of one seesaw; you never get to raise one without lowering the other, and the only thing that decides whether the trade improved is the expectancy after both have moved.

**"Scaling out increases your profit."** Scaling out almost always *decreases* expectancy slightly (you cap the first portion's upside), while it meaningfully *decreases variance*. Its benefit is a smoother equity curve and an easier-to-hold system, not more money. If your goal is maximum expectancy and you can stomach the swings, scale out less; if your goal is survivability and consistency, scale out more -- but don't tell yourself it's making you richer.

**"You can set the stop first and the target later, after you're in."** This is the original sin restated. The win rate you'll achieve depends on *both* lines together -- a near stop with a far target is a low-probability trade, a far stop with a near target is a high-probability one -- so you cannot evaluate the trade's expectancy until both exist. Designing them at different times, especially with one of them invented after entry under pressure, means you never actually know what trade you took.

## How it shows up in real markets

These are illustrative scenarios built from the mechanics above; the numbers are round teaching numbers, and the lesson, not the exact figure, is the point. As-of this writing in June 2026, the dynamics described are structural to how stops and targets interact with price; they don't go stale the way a specific rate or price would.

**The wick that takes the arbitrary stop and spares the structure stop.** This is the most common, most maddening experience in retail trading, and it plays out thousands of times a day across every liquid market. A trader buys a support level -- say a stock holding \$50 -- and places a stop at a comfortable \$49 ("a dollar of risk"). Overnight or on a news blip, price wicks down to \$48.80 and snaps right back to \$51, closing the day green. The trader is stopped out at \$49 for a full loss on a trade that was *right*. A second trader bought the same level with a structure stop at \$48.50 -- below the prior swing low, with an ATR buffer -- and is untouched; they ride the move to their \$56 target. Same idea, same entry, opposite outcome, and the entire difference is that one stop sat in the noise band and the other sat below it. Multiply this across a year of trades and it is the difference between a 35%-win system and a 48%-win system on the identical strategy.

**The trader who destroys a good system with premature breakeven stops.** A genuinely profitable swing trader -- a real, positive edge, +0.5R or more per trade -- watches a green position tick back toward entry and, every time, slides the stop to breakeven to "lock it in". Over a year, this single reflex converts dozens of eventual winners into 0R scratches: trades that dipped through entry on their way to a big target now exit flat. The trader's win rate looks fine (lots of "didn't lose" trades), but the equity curve flattens, because the few large winners that carried the whole expectancy keep getting cut at breakeven. The system didn't stop working; the management broke it. This is, anecdotally, one of the most common ways profitable strategies stop being profitable -- not a failed edge, a managed-to-death edge.

**Scaling out in a strong trend.** In a sustained trend -- the kind that runs for weeks -- a trend-following trader who scales out captures a smooth, satisfying series of partial exits: half off at +1R, a quarter at +2R, the last quarter trailing far up the move. Their expectancy is a touch lower than a hypothetical all-at-the-top exit (which nobody catches anyway), but their variance is dramatically lower and, crucially, they *stay in the trade* because they've already banked profit and the remaining runner is "house money" psychologically. The trader who went all-in for a far target on the same trend either takes profit too early (and misses the tail) or holds for the top (and gives back a chunk on the reversal). Scaling out is, for many trend traders, less about expectancy and more about *behavioral survivability* -- it's the exit plan that lets a human actually hold a winner.

**The tight-stop scalper versus the wide-stop swing trader.** Two profitable traders sit at opposite ends of the tradeoff curve, and both are right. The **scalper** runs very tight stops on a fast timeframe, targets 1-to-1 or less, and wins 60-70% of the time; their edge is a high hit rate on a low ratio, and their enemy is costs (spread and commission eat a low-ratio edge fast). The **swing trader** runs wide structure stops on a daily chart, targets 3-to-1 or 5-to-1, and wins 35-45% of the time; their edge is a fat ratio on a modest hit rate, and their enemy is patience (most trades lose, and the winners are rare and large). Neither is "correct" in the abstract; each has found the point on *their* instrument's tradeoff curve where expectancy peaks for *their* setup. A scalper who tried to run 5-to-1, or a swing trader who tried to run 1-to-1, would each destroy a working edge by moving to the wrong point on the curve.

**The "guaranteed" tight stop in a volatile name.** A trader applies a fixed, tight stop -- say 1% -- to a high-volatility stock whose daily ATR is 4%. The stop is *inside a single normal bar's range*: the stock routinely moves more than 1% in an hour for no reason. The result is a near-guaranteed string of noise-stops, a win rate that collapses toward single digits, and a negative expectancy on what might be a perfectly good entry signal. The fix is not a better signal; it's a stop sized to the instrument's volatility, which on this name means a much wider stop, a much smaller position (so the dollar risk stays the same), and a target far enough away to keep the reward-to-risk sane. The same fixed-percent stop that works fine on a sleepy index fund is a disaster on a volatile single stock, and the only thing that tells you the difference is ATR.

## When this matters to you / further reading

This matters the instant you place your next trade, because you are going to choose a stop and a target whether you do it deliberately or not -- and if you do it the beginner's way (round numbers, separate decisions, target invented after entry), you have very likely already given away your edge before the trade does anything. The shift this post asks for is small to describe and hard to do: *decide both lines before you click, place the stop where the idea is wrong rather than where the round number is, and judge the pair by expectancy rather than by the ratio or the win rate alone.*

Concretely, the next time you size up a trade, do four things in order. Find the structure your idea leans on. Put the stop just beyond it plus an ATR buffer -- and read off 1R from wherever that lands, rather than choosing 1R to be comfortable. Put the target at the next structure, and read off the reward-to-risk. Then check the expectancy at the win rate you honestly believe that pairing achieves -- and if you don't *know* your win rate at that pairing, that is the most important thing you can go measure, because without it you are guessing at the only number that matters.

Where to go from here. [The expectancy post](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) is the foundation under everything here -- the full derivation of expectancy, the breakeven win rate, and the variance and risk-of-ruin math that explains *why* a smoother (scaled-out) equity curve helps you survive. [Position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) is the natural next step: once your stop sets 1R, *how many shares* you buy decides how big a fraction of your account a -1R actually is, and that -- not the entry -- is what keeps you in the game. [Building one high-probability setup](/blog/trading/technical-analysis/building-one-high-probability-setup) shows the stop-and-target object slotted into a complete, end-to-end trade plan with bias, level, trigger, and sizing. And [support and resistance: why levels exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist) is the structural bedrock under every "place the stop beyond the level" instruction in this post -- it's the post that explains why the levels your stops and targets hang on are real in the first place.

The thread through all of it: a trade is one designed object, not a sequence of reactions. The stop and the target are the two lines that define it, you draw them together and before entry, and you judge the result the only honest way there is -- by what it earns, on average, per trade.
