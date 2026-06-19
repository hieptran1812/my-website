---
title: "The Gambler's Ruin and Bet Sizing: The Math of Staying Solvent"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Turn the gambler's-ruin formula into concrete bet-sizing rules: given an edge and a bankroll, how to find the largest bet that keeps your probability of ruin below an acceptable budget like one percent."
tags: ["risk-management", "gamblers-ruin", "bet-sizing", "position-sizing", "bankroll-management", "risk-of-ruin", "kelly-criterion", "survival"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **One sentence:** The gambler's-ruin formula turns a vague fear of blowing up into an exact number, and once you can compute your probability of ruin you can run the calculation backwards to find the largest bet that keeps that number below a budget you choose.
> - **Ruin is a computable probability, not a feeling.** With a per-bet edge, a bankroll, and a bet size, the classic formula P(ruin) = (r^i − r^N)/(1 − r^N) with r = q/p gives you the exact odds you go broke.
> - **The edge–bankroll–betsize triangle.** Ruin depends on three knobs. You can buy survival with more edge, a deeper bankroll relative to your bet, or a smaller bet — and bet size is the one you fully control.
> - **A bad streak is survivable only if your bankroll is deep enough in units.** A bankroll that is k bet-units deep survives any k-loss run; deepen the bankroll (shrink the bet) and the longest streak you can absorb grows one-for-one.
> - **Set a ruin budget, then back out the max bet.** Pick an acceptable ruin probability (say 1%), solve the formula for the smallest bankroll-in-units that clears it, and the largest bet you may risk is one unit out of that many.
> - **This is sizing, not edge-hunting.** A real one-percentage-point edge can still ruin you 88% of the time if you start under-capitalized relative to your bet; the same edge is nearly ruin-proof once the bet is small enough.

A trader with a genuine edge can still go broke with near-certainty. That is not a paradox or a trick of words — it is arithmetic, and it has a closed-form formula attached to it that has been understood since the seventeenth century. The formula is the gambler's-ruin problem, and once you can write it down and plug numbers into it, the whole murky question of *"how much should I bet?"* collapses into something you can actually solve.

Most traders never solve it. They size by feeling — a position that "feels right," a round number of contracts, a percentage they heard on a podcast — and then they're surprised when a winning system hands them a losing account. The reason is that sizing-by-feeling has no error bars: it gives you no way to know whether you're betting twice the safe amount or a hundredth of it, and the penalty for those two mistakes could not be more different. Bet a hundredth of the safe amount and you give up some growth. Bet twice the safe amount and you might be sitting on a coin flip with your career. The gambler's-ruin formula is the instrument that replaces the feeling with a number, and the whole point of this post is to hand you that instrument and teach you to read it both forwards and backwards.

This post is the practical sizing closer for the survival track of this series. The conceptual groundwork is already laid: the [risk-of-ruin post](/blog/trading/risk-management/risk-of-ruin-why-positive-expectancy-is-not-enough) established that a positive edge tells you a bet is worth taking but says nothing about size, and that zero is an absorbing barrier you never come back from; the [ergodicity post](/blog/trading/risk-management/ergodicity-time-average-vs-ensemble-average-and-the-coin-flip-that-ruins-you) established why your personal path through time can be ruined even when the average across a crowd of traders looks fine. We are not going to re-derive those ideas. We are going to *apply* them — to turn the math into a number on a calculator, and then into a rule you size every position by.

Here is the destination, stated up front so you know where we are heading. By the end you will be able to do three things. First, given an edge, a bankroll, and a bet size, compute your exact probability of ruin. Second, run that calculation in reverse: pick a ruin probability you can live with — your *ruin budget* — and back out the largest bet that respects it. Third, translate all of it onto a real trading account, so the textbook "i units toward a target N" becomes a dollar figure you actually risk per trade. That is the entire job of bet sizing: keep the probability of touching zero below a budget you set in advance, and only then go collect your edge.

![Gambler's ruin probability versus starting bankroll in units for a slight edge, a fair coin, and a slight disadvantage, with the disadvantage case ruining about 88 percent of the time from the midpoint](/imgs/blogs/the-gamblers-ruin-and-bet-sizing-the-math-of-staying-solvent-1.png)

Look at the figure above before reading on. It plots the exact probability of ruin against how deep your bankroll is, measured in bet-units, for three coins that differ by a single percentage point. The red curve is a 49% coin — a hair *worse* than fair. Starting from the midpoint, with 50 units of bankroll and a target of 100, it goes broke 88% of the time. The green curve is its mirror, a 51% coin, and from the same midpoint it goes broke only 12% of the time. One percentage point of per-bet edge is the entire difference between "almost certainly ruined" and "almost certainly fine." That sensitivity is the whole game, and the formula behind that curve is what we are going to learn to wield.

## Foundations: the gambler's-ruin formula from first principles

Before we can size anything, we need the formula that produces the curve above, and we need every symbol in it defined from zero. Skip nothing here; the entire sizing playbook is just this formula read in different directions.

### The setup: a unit-bet random walk between two walls

The cleanest version of the problem is a betting game played one unit at a time. You start holding **i** units of money. Each round you stake exactly one unit. With probability **p** you win and your stake goes up by one unit; with probability **q = 1 − p** you lose and it goes down by one unit. You keep playing until one of two things happens: your stake climbs to a target **N** units (you have "won the game," reached your goal), or it falls to **0** (you are ruined). The number you want is the probability that you hit 0 before you hit N.

This is a random walk between two walls — a wall at 0 and a wall at N — and both walls are *absorbing*: once the walk touches either one, the game stops. The wall at zero is the one that matters for survival, because in real markets it is the only one you can't walk back from. We covered why zero is absorbing in the [risk-of-ruin post](/blog/trading/risk-management/risk-of-ruin-why-positive-expectancy-is-not-enough); here we just need the consequence: ruin is the event that the path *first touches* zero, ever, before reaching the goal.

A unit bet sounds artificial — nobody risks "one unit" — but it is the right simplification, and the payoff is enormous. Your real bankroll is not "one trade." It is *many bets deep*. If you risk \$1,000 per trade on a \$100,000 account, your account is 100 units deep. The unit-bet model says: forget the dollars, just count how many bets of loss your bankroll can absorb before it is gone. That count — your bankroll measured in bet-units — turns out to be the single most important number in sizing. Bet smaller, and the same dollar account becomes more units deep, and ruin falls. That is the lever, and the formula tells you exactly how hard it pulls.

### The formula itself

Here is the result, the centerpiece of the entire post:

P(ruin) = (r^i − r^N) / (1 − r^N),  where  r = q / p

Let me unpack every piece. The ratio **r = q/p** is the *odds against you per bet* — how much more likely a loss is than a win. If the game is fair, p = q = 0.5, so r = 1 and the formula above divides by zero; we handle that fair case separately (below). If you have an edge, p > q, so r < 1. If you are at a disadvantage, p < q, so r > 1. The whole behavior of the formula is governed by whether r is above or below 1, because raising a number to a high power either shrinks it toward zero (r < 1) or explodes it (r > 1).

- **i** is your starting bankroll in units — how many one-unit bets of losses you can absorb.
- **N** is your target in units — where you'd stop, "having won."
- **r^i** and **r^N** are r raised to those powers.

For the **fair game** (p = 0.5 exactly), the formula has a beautifully simple limit:

P(ruin) = 1 − i/N

So a fair coin starting at the midpoint, i = 50 of N = 100, ruins exactly 1 − 50/100 = 50% of the time. That is the slate dotted line on the cover figure — a straight line, because for a fair coin the probability of reaching either wall is just proportional to how close you start to it. The instant you tilt the coin even slightly, that straight line bends into the dramatic curves you see in green and red.

### Where the formula comes from, in two lines

You don't have to take the formula on faith — it falls out of one simple observation, and seeing it makes the symbols stick. Let P(i) be the probability of eventually being ruined starting from i units. From i, your very next bet either wins (probability p, leaving you at i+1) or loses (probability q, leaving you at i−1). After that bet you're in exactly the same kind of problem, just from a new starting point. So the ruin probability from i must equal the probability of winning times the ruin probability from i+1, plus the probability of losing times the ruin probability from i−1:

P(i) = p · P(i+1) + q · P(i−1)

That is a *recurrence relation*, and it comes with two boundary conditions that are obvious once stated: P(0) = 1 (if you're already at zero, you're already ruined, with certainty), and P(N) = 0 (if you've already reached the target, you can't be ruined). A recurrence of this exact form has a known solution: a geometric one, P(i) = A + B·r^i with r = q/p, where A and B are fixed by the two boundary conditions. Plugging the boundaries in and solving the two simple equations gives precisely P(i) = (r^i − r^N)/(1 − r^N) for the edge case, and the straight-line P(i) = 1 − i/N for the fair case where r = 1. That is the whole derivation: a one-step look-ahead, two boundary facts, and a little algebra. The heavier sequential-betting version of this argument lives in the [quant-interview Kelly post](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews); here the recurrence is all we need, and what matters is the lesson it encodes — the r^i term is your bankroll depth doing exponential work, for you if r < 1 and against you if r > 1.

### Three numbers that show why one point of edge is everything

Let's plug in the canonical case, the one annotated on the cover, computed exactly with the series' `dr.ruin_prob_unit_bet(p, i, N)` helper so the same numbers recur across every post:

- **p = 0.49, i = 50, N = 100** → P(ruin) = **0.88**. A coin a single point *worse* than fair, starting at the midpoint, goes broke 88% of the time.
- **p = 0.50, i = 50, N = 100** → P(ruin) = **0.50**. The fair coin: a coin flip on your survival.
- **p = 0.51, i = 50, N = 100** → P(ruin) = **0.12**. A coin a single point *better* than fair goes broke only 12% of the time.

Stop and feel how violent that is. Moving the per-bet win probability from 49% to 51% — two percentage points — drops the probability of ruin from 88% to 12%. The reason is that small per-bet advantages *compound over many bets*. Each bet's tiny tilt gets raised to a power as high as your bankroll depth, and exponentiation turns small inputs into enormous output differences. This is the mathematical engine behind everything that follows: **ruin is hypersensitive to the per-bet edge and to your bankroll depth, and you control the second of those directly through bet size.**

#### Worked example: the gambler's ruin on a real account

Make it concrete on the recurring **\$100,000 account**. Suppose you have a strategy with a genuine but slim edge — call it a 51% chance of a winning trade at 1:1 odds (you make what you risk when you win, lose what you risk when you lose). You decide to risk \$2,000 per trade and you'll consider yourself "done" — having doubled — at \$200,000.

Translate to units. Your bet is \$2,000, so one unit = \$2,000. Your starting bankroll is \$100,000 / \$2,000 = **50 units**. Your target of \$200,000 is **100 units**. That is exactly i = 50, N = 100, p = 0.51 — so your probability of *ruin before doubling* is 0.12, or **12%**.

Twelve percent. Roughly one chance in eight that a strategy with a real, positive edge takes your \$100,000 to zero before it ever reaches \$200,000. Now cut the bet in half, to \$1,000. One unit = \$1,000, so your account is 100 units deep and your target is 200 units. Plug in p = 0.51, i = 100, N = 200, and ruin falls to **1.8%**. You halved the bet and ruin dropped by roughly a factor of seven.

*The edge didn't change at all between those two cases — only the bet size — and yet the probability of going broke moved from one-in-eight to one-in-fifty. Bet size, not edge, was doing the work.*

## The edge–bankroll–betsize triangle

Three knobs control ruin, and they trade off against each other. Hold ruin fixed and you can move along a surface where more of one lets you spend less of another. Seeing this triangle clearly is what lets you stop treating sizing as a gut feeling.

The three knobs:

1. **Edge** — the per-bet win probability p (equivalently, r = q/p). More edge crushes ruin, because r drops further below 1 and r raised to your bankroll depth collapses faster.
2. **Bankroll depth** — i, your bankroll measured in bet-units. A deeper bankroll means more losses you can absorb, which means a higher power on a number less than 1, which means lower ruin.
3. **Bet size** — the dollar fraction you risk per bet. This is the knob that *sets* your bankroll depth: depth i = (bankroll) / (bet size). Halve the bet, double the depth.

Notice that bet size and bankroll depth are two faces of the same coin. You rarely change your actual capital trade-to-trade, so in practice the lever you pull is **bet size**, and it moves bankroll depth inversely. That is why this whole discipline is called *bet sizing* and not *bankroll growing*: the number you tune every day is the bet, and tuning it down is how you buy survival without needing more edge or more money.

![Heatmap of ruin probability across per-bet edge on the horizontal axis and bankroll depth in units on the vertical axis, with a green safe corner of high edge and deep bankroll and a red ruin zone of thin bankroll and no edge](/imgs/blogs/the-gamblers-ruin-and-bet-sizing-the-math-of-staying-solvent-3.png)

The heatmap above is the triangle drawn as a surface. The horizontal axis is the per-bet edge (win probability), the vertical axis is bankroll depth in units, and the color is the exact probability of ruin from the formula. The **safe corner** is top-right: real edge plus a deep bankroll, where ruin is green and near zero. The **ruin zone** is bottom-left: a thin bankroll with no edge, where ruin is red and near one. The white contour lines mark the 1%, 10%, and 50% ruin boundaries — those contours are exactly the "budget" lines we will learn to live on. To get into the safe corner you can move right (find more edge) or move up (deepen your bankroll by shrinking your bet). The fastest, most reliable route for a trader is *up*, because you control your bet today and your edge only over months of research.

Two more features of the surface are worth reading off, because they correct common sizing instincts. First, notice how the contours *bunch up* on the left, near the fair-coin line at p = 0.50: when your edge is thin, ruin probability changes violently with small changes in either edge or depth, so the surface is steep and treacherous and you have almost no margin for error. Out on the right, with a fat edge, the contours spread apart and the surface flattens — a strong edge is *forgiving* of sizing mistakes, while a thin edge punishes them. This is the quantitative reason thin-edge strategies (most retail strategies) demand more sizing discipline, not less: you are operating on the steep part of the surface where a small misjudgment of your edge slides you across several ruin contours at once. Second, notice that no amount of bankroll depth fully rescues you on the *wrong* side of the fair line — to the left of p = 0.50 the surface stays red even at the top, because a negative-edge game ruins you eventually no matter how deep you start (the red cover curve at 88% was exactly this). Depth buys you *time* against a bad edge, never immunity. The only durable home is the green corner, and you get there by pairing a real edge with a bet small enough to keep you many units deep.

#### Worked example: buying survival two different ways

You run the **\$10,000,000 book** and you're unhappy that your current ruin probability is too high. You have two ways to fix it, and the heatmap shows both.

Suppose your strategy is a 51% edge and you currently risk \$200,000 per trade, so your book is 50 units deep, targeting 100 units (doubling). From the formula, p = 0.51, i = 50, N = 100 gives ruin = 12%. Too high. Two fixes:

- **Move right (more edge).** If you could lift your win rate to 55% — a huge, hard-won research improvement — then at the same 50-unit depth, p = 0.55, i = 50, N = 100, ruin collapses to essentially **0%**. Wonderful, but a four-point edge improvement is a multi-year project and might be impossible.
- **Move up (deeper bankroll via a smaller bet).** Keep the 51% edge, but cut the bet to \$100,000. Now the book is 100 units deep, target 200, and p = 0.51, i = 100, N = 200 gives ruin = **1.8%**. You bought a 7× reduction in ruin overnight, with no new edge, purely by halving the bet.

*Finding more edge and deepening the bankroll both move you toward the safe corner, but only one of them is available to you this afternoon — and it is the bet size.*

## The absorbing barrier: why solvency is a first-passage problem

It is worth being precise about *what* the formula computes, because the precision changes how you think about risk. The gambler's-ruin probability is a **first-passage probability**: the chance that the equity path *first touches* zero at some point before it reaches the goal. It is not about where you end up on average. It is about whether you ever, even once, brush against the absorbing barrier — because brushing it once is fatal.

![Flow diagram showing a bankroll path that wanders up and down with each bet, branches into either reaching the target or a bad run that first touches zero and is absorbed forever](/imgs/blogs/the-gamblers-ruin-and-bet-sizing-the-math-of-staying-solvent-5.png)

The diagram traces the logic. You start with i units and an edge. The equity wanders, plus one on a win and minus one on a loss. Most of the time, with an edge, it drifts up and reaches the target — you survived the run and banked the win. But some paths get dragged down by a bad streak, and the moment one of them *first touches zero*, it is absorbed: there is no stake left to bet, the edge becomes irrelevant, and the game is over forever. The probability the formula gives you is the weight of all the paths that end in that bottom-right absorbing node rather than the top-right goal node.

This is why averaging is the wrong instinct. A trader who reasons "my expected return per trade is positive, so over time I'll be fine" is computing the *destination* of the average path. But ruin is decided by the *minimum* the path reaches along the way, not its average endpoint. A path can have a wonderful positive expected endpoint and still, on its way there, dip down and touch zero — at which point the wonderful endpoint never arrives. First passage, not average, is what kills accounts. This is the same time-versus-ensemble distinction the [ergodicity post](/blog/trading/risk-management/ergodicity-time-average-vs-ensemble-average-and-the-coin-flip-that-ruins-you) develops at length; here it shows up as: *the formula measures a property of your worst moment, not your average outcome.*

## Fixed-fractional sizing: the safe zone and the cliff

So far the model bets a constant dollar amount each round, which can literally reach zero. Most real traders instead bet a constant *fraction* of their current account — risk 1% of equity per trade, say. This is **fixed-fractional sizing**, and it behaves differently in an important and reassuring way: because each bet shrinks as the account shrinks, you can never quite reach exactly zero. You can, however, get dragged so far down that you are finished for all practical purposes — a *practical ruin* at, say, −90%, which under the recovery asymmetry needs a +900% gain to undo and is, realistically, the end.

The shape of how practical ruin depends on the bet fraction is the single most useful picture in sizing.

![Probability of practical ruin rising with bet fraction under fixed-fractional sizing, flat and low in a safe zone up to about half-Kelly then climbing steeply toward near-certain ruin past full-Kelly](/imgs/blogs/the-gamblers-ruin-and-bet-sizing-the-math-of-staying-solvent-2.png)

The figure above is a seeded simulation of a real positive edge — a 55% coin at 1:1 odds, whose growth-optimal full-Kelly bet is 10% — sized at every fraction from a sliver up to half your account, with the share of paths dragged to a −90% practical ruin measured at each fraction. The shape is the lesson. There is a **safe zone** on the left where ruin stays low and nearly flat: betting somewhere up to around half-Kelly barely moves your ruin probability. Then there is a **steep climb** as the bet grows past the growth-optimal point, and past roughly twice-Kelly the probability of practical ruin marches toward near-certainty. The curve is not gentle. It is a plateau followed by a cliff.

Two practical consequences fall out of that shape. First, **there is almost no penalty for betting too small.** In the safe zone the ruin curve is flat, so shading your bet down a notch costs you essentially nothing in ruin terms (it costs a little growth, which is a trade most survivors gladly make). Second, **there is a savage penalty for betting too large.** Once you're on the cliff, a small increase in bet size produces a large increase in ruin. The asymmetry of those two penalties — trivial on the small side, catastrophic on the large side — is the entire argument for erring small. This is also exactly why practitioners bet *fractional* Kelly, a point developed in the dedicated [Kelly criterion post](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge): the growth-optimal bet sits right at the foot of the cliff, and you want a safety margin between you and the edge of it.

#### Worked example: where on the cliff are you?

Back to the **\$100,000 account** with the 55% / 1:1 strategy, whose full-Kelly fraction is 10% (\$10,000 per trade) and half-Kelly is 5% (\$5,000). Three sizing choices:

- **Risk \$2,000 per trade (2%).** You're deep in the safe zone, well left of half-Kelly. Practical ruin is negligible and you give up only a little long-run growth. This is where most disciplined traders live.
- **Risk \$5,000 per trade (5%, half-Kelly).** Still in the safe zone, near its right edge. Ruin is low; growth is close to its max. A reasonable aggressive choice.
- **Risk \$20,000 per trade (20%, twice-Kelly).** You're over the cliff. Despite a genuine, strong 55% edge, a meaningful fraction of equity paths get dragged to practical ruin — this is precisely the scenario the [risk-of-ruin post's](/blog/trading/risk-management/risk-of-ruin-why-positive-expectancy-is-not-enough) cover simulation shows, where one path in five collapses *with the edge fully intact.*

*The same edge is nearly ruin-proof at 2% and a coin flip on your survival at 20% — the bet fraction alone decides which side of the cliff you stand on.*

## How long can you survive a bad streak?

Ruin probability is the headline, but there is a second, more visceral question every trader asks during a drawdown: *how long can I keep losing before I'm done?* The unit-bet model answers it with brutal simplicity, and the answer is the most intuitive sizing rule of all.

The worst case is a pure losing streak — losses back-to-back with no wins in between. If your bankroll is **i** units deep and you bet one unit each time, then **i** losses in a row take you from i straight to zero. So the longest losing streak you can survive is *exactly* your bankroll depth in units. A bankroll 100 units deep survives any 99-loss run and dies on the 100th. A bankroll 20 units deep dies on the 20th consecutive loss. There is nothing subtle here: depth in units *is* the length of the worst streak you can absorb.

![Survival length versus starting bankroll showing the worst-case survivable losing streak equal to bankroll depth and the expected number of bets to absorption rising with a deeper bankroll](/imgs/blogs/the-gamblers-ruin-and-bet-sizing-the-math-of-staying-solvent-4.png)

The figure plots two survival-length facts against bankroll depth. The amber dashed line is the worst-case survivable streak — it is just the 45-degree line, "a k-unit bankroll survives a k-loss run," because that relationship is exact. The colored curves are the *expected* number of bets until the game ends (at either wall), the classic gambler's-ruin duration. With an edge (green) you tend to survive longer from a deep start because the edge keeps pushing you away from the zero wall; with a disadvantage (red) the duration is shorter because the drift carries you toward ruin. But the simplest, most actionable takeaway is the amber line: **deepen your bankroll in units — by shrinking your bet — and the longest streak you can ride out grows one-for-one.**

There's a psychological dimension the math doesn't capture but the playbook must. A long losing streak doesn't just drain capital — it drains conviction, and a trader bleeding through a drawdown is exactly the trader most likely to do something stupid: double the bet to "make it back," abandon the system at the worst possible moment, or freeze. Sizing for streak-survival isn't only about the arithmetic of the bankroll; it's about choosing a bet small enough that the inevitable bad run is *boring* rather than terrifying, so you can keep executing the system mechanically while the edge reasserts itself. A bet that's correct on the ruin math but large enough to keep you awake at night is, in practice, too large, because the real failure mode isn't usually the account hitting zero — it's you abandoning a winning system three losses before it would have turned. The deeper your bankroll in units, the longer and calmer the streak you can sit through, and that calm is itself part of the edge.

#### Worked example: sizing to survive a plausible streak

You have the **\$100,000 account** and you want to be sure you can survive the worst losing streak you'll realistically face. Suppose your win rate is 51%, so each trade loses with probability q = 0.49. How long a streak should you plan for? A 10-loss streak has probability 0.49^10 ≈ **0.08%** — rare but not freakish over a few thousand trades. A 20-loss streak is 0.49^20 ≈ **0.00006%** — essentially never.

Now size for it. If you want to survive any 10-loss run, your bankroll must be at least 10 units deep, so your bet can be at most \$100,000 / 10 = \$10,000 (10% — and at 10% you survive the streak but, as the cliff figure warned, your *ruin* probability over many trades is still ugly). If you want to survive any 20-loss run, your bet drops to \$5,000 (5%). If you want a comfortable 50-unit cushion, your bet is \$2,000 (2%). Notice these are the *same bet sizes* the ruin-probability analysis pushed you toward — the streak view and the probability view agree, because they're two readings of the same depth-in-units number.

*The streak you can survive is just your bankroll counted in bets, so "how small should I bet?" and "how long a losing run must I outlast?" are the identical question asked from two directions.*

## From the coin to the trade: making the model fit real returns

The clean unit-bet model assumes every win pays exactly one unit and every loss costs exactly one unit. Real trades aren't like that. A trade might risk \$1,000 to make \$2,000, or win 0.8% and lose 1.2%; the wins and losses are different sizes, and the win probability isn't a tidy 51%. The model still applies — you just have to map your real strategy onto it first, and the mapping is worth doing carefully because a sloppy translation is how people convince themselves an over-sized bet is safe.

The bridge is the **R-multiple**. Define one R as the amount you risk per trade — your stop distance times your position size, the dollar figure you lose when the trade goes against you. Then express every outcome as a multiple of R: a winner that makes twice what you risked is +2R, a loser that hits your stop is −1R, a partial loss is −0.5R. Now your strategy is summarized by a *distribution of R-multiples* and a win rate. Your per-trade expectancy in R is simply the probability-weighted average of those multiples, and your bankroll, measured in R, is your account divided by the dollar value of one R. That bankroll-in-R is the direct analogue of "bankroll in units" from the coin model — it is how many full risk-units of loss your account can absorb.

Here's why the translation matters for sizing. When wins and losses are different sizes, the relevant "odds against you" is no longer just q/p — it folds in the payoff ratio b (how many units a win pays per unit risked). The growth-optimal Kelly fraction for such a bet is f\* = (p·(b+1) − 1)/b, the formula the [Kelly post](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge) builds out; what matters here is that a bigger b (wins much larger than losses) buys you a higher safe bet fraction, the same way a bigger p did in the coin model. A strategy that wins only 40% of the time can still be deeply survivable if its winners are 3R and its losers are 1R, because each win refills more of the bankroll than each loss drains. The gambler's-ruin lesson survives the translation intact: it's the *combination* of edge (now a blend of win rate and payoff size) and bankroll depth in R that sets ruin, and bet size is what sets the depth.

#### Worked example: an asymmetric strategy on the \$100,000 account

Suppose your strategy wins only 45% of the time, but winners average +2R and losers average −1R. Is this survivable? First, expectancy: per trade you make 0.45 × 2R − 0.55 × 1R = 0.90R − 0.55R = **+0.35R**. A healthy positive edge despite the sub-50% win rate. Now size it. If one R is \$1,000 on your \$100,000 account, your bankroll is 100R deep. The growth-optimal Kelly fraction here is f\* = (0.45 × 3 − 1)/2 = (1.35 − 1)/2 = **0.175** — full-Kelly would risk 17.5% of equity, or about 17.5R, per trade. That is a *huge* bet, and it sits right at the foot of the cliff. A survival-minded trader bets a fraction of it: at half-Kelly you'd risk roughly 8.75R (\$8,750), and most would go smaller still, holding the bet near 1–2R (\$1,000–\$2,000) so the bankroll stays 50–100R deep and the worst plausible losing streak is easily survived.

*The lopsided 45%-win strategy is genuinely profitable and genuinely survivable — but only if you size by its full risk-unit depth and ignore the temptation to bet the fat 17.5% the growth-optimal formula technically allows.*

This is also the cleanest way to see why the discrete model isn't a toy. Translate to R-multiples and bankroll-in-R, and a real trading account *is* a gambler's-ruin problem — a random walk with unequal step sizes between an absorbing barrier at zero and the open-ended goal of compounding. Everything we derived for the coin carries over: ruin is hypersensitive to the edge, it falls steeply as the bankroll deepens in R, and the lever you actually pull is the size of one R relative to your account. With that bridge in hand, we can finally do the move that makes this whole post operational.

## Backing the max bet out of a ruin budget

Now we run the formula in reverse — the move that turns all this analysis into an actual rule. Instead of *plugging in* a bet size and reading out the ruin probability, we *fix* the ruin probability we're willing to accept and solve for the largest bet that respects it.

First, a clean simplification. For a favorable game (p > 0.5) played toward a distant goal, the probability of *ever* being ruined from a bankroll of i units approaches a simple expression: **r^i**, with r = q/p < 1. This is the conservative "ruin-ever" version — it ignores the protection of a finite target and asks only "what's the chance I ever touch zero?" — which is exactly the pessimistic bound a survival-minded trader should size against. Now set a **ruin budget** B (your acceptable probability of ruin, say 1%) and solve:

r^i ≤ B  →  i ≥ ln(B) / ln(r)

The right-hand side is the *minimum bankroll depth in units* that keeps ruin under budget. Round it up to a whole number — call it i\* — and the largest bet fraction you may risk per bet is one unit out of i\*:

f_max = 1 / i\*

That's the whole rule. Pick a ruin budget, compute how many units deep you must be, and your max bet is the reciprocal of that depth. Let's see what it demands.

![Largest allowed bet fraction per bet versus per-bet edge for a one percent and a five percent ruin budget, showing a bigger edge earns a bigger allowed bet while a thin edge forces tiny bets](/imgs/blogs/the-gamblers-ruin-and-bet-sizing-the-math-of-staying-solvent-6.png)

The figure plots f_max against the size of your edge, for a 1% ruin budget (green) and a looser 5% budget (amber). Two things jump out. First, **a thin edge forces a tiny bet.** A one-percentage-point edge (p = 0.51) under a 1% ruin budget allows at most about a **0.86%** bet — under one percent of your account per trade. That feels absurdly small to most traders, which is exactly why most traders are over-sized. Second, **a bigger edge earns a bigger allowed bet,** and steeply: a 5% edge (p = 0.55) allows about 4.35% per bet under the same 1% budget, and a 10% edge (p = 0.60) allows about 8.3%. The allowed bet is roughly proportional to your edge — which, not coincidentally, is the same message the Kelly criterion delivers, by a different route, in its [own post](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge).

#### Worked example: backing out the max bet on a real account

Take the **\$100,000 account** with the slim, realistic 51% edge, and set a strict ruin budget of **1%**. Compute the required depth: r = 0.49/0.51 = 0.9608, and i\* = ⌈ln(0.01)/ln(0.9608)⌉ = ⌈115.2⌉ = **116 units**. So you must be 116 bets deep, which means your max bet is \$100,000 / 116 ≈ **\$862**, or 0.86% of the account.

Now sanity-check against over-sizing. If you instead risked the "feels reasonable" 5% — \$5,000 per trade — you'd be only 20 units deep, and the ruin-ever bound r^20 ≈ **45%**. A coin-flip-and-then-some chance of eventually going broke, on a real edge, because the bet was six times too big for a 1% budget. Drop to \$1,000 (100 units), and r^100 ≈ **1.8%** — close to budget but not quite there, which is why the exact rule said \$862, not \$1,000.

*A genuine edge bought you almost nothing in permissible bet size — under one percent of the account — because the edge was thin; the rule is blunt about the fact that small edges demand small bets.*

#### Worked example: the same rule on the institutional book with a stronger edge

Now the **\$10,000,000 book**, but with a stronger, 55% edge, same 1% ruin budget. Here r = 0.45/0.55 = 0.8182, and i\* = ⌈ln(0.01)/ln(0.8182)⌉ = ⌈22.95⌉ = **23 units**. So the book must be 23 bets deep, and the max bet is \$10,000,000 / 23 ≈ **\$434,783**, about 4.35% of the book. Check it: r^23 ≈ 0.99%, just under the 1% budget, as designed.

Compare that 4.35% ceiling to the strategy's *growth-optimal* full-Kelly bet, which for a 55% / 1:1 edge is 10% (\$1,000,000) and half-Kelly is 5% (\$500,000). The ruin-budget ceiling of 4.35% sits just *below* half-Kelly — which is a lovely consistency check. The two completely independent ways of thinking about size — "maximize long-run growth, then back off for safety" (Kelly) and "cap the probability of ruin at a budget" (this post) — land in the same neighborhood: bet a few percent of the book, well under the growth-optimal point.

*A four-point stronger edge let the institutional book bet five times the fraction the retail account could — 4.35% versus 0.86% — for the identical 1% ruin budget, because permissible bet size scales with edge.*

## Where the model lies to you, and how to size for it anyway

The gambler's-ruin formula is exact — but it is exact about a *model*, and the model makes three assumptions that real markets violate. Knowing where it bends keeps you from trusting a max-bet number that was computed on fantasy inputs. The fix in every case is the same: shade your bet *smaller* than the clean formula says, because every one of these violations pushes real ruin *higher* than the model.

**Assumption one: each bet is independent.** The model treats every trade as a fresh coin flip, uncorrelated with the last. Markets don't oblige. Losses cluster — a bad regime produces a run of correlated losers, because the same condition that broke your edge today is still there tomorrow. Worse, traders run *many positions at once*, and in a crisis those positions stop being separate bets and become one bet, because [correlations go to one](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis). A book that looks 100 units deep across ten "independent" strategies can be 10 units deep when all ten move together. Serial correlation and cross-correlation both shrink your *effective* bankroll depth far below the count you'd get by treating bets as independent — so the real losing streak you must survive is longer than the independent model predicts, and your safe bet is smaller.

**Assumption two: the loss size is fixed.** The unit-bet model loses exactly one unit per losing bet, and even the R-multiple version assumes your stop holds. Real losses gap *through* stops. A position you intended to risk 1R on can lose 5R when the market jumps overnight or a circuit breaker traps you — the [fat tails](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants) the normal model ignores are exactly the multi-R losses that turn a survivable streak fatal. If a single trade can secretly cost 5R, then a bankroll that's "100R deep" is really 20 worst-case-losses deep, and your ruin math should use that pessimistic number. This is the whole reason the [options position-sizing post](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) insists you size short-option books against the tail, not the typical move: the payoff's fat left tail makes the realized loss-per-bet much larger than the average.

**Assumption three: the edge is constant and known.** The formula takes p as a fixed, known input. In reality you *estimate* p from past results, with error, and the true edge *drifts* — strategies decay, regimes change, and the edge you measured last year may be gone. Because ruin is hypersensitive to p (recall: a two-point swing took ruin from 88% to 12%), a small overestimate of your edge produces a large underestimate of your ruin. The discipline is to plug a *conservatively discounted* edge into the formula and to monitor whether the live edge still matches the assumed one — and to cut size the moment it doesn't.

#### Worked example: sizing for a fat-tailed loss on the institutional book

Return to the **\$10,000,000 book** with the 55% edge, where the clean formula permitted a 4.35% bet (\$434,783) under a 1% ruin budget. Now suppose realistically that in a tail event your "1R" stop fails and you actually lose 4R — a fourfold gap-through. Your effective bankroll depth in worst-case-losses is one-quarter of the nominal count: the 23-unit requirement becomes a *92*-unit requirement against the real worst-case loss, so the prudent max bet drops to \$10,000,000 / 92 ≈ **\$108,700**, about 1.1% of the book. The clean model said 4.35%; reality, with gapping losses, says roughly a quarter of that.

*The formula's max bet is a ceiling computed on a forgiving model, so the honest move is to treat it as the number you size comfortably below — never the number you size up to.*

## Common misconceptions

**"A positive edge means I can't go broke."** False, and it's the most expensive belief in trading. A 51% edge starting 50 units deep toward a 100-unit goal goes broke 12% of the time; the same edge over-sized to 5-units-deep goes broke far more often. The cover figure's red curve is a *negative*-edge case at 88% ruin, but the green positive-edge curve still shows real ruin probabilities at thin bankroll depths. Edge sets the *direction* of drift; bet size sets whether you survive the journey.

**"Bet bigger when you have a bigger edge — go all-in on your best setups."** Half right, fatally incomplete. A bigger edge *does* permit a bigger bet — f_max scales with edge, from 0.86% at a 1-point edge to 8.3% at a 10-point edge. But "go all-in" means f = 100%, which is i = 1 unit of depth, and at one unit deep a *single* loss is ruin. The rule permits a *proportionally* bigger bet, never a reckless one; even a 10-point edge caps you near 8% under a 1% budget, not 100%.

**"If I bet a fraction of my account, I can never go broke, so sizing doesn't matter."** Technically true that fixed-fractional sizing never reaches exactly zero, but practically false. The fixed-fractional ruin curve shows that an over-sized fraction drags a large share of paths to −90% — a *practical* ruin that needs a +900% recovery and ends careers just as surely as literal zero. "Can't reach zero" is cold comfort at −95%.

**"Surviving a 10-trade losing streak means I'm safe."** It means you survived *that streak*, not that your ruin probability is low. Streak-survival is a worst-case-depth check (be at least 10 units deep to survive 10 losses); ruin probability is a whole-distribution check over thousands of trades. You can clear the streak bar at 10% bet size and still carry an ugly multi-year ruin probability, because losses don't have to come consecutively to bleed you out. Size for the *probability*, and the streak takes care of itself.

**"The right bet is the one that grows my money fastest."** That's the growth-optimal (full-Kelly) bet, and it sits exactly at the foot of the cliff in the fixed-fractional figure — the point of maximum growth is also the point where ruin starts climbing steeply. Betting for fastest growth and betting to stay solvent are *different objectives* with different answers, and survival has to win, because you can only compound if you're still in the game.

**"My bankroll is my net worth, so I'm always deep enough."** Your bankroll is the capital you'll actually keep deploying through a drawdown — not your net worth, and definitely not your capital *before* a margin call or a redemption shrinks it. Leverage makes this worse by quietly cutting your effective depth in units, as the [leverage post](/blog/trading/risk-management/leverage-and-the-arithmetic-of-ruin) details: borrowing to bet bigger reduces i, raising r^i, raising ruin. The depth that matters is the one measured against the bet you're *really* placing, after leverage.

## How it shows up in real markets

The gambler's-ruin math is asset-agnostic arithmetic, but blow-ups are where it stops being a formula and starts having names and dates. In every case the pattern is the same: a real or perceived edge, sized far past any sane ruin budget, meeting a losing streak the bankroll couldn't absorb.

**Long-Term Capital Management, August–September 1998.** LTCM ran convergence trades with a genuine statistical edge and roughly 25-to-1 balance-sheet leverage on about \$4.7 billion of equity, with around \$1.25 trillion in gross derivative notional. In gambler's-ruin terms, that leverage drove their effective bankroll depth in units down to a tiny number — a single adverse move in correlated positions was a multi-unit loss. When Russia defaulted and their correlations [went to one](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis), the "diversified" book moved as one bet, and roughly \$4.6 billion of capital evaporated in about four months, forcing a \$3.6 billion Fed-organized recapitalization. The edge was real; the bankroll depth, after leverage, was one bad streak from zero.

**Amaranth Advisors, September 2006.** Amaranth held concentrated, levered natural-gas calendar spreads. The position was so large relative to the fund's capital — a bankroll only a few units deep against the volatility of that spread — that one bad week of gas moves cost roughly \$6.6 billion and ended the fund. A single concentrated bet at low effective depth is the gambler with i = 2 units: one streak from ruin.

**Archegos Capital Management, March 2021.** Archegos held concentrated single-stock exposure financed with total-return swaps at perhaps 5-times-plus leverage, hidden across multiple prime brokers who each saw only their slice. When the underlying stocks fell, the levered book had almost no depth in units to absorb the drawdown; the family office was wiped out and its banks lost over \$10 billion in aggregate, Credit Suisse alone roughly \$5.5 billion. Leverage had quietly cut everyone's bankroll-in-units to a sliver, and the formula does the rest.

**The XIV blow-up (Volmageddon), 5 February 2018.** The short-volatility carry trade had a real edge — selling volatility pays most of the time. But the inverse-VIX product XIV was structurally over-sized against the tail: when the VIX jumped from 17.3 to 37.3 in a day, the largest one-day VIX percentage rise on record, XIV's net asset value fell about 96% after the close and it was terminated. A −96% is past any conceivable practical-ruin floor — the trade was sized as if the tail couldn't happen, which is sizing with no ruin budget at all. The [options-volatility post on Volmageddon](/blog/trading/options-volatility/case-study-volmageddon-2018-and-the-short-vol-blowup) traces the reflexive feedback loop in detail.

The thread through all four: none of these died from a *bad edge*. They died from sizing — leverage and concentration that drove their effective bankroll depth so low that an ordinary-to-severe streak reached the absorbing barrier before the edge could pay. Every one of them would have survived the same shock at a fraction of the size. That is the entire content of the gambler's-ruin formula, written in capital and headlines.

And note the asymmetry in the public memory of these events. We remember the blow-ups because they have names; we never hear about the thousands of funds and traders running the *same* edges at survivable size, who quietly compounded through the same shocks because their bankroll-in-units was deep enough to absorb the streak. The survivors are invisible precisely because nothing dramatic happened to them — which is the point. The gambler's-ruin formula is not a theory of disaster; it is the dividing line between the trader who shows up in a textbook of catastrophes and the trader who is still at the desk a decade later, both of whom may have started with the identical edge. The only difference between the two columns is the size of the bet, and the size of the bet is the one thing entirely within your control before the streak arrives.

## The bet-sizing playbook

The math reduces to a short, blunt checklist. Run it before you size anything.

**1. Set a ruin budget first — before you think about returns.** Pick the probability of catastrophic loss you can live with as a hard ceiling. For most traders 1% is a sensible budget; for a fund with outside capital and redemption risk, often less. This number is a *decision*, made in calm, not a market output. Everything else is derived from it.

**2. Estimate your edge honestly, then discount it.** You need a per-bet win probability p (or, more generally, an edge) to feed the formula. Estimate it from real, out-of-sample results, not hope — and then shade it *down*, because measured edges are usually optimistic and ruin is hypersensitive to p. A one-point error in p moves ruin enormously, so be conservative on the input.

**3. Back out the max bet from the budget.** Compute the required bankroll depth i\* = ⌈ln(B)/ln(r)⌉ with r = q/p, and set your maximum bet fraction at f_max = 1/i\*. On the \$100,000 account at a 51% edge with a 1% budget, that was \$862 per trade; on the \$10,000,000 book at a 55% edge it was about \$434,783. This is a *ceiling*, not a target — you may bet less, never more.

**4. Cross-check against the streak you must survive.** Confirm your bankroll is deep enough in units to ride out a plausible worst losing streak (a k-unit bankroll survives k losses in a row). If the two checks disagree, take the smaller bet. They usually agree, because both are readings of your depth in units.

**5. Stay in the safe zone, below the cliff.** Keep your bet well left of the growth-optimal (full-Kelly) point — half-Kelly or less is the practitioner default — because the penalty for betting too small is trivial and the penalty for betting too large is catastrophic. The asymmetry only points one way.

**6. Recompute on every equity change, and install a kill-switch.** Bankroll depth in units changes as your account changes; a drawdown *shrinks* your depth and *raises* your ruin probability at a fixed dollar bet, so re-derive your max bet as equity moves. And set a hard drawdown kill-switch — a level (say −20%) at which you cut size or stop entirely — so that a bad streak can never silently walk you off the cliff while you tell yourself the edge will come back. The edge can only come back if you are still solvent to deploy it.

The discipline underneath all six rules is a single reframing, and it is the one the survival thesis of this whole series rests on.

![Before and after comparison of betting for return by maximizing each bet versus betting to keep ruin below budget by backing the max bet out of a chosen ruin budget](/imgs/blogs/the-gamblers-ruin-and-bet-sizing-the-math-of-staying-solvent-7.png)

The figure contrasts the two mindsets. On the left, *betting for return*: the goal is the biggest expected gain on this trade, you size by the edge and bet near or past full-Kelly, no ruin budget is ever set, and one bad streak touches the barrier. On the right, *betting to keep ruin below budget*: the goal is to hold P(ruin) under your chosen ceiling first, you back the max bet out of that budget, you size at or below the cap, and streaks stay survivable while the edge compounds for years. Same edge, same account, opposite outcomes — because survival was the objective instead of an afterthought. That is the whole job: keep the probability of touching zero below a budget you choose, and only then go collect your edge.

### Further reading

- [Risk of ruin: why positive expectancy is not enough](/blog/trading/risk-management/risk-of-ruin-why-positive-expectancy-is-not-enough) — the conceptual groundwork this post turns into rules: the absorbing barrier and why edge alone doesn't save you.
- [The Kelly criterion: how much to bet when you have an edge](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge) — the growth-optimal sizing rule and why practitioners bet a fraction of it, the natural complement to the ruin-budget approach here.
- [Leverage and the arithmetic of ruin](/blog/trading/risk-management/leverage-and-the-arithmetic-of-ruin) — how borrowing cuts your effective bankroll depth in units and multiplies ruin.
- [Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) — the heavier sequential-betting math behind growth-optimal sizing, for the derivations.
- [Position sizing and risk of ruin in options trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) — how the same ruin arithmetic applies when the payoff is an option's asymmetric, fat-tailed distribution rather than a coin flip.
