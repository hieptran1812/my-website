---
title: "Case Study: How a Regime Change Quietly Killed a Winning Strategy"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A mean-reversion strategy prints a smooth, beautiful equity curve in a choppy range -- high win rate, small frequent wins -- and then the same unchanged rules buy every dip into a falling knife once the market trends one way, and the curve cliffs. This is the whole life of that strategy, with illustrative R-multiple numbers: why the good years felt like skill, why each losing trade looked normal, why the backtest lied, and how to detect a regime change instead of mistaking a dead edge for an ordinary drawdown."
tags:
  [
    "regime-change",
    "mean-reversion",
    "market-regime",
    "technical-analysis",
    "backtesting",
    "expectancy",
    "r-multiple",
    "overfitting",
    "drawdown",
    "risk-management",
    "trading-strategy",
    "case-study",
  ]
category: "trading"
subcategory: "Technical Analysis"
author: "Hiep Tran"
featured: true
readTime: 51
---

> [!important]
> **TL;DR** -- a winning mean-reversion strategy did not stop working because the trader made a mistake. It stopped working because the market underneath it changed from a choppy range into a one-way trend, and the exact same rules that banked small frequent wins in the range now buy every dip into a falling knife.
>
> - **In the range it printed beautifully.** Fade the oscillator extremes, buy the dip, take profit at the middle of the band. About a 70% win rate, small +0.6R winners against -1R losers, a smooth rising equity curve, +12R over the good period. It felt like skill.
> - **The regime changed and the same rules turned lethal.** Price broke its band and trended one way. Each individual trade still looked normal at entry, so the trader did not notice -- but the win rate collapsed and the rare big loss that a high-win-rate edge always carries arrived again and again, stacking into -15R over the next stretch.
> - **The backtest lied because it only covered the ranging regime.** A test that samples one regime and reports +0.4R per trade is overfit to that regime -- a hidden form of overfitting -- and says nothing about the regime it never saw, where the same rules ran -0.2R.
> - **Strategies are regime-conditional.** Every edge needs a particular kind of market. Mean-reversion needs oscillation; it blows up in a trend. The honest job is to monitor the regime tells (structure, volatility, autocorrelation), cut size when the edge degrades, and tell a normal drawdown apart from a dead edge instead of assuming the slump will pass.
> - This is educational, not advice. All R-multiple and equity numbers below are **illustrative** -- round figures chosen to make the mechanism legible, not a record of any specific account.

Here is a story that has happened to more traders than almost any other, and it almost never gets told honestly because it has no villain. A trader builds a mean-reversion strategy. It works. For a year and a half the equity curve climbs in a smooth, almost boring line -- a string of small wins, the occasional small loss, drawdowns so shallow they barely register. The trader feels, reasonably, like they have figured something out. Then, over a few weeks, the same strategy gives back more than a year of gains. The trader did not change a single rule. They did not get sloppy or emotional or unlucky in any obvious way. Each trade they took looked exactly like the trades that had been winning. And yet the account bled out.

What killed the strategy was not a flaw in the rules. It was a change in the *market* the rules were running on. The strategy was a key cut precisely for one lock, and one day the lock was quietly swapped for a different one. The key did not break. It just no longer fit. This is the single most under-appreciated failure mode in all of technical analysis, and the whole point of this post is to walk one strategy through its entire life -- birth, glory, death -- so you can see the mechanism with your own eyes and recognize it before it costs you a year.

![An equity curve that climbs smoothly to plus twelve R over sixty ranging trades and then cliffs to minus fifteen R over twenty trend trades after a single amber regime-change line, while the rules never change.](/imgs/blogs/case-study-regime-change-killed-a-strategy-1.png)

The diagram above is the whole post in one picture, and we will keep coming back to it. On the left, the good years: sixty closed trades, about a 70% win rate, a smooth green climb to +12R. Then a single amber line marks the moment the market's behavior changed from a range to a trend. The rules did not move. The strategy file is byte-for-byte identical on both sides of that line. But on the right, the curve cliffs -- twenty trades, the win rate collapses, the losers balloon, and the account gives back -15R, more than the entire good run. Notice the annotation under the green block: *each individual losing trade still looked normal*. That is why the trader did not pull the cord. Nothing screamed. The disaster arrived as a series of ordinary-looking trades.

A note before we start: this is an educational piece about how a class of strategy behaves across market regimes, not advice to trade any of it. Every number here -- the win rates, the R-multiples, the equity totals -- is **illustrative**, chosen round and clean so the mechanism is easy to follow, not lifted from any particular account or instrument. The mechanism is real and recurring; the specific figures are a teaching scaffold.

## Foundations: two regimes, two opposite edges

Before we can watch a strategy die, we need to define -- from zero, assuming no trading background -- what the strategy *is*, what it *needs* to work, and what a "regime" even means. We will build every term as we go.

### What a strategy and an edge are

A **trading strategy** is a set of rules precise enough that you (or a computer) could follow them with no judgment: when to enter a position, where to place a *stop-loss* (the price at which you admit you were wrong and exit for a small, pre-defined loss), and when to take profit. A strategy has an **edge** if, over many trades, its wins outweigh its losses on average. An edge is not a promise about any single trade -- markets are far too noisy for that. It is a statistical tilt: repeat the same kind of trade a thousand times and you come out ahead.

We measure that tilt with one number, **expectancy** -- the average profit or loss per trade, counting winners and losers together. A strategy makes money over time if and only if its expectancy is positive after costs. There is a [full treatment of why expectancy matters and why win rate alone lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) in a companion post; for now hold the one-line version: *the average outcome per trade is what determines whether you make money, and that average is built from how often you win and how much you win versus lose.*

### The R-multiple: one ruler for every trade

To compare trades and strategies fairly, we put everything on a common ruler called the **R-multiple**. **R is the amount you risk on a single trade** -- the distance from your entry price to your stop-loss, times your position size. If you buy at \$100 and your stop is at \$98, then R = \$2 per share. A trade that makes \$4 per share is a +2R win. A trade that loses the full \$2 is a -1R loss. A trade you exit early for \$1.20 is a +0.6R win.

The beauty of R-multiples is that they strip out dollars and position size, so a trader risking \$100 a trade and a fund risking \$10 million a trade can be compared line for line. Throughout this case study, we talk almost entirely in R. When we say the strategy made "+12R over the good period," we mean it earned twelve times its per-trade risk -- if that risk was \$100 a trade, that is \$1,200; if it was \$10,000 a trade, that is \$120,000. The shape of the story is the same at any size.

### A regime is the market's current behavior

A **market regime** is the kind of behavior price is exhibiting over a stretch of time -- the character of the tape, not its level. There are two regimes that matter for this whole post, and they are opposites.

A **range** (also called a sideways or choppy market) is when price oscillates between a rough floor and a rough ceiling without going anywhere on net. It bounces up, bounces down, and keeps returning to a middle. Months can pass and price ends roughly where it started, having traveled a great deal in between. In a range, a move that stretches too far tends to *snap back* -- the statistical property here is **negative autocorrelation**, a technical way of saying today's move is, on average, slightly predictive of *the opposite* move tomorrow. Up days tend to be followed by down days. That alternation is what makes a range a range.

A **trend** is the opposite: price moves persistently in one direction, making higher highs and higher lows (an uptrend) or lower highs and lower lows (a downtrend), and the move *keeps going* far longer than feels reasonable. The statistical property is **positive autocorrelation** -- today's move is, on average, slightly predictive of *the same* move tomorrow. Down days cluster with down days. That clustering is what makes a trend a trend.

### Two edges, exact opposites

There are exactly two fundamental ways to extract money from price movement, and they map onto these two regimes as mirror images. This is the deepest idea in technical analysis, and there is a full companion post on [trend-following versus mean-reversion and why the same setup wins in one regime and dies in the other](/blog/trading/technical-analysis/trend-following-vs-mean-reversion). The short version, because the whole case study turns on it:

- **Trend-following** bets the move continues. When price is rising and strong, a trend-follower *buys* -- buying strength, expecting more strength. It needs a trend (positive autocorrelation). It wins a minority of the time (often 35-45%) but its winners dwarf its losers, so its equity curve is a choppy grind punctuated by big jumps.
- **Mean-reversion** bets the stretch snaps back. When price has dropped sharply and looks oversold, a mean-reversion trader *buys the dip* -- fading the move, expecting a bounce back toward the average. It needs a range (negative autocorrelation). It wins a majority of the time (often 65-75%) but its losers, though rare, are larger than its wins, so its equity curve glides up smoothly and then *cliffs*.

Stand at one moment in time -- price has just dropped hard and an oscillator reads "oversold." A trend-follower looking at that exact chart wants to *sell* (the weakness confirms the downtrend continues). A mean-reversion trader looking at the exact same chart wants to *buy* (the stretch should snap back). They take opposite sides of the same trade at the same price, and *both can be right* -- just not in the same regime. Which one is correct is decided not by the signal in front of you but by the regime the market is in.

Our trader runs a **mean-reversion strategy**. It is built to fade extremes and buy dips. It needs a range. Remember the cliff in mean-reversion's equity curve -- *smooth climb, then a sharp cliff*. That cliff is not a footnote. It is the whole story. The cliff is what happens when the regime mean-reversion needs disappears and the regime that destroys it arrives.

## The good years: mean-reversion in a range

Let us meet the strategy in its prime. We will write its rules precisely, watch its equity curve, and -- this is the important part -- understand exactly *why it felt like skill* when it was really regime-luck.

### The rules

Our trader's strategy is a textbook mean-reversion system on a liquid instrument -- say a large stock index ETF or a major currency pair, the kind of thing that spent a long stretch chopping sideways. The rules, in plain language:

1. **Bias / universe.** Trade one liquid instrument that has been ranging -- oscillating inside a visible band -- for the lookback period.
2. **Signal.** Use a momentum oscillator, the **Relative Strength Index (RSI)** -- a number from 0 to 100 that measures how stretched recent up-moves are versus down-moves (there is a [full explainer on RSI and momentum oscillators](/blog/trading/technical-analysis/rsi-and-momentum-oscillators) if it is new to you). An RSI below 30 is conventionally "oversold"; above 70 is "overbought."
3. **Entry.** When RSI drops below 30 *and* price touches the lower edge of the range (a common second filter is the lower **Bollinger Band**, a volatility band two standard deviations below a moving average), **buy the dip**.
4. **Stop-loss.** Place the stop 1R below the entry -- far enough to survive normal noise, close enough that a real breakdown costs only one unit of risk.
5. **Target.** Exit when price reverts to the middle of the range (the moving average), typically a **+0.6R** gain.
6. **Size.** Risk a fixed fraction of the account per trade -- say 1% -- so every trade risks the same 1R.

That is the entire strategy. It is simple, mechanical, and -- crucially -- it has a real edge *in a range*. The negative autocorrelation of a ranging market means that a price stretched to its lower band, with RSI oversold, genuinely does tend to bounce. The trader is selling insurance against panic and getting paid a small premium each time the panic proves overdone.

It is worth pausing on *why* this edge exists at all, because understanding the mechanism is the only way to predict when it disappears. In a ranging market with no new directional information, price is pushed around by the ordinary noise of order flow -- a large seller needs to liquidate, a piece of short-term news gets overreacted to, a cluster of stop-losses gets triggered. Each of these shoves price away from where supply and demand are actually balanced. Market-makers and liquidity providers lean against those shoves: they sell into spikes and buy into dips to capture the *bid-ask spread* (the small gap between the price you can buy at and the price you can sell at), and their buying-the-dip pushes price back toward the balance point. Our trader's strategy is, in effect, riding alongside those liquidity providers -- buying the same dips, betting on the same reversion to balance. The edge is real because the mechanism is real: in a range, there *is* a balance point, and stretches away from it *are* corrected. The instant that stops being true -- the instant new directional information means the old balance point is wrong and price is *supposed* to move to a new level -- the mechanism reverses, and the trader is now leaning against a move that has a reason to continue.

This is the single most important sentence in the foundations: *the mean-reversion edge depends on there being a stable balance point that stretches snap back to.* A trend is, by definition, the market repricing to a new balance point -- there is no fixed level to revert to, because the level itself is moving. The same dip-buying that catches reversion to a stable level catches a falling knife when the level is collapsing. Hold that, and everything that follows is just watching the consequence play out.

### The equity curve that felt like skill

![Left panel, a ranging price tape oscillating between a dashed sell edge and a dashed buy edge so every dip reverts to the band; right panel, the band breaks and price trends one way so each dip sets a lower low.](/imgs/blogs/case-study-regime-change-killed-a-strategy-2.png)

The left panel above is the market the strategy was born into. Price oscillates between a ceiling ("sell edge") and a floor ("buy the dip"), and every stretch toward an edge snaps back toward the middle. In this tape, the rules are a money machine. Look at the right panel for a moment too -- that is the future, the regime change we are building toward, where the band breaks and price trends one way. But for now, we live in the left panel.

Here is what the equity curve looked like over the good period -- and remember, these numbers are **illustrative**. Over roughly eighteen months the strategy took about 60 trades. It won about 70% of them. The winners were small and consistent at +0.6R; the losers, the 30% that did not revert in time, cost the full -1R. The result was a smooth, rising line. Let us compute exactly why it rose, because the math is the heart of why this strategy is dangerous *precisely because it works so well*.

#### Worked example: the mean-reversion expectancy in the range

Let us compute the strategy's expectancy in the ranging regime, step by step. Expectancy in R-multiples is:

$$E[R] = (P_{win} \times R_{win}) - (P_{loss} \times R_{loss})$$

where $P_{win}$ is the probability of a win, $R_{win}$ is the average winner in R, $P_{loss}$ is the probability of a loss, and $R_{loss}$ is the average loser in R (written as a positive number we subtract).

Plug in the good-years numbers (illustrative):

- $P_{win} = 0.70$ (wins 70% of the time)
- $R_{win} = +0.6$ (average winner)
- $P_{loss} = 0.30$ (loses 30% of the time)
- $R_{loss} = 1.0$ (average loser, full stop-out)

So:

$$E[R] = (0.70 \times 0.6) - (0.30 \times 1.0) = 0.42 - 0.30 = +0.12 \text{ R per trade}$$

Every trade, on average, earns +0.12R. That does not sound like much, but over 60 trades it compounds into a meaningful number, and the curve is *smooth* because most trades are small wins. Run it forward: 60 trades $\times$ +0.12R $\approx$ +7.2R from the raw expectancy. In a particularly clean ranging stretch the realized number ran higher -- our illustrative curve hits **+12R** over the good period -- because the trader caught a run of reversions with very few stop-outs, which is exactly the kind of good variance a high-win-rate strategy throws off when its regime is cooperating. **The intuition this teaches:** a mean-reversion edge is *small per trade and positive*, and it pays out as a steady drip of little wins -- which is exactly what makes it feel like a smooth, reliable skill rather than a statistical tilt that depends entirely on the market staying in a range.

### Why a smooth curve is so seductive -- and so misleading

The smoothness is the trap. Human beings read a smooth rising line as *competence* and a jagged one as *luck*. This is backwards in trading. A smooth mean-reversion curve is smooth because the strategy wins often and small -- and a strategy that wins often and small is *carrying a fat tail it has not paid yet*. The 70% win rate is real, but it is financed by the 30% losers and, lurking underneath, by the rare catastrophic loss that arrives only when the regime breaks. During the good years that catastrophe simply has not happened, so the curve looks even better than the edge justifies.

Three psychological forces conspire here, and they matter because they are why the trader will fail to react when the regime turns:

- **Recency and confirmation.** Each small win confirms the strategy works. Sixty confirmations in a row builds near-total conviction. The trader is not being stupid; they are being *Bayesian on a biased sample*.
- **Attribution.** A run of wins gets attributed to skill ("I read the market well"), not to regime ("the market was in the one state my strategy needs"). This is the single most expensive misattribution in trading.
- **Anchoring on the shallow drawdown.** The worst drawdown in the good period was tiny -- maybe -2R, a couple of consecutive stop-outs. The trader anchors on that as "the worst it gets," and sizes and emotionally prepares for a -2R world. The regime change will hand them a -15R world.

The honest framing the trader *should* have held -- and almost nobody does -- is this: *I have a real edge, conditional on the market staying in a range. My smooth curve is evidence of the edge and evidence that the range has persisted. It is not evidence that the range will continue, and my strategy has no defense if it doesn't.* That sentence, internalized, is the difference between giving back -2R and giving back -15R.

There is a deeper statistical point hiding in the smoothness, and it is worth making precise because it is the reason the danger is *invisible* rather than merely ignored. A strategy's equity curve has two properties that beginners conflate: its *expected return* and its *risk*. A smooth, high-win-rate curve has low *visible* risk -- the day-to-day variation is small, the drawdowns are shallow, the curve looks almost like a savings account. But the *true* risk of a mean-reversion strategy lives almost entirely in the tail -- the rare large loss -- and the tail is, by definition, the part you have not observed yet during a good run. So the very smoothness that the trader reads as "low risk" is actually "risk that has not yet been realized." Statisticians call this the difference between *realized* volatility and *tail* risk, and the gap between them is exactly the size of the disaster waiting in the wrong regime. A strategy that looks safest by its observed track record can be the one carrying the most unpaid tail -- and a high-win-rate mean-reversion strategy in a persistent range is the textbook case of a curve that looks safe precisely because it is about to be expensive.

Consider the asymmetry one more way, with the numbers in front of you. To make +0.6R, the strategy needs price to revert -- a modest, common event. To lose -1R (or -2R or -3R), the strategy needs price to *not* revert and keep going -- a rarer event in a range, but the *defining* event in a trend. So the payoff is structured so that the common outcome is a small win and the rare outcome is a larger loss. In a range, the common outcome dominates and the curve climbs. The strategy is, in the most literal sense, *picking up small coins in front of a steamroller that is currently parked.* During the good years the steamroller is parked, and picking up coins is pleasant and profitable. The regime change is the steamroller starting to move. Nothing about the coins changed; the steamroller did.

## The regime change: the market goes one-way

Now the lock gets swapped. Over a few weeks -- there is rarely a bell that rings -- the market the strategy needs disappears and the market that destroys it arrives. To make this concrete and recognizable, anchor it (as an illustrative scenario) to a **buy-the-dip book meeting a sustained downtrend**, the way countless mean-reversion approaches that thrived in the calm, range-bound stretch of 2020-2021 ran straight into the relentless one-way decline of **2022**, when major equity indices fell through level after level for most of the year. (As of mid-2026 that 2022 drawdown is the canonical modern example; the mechanism, not the specific year, is the lesson.)

### Each trade still looks normal

This is the cruel part, and the reason the trader does not notice. Re-read the strategy rules: *RSI below 30, price at the lower band, buy the dip, stop 1R below, target the middle.* When a downtrend begins, those conditions fire *constantly* -- a falling market is oversold almost all the time. So the strategy keeps taking trades, and at the moment of entry, **each trade looks identical to the winning trades**. Same oscillator reading, same band touch, same setup on the chart. There is no signal that says "this dip is different." The trader, following the rules they have followed sixty profitable times, buys.

![One buy-the-dip rule with two outcomes shown as before and after columns: in a range the dip reverts to a plus zero point six R win; in a trend the same dip is a falling knife that runs to a minus two or minus three R loss.](/imgs/blogs/case-study-regime-change-killed-a-strategy-3.png)

The figure above puts the two outcomes of *the identical rule* side by side. On the left, the range: RSI prints 28, the rule fires, price reverts to the middle, exit at +0.6R, and this happens about 70% of the time -- small frequent wins. On the right, the trend: RSI prints 28 (same reading), the same rule fires, but price *keeps falling*, gaps through the stop, and the trader exits late for a -2R or -3R loss -- and this happens trade after trade. The entry signal is byte-for-byte the same. Only the market's response is different, and the market's response is the only thing that pays you.

### The rare big loss the edge always carried

Recall from the foundations that mean-reversion's signature is "wins often and small, loses rarely but large." During the good years, the "loses rarely but large" part was dormant -- the losers were just -1R stop-outs, not catastrophes. In a trend, that dormant tail wakes up and becomes the *main event*. Why does the loss balloon from -1R to -2R or -3R?

Two mechanisms, both mechanical:

- **The stop gets run, and then some.** In a range, the -1R stop is rarely hit because price reverts before reaching it. In a trend, price blows straight through the stop. If the trader's exit slips (gaps open below the stop, fast market, the stop is a mental one they hesitate on), the realized loss is worse than the intended -1R -- it becomes -2R or -3R.
- **Averaging down.** Many discretionary mean-reversion traders, conditioned by a range where dips always recover, *add to the position* as it falls ("it's an even better discount now"). In a range this is genius; in a trend it converts a -1R loss into a -3R or worse loss by buying more of a falling knife. Even a strictly mechanical trader who never averages down still suffers the gap-through losses; the discretionary one suffers both.

So the rare big loss the high-win-rate edge always carried -- the price of selling insurance against panics -- finally comes due, and it comes due not once but repeatedly, because in a trend *every* dip is the panic that does not recover.

There is a third, subtler mechanism that compounds the first two: **the trader's own conditioning works against them.** Eighteen months of being rewarded for buying dips has trained a reflex. The reflex says "dip equals discount equals buy," and it fires faster than conscious deliberation. When the regime turns, the reflex does not turn with it -- it keeps firing on exactly the same visual cue (price down, oscillator oversold), and now it is firing into a trend. So even a trader who *intellectually* knows about regime risk finds themselves taking the trades anyway, because the pattern-recognition that made them money is now an automated liability. This is why "just be aware of regime risk" is insufficient as a defense; awareness lives in the slow, deliberate part of the mind, and the trades are placed by the fast, conditioned part. The defense has to be *mechanical* -- a regime filter that refuses to let the entry rules fire, or a size cut that runs automatically when the tells flash -- precisely because willpower arrives too late.

It is also worth being honest about the *timing*. Regime changes do not announce themselves with a clean break. The first lower low after a long range looks, at the moment, exactly like the dozens of dips that recovered before it. The second lower low looks like a slightly deeper dip. By the time the pattern is unambiguous -- five or six lower lows in a row, the band decisively broken -- the trader has already taken several losing trades and is several R into the drawdown. So even perfect monitoring does not catch the regime change at trade one; it catches it at trade three or four, after a few real losses. The goal of detection is not to avoid all damage -- that is impossible -- but to convert a -15R catastrophe into a -3R or -4R early exit. The difference between those two outcomes is not foresight; it is *reaction speed*, and reaction speed is exactly what a mechanical overlay buys you.

#### Worked example: the same rules in the trend

Let us recompute expectancy with the trend-regime numbers (illustrative) and watch the edge flip sign. Same formula:

$$E[R] = (P_{win} \times R_{win}) - (P_{loss} \times R_{loss})$$

Now plug in the trend regime:

- $P_{win} = 0.30$ (the win rate collapses -- in a downtrend, only the occasional dip bounces enough to hit the +0.6R target before resuming down)
- $R_{win} = +0.6$ (the winners, when they happen, are unchanged -- same target)
- $P_{loss} = 0.70$ (loses 70% of the time now)
- $R_{loss} = 2.5$ (the average loser balloons -- stops run, slippage, the occasional -3R; call it -2.5R on average)

So:

$$E[R] = (0.30 \times 0.6) - (0.70 \times 2.5) = 0.18 - 1.75 = -1.57 \text{ R per trade}$$

The expectancy has not just dropped -- it has *inverted*, from +0.12R to roughly **-1.57R per trade** at the worst of it. (Across the whole trend stretch, blending the early trades where losses were nearer -2R, the realized average works out closer to **-0.75R per trade** over the 20 trades, which is how -15R accumulates -- 20 $\times$ roughly -0.75R. The exact blend is illustrative; the sign and the magnitude of the flip are the point.) **The intuition this teaches:** an edge built on a high win rate is brittle, because when the regime flips, *both* the win rate falls *and* the loss size grows at the same time -- the two terms that protected you both turn against you at once, so the expectancy doesn't sag, it collapses through zero.

#### Worked example: the equity curve, good period then trend

Now let us assemble the full equity curve that the first figure shows, trade block by trade block, in R (illustrative throughout).

**The good period (60 trades, ranging):**

- Wins: 60 $\times$ 0.70 $\approx$ 42 winners $\times$ (+0.6R) = +25.2R
- Losses: 60 $\times$ 0.30 $\approx$ 18 losers $\times$ (-1.0R) = -18.0R
- Net over the good period: +25.2R - 18.0R = **+7.2R** from base expectancy

In our illustrative curve the trader's realized result over the good period was a bit luckier than base expectancy -- a clean stretch with fewer stop-outs -- landing at **+12R**. The account is up twelve units of risk, the curve is smooth, and confidence is total.

**The trend period (20 trades, one-way down):**

- Wins: 20 $\times$ 0.30 = 6 winners $\times$ (+0.6R) = +3.6R
- Losses: 20 $\times$ 0.70 = 14 losers $\times$ (average -1.4R, blending early -2R+ knives with some that hit the intended stop) = -19.6R
- Net over the trend period: +3.6R - 19.6R = **-16.0R**, which we round to the illustrative **-15R** the figure shows.

**The round trip:** +12R, then -15R, for a net of -3R *below where the strategy started* -- and a peak-to-trough drawdown of 27R (from +12R down to -15R). The trader did not just give back the good year; they ended underwater, having handed back everything plus more. **The intuition this teaches:** a high-win-rate strategy can spend eighteen months building a smooth +12R and then erase all of it plus more in a few weeks, because the gains came in as +0.6R drips and the losses come out as -2R and -3R gushes -- the asymmetry that was invisible in the range is the entire story in the trend.

### Why the alarm never rang

Step back and notice what *did not* happen. There was no single -10R trade that would have screamed "stop." There was no rule violation. There was no day the trader could point to and say "that was the disaster." The -15R arrived as roughly fourteen losing trades each of which, individually, looked like a normal trade that just happened to lose -- a -2R here, a -3R there, a couple that bounced and won to keep hope alive. The disaster was distributed across many ordinary-looking events, which is exactly why human risk-detection fails on it. We are wired to notice the single dramatic loss, not the slow accumulation of slightly-larger-than-usual losers that, summed, are catastrophic. The regime change is invisible at the resolution of a single trade and only visible at the resolution of the equity curve and the regime tells -- which we will get to.

The "couple that bounced and won to keep hope alive" deserves its own emphasis, because it is the cruelest feature of the whole episode. Even in a strong downtrend, price does not fall in a straight line -- it has counter-trend bounces, and some of them are sharp enough to hit the +0.6R target before the decline resumes. So the trader, in the middle of the bleed, gets the occasional win. That win is poison, because it *confirms the strategy still works* at exactly the moment the trader most needs to conclude it has stopped working. A pure string of losses would, eventually, force a rethink; an intermittent reinforcement schedule -- mostly losses with the occasional win -- is the most behaviorally sticky pattern there is, the same schedule that makes slot machines compulsive. The trader keeps pulling the lever because it pays off just often enough to sustain the belief. The intermittent winner is not evidence the edge survives; it is the noise of a counter-trend bounce in a regime where the edge is dead, and mistaking it for the former is how the drawdown runs all the way to -15R instead of stopping at -4R.

There is one more thing the trader could have watched that the trades themselves obscured: the *shape* of the losers. In the good years, losses clustered tightly at -1R -- the stop did its job. In the trend, the losers spread out: some -1.5R, some -2R, a couple -3R. That *widening distribution of loss sizes* is itself a tell, independent of the win rate. A strategy whose losses are growing larger than its design even while it occasionally still wins is a strategy whose stop is no longer protecting it -- which is precisely what happens when price gaps through the stop in a fast one-way move. Watching the *largest* recent loss, not just the average, is a cheap and early alarm: the first -3R in a strategy designed for -1R losses is a louder signal than three more -1R losses, because it says the failure mode has changed character, not just frequency.

## Why the backtest lied

Here is the question that haunts every trader after a blow-up: *the backtest looked great -- how did it not warn me?* The answer is the deepest lesson in this whole case study, and it connects directly to the post on [backtesting without fooling yourself](/blog/trading/technical-analysis/backtesting-without-fooling-yourself). The backtest did not lie about the data it saw. It lied by *omission*: it only ever saw one regime.

### A backtest only tells you about the regime it sampled

A **backtest** replays your rules over historical price data and reports what would have happened. Its output is only as representative as its data. If you build and test a mean-reversion strategy on three years of data, and those three years happened to be a long range, your backtest will report a beautiful positive expectancy -- and it will be *correct about that history*. What it cannot tell you is anything about how the rules behave in a regime that is not in the sample. A test run on range-only data is silent about trends the way a test of a car only ever driven in summer is silent about ice.

![A backtest equity curve climbing on range-only data to plus zero point four R per trade, then a dashed amber live-start line after which a red live curve falls into a trend the test never sampled, ending at minus zero point two R per trade.](/imgs/blogs/case-study-regime-change-killed-a-strategy-5.png)

The figure shows the trap precisely. The blue backtest window covers three years of range data and the green curve climbs to a reported **+0.4R per trade**. Then the dashed amber line marks the start of live trading. Soon after, the regime flips, the red live curve falls, and live trading prints **-0.2R per trade** -- a regime the backtest never sampled and therefore could never warn about. The backtest was not fraudulent. It was *incomplete*, and incomplete in the one way that mattered.

### Regime-overfitting is a hidden form of overfitting

You have probably heard of **overfitting** -- tuning a strategy's parameters until it fits the past so tightly that it has memorized noise rather than captured a real edge, so it looks perfect in-sample and fails out-of-sample. The classic overfitting is *parameter* overfitting: turn enough knobs (the RSI threshold, the band width, the lookback) and you can make any past look great. The post on [the indicator trap](/blog/trading/technical-analysis/the-indicator-trap) walks through how indicator-tuning produces exactly this illusion, and the post on [backtesting without fooling yourself](/blog/trading/technical-analysis/backtesting-without-fooling-yourself) covers the honest workflow that detects it.

But there is a subtler, sneakier form: **regime overfitting**. Even if you never touch a parameter -- even if your strategy has zero free knobs and you test it honestly with strict in-sample and out-of-sample splits -- you can still be overfit *to a regime* if your entire dataset, in-sample and out-of-sample alike, is one regime. Splitting range-only data into in-sample and out-of-sample does not help, because both halves are ranges. Your "out-of-sample" test confirms the strategy works in a range, which you already knew. It is overfitting in the sense that matters -- you have fit a model to a feature of the world (the range) that is not permanent, and you have mistaken it for a feature that is (a durable edge).

This is why a strategy can pass every standard backtesting check -- no look-ahead bias, no survivorship bias, realistic costs, clean walk-forward -- and still die. The standard checks protect you from *lying about the data you have*. They do nothing about *the regime your data doesn't contain*. The only defenses against regime overfitting are (a) deliberately testing across data that spans *both* regimes (find the trends in history and see what your range strategy does in them), and (b) understanding the *mechanism* of your edge well enough to know which regime it needs -- so you can predict, not just measure, when it will fail.

#### Worked example: the backtest's +0.4R versus the live -0.2R

Let us make the omission concrete with numbers (illustrative). Suppose the backtest covered three years that were 100% ranging, and over 200 simulated trades it produced:

- Win rate: 70%, average win +0.6R, average loss -1.0R
- Expectancy: $(0.70 \times 0.6) - (0.30 \times 1.0) = 0.42 - 0.30 = +0.12$R per trade on base assumptions; with the clean ranging stretch the realized figure ran to about **+0.4R per trade** in the rosiest sub-windows the trader fixated on.

The trader reads +0.4R per trade, multiplies by their expected trade frequency, and projects a wonderful year. Now live trading begins, and after a few months the regime flips for the back half of the live sample. Blending the early live range trades (still +0.4R-ish) with the trend trades (about -1.5R each), the *realized live expectancy* over the full live period works out to roughly **-0.2R per trade**. The strategy went from a reported +0.4R to a lived -0.2R -- a swing of 0.6R per trade -- without a single rule changing, purely because the live period contained a regime the backtest's history did not. **The intuition this teaches:** a backtest number is a measurement of the past regimes in your sample, not a prediction of the future; +0.4R "in-sample on a range" and -0.2R "live across a regime change" are both true, and the gap between them is the size of the regime your test never saw. (For why the +0.4R figure itself can mislead even within one regime, see [expectancy and why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies).)

### The two expectancies side by side

![A matrix comparing the same rules across range and trend regimes, with win rate, average winner, average loser, expectancy per trade, and verdict rows, showing plus zero point four two R in the range flipping to minus zero point two zero R in the trend.](/imgs/blogs/case-study-regime-change-killed-a-strategy-4.png)

The matrix above lays the two expectancies side by side, row for row, so you can see exactly which terms break. The win rate flips from 70% to 30%. The average winner is *unchanged* at +0.6R -- the target never moved. But the average loser balloons from -1.0R to -2.5R as stops get run and knives get caught. Run the arithmetic in each column and the bottom rows tell the story: **+0.42R per trade** in the range (a real edge) becomes **-0.20R per trade** in the trend (a bleeding loss). The verdict row says it plainly: a real edge becomes a bleeding loss, and the only thing that changed is the column header -- the regime.

Notice the structure of the failure once more, because it is the generalizable lesson: *two of the three terms turned against the trader simultaneously.* The win rate fell (fewer dips revert) and the loss size grew (the dips that don't revert run further). Only the winner size held. When a strategy's protection comes from a high win rate paired with controlled losses, a regime change is uniquely lethal because it attacks both pillars at once.

## Detecting and surviving a regime change

Everything so far has been diagnosis. Now the useful part: how would the trader have caught this, and what should they have done? The answer is not "be smarter at picking trades." Each trade was correctly selected by the rules. The answer is a layer *above* the trades -- a regime-monitoring and risk-management overlay that watches whether the strategy's required regime is still present and responds when it degrades.

### The regime tells: structure, volatility, autocorrelation

A regime change is not invisible at the level of the *market*; it is only invisible at the level of a single *trade*. There are observable tells, and the honest trader watches them every day, not just when the account is bleeding.

![A three-by-two grid of regime tells, with rows for market structure, volatility, and autocorrelation, and columns for the range tell with the edge alive in green and the trend tell with the edge dead in red.](/imgs/blogs/case-study-regime-change-killed-a-strategy-6.png)

The grid above lays out the three most reliable tells, each reading one way in a range (edge alive, green) and the opposite way in a trend (edge dead, red):

- **Market structure.** In a range, price holds a band -- highs cluster near a ceiling, lows near a floor, and neither marches in one direction. The tell that the regime is changing is *structure breaking*: a sequence of **lower lows** (in a forming downtrend) or higher highs (in an uptrend) that the range used to prevent. The first time price closes meaningfully below the floor that contained it for months and then makes *another* lower low instead of recovering, the range is on notice. There is a dedicated post on reading [trend and market structure](/blog/trading/technical-analysis/trend-and-market-structure) if you want the full toolkit. For our trader, the broken floor in the second figure's right panel is exactly this tell, and it fired before the worst of the -15R.
- **Volatility.** In a range, volatility is steady and mean-reverting -- a spike fades quickly, bars are roughly the same size. A regime change announces itself with **expanding, directional volatility**: the range bars get bigger *and* they point the same way. Rising volatility that is symmetric (big up and down bars) is just a noisier range; rising volatility that is *one-directional* (big down bars, small up bars) is a trend forming. The Bollinger Bands the strategy already uses are a volatility measure -- when the bands expand and price *walks down the lower band* instead of bouncing off it, that walk is the trend announcing itself. (See [Bollinger Bands and volatility](/blog/trading/technical-analysis/bollinger-bands-and-volatility) for why "walking the band" is the classic regime tell.)
- **Autocorrelation.** This is the most direct measure, because it is the literal statistical property that distinguishes the regimes. In a range, returns are *negatively* autocorrelated -- an up day tends to be followed by a down day. In a trend, returns are *positively* autocorrelated -- a down day tends to be followed by a down day. You can measure this on a rolling window (the correlation of each day's return with the prior day's, over the last 20 or 40 days). When that rolling autocorrelation crosses from negative to positive and stays there, the very thing your mean-reversion edge feeds on has reversed sign. That is the cleanest quantitative regime alarm there is.

No single tell is a perfect oracle -- markets are noisy, and you will get false alarms. But watching all three *together* is powerful: when structure breaks *and* volatility expands one-way *and* autocorrelation flips positive, the probability that you are in a genuine regime change rather than a normal wiggle is high enough to act on.

#### Worked example: the autocorrelation tell flipping sign

Let us make the autocorrelation tell concrete, because it is the cleanest quantitative alarm and the one most traders never compute. **Autocorrelation** here means: take each day's return, pair it with the *previous* day's return, and measure the correlation across a rolling window of, say, the last 20 days. The correlation is a number between -1 and +1.

In our trader's range, suppose the rolling 20-day autocorrelation hovered around **-0.25** for the whole good period. That negative number is the statistical fingerprint of mean-reversion: a positive return day tends to be followed by a negative return day, and vice versa. An autocorrelation of -0.25 is modest but persistent, and it is exactly the food the strategy eats -- it is *why* the dips revert.

Now watch what happens as the regime turns (illustrative numbers):

- Two weeks before the break, the rolling figure is still **-0.20** -- range intact, edge alive.
- As the first lower lows print, it drifts to **-0.05** -- the reversion is weakening; this is the first quantitative whisper.
- A week into the trend, it crosses to **+0.15** -- a down day is now followed by another down day; the regime has flipped sign.
- At the worst of the trend, it reads **+0.35** -- strong positive autocorrelation, the fingerprint of momentum, the precise opposite of what the strategy needs.

The moment that rolling number crossed from negative to positive and *stayed* there is the cleanest regime alarm the trader had. It is computable from price alone, it requires no judgment, and it can be wired into the strategy as a hard filter: *if rolling autocorrelation is positive, the mean-reversion entry rules do not fire.* **The intuition this teaches:** the regime your strategy needs is not a vibe -- it is a measurable statistical property of returns, and you can watch that property cross zero in real time and treat the crossing as the alarm that the slow, conditioned part of your brain will never sound on its own.

### Reduce size when the edge degrades

The first response to a possible regime change is not to flip your whole strategy on a hunch -- it is to **cut size**. The honest logic: you are uncertain whether the recent losses are normal variance or a real shift, and uncertainty itself is a reason to risk less. If your rule was to risk 1% (1R) per trade, cut to 0.5R the moment the tells start flashing and the trade results start to disappoint (win rate slipping under 55%, losers exceeding the -1R you modeled). This does two things: it mechanically halves the bleed while you figure out what is happening, and it removes the all-or-nothing pressure of a single big decision. You do not have to be *right* about the regime to benefit; you only have to be appropriately *humble* about your uncertainty. This connects to the broader discipline of [position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion): when your estimated edge falls, the mathematically correct bet size falls with it, and a degrading edge is exactly a falling estimated edge.

Consider how much this single discipline would have saved in our case study (illustrative). Recall the trend period was 20 trades for -15R at full 1R size. Suppose the trader had a rule: *after three consecutive losses larger than the modeled -1R, halve the size.* Those three losses might cost roughly -6R at full size before the rule triggers. But the remaining 17 trades, now at half size, would lose roughly half of what they did -- turning a -9R back-half into a -4.5R back-half. The total becomes about -6R minus -4.5R, roughly **-10.5R instead of -15R** -- and that is *without* the trader ever correctly diagnosing the regime, purely from a mechanical response to the loss pattern. Add a second rule that flattens entirely after the autocorrelation flips positive, and the total drops toward -5R or -6R. The size cut is not a magic exit; it is a *circuit breaker* that limits the damage of being wrong while you work out whether you are wrong. That is its whole value: it converts a decision you will make too late (stop trading the strategy) into a response that fires on time (risk less, automatically).

The reason size reduction is the *first* move, before any decision to abandon the strategy, is that it is robust to the one thing you cannot do reliably in the moment: tell variance from regime change with certainty. If the losses turn out to be normal variance, you cut size for a few trades, missed a little upside, and resumed -- a small cost. If the losses turn out to be a regime change, you cut size and dramatically limited the catastrophe -- a huge benefit. The expected value of cutting size on a degrading edge is positive *under both interpretations of the data*, which is exactly the property you want from a response made under uncertainty. Holding full size, by contrast, is only correct if it is definitely variance, and "definitely" is a word you are not entitled to use mid-drawdown.

### Telling a normal drawdown from a dead edge

Here is the hardest judgment in all of trading, and the one the trader in our story got wrong: *is this slump a normal drawdown that I should wait out, or has my edge died and I should stop?* Both look identical at the start -- a string of losses. The mistake is to assume, by default, that it is the former ("every strategy has drawdowns; I just have to be disciplined and keep taking the trades"). That assumption is correct in a range and lethal in a regime change.

![A matrix distinguishing a normal drawdown from a dead edge across rows for depth versus model, win rate, loss size, regime tells, and what to do, with normal-drawdown cells in green and dead-edge cells in red.](/imgs/blogs/case-study-regime-change-killed-a-strategy-8.png)

The matrix above gives you the discriminators. The key tool in the top row is the **Monte Carlo band** -- the range of equity outcomes you get when you take your backtested trades and *shuffle their order* thousands of times to see the realistic spread of drawdowns a strategy with your win rate and payoff naturally produces. (The [backtesting post](/blog/trading/technical-analysis/backtesting-without-fooling-yourself) covers this technique in depth.) A normal drawdown stays *inside* that simulated worst-run band; a dead edge breaks *beyond* every run the simulation ever produced. Read the matrix top to bottom:

- **Depth versus model:** normal drawdown is inside the worst Monte Carlo run; a dead edge is beyond every simulated run. This is your most objective check -- if your current drawdown is deeper than the 1-in-1,000 worst shuffle, something has changed that the backtest never contained.
- **Win rate:** in a normal drawdown the win rate is still near its modeled 70% -- you are just on the unlucky side of variance. In a dead edge it has collapsed under 40%.
- **Loss size:** normal losses are still around the -1R you designed for; dead-edge losses have ballooned to -2R and -3R.
- **Regime tells:** in a normal drawdown the market is still ranging (structure intact, autocorrelation still negative); in a dead edge the tells have flipped.
- **What to do:** if it is a normal drawdown, *hold size and keep going* -- discipline is correct here. If it is a dead edge, *stop and switch playbook* -- discipline applied to a dead edge is just a faster way to lose.

The trader in our story applied range-appropriate discipline ("drawdowns are normal, keep taking the trades") to a dead edge. The discipline was real and would have been correct in a range. It was the diagnosis that was wrong. The whole skill is in the diagnosis, and the diagnosis is exactly the four discriminators above.

### Have a trend playbook too

The deepest fix is structural: do not run only one edge. A trader who *only* knows how to mean-revert is defenseless in a trend by construction -- the only tool they have is the one the regime punishes. The mature answer is to hold *both* playbooks and switch based on the regime tells.

![A five-stage pipeline for surviving a regime change: monitor the tells daily, detect when the edge degrades, cut size, distinguish a drawdown from a dead edge, and switch to the trend playbook.](/imgs/blogs/case-study-regime-change-killed-a-strategy-7.png)

The pipeline above is the full detect-and-survive workflow, stage by stage. Monitor the tells daily (structure, volatility, autocorrelation). When the edge degrades -- win rate under 55%, losers exceeding the modeled -1R -- cut size while you decide whether it is noise or a shift. Use the Monte Carlo band and the regime tells to distinguish a normal drawdown from a dead edge. And if it is a genuine regime change, *switch playbooks*: in a trend, you stop buying dips and instead buy *pullbacks in the direction of the trend*, trail your stops wide to let winners run, and -- above all -- you **stop fading the extremes**, because fading a real trend is the single most expensive mistake there is. A trend-following overlay would have done more than save the -15R; in a sharp one-way move it would have *made* money on the very price action that was destroying the mean-reversion book. The trade you take at a broken-down lower low is the exact opposite under the two playbooks: the mean-reverter buys it (and dies), the trend-follower sells it short (and profits). Same price, opposite trade, and the regime decides who is right.

## Common misconceptions

Let us correct the beliefs that turn this recoverable situation into a blow-up. Each is something a thoughtful trader genuinely believes, and each is wrong in a specific, costly way.

**"A losing streak just means I have to wait it out."** This is true in a range and lethal in a regime change, and the whole skill is telling which one you are in. A normal drawdown -- a streak of losses inside the spread your strategy naturally produces -- absolutely should be waited out; bailing on a good strategy mid-drawdown is how amateurs destroy edges. But a regime change is *not* a drawdown; it is the disappearance of the conditions your edge requires. Waiting out a dead edge does not return you to the good times; it just funds the trend with your account. The discriminators in the dead-edge matrix -- the Monte Carlo band, the win rate, the loss size, the regime tells -- are precisely how you tell "wait it out" from "stop now." Defaulting to "wait it out" because that is the disciplined-sounding choice is how the -15R happened.

**"A good backtest means a durable edge."** A backtest measures the regimes in its sample, nothing more. A glorious backtest on three years of range data is genuine evidence that the strategy works *in a range* and zero evidence about how it behaves in a trend the sample did not contain. Durability is not something a single-regime backtest can establish, no matter how clean. The only paths to confidence about durability are testing across data that spans both regimes and understanding the *mechanism* of the edge well enough to predict which regime it needs. A backtest answers "did this work on this history"; durability asks "will this work across futures that include regimes this history lacked," and those are different questions.

**"The strategy stopped working for no reason."** It always has a reason, and the reason is almost never random. "For no reason" is what a regime change feels like from inside a single-strategy worldview, because nothing in the *strategy* changed -- so the cause must be external and is therefore invisible to someone looking only at their rules. But the market changed, observably, in structure and volatility and autocorrelation. The reason was there the whole time, written in the tape; it was just outside the frame the trader was looking at. "For no reason" is a confession that you were not monitoring the regime, not a fact about the market.

**"One strategy fits all markets."** No strategy fits all markets, because the two fundamental edges are *opposites* and the regimes that reward them are mutually exclusive. A strategy optimized to fade extremes will, by mathematical necessity, lose in a trend -- it is the precise wrong tool, not a slightly-suboptimal one. There is no single rule set that buys dips profitably in a range *and* survives a one-way trend, because in the trend the profitable action is to stop buying dips entirely. Believing one strategy fits all markets is believing one key fits all locks; the moment the lock changes, the belief costs you.

**"More confirmation from the indicators would have helped."** Adding more oscillators to a mean-reversion strategy does not help in a trend, because in a trend *all* the oscillators are oversold *all* the time -- they confirm each other into buying every step down. Confirmation among correlated indicators is an illusion of independence (the [confluence post](/blog/trading/technical-analysis/confluence-stacking-independent-factors) covers why stacking correlated signals does not stack edge). The thing that would have helped is not more signals of the same kind; it is a signal of a *different* kind -- a regime filter that asks "is this even a range?" before the entry rules get a vote.

**"Cutting size when I'm losing is just fear talking."** Cutting size in the face of a degrading edge is not fear; it is the correct response to increased uncertainty about your edge. The mathematically optimal bet size scales with your estimated edge, and a degrading edge is a falling estimate -- so the optimal size falls. Refusing to cut size because "that's just fear" is confusing the discipline of *holding through normal variance* (correct) with the recklessness of *betting full size on an edge you can no longer verify* (a mistake). The skill is to cut size on genuine edge degradation and *not* to cut it on normal variance, and the regime tells are how you tell those apart.

## How it shows up in real markets

The mechanism is not a toy. It recurs across instruments, decades, and trader types. Here are several named, concrete episodes -- with as-of caveats, since the specifics matter and market facts go stale.

**Buy-the-dip equity books meeting 2022.** Through the calm, range-and-grind stretch of 2020 into late 2021, "buy the dip" in US equity indices was close to a free lunch -- shallow pullbacks reverted quickly and the dip-buyer was rewarded again and again, exactly the smooth high-win-rate curve our case study describes. Then 2022 arrived: major US indices fell through level after level for most of the calendar year as central banks raised interest rates aggressively to fight inflation, and the S&P 500 ended the year down roughly 19% (as of the end of 2022). Every dip that year was a falling knife. Dip-buyers conditioned by 2021 bought each one and watched it keep falling -- the textbook range-to-trend flip, with the identical entry rule winning in one regime and bleeding in the next. (As of mid-2026, 2022 remains the canonical recent example of this exact failure; the lesson is the mechanism, not the specific year or percentage.)

**A mean-reversion fund undone by a one-way move.** Systematic mean-reversion strategies -- statistical-arbitrage and short-term reversal books that bet stretched moves snap back -- have repeatedly suffered sharp, concentrated losses when a normally-mean-reverting market suddenly trended hard. The well-documented "quant quake" of August 2007 is one such episode: a cluster of quantitative funds running similar mean-reversion-flavored books saw their usual reversion patterns invert violently over a few days as forced deleveraging pushed prices in one direction, turning the small frequent wins into rapid large losses. The details are debated and the precise mechanism was partly a crowding-and-deleveraging spiral rather than a pure macro trend, but the equity-curve signature -- a long smooth climb then a sudden cliff -- is the same one our case study draws. (As-of caveat: this is a much-studied event with multiple interpretations; cite it as illustrative of the *shape* of mean-reversion failure, not as a settled single-cause story.)

**An oscillator-fader caught in a parabolic rally.** The mirror image of the falling-knife problem happens to the *short* side. A trader who fades overbought readings -- selling when RSI prints above 70, betting on a pullback -- prints a lovely curve in a range, where every overbought reading does mean-revert. Put that same trader into a parabolic *uptrend* -- a runaway move like a hot growth stock or a crypto asset in a mania, where RSI can stay pinned above 80 for weeks -- and they short every step up, getting run over repeatedly as the thing they keep calling "overbought" keeps going higher. Individual crypto manias (for example the sharp 2017 and 2020-2021 run-ups in major tokens, as of those dates) produced exactly this carnage for short-the-extreme traders. The oscillator was not broken; it was regime-mismatched, the same defect as buying dips in a downtrend, flipped.

**A volatility-selling book in a spike.** The purest expression of "high win rate, fat tail you haven't paid yet" is selling options or selling volatility -- collecting a small steady premium most of the time in exchange for a rare large loss when volatility spikes. In a calm, range-like volatility regime, a vol-selling book prints the smoothest equity curve in finance, which is exactly why it attracts capital. Then a volatility spike arrives and the rare large loss comes due all at once. The "Volmageddon" episode of early February 2018 is the textbook case: short-volatility products that had quietly compounded gains for over a year lost the overwhelming majority of their value in a matter of days when volatility spiked, and at least one large short-vol exchange-traded note was effectively wiped out (as of February 2018). It is the same equity-curve shape as our mean-reverter -- smooth climb, sudden cliff -- because it is the same underlying bet: win often and small, lose rarely and catastrophically, and pray the regime that produces the catastrophe stays away.

**A trend-following CTA in a chop.** For completeness and symmetry, the failure runs the other way too. A managed-futures trend-following fund (a "CTA") that thrives on sustained trends will bleed steadily during a long, choppy, sideways year -- every breakout it buys reverses, every position gets whipsawed, and the smooth-loser-many-small-cuts pattern grinds the account down. Several stretches in the 2010s (as of those years) were difficult for trend-followers for exactly this reason: not enough sustained trends, too much chop. It is the same lesson with the regimes swapped -- a trend-follower needs trends the way our mean-reverter needs ranges, and the wrong regime is corrosive to either one.

The common thread across all five: the strategy was correct *for a regime*, the regime changed, the rules did not, and the equity curve told the truth long before the trader admitted it. Each is, at root, the same story this whole post tells, dressed in a different instrument.

## When this matters to you / further reading

This matters to you the first time you build something that works. The danger is not the strategy that fails immediately -- you abandon that one and move on. The danger is the strategy that *succeeds*, because success in one regime builds exactly the conviction that leaves you defenseless when the regime turns. The smoother your equity curve, the more important it is to ask the uncomfortable question: *which regime is this curve a record of, and what happens to these rules in the opposite one?* If you cannot answer that, you do not yet understand your own edge; you have only observed it under favorable conditions.

The practical takeaways, stated plainly and without advice to trade anything:

- **Know which regime your edge requires.** Every strategy needs a particular kind of market. Mean-reversion needs oscillation; trend-following needs persistence. If you cannot name the regime your edge feeds on, you cannot know when it will fail.
- **Distrust a single-regime backtest.** A backtest is a measurement of the regimes in its sample. Deliberately test across history that contains *both* regimes, and weight your confidence by how the strategy behaved in the regime that hurts it, not the one that flatters it.
- **Monitor the tells, not just the trades.** Structure, volatility, and autocorrelation tell you about the regime above the level of any single trade. Watch them when you are *winning*, not only when you are bleeding.
- **Separate a drawdown from a dead edge with the Monte Carlo band and the regime tells.** "Wait it out" is correct for a normal drawdown and lethal for a dead edge, and the discriminators are objective -- use them.
- **Hold more than one playbook.** A trader with only a mean-reversion tool is, by construction, defenseless in a trend. The mature posture is two edges and a regime filter that decides which one is allowed to trade.

To go deeper into the ideas this case study leans on: the foundational distinction is in [trend-following versus mean-reversion](/blog/trading/technical-analysis/trend-following-vs-mean-reversion), which explains why the two edges are exact opposites and why the same setup wins in one regime and dies in the other. The math of why a high win rate is not the same as an edge -- and why a smooth curve can hide a fatal asymmetry -- is in [expectancy and why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies). The honest workflow for testing a strategy without fooling yourself, including the Monte Carlo technique for finding the realistic worst drawdown, is in [backtesting without fooling yourself](/blog/trading/technical-analysis/backtesting-without-fooling-yourself). And the specific way that tuning and trusting indicators produces the illusion of a durable edge is in [the indicator trap](/blog/trading/technical-analysis/the-indicator-trap). Read together, those four are the toolkit that turns the story in this post from something that happens *to* you into something you can see coming -- and the difference between giving back a couple of R and giving back a year.
