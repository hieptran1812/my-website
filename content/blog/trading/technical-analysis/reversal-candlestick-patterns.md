---
title: "Reversal Candlestick Patterns That Earn Their Keep: Engulfing, Pin Bars, and Doji in Context"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A first-principles, honest guide to the three reversal candlestick patterns worth knowing: the bullish and bearish engulfing, the pin bar (hammer and shooting star), and the doji. What order-flow story each one encodes, its real base rate, the context filters that turn it from noise into a usable edge, and how to define entry, stop, and target so the expectancy math can actually work."
tags: ["candlestick-patterns", "price-action", "technical-analysis", "engulfing-candle", "pin-bar", "doji", "reversal-patterns", "risk-reward", "expectancy", "trading-psychology", "support-and-resistance", "order-flow"]
category: "trading"
subcategory: "Technical Analysis"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A handful of reversal candlestick patterns are genuinely worth knowing, but only because of the *order-flow story* each one encodes, and only when it forms at a price level that matters with the higher-timeframe context behind it. In isolation, their edge is small and usually gone after costs.
>
> - A **reversal pattern** is a single bar (or two) claiming that the prior move's momentum just failed. The claim is *probabilistic*, never a guarantee.
> - The three keepers: the **engulfing** (a same-bar takeover, where one side seizes the whole prior range), the **pin bar** — hammer or shooting star — (a long wick marking a price the market tested and rejected), and the **doji** (a stalemate that resolves nothing until the next bar).
> - On their own, all three barely beat a coin flip. A bare bullish engulfing follows through maybe 50 to 53 percent of the time, and after the spread and slippage that edge is gone.
> - Context is what adds the edge: the pattern at a tested **support or resistance** level, aligned with the **higher-timeframe trend**, on rising **volume**, with a **confirming** next bar. Stack those filters and the same pattern can run to 60 to 65 percent follow-through — a real, if modest, tilt.
> - The pattern is not the trade. The trade is **entry / stop / target**: entry on the break of the pattern's extreme, stop just beyond the rejected wick, target at the next level. Get a **3:1 reward-to-risk** and you only need to be right one time in four to break even.
> - The single number to remember: a setup that looks like a **62% winner in hindsight** is often a **50% setup** once you must trade it in real time and pay a **0.1R** cost per trade — which can cut the edge by roughly **73%**. Honesty about that gap is the whole skill.

Here is a scene every chart-reader knows. A market has been falling for days. Then one candle prints that looks different from the rest: a long spike down that snaps back, or a big green bar that swallows the red one before it. Something in you says *that's the bottom*. People have been trading on that feeling for three hundred years — the candlestick chart was invented by Japanese rice traders in the 1700s precisely to read this kind of moment — and the feeling is not wrong. It is just badly calibrated. Most of the time that candle means nothing. Some of the time it means a great deal. The entire job of this post is to teach you which is which.

We are going to take the three reversal candlestick patterns that actually survive honest testing — the **engulfing**, the **pin bar**, and the **doji** — and do four things with each. First, understand its *mechanism*: what just happened in the order book to make the candle look like that. Second, state its *honest base rate*: how often it actually leads to a reversal, with no flattery. Third, name the *context filters* that lift it from noise to signal. And fourth, turn it into a *trade* — an entry, a stop, and a target — so the expectancy math has a chance to work in your favour. If you have read [the post on what a price chart is and how candles are born](/blog/trading/technical-analysis/how-a-price-chart-is-born), you already know how to read a candle's body and wick; here we put that reading to work.

The diagram below is the mental model for everything that follows. Three candle shapes, three stories. The engulfing is a *takeover* — one side seizes the whole prior range in a single bar. The pin bar is a *rejection* — price probes a level and gets thrown back, leaving a long wick like a footprint. The doji is a *stalemate* — buyers and sellers fight to a draw and nobody wins the session. Notice that none of these is a chart pattern in the geometric sense; each is a compressed record of a *fight*, and the candle's shape is the scoreboard.

![Three reversal candle shapes side by side: a green engulfing candle that swallows a red one, a blue pin bar with a long lower wick, and an amber doji with a tiny body, each labelled with the order-flow story it tells](/imgs/blogs/reversal-candlestick-patterns-1.png)

A word of honesty up front, because this is a field thick with overpromising. There is no candlestick pattern that "works" in the sense beginners hope — none that reverses the market most of the time on its own. The published academic record on candlestick patterns is, charitably, mixed: many studies find no exploitable edge once you account for transaction costs and data-snooping. What survives is narrower and more boring: certain patterns, *at certain locations*, *in certain trends*, with a *confirming* bar, offer a small probabilistic tilt and — crucially — a place to put a tight stop, which lets you build a trade with good reward-to-risk. That is the honest claim. We will defend exactly that claim and nothing more. Nothing here is financial advice; it is a mechanical explanation of how these patterns behave.

## Foundations: what a reversal pattern is claiming

Before the three patterns, we need to agree on what the word "reversal" is even asserting, because most of the confusion in this topic comes from taking the word too literally.

A **trend** is just the direction price has been moving over some window: a sequence of higher highs and higher lows is an uptrend, lower highs and lower lows is a downtrend. (If that framing is new, the post on [trend and market structure](/blog/trading/technical-analysis/trend-and-market-structure) builds it from scratch.) A **reversal** is the claim that this direction is about to *flip* — that an uptrend is about to become a downtrend, or vice versa. A **reversal candlestick pattern** is a single candle, or a small cluster of two or three, whose shape is being read as evidence that the flip is happening *right now, at this bar*.

So when someone points at a candle and says "that's a reversal signal," the literal content of that claim is: *the momentum that was carrying price in the prior direction just failed at this price, and the other side is now in control.* That is a strong claim to read off one bar. Sometimes it is true. Often it is not. The pattern is **evidence**, in the same sense a single witness is evidence — informative, fallible, and far more credible when other evidence agrees with it.

### Body and wick: a thirty-second refresher

Every candlestick compresses one period — a minute, an hour, a day — into four numbers: the **open** (first traded price of the period), the **high** (highest), the **low** (lowest), and the **close** (last traded price). (For the full treatment of how to read a candle and what the evidence honestly supports, see [candlestick anatomy and the honest evidence](/blog/trading/technical-analysis/candlestick-anatomy-and-the-honest-evidence).) The **body** is the rectangle between the open and the close. If price closed higher than it opened, the body is drawn green (or hollow) and we call it an *up-candle*; if it closed lower, the body is red (or filled), a *down-candle*. The thin lines sticking out of the body are **wicks** (also called *shadows* or *tails*): the upper wick runs from the body up to the high, the lower wick runs from the body down to the low.

Why does this encoding matter so much for reversals? Because the body tells you *who won the period* and the wick tells you *where price went but could not stay*. A long lower wick means price fell to that low but buyers dragged it back up before the close — the low was *rejected*. A long upper wick means price rose to that high but sellers pushed it back down — the high was rejected. The body is the net result; the wick is the failed excursion. Every reversal pattern in this post is a particular arrangement of bodies and wicks that says *the prior side pushed, and failed*.

### Why "reversal" is probabilistic, not a promise

Here is the single most important mental adjustment to make. A reversal pattern does not *cause* a reversal and does not *predict* one with certainty. It marks a moment where the probability of a reversal is *somewhat higher than baseline* — and "somewhat" is doing a lot of work. If a market reverses out of a random bar 50 percent of the time over the next N bars (the rough baseline for a coin-flip market), and a bullish engulfing at a random location reverses 52 percent of the time, that 2-point edge is real but *tiny*, and the bid-ask spread you pay to enter and exit can easily exceed it. The pattern only becomes interesting when context pushes that number up to 58, 60, 63 percent — and even then, the bigger contribution of the pattern is often not the win rate but the *clean place it gives you to put a stop*.

Hold that thought, because it is the hinge of the whole post: **a pattern's value is partly its slightly-better-than-baseline win rate, and partly the tight, well-defined risk it lets you take.** A pattern that only nudges your win rate from 50 to 55 percent but lets you risk \$2 to make \$6 is a *better trade* than a pattern that wins 65 percent of the time but forces you to risk \$10 to make \$5. We will make that precise when we get to expectancy.

## The engulfing pattern

The engulfing pattern is the most intuitive of the three because the picture matches the story exactly: one candle's body *engulfs* — completely covers — the body of the candle before it.

### Construction

A **bullish engulfing** is a two-candle pattern. The first candle is a down-candle (red, a small-ish body). The second candle is an up-candle (green) whose body *opens below the prior candle's close (ideally below its low) and closes above the prior candle's open (ideally above its high)* — so the green body swallows the red one whole. The strict definition only requires the second body to cover the first body; the strongest versions also engulf the wicks. A **bearish engulfing** is the mirror image at a top: a small green candle, then a big red candle whose body opens above the prior high and closes below the prior low.

The figure below shows both constructions, candle by candle, so you can see what "engulf" means precisely.

![Two side-by-side two-candle constructions: a bullish engulfing where a large green body swallows a small red body at a bottom, and a bearish engulfing where a large red body swallows a small green body at a top, each annotated with how the second bar opens and closes relative to the first](/imgs/blogs/reversal-candlestick-patterns-2.png)

A few construction notes that matter in practice. The size relationship is the point: a green body that barely covers a tiny red body is weak; a green body two or three times the size of the red one, closing on its highs, is strong. The **close** of the engulfing bar matters more than any other single point — a bullish engulfing that closes near its high is a far stronger statement than one that closes back in the middle of its range, because the close is where the period's verdict is sealed. And the pattern is only meaningful as a *reversal* if there was a prior move to reverse: a bullish engulfing only "reverses" something if price was falling into it.

### The order-flow story

Why would this shape carry any information at all? Read it as a record of what happened in the order book during those two bars.

During the first (red) bar, sellers were in control: every time buyers tried to lift price, more sell orders met them, and the bar closed lower. The prevailing belief was "this goes down." Then the second bar opens — often *gapping lower* or opening near the prior close, confirming the bearish mood — and something changes. A wave of buying hits. Not only does it absorb all the sell orders that were happily pushing price down a moment ago, it overwhelms them so completely that price closes *above where the entire prior bar opened*. In one bar, the buyers did not just stop the decline; they erased it. That is the order-flow event the green body records: **a decisive absorption of supply followed by aggressive demand seizing the whole range.** When that happens *after a sustained downtrend, at a level where buyers had reason to step in*, it is genuinely informative — the people who were short are now offside and may have to cover, which adds fuel.

The bearish engulfing is the same story flipped: buyers were in control, then a wall of selling absorbed all the buying and drove the close below the prior bar's open, trapping the late longs.

### Honest reliability and the confirmation rules

Now the uncomfortable numbers. Studies and large-sample backtests of the bare engulfing pattern — taken anywhere, in any context — tend to find follow-through rates clustered around **50 to 55 percent** over a short horizon, which after costs is not an edge you can trade. Thomas Bulkowski's widely-cited pattern statistics, for instance, rank the bullish and bearish engulfing in the *middle* of the candlestick pack, not the top, and his "reversal" success rates depend heavily on how you define follow-through and over what window. The honest summary: **a context-free engulfing candle is close to a coin flip.**

What lifts it? Three filters, which we will treat formally in their own section, but in brief:

- **Location.** A bullish engulfing *at a tested support level* is a different animal from one in the middle of a range. At the level, the buyers stepping in have a *reason* — there are resting bids, trader memory, prior structure (see [why support and resistance levels exist at all](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist)). Mid-range, the same candle is just noise that happened to look dramatic.
- **Trend alignment.** A bullish engulfing that forms on a pullback *within* a larger uptrend (buying the dip in an up-market) follows through more often than one trying to call the bottom of a strong downtrend (catching a falling knife).
- **Confirmation.** Requiring the *next* bar to close in the engulfing's direction before you act filters out a large share of failures — at the cost of a worse entry price. More on this trade-off later, because it is where most of the hindsight illusion lives.

The single most useful mechanical rule for the engulfing: **act on the close, not the wick.** An engulfing candle is only an engulfing candle once it has *closed* engulfing. Intra-bar, price can look like a monster bullish engulfing and then sell off in the last ten minutes to close as a doji. Many beginners enter early on what they *think* is forming and get caught when the bar finishes telling a different story.

## The pin bar (hammer, shooting star, hanging man)

The pin bar is the rejection pattern, and it is the one where *location is the entire signal*. The shape on its own tells you almost nothing; the shape *at a level* can tell you a lot.

### Construction and the names

A **pin bar** (short for "Pinocchio bar," because its long nose "lies" about where price wanted to go) is a single candle with a **small body at one end and a long wick at the other**, with little or no wick on the body side. The rule of thumb: the wick (tail) should be at least *two-thirds* of the candle's total range, and ideally the body should sit in the top or bottom third.

The names depend on which way the wick points and where the candle appears:

- A **hammer** has a long *lower* wick and a small body up top — it looks like a hammer or a mallet. It appears at the *bottom* of a down-move and is read as bullish (a failed push lower).
- A **shooting star** has a long *upper* wick and a small body at the bottom. It appears at the *top* of an up-move and is read as bearish (a failed push higher).
- A **hanging man** has the same shape as a hammer (long lower wick) but appears at the *top* of an up-move; it is read as a warning that the uptrend may be tiring. The shape is identical to a hammer — the location flips the meaning. This is your first big clue that the candle shape alone is not the signal.

The figure shows the anatomy of a hammer and a shooting star precisely: where the body sits, where the long wick goes, and which price it is rejecting.

![Anatomy of a hammer and a shooting star: the hammer has a small green body at the top and a long lower wick piercing a support level, the shooting star has a small red body at the bottom and a long upper wick into a resistance level, each labelled with the rejected price and the order-flow story](/imgs/blogs/reversal-candlestick-patterns-3.png)

### The order-flow story: a long wick is a rejected price

The long wick is the whole point, so let us be precise about what it means. Take the hammer. During the bar, price *fell* — it went all the way down to the low (the bottom of the lower wick). At that low, something happened: buyers stepped in hard enough not just to stop the fall but to *drag price back up* so that it closed near the top of the bar. The lower wick is a **footprint of a failed sell-off**: price went there, and it could not stay. The market *tested* the lower price and *rejected* it.

That rejection is meaningful for a specific reason: it tells you there was real demand waiting at that low. Someone — or many someones — was willing to buy aggressively enough to absorb all the sellers and reverse the bar. If that low coincides with a price where you would *expect* demand (a support level, a prior swing low, a round number), the wick is corroborating evidence that the level is being defended. The shooting star is the mirror: a long upper wick is a footprint of a failed rally, demand exhausted and supply taking over at a high.

### Why location is everything

Here is the hammer's dirty secret, and it is the single most important idea in this entire post. **A hammer in the middle of a range is almost meaningless. A hammer at a tested support level is a real, if modest, signal.** Same shape, completely different value.

Why? Because the long lower wick only *means* something if the price it rejected is a price other traders care about. At support, the wick says "the level held — buyers defended it, just as the chart's memory predicted." There are resting bid orders there, traders remember the level, stops of short-sellers may be clustered just below it. The rejection has *causes*. In the middle of a range, the same wick rejected a price that nobody was defending in particular; it is just normal intrabar noise that happened to leave a long tail. There is no level, no memory, no resting demand — so there is no reason to expect the rejection to continue.

The figure below makes this concrete by putting the identical hammer in two places and showing what changes.

![A before-and-after comparison: the same hammer candle at a tested support zone reads as a tradeable rejection with a tight stop, while the identical hammer in the middle of a range with no level nearby reads as noise with a coin-flip follow-through and a wide, ill-defined stop](/imgs/blogs/reversal-candlestick-patterns-6.png)

There is a second, mechanical reason location matters: **the stop.** A hammer at support gives you a natural, *tight* place to put your stop — just below the wick, which is just below the level. If the level fails, you are wrong, and you find out cheaply. A hammer mid-range gives you nothing to lean on; your stop is arbitrary, and an arbitrary stop is usually either too tight (you get shaken out of a good trade) or too wide (one loss erases several wins). The level is not just a probability filter; it is what makes the *risk side* of the trade definable. We will return to this when we build the actual trades.

## The doji and indecision

The doji is the subtlest of the three and the most misunderstood. It is not, on its own, a buy signal or a sell signal. It is a statement of *indecision*, and indecision resolves into a direction only on the *next* bar.

### What a doji is

A **doji** is a candle whose open and close are *essentially equal* — the body is a thin line or nonexistent, because price ended the period right where it began. The wicks can be short or long. The defining feature is the near-zero body: whatever happened during the bar, neither side won. Buyers and sellers fought all session and ended in a draw.

There are four common flavours, distinguished by where the wicks sit. The figure catalogues them.

![A catalogue of four doji types in a grid: standard doji with small wicks both sides, long-legged doji with very long wicks both sides, gravestone doji with only a long upper wick, and dragonfly doji with only a long lower wick, each with its order-flow hint](/imgs/blogs/reversal-candlestick-patterns-4.png)

- A **standard doji** has a small body and modest wicks on both sides: pure, balanced indecision, no lean either way.
- A **long-legged doji** has a tiny body and *very long* wicks on both sides: a violent two-way fight. Price ranged far up and far down during the bar and still closed flat — maximum disagreement, maximum volatility, still no winner.
- A **gravestone doji** has the body at the *bottom* and a long *upper* wick only: price rallied hard during the bar and was driven all the way back down to the open. Buyers tried, buyers failed. It has a *bearish* lean, especially at a top.
- A **dragonfly doji** has the body at the *top* and a long *lower* wick only: price fell hard and was bought all the way back up to the open. Sellers tried, sellers failed. It has a *bullish* lean, especially at a bottom. (You will notice the dragonfly is essentially a hammer with no body, and the gravestone is a shooting star with no body — the family resemblance is real.)

### A doji resolves nothing until the next bar

The cardinal rule of the doji: **by itself it tells you the trend is *stalling*, not that it is *reversing*.** A doji in the middle of a strong trend often just means the market paused to catch its breath and then continued. The information in a doji is conditional: it says "the prior momentum has, for one bar, evaporated." Whether that stall becomes a reversal or a continuation is told by the *next* bar.

This is why experienced readers treat the doji as a *trigger to pay attention*, not a trigger to act. A doji at a meaningful level — say, a long-legged or gravestone doji right at a resistance zone after a long rally — is a flag that momentum has died exactly where you would expect a reversal. But you wait. If the next bar closes down through the doji's low, the stall has resolved into a reversal and *now* you have a signal. If the next bar closes up, the trend has resumed and the doji was just a breath. The doji's honest base rate as a standalone reversal signal is essentially baseline — around 50 percent, sometimes worse — precisely because half the time the stall resolves back into the trend. Its value is entirely in flagging the *location* of possible exhaustion and forcing you to wait for confirmation.

## Context filters that actually add edge

We have now met all three patterns and kept saying "in context." It is time to make "context" precise. There are four filters, and they are roughly additive: each one you stack on lifts the follow-through rate and, just as importantly, sharpens the trade you can build. The matrix below shows, honestly, how the numbers move as you add filters — from near-coin-flip for the bare pattern to a real-if-modest tilt once everything aligns.

![A reliability matrix with the three patterns as rows and four context columns: bare pattern, at a tested level, at a level with the trend, and at a level with the trend and confirmed, showing follow-through percentages rising from roughly 50 percent to roughly 60 to 65 percent as filters are stacked](/imgs/blogs/reversal-candlestick-patterns-7.png)

A caveat on these numbers before we lean on them: the percentages in that figure are *illustrative ranges* synthesised from the general direction of published candlestick research and practitioner backtests, not a precise claim about any one market or period. They are meant to convey the *shape* of the effect — bare patterns near baseline, context lifting them into the high 50s and low 60s — not to be quoted as exact. Real follow-through rates depend on the market, the timeframe, the trend regime, and how you define "follow-through." The honest, defensible claim is the *direction* of the effect, and that direction is robust.

### Filter 1: at a support or resistance level

This is the heavyweight filter — the one that does the most work, especially for the pin bar. A reversal pattern is asking "is the prior move failing here?" and a support/resistance level is a *prior answer* to "where might it fail?" When the two agree — a bullish pattern at support, a bearish pattern at resistance — you have two independent pieces of evidence pointing the same way. The pattern says "momentum just failed"; the level says "and here is exactly why it would fail *here*." A pin bar whose long wick rejects a price *that is also a tested support level* is the canonical example of this agreement.

### Filter 2: with the higher-timeframe trend, or at a clear extreme

There are two honest ways to use trend context, and they pull in slightly different directions:

The first and statistically safer use is **trend-aligned reversals**, which are really *continuation* trades wearing a reversal costume. In an uptrend, price pulls back, makes a bullish engulfing or hammer at a support level, and resumes up. You are not calling a top or bottom of the whole market; you are buying a dip *in the direction the market is already going*. This is high-percentage because you have the dominant force on your side. The reversal you are trading is the reversal of the small *pullback*, not of the major trend.

The second use is the **counter-trend reversal at an extreme**: trying to catch the actual turn of a trend. This is lower-percentage and more dangerous — "catching a falling knife" — but the payoff when right is larger because you enter near the extreme. The honest filter here is to demand a *clear extreme*: an exhausted, climactic move into a major level, not just any pullback. Counter-trend reversal trades should be rarer and require more confirmation than trend-aligned ones.

### Filter 3: volume

**Volume** is the number of shares or contracts traded during the bar — a measure of *participation*. A reversal pattern on high volume is more credible than the same pattern on thin volume, because high volume means the absorption and rejection the candle depicts actually involved a lot of orders changing hands, not a few trades in a quiet hour. A bullish engulfing on the highest volume of the week is a stronger statement than one on a sleepy afternoon: more sellers were absorbed, more buyers committed. Volume is a corroborating filter, not a standalone one — it strengthens an already-located pattern rather than rescuing a bad one. (Not every market gives you reliable volume — spot foreign exchange, famously, has no central volume figure — so this filter is more available in stocks and futures than in some others.)

### Filter 4: the confirmation bar

The **confirmation bar** is the simplest and most underrated filter, and the one that creates the most hindsight illusion. The rule: do not act on the pattern bar itself; wait for the *next* bar to confirm the direction before you enter. For a bullish engulfing or hammer, "confirmation" means the next bar trades above the pattern's high (or closes up). For a bearish pattern, the next bar breaks the pattern's low.

Confirmation works because it filters out the patterns that looked great and then immediately failed. A hammer that gets confirmed by a strong up-bar is a different population from all hammers — you have conditioned on the ones that actually started moving. The cost is real: you enter *later* and at a *worse price*, which shrinks your reward-to-risk because your stop is now further from your entry. There is no free lunch. Confirmation trades a better win rate for a worse payoff, and whether that is worth it depends on the specific setup. The crucial honest point is this: **the impressive win rates people quote for these patterns are almost always measured with the benefit of hindsight — knowing how the bar resolved — and the win rate you can actually capture, in real time, requiring confirmation, is meaningfully lower.** We will quantify that gap in the worked examples.

### Multi-factor confluence

When several filters align — a bullish engulfing, at a tested support level, on a pullback within an uptrend, on rising volume, confirmed by the next bar — you have what traders call **confluence**: multiple independent reasons pointing the same way. Confluence is where these patterns stop being a gimmick and start being a usable, if modest, edge. The follow-through rate climbs into the low-to-mid 60s, *and* the trade is clean (a tight stop below the wick, a clear target at the next level). The post on [breakouts versus fakeouts](/blog/trading/technical-analysis/breakouts-vs-fakeouts) explores the related idea from the other side — when a level *breaks* rather than holds — and the two together are how you read what price does *at* a level. The lesson is the same in both: the obvious, lone signal is the one that gets faded; the confluent signal is the one with a story.

## Turning a pattern into a trade

A pattern is not a trade. A trade is three prices: where you get in (**entry**), where you admit you were wrong (**stop**), and where you take your money (**target**). The pattern only earns its keep if those three prices produce a favourable **reward-to-risk ratio** — the size of your potential win divided by the size of your potential loss.

### Entry, stop, target

**Entry.** Two common choices. The aggressive entry is *at the close of the pattern bar* — you act the moment the engulfing or pin bar completes. The conservative entry is *on the break of the pattern's extreme* — for a bullish pattern, you buy only when price trades above the pattern's high (this is the confirmation bar in action). The conservative entry has a higher win rate and a worse price; the aggressive entry has a better price and more failures. Most disciplined approaches use the break-of-extreme entry for exactly the reason in the last section.

**Stop.** This is where the pattern shines. The stop goes *just beyond the rejected wick* — below the low of a hammer or bullish engulfing, above the high of a shooting star or bearish engulfing. The logic is airtight: the wick is the price the market rejected. If price goes *back* beyond it, the rejection has failed — the level did not hold, your read was wrong, and you should be out. The wick gives you a *natural, non-arbitrary* place to be wrong, and a tight one. This is the single biggest mechanical gift these patterns give you.

**Target.** The target is the *next meaningful level* in the direction of the trade — the next resistance for a long, the next support for a short — or a *measured move* (projecting the size of the prior swing). The target is what determines whether the trade is worth taking at all, because it sets the reward against the stop's risk.

The figure walks through a complete bullish-engulfing-at-support trade with real numbers, so you can see entry, stop, and target as three concrete prices and the reward-to-risk that falls out of them.

![A price chart showing a downtrend into a green support zone where a bullish engulfing forms, with a dashed entry line at the break of the engulfing high, a stop line below the wick in an amber stop zone, and a target line at a red resistance zone, annotated with a 6.00 reward over 2.00 risk equals 3-to-1 reward-to-risk that breaks even at a 25 percent win rate](/imgs/blogs/reversal-candlestick-patterns-5.png)

### Tie it to expectancy

Why do we obsess over reward-to-risk instead of win rate? Because **expectancy** — the average amount you make per trade over many trades — depends on *both*, and reward-to-risk is the lever the pattern hands you. The formula, in units of **R** (where 1R = the dollar amount you risk per trade, i.e. the distance from entry to stop):

$$\text{Expectancy} = (W \times R_{\text{win}}) - (L \times R_{\text{loss}})$$

where $W$ is your win rate (the fraction of trades that hit the target), $L = 1 - W$ is your loss rate, $R_{\text{win}}$ is the reward in R-multiples, and $R_{\text{loss}}$ is the loss in R-multiples (normally 1, since you risk 1R and lose it on a stop-out). A positive expectancy means you make money on average; negative means you bleed. The deep treatment of why win rate alone is misleading lives in [the post on expectancy and why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) — here we just need the punchline: with a 3:1 reward-to-risk, you break even at a *25 percent* win rate, so a pattern that wins even 45 percent of the time with 3:1 payoff is strongly profitable, while a pattern that wins 65 percent of the time at 1:2 payoff (risking \$2 to make \$1) is a slow death.

That is the resolution of the paradox we set up at the start. The reversal patterns do not need to win most of the time. They need to (a) win *somewhat* more than their reward-to-risk requires for breakeven, and (b) give you the tight, defined stop that *makes* a good reward-to-risk possible. Both of those come from forming *at a level* — which is why we keep coming back to location.

#### Worked example: a bullish engulfing at support, priced as a 3:1 trade

Let us build the trade in the figure, step by step, with round numbers.

A stock has been falling and reaches a support zone around \$99.50 to \$100.00 — a level it bounced from twice before, so there is memory and resting demand there. Price dips to a low of \$99.20, and the day prints a **bullish engulfing**: a green candle whose body opens at \$99.40 and closes at \$100.40, swallowing the prior red bar, with the low wick down at \$99.20. The pattern has formed *at support*, on a pullback within what is still a broader uptrend on the weekly chart. That is three filters: location, trend, and the pattern itself.

Now we price the trade:

- **Entry:** We use the conservative break-of-extreme entry. The engulfing's high is \$100.40, so we buy when price trades above it — call it **\$100.50** to allow a tick of room.
- **Stop:** Just below the rejected wick. The low was \$99.20; we put the stop at **\$98.50**, a bit below, so a marginal poke through the wick does not knock us out. Our risk per share is \$100.50 − \$98.50 = **\$2.00**. That is our 1R.
- **Target:** The next resistance overhead sits at **\$106.50** (a prior swing high). Our reward per share is \$106.50 − \$100.50 = **\$6.00**.

The reward-to-risk is \$6.00 / \$2.00 = **3:1**. Breakeven win rate is $1 / (1 + 3) = 25\%$. So even if this exact setup only works *one time in three* (a 33 percent win rate), the expectancy is positive:

$$\text{Expectancy} = (0.33 \times 3R) - (0.67 \times 1R) = 0.99R - 0.67R = +0.32R$$

per trade. On a \$2.00 risk, that is about \$0.64 of expected profit per share, per trade, *if* the win rate is 33 percent. If context pushes the realistic win rate to 45 percent, the expectancy jumps to $(0.45 \times 3) - (0.55 \times 1) = 1.35 - 0.55 = +0.80R$ — a genuinely strong trade. **The one-sentence intuition: at a level, the tight stop below the wick is what lets a sub-even win rate still print money, because it buys you a 3:1 payoff.**

#### Worked example: the same hammer at support versus mid-range

Now the base-rate comparison that proves location is the signal. Suppose we have a hammer — a candle with a \$1.00 lower wick and a small \$0.20 body up top, identical shape in both cases.

**Case A: the hammer at support.** Price has fallen into a support zone at \$50.00 that held twice before. The hammer's low is \$49.60 (the wick pierces the level), and it closes back at \$50.40. We buy the break of the high at \$50.50, stop below the wick at \$49.40 (risk = \$1.10), target the next resistance at \$53.80 (reward = \$3.30). Reward-to-risk ≈ 3:1. And the *win rate* of this located, trend-aligned, confirmed setup is, realistically, around **58 to 62 percent** based on the direction of the research. Expectancy at 60 percent: $(0.60 \times 3) - (0.40 \times 1) = 1.80 - 0.40 = +1.40R$. Excellent.

**Case B: the identical hammer mid-range.** Same shape, same \$1.00 wick, but it forms at \$50.00 in the *middle* of a sideways range with no level nearby. There is nothing for the stop to lean on — you might place it \$1.10 below at \$49.40, but you are guessing. And the win rate of a mid-range hammer with no level is, honestly, around **50 to 52 percent** — a coin flip. Even at the same 3:1 you might construct, the *realised* win rate is so much lower that, after the spread, you are barely above water; and in practice your stop placement is so arbitrary that your true reward-to-risk is fuzzier than the clean Case A. **The one-sentence intuition: the candle is identical, but only at the level does the wick reject a price with a reason behind it — so location, not shape, moved the win rate from a coin flip to a real edge.**

#### Worked example: a shooting star at resistance, shorted

The bearish mirror, with numbers. A crypto asset has rallied hard into a resistance zone at \$30,000 — a round number and a prior high, so there is real supply memory there. The bar prints a **shooting star**: a small red body near \$30,100 at the bottom, a long upper wick spiking to \$31,200 (price tried to break out and got slammed back), closing at \$30,100. The high (\$31,200) was rejected at resistance.

We price the short:

- **Entry:** Sell the break of the shooting star's low. The bar's low is \$30,050; we short when price trades below it, at **\$30,000** (which is also the round-number level breaking).
- **Stop:** Just above the rejected wick, at **\$31,400** — above the \$31,200 high, with room. Risk = \$31,400 − \$30,000 = **\$1,400**. That is 1R.
- **Target:** The next support below sits at **\$25,800** (a prior consolidation shelf). Reward = \$30,000 − \$25,800 = **\$4,200**.

Reward-to-risk = \$4,200 / \$1,400 = **3:1**. Breakeven win rate 25 percent. If this located-and-confirmed bearish setup wins around 55 percent of the time, expectancy is $(0.55 \times 3) - (0.45 \times 1) = 1.65 - 0.45 = +1.20R$ — \$1,680 of expected profit per unit risked, per trade. **The one-sentence intuition: a shooting star at resistance lets you short *with* a tight stop above the rejected high, so the asymmetric payoff does the work even at a modest win rate.**

#### Worked example: the "looks great in hindsight" trap

This is the most important worked example in the post, because it quantifies the gap between what these patterns *look like* they do and what you can actually capture. The figure lays it out as bars.

![A bar comparison in two panels: on the left, the hindsight win rate of 62 percent shrinking to a tradeable 50 percent once a confirming bar is required, and on the right, the expectancy at 1.5-to-1 payoff falling from plus 0.55R in hindsight to plus 0.25R when tradeable to plus 0.15R after a 0.1R cost, with a note that this is a roughly 73 percent haircut from honesty alone](/imgs/blogs/reversal-candlestick-patterns-8.png)

Suppose you backtest a hammer-at-support pattern *the naive way*: you scroll through historical charts, find every hammer at support, and check whether price was higher some bars later. You get an exciting **62 percent** win rate. You feel like you have found gold.

Here is the trap. That 62 percent is measured *in hindsight* — you are counting hammers you can only identify *after* seeing how the bar and the next few bars resolved, and you are implicitly cherry-picking the clean ones. To actually trade this, you must (a) identify the hammer in real time, and (b) wait for a confirming bar before entering, because without confirmation a chunk of those "hammers" immediately failed. Once you require real-time identification and confirmation, the *tradeable* win rate drops to about **50 percent**. Same pattern, honest measurement, ten percentage points gone.

Now the expectancy, at a realistic **1.5:1** reward-to-risk (the conservative confirmation entry shrinks your payoff because you enter later, further from the stop):

- **Hindsight (62%, no real cost modelled):** $(0.62 \times 1.5) - (0.38 \times 1) = 0.93 - 0.38 = \mathbf{+0.55R}$. Looks fantastic.
- **Tradeable (50%, confirmed):** $(0.50 \times 1.5) - (0.50 \times 1) = 0.75 - 0.50 = \mathbf{+0.25R}$. Still positive — good! — but less than half of the hindsight figure.
- **After costs:** every trade pays the bid-ask spread plus slippage, conservatively **0.1R** per round trip. Subtract it: $0.25R - 0.10R = \mathbf{+0.15R}$.

So the *real* edge of this pattern, honestly measured, is **+0.15R per trade** — not the +0.55R the hindsight backtest screamed. That is a **roughly 73 percent haircut** from the paper figure, and the trade is *still worth taking* (+0.15R compounded over hundreds of trades is real money), but you would size and expect it completely differently than the hindsight number implies. **The one-sentence intuition: most of the "edge" in a naive candlestick backtest is hindsight and unpaid costs; the honest, tradeable edge is a fraction of it — but a positive fraction, which is the whole game.**

## Common misconceptions

These are the beliefs that cost beginners the most, each corrected with the *why*.

**"An engulfing candle always reverses the trend."** No — a bare engulfing candle follows through only around half the time. The engulfing is *evidence* of a momentum shift, not a guarantee of one. In a strong trend, engulfing candles against the trend fail constantly because the dominant force reasserts itself. The pattern is only worth trading *at a level, with the trend behind the move you are fading the pullback of, and confirmed.* Treating any engulfing as an automatic reversal is the fastest way to get run over by a trend that was never going to turn.

**"The bigger the wick, the stronger the signal, regardless of location."** No — wick size is meaningless without location. A two-inch lower wick in the middle of a range is just an intrabar excursion that snapped back; it rejected a price nobody cared about. The same wick *at a tested support level* is a rejection of a price with resting demand and trader memory behind it. The signal is the *coincidence of the rejection with a level*, not the raw length of the tail. A giant wick mid-nowhere is noise dressed up as drama.

**"You can trade these patterns anywhere."** No — this is the central error the whole post is built to correct. These patterns have near-zero edge in isolation and a modest edge only with context (location, trend, volume, confirmation). Trading them "anywhere" — scanning for every hammer or engulfing on the chart and taking them all — is a recipe for paying spreads and slippage on a coin-flip until your account bleeds out. The location filter is not optional polish; it is most of the signal.

**"A doji is a sell signal."** No — a doji is an *indecision* signal, not a directional one. It tells you the prior momentum stalled for one bar; it does not tell you which way the resolution goes. A doji in a strong uptrend is, more often than not, a pause before continuation, not a top. The doji's job is to flag *where* to pay attention (especially at a level) and to make you *wait* for the next bar to reveal the direction. Selling automatically on a doji means selling into a trend that resumes half the time.

**"Confirmation is just being slow — the aggressive entry is better because the price is better."** Half-true and dangerous. Yes, the confirmation entry gives a worse price and a smaller reward-to-risk. But it also filters out a large share of immediate failures, and the *net* expectancy of the confirmed entry is often higher despite the worse price, because the win-rate improvement outweighs the payoff shrinkage. Whether aggressive or confirmed wins depends on the specific setup and market — the point is that "better entry price" is not automatically "better trade," and assuming it is leads straight into the hindsight trap.

**"These patterns work the same in every market and timeframe."** No — a hammer on a daily chart of a liquid index and a hammer on a 1-minute chart of a thin micro-cap are not the same signal. Lower timeframes have more noise and more false patterns per real one; thin markets get pushed around by single orders, so wicks there carry less information. The order-flow story behind each pattern requires *enough participants* for the absorption and rejection to mean something. Treat the pattern's reliability as scaling with the liquidity and timeframe, not as a constant.

## How it shows up in real markets

The mechanics are easier to trust when you see them in named episodes. These are illustrative descriptions of recurring, well-documented *types* of events, with the caveat that exact prices and dates drift and you should treat any specific figure as approximate and as-of when it occurred, not as a live quote.

**A textbook bullish engulfing at a major index bottom.** Across many market bottoms in major equity indices — the kind that end a multi-week sell-off — the turn is frequently marked by a large bullish engulfing or a hammer on a daily chart, on the highest volume of the decline, right at a level the market had bounced from before. The March 2020 COVID-crash bottom in the S&P 500, for instance, was followed within days by enormous up-bars that engulfed prior sessions on record volume, at a level near prior 2018-2019 support. The mechanism from this post in action: panic selling exhausted itself, a wall of buying absorbed it at a remembered level, and the engulfing close sealed the verdict. The lesson is not that "engulfing candles call bottoms" — most do not — but that *real* bottoms, when they happen, tend to *leave* engulfing or hammer footprints at levels on high volume, which is exactly the confluence the post describes. The pattern is necessary-ish, not sufficient.

**A shooting star at a crypto top.** Bitcoin and other crypto assets, which trade 24/7 and are heavily driven by leverage and round-number psychology, repeatedly print long upper-wick rejection candles (shooting stars and gravestone dojis) at round-number resistance after parabolic rallies — near \$20,000 in late 2017, near \$64,000-\$69,000 in 2021, near \$30,000 on various retests. The order-flow story is vivid: leveraged longs pile in chasing the breakout, price spikes to a new high, there are not enough fresh buyers to sustain it, a cascade of long liquidations and profit-taking slams it back down, and the bar closes near its low — a long upper wick at the round number. The rejection at a level that *everyone was watching* is what gives the wick meaning. Note the as-of caveat hard here: crypto levels and prices move violently and these figures are historical, not current.

**A hammer that failed because it was mid-range.** For every clean hammer-at-support that worked, there is a hammer that printed in the middle of a choppy range and went nowhere — or reversed and stopped out traders who took it. This is the *most common* real outcome and the least talked about, because losing trades do not get screenshotted into trading courses. A hammer in the middle of a multi-week chop is just one of dozens of long-wick bars the range produces; it has no level behind it, no trend to lean on, and its "rejection" rejected a price the market did not care about. Traders who took every hammer they saw, regardless of location, are the empirical reason the bare-pattern win rate sits near 50 percent. The failures are the base rate; the textbook winners are the survivors.

**How the same setups get stop-hunted.** The cruelest real-market behaviour is that the *cleanest-looking* setups are the ones most likely to be faded. Because everyone learns to put their stop "just below the hammer's wick at support," a large cluster of stop-loss orders builds up just below the level. Larger participants can see (or infer) that cluster, and a quick spike down through the level triggers all those stops — filling the big players' buy orders cheaply with the retail stop-sells — and *then* price reverses up, leaving an even longer wick. The hammer "worked," but everyone who placed the obvious stop got knocked out at the low first. This is why disciplined traders put stops a little *beyond* the obvious wick, accept a slightly larger risk, and treat the most obvious level as the most likely to get hunted. The related dynamic on breakouts — the fakeout through a level before the real move — is the same liquidity mechanism viewed from the other side, and is covered in [breakouts versus fakeouts](/blog/trading/technical-analysis/breakouts-vs-fakeouts).

**An earnings gap that "engulfed" but meant nothing.** A frequent trap in single stocks: a company reports earnings, the stock gaps up violently the next morning, and the daily candle technically "engulfs" several prior bars — a giant bullish engulfing. Beginners read it as a screaming reversal signal. But this engulfing was caused by an *information event*, not by the order-flow story of gradual absorption the pattern is supposed to encode. The gap is a repricing to new fundamentals, and the "engulfing" is an artefact of the gap, not a battle won at a level. The pattern's meaning depends on the *mechanism* that produced it; an engulfing produced by a news gap is not the same signal as one produced by intraday absorption at support, even though the shapes look identical. Always ask *what produced this candle* before trusting its shape.

## When this matters to you, and where to go next

If you take one thing from this post, make it this: **the candle is never the signal by itself.** The shape — engulfing, pin bar, doji — is a compressed record of a fight, and a fight only matters when it happens *somewhere that matters*, in a *context* that gives it meaning, *confirmed* by what comes next. The honest edge of these patterns is small, it lives almost entirely in location and confluence, and a large fraction of the win rate people quote is hindsight and unpaid costs. None of that makes them useless — a +0.15R real edge, repeated with discipline and a tight stop, is exactly the kind of small, durable advantage that compounds. It just means you have to be ruthlessly honest about what they do.

This matters to you the moment you find yourself about to act on a candle. Before you do, run the checklist this post built: *Is it at a tested level? Is it aligned with the higher-timeframe trend, or at a genuine extreme? Is there volume behind it? Has the next bar confirmed? And can I place a tight stop just beyond the wick that gives me at least 2:1, ideally 3:1, to the next level?* If the answer to most of those is no, the candle is noise, and the most profitable thing you can do is nothing.

To go deeper from here: the post on [how a price chart is born](/blog/trading/technical-analysis/how-a-price-chart-is-born) grounds everything in how candles are constructed from raw trades; [support and resistance: why levels exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist) is the indispensable companion, because location is most of the signal and that post explains *why* levels are real; [trend and market structure](/blog/trading/technical-analysis/trend-and-market-structure) is how you read the higher-timeframe context that filters these patterns; [breakouts versus fakeouts](/blog/trading/technical-analysis/breakouts-vs-fakeouts) covers what happens when a level *breaks* instead of holds, the mirror of everything here; and [expectancy: why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) is the math that turns a pattern with a tight stop into a profitable system. Read those four and the candle stops being a mystery and becomes what it always was: a small, honest piece of evidence about a fight at a price. This is educational material about how markets behave, not advice to buy or sell anything.
