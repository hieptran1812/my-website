---
title: "Candlestick anatomy and the honest evidence: what one candle tells you, and what the statistics say"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A first-principles tour of the Japanese candlestick: how open, high, low, and close compress into a body and two wicks, how to read the battle inside one bar, and why the large-sample evidence says single-candle patterns have only modest, context-dependent edges."
tags: ["technical-analysis", "candlesticks", "price-action", "ohlc", "bulkowski", "doji", "hammer", "candlestick-patterns", "market-structure", "expectancy", "trading-psychology"]
category: "trading"
subcategory: "Technical Analysis"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A Japanese candlestick compresses four numbers — open, high, low, close — into one shape that shows the *battle* inside a single period. Learn to read it fluently, but be honest about what it predicts: not much, on its own.
>
> - A candle has a **body** (the distance from open to close) and two **wicks** or *shadows* (the thin lines reaching up to the high and down to the low). Green/up means it closed above its open; red/down means it closed below. That's the entire alphabet.
> - A **long body** is conviction, a **long wick** is rejection (price went there and got pushed back), and a **tiny body** — a *doji* — is indecision. Where the candle *closes* inside its range — near the high or near the low — tells you who held the floor at the bell.
> - The romantic lore ("this pattern predicts a reversal") has weak, context-dependent evidence. Large-sample studies — most famously Thomas Bulkowski's — find single candles resolve the "predicted" way only a little more than half the time, and the edge mostly comes from *context*: a level, the trend, and the next bar's confirmation.
> - The single most useful reframe: **a candle is information about who won the period that just ended, not a fortune cookie about the period that hasn't started.** The same hammer is meaningful at a tested support level and almost meaningless floating in the middle of nowhere.
> - The one number to remember: a pattern that is "right" 55% of the time, with average wins equal to average losses, earns about **+0.10R per trade before costs** — and a typical round-trip cost of ~0.10R erases the whole edge. Reliability percentages are not the same as profit.

Open any trading app and you will see them: rows of little colored rectangles with thin lines poking out the top and bottom, marching left to right. Each one is a *candlestick*, and around them has grown one of the most seductive bodies of folklore in all of finance. A candle that looks like a hammer "signals a bottom." A "shooting star" warns of a top. A "doji" means the trend is about to turn. There are books with hundreds of named patterns, each with a confident story about what it foretells. It is beautiful, it is memorable, and most of it is, to put it gently, not supported by the evidence at the strength people believe.

This post does two things at once, and refuses to let go of either. First, it teaches you to read a candle *fluently* — because the candle itself is genuinely useful information, a compact and honest record of a real fight between buyers and sellers. Second, it tells you the truth about the predictive lore: when researchers run the named patterns across tens of thousands of historical cases, the edges are small, conditional, and easy to eat with trading costs. The two halves are not in tension. The candle is real and worth reading; the fortune-telling laid on top of it is mostly decoration. By the end you should be able to look at any single bar and say precisely what it does and does not tell you.

![Anatomy of one candle showing a green bull candle that closes above its open and a red bear candle that closes below its open, each with a body spanning open to close and thin wicks reaching up to the high and down to the low](/imgs/blogs/candlestick-anatomy-and-the-honest-evidence-1.png)

The diagram above is the mental model for the whole post: a candle is four prices — open, high, low, close — drawn as one shape. The thick part, the *body*, spans from the open to the close. The thin lines, the *wicks*, reach from the body up to the highest price traded and down to the lowest. The color encodes one bit of information: did it close above where it opened (green, an up bar) or below (red, a down bar)? That is the complete alphabet. Everything else — every named pattern, every reversal story — is just words spelled out of those few letters. Let's learn the letters first, then read the words honestly.

This is the first post in Track 2 of the series [Technical Analysis, Honestly](/blog/trading/technical-analysis/what-technical-analysis-really-is), and it builds directly on the foundations from Track 1. If you have not yet read [how a price chart is born](/blog/trading/technical-analysis/how-a-price-chart-is-born), it is worth a detour, because a candle only makes sense once you know it is a *summary* of thousands of individual trades, not a fact handed down from on high.

## Foundations: the four prices in every candle

Before any pattern, any story, any prediction, there is the raw material: four numbers per period. Let's define each one from scratch, because the entire rest of the post is built on them.

Pick a period — say, one trading day. Over that day, a stock trades thousands of times, each trade *printing* a price (a print is just a recorded trade; see [how a price chart is born](/blog/trading/technical-analysis/how-a-price-chart-is-born) for the machinery). From that frantic stream of prints, we extract exactly four numbers:

- The **open** is the price of the *first* trade of the period — where the auction started for the day.
- The **high** is the *highest* price any trade printed during the period — the most anyone paid, even for an instant.
- The **low** is the *lowest* price any trade printed — the least anyone accepted.
- The **close** is the price of the *last* trade of the period — where the auction ended for the day.

These four numbers — abbreviated **OHLC** — are the candle. They throw away an enormous amount of detail (the order of the trades, the volume at each price, the path price took to get from open to close), but they keep the four corners of the fight, and it turns out those four corners carry a lot.

### Body, wick, and color: drawing the four prices

Now we turn four numbers into one shape, and the convention is simple and consistent across every charting platform in the world.

The **body** is a rectangle drawn between the open and the close. If the close is *above* the open, the period ended higher than it started — buyers won the net battle — and the body is drawn **green** (or hollow/white in older conventions). This is an **up candle** or **bull candle**. If the close is *below* the open, the period ended lower than it started — sellers won the net battle — and the body is drawn **red** (or filled/black). This is a **down candle** or **bear candle**. The color is not about whether the price went up *versus yesterday*; it is purely about close-versus-open *within this one period*. A stock can close green on a day it fell sharply from the prior close, as long as it closed above where *today* opened.

The **wicks** — also called **shadows** or **tails** — are the thin vertical lines extending above and below the body. The **upper wick** reaches from the top of the body up to the high; the **lower wick** reaches from the bottom of the body down to the low. The wicks record *the extremes price reached but did not hold*. If a stock opens at \$100, spikes to \$105, but closes at \$101, the body spans \$100 to \$101 and a long upper wick stretches up to \$105 — a visible record that price *visited* \$105 and got rejected back down.

It is worth pausing on *why* the candlestick form, invented by Japanese rice traders centuries ago, beat the plainer Western "OHLC bar" (a vertical line with two little ticks for open and close). The candle's edge is purely visual: the *filled body* makes the open-to-close distance and direction jump out at a glance, and the *color* lets you scan a hundred bars and instantly see the rhythm of green and red — the ebb and flow of who has been winning. The same four numbers are in a Western bar, but you have to *read* them; in a candle you *see* them. That visual immediacy is the candle's genuine contribution, and it is entirely about presentation, not prediction. The candle does not know anything the four numbers don't; it just shows the four numbers in a way your eye parses faster.

There is one beautiful asymmetry worth internalizing immediately. On a **green** candle, the open is the *bottom* of the body and the close is the *top* (because close > open). On a **red** candle, it flips: the open is the *top* of the body and the close is the *bottom* (because close < open). So the body always shows you open and close — but which edge is which depends on the color. The high is always the top of the upper wick and the low is always the bottom of the lower wick, regardless of color.

That is the whole foundation. Four prices, one body, two wicks, one color bit. A practitioner can skim this section; a beginner cannot proceed without it, because every single thing that follows is a sentence written in this alphabet. Before we read sentences, let's make sure we can read individual letters with total fluency — which is what reading the "balance of power" in one candle really means.

## Reading one candle: the balance of power

Here is the central idea that separates people who *read* candles from people who merely *memorize patterns*: a candle is a record of a **negotiation**, and its shape tells you how the negotiation went. Buyers push price up; sellers push it down; the open is where they started and the close is where they finished. Everything the candle shows you is about the *balance of power* between those two sides over the period.

![Four mini-candles illustrating the balance of power: a long body showing conviction, a small body with a long upper wick showing rejection, a doji with open equal to close showing indecision, and a candle closing near its high showing buyers held control](/imgs/blogs/candlestick-anatomy-and-the-honest-evidence-2.png)

The figure above lays out the four readings you should be able to make instantly. Let's walk each one.

**A long body is conviction.** When the body is large — the close is far from the open — one side dominated the whole period. A long green body means buyers took control at the open and never gave it back, pushing the close far above the open. A long red body means sellers did the same in reverse. The longer the body, the more decisive the period. This is the single most reliable piece of information in a candle, and it is also the least romantic: a big green bar means "buyers won this period decisively," full stop. It does not mean buyers will win tomorrow.

**A long wick is rejection.** A wick records a price that was *reached and abandoned*. A long upper wick means price pushed up into some level, found sellers waiting, and got shoved back down before the close — the high was *rejected*. A long lower wick means price fell into some level, found buyers, and got pushed back up — the low was rejected. Rejection is genuinely informative because it shows you where the *other side* stepped in with force. A long lower wick at a price level is the market saying "buyers defended here." Whether that defense holds is a separate question — but the wick is real evidence that someone with size showed up.

**A tiny body is indecision.** When the open and close are almost equal — the body is a thin sliver — neither side won. Price may have ranged widely within the period (long wicks both ways) but ended right back where it started. This is a *doji* (we will name it formally in a moment), and its message is honest and humble: "this period was a stalemate." Crucially, a stalemate is *not* a prediction of reversal. It is a statement that the prior momentum paused. What happens next is genuinely undecided — which is exactly why the next bar matters so much, as we will see.

**Where it closes inside the range tells you who held the floor.** This is the subtlest and most useful reading. Forget the open for a second and ask: of the entire range from low to high, *where did the close land?* A candle that closes in the top 10% of its range — right up near the high — means that whatever happened during the period, buyers were in control *at the bell*. A candle that closes near its low means sellers had the last word. The close is special among the four prices because it is the only one that represents a *settled* price — the level both sides agreed on when the period ended. A long green body that *also* closes right at its high is doubly bullish for the period; a long green body that closes well off its high (a meaningful upper wick) is bullish but with a hesitation stamped into it.

### Relative size is the whole game

One more principle, and it is the one beginners most often miss: **the size of a candle only means something relative to the candles around it.** A "long" body is long *compared to the recent bodies*. A two-point body is enormous on a stock that usually moves half a point in a day and trivial on one that swings ten points. There is no absolute scale. When you read a candle, you are always asking "is this body bigger or smaller than usual? Is this wick longer than the recent wicks?" A big green candle in a series of tiny ones is a genuine surge in conviction; the same candle in a series of equally big ones is just Tuesday. This is why context-blind pattern recognition fails so often: the *meaning* of a shape is relative, and a rule that says "a candle with a body 3× its wick is a marubozu" is measuring an absolute that the market does not respect.

> A candle's shape tells you the balance of power *inside the period that just ended*: long body is conviction, long wick is rejection, tiny body is indecision, and the close shows who held the floor at the bell. None of that is a prediction — it is a high-fidelity readout of a fight that is already over.

### The named single candles, defined

Now that you can read shape fluently, the named candles are easy — each is just a particular ratio of body to wick, and the name is a *description*, not a prophecy.

![A taxonomy tree of single candles sorted by body-to-wick ratio: marubozu and long candle under body-dominates conviction, and spinning top, doji, hammer, and shooting star under wicks-dominate indecision](/imgs/blogs/candlestick-anatomy-and-the-honest-evidence-3.png)

The taxonomy above sorts the named single candles by the one thing that actually distinguishes them — how much of the range is body versus wick.

- A **marubozu** (Japanese for "bald" or "shaved head") is a candle with a full body and essentially *no wicks*: it opened at one extreme and closed at the other, never reversing. A green marubozu opened at the low and closed at the high — total buyer control for the entire period. It is the purest conviction candle.
- A **long candle** (sometimes "long line") is a big body with small wicks — strong conviction, a little hesitation at the edges.
- A **spinning top** is a small body with *both* wicks long — price ranged widely up and down but ended near where it began. Indecision with volatility.
- A **doji** is the limiting case where the open and close are essentially *equal*, so the body is a horizontal line and the candle looks like a cross or plus sign. Pure indecision. There are sub-types — the *dragonfly doji* (open and close at the high, a long lower wick), the *gravestone doji* (open and close at the low, a long upper wick) — but they are all "open ≈ close."
- A **hammer** is a small body sitting at the *top* of its range with a long *lower* wick (at least twice the body). It says price fell hard during the period and then buyers shoved it all the way back up — a rejection of the lows.
- A **shooting star** is the mirror image: a small body at the *bottom* of its range with a long *upper* wick. Price rallied and then got slammed back down — a rejection of the highs.

Notice that *every one of these is a description of the body-to-wick ratio and the body's position in the range.* "Hammer" is not a verb meaning "will reverse." It is a noun meaning "small body, long lower wick." The leap from that description to a prediction is exactly where the evidence gets thin — which is the subject of the next section.

## The honest evidence: what Bulkowski and large samples actually show

Here is where most candlestick content quietly lies to you, usually by omission. It tells you a hammer "is a bullish reversal signal" and stops there, as if the shape *causes* the reversal. The honest question is: **across thousands of real hammers in real price history, how often did price actually go up afterward — and by how much, after costs?** That is a question you can answer with data, and people have.

The most thorough public effort is by **Thomas Bulkowski**, a researcher who hand-identified and tracked tens of thousands of chart and candlestick patterns and published the results in books like *Encyclopedia of Candlestick Charts*. His work is not gospel — it has methodology choices you can argue with, and his samples come from a particular era of U.S. equities — but it is one of the few large-sample, systematically-measured attempts to ask "do these patterns actually do what the folklore claims?" And the headline finding, stated plainly, is: **the edges are real but small, and they are mostly conditional on context.**

![A matrix of illustrative large-sample results showing hammer, shooting star, doji, spinning top, and bullish engulfing patterns, each with what it claims, how often it resolves that way at roughly 50 to 63 percent, and an honest caveat about needing a level, a trend, or confirmation](/imgs/blogs/candlestick-anatomy-and-the-honest-evidence-4.png)

The matrix above gives the shape of the result. (The percentages are *illustrative and approximate* — they vary by study, by market, by exactly how you define the pattern and the holding period, and they shift over time as markets change; treat them as "roughly this, as of the studies through the early 2020s," not precise constants.) A few things jump out, and they are the things the folklore leaves out.

**"Reliability" numbers are barely-above-coin and highly conditional.** When you read that a pattern is "60% reliable," the natural assumption is that it predicts the move 60% of the time in a way you can trade. But dig into how those numbers are computed and the picture gets murkier. The percentage is usually "how often price reached *some* target in *some* direction within *some* window" — and that target, direction, and window are chosen by the researcher. Worse, many patterns only show their modest edge *in the right context*: a hammer's bullish tendency mostly appears when it forms in a downtrend at a level, not floating in mid-range. Strip away the context and many single candles drift toward a 50/50 coin flip. A doji on its own is close to a literal coin: it tells you the trend paused, not which way it will resolve.

**Survivorship and selection haunt the lore.** Candlestick patterns were catalogued by people looking at charts and noticing "every time price reversed, there was often a hammer-ish thing near the bottom." That is selecting on the outcome. The reverse question — "of all the hammers, how many were followed by a reversal?" — gives a much less flattering answer, because the chart is full of hammers that led nowhere, and nobody circled those in the textbook. The named patterns survived into the canon partly *because* a few dramatic examples were memorable, not because a careful prospective study found them reliable. When Bulkowski did the prospective counting, the drama deflated.

**A pattern *appearing* is not a pattern *paying*.** This is the distinction that matters most for your money. Suppose a pattern really does resolve "correctly" 58% of the time. Fantastic — but a tradeable edge depends on *how much you make when right versus how much you lose when wrong*, and on *what it costs to trade*. If your average win equals your average loss, 58% is a thin edge; subtract the bid-ask spread, slippage, and commissions on every round trip, and a thin edge can become a negative one. The reliability percentage lives in a world without costs. Your account does not. We will make this concrete with arithmetic in a worked example below, and it ties directly to [why win rate lies and expectancy is what matters](/blog/trading/technical-analysis/expectancy-why-win-rate-lies).

**Context does the real work.** The honest summary of the whole literature is that the *candle* contributes a little information and the *context* contributes most of it. A pattern at a tested support or resistance level (see [why levels exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist)), in line with the prevailing trend, confirmed by the next bar, has a measurably better hit rate than the same shape in isolation. Take away the level, the trend, and the confirmation, and the residual edge of the bare shape is small enough that costs can swallow it. This is not a counsel of despair — it is a relocation of where the information lives. Stop asking "what does this candle predict?" and start asking "what does this candle, *here, in this context*, tell me about the balance of power?"

### A note on statistical honesty

If you want to be rigorous about whether a pattern has an edge at all, the tool is hypothesis testing: you frame a *null hypothesis* (the pattern has no edge — outcomes are 50/50 or match the base rate) and ask whether the observed hit rate is far enough from that null to be unlikely by chance. With a large enough sample, even a tiny real edge becomes "statistically significant" — but *statistically significant* and *economically significant after costs* are completely different bars. A pattern can clear the first and fail the second easily. If you want the machinery for thinking about this honestly — p-values, sample size, what significance does and does not mean — see [hypothesis testing and p-values](/blog/trading/math-for-quants/hypothesis-testing-pvalues-math-for-quants). The short version: with enough data you can *prove* a candle pattern has a 1% edge, and a 1% edge is worth exactly nothing once you pay the spread.

## Candles in context, not isolation

If there is one section to tattoo on your forearm, it is this one. **The same candle means opposite things in different places.** The shape is fixed; the meaning is supplied by where it appears.

![A before-and-after comparison showing the same hammer shape: in mid-range with no nearby level it has a base rate near 50 percent and barely beats a coin, while at a tested support level where buyers defended before it has an edge around 55 to 60 percent because the rejection confirms the level](/imgs/blogs/candlestick-anatomy-and-the-honest-evidence-5.png)

The figure above is the whole argument in one picture. On the left, a hammer floats in the middle of a range — no nearby level, no obvious reason for buyers to have stepped in. The long lower wick is real, but it is rejection *of nothing in particular*, and its base rate of "leads to a bounce" is close to a coin flip. On the right, the *identical shape* sits right on a support level that buyers have defended before. Now the long lower wick is rejection *of a level that matters* — it is the visible footprint of the same buyers showing up again. The shape did not change. The location did, and the location is most of the information.

This generalizes. A doji at the top of a long uptrend (momentum stalling at a high) is a different animal from a doji in the middle of a quiet range (nobody cares). A long red bar that closes below a support level (a *break*) means something different from the same bar bouncing off support (a *hold*). You cannot read a candle in isolation any more than you can understand a single word of a sentence without the words around it.

### Multi-candle beats single-candle

This is also why the slightly more reliable patterns in the literature tend to be *multi-candle* patterns rather than single candles. An "engulfing" pattern — where one candle's body completely engulfs the previous candle's body — is two bars of information: yesterday's failure plus today's decisive reversal. A "morning star" is three bars: a down move, a doji (indecision), then an up move (resolution). These carry more information than any single candle because they encode a small *sequence* — a turn in the balance of power over several periods, not a single snapshot. The general principle: more bars, more context, more information. A single candle is the weakest unit of evidence; it is genuinely informative about *its own period*, and only weakly suggestive about the next.

### The role of the next bar's confirmation

The cleanest way to add context to a single candle is to *wait for the next bar*. A doji says "indecision" — fine, but indecision *resolves*, and the bar after the doji is what resolves it. If the candle after a doji closes strongly up, through the top of the doji's range, the stalemate broke in the buyers' favor and you now have real information. If it closes strongly down, it broke the other way. The doji didn't predict anything; the confirmation bar *supplied* the prediction. This is why disciplined readers talk about "waiting for confirmation" — it is not superstition, it is the recognition that a single candle's information is incomplete and the next bar completes it. The cost of waiting is that you enter later and give up some of the move; the benefit is that you are trading on resolved information instead of a guess. That tradeoff is real and unavoidable, and pretending the single candle already told you the answer is how people lose money on "signals."

## Timeframe and the candle

A candle is always a candle *of some period*, and the period changes everything about how much the candle's fight was worth.

A **daily candle** is six and a half hours of continuous fighting (for U.S. stocks; a full 24 hours for crypto and most FX). A great deal happened inside it — thousands of trades, multiple pushes and rejections, the whole day's worth of news and emotion — and the four numbers are a genuine compression of all of it. A daily candle's body and wicks are a meaningful summary because there is a meaningful amount of activity to summarize.

A **one-minute candle**, by contrast, is sixty seconds. Far fewer trades, far less information, and a much larger fraction of the movement is *noise* — random jostling from individual orders rather than a real shift in the balance of power. A "hammer" on a one-minute chart is mostly the random walk of a few orders; the "rejection" it shows might be one large order that happened to print and reverse. The lower the timeframe, the noisier each candle, and the weaker any inference you draw from its shape. This is not a small effect: a huge amount of the candlestick-pattern disappointment people experience comes from applying daily-chart folklore to one-minute or five-minute bars, where the patterns are almost pure noise.

There is a deeper subtlety, too: the candle's boundaries are *arbitrary*. A "daily" candle for a U.S. stock starts at the open and ends at the close, but a "daily" candle in crypto might be drawn from midnight UTC to midnight UTC — and a pattern that looks like a clean hammer under one boundary convention vanishes if you shift the day boundary by a few hours. The shape is partly an artifact of where you chose to slice time. This is a humbling fact: some of the "patterns" you see are created by the chosen period boundaries, not by the market.

To feel this concretely, take the same 24 hours of trading sliced two ways. Slice it into one daily candle and you might get a clean bullish bar: it opened low, closed high, simple. Slice the *same* 24 hours into 24 hourly candles and the story fragments into a dozen little fights — three red bars, a doji, a long-wicked rejection, then a run of green — none of which individually says "bullish day." Neither view is wrong; they are different compressions of the identical underlying prints. But a pattern that exists in one slicing can be invisible in another, which is a direct warning against treating any single candle's shape as a fundamental fact about the market rather than a fact about *your chart settings*. The market did not draw the candle; your timeframe choice did. Two traders looking at the "same" stock on different timeframes are, quite literally, reading different evidence.

### The close is the price that matters

Among the four prices, the **close** has a special status that grows more important the higher the timeframe. The open, high, and low are all *intraperiod* facts — they happened at some moment and then the market moved on. The close is the *settled* price: the level at which the period's auction finally cleared, the number that goes into the official record, the price against which positions are marked. When traders say "it closed above resistance," they mean something stronger than "it touched above resistance intraday," because a *close* through a level survived the entire period's worth of buying and selling pressure, while an intraday spike (a wick) did not. A breakout that only shows up as a wick — price poked through and got rejected back — is much weaker evidence than a breakout that *closes* through. The body's edges (open and close) are the prices both sides committed to; the wicks (high and low) are prices that were tested and rejected. When in doubt, weight the close.

## Worked examples

Reading candles is a skill, and skills are built by working through concrete cases. Here are four, each isolating one of the ideas above with explicit numbers. *These are educational illustrations of how to reason about candles, not trading advice.*

#### Worked example: decoding one candle from its OHLC

You are handed a single daily candle with these four prices: **open \$100, high \$105, low \$99, close \$104.5.** Without seeing the chart, what does this candle tell you? Let's decode it from the numbers alone.

![A single annotated candle decoding open 100, high 105, low 99, close 104.5: the body spans 100 to 104.5 which is 4.5 points, sitting inside a 6-point range, with a half-point upper wick and a one-point lower wick, and a verdict box reading strong bull bar that closes in the top of its range](/imgs/blogs/candlestick-anatomy-and-the-honest-evidence-8.png)

Step one: **is it green or red?** The close (\$104.5) is above the open (\$100), so it is a **green/up candle**. Buyers won the net battle.

Step two: **how big is the body?** Body = close − open = \$104.5 − \$100 = **\$4.5**. Compared to the total range (high − low = \$105 − \$99 = **\$6.0**), the body is \$4.5 / \$6.0 = **75% of the range**. That is a large body — most of the period's movement was decisive, directional buying, not back-and-forth chop.

Step three: **how long are the wicks?** Upper wick = high − close = \$105 − \$104.5 = **\$0.5**. Lower wick = open − low = \$100 − \$99 = **\$1.0**. Both are small relative to the \$4.5 body. The tiny upper wick means price pushed to \$105 and only gave back fifty cents — buyers barely got rejected at the top. The one-point lower wick means price dipped to \$99 early but buyers reclaimed it.

Step four: **where did it close?** The close at \$104.5 sits \$0.5 below the \$105 high — it closed in the **top ~8% of its range** ((104.5 − 99) / (105 − 99) = 5.5/6.0 = 92% of the way up). Buyers were firmly in control *at the bell*.

The verdict: this is a **strong bull bar.** A big body (75% of range), closing near the high (top 8%), with only a small upper wick. The honest reading is: *during this period, buyers took control early, defended a brief dip, pushed price up decisively, and held it near the highs into the close.* What it tells you about *tomorrow* is much less — but what it tells you about *today's fight* is crisp and unambiguous. **A candle is a high-resolution readout of the period that just ended; read that period precisely and don't over-claim about the next one.**

#### Worked example: a hammer at support versus the identical shape mid-range

Now the central lesson, with numbers. You see a **hammer**: small body, long lower wick. Say it opens at \$50.20, falls to a low of \$48.00, then recovers to close at \$50.00 — a tiny body (open \$50.20, close \$50.00, a 20-cent body) sitting atop a long \$2.00 lower wick. Identical shape in two scenarios.

**Scenario A — mid-range.** The stock has been chopping between \$45 and \$55 with no particular level near \$48. The hammer's low at \$48 is not a level anyone defended before; it is just where this particular dip happened to stop. What is the realistic base rate that price is higher a few days later? Roughly the unconditional base rate of the stock itself — call it ~50%, a coin flip. The long wick is real, but it is rejection of nothing structural. There is essentially no edge here; you would be trading noise.

**Scenario B — at tested support.** Now suppose \$48 is a level the stock bounced off *twice before* in the last two months — a genuine support level where buyers have repeatedly stepped in. The *same* hammer now means something: the long lower wick is the visible footprint of those same buyers defending \$48 *again*. The hit rate conditional on "hammer at a twice-tested support, in an overall uptrend" is meaningfully better than the mid-range case — call it ~55–60% in the kind of data Bulkowski reports, *because the context, not the shape, supplies the edge.*

The numbers that matter: the shape contributed the *same* zero-to-small base information in both scenarios. The difference between a coin flip and a ~55–60% read came entirely from the level. **The hammer is not the signal; the hammer at a level is the signal, and the level is doing most of the work.** This is why context-free pattern trading disappoints, and why the same shape can be a great read or a useless one depending only on where it sits.

#### Worked example: the "55% reliable" trap

This is the most important arithmetic in the post, because it is where "reliability" and "profit" part ways. Suppose you have found a genuinely 55%-reliable pattern: across a large sample, price went your way 55% of the time and against you 45% of the time. Suppose further — to keep it clean — that your average win equals your average loss in size: you risk 1R (one unit of risk) to make 1R. Is this a money-maker?

![A bar chart showing why 55 percent reliable can be break-even: wins of 0.55 times plus 1R equal plus 0.55R, losses of 0.45 times minus 1R equal minus 0.45R, giving a gross edge of plus 0.10R, from which a cost of minus 0.10R for spread and slippage leaves a net edge of 0.00R, exactly break-even](/imgs/blogs/candlestick-anatomy-and-the-honest-evidence-6.png)

Let's compute the *expectancy* — the average profit per trade, measured in R (see [expectancy and why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) for the full treatment).

$$E[R] = p \cdot W - (1-p) \cdot L$$

where $p$ is the win probability, $W$ is the average win in R, and $L$ is the average loss in R. Plugging in $p = 0.55$, $W = 1$, $L = 1$:

$$E[R] = 0.55 \times 1 - 0.45 \times 1 = 0.55 - 0.45 = +0.10\text{R}$$

So *before costs*, your 55%-reliable pattern earns **+0.10R per trade** — a tenth of your risk unit, on average. That is a real, positive edge. But now subtract the cost of actually placing the trade: the bid-ask spread you pay on the way in and out, plus slippage, plus any commission. On a liquid stock with a tight spread these costs might be small; on a less liquid one, or trading frequently, a realistic round-trip cost can easily be on the order of **0.10R**. Subtract it:

$$E[R]_{\text{net}} = +0.10\text{R} - 0.10\text{R} = 0.00\text{R}$$

**Break-even.** Your 55%-reliable pattern, after realistic costs, makes you exactly nothing — and if costs run a hair higher than 0.10R, it makes you *less* than nothing while you do all the work and take all the stress. This is the trap in one line: **a reliability percentage tells you the win rate, but the win rate is not the edge.** The edge is expectancy, expectancy depends on the win/loss *sizes* as well as the rate, and the *net* edge depends on costs. A pattern can be "reliable" and unprofitable at the same time, and most of the marketed candlestick patterns live in exactly that zone. **Never evaluate a pattern by its hit rate alone; compute the expectancy, then subtract what it costs to trade it.**

#### Worked example: a doji resolved by the next bar

Finally, the confirmation lesson with numbers. You are watching a stock in a steady uptrend. It prints a **doji**: open \$80.05, close \$80.00 — a 5-cent body, essentially open = close — with wicks up to \$81 and down to \$79. The period was a perfect stalemate: buyers pushed to \$81, sellers pushed to \$79, and they finished dead even at \$80.

![A three-candle figure showing a prior up bar, then a doji where open equals close in the middle, with two possible next bars: answer A is a green candle that closes up meaning buyers win, answer B is a red candle that closes down meaning sellers win, illustrating that the next bar supplies the direction the doji did not](/imgs/blogs/candlestick-anatomy-and-the-honest-evidence-7.png)

What does the doji predict? *Nothing, on its own.* The folklore says "a doji after an uptrend signals a reversal," but the honest reading is "the uptrend's momentum paused — the relentless buying stopped for one period." Whether that pause becomes a reversal or just a breather is genuinely undecided. The doji is a question, not an answer.

Now the next bar resolves it. **Answer A:** the following candle opens at \$80.10 and closes strongly at \$82 — a big green body that closes *above the top of the doji's range* (\$81). The stalemate broke decisively in the buyers' favor; the uptrend resumed, and you now have real information (the pause was a breather). **Answer B:** the following candle opens at \$79.90 and closes at \$78 — a big red body closing *below the bottom of the doji's range* (\$79). The stalemate broke in the sellers' favor; the doji marked the moment the uptrend's momentum died, and the reversal the folklore *hoped* for actually materialized — but you only knew it *after* the confirmation bar, not from the doji.

The lesson in numbers: the doji's body (5 cents) carried almost no directional information. The next bar's body (a \$2 move that closed beyond the doji's range, in one direction or the other) carried *all* of it. **A doji tells you the fight paused; the bar after the doji tells you who won the next round — wait for it, because the candle alone is genuinely a coin flip.**

## Common misconceptions

Beginners (and plenty of non-beginners) carry a handful of beliefs about candles that are wrong in ways that cost money. Here are the most common, corrected.

**"A green candle means buyers are in control of the future."** A green candle means buyers were in control of *the period that just ended* — close finished above open. It is a fact about the past, not a forecast. The very next candle can be a long red one, and frequently is. The candle's color is a scoreboard for a game that is already over, not a prediction of the next game. Confusing "won the last period" with "will win the next period" is the single most common candle error, and it is why people buy the top of a strong green bar and then watch it reverse.

**"Candlestick patterns predict reversals."** Most named patterns have, in large samples, a modest and conditional tendency that is easy to overstate and easy to lose to costs. A pattern "predicting" a reversal usually means "in a particular context, price reversed somewhat more often than chance" — which is a far weaker claim than the textbook implies. Strip away the context (the level, the trend, the confirmation) and most single-candle patterns drift toward 50/50. The pattern is not a crystal ball; at best it is a small nudge to a probability, and only in the right setting.

**"The wick shows where price is *going*."** A wick shows where price *went and got rejected* — it is a record of the past, specifically of a level that was tested and did not hold *for that period*. A long upper wick does not mean price will go back up to the wick's tip; if anything it is evidence that sellers defended that area. People sometimes treat the wick's extreme as a target for the next move, which inverts its meaning: the wick is the high-water mark of a tide that already went out, not an arrow pointing where the next tide will reach.

**"Bigger-timeframe candles are *always* more reliable."** Higher timeframes are *less noisy* per candle, which is genuinely valuable — a daily candle summarizes far more real activity than a one-minute candle, and its shape carries more signal. But "less noisy" is not the same as "reliable enough to trade profitably." A weekly hammer at a major level is better evidence than a one-minute hammer, but it is still just one piece of context-dependent information, and it can still fail. The right statement is "lower timeframes are mostly noise; higher timeframes carry more signal but are not oracles." And higher timeframes come with their own cost: far fewer signals, so far less data to ever know whether your read has an edge.

**"A doji always signals a reversal."** A doji signals *indecision* — the prior momentum paused. Sometimes that pause precedes a reversal; very often it precedes a continuation after a one-period breather. The doji itself does not distinguish the two; only the next bar does. Treating every doji as a reversal signal means fading a lot of trends that simply caught their breath.

**"More patterns memorized means a better trader."** The number of named patterns you can recite is nearly uncorrelated with profitability. What matters is reading the *balance of power* and the *context* — and those are a handful of principles (body = conviction, wick = rejection, close = who held the floor, location = most of the meaning), not a flashcard deck of hundreds of names. A trader who reads four principles fluently in context will beat one who has memorized two hundred patterns and applies them blind.

## How it shows up in real markets

The principles from the sections above become vivid when you watch them play out. Here are several named episodes and recurring real-market phenomena. *Specific historical figures are cited as-of the dates given and are approximate; the point is the mechanism, not the exact tick.*

### A famous reversal day that was "obvious" only in hindsight

Pick almost any major market bottom and you will find a dramatic single candle near the low — a long-lower-wick "hammer" or a giant green "engulfing" bar — that, *after the fact*, looks like it screamed "buy." The reversal off the COVID-crash lows in late March 2020 is a textbook example: around March 23, 2020, the S&P 500 carved out a low and then ripped higher, and the daily candle near the bottom had the long-lower-wick shape the folklore loves. The seductive story is "the hammer called the bottom." The honest story is that *thousands* of similar-looking hammers had printed on the way down over the prior month, and every one of them failed — price kept falling. The one at the actual bottom was indistinguishable *in shape* from the failed ones; what made it "the bottom" was everything else (unprecedented policy response, capitulation selling exhausting, valuations) that you could not read off the candle. Survivorship makes the winning hammer famous and buries the dozens of identical losers. The lesson: the bottom candle is obvious *only* in a chart annotated after the fact.

### A doji at a major top

The same survivorship runs in reverse at tops. Major tops are often marked, in hindsight, by a doji or a small-bodied indecision candle right at the high — the moment the relentless buying finally stalled. You can find one near the late-2021/early-2022 highs in many growth stocks and indices: a doji or spinning top at the peak, followed by a long decline. The pattern-seller's pitch is "the doji warned you." The reality is that dojis print constantly throughout an uptrend — at minor pauses, at midday lulls, at dozens of points that were *not* the top — and only the one that happened to sit at the actual peak gets circled in the retrospective. A doji genuinely tells you momentum paused; it does not tell you *this* pause is the one that becomes a top. The confirmation (a decisive close below the prior support, a break of the trend's structure) is what distinguished the real top, and that came *after* the doji.

### How algos exploit retail candle-pattern triggers

Here is a less comfortable real-market fact: because so many retail traders are taught the same candlestick rules, those rules become *predictable behavior* that more sophisticated participants can anticipate and trade against. If thousands of retail traders are taught to "buy when a hammer forms at support" and to "put your stop just below the wick's low," then there is a known cluster of buy orders and a known cluster of stop-loss orders at predictable prices. Faster, better-capitalized players know exactly where those orders sit. A common pattern: price is pushed *just* below the hammer's low — far enough to trigger the retail stops (a cascade of forced selling) — and then snaps back up, having harvested the liquidity. The retail trader, who did everything the book said, gets stopped out at the worst possible tick right before the move they predicted. This is not a conspiracy theory; it is the natural consequence of a widely-taught, mechanical rule creating a predictable order cluster. The more mechanically a pattern is traded by a crowd, the more it becomes a *target* rather than an edge.

### The gap between backtested candle patterns and live results

When quantitative researchers backtest pure candlestick patterns — coding up "hammer," "engulfing," "morning star" as precise rules and running them across decades of data — the results are famously underwhelming, and the gap between backtest and live trading is even worse. Several effects compound. First, the in-sample edge is usually small to begin with (consistent with Bulkowski's modest numbers). Second, costs eat much of it, exactly as in the 55%-reliable worked example. Third, the precise rule needed to backtest a pattern ("body < 30% of range, lower wick > 2× body") is more rigid than the fuzzy human pattern, and small definition changes swing the results — a sign the edge is fragile, not robust. Fourth, *live* execution adds slippage and the algos-front-running effect above, so even a backtest that showed a thin edge often goes negative live. The recurring finding across this literature is that single-candle patterns, traded mechanically, rarely survive contact with realistic costs — which is precisely why the honest framing is "read the candle for what it tells you about the period, and let context and risk management, not the pattern, carry the decision." To do this kind of evaluation without fooling yourself, the [hypothesis-testing](/blog/trading/math-for-quants/hypothesis-testing-pvalues-math-for-quants) and [expectancy](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) frameworks are the right tools.

### When candles genuinely add value

To be fair to the candle: there are real settings where reading one well *does* help, and they are all context-rich. A long-lower-wick rejection of a major, multiply-tested support level, in an uptrend, confirmed by a strong next bar, is a genuinely better-than-coin read — not because the wick is magic, but because it is the visible confirmation that the *level* (which has the real information; see [why levels exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist)) is being defended again. A decisive marubozu close *through* a long-standing resistance level is meaningful evidence of a real breakout, because a full-bodied close through a level survived an entire period's selling. In both cases the candle is *confirming a structural fact* (a level holding or breaking), not predicting out of thin air. That is the candle at its honest best: a high-resolution readout of how the current period's fight resolved *at a place that already mattered.*

## When this matters to you and further reading

Here is where this actually touches your decisions. The next time you see a candle — in a chart, in a tweet, in a "this pattern signals X" post — run it through the honest checklist this post built. **First, read the period precisely:** is the body long (conviction) or short (indecision)? Are the wicks long (rejection) and on which side? Where did it *close* in its range (who held the floor)? That reading is reliable, because it is just a description of a fight that already happened. **Second, locate it:** is it at a tested level, in line with the trend, or floating in mid-range? The location supplies most of the meaning; the same shape is a real read at a level and noise in the middle. **Third, demand confirmation:** has the next bar resolved the question the candle posed, or are you guessing? **Fourth, do the expectancy math:** even if the pattern is "reliable," what is the expectancy after the win/loss sizes and the cost of trading it? A reliability percentage with no expectancy attached is marketing, not evidence.

The reframe to keep is the one from the top: **a candle is information about who won the period that just ended, not a fortune cookie about the period that hasn't started.** Read it for what it is — a compact, honest record of a real negotiation — and you will get genuine value from it. Read it as prophecy, and you will join the long line of traders who memorized two hundred patterns and lost money to the spread.

Where to go next. The sibling post in this track, [reversal candlestick patterns](/blog/trading/technical-analysis/reversal-candlestick-patterns), takes the multi-candle patterns (engulfing, morning/evening star, tweezer tops and bottoms) and runs them through the same honest lens — more bars, a little more information, the same demand for context and confirmation. To understand *why* the levels that give candles their meaning exist at all, read [support and resistance: why levels exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist). To make sure you never confuse a reliability percentage with a profitable edge, [expectancy: why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) is the essential companion to this post — the 55%-reliable trap is its whole subject. To ground everything in where a candle even comes from, revisit [how a price chart is born](/blog/trading/technical-analysis/how-a-price-chart-is-born). And to test any pattern claim with statistical honesty rather than wishful thinking, [hypothesis testing and p-values](/blog/trading/math-for-quants/hypothesis-testing-pvalues-math-for-quants) gives you the tools to ask "is this edge real, and is it big enough to matter after costs?" — the two questions that separate reading candles from being read by the people selling them.

*This post is educational and explains mechanisms and evidence; it is not financial advice and recommends no trades.*
