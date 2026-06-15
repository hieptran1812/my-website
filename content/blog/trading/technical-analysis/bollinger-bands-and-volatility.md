---
title: "Bollinger Bands and Volatility: The Squeeze, the Walk, and What the Bands Really Measure"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Bollinger Bands are a volatility envelope, not a magic reversal signal. This is the honest, statistical reading: how the bands are built from a moving average and standard deviation, why a band touch is common rather than rare, the two edges that survive scrutiny (the squeeze and the walk), and how ATR turns volatility into a stop and a position size."
tags:
  [
    "bollinger-bands",
    "volatility",
    "standard-deviation",
    "atr",
    "average-true-range",
    "the-squeeze",
    "mean-reversion",
    "position-sizing",
    "technical-analysis",
    "moving-averages",
    "fat-tails",
    "risk-management",
  ]
category: "trading"
subcategory: "Technical Analysis"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** -- Bollinger Bands wrap a moving average in two lines set a fixed number of standard deviations away, so they are literally a *volatility envelope*: the bands flare wide when the market is noisy and pinch narrow when it is calm. Their whole content is "how far is price from its recent average, measured in units of recent volatility."
>
> - The **middle band** is a 20-period simple moving average; the **outer bands** are that average plus and minus $k$ times the standard deviation of the same 20 closes, with $k = 2$ the default. A *standard deviation* (sigma) is just the typical distance of the data from its own average.
> - For a perfect bell curve about **95%** of values sit within two standard deviations -- but real market returns are **fat-tailed**, so only about **88 to 89%** sit inside the 2-sigma bands. A band touch is roughly one bar in nine: common, *not* a reversal signal.
> - The two edges that survive honest scrutiny are the **squeeze** (volatility is mean-reverting, so unusually narrow bands often precede a large move -- though the direction is unknown) and recognizing the **walk** (in a strong trend price rides the upper or lower band for a long time, and fading every touch is the classic losing trade).
> - **Average True Range (ATR)** is the other volatility ruler: the typical dollar range of a bar. An ATR-based stop adapts to volatility, so the same percentage risk gives you a wider stop in chop and a tighter one in calm -- which is why volatility-based stops beat fixed-dollar stops.
> - The single number to remember: on a \$10,000 account risking 1% (\$100) per trade, a 2-ATR stop of \$2 sizes you to exactly **50 shares**. Volatility, not your gut, sets the size.

A trader puts Bollinger Bands on a chart for the first time and within an hour has a theory of everything. Price touched the upper band -- sell, it is overbought. Price touched the lower band -- buy, it is oversold. The bands "contain" price, so price must bounce between them like a ball in a box. It feels like a machine that prints the future. Then the trader shorts an upper-band touch in a roaring bull market, gets run over for two weeks straight, and concludes the indicator is broken.

The indicator is not broken. The theory was wrong. Bollinger Bands do exactly one thing, and it is not predict reversals. They draw a *volatility envelope* around a moving average -- a band whose width is a direct, mechanical readout of how spread-out recent prices have been. Everything useful about them follows from understanding that they measure volatility, and almost everything harmful follows from mistaking the envelope for a fence that price cannot leave. This post is the honest version: how the bands are built from arithmetic you can do by hand, what fraction of price action actually lands inside them, the two situations where they genuinely help, and the misconceptions that cost people money.

![Bollinger Bands are a volatility envelope: the upper and lower bands flare wide when prices are noisy and pinch narrow when prices are calm, while price wanders inside.](/imgs/blogs/bollinger-bands-and-volatility-1.png)

The diagram above is the mental model for the whole post. The middle line is a moving average; the two outer lines are the same average pushed up and down by a multiple of recent volatility. When the market is noisy, that volatility is large and the bands flare apart. When the market goes quiet, the volatility collapses and the bands pinch toward the average -- the *squeeze* in the middle of the figure. Price itself just wanders around inside, occasionally poking a band and coming back, occasionally riding one for a long stretch. The bands do not push price around; they *describe* where price has been relative to its own recent average. Keep that image in mind and the rest of this post is detail.

A note before we start: this is educational. It explains the mechanics and statistics of a volatility indicator so you can read any "the bands say sell" claim honestly. It is not advice to trade anything, and every method described here can lose money in ways we will be specific about.

## Foundations: a band is a volatility envelope

We are going to build a Bollinger Band from nothing, one piece at a time, with numbers small enough to check in your head. By the end of this section you will be able to compute a band by hand, which is the only way to truly stop believing it is magic.

### The middle band: a moving average

The center line of a Bollinger Band is a **moving average** -- specifically a 20-period **simple moving average (SMA)**. A simple moving average is exactly what it sounds like: take the last 20 closing prices, add them up, divide by 20. That is the average price over the window. "Moving" means that on each new bar you drop the oldest close, add the newest, and recompute -- the window slides forward one step at a time, so the average moves along with price.

Why an average at all? Because a raw price chart is jagged -- it jumps around on every bar from noise that has nothing to do with the underlying drift. Averaging the last 20 closes smooths that jitter into a single line that represents "where price has been lately, on balance." The 20-period choice is a convention (roughly a month of trading days); a shorter window hugs price more tightly and reacts faster, a longer one is smoother and slower. The middle band is just this smoothed centerline, and we cover the honest behavior of moving averages on their own in [the moving-averages backtest](/blog/trading/technical-analysis/moving-averages-honest-backtest). For Bollinger Bands, the moving average is only step one -- the half that everyone already knows. The interesting half is the width.

### Standard deviation: measuring spread

Here is the term that makes Bollinger Bands what they are. The **standard deviation** -- written with the Greek letter sigma, $\sigma$ -- is a measure of how *spread out* a set of numbers is around their own average. Small standard deviation means the numbers cluster tightly near the average; large standard deviation means they are scattered far from it.

Concretely, you compute it like this. Take your 20 closes. Find their average (that is the middle band). For each close, measure how far it is from the average, and square that distance (squaring makes every distance positive and punishes big deviations more). Average those squared distances -- that average is the **variance**. Then take the square root of the variance to get back to the original units (dollars). That square root is the standard deviation. We work through this arithmetic carefully, with the meaning of variance and the higher *moments* of a distribution, in [the expectation, variance, and moments primer](/blog/trading/math-for-quants/expectation-variance-moments-math-for-quants).

In one formula, the standard deviation of $N$ closes $p_1, \dots, p_N$ with average $\bar{p}$ is:

$$\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(p_i - \bar{p}\right)^2}$$

where $p_i$ is the $i$-th closing price, $\bar{p}$ is the 20-period average (the middle band), and $N = 20$. The term inside the sum is each close's squared distance from the average; the sum-then-divide is the variance; the square root brings it back to dollars. (Bollinger's original calculation uses the *population* standard deviation -- divide by $N$, not $N-1$ -- which barely matters at $N = 20$.)

The single most important fact about standard deviation, for our purposes: **it is a volatility measure in the units of the price.** If a \$100 stock has been bouncing in a tight 50-cent range, its standard deviation might be \$0.40. If the same stock starts swinging in \$4 ranges, its standard deviation jumps to maybe \$2.50. The number directly reflects how violently price has been moving.

### Building the outer bands

Now we have both ingredients. The outer bands are the middle band pushed up and down by a multiple of that standard deviation:

$$\text{upper band} = \text{SMA}_{20} + k\,\sigma, \qquad \text{lower band} = \text{SMA}_{20} - k\,\sigma$$

where $\text{SMA}_{20}$ is the 20-period average (the middle band), $\sigma$ is the standard deviation of those same 20 closes, and $k$ is the band multiplier -- conventionally $k = 2$. So the default Bollinger Band is "the 20-day average, plus and minus two standard deviations." That is the entire construction.

![A Bollinger Band is a 20-period moving average wrapped in two lines set two standard deviations out, so it is literally a volatility envelope that widens and pinches with the spread of recent prices.](/imgs/blogs/bollinger-bands-and-volatility-2.png)

The figure above is the construction as a four-step pipeline: average the closes, measure their spread, push the bands two deviations out, and watch the envelope breathe. Notice what falls out of this for free: because $\sigma$ is in the bands' width, **the bands automatically widen when volatility rises and pinch when it falls.** Nobody has to adjust anything. A calm market has a small $\sigma$, so $2\sigma$ is small, so the bands sit close to the average. A wild market has a large $\sigma$, so the bands flare far apart. The band width *is* a volatility chart. That is the one true thing Bollinger Bands tell you, and it is genuinely useful -- as long as you read it as a volatility statement and nothing more.

### The bands breathe with volatility

Step back and look at what we have built. The middle line tracks the recent average price. The gap between the bands tracks the recent volatility. As volatility cycles -- calm, then violent, then calm again, which is how markets actually behave -- the bands rhythmically pinch and flare. People describe this as the bands "breathing," and it is the most honest one-word summary of the indicator. Width up means volatility up; width down means volatility down. The bands are a breathing chart of fear and calm, drawn in standard deviations.

### Where price sits inside the bands: %b

There is one more derived quantity worth knowing because it turns "price touched the band" into a precise number. It is called **%b** (percent-b), and it measures where the latest price sits *within* the band channel on a 0-to-1 scale: 0 means price is exactly on the lower band, 1 means exactly on the upper band, 0.5 means exactly on the middle band. The formula is just a rescaling:

$$\%b = \frac{\text{price} - \text{lower band}}{\text{upper band} - \text{lower band}}$$

where price is the current close, and the bands are today's upper and lower. If price is at \$103, the lower band at \$97, and the upper at \$109, then %b is (\$103 - \$97) / (\$109 - \$97) = \$6 / \$12 = 0.5 -- price is sitting right on the middle band. If price climbs to \$108 with the same bands, %b is (\$108 - \$97) / \$12 = 0.92 -- price is most of the way to the upper band. A %b above 1 means price has poked *above* the upper band; below 0 means *below* the lower band. We mention it because it is the honest way to talk about "band touches": instead of a fuzzy "price is at the band," %b gives you a clean number, and you will see later that even a %b of 1.0 (a clean upper-band touch) is, by itself, not a signal -- it is a measurement.

Everything from here is about reading that breathing correctly -- and resisting the three or four very tempting misreadings that the picture invites.

## What the bands actually mean statistically

The most common belief about Bollinger Bands is that they "contain" price -- that price lives between the bands and a touch of a band is therefore a rare, extreme event worth fading. To see how wrong this is, we have to look at the statistics of where price actually lands relative to the bands. The honest reading is humbling.

### The 95% that isn't there

Start with the textbook. If a set of numbers follows a **normal distribution** -- the symmetric bell curve, the workhorse of introductory statistics -- then a precise fraction of the values falls within each band of standard deviations from the mean. About 68% of values sit within one standard deviation, about **95% within two standard deviations**, and about 99.7% within three. This is the famous "68-95-99.7 rule." Since Bollinger Bands are set at two standard deviations, the naive inference is: 95% of price action should be inside the bands, so a touch of the band is a 1-in-20 event -- rare enough to fade.

There are two problems with that inference, and both matter.

The first is that the 95% figure is *exactly* true only for a perfect normal distribution. Market returns are not normally distributed. They have **fat tails** -- a fancy way of saying that large moves happen far more often than a bell curve predicts. A bell curve says a five-standard-deviation day should happen roughly once every several thousand years; markets produce them every few years. The distribution of returns has a taller, thinner peak (lots of small, quiet days) and fatter tails (rare but real violent days) than the normal curve. Because the tails are fatter, *less* of the mass sits inside two standard deviations.

![Under a normal distribution about 95% of values fall within two standard deviations, but fat-tailed market returns put only roughly 88 to 89% inside two-sigma bands, which makes a band touch an ordinary event rather than an extreme.](/imgs/blogs/bollinger-bands-and-volatility-3.png)

The matrix above lays out the coverage. The normal-distribution row gives the textbook 68/95/99.7. The real-market row gives what empirical studies of stock and index returns actually find: roughly 60 to 65% inside one sigma, about **88 to 89% inside two sigma**, and around 97 to 98% inside three. The bottom row translates each into trading terms.

![The fat-tailed market curve has a thinner peak and fatter tails than the normal one, so less of its mass sits inside two standard deviations and the band is touched more often.](/imgs/blogs/bollinger-bands-and-volatility-8.png)

The distribution figure makes the mechanism visual: the market curve (the bold one) is taller in the middle and fatter at the edges than the normal curve. The shaded note is the punchline -- because more mass lives in the tails, price spends more time outside the two-sigma envelope. If only about 88 to 89% of bars are inside the bands, then about **11 to 12%** are outside, which means a bar touches or exceeds a band roughly **one time in nine**, not one time in twenty. A 1-in-9 event is not rare. It is something you should expect to see several times a week.

### The second problem: the bands move

There is a subtler issue that makes "price reverts from the band" even less impressive than the raw frequency suggests. The bands are not fixed lines -- they are recomputed every bar from the most recent 20 closes. When price makes a big move toward a band, that very move *increases* the recent standard deviation, which *widens* the band, which can pull the band out of the way of the price that was chasing it. And when price then drifts back toward the average -- which it tends to do in a sideways market simply because the average is the center of the recent range -- it looks like price "bounced off the band." But often the band did half the work by moving, and the drift back to the middle was ordinary **mean reversion**: the tendency of a quantity that has wandered far from its average to drift back toward it, not because of any force but because the average is, by construction, the middle of where it has recently been.

So when someone shows you a chart where price tagged the lower band and rallied, the honest decomposition is: some of that was the band widening to meet an extreme close, and most of the rally was garden-variety mean reversion within a range that would have happened with or without the band drawn on the chart. The band did not cause the bounce. It marked a spot that was already far from the average, and far-from-the-average tends to revert in a range. The band is a *coincident description* of an extreme, not a *cause* of the reversal.

### Why "two standard deviations" is the wrong question

There is one more layer to the statistics that is worth getting straight, because it dissolves a lot of confusion. When we say "about 88 to 89% of bars are inside the two-sigma bands," we are describing a *long-run frequency* over many bars and many regimes. It is not a probability you can attach to any single bar in any single moment. In a quiet range, the fraction inside the bands might genuinely be 92 or 93% -- the market is well-behaved and rarely strays. In a strong trend, the fraction inside on the trending side might be far lower because price is walking the band. The 88 to 89% is an average across regimes that papers over enormous variation between them. This is the deepest reason "a touch is a 1-in-9 reversal" is wrong: the 1-in-9 is a blended frequency, and the *conditional* frequency -- given that you are in a trend, given that price is walking -- is completely different. Averages over mixed regimes are dangerous to trade on, because you are never in "the average"; you are always in one specific regime, and which one decides everything.

There is also a sampling subtlety. The standard deviation in the band formula is computed from only 20 closes. Twenty is a small sample. The standard error of an estimate of standard deviation from $N=20$ points is roughly $\sigma / \sqrt{2N} \approx \sigma / 6.3$ -- meaning the bands' own width is uncertain by something like 15% just from the smallness of the window. So the precise position of the band on any given bar is itself a noisy estimate. Reading great significance into whether price tagged the band exactly or fell a few cents short is false precision; the band is a fuzzy region, not a hard line, and treating its exact pixel as a trigger is reading noise.

#### Worked example: computing a 20-SMA and a 2-sigma band by hand

Let us build a band from scratch so the arithmetic is concrete. We will use a tiny 5-close window instead of 20 (the method is identical; 5 numbers are checkable by hand). Suppose the last five closes of a \$100 stock are: \$98, \$101, \$99, \$102, \$100.

Step 1 -- the middle band (the average). Add them: \$98 + \$101 + \$99 + \$102 + \$100 = \$500. Divide by 5: the average is \$100. That is the middle band.

Step 2 -- the deviations from the average. Subtract \$100 from each close: -\$2, +\$1, -\$1, +\$2, \$0.

Step 3 -- square each deviation: \$4, \$1, \$1, \$4, \$0. Sum them: \$10. Divide by 5 (the population variance): \$2. The **variance** is 2 (in squared dollars).

Step 4 -- the standard deviation: take the square root of the variance. $\sqrt{2} \approx \$1.41$. So $\sigma \approx \$1.41$.

Step 5 -- the bands. Upper = average + $2\sigma$ = \$100 + 2 × \$1.41 = \$100 + \$2.83 = **\$102.83**. Lower = average - $2\sigma$ = \$100 - \$2.83 = **\$97.17**.

So with this calm little series, the bands sit at \$102.83 and \$97.17 -- a total width of about \$5.66. Now suppose volatility doubles: the next five closes swing harder, say \$96, \$104, \$98, \$106, \$96 (still averaging \$100, but more spread out). Redo steps 2 to 4: deviations -\$4, +\$4, -\$2, +\$6, -\$4; squares 16, 16, 4, 36, 16; sum 88; variance 17.6; $\sigma \approx \$4.20$. The new bands are \$100 ± \$8.40, i.e. **\$108.40 and \$91.60** -- a width of \$16.80, three times wider. *The intuition: the bands' width is nothing but the standard deviation scaled up; double the volatility and the envelope roughly doubles, no human judgment required.*

## The squeeze: low volatility precedes expansion

If most of what Bollinger Bands "predict" is an illusion built on mean reversion and moving goalposts, is there anything genuinely useful in them? Yes -- and it comes from a real, well-documented property of volatility itself.

### Volatility clusters and mean-reverts

Two empirical facts about market volatility are about as close to laws as anything in trading. First, **volatility clusters**: calm periods follow calm periods and violent periods follow violent periods. A quiet day is more likely to be followed by another quiet day; a wild day by another wild day. Volatility comes in regimes, not random sprinkles. Second, **volatility is mean-reverting**: it does not trend forever in one direction. Unusually low volatility tends to rise back toward normal, and unusually high volatility tends to fall back toward normal. Volatility cannot stay pinned at zero (markets always wake up) and it cannot stay at crisis levels forever (panics exhaust themselves).

Put those two facts together and you get the one Bollinger signal worth taking seriously. When the bands pinch to an unusually narrow width -- their narrowest in many months -- volatility has compressed to an extreme. Because volatility mean-reverts, that compression is unlikely to last. The bands are telling you the market is coiled, and a coiled market tends to spring. This is the **squeeze**.

### What the squeeze does and does not tell you

The squeeze is genuinely informative about *one* thing: the *magnitude* of what is coming. Narrow bands say a large move is more likely than usual in the near future, because the abnormal calm is statistically due to break. Traders quantify this with a derived indicator called **BandWidth** -- simply the gap between the upper and lower bands divided by the middle band, which turns the visual pinch into a number you can track and compare to its own history. When BandWidth hits a multi-month low, you have a squeeze.

But here is the honest and crucial limit: the squeeze tells you nothing about *direction*. A coiled market can spring up or down with roughly equal probability. The narrow bands are a volatility forecast, not a price forecast. This is why trading the squeeze is hard: you know a move is coming, you do not know which way, so you either wait for price to break out of the squeeze and follow the breakout (accepting that some breakouts are false -- see [breakouts vs. fakeouts](/blog/trading/technical-analysis/breakouts-vs-fakeouts)), or you straddle it somehow. There is no "the squeeze means buy." There is only "the squeeze means brace."

![When the bands pinch to their narrowest the market is coiling, and the move out of the squeeze is large though its direction is not known in advance.](/imgs/blogs/bollinger-bands-and-volatility-4.png)

The figure shows the structure: the amber pinch is the squeeze, where the bands are at their narrowest; then price breaks out, volatility expands, and the bands flare. The trade drawn here is one disciplined way to play it -- wait for price to close beyond the high of the squeeze, enter on that break, place the stop back inside the squeeze (where the trade is wrong), and target a multiple of the risk.

#### Worked example: a squeeze-to-breakout trade with entry, stop, and target

Suppose a \$100 stock has gone quiet. Over the last few weeks its bands have pinched to their narrowest in six months: the upper band sits at \$101 and the lower band at \$99, a total width of just \$2 around the \$100 average. That is a squeeze. We do not know which way it breaks, so we let price tell us.

Price closes at \$101.20, decisively above the squeeze's high near \$101. We treat that as the breakout and enter long at **\$101**. The trade is "wrong" if price falls back into the squeeze, so we put the **stop at \$98** -- just below the squeeze's low. Our risk per share is \$101 - \$98 = **\$3** (this is our 1R, the one unit of risk; see [expectancy and R-multiples](/blog/trading/technical-analysis/expectancy-why-win-rate-lies)). For a target, we want at least twice our risk, so 2R = 2 × \$3 = \$6 above entry, a **target of \$107**.

Now the honest accounting. If the breakout is real and price runs to \$107, we make \$6 per share, or +2R. If it is a fakeout and price collapses back through the squeeze to our \$98 stop, we lose \$3 per share, or -1R. We do not need to be right most of the time for this to work: at a 2-to-1 reward-to-risk ratio, the breakeven win rate is $\frac{1}{1+2} = 33\%$, so even if only 4 in 10 squeeze breakouts follow through, the expectancy is positive. *The intuition: the squeeze does not tell you the direction, so you let the breakout pick it and size the trade so a false break costs exactly one small, planned R.*

#### Worked example: quantifying a squeeze with BandWidth

"The bands look narrow" is a feeling. To trade a squeeze you want a number, and that number is **BandWidth**. BandWidth is the gap between the upper and lower bands, expressed as a fraction of the middle band so that it is comparable across price levels and across instruments:

$$\text{BandWidth} = \frac{\text{upper band} - \text{lower band}}{\text{middle band}}$$

Let us compute it for both states of our earlier hand-built example. In the calm state, the bands were at \$102.83 and \$97.17 around a \$100 average. The gap is \$102.83 - \$97.17 = \$5.66. BandWidth is \$5.66 / \$100 = **0.0566**, or about **5.7%**. In the volatile state, the bands were at \$108.40 and \$91.60, a gap of \$16.80, so BandWidth is \$16.80 / \$100 = **0.168**, or about **16.8%** -- three times wider, exactly mirroring the tripled volatility.

Now the squeeze signal is concrete and mechanical. You track BandWidth over time and watch for it to fall to its *lowest reading in, say, the last 125 bars* (about six months of daily data). Suppose over that window BandWidth has ranged between 6% and 30%, and today it prints **5.5%** -- a fresh six-month low. That is a squeeze by a precise, repeatable definition, not a vibe. You now have a dated, falsifiable condition: "BandWidth made a 125-bar low," after which you wait for price to close beyond the squeeze's high or low and trade the resulting expansion. *The intuition: turning the visual pinch into BandWidth lets you define a squeeze with a number you can backtest, instead of arguing about whether the bands "look" tight.*

## Walking the band: the trend trap

The squeeze is the friendly face of Bollinger Bands. The walk is the trap that empties accounts, and it is the direct consequence of the misconception that a band touch is a reversal signal.

### Price rides the band in a strong trend

In a genuinely strong trend, price does not bounce politely between the bands. It *hugs* one of them. In a powerful uptrend, price closes near or on the **upper band** bar after bar after bar, for days or weeks. This is called **walking the band** (or "riding the band"). It happens because in a strong trend, price is making new highs faster than the moving average can follow, so price stays persistently far above the average -- and "persistently far above the average, on the high side" is exactly the definition of "near the upper band." The upper band is not a ceiling in a trend; it is the path.

The mirror image happens in a strong downtrend: price walks the lower band down, closing near it day after day as it grinds to new lows.

![In a powerful uptrend the price hugs the upper band bar after bar, so each upper-band touch is momentum rather than exhaustion and fading it is the classic losing trade.](/imgs/blogs/bollinger-bands-and-volatility-5.png)

The figure tells the whole story. Price walks up the upper band while the lower band falls away below, untouched and irrelevant. Each red box marks a trader who saw an upper-band touch, called it "overbought," shorted -- and got stopped out as the trend simply continued. Three shorts, three stop-outs, while a trend-follower who bought the strength rode the entire move.

### Why fading the band touch is the classic losing trade

Here is the cruel logic. The very thing that makes a band touch look like an extreme -- price being two standard deviations above the average -- is also the *signature of strong upward momentum*. A market that can push price two standard deviations above its 20-day average and *keep doing it* is not exhausted; it is powerful. Fading that touch means betting against the strongest force in the chart. You are selling strength into a market that is demonstrating, touch by touch, that it can absorb sellers and keep rising.

This is why "sell the upper-band touch" is the single most reliable way to lose money with Bollinger Bands. In a range it is a coin flip; in a trend it is a slow, repeated bleeding as you short every rung of a ladder going up. The band touch carries no edge on its own. Its meaning is entirely conditional on the regime: in a range, a touch is a far-from-average extreme that *might* mean-revert; in a trend, the same touch is *momentum confirming itself*. Without knowing the regime, the touch tells you nothing tradable.

The honest reading of an upper-band touch is therefore not "sell." It is "price is far above its recent average -- now, is this a quiet range that tends to revert, or a strong trend that tends to continue?" That second question is the whole game, and the band cannot answer it. You answer it from the trend and market structure (covered in [trend and market structure](/blog/trading/technical-analysis/trend-and-market-structure)) and, frankly, from accepting that one indicator will never be enough -- the theme of [the indicator trap](/blog/trading/technical-analysis/the-indicator-trap).

There is a constructive flip side to this, and it is the more useful one. Bollinger himself argued that band walks can be *confirmation of trend strength*, not exhaustion. A market that walks the upper band -- repeated %b readings near or above 1.0, with the middle band sloping up beneath it -- is demonstrating in real time that demand keeps overwhelming supply. Read that way, the band touch in a trend is a reason to *stay long* (or to look for pullback entries on the rare dips back toward the middle band), not a reason to short. The same evidence -- price two standard deviations above its average -- supports opposite trades depending on whether you have correctly identified the regime. That is precisely why the band touch is not a standalone signal: its sign is set by context you must supply, and getting the context wrong inverts the trade. The single most expensive mistake in technical analysis is taking a measurement that means "strong" and trading it as if it meant "extreme."

#### Worked example: counting band touches in a trend versus a range

Let us put numbers on the trap. We will compare two 100-bar stretches.

First, a **quiet range**. Price oscillates around \$100 in a sideways channel. Over 100 bars, suppose price touches the upper band 6 times and the lower band 7 times -- about 13 touches total, consistent with the ~1-in-9 frequency we computed earlier, split roughly evenly because the market has no direction. Now suppose we fade every touch (short the upper, buy the lower) with a tight stop and a target back at the middle band. Because the market is genuinely mean-reverting here, maybe 7 of those 13 fades work and 6 fail -- a win rate around **54%**. With a modest reward-to-risk, that is a small positive edge. Fine.

Second, a **strong uptrend**. Price grinds from \$100 to \$130 over 100 bars. Now price touches the upper band 22 times (it is walking it) and the lower band twice. We fade every upper-band touch with the same tight stop. But this market does not revert -- it walks. Maybe 6 of the 22 upper-band shorts work (brief pullbacks) and **16 get stopped out** as the trend continues. That is a win rate around **27%**. With losers at -1R and the occasional small winner, the expectancy is firmly negative: we lose money steadily, one stop-out at a time, while the stock we are shorting goes up \$30.

![Fading a band touch wins roughly half the time in a sideways range but loses badly in a strong trend where price walks the band, so the touch by itself is not a tradable signal without the trend context.](/imgs/blogs/bollinger-bands-and-volatility-7.png)

The matrix above generalizes the example: the win rate of "fade the touch" swings from a coin-flippy ~50-55% in a quiet range to a losing ~25-35% in a strong trend, with the volatility-spike regime being outright treacherous. *The intuition: the identical action -- fade the band touch -- is mildly profitable in one regime and steadily ruinous in another, which means the band touch alone is not a signal; the regime is.*

## ATR: the other volatility ruler

Bollinger Bands measure volatility with standard deviation. There is a second, complementary volatility ruler that every risk manager reaches for, and it solves problems the bands do not: **Average True Range (ATR)**. Where the bands turn volatility into an envelope on the chart, ATR turns volatility into a single number you can use to set a stop and a position size.

### What true range and ATR are

To define ATR we first need **true range**, the honest measure of how much a single bar moved. The naive measure of a bar's range is just its high minus its low. But that misses overnight gaps: if a stock closes at \$100 and opens the next morning at \$95 (a \$5 gap down) before trading in a 50-cent range, the high-minus-low says "the bar moved 50 cents," which badly understates the real move. True range fixes this by taking the largest of three distances: (1) today's high minus today's low, (2) today's high minus *yesterday's close*, and (3) yesterday's close minus today's low. The biggest of those three captures the full move including any gap.

**Average True Range** is then simply the average of the true range over the last $N$ bars -- conventionally 14. In one line:

$$\text{ATR}_{14} = \text{average of the true range over the last 14 bars}$$

where the true range of each bar is the largest of (high - low), (high - prior close), and (prior close - low). ATR comes out in dollars (or points): it answers "how much does this instrument typically move in one bar, right now?" A \$200 stock might have an ATR of \$4; a \$20 stock might have an ATR of \$0.30. Like the bands' width, ATR rises in volatile regimes and falls in calm ones -- but unlike a standard deviation around a mean, it is **non-bounded** and makes no assumption about a center. It is a pure "typical move size," which is exactly what you want for setting a stop.

![Average True Range measures the typical bar range in dollars, so a stop set a few ATR below entry adapts to volatility instead of guessing a fixed dollar amount.](/imgs/blogs/bollinger-bands-and-volatility-6.png)

The two-panel figure shows the idea. The top panel is price with a stop placed a couple of ATR below the entry. The bottom panel is ATR itself, drawn as its own line: when the market is calm, ATR is low (here \$2) and the stop sits close; when the market gets choppy, ATR rises (here \$5) and the stop automatically widens to give the trade room. The ruler on the right is the whole point -- you measure your stop in ATR, not in dollars.

### Why volatility-based stops beat fixed-dollar stops

A common beginner habit is to set a fixed stop: "I always risk \$1 per share" or "I always use a 2% stop." The problem is that a fixed stop ignores how much the instrument is actually moving. In a calm market, a \$1 stop might be ten times the typical bar range -- absurdly wide, you are risking far more than you need. In a volatile market, that same \$1 stop might be a quarter of a single bar's range -- so tight that ordinary noise stops you out instantly, before your idea has a chance to work. A fixed stop is right-sized only by luck.

An ATR-based stop fixes this by *scaling with volatility*. You set the stop a fixed *number of ATR* away from entry -- 2 ATR is common. In a calm market where ATR is \$0.50, your 2-ATR stop is \$1 away. In a volatile market where ATR is \$3, your 2-ATR stop is \$6 away. The stop is always sized to the market's current noise: wide enough to survive normal wiggles, tight enough not to risk more than the volatility warrants. Crucially, this means your stop distance is *consistent in risk terms* even as the market's character changes. That is the core reason trend-following and systematic strategies almost universally size stops in ATR rather than dollars.

#### Worked example: ATR-based position sizing on a \$10,000 account

This is where ATR turns into actual share counts, and it is the most useful calculation in this whole post. The logic is: decide how much money you are willing to lose on the trade, decide where the stop goes (in ATR), and let those two numbers determine the position size. The size is an *output*, never a guess.

Start with the account and the risk budget. You have a **\$10,000 account** and you have decided to risk **1% per trade**. One percent of \$10,000 is **\$100** -- that is the most you will lose if the stop is hit. (Risking a small, fixed fraction per trade is the heart of surviving long enough for an edge to compound; we cover why in [expectancy and risk of ruin](/blog/trading/technical-analysis/expectancy-why-win-rate-lies).)

Now the stop. The stock has an ATR of **\$1**, and you have decided to place your stop **2 ATR** below entry. So your stop distance is 2 × \$1 = **\$2 per share**. That \$2 is your risk per share -- your 1R.

The position size is now forced. You are willing to lose \$100 total, and you lose \$2 per share if stopped out, so you can hold:

$$\text{shares} = \frac{\text{dollar risk}}{\text{risk per share}} = \frac{\$100}{\$2} = 50 \text{ shares}$$

You buy **50 shares**. If the stop is hit, 50 shares × \$2 = \$100 lost -- exactly your 1% budget. Not a penny more, by construction.

Watch what happens when volatility changes. Suppose instead the stock is in a choppy regime with an ATR of **\$2.50**, so your 2-ATR stop is \$5 per share. The same \$100 risk budget now buys $\$100 / \$5 = 20$ shares. *The intuition: as volatility rises, ATR-based sizing automatically makes your position smaller, so your dollar risk stays pinned at \$100 no matter how wild the market is.* The fixed-dollar trader, by contrast, would have bought the same number of shares in both regimes and silently taken on far more risk in the choppy one. ATR sizing keeps your risk constant; that is its entire job.

## Common misconceptions

Bollinger Bands attract more confident wrong beliefs than almost any indicator. Here are the ones that cost the most.

### "A touch of the upper band is a sell signal"

This is the headline error, and the whole "walking the band" section is its refutation. A band touch means price is two standard deviations above its 20-day average -- which in a range *might* mean overextension, but in a trend means *strong momentum*. Selling every upper-band touch works fine in a sideways market and is a steady bleed in a trending one. The touch is not a signal; it is a measurement of distance-from-average, and its trading meaning flips entirely depending on the regime. The same applies in reverse to "a touch of the lower band is a buy."

### "The bands contain 95% of price action"

This comes from applying the normal-distribution 68-95-99.7 rule to markets. But market returns are fat-tailed, so the real figure for two-sigma bands is about 88 to 89%, not 95%. That difference matters enormously for how you read a touch: at 95% inside, a touch is a 1-in-20 event (genuinely rare); at 88 to 89% inside, a touch is roughly 1-in-9 (common). The bands do not contain price; price leaks out of them several times a week, exactly as a fat-tailed distribution predicts. Treating a band touch as a rare extreme is treating an ordinary event as a special one.

### "Wider bands mean a reversal is coming"

Wide bands mean volatility is *currently high* -- nothing more. High volatility happens in the middle of crashes, in the middle of melt-ups, and at major tops and bottoms. The width tells you the *amount* of recent movement, not its *direction* or whether it is about to reverse. Inferring "the bands are wide, so price will snap back" confuses high volatility with exhaustion. Sometimes a volatility expansion is the *start* of a big trend, not the end of one. Band width is a thermometer, not a forecast.

### "The bands cause the bounce"

When price tags a band and reverses, it is tempting to think the band acted as a barrier that repelled price. But the band is a line drawn from a formula; it exerts no force on anything. What you are usually seeing is mean reversion within a range (price was far from its average and drifted back, as far-from-average tends to do) combined with the band itself widening to meet the extreme close. The band is a coincident marker of an extreme, not the cause of the reversal. Believing the band "holds" price is like believing the speed-limit sign is what slows the car.

### "The default 20-period, 2-sigma settings are optimal"

The 20 and the 2 are reasonable, well-tested conventions -- they are not laws of nature, and they are not "tuned" to your market. Some traders use tighter bands ($k = 1.5$) for choppier instruments or wider ($k = 2.5$) for noisier ones, and shorter or longer averages for faster or slower signals. But chasing the "perfect" settings by optimizing over past data is a classic way to fool yourself: you can always find settings that would have worked beautifully on history and fail going forward (the overfitting trap from [the indicator trap](/blog/trading/technical-analysis/the-indicator-trap)). The defaults are fine. The settings are not where the edge is.

### "ATR tells you which way price will go"

ATR measures how *much* price moves, never which *direction*. A rising ATR in a falling market means the decline is accelerating; a rising ATR in a rising market means the advance is accelerating. ATR is direction-blind by construction -- it is built from the *size* of bars, not their sign. Use it for stops and sizing, never for picking direction.

## How it shows up in real markets

Abstractions become real when you watch them play out. Here are four named patterns -- described as recurring market behaviors rather than specific dated trades, since exact levels go stale. Treat these as the *shapes* to recognize, not as predictions; the as-of caveat is that any particular instrument's numbers change daily.

### A low-volatility squeeze before a major breakout

The classic squeeze setup appears again and again across instruments. A stock or index goes unusually quiet -- daily ranges shrink, BandWidth falls to a multi-month or multi-year low, and the Bollinger Bands pinch into a tight ribbon hugging the moving average. Then, often catalyzed by an earnings report, an economic release, or simply the resolution of an accumulation pattern, volatility explodes: a large bar punches out of the ribbon, the bands flare apart, and a sustained move follows. Traders who tracked the squeeze were braced for *a* move (they knew the compression was statistically unsustainable) even though they could not know the direction until the breakout bar printed. This is the squeeze doing exactly what it honestly can: forecasting magnitude, not direction. The risk it carries is the false breakout -- price punches out, sucks in breakout buyers, then reverses back into the ribbon, which is why the disciplined version waits for a *close* beyond the squeeze and keeps the stop tight inside it.

### A parabolic move walking the upper band for weeks

In every major bull run -- in individual high-flying stocks, in commodities during a supply shock, in speculative manias -- there is a phase where price simply walks up the upper Bollinger Band, closing on or near it for ten, twenty, thirty bars in a row. To a mean-reversion trader this looks like the most overbought market imaginable, and the temptation to short "the obvious top" is enormous. It is also where fortunes are lost on the short side: the band walk is the visual signature of momentum so strong that price stays two standard deviations above its own average for weeks. Every upper-band touch that a fader shorts is another rung the market climbs. The honest lesson is that in a confirmed strong trend, the upper band is a *trail* the trend follows, not a *ceiling* it bumps against -- and the same logic, inverted, governs the parabolic *declines* that walk the lower band down in a crash.

### A volatility spike blowing the bands wide in a crash

When a market crashes, volatility doesn't rise gently -- it detonates. Standard deviation over the trailing 20 bars spikes as a few enormous down-days enter the window, and the Bollinger Bands blow apart to widths that would have looked impossible during the preceding calm. Price often closes *outside* the lower band for several consecutive days -- a "three-standard-deviation event" that the normal distribution says should essentially never happen, occurring in a cluster. This is fat tails in their purest form: the violent regime that the bell curve underweights and that the market periodically delivers anyway. The lesson here is defensive: when the bands are wildly wide and price is closing outside them, that is *not* a high-probability mean-reversion buy. It is a market in a high-volatility regime where moves beget moves, and the historical standard deviation that the bands are built from is, for the moment, badly underestimating the risk. Volatility expansions can keep expanding.

### ATR-based stops in a trend-following system

Systematic trend-following strategies -- the kind run by managed-futures funds across dozens of markets -- almost never use fixed-dollar stops. They size everything in ATR. A typical rule places the initial stop a fixed multiple of ATR (often 2 to 3) from entry and sizes each position so that a stop-out costs the same small fraction of the portfolio regardless of which market the trade is in. The effect is that a quiet, low-ATR market (say a stable currency pair) gets a larger share count with a tight dollar stop, while a wild, high-ATR market (say a commodity in a supply panic) gets a small share count with a wide dollar stop -- and *both* trades risk the same 1% of the account if stopped. This is ATR doing the job standard deviation cannot do as cleanly: translating "how much does this market move" directly into "how many shares should I hold." The risk it does not remove is the gap: a market can leap past an ATR-based stop overnight on news, so the realized loss is occasionally larger than the planned 1R -- which is why even ATR-sized risk is a target, not a guarantee.

### Bands and ATR disagreeing at a turning point

A subtle and instructive pattern shows up at major market turns, when the two volatility rulers briefly tell different stories. Near the end of a long, grinding uptrend, price can keep drifting up while the *trailing* standard deviation stays low -- the moves are small and orderly, so the Bollinger Bands stay narrow and price hugs the upper band. Everything looks calm and controlled. Then the first sharp reversal day arrives: a single large down-bar. ATR, which is sensitive to the *true range* of that one bar (including any overnight gap), jumps immediately, while the 20-day standard deviation -- diluted across 19 still-quiet prior closes -- barely moves yet. So ATR is screaming "volatility just changed" a few bars before BandWidth confirms it. Traders who watch both rulers treat that divergence as an early warning: when the per-bar volatility ruler (ATR) spikes but the smoothed-envelope ruler (the bands) has not caught up, the character of the market may be shifting. It is not a precise signal -- ATR also spikes on isolated one-off shocks that go nowhere -- but it illustrates that "volatility" is not one number, and the two common ways of measuring it can lead by a few bars. The as-of caveat applies sharply here: which ruler moves first depends on the specific gap and bar sizes, which differ every time.

## When this matters to you / further reading

If you take one thing from this post, take the reframe: Bollinger Bands are a *volatility chart in disguise*, and once you read them that way most of the folklore evaporates. The bands are not a fence price bounces inside. They are a breathing envelope whose width is recent volatility and whose distance from price is "how far from average, in standard deviations." A touch is common, not rare. A pinch is a magnitude forecast, not a direction call. A walk is momentum, not a reversal waiting to happen. And the genuinely useful applications -- recognizing a squeeze, refusing to fade a trend, and using ATR to size stops and positions -- all flow from treating the bands as a measurement of volatility rather than an oracle.

This matters to you the moment you put any volatility indicator on a chart, because the cost of the misreadings is not abstract. Shorting band touches in a trend is a real way to lose real money, repeatedly and predictably. Sizing positions by gut instead of by ATR is how a single volatile trade wipes out a month of careful gains. The bands and ATR are honest tools when used for what they measure; they are account-killers when asked to predict what they cannot.

For the pieces that surround this one: [the moving-averages backtest](/blog/trading/technical-analysis/moving-averages-honest-backtest) covers the middle band's behavior on its own; [the indicator trap](/blog/trading/technical-analysis/the-indicator-trap) is the broader argument about why no single indicator is a system; [expectancy and why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) is the math that makes the squeeze trade and the ATR sizing add up to a real edge; and [expectation, variance, and moments](/blog/trading/math-for-quants/expectation-variance-moments-math-for-quants) is the statistical foundation under the standard deviation that the whole indicator is built from. None of this is advice to trade. It is the honest reading of what the bands measure -- so that whatever you do next, you do it with the right picture in your head.
