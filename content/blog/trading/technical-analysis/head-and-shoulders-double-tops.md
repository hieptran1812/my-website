---
title: "Head and Shoulders, Double Tops and Bottoms: Reversal Patterns, Honestly Measured"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A structural, honest account of the head-and-shoulders and double top/bottom: why they are a failure of market structure, how the neckline and measured target really work, what the hit rate actually is, and why a pattern is not a pattern until the neckline breaks."
tags: ["technical-analysis", "chart-patterns", "head-and-shoulders", "double-top", "double-bottom", "reversal-patterns", "neckline", "price-action", "measured-move", "market-structure"]
category: "trading"
subcategory: "Technical Analysis"
author: "Hiep Tran"
featured: true
readTime: 43
---

> [!important]
> **TL;DR** — The head-and-shoulders and the double top/bottom are the most famous reversal chart patterns, and unlike most candlestick lore they have a clear structural logic: they mark the moment the trend's higher-highs-and-higher-lows sequence breaks.
>
> - A *reversal pattern* is a shape on the chart that, once confirmed, says the prior trend has ended and a new one in the opposite direction has begun. The head-and-shoulders and the double top/bottom are the two best-known.
> - Their honest meaning is a **failure of market structure**: in an uptrend, each push is supposed to make a higher high. When a push makes a *lower* high (the right shoulder, or the second top) and then price breaks the level that had been holding it up, the up-structure is broken.
> - The *neckline* is a real support/resistance level — the price that connected the lows between the peaks. The pattern is only valid once price closes through that neckline **with follow-through**; the shape alone is a guess, not a signal.
> - The textbook *measured target* projects the pattern's height (head-to-neckline) from the break. It has a rough statistical basis — in Thomas Bulkowski's large samples the target is reached roughly **55–72%** of the time depending on the pattern — but it is a probability, not a destination.
> - The big honest catch is *pareidolia*: the human eye finds these shapes everywhere, most of which never confirm. The discipline is to wait for the break and to size the trade to the real hit rate, not the legend. This is educational material about how patterns are described and measured, not advice to buy or sell anything.

"It's forming a head and shoulders." You have heard someone say it over a chart, with the confidence of a weather forecaster, and then watched the price do something else entirely. Reversal patterns have a reputation problem: they are simultaneously the most famous thing in technical analysis and the most abused. Half the screenshots online label a shape that never confirmed; the other half show a textbook example *after the fact*, when hindsight has done all the hard work. So let us do the unglamorous thing and treat these two patterns — the head-and-shoulders and the double top/bottom — the way they deserve: define them mechanically, explain *why* they would carry information at all, measure them honestly against real data, and be clear about exactly when they mean nothing.

The diagram below is the mental model for the whole post. A head-and-shoulders top is not a face the market drew for you; it is the visible signature of an uptrend running out of buyers. Price pushes up (the left shoulder), pulls back, pushes higher (the head), pulls back to the same place, and then pushes up a *third* time — but this time it fails to reach the previous peak (the right shoulder). Three pushes, and the last one is weaker than the one before. The line connecting the two pullback lows is the *neckline*, and when price finally breaks below it, the staircase of higher highs and higher lows is mechanically broken: the trend is over.

![A head and shoulders top with a left shoulder, a taller head at 120 dollars, a lower right shoulder, a neckline at 100 dollars, and a projected target at 80 dollars below the break](/imgs/blogs/head-and-shoulders-double-tops-1.png)

Everything else in this article is detail hung on that one picture. We will recover the structural logic from first principles, walk the head-and-shoulders and its inverse, do the same for the double top and bottom, draw the neckline and compute the measured target with real dollar numbers, confront the honest hit-rate data, and then spend real time on *failure* — because the single most important fact about these patterns is that most of the shapes your eye finds are not patterns at all.

This piece is the fifth leg of a series. It assumes you already know how a chart is built and what a trend is. If you have not read them, [trend and market structure](/blog/trading/technical-analysis/trend-and-market-structure) is the prerequisite — these patterns are *defined* as a break of the structure described there — and [support and resistance, and why levels exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist) explains why the neckline is a real level and not a doodle. The companion piece [reversal candlestick patterns](/blog/trading/technical-analysis/reversal-candlestick-patterns) covers the single-bar and two-bar reversal signals; chart patterns like these are the same idea at a larger scale. And because every pattern is ultimately a bet, [expectancy, and why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) is where the honest math of "is this worth trading" lives.

## Foundations: a reversal is a broken structure

Before we can read a reversal pattern we need three ideas that the series has already built, restated in the exact form this post will use. If any of these is fuzzy, the patterns below will look like magic; if they are clear, the patterns are almost obvious.

### Trend as a structure, not a vibe

A *trend* is not a feeling and not a line you drew at a flattering angle. It is a mechanical sequence of *swing points*. A *swing high* is a local peak — a bar whose high tops the bars on either side. A *swing low* is a local trough — a bar whose low bottoms the bars on either side. An *uptrend* is the pattern "each swing high is higher than the last (a *higher high*, HH) and each swing low is higher than the last (a *higher low*, HL)." A *downtrend* is the mirror: lower highs (LH) and lower lows (LL). Anything else is a *range*. That is the whole definition, drawn straight from [trend and market structure](/blog/trading/technical-analysis/trend-and-market-structure).

Why does this matter for reversals? Because a reversal is defined *as the breaking of that sequence*. An uptrend continues as long as the HH/HL staircase holds. The instant a swing high comes in *lower* than the previous one — the first *lower high* inside an uptrend — the market has, for the first time, failed to do the one thing an uptrend requires. Practitioners call that first crack a *change of character* (often abbreviated ChoCh). It is a warning, not a verdict. The verdict comes when price then breaks below the most recent higher low, closing the door: that is a *break of structure* (BOS), and it is the mechanical moment a trend ends.

Hold those two events in your head — change of character, then break of structure — because **a head-and-shoulders top is exactly a change of character (the right shoulder is the first lower high) followed by a break of structure (the neckline break is the close below the prior low).** The pattern is not a separate thing layered on top of structure. It *is* a particular, recognizable way the structure breaks.

### The neckline is a real support/resistance level

The second foundation is that the *neckline* — the line that will define the pattern — is not arbitrary. A *support* level is a price where falling prices have repeatedly stopped and reversed up, because enough buyers wait there; a *resistance* level is a price where rising prices have repeatedly stopped and reversed down, because enough sellers wait there. The full account of why these levels exist — memory, round numbers, prior swing points, and the order book — is in [support and resistance, and why levels exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist). The one fact we need here: a level becomes meaningful precisely *because price has reacted to it more than once.*

In a head-and-shoulders top, the pullbacks after the left shoulder and after the head bottom out at roughly the *same* price. That price is therefore a support level the market has tested twice — a real floor where buyers have stepped in. The neckline is just the line drawn through those lows. When price finally falls through it, a tested floor has given way. And there is a well-documented behavior — the *polarity flip* — where a broken support becomes resistance: the price that used to attract buyers now attracts sellers on the way back up. That is why the *retest* of a broken neckline (price popping back up to it and being rejected) is such a clean entry, and we will use it in a worked example.

### A top forms when the up-structure fails

Put the two foundations together and the whole subject collapses into one sentence. **A topping pattern is the picture of an uptrend's HH/HL sequence failing at a level the market keeps defending.** The buyers who powered the higher highs run out; the last push can't make a new high; the floor that held the pullbacks finally cracks; and the structure flips from "higher highs and higher lows" to "lower highs and lower lows." The head-and-shoulders and the double top are just the two most common, most recognizable *shapes* that failure takes. A bottoming pattern is the same story upside down: a downtrend's LH/LL sequence failing at a level buyers keep defending, flipping to higher highs and higher lows.

Everything below is making that precise — naming the parts, drawing the lines, measuring the move, and then being ruthlessly honest about how often it actually works.

## The head and shoulders top (and inverse bottom)

The head-and-shoulders top is the most famous reversal pattern in all of technical analysis, and it earns its fame by being genuinely structural. Let us name every part and tell the order-flow story that makes the shape mean something.

### The three peaks and the neckline

Reading left to right, a head-and-shoulders top has four landmarks:

- **The left shoulder.** Price, in an existing uptrend, makes a swing high (the latest of many higher highs), then pulls back. Nothing is wrong yet — this is just an ordinary swing in an uptrend.
- **The head.** Price pushes up again and makes a *higher* high than the left shoulder. The uptrend is, so far, perfectly healthy: this is another HH. Then it pulls back, and crucially the pullback lands at roughly the same low as the left shoulder's pullback. Those two lows define the neckline.
- **The right shoulder.** Price pushes up a third time — but this push *fails to exceed the head*. It tops out near the level of the left shoulder, making a *lower high*. This is the change of character: the first time in the whole advance that a push did not make a new high. The buyers are visibly weaker.
- **The neckline break.** Price falls from the right shoulder and breaks below the neckline (the support line through the two pullback lows). This is the break of structure. The pattern is now confirmed, and only now.

In our running numbers, the head tops at \$120, the neckline sits at \$100, and the right shoulder tops around \$110 — a clearly lower high than the head. Look back at figure 1: the three peaks and the neckline are exactly those four landmarks.

### The order-flow story: why each push tells you something

A shape is only worth trading if it reflects something real about supply and demand. Here is the order-flow narrative, told plainly. In an uptrend, buyers are in control: every dip is bought, every push makes a new high. The *left shoulder* is one such push. The *head* is a stronger push — demand is still winning, and it drives price to a new high. But notice what the pullback after the head tells us: sellers were able to push price all the way back down to the neckline, the same floor as before. Demand absorbed them there, as it had before, and price bounced — but the *right shoulder* push that follows is the tell. It cannot reach the head. Demand has weakened: the buyers who powered the higher highs are spent, or have become sellers. When price returns to the neckline a third time and this time *breaks through it*, the floor of waiting buyers is gone. The people who bought at the neckline twice are now underwater and become sellers themselves; the people who shorted the right shoulder are validated and press their bets. Supply overwhelms demand, and the structure flips down.

That is the honest mechanism: **three pushes with the last one weakest, and a tested floor that finally gives way.** Every part of the shape corresponds to a real change in who is winning. This is what separates the head-and-shoulders from most candlestick folklore — there is a coherent reason it would carry information, not just a memorable name.

It is worth being precise about what the *neckline* represents in order-flow terms, because that is the crux of why the break matters. Each time price fell to the neckline and bounced, it did so because resting buy orders sat there — limit orders from traders who wanted to buy that dip, plus stop-buy orders and the natural demand of anyone who considers the asset "cheap" at that price. That stack of buy interest is what we *see* as support. The first pullback (after the left shoulder) consumed some of it; the second pullback (after the head) consumed more. By the third visit — the move down from the right shoulder — much of that resting demand has already been eaten in the prior two bounces, and whatever remains is thinner. So the third test is the one most likely to break, and when it does, the absence of buyers is not a surprise; it was being depleted in plain sight. This is the genuinely useful intuition the shape encodes: **a level that has been tested repeatedly is not getting stronger with each test — it is getting weaker, because each bounce spends some of the orders that made it a level.** A two-or-three-times-tested neckline is closer to breaking than a fresh one, not further.

There is also a *positioning* story that compounds the order-flow one. By the time the right shoulder forms, a lot of traders are *long* — they bought the uptrend, they bought the dips to the neckline, and they are sitting on profits they do not want to give back. Those longs have stop-loss orders, and a great many of them are placed just below the neckline, because that is the obvious "if this breaks, I'm wrong" level. So below the neckline sits not just an absence of buyers but a cluster of *sell* orders waiting to be triggered. When price breaks the neckline, those stops fire, adding fuel to the move down — which is part of why neckline breaks can be sharp and fast. It is also, as we will see, part of why they can be *faked*: a market-maker or a large seller who knows those stops are there has an incentive to push price just below the neckline to trigger them, then buy the resulting flush. Structure and order flow are two views of the same thing.

### The inverse head and shoulders (a bottom)

Flip the whole story vertically and you have the *inverse* head-and-shoulders, a bottoming pattern that signals the end of a *downtrend*. Reading left to right: in an existing downtrend, price makes a swing low (the left shoulder), bounces, falls to a *lower* low (the head — the lowest point), bounces to roughly the same high as before (defining the neckline as a resistance level), then falls a third time but makes a *higher* low (the right shoulder), failing to reach the head. The break of the neckline *upward* confirms the bottom. The order-flow story is the mirror: each sell-off is weaker than the last, the final low cannot reach the head, and when buyers finally push price up through the resistance line, the sellers who defended it are overwhelmed.

![An inverse head and shoulders bottom with a left shoulder, a lower head at 80 dollars, a higher right shoulder, a neckline at 100 dollars resistance, and a projected target at 120 dollars above the break](/imgs/blogs/head-and-shoulders-double-tops-2.png)

In figure 2 the head is the lowest low at \$80, the neckline is resistance at \$100, the right shoulder is a higher low, and the break above \$100 projects a target of \$120 — the head-to-neckline height (\$20) added to the neckline. One honest asymmetry worth flagging now: bottoms and tops are *not* perfect mirrors in practice. Tops often form faster and more violently (fear is sharper than greed), while bottoms tend to be slower, rounder, and to take more time to build. The geometry is symmetric; the emotional tempo is not.

## The double top and double bottom

If the head-and-shoulders is three pushes with the middle one highest, the double top is the simpler cousin: *two* pushes to the same level, the second one failing. It is the most common reversal shape precisely because it is so simple — and, for the same reason, the most over-recognized.

### Two failed attempts at the same level

A *double top* forms when price, in an uptrend, pushes up to a high, pulls back, then pushes up a *second* time to *roughly the same high* — and fails to break above it. Two peaks at the same ceiling, with a valley between them. The trough of that valley — the pullback low between the two peaks — is the *neckline* (sometimes called the *middle low* or the *confirmation level*). The shape on the chart traces a clear **M**: up, down, up, down.

A *double bottom* is the mirror: two pushes down to roughly the same floor, the second failing to break lower, with a peak between them. It traces a **W**: down, up, down, up. The peak between the two lows is the neckline (the *middle high*).

![A double top traced as an M shape with two peaks at 120 dollars and a middle low at 108, beside a double bottom traced as a W shape with two lows at 50 dollars and a middle high at 58](/imgs/blogs/head-and-shoulders-double-tops-3.png)

Figure 3 lays the two side by side so the symmetry is unmistakable. The double top (left, the M) is two failed pushes at a ceiling; the double bottom (right, the W) is two failed pushes at a floor. The order-flow story is the same as before in miniature: at the double top, the first peak is sellers rejecting price at a level; the second peak is buyers trying again and being rejected at the *same* level — demand could not push price any higher than last time, which is information. When the valley between gives way, the buyers who bought the dip are trapped.

### Why the confirmation is the break, not the second peak

Here is the single most common and most expensive mistake people make with these patterns, so we will state it as plainly as possible: **a double top is not confirmed when the second peak forms. It is confirmed only when price breaks below the middle low.** Two peaks at the same level is *not yet* a reversal — it is a perfectly ordinary feature of a strong uptrend, which makes higher highs by definition. Price can touch a ceiling twice and then break clean through it on the third try; that is a *continuation*, not a reversal, and it happens constantly.

The logic comes straight from the foundations. The uptrend's structure is intact as long as the higher-low sequence holds. The second peak failing to exceed the first is, at most, a change of character (a potential lower high). But the *break of structure* — the close below the middle low — is what actually breaks the higher-low sequence. Until that close, the most recent higher low is still holding, and the uptrend is still, mechanically, an uptrend. The same is true of the double bottom: it is confirmed only when price breaks *above* the middle high, not when the second low forms.

This is not pedantry; it is the entire edge. People who trade the second peak are trading a guess; people who wait for the middle-low break are trading a structural fact. The difference shows up directly in the hit rate, and it is why every honest source insists the pattern "does not exist" until the neckline breaks.

## The neckline, the break, and the measured target

We now have the shapes. The practical questions are: how do you draw the neckline, what counts as a real break, where is the target, and — the honest part — how often does any of it actually work?

### Drawing the neckline

The *neckline* is the support (for a top) or resistance (for a bottom) line connecting the reaction lows (or highs) of the pattern. For a head-and-shoulders top, you draw it through the two pullback lows — the low after the left shoulder and the low after the head. For a double top, the neckline is the single middle low. Two practical points that the textbooks gloss over:

- **The neckline is rarely perfectly horizontal.** The two lows are often at slightly different prices, so the neckline slopes. A *down-sloping* neckline on a top (the second low lower than the first) is generally read as more bearish; an *up-sloping* one, less so. For our worked examples we keep it horizontal at round numbers so the arithmetic is clean, but do not expect that on a real chart.
- **The exact neckline price is fuzzy.** Whether you draw it through the wicks (the extreme lows) or the bar closes changes the level by a little, and that little matters when you are deciding whether a "break" has happened. There is no single correct choice — this is one of several places where the pattern is less precise than it looks, and we will be honest about that in the validity section.
- **The two shoulders are rarely symmetric.** The textbook picture has a left and right shoulder at the same height and the same width. Real patterns are lopsided: one shoulder is taller, one takes longer to form. A modest asymmetry does not invalidate the pattern — what matters structurally is that the right shoulder is a *lower high than the head*, not that it mirrors the left shoulder. Demanding visual symmetry is a way to talk yourself out of valid patterns and into invalid ones.
- **Where you anchor the projection changes the target.** The measured move is projected from the *break point on the neckline*, but because the neckline can slope and the break can happen at different x-positions, two analysts can compute targets that differ by a few percent from the same chart. Again, this is a reason to treat the target as a zone, not a line — a useful discipline is to mark the target as a *band* (say, \$78–\$82 rather than exactly \$80) and plan to scale out across it.

### What counts as a real break

A *break* is not price touching the neckline. It is not price poking a wick through it for an hour. The conventional, conservative definition is a **decisive close beyond the neckline** — a full candle (on whatever timeframe you trade) closing below the neckline for a top, above it for a bottom — *with follow-through*, meaning price continues in that direction rather than immediately snapping back. The follow-through requirement exists because *false breaks* (price closing just beyond the level and then reversing) are extremely common, especially at obvious, widely-watched levels where stop orders cluster. We will return to false breaks as the central failure mode.

### The measured target: the height projection

The famous *measured move* (or *measured target* or *price target*) is a simple geometric rule: **measure the height of the pattern — from the head (or the peak) to the neckline — and project that same distance from the point where price breaks the neckline.**

![The measured move shown as the head-to-neckline height of 20 dollars projected downward from the break point at the 100 dollar neckline to a target of 80 dollars](/imgs/blogs/head-and-shoulders-double-tops-4.png)

Figure 4 isolates the projection so the arithmetic is unmistakable. In our head-and-shoulders top: the head is at \$120, the neckline at \$100, so the height is \$120 − \$100 = \$20. Project \$20 down from the neckline break at \$100, and the target is \$100 − \$20 = **\$80**. For the inverse head-and-shoulders bottom, the head is at \$80, the neckline at \$100, the height is \$20, and the target is \$100 + \$20 = \$120. For a double top with peaks at \$120 and a middle low at \$108, the height is \$12 and the target is \$108 − \$12 = \$96. The rule is always the same: *neckline minus height for a top, neckline plus height for a bottom.*

The measured move has a genuine logic — the energy that drove price from the neckline up to the head is, roughly, the energy that should drive it from the neckline down by the same amount once the structure flips. But "roughly" is doing enormous work in that sentence, which brings us to the honest numbers.

### Bulkowski's honest hit-rate and average-move numbers

Thomas Bulkowski spent years cataloging tens of thousands of chart-pattern occurrences and publishing the statistics in *The Encyclopedia of Chart Patterns*. His work is the closest thing the field has to an honest empirical baseline, and the numbers are far more sobering than the folklore.

![A reliability matrix showing for the head and shoulders top, the inverse head and shoulders, and the double top or bottom the share that reach the measured target, the average move on the winners, and an honest caveat for each](/imgs/blogs/head-and-shoulders-double-tops-7.png)

Figure 7 summarizes the honest picture. The exact figures vary by edition, by how he defines a "break", and by whether the broader market is in a bull or bear phase, so treat these as *orders of magnitude*, not precise constants (Bulkowski's published studies span roughly the 1990s through the 2010s — as-of caveats apply, and his samples are US equities):

- **The head-and-shoulders top** reaches its measured target somewhere around **55–63%** of the time after a confirmed break, with an average decline (on the moves that work) on the order of **20%**.
- **The inverse head-and-shoulders bottom** does somewhat better, hitting target around **60–74%** of the time, with average rises that are larger and more dispersed (bull moves run further than bear moves).
- **Double tops and bottoms** land in a similar band — very roughly **65–72%** reaching target in his samples — but with enormous dispersion and a strong dependence on the size of the prior trend.

Three honest takeaways from those numbers. First, these patterns are *better than a coin flip* — that is real, and it is more than you can say for most candlestick lore. Second, the target is hit only **a bit more than half to two-thirds of the time**, which means roughly a third or more of confirmed breaks *fail to reach target*. Third, the average move is an average of a wildly skewed distribution: a few patterns run far past target and many fall short, so "the average move is 20%" does not mean "expect 20%." The right way to use these numbers is not "the target will be hit" but "the target is hit often enough that, at a good reward-to-risk ratio, the trade has positive expectancy" — which is exactly what [expectancy, and why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) is about, and which we compute below.

### The retest of the broken neckline

One more mechanic before the worked examples. After price breaks the neckline, it very often *retests* it: price pops back up (for a top) to the broken neckline before resuming the move down. This is the polarity flip in action — the old support is now resistance. The retest is useful for two reasons. It gives a *second entry* for anyone who missed the break, and because the stop can go just above the retest high, it gives a *tighter stop* and therefore a better reward-to-risk ratio. The cost is that not every pattern retests; sometimes price just runs, and waiting for a retest means missing the trade entirely. There is no free lunch — the retest trades a better price for a lower fill rate.

![The life of a reversal pattern as four stages: form, then break the neckline with follow-through, then retest the broken neckline, then resolve toward the measured target or fail](/imgs/blogs/head-and-shoulders-double-tops-8.png)

Figure 8 puts the whole lifecycle in order: the pattern *forms* (still just a shape, a guess), then it *breaks* the neckline with follow-through (now confirmed and tradable), then it often *retests* the broken neckline (the lower-risk entry), then it *resolves* toward the measured target — about 55–60% of the time — or it fails. Notice that only the second stage onward is tradable. The form stage is where most people make their mistake: they trade the shape before it has confirmed anything.

## Validity and failure

This is the most important section in the post, because the honest truth about reversal patterns is mostly a truth about how and how often they *fail*. A pattern that you only recognize after it works is worthless; the discipline is entirely in the rules for what counts and what doesn't.

### A pattern isn't a pattern until the neckline breaks

We have said it twice and we will say it a third time because it is the whole game: **the shape is not the signal. The break is the signal.** A head-and-shoulders that has formed its right shoulder but has *not* broken the neckline is not a head-and-shoulders — it is a candidate, and most candidates fail. The same shape resolves two completely different ways, and only the neckline break tells you which.

![Valid break versus failed pattern shown side by side: on one side the neckline holds and price reclaims the right shoulder and rips higher, on the other a full candle closes below the neckline with follow-through toward the measured target](/imgs/blogs/head-and-shoulders-double-tops-6.png)

Figure 6 shows the fork. On the right, the valid case: the right shoulder is a clear lower high, a full daily candle closes below the \$100 neckline, there is follow-through with no immediate reclaim, and the pattern targets \$80 with a stop above \$110. On the left, the failure: the shape looks textbook, but price stalls at the neckline without a decisive close, then buyers *reclaim* above the right shoulder at \$110 — and the failed pattern often rips higher precisely *because* the shorts who sold the "obvious" head-and-shoulders are now trapped and forced to buy back (cover) their positions. A failed bearish pattern is a bullish event. This is not a footnote; it is one of the most reliable ways these patterns lose money for the people who trade them early.

### The failed head and shoulders

Worth naming specifically: the *failed* head-and-shoulders is when the shape forms, maybe even breaks the neckline briefly, and then price reverses and climbs above the right shoulder (for a top) — invalidating the pattern. There are two flavors. The *no-break* failure: the neckline never gives a decisive close, and price resumes the uptrend. The *failed-break* (or *bear trap*): price closes below the neckline, triggers a wave of short entries and long stop-losses, and then snaps back above the neckline within a bar or two, trapping everyone who acted on the break. The bear trap is especially vicious because it punishes the disciplined trader who waited for the close, not just the impatient one. The only defenses are follow-through confirmation (wait an extra bar or two) and a stop that invalidates the trade cleanly (above the right shoulder), so a failure costs you a known, small amount.

How do you know early that a pattern is failing rather than working? There is no certainty, but there are tells, and they are all about *follow-through*. A healthy break accelerates *away* from the neckline: the bars after the break are decisive, they close near their extremes in the break direction, and price does not immediately return to the level. A failing break is *hesitant*: price closes just barely beyond the neckline, the next bar stalls, and then price drifts back toward the level — the move "lacks conviction." When you see a break that pokes through and then immediately starts climbing back, the bear-trap risk is high, and a trader who has already entered should be watching their invalidation closely rather than mentally spending the measured-move target. The honest framing is that the *failure rate is built into the hit rate we already quoted*: when Bulkowski says the target is reached 57% of the time, the other 43% includes both the patterns that fizzle short of target and the outright failures that stop you out. You do not get to assume you are in the 57%; you size and stop as though you might be in the 43%.

There is also a subtler failure that the statistics hide: the pattern that *works but not enough to pay you*. Price breaks the neckline, moves in your favor, but stalls at, say, half the measured distance — \$90 instead of the \$80 target — and then ranges or reverses. You were "right" about direction and still lost or barely broke even, because you were waiting for a target the move never reached. This is why fixed all-or-nothing targets are dangerous and why scaling out (taking partial profit along the way, trailing a stop on the rest) is the practical answer to a probabilistic target. The pattern told you direction with decent odds; it did not promise you a destination.

### Volume's classic — and unreliable — role

The textbooks make a big deal of *volume* — the number of shares or contracts traded in a period. The classic claim is that in a valid head-and-shoulders top, volume should be highest on the left shoulder and the head, lower on the right shoulder (confirming that demand is fading), and should *expand* on the neckline break (confirming conviction). For the inverse pattern, a volume surge on the upside break is supposed to confirm the bottom.

The honest version: volume *sometimes* behaves this way and the story is plausible — a breakout on heavy volume is more convincing than one on light volume, because it means more participants acted. But the relationship is weak and unreliable, especially in 24-hour markets like crypto where "volume" is fragmented across venues, and in any market where a single large order can distort the print. Bulkowski's own data shows volume confirmation improves the odds only modestly. Treat volume as a *tiebreaker*, not a requirement. A common error is "the volume didn't expand on the break, so the pattern is invalid" — that is over-reading a noisy signal. The price structure (the actual close beyond the neckline with follow-through) is the load-bearing evidence; volume is a soft corroborant at best.

### Pareidolia: the eye over-finds these

Here is the deepest honesty problem, and it is psychological. *Pareidolia* is the human tendency to see meaningful patterns in random noise — faces in clouds, animals in inkblots. Price charts are extraordinarily noisy, and the head-and-shoulders and double top are *visually simple* shapes, which means the eye finds them constantly, including in pure randomness. If you stare at any chart long enough you will "see" a head-and-shoulders forming; you will see two peaks at a level and call it a double top. The overwhelming majority of these never confirm.

This is why the discipline of *waiting for the break* is not a nicety — it is the only thing standing between you and trading your own pattern-matching machinery. A useful mental exercise: generate a chart of pure random coin-flips and you will find dozens of "perfect" head-and-shoulders patterns in it, most of which "fail." The shape carries information *only* in the specific structural context (a real prior trend, a tested neckline, a decisive break) and *only* after confirmation. Outside that context it is a Rorschach test. Survivorship bias compounds the illusion: the examples that get shared and remembered are the ones that worked, so your sense of how reliable the pattern is comes from a curated highlight reel, not the full population of attempts.

## Worked examples

Patterns are abstractions until you put dollars on them. Here are four concrete walkthroughs, each ending in the one idea it is built to teach. These use round numbers so you can do the arithmetic in your head; real charts are messier.

#### Worked example: a head and shoulders top, entry to target

You are watching a stock that has been in an uptrend and has now traced a head-and-shoulders top. The head printed at \$120. The two pullback lows — after the left shoulder and after the head — both bottomed near \$100, so you draw the **neckline at \$100**. The right shoulder topped at \$110, a clear lower high.

First, the **measured target**. The pattern height is the head minus the neckline: \$120 − \$100 = **\$20**. Project that down from the neckline: the target is \$100 − \$20 = **\$80**.

Now the **trade**. You do not act on the right shoulder; you wait. A full daily candle closes at \$99, below the \$100 neckline, and the next day price continues lower — a confirmed break with follow-through. You enter a short at the neckline break, call it **\$100**. Your stop goes just above the right shoulder, at **\$110** (above the right shoulder is the cleanest invalidation: if price climbs back above it, the pattern has failed and you want out).

- **Risk per share** = entry − stop = \$100 − \$110 = −\$10. You are risking \$10 per share.
- **Reward per share** = entry − target = \$100 − \$80 = \$20. You are aiming for \$20 per share.
- **Reward-to-risk ratio (R:R)** = \$20 / \$10 = **2:1**.

![Trading a head and shoulders top showing the entry at the neckline break at 100 dollars, the retest entry, the stop above the right shoulder at 110 dollars in the risk band, and the target at 80 dollars in the reward band giving a two to one reward to risk](/imgs/blogs/head-and-shoulders-double-tops-5.png)

Figure 5 shows the trade: the entry at the break, the stop band above \$110 (the \$10 of risk, in amber), and the reward band down to \$80 (the \$20 of reward, in green), for a 2:1 reward-to-risk. **The intuition: the measured target and the structural stop together hand you a defined reward-to-risk before you ever take the trade — you know exactly what you are risking and what you are aiming for.**

#### Worked example: a double bottom, entry to target

Now a bottom. A stock has been falling and traces a double bottom: it made a low at **\$50**, bounced to a middle high of **\$58**, fell again to **\$50** (holding the same floor — the second low did not break below the first), and is now rising. The neckline is the middle high at **\$58**.

The **measured target**: the pattern height is the neckline minus the floor: \$58 − \$50 = **\$8**. Project that *up* from the neckline (it is a bottom): \$58 + \$8 = **\$66**.

The **trade**: you wait for confirmation. A daily candle closes at \$59, above the \$58 neckline, with follow-through. You enter a long at the break, **\$58**. Your stop goes below the second low, at **\$50** — if price falls back below the floor, the double bottom has failed.

- **Risk per share** = entry − stop = \$58 − \$50 = \$8.
- **Reward per share** = target − entry = \$66 − \$58 = \$8.
- **R:R** = \$8 / \$8 = **1:1**.

Notice that this double bottom offers only a **1:1** reward-to-risk if you put the stop all the way at the second low — because the pattern is short (an \$8 height) relative to the distance to the stop. This is a real and common problem: not every valid pattern is a good *trade*. A 1:1 reward-to-risk needs a win rate well above 50% just to break even after costs.

Can you rescue the trade? Sometimes, by tightening the stop. Instead of placing it below the second low at \$50, you might place it below the most recent *minor* swing low formed on the breakout — say \$54, if price made a small higher low at \$54 just before clearing the neckline. Then risk = \$58 − \$54 = \$4, reward = \$66 − \$58 = \$8, and the R:R doubles to **2:1**. The trade-off is real and worth stating plainly: a tighter stop improves the reward-to-risk but *raises the chance of being stopped out by ordinary noise* before the move plays out, because \$54 is a closer, more easily-touched level than \$50. There is no free lunch — you are buying a better payoff ratio with a lower win rate, and only the expectancy math (next example) tells you whether the trade improved on net. **The intuition: a confirmed pattern and a good trade are not the same thing — the geometry has to give you a reward-to-risk worth taking, and where you put the stop is a genuine trade-off between payoff and hit rate, not a free optimization.**

#### Worked example: the retest entry for a tighter stop

Return to the head-and-shoulders top from the first example: neckline \$100, target \$80, right shoulder \$110. In the first version you entered at the break (\$100) with a stop at \$110, risking \$10 for a 2:1.

Now suppose you wait for the **retest**. After breaking below \$100, price pops back *up* to the broken neckline — the polarity flip, old support now resistance — and stalls at **\$101**, then rolls over. You enter your short on that rejection at **\$101**, and because the retest gives you a nearby invalidation, you place your stop just above the retest high, at **\$104** (instead of all the way up at the \$110 shoulder).

- **Risk per share** = \$101 − \$104 = −\$3. You are now risking only \$3.
- **Reward per share** = \$101 − \$80 = \$21.
- **R:R** = \$21 / \$3 = **7:1**.

The retest improved the reward-to-risk from 2:1 to 7:1 — same target, same idea, but a far tighter stop because the entry is closer to the invalidation level. The catch, again: **not every pattern retests.** If price breaks \$100 and never looks back, the retest trader gets no fill and misses the whole move. **The intuition: the retest trades a much better reward-to-risk for a lower probability of getting filled at all — a different bet, not a strictly better one.**

#### Worked example: measured-move expectancy

Now the honest math that ties it together. *Expectancy* is the average profit or loss you can expect per trade, computed as `expectancy = (win rate × average win) − (loss rate × average loss)`, the central tool from [expectancy, and why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies). Let us apply it to the head-and-shoulders trade.

Take the first trade's numbers: you risk \$10 to make \$20 (2:1). Suppose, consistent with Bulkowski-style data, the target is reached about **57%** of the time (call it the win rate), and when it fails you lose your full \$10 risk (you are stopped out). So:

- **Win rate** = 57%, **average win** = \$20 (you hit the \$80 target).
- **Loss rate** = 43%, **average loss** = \$10 (you are stopped at \$110).
- **Expectancy per trade** = (0.57 × \$20) − (0.43 × \$10) = \$11.40 − \$4.30 = **+\$7.10 per share**.

A positive expectancy of \$7.10 per share means that *across many such trades*, even though you lose 43% of the time, the 2:1 payoff on the winners makes the strategy profitable. Now watch how fragile that is. Suppose you trade the *unconfirmed* pattern (the right shoulder, before the break) and that drops your win rate to 40% because half your "patterns" never confirm:

- **Expectancy** = (0.40 × \$20) − (0.60 × \$10) = \$8.00 − \$6.00 = **+\$2.00 per share** — barely positive, and trading costs and slippage could erase it entirely.

And the 1:1 double bottom from the second example, at the same 57% win rate:

- **Expectancy** = (0.57 × \$8) − (0.43 × \$8) = \$4.56 − \$3.44 = **+\$1.12 per share** — positive but thin, and a few points lower in win rate flips it negative.

**The intuition: the edge in these patterns is not the shape — it is the combination of a decent hit rate with a good reward-to-risk, and it survives only if you wait for confirmation and only take the trades whose geometry pays you enough. Win rate alone tells you nothing; expectancy is the whole story.**

## Common misconceptions

Five beliefs that beginners hold about reversal patterns, each wrong, each corrected with the *why*.

**"The pattern is confirmed when the right shoulder (or second peak) forms."** No. The shape is a candidate, not a signal. Confirmation is the *neckline break with follow-through*, full stop. The right shoulder is, at most, a change of character — a warning that the structure *might* be failing. Trading the unconfirmed shape is trading your own pareidolia, and the worked example above shows it can cut your expectancy by more than half. The market makes two peaks at a level constantly without reversing; only the break of the middle low (or the neckline) makes it a reversal.

**"The measured target is a precise destination."** No. It is reached only roughly 55–72% of the time depending on the pattern, and the move is drawn from a wildly skewed distribution, so even the "average move" is misleading. Treat the target as a *probabilistic reference* for setting a reward-to-risk and a place to consider taking profit, not a price the market is obligated to reach. A third or more of confirmed patterns fall short. Plan for partial exits and trailing stops, not a single all-or-nothing target.

**"Every three-peak shape is a head and shoulders."** No. A valid head-and-shoulders requires a *prior uptrend* to reverse (you cannot top a market that wasn't rising), a head that is a genuine higher high, a right shoulder that is a clear *lower* high, and a neckline drawn through real reaction lows. A random three-bump squiggle in a sideways range is not a head-and-shoulders — it is noise that happens to have three bumps. Context (the prior trend and a tested neckline) is part of the definition, not optional decoration.

**"Volume must confirm or the pattern is invalid."** No. The classic volume story (fading on the right shoulder, expanding on the break) is a *soft corroborant*, and the empirical relationship is weak. Plenty of valid patterns break on unremarkable volume, and volume data is especially noisy in fragmented or 24-hour markets. The load-bearing evidence is the price structure — the decisive close beyond the neckline with follow-through. Rejecting a clean structural break because volume "didn't confirm" is over-reading a noisy variable.

**"Bigger patterns and bigger timeframes are always more reliable."** Partly true, but often over-stated. A head-and-shoulders on the daily or weekly chart, built over months, does generally carry more weight than one on the 5-minute chart, built over an hour — bigger patterns involve more participants and more real positioning. But "bigger is better" has limits: a sprawling, ambiguous pattern whose neckline you can draw three different ways is *less* tradable than a clean small one, and the higher timeframe's target is proportionally farther away, which can wreck the reward-to-risk if your stop is fixed. Timeframe matters, but clarity and reward-to-risk matter more.

## How it shows up in real markets

Reversal patterns are not just textbook diagrams; they have printed in some of the most-watched charts in history, and — just as instructively — they have failed loudly. These are illustrative episodes, described from public price history; exact levels depend on the data source and the timeframe, and all dates and figures are as-of the events described, not current prices.

**A major index top that printed a textbook head-and-shoulders (the Nikkei 225, around 1990).** Japan's Nikkei 225 index peaked near 38,900 at the end of 1989 after a historic bubble. Over the following months the chart traced what many technicians read as a large head-and-shoulders top, and the break of its neckline preceded a multi-year collapse that took the index down by more than half and, eventually, by far more over the following decade. The pattern did not *cause* the crash — the bubble's fundamentals did — but it is a clean illustration of the structural logic: a historic uptrend, a failure to make new highs, and a break of a tested floor that confirmed the trend had ended. The lesson is the one from the foundations: the pattern is the visible signature of a structure breaking, not an independent force.

**A famous double bottom that launched a bull run (the S&P 500, March 2009).** After the 2008 financial crisis, the S&P 500 bottomed near 666 in early March 2009, bounced, and — on some readings of the late-2008/early-2009 price action — formed a double-bottom-like structure before launching one of the longest bull markets in history. The "W" shape, with the break above the middle high confirming, is a textbook bottoming signature: two tests of a floor, the second holding, then a decisive break upward. As always, the confirmation was the *break of the middle high*, not the second low itself — anyone who "bought the double bottom" at the second low was guessing; the structural buyers waited for price to clear the intervening peak.

**A head-and-shoulders that failed and ripped higher (the bear trap).** This pattern recurs across markets and decades: a widely-watched index or stock forms an "obvious" head-and-shoulders top, financial media calls it, traders pile in short on the neckline break — and then price snaps back above the neckline and the right shoulder, squeezing the shorts and accelerating *upward*. The S&P 500 has produced several of these "failed tops" during strong bull markets (mid-cycle scares that looked like reversals and weren't). The mechanism is exactly the trapped-shorts dynamic from the validity section: a failed bearish pattern becomes a bullish event because the people positioned for the down-move are forced to buy back. The defensive lesson: the stop above the right shoulder is not optional, because the failed pattern is one of the most reliable ways to lose money on a chart pattern.

**How widely-watched necklines become self-fulfilling, then get faded.** When a head-and-shoulders is so clean that everyone sees the same neckline — say, a round-number level on a major index that financial TV has been pointing at for a week — two things happen in sequence. First, the level becomes partly *self-fulfilling*: so many stop-loss orders and breakout-short orders cluster just below it that a break does trigger a cascade, and price drops sharply, "confirming" the pattern. But then, precisely because the move was driven by mechanical order flow rather than fresh selling, it often *reverses* — the sellers are exhausted, the level gets reclaimed, and the obvious pattern gets faded by traders who specialize in fading obvious patterns. This is the modern reality of any widely-watched chart pattern: the more obvious it is, the more it is both self-fulfilling in the short run and a fade candidate in the slightly longer run. It is why follow-through and a clean invalidation matter more than the beauty of the shape.

**A double top that capped a stock and the role of the middle low (large-cap equities).** Individual stocks print double tops regularly at the end of strong runs: price drives to a high on earnings or momentum, pulls back, rallies again on a second wave of buying, fails to make a new high, and then breaks the intervening low as the late buyers give up. The instructive part is almost always the *middle low*. Traders who shorted the second peak (the visually obvious "it failed at the same level" entry) were repeatedly squeezed when the stock made a marginal new high before topping; traders who waited for the close below the middle low got a far cleaner signal and a far cleaner invalidation. The recurring lesson across countless individual names is the one from the double-top section: the second peak is the bait, the middle-low break is the signal, and the gap between them is where most of the losses happen. The exact tickers and levels are left general here because the pattern is generic and the specific levels are as-of whenever each episode occurred, not current.

**Crypto's pattern-heavy, low-reliability environment.** Cryptocurrency charts are saturated with head-and-shoulders and double-top callouts, partly because the market is 24/7, retail-heavy, and chart-pattern-literate, and partly because high volatility produces lots of swings for the eye to connect. Bitcoin's major tops (late 2017 near \$20,000, late 2021 near \$69,000) were each retroactively fit with topping patterns, and the down-moves did roughly approximate measured targets in some readings. But crypto is also where pareidolia and false breaks run wild: the same volatility that produces clean-looking patterns also produces frequent neckline whipsaws, and volume data is fragmented and unreliable across exchanges. Crypto is the clearest case for the whole post's thesis: the shapes are everywhere, most are noise, and the only ones worth anything are the ones that confirm with a decisive break and follow-through.

## When this matters to you / further reading

If you take one thing from this post, take this: **a reversal pattern is a hypothesis about structure, and it is worthless until the neckline confirms it.** The head-and-shoulders and the double top/bottom are the rare chart patterns with a real structural logic — they are the picture of a trend's higher-highs-and-higher-lows sequence failing at a level the market keeps defending. That makes them worth understanding even if you never trade them, because they are how price *tells you* a trend has ended. But the same structural honesty that makes them meaningful also demands discipline: wait for the break, demand follow-through, set a stop that invalidates the idea cleanly, and size the trade to the real hit rate (55–72%, not 100%) and the real reward-to-risk, not the legend.

Where this touches you: even if you are an investor, not a trader, you will hear these patterns invoked constantly — in financial media, in market commentary, in the confident pronouncements of people who "see a head-and-shoulders forming." Now you can evaluate the claim. Has the neckline actually broken? Was there a real prior trend? Is the right shoulder a genuine lower high? Is the target a probability or a promise? The ability to ask those four questions is most of what separates someone who understands chart patterns from someone who is being sold one.

For the next steps, the rest of this series builds the scaffolding these patterns stand on. Re-read [trend and market structure](/blog/trading/technical-analysis/trend-and-market-structure) if the change-of-character and break-of-structure language was new — the patterns here are *literally defined* in those terms. Read [support and resistance, and why levels exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist) for why the neckline is a real level and why the retest works. See [reversal candlestick patterns](/blog/trading/technical-analysis/reversal-candlestick-patterns) for the single-bar and two-bar version of the same reversal idea, which often appears *at* the neckline and *at* the shoulders, giving you a finer-grained entry. And above all, work through [expectancy, and why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies), because the entire value of these patterns lives in the expectancy math, not the picture. A beautiful head-and-shoulders with a 1:1 reward-to-risk and a 55% hit rate is a worse trade than an ugly one with a 3:1 and the same hit rate — and only the expectancy framework tells you which is which. This has been educational material about how reversal patterns are described and measured; none of it is a recommendation to buy or sell any security.
