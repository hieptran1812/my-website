---
title: "The Term Structure of Volatility: Contango, Backwardation, and the VIX Curve"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How implied volatility differs by expiry, why the curve slopes up in calm and inverts in panic, and how to trade the term structure with calendars, roll-down, and forward vol."
tags: ["options", "volatility", "term-structure", "vix", "contango", "backwardation", "calendar-spread", "forward-vol", "roll-cost", "implied-volatility"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Implied volatility is not one number; it is a curve across expiries, and the *shape* of that curve is itself a tradeable view on whether the market expects calm or panic.
>
> - **Contango (upward slope) is the normal state:** near-term vol sits low, longer-dated vol higher, because vol mean-reverts toward a long-run average near 19.5 and the front is below it. **Backwardation (inverted) is the panic state:** the front month spikes above the back as near-term fear blows out.
> - **The shape is set by two forces:** mean reversion (short-dated vol can roam far above or below the mean; long-dated vol is anchored near it) and event humps (a known catalyst lifts the implied vol of every expiry that *spans* its date).
> - **An upward curve costs the long-vol holder money to carry.** A constant-maturity long-vol product like VXX must keep rolling down a contango curve, bleeding each month — about 6.5% on the front roll in our curve. That roll is the engine behind the long-run decay of every long-vol ETP.
> - **The one number to remember:** in our curated curves the calm spread is +3.1 vol points from the 1-month to the 6-month, while a stressed curve is inverted by 14 vol points (38 down to 24). The same instrument, two regimes, opposite slopes — and the slope is the trade.

On the morning of August 5, 2024, a short-volatility book that had been quietly printing money for two years detonated. The yen carry trade — borrow cheap yen, buy almost anything that yields more — unwound in a global margin call, and equity volatility went vertical. The cash VIX, which had closed the previous Friday in the low teens, gapped to a closing print of 38.57 and traded far higher intraday. But the number that actually destroyed positions was not the *level* of the VIX. It was the *shape* of the curve.

For the prior two years, the VIX futures curve had been in placid contango: each month further out priced a little higher than the month before, the way it does in calm markets. Traders who were short the front month and long a later month — or short volatility outright and "rolling down" the curve to collect carry — had been harvesting that upward slope month after month. It felt like free money. Then in a single session the curve violently *inverted*: the front month rocketed above the back, backwardation slammed in, and every position that depended on the curve sloping the friendly way took the loss on the wrong side, all at once. The front month, the one closest to the panic, moved the most. The back months barely flinched. The people who got hurt were not wrong about the *level* of vol over the next year. They were wrong about the *shape* of the curve over the next two weeks.

That is what this post is about. We have spent earlier posts treating implied volatility as a single input — the number you back out of an option's price. But there is no such thing as "the" implied vol. There is a one-month implied vol, a three-month implied vol, a one-year implied vol, and they are usually all different. Plot implied vol against expiry and you get a curve: the **term structure of volatility**. Reading that curve — knowing why it slopes the way it does, what it costs you to hold a position across it, and how to trade its shape — is one of the highest-leverage skills in the whole options game. It is also where some of the most reliable carry and some of the most spectacular blowups both live.

![VIX term structure showing an upward contango curve in calm markets and an inverted backwardation curve in stressed markets on one axis](/imgs/blogs/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve-1.png)

We built the foundation for this in [vega, your exposure to implied volatility](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol), where we saw that vega *grows with maturity* — a longer-dated option carries more exposure to a change in implied vol than a near-dated one. That post mapped vega *within* a single expiry. This post maps implied vol *across* expiries. The two ideas fit together: because longer-dated options have more vega, the term structure tells you how the *price of vol* changes as you walk out the maturity axis, and the slope of that walk is something you can be long or short.

## Foundations: what the term structure of volatility actually is

Start with the everyday version. A weather forecast is more confident about tomorrow than about the same date two months from now. You can say with some precision whether it will rain tomorrow; for a day eight weeks out, all you can honestly say is "it'll be roughly seasonal." Now flip it around to the *variability* of the forecast. The day-to-day forecast can be anything — a freak storm, a perfect clear sky — so its possible range is wide and jumpy. The two-month-out forecast settles toward the seasonal average, because over a long horizon the unusual days wash out and you converge on the climate. Short horizons are volatile; long horizons revert to normal.

Implied volatility behaves exactly the same way, and the term structure is the chart of that behavior. **The term structure of volatility is the set of implied volatilities for options of the same underlying but different expiries, plotted against time to expiration.** A one-week option might imply 14% vol; a one-month option 16%; a six-month option 18%. Connect those points and you have a curve.

A crucial clarification on units before we go further, because it confuses everyone the first time. Every one of those numbers — 14%, 16%, 18% — is an *annualized* volatility. They are all expressed on the same per-year scale even though the options expire at wildly different times. The annualization is what makes them comparable: it strips out the "more time means more total movement" effect (a six-month option will obviously have a bigger expected *dollar* move than a one-week option) and leaves only the *rate* of expected movement. So when the one-week implies 14% and the six-month implies 18%, that genuinely means the market expects the *pace* of volatility to be slower in the near term than over the next half year. The term structure is a chart of the expected *rate* of movement at different horizons, all on one annualized axis.

### The VIX is one point on this curve

The number people quote as "the VIX" is itself a point on a term structure. The CBOE Volatility Index is constructed to measure the 30-day implied volatility of the S&P 500, blended from a strip of SPX options. But the CBOE publishes a whole family of these: a 9-day index (the VIX9D, sometimes informally "the one-week VIX"), the headline 30-day VIX, a 3-month index (VIX3M), and a 6-month index (VIX6M). Lay those four out and you have read the term structure of S&P implied vol straight off the screen.

In calm markets the ordering is almost always VIX9D < VIX < VIX3M < VIX6M — short-dated implied vol below long-dated. That upward slope has a name borrowed from commodities futures.

It is worth dwelling on *why* the index is built from a strip of options rather than a single contract, because it explains what the term structure is measuring. A single SPX option's implied vol mixes together two things you would rather separate: the market's view on the *level* of volatility and its view on the *strike* (the skew — out-of-the-money puts trade richer than calls, which we treat in the surface posts). The VIX construction integrates across all the strikes at a given expiry to produce one number that represents the expected volatility *at that maturity*, washing out the strike dimension. So each VIX-family index is a clean read on one point of the term structure: 9-day, 30-day, 3-month, 6-month expected vol, strike-averaged. When you lay the four of them side by side you are reading the term structure with the skew already collapsed out — which is exactly the slice this post cares about. The skew is the *other* axis, and the two together form the vol surface; here we hold strike fixed (or strike-average it away) and walk only the time axis.

One more foundational point, because it trips up newcomers who come from the stock world. A stock has one price. A bond has a yield for each maturity — the yield curve — and nobody is surprised that the 2-year and the 10-year yield differ. Volatility is like the bond, not like the stock: there is a different implied vol for every horizon, and the *curve* of those vols is the object, not any single point. The mistake is to ask "what is the implied vol of this stock?" as if it were a scalar. The honest answer is "which expiry?" — because the one-week and the one-year can easily be 20 points apart, and the gap between them is itself the most tradeable thing on the screen.

### Contango and backwardation, defined

- **Contango** is an *upward-sloping* term structure: longer-dated implied vol is higher than shorter-dated. The market is calm at the front and prices a gentle rise in uncertainty as you go further out. This is the normal state; the curve is in contango the large majority of the time.
- **Backwardation** is an *inverted* term structure: shorter-dated implied vol is *higher* than longer-dated. The front month is the most afraid. This happens in stress — a crash, a panic, a known imminent shock — when near-term fear blows out above the longer-run expectation.

The words come from the futures world (where contango means far-dated futures trade above spot and backwardation the reverse), and they map cleanly onto the VIX *futures* curve, which we will get to. For the implied-vol curve, just hold the pictures: contango slopes *up* to the right, backwardation slopes *down*. The cover chart above shows both on one axis using our curated curves — a calm contango curve rising gently from 14.5 at one month to 17.6 at six months, and a stressed backwardation curve plunging from 38 at the front to 24 at the back.

#### Worked example: reading contango vs backwardation off the curve

Here are two real-shaped VIX term structures, in vol points, by months to expiry:

```
    months out:   1      2      3      4      5      6
    CALM curve:   14.5   15.5   16.3   16.9   17.3   17.6
    STRESS curve: 38.0   32.0   28.5   26.5   25.0   24.0
```

Read the calm curve left to right: it rises monotonically. The 1-month is 14.5; the 6-month is 17.6. The slope, end to end, is +3.1 vol points and it is *upward* — that is textbook contango. Every month further out costs more implied vol than the month before it. The front (14.5) sits well below the long-run average near 19.5, and the back is climbing toward it.

Now the stress curve: it *falls* monotonically. The 1-month is 38.0 — far above the long-run average — and the 6-month is 24.0. The slope is −14.0 vol points, *downward* — textbook backwardation. The front month is screaming; the back is elevated but far calmer. Said as a single number: the calm curve is in **+3.1-point contango**, the stress curve in **14-point backwardation**.

The intuition: in calm markets the front is below the long-run mean and the curve slopes up toward it; in panic the front spikes far above the mean and the curve slopes down from it. The slope is just where the front sits relative to where vol "wants" to be.

## Why the curve has the shape it does

A term structure is not arbitrary. Two forces almost entirely determine its shape, and understanding them is what lets you predict how the curve will move when conditions change. Here they are.

![A cause diagram showing mean reversion and event humps producing contango in calm markets and backwardation in panic](/imgs/blogs/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve-3.png)

### Force one: mean reversion of volatility

Volatility is one of the most strongly mean-reverting quantities in all of finance. It does not wander off to infinity or drift to zero; it gets yanked back toward a long-run average — for the S&P that average is around 19.5 vol points over the last few decades. When vol spikes to 40 it does not stay there; within weeks or months it grinds back toward 20. When it sinks to 11 in a sleepy summer it does not stay there either; eventually something happens and it pops back up.

Now ask what mean reversion does to the *term structure*. The short-dated implied vol is a forecast of vol over a short window, so it can be anything — if there is a crash *right now*, the one-week vol can be 60, and if it is a dead-calm August it can be 11. The short end is free to roam. But the long-dated implied vol is a forecast of *average* vol over a long window, and mean reversion guarantees that whatever is happening now will fade and vol will spend most of that long window near its average. So the long end is *anchored* near the long-run mean, almost regardless of what the front is doing.

That single fact produces the slope:

- When the front is **low** (calm), it sits *below* the anchored back end, so the curve slopes **up**: contango.
- When the front is **high** (panic), it sits *above* the anchored back end, so the curve slopes **down**: backwardation.

The chart below makes the mechanism visible: the plausible *range* of implied vol is wide at the front and narrows toward the back, all centered on the long-run mean. The front month can print 38 in a panic or 11 in calm; the six-month barely strays from the mean either way.

![Mean reversion chart showing a wide plausible range of short-dated implied vol narrowing toward an anchored long-run mean for long-dated vol](/imgs/blogs/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve-2.png)

This is the same square-root-of-time logic that governs vega: a longer horizon has more time for the random ups and downs to average out, so the *uncertainty about the average vol* shrinks with maturity. Conceptually, the term structure is the market's mean-reverting forecast laid out across horizons, and the dashed bands in that chart show how far each horizon can credibly stray from the mean.

There is a useful number that quantifies how fast the reversion happens: the **half-life** of a vol shock. Empirically, an S&P vol spike decays with a half-life on the order of a few weeks — roughly speaking, a shock that takes the VIX from 20 to 40 will, absent fresh news, give back about half of that excess (the 20 points above its starting level) within a few weeks, and most of it within a couple of months. That half-life is precisely why the *back* of the curve refuses to move much in a panic: a six-month forecast that knows a shock half-lives away in three weeks will price almost no extra vol into months four, five, and six, because by then the shock is long gone. The half-life is the hidden parameter behind the whole shape of the curve. A *short* half-life (fast reversion) gives a steep curve that whips violently at the front and stays flat at the back; a *long* half-life (slow reversion) would give a flatter, more parallel curve. Equity vol has a short half-life, which is exactly why equity term structures are so front-loaded and why the front-month whip is so dramatic.

This also explains an asymmetry worth internalizing: the curve inverts *fast* and *normalizes slow*. Inversion happens in the single session a shock arrives, because the front-end whip is near-instantaneous. Normalization — the return to contango — happens over the half-life as the shock decays, which is days to weeks. So backwardation is a sharp spike followed by a gradual slide back to contango, not a symmetric move. The trader's edge is that the *normalization* is more predictable than the *inversion*: you rarely see the inversion coming, but once it has happened, mean reversion gives you a reasonable map for how the curve will re-flatten.

#### Worked example: why mean reversion forces the slope

Suppose vol mean-reverts and the market believes the long-run average is 19.5. Today there is no crisis; the realized vol over the past few weeks has been a sleepy 13. The market's one-month implied vol forecast can lean on that recent calm — call it 14.5. But the six-month implied vol has to forecast the *average* vol over the next 180 days, and the market knows that over six months *something* usually happens; vol will mean-revert up toward 19.5 well before the window closes. So the six-month sensibly prices around 17.6 — higher than the one-month, pulling toward the long-run mean. Result: 14.5 at one month, 17.6 at six months, an upward (contango) curve, purely from mean reversion with a calm front.

Now flip it: a crash hits and realized vol jumps to 50. The one-month implied vol can be 38 — it must forecast the immediate chaos. But the six-month still has to average over 180 days, and the market expects the panic to fade and vol to revert toward 19.5 long before then, so the six-month prices only 24. Result: 38 at one month, 24 at six, a downward (backwardation) curve. Same mean reversion, panicked front, inverted slope. **The slope of the curve is just a readout of where the front sits relative to the long-run mean.**

### Force two: event humps

Mean reversion sets the smooth background slope. The second force adds *local bumps* on top of it. Markets know the dates of certain events in advance — an earnings report, a Federal Reserve rate decision, a CPI inflation print, a court ruling, a drug-trial readout. Each of those is a scheduled burst of uncertainty. And here is the key mechanical fact:

> **Any option expiry that *spans* a known event date must price in the extra movement that event can cause; any expiry that *settles before* the event does not.**

So if a company reports earnings on, say, the 20th of the month, then the option expiry on the 15th (before earnings) prices only the ordinary day-to-day movement, while the expiry on the 22nd (after earnings) has to bake in the earnings jump. The post-earnings expiry's implied vol gets *lifted* — an **event hump** sitting on the term structure. The hump appears exactly at the first expiry that captures the event and persists in every expiry beyond it.

![An event hump chart showing implied vol rising on the expiry that spans an earnings date, above the smooth base curve](/imgs/blogs/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve-4.png)

This is why single-stock term structures around earnings look so distinctive: there is a smooth base curve, and then a step up at the first expiry that catches the report. The same thing happens at the index level around FOMC meetings and CPI dates — the VIX term structure develops a little kink at the expiry that spans the meeting. Traders read those kinks to back out how much *event-specific* vol the market is pricing, which connects directly to [the expected move of an event](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options).

#### Worked example: pricing the event hump across two expiries

Take a stock at \$100 that reports earnings in week 3. Use an at-the-money \$100 straddle (long the call and the put) as our vol thermometer, and the Black-Scholes model from this series.

- **Pre-event expiry (2 weeks out, settles before earnings).** It prices only ordinary movement; on the base curve its implied vol is about 34.2%. The 2-week ATM straddle is worth about **\$5.35**.
- **Post-event expiry (4 weeks out, spans earnings).** It must carry the earnings jump on top of the base curve. The base 4-week vol would be 36.0% — but the event lifts it about 9 points to roughly 45.0%. The 4-week ATM straddle is therefore worth about **\$9.93**.

Now isolate what the event is worth. Price that same 4-week straddle on the base curve *without* the hump (36.0% vol): it would be about **\$7.94**. The difference, \$9.93 − \$7.94 ≈ **\$1.98**, is the dollar value the market has assigned to the earnings event itself — the event premium baked into the straddle that spans the report. **The hump is not noise; it is the market's price tag on a scheduled burst of uncertainty, and you can read it straight off the gap between the expiry that spans the event and the one that does not.**

## The VIX futures curve versus the cash VIX

So far we have talked about the implied-vol curve you read off options. There is a closely related and even more directly tradeable object: the **VIX futures curve**. This is where the words *contango* and *backwardation* originally come from, and it is the engine behind a whole class of products and trades.

The cash VIX — the headline index — is not directly tradeable. You cannot buy "the VIX" the way you buy a share of stock, because it is a calculated index, a snapshot of 30-day SPX implied vol at this instant. What *is* tradeable is **VIX futures**: contracts that settle to the value of the VIX on a specific future date. There is a front-month VIX future, a second-month, a third-month, and so on, each settling to where the (then-spot) VIX prints on its settlement date.

Plot the prices of those futures by settlement month and you get the VIX futures curve, and it has the same two regimes:

- **Contango:** each further-out future trades above the nearer one (and above the cash VIX). This is the normal state — the futures price in mean reversion *up* from a calm spot toward the long-run average, exactly like the implied-vol term structure.
- **Backwardation:** the front future trades above the back (and above is the cash VIX). This is the panic state — spot VIX has spiked above where the market thinks it will settle, so the futures price in mean reversion *down*.

The relationship between the cash VIX and its futures is the whole game. When the curve is in contango and nothing changes, each VIX future must *converge down* to the (lower) cash VIX as it approaches settlement, because at settlement the future equals the index. That downward convergence is a structural headwind for anyone holding a long VIX-futures position — it is the source of the roll cost we turn to next.

A subtlety that confuses even experienced traders: the VIX futures curve and the SPX implied-vol term structure are *related but not identical*. The implied-vol term structure is a curve of *spot* implied vols — the vol implied by options expiring at each date, as of today. The VIX futures curve is a curve of *forward* expectations — each future is the market's bet on where the 30-day VIX will *be* at that future settlement date. The futures curve is, in a precise sense, the term structure of *forward* 30-day vol. The two move together — both are in contango when calm, both invert in panic — but they are answering slightly different questions, and the gap between them is itself information. When VIX futures trade at a large premium to the cash VIX, the market is saying "vol is low now but we expect it higher soon"; that premium is the carry a short-vol position collects and the cost a long-vol position pays.

The settlement convergence is what makes this mechanical rather than a matter of opinion. On its settlement morning, a VIX future is *defined* to equal a special opening quotation of the VIX index — there is no daylight between them. So whatever premium the future carried over the cash index has to bleed out, day by day, over the contract's life. In contango that bleed is a loss for the long and a gain for the short; the curve does not need to be *right* about anything for the convergence to happen, it happens by the rules of settlement. That is why the roll-down is often called the *most reliable* feature of vol products — and also why it is so dangerous, because reliability breeds complacency, and the one regime where it reverses (backwardation) is precisely the regime that arrives without warning.

### Constant-maturity exposure and why it has to roll

Most people do not trade individual VIX futures; they trade **exchange-traded products** built on them — VXX, UVXY, and their kin. These products target a *constant maturity* (VXX targets roughly a constant 30-day VIX-futures exposure). But individual futures do not stay at a constant maturity; they age toward settlement every single day. So to hold a steady 30-day exposure, the product must continuously *sell* the front-month future as it ages and *buy* the next-month future. That daily rebalancing is "the roll," and on a contango curve it is a guaranteed slow bleed, because you are repeatedly selling the cheaper near future and buying the more expensive far one.

We set this up here and give it the full treatment — the VXX bleed, UVXY's leverage, the cost of the roll, and why these products are designed to decay — in [the VIX and vol products](/blog/trading/options-volatility/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll). For now, the key idea is just that an upward curve is *expensive to hold long* and *profitable to hold short*, and the cost of holding is set by the steepness of the curve.

## The roll-down: the theta of the curve

The roll cost deserves its own treatment because it is the single most important consequence of curve shape for a position holder. Think of it as the **theta of the curve** — a steady decay you pay (or collect) just for holding a vol position across a sloped term structure, even if nothing moves.

Here is the mechanism in slow motion. Suppose the curve is in contango and you are long a VIX future that settles in two months, priced at, say, 15.5. One month passes and *nothing changes about the world* — the curve is still in the same shape. But your future is now a one-month future, and on an unchanged contango curve a one-month future is priced lower, at 14.5. You have lost 1.0 vol point not because vol fell, not because you were wrong about anything, but purely because your contract slid one month down a downward-converging curve. That is roll-down. It is the carry cost of being long an upward-sloping vol curve.

For a short position it is the mirror image: you *collect* that roll-down. Selling the richer near-future and watching it converge down to the cheaper spot is the harvest that short-vol carry trades live on — and it is exactly the harvest that blows up when the curve inverts.

![A bar chart of the monthly roll loss rolling down a contango VIX curve, steepest at the front month](/imgs/blogs/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve-5.png)

#### Worked example: the roll-down on a contango curve

Use the calm curve again, in vol points: 14.5, 15.5, 16.3, 16.9, 17.3, 17.6 for months 1 through 6. A long-vol holder rolling down it loses, each month, the step between adjacent points expressed as a fraction of what they are holding:

- Rolling the 2-month down to the 1-month: from 15.5 to 14.5, a loss of 1.0 vol point. As a fraction of the 15.5 you held, that is 1.0 / 15.5 ≈ **−6.5%** in a month. That is the front-month roll, and it is the steepest.
- The 3-month rolling to the 2-month: 16.3 to 15.5, a loss of 0.8, about **−4.9%**.
- Further out the steps flatten: −3.6%, −2.3%, −1.7% as you walk back through the curve.

Notice the shape: the roll-down is *steepest at the front*, where the curve bends hardest, and flattens further out. This is why constant-maturity products that hug the front (UVXY targets the very front of the curve and applies leverage) bleed fastest, while products that sit further out the curve bleed more slowly. Stack the whole strip of monthly roll steps and the curve costs roughly **19% over the six-month strip** to hold long in this contango — a brutal headwind. **The roll-down is theta you pay to the curve's shape, not to the calendar, and on a steep contango it dwarfs almost everything else about a long-vol position.**

## How the whole curve moves: shifts, twists, and the front-end whip

Reading a static curve is half the skill; the other half is anticipating how the *whole* curve moves when conditions change. Term structures do not slide up and down rigidly. They move in characteristic ways, and a trader who knows the modes can position for the move rather than just the level.

The dominant mode is the **front-end whip**. Because the back of the curve is anchored near the long-run mean by mean reversion and the front is free to roam, almost all of the *motion* of the curve happens at the short end. When a shock hits, the one-month vol can jump 20 points in a session while the six-month moves 3. When calm returns, the front collapses back while the back barely budges. This is exactly what the August 2024 chart shows. The practical consequence: **the front of the curve is where the action is, and where the leverage and the danger both live.** A position concentrated in the front month has enormous sensitivity to the curve's motion; a position in the back is comparatively sleepy. UVXY targets the very front *and* applies leverage, which is why it is the most violent instrument on the board — it sits exactly where the whip cracks hardest.

The second mode is the **parallel shift plus steepening or flattening**. In a mild risk-off episode the whole curve lifts a few points and the front lifts a bit more, *flattening* the contango without fully inverting it. In a recovery the whole curve drifts down and the front drops most, *re-steepening* the contango. These are the bread-and-butter moves between the extremes, and they are what a calendar or a curve spread is really trading: not the level of vol, but whether the curve steepens or flattens from here.

Then there is the regime *flip* — contango to backwardation and back. This is not a smooth move; it is a snap, and it is the move that matters most for risk. A curve that has been gently flattening for weeks can invert in a single session when a shock arrives, because the front-end whip overshoots the back. The flip is discontinuous in its consequences: every position whose edge was "the curve slopes up" loses on the wrong side at the moment of inversion, all at once, which is why the blowups cluster on flip days rather than on merely high-vol days.

#### Worked example: the front-end whip versus the anchored back

Take the calm curve (front 14.5, six-month 17.6) and apply a shock that drags it to the stress shape (front 38.0, six-month 24.0). Look at how much each tenor moved:

- **Front month:** 14.5 to 38.0, a move of **+23.5 vol points**.
- **Three-month:** 16.3 to 28.5, a move of **+12.2 vol points**.
- **Six-month:** 17.6 to 24.0, a move of **+6.4 vol points**.

The front moved nearly *four times* as much as the back. A long-vol position in the front month gained 23.5 points of vol on the move; the same dollar of vega placed in the six-month gained only 6.4. That ratio — roughly 3.7-to-1 front-to-back in this shock — is why front-month vol is both the best long-vol hedge for a crash (it whips up the most) and the most expensive thing to hold in calm (it rolls down the most). **The front end is high-beta to the curve; the back end is low-beta and anchored, and knowing which end you are exposed to tells you how violently your position will move when the regime turns.**

## Forward volatility: the vol of a future window

Here is a subtle but powerful idea the term structure lets you extract. The one-month implied vol is the market's forecast of vol over the next month. The two-month implied vol is the forecast over the next two months. Buried inside those two numbers is a *third*: the market's forecast of vol over the window that runs *from* one month *to* two months — the **forward volatility** of that future month. It is the vol-world analogue of a forward interest rate.

The mechanics rest on the fact that *variance is additive over time* (variance is vol squared times time). The two-month variance must equal the one-month variance plus the forward variance of the second month:

```
    sigma_2m^2 * T_2m  =  sigma_1m^2 * T_1m  +  sigma_fwd^2 * (T_2m - T_1m)
```

Solve for the forward vol:

```
    sigma_fwd  =  sqrt( ( sigma_2m^2 * T_2m  -  sigma_1m^2 * T_1m ) / (T_2m - T_1m) )
```

This matters because the *spot* implied vols can hide a violent forward view. A modest-looking 6-month vol stacked on a low 1-month vol can imply that the market expects a *much* higher vol in the months 1-to-6 window. The forward vol is where you see what the curve is really saying about the future.

#### Worked example: extracting forward vol from two expiries

Take the calm curve. The 1-month implied vol is 14.5% (T = 1/12 year) and the 3-month is 16.3% (T = 3/12). Plug into the formula, with vols as decimals:

```
    sigma_fwd  =  sqrt( ( 0.163^2 * (3/12)  -  0.145^2 * (1/12) ) / (3/12 - 1/12) )
               =  sqrt( ( 0.006642  -  0.001752 ) / (2/12) )
               =  sqrt( 0.004890 / 0.16667 )
               =  sqrt( 0.029340 )  =  0.1713
```

So the **forward vol over the 1-to-3-month window is about 17.1%** — meaningfully higher than the 14.5% the front month implies, and even a touch above the 16.3% three-month, because the calm front is dragging the headline numbers down. The curve is quietly telling you the market expects vol to run hotter once you clear the calm front month.

Now do the same on the stress curve (1-month 38%, 3-month 28.5%): the forward 1-to-3-month vol works out to about **22.3%** — *far below* the 38% front month, because the inverted curve is pricing the panic to fade fast. **Forward vol is the cleanest single number for "what does the curve think happens *after* the front month?", and it can differ wildly from the spot implied vols you read off the screen.**

You can push this all the way to a single event. Recall the event-hump example: a 2-week expiry implied 34.2% and a 4-week expiry (spanning earnings) implied 45.0%. The forward vol over that 2-to-4-week window — the window that *contains* the earnings date — works out to roughly **53.6%**, versus about 37.7% if there were no event. That ~54% forward vol is, in effect, the market's implied vol *for the earnings event itself*, isolated by differencing two expiries. This is precisely how desks price [event volatility and the vol-crush](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush).

## Trading the term structure

Everything above is description. Here is the part that pays: the term structure is not just a thing to read, it is a thing to *trade*. There are three broad ways to do it.

### One: calendar and diagonal spreads

The purest term-structure trade is the **calendar spread**: sell the near-term option and buy the longer-term option at the *same strike*. You are explicitly trading the shape of the curve — short the front-month vol, long the back-month vol — with the directional bets largely canceling near the strike. A **diagonal** is the same idea with *different* strikes, adding a directional tilt.

![A structure diagram of a calendar spread selling the rich front month and owning the cheaper back month at the same strike](/imgs/blogs/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve-6.png)

Why does this work? Two reasons, both straight out of the term structure. First, **theta**: the near-term option you sold decays faster per day than the longer-term option you own (decay accelerates into expiry, as we covered in [theta and the clock](/blog/trading/options-volatility/theta-trading-the-clock-and-the-price-of-being-long-options)), so the *spread* collects net time decay if the stock sits still. Second, **term structure**: if you sold a front month puffed up by an event hump and own a quieter back month, you profit when that hump deflates after the event and the curve *normalizes*. The full mechanics — how to leg in, manage, and adjust — live in [calendars and diagonals](/blog/trading/options-volatility/calendars-and-diagonals-trading-time-and-term-structure); here we just price the term-structure edge.

#### Worked example: a calendar that profits from term-structure normalization

A stock sits at \$100 with an event-rich front month. Set up an at-the-money \$100-strike calendar:

- **Sell** the 2-week \$100 call at an event-rich 45% implied vol. Price ≈ **\$3.59**. You collect this.
- **Buy** the 6-week \$100 call at a quieter 33% implied vol. Price ≈ **\$4.69**. You pay this.
- **Net debit paid = \$4.69 − \$3.59 = \$1.10.** That is your max loss.

Now the event passes, two weeks go by, and the stock pins right at \$100. Two things happen, both helping you. The front-month call you sold expires at the money and worthless — you keep the full \$3.59 you collected on it. And the back-month call you own now has 4 weeks left, the event hump is gone, and its implied vol has normalized down to about 30%. Its price is now about **\$3.46**.

Your spread is worth \$3.46 − \$0 = **\$3.46** at exit, against the \$1.10 you paid. P&L ≈ \$3.46 − \$1.10 = **+\$2.37 per share — a 216% return on the debit.** The trade made money on two fronts at once: the rich front-month vol decayed away (you were short it), and the curve normalized (the hump you sold deflated). **A calendar is a bet that the front month is too expensive relative to the back; you win when the curve flattens or the stock pins the strike, and you lose on a big move away or a vol spike that re-inverts the curve.**

### Two: selling rich front-month event vol

A simpler, related trade: when an event hump puffs up the front month, *sell it* — outright, or hedged. The market consistently overpays for known-event vol, and after the event resolves, the front-month implied vol collapses in the "vol crush." If you are short that front-month vol (a short straddle, a short strangle, an iron condor over the event), you collect the deflation. This is a high-win-rate, fat-tail-risk trade: you win small and often when the event is a non-event, and you lose big on the rare gap. It is the single-name cousin of the variance-risk-premium harvest.

The term-structure framing sharpens this trade in two ways the level-only view misses. First, the event hump tells you *exactly* how much vol you are being paid to sell — it is the gap between the expiry that spans the event and the smooth base curve, the ~9-point hump from our example. If that hump is fat relative to the stock's history of actual earnings moves, the front-month vol is rich and worth selling; if the hump is thin, the market is *underpricing* the event and you might want to be a buyer instead. The hump is the price, and your edge is having a better estimate of the realized move than the price implies. Second, the curve tells you *where* the vol-crush risk concentrates: it is entirely in the front, because the back-month vol barely carries the event premium. So a trade that sells the front and is flat or long the back — a calendar — keeps the vol-crush gain while neutralizing most of the directional and longer-dated vol risk. Selling the front-month event vol naked captures the full crush but eats the full gap risk; structuring it against a back month is how desks keep the edge while capping the tail.

A practical caution: the vol crush is so reliable that it is itself crowded. Everyone knows the front-month implied vol collapses after earnings, so the *price* of that front-month vol already reflects the expected crush. Your edge is not "vol falls after earnings" — that is priced — but "this particular hump is fatter (or thinner) than this particular stock's realized-move history justifies." Selling every earnings hump indiscriminately collects the average variance risk premium and the occasional catastrophic gap; selling only the *richest* humps, sized for the gap, is where the actual edge lives.

### Three: harvesting (and surviving) the roll

The third trade is structural: in persistent contango, *short the front-month vol product* (or be short VIX futures, hedged) and collect the roll-down — the theta of the curve. This is the trade that printed money for years and then detonated in the carry unwind. The roll-down is real carry; the catastrophic risk is that the curve inverts and the front gaps up against you. Sizing and a hard stop on backwardation are what separate a carry harvester from a future cautionary tale. Volatility-as-an-asset treats this explicitly in [owning fear as an asset](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear).

The deceptive thing about the roll harvest is its P&L *shape*. Day after day it produces small, steady gains — the front-month roll of about 6.5% a month drips in as positive carry, and the equity curve looks like a smooth diagonal line up and to the right. That smoothness is exactly what makes it dangerous: it feels like a low-risk yield, it backtests beautifully through any period without a flip, and it lulls position-sizing discipline. Then the curve inverts and the entire position takes a multiple of all the accumulated carry in a single session. The honest way to think about the roll harvest is as *selling insurance on the curve's shape*: you collect a small premium continuously and pay a large claim rarely. The premium is real and the strategy has positive expectancy across most environments — but the claim, when it comes, can exceed years of premium, and it arrives without a phone call. Size it as insurance written, not as a yield earned, and keep the position small enough that one flip is a survivable loss rather than a terminal one.

## Common misconceptions

**"A higher VIX means the curve is in backwardation."** No — *level* and *shape* are different things. The VIX can be elevated at, say, 25 while the curve is still in contango (front 25, back 27) if the market expects vol to keep rising. Conversely the curve can be in backwardation at a *modest* level. What inverts the curve is the *front* spiking *above the back*, which is about the relationship between near and far, not the absolute level. In our stress curve the front is 38 and the back 24 — backwardation by 14 points — but a "slow-burn" stress could invert with the front at only 22 over a 20 back. Shape is the relationship; level is one point.

**"VXX tracks the VIX, so if the VIX is flat, VXX is flat."** Wrong, and it costs people real money. VXX holds VIX *futures*, not the cash VIX, and in contango those futures roll down toward the lower cash index every day. The cash VIX can sit perfectly still at 15 for a month while VXX bleeds several percent from pure roll-down. Recall our roll-down example: the front-month roll alone was about **−6.5% in a month** on a calm contango curve. That decay is structural, not a tracking error — it is the curve's theta, and VXX pays it whether or not the VIX moves.

**"Backwardation means I should buy volatility — the trend is up."** This is the most expensive misconception of all, and it is exactly backwards. Backwardation is the panic state, and it usually arrives *after* the spike, not before it. Buying the front month when the curve is already deeply inverted means buying the most expensive, fastest-decaying vol on the board, right when mean reversion is about to drag it back down. The roll *into* backwardation is positive carry for a long (the front converges *up* toward... no — the front is highest, so a long front future rolls *down* as the panic fades). When backwardation normalizes, the front collapses, and a late long-vol buyer gets run over by the same mean reversion that set the shape. Backwardation is a signal that fear is *already* priced, not that you should pay up for more.

**"Forward vol is just the average of the two implied vols."** No — variance is additive, not vol, so you cannot average the vols. From the example, a 14.5% one-month and a 16.3% three-month do *not* give a forward vol of (14.5 + 16.3)/2 = 15.4%. The correct variance-weighted calculation gives **17.1%** — higher than either input might naively suggest, because you subtract the small near-term variance and re-scale over the short forward window. Averaging the vols understates the forward view, sometimes badly, especially around event humps where the difference can be tens of points.

**"The term structure tells me where vol is going."** It tells you where the market is *pricing* vol, which is not a forecast you can trust blindly. The curve is the consensus, and the consensus is often wrong — contango persists because sellers demand a premium for bearing crash risk (the variance risk premium), not because vol is genuinely about to rise every month. You can be short the roll and right for years; you can also be short the roll and wiped out in one session. The shape is information about *what is priced*, and your edge comes from having a *different, better* view about whether that price is too rich or too cheap — never from assuming the curve is a crystal ball.

## How it shows up in real markets

### February 2018: Volmageddon

For years leading into early 2018, the VIX curve sat in steep, placid contango, and an enormous short-vol complex had grown up to harvest the roll-down. The most infamous vehicle was an inverse-VIX ETP (XIV) that was effectively *short* the front-month VIX futures and collected the roll every day. It worked beautifully — until February 5, 2018, when the VIX closed at 37.32 (from the low teens days earlier) and, more importantly, the curve *snapped into backwardation*. The front-month future, which the short-vol products were short, gapped up violently against them. XIV lost about 96% of its value in a day and was terminated. The lesson was not that vol got high; it was that the *curve inverted*, and every position whose edge was "the curve slopes up" lost on the wrong side at once. The roll-down that had been carry for years became a one-day catastrophe when the shape flipped.

### August 2024: the yen carry unwind

The opening story. On August 5, 2024, the cash VIX closed at **38.57** (per CBOE) and traded far higher intraday, in a global deleveraging triggered by an unwind of the yen carry trade. As before, the curve inverted hard — the front month rocketed while the back stayed comparatively anchored near the long-run mean. The chart below contrasts a calm contango curve with the inverted stress curve and marks that 38.57 print at the front.

![VIX curve on a calm date in contango versus an inverted curve on a spike date, marked with the August 2024 cash VIX close of 38.57](/imgs/blogs/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve-7.png)

What is striking, and what the chart shows, is how *localized* the move was: the front end whipped up by more than 20 vol points while the six-month barely moved. That is mean reversion doing its job — the market believed the panic would fade, so it refused to re-price the back end. Within weeks the curve had re-flattened and slid back into contango, exactly as the mean-reversion model predicts. Traders who shorted the front-month spike (rather than chased it) and who had survived the gap were paid as the inversion normalized.

### The everyday case: an earnings event hump

You do not need a crisis to see the term structure at work. Every earnings season, single-stock term structures sprout event humps. A liquid name reporting next week will show its front-week expiry bid up — sometimes implying 60–80% vol on a stock whose back-month vol is 35% — purely because that expiry spans the report. Read the gap between the expiry that spans earnings and the one that does not, run the forward-vol calc, and you have extracted the market's implied earnings move. The day after the report, that hump collapses in the vol crush, and the front-week vol snaps back toward the base curve. This is the most reliable, most repeatable term-structure pattern in all of options, and it is the bread and butter of event-vol selling.

## The playbook: how to trade the term structure

Here is the operational summary — what to look at, what to put on, and what kills you.

**Read the curve first, every time.** Before any vol trade, pull the term structure: the VIX9D / VIX / VIX3M / VIX6M for the index, or the option-chain implied vols by expiry for a single name. Is it in contango (upward) or backwardation (inverted)? How steep? Where is the front relative to the long-run mean near 19.5? The shape is your map; do not trade vol without it.

**Match the structure to the shape.**

- **Curve in steep contango, you think it stays calm:** the roll-down is your friend. Short front-month vol (short straddle/strangle over a quiet period, or short the front vol product, hedged) and collect the theta of the curve — about 6.5% a month on the front roll in our example. *Size for the gap*, because contango ends in one violent session.
- **Front-month event hump (earnings, FOMC, CPI):** sell the rich front and own a quieter back as a **calendar** — our example turned a \$1.10 debit into \$2.37 on normalization, a 216% return — or sell the front-month event vol outright and harvest the post-event crush. You win when the hump deflates; you lose on a big move away from the strike.
- **Curve deeply inverted (backwardation), panic already priced:** do *not* chase long vol. Fade the front — short the spiked front-month vol (hedged, sized small) and bet on the curve normalizing as mean reversion drags the front back down. This is the "sell the spike" trade, and it pays when the inversion re-flattens.

**Use forward vol to find the real view.** When the spot implied vols look unremarkable, compute the forward vol between two expiries — it can reveal a violent expectation hiding in a calm-looking curve (our 1-to-3-month forward came out at 17.1% off a 14.5% front, and the earnings-window forward at ~54%). Trade the forward vol you think is mispriced, not the headline number.

**Know your invalidation cold.** For every term-structure trade, the killer is the curve *flipping*: a contango roll-harvest dies when the curve inverts; a calendar dies on a big directional move or a vol spike that re-steepens the wrong way. Define the inversion or the move that ends the thesis *before* you put the trade on, and have the hedge or the stop ready. The roll-down is real carry, but the carry-to-catastrophe ratio is exactly why Volmageddon and the 2024 unwind are remembered. The shape is your edge; the shape flipping is your ruin. Trade it knowing both.

**The one rule to carry out the door:** the level of vol is one number, but the *shape* of the curve is the trade — contango is carry you collect, backwardation is the fear that's already priced, and the moment the two swap is the moment positions blow up.

## Further reading & cross-links

- [Vega: your exposure to implied volatility and the vol-of-vol](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) — vega grows with maturity; this post maps vol *across* the maturities vega lives on.
- [Theta: trading the clock and the price of being long options](/blog/trading/options-volatility/theta-trading-the-clock-and-the-price-of-being-long-options) — the decay that makes the front month of a calendar bleed faster than the back.
- [Implied vs realized volatility: the trade at the heart of options](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options) — the gap the whole vol business trades; the curve is its term-by-term map.
- [What sets an option's price: the five inputs and the intuition](/blog/trading/options-volatility/what-sets-an-options-price-the-five-inputs-and-the-intuition) — where implied vol enters the price, before we split it by expiry.
- [The VIX and vol products: VIX, VXX, UVXY, and the cost of the roll](/blog/trading/options-volatility/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll) — the full treatment of the roll cost we set up here (forward reference).
- [Calendars and diagonals: trading time and term structure](/blog/trading/options-volatility/calendars-and-diagonals-trading-time-and-term-structure) — the calendar mechanics in depth (forward reference).
- [Reading the vol surface like a trader: the 3D map of fear](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) — the term structure plus the skew, in one surface (forward reference).
- [The volatility surface as a no-arbitrage object](/blog/trading/quantitative-finance/volatility-surface) — the theory of why the surface (term structure × skew) must be arbitrage-free.
- [The expected move: pricing event risk with options](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options) — how the event hump translates into a tradeable expected move.
