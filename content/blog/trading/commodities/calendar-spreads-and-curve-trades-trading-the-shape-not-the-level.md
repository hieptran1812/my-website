---
title: "Calendar Spreads and Curve Trades: Trading the Shape, Not the Level"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How to trade the forward curve directly instead of the flat price: a calendar spread is long one delivery month and short another, so the level cancels and you profit only from a change in the shape — lower margin, smaller risk, and the famous blow-ups that prove it is not riskless."
tags: ["commodities", "calendar-spread", "curve-trade", "forward-curve", "contango", "backwardation", "futures", "spread-margin", "roll", "amaranth", "crude-oil"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A flat-price bet is a wager on the *level* of a commodity (up or down). A **calendar spread** is a wager on the *shape* of the forward curve: you go long one delivery month and short another, and you profit only when the **gap between the two contracts changes**. The parallel move in the level cancels between the two legs, so you have isolated the one thing you actually have a view on.
>
> - Because the legs largely offset, a calendar spread carries far **less directional risk** than an outright position — and exchanges know it, charging **spread margin** that is a small fraction of two separate outright margins.
> - A **bull spread** is long the front and short the back; it wins when the prompt tightens (the curve moves toward backwardation). A **bear spread** is the reverse; it wins when the prompt loosens (the curve moves toward contango).
> - "Lower risk" is not "no risk." A spread that looks tiny tempts traders to size it huge, and when the gap moves further than any model said, the loss scales with the size — that is exactly how **Amaranth lost about \$6.5 billion** on a natural-gas calendar spread in 2006, and how the **April 2020 prompt WTI spread** blew out as the front went negative.
> - The one idea to remember: a calendar spread lets you express a precise view — *"this market is about to get tight"* — while throwing away the part of the bet you have no edge on, the flat price.

In September 2006, a single hedge fund called Amaranth Advisors held a position in natural-gas futures that, on paper, looked almost boring. The fund was not betting that gas would simply rise or fall. It was betting on the *relationship* between two delivery months — long the March contract, short the April contract, a bet that the difference between winter gas and spring gas would widen. This is a **calendar spread**, and the whole appeal of the trade is that it is supposed to be *safer* than an outright bet: the two legs move together, so the big directional swings cancel, and you are left exposed only to the small, slow drift of the gap between them. Amaranth's star trader had made a fortune on exactly this structure the year before.

Then the gap moved the wrong way, fast. The "safe," offsetting position lost about \$6.5 billion in a matter of weeks — one of the largest hedge-fund collapses in history — on a trade that, leg for leg, was about as far from a wild directional gamble as a commodity position gets. How does a "limited-risk" spread vaporize a multi-billion-dollar fund? The answer is the entire subject of this post, and it is also the reason calendar spreads are one of the most useful and most misunderstood tools a commodity trader has.

This is the first post in the series' track on the *mechanics of trading* commodities, and it builds directly on the curve. We have spent earlier posts learning to *read* the forward curve — its two shapes, [contango vs backwardation](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means), and the [convenience yield and cost of carry](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape) that give it that shape. Now we learn to *trade the curve directly* — to take a position on the shape itself, deliberately stripping out the flat price. By the end you will understand why a curve trade is lower-risk than a flat-price bet, when it is the right tool, and why "limited risk" is a phrase that has bankrupted some very smart people.

![A calendar spread is long the back month and short the front, profiting if the curve shape shifts](/imgs/blogs/calendar-spreads-and-curve-trades-trading-the-shape-not-the-level-1.png)

## Foundations: what a calendar spread actually is

Start from zero, because the whole idea rests on one distinction that beginners constantly blur: the difference between the *level* of a commodity's price and the *shape* of its forward curve.

Recall the setup from the rest of this series. A commodity does not have one price. It has a *menu* of prices — one futures price for each delivery month, stretching out into the future. Plot those prices against their delivery dates and connect the dots, and you get the **forward curve**. The *level* is roughly how high the whole curve sits (is oil around \$40 or around \$100?). The *shape* is the slope and curvature of the line — does it rise to the right (contango), fall to the right (backwardation), or bend? Two completely separate facts. Oil can be expensive and in contango, or cheap and in backwardation, or any mix.

A normal, "outright" futures position bets on the **level**. If you buy one December crude contract, you make money when December crude goes up and lose when it goes down — full stop. Your profit or loss is whatever happens to that one price. You are *long the level* (at one point on the curve), and the whole curve sliding up or down is what moves you.

A **calendar spread** — also called a *time spread* or *horizontal spread* — bets on the **shape** instead. You hold *two* offsetting legs at once: you buy (go long) one delivery month and sell (go short) a different delivery month *of the same commodity*. For example, long December crude and short June crude. Now ask what happens when the *whole* curve moves up by, say, \$3. Your long December leg gains \$3. Your short June leg loses \$3. They cancel. The parallel shift in the level — the very thing an outright position lives and dies on — does *nothing* to your spread. What is left? Only the *difference* between the two contracts, the **spread** itself: December price minus June price. You make or lose money only when that gap changes.

That is the entire trick, and it is worth saying slowly because everything follows from it. **A calendar spread converts a bet on the level into a bet on the shape.** You have surgically removed the part of the trade you may have no edge on (where oil is going overall) and kept the part you do have a view on (whether the prompt market is about to tighten or loosen relative to later months).

The figure above lays the two side by side: the outright bettor *owns the level* and needs the whole curve to move their way; the spread trader *owns the shape* and only needs the gap between two contracts to change. The chart below shows where the two legs actually sit on the curve, and the gap they trade.

![Two forward curve shapes with the two legs of a calendar spread and the gap marked](/imgs/blogs/calendar-spreads-and-curve-trades-trading-the-shape-not-the-level-2.png)

The orange line is a contango curve (back months dearer); the green is backwardation (front months dearer). The two red circles mark a spread's legs — short the front (M0), long the back (M12) — and the red double-arrow is the **gap**, the spread you actually own. Notice that this gap is a property of the *shape*, not the *height*: you could slide the whole orange curve up or down by \$20 and the gap between M0 and M12 would be unchanged. That invariance is the source of every advantage a spread has over an outright.

#### Worked example: the spread P&L on a December-June crude calendar

Suppose you put on a calendar spread in crude oil. You go **long December at \$74** and **short June at \$76**. The spread (front-leg-minus-back, by the convention we'll use — long-the-cheaper-deferred, short-the-dearer-prompt) is December minus June: \$74 − \$76 = **−\$2**. You are betting this gap will *narrow* — that December will gain on June.

Now a few weeks pass. The whole oil complex sells off \$8: December falls to \$66, June falls to \$68. Has your spread changed? December minus June is now \$66 − \$68 = −\$2. Identical. The \$8 collapse in the level did *nothing* to you — your long December lost \$8, your short June made \$8, and they cancelled exactly. You are completely insulated from the direction of oil.

Then the shape shifts: the prompt tightens, and December climbs relative to June. Say December is \$69 and June is \$70.50 — the spread is now \$69 − \$70.50 = **−\$1.50**. It narrowed from −\$2 to −\$1.50, a move of **+\$0.50 in your favor**. Crude contracts are 1,000 barrels each, so on one pair of contracts that is \$0.50 × 1,000 = **\$500 of profit**. If you held five pairs, \$2,500. And you earned it without having any idea where oil itself was heading.

The intuition: your entire profit came from the \$0.50 change in the gap between two contracts — the *shape* — while the \$8 swing in the level passed through you without a trace.

### Who actually trades the shape, and why

It is fair to ask: who *wants* a position that is deliberately blind to the price of oil? Quite a lot of the market, it turns out, and for good reasons that are worth knowing because they tell you who is on the other side of your spread.

**Physical traders and merchant houses** are the natural home of calendar spreads. A firm that actually moves barrels, cargoes, and tonnes — the [commodity trading houses](/blog/trading/finance/commodity-trading-houses-glencore-vitol-trafigura) and the trading arms of producers and refiners — sees the *physical* market directly: which tanks are filling, which cargoes are stranded, where the prompt is tight. Their edge is almost never a view on the flat price (nobody reliably knows where oil is going); it is a view on the *relative* value of prompt vs deferred, which is exactly what a calendar spread expresses. When such a firm has surplus barrels and contango pays to store, they put on a *cash-and-carry* — buy the cheap physical, sell the dear deferred future, store, and lock the gap — which is a calendar spread with a physical leg. The curve's shape is, quite literally, their bread and butter.

**Hedgers** create spreads as a by-product. A producer who sold forward and a consumer who bought forward both sit at specific points on the curve, and as their hedges age and need rolling, they transact calendar spreads to move their exposure from one month to the next. A long-only commodity index fund is the most visible example: every month it must roll its position down the curve, and that roll *is* a giant calendar-spread transaction repeated across the whole market — which is why the roll's timing and cost (the subject of [roll yield](/blog/trading/commodities/roll-yield-and-why-long-only-commodity-etfs-bleed)) is itself a tradable event.

**Relative-value speculators** trade spreads precisely because they offer a cleaner risk for the same research effort. If you have spent a month understanding a market's storage and supply picture, you have formed a view on its *shape*, not necessarily its level — so a spread monetizes your actual research while throwing away the level risk you did not study. Many systematic and discretionary funds run entire books of nothing but calendar and inter-commodity spreads, because spread risk is more forecastable than flat-price risk: the physical relationships that drive a curve's shape are slower-moving and more analyzable than the macro chaos that drives the level.

The common thread: the people who trade the shape are the people who *know the physical market* and have a view on relative scarcity, not the people guessing the next macro headline. When you put on a spread, that is who you are trading against — which is a useful, humbling thing to remember.

## Why the spread cancels the level

Let us make the cancellation airtight, because it is the foundation of the whole "lower risk" claim and it is easy to half-understand.

A futures position's profit is (in the simplest form) the change in the contract's price times the contract size, with a plus sign if you are long and a minus sign if you are short. Stack the two legs of a calendar spread:

- **Long leg** (say long the back month): profit = +(change in back-month price) × size
- **Short leg** (say short the front month): profit = −(change in front-month price) × size

Add them. The total spread P&L = [(change in back) − (change in front)] × size = (change in *the gap*) × size. The individual price *levels* dropped out of the formula entirely; only their *difference* survives. Whatever happens to both prices together — a Fed surprise, a demand-destruction shock, a risk-off panic that drags every commodity down — affects both legs equally and nets to zero. Only a move that *separates* the two contracts shows up in your P&L.

This is why traders say a calendar spread "hedges out the flat price." You are not hoping the legs offset; the offset is mechanical, baked into the arithmetic. The figure below walks the cancellation through a concrete \$10 parallel drop, contrasting it with the outright long that eats the full move.

![Before and after comparison showing a parallel curve move cancels in a spread but hits an outright long](/imgs/blogs/calendar-spreads-and-curve-trades-trading-the-shape-not-the-level-3.png)

On the left, the outright long owns the level: the whole curve drops \$10 and the position loses the full \$10 per barrel. On the right, the calendar spread takes the same \$10 drop — the long leg loses \$10, the short leg gains \$10, they cancel, and the *only* thing that can move the spread's P&L is a change in the gap. You isolated the shape.

There is a beautiful consequence here that connects straight back to the series' spine. A commodity price, we keep saying, is *a physical thing forced through a financial contract* — and the forward curve, storage cost, and convenience yield are the gears that set the curve's shape. A calendar spread is the purest possible trade on those gears. When you trade the shape, you are betting directly on the physical reality the curve encodes: is the commodity scarce *right now* relative to later (which steepens backwardation, narrowing a long-front spread in your favor), or is there a glut piling up in storage (which steepens contango, widening it)? An outright position muddies that view with the noise of the overall price level. A spread distils it.

### Two technical notes that keep the trade honest

Before we go further, two refinements that separate a clean spread trade from a sloppy one.

**The legs must be the same commodity, same exchange, same contract spec.** A calendar spread is long and short *the same underlying* in different months. Long December WTI and short June WTI is a calendar spread. Long December WTI and short June *Brent* is something else entirely — an *inter-commodity* spread (more on those later) — and it does *not* fully cancel the level, because WTI and Brent can diverge. Keep the calendar spread within one contract or you reintroduce exactly the risk you were trying to remove.

**Near expiry, the front leg gets twitchy.** The front month converges to spot and sees a flood of roll activity in its final days, which can jerk the prompt spread around for reasons that have nothing to do with the curve's underlying shape. Practitioners who hold spreads through expiry either roll the front leg early or measure their "view" off the second and third months once the front is within a week of dying — the same caution we met when reading the curve itself. A spread is cleaner than an outright, but it is not immune to the mechanical noise of expiry.

### A spread trades as one instrument, not two trades

A subtlety that trips up beginners: you do *not* usually put on a calendar spread by buying one outright contract and then, separately, selling another. You can — and a retail trader often does — but the professional way is to trade the **spread itself as a single, listed instrument**. Exchanges list calendar spreads directly: there is a quoted market for "December minus June crude," with its own bid and offer expressed as the *price difference*, and you trade that one number.

Why does this matter? Two reasons, both practical. First, **execution risk**. If you "leg in" — buy the December contract, then turn to sell June — the market can move in the seconds between the two fills, leaving you with a worse spread than you intended (or, briefly, an unhedged outright leg). Trading the spread as one instrument fills both legs *simultaneously* at the agreed difference, so you never carry naked level risk even for an instant. Second, **liquidity and cost**. The spread market is often deeper and tighter than legging the two outrights, because it is exactly where the spread-trading desks concentrate. The bid-offer on a near-dated crude calendar spread might be a penny or two, far tighter than the combined cost of crossing two separate outright spreads.

The mechanics also explain a quoting convention worth internalizing. A calendar spread is quoted as *the price of the nearer leg minus the price of the farther leg* (or by an exchange-specified convention). A *positive* quote means the front is dearer (backwardation); a *negative* quote means the front is cheaper (contango). When a trader says "I bought the spread at minus two," they bought the structure when the front was two dollars below the back — long the cheaper deferred or short the dearer back depending on the contract's convention — and they profit if that number rises toward zero and beyond. Getting the sign convention right is not pedantry: a flipped sign means you put on the *opposite* trade, which is how careless desks turn a bull spread into a bear spread by accident.

### The spread ratio: keeping the legs balanced

One more piece of plumbing. A *one-to-one* calendar spread — one contract long, one short — is balanced in *contracts* but, strictly, you might want it balanced in *dollar sensitivity*. For two months of the same commodity the contract sizes are identical (a December crude contract and a June crude contract are both 1,000 barrels), so one-to-one is genuinely balanced and the level cancels cleanly. But for *inter-commodity* spreads the legs often differ in size or in how much each moves per unit, so traders use a **ratio** — the 3-2-1 crack uses three crude contracts against two gasoline and one distillate precisely to match the physical conversion. For a plain calendar spread you can ignore ratios; for its inter-commodity cousins, the ratio is the whole design. Keep that distinction filed away — it is why a calendar spread cancels the level perfectly and a crack spread only approximately.

## The first big payoff: lower risk

Now the practical question — *why bother?* The headline answer is **risk**. A calendar spread is, in almost every case, far less risky than either of its legs traded outright. Here is why, in two layers.

**Layer one: the legs offset, so day-to-day volatility collapses.** An outright crude position swings with the full daily move of oil, which is often \$1–\$3 a barrel and occasionally far more. A calendar spread only swings with the *change in the gap* between two nearby months, which is usually a fraction of that — typically a few cents to a few dimes a day in a calm market. The two legs are highly correlated (they are the same commodity weeks apart), so most of the volatility cancels. Empirically, a near-dated crude calendar spread might have one-fifth to one-tenth the daily volatility of the outright. You have taken a position whose worst plausible day is a small fraction of an outright's worst plausible day.

**Layer two: the exchange knows this, and charges you accordingly — spread margin.** This is the part that makes spreads economically attractive, not just psychologically comfortable. When you trade futures, the exchange makes you post **margin** — a good-faith deposit sized to the position's risk, so that if the market moves against you overnight the clearinghouse is covered. For an outright position, margin reflects the full single-leg risk. But for a *recognized calendar spread*, the exchange's risk engine sees the two offsetting legs and charges a much smaller **spread margin** — because the position's real risk is much smaller.

#### Worked example: the margin saving on a crude calendar spread

Suppose the exchange's initial margin on one outright WTI crude contract is about \$6,000 (margins move with volatility; this is illustrative). To trade the December and June legs *separately* as two outright positions, you would post margin on both: roughly \$6,000 + \$6,000 = **\$12,000** of capital tied up.

But put them on as a *recognized calendar spread*, and the exchange's risk system charges **spread margin** — often on the order of \$1,000 to \$1,500 for a near-dated crude calendar, because the two legs offset. Call it \$1,200. So the *same economic position* — long December, short June — ties up about **\$1,200 instead of \$12,000**, roughly a tenth of the capital.

That is a tenfold improvement in capital efficiency, and it is not a loophole — it is the clearinghouse correctly pricing the fact that your offsetting legs carry a fraction of the risk. The same \$50,000 of trading capital that supports a handful of outright contracts can support a large book of spreads.

The intuition: lower real risk earns you lower margin, and lower margin means the same capital expresses a much larger spread view — which is exactly why spread desks can run enormous notional positions on modest capital.

That capital efficiency is a double-edged sword, and we will come back to it with a vengeance: it is precisely the temptation that turned Amaranth's "safe" spread into a catastrophe. Low margin invites large size, and large size on a small-looking trade is how limited-risk positions blow up. Hold that thought.

## Bull spreads and bear spreads: expressing a view

So far the spread has been a vague "long one month, short another." But *which* month you go long encodes a specific view about the market, and the two directions have names.

A **bull spread** (in this calendar context) is **long the front month, short the back month**. You are betting the prompt will *gain* on the deferred — that near-term scarcity will tighten the curve, pulling it toward backwardation. You go long the front because you expect *near-term strength*: a supply disruption, a demand surge, inventories drawing down fast. The name comes from being bullish on the *prompt*.

A **bear spread** is the reverse: **long the back month, short the front month**. You are betting the prompt will *lose* relative to the deferred — that a building glut will loosen the front, pushing the curve toward contango. You favor the deferred because you expect *near-term weakness*: storage filling up, demand softening, the prompt becoming a burden.

Crucially, both are bets on the *shape*, not the level. A bull spread can make money even if oil is falling overall (as long as the front falls *less* than the back). A bear spread can make money even if oil is rising (as long as the front rises *less* than the back). The matrix below lays out which leg you are long, the view it expresses, and the curve move that pays it off.

![Matrix comparing bull spread and bear spread legs views and winning conditions](/imgs/blogs/calendar-spreads-and-curve-trades-trading-the-shape-not-the-level-5.png)

Read the matrix row by row. The bull spread (long front, short back) is the trade you put on when you think *the market is about to get tight right now* — it wins as the curve moves toward backwardation. The bear spread (long back, short front) is the trade for *a glut is coming and storage will have to be paid for* — it wins as the curve moves toward contango. The bottom row gives the read: the spread's P&L is the change in the front-minus-back gap times contract size, and nothing else.

#### Worked example: a contango-widening bear spread in natural gas

Natural gas in the United States is the textbook chronic-contango market, because seasonal storage means the curve usually slopes up. Suppose you believe a mild winter is coming and storage will end the season unusually full — a building glut that should *widen* the contango (loosen the prompt relative to deferred months). You put on a **bear calendar spread**: long the deferred month, short the front month.

Say you enter with the front month at \$3.00/MMBtu and the deferred month at \$3.40/MMBtu. The spread (front minus deferred) is \$3.00 − \$3.40 = **−\$0.40**, and you are positioned for it to get *more* negative (the contango to widen). The mild winter arrives, storage swells, and the prompt sags: the front drops to \$2.60 while the deferred holds at \$3.35. The spread is now \$2.60 − \$3.35 = **−\$0.75**. It widened from −\$0.40 to −\$0.75, a move of **−\$0.35** — and because you were short the front and long the deferred, that widening is *in your favor*.

A NYMEX natural-gas contract is 10,000 MMBtu, so \$0.35 per MMBtu × 10,000 = **\$3,500 of profit per spread pair**. You never had to predict the absolute price of gas; you only had to be right that the glut would loosen the front relative to the back.

The intuition: a bear spread is a clean way to bet "the prompt is too rich and a glut will cheapen it" — you profit from the contango *widening*, regardless of where the overall gas price goes.

### Where the spread's edge actually comes from

It is worth being honest about *why* a spread trader expects to make money, because "I think the curve will tighten" is a view, not an edge. Real spread edges tend to come from one of a few places:

- **A physical read.** You watch inventories, refinery runs, weather, shipping, and OPEC+ chatter, and you form a view on whether the *prompt* market specifically is about to get tight or loose. This is the domain of physical traders and merchant houses — [commodity trading houses](/blog/trading/finance/commodity-trading-houses-glencore-vitol-trafigura) like Glencore, Vitol, and Trafigura run enormous spread books precisely because they *see the physical flows* and can read the prompt's tightness before the screen does.
- **A seasonal pattern.** Some calendar spreads have reliable seasonal tendencies — heating-oil and gas spreads tighten into winter, gasoline cracks widen into summer driving season. These are not free money (everyone knows them, so they are partly priced in), but they are a starting hypothesis.
- **A carry/roll view.** This connects to [roll yield](/blog/trading/commodities/roll-yield-and-why-long-only-commodity-etfs-bleed): a backwardated front structure hands a positive carry to a long-front position, so a bull spread in a backwardated market collects roll as well as betting on further tightening.

Without one of these, a spread is just a lower-variance way to be wrong. The lower risk is real, but it is not the same as positive expected return.

## How the P&L actually accrues: it is the change in the gap

Let us watch a spread trade play out over time, because seeing the P&L *accrue from the gap* — not from the level — is what finally makes the concept stick.

Follow a trader who looks at a deeply contangoed crude curve, decides the prompt is artificially cheap (a temporary storage scare that will clear), and puts on a **bull spread**: long the front, short a deferred month. At entry, the front-minus-back gap is deeply negative — the front trades well below the deferred. The bet is that the gap will *narrow and then flip positive* as the prompt tightens. The chart below traces an illustrative path of that gap from entry to exit.

![Illustrative path of a calendar spread gap tightening from contango to backwardation over time](/imgs/blogs/calendar-spreads-and-curve-trades-trading-the-shape-not-the-level-4.png)

The blue line is the spread (front minus back) over the weeks the trade is held. It starts in contango (negative, amber region) and tightens steadily until it flips into backwardation (positive, green region). The trader's P&L is *the vertical distance the line traveled* — the total change in the gap — times the contract size, with nothing contributed by the level of oil at all. Whether crude was at \$60 or \$90 throughout this period is irrelevant to the picture; only the gap's journey from deep negative to positive matters.

This is the single most important mental shift in moving from flat-price trading to curve trading. On an outright chart, you watch *the price* and your P&L is the price's path. On a spread chart, you watch *the gap* and your P&L is the gap's path. The level becomes scenery. A spread trader can be completely indifferent to a \$30 move in oil and intensely focused on a 40-cent move in a calendar gap, because the 40 cents is their entire world.

#### Worked example: marking a spread to market day by day

You are long the December-June crude bull spread, entered with the spread at **−\$2.00** (December \$74, June \$76), one pair of contracts (1,000 barrels each).

- **Day 5:** December \$75, June \$76 → spread −\$1.00. Change from entry: −\$2.00 → −\$1.00 = **+\$1.00**. Mark-to-market profit: \$1.00 × 1,000 = **+\$1,000**.
- **Day 12:** December \$76, June \$76 → spread \$0.00 (the curve flattened). Change from entry: **+\$2.00**. Profit: **+\$2,000**.
- **Day 20:** December \$77, June \$75.50 → spread +\$1.50 (now backwardated). Change from entry: −\$2.00 → +\$1.50 = **+\$3.50**. Profit: \$3.50 × 1,000 = **+\$3,500**.

At every step the level of oil did different things — December went from \$74 to \$77, June *fell* from \$76 to \$75.50 — but your P&L tracked only the gap: −\$2.00 → −\$1.00 → \$0.00 → +\$1.50. The curve went from \$2 of contango to \$1.50 of backwardation, a \$3.50 tightening, and you collected every cent of it as \$3,500 per pair.

The intuition: your profit *is* the change in the spread, marked continuously — you are paid for the curve tightening exactly the amount it tightens, and the absolute price of oil never enters the calculation.

## Inter-commodity spreads: the cousins

A calendar spread trades the same commodity in two months. Its close cousins trade *two different but related commodities* against each other, and the logic is similar — you are betting on a *relationship*, not a level — but the cancellation is no longer perfect, because the two commodities can genuinely diverge. We link out for the deep dives, but it is worth seeing the family.

The **WTI-Brent spread** is long one benchmark crude and short the other. It is a bet on the relationship between America's landlocked WTI and the seaborne Brent — driven by US shale output, pipeline bottlenecks, and export economics. Unlike a true calendar spread, this one *can* trend a long way (the shale glut pushed Brent to a structural premium over WTI for years), because the two crudes are different physical streams in different places. The full story is in [WTI vs Brent](/blog/trading/commodities/crude-oil-wti-vs-brent-the-worlds-two-benchmark-barrels).

The **crack spread** is long crude and short the refined products (gasoline, diesel) it becomes — or the other way around. It is the refiner's *margin*, and trading it is a bet on whether refining is about to get more or less profitable. The classic structure is the 3-2-1 crack (three barrels of crude in, two of gasoline and one of distillate out). The full mechanics live in [refining and crack spreads](/blog/trading/commodities/refining-and-crack-spreads-turning-crude-into-products).

The **crush spread** is the agricultural analog: long soybeans and short the soybean meal and oil they get crushed into (or the reverse). It is the soybean processor's margin. Same idea, different commodity chain.

Why group these with calendar spreads at all? Because they share the same *mindset*: you have decided you have a view on a **relationship** but not on the **level**, so you construct a position that profits from the relationship and is largely (crack, crush) or partly (WTI-Brent) immune to the level. The key difference to keep straight: a *calendar* spread's legs are the same commodity, so the cancellation of the level is essentially perfect; an *inter-commodity* spread's legs are different commodities, so the cancellation is only approximate and the spread can drift much further than a calendar gap ever would. That extra freedom is exactly why inter-commodity spreads can be both more profitable and more dangerous.

There is a useful way to rank the whole family by *how completely the level cancels*, because that ranking is also a ranking of risk. At one extreme sits the calendar spread: same commodity, identical contract size, near-perfect cancellation, smallest residual risk, lowest margin. A step out sits the *location* or *grade* spread within one commodity (WTI at Cushing vs WTI at the Gulf Coast, or a sour crude vs a sweet one) — same broad commodity but a real physical difference that can widen with pipelines and refinery demand. Another step out sits the crack and crush — different commodities, but bound together by a physical *conversion process* (crude becomes products, beans become meal and oil), so the relationship is anchored by the economics of running the plant. At the far extreme sits a pure inter-market spread between loosely related commodities (gold vs silver, corn vs wheat) — here the "relationship" is statistical and behavioral, not a hard physical link, so it can diverge almost without limit. The further down this ladder you go, the more of the level *leaks back in*, the wider the spread can roam, and the more margin the exchange charges — because the offset is doing less and less of the work.

The practical upshot: a calendar spread is the *safest* member of the family precisely because its legs are physically identical and only weeks apart. Every step toward a looser relationship trades some of that safety for a wider opportunity. Knowing where your spread sits on this ladder tells you, before you put it on, roughly how far it can move against you and therefore how to size it.

#### Worked example: a WTI-Brent inter-commodity spread vs a calendar spread

Compare the two side by side on the same capital, so the ladder becomes concrete. Suppose you think US export bottlenecks will *widen* Brent's premium over WTI. You buy Brent and sell WTI — an inter-commodity spread — entered when Brent is \$82 and WTI is \$78, a Brent-minus-WTI spread of **+\$4**. Your thesis plays out and the spread widens to **+\$7** (Brent \$84, WTI \$77). Both crude contracts are 1,000 barrels, so the gain is \$3 × 1,000 = **\$3,000 per pair**.

Now the catch: because Brent and WTI are *different physical streams in different places*, this spread can also keep moving — for years it sat anywhere from near zero to over \$10 as shale output and pipeline capacity shifted. The exchange knows the legs do not cancel as cleanly as a calendar spread's, so it charges *more* margin than a calendar spread (the offset is only partial) and the position can roam much further against you. Run the same \$3 *adverse* move and you lose \$3,000 a pair — but unlike a calendar gap that is anchored by storage arbitrage at a few dollars, a WTI-Brent gap has historically traveled \$10 or more, so the tail is genuinely fatter.

The intuition: a calendar spread's legs are the same barrel weeks apart, so the gap is tethered by storage economics and stays narrow; an inter-commodity spread's legs are different barrels in different places, so the relationship can stretch far — more opportunity, but a wider tail and more margin, exactly as the ladder predicts.

## Common misconceptions

A handful of myths about spreads are so common — and so costly — that they deserve to be named and corrected with numbers.

**Myth 1: "A spread is hedged, so it can't lose much."** This is the Amaranth myth, and it is lethal. A spread *cancels the level*, but it is fully exposed to the *gap*, and the gap can move enormously. Amaranth's March-April 2007 gas spread moved several dollars per MMBtu against them — on a position sized in the tens of thousands of contracts, that is billions. The legs offset the *level*; they do *nothing* to offset a move in the *spread itself*, which is the only thing you are exposed to. A spread is not a hedge against its own thesis being wrong.

**Myth 2: "Lower margin means lower risk."** Margin reflects *normal* risk, not *tail* risk. The exchange's spread margin assumes the gap behaves like it usually does — small, slow moves. When the gap dislocates (storage runs out, a delivery squeeze hits the prompt), it can move many times the "normal" range the margin was sized for. Low margin makes a spread *capital-efficient*, which tempts traders to size it large, which makes the tail risk *worse*, not better. The April 2020 prompt WTI spread went from a few dollars of contango to *tens of dollars* as the front went to −\$37 — a move no margin model had priced.

**Myth 3: "The spread will mean-revert, so just hold it."** Spreads do mean-revert *until they don't*. A curve that is "too steep" can get steeper for weeks if the underlying physical situation keeps deteriorating — tanks keep filling, the squeeze keeps tightening. "It has to come back" is not a risk-management plan; it is the famous last words of every blown-up spread book. The market can stay dislocated longer than you can meet margin calls.

**Myth 4: "Spreads are only for the prompt; the back end is dead."** Far-deferred calendar spreads (a year-out vs two-years-out) are less volatile but very much alive — they are where structural views on the energy transition, long-run supply, and capex cycles get expressed. The far curve moves less day to day, but it carries real information about the market's multi-year supply picture.

**Myth 5: "A calendar spread is a free way to collect roll yield."** Partly true, partly dangerous. In backwardation, a long-front spread does collect positive carry as the front rolls up the curve — but you are *also* exposed to the spread moving against you. The carry is a tailwind, not a guarantee; if the curve flips to contango while you hold, the carry reverses and you lose on the gap too. Roll yield and spread P&L are linked but not the same thing.

## How it shows up in real markets

The theory becomes real at the moments it breaks. Three episodes show calendar and curve trades doing exactly what the textbook says — and then doing the thing the textbook warns about.

### Amaranth, 2006: when a "limited-risk" spread ate a fund

Amaranth Advisors was a multi-strategy hedge fund whose energy desk, run by trader Brian Hunter, made roughly \$1 billion in 2005 betting on natural-gas calendar spreads — chiefly the **March-minus-April** spread. The trade has real logic: March is the last full month of winter heating demand, April is the first month of spring, and gas in storage going into the shoulder season can leave the March-April gap wide. Hunter was long March, short April — betting the winter premium would hold or widen.

In 2006, the bet went the other way. A mild winter and ample storage *narrowed* the March-April spread instead of widening it. Because the position looked "hedged" and carried low spread margin, it had been sized to a staggering scale — Amaranth at one point held a dominant share of the open interest in some gas contracts. When the spread moved against them and they tried to exit, there was no one to take the other side at anything like a fair price; their own exit pushed the spread further against them. The fund lost about **\$6.5 billion** in September 2006 and collapsed. The position was, contract for contract, a low-volatility spread. The *size* turned a modest adverse move in the gap into one of the largest trading losses in history.

The lesson is the whole point of this post: the spread cancelled the level, exactly as designed — and it did not matter, because the risk was never the level. The risk was the gap, the gap moved, and the size did the rest.

### April 2020: the prompt spread that went to minus thirty-seven

On 20 April 2020, the expiring May WTI contract settled at **−\$37.63** — the first negative oil price in history. Demand had collapsed in the pandemic, a Saudi-Russian price war flooded the market, and the storage tanks at Cushing, Oklahoma — WTI's delivery point — were nearly full. Anyone long the prompt contract who could not take physical delivery had to sell at *any* price before expiry, and with no tanks to put the oil in, "any price" went below zero: sellers paid buyers to take the barrels.

For a calendar spread, this was a curve event of historic magnitude. The chart of that "super-contango" tells the story.

![Illustrative April 2020 super-contango WTI curve with the front collapsing and the front-back gap exploding](/imgs/blogs/calendar-spreads-and-curve-trades-trading-the-shape-not-the-level-6.png)

The front month collapsed toward (and through) zero while contracts a year out still fetched over \$35 — a front-to-back gap of *more than \$50* at the extreme, an order of magnitude wider than a normal crude calendar gap. A bear spread (short the front, long the back) put on before the crash would have made a fortune as the contango exploded. A bull spread (long the front) caught on the wrong side would have been annihilated, because the "limited" downside on the front leg turned out to include *negative prices* — a possibility no risk model had contemplated. The very assumption that makes a long position feel safe (a price can't go below zero) was false, and the spread paid for that false assumption in full.

This is the cleanest possible illustration of Myth 2: the margin on that prompt spread had been sized for a world where the gap moved in dollars, not tens of dollars, and where prices stayed positive. When the physical reality — no storage left — broke both assumptions at once, the spread did what no model said it could.

And it is the cleanest illustration of the series' spine, too. The April 2020 super-contango was not a financial accident; it was the *physical* world asserting itself through the paper. Every tank at Cushing was full, so a barrel for May delivery was a genuine liability — you would have had nowhere to legally put it — while a barrel a year out, when the glut would surely have cleared, was a normal asset worth \$35. The curve's shape was screaming a physical fact (we have run out of room), and the calendar spread between those two months was the precise number that priced it. A trader who had read the storage data and the tank-capacity numbers in the weeks before could have put on the bear spread and watched it pay off as the physics forced the front below zero. The level — where oil was "worth" — became almost meaningless; the *shape*, driven entirely by storage, was everything. That is the whole thesis of trading the curve: when the physical and the paper diverge, the spread is where you bet on the physical truth.

### The natural-gas "widow-maker" and the chronic-contango market

US natural gas is the market where calendar spreads are most actively, and most dangerously, traded. Because gas is expensive to store and demand is brutally seasonal (heating in winter), the curve is usually in contango and the *winter-summer* spreads can be huge and volatile. The March-April spread that killed Amaranth is nicknamed the **"widow-maker"** for good reason: it concentrates the entire winter-vs-spring storage question into one number, and that number can move violently when a cold snap or a warm winter rewrites the storage outlook overnight.

To see *why* this one spread is so loaded, walk the physical logic. March is the last full month a cold winter can drain storage; April is the first month of the spring refill, when demand collapses and inventories start rebuilding. If the winter is cold and storage ends March nearly empty, March gas commands a huge premium over April — the gap blows out. If the winter is mild and storage stays comfortable, March and April converge — the gap collapses. So the March-April spread is, in effect, a single tradable number that prices the market's entire view of *whether winter will exhaust storage*. A trader with a genuine read on the weather-and-storage trajectory can express it with surgical precision; a trader without one is holding a coiled spring. The reason it earns the name "widow-maker" is that the storage question can be answered abruptly — one polar vortex, one warm spell — and when the answer changes, the spread can move several dollars per MMBtu in days, which on a large position is the kind of move that ends careers.

The deeper point is that the gas curve's shape is *constantly being repriced by storage and weather*, and each repricing is a P&L event for spread holders. Every weekly storage report from the EIA is a moment where the market's view of the winter-vs-spring balance updates, and the March-April spread reprices accordingly. This is genuinely *information-rich* trading — you are not betting on the level of gas, you are betting on the *seasonal storage balance*, which is a much more tractable physical question if you do the work. But the same richness is what makes it lethal to the unprepared: the spread looks calm for weeks, lulling a trader into size, and then a single cold snap detonates it. The discipline is, once again, in the sizing — and in respecting that a "low-volatility" spread is low-volatility right up until the storage question gets answered.

## The playbook: how to express a tightness view as a spread

Pull it all together into something you can actually use. Suppose you have formed a view — you think a particular commodity's *prompt* market is about to get tight (a supply disruption, an inventory draw, a demand surge), but you have *no* strong view on where the overall price is heading. How do you express that cleanly?

**Step 1 — Decide it is a shape view, not a level view.** If you genuinely don't know whether oil is going to \$60 or \$90 but you *do* think the front is about to tighten relative to later months, an outright position is the wrong tool — it loads you with the level risk you have no edge on. A calendar spread is the right tool: it keeps your real view (tightening) and discards the noise (the level).

**Step 2 — Choose the direction.** Tightening view → **bull spread**, long the front, short a deferred month. Loosening/glut view → **bear spread**, long the deferred, short the front. The leg you go long is the leg you think will *outperform* the other.

**Step 3 — Choose the months.** Near-dated spreads (front vs second or third month) are the most sensitive to *prompt* physical conditions — inventories, refinery runs, weather — and move the most per unit of physical news. Deferred spreads (a year out vs two) express structural, slower-moving views (supply cycles, the transition) and are calmer. Match the spread's tenor to the tenor of your view.

**Step 4 — Size for the gap's tail, not its average.** This is the step that bankrupts people who skip it. The exchange will let you put on a large spread for little margin because the *normal* gap volatility is low. Do *not* let the low margin set your size. Size the position so that a *tail* move in the gap — three or four times the normal range, the kind of move that shows up in storage squeezes and delivery dislocations — is survivable. Ask: "If this gap moves like April 2020, am I still solvent?" If the answer is no, you are sized like Amaranth.

**Step 5 — Define the invalidation.** A spread thesis is "the prompt will tighten." It is wrong if the prompt *loosens* instead — if inventories build when you expected a draw. Decide in advance what gap level, or what physical data point (an inventory report, a storage number), tells you the thesis is broken, and exit there. "It has to mean-revert" is not an invalidation; it is denial.

The decision flow below captures the discipline that separates a spread trade from a spread accident.

![Timeline showing how a limited-risk spread gets sized too large and blows up when the gap dislocates](/imgs/blogs/calendar-spreads-and-curve-trades-trading-the-shape-not-the-level-7.png)

Read it left to right: the spread *looks* tiny and safe, so it tempts large size; then a shock moves the gap further than any model said; and because loss equals the move times the size, the oversized position turns a moderate adverse gap move into a disaster. Every blown-up spread in history walked this exact path. The defense is entirely in Step 4 — refusing to let low margin dictate size.

#### Worked example: a spread blow-up, sized two ways

Take the same adverse move and run it through a disciplined size and an Amaranth-style size, so the difference is in dollars.

You put on a crude bull spread, long the front, short the deferred, expecting the curve to tighten. Instead a storage glut hits and the spread moves **\$4 against you** — a tail move, the kind that happens a couple of times a decade. Crude contracts are 1,000 barrels, so the loss is \$4 × 1,000 = **\$4,000 per pair**.

- **Disciplined size:** you hold **10 pairs**, sized so a tail move is survivable against your \$200,000 of capital. Loss = 10 × \$4,000 = **\$40,000**, or 20% of capital. Painful, survivable, you live to trade again.
- **Amaranth-style size:** the spread margin was only ~\$1,200 a pair, so your \$200,000 of capital "allowed" you to hold **160 pairs**. Loss = 160 × \$4,000 = **\$640,000** — more than three times your capital. You are not down 20%; you are insolvent, facing margin calls you cannot meet, forced to liquidate into a market that moves further against you as you sell.

The *trade* was identical. The *direction* was identical. The *adverse move* was identical. The only difference was size — and size, not the level and not even the gap, is what determines whether a spread is a precision instrument or a loaded gun.

The intuition: a calendar spread genuinely lowers your *per-unit* risk, but it lowers your margin by even more, so the discipline that keeps you alive is refusing to convert the margin saving into extra size — the lower risk is meant to be *kept*, not spent.

## The takeaway

Here is the whole post in one breath. A flat-price futures position bets on the *level* of a commodity — where the whole curve sits. A **calendar spread** bets on the *shape* — the gap between two delivery months — by going long one month and short another, so the level cancels and only the change in the gap moves your P&L. That cancellation is mechanical, not hopeful: a parallel move in the level affects both legs equally and nets to zero, leaving you exposed only to the one relationship you actually have a view on.

This buys you two things. First, genuinely **lower risk**: the legs offset, so a spread's daily volatility is a fraction of an outright's. Second, **lower margin**: the exchange's risk engine sees the offset and charges spread margin that can be a tenth of two separate outright margins. Together they make a spread a precise, capital-efficient way to express a view like *"this market is about to get tight,"* without taking on the flat-price risk you have no edge on. A **bull spread** (long front) expresses near-term tightening toward backwardation; a **bear spread** (long back) expresses a coming glut and widening contango.

And here is the insight to carry out the door, the one that ties back to this series' spine. A commodity price is *a physical thing forced through a financial contract*, and the curve's shape — contango or backwardation — is the market's running readout of the physical balance: glut or scarcity, storage full or storage tight. A calendar spread is the purest financial expression of a *physical* view. When you trade the shape, you are betting directly on the gears — storage, convenience yield, the prompt's tightness — that this whole series is about, and you are deliberately throwing away the part of the bet (the level) that those gears don't cleanly control.

But never let "limited risk" lull you. The spread removes the level risk you didn't want; it does *nothing* to the gap risk you bought on purpose, and the gap can dislocate violently when the physical reality breaks — tanks running out, a delivery squeeze, a negative print. Amaranth's spread cancelled the level exactly as designed and still vaporized \$6.5 billion, because the risk was never the level and the size was sized for a calm gap. The low margin that makes spreads attractive is the same low margin that tempts the size that kills you. Trade the shape, keep the risk you lowered *lowered*, and you have one of the most elegant tools in commodities. Spend the margin saving on size, and you have rebuilt the gun that has shot so many before you.

## Further reading & cross-links

- [Contango vs backwardation: what the shape of the curve means](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) — the two shapes a calendar spread trades, in depth.
- [Roll yield and why long-only commodity ETFs bleed](/blog/trading/commodities/roll-yield-and-why-long-only-commodity-etfs-bleed) — the carry the front leg of a spread collects or pays, and why it compounds.
- [Refining and crack spreads: turning crude into products](/blog/trading/commodities/refining-and-crack-spreads-turning-crude-into-products) — the inter-commodity cousin: long crude, short products.
- [Crude oil: WTI vs Brent, the world's two benchmark barrels](/blog/trading/commodities/crude-oil-wti-vs-brent-the-worlds-two-benchmark-barrels) — the WTI-Brent spread, an inter-commodity relationship trade.
- [The term structure of volatility: contango, backwardation, and the VIX curve](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve) — the same curve-shape thinking applied to volatility, where calendar spreads also live.
- [Gold futures, COMEX, contango, backwardation, and paper vs physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) — how curve shape and spreads work in the monetary metal, and how it differs from an industrial commodity.
