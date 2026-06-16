---
title: "Analyzing Perp DEXs and On-Chain Derivatives: Open Interest, Funding, and Liquidations"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "How to read on-chain perpetual-futures data — funding rates, open interest, long/short skew, and liquidation levels — as a positioning gauge and a cascade early-warning, using Hyperliquid, Coinglass, and Dune."
tags: ["onchain", "crypto", "perpetuals", "derivatives", "funding-rate", "open-interest", "liquidations", "hyperliquid", "dydx", "gmx"]
category: "trading"
subcategory: "Onchain Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — On-chain perpetual-futures DEXs put leverage data that used to live inside exchange servers onto a public ledger: open interest, funding rates, long/short skew, and liquidation levels are all readable, and together they form a positioning map and a cascade early-warning.
>
> - A **perpetual future** (perp) is a leveraged long or short with no expiry date; the **funding rate** is the mechanism that keeps its price glued to spot — when the perp trades above spot, longs pay shorts, and vice versa.
> - Read **funding** as a contrarian crowd gauge, **open interest** as the leverage in the system, **skew** as who is crowded, and **liquidation levels** as the fuel for the next violent move.
> - What you DO with it: when funding is extreme-positive and open interest is climbing into a thin liquidity zone, the crowd is long and fragile — fade it or wait for the flush, don't chase. The data is a sentiment edge, not a crystal ball.
> - The number to remember: a perp paying **+0.1% funding every 8 hours** costs a long roughly **0.3% per day** to hold — about **110% annualized**. Nobody pays that for long without a reason, and the reason is usually a crowd that's about to get squeezed.

On 26 March 2024, a single wallet on Hyperliquid — a fully on-chain perpetuals exchange — opened a long position in PEPE that the entire market could watch in real time. Not "watch" in the way you watch a centralized exchange, where you see an anonymous green candle and guess who's behind it. On Hyperliquid the order book and every open position settle on-chain, so traders could see the exact size, the exact entry, the unrealized profit ticking up, and — most importantly — the precise price at which that position would be force-closed. The wallet became a spectator sport. People built dashboards just to track it. When it was finally liquidated months later in a different position, the news traveled before the price even finished moving, because everyone already knew the liquidation price was sitting right there, on a public ledger, waiting.

That is the thing that makes on-chain derivatives different. On a centralized exchange like Binance or a traditional futures venue like CME, the positions are private. You get aggregate statistics — total open interest, an average funding rate — published by the exchange, and you trust that the exchange is reporting them honestly. On an on-chain perp DEX, the positions themselves are on the chain. The leverage is public. The skew is public. The liquidation map is, with a bit of work, reconstructable from public data. For the first time in derivatives history, you can read the actual book instead of a press release about the book.

This post teaches you to read that data the way a trader reads it: not as a prediction machine, but as a positioning map. We'll build every concept from zero — what a perp even is, why funding exists, what open interest measures — and then go deep on how to combine the four public dials into a single read, and how to use that read to spot when the market is crowded, fragile, and one shove away from a liquidation cascade.

![Perp dashboard mental model with funding, open interest, skew, and liquidation levels feeding a positioning map](/imgs/blogs/analyzing-perp-dexs-and-onchain-derivatives-1.png)

## Foundations: what a perpetual future actually is

Start with a normal future, because a perp is a mutation of it. A **future** is a contract to buy or sell an asset at a set price on a set future date. If you buy a one-month Bitcoin future at \$60,000, you've agreed to pay \$60,000 for one BTC in a month, regardless of where the price goes. Futures are how miners, institutions, and speculators take a position on a future price without holding the underlying asset today. They expire — that's the defining feature. On the expiry date, the contract settles to the spot price and ceases to exist.

A **perpetual future**, or **perp**, removes the expiry. It's a future contract that never settles to a date — you can hold it forever (hence "perpetual"). That sounds like a small change, but it creates a problem and the solution to that problem is the single most important concept in this entire post.

The problem: a normal future is anchored to spot by its expiry. As the date approaches, arbitrageurs force the future's price to converge with spot, because at expiry they're the same thing. Remove the expiry and you remove the anchor. A perp could, in principle, drift arbitrarily far from the spot price — perpetual contracts trading at \$65,000 while spot BTC is \$60,000, with nothing forcing them back together. That would make the perp useless as a way to bet on the spot price.

The solution is the **funding rate**, and we'll spend the whole next section on it because it's the lever everything else hangs on.

But first, two more foundational terms.

**Leverage.** A perp lets you control a large position with a small amount of collateral. If you post \$1,000 of collateral and take 10x leverage, you control a \$10,000 position. A 1% move in the underlying becomes a 10% move in your equity. This is the appeal and the danger: leverage amplifies both the gain and the loss, and it's why perp positions can be wiped out by moves that a spot holder would barely notice.

**Long and short.** Going **long** means you profit if the price rises. Going **short** means you profit if it falls — you're effectively borrowing the asset, selling it, and hoping to buy it back cheaper. On a perp, both are equally easy: you just open a position in either direction. This symmetry matters because at any moment the market has a population of longs and a population of shorts, and the balance between them — the **skew** — is one of the dials we'll learn to read.

### Margin, collateral, and the liquidation price

Two more mechanics, because they're what make liquidations *computable*, which is the whole basis for liquidation heatmaps later.

**Initial margin** is the collateral you must post to open a position. At 10x leverage, the initial margin is 10% of the position notional — \$1,000 of collateral backs a \$10,000 position. **Maintenance margin** is the smaller amount you must keep to *avoid* liquidation, often something like 0.5%–5% of notional depending on the asset and the size tier. When your equity (collateral plus or minus unrealized profit) falls to the maintenance margin, the position is liquidated.

The key consequence is that the **liquidation price is deterministic** — a fixed function of your entry price, your leverage, and your collateral. For an isolated-margin long, the liquidation price sits roughly `entry × (1 − 1/leverage + maintenance%)` below entry. A 10x long entered at \$60,000 liquidates near \$54,000 (a ~10% drop, nudged a little higher by the maintenance buffer). A 20x long entered at the same \$60,000 liquidates near \$57,000 (a ~5% drop). A 50x long liquidates on roughly a 2% move.

This is *the* fact that on-chain transparency exposes. Because the venue stores your entry, leverage, and collateral on-chain, anyone can compute your liquidation price. Multiply that across every open position and you get a map of where the forced selling and buying sits — the liquidation heatmap we'll build toward.

**Cross vs isolated margin** is a wrinkle worth naming. With **isolated** margin, each position has its own collateral and its own liquidation price — clean to read. With **cross** margin, all your positions share one collateral pool, so a position's effective liquidation price depends on your *whole book*, which is harder to read from outside even on-chain. Most heatmap estimates assume isolated-style liquidation prices; treat them as approximations for the cross-margin crowd.

#### Worked example: the same trade at 10x versus 50x

Take two traders, each posting **\$10,000 of collateral** on a BTC long entered at **\$60,000**. Trader A uses **10x** — a **\$100,000** position that liquidates near **\$54,000** (needs a ~10% drop). Trader B uses **50x** — a **\$500,000** position that liquidates near **\$58,800** (a ~2% drop wipes the \$10,000).

Now BTC dips a mild **2.5%** to **\$58,500**. Trader A is down **\$2,500** on paper but very much alive. Trader B is *liquidated* — the position is force-sold, the entire **\$10,000** collateral gone, and \$500,000 of notional just hit the market as a forced sell.

The intuition: the same \$10,000 and the same \$60,000 entry produce wildly different fragility depending only on leverage, and the high-leverage trader is the kindling — they're the first to liquidate, and they're sitting right next to spot.

### Why on-chain perps are special

You can trade perps on a centralized exchange (CEX) — Binance, Bybit, OKX all run enormous perp markets, far bigger than the on-chain ones. So why do on-chain perp DEXs matter for analysis?

Because on a CEX, the data you get is *curated by the exchange*. The exchange knows every position, but it shows you only aggregates: total open interest, an estimated funding rate, sometimes a liquidation feed (which several exchanges have been caught underreporting). You're reading a summary written by the house.

On an on-chain perp DEX — Hyperliquid, dYdX, GMX, and others — the matching and settlement happen on a blockchain. The consequences:

- **Hyperliquid** runs a fully on-chain central limit order book on its own L1. Orders, fills, and open positions are on-chain state. You can query the actual size of the actual positions, including very large individual ones, in close to real time.
- **GMX** uses a shared liquidity pool model: traders take positions against a pool of liquidity providers, and every position, its collateral, and its liquidation price are stored in contracts you can read directly on Arbitrum or Avalanche.
- **dYdX** (v4) runs as its own Cosmos app-chain with an on-chain order book, again exposing the book and positions as chain state.

The practical upshot: the leverage in the system is no longer a number the exchange chooses to publish. It's a number you can compute. That's the edge this post is about — and, as always in this series, also the noise trap, because public data invites manipulation, fake positions used as bait, and over-reading of a single whale's moves.

The three architectures differ in ways that change *what* you can read, so it's worth being precise:

- **Order-book DEXs (Hyperliquid, dYdX v4).** These rebuild the central-limit-order-book model on-chain or on a dedicated app-chain. Longs are matched against shorts directly, just like a CEX, so the open interest is genuinely two-sided trader-versus-trader. Funding is the usual longs-pay-shorts mechanism. Because the book and positions are chain state, you can read the actual bids, asks, and the largest open positions. Hyperliquid's design — its own L1 with the book in consensus — is why the famous large-trader-wallet watching happens there: the positions are first-class on-chain objects.
- **Pool-based / oracle-priced DEXs (GMX, early perps).** Here traders don't trade against each other — they trade against a shared liquidity pool of LPs, and the price comes from an oracle rather than an internal order book. There's no order book to read, but every position, its collateral, and its liquidation price live in contracts you can query directly. The risk profile is different too: the LP pool is effectively the counterparty to all traders, so when traders win big, the pool bleeds, and the protocol uses *borrow fees* (a funding-like cost) to balance open interest between longs and shorts.
- **Hybrid / vault-backed designs.** Several venues mix the two — an order book for matching with a backstop vault that absorbs liquidations. Hyperliquid's HLP vault, for instance, can act as a liquidity backstop. The detail matters when you read liquidation data, because a liquidation absorbed by a vault behaves differently in the market than one dumped into a thin order book.

For reading *positioning*, order-book venues like Hyperliquid give you the richest data — actual two-sided OI, real funding, and visible individual positions. Pool-based venues give you clean per-position liquidation prices but a funding-substitute (borrow fees) and no order book. Match the venue to the question.

## The funding rate: the mechanism that pegs a perp to spot

The funding rate is a periodic payment exchanged directly between longs and shorts. It is *not* a fee paid to the exchange (though exchanges take trading fees separately). It's a transfer from one side of the market to the other, and its sole job is to keep the perp price tethered to the spot price.

Here's the mechanism. The exchange compares the perp's price to a spot index price, typically every 8 hours (some venues do it hourly, Hyperliquid does it hourly).

- If the perp trades **above** spot (a premium), funding is **positive**: **longs pay shorts**.
- If the perp trades **below** spot (a discount), funding is **negative**: **shorts pay longs**.

Why does this peg the price? Follow the incentive. When the perp is above spot, it means more aggressive buying pressure on the perp than on spot — too many people want to be long. Positive funding makes being long *cost money every 8 hours*. That cost discourages new longs and pressures existing longs to close, which sells the perp down toward spot. Simultaneously, it pays shorts to exist, attracting arbitrageurs who short the perp and buy spot to capture the funding risk-free. Both forces drag the perp back to spot.

![Funding mechanism showing perp premium, longs paying shorts, and the perp re-pegging to spot](/imgs/blogs/analyzing-perp-dexs-and-onchain-derivatives-2.png)

The figure above shows the loop. Spot BTC sits at \$60,000. The perp is bid up to \$60,300 — a 0.5% premium, the crowd is long. Funding goes positive, say +0.05% per 8 hours. Now longs are bleeding that 0.05% three times a day to the shorts, the cost of leverage makes the marginal long close, and an arbitrageur shorts the perp while buying spot to harvest the funding. The selling pressure on the perp closes the gap and it re-pegs near \$60,000. The funding rate is the rubber band.

**How the rate is actually computed.** You don't need the exact formula to trade, but knowing the shape protects you from misreading it. Most venues set funding from two components: a **premium index** (how far the perp's price sits above or below the spot index, averaged over the period) plus a small fixed **interest-rate component** (often a token 0.01% per 8h that nudges funding slightly positive by default, reflecting the cost of holding the quote currency). The premium component dominates in any market that's actually moving. Venues also **clamp** funding to a maximum per period — when a market is wildly one-sided, funding pins to that cap, which is itself a signal: a pinned-cap funding rate means the rubber band is stretched as far as the venue allows.

The period matters when you compare venues. A "+0.01%" reading means very different things at an 8-hour cadence versus an hourly one. Hyperliquid pays funding **hourly**; most CEXs pay every **8 hours**. Always normalize to the same window before comparing — a useful habit is to convert everything to an annualized figure (multiply the 8h rate by 3 × 365, or the hourly rate by 24 × 365) so you're comparing like with like.

#### Worked example: what positive funding costs a long

Say the BTC perp on a venue has **\$2 billion of open interest** (we'll define open interest precisely in a moment — for now, it's the total size of all open positions). Funding is running hot at **+0.1% every 8 hours**, which means the long side as a whole is paying the short side 0.1% of notional, three times a day.

That's 0.1% × 3 = **0.3% of \$2B per day = \$6 million per day** flowing from longs to shorts. Wait — that overstates it, because funding is paid on the *net* skew, not the gross open interest. If longs and shorts are balanced, the payment nets out within the population. Take the realistic case where the book is skewed, say **\$2 billion of longs versus \$1.6 billion of shorts**, a net long imbalance of \$400 million. The net funding transfer is roughly 0.3% × \$400M ≈ **\$1.2 million per day** out of the crowded longs' pockets. Round numbers, the long side is paying north of **\$1 million a day** for the privilege of staying long.

The intuition: when funding is this hot, the crowd is paying real money — millions a day — to hold a leveraged long, and crowds rarely pay that premium right before a rally; they pay it right before they get squeezed.

### Funding as a contrarian sentiment gauge

This is the first genuinely tradeable read. Funding is a thermometer for crowd positioning.

- **Strongly positive funding** = the perp trades at a premium = the crowd is aggressively long and *paying* to stay long. This is "overheated." It tells you longs are crowded, leverage is one-sided, and the market is vulnerable to a long squeeze — a downdraft that liquidates the leveraged longs.
- **Strongly negative funding** = the crowd is aggressively short and paying to stay short. This is the mirror image: shorts are crowded, and the market is vulnerable to a short squeeze — a violent upward move that liquidates the shorts.

The contrarian logic: extreme funding marks extreme crowding, and extreme crowding is fragile. It doesn't tell you the *timing* — funding can stay hot for weeks during a strong trend — but it tells you the *fuel load*. The most violent moves in crypto happen when funding is at an extreme and then reverses.

![Funding rate oscillating around zero with crowded-long and crowded-short extreme zones shaded](/imgs/blogs/analyzing-perp-dexs-and-onchain-derivatives-4.png)

The chart above is an illustrative shape, not a dated series — the point is the *pattern*, not specific dates. Funding oscillates around zero. The shaded red band at the top marks the "crowded longs, overheated, fade" zone; the shaded green band at the bottom marks the "crowded shorts, oversold, squeeze fuel" zone. Notice that the most extreme funding always coincides with the points where the crowd is most one-sided — and that's exactly where the subsequent reversal tends to be sharpest. You read it as a fade signal at the extremes and as noise in the middle.

#### Worked example: funding flipping negative before a short squeeze

Suppose a token has sold off hard and the crowd has piled into shorts. The perp now trades at a **0.4% discount to spot**, and funding has flipped to **−0.08% per 8 hours** — shorts are paying longs to keep their bearish bets open. Open interest has *grown* during the selloff to **\$800 million**, mostly fresh shorts. Skew is heavily short.

The setup: shorts are now *paying* ~0.24% per day to stay short, the book is one-sided, and a wall of short liquidations sits just above the current price. If the price ticks up even 3–4%, those shorts start getting force-closed — and a short liquidation is a forced *buy*, which pushes the price higher, into the next cluster of shorts. A \$50 million wave of short liquidations buying at market into thin liquidity can move price several percent in minutes.

The intuition: negative funding plus rising open interest plus a crowded short book is a loaded spring — the shorts have already done the work of building the fuel pile, and any upside catalyst lights it.

## Open interest: the leverage in the system

**Open interest (OI)** is the total notional value of all open perp positions — every long and every short that hasn't been closed yet. Crucially, it's *not* volume. Volume counts trades (a contract bought and sold in the same hour is counted twice). Open interest counts *positions still open*. If I open a \$10,000 long and you open the \$10,000 short on the other side, open interest rises by \$10,000. When either of us closes, it falls.

So OI is a direct measure of how much leveraged exposure is currently live in the market. Rising OI means new positions are being opened — fresh money, fresh leverage. Falling OI means positions are being closed or liquidated — leverage leaving the system.

The mistake beginners make is reading OI in isolation. A big OI number isn't bullish or bearish by itself. The signal is in **OI direction combined with price direction**.

![Matrix of open interest direction against price direction showing four regimes from real trend to capitulation](/imgs/blogs/analyzing-perp-dexs-and-onchain-derivatives-3.png)

The matrix above lays out the regimes. Read it as OI on the rows, price on the columns:

- **OI rising + price rising** = a real uptrend. New longs are opening *and* pushing price up. The move is funded by genuine new positioning, not just short covering. This is the healthiest version of a rally — but watch funding, because if it's also overheating, the new longs are leveraged and fragile.
- **OI rising + price flat** = building tension. Leverage is stacking up but price isn't going anywhere. Both sides are adding positions and neither is winning. This is a coiled spring — when it breaks, it breaks hard, because there's a large leveraged book to liquidate in whichever direction loses.
- **OI falling + price rising** = a short squeeze. Price is rising *because* shorts are being closed (forced or voluntary), not because of fresh longs. These rallies are often sharp but unsustainable — once the shorts are gone, the buying that was driving the move disappears.
- **OI falling + price falling** = an OI flush / capitulation. Longs are being liquidated, leverage is purging out of the system. This is painful while it happens but it's also how a market resets — after a flush, the leverage is gone and the next move starts from a cleaner base.

#### Worked example: rising OI with flat price, then the break

A perp market on a mid-cap token sits at **\$50 per token** for two weeks. Open interest climbs from **\$120 million to \$300 million** over those two weeks — a \$180 million build — while the price barely moves. Funding is mildly positive, around +0.02% per 8h. Both longs and shorts keep adding; nobody's winning.

This is the "building tension" cell. The market is now carrying \$300 million of leveraged positions with no directional resolution. When the price finally breaks — say it drops to \$47, a 6% move — the longs who entered with 10x leverage near \$50 are underwater enough to start hitting liquidations. If \$80 million of those longs liquidate, that's \$80 million of forced market-selling hitting an already-falling market.

The intuition: a quiet price with a swelling OI isn't calm — it's a larger and larger bet stacked on a coin flip, and the bigger the OI, the more violent the resolution when it finally comes.

## Long/short skew and the pain trade

**Skew** is the imbalance between longs and shorts. If \$2 billion is long and \$1.2 billion is short, the market is **long-skewed**. Some venues publish a long/short ratio directly; on an on-chain venue you can often compute it from positions.

Skew matters because of the **pain trade** — the idea that markets tend to move in the direction that hurts the most participants, because that's where the leverage (and therefore the forced-buying or forced-selling fuel) is concentrated. When the book is heavily long, the pain trade is *down*: a drop liquidates the crowded longs, and their forced selling extends the drop. When the book is heavily short, the pain trade is *up*.

Skew and funding are two views of the same thing — crowding — but they're not identical. Funding tells you who's *paying* (the urgency of the crowd), skew tells you who's *positioned* (the size of the crowd). You want both pointing the same way to have conviction. A market that's long-skewed *and* paying high positive funding is doubly crowded long. That's the cleanest fade setup.

There's a critical subtlety with on-chain skew specifically: a few very large wallets can dominate the number. On a CEX, the skew is an aggregate of millions of accounts. On a smaller on-chain venue, one \$50 million position can tilt the whole skew reading. So on-chain skew is *more granular* (you can see the individual positions) but also *more noise-prone* (one whale isn't a "crowd"). We'll come back to this when we talk about the on-chain advantage and its traps.

A further trap with the published "long/short ratio": there are two different numbers and people confuse them. The **account ratio** counts how many *accounts* are long versus short — useful for gauging retail sentiment, since most accounts are small. The **position ratio** weights by *notional size* — useful for gauging where the actual money is. They can disagree sharply: 90% of accounts can be long while 60% of the notional is short, because a handful of large shorts outweigh a crowd of small longs. When you read a "70% long" stat, always ask *70% of what* — accounts or dollars. For a squeeze read, the dollar-weighted position ratio is what matters, because liquidations are sized in dollars, not accounts.

#### Worked example: account ratio versus dollar skew

A perp market reports **80% of accounts are long**. That sounds like a crowded-long fade. But drill into the notional: 8,000 retail longs averaging **\$2,000 each** = **\$16 million** of longs, while 200 larger accounts are short, averaging **\$150,000 each** = **\$30 million** of shorts. By *dollars*, the book is actually **65% short** despite being 80%-long by account count.

The intuition: a squeeze is driven by the dollars that get force-closed, not by how many people are on each side — here the real fuel pile is the \$30M of shorts, so the pain trade is *up*, the exact opposite of what the headline account ratio suggested.

## Liquidations: the cascade mechanism

A **liquidation** is what happens when a leveraged position runs out of collateral. Recall the 10x long: \$1,000 of collateral controls a \$10,000 position. If the price falls 10%, the position loses \$1,000 — the entire collateral — and the exchange force-closes it to avoid the position going negative. (In practice liquidation triggers a bit before the collateral is fully gone, to leave a buffer.) Higher leverage means a smaller move wipes you out: a 20x long liquidates on a ~5% adverse move, a 50x long on a ~2% move.

A liquidation is a **forced market order**. The exchange doesn't politely ask the market for a good price — it sells (for a long being liquidated) or buys (for a short being liquidated) immediately, at whatever the market will bear. This is the seed of the cascade.

![Pipeline of a long-squeeze cascade from a price dip through forced selling to a flush and bounce](/imgs/blogs/analyzing-perp-dexs-and-onchain-derivatives-5.png)

The pipeline above traces a **long-squeeze cascade**:

1. Price dips a couple of percent into a cluster of long liquidation prices.
2. Those longs get liquidated — the exchange force-sells them at market.
3. That forced selling pushes price *lower*, into the next cluster of liquidation prices below.
4. Which liquidates those, which sells more, which pushes lower still — a self-feeding waterfall. Hundreds of millions in longs can evaporate in under an hour.
5. Eventually the leverage is exhausted, the flush ends, and OI collapses as positions are gone.
6. With the forced sellers cleared out and no leverage left to liquidate, price often snaps back violently — the V-shaped reversal that traps anyone who panic-sold at the bottom.

A **short squeeze** is the same machine in reverse: price rises into short liquidations, which are forced *buys*, which push price higher into the next shorts. The asymmetry to remember: short squeezes can be more violent on the upside because there's theoretically no ceiling on price, whereas a long cascade is bounded by price reaching zero.

#### Worked example: a \$300M long flush in one hour

A token is trading at **\$2.00** with **\$1.5 billion of open interest**, heavily long-skewed (say **\$900 million long vs \$600 million short**) and funding running hot at +0.09% per 8h — textbook overheated. A macro headline drops and BTC sells off, dragging the token to **\$1.90**, a 5% dip.

That 5% move is enough to liquidate the 20x longs. Suppose **\$120 million of longs** liquidate in the first cluster. The exchange force-sells \$120M at market into a falling tape, which pushes the token to **\$1.82**. That 9% total drop now catches the 10x longs — another **\$180 million** liquidated. Total: **\$300 million of longs force-closed in under an hour**, dragging price to **\$1.78** before the selling exhausts and it bounces back to \$1.90 within the day, having flushed the leverage.

The intuition: the move wasn't \$300 million of people *deciding* to sell — it was \$300 million of leverage being *forced* to sell, and the difference is everything: forced selling doesn't care about price, so it overshoots, then reverses.

### Liquidation heatmaps

Because each leveraged position has a known liquidation price (it's a function of entry, leverage, and collateral), and because on-chain venues expose positions, analysts build **liquidation heatmaps**: a chart of how much leverage would be liquidated at each price level. Bright bands are clusters of stops. These clusters act like *magnets* — price has a tendency to gravitate toward large liquidation clusters, because once it gets close, the cascade pulls it the rest of the way (and because, cynically, large players have an incentive to push price toward known liquidation pools to harvest them).

![Liquidation heatmap concept grid of leverage tier against price level showing a dense cluster just below spot](/imgs/blogs/analyzing-perp-dexs-and-onchain-derivatives-6.png)

The heatmap above is an illustrative grid, not live data — it teaches what you're looking for. The x-axis is the liquidation price level, the y-axis is leverage tier, and the color intensity is how much notional would liquidate there. The dashed line marks the current spot price at \$62k. Notice the dense dark cluster of long liquidations sitting *just below* spot, concentrated in the high-leverage (20x–50x) tiers — those positions liquidate on a small move. That cluster is the read: a 2–3% dip would tap it, and the cascade could carry price down through it. A trader sees that and either (a) avoids being long with tight leverage right above the cluster, or (b) waits for the flush and buys the overshoot.

Real heatmaps come from **Coinglass** (aggregate across CEXs and some DEXs, estimated from public OI and assumed leverage distributions) and from on-chain venues directly where the actual positions are queryable. Treat the CEX-aggregate ones as *estimates* — they assume a leverage distribution that may be wrong — and the on-chain ones as more literal, since they're built from real positions.

## The on-chain advantage: seeing the actual positions

Here is where on-chain perps genuinely change the game. On a CEX, you read aggregates. On Hyperliquid, you can read *positions* — including individual large ones — because the order book and open positions are on-chain state.

This created an entirely new spectator sport: large-trader-wallet watching. Because a big wallet's position, leverage, entry, and *liquidation price* are all public, the market can see exactly when a whale is offside and exactly where they'll be force-closed. This cuts both ways. Sometimes the crowd front-runs the whale's liquidation, pushing price toward the known liquidation level to trigger it. Sometimes the whale uses their visibility deliberately — a publicly visible large position can itself be a signal designed to influence others.

#### Worked example: a visible \$50M leveraged long and its liquidation price

Suppose a wallet opens a **\$50 million long** on a BTC perp at an entry of **\$60,000** with **5x leverage**, posting **\$10 million of collateral**. On a CEX this is invisible. On an on-chain venue, anyone can read it.

With 5x leverage, the position is liquidated when losses eat the collateral — roughly a **20% adverse move**, so the liquidation price is around **\$48,000** (in practice a bit higher due to the maintenance-margin buffer, call it **\$49,000**). Now the entire market knows: there's a \$50 million long that gets force-sold if BTC touches \$49,000. That \$49,000 level becomes a target. If price drifts down toward it, traders pile in short to push it the last bit and trigger the \$50M forced sale — which they then buy back cheaply.

The intuition: on-chain transparency turns a whale's liquidation price into a public coordinate, and a known liquidation level isn't a secret support — it's a bullseye the rest of the market aims at.

This is the deanonymizing logic of the whole series applied to leverage: the position isn't anonymous, it's pseudonymous, and its mechanical vulnerability (the liquidation price) is fully public. Visibility is an edge for the watcher and a liability for the watched.

### The traps in the transparent data

Transparency creates second-order games that you have to account for, or the edge becomes a trap.

**Bait positions.** A sophisticated player can open a large, visible position *as a signal* — a \$50M long meant to make others think a whale is bullish, drawing in followers, while the real position is hedged elsewhere or about to be flipped. On-chain you see the position; you don't see the intent. A visible position is a fact about exposure on that venue, not a sincere directional opinion. Treat a single large position as a *hypothesis to confirm* (does the wallet have a track record? is the position hedged on another venue you can also read?), not a signal to copy.

**Stop-hunting toward known levels.** Because liquidation prices are public, clusters become coordination points. Large players with the capital to push price the last few percent into a known cluster can deliberately trigger the cascade and buy the overshoot — a form of MEV-adjacent extraction. This is why a big liquidation cluster *just below* price is dangerous to sit on top of with high leverage: you're not just betting on direction, you're betting that nobody finds it worth their while to push you into your own stop.

**Split positions and masking.** A whale who *doesn't* want to be read can split one \$50M position across ten wallets of \$5M each, or use cross-margin to obscure their true liquidation price, or open offsetting positions on different venues. So the absence of a visible whale doesn't mean there isn't one — it may mean a careful one. The clustering and attribution techniques from elsewhere in this series apply directly: funding source, timing, and behavior link wallets that a single position view wouldn't.

**Reflexivity of the data itself.** Once enough traders watch the same liquidation heatmap, the map starts to move the market — everyone positions around the same clusters, which changes where the clusters form next. The signal is real, but it's not a static measurement of an indifferent system; it's a measurement of a system that's watching itself. That's a reason to use the data for *fragility and fuel-load* reads (which are structural and slow to game) rather than precise *timing* calls (which the crowd front-runs).

## Funding arbitrage and the basis

One more concept rounds out the picture, because it explains *why* funding behaves the way it does and gives a sense of who's on the other side of the crowded trade.

The **basis** is the gap between the perp price and the spot price. When the perp trades at a premium (positive funding), there's a risk-free-ish trade available: **short the perp and buy the equivalent spot**. You're now market-neutral — if BTC goes up, your spot gains offset your perp loss, and vice versa — but you *collect the positive funding* every 8 hours for holding the short perp leg. This is the **cash-and-carry** or **funding arbitrage** trade.

#### Worked example: harvesting positive funding cash-and-carry

Funding on a BTC perp is running at **+0.05% per 8 hours** = 0.15% per day = roughly **~55% annualized** (0.15% × 365, ignoring compounding). An arbitrageur deploys **\$1 million**: buys \$1M of spot BTC and shorts \$1M of the BTC perp. The position is delta-neutral — price moves wash out — but it collects **\$500 per 8 hours** in funding (0.05% of \$1M), or about **\$1,500 per day**, **~\$45,000 over a month** if funding holds (it won't hold that high for a month, but you get paid until it normalizes).

The intuition: funding is the market *paying* the arbitrageurs to do the job of pulling the perp back to spot — high funding is a recruiting bonus, and the more the crowd crowds, the bigger the bonus, which is exactly why extreme funding is self-correcting and why fading the crowd at the extreme is a structurally favored bet.

This also tells you who's typically short during a euphoric, high-funding rally: not bears betting on a crash, but **basis traders** harvesting funding while staying market-neutral. That's why a long squeeze can be so violent — a chunk of the "short" side is hedged arb that isn't actually betting against price and won't add to a rally, leaving the directional shorts thin.

## A dated cascade: the 19 May 2021 long flush

To see the whole machine fire at once, look at one of the most-studied liquidation events in crypto history. By mid-May 2021, the market was the textbook overheated setup: BTC had run from under \$30,000 to nearly \$65,000 in months, perp funding had been pinned strongly positive for weeks (the crowd was paying handsomely to stay long), and open interest across perp venues sat near record highs. The book was massively long-skewed and leveraged.

On 19 May 2021, BTC dropped hard intraday — from around \$43,000 toward the low \$30,000s, a move of roughly 30% at the lows. The mechanism was exactly the cascade pipeline: the initial drop tapped the nearest cluster of leveraged long liquidations, those forced sales pushed price into the next cluster, and the waterfall ran. Across exchanges, liquidations that day ran into the **billions of dollars** — one of the largest single-day liquidation events on record at the time, with the overwhelming majority being *longs* getting force-closed. Open interest collapsed as the leverage purged out of the system.

Read it through the four dials and the whole thing was legible *before* the drop, not just after:

- **Funding** had been extreme-positive for weeks — the crowd was paying to be long (overheated, fade-risk).
- **Open interest** was near record highs — a huge leveraged book to liquidate (large fuel load).
- **Skew** was heavily long — the pain trade was down.
- **Liquidation levels** clustered below as leverage stacked into the rally — the magnet was beneath price.

#### Worked example: sizing the 19 May fuel load

Suppose, on the eve of the drop, a BTC perp venue showed **\$12 billion of open interest**, skewed roughly **\$7 billion long vs \$5 billion short**, with funding pinned near its cap. The net long imbalance is **\$2 billion**. When price fell ~10% into the dense long-liquidation band, a conservative quarter of the leveraged longs — call it **\$1.75 billion** — got force-sold into a falling market within hours.

That \$1.75 billion of forced selling wasn't anyone choosing to exit; it was margin engines hitting bids regardless of price, which is precisely why the move overshot to the downside before snapping back. The intuition: the size of the flush was readable in advance from the OI and the skew — the bigger and more one-sided the leveraged book, the bigger the eventual forced unwind, and 19 May was a near-record book meeting its near-record flush.

This is the pattern that repeats — 2021, the 2022 deleveraging events, the 2024–2025 perp-era flushes on venues like Hyperliquid where you can now watch it position-by-position. The specifics change; the machine doesn't. Crowded, leveraged, one-sided book plus a catalyst equals a cascade toward the nearest liquidation cluster.

## How to read it: a walkthrough on Hyperliquid, Coinglass, and Dune

Enough theory. Here's a concrete pass through a real perp market using the actual tools, in the order you'd do it.

**Step 1 — Pull up the market on the venue (Hyperliquid).** Open the perp you care about. The trading interface shows you, right there: the current funding rate (and a countdown to the next funding), the open interest, and on Hyperliquid specifically, a feed of recent trades and — via the API or community dashboards — the largest open positions. Note the funding sign and magnitude first. Positive and large? The crowd is long. Negative and large? The crowd is short.

**Step 2 — Check funding history and cross-venue funding (Coinglass).** Coinglass aggregates funding rates across exchanges and DEXs. Look at two things: (a) how extreme is current funding versus its own recent history — is +0.05% normal for this asset or a 90th-percentile reading? and (b) is funding aligned across venues, or is one venue an outlier (which can signal a localized squeeze or a venue-specific quirk)? Extreme, broad-based positive funding is the strongest "crowded long" read.

**Step 3 — Read the open-interest trend.** On Coinglass or the venue, pull up OI over the last few days alongside price. Apply the regime matrix: is OI rising with price (real trend), rising with flat price (building tension), or falling with falling price (a flush in progress)? This tells you whether the funding crowding is *fresh* (rising OI) or *exhausting* (falling OI).

**Step 4 — Look at the liquidation map.** Coinglass publishes aggregate liquidation heatmaps; the on-chain venues let you (or a Dune dashboard) reconstruct clusters from real positions. Find the nearest large cluster relative to current price. That's your nearest cascade trigger. Note which side it's on: a big cluster *below* current price is long-liquidation fuel (downside risk); a big cluster *above* is short-liquidation fuel (upside risk / squeeze setup).

**Step 5 — On Hyperliquid, check the big positions.** Use a community dashboard or the API to see whether the OI and skew are driven by *many* positions or *one whale*. If a single \$50M+ position dominates the skew, the "crowd" reading is really a "one trader" reading — far less reliable as sentiment. If the skew is spread across thousands of positions, it's a genuine crowd signal.

**Step 6 — Query the raw data with Dune (for the rigorous read).** Hyperliquid, GMX, and dYdX all have public Dune dashboards and queryable tables. You can write SQL to compute OI, funding accrued, liquidation volume by hour, and skew directly from on-chain data rather than trusting a dashboard's summary. A sketch of the kind of query you'd run against a perp-trades table:

```sql
-- liquidation volume by hour for one perp market (illustrative schema)
select
  date_trunc('hour', block_time)  as hr,
  sum(case when is_liquidation then notional_usd else 0 end) as liq_usd,
  sum(case when side = 'long'  and is_liquidation then notional_usd else 0 end) as long_liq_usd,
  sum(case when side = 'short' and is_liquidation then notional_usd else 0 end) as short_liq_usd
from perp_trades
where market = 'BTC'
  and block_time > now() - interval '7' day
group by 1
order by 1;
```

The spikes in `long_liq_usd` are your cascades; line them up against the price chart and you'll see the forced-selling waterfalls directly in the data — the thing a CEX would only summarize for you, you computed yourself from the ledger.

**Step 7 — Synthesize into a positioning read.** Put the four dials together: funding sign and extremity (who's paying, how badly), OI direction (fresh or exhausting), skew (how one-sided, and is it real crowd or one whale), and the nearest liquidation cluster (where the fuel is). That synthesis is your map. We'll formalize it into a decision table in the playbook.

## Common misconceptions

**"High open interest is bullish."** No — OI is direction-agnostic. High OI just means a lot of leverage is in the system. Whether that's bullish or bearish depends entirely on price direction and funding. High OI with overheated funding into a thin liquidity zone is *dangerous*, not bullish — it's a large fragile book. The 2021 and 2024 tops both featured record OI right before record liquidation cascades.

**"Positive funding means price will go up."** It's the opposite tendency, if anything. Positive funding means the crowd is *already* long and *paying* to stay long — the bullish bet is already on the table and leveraged. Extreme positive funding is a contrarian *caution* flag, not a buy signal. (Mild positive funding in a healthy uptrend with rising OI is fine — it's the *extremes* that flip contrarian.)

**"On-chain perp data can't be faked, it's on the blockchain."** The data is *real* — the positions genuinely exist — but it can be *staged*. A large player can open a visible position specifically to influence others (a public \$50M long that's really a feint), or split a position across wallets to hide skew, or wash-trade to inflate volume (though OI is harder to fake than volume since it requires real collateral). On-chain doesn't mean honest-intent; it means honest-record. Read positions as *facts about exposure*, not as *sincere opinions*.

**"Liquidations are random spikes."** They're the opposite of random — they happen at *known, computable price levels*. The whole point of a liquidation heatmap is that the triggers are predictable. A cascade isn't an act of God; it's the mechanical consequence of price reaching a cluster of liquidation prices that were sitting there in plain sight.

**"Funding is just an exchange fee."** Funding is a transfer *between traders*, not a fee to the venue. The venue takes separate trading fees. Confusing the two leads people to underestimate the *contrarian information* in funding — it's not a cost the house imposes, it's the crowd revealing how badly it wants to be on one side.

## Risk: what the perp data can't tell you

Perps amplify everything — that's the first risk and it's structural. A spot trader who's wrong loses some money slowly. A leveraged trader who's wrong gets *forced out* at the worst possible price, and the very transparency that lets you read others' liquidation prices lets others read yours. Before you act on any of these signals, internalize what the data does *not* do.

**It does not give you timing.** Funding can stay pinned positive for weeks during a powerful trend; "overheated" markets can get more overheated. The positioning map tells you the *fuel load and the fragility*, not when the match gets struck. Fading an extreme too early is one of the most reliable ways to get liquidated in the direction you correctly predicted. If you trade on a crowding read, you must either be unleveraged enough to survive the trend continuing, or wait for a confirming catalyst (price actually rolling over, OI starting to fall) before sizing up.

**It does not tell you about off-venue and cross-venue exposure.** A wallet that looks heavily long on Hyperliquid may be perfectly hedged with a short on a CEX you can't see, or with spot, or with options. The on-chain view is one venue's slice. The "skew" you read on a single DEX is real for that DEX but may not represent the trader's *net* position at all.

**It does not account for the protocol's own risk.** Perp DEXs are smart contracts and chains, and they fail in ways a CEX doesn't: an oracle that feeds a wrong price (covered in the spoofing-and-oracle post) can trigger liquidations that *shouldn't* happen; a contract bug can drain the LP vault; a chain halt can freeze you in a losing position with no way to add margin. The clean public data assumes the machine underneath is working — sometimes it isn't.

**It is reflexive and gameable on short timeframes.** As covered above, once the crowd watches the same heatmap, the heatmap moves the market. Precise, short-horizon reads ("price will sweep this cluster in the next hour") are exactly the kind the crowd front-runs and large players hunt. The durable edge is in the slow, structural read — *how crowded and fragile is this market, and where is the fuel* — not in calling the next candle.

#### Worked example: getting the direction right and still being liquidated

You correctly read a market as overheated: funding pinned at +0.1% per 8h, long-skewed, OI at highs. You short **\$100,000** of the perp with **10x** leverage, posting **\$10,000**, expecting the long flush. But the trend has one more leg: price grinds **up 8%** over three days before it finally cracks. Your short is down **\$8,000** of your **\$10,000** at the highs — one more 2% tick up and you're liquidated, force-bought at the top, *adding fuel to the very squeeze you predicted*. Then it dumps 15% the next day, exactly as you called.

The intuition: you were right about the destination and still got carried out, because leverage charges you for the path, not just the outcome — the positioning read was correct, but the size and timing turned a good thesis into a liquidation.

## The playbook: what to do with it

Here's the if-then synthesis. The cross of funding and skew gives you a lean; OI tells you if it's fresh or exhausting; the liquidation map tells you where the risk is and sizes the trade.

![Decision matrix crossing funding sign and skew with the read, the action, and the invalidation](/imgs/blogs/analyzing-perp-dexs-and-onchain-derivatives-7.png)

The decision matrix above formalizes it:

- **Funding strongly positive + long-skewed (overheated longs).** The read: late, leveraged longs, squeeze risk to the downside. The action: don't chase longs — fade the move, hedge an existing long, or wait for a long-liquidation flush to buy the overshoot. The invalidation: if OI keeps rising *with* price *and* funding cools off, the trend is absorbing the leverage healthily — your fade is wrong, stand down.
- **Funding near zero + balanced book.** The read: the trend, if any, is driven by spot, not leverage — less fragile. The action: trade the spot trend normally; the perp data isn't flagging a crowding edge here. The invalidation: funding starts spiking one way as a crowd builds — re-assess.
- **Funding strongly negative + short-skewed (oversold, crowded shorts).** The read: squeeze risk to the *upside*; the shorts are the fuel. The action: lean long into the setup or wait for the short squeeze to fire and ride it. The invalidation: OI keeps rising *as price falls* — that's a genuine, conviction-backed downtrend with fresh shorts winning, not a crowded-short bottom; don't fight it.

The general rules of thumb to carry:

1. **Extreme funding + crowded skew + fresh OI = fade the crowd, or at least don't join it.** The market is loaded one-sided and fragile.
2. **A nearby liquidation cluster is a magnet, not a wall.** Don't treat a big cluster below you as support — treat it as a trap door. Position above it with tight leverage at your peril.
3. **A flush is an opportunity, not a catastrophe.** When OI collapses on a price crash and funding resets, the leverage is gone and the next move starts from a clean base. The V-reversal after a cascade is one of the most repeatable patterns in leveraged markets.
4. **One whale is not a crowd.** On a smaller on-chain venue, check whether the skew is thousands of positions or one \$50M wallet before treating it as sentiment.
5. **The data is a sentiment edge, not a crystal ball.** Funding can stay hot for weeks in a strong trend; crowded can get more crowded. The positioning map tells you the *fuel load and the fragility*, not the *timing*. Combine it with everything else — spot flows, macro, narrative — and size for the fact that leverage amplifies your mistakes as much as the market's.

The deepest point: perps put a layer of leverage *on top of* the spot market, and that leverage is both the source of the most violent moves and — uniquely, on-chain — the most readable. You're not predicting price. You're reading where the leverage is, how crowded and how fragile, and where it gets forced out. That read won't tell you what happens next. It will tell you what *can't* happen quietly — and in a leveraged market, the moves that aren't quiet are the ones that matter.

## Further reading & cross-links

- **[Following smart-money wallets](/blog/trading/onchain/following-smart-money-wallets)** — the whale-watching toolkit that pairs directly with reading a visible large perp position on Hyperliquid.
- **[Fake depth, spoofing, and oracle attacks](/blog/trading/onchain/fake-depth-spoofing-and-oracle-attacks)** — why a thin or manipulated price feed is so dangerous for leveraged markets, and how a manipulated oracle can trigger liquidations that shouldn't happen.
- **[Reading DEX liquidity and pools](/blog/trading/onchain/reading-dex-liquidity-and-pools)** — the spot-side liquidity that funding arbitrage and forced liquidations trade *into*; thin spot depth is what makes a cascade overshoot.
- **[Crypto mining, staking, and MEV](/blog/trading/crypto/crypto-mining-staking-and-mev)** — how block builders and searchers can position around predictable liquidations, an MEV angle on the cascade mechanism.
- **[DeFi protocols: Uniswap, Aave, MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao)** — the lending-and-collateral plumbing whose liquidations share the same forced-selling logic as perps.
- **[Centralized crypto exchanges: Binance, Coinbase](/blog/trading/crypto/centralized-crypto-exchanges-binance-coinbase)** — the CEX perp markets that dwarf the on-chain ones in size, and where the aggregate funding/OI data still has to be taken on trust.
