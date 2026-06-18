---
title: "Stop-Hunts, Liquidation Cascades, and the Predator"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "A from-scratch predator-prey game of clustered stops and forced liquidations: where stops pile up, why a larger player hunts them, how a leverage cascade self-reinforces, and how to stop being the prey."
tags: ["game-theory", "trading", "stop-hunt", "liquidation-cascade", "leverage", "market-microstructure", "risk-management", "crypto", "forced-selling"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A clustered stop or liquidation level is not just your exit; it is a *target* a larger player can aim the price at, because forced selling is price-insensitive supply that the hunter gets to buy cheap.
>
> - Stops and liquidation levels cluster at the same obvious places — below support, on round numbers, at recent lows, at the standard leverage levels — which concentrates forced supply at a handful of predictable prices.
> - The hunt is a predator-prey game with no pure equilibrium: the prey wants to do the opposite of the predator, so the only stable play is to *hide your stop often enough* that hunting stops paying.
> - In leveraged markets each forced sale knocks the price into the next cluster of liquidation levels, firing the next wave — a self-reinforcing cascade that only ends when forced supply runs out or patient bids absorb it.
> - The one rule: do not be forced and do not be obvious. Size so a normal wick cannot liquidate you, and put your stop where the crowd is not.

You are long a coin at \$100. There is a clean support line at \$100 that every chart-watcher can see, and you did the responsible thing: you placed a stop-loss one tick under it, at \$99.90, so you "get out if support breaks." For two hours nothing happens. Then, in ninety seconds, the price slices straight through \$100, spikes down to \$96 on a long ugly wick, your stop fills at \$96.20 in the chaos — and within five minutes the price is back above \$100 as if nothing happened. You took the maximum loss exactly at the bottom tick, and the move that hurt you reversed the instant you were gone.

That was not bad luck. That was a *stop-hunt*, and you were the prey. Someone with more size than you pushed the price the last little distance to where they knew the stops were resting, the stops fired as forced market-sell orders, and that wave of price-insensitive selling created a pocket of cheap supply that the hunter bought. Your stop did not protect you from a real breakdown; it *advertised your pain point* and handed it to the only player at the table who could profit from it.

This post builds that game from zero. We will define a stop, a liquidation, and what "clustering" means; we will show where forced orders pile up and why; we will build the **liquidation cascade** that turns one wave of forced selling into a chain reaction in leveraged markets; we will model the predator-vs-prey interaction as a formal game with a payoff matrix; and we will end on how *not* to be the prey. The chart below is the mental model for the whole post: price drifting to an obvious cluster, the sweep through it, and the predator buying the cheap supply the cascade leaves behind.

![Stylized price path drifting to a stop cluster below support, spiking down through it where stops fire, then snapping back as the predator buys](/imgs/blogs/stop-hunts-liquidation-cascades-and-the-predator-1.png)

This is, of course, educational — a description of a mechanism, not advice to trade any particular way. But it is one of the most important mechanisms in markets, because it is the place where the abstract idea that "every trade has a counterparty who wants you to be wrong" becomes a concrete, repeatable pattern you can learn to spot and avoid.

## Foundations: stops, liquidations, clustering, and the predator-prey game

Before we can play the game we have to define the pieces. Take them one at a time; none of them is complicated on its own, and the whole danger comes from how they interact.

**A stop-loss order** is a resting instruction you leave with the exchange: "if the price trades down to \$99.90, sell my position at market." It is a safety device — it caps your loss without you having to watch the screen. The crucial feature is what happens when it triggers: it becomes a *market order*, meaning it sells at whatever price is available right now, not at a price you choose. A stop is price-*insensitive*. Once it fires, it will take any bid, however low. That word — insensitive — is the entire vulnerability, so hold onto it.

**A liquidation** is the same idea, but forced on you by leverage. *Leverage* means trading with borrowed money: with 10x leverage you put up \$10 of your own money (the *margin*) and control a \$100 position. If the price moves against you enough that your \$10 of margin is nearly gone, the exchange does not call you to ask politely — it *force-closes* your position to protect the money it lent you. The price at which that happens is your **liquidation price**, and the closing order, like a stop, is a price-insensitive market order. The higher your leverage, the smaller the move that wipes out your margin, so the closer the liquidation price sits to your entry.

Here is the simple version of the arithmetic. For a long position with leverage $L$, entry price $E$, and a small *maintenance margin* requirement $m$ (the exchange's safety buffer, often around 0.5%), the liquidation price is approximately:

$$P_\text{liq} \approx E \cdot \left(1 - \frac{1}{L} + m\right)$$

The distance from your entry down to liquidation is roughly $\frac{1}{L} - m$. At 2x leverage that is about 49.5% — the price has to halve before you are forced out. At 50x it is about 1.5% — a routine intraday wiggle. We will come back to this; it is the fuel gauge of the whole cascade.

**Clustering** is the observation that thousands of independent traders, choosing their own stops and their own leverage, do not spread those orders evenly across all prices. They pile them up at the *same* few prices — the obvious ones. We will spend a whole section on why, but the headline is: because everyone looks at the same chart, reads the same round numbers, and uses the same standard leverage tiers, the forced orders concentrate. A *cluster* is a price level holding an unusually large amount of resting forced supply.

**The predator and the prey.** The prey is the crowd of ordinary traders whose stops and liquidations form the clusters. They are not coordinating; each one independently placed a "reasonable" order at an obvious level, and in doing so they collectively built a target. The *predator* is a larger player — a well-capitalized desk, a whale, sometimes a coordinated group — with two things the prey lacks: enough size to *move* the price the last distance to a cluster, and enough patience to *buy* the cheap supply the cluster releases. The predator does not forecast the future. It engineers a moment in the present.

And **the game** is this: a trade is a strategic interaction, not a bet against nature. The prey reveals its pain point by placing an obvious, visible order. The predator, reasoning one level deeper, asks "where are the stops, and is it worth pushing to them?" The whole rest of this post is working out the payoffs of that interaction and what it means for which side of it you want to be on.

One more building block makes the whole mechanism click: how a market sell order actually moves the price. A market sells by "eating the book" — it fills against the highest resting buy orders (bids) first, then the next-highest, and so on down the ladder, until the whole order is filled. *Depth* is just how much buying is resting at each price. If \$50M of bids sit between \$100 and \$99, a \$50M market sell drops the price one dollar; if only \$10M sits there, the same order drops it five dollars, because it has to reach down five rungs to find enough buyers. This is why a *thin book* — little resting depth — turns a modest forced sale into a large price drop, and it is the lever that makes cascades possible. A stop or liquidation is a market order; into a thick book it barely registers, into a thin one it gaps the price into the next cluster. Keep this model in mind: forced supply meeting thin depth is the entire engine.

> A note on ethics and framing, because this matters: this is a *defender's* guide. The point of understanding the hunt is so you can recognize it, stop feeding it, and stop being the cheap supply — not so you can run it. Pushing a price into a known cluster to trigger forced selling is, in many regulated venues, manipulation; in fragmented or thin markets it lives in a gray zone. We explain the mechanism the way a self-defense class explains a mugging: so the victim stops walking into it.

## Where stops and liquidations cluster, and why

The single most important fact about stop-hunting is that the prey chooses its own pain points, and it chooses them *predictably*. If stops were scattered uniformly across every possible price, there would be no cluster to hunt and no profit in pushing — you would trigger a trickle of selling everywhere and nothing concentrated anywhere. The hunt only works because the crowd parks its forced orders at the same obvious places. The grid below is the liquidity map every predator reads before deciding whether to push.

![Grid mapping the four places forced orders cluster, why the crowd picks each level, and the forced supply parked there](/imgs/blogs/stop-hunts-liquidation-cascades-and-the-predator-2.png)

Why those four places? This connects directly to two ideas from elsewhere in this series — *focal points* (the obvious answer everyone independently converges on when they are trying to match each other) and the chart-reading habits of *support and resistance*. The cluster forms wherever a focal point and a charting rule agree on the same price.

**Below support (and above resistance).** A *support* level is a price where the chart has bounced before, and the universal trading-book rule is "if support breaks, you were wrong, so exit." That means a wall of long stops sits just *under* support — not at it, but a tick or two below, because the trader wants to give support "a little room." Everyone gives it the same little room, so the stops stack in a tight band right beneath the line. The predator does not have to guess where; the chart drew the target for them.

**Round numbers.** Humans anchor on whole numbers. Bitcoin at \$50,000, a stock at \$100, gold at \$2,000 — these attract stops and limit orders far out of proportion to anything fundamental, purely because a round number is the easiest focal point in the world. "I'll get out if it loses \$100" is a sentence thousands of people say independently, and they all mean the same price.

**Recent lows and highs.** Yesterday's low, the week's low, the low of the last big candle — these are the natural "if it makes a new low, I'm wrong" levels. They are visible to everyone on every timeframe, so swing-low stops cluster just beneath the most recent obvious low.

**Leverage liquidation levels.** This is the cluster unique to leveraged markets, and it is the most mechanical of all. Exchanges compute liquidation prices by *formula* from your leverage. Because traders gravitate to round leverage tiers — 5x, 10x, 25x, 50x, 100x — and the formula maps each tier to a fixed *percentage* below entry, the liquidation levels of everyone who entered around the same price stack at the same handful of percentages. You do not even need to know individual traders' stops; you can read the leverage map directly off the standard tiers. This is why the densest, most reliable clusters in all of trading are in crypto perpetual futures, where retail leverage is high and the liquidation formula is public.

#### Worked example: how much money sits in a support cluster

Suppose a coin is trading at \$104 with clean support at \$100, and you want to estimate the forced supply waiting just below. Say \$2 billion of long positions opened on the recent bounce off \$100, and from surveys of where traders place stops, roughly 30% of them set a stop in the \$96–\$99.90 band (just under support). That is \$2,000M × 0.30 = \$600M of stop-driven sell orders parked in a \$4 band.

Now add the leverage. Of those longs, say \$400M is held at 20x or higher, with liquidation prices in that same band (recall 20x liquidates about 4.5% below entry — right at \$95.50 for a \$100 entry). Those are *forced* sells on top of the discretionary stops. So the cluster below \$100 holds on the order of \$600M + \$400M ≈ \$1B of price-insensitive supply waiting to fire if the price trades down four percent.

The intuition: the cluster is not a vague "support might break" feeling — it is a roughly knowable *dollar amount* of selling that the predator can trigger and then buy, and a knowable amount is a knowable target.

## Why forced selling is the predator's prize

We keep saying "price-insensitive," so let us make the predator's incentive precise. There are two kinds of selling in a market. *Discretionary* selling is someone who chooses to sell because they think the price is too high; they will sell at \$99 but maybe not at \$96, because at \$96 they would rather hold. *Forced* selling — a triggered stop or a liquidation — has no such floor. The seller is not choosing; the order is a market order that will hit any bid down to zero. It *must* transact, and it does not care about the price.

To a buyer, forced supply is the best supply in the world, because it is desperate. If you are the only large bid sitting under a cluster when it fires, you buy from sellers who have no choice and no patience, at prices far below where they would voluntarily sell. The predator's entire edge is being the *patient capital on the other side of forced impatience*. This is exactly the de-grossing logic from the [exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game): when a crowd is forced through the same narrow door at the same time, whoever is calm and liquid gets to set the price of their panic.

#### Worked example: the predator's profit from one sweep

Use the numbers from the cover figure. The price drifts to the \$96–\$99.50 stop cluster, the predator spends to push the last leg, and the stops fire into a thin book — the wick prints \$96. The predator has resting bids and buys, say, \$300M of the forced supply at an average of \$96.50. The cascade exhausts itself; bids absorb the rest; the price snaps back to \$100 within minutes.

The predator's profit on the snap-back alone: \$300M bought at \$96.50 and marked back at \$100 is a gain of $(100 - 96.50)/96.50 \approx 3.6\%$, or about \$10.9M, in minutes — before counting whatever it cost to push the price down there (the "ammunition," which we will price next). The prey's side of the same ledger: the long who stopped out at \$96.20, having entered at \$100, locked a 3.8% loss at the *exact tick* the move reversed.

The intuition: the predator's gain and the prey's loss are the same trade. The cheap supply the cascade created did not vanish — it changed hands, from the forced seller to the patient buyer.

The cost of pushing is the predator's risk. Moving the price the last \$1–\$4 to the cluster takes real buying-then-selling that loses a little to the spread and to slippage. The hunt is profitable only when the supply released by the cluster is worth more than the ammunition spent to trigger it. That is precisely why clusters have to be *dense and obvious*: a thin, scattered set of stops is not worth the push. The predator hunts the fat, focal clusters and ignores the rest — which, helpfully, tells you exactly where *not* to put your stop.

### How the predator reads the cluster before pushing

A rational predator does not push on a hunch; it pushes when the expected value of the hunt is positive, and it can estimate that EV surprisingly well from public information. Three signals tell it where the fat clusters are and how dense they are.

The first is the *chart itself*. Support, round numbers, and recent lows are visible to the predator on the same screen as everyone else — that is the whole point of a focal point. If a clean support line sits at \$100 with a long uptrend resting on it, the predator knows the stops are a tick or two beneath, because the trading-book rule that put them there is universal. The chart is a confession of where the crowd's pain lives.

The second, in leveraged markets, is the *liquidation map*. Several data services publish estimated liquidation levels by aggregating exchange open-interest and the standard leverage tiers. Because the liquidation price is a public formula of entry and leverage, you can reconstruct, with real accuracy, how much notional will be force-sold at each price band below the current price. A predator does not have to guess that "there are probably some stops down there" — it can read an estimate that says "\$420M of longs liquidate between \$95 and \$96." That is a targeting solution.

The third is *order-book and tape behavior*. Even hidden stops leak: as price approaches a cluster, you often see the book thin out just below the level (market-makers pull bids to avoid being run over by the coming forced flow) and you see the trade tape speed up. A predator watching the book can sense the air-pocket forming under the cluster and time the push to coincide with the thinnest depth.

#### Worked example: the predator's expected value of a hunt

Put the whole decision in dollars. The predator is considering a hunt on the \$100 cluster. From the liquidation map and the chart it estimates the cluster holds about \$600M of forced supply that, once it fires into a thin book, can be bought at an average of roughly 3.5% below the trigger. The push to get there costs ammunition — say it must buy and then unwind about \$120M of position to drag the price the last 4%, losing about 1.2% of that to spread and slippage, a cost of roughly \$1.4M.

The reward: capturing, say, half the released supply, \$300M, at a 3.5% discount, then marking it back as price recovers, is about \$300M × 3.5% = \$10.5M of upside. Net expected value ≈ \$10.5M − \$1.4M = \$9.1M, for a hunt that resolves in minutes. But weight it by the chance the prey *hid* — if there is a 30% chance the cluster is thinner than estimated and the push fizzles (a −\$1.4M outcome with no reward), the probability-weighted EV is $0.7 \times 9.1\text{M} + 0.3 \times (-1.4\text{M}) \approx \$6.0\text{M}$. Still strongly positive.

The intuition: the hunt is not gambling — it is a positive-EV operation the predator can size and price like any other trade, and the thing that flips its EV negative is *uncertainty about whether the cluster is really there*. That uncertainty is the prey's only defense, and it is the whole reason the equilibrium is mixed.

## Building the liquidation cascade from zero

Everything so far is a single wave: push to a cluster, stops fire, supply is bought. The *cascade* is what happens when one wave's selling is large enough to trigger the *next* cluster, which triggers the next — a chain reaction that is the signature failure mode of leveraged markets. Let us build it one link at a time.

![Six-stage pipeline of a liquidation cascade, from the initial trigger through successive forced-selling waves to exhaustion](/imgs/blogs/stop-hunts-liquidation-cascades-and-the-predator-3.png)

Start with the key fact from the foundations: liquidation levels cluster at the standard leverage tiers, which map to fixed percentages below entry. For a \$100 entry, the 20x longs liquidate around \$95.50, the 10x longs around \$90.50, the 5x longs around \$80.50. These are *stacked clusters* — a ladder of forced supply at known rungs.

Now introduce *market depth*: the amount of resting buy orders available to absorb selling per unit of price. Say the order book holds \$50M of bids for each \$1 of price (a thin book — in a calm, deep market it would be far more). Depth is what determines whether a wave stops or spreads. Walk the cascade:

```
Trigger:  price dips 4.5% to $95.50, the 20x liquidation band.
Wave 1:   $200M of 20x longs force-liquidate at market.
          $200M of forced sells / $50M of bids per $1 = a $4.0 drop.
          Price falls from $95.50 toward ~$92 ... but the path runs
          straight through the next cluster.

Wave 2:   the drop reaches $90.50, the 10x liquidation band.
          $500M of 10x longs force-liquidate.
          $500M / $50M per $1 = a $10.0 drop. Price -> ~$80.50.

Wave 3:   that lands on the 5x liquidation band at $80.50.
          $900M of 5x longs liquidate -> $900M / $50M = an $18 drop.
          Price -> ~$62, far below where any of this started.
```

Read what just happened. A 4.5% nudge — the kind of move that happens twice a week — set off \$1.6B of forced selling and dragged the price down nearly 40%, not because anyone's *view* changed but because each wave of price-insensitive selling reached down and pulled the trigger on the next cluster. This is a *self-reinforcing* feedback loop, the same structural shape as a bank run or a fire sale: the act of selling lowers the price, and the lower price forces more selling. The cascade only ends when one of two things happens — the ladder of clusters runs out (everyone leveraged is already liquidated), or *patient bids show up* in enough size to absorb a wave without dropping into the next cluster.

#### Worked example: when does a wave stop versus spread?

The cascade is a chain reaction with a *critical condition*, exactly like the difference between a fizzle and an explosion. A wave spreads to the next cluster if the price drop it causes is at least as large as the gap to that cluster. Drop caused = (forced notional) / (depth per \$1). Gap = distance to the next cluster.

Wave 1 caused a \$4 drop from \$95.50, and the gap to the next cluster (\$90.50) was \$5. Four is less than five — so with that depth, Wave 1 would have *fizzled* and the cascade would have stopped at \$91.50, one cluster in. But drop the depth to \$50M per \$1 and raise the 20x notional to \$250M, and the drop becomes \$5 — exactly enough to reach \$90.50 and ignite Wave 2. The cascade lives or dies on the ratio of *forced supply* to *book depth*.

The intuition: leverage builds the fuel (dense, close clusters) and thin liquidity removes the firebreaks (shallow depth between rungs). A cascade is not a price falling; it is a price *falling through its own safety levels because the selling at each level is larger than the buying between levels*.

This is why cascades cluster in time and in venue. They happen during low-liquidity windows (weekends, holidays, the dead of night in the dominant timezone), in the most-leveraged instruments (crypto perps above all), and after a long calm has let leverage build up — because a calm market is one where the clusters have grown fat and the depth has grown thin from complacency. If you want the chain-level mechanics of *how* a lending protocol's liquidations actually execute on-chain, that is the subject of [analyzing lending and liquidations](/blog/trading/onchain/analyzing-lending-and-liquidations); here we care about the game-theoretic shape, not the smart-contract plumbing.

### The cascade as a branching process: why it is all-or-nothing

There is a deeper reason cascades feel sudden and binary — quiet for weeks, then a 40% gap in an hour. A cascade is mathematically a *branching process*, the same family of model as an epidemic or a nuclear chain reaction. The key quantity is a *reproduction number*: how many new liquidations does each liquidated dollar trigger on average? Call it $R$. If each \$1 of forced selling drops the price enough to force *less* than \$1 of new liquidations ($R < 1$), the chain dies out quickly — a wave fires, fizzles, and the price stabilizes. If each \$1 forces *more* than \$1 of new liquidations ($R > 1$), the chain explodes — every wave is bigger than the last, and the price runs until it exhausts the leveraged supply entirely.

What sets $R$? The density of clusters (how much forced notional sits per dollar of price) divided by the depth (how much resting bid absorbs each dollar). Build leverage and $R$ rises; thin the book and $R$ rises. The dangerous feature of a branching process is that $R$ crosses 1 *invisibly*: nothing looks different at $R = 0.95$ versus $R = 1.05$ on a calm day, but the first is a non-event and the second is a catastrophe. That is why cascades are so hard to see coming and so violent when they arrive — the system sits just below criticality for a long time as leverage builds, then a small trigger tips $R$ past 1 and the whole stack goes at once.

#### Worked example: how much depth would have stopped the cascade

Return to the three-wave cascade. Wave 1 was \$200M of forced selling into \$50M of depth per \$1, causing a \$4 drop that reached toward the next cluster. How much extra depth would have killed the chain? To keep the drop *under* the \$5 gap to the \$90.50 cluster, you need depth such that $200\text{M} / d < 5$, i.e. $d > \$40\text{M}$ per \$1. The market had \$50M, which sounds sufficient — but the \$4 drop landed inside the *fat* part of the next cluster's range, and a fresh wave started before the price fully stabilized.

To stop it cleanly, a patient buyer would have needed to *add* depth right under the trigger: stepping in with, say, \$300M of bids stacked between \$95.50 and \$94 would have absorbed Wave 1 entirely within a \$1.50 band, never reaching \$90.50, and the cascade dies at wave one. That \$300M buyer, of course, is the predator — buying the cheap supply *is* what stops the cascade, which is why the patient capital that ends the fall is the same capital that profited from it.

The intuition: a cascade is stopped by depth, and the only player with both the size and the incentive to supply that depth at the bottom is the predator. The crowd's forced selling and the predator's patient buying are two halves of the same event — the cascade is the mechanism that transfers the cheap supply from the first to the second.

## Leverage is the dial that builds the fuel

The cascade's intensity is set almost entirely by one number: how much leverage the crowd is carrying. We can see why from the liquidation-distance arithmetic. The chart below plots how far the price has to fall to force-liquidate a long, for each leverage tier, computed straight from $P_\text{liq} = E(1 - 1/L + m)$.

![Bar chart of the percentage price fall needed to force-liquidate a long position at each leverage tier from 2x to 100x](/imgs/blogs/stop-hunts-liquidation-cascades-and-the-predator-4.png)

The shape is a cliff. At low leverage the liquidation level is *far* away — a 2x long survives a 49.5% crash, a 5x long survives 19.5%. Those positions are not cascade fuel; you cannot hunt them with a normal wick, and a small dip does not force them to sell. But the curve falls off a cliff at the high end: 20x liquidates 4.5% from entry, 50x at 1.5%, 100x at half a percent. A 100x long is liquidated by a move smaller than the typical bid-ask bounce. Those positions are *pure* cascade fuel — densely clustered (everyone uses the round leverage tiers), close to entry (so a tiny push reaches them), and forced (no discretion when they fire).

#### Worked example: how leverage turns a normal day into a liquidation

You go long at \$100 in three scenarios: 5x, 20x, and 50x. A perfectly ordinary 2% intraday dip to \$98 occurs — the kind of move that happens most days and reverses by the close.

- **5x:** liquidation at \$80.50. The \$98 print is a 2% paper loss on the position, magnified 5x to a 10% drawdown on your margin. Uncomfortable, but you are nowhere near forced out; you hold, and you recover.
- **20x:** liquidation at \$95.50. The \$98 print is a 2% move, magnified 20x to a 40% margin drawdown. You are sweating but alive — a push to \$95.50 (4.5%) ends you, and that is now within a single bad candle.
- **50x:** liquidation at \$98.50. The ordinary 2% dip *already liquidated you*. You did not get to hold and recover; the move that everyone else shrugged off force-sold your position at the bottom and you were the cheap supply.

The intuition: leverage does not just amplify your gains and losses, it *shortens the distance to forced selling* until a routine wiggle becomes a liquidation. The same \$2 dip is a non-event at 5x and a death at 50x — and the deaths are exactly the clustered, close, forced supply the predator and the cascade feed on.

This is the connection back to the spine of the series. Your edge is never just your forecast; it is knowing what game you are in and who is on the other side. A maxed-out leveraged long is not "bullish" — strategically, it is a *pre-committed forced seller* whose exit price is published in advance by the exchange's formula. You have told the whole market the exact price at which you will be made to sell, regardless of your opinion. That is the worst possible hand to show.

### Margin mode and auto-deleveraging: the cascade's hidden plumbing

Two more pieces of leverage plumbing decide how brutal a cascade gets, and they are worth defining because the prey usually does not know which one it has chosen. The first is *margin mode*. With **isolated margin**, only the collateral you assigned to a single position is at risk: if it liquidates, the rest of your account is untouched, but the liquidation point is fixed and close. With **cross margin**, your whole account balance backs the position, so the liquidation price is *further away* (more collateral to burn through) — but if it ever does liquidate, it can take your entire account with it, and a cross-margined trader who is also long several correlated coins can have *all* of them liquidate together in a cascade. Cross margin trades a closer-but-survivable single liquidation for a farther-but-catastrophic joint one.

The second is **auto-deleveraging** (ADL). When a cascade is so violent that the exchange cannot liquidate a position at a price that keeps the insurance fund solvent — the bankrupt position's losses exceed its margin and the book is too thin to absorb it — the exchange reaches in and force-closes the *winning* counterparties on the other side, at the bankruptcy price, to balance the books. If you were short and *right* during a long-liquidation cascade, ADL can rip you out of your winning trade at the worst moment of the move, exactly when it was about to pay the most. ADL is rare, but it means a cascade can punish the people who positioned *correctly* against the crowd, which is its own grim piece of game theory: being right is not enough if the venue itself becomes a forced participant.

#### Worked example: isolated versus cross liquidation distance

You hold a \$10,000 long at \$100 with \$1,000 of margin assigned (10x on that position) in an account that also holds \$4,000 of spare cash.

- **Isolated:** only the \$1,000 backs the trade. Liquidation sits about 9.5% below entry, near \$90.50 (from $P_\text{liq} = E(1 - 1/L + m)$). A drop to \$90 force-closes the position and you lose the \$1,000 — but your \$4,000 cash is safe.
- **Cross:** all \$5,000 backs the trade, so the *effective* leverage is 2x (\$10,000 position on \$5,000 collateral) and liquidation falls to roughly 49.5% below entry, near \$50.50. You survive far deeper drawdowns — but if the price ever does reach \$50.50, the liquidation consumes the whole \$5,000, not just \$1,000, and any other cross-margined longs go with it.

The intuition: cross margin buys you distance from liquidation at the cost of *correlation* — it converts many small, isolated deaths into one large, joint one, which is exactly the kind of synchronized forced selling a cascade feeds on.

## The predator-vs-prey game, formally

Now we can write the interaction as a game and solve it. There are two players. The **predator** chooses to HUNT (spend ammunition to push the price into the cluster) or PASS (leave it alone). The **prey** — really the representative crowd member — chooses to CLUSTER (place a tight, visible stop at the obvious level) or HIDE (use a wider, mental, or off-level stop the book cannot see). The payoffs, in abstract "risk units," come from `nash_2x2` in the series model.

![Two-by-two payoff matrix of the hunt-versus-hold game showing no pure equilibrium and a mixed-strategy solution](/imgs/blogs/stop-hunts-liquidation-cascades-and-the-predator-5.png)

Read the four cells:

- **Hunt vs Cluster** (predator +6, prey −5): the predator's dream. The stops are exactly where expected, the push triggers them, the cascade fires, and the predator buys the cheap supply. The prey takes the maximum loss at the worst price.
- **Hunt vs Hide** (predator −3, prey +2): the predator pushes, spends the ammunition... and nothing fires, because the prey's stop was below the cluster or invisible. The predator eats the cost of the push for no payoff; the prey survives the fake-out and often re-enters cheaper.
- **Pass vs Cluster** (predator 0, prey +1): the predator declines to hunt, the prey's tight stop is never triggered, and the tight stop was actually efficient this time (minimal slippage). Fine — but the prey got lucky that no one came.
- **Pass vs Hide** (predator 0, prey 0): nothing happens; the prey paid a tiny bit of extra slippage for a wide stop that was never needed.

#### Worked example: solving the game with `nash_2x2`

Feed the matrix to the model — row player (predator) payoffs $A = [[6, -3], [0, 0]]$, column player (prey) payoffs $B = [[-5, 2], [1, 0]]$ — and it reports **no pure equilibrium** and a **mixed equilibrium of (0.125, 0.333)**.

Why no pure equilibrium? Because the players want to *mismatch*. If the prey always clusters, the predator always hunts (+6 beats 0). But if the predator always hunts, the prey should always hide (+2 beats −5). And if the prey always hides, the predator should pass (0 beats −3). And if the predator passes, the prey can safely cluster again (+1 beats 0). The best responses chase each other in a circle — exactly the structure of [matching pennies](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once) or rock-paper-scissors. There is no resting point in pure strategies.

The mixed solution: the prey should **cluster with probability 0.125 and hide with probability 0.875** (the model's 0.333 is the predator's hunt frequency that makes the prey indifferent; the prey's own indifference, balancing the predator, lands at clustering only one time in eight). In words: *hide your stop seven times out of eight.* Cluster only rarely, so that the predator can never reliably assume your stop is at the obvious level, which makes the push unprofitable in expectation.

The intuition: there is no rule that makes you un-huntable, but there is a *frequency* of hiding that makes hunting you a money-loser. Unpredictability is the defense.

#### Worked example: the prey's expected value, hunt-rate by hunt-rate

Suppose the predator hunts with probability $h$. The prey can compute its expected value (using `expected_value`) for each choice:

- **Cluster:** $\text{EV} = h \cdot(-5) + (1-h)\cdot(+1) = 1 - 6h$.
- **Hide:** $\text{EV} = h \cdot(+2) + (1-h)\cdot(0) = 2h$.

Set them equal to find where the prey is indifferent: $1 - 6h = 2h \Rightarrow h = 1/8 = 0.125$. So if the predator hunts *more* than one time in eight, hiding is strictly better; if less, clustering is fine. Plug in a realistic hunt-rate of $h = 0.5$ in a thin, hunt-prone market: clustering returns $1 - 3 = -2$ and hiding returns $+1$. The gap is three full risk units — hiding is dramatically better whenever hunting is at all common.

The intuition: in any market liquid and leveraged enough that hunting happens even occasionally, the math says hide. The "place a tight stop at the obvious level" advice from beginner trading books is, in game-theoretic terms, choosing the dominated strategy in a market with predators.

## Common misconceptions

**"Stop-hunting is a conspiracy theory — markets are too big to manipulate."** Two errors in one. First, you do not need a conspiracy: the clusters are an *emergent* property of independent traders converging on the same focal points, and any single large player can choose to push into one. Second, "too big to manipulate" confuses the whole market with a *moment*. Nobody moves Bitcoin's trend, but pushing the price \$2 into a stop cluster at 3 a.m. on a thin weekend book is entirely feasible for a desk with size. The hunt is local and brief, not global and sustained.

**"My stop protects me, so a tight stop is the safe choice."** A stop protects you from a *real, sustained* adverse move. It does *not* protect you from a fake-out wick designed to trigger it, and a tight stop at the obvious level maximizes your exposure to exactly that. The safe choice is a stop placed where a hunt would not reach and sized so the position can survive the wick — which usually means a *wider* stop on a *smaller* position, not a tight stop on a big one.

**"I'll just use very high leverage with a tight stop — same risk, more profit."** This is the most expensive misconception in crypto. High leverage moves your *liquidation* price right up next to your entry, and the liquidation fires before — or instead of — your stop. You think your risk is the distance to your stop; your real risk is the much shorter distance to your liquidation, a price the exchange publishes for the whole market to aim at. You have not kept the same risk and added profit; you have converted yourself into pre-advertised cascade fuel.

**"The cascade is just panic selling — it's irrational."** The forced sellers are not panicking; they have no choice — the orders are mechanical. And the predator buying the lows is the opposite of panicked: it is the calmest, best-informed capital in the room. A cascade is not a failure of rationality; it is the *rational* consequence of a system that pre-commits a wall of price-insensitive sellers and then removes the depth that would have absorbed them.

**"If I see the hunt coming, I can front-run it and profit."** Maybe — but now *you* are playing the predator's game with the predator's capital requirements, against players who do it full-time with better latency and bigger size. Recognizing the hunt is enormously valuable for *not being the prey*. Trying to become the predator with retail size is how you discover that the person you thought you were hunting was hunting you. The defensive use of this knowledge has a far better risk-reward than the offensive one.

**"Wicks below my stop prove the market is rigged against me personally."** It can feel personal, but the mechanism is impersonal and statistical. You were not singled out — you placed an order at the same focal level as thousands of strangers, and the cluster, not you, was the target. The fix is not to feel persecuted; it is to stop standing in the crowd. The market is not rigged against *you*; it is structurally biased against the *obvious order*, and you get to choose whether your order is obvious.

**"A guaranteed stop or a tighter exchange solves this."** Some venues offer guaranteed stops (filled at your price for a fee) or better liquidation engines, and they help at the margin. But a guaranteed stop still has to be *placed somewhere*, and if you place it at the obvious level you still exit into the hunt — you just pay a known fee to do so instead of taking slippage. The order's *location*, not its type, is what makes it prey. No product fixes a stop parked one tick under support.

## How it shows up in real markets

The single largest crypto liquidation events on record show the cascade at industrial scale. The chart below collects the worst single days from public aggregate liquidation data.

![Bar chart of reported one-day crypto liquidation totals across major cascade episodes from 2020 to 2025](/imgs/blogs/stop-hunts-liquidation-cascades-and-the-predator-7.png)

**May 19, 2021 — the China-mining-ban cascade.** Bitcoin fell from around \$43,000 toward \$30,000 in hours. Aggregate liquidations across derivatives venues exceeded \$8 billion in a single day (Coinglass data, widely reported). The trigger was a news headline, but the *depth* of the move was the cascade: each band of leveraged longs liquidated into the next, and exchanges briefly went unresponsive as their liquidation engines saturated. It is the textbook leverage-built fuel meeting a thin Sunday-into-Wednesday book.

**October 10, 2025 — the record liquidation day.** A sharp risk-off shock set off the largest single-day liquidation cascade ever recorded, with reported totals around \$19 billion across venues (Coinglass and press coverage). Whatever the proximate trigger, the magnitude is a cascade signature: a normal-sized initial move detonating an enormous stack of clustered, high-leverage positions that had built up over a long calm.

**March 12, 2020 — "Black Thursday."** As COVID panic hit every market, Bitcoin fell ~50% in a day, and on Ethereum's then-young DeFi lending markets the cascade went *on-chain*: collapsing collateral values triggered automatic liquidations, but network congestion meant some liquidation auctions cleared at *near-zero* prices because no bidders could get transactions through. The forced sellers were mechanical; the missing depth was a clogged blockchain. The chain-level version of this is exactly the [lending-and-liquidations](/blog/trading/onchain/analyzing-lending-and-liquidations) mechanism.

**The classic equity stop-run.** In single stocks and futures, the gentler cousin of the cascade is the intraday stop-run: price grinds to a hair above the day's high (or below the low), triggers the clustered breakout-stops and short-stops, prints a quick spike, and reverses. No leverage cascade — just a sweep of the discretionary stop cluster at the obvious level. It is so routine that a whole school of intraday trading is built around *fading* it: waiting for the obvious level to be swept and then taking the other side of the forced flow.

**The "max pain" and options-expiry pin.** A related clustering effect: near a big options expiry, dealers hedging their books can nudge the underlying toward the strike where the most options expire worthless. It is not a stop-hunt, but it is the same shape — a known cluster of pre-committed levels that a larger, hedging player has an incentive to push the price toward. The lesson generalizes: any pre-committed, clustered, mechanical order flow is a target.

**The August 5, 2024 yen-carry unwind.** When the Bank of Japan nudged rates higher and the yen spiked, a vast carry trade — borrow cheap yen, buy risk assets — unwound violently across global markets, and crypto caught the spillover. Bitcoin fell sharply over a weekend into Monday, with roughly \$1 billion of crypto liquidations as leveraged longs that had built up during the summer calm were force-sold. It is a clean cross-market example: the trigger was macro (a funding-currency move), but the *amplitude* in crypto was the leverage cascade, because the longs were clustered and the weekend book was thin. The lesson generalizes beyond any one asset: a long calm builds the fuel everywhere, and the spark can come from a completely different market.

**The recurring 3 a.m. weekend wick.** The most common stop-hunt is not a famous event at all — it is the quiet, repeated weekend wick. Crypto trades 24/7, but liquidity is far thinner on weekends and overnight in the dominant timezone, when desks are offline and market-makers run smaller books. That is precisely when a relatively small push moves the price furthest, so the obvious clusters get swept on low-liquidity wicks that reverse by Monday. Traders who place a tight stop on Friday and check their phone on Monday to find they were stopped out at a price that never traded again are not unlucky — they left a visible order in the thinnest book of the week. The defense is structural: do not hold a tight, obvious, resting stop through the thinnest liquidity window.

**The professional execution angle.** The mirror image of being hunted is *executing without being hunted*. A large fund that must sell does not dump into the book; it hides its size precisely so predators cannot detect and front-run the forced-looking flow. That is the whole subject of [execution as a game](/blog/trading/game-theory/execution-as-a-game-vwap-twap-and-hiding-from-predators) — and it is the institutional version of the same defensive principle this post ends on: do not be the obvious, detectable, forced order.

## The playbook: how not to be the prey

The point of all of this is a small set of decisions you control. Here is the defender's playbook, drawn straight from the game. The before-and-after below is the whole thing in one figure.

![Before-and-after comparison of an obvious clustered stop versus a stop placed away from the crowd with appropriate sizing](/imgs/blogs/stop-hunts-liquidation-cascades-and-the-predator-6.png)

**Who's on the other side.** When your stop or liquidation is at an obvious cluster, the player on the other side is a patient, well-capitalized predator (or just the mechanical cascade) who profits *specifically* from your forced, price-insensitive exit. You are not selling to a willing buyer at a fair price; you are handing cheap supply to the one counterparty engineered to receive it.

**The game you're in.** Hunt-vs-hold, a mixed-strategy game with no pure equilibrium. You cannot make yourself un-huntable, but you can make hunting *you* unprofitable by being unpredictable about where your exit sits. The equilibrium prescription is concrete: hide your stop the large majority of the time (the model said roughly seven times in eight) so the predator can never assume your pain point is at the obvious level.

**Don't cluster your stop at the obvious level.** If the whole crowd's stop is a tick under \$100, do not put yours there. Either place it meaningfully *below* the cluster — far enough that the hunt wick does not reach it — or above where you would actually be proven wrong. The worst place for a stop is the most popular place for a stop. In the worked figures, the difference was a stop at \$99.90 (swept on the \$96 wick) versus \$94 (survived it).

**Size so you are never forced.** This is the single most important rule, because it removes you from the cluster entirely. Choose your position size from your stop distance, not the other way around: pick where you would genuinely be wrong, then size the position so that loss is acceptable. A wide stop on a small position is strictly safer than a tight stop on a big one against predators, because it cannot be swept by a wick. And in leveraged markets, keep leverage low enough that your *liquidation* price is nowhere near a normal move — a 2x–5x position has its liquidation 20–50% away and is simply not cascade fuel.

#### Worked example: sizing from the stop instead of the leverage

You have a \$10,000 account and you want to risk at most 1% — \$100 — on a long at \$100, with your real invalidation (where you would genuinely be wrong) at \$94, below the cluster. The honest way to size: your stop is \$6 away, or 6% of the entry price, and you are willing to lose \$100, so your position should be \$100 / 0.06 ≈ \$1,667 of notional. On a \$10,000 account that is about 0.17x of your equity — barely any leverage, with liquidation effectively irrelevant. The wick to \$96 does not touch your \$94 stop, and even your true stop only costs the \$100 you budgeted.

Now the prey's way, sizing from leverage instead: "I'll use 25x to make it interesting," so \$10,000 of margin controls \$250,000 of notional with liquidation about 4% away at \$96. The \$96 wick *is* your liquidation; you lose far more than \$100, and you lose it at the exact bottom. Same account, same view, same entry — the only difference is whether you sized from your stop or from your greed.

The intuition: sizing from the stop makes the position small enough that no wick can force you; sizing from the leverage makes the position large enough that an ordinary wick liquidates you. The arithmetic is the entire difference between being the buyer of the cheap supply and being the cheap supply.

**Use mental stops or options instead of resting orders.** A resting stop order is *visible infrastructure* — even if the exchange does not leak it, its location is inferable from the cluster, and once it fires it is detectable forced flow. A *mental stop* (you watch and exit manually on a real break) leaves nothing in the book to hunt. Better still for a defined risk, a *protective put* (an option that pays off if the price falls below a strike) caps your loss without any resting sell order at all — you exit on *your* terms, on a real move, not on a wick. The tradeoff is that mental stops require discipline and puts cost a premium; both are usually cheaper than being the cheap supply.

**Avoid max leverage, especially after a long calm.** The cascade fuel is highest exactly when it feels safest — after a quiet stretch that has let leverage build and depth thin out. Treat high aggregate leverage and a long calm as the *setup* for a cascade, not the all-clear. If you must hold leverage into that, hold less of it.

**The defender's view, and the rare offensive note.** Defensively, all of the above reduces to: *do not be forced and do not be obvious.* Offensively — and this is the part to treat with caution — recognizing a swept cluster *after* it fires is one of the cleaner setups in trading: the forced supply is gone, the predator has bought, and the snap-back is often fast. Fading a completed stop-run (buying after the wick has swept the obvious low and reversed) is taking the patient side *after* the forced flow has cleared, rather than trying to push it yourself. That is a defensible edge for ordinary size; trying to *cause* the cascade is not, both because you lack the capital and because in most venues it is manipulation.

**Invalidation.** This whole frame is wrong when the move is *real* — when the price breaks the cluster and keeps going because the fundamentals genuinely changed, not because someone swept the stops and bought. The tell is the snap-back: a hunt sweeps and reverses; a real breakdown sweeps and *continues*, because there was no patient bid waiting, only more sellers. If you faded a swept low and it keeps falling, you were wrong about which one it was — and your *own* stop (placed below the cluster, sized small) is what saves you. The defense is never a single trick; it is the discipline of not being the forced, obvious order in the first place.

**The one number to carry.** If you remember a single figure from this post, make it the *distance to forced selling*. Before any leveraged trade, ask: how far is the price from the point where I am made to sell regardless of my opinion — my stop if it is at the obvious level, my liquidation if my leverage is high? If that distance is smaller than a normal day's range, you are cascade fuel and you should either widen it (lower leverage, a stop below the cluster) or stand aside. The predator's entire business is the gap between where you *think* your risk is (your view) and where your risk *actually* is (the published price at which you will be forced out). Close that gap, and you stop being the cheap supply — which, in a game with no pure equilibrium and a patient hunter on the other side, is the only durable edge there is.

## Further reading & cross-links

- [Execution as a game: VWAP, TWAP, and hiding from predators](/blog/trading/game-theory/execution-as-a-game-vwap-twap-and-hiding-from-predators) — the institutional mirror image: how a large player executes *without* becoming the detectable, huntable forced flow.
- [Crowded trades and the exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game) — the de-grossing logic behind why everyone forced through the same door at once gets a terrible price, and why patient capital sets the price of their panic.
- [Analyzing lending and liquidations](/blog/trading/onchain/analyzing-lending-and-liquidations) — the on-chain, smart-contract-level mechanics of how forced liquidations actually execute, the chain-level version of the cascade in this post.
- [The prisoner's dilemma in markets: why everyone sells at once](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once) — the coordination math behind forced, simultaneous selling, and why "I'll get out first" usually fails.
