---
title: "Order Types as Strategic Moves: Market, Limit, Hidden, and Pegged"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Every order type is a move in a game that trades off urgency, price, queue priority, and how much of your intent you reveal to the other side."
tags: ["game-theory", "trading", "order-types", "market-microstructure", "limit-order", "market-order", "iceberg-order", "execution", "adverse-selection"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — An order type is not a button; it is a strategic move, and the moment you pick one you have already chosen where you sit on two trade-offs that the other side is reading off your screen.
>
> - **Urgency vs patience:** a market order pays the spread now for a certain fill; a limit order saves the spread but risks adverse selection and never filling at all.
> - **Reveal vs conceal:** a resting limit broadcasts a price and a size to everyone; hiding that intent (iceberg, hidden, dark, pegged) costs you queue priority, a fee, or fill certainty.
> - **Concealment is never free.** Every order that hides something pays for the silence somewhere — that is the central law of this whole post.
> - **The one rule:** before you click, ask *who reads this order and what does it tell them* — because the order is a signal, and a signal you did not mean to send is the most expensive thing on the screen.

A few years ago a trader I know wanted to buy 50,000 shares of a thinly traded stock. He typed it into a market order, hit send, and watched the price he paid climb a full 1.2% above where it had been quoting one second earlier. He had not been front-run by a villain. He had simply announced, in the loudest possible voice the market has, *I need 50,000 shares and I need them now* — and the people on the other side, who read that announcement the instant his order crossed the spread, charged him for the privilege of his own urgency. He paid roughly \$9,000 more than the mid-price for a decision he made without thinking, by choosing the wrong order type.

Here is the thing almost nobody tells a new trader: the order-entry ticket is a menu of *strategies*, not a menu of *conveniences*. Market, limit, stop, iceberg, pegged, post-only, IOC, midpoint — these are not seven slightly different ways to do the same thing. Each one is a different move in a game you are playing against a specific, adaptive counterparty who can see (or infer) what you did and will respond. The market maker on the other side, the high-frequency firm scanning the book, the other institution trying to buy the same thing you are — they all read your order as information, and your fill quality is the price they charge for what you revealed.

The diagram above is the mental model for the entire post. It lays every order type onto two axes. The horizontal axis is **how much you reveal** — from a market order that screams your intent, to a dark order nobody sees until after it prints. The vertical axis is **how urgent you are** — from "cross the spread this millisecond" to "post a price and wait all day." Where an order sits on that grid *is* its strategy. The rest of this post is about reading that grid: what each order tells the other side, what it costs to stay quiet, and how to choose your move when you actually know who is reading it.

![Grid of order types mapped on urgency and reveal axes](/imgs/blogs/order-types-as-strategic-moves-market-limit-hidden-and-pegged-1.png)

## Foundations: the order book, the spread, and the two trade-offs

Before any order type makes sense, you need three plain-English ideas: what a *limit order book* is, what the *bid-ask spread* costs, and the two master trade-offs every order navigates. We build each from zero.

### What an order actually is

An **order** is an instruction you send to an exchange: *buy or sell this many shares of this thing, under these conditions*. The exchange does not care who you are; it cares only about the instruction. The whole art of trading well is choosing the *conditions* — and the conditions are exactly what define the order type.

There are two fundamental things an order can do, and they sit on opposite sides of every trade:

- A **maker** (also called *adding* or *posting* liquidity) is an order that *waits*. It sits in the book at a price you name and offers to trade with anyone who comes along. You are supplying the option for someone else to trade; you are the patient one.
- A **taker** (also called *removing* or *crossing*) is an order that *acts now*. It reaches across and trades against an order that was already resting. You are demanding immediacy; you consume the liquidity someone else supplied.

Every trade has exactly one maker and one taker. When you trade, you are always one or the other. That single distinction — *did I supply the trade or demand it?* — is the root of the whole taxonomy.

### The limit order book

Start with a list of two columns. On the left, every standing offer to **buy**, sorted from highest price down. These are the **bids**. On the right, every standing offer to **sell**, sorted from lowest price up. These are the **asks** (or *offers*). This list is the **limit order book** — "limit" because each entry names a *limit price*, the worst price its owner will accept.

The single most important pair of numbers in the book:

- The **best bid** is the highest price anyone is currently willing to pay.
- The **best ask** is the lowest price anyone is currently willing to sell at.

The gap between them is the **bid-ask spread** — the distance between the best price you can sell at and the best price you can buy at. It is the single most important cost in trading, and most beginners never see it because their broker hides it inside the fill price.

A concrete book for a stock trading around \$100.00:

| Side | Price | Size (shares) |
|---|---|---|
| Ask (sell) | \$100.03 | 800 |
| Ask (sell) | \$100.02 | 500 |
| Ask (sell) | **\$100.01** ← best ask | 600 |
| — spread = \$0.02 — | | |
| Bid (buy) | **\$99.99** ← best bid | 700 |
| Bid (buy) | \$99.98 | 1,100 |
| Bid (buy) | \$99.97 | 900 |

The **mid-price** — the fair reference everyone agrees on — is halfway between best bid and best ask: \$100.00 here. The spread is \$0.02. If you want to buy *right now*, you pay \$100.01 (the best ask), which is \$0.01 — half the spread — above the mid. That \$0.01 is the **half-spread**, and it is the toll for immediacy.

To see how the order book is itself a continuous auction where bids ladder up and asks ladder down, read the sibling post on [the double auction of the order book](/blog/trading/game-theory/every-market-is-an-auction-the-double-auction-of-the-order-book). Here we take that auction as given and ask: *which move do you make inside it?*

### Trade-off one: urgency vs patience

The first axis. You can fill *now* or fill *cheaply*, but rarely both.

- **Cross the spread now** (a market order): you pay the half-spread (or more) and the fill is certain. You bought immediacy.
- **Post a price and wait** (a limit order): you save the half-spread — you might even get *paid* the spread — but the fill is uncertain, and while you wait the price might run away from you, leaving you chasing it later at a worse price.

This is the patience-vs-urgency trade-off, and there is no free lunch in it. The faster you must fill, the more you pay; the longer you can wait, the cheaper you can fill — but the more you risk filling at the wrong moment or not at all. Every order type is, at bottom, a position on this axis.

### Trade-off two: reveal vs conceal

The second axis, and the one this series cares about most, because it is pure game theory. When you post an order to the book, that order is **information**. Everyone watching can read it. A resting limit to buy 10,000 shares tells the whole market: *somebody wants 10,000 shares at this price.* That is a signal — and the other side trades against signals.

- **Reveal** (post a visible limit): you get queue priority and the chance to earn the spread, but you have told everyone your price and your size. They can step in front of you, fade away from you, or pick you off when the news moves against you.
- **Conceal** (iceberg, hidden, dark, pegged): you hide your price, your size, or both, so the other side cannot read your intent and move against you — but concealment *always* costs you something: queue priority, a fee, or the certainty of a fill.

That last clause is the law of the whole post: **concealment is never free.** A market that lets you hide your hand makes you pay for the silence, because the information you are withholding has value to the people you are withholding it from. Hold that thought; we will price it out, in dollars, several times.

These two trade-offs connect directly to two ideas from elsewhere in this series. The reveal-vs-conceal axis is exactly the logic of [mixed strategies and being unpredictable](/blog/trading/game-theory/mixed-strategies-and-the-value-of-being-unpredictable): the value of hiding your intent is that the opponent cannot best-respond to a move they cannot read. And the cost of being read — of having an informed-looking order picked off — is the engine of [Kyle's model, where an informed trader hides in the noise](/blog/trading/game-theory/kyles-model-how-an-informed-trader-hides-in-the-noise). Keep both in the back of your mind; we cash them out at the end.

Every order type we are about to meet falls onto one map. The first fork is whether the order **takes** liquidity (demands an immediate fill and pays the spread) or **makes** liquidity (posts and waits to be hit); the second fork is how much it leaves you visible on the book. The whole menu — market, limit, iceberg, hidden, pegged, stop, IOC, post-only — is just different answers to those two questions.

![Order type taxonomy splitting takers from makers and then by how visible each leaves you](/imgs/blogs/order-types-as-strategic-moves-market-limit-hidden-and-pegged-7.png)

## The market order: maximum urgency, maximum noise

Start with the bluntest move on the board. A **market order** says: *fill me immediately, at whatever price the book has right now.* It is the order with the highest urgency and, paradoxically, the loudest signal — because to fill immediately you must *cross the spread*, and crossing the spread is the single most legible action in markets.

### What it costs

The instant your market buy hits the exchange, it trades against the best ask, then the next ask, then the next, eating up the book until your size is filled. If the book is deep, you pay the half-spread. If the book is thin, you "walk the book" and pay much more — this is **market impact**, the amount you move the price by demanding liquidity faster than it is supplied.

#### Worked example: the cost of crossing the spread

You want to buy 600 shares of our \$100.00 stock. The book above has 600 shares resting at the best ask of \$100.01. You send a market buy.

- All 600 fill at \$100.01. You paid \$100.01 × 600 = \$60,006.
- The mid-price was \$100.00, so the "fair" cost was \$60,000.
- Your cost of immediacy: \$60,006 − \$60,000 = a \$6 toll, which is exactly the half-spread (\$0.01) × 600 shares.

Now scale it up. You want 1,500 shares, but only 600 rest at \$100.01. Your order eats all 600 at \$100.01, then 500 at \$100.02, then 400 at \$100.03. Your average fill is (600×100.01 + 500×100.02 + 400×100.03) / 1,500 = \$100.0187. Against the \$100.00 mid you paid \$0.0187 × 1,500 = a \$28 toll — far more than the half-spread, because you walked the book. The intuition: a market order's cost is not the half-spread; it is the half-spread *plus the impact of your own size*, and the thinner the book, the more your urgency costs you.

### What it signals

Here is the strategic part. A market order does not just cost you the spread — it *tells the other side you were willing to pay it.* And a trader who is willing to pay the spread to fill *right now* is, on average, a trader who knows something or fears something. Why else the rush?

This is the heart of [adverse selection](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news): the market maker who sold you those 600 shares now suspects you may know the price is about to rise, so they re-quote higher to protect themselves. Your urgency is information, and the market prices it in against you. The market maker's whole job is to read order flow this way; for the dealer's-eye view of exactly this inference, see [how an options market maker thinks about the other side of your trade](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade).

So the market order is the move you make when *immediacy is worth more to you than the information you leak by demanding it.* That is a real and common situation — a stop got hit, a hedge must go on, the close is in ten seconds. But it is a choice, and you should make it knowing the loudness is the cost.

There is a subtler version of the leak that catches even experienced traders: a *sequence* of small market orders. You might think slicing a large buy into twenty small market orders disguises it. It does the opposite over time — twenty consecutive buys all crossing the spread in the same direction is a flashing sign that reads *persistent buyer, more to come.* The market makers update their quotes upward after each one, and you walk yourself up the book even though no single order was large. This is precisely the price-impact relationship at the heart of [Kyle's model](/blog/trading/game-theory/kyles-model-how-an-informed-trader-hides-in-the-noise): cumulative net order flow in one direction moves the price linearly, and a string of same-side market orders is cumulative net flow in its purest form. Concealing a large order is therefore not about making each piece small; it is about breaking the *correlation* the other side reads — randomizing the timing, the size, and sometimes the venue, so the flow looks like noise rather than a campaign. That randomization is a [mixed strategy](/blog/trading/game-theory/mixed-strategies-and-the-value-of-being-unpredictable) in action, and it is exactly what execution algorithms automate.

## The limit order: patience, a named price, and a visible signal

The opposite move. A **limit order** says: *I will trade, but only at my price or better — and until someone meets it, I wait.* You name a limit price; the order rests in the book; it fills only when the market comes to you.

A limit buy posted *below* the best ask (say, at the best bid of \$99.99) is passive — it adds liquidity and waits. A limit buy posted *at or above* the best ask is **marketable** — it crosses immediately, behaving like a market order but with a price cap to protect you from walking too far up the book. Most of the strategic interest is in the passive, resting kind, so that is what we focus on.

### The three things a resting limit buys you

1. **You save the spread.** Instead of paying \$100.01, you offer \$99.99 and wait. If you fill, you bought \$0.02 cheaper than a market order would have.
2. **You might earn the spread.** If a seller comes to you and hits your \$99.99 bid, and you later sell to a buyer lifting your \$100.01 offer, you pocket the whole \$0.02 spread. This is the market maker's bread and butter.
3. **Queue priority.** Most exchanges fill resting orders by **price-time priority**: best price first, and within a price, *first come, first served*. Post early and you are near the front of the line at your price — a real, valuable asset, because being at the front means you fill before the price moves on.

### The two risks a resting limit bears

Nothing is free, and the limit order's costs are subtle precisely because they are *probabilistic* — they do not show up on every trade, so beginners discount them.

1. **Non-execution risk.** The market might simply never come to your price. You bid \$99.99, the stock rallies to \$101 on news, and you never filled — you missed the move entirely and now must chase at a worse price. Your "saving" turned into a much larger opportunity cost.
2. **Adverse-selection risk.** This is the deep one, and it is pure game theory. *When* does your resting bid get filled? It gets filled exactly when someone wants to sell to you at \$99.99 — and the people most eager to sell to you are the people who think \$99.99 is *too high*, i.e., who have information that the price is about to fall. So your resting limit fills selectively, on the trades where you are on the wrong side. You "win" the fill precisely when winning it is bad news.

That second risk is why a resting limit order is *a visible signal that the informed can exploit.* You posted a price; you advertised exactly where you will trade; and the informed trader, who can see your bid sitting there, sells into it the moment they know it is mispriced. This is the [Glosten-Milgrom adverse-selection game](/blog/trading/game-theory/the-bid-ask-spread-as-an-adverse-selection-game-glosten-milgrom) running against *you* personally.

### Pricing the choice: market vs limit, in dollars

Now we can put numbers on both moves and compare them directly. The figure below does exactly this for a single-share buy.

![Bar chart of expected cost for a market order versus a limit order](/imgs/blogs/order-types-as-strategic-moves-market-limit-hidden-and-pegged-2.png)

#### Worked example: the expected cost of patience

Stock at \$100.00, spread \$0.02, so the half-spread is \$0.01 per share. You want to buy.

The **market order** is simple and certain: you cross and pay the half-spread. Cost = **\$0.01 per share** (1.00 cent), guaranteed.

The **limit order** is a gamble with three pieces. Suppose, from watching this stock, you estimate:

- Your resting limit fills with probability **55%** (it does not always fill).
- When it fills, you save the half-spread you would have paid: a credit of **−\$0.010** per filled share.
- But a filled limit is adversely selected — on average you lose **\$0.012** per filled share to the fact that you fill when the price is about to tick down against you.
- When it *does not* fill (45% of the time), the price runs away and you must chase it later, costing **\$0.030** per share.

Expected limit-order cost per share:

$$
E[\text{cost}] = 0.55 \times (-0.010 + 0.012) + 0.45 \times 0.030 = 0.55 \times 0.002 + 0.0135 = \$0.0146
$$

So the limit order's *expected* cost is about **1.46 cents** per share — but look at what is hiding in that number. The pure spread-saving (−1.0 cent × 0.55 = −0.55 cent) makes the limit look like a free win. Adverse selection (+1.2 cent × 0.55) eats most of that. And the chase-on-a-miss (+3.0 cents × 0.45 = +1.35 cents) is the real killer: it is the single biggest term. The intuition: the limit order looks cheaper than the market order on the spread alone, but once you charge it for the fills it misses and the toxicity of the fills it gets, "patience" can quietly cost you *more* than just paying the spread — and whether it does depends entirely on your fill probability.

Notice how sensitive this is. If your fill probability were 90% instead of 55%, the chase term shrinks to 0.10 × 3.0 = 0.30 cents and the limit order wins comfortably. If it were 30%, the chase term balloons to 2.1 cents and patience is a disaster. *The whole decision rides on how likely you are to fill* — which is itself a function of how aggressively you priced the limit and how fast the market is moving. That is the patience-vs-urgency trade-off made quantitative.

## The patience–urgency curve: there is an optimal aggressiveness

The market order and the resting limit are the two endpoints of a continuum. In between lie all the prices you could post: deep below the mid (very patient, cheap if it fills, likely to miss), just inside the spread (semi-aggressive), or marketable (basically a market order). Each point on that continuum is a different blend of the two costs, and somewhere in the middle is the blend that minimizes your *total* expected cost.

![Line chart of immediacy cost timing cost and total cost versus aggressiveness](/imgs/blogs/order-types-as-strategic-moves-market-limit-hidden-and-pegged-3.png)

The curve shows it cleanly. The **immediacy cost** (red, dashed) rises with aggressiveness: the more you cross, the more spread and impact you pay. The **timing cost** (amber, dashed) falls with aggressiveness: the more you cross, the less you risk the price running away before you fill. Their sum (blue) is U-shaped, and the bottom of the U is the cheapest move — neither maximally patient nor maximally urgent.

#### Worked example: finding the cheapest blend

Model the two costs as functions of aggressiveness $a$, running from 0 (a deep patient limit) to 1 (a market order), both in cents per share:

$$
\text{immediacy}(a) = 1.6\,a^{1.4}, \qquad \text{timing}(a) = 2.4\,(1-a)^{1.7}
$$

Total cost is their sum. Evaluate at a few points:

- $a = 0$ (fully patient): immediacy = 0, timing = 2.4 → **total 2.40 cents.** You pay nothing to cross but bear the full risk of the price running.
- $a = 1$ (market order): immediacy = 1.6, timing = 0 → **total 1.60 cents.** You pay the full spread-plus-impact but never miss.
- $a = 0.55$ (a limit posted partway into the spread): immediacy = 1.6 × 0.55^1.4 ≈ 0.70, timing = 2.4 × 0.45^1.7 ≈ 0.61 → **total ≈ 1.31 cents.**

The middle blend (\$0.0131) beats both pure extremes — cheaper than the patient limit (\$0.0240) *and* cheaper than the market order (\$0.0160). The intuition: optimal execution is almost never "always cross" or "always wait"; it is a calibrated point on the curve, and the smartest traders spend their lives finding it for each name and each moment. This is exactly the problem that [execution algorithms like VWAP and TWAP](/blog/trading/game-theory/execution-as-a-game-vwap-twap-and-hiding-from-predators) — the subject of the next post — automate: they walk this curve over time instead of choosing one point.

## Concealment orders: iceberg, hidden, and the price of silence

So far the reveal-vs-conceal axis has been theoretical. Now it gets concrete, because exchanges sell you concealment directly — and they charge for it, exactly as the law predicts.

The figure below catalogs what each order type reveals, what it hides, and the bill it pays for hiding.

![Grid of order types showing what each reveals what it hides and the cost of hiding](/imgs/blogs/order-types-as-strategic-moves-market-limit-hidden-and-pegged-4.png)

### Why you would ever want to hide

Recall the problem: a large resting limit is a billboard. If you post a visible bid for 100,000 shares, you have told every other participant that there is a huge buyer at this level. Three bad things follow:

1. **Others step in front.** A faster trader posts a bid one tick above yours, takes the next seller, and leaves you. You provided the information; they captured the value.
2. **The price fades up.** Sellers see the wall of demand, infer the price should be higher, and pull their offers. You moved the market against yourself just by showing up.
3. **You get picked off.** When bad news hits, the informed sell into your stale bid before you can pull it, because they can see exactly where it sits.

Concealing your size defeats all three — *if* you are willing to pay for it.

### The iceberg order

An **iceberg** (or *reserve*) order shows only a small slice of your true size to the book, while the bulk stays hidden in the exchange's matching engine. When the visible slice fills, the engine automatically replenishes it from the hidden reserve. Only the tip of the iceberg shows; the mass is underwater. The name is the mechanism.

The catch — and it is the whole point — is **queue priority on the hidden part.** On most exchanges, the visible slice holds normal time priority, but each time it refills, the replenished slice goes to the *back* of the queue at that price, behind everyone who arrived in the meantime. You are trading priority for invisibility.

![Before and after comparison of a full visible limit versus an iceberg order losing queue priority](/imgs/blogs/order-types-as-strategic-moves-market-limit-hidden-and-pegged-6.png)

#### Worked example: the queue-priority cost of an iceberg

You want to buy 10,000 shares at \$99.99. Compare two ways.

**Full visible limit:** you post all 10,000 at \$99.99. You arrived first, so you hold one priority spot for the entire size. When sellers hit the \$99.99 bid, you fill first, all 10,000, at time priority. The cost: you broadcast 10,000 shares of demand, and the market may fade away from you before you fill much of it.

**Iceberg showing 1,000:** you show 1,000 and hide 9,000. The first 1,000 fills at your priority. Then the engine refills the next 1,000 — but that slice now sits *behind* every other order that arrived at \$99.99 in the meantime. Say 4,000 shares of other orders queued up at \$99.99 while your first slice was filling. Your second slice waits behind those 4,000. Across all nine refills, you repeatedly go to the back of the line.

Put a number on it. Suppose at \$99.99 the queue turns over at 2,000 shares per minute and, each time you refill, an average of 3,000 shares of competing orders sit ahead of you. Each refill therefore waits an extra 3,000 / 2,000 = 1.5 minutes versus the visible order's instant priority. Nine refills × 1.5 minutes = **13.5 extra minutes** of exposure. If the price drifts up \$0.005 per minute during that window, your delayed fills cost roughly 13.5 × \$0.005 ≈ **\$0.0675 of slippage per delayed share** — call it \$0.04 averaged over the whole 10,000, or about a **\$400** cost on the order. The intuition: the iceberg bought you invisibility, and the bill arrived as queue position — you traded a known billboard for an unknown wait, and the wait is not free.

So whether to iceberg comes down to a comparison: is the market-impact you avoid by hiding your 10,000-share intent *larger* than the queue-priority slippage you pay for hiding it? For a genuinely large order in a name that would move on the news of your size, yes — the impact you dodge dwarfs the queue cost. For a small order nobody would have reacted to anyway, the iceberg is pure loss: you paid for silence about a secret that was not worth keeping.

### The fully hidden order

A **hidden** order goes one step further: it shows *nothing* in the book — no price, no size. It rests in the matching engine and trades when someone crosses into it, but no one watching the book knows it is there. The price for total invisibility is steep on most venues: hidden orders rank *behind all visible orders* at the same price (visible orders always have priority over hidden ones, regardless of time), and many exchanges charge hidden executions the *taker* fee even though you were technically resting. You become last in line and you may forfeit the maker rebate. Total concealment, maximum cost.

## Pegged orders: stay competitive without naming a number

There is a clever middle move that sidesteps part of the reveal problem. A **pegged** order does not name a fixed price; instead it *tracks* a reference — the best bid, the best ask, or the midpoint — and automatically re-prices as that reference moves. You say "peg my buy to the best bid" and the order rides up and down with the market's best bid without you re-sending it.

Why is this strategic? Two reasons.

1. **You stay competitive automatically.** A fixed limit at \$99.99 becomes stale the moment the market moves to \$100.05 — you are now far from the action and will not fill. A bid pegged to the best bid is always *at* the best bid, no matter where it goes. You never have to chase.
2. **You reveal less about your conviction.** A fixed price says *I specifically value this at \$99.99* — a strong, readable opinion. A pegged order says only *I want to participate near the going rate* — a much weaker signal. You have told the market you are a buyer, but not the precise price at which you would walk away, which is the more valuable secret.

The most strategically interesting peg is the **midpoint peg**: an order that sits at the mid-price, halfway between bid and ask. It is invisible to the lit book (the mid is not a quoted price), and it offers *price improvement* — a midpoint buy fills at the mid, saving the half-spread versus crossing. Midpoint orders trade against each other in the gap where no visible order can reach, which is the bridge to dark trading below.

#### Worked example: the price improvement of a midpoint peg

Stock at \$100.00, spread \$0.02, so best bid \$99.99 and best ask \$100.01. You want to buy 1,000 shares.

- **Cross the spread (market order):** pay \$100.01 × 1,000 = \$100,010. Cost vs mid = +\$10.
- **Post a visible limit at \$99.99:** if it fills, pay \$99.99 × 1,000 = \$99,990. Saving vs mid = +\$10 in your favor — but you bear non-fill and adverse-selection risk, and you advertised a 1,000-share bid.
- **Midpoint peg:** if a midpoint seller crosses with you, you fill at \$100.00 × 1,000 = \$100,000. Cost vs mid = \$0 — you split the spread with the counterparty, each saving the half-spread, and *neither of you showed a price in the lit book.*

The midpoint peg captures half the spread-saving of a passive limit while revealing nothing in the visible book. The intuition: pegging to the mid is the order type for the trader who wants to be quiet *and* fairly priced, accepting only that the fill happens when another hidden counterparty happens to be there — concealment paid for, this time, in fill uncertainty rather than queue position.

## Stop orders: a conditional move that becomes a target

A **stop order** is different in kind: it is *conditional*. It does nothing until the price touches a trigger you set; then it springs into life as a market order (a *stop-market*) or a limit order (a *stop-limit*). A stop-loss sell sits dormant below the current price and fires — selling you out — only if the market falls to your trigger. It is insurance against a move you fear.

Strategically, the stop has a dangerous property the other order types do not: **stops cluster, and clusters are targets.** Traders place stops at obvious levels — round numbers, recent lows, just below a chart's support. So just under an obvious support level sits a pile of stop-sell orders, invisible until triggered but entirely predictable to anyone who has watched the same chart you have.

This is where the predator enters. A trader with enough size can push the price down *just far enough* to tag that cluster of stops. The stops fire, dumping market-sell orders into the book, which pushes the price down further, triggering more stops — a self-reinforcing cascade. The predator, having sold to trigger it, buys back cheaply at the bottom of the cascade they engineered. This is **stop-hunting**, and it is a real, documented behavior. The defensive lesson (this series always frames manipulation as *detection and defense*, never how-to): a stop placed at the obvious level is a free option you have written to the rest of the market — *they* can see where it must be even though *they* cannot see the order. Place stops away from the herd's levels, or use mental stops you execute manually, precisely because a clustered stop is the most predictable order in the market. This connects to [crowded trades and the exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game): when everyone's stop is at the same place, everyone's forced sell hits at the same moment, and the exit is a stampede.

## The fast-or-nothing orders: IOC, FOK, and post-only

Three more moves round out the practical taxonomy, each pinning down one dimension of the trade-off precisely.

- **IOC (Immediate-Or-Cancel):** fill whatever you can *right now* against resting orders, and cancel the rest instantly. It never rests in the book, so it leaves no visible signal and bears no resting-order risk. It is the move for "take what is available this instant, but do not advertise a leftover." A taker who refuses to become a maker.
- **FOK (Fill-Or-Kill):** fill the *entire* size immediately or cancel the *whole* thing — no partial fills. It is the all-or-nothing move: useful when a half-fill is useless to you (one leg of a paired trade, for instance) and you would rather have nothing than half.
- **Post-only:** the mirror image. It *guarantees you are a maker* — if the order would cross the spread and take liquidity (making you a taker), the exchange rejects or re-prices it instead of executing. You use post-only when capturing the **maker rebate** (more on this next) and avoiding the taker fee matters more than an instant fill. It is the order type that says *I will only ever supply liquidity, never demand it.*

Each is a way of refusing one half of a trade-off outright: IOC refuses to rest, FOK refuses to partial-fill, post-only refuses to take. Knowing they exist means you can express your exact strategic intent instead of hoping a plain order behaves the way you want.

## The maker rebate and the queue: where order types meet hard cash

Two threads have run through everything above — *queue priority* and the *maker/taker* split. They deserve their own section, because together they turn the abstract reveal-vs-conceal trade-off into a fee you can read off an exchange's price list to the tenth of a cent.

### Price-time priority, in detail

When you rest a limit order, you join a queue at your price, and the queue is ordered by a simple rule almost every exchange uses: **price-time priority.** Better prices fill first; within a single price, earlier orders fill before later ones. Your position in that queue is an asset with real value, because the orders ahead of you fill first — and if the price ticks away before the queue reaches you, you may never fill at all.

This is why being *early* matters so much, and why every order type that touches the queue is, in part, a bet on your place in line. A plain visible limit posted first holds the best spot. An iceberg surrenders its spot on every refill. A hidden order starts *behind* every visible order regardless of when it arrived. A pegged order that re-prices when the reference moves often loses its time priority on the re-price (it counts as a new order at the new level). Each concealment choice is also a *queue* choice, and the queue is where fill probability — the variable that dominated our market-vs-limit worked example — is actually decided.

### The maker-taker game

Now layer the fees on top. Under the **maker-taker** model that most US equity exchanges run, the exchange *pays* you a rebate when your order adds liquidity (you posted and waited — you made) and *charges* you a fee when your order removes liquidity (you crossed — you took). The exchange pockets the difference. This flips the spread economics: a maker not only saves the spread, they get *paid a rebate on top*; a taker not only pays the spread, they pay a *fee on top*. The order-type choice — maker or taker — is now worth hard cash on every single share, set explicitly by the exchange.

#### Worked example: what the maker/taker choice is worth

Use a representative US schedule (NYSE Arca, mid-2026): a maker rebate of about \$0.0022 per share for adding liquidity, and a taker fee of about \$0.0030 per share for removing it. The swing between being a maker and a taker is \$0.0022 + \$0.0030 = **\$0.0052 per share** — a 0.52-cent gap that has nothing to do with the spread and everything to do with which side of the order-type choice you landed on.

Now trade 200,000 shares over a day. If you reflexively use market orders and take every time, you pay 200,000 × \$0.0030 = **\$600** in taker fees. If you instead post passively and earn the rebate, you *receive* 200,000 × \$0.0022 = **\$440** in rebates. The difference between the two habits is \$600 + \$440 = **\$1,040** on the same 200,000 shares — and that is before counting the spread you saved by posting rather than crossing. The intuition: the maker-taker fee turns "be patient" into a directly priced strategy, which is exactly why **post-only** orders exist and why entire high-frequency firms are built around capturing the rebate rather than the spread.

But — and this is the strategic sting — the rebate is not free money either, because to earn it you must *rest a visible order*, which puts you right back into the adverse-selection and reveal problems from the limit-order section. The rebate compensates you for supplying liquidity precisely because supplying it is dangerous: you get picked off by the informed, and the rebate is the market's payment for bearing that risk. So the maker-taker fee is not a loophole; it is the price of the reveal-vs-conceal trade-off, made explicit by the exchange. You earn the rebate by accepting the risk of being read — which is the whole law of this post, now denominated in tenths of a cent.

## The take-or-make game: why everyone races to cross

Now we earn the game-theory payoff. Step back from a single trader and consider *two* traders who both want to buy the same thing from a thin book — a crowded entry into a popular trade. Each independently chooses a move: **TAKE** (cross the spread now, certain fill) or **POST** (rest a passive limit, save the spread but risk the other one grabs the liquidity first and the price runs).

This is a 2×2 game, and its structure is the most famous one in all of game theory. The payoff matrix below — computed from the model, with payoffs in cents per share of net edge — lays out all four outcomes.

![Two by two payoff matrix for the take versus post execution game](/imgs/blogs/order-types-as-strategic-moves-market-limit-hidden-and-pegged-5.png)

#### Worked example: the execution prisoner's dilemma

The four cells, with your payoff first and the rival's second, in cents per share of edge:

- **Both POST** → (+3, +3). Neither pays the spread, the price does not run, and you both fill cheaply against incoming sellers. This is the *jointly best* outcome — the cooperative optimum.
- **You TAKE, rival POSTS** → (+5, −2). You snatch the thin offer first, fill is certain, and the rival is left chasing a now-higher price. You win the race.
- **You POST, rival TAKES** → (−2, +5). The mirror image: the rival grabs it, you chase and miss.
- **Both TAKE** → (−1, −1). You both cross at the same instant, the thin offer evaporates, the price gaps up, and you *both* overpay the impact. The worst joint outcome.

Now find the equilibrium by checking your best response. *If the rival POSTS,* you compare your TAKE payoff (+5) to your POST payoff (+3) — you prefer to TAKE. *If the rival TAKES,* you compare your TAKE payoff (−1) to your POST payoff (−2) — you *still* prefer to TAKE. So no matter what the rival does, taking beats posting for you. TAKE **strictly dominates** POST. By symmetry the rival reasons identically. Both take. The equilibrium is **(TAKE, TAKE) = (−1, −1).**

And here is the tragedy, in one line: the equilibrium (−1, −1) is *worse for both of you* than the outcome you could have reached by both posting (+3, +3). Individual rationality drives you both to the spread-paying, price-gapping race, even though mutual patience would have paid you both more. This is the [prisoner's dilemma in markets](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once), and it is *exactly* why, in a crowded trade or a fast tape, everyone reaches for the market order at once and everyone overpays. The intuition: when you fear the other side will grab the liquidity first, the safe individual move is to grab it yourself — and when everyone reasons that way, the collective result is a stampede of takers paying impact that patience would have spared them all.

This single game explains a huge fraction of market behavior: the open and close where everyone crosses at once, the post-news scramble, the "everyone hits the bid" flash of a selloff. The order-type choice is the move; the dilemma is why the aggressive move wins even when it is collectively dumb.

## Common misconceptions

**"A market order gets me a better fill because it's instant."** No — instant and *good* are opposites here. A market order gets you the *certain* fill, at the cost of paying the spread and broadcasting your urgency. "Better" depends on whether immediacy is worth more to you than the spread plus the information you leak. For a patient order in a liquid name, a market order is often the *worst* fill, not the best.

**"A limit order is free — I just save the spread."** This is the most expensive belief on the list. A resting limit bears two real costs that do not appear on any single fill: adverse selection (you fill exactly when the price is about to move against you) and non-execution (you miss, and the chase costs you more than the spread you saved). The worked example showed a "patient" limit with a 1.46-cent expected cost — *higher* than the 1.00-cent market order — once you charge it honestly for the misses. Patience is a position with risk, not a free lunch.

**"Hidden orders let me trade with no downside — I just stay invisible."** Concealment is never free; that is the law of this post. A hidden order forfeits queue priority (it ranks behind every visible order at its price) and often pays the taker fee instead of earning the maker rebate. An iceberg goes to the back of the line on every refill. A dark order may never fill at all. You always pay for the silence — in priority, in fees, or in fill probability.

**"If I use a stop-loss, I'm protected — the worst case is capped."** A stop-loss caps your *intended* exit price, not your *realized* one. A stop-market becomes a market order when triggered, so in a fast move it can fill far below your trigger (slippage). Worse, stops cluster at obvious levels and become targets, so the very move that triggers your stop may be partly *caused by* the cascade of stops around yours. Your "protection" can be the thing that sells you out at the bottom.

**"Order type doesn't matter for a small retail trade — it all fills the same."** For a small order in a liquid name, the *direct* cost difference is tiny, true. But the *habit* matters: the trader who reflexively uses market orders pays the spread on every trade, and over thousands of trades that is a real drag. And the moment the order is large, or the name is thin, or the tape is fast, the order-type choice swings from pennies to thousands of dollars — as the trader in the opening story learned for \$9,000.

**"A hidden or dark order means nobody can ever read my intent."** Concealment hides the order from the *lit book*, but it does not make you unreadable. Sophisticated counterparties detect hidden liquidity by *pinging* — sending tiny IOC orders to probe for resting size, then inferring a hidden buyer from which pings fill. And the moment your hidden order *trades*, the print is reported, so the other side learns after the fact that someone with size was there. Concealment buys you a delay and a discount on what you reveal, not true invisibility — and against a determined, faster opponent, even the delay can be short.

## How it shows up in real markets

**Maker-taker pricing and the rebate war (US equities, ongoing).** Most US exchanges run a **maker-taker** fee model: they *pay you a rebate* for adding liquidity (posting a limit that rests) and *charge you a fee* for removing it (crossing the spread). As of mid-2026, NYSE Arca's schedule pays makers a rebate around \$0.0022 per share for adding liquidity and charges takers around \$0.0030 per share for removing it ([NYSE Arca fee schedule](https://www.nyse.com/publicdocs/nyse/markets/nyse-arca/NYSE_Arca_Marketplace_Fees.pdf)). That \$0.0052-per-share gap is precisely why **post-only** orders exist: on a 100,000-share fill, choosing to be a maker rather than a taker is worth about 100,000 × \$0.0052 ≈ **\$520** in fees alone, before any spread savings. Whole trading strategies — and the SEC's long-running debate about whether maker-taker distorts order routing ([SEC memo on maker-taker fees](https://www.sec.gov/spotlight/emsac/memo-maker-taker-fees-on-equities-exchanges.pdf)) — exist entirely because the order-type choice (maker vs taker) carries a hard-cash fee that the exchange sets.

**Dark pools and the conceal trade-off at scale (US, 2024).** The reveal-vs-conceal axis is so valuable that an entire parallel market exists for hiding. In Q1 2024, off-exchange venues — dark pools and internalizers, where orders are concealed from the lit book until after they print — handled about **44.5%** of all US equity trading volume ([market-structure data, 2024](https://www.rblt.com/market-structure-reports/let-there-be-light-us-edition-54)). Nearly half of all trading now happens in the dark. Institutions route there to hide large orders from exactly the front-running, fading, and pick-off dynamics described above — and they accept the dark pool's costs (uncertain fills, the risk of trading against a faster informed counterparty inside the pool) as the price of concealment. The size of this market is the clearest possible evidence that hiding your intent is worth real money, and that it is never free.

**Spoofing: weaponizing the visibility of resting orders (JPMorgan, 2020).** Because a resting limit *is* a signal, a manipulator can fake the signal. **Spoofing** is placing large visible limit orders you never intend to fill — to create a false impression of supply or demand — then canceling them once others react, while you trade the opposite way on your real (often hidden) order. In September 2020 the CFTC fined JPMorgan a record **\$920 million** for years of spoofing in precious-metals and Treasury futures, where traders entered hundreds of thousands of orders intending to cancel them before execution ([CFTC press release](https://www.cftc.gov/PressRoom/PressReleases/8260-20)). The defense, for you: a sudden wall of visible size that appears and vanishes is often *not* real liquidity — it is a fake signal designed to be read. Treat the visibility of the book as information that can be deliberately corrupted, and do not let a resting order you can see stampede you into a move.

**The flash crash and clustered stops (May 6, 2010).** Navinder Sarao, the "flash-crash trader," ran a dynamic-layering spoof in E-mini S&P futures — placing exceptionally large sell orders, modifying them hundreds of times to avoid execution, then canceling — and was ordered to pay over **\$38 million** in penalties and disgorgement ([CFTC press release](https://www.cftc.gov/PressRoom/PressReleases/7486-16)). His fake visible sell pressure helped tip the order book during the May 6, 2010 crash, when the Dow fell roughly 1,000 points in minutes as triggered stop orders and fleeing liquidity fed on each other. The episode is the textbook case of the stop-cascade: clustered conditional orders, all firing at once, turning a wobble into a plunge. It is the take-or-make dilemma and the stop-cluster target rolled into a single afternoon.

**The closing auction: everyone's order type collides (US equities, daily).** Every trading day ends with a single-price **closing auction** where an enormous volume — often 5–10% of the day's total — prints at one official closing price. Here every order type's strategic character is on display at once: market-on-close orders demand immediacy at any price, limit-on-close orders name a price and risk not participating, and the imbalance between them sets the close. The auction is the most strategic moment of the day precisely because the patience-vs-urgency and reveal-vs-conceal choices all settle in one instant. The series covers this in depth in [the opening and closing auction](/blog/trading/game-theory/the-opening-and-closing-auction-the-most-strategic-moment-of-the-day) — it is the single best place to watch order types behave as the strategic moves this post says they are.

## The playbook / How to play it

You now have the grid. Here is how to actually use it when your finger is over the button.

**Who is on the other side.** When you post a visible limit, your counterparty is whoever *chooses* to trade against your advertised price — and they choose to do so exactly when it benefits them, which is when it hurts you (adverse selection). When you send a market order, your counterparty is the market maker who quoted the spread, and they have just learned you were willing to pay it (they re-quote against your urgency). When you go dark or pegged, your counterparty is another hidden order you cannot see and cannot read — safer from the lit-book predators, but you have less information about who you traded with. *Always know which of these three you are facing,* because each reads you differently.

**The game you are in.** Two trade-offs, every time. **Urgency vs patience:** how much is filling *now* worth, against the spread and impact you pay to get it? **Reveal vs conceal:** how much would the other side move against you if they read your true size and price, against the queue priority, fee, or fill certainty you sacrifice to hide it? Your order type is your simultaneous answer to both questions, whether you thought about it or not.

**Your edge.** It comes from matching the move to the situation, while most traders reflexively use one order type for everything:

- *Small order, liquid name, no rush:* a passive limit or post-only, capturing the spread and the maker rebate. The direct cost is small but the habit compounds.
- *Large order, would move the market:* hide it — iceberg, midpoint peg, or work it through a dark venue or an execution algorithm — and accept the queue/fee/fill cost, because the impact you avoid is larger. Verify that inequality before you assume hiding helps.
- *Genuine urgency (a hedge, a stop, the close in seconds):* cross the spread with a market or IOC order, knowing you are paying for immediacy and leaking your urgency. That is a fine trade when immediacy genuinely outweighs the leak.
- *Want to stay competitive without committing to a price:* peg to the bid or the mid, revealing less conviction while never going stale.

**The invalidation.** Your order-type choice is wrong when the assumption behind it breaks. A passive limit is wrong the moment your fill probability collapses (fast tape, news pending) — then the chase cost dominates and you should have crossed. An iceberg is wrong when the order is too small to have moved the market anyway — then you paid queue cost for a secret nobody cared about. A stop at the obvious level is wrong because it is the most predictable order in the market — move it off the herd's level. Re-check the assumption, not just the price.

**The sizing and exit.** Scale your concealment to your footprint: the bigger your order relative to the book's depth, the more the reveal-vs-conceal trade-off tilts toward hiding, and the more it pays to slice the order over time rather than show it at once. And size the *urgency* to the cost of waiting: if missing the trade entirely is catastrophic, pay for immediacy; if it is merely inconvenient, be patient and let the cheaper fill come to you. The exit follows the same logic in reverse — getting *out* of a crowded position is the take-or-make dilemma at its most dangerous, because that is when everyone reaches for the market order at once. None of this is advice to trade; it is the structure of the decision, so that whichever move you make, you make it knowing what it costs and what it tells the other side.

The deeper truth under all of it: an order is the smallest unit of strategic communication in markets. You cannot trade without sending one, and you cannot send one without revealing *something* — at minimum, that you traded. The whole skill is controlling what you reveal, paying the minimum to conceal the rest, and reading the same signals coming off everyone else's orders. Master that, and you stop being the trader who paid \$9,000 to announce his own urgency.

## Further reading & cross-links

- [Every market is an auction: the double auction of the order book](/blog/trading/game-theory/every-market-is-an-auction-the-double-auction-of-the-order-book) — where these orders go and how bids and asks ladder into a continuous auction.
- [Mixed strategies and the value of being unpredictable](/blog/trading/game-theory/mixed-strategies-and-the-value-of-being-unpredictable) — the formal reason concealment has value: the other side cannot best-respond to a move they cannot read.
- [Kyle's model: how an informed trader hides in the noise](/blog/trading/game-theory/kyles-model-how-an-informed-trader-hides-in-the-noise) — the price-impact math behind why a large order must hide, and how much impact size creates.
- [Who is on the other side of your trade?](/blog/trading/game-theory/who-is-on-the-other-side-of-your-trade) — the counterparty map every order-type choice is a move within.
- *Next in the series:* [Execution as a game: VWAP, TWAP, and hiding from predators](/blog/trading/game-theory/execution-as-a-game-vwap-twap-and-hiding-from-predators) — how algorithms walk the patience-vs-urgency curve over time and conceal a large order from the front-runners.

*This post is educational, not financial advice. It explains the mechanics and strategy of order types; it does not recommend any trade, order, or position.*
