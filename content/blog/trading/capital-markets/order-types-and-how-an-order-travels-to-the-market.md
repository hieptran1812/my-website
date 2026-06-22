---
title: "Order Types and How an Order Travels to the Market: The Life of an Order, From Your App to the Book"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "What market, limit, and stop orders actually do, the maker-taker fee model, and the six hops an order makes from your phone to a settled trade."
tags: ["capital-markets", "order-types", "market-order", "limit-order", "stop-loss", "maker-taker", "best-execution", "smart-order-router", "secondary-market", "market-microstructure"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — When you tap "Buy," you are not buying from the company; you are sending a tiny instruction into a global relay race, and the *type* of instruction you choose decides whether you control the price or control the fill — you almost never get both.
>
> - A **market order** says "fill me now at whatever's available" — it guarantees you trade but not at what price (you pay the spread and risk slippage). A **limit order** says "fill me only at my price or better" — it guarantees the price but not that you trade at all.
> - A **stop order** is a trigger, not a price: it sits dormant until the stock hits a level, then fires a market (or limit) order — which is exactly why stops can fill far below where you set them in a fast-falling market.
> - You are either a **maker** (you post a resting order and *add* liquidity, sometimes earning a rebate) or a **taker** (you hit an existing order and *remove* liquidity, paying a fee). This maker-taker model literally pays people to quote.
> - The one fact to remember: your order passes through your **app → broker → smart order router → venue → matching engine → execution report → clearing**, and at every hop someone makes a decision that costs or saves you money. About **45%** of US equity volume never touches a public exchange at all.

## The morning a market order ate someone's lunch

On the morning of August 24, 2015, US stocks opened in chaos. Worries about China had built up over the weekend, and at 9:30 a.m. the market gapped down hard. In the first few minutes, hundreds of stocks and exchange-traded funds (ETFs) traded at prices that made no sense — some ETFs that should have been worth around \$100 briefly printed trades near \$5 or \$10. The values snapped back within minutes, but the trades were real. People who had set "stop-loss" orders to protect themselves woke up to discover those very orders had sold them out at the bottom of the flash, locking in losses of 30%, 50%, sometimes more, on positions that recovered before lunch.

Nobody hacked anything. No broker cheated. Those investors simply did not understand what their order *type* would actually do when the order book thinned out. They thought a stop-loss was a safety net. It was actually a trapdoor: a dormant instruction that, once triggered, became a *market* order — "sell at any price available" — into a book that, for those terrifying few minutes, had almost no buyers near the last price.

This post is about the difference between what you *think* you are telling the market and what you are *actually* telling it. We will walk an order through its whole life — from the tap on your phone to the moment cash and shares change hands two days later — and at each step we will ask the only question that matters: *who decides the price, who decides whether you fill, and who gets paid?* That is the secondary market in miniature. And the secondary market, remember, is the engine that makes the whole capital-markets machine run: nobody would fund a company's risky 30-year future by buying its newly issued shares if they couldn't turn around and sell those shares to someone else tomorrow morning. **Liquidity is the product, and an order is how you buy or sell it.**

![Order lifecycle pipeline from app to broker to router to venue to fill to clearing](/imgs/blogs/order-types-and-how-an-order-travels-to-the-market-1.png)

## Foundations: what an order actually is

Before any jargon, an everyday analogy. Consider a busy farmers' market for one specific thing — say, used bicycles of one exact model. Sellers stand on one side holding signs with the price they'll accept ("\$210," "\$215," "\$220"). Buyers stand on the other side holding signs with what they'll pay ("\$200," "\$205," "\$208"). The lowest seller's sign (\$210) and the highest buyer's sign (\$208) are the two prices everyone watches. The gap between them — \$2 — is the **spread**. Nothing trades *inside* that gap, because no buyer and seller agree yet. A trade happens only when someone crosses: a buyer agrees to pay \$210, or a seller agrees to take \$208.

That stack of signs is the **order book**. Each sign is a **resting limit order** — a standing offer to trade a set quantity at a set price. The market for a stock is exactly this, just electronic and updated thousands of times a second. (We cover the book's mechanics in depth in [inside an exchange: the matching engine and the order book](/blog/trading/capital-markets/inside-an-exchange-the-matching-engine-and-the-order-book); here we care about what *your* order does when it arrives.)

An **order** is a precise instruction with four ingredients:

1. **Side** — buy or sell.
2. **Quantity** — how many shares (e.g. 100, or 10,000).
3. **Price condition** — at the market (any price), or a limit (this price or better), or armed by a trigger (a stop).
4. **Handling flags** — how long it lives, whether it can fill partially, whether it's hidden, and so on (the "time-in-force" and special instructions).

Two foundational terms run through everything below:

- **The bid** is the highest price any buyer is currently willing to pay. **The ask** (or offer) is the lowest price any seller will accept. The **National Best Bid and Offer (NBBO)** is the best bid and best ask across *all* US exchanges combined — the official "current market" your broker is generally measured against.
- **Liquidity** is how easily you can trade a meaningful size without moving the price. A mega-cap like Apple is deeply liquid — thousands of shares rest at every penny. A tiny micro-cap may have a single buyer fifty cents below the last sale. Liquidity is not a fixed property; it evaporates in a panic, which is the whole moral of the August 2015 story.

One more piece of vocabulary before we go deep, because it underlies the maker-taker model and best execution alike: **adding** versus **removing** liquidity. When your order *rests* in the book waiting for someone else to trade against it, you have *added* a quote to the market — you've made it easier for the next person to trade. When your order *crosses the spread* and hits a resting quote, you have *removed* a quote — you've consumed liquidity that was sitting there. This add/remove distinction is not academic: it determines whether you pay a fee or earn a rebate, whether you signal your intentions to the market, and whether the exchange treats you as a supplier or a consumer of its core product. Hold onto it; half of this post hangs off it.

The single most important idea in this entire post is a trade-off you cannot escape:

> **You can have price certainty or fill certainty, but not both at once.** Every order type is just a different point on that spectrum.

It helps to see *why* this trade-off is fundamental rather than a quirk of any one exchange. A trade requires two parties to agree on a price *and* on the timing. If you fix the timing ("right now"), you must accept whatever price the other side currently demands — that's a market order. If you fix the price ("\$49.90, not a penny more"), you must accept whatever timing the other side eventually offers, which might be never — that's a limit order. There is no third option that pins down both, because the counterparty has a vote. The only way to get both price *and* timing certainty is to *be* the counterparty — to post a resting quote and wait, which is what a market maker does for a living, and which is why they get paid for it. Everyone else is choosing which half of the certainty they're willing to give up.

![Matrix of order types showing price certainty versus fill certainty](/imgs/blogs/order-types-and-how-an-order-travels-to-the-market-2.png)

## The core order types

### Market orders: take liquidity now, pay the spread

A **market order** says: *fill me immediately at the best price currently available, whatever that is.* It is the most aggressive instruction — you are *demanding* a trade right now and accepting whatever the book gives you. In market-structure language, a market order is **marketable**: it crosses the spread and removes (takes) liquidity that someone else posted.

What you give up is price certainty. You will fill — almost always instantly, in a liquid name — but you do not get to name the price. At minimum you pay the spread: if the bid is \$49.98 and the ask is \$50.00, a market buy pays \$50.00 and a market sell receives \$49.98. That two-cent gap is the **cost of immediacy**, and it goes to whoever was posting on the other side (often a market maker — see [market makers and the spread: who provides liquidity](/blog/trading/capital-markets/market-makers-and-the-spread-who-provides-liquidity)).

Worse, if your order is bigger than the size resting at the best price, it "walks the book": it eats the best ask, then the next-best, then the next, climbing to worse and worse prices until it's filled. The difference between the price you expected and the average price you actually got is called **slippage**.

#### Worked example: a market order walking the book

You market-buy 5,000 shares of a mid-cap stock. The offer side of the book looks like this:

- 1,000 shares offered at \$50.00
- 1,500 shares offered at \$50.05
- 1,500 shares offered at \$50.12
- 2,000 shares offered at \$50.20

Your 5,000-share order sweeps: 1,000 @ \$50.00, 1,500 @ \$50.05, 1,500 @ \$50.12, and 1,000 @ \$50.20. Total cost:

```
1000 x 50.00 = 50,000.00
1500 x 50.05 = 75,075.00
1500 x 50.12 = 75,180.00
1000 x 50.20 = 50,200.00
-------------------------
5000 shares  = 250,455.00
avg price = 250455 / 5000 = 50.091
```

You paid an average of \$50.091, even though the screen showed \$50.00 when you clicked. That \$0.091 × 5,000 = **\$455** of slippage is the price of demanding an instant fill of size the top of the book couldn't absorb. The intuition: a market order is wonderful in a deep, liquid name and dangerous in a thin one — its cost is invisible until the book is too shallow to swallow your size.

There is a deeper cost to market orders that the slippage number understates: **market impact**. When you sweep the book, you don't just pay worse prices on the way up — you also *move the price* and *signal* to the rest of the market that a large buyer is active. Other participants' algorithms detect the sweep within milliseconds and may pull their offers higher or buy ahead of you, anticipating more demand. For a 100-share retail order this is irrelevant noise; for an institution trying to buy 500,000 shares, naively sending one giant market order would be financial self-harm — they'd move the price against themselves and pay a fortune. That's why large orders are *worked* over time in small slices (using algorithms with names like VWAP and TWAP, and the iceberg and peg flags we'll meet below) rather than dumped at market. The market order is the bluntest instrument in the toolkit: maximally fast, maximally aggressive, and maximally expensive when used at size.

When *should* you use a market order, then? When the value of certainty-of-fill genuinely exceeds the spread cost: exiting a position before a binary event (an earnings release, an FDA decision), trading a deeply liquid name where the spread is a single penny, or closing a small position where the dollar slippage is trivial. The discipline is to ask, every time, "is being filled *this second* worth the spread plus the slippage I might eat?" In a mega-cap, almost always yes. In a micro-cap, almost never.

### Limit orders: name your price, wait

A **limit order** says: *fill me only at my limit price or better — never worse.* A limit buy at \$49.90 will fill at \$49.90 or less, never more. A limit sell at \$50.10 will fill at \$50.10 or more, never less. You have bought total price certainty.

The catch is the mirror image of the market order: you've given up fill certainty. If you post a limit buy at \$49.90 while the stock trades at \$50.00, your order just *sits* in the book as a resting order. It fills only if a seller comes down to \$49.90. If the stock runs away upward, you never trade — you "missed the fill." Your protection against a bad price is also your exposure to no trade at all.

A limit order can be either side of the liquidity divide. If you post it *away* from the current price (a buy below the bid, a sell above the ask), it rests in the book and *adds* liquidity — you become a **maker**. If you post a limit buy *at or above* the current ask, it's immediately **marketable** and behaves like a market order with a price ceiling — it *takes* liquidity. So "limit order" describes the price rule, not whether you add or remove; that depends on where you place it.

Where a resting limit order sits in the queue matters enormously, and it's governed by **price-time priority** — the matching rule used by nearly every major exchange. First, orders sort by price: the most aggressive buy (highest bid) and most aggressive sell (lowest offer) are first in line to trade. Second, among orders at the *same* price, the one that arrived *earliest* trades first. This is why traders care about being early to a price level: if 50,000 shares are bid at \$49.90 ahead of you, an incoming sell of 10,000 fills *them*, not you, and you keep waiting. Time priority is also why microseconds and co-location (putting your servers physically next to the exchange's) became worth billions to high-frequency firms — being first in the queue at a price is a real, monetizable edge. For a retail limit order, the practical takeaway is humbler: a limit order at the inside quote is not guaranteed to fill *even if the stock trades at your price*, because the orders ahead of you in the queue may absorb all the volume. Your fill probability depends on your queue position, not just your price.

A limit order also leaves a **footprint**. A resting buy order at \$49.90 is visible in the public book (unless you hide it — see the iceberg flag below), and it tells the market "someone wants to buy here." Sophisticated participants read the book and may trade around your visible order. This is the cost of posting: you provide a free option to the rest of the market. If the stock is about to fall, the people who *know* that will trade against your resting bid before it drops — a phenomenon called **adverse selection**, and the central risk every market maker manages. The formal models of this live in [the market-making simulator](/blog/trading/quantitative-finance/market-making-simulator-quant-research); here it's enough to know that posting liquidity is not free money — you're being paid the spread and the rebate precisely *because* you bear adverse-selection risk.

#### Worked example: market versus limit on the same name

Same mid-cap, trading \$49.98 bid / \$50.00 ask. You want 5,000 shares. Two paths:

- **Market order:** fills immediately at an average of \$50.091 (from the sweep above). Cost: \$250,455. You own the stock in under a second.
- **Limit order at \$50.00:** you fill the 1,000 shares resting at \$50.00 (\$50,000), and the remaining 4,000 shares rest in the book at \$50.00 *unfilled*. If the stock then ticks up to \$50.30 and never looks back, those 4,000 shares never fill. You bought 1,000 shares for \$50,000 — and *missed* 4,000 shares that you'd now have to chase at \$50.30, costing an extra \$0.30 × 4,000 = \$1,200 versus the original ask if you re-enter.

So the limit order saved you the \$455 of slippage *on the part that filled*, but exposed you to a \$1,200 opportunity cost on the part that didn't. The intuition: limit orders win when you're patient and the price is likely to come to you; market orders win when *being filled* is worth more than shaving a few cents — like exiting a position before an earnings report.

### Stop and stop-limit orders: a trigger that arms an order

A **stop order** is not a price you trade at — it is a *trigger* that arms a different order. A **stop-loss** sell at \$50 means: "do nothing while the stock is above \$50; the moment it trades at or below \$50, send a *market* sell order." A **stop-limit** is the same trigger, but it arms a *limit* order instead of a market order: "if it hits \$50, send a limit sell at \$49.50" (you set both the stop and the limit).

This distinction is where people get hurt. A plain stop converts to a *market* order at the trigger, so it inherits all of the market order's price uncertainty — and it triggers precisely when the market is moving against you, which is exactly when liquidity is thinnest and slippage is worst. That is what happened in August 2015: stops triggered into a gap and filled far, far below the stop price.

A stop-*limit* protects you from a runaway-bad fill (you set a floor) but reintroduces fill uncertainty: if the stock gaps straight through your limit, the limit order never executes and you're left holding a falling position with no protection at all. There is no free lunch — only a choice of which risk you'd rather carry.

#### Worked example: a stop-loss that fills in a gap

You hold 1,000 shares bought at \$55. To "limit your loss," you place a stop-loss sell with a stop at \$50, expecting to lose at most about \$5/share (\$5,000). Overnight, the company reports terrible news. The next morning the stock *opens* at \$43 — it never traded between \$50 and \$43, it gapped.

Your stop triggers (price is below \$50) and becomes a market sell. It fills at the opening price, roughly \$42 after a few cents of slippage into a thin pre-open book. Your realized loss:

```
buy:  1000 x 55 = 55,000
sell: 1000 x 42 = 42,000
loss = 55,000 - 42,000 = 13,000
```

You lost **\$13,000**, not the \$5,000 you "planned." The stop did exactly what you told it — sell at market once \$50 traded — but a stop cannot defend a price that the market skips over. The intuition: a stop-loss controls *when* you sell, never *at what price*; in a gap, "when" and "what price" come apart violently. A stop-limit at, say, \$48 would have *not filled* here (price gapped below \$48), leaving you still holding — which might be better or worse, but is a genuinely different bet.

![Timeline of a stop-loss triggering and filling in an overnight gap down](/imgs/blogs/order-types-and-how-an-order-travels-to-the-market-6.png)

There is a subtler point about *how* a stop triggers. Brokers and exchanges differ on the **trigger condition**: some arm the stop on the first *trade* at or through the level, others on the *quote* (the bid or ask) touching it, and a few wait for a trade to *print*. In a fast market these can differ by pennies — and a single bad print can trigger a wave of stops that wouldn't have triggered on quotes. There is also the matter of *where* the stop lives: a stop held on the exchange behaves differently from a "synthetic" stop held on your broker's servers, which only sends the order once the broker sees the trigger. The synthetic stop is invisible to the market (it can't be "hunted" by someone reading the book) but adds a few milliseconds of latency between trigger and order. For a long-term investor those milliseconds are irrelevant; for an active trader in a volatile name, they are the difference between a clean exit and a chase. None of this changes the core lesson — a stop is a delayed market (or limit) order — but it explains why two people with "the same" stop can get very different fills.

A few related guardrails are worth naming because they protect you when an order type would otherwise run wild. **Limit up / limit down (LULD)** bands halt trading in a single stock for five minutes if it moves more than a set percentage (5%, 10%, or 20% depending on the tier) away from a rolling reference price within a short window. These bands were strengthened directly because of the August 2015 episode: they are a circuit breaker against exactly the "market order sweeps an empty book" failure mode. **Marketable limit orders with a collar** are another defense — some brokers automatically convert a naked market order into a limit order a few percent away from the last price, so a fat-finger or a flash can't fill you at an absurd level. If you only remember one operational habit from this section: in any name you don't trade every day, prefer a marketable limit (a limit set at or just through the current quote) over a raw market order. You keep almost all of the fill certainty and cap the disaster scenario.

### Time-in-force and handling flags

Beyond the three core types, a handful of flags control *how* an order lives and fills. These are the knobs professionals actually turn:

- **Day** — the order is good only until the close of today's session; unfilled, it's cancelled. (The default at most brokers.)
- **GTC (Good-'Til-Cancelled)** — stays alive across days (brokers usually cap it at 30–90 days) until it fills or you cancel it.
- **IOC (Immediate-Or-Cancel)** — fill whatever you can *right now*, cancel the rest. Partial fills allowed. Used to grab available liquidity without leaving a resting order that signals your intent.
- **FOK (Fill-Or-Kill)** — fill the *entire* quantity immediately or cancel the whole thing. No partial fills. Used when a partial position is useless to you.
- **AON (All-Or-None)** — fill the full size or none, but (unlike FOK) it can wait in the book for that full size to become available.
- **Hidden / iceberg** — a large order whose displayed size is only a small "tip" (the iceberg) while the bulk stays hidden; as the tip fills, more surfaces. Used to work big size without showing the whole hand to the market.
- **Peg** — an order that automatically re-prices to track a reference, e.g. "peg to the NBBO midpoint" so it always sits at the middle of the spread.
- **MOC (Market-On-Close)** — execute at the official closing price in the closing auction. Index funds rebalancing to a benchmark live here, because their job is to match the closing print, not to beat it.

#### Worked example: an IOC partial fill

You send an **IOC buy** for 10,000 shares at a \$50.00 limit. At that instant the book shows 3,000 shares offered at \$50.00 and the next offer is at \$50.04. The IOC fills the 3,000 shares it can get at \$50.00 or better, then *cancels the remaining 7,000* rather than resting or chasing up to \$50.04:

```
filled:    3,000 @ 50.00 = 150,000
cancelled: 7,000 (no resting order left)
```

You spent \$150,000 and own 3,000 shares; the other 7,000 simply vanished from the market with no footprint. The intuition: an IOC is for a trader who wants *available liquidity at a price, with zero information leakage* — it never sits in the book advertising "someone wants 7,000 more here," which would invite others to trade ahead of you.

These flags compose, and the compositions are where institutional trading lives. An iceberg order with a peg-to-midpoint price quietly works a large position at the middle of the spread while showing only a tiny tip; a series of IOCs sweeps available liquidity across venues without leaving resting footprints; an MOC order parks size for the closing auction where the day's largest pool of liquidity gathers. Each flag is a tool for managing the same two adversaries every trader faces: **the spread** (the explicit cost of immediacy) and **information leakage** (the implicit cost of revealing your intent). A retail investor rarely needs more than day and GTC. But understanding that the flags *exist* explains a lot about why the order book behaves the way it does — much of the size you'd want to trade against is deliberately hidden, pegged, or routed off-exchange, so the visible book is only the surface of the real liquidity.

## Maker vs taker: who adds liquidity, and who pays

Every fill has two sides, and modern US equity markets label them. The **maker** is the side whose order was *resting* in the book first — they *made* (provided) the liquidity. The **taker** is the side whose order arrived and *crossed the spread* to hit that resting order — they *took* (removed) the liquidity. A resting limit order is the classic maker; a market order or marketable limit order is the classic taker.

Why does the market bother to label this? Because of the **maker-taker fee model**, one of the strangest and most consequential designs in market structure. Most US exchanges *pay you a rebate* to post a resting order (to make liquidity) and *charge you a fee* to take it. The numbers are tiny per share but enormous in aggregate: a typical exchange might charge the taker about \$0.0030 per share and pay the maker a rebate of about \$0.0025 per share, pocketing the ~\$0.0005 difference as its revenue.

The logic is that liquidity is valuable and scarce, so the exchange subsidizes the people who supply it (market makers, high-frequency firms) by taxing the people who consume it. This is why a market maker can profitably quote a one-cent spread on a mega-cap: they earn the spread *plus* the rebate on the enormous volume they post.

But maker-taker has a dark side that drives much of the industry's structural controversy. Because exchanges compete for order flow, they compete on rebates — and a broker routing your order has an incentive to send it to whichever venue pays *the broker* the fattest rebate, which is not necessarily the venue that fills *you* best. This is a textbook conflict of interest, and it's why a parallel "inverted" or **taker-maker** model also exists (some venues *pay* takers and *charge* makers, to attract aggressive order flow). The result is a thicket of fee schedules that a smart order router must navigate in microseconds, weighing displayed size, fees, rebates, and fill probability at a dozen venues at once. Regulators have run pilot programs to test whether capping or banning rebates would tighten spreads or change routing — evidence so far is mixed, which is itself telling: the rebate is so woven into the economics of quoting that removing it changes who posts, how tightly, and where. The honest summary is that maker-taker *does* subsidize tight quotes (good for everyone trading) *and* creates routing conflicts (bad for trust), and reasonable people disagree about the net.

There's a second-order effect worth naming: maker-taker fees mean the **effective spread you pay is not the quoted spread**. If you take liquidity at a one-cent spread but pay a \$0.0030 taker fee, your real round-trip cost is wider than a penny. Conversely, if you make and earn a rebate, your effective cost is narrower than the quoted spread. The quoted bid-ask is the *advertised* price of immediacy; the maker-taker fee is the *hidden* adjustment. Professionals measure execution quality in **effective spread** and **realized spread** (which also accounts for where the price goes right after the trade) precisely because the quoted spread lies about the true cost once fees are included.

![Maker versus taker before and after comparison with rebate and fee](/imgs/blogs/order-types-and-how-an-order-travels-to-the-market-4.png)

#### Worked example: maker rebate vs taker fee on a 10,000-share order

You need to buy 10,000 shares of a liquid stock quoting \$49.99 / \$50.00. Two ways to get filled, and the exchange fees flip your economics:

- **Take liquidity** (marketable buy that lifts the \$50.00 offer): you pay \$50.00 × 10,000 = \$500,000 *plus* a taker fee of \$0.0030 × 10,000 = **\$30**. All-in cost: \$500,030. Filled instantly.
- **Make liquidity** (rest a limit buy at \$49.99 and wait to be hit): if it fills, you pay \$49.99 × 10,000 = \$499,900 *and earn* a maker rebate of \$0.0025 × 10,000 = **\$25**. All-in cost: \$499,875.

The difference is \$500,030 − \$499,875 = **\$155** in your favor by making — \$100 from buying a penny cheaper plus \$55 from the rebate-vs-fee swing — *if* the resting order fills. The intuition: maker-taker turns the price-vs-fill trade-off into a money-vs-fill trade-off too. Posting is cheaper *and* paid, but only if the market comes to you; demanding immediacy costs the spread *and* the fee. For a retail investor the per-share amounts are trivial; for a firm trading hundreds of millions of shares a day, rebates are a core revenue line and *drive routing decisions* — which is where best execution gets complicated.

## The order lifecycle: every hop, and what happens there

Now the relay race. When you tap "Buy 100 shares," the order does not teleport to an exchange. It passes through a chain of handlers, each adding a decision, a check, or a delay measured in microseconds. Here is each hop.

**1. Your app (the order entry).** Your broker's app or website packages your instruction into a standardized electronic message (the industry uses a protocol called FIX). It validates the obvious things locally — you can't sell shares you don't have in a cash account, the symbol exists, the quantity is a whole number — then transmits it to the broker's systems.

**2. Your broker (risk checks and the routing decision).** The broker is the regulated entity that *holds your account* and is legally responsible for your order. It runs **pre-trade risk checks** mandated by regulators (the "market access rule"): is this order within your buying power? Is it absurdly large (a fat-finger 1,000,000-share order where you meant 1,000)? Would it breach a position limit? If it passes, the broker must decide *where to send it* — to which exchange, wholesaler, or dark pool. This routing decision is governed by **best execution** (next section) and is where most of the hidden economics live.

**3. The smart order router (SOR).** Modern markets are *fragmented*: there are roughly a dozen+ public US stock exchanges and dozens of off-exchange venues, all trading the same stocks. The **smart order router** is software that, in microseconds, decides how to slice and route your order across venues to get the best result — checking which venue has size at the best price, accounting for fees and rebates, and sometimes splitting one order across several venues simultaneously. (We cover the fragmented landscape in [lit markets, dark pools, and the fragmented tape](/blog/trading/capital-markets/lit-markets-dark-pools-and-the-fragmented-tape).)

What does the SOR actually weigh? In the moment your order arrives, it sees a snapshot of every venue's displayed quote, knows each venue's fee/rebate schedule, and has statistical models of each venue's *hidden* liquidity and fill probability. It then solves a tiny optimization: given that I want to buy 5,000 shares, which combination of venues, in which order, at which price levels, minimizes my expected total cost — including the chance that quotes move while I'm routing? If the best offer is split across three exchanges (1,000 shares each) plus likely hidden size in a dark pool, the SOR may fire simultaneous IOC slices at all of them. The hard problem is that the moment it takes liquidity on one venue, the others may react — so routing is a game against other algorithms, not a static lookup. This is the operational heart of a *fragmented* market: fragmentation creates the need for the SOR, and the SOR's quality is a real, if invisible, component of your execution price.

#### Worked example: a smart order router splitting an order

You market-buy 4,000 shares. The SOR sees the offer side spread across venues at the \$50.00 NBBO: Exchange A shows 1,500 shares, Exchange B shows 1,000, and a dark pool historically fills ~1,500 at the midpoint \$49.995. The router fires three slices at once:

```
Exchange A: 1,500 @ 50.00  = 75,000.00  (taker fee ~0.0030/sh = $4.50)
Dark pool:  1,500 @ 49.995 = 74,992.50  (midpoint, no taker fee)
Exchange B: 1,000 @ 50.00  = 50,000.00  (taker fee ~0.0030/sh = $3.00)
-----------------------------------------------------------------
total:      4,000          = 199,992.50  + ~$7.50 fees
avg price = 199992.50 / 4000 = 49.998
```

By pulling 1,500 shares from the dark pool at the midpoint, the SOR beat the \$50.00 lit offer on 38% of the order, landing an average of \$49.998 — *better* than the displayed best offer. The intuition: a good router doesn't just take the visible best price; it harvests hidden midpoint liquidity and splits to avoid moving any one venue. The same 4,000-share market order, routed naively to a single exchange, would have walked that venue's book to a worse average.

**4. The venue and its matching engine.** The order arrives at a venue — a lit exchange like Nasdaq or NYSE, a dark pool, or a wholesaler. The venue's **matching engine** is the program that pairs buy and sell orders by strict rules, almost always **price-time priority**: the best price trades first, and among orders at the same price, the one that arrived earliest trades first. If your order is marketable, it matches against resting orders instantly; if it's a resting limit order, it joins the book and waits its turn.

**5. The execution report (the fill).** When a match happens, the venue sends back an **execution report**: filled quantity, price(s), and a timestamp. Your broker relays it to your app, and you see "Filled: 100 @ \$50.00." The trade is also reported to the public **consolidated tape** so the whole market sees the print. At this instant you are economically committed — but no shares or cash have actually moved yet.

A note on *what the broker is actually doing* at hop 2, because "broker" hides a lot. Most retail brokers do not route your order to an exchange themselves at all — they hand it to a **wholesaler** (a large electronic market maker) or run it through their own internal systems first, and only what can't be filled there reaches a public venue. The broker is simultaneously your agent (legally bound to seek your best execution), a risk manager (it must not let your order blow through its own capital limits, because *the broker* is on the hook to the clearinghouse if you default), and a business optimizing its own routing economics. Those three roles mostly align, but not perfectly — which is the entire subject of the best-execution section below. The broker also handles the *give-up*: it tells the clearing system which clearing member stands behind your trade, so the CCP knows whose collateral backs it.

**6. Clearing and settlement.** The trade now enters the *plumbing*. A **central counterparty (CCP)** — for US equities, the National Securities Clearing Corporation — steps between buyer and seller through **novation**, becoming the buyer to every seller and the seller to every buyer, so neither side bears the other's default risk. It nets down the day's millions of trades to tiny net obligations, and on **T+1** (one business day after the trade, since the US shortened the cycle in May 2024) the shares move to your account and the cash moves out, via the Depository Trust Company. Only now is the trade truly *done*. (The exchange-and-clearinghouse machinery is detailed in [stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses).)

That entire journey — six hops, multiple venues, a matching engine, a CCP — typically completes its *execution* portion in well under a second, often in microseconds at the venue. The settlement tail takes a day. And at every hop, someone is making a decision that affects your price.

## Best execution: the broker's duty and where PFOF fits

Here's the question that should bother you: if the broker decides *where* to route your order, and venues *pay rebates* and *some buyers pay brokers for order flow*, how do you know the broker is routing to get *you* the best deal rather than the broker the best rebate?

The answer, in principle, is **best execution** — a legal duty. A broker must "use reasonable diligence" to obtain the most favorable terms reasonably available for a customer's order. "Favorable" is not just price; it's a blend of:

- **Price** — can you fill at or inside the NBBO?
- **Speed** — how fast does it fill?
- **Likelihood of execution** — will it actually fill, and fully?
- **Total cost** — fees, and the chance of price improvement.

The broker must regularly evaluate its routing against these factors and document it. In practice, best execution is a *process* duty, not a guarantee of the best possible price on every single order — which leaves room for genuine tension.

To make this auditable, US rules require disclosure: brokers and venues publish standardized reports (historically known by their rule numbers, 605 and 606) showing execution quality and where orders were routed. A "606 report" tells you which venues your broker sent orders to and how much PFOF it received from each; a "605 report" tells you a venue's execution-quality statistics — effective spreads, price improvement rates, speed. These reports are dense and rarely read by retail investors, but they are the mechanism by which best execution is supposed to be *checkable* rather than merely promised. The existence of mandatory disclosure is itself the regulatory philosophy of this whole series in miniature: you don't ban the conflict, you *force it into the sunlight* and let competition and oversight discipline it. Whether that's sufficient is exactly what the PFOF debate is about.

That tension has a name: **payment for order flow (PFOF)**. Some "wholesalers" (large electronic market makers) pay retail brokers a small sum to send them the broker's customer orders, rather than routing those orders to a public exchange. The wholesaler then fills the order itself, often offering a hair of **price improvement** over the NBBO (say, \$49.999 instead of the \$50.00 ask), and profits from the spread and the order flow's predictability.

The defenders' case: retail orders are "uninformed" (not from someone with inside knowledge), so wholesalers can fill them at a better price than the public quote *and* still profit, and the PFOF revenue is what lets brokers offer \$0 commissions. The critics' case: PFOF creates a conflict — the broker is paid by the venue, not the customer, so the broker's incentive to route for *its* best rebate can collide with its duty to route for *your* best price; and routing retail flow off-exchange to wholesalers is a big reason so much volume never reaches a public, price-setting exchange.

![US off-exchange share of equity volume over time](/imgs/blogs/order-types-and-how-an-order-travels-to-the-market-5.png)

The chart above shows the long climb: off-exchange (dark pools plus wholesaler internalization) has grown from around 30% of US equity volume in 2010 to roughly 47% by 2024. A near-majority of trading no longer happens on the lit exchanges whose quotes *set* the public price — a structural fact that PFOF and best-execution debates orbit around.

#### Worked example: price improvement vs the rebate the broker keeps

You market-buy 200 shares with the NBBO at \$49.98 / \$50.00. Your zero-commission broker routes to a wholesaler that fills you at \$49.995 — \$0.005 of price improvement per share:

```
your improvement: 200 x 0.005 = $1.00 saved vs the $50.00 ask
```

You saved a dollar versus the public offer, and paid no commission — genuinely good for you on this order. Meanwhile the wholesaler captured most of the \$0.02 spread (it sold to you near \$49.995 having likely bought near \$49.98) and paid your broker a fraction of a cent per share in PFOF. Everyone got a sliver. The intuition: best execution and PFOF can coexist for small, uninformed orders *and* still distort the bigger picture — the question is never "did this one order improve?" but "is the routing system, in aggregate, working for customers or for the intermediaries' rebate math?"

![Where US equity orders execute, lit exchange versus off-exchange](/imgs/blogs/order-types-and-how-an-order-travels-to-the-market-7.png)

## Common misconceptions

**"A market order fills at the price I see on the screen."** No. The price you see is the *last trade* or the current NBBO, and by the time your order arrives the book may have moved or your size may exceed what's resting there. In a liquid mega-cap the slippage is a penny; in a thin name it's the \$455 from our worked example. The screen price is an *estimate*, never a promise — only a limit order makes the price a promise.

**"A stop-loss caps my loss at the stop price."** No — and this myth costs people real money. A stop-loss becomes a *market* order at the trigger; it caps *nothing* about the fill price. In a gap or a flash crash it can fill far below the stop, exactly as it did in August 2015. The spread of these losses is widest in illiquid names. If you want a price floor, use a stop-*limit* — and accept that it might not fill at all.

**"Limit orders always get me a better price."** Only on the shares that fill. A limit order's hidden cost is the trade you *miss*. Set the limit too tight and a fast market leaves you behind, forcing you to chase at a worse price than a market order would have paid in the first place. Price certainty has a fill-risk premium.

**"Free trading means there are no costs."** Zero commission is not zero cost. You still pay the spread, you may bear slippage, and the broker is often paid by a wholesaler (PFOF) to handle your order. The cost moved from a visible commission line to invisible spread and routing economics. "Free" describes the commission, not the trade.

**"All my volume hits a real exchange."** Roughly 45–47% of US equity volume executes *off* the lit exchanges — in dark pools and wholesaler internalization. Your retail order, in particular, very often never touches Nasdaq or NYSE at all; it's filled by a wholesaler.

**"A limit order at the inside quote will definitely fill if the stock trades there."** No — time priority means there may be a queue ahead of you at that price. If the stock prints a few trades at \$49.90 but the volume is absorbed by orders that arrived before yours, you can sit unfilled while the tape shows trades at exactly your price. Fill probability is about your *queue position*, not just the price you named.

**"Order types are the same at every broker and exchange."** Mostly, but not entirely. The available flags, the stop *trigger condition* (trade vs quote), whether stops are held at the exchange or synthetically on the broker's servers, and the routing logic all vary. Two investors entering "the same" stop-loss at different brokers can get materially different fills. Read your broker's order-handling disclosures before you rely on an exotic flag.

## How it shows up in real markets

**The August 2015 ETF flash (the stop-loss massacre).** When the market gapped down at the open, market-wide circuit breakers and the sudden evaporation of liquidity meant many ETFs had almost no resting buy orders near fair value for a few minutes. Stop-loss orders triggered en masse, converted to market sells, and swept down through an empty book — printing trades 30–50% below where the ETFs traded minutes later. The regulatory response reshaped the "limit up / limit down" bands and made the industry far more cautious about plain stop orders. The lesson the market learned: *a market order is only as safe as the liquidity behind it, and a stop is just a delayed market order.*

**The shift to T+1 (May 28, 2024).** For decades US equities settled at T+2 (two days after the trade). On May 28, 2024, the US moved to **T+1** — one day. Why it matters for an order's life: the faster the settlement, the less time a CCP (and your broker) bears the risk that a counterparty fails between trade and settlement, which reduces the collateral the system must post. It also compressed the operational window for everything downstream of your fill — currency conversion for foreign buyers, securities lending recalls, error corrections — into a single overnight. Your *execution* didn't change, but the plumbing that finishes the job got a day shorter. (The settlement-cycle history is part of why this series treats clearing as a first-class engine.)

**The 2021 meme-stock PFOF spotlight.** When trading in names like GameStop exploded in early 2021, a retail broker briefly restricted buying — and the ensuing furor put *payment for order flow* on the front page. Regulators examined whether the routing of retail orders to a handful of wholesalers, and the PFOF those wholesalers paid, served customers' best execution or the brokers' revenue. No single villain emerged, but the episode crystallized the core tension of this post: the order you send travels through intermediaries whose incentives are not automatically aligned with yours, and "free" trading is paid for somewhere in that chain.

**A block trade and why order types scale.** Now flip from retail to institutional. Suppose a pension fund must buy a \$40 million position — roughly 800,000 shares of a \$50 stock. Sending one market order would be malpractice: it would sweep the book up several percent and signal the whole market. Instead the fund's trader breaks it into hundreds of child orders worked over hours, mixing passive limit orders (to earn the spread and rebate when the market comes to them) with opportunistic IOCs and dark-pool midpoint orders (to harvest hidden liquidity without showing the hand), and perhaps a negotiated block crossed off-exchange with a single counterparty. The fund might also use a VWAP algorithm that aims to match the volume-weighted average price over the day, so its execution looks "average" rather than aggressive. Every concept in this post — maker vs taker, the flags, the SOR, off-exchange venues — is a tool the trader composes to move \$40 million without the market noticing. The retail investor's single tap and the institution's day-long campaign are the *same machinery* at different scales; the institution just has more knobs and a bigger reason to turn them carefully.

**Why liquidity is the secondary market's whole product.** Tie it back to the spine. The reason an investor will buy a company's freshly issued shares (the *primary* market) is the confidence they can sell those shares to someone else tomorrow at a fair price with low friction (the *secondary* market). Order types, the maker-taker model, smart routing, and best execution are all just machinery for delivering that confidence — for keeping the spread tight and the fill reliable. When that machinery fails (August 2015, a flash crash), liquidity vanishes, spreads blow out, and the secondary market briefly stops doing its job — which is exactly why the next morning's IPO would have struggled to price. **Secondary-market liquidity is what makes primary issuance possible**, and an order is the atom of that liquidity.

![US equity market cap by year showing the size of the secondary market](/imgs/blogs/order-types-and-how-an-order-travels-to-the-market-8.png)

The scale here is the point: the US equity market alone grew from roughly \$26 trillion in 2014 to about \$58 trillion in 2024 of tradable value. Every dollar of that is only *worth* its quoted price because someone can sell it via an order that travels the chain we just walked — and the spread they'll pay to do so depends entirely on the liquidity tier of the name.

![Bid-ask spread by liquidity tier in basis points](/imgs/blogs/order-types-and-how-an-order-travels-to-the-market-3.png)

The spread-by-tier chart makes the order-type advice concrete. In a mega-cap with a ~1 basis-point spread, a market order costs you almost nothing — use it freely. In a micro-cap with an ~80 basis-point spread (0.80%), a market order hands away nearly a full percent before the stock even has to move against you, and a market order of size will walk a thin book catastrophically. *The right order type is a function of the liquidity tier*: market orders for the deep names, patient limit orders for the thin ones, and great caution with stops in anything illiquid.

## The takeaway: how to read your own order

Once you see an order as an instruction traveling a chain of decision-makers, the practical rules fall out on their own:

1. **Choose your certainty deliberately.** Decide before you click whether you care more about *being filled* (market / marketable limit) or about *the price* (resting limit). You are picking which risk to bear; there is no order type that removes both.

2. **Match the order type to the liquidity tier.** Market orders are fine in deep, penny-spread names and dangerous in thin ones. In an illiquid stock, a limit order isn't caution — it's the only sane choice, because a market order will walk a sparse book.

3. **Respect what a stop really is.** A stop is a *delayed market order*; a stop-limit is a *delayed limit order with a floor that might never fill.* Neither caps your loss in a gap the way the name suggests. Size your stop distance for the volatility of the name, not for your comfort.

4. **Know that "free" is paid somewhere.** Zero commission moved the cost into the spread and the routing economics. That's not a scandal by itself — price improvement is real — but it means *best execution is a process you're trusting*, and it's worth understanding who pays your broker.

5. **Remember the chain.** App → broker → router → venue → matching engine → fill → clearing. Every hop is a decision and a potential cost. The more you understand the chain, the less the market can surprise you with a fill you didn't expect.

6. **Measure the right cost.** The visible commission is usually zero and the quoted spread is usually small, but your true execution cost is the *effective spread* — the quoted spread adjusted for exchange fees, slippage, and price improvement. A "free" trade in a thin name can quietly cost you a full percent; a well-routed trade in a deep name can fill *inside* the quote. Judge by the fill you got versus the NBBO at the moment you sent the order, not by the commission line.

A final reframing of all six rules: each one is really a question about *who is on the other side of your order and what they're being paid.* When you take liquidity, a maker (often a market maker or wholesaler) is paid the spread plus a rebate to be your counterparty, and you pay a fee for the privilege of immediacy. When you make liquidity, *you* are the one being paid the spread — and bearing the adverse-selection risk that the person who finally trades against you knows something you don't. Best execution is the rulebook that tries to keep the intermediaries between you and that counterparty honest; the maker-taker schedule is the incentive system that determines how tightly anyone is willing to quote; and the order type is your one lever on the whole apparatus. You can't change the fee schedules or the routing tables, but you *can* choose, every single time, whether you're demanding immediacy or offering patience — and that single choice is the largest determinant of your execution cost that's actually within your control.

The deepest point is the one the spine keeps returning to: an order is how an individual saver touches the great machine that turns savings into investment. When you post a limit order, you are *providing* the liquidity that lets the next person trade — you are, for a moment, a tiny market maker. When you send a market order, you are *consuming* it. The whole secondary market is just billions of these instructions colliding under a set of priority rules, and the quality of that collision — tight spreads, reliable fills, honest routing — is precisely what gives the primary market the confidence to keep funding the future. Understand your order, and you understand the smallest working part of capitalism's allocation engine.

## Further reading & cross-links

- [Inside an exchange: the matching engine and the order book](/blog/trading/capital-markets/inside-an-exchange-the-matching-engine-and-the-order-book) — where your order goes to match, and how price-time priority actually works.
- [Market makers and the spread: who provides liquidity](/blog/trading/capital-markets/market-makers-and-the-spread-who-provides-liquidity) — the counterparty on the other side of your market order, and why the spread exists.
- [Lit markets, dark pools, and the fragmented tape](/blog/trading/capital-markets/lit-markets-dark-pools-and-the-fragmented-tape) — the venue landscape your smart order router navigates, and why ~45% of volume is off-exchange.
- [Stock exchanges and clearinghouses](/blog/trading/finance/stock-exchanges-and-clearinghouses) — the clearing-and-settlement plumbing that finishes your trade on T+1.
- [Order book simulator (quant research)](/blog/trading/quantitative-finance/order-book-simulator-quant-research) — build the book yourself and watch orders match.
