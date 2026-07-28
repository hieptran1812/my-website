---
title: "OTC Desks: Moving Size Without Moving Price"
date: "2026-07-27"
publishDate: "2026-07-27"
description: "A build-from-zero guide to crypto's over-the-counter block market: why a $50M market order destroys itself on a thin book, how principal and agency desks price the same trade differently, how a desk hedges and unwinds what it just bought — and why that unwind is the channel through which invisible off-book flow becomes very visible on-book pressure."
tags: ["crypto", "otc", "block-trading", "market-microstructure", "liquidity", "slippage", "dark-pools", "settlement", "crypto-players", "execution"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 52
---

> [!important]
> **TL;DR** — Crypto's largest trades do not happen on the screen you are watching. They are negotiated privately with an **over-the-counter (OTC) desk**, printed nowhere, and then quietly digested into the public market over the following hours or days. The trade is invisible; its price impact is not.
>
> - **Big market orders destroy themselves.** In the illustrative order book we walk through below, a $50M market buy for 500 BTC eats six price levels and pays a blended $100,543.80 against a $100,000 mid — **$271,900 of slippage, 54 basis points** — and leaves a visible 1.8% wick behind for everyone to see.
> - **An OTC desk sells you one number instead of a ladder.** Quote 500 BTC at $100,250 — 25 bps over mid — and you are done in seconds, with no public print. The desk absorbs the ladder on its own book.
> - **Two models, opposite products.** A **principal** desk takes your price risk and charges you a spread; an **agency** desk keeps you exposed and charges a commission. Principal wins the tail, agency wins the average.
> - **The unwind is the transmission channel.** The desk hedges instantly with perpetual futures and then buys or sells the actual coins in small clips across venues for hours or days. That is how "off-book" flow becomes on-book pressure — and it is why the absence of a wick is *not* the absence of a seller.
> - **The one thing to remember:** every comparable traditional market publishes its block trades — US equity dark pools within **10 seconds**, CME bitcoin futures blocks within **15 minutes**. Crypto spot OTC publishes nothing, ever. There is no tape to be dark relative to.

Here is a puzzle worth sitting with.

In August and September 2020, MicroStrategy bought $425 million of bitcoin. By the standards of the market at the time this was an enormous amount of coin — enough, if dumped, to move the price by several percent. Yet if you pull up a bitcoin chart for those weeks, you will not find the purchase. There is no candle with a spike. There is no volume anomaly you could point at and say *there, that's them*. The buying happened, the coins moved, and the chart shrugged.

According to Coinbase Institutional's own published case study of the trade (December 2020), the reason is almost comically mundane: the order was sliced into roughly **200,000 individual fills**, each averaging under **0.3 BTC**, executed over five days by a time-weighted average price algorithm. The largest bitcoin purchase of its era arrived in the market in bites smaller than a retail investor's.

That is the entire subject of this article. There is a second crypto market sitting alongside the one on your screen — bigger per trade, invisible per trade, and populated by exactly the participants whose behaviour you most want to know about. It is called the over-the-counter market, and the firms that run it are called **OTC desks**.

The point of this piece is not to tell you the OTC market is spooky. It is to show you, with arithmetic you can check line by line, exactly *why* size routes there, exactly *what the desk does with your trade afterwards*, and exactly *how that off-book trade finds its way back onto the book you are watching*. Because it always does. The market never forgets a coin that needs to change hands; it only decides how loudly to announce it.

![How a $50M order reaches the market either way — swept in one visible burst, or absorbed by a desk and dispersed over hours](/imgs/blogs/otc-desks-and-moving-size-without-moving-price-1.webp)

The diagram above is the mental model for everything that follows. A buyer needs 500 BTC. There are two routes. The top route sweeps the public order book: fast, visible, and expensive — six price levels consumed, 54 basis points of slippage, a wick anyone can screenshot. The bottom route calls a desk: one firm price, 25 basis points, no public print at all. But look at where the bottom route *ends*. The desk, having sold you coins it did not own, is now short 500 BTC, and it must go buy them. It buys them on the same public book the first route swept — just slowly, in small pieces, spread across venues, over the next six hours. The pressure did not disappear. It changed shape.

Hold that image. We are going to build every piece of it from zero.

## Foundations: the building blocks

If you already know what a bid-ask spread and a basis point are, skim this section. If you don't, nothing later will make sense without it, so let's be patient.

### An order book is a queue of promises

An **exchange order book** is a public list of standing offers. On one side are **bids** — people promising to buy a certain quantity at a certain price. On the other are **asks** (also called offers) — people promising to sell. The book is sorted so the best prices sit at the top: the highest bid and the lowest ask.

The **spread** is the gap between the best bid and the best ask. If the best bid for bitcoin is $99,990 and the best ask is $100,010, the spread is $20, and the **mid price** — the arithmetic midpoint, the number people usually mean when they say "the price" — is $100,000.

Two more terms and we're equipped.

A **basis point** (bp, pronounced "bip") is one hundredth of one percent: 0.01%. One hundred basis points is 1%. Finance uses this unit obsessively because the interesting differences in execution quality live in the third decimal place of a percentage, and saying "twenty-five basis points" is less error-prone than "nought point two five percent."

**Depth** is how much quantity is available near the top of the book. A market is "deep" if you can trade a lot without walking far down the ladder, and "thin" if you can't. Depth is the single most important and least appreciated property of a market, because it is what converts your *intention* to trade into an actual *price*.

### The two order types that matter

A **limit order** says: *buy me 10 BTC, but never pay more than $100,020.* It joins the book and waits. You control your price; you do not control whether you trade at all.

A **market order** says: *buy me 10 BTC right now, whatever it costs.* It does not join the book — it consumes the book, starting at the best ask and walking down the ladder until it is filled. You control whether you trade; you do not control your price.

That asymmetry is the whole game. And it is why the first thing anyone with real size learns is that a market order is not a way of buying an asset. It is a way of buying *the order book*.

### Worked example 1: what a $50M market buy actually costs

Let's make this concrete. Below is an **illustrative order book** — I have made these numbers up so the arithmetic is clean and checkable, because real order-book snapshots are proprietary, change every millisecond, and would obscure the point. The shape, however, is realistic: size gets thinner and prices get worse as you go down the ladder.

Bitcoin's mid price is $100,000. The ask side looks like this:

| Level | Price | Size (BTC) | Cost of that level | Cumulative BTC | Cumulative cost |
| --- | --- | --- | --- | --- | --- |
| 1 | $100,010 | 40 | $4,000,400 | 40 | $4,000,400 |
| 2 | $100,050 | 60 | $6,003,000 | 100 | $10,003,400 |
| 3 | $100,150 | 90 | $9,013,500 | 190 | $19,016,900 |
| 4 | $100,400 | 120 | $12,048,000 | 310 | $31,064,900 |
| 5 | $100,900 | 150 | $15,135,000 | 460 | $46,199,900 |
| 6 | $101,800 | 40 | $4,072,000 | 500 | $50,271,900 |

You want 500 BTC. You send a market order.

Step 1. You take all 40 BTC at level 1: `40 × $100,010 = $4,000,400`. You need 460 more.

Step 2. You take all 60 at level 2: `60 × $100,050 = $6,003,000`. Running total $10,003,400. You need 400 more.

Step 3. Level 3 gives you 90 at $100,150: `90 × $100,150 = $9,013,500`. Running total $19,016,900.

Step 4. Level 4 gives you 120 at $100,400: `120 × $100,400 = $12,048,000`. Running total $31,064,900.

Step 5. Level 5 gives you 150 at $100,900: `150 × $100,900 = $15,135,000`. Running total $46,199,900.

Step 6. You still need 40 BTC. The next resting offer is at $101,800: `40 × $101,800 = $4,072,000`. Running total **$50,271,900**. Done.

Now the accounting. Your **volume-weighted average price** — the VWAP, which is simply total dollars divided by total coins — is:

`$50,271,900 ÷ 500 = $100,543.80`

At the mid price of $100,000, those 500 BTC "should" have cost $50,000,000. You paid $50,271,900. The difference, **$271,900**, is your **slippage** — the cost of your own market impact. As a fraction: `$271,900 ÷ $50,000,000 = 0.005438 = 54.4 basis points`.

![Walking a $50M market buy up an illustrative order book: each level you consume is worse than the last](/imgs/blogs/otc-desks-and-moving-size-without-moving-price-2.webp)

Two things about this deserve emphasis, because beginners consistently underrate both.

**First, nobody charged you the $271,900.** There is no line item. No fee was levied. The exchange took its normal commission on top of this. The $271,900 is pure arithmetic — the mechanical consequence of consuming a ladder whose rungs get worse. It is the most expensive thing in institutional trading and it appears on no invoice.

**Second, you told everybody.** Your final fill printed at $101,800, 1.8% above where the market was thirty seconds earlier. That print is public. Every algorithm watching that book now knows that someone with size just arrived and is probably not finished. Some of them will front-run your remainder; others will simply widen their quotes. You have not only paid 54 bps, you have made your *next* 500 BTC more expensive.

> A market order is a public announcement of private information, and you pay for the privilege of broadcasting it.

That is the problem OTC exists to solve.

## 1. The OTC alternative: one price for the whole clip

**Over-the-counter** means, literally, "not on an exchange." An OTC trade is a bilateral, privately negotiated transaction between two parties. In crypto, one of those parties is almost always a professional dealer — an **OTC desk** — whose business is standing ready to quote a single price for a quantity far larger than the visible book could absorb.

The mechanism is the **request for quote**, universally shortened to **RFQ**. It is worth understanding in detail because it is, functionally, the entire user interface of the OTC market.

![The RFQ workflow: a $50M negotiation compressed into a few seconds of firm, private, two-way pricing](/imgs/blogs/otc-desks-and-moving-size-without-moving-price-3.webp)

**Stage 1 — the request.** You message the desk, on chat or via API: *"500 BTC, two-way."* Note what you did *not* say: whether you are buying or selling. "Two-way" means you are asking the desk to quote both a bid and an offer. This is deliberate. If the desk knew you were a buyer, it would skew its price against you. By hiding your direction you force it to price honestly on both sides.

Galaxy Digital's SEC filings describe exactly this bifurcation in its own trading business: an **electronic OTC** channel over API connectivity, and a **high-touch OTC** channel operating primarily through chat. The chat channel is not a legacy artifact. When the trade is large, unusual, or in an illiquid token, a human negotiation is still the highest-bandwidth way to transfer information about what is actually possible.

**Stage 2 — the desk prices.** In the two or three seconds before it responds, the desk is computing four things:

1. **Where is mid, right now, across every venue it can trade on?** Not one exchange's mid — a synthesised, liquidity-weighted view.
2. **What is my current inventory?** If the desk is already long 800 BTC and wants to be flat, it will quote you an aggressive offer to sell you some. If it is already short, its offer will be defensive. Your price depends on a balance-sheet position you cannot see.
3. **What will it cost me to get out of this?** This is the big one, and section 3 is entirely about it.
4. **What is the risk premium?** How volatile is the market in the next ten minutes, and how much do I need to be paid for being exposed to it?

**Stage 3 — the firm quote.** The desk returns something like **bid $99,750 / offer $100,250**. This is a *firm* price, meaning the desk is contractually obliged to trade at it if you accept — but it is live for only a handful of seconds, typically five to fifteen. That expiry is not rudeness; it is the desk protecting itself from being picked off if the market moves while you deliberate. A quote that lives for a minute in a market that can move 1% in a minute is a free option, and desks do not give away free options.

**Stage 4 — you hit or you walk.** No obligation. If you don't like the price you say nothing and it expires. Many RFQs end this way, and desks price accordingly — they know most quotes die.

**Stage 5 — trade capture.** You lift the offer. `500 × $100,250 = $50,125,000`. The trade is confirmed, usually within seconds, with a ticket both sides sign off on.

**Stage 6 — settlement.** Coins move one way, dollars the other. This is the step that has historically bankrupted people, and it gets its own section.

### Worked example 2: the risk price versus the sweep

Now compare the two routes on identical size.

**Route A — sweep the book.** From worked example 1: 500 BTC for **$50,271,900**, a VWAP of $100,543.80, and a public 1.8% wick.

**Route B — take the desk's risk price.** 500 BTC at $100,250 = **$50,125,000**, VWAP $100,250 by construction, no public print at all.

The saving is `$50,271,900 − $50,125,000 = $146,900`. Expressed against the $50,000,000 mid-value of the trade, you paid 25 bps instead of 54.4 bps — you saved **29.4 basis points**.

But the dollar saving is the smaller half of the benefit. The larger half is *certainty*. When you lifted the desk's offer, you knew your price. When you sent the market order, you did not — you found out afterwards. If the book had been a little thinner that morning, or if a large seller had pulled their offers ten seconds before you pressed the button, route A might have cost you 90 bps instead of 54. Route B would still have cost 25.

**The intuition:** an OTC desk sells you a number, and the number's chief virtue is that it exists before you trade rather than after.

## 2. Two business models: principal and agency

Almost every complaint about OTC pricing traces back to a reader not knowing which of two completely different products they bought. The industry uses one word — "desk" — for two businesses that are structurally opposite.

![Principal and agency desks sell opposite things: certainty of price versus certainty of cost](/imgs/blogs/otc-desks-and-moving-size-without-moving-price-4.webp)

### The principal desk sells you a price

A **principal** desk — trading "on risk," quoting a "risk price" — buys your block onto its own balance sheet. When you lift its offer at $100,250, that desk does not have a matching seller lined up. It has just gone short 500 BTC, and every dollar the price rises before it covers is a dollar out of its pocket.

You pay for that in the **spread**: the difference between the desk's quote and where the market actually is. There is no separate commission. The 25 bps is embedded in the price you see, which is why the model is sometimes criticised as opaque — you cannot easily decompose what you paid into "market" and "fee." (You can approximate it: compare your fill to the mid at the moment you traded. That comparison, called **arrival slippage**, is the standard institutional yardstick.)

What you are buying, precisely, is **risk transfer**. From the instant you accept, the market's next move is the desk's problem. GSR, Cumberland and B2C2 built their businesses on this model, and the trade-off is covered in more depth in [GSR, Cumberland, and the established OTC desks](/blog/trading/crypto-players/gsr-cumberland-and-the-established-otc-desks).

### The agency desk sells you an execution

An **agency** desk does not take the other side. It takes your order and works it into the market on your behalf — slicing it, routing it, timing it — and charges you an explicit **commission** for the service. The coins you end up with were bought from the market, not from the desk. Your price is whatever the market gave, plus the fee.

The agency desk's incentive is entirely different. A principal desk wants to quote wide enough to survive its own unwind. An agency desk wants to **beat a benchmark** — typically arrival price (where the market was when you handed over the order) or interval VWAP (the market's own average over the execution window) — because beating benchmarks is how it gets re-hired. Coinbase's execution of the MicroStrategy order is the canonical published example of the agency model working: per Coinbase's case study, the achieved average price was *below* the price at which buying started, which the firm quantified as roughly **$4.25 million of savings** against the alternative.

### Worked example 3: agency versus principal on the same $50M

Same 500 BTC, same $100,000 arrival mid, two products. Assume the agency desk charges 8 bps commission — an illustrative rate; real institutional schedules are negotiated and confidential.

**Scenario 1 — a calm market.** The desk works your order over six hours. Its own buying pushes the price up modestly; it achieves a VWAP of $100,090, nine basis points of impact.

- Coins cost: `500 × $100,090 = $50,045,000`
- Commission: `$50,045,000 × 0.0008 = $40,036`
- **Total: $50,085,036**

Against the principal route's $50,125,000, you saved **$39,964** — about 8 bps. Agency won.

**Scenario 2 — the market moves while you're working.** Two hours in, a macro headline lands and bitcoin rallies. Your remaining size gets filled into a rising market. Achieved VWAP: $100,650.

- Coins cost: `500 × $100,650 = $50,325,000`
- Commission: `$50,325,000 × 0.0008 = $40,260`
- **Total: $50,365,260**

Against the principal route's $50,125,000, you paid **$240,260 more** — about 48 bps worse. Principal won, decisively.

Here is the table that makes the trade-off legible:

| | Principal (risk price) | Agency (worked order) |
| --- | --- | --- |
| Who owns the price risk | The desk, from the second you hit | You, until the last clip fills |
| What you pay | A spread inside the price (25 bps, all-in) | A commission (~8 bps) plus whatever the market gives you |
| When you know your price | At t = 0, before any impact | At t = end, hours or days later |
| Desk's incentive | Quote wide enough to survive its own unwind | Beat the benchmark, get re-hired |
| Calm market outcome | $50,125,000 | $50,085,036 |
| Adverse market outcome | $50,125,000 | $50,365,260 |
| Best when | You need certainty, or size far exceeds the book | You have time and the market is calm |

**The intuition:** agency execution wins on the average outcome; principal execution wins on the bad one. The desk's spread is the premium on an insurance policy against exactly the second scenario — and like any insurance, it looks overpriced right up until the day you need it.

Whether that premium is fair is an empirical question you can actually answer, which is why serious institutional traders keep **transaction cost analysis** (TCA) records: every fill, benchmarked against arrival mid, aggregated over hundreds of trades. Over a large enough sample, you can see whether a given desk's risk prices were systematically better or worse than working the order yourself. Most crypto participants never do this, which is precisely why spreads in crypto OTC are wider than in comparable traditional markets.

## 3. How the desk gets flat: hedge first, unwind second

This section is the mechanical heart of the article. If you take one thing from this piece, take this: **the desk's problem starts the moment your trade ends.**

You have your 500 BTC. The desk has $50,125,000 and a short position in bitcoin it did not want and does not intend to keep. Its entire business now consists of getting back to **flat** — owning neither a long nor a short — for less than the $125,000 spread it charged you.

It does this in two distinct stages, and confusing them is the most common error in amateur analysis of OTC flow.

![The desk's P&L on a $50M block: hedge in seconds, unwind over hours, and watch its own covering eat the spread](/imgs/blogs/otc-desks-and-moving-size-without-moving-price-5.webp)

### Stage one: hedge the delta, in seconds

**Delta** is simply your exposure to the price of the underlying asset. Short 500 BTC means a delta of −500: if bitcoin rises $1,000, you lose $500,000.

The desk cannot buy 500 BTC of spot in two seconds — that would be exactly the market order it just saved you from. But it *can*, in two seconds, buy 500 BTC worth of **perpetual futures**.

A perpetual future ("perp") is a derivative contract that tracks the spot price without ever expiring. Its price is tethered to spot by a **funding rate**: a small payment exchanged between longs and shorts every eight hours, whose sign flips depending on whether the perp is trading above or below spot. Perp markets in bitcoin are dramatically deeper than spot markets and trade around the clock, which makes them the natural instrument for instantaneous risk transfer.

So: at t+2 seconds, the desk buys 500 BTC of perps. Its spot position is −500; its futures position is +500; its net delta is approximately zero. **It is now hedged, but not finished.** It still has to source the actual coins, and it still owes funding on the perp position for as long as it holds it. The mechanics of running this kind of book continuously are the subject of [inventory risk, hedging, and delta-neutrality](/blog/trading/crypto-players/inventory-risk-hedging-and-delta-neutrality).

Why does the two-second gap matter so much? Because that is the window in which desks die. If a headline drops in the interval between the trade confirming and the hedge going on, a 1% adverse gap on $50M is **$500,000** — four times the entire spread the desk charged. The rest of the business is a game of small edges; this is the one place where a single bad second erases a month.

### Stage two: unwind into the real market, over hours

Now the desk sources 500 actual bitcoin. It has three tools and it uses all of them.

**Internalisation.** The desk's most profitable outcome is finding another client who wants the other side. Suppose a mining company calls that afternoon wanting to sell 150 BTC. The desk buys them at $99,950 — slightly below mid, because the miner is the one asking for liquidity now. Cost: `150 × $99,950 = $14,992,500`. This 150 BTC never touched a public market at all. It went from a miner's wallet to your custody account, and the order book never knew.

This is why the *number of counterparties* is a desk's real asset. Galaxy Digital reported more than **1,300 trading counterparties** at the end of Q1 2025, more than **1,500** by the end of Q3 2025, and more than **1,600** by year-end 2025, per its quarterly filings. That growth is not vanity — every additional counterparty is another chance to cross a trade internally instead of paying the market to unwind it. The Asian desks compete on exactly this axis; see [Amber, Galaxy, and the Asia market-making landscape](/blog/trading/crypto-players/amber-galaxy-and-the-asia-mm-landscape).

**Algorithmic execution.** The remaining 350 BTC gets bought through the same machinery an agency desk would use: a **TWAP** (time-weighted average price) algorithm that slices the order evenly across a time window, or a **VWAP** algorithm that slices it in proportion to expected volume so it participates more when the market is busy. Fills are spread across four or five venues so no single book sees a suspicious pattern. Suppose the desk achieves a VWAP of $100,120 over six hours: `350 × $100,120 = $35,042,000`.

**Cross-venue sourcing.** Prices differ slightly between exchanges at any instant. A desk with inventory and connectivity everywhere buys wherever it is cheapest, which simultaneously does the market a small service by pulling prices back together — the same mechanic explored in [cross-exchange arbitrage and the latency game](/blog/trading/crypto-players/cross-exchange-arbitrage-and-the-latency-game).

### Worked example 4: the desk's P&L on your block

Let's total it up. All figures illustrative but internally consistent; fee levels are typical institutional-tier magnitudes rather than any specific venue's schedule.

**Revenue.** You paid `500 × $100,250 = $50,125,000`.

**Cost of covering.**
- Internalised from the miner: `150 × $99,950 = $14,992,500`
- Bought via algo: `350 × $100,120 = $35,042,000`
- **Total: $50,034,500**, a blended cover price of `$50,034,500 ÷ 500 = $100,069`

**Gross spread:** `$50,125,000 − $50,034,500 = $90,500`, which is `$90,500 ÷ $50,125,000 = 18.1 bps`.

Pause here, because this line is the point of the whole article. The desk quoted you 25 bps — $125,000 above the $50,000,000 mid-value. It captured only $90,500. The missing **$34,500** — more than a quarter of the spread it quoted — was consumed by the price *it moved while covering*. Its own buying pushed the blended cover price to $100,069, 6.9 bps above the mid where it started.

The off-book trade moved the on-book price. Not as a wick. As a drift.

**Explicit costs.**
- Spot taker fees on the $35,042,000 algo execution at 3 bps: `$35,042,000 × 0.0003 = $10,513`
- Perpetual futures round-trip taker fees on $50,000,000 notional at 2.5 bps each way: `$50,000,000 × 0.00025 × 2 = $25,000`
- Funding paid on the long perp position for six hours, at 0.01% per eight-hour interval: `$50,000,000 × 0.0001 × (6 ÷ 8) = $3,750`
- **Total costs: $39,263**

**Net profit:** `$90,500 − $39,263 = $51,237`, which is `$51,237 ÷ $50,125,000 = 10.2 bps`.

So the desk quoted 25 bps and kept 10.2. It carried $50 million of directional bitcoin exposure for two seconds and residual basis risk for six hours, to earn fifty-one thousand dollars. And a single 1% adverse gap during the unhedged window would have cost it $500,000 — nearly ten times the profit.

**The intuition:** an OTC spread is not a fee, it is rent on the desk's balance sheet — and the desk's own unwind is one of the largest costs it has to pay out of that rent.

This is also, incidentally, why desks are so aggressive about internalisation. If the desk had internalised all 500 BTC instead of 150, it would have paid roughly $99,950 across the whole clip, captured about 30 bps gross, and avoided nearly all the fees and funding. The dream trade is the one that never touches an exchange. Most trades are not the dream trade.

## 4. The unwind is the transmission channel

Everything so far has been about a single $50M bitcoin trade, where the market is deep enough that a six-hour unwind is a rounding error. Now let's do the case that actually matters for reading charts: a large position in an asset that is *not* deep.

Suppose a token foundation needs to sell $24 million of its own token to fund two years of payroll. This is a real and recurring situation — see [token foundations and treasuries: the on-chain central banks](/blog/trading/crypto-players/token-foundations-and-treasuries-the-on-chain-central-banks) for why these entities are structurally always on the sell side.

All the numbers in this scenario are illustrative — a constructed token with constructed volumes, chosen so the arithmetic stays checkable. The token trades at $2.00. Its genuine average daily volume — after you discount the wash-traded portion, which is a whole discipline of its own covered in [wash trading, spoofing, and manufactured volume](/blog/trading/crypto-players/wash-trading-spoofing-and-manufactured-volume) — is 20 million units a day, roughly $40 million notional. The foundation wants to sell 12 million units. That is **60% of a full day's real volume**, in one clip.

There is no world in which that can be market-sold. So the foundation calls a desk, and the desk quotes a risk price of **$1.94** — a 3% discount to the $2.00 mid. The foundation accepts, receives `12,000,000 × $1.94 = $23,280,000`, and walks away with certainty.

The desk now owns 12 million units of a token with $40 million of daily volume, and no perpetual futures market deep enough to hedge in. It cannot get flat in seconds. It has to sell, slowly, and eat the price it moves.

### Worked example 5: the six-day unwind

The desk sets a **participation cap** of 10% of volume — an execution constraint meaning "never be more than one tenth of what trades." Below that threshold, the desk's flow is statistically hard to distinguish from ordinary market activity; above it, everyone notices. Ten percent of 20 million units is 2 million units a day. Twelve million units at 2 million a day is **six trading days**.

| Day | Units sold | Achieved VWAP | Proceeds | Cumulative proceeds |
| --- | --- | --- | --- | --- |
| 1 | 2,000,000 | $1.985 | $3,970,000 | $3,970,000 |
| 2 | 2,000,000 | $1.972 | $3,944,000 | $7,914,000 |
| 3 | 2,000,000 | $1.955 | $3,910,000 | $11,824,000 |
| 4 | 2,000,000 | $1.948 | $3,896,000 | $15,720,000 |
| 5 | 2,000,000 | $1.930 | $3,860,000 | $19,580,000 |
| 6 | 2,000,000 | $1.918 | $3,836,000 | $23,416,000 |

Blended realised price: `$23,416,000 ÷ 12,000,000 = $1.9513`.

Desk P&L: `$23,416,000 − $23,280,000 = $136,000`, or `$136,000 ÷ $23,280,000 = 58 bps`. Six days of risk, six days of balance sheet, for fifty-eight basis points. If the token had fallen 5% on day two for reasons entirely unrelated to the desk, the trade would have been a substantial loss.

![Six days of steady selling leaves no wick — only a slope](/imgs/blogs/otc-desks-and-moving-size-without-moving-price-6.webp)

Now look at that chart the way a screen trader would have experienced it in real time.

Day 1: the token drifts down 0.75%. Nothing. Day 2: down again. Day 3: down again. By day 6 the price is 4.1% lower than where it started and there has been **no single candle you could point at**. No wick. No volume spike — the desk was never more than 10% of volume. No news. Just six sessions of unexplained heaviness that every chart-reader in the token's community will attribute to "weak sentiment," "the market is risk-off," or "whales accumulating below."

The actual explanation is that a foundation sold $24 million to a desk on day zero, off-book, and the desk has been feeding it out ever since.

**The intuition:** the absence of a wick is not the absence of a seller. A block trade does not remove selling pressure from the market — it converts a vertical drop into a diagonal one. Learn to read the slope.

This is the single most useful idea in the article, and it inverts the naive reading of OTC. People assume OTC "protects" the market from big trades. It does not. It *launders* them, in the neutral sense: it converts a legible, dateable, single event into an illegible, undateable, distributed one. The dollars of pressure are identical. Only the forensic signature changes. The broader question of what genuinely moves a crypto price, and over what horizon, is the subject of [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move).

## 5. Settlement: the part that actually kills people

Trading is the fun part. Settlement is where the money is lost.

Every OTC trade has a moment of maximum vulnerability: the interval between "we agree" and "we have both been paid." In a bilateral trade with no exchange in the middle, somebody has to go first. If you send your $50 million before receiving your 500 BTC, you are — for however long that takes — an unsecured creditor of the desk. If the desk goes bankrupt in that window, you are not a trader with a bad fill. You are a bankruptcy claimant with a lawyer.

![Off-exchange settlement cuts a counterparty failure from a total loss of principal to a few hours of unsettled P&L](/imgs/blogs/otc-desks-and-moving-size-without-moving-price-7.webp)

### The old model: pre-funding, and hope

Before late 2022, the dominant model was **pre-funding**. You maintained a balance with your desk or your exchange. You wired money in, it sat on their balance sheet, and you traded against it. Operationally this was wonderful — instant settlement, no wire delays, no failed trades. Legally it was catastrophic, because in most jurisdictions those assets were not segregated. They were the desk's assets, and you were a creditor.

The industry discovered this the hard way. In November 2022, following the collapse of FTX, **Genesis Global Capital suspended all withdrawals on 16 November 2022**, citing market dislocation and redemption demands exceeding $1 billion (CNBC, November 2022). The lending unit's exposures included insolvent counterparties. Genesis and affiliated entities filed for **Chapter 11 bankruptcy on 19 January 2023** (CNBC, January 2023). Genesis Global Trading — a separate entity, and one of the most established OTC desks in the business — went on to shut down its spot trading platform on **18 September 2023** (The Block, citing CoinDesk).

The lesson institutions drew was not "pick better counterparties." It was "stop needing to."

### The new model: off-exchange settlement

The post-FTX architecture is called **off-exchange settlement** (OES), and the idea is elegant. Your assets stay with an independent, regulated **custodian** — a firm whose only job is holding assets, typically under a trust structure that survives the custodian's own insolvency. The custodian *mirrors* your balance to the trading venue, which grants you trading credit against it. You trade on the mirror. Actual assets move only at periodic net settlement.

The result: if the venue fails, your assets are not there to lose. Your exposure is limited to **unsettled profit and loss** since the last settlement cycle — hours, not months, and a fraction of principal rather than all of it.

The main implementations, as of 2025:

- **Copper's ClearLoop**, where assets remain in Copper's multi-party-computation custody under an English law trust. Copper has reported ClearLoop facilitating over **$50 billion in monthly notional trading volume**, with live connections to venues including Coinbase, OKX, Bybit, Deribit and Bitget.
- **Fireblocks off-exchange**, using segregated MPC wallets. Deribit became the first exchange to fully integrate it in February 2024; HTX followed in April 2025.
- **Ceffu's MirrorX**, which mirrors custodied balances into Binance liquidity.
- **BitGo's Go Network**, which went live for Deribit in February 2025 alongside ClearLoop, with assets held at BitGo Trust (CoinDesk, 20 February 2025). OKX integrated BitGo off-exchange settlement for US institutional clients.

For pure bilateral OTC trades — where there is no exchange at all, just you and a desk — the analogous solutions are **tri-party settlement** (a neutral custodian holds both legs and releases them simultaneously, achieving true delivery-versus-payment) and **escrow** arrangements for very large or unusual trades.

### What is still risky

Three things, honestly stated.

**Stablecoin leg risk.** Most crypto OTC settles the fiat leg in stablecoins rather than bank wires, because wires are slow and crypto is not. Finery Markets' analysis of over 15 million institutional spot trades executed on its platform between January 2024 and December 2025 found stablecoins accounted for **78% of all OTC trades in 2025, up from 26% two years earlier**. That is an enormous concentration of settlement risk in a small number of token issuers. It is a different risk than exchange failure, not an absence of risk.

**Credit lines.** Desks extend credit to good clients, and credit is the mechanism by which a single counterparty failure becomes several. This is the transmission channel that turned FTX's collapse into an industry-wide event rather than one firm's problem.

**Legal enforceability.** OES structures depend on the trust or segregation arrangement actually holding up in the relevant jurisdiction's insolvency proceedings. Several of these structures have never been tested in a real bankruptcy. "Legally segregated" is a claim about a document, and documents are tested only when someone fails.

## 6. Who the structural sellers actually are

The OTC market exists because certain participants must trade size *on a schedule*, regardless of price. Understanding who they are turns OTC from a mystery into a calendar.

![Crypto's persistent sellers are structural, not emotional — they sell whether or not anyone wants to buy](/imgs/blogs/otc-desks-and-moving-size-without-moving-price-8.webp)

### Miners: the metronome

Bitcoin miners receive newly issued coins and pay their costs — electricity, hosting, debt service, staff — in fiat currency. That mismatch forces a recurring conversion whose size is set by protocol, not sentiment.

The April 2024 halving cut the block subsidy from 6.25 BTC to 3.125 BTC, reducing daily issuance from roughly **900 BTC to roughly 450 BTC**. At a $100,000 bitcoin price, that is around $45 million of new supply per day that has to find a buyer, forever, whether or not anyone is in the mood.

Most of it does not hit exchange order books. It goes to desks, because miners are precisely the sophisticated, repeat, size-constrained sellers that OTC was built for — and because a miner that market-sold its production daily would be donating slippage to the market every single day.

Miner behaviour around the halving was widely tracked: CoinDesk reported on 12 June 2024 that miners sold at least 1,200 BTC on 10 June, the highest daily total in two months, as post-halving revenue pressure bit. Longer-run, on-chain analytics firms tracking address clusters they label as miner-linked OTC desks have reported a decline in those balances from roughly 500,000 BTC in November 2021 to roughly 139,700 BTC — a fall of about 72% (AMBCrypto, citing CryptoQuant).

Treat that last figure with the caution it deserves. "Miner OTC desk balances" is an *inference* from address clustering heuristics, not an audited disclosure. Nobody outside the desks knows their actual inventory. It is a useful directional signal and a terrible precise one. The economics of mining, staking rewards and the other structural supply streams are laid out in [crypto mining, staking, and MEV](/blog/trading/crypto/crypto-mining-staking-and-mev).

### Token foundations and treasuries

A foundation holding a treasury denominated in its own token must periodically convert some of it into fiat or stablecoins to pay salaries, grants and audits. This is worked example 5, and it recurs on a quarterly or annual cadence at essentially every large token project.

### Early investors at unlock

A vesting cliff is a date on a contract. When it passes, tokens become transferable, and a fund with a fiduciary duty to realise returns will realise them. The decision was made years earlier at the term sheet; the calendar simply executes it. How that flow is structured and priced is covered in [the crypto VC and market-maker relationship](/blog/trading/crypto/crypto-vc-and-market-makers).

### ETF authorised participants: the two-way flow

The spot bitcoin ETFs approved on 10 January 2024 introduced a genuinely new and enormous OTC user. An **authorised participant** (AP) is a large financial institution permitted to create and redeem ETF shares directly with the fund. Under the cash-creation model — the only model permitted for US spot bitcoin ETFs until the SEC approved in-kind creations and redemptions on 29 July 2025 (CoinDesk) — the AP delivers cash, and the fund (or its designated **execution agent**) must go buy actual bitcoin.

Those purchases are exactly the kind of scheduled, size-constrained, price-sensitive flow that routes to desks. The prospectuses say so explicitly: the Invesco Galaxy Bitcoin ETF's SEC filings name Galaxy Digital Funds LLC as execution agent; Coinbase serves as prime broker and bitcoin counterparty for several trusts. The full architecture of that bridge is the subject of [bitcoin ETFs and the TradFi bridge](/blog/trading/crypto/bitcoin-etfs-and-the-tradfi-bridge).

Critically, this flow runs both directions. On a heavy-inflow day, APs are structural *buyers* of size. On an outflow day, structural sellers. The ETF wrapper converted a category of investor demand that previously never touched crypto into a daily, mechanical, OTC-routed flow.

### Corporate treasuries: the standing bid

MicroStrategy — since renamed Strategy — is the archetype, but the category has grown well beyond it. A corporate treasury buying bitcoin as a reserve asset is a patient, price-insensitive, repeat buyer of size, and it uses exactly the same machinery in reverse.

## 7. Premium and discount as a demand signal

Because OTC trades are unreported, you cannot observe OTC demand directly. What you can sometimes observe is a **premium** or **discount**: a persistent price gap between two places where the same asset trades. Gaps are informative, but only if you understand what creates them.

![The size of a premium measures how blocked the arbitrage is; only its sign measures demand](/imgs/blogs/otc-desks-and-moving-size-without-moving-price-9.webp)

The governing principle is this: **in an open market, arbitrage kills price gaps almost instantly.** If bitcoin is $100,000 in one place and $100,500 in another, someone buys the first and sells the second until the gap closes. Therefore a *persistent* gap is not evidence of demand. It is evidence that something is **blocking the arbitrage** — and the size of the gap measures the strength of the blockage, not the strength of the demand.

### The Coinbase premium: small, fast, and about sign

The **Coinbase premium** is the percentage difference between bitcoin's price on Coinbase (a US dollar pair, dominated by US institutional flow) and on Binance (a USDT pair, dominated by offshore flow).

Nothing blocks this arbitrage. Any firm with inventory on both venues can trade it in seconds. So the gap is tiny — single-digit basis points — and mean-reverts within minutes.

Suppose bitcoin is $100,000 on Binance and $100,050 on Coinbase: a 5 bp premium. On a $50 million clip that is $25,000, which sounds like real money until you remember the arbitrageur needs inventory on both venues, pays fees on both legs, and is competing with everyone else who noticed.

Because the gap is arbitraged so efficiently, its *magnitude* carries almost no information. What carries information is its **sign and persistence**: a Coinbase premium that stays positive for days means US dollar buying is continuously outrunning the arbitrageurs' capacity to flatten it, which is a genuine, if crude, read on where flow is coming from.

### The Korea premium: large, persistent, and about capital controls

The **kimchi premium** is the gap between bitcoin's price on Korean won exchanges and the global price. It has at times exceeded 20% during Korean retail manias.

Why doesn't arbitrage close it? Because the closing trade requires you to buy bitcoin offshore, sell it in Korea for won, and then convert those won back to dollars and move them out. That last leg runs into Korea's foreign exchange rules and Korean banks' reluctance to process crypto-linked outbound remittances. Access to Korean exchanges requires a real-name domestic bank account, which effectively restricts participation to residents.

So the premium can persist for weeks. It is a real and useful signal of local demand — but what it signals is *demand that cannot be met by global supply*, which is a statement about capital controls at least as much as a statement about bitcoin.

### The Grayscale discount: the biggest one of all

The most instructive premium/discount episode in crypto history ran in the opposite direction.

The Grayscale Bitcoin Trust (GBTC) was, before January 2024, a closed-end trust. You could create shares but you could not redeem them for the underlying bitcoin. Without a redemption mechanism, nothing tethered the share price to net asset value, and the two came apart badly. GBTC had traded at a large *premium* in the 2020–2021 bull market, then flipped to a discount that reached **nearly 50% in December 2022** — meaning the market valued a claim on one bitcoin at roughly half of one bitcoin.

The resolution came through the courts and then the regulator: Grayscale prevailed in its legal challenge to the SEC during 2023, the SEC approved spot bitcoin ETFs on 10 January 2024, and GBTC converted to an ETF the following day. With authorised participants finally able to create and redeem at NAV, the discount **closed to approximately zero on 11 January 2024**, its first time at parity since February 2021 (CoinDesk, 11 January 2024).

#### Worked example 6: trading the GBTC discount

Suppose that in December 2022 you buy $10,000,000 of GBTC at a 48% discount to NAV — a level inside the "nearly 50%" record lows CoinDesk reported for that month. The share price is 52% of the bitcoin backing it, so your $10,000,000 buys a claim on:

`$10,000,000 ÷ 0.52 = $19,230,769` of bitcoin at net asset value.

If the discount closes to zero and bitcoin's price is unchanged, your position is worth $19,230,769 — a gain of **92.3%** from the discount alone.

Now subtract the frictions. A closed-end trust charges an annual **sponsor fee** that is deducted from net asset value, so your claim shrinks slightly every day you hold it. Assume 2% a year — I am flagging this as an *assumption*, not a sourced input, because I could not verify Grayscale's historical fee schedule directly for this article; GBTC has been reported as the highest-fee product in the spot bitcoin ETF launch cohort. Over the roughly 13 months from December 2022 to 11 January 2024, a 2% annual fee drags NAV by about `2% × (13 ÷ 12) = 2.17%`:

`$19,230,769 × (1 − 0.0217) = $18,814,038`

Return on the discount trade: **+88.1%**.

The conclusion is robust to the assumption. At a 1.5% annual fee the drag is 1.62% and the return is **+89.2%**; at 3% the drag is 3.25% and the return is **+86.1%**. Across any plausible fee, the answer is "somewhere in the high eighties" — the fee is a rounding error against a 92-point discount. What is *not* a rounding error is everything in the next paragraph.

Note carefully what this is and is not. It is *not* additive to bitcoin's own return — it is multiplicative. If bitcoin had also doubled over that period, your position would have been worth roughly `2 × $19,230,769 × 0.978 = $37.6 million`. The discount closing was a multiplier on whatever bitcoin did.

And now the honest part, which is the reason this is a worked example rather than a recommendation. In December 2022 there was **no redemption mechanism and no date**. You were not arbitraging a spread with a known convergence — you were making a bet on a legal outcome and a regulatory decision, either of which could have gone the other way. The discount had widened for a full year before it narrowed, and holders who bought at a 30% discount in mid-2022 watched it go to 48% before it came back. Being right about the destination is not the same as surviving the journey.

**The intuition:** the size of a persistent premium or discount tells you how completely the arbitrage is blocked; only its sign tells you about demand. Confusing the two is the most common analytical error in this entire domain.

## 8. The dark-pool comparison: what TradFi must disclose

Crypto's OTC market is routinely compared to traditional finance's **dark pools**, and the comparison is genuinely useful — but the differences matter more than the similarities, and they run in the opposite direction to what most people assume.

A dark pool is an **alternative trading system** (ATS): a private venue where institutions match large orders without displaying them publicly beforehand. Same motivation as crypto OTC — hide size, avoid impact. And dark pools are genuinely controversial in equities, criticised for fragmenting liquidity and for advantaging participants who can see flow others cannot.

But note the precise nature of the criticism. It is that dark pools hide orders **before** execution. Not after.

![Every comparable traditional block market publishes the trade; crypto spot OTC publishes nothing, ever](/imgs/blogs/otc-desks-and-moving-size-without-moving-price-10.webp)

| Market | Must the trade be reported? | How fast, and to whom? | Venue volume published? |
| --- | --- | --- | --- |
| US equities dark pool / ATS | Yes — FINRA Rule 6282 | Within 10 seconds of execution, to a FINRA Trade Reporting Facility, then the public tape | Yes — per security, on a roughly two-week delay (FINRA Rule 4552) |
| US corporate bonds | Yes — TRACE | Within minutes of execution, publicly disseminated | Yes, aggregated |
| CME bitcoin futures block | Yes — CME Rule 526 | Within 15 minutes, to the exchange, then published in exchange data | Yes — included in exchange volume |
| Crypto **spot** OTC block | No — no reporting obligation applies | Never; only the two counterparties ever know | No consolidated tape exists |

Read that bottom row again. In US equities, a dark pool execution is invisible for the milliseconds before it happens and then becomes public within ten seconds. FINRA additionally publishes ATS-by-ATS volume for each security, so you can see how much of a stock's activity happened in the dark and where. In corporate bonds, TRACE did the same thing to a market that was previously entirely opaque. Even in crypto *futures* — where the CFTC regulates and CME operates the venue — a block trade must clear a minimum size threshold (5 contracts for standard bitcoin futures, 10 for micro contracts, per CME Rule 526) and be reported to the exchange within a 15-minute window, after which it appears in published volume.

Crypto spot OTC has none of this. There is no trade reporting rule, no reporting facility, no consolidated tape, and no aggregate venue statistics. A $500 million bitcoin block can trade between two firms and be known to precisely two firms, permanently.

The consequence is not merely that individual trades are hidden. It is that **the market's total size is unknown**, including to the people in it. The best available figures are single-venue disclosures that their publishers correctly frame as partial. Finery Markets, reporting on its own institutional platform, found crypto spot OTC volumes grew **109% year over year in 2025**, against roughly 9% growth in top-20 centralised exchange volumes over a comparable period (The Block data, cited by Finery). That is a striking divergence — but it is one platform's flow, not the market's, and it should be read as a directional indicator rather than a measurement.

Nobody knows how big crypto OTC is. Anyone who tells you a precise number is quoting an estimate whose methodology you should ask about.

Some venues have built hybrid structures that split the difference. Deribit, the dominant crypto options exchange, supports on-exchange **block trading** with published minimum sizes, and launched a **Block RFQ** interface in March 2025 supporting multi-leg structures of up to 20 legs (CoinDesk, 6 March 2025). Paradigm operates as a communication and workflow layer where institutions negotiate derivative blocks bilaterally, with the agreed trade then submitted to an exchange like Deribit for execution and clearing. These trades *do* reach exchange data, which is why crypto derivatives block flow is dramatically more visible than crypto spot block flow. Deribit's published block minimums have changed over time — its documentation has listed thresholds such as 25 BTC or 250 ETH for certain block methods and a $200,000 minimum for futures blocks — so check the current schedule before relying on any specific figure. The role exchanges play as active participants rather than neutral venues is explored in [exchanges are players, not just venues](/blog/trading/crypto-players/exchanges-are-players-not-just-venues).

## Common misconceptions

**"OTC trades don't affect the price."** This is the big one, and it is wrong in the most useful possible way. Every worked example above shows the mechanism: the desk that bought your block has to sell it, and it sells it into the same public market you are watching. Worked example 4 quantified it — more than a quarter of the desk's quoted spread was consumed by the price its own covering moved. Worked example 5 showed the extreme version: 4.1% of drift over six days with no attributable candle. OTC changes the *timing and shape* of price impact. It does not eliminate it.

**"A big trade always leaves a wick."** No. A wick is the signature of *impatient* size. Patient size — the kind that goes through a desk and gets fed out at 10% of volume — leaves a slope, not a wick. Traders who scan for volume spikes and long wicks are, by construction, only finding the least sophisticated participants.

**"The OTC price is better than the exchange price."** Sometimes; it depends entirely on your size. For a $10,000 trade the exchange is obviously better — the book absorbs you completely and you pay a two-basis-point spread. Somewhere between there and $50 million the calculus flips. The crossover point is where your order's slippage exceeds the desk's spread, and it moves constantly with market depth.

**"Principal desks are ripping clients off with hidden fees."** The spread is not hidden — it is *embedded*, which is different, and you can measure it exactly by comparing your fill to arrival mid. Whether 25 bps is fair for instant risk transfer on $50 million is a legitimate empirical question that you answer with transaction cost analysis over many trades. Most clients never measure, which is genuinely why spreads are wider than they need to be. The remedy is measurement, not indignation.

**"OTC is where the manipulation happens."** OTC is where *size* happens, which is not the same thing. The absence of a reporting requirement certainly makes some misconduct harder to detect, and claims about specific desks' conduct — where they exist — should be treated as reported allegations rather than established fact unless adjudicated. But the mechanics described in this article are ordinary, legal, and identical in function to institutional block trading in every other asset class. The distinguishing feature of crypto OTC is not its ethics; it is its complete absence of post-trade transparency.

**"Since I can't see OTC, I should ignore it."** The opposite. You can't see the trades, but you *can* often see the structural calendar that generates them — halving dates, vesting cliffs, foundation treasury policies, ETF flow reports. The flow is invisible; the schedule frequently is not.

## How it shows up in real markets

### 1. MicroStrategy's $425 million, and the chart that never noticed

Per Coinbase Institutional's published case study (December 2020), MicroStrategy's initial bitcoin purchase comprised $250 million executed over five days in August 2020 through Coinbase Prime, followed by $175 million in September. The execution combined human oversight with a time-weighted average price algorithm, slicing the order into roughly 200,000 fills averaging under 0.3 BTC each, with instructions adapted in real time — accelerating into weakness, slowing into strength. Coinbase reported an achieved average price below the price at which buying started, quantified as approximately $4.25 million in savings.

This is the agency model in its purest form, and it demonstrates the article's central asymmetry: the buyer got a good price *because* it accepted duration risk. Five days of exposure to bitcoin's price, in exchange for near-zero market impact. A principal desk would have quoted the whole $425 million instantly at a spread — and given late-2020 liquidity, that spread would have been substantially wider than the impact the algorithm actually caused.

### 2. Genesis, November 2022, and the death of pre-funding

On 16 November 2022, in the immediate aftermath of FTX's collapse, Genesis Global Capital suspended withdrawals, citing redemption demands exceeding $1 billion (CNBC). Genesis and affiliates filed Chapter 11 on 19 January 2023 (CNBC). Genesis Global Trading later wound down its spot trading platform in September 2023.

The mechanism was counterparty contagion: Genesis's lending book had exposure to insolvent counterparties, and when those failed, its own creditors could not be paid. For OTC clients the lesson was structural rather than moral. A firm can be perfectly competent at trading and still fail because of what sits on the other side of its balance sheet, and if your assets are on that balance sheet when it happens, your trading skill is irrelevant. This single episode is why off-exchange settlement went from a niche product to a default institutional requirement in under two years.

### 3. The 2024 halving and a forced seller with no discretion

On 20 April 2024 the bitcoin block subsidy halved from 6.25 to 3.125 BTC, cutting daily issuance from roughly 900 to roughly 450 BTC. Miners' revenue halved overnight while their electricity bills did not.

The observable consequence was a period of elevated miner distribution: CoinDesk reported on 12 June 2024 that miners sold at least 1,200 BTC on 10 June, the highest daily total in two months, with exchange transfers reaching a two-month high. Longer-run on-chain estimates suggest miner-linked OTC balances have declined substantially since 2021 — roughly 500,000 BTC in November 2021 to roughly 139,700 BTC, about a 72% fall (AMBCrypto citing CryptoQuant), though these are clustering inferences rather than disclosures.

The instructive part is the *predictability*. Nobody knew which desk handled which miner's flow, but everybody knew that halved revenue against unchanged fiat costs meant more selling per coin held. The calendar was public even though the trades were not.

### 4. The spot ETF launch and a permanent new OTC client

The SEC approved 11 spot bitcoin ETFs on 10 January 2024. Because US spot bitcoin ETFs operated on a cash-creation model until the SEC approved in-kind creations and redemptions on 29 July 2025 (CoinDesk), every dollar of net inflow had to be converted into actual bitcoin by a fund's execution agent — Galaxy Digital Funds LLC for the Invesco Galaxy Bitcoin ETF, per its SEC filings, with Coinbase serving as prime broker and bitcoin counterparty for several trusts.

This created something crypto had never had: a large, recurring, professionally executed, *reported-in-aggregate* flow. Daily ETF flow figures are published. The individual trades are not. So a screen trader in 2024 could, for the first time, know roughly how much size needed to be bought on a given day without knowing a single trade that did the buying — an unusually clean illustration of the schedule being visible while the flow is not.

### 5. The GBTC discount and the 13-month arbitrage with no delivery date

GBTC's discount to net asset value reached nearly 50% in December 2022 and closed to approximately zero on 11 January 2024 upon its ETF conversion — its first time at parity since February 2021 (CoinDesk).

The structural cause was the absence of a redemption mechanism: shares could be created but not redeemed, so nothing forced convergence. That is a purely mechanical fact about the wrapper, not a view on bitcoin, and it is worked through numerically in worked example 6. The lesson for reading premiums generally: before interpreting a price gap as sentiment, establish whether an arbitrage *could* close it. If it could and didn't, look for the blockage. If it structurally couldn't, the gap is telling you about plumbing.

### 6. Off-exchange settlement becoming the default

By 2025 the architecture built after FTX was operating at meaningful scale. Copper has reported ClearLoop facilitating over $50 billion in monthly notional trading volume across connected venues including Coinbase, OKX, Bybit, Deribit and Bitget. Deribit fully integrated Fireblocks off-exchange settlement in February 2024, with HTX following in April 2025. In February 2025, BitGo and Copper delivered off-exchange settlement for Deribit, with assets held at BitGo Trust and settled through ClearLoop and the Go Network (CoinDesk, 20 February 2025).

The economic effect is subtle but large. When institutions no longer need to pre-fund venues, capital that previously sat idle as trading collateral becomes available for actual trading. Reduced counterparty risk does not merely prevent losses; it increases effective liquidity, which narrows spreads, which lowers the cost of exactly the block trades this article is about.

## What a screen trader can actually infer — and the traps

Here is the honest inventory of what this knowledge buys you.

**What you can genuinely infer.**

*Slope without news is a candidate for an unwind.* Multi-session, low-volatility, one-directional drift with no catalyst and no volume spike is the exact signature of algorithmic execution at a participation cap. It is not proof — markets drift for many reasons — but it is a hypothesis worth holding, and it is a much better hypothesis than "sentiment."

*Structural calendars are public.* Halving dates are in the protocol. Vesting cliffs are usually in the token's public documentation. ETF flows are published daily. Foundation treasury policies are often disclosed. You cannot see the trades, but you can frequently see the obligation that will generate them.

*Persistent premiums locate blocked arbitrage.* A gap that survives is telling you where capital cannot flow. That is real information about market structure even when it says nothing about demand.

*Depth is observable and predictive.* You can measure order book depth yourself in real time. Thin books mean any given block will take longer to unwind, which means a longer, flatter, more persistent price effect.

**The traps, which are numerous.**

*Attribution is nearly impossible.* You may correctly detect that someone is working a large order and be completely wrong about who, why, or which direction they started from. A desk unwinding a long looks identical to a fund building a short.

*On-chain "OTC desk" labels are inferences.* Every figure you see about desk inventories or miner OTC balances comes from address clustering heuristics maintained by analytics firms. They are educated guesses that get revised. Treat them as directional and never as measurements — including the ones cited in this article.

*Exchange inflow spikes are ambiguous.* Coins moving to an exchange might be about to be sold, or might be collateral for a loan, or might be a custody migration, or might be a desk rebalancing inventory across venues. The single most over-interpreted signal in crypto analytics.

*Absence of evidence is not evidence of absence — this is the whole thesis.* The chart looked calm during MicroStrategy's $425 million. It looked calm during our hypothetical foundation's $24 million. Quiet tape is entirely consistent with enormous flow. If your framework treats "nothing visible happened" as "nothing happened," it will be wrong at exactly the moments that matter most.

*Survivorship bias in the stories.* The OTC episodes you know about are the ones that went wrong — Genesis, FTX-adjacent failures, the disputes that reached court. The overwhelming majority of block trades settle uneventfully and are never mentioned by anyone. Do not calibrate your view of the market on its litigation record.

> The market is a machine for converting private decisions into public prices. OTC does not stop that machine. It just lengthens the conveyor belt.

## When this matters to you

If you are trading retail size, none of this changes what you should do — your orders are absorbed by the book, and the desk economics above are simply not your problem. What it changes is how you *read*. When a token bleeds for six sessions with no news, "an unwind is running" belongs in your list of hypotheses, and it should make you more patient about calling a bottom, because an unwind ends on a schedule you cannot see. When a chart is calm, you should not conclude that nothing is happening.

If you work anywhere near institutional execution, the actionable content is measurement. Compare every fill to arrival mid. Keep the records. Over fifty trades you will know, quantitatively, whether your desk's risk prices have been fair, and that knowledge is the only thing that reliably tightens a spread.

If you are evaluating a counterparty, the settlement section is the part that protects you. Ask where your assets sit between trade and settlement, ask whether the segregation structure has been tested in an insolvency, and ask what your exposure is at the maximum point of the cycle. Those three questions would have saved a great many people a great deal of money in November 2022.

And if you are simply trying to understand why crypto prices do what they do: the most important flows in this market are the ones you cannot see, executed by firms you have never heard of, on a schedule set by protocol issuance, vesting contracts and fund redemption cycles. The screen shows you the market's voice. The OTC market is where it does most of its thinking.

*This article is educational. It describes market mechanics and historical episodes; it is not investment advice, and nothing in it is a recommendation to buy or sell anything.*

## Sources & further reading

**Primary sources behind the headline figures**

- Coinbase Institutional, *MicroStrategy case study* (December 2020) — the $425M purchase, ~200,000 fills averaging under 0.3 BTC, TWAP execution over five days, and the ~$4.25M reported saving. See also [Coinbase brokered MicroStrategy's $425M bitcoin purchase](https://www.coindesk.com/markets/2020/12/01/coinbase-brokered-microstrategys-425m-bitcoin-purchase-exchange-says) (CoinDesk, 1 December 2020).
- [Genesis lending unit halts withdrawals as FTX contagion spreads](https://www.cnbc.com/2022/11/16/genesis-lending-unit-halts-withdrawals-in-aftermath-of-ftx-collapse.html) (CNBC, 16 November 2022) and [Crypto lender Genesis Trading files for bankruptcy protection](https://www.cnbc.com/2023/01/20/crypto-lender-genesis-trading-files-for-bankruptcy-barry-silbert-digital-currency-group.html) (CNBC, 20 January 2023).
- [Grayscale's GBTC discount closes to zero for the first time since February 2021](https://www.coindesk.com/markets/2024/01/11/grayscales-gbtc-discount-closes-to-zero-for-first-time-since-february-2021) (CoinDesk, 11 January 2024) — the ~48% December 2022 discount and its close at conversion.
- [CME Group Rule 526 — Block Trades](https://www.cmegroup.com/rulebook/files/cme-group-Rule-526.pdf) and [CME cryptocurrency futures FAQ](https://www.cmegroup.com/articles/faqs/frequently-asked-questions-cryptocurrency-futures.html) — the 5-contract minimum for bitcoin futures blocks, 10 for micro contracts, and the 15-minute reporting window.
- [FINRA trade reporting FAQ](https://www.finra.org/filing-reporting/market-transparency-reporting/trade-reporting-faq) — the 10-second TRF reporting obligation (Rule 6282) and ATS volume publication with a two-week delay (Rule 4552).
- Galaxy Digital Inc., [Form 10-Q for the quarter ended 31 March 2025](https://www.sec.gov/Archives/edgar/data/1859392/000185939225000005/glxy-20250331.htm) and subsequent filings — trading counterparty counts (>1,300 in Q1 2025, >1,500 in Q3 2025, >1,600 at year-end 2025) and the electronic/high-touch OTC split.
- [Invesco Galaxy Bitcoin ETF, Form 424B3 (2024)](https://www.sec.gov/Archives/edgar/data/1855781/000119312524018217/d714048d424b3.htm) — the execution agent structure for cash creations.
- [SEC approves in-kind redemptions for all spot bitcoin and ethereum ETFs](https://www.coindesk.com/markets/2025/07/29/sec-approves-in-kind-redemptions-for-all-spot-bitcoin-ethereum-etfs) (CoinDesk, 29 July 2025).
- [Crypto custody firms BitGo and Copper deliver off-exchange settlement for Deribit](https://www.coindesk.com/business/2025/02/20/crypto-custody-firms-bitgo-and-copper-deliver-off-exchange-settlement-for-deribit) (CoinDesk, 20 February 2025) — the ClearLoop and Go Network integration.
- [Deribit launches Block RFQ system for large over-the-counter trades](https://www.coindesk.com/markets/2025/03/06/deribit-launches-block-rfq-system-to-improve-liquidity-for-large-over-the-counter-trades) (CoinDesk, 6 March 2025) and [Deribit Block RFQ product documentation](https://statics.deribit.com/files/Block_RFQ_DPD.pdf) (May 2025).
- [Finery Markets, *Crypto OTC Report: 2025 Results & Trends*](https://finerymarkets.com/blog/crypto-otc-report-2025-results-trends) — 109% year-over-year growth in spot OTC volumes and the 78% stablecoin share, based on 15M+ institutional trades on its own platform between January 2024 and December 2025. Read as single-platform data, not a market measurement.
- [Bitcoin miners cash in as exchange transfers hit two-month high](https://www.coindesk.com/business/2024/06/12/bitcoin-miners-cash-in-on-btc-rally-as-exchange-transfers-hit-two-month-high) (CoinDesk, 12 June 2024) — the 1,200 BTC single-day miner sale.

**Where the numbers are deliberately illustrative**

The order book in worked example 1, the risk prices, commissions, exchange fee tiers, funding rates, and the entire six-day unwind in worked example 5 are **constructed examples**, labelled as such in the figures. Real desk spreads, real fee schedules and real order-book snapshots are confidential and change constantly. The arithmetic is exact; the inputs are chosen for clarity.

One further flag, in the same spirit: the 2% annual GBTC sponsor fee used in worked example 6 is an **assumption**, not a sourced figure — Grayscale's fee page was not retrievable while writing this. The example shows the sensitivity, and the answer lands in the high eighties of percent across any plausible fee.

**Crypto OTC volumes are genuinely unreported.** There is no consolidated tape, no trade reporting obligation, and no regulator collecting the data. Every aggregate figure in circulation — including the Finery Markets growth rates cited above — is one venue's or one vendor's partial view. Any number in this article presented as a fact about the world carries a source above; where a figure is an on-chain inference (miner OTC balances) or a single-platform disclosure, that is stated at the point of use.

**Further reading on this blog**

- [How crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) — the broader mechanics of price formation this article plugs into.
- [What a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) — the continuous-quoting sibling of the block business.
- [GSR, Cumberland, and the established OTC desks](/blog/trading/crypto-players/gsr-cumberland-and-the-established-otc-desks) — the firms running the principal model.
- [Inventory risk, hedging, and delta-neutrality](/blog/trading/crypto-players/inventory-risk-hedging-and-delta-neutrality) — the hedging machinery in section 3, in depth.
- [Bitcoin ETFs and the TradFi bridge](/blog/trading/crypto/bitcoin-etfs-and-the-tradfi-bridge) — the creation/redemption architecture behind the newest OTC flow.
- [Crypto mining, staking, and MEV](/blog/trading/crypto/crypto-mining-staking-and-mev) — where the structural miner supply comes from.
</content>
