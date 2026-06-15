---
title: "How a Price Chart Is Born: Price, Volume, and the Auction Beneath Every Candle"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A first-principles tour of where a price chart actually comes from: the bid and ask, the order book, market versus limit orders, how a single trade prints, how prints aggregate into OHLC candles, and what volume really measures."
tags: ["technical-analysis", "order-book", "bid-ask-spread", "market-microstructure", "candlesticks", "volume", "liquidity", "slippage", "limit-orders", "ohlc"]
category: "trading"
subcategory: "Technical Analysis"
author: "Hiep Tran"
featured: true
readTime: 50
---

> [!important]
> **TL;DR** — Every candle on a chart is the compressed record of a continuous auction between buyers and sellers; to read charts honestly, you have to understand the machinery that produces them.
>
> - A market is a never-ending auction. At any instant there is a highest price someone will pay (the *bid*), a lowest price someone will accept (the *ask*), and a gap between them (the *spread*). The list of all the resting offers, stacked by price, is the *order book*.
> - A *limit order* says "I'll trade, but only at my price or better" and waits in the book. A *market order* says "fill me now at whatever's there" and trades immediately against the resting limit orders. A trade *prints* the instant the two meet.
> - The "price" you see is just the price of the **last trade**. A candle bundles every print in a time window into four numbers: **open, high, low, close** (OHLC). The body is open-to-close; the wicks reach to the high and low.
> - **Volume is the number of contracts (or shares) that changed hands** — nothing more. Every trade has a buyer *and* a seller, so volume is never "more buyers than sellers." It measures participation and conviction, not direction.
> - The **spread is a real cost** you pay every round trip, and **liquidity** — how many shares rest near the touch — is why price levels hold or break. A thin book lets the same order *walk* far; a deep book absorbs it. The single number to remember: on a 0.05% spread, a \$10,000 round trip costs you about \$5 before the price even moves.

Pull up any price chart — a stock, a currency pair, Bitcoin — and you'll see a row of little candles marching left to right. Most people treat each candle as a fact handed down from somewhere: the price *was* this, then it *became* that. But a candle is not a fact about the world the way a thermometer reading is. It is a *summary statistic* of a frantic, continuous negotiation that happened underneath it — thousands of individual trades, each one a buyer and a seller agreeing on a number for an instant. If you don't know how that negotiation works, the chart will quietly lie to you in ways you can't see.

This is the first real building block in the series [Technical Analysis, Honestly](/blog/trading/technical-analysis/what-technical-analysis-really-is). Before we can talk about support, resistance, trends, or any pattern, we have to answer a more basic question: *where does the chart come from?* What is actually happening in the half-second a candle represents? Once you can see the auction beneath the candle, half of technical analysis stops being mysticism and starts being plumbing.

![A graph showing buyers and sellers feeding an order book and matching engine, which prints trades onto the tape and aggregates them into one OHLC candle](/imgs/blogs/how-a-price-chart-is-born-1.png)

The diagram above is the mental model for this whole post: buyers and sellers post orders, those orders rest in an *order book*, a *matching engine* pairs the best buyer with the best seller, each pairing *prints* a trade, the prints stream onto a public record called the *tape*, and at the end of each time window the prints get squeezed into a single candle with an open, a high, a low, a close, and a volume. Everything else — the spread, slippage, gaps, why levels exist — falls out of that one machine. Let's build it from the bottom up.

## Foundations: a market is a continuous auction

Forget charts for a moment. Picture a marketplace — a literal one, with stalls. You want to buy apples. You walk around and find that the lowest price anyone is *currently willing to sell* an apple for is \$1.02. That's the **ask** (also called the *offer*): the cheapest price at which you can buy right now. Meanwhile, the *most* anyone is *currently willing to pay* for an apple is \$1.00. That's the **bid**: the highest price at which you can sell right now.

Notice the asymmetry that trips up every beginner: you cannot buy at the bid and you cannot sell at the ask. If you want to buy *immediately*, you pay the ask (\$1.02). If you want to sell *immediately*, you receive the bid (\$1.00). The two-cent gap between them — \$1.02 minus \$1.00 — is the **bid-ask spread**, and it is the first cost of trading nobody warns you about.

A financial market is exactly this marketplace, except (a) it never closes the negotiation — it runs continuously, microsecond by microsecond — and (b) instead of stalls, every buyer and seller posts their willingness to trade into a single shared list called the **order book**. The order book is the heart of the machine. Everything visible on a chart is downstream of it.

The technical name for this mechanism is a **continuous double auction**. "Double" because both sides bid — buyers compete to offer the highest price, sellers compete to offer the lowest — unlike a normal auction where only buyers bid against a single seller. "Continuous" because it never pauses to clear; a trade happens the instant a buyer's price meets a seller's price, and then the auction simply keeps going. Almost every modern exchange — for stocks, futures, options, and most crypto — runs some flavor of this continuous double auction over a central order book. (There are other designs — *dealer markets* where you trade against a single market maker's quote, and *automated market makers* in crypto that price trades off a formula instead of a book — but the order-book auction is the default mental model, and the one that produces the candlestick charts we're decoding.) Once you internalize "continuous double auction over a shared book," the rest of this post is just watching that machine run.

### The two kinds of orders

There are only two fundamental ways to express what you want, and the entire mechanism is built from them.

A **limit order** is a *conditional* instruction: "Buy me 100 shares, but pay no more than \$100.00," or "Sell my 200 shares, but accept no less than \$100.05." A limit order names a price and a quantity. If nobody is willing to trade with you at that price right now, your order doesn't vanish — it *rests* in the order book, advertising your willingness to trade, until either someone trades against it or you cancel it. A limit order is patient. It gives liquidity to the market: it's a standing offer other people can take.

A **market order** is an *unconditional* instruction: "Buy me 100 shares right now, whatever the going price is." A market order names a quantity but *not* a price. It doesn't rest in the book; it executes immediately by consuming the best resting limit orders on the opposite side. A market order is impatient. It *takes* liquidity: it removes standing offers from the book.

This buyer/taker distinction is the most useful lens in all of market microstructure (*microstructure* just means "the detailed mechanics of how trades actually happen"). Limit orders *make* the market — they're the resting offers that define the bid and the ask. Market orders *take* the market — they reach across and grab those offers. Most of the time, both sides of every trade are doing one of these two things.

> A *limit order* posts a price and waits; a *market order* names only a size and trades immediately against whatever's resting. That single choice — patience versus immediacy — determines whether you pay the spread or earn it.

### The order book as a price ladder

Now stack every resting limit order by price and you get the order book, often drawn as a **price ladder**. On one side are all the buy limit orders (the *bids*), sorted from highest price down. On the other side are all the sell limit orders (the *asks*), sorted from lowest price up. In the middle, where the two sides almost-but-don't-quite meet, sits the spread.

![A price ladder matrix with ask levels shaded red above the amber spread band and bid levels shaded green below, each row showing price resting size and side](/imgs/blogs/how-a-price-chart-is-born-2.png)

Read the ladder above from the middle outward. The **best bid** is \$100.00 with 400 shares behind it: that's the most anyone will currently pay, and there's demand for 400 shares at that level. The **best ask** is \$100.02 with 300 shares: the cheapest anyone will currently sell, with 300 shares available. The two-cent gap between them — \$100.00 to \$100.02 — is the spread. Together, the best bid and best ask are called the **touch** or the **inside market** or the **top of book**: the prices you can actually trade at right now.

Above the best ask, the asks get *more expensive* (\$100.03 with 500 shares, \$100.04 with 800 shares). Below the best bid, the bids get *cheaper* (\$99.99 with 600 shares, \$99.98 with 900 shares). The further you go from the touch in either direction, the worse the price — but usually the *more* shares are resting there, because traders are happy to post big orders at prices they consider bargains or premiums.

That word **liquidity** gets thrown around constantly, and here is its concrete meaning: *liquidity is how many shares you can trade near the current price without moving it much.* A market with a tight spread and large sizes resting close to the touch is **liquid** — you can buy or sell a lot without pushing the price around. A market with a wide spread and only a few shares at each level is **illiquid** or **thin** — even a modest order shoves the price. Liquidity is not a vibe; it is the literal depth of the book, and you can read it straight off the ladder.

### Who's actually in the book

It helps to know whose orders you're looking at, because the ladder is a crowd, not a single counterparty. Broadly, three kinds of participants populate the book:

- **Market makers** (and modern *high-frequency* liquidity providers) post limit orders on *both* sides at once — a bid and an ask — and try to earn the spread by buying at the bid and selling at the ask over and over, staying roughly flat. They are the reason a liquid stock has a tight, two-sided quote at all times: it's their job to be there. They are also the first to *pull* their quotes when things get scary, which is exactly why liquidity can vanish in a heartbeat.
- **Institutions** — pension funds, mutual funds, hedge funds — move large size and mostly use patient strategies (slicing big orders into many small ones, often as resting limit orders) to avoid walking the book against themselves. When you see a thick wall of size at one price, it's often an institution.
- **Retail traders** — individuals — are usually a small slice of the resting size and frequently use market orders, which means they *take* liquidity and pay the spread. If you trade from a phone app and tap "buy," you are almost certainly sending a market order and lifting someone's ask.

The book, then, is a live tension between liquidity *makers* (resting limit orders, earning the spread, supplying immediacy to others) and liquidity *takers* (market orders, paying the spread, demanding immediacy now). Price moves when takers on one side overwhelm the resting orders on the other.

### Price-time priority: who gets filled first

One more rule makes the auction deterministic. When several limit orders rest at the *same* price, which one trades first against an incoming market order? Almost every exchange uses **price-time priority**: orders are filled best-price-first, and *within* a price level, first-come-first-served by the time they were posted. So if you post a bid for 100 shares at \$100.00 and someone else posts a bid for 100 shares at \$100.00 a second later, an incoming market sell will fill *your* order first because you got there earlier. This is why being early to a price level matters, and why high-frequency traders fight over microseconds: at a busy level, the front of the queue gets filled and the back of the queue waits. It's a literal line, and the auction respects it.

## How a single trade prints

Here is the moment everything hinges on: how does a trade actually happen, and how does it become "the price"?

When you send a **market buy** order, the matching engine takes your order and fills it against the lowest-priced resting sell limit orders — the best ask first, then the next ask up, then the next, until your full quantity is filled. The instant your order matches against a resting order, a **trade prints**: a record is created saying "X shares traded at price P at time T," and that record streams onto the public **tape** (the time-and-sales feed). The price of that print becomes the **last trade price** — which is what most charts, tickers, and apps mean when they show you "the price."

So *the price is just the price of the most recent trade.* It is not an average, not a fair value, not a consensus — it is a single number, the level at which the last buyer and last seller happened to cross. The next trade can print at a different price a microsecond later. The chart is a flipbook of these last-trade prices.

It's worth being precise about *which* price a trade prints at. When a market buy lifts a resting ask at \$100.02, the trade prints at \$100.02 — the *resting* order's price, not the incoming order's (a market order had no price to begin with). The party who posted the limit order set the price; the party who sent the market order accepted it. This is why the resting side is said to *make* the price and the aggressing side is said to *take* it. The taker chose *when* and *whether* to trade; the maker chose *at what level*. Every print is a tiny contract between an impatient taker and a patient maker, executed by the engine at the maker's price.

### The matching engine, step by step

The **matching engine** is the piece of software at the exchange that does all this. It's simple in principle, even though real ones are heroically optimized. When an order arrives, the engine asks: *can this trade right now?* For an incoming market buy, "right now" means "is there any resting sell order?" — and there always is, at *some* price. The engine then:

1. Takes the incoming order and looks at the opposite side of the book (asks, for a buy).
2. Matches against the **best** (lowest) ask first, filling as many shares as rest there.
3. If the incoming order still has shares left to fill, moves to the *next* ask level and matches there.
4. Repeats until the incoming order is fully filled (or, for some order types, until a price limit is hit).
5. Emits a **print** for each level it matched at, each one streaming to the tape with its price, size, and timestamp.

A single large market order can therefore generate *several* prints at *several* prices in one logical action — which is exactly the "walking the book" behavior we're about to quantify. A limit order that *crosses* the spread (e.g., a buy limit at \$100.05 when the best ask is \$100.02) behaves like a market order up to its limit price: it matches everything from \$100.02 up to \$100.05, then rests whatever's left at \$100.05. A limit order that *doesn't* cross (a buy limit at \$99.95 when the best bid is \$100.00) simply joins the book and waits. The engine is the same; the only question is whether the new order can immediately meet an existing one.

### Walking the book and slippage

If your market order is small enough to be fully filled by the shares resting at the best ask, you pay exactly the touch and you're done. But if your order is *larger* than what's resting at the best ask, something important happens: your order **walks the book.** It eats all the shares at the best ask, then climbs to the next ask level and eats those, then the next, until it's filled. Each level you climb to is a *worse* price than the touch. The gap between the price you *expected* (the touch) and the average price you *actually paid* is called **slippage**.

Slippage is the second cost of trading, and it's purely a function of (a) how big your order is and (b) how deep the book is. A small order in a deep book barely slips. A big order in a thin book can slip enormously. Let's make it concrete.

![A bar chart showing a 1,000-share market buy filling 300 shares at 100.02, 500 at 100.03, and 200 of 400 at 100.04, averaging 100.029 with positive slippage versus the touch](/imgs/blogs/how-a-price-chart-is-born-3.png)

#### Worked example: a market order walks a five-level book

Suppose the ask side of the book looks like this:

| Ask price | Shares resting | Cumulative shares |
|---|---|---|
| \$100.02 (best ask / touch) | 300 | 300 |
| \$100.03 | 500 | 800 |
| \$100.04 | 400 | 1,200 |
| \$100.05 | 600 | 1,800 |
| \$100.06 | 700 | 2,500 |

You send a **market buy for 1,000 shares.** Here's exactly how it fills, level by level:

- **300 shares at \$100.02** = \$30,006. (You've eaten the entire best ask. 700 shares left to fill.)
- **500 shares at \$100.03** = \$50,015. (Best ask is now \$100.04. 200 shares left.)
- **200 shares at \$100.04** = \$20,008. (You only need 200 of the 400 resting here. Done.)

Total cost = \$30,006 + \$50,015 + \$20,008 = **\$100,029** for 1,000 shares.

Your **average fill price** is \$100,029 ÷ 1,000 = **\$100.029 per share.**

You *expected* to pay the touch, \$100.02. You *actually* paid \$100.029 on average. The difference, \$100.029 − \$100.02 = **\$0.009 per share** — nine-tenths of a cent — is your slippage. Across 1,000 shares that's \$9 you handed over simply because your order was bigger than the top level of the book. And note what you did to the chart: the *last* trade printed at \$100.04, so the ticker now reads \$100.04 even though the touch was \$100.02 when you started. You moved the price up two cents by buying — not because of "buying pressure" in some mystical sense, but because you mechanically consumed two price levels of resting sellers.

**The intuition:** a market order's true cost isn't the touch you see — it's the *average* of every level it has to climb to get filled, and that climb is the price you pay for immediacy.

### The tick size sets the finest grain

Notice every price in that book moved in **one-cent** steps: \$100.02, \$100.03, \$100.04. That step is the **tick size** — the smallest increment a price is allowed to change by, set by the exchange. U.S. stocks above \$1 generally trade in penny ticks; some futures trade in quarter-point ticks; some crypto pairs have far finer ticks. The tick size matters more than beginners expect: it sets a *floor* on the spread (the bid and ask can't be closer than one tick), so a one-cent tick on a \$100 stock means the *tightest possible* spread is 0.01%, while a one-cent tick on a \$2 stock means the tightest possible spread is 0.5% — fifty times wider as a percentage, purely because of the price level relative to the tick. This is one quiet reason cheap stocks are proportionally more expensive to trade: the tick is a bigger fraction of the price. When you look at a chart, the candles can only land on the tick grid; the smoothness is an illusion of zooming out.

### What the book doesn't show: hidden and iceberg orders

The ladder you can see is not always the whole truth. Two common features hide liquidity from the public view. An **iceberg order** displays only a small "tip" of its true size in the book and automatically reloads more as the visible portion gets filled — so a level that looks like 100 shares might actually have 10,000 behind it. **Hidden orders** don't display at all; they rest invisibly and only reveal themselves when a print occurs against them. The practical consequence: the *visible* depth is a lower bound on the real depth, and a market order can sometimes fill *better* than the visible book suggested (because hidden size was waiting), or a level that "looked thin" can absorb far more than expected. This is why even a perfect read of the visible ladder can't fully predict your fill — and why the chart, which only ever shows you *executed* prints, is downstream of a book whose true state you can never completely observe.

## From ticks to candles

We now have a stream of prints — each one a "ticked" price, a *tick* being the smallest recorded movement, and a print being a single executed trade. On a busy stock that stream can be hundreds of prints per second. No human can read a firehose of individual trades, so we *aggregate* them into bars. The most common bar is the **candlestick**, and the most common aggregation is by **time**: chop the trading day into equal windows (1 minute, 5 minutes, 1 hour, 1 day) and summarize all the prints in each window with four numbers.

Those four numbers are the **OHLC**:

- **Open** — the price of the *first* print in the window.
- **High** — the *highest* print price during the window.
- **Low** — the *lowest* print price during the window.
- **Close** — the price of the *last* print in the window.

That's the entire definition of a candle. Everything you've ever heard about candlestick "patterns" is built on top of just these four numbers per bar.

The *drawing* is a clever bit of visual encoding. The **body** of the candle is the rectangle spanning from the open to the close. If the close is *above* the open, the candle went up over the window, and it's drawn hollow or green (an *up* candle). If the close is *below* the open, it went down, and it's drawn filled or red (a *down* candle). The thin lines sticking out of the body — the **wicks** (or *shadows*) — reach up to the high and down to the low. So a single candle tells you, at a glance: where the window opened and closed (the body), and how far price stretched in each direction before settling (the wicks).

![A diagram showing eight blue print dots scattered over a minute, with the resulting candlestick on the right: green body from open 100.10 to close 100.15, upper wick to high 100.22, lower wick to low 100.08](/imgs/blogs/how-a-price-chart-is-born-4.png)

#### Worked example: building one 1-minute candle from eight prints

Suppose during the 10:00–10:01 minute, these eight trades printed, in order:

| Time | Print price |
|---|---|
| 10:00:03 | \$100.10 |
| 10:00:11 | \$100.14 |
| 10:00:19 | \$100.22 |
| 10:00:27 | \$100.16 |
| 10:00:35 | \$100.08 |
| 10:00:44 | \$100.12 |
| 10:00:51 | \$100.18 |
| 10:00:58 | \$100.15 |

Let's derive the OHLC:

- **Open** = the first print = **\$100.10** (at 10:00:03).
- **High** = the maximum across all prints = **\$100.22** (at 10:00:19).
- **Low** = the minimum across all prints = **\$100.08** (at 10:00:35).
- **Close** = the last print = **\$100.15** (at 10:00:58).

Because the close (\$100.15) is above the open (\$100.10), this is an **up candle** — green, with a body from \$100.10 to \$100.15. The **upper wick** stretches from \$100.15 up to the high of \$100.22; the **lower wick** stretches from \$100.10 down to the low of \$100.08. Notice how much the candle throws away: eight distinct prices, the entire path of the minute — which went up, then way up, then crashed to the low, then recovered — gets crushed into a small green body with two wicks. The candle is *lossy compression.* It keeps the four extremes and discards the journey.

**The intuition:** a candle's body is "where the window started and ended"; its wicks are "how far it reached and got rejected." A long upper wick means price tried to go higher and sellers slapped it back down; a long lower wick means buyers stepped in below. That's the entire grammar of a candle, and it comes straight from OHLC.

A subtle but important consequence: *two candles can look identical and have come from completely different paths.* Our minute could have ticked up-down-up-down or smoothly drifted — same OHLC, same candle. The candle does not record the order of the prints or how the price wandered between the extremes. When someone reads a deep narrative into a single candle, remember: it's four numbers. There's only so much it can mean.

### Reading the shape of a candle

Even with just four numbers, the *shape* of a candle carries a rough story, because the relative sizes of the body and the two wicks encode the balance of the window:

- **A long body with tiny wicks** means price opened, marched in one direction, and closed near its extreme — a decisive window where one side pressed all the way through. A long green body says buyers lifted offers steadily; a long red body says sellers hit bids steadily.
- **A tiny body with long wicks on both sides** (sometimes called a *doji*) means price thrashed up and down but finished right back where it started — lots of motion, no resolution. The window was a fair fight.
- **A long upper wick with a small body near the low** means price spiked up and got *rejected* — buyers pushed, sellers overwhelmed them, and the close came back down. The wick is the high-water mark of a failed advance.
- **A long lower wick with a small body near the high** means price dropped and got *bought back up* — sellers pushed, buyers stepped in below, and the close recovered. The wick marks where demand appeared.

These are descriptions of *what the four numbers imply*, not predictions. A long lower wick tells you buyers showed up *in that window*; it does not promise they'll show up again. Candlestick "patterns" — the named multi-candle formations you'll read about elsewhere — are just labeled combinations of these body-and-wick shapes across two or three bars. Every one of them reduces to OHLC arithmetic. Knowing that keeps you honest: a "hammer" or an "engulfing candle" is a shorthand for a particular OHLC relationship, useful as a *summary*, not a spell.

### Why the close is special

Of the four numbers, the **close** does the most work, and it's worth knowing why. The close is the last agreed price of the window — the market's most recent verdict — so it's the number most charts use to draw line charts, compute moving averages, and define "the price" of a day. The *daily* close is especially load-bearing: it's the price used to mark portfolios to market, settle many derivatives, and compute returns. That's why the final minutes of a trading day often see a surge of volume (the *closing auction*): a lot of participants specifically want to transact at, or influence, the official close. So while every candle has an open, high, low, and close, in practice the close is the number the rest of the world quotes back to you.

## What volume actually measures

Underneath the price candle, almost every chart shows a second series: **volume**, usually a bar for each time window. And volume is where the largest and most stubborn misunderstanding in all of charting lives. So let's be very precise.

**Volume is the number of shares (or contracts, or coins) that changed hands during the window.** That's it. If 4.5 million shares traded between 10:00 and 10:01, the volume bar for that minute is 4.5 million. Volume counts *transacted quantity.* It is a measure of *activity*, of how much trading happened.

Now the crucial fact that destroys the most common myth: **every single trade has a buyer and a seller.** They are two sides of the *same* transaction. When 4.5 million shares trade, that is 4.5 million shares *bought* and *simultaneously* 4.5 million shares *sold*. It is logically impossible for more shares to be bought than sold, or vice versa — a share doesn't leave one hand without entering another. So volume is *never* "buyers outnumbering sellers." For every buyer there is, by construction, an exactly equal seller. Volume measures the *size of the agreement*, not which side "won."

![Two stacked panels sharing an x-axis: a price panel of five candles trending up, and a volume panel below where the big up-candle prints on a 4.5M-share amber spike while the others print on thin gray volume](/imgs/blogs/how-a-price-chart-is-born-6.png)

So what *does* volume tell you? It tells you about **participation** and **conviction**. A price move on heavy volume means a *lot* of shares changed hands to get there — many participants were willing to transact, so the move reflects broad agreement and is, loosely, "better supported." A price move on thin volume means *few* shares moved it — it took very little real trading to push the number around, so the move is more fragile and easily reversed. Volume doesn't have a direction; it has a *weight*. It tells you how much the market "meant it."

#### Worked example: same price move, different volume

Imagine two stocks, both trading at \$100.00, and both rise to \$100.20 over the same hour — an identical +\$0.20 (+0.2%) move on the chart. If you only looked at the price candle, they'd be twins.

- **Stock A** rose to \$100.20 on **4.5 million shares** traded. That's heavy participation: lots of real buyers lifting lots of real sellers, level after level. The move is *backed by* a large amount of transacted demand. If the price holds, the \$100.20 level now has a lot of people who transacted around it — it's a meaningful level.
- **Stock B** rose to \$100.20 on **120,000 shares** traded — a thin, sleepy book where a handful of orders nudged the last-trade price up through nearly-empty levels. The +0.2% is *real* in the sense that trades printed there, but it's *unsupported*: almost nobody participated. A single modest sell order could erase it.

Same +\$0.20 on the price panel; wildly different stories underneath. Volume is the second dimension that tells them apart.

**The intuition:** price tells you *where* the last trade happened; volume tells you *how much trading it took to get there.* A move without volume is a move without a crowd behind it.

There's a popular phrase, "volume is buying pressure," and you should retire it. There is no such thing as net buying pressure in the volume number, because buys and sells are mechanically equal. What people *mean* — and what's defensible — is that *aggressive* market orders on one side (impatient takers lifting the ask, or hitting the bid) tend to move price, and heavy volume *accompanying* a directional move signals that the move had real takers behind it. But the volume bar itself is just transacted quantity. Direction comes from *price*; volume only tells you the size of the crowd.

### The one true thing close to "buying pressure": order flow

There *is* a defensible cousin of "buying pressure," and it has a name: **order flow** (or *trade direction*). The trick is to classify each print by *who was the aggressor* — the taker who crossed the spread. A trade that prints at (or near) the *ask* was almost certainly initiated by an impatient *buyer* lifting an offer; a trade that prints at (or near) the *bid* was almost certainly initiated by an impatient *seller* hitting a bid. By tagging each print this way (a method known as the *tick rule* or *Lee-Ready algorithm*), you can compute **net order flow**: buyer-initiated volume minus seller-initiated volume.

This is the closest legitimate thing to "buying versus selling," and it's genuinely informative — sustained buyer-initiated flow tends to lift price, because takers are walking *up* the ask. But notice three caveats. First, even here the *total* shares bought still equals the total sold; what's imbalanced is the *aggression*, not the count. Second, this requires more than the raw volume bar — you need the trade-direction tags, which not every chart shows. Third, classification is imperfect (a trade printing *between* bid and ask is ambiguous). So when a sophisticated trader talks about "buying pressure," they usually mean *net aggressive order flow*, an actual measured quantity — not the volume bar, and not a hand-wave about there being "more buyers."

### Volume and liquidity are cousins, not twins

One last distinction, because beginners conflate them. **Liquidity** is how many shares are *resting* in the book right now, available to trade — a snapshot of *standing offers*. **Volume** is how many shares *actually traded* over a window — a record of *completed transactions*. A market can be highly liquid (deep book) but have low volume (few people choosing to trade), or have a thin book but a burst of volume (a few big market orders mopping up what little was there). They're related — liquid markets *tend* to have higher volume — but they answer different questions: liquidity is "how much can I trade right now without moving price?"; volume is "how much got traded?" Keep them separate and a lot of chart commentary stops being confusing.

## The spread, liquidity, and the cost of trading

We've met the spread twice now — as the gap on the ladder and as a cost. Let's pin down exactly how much it costs, because it is the single most underrated drag on a trader's results.

The spread is a cost because of the asymmetry we started with: you *buy at the ask* and *sell at the bid.* To get into and out of a position — a **round trip** — you cross the spread twice in effect: you pay up to get in, and you give up to get out. If the price doesn't move at all between your entry and exit, you still lose the spread.

![A before-after diagram contrasting crossing the spread, where buying at ask 100.02 and selling at bid 100.00 loses 2.00 on 100 shares, against the mid price view where the chart shows break-even but you are down the spread](/imgs/blogs/how-a-price-chart-is-born-5.png)

The figure above makes the trap visible. Suppose the bid is \$100.00 and the ask is \$100.02 — a two-cent spread, with a *mid price* of \$100.01 right in between. You buy 100 shares at the ask, paying \$100.02 × 100 = \$10,002. A moment later, with the book unchanged, you sell those 100 shares at the bid, receiving \$100.00 × 100 = \$10,000. You're out \$2 — the full spread (\$0.02) times your 100 shares — even though the *mid price* the chart often plots never budged from \$100.01. The chart says "break-even." Your account says "down \$2." That gap is the spread you paid for the privilege of immediacy on both legs.

#### Why does the spread exist at all?

It's natural to ask: if the spread is pure cost to me, who's collecting it, and why don't they just compete it to zero? The collector is whoever posted the resting limit orders — typically a market maker quoting both sides. They earn the spread by buying at the bid from impatient sellers and selling at the ask to impatient buyers, pocketing the difference when both happen. But the spread isn't free money for them; it compensates two real risks. The first is **inventory risk**: a market maker who buys at the bid is now *holding* shares they didn't want, exposed to the price moving against them before they can sell. The second, deeper one is **adverse selection**: the impatient trader lifting your ask might be lifting it *because they know something you don't* — they have information that the price is about to rise, and you just sold to them right before it does. Market makers widen the spread precisely to cover the losses they take to better-informed traders. This is why spreads *widen* when uncertainty spikes (around news, at the open, during turmoil): the risk of trading against someone informed goes up, so the compensation demanded goes up, and the toll you pay to cross gets steeper. The spread, in other words, is the price of *immediacy* and the market maker's premium for *bearing risk and ignorance* on the other side of your trade.

#### Worked example: the cost of the spread on a \$10,000 round trip

Let's scale it to a realistic position and express the spread as a percentage, which is how costs are usually compared. Say you trade a stock where the spread is **0.05%** of the price — a typical figure for a liquid mid-cap (it would be far tighter for a mega-cap like Apple, far wider for a small-cap). You put **\$10,000** into a round trip.

- A 0.05% spread on a \$10,000 position is 0.0005 × \$10,000 = **\$5.** That's the spread cost on *one leg* expressed against your position size — but the standard way to think about it is that **crossing the spread once costs you half the spread relative to the mid, and a full round trip costs you the whole spread.** So the round-trip spread cost is approximately **\$5** on your \$10,000.
- Put differently: if the mid price is \$100 and the spread is \$0.05 (which is 0.05%), you buy at \$100.025 and sell at \$99.975. On 100 shares (\$10,000), that's \$10,002.50 paid and \$9,997.50 received — a **\$5 round-trip loss** before any price movement.

Five dollars on \$10,000 sounds trivial — until you do it often. If you make **one round trip a day** for a year, that's roughly 250 × \$5 = **\$1,250**, or **12.5% of your \$10,000**, vaporized into spreads alone. If you're a high-frequency day-trader doing twenty round trips a day, the spread can dwarf every other consideration on the chart. This is *why* costs matter so much more for active traders than for buy-and-hold investors, and it's a major reason most frequent traders underperform — they're paying the spread (and slippage, and commissions) over and over.

**The intuition:** the spread is a toll booth you pass through on every round trip, and the chart's mid-price line hides it. The faster you trade, the more tolls you pay.

### The four costs of a round trip

The spread is the most invisible cost, but it's not the only one. To trade *honestly* you should hold all four in your head, because together they explain why active trading is so much harder than it looks:

1. **The spread** — the gap between bid and ask, paid because you buy at the ask and sell at the bid. Tighter for liquid names, wider for illiquid ones. Always present, always against you.
2. **Slippage** — the *extra* cost when your order is large enough to walk the book past the touch, paying worse prices at each level. Zero for tiny orders in deep books; potentially enormous for large orders in thin ones.
3. **Commissions / fees** — what your broker or the exchange charges per trade. Many retail brokers advertise "zero commissions," but the cost hasn't vanished; it's often recouped through *payment for order flow*, where your order is routed to a market maker who pays the broker for the privilege of filling it — which can quietly widen your effective spread.
4. **Market impact** — the degree to which *your own* trading moves the price against you and signals your intentions to others, who may trade ahead of you. For a retail-sized order this is negligible; for an institution moving size, it's the dominant cost and the entire reason VWAP-style slicing exists.

A buy-and-hold investor who trades twice a decade barely notices any of these. A day-trader who round-trips twenty times a day pays all four, twenty times a day, every day — and the chart, which plots a clean mid-price line, shows none of it. This is the single most important reason the *same chart* is a benign backdrop for an investor and a minefield for a frequent trader.

### When liquidity vanishes: slippage and gaps

Liquidity isn't constant. The book is deep and the spread is tight when many participants are active and confident; it thins out — sometimes instantly — when they pull their orders. And when the book thins, two things get worse fast: **slippage** balloons, and the price can **gap.**

![A before-after diagram contrasting a thin book where a 1,000-share order walks up eight levels for 2 percent slippage against a deep book where 20,000 shares rest at the touch and the same order fills flat](/imgs/blogs/how-a-price-chart-is-born-8.png)

The figure contrasts the two regimes with the *same order* — a 1,000-share market buy. In a **deep book** (a large-cap stock, say), there might be 20,000 shares resting right at the best ask; your 1,000-share order doesn't even clear the top level, so you fill at the touch with essentially zero slippage. In a **thin book** (an illiquid small-cap), there might be only 100 shares at each level; your 1,000-share order has to climb *eight levels* to get filled, paying steadily worse prices the whole way and ending with an average fill 2% above where it started. Same order, same intent — radically different cost, entirely because of the depth of the book.

A **gap** is the extreme version of this. A gap is when the price *jumps* from one level to a distant one with no trades in between — the chart literally has a hole. Gaps happen when liquidity on one side of the book evaporates faster than it can be replenished: a flood of sell orders hits a book with almost no resting bids, so the last-trade price tumbles through empty levels until it finds buyers far below. The most common everyday gap is the **overnight gap**: news breaks while the market is closed, and when it reopens, the first trade prints far from the previous close because the resting orders have all been repriced. We'll see real examples shortly.

This is also, by the way, *why support and resistance levels exist at all* — a topic we go deep on in [Support and Resistance: Why Levels Exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist). A "level" on a chart is usually a price where a lot of liquidity is resting (a wall of bids or asks) or where a lot of people have a reason to transact. Price pauses there because there's genuinely something in the book to chew through. Levels aren't magic lines; they're places where the order book is thick.

### The tape: where prints become public

Every print, the moment it happens, streams onto the **tape** — the public **time-and-sales** record of every executed trade, in order, with its price, size, and timestamp. ("Reading the tape" is one of the oldest skills in trading, from the era of literal ticker tape spooling out of a machine.) The tape is the *raw* truth: it's the unaggregated firehose of individual prints, before anything compresses them into candles. A chart is a *summary* of the tape; the tape is the source.

Why care, if you only read charts? Because everything on the chart is a lossy view of the tape, and knowing the source keeps you grounded. When a candle shows a long lower wick, the tape would show you the exact sequence of prints that pushed down and then bought back up — the speed, the sizes, whether it was one big order or a thousand small ones. The candle throws all of that away and keeps four numbers. You don't need to read the tape to trade off charts, but you should *remember it exists*, because it's the reason a candle can only ever be an approximation: it's a four-number sketch of a stream that contained thousands of distinct facts.

### Price discovery: what the auction is actually *for*

Step back and ask what this whole machine accomplishes. The continuous double auction performs **price discovery**: it is society's mechanism for aggregating the scattered opinions, information, and needs of thousands of participants into a single, continuously-updated number — the last-trade price. Nobody decrees the price; it *emerges* from the auction as impatient takers meet patient makers. When new information arrives (an earnings report, a rate decision), participants reprice their resting orders, the book shifts, takers cross the new spread, and within seconds the last-trade price reflects the news. The chart is a recording of price discovery in action — a time-lapse of the market figuring out what something is worth, trade by trade, never finished. This is why the price is *informative* (it aggregates a lot of opinion) but *not* the final word on value (it's just the latest marginal transaction). Holding both ideas at once — the price is meaningful *and* it's only the last trade — is the mark of reading charts honestly.

## Bar types

We've been assuming **time bars** — one candle per fixed clock interval. They're the default everywhere, and they're intuitive. But they have a quirk worth understanding, and there are two alternatives that fix it.

The quirk: a time bar closes on the *clock*, regardless of how much trading happened. During a sleepy lunch hour, a 1-minute bar might contain three trades; during the opening surge, the *same* 1-minute bar might contain fifty thousand. Time bars therefore *over-sample* quiet periods (lots of near-empty bars) and *under-sample* violent ones (a single bar swallows a huge, information-rich burst). The clock and the action are out of sync.

![A matrix comparing time, tick, and volume bars across what closes a bar, behavior in a quiet five-minute window, and behavior in a five-minute opening burst](/imgs/blogs/how-a-price-chart-is-born-7.png)

The two alternatives change *what closes a bar*:

- **Tick bars** close after a fixed *number of trades* (say, every 500 prints), regardless of how long that takes. In a quiet patch, one tick bar might span ten minutes; in the opening frenzy, you might print forty tick bars in five minutes. Tick bars sample by *activity*, so each bar carries roughly equal information.
- **Volume bars** close after a fixed *number of shares* traded (say, every 50,000 shares). Like tick bars, they speed up when the market is busy and slow down when it's quiet, but they normalize by *quantity transacted* rather than *number of trades* — which matters when trade sizes vary a lot.

There are two more variants worth a sentence each, because you'll meet them:

- **Dollar (or "value") bars** close after a fixed *dollar amount* changes hands (say, every \$10 million traded) rather than a fixed share count. They're like volume bars but normalize for price level, which matters over long horizons or across assets — \$10 million is \$10 million whether the stock is \$5 or \$500, whereas "50,000 shares" means very different value at those two prices.
- **Range bars** close whenever price has moved a fixed *distance* (say, every \$0.10 of travel), ignoring both time and volume. A flat market produces almost no range bars; a trending market produces a steady stream. They make trends visually cleaner at the cost of throwing away the time axis entirely.

Why does any of this matter? Because most chart statistics and patterns implicitly assume each bar carries comparable information, and time bars violate that assumption badly around opens, closes, and news. Many quantitative traders prefer tick, volume, or dollar bars precisely because the resulting series has nicer statistical properties (more stable variance, less clustered activity) — a point made at length in the literature on *financial machine learning*. For a beginner reading a chart, the lesson is narrower but real: **a "1-minute candle" at 9:30 a.m. and a "1-minute candle" at noon are not the same kind of object** — one is a hurricane compressed into a body and wicks, the other is a near-empty box. Knowing what closed the bar tells you how much to trust it. And whenever someone shows you a chart, it's fair to ask: *what kind of bars are these, and what closed them?*

## Common misconceptions

This is where reading charts honestly pays off. Each of these is a belief that sounds right, is widely repeated, and is mechanically wrong.

**"More buyers than sellers pushes the price up."** Impossible, by construction. Every trade is a buyer *and* a seller in equal measure — a share can't be bought without being sold. What actually moves price up is *aggressive* buyers: takers willing to lift the ask with market orders, consuming resting sell limit orders and climbing the book. The number of buyers equals the number of sellers exactly; what differs is which side is *impatient*. Price rises when impatient buyers walk *up* the ask side; it falls when impatient sellers walk *down* the bid side. "More buyers than sellers" is a description of *aggression*, sloppily phrased as a description of *count*.

**"Volume is buying pressure."** No — volume has no direction. It's the total quantity transacted, and since every trade has matched buyer and seller, you cannot decompose a raw volume bar into "buying" and "selling." What you *can* sometimes infer is whether the *aggressor* was more often the buyer or the seller (some data feeds tag each trade as buyer-initiated or seller-initiated), and *that* asymmetry of aggression relates to direction. But the volume bar itself is just a count of shares. High volume on an up-move means "a big crowd transacted as price rose," not "buying beat selling."

**"The price is the value."** The price is the level of the *last trade* — the most recent point where one impatient party crossed the spread to meet a resting one. It is a real, executed number, but it's a single transaction at the margin, not a verdict on what the asset is *worth*. In a thin market, the last-trade price can be wildly unrepresentative — one small order printing through empty levels can set a "price" that no one else agrees with. Price is *where the last deal happened*; value is a separate question entirely.

**"A green candle means the buyers won."** A green candle only means the *close* was above the *open* for that window — nothing about who "won." Concretely: say a minute opens at \$100.00, immediately rockets to \$100.50 (high), then sellers crush it all the way down to \$99.60 (low), and it limps back to close at \$100.01. That candle is *green* — the close (\$100.01) edged the open (\$100.00) — yet sellers controlled almost the entire window and dragged price 90 cents off its high. The body is a tiny green sliver; the wicks are huge. The candle keeps four numbers and discards the fight. A green body tells you the endpoint beat the start by a hair; it does not tell you the story in between, and it certainly doesn't crown a winner. (The wicks hint at the struggle — that long upper wick says buyers tried higher and got rejected hard — but even that is a coarse summary.)

**"If price moved, someone with information moved it."** Sometimes. But price also moves for entirely mechanical reasons: a large order forced to execute regardless of price (an index fund rebalancing, a fund meeting redemptions), a stop-loss cascade where falling prices trigger more sell orders that push prices lower still, or simply a thin book where tiny orders have outsized effects. Not every tick is a signal; a lot of price movement is just the plumbing — liquidity, forced flows, and the mechanical consequences of the auction — rather than anyone "knowing" anything.

**"A bigger volume bar means a more important price."** Volume measures participation, which is *related* to importance but not the same thing. A huge volume spike can be a single block trade crossed between two institutions at a pre-agreed price — enormous volume, but it tells you little about where the *next* trade will print. Context matters: volume confirms a move when it accompanies price travel through the book, but a volume number in isolation is just activity, not significance.

## How it shows up in real markets

The auction beneath the candle is not a metaphor. It is what *actually happens*, and when it behaves strangely, the chart shows it. Here are real, named episodes where understanding the machinery is the difference between confusion and clarity. (These are historical illustrations, not trading recommendations; the point is the mechanism, not the outcome.)

### The May 6, 2010 "Flash Crash" — liquidity vanishing in minutes

On the afternoon of May 6, 2010, the U.S. stock market fell roughly 9% and then largely recovered within about 36 minutes. The Dow Jones Industrial Average dropped nearly 1,000 points intraday before snapping back. What happened, mechanically, is exactly the gap-and-slippage story above, at the scale of the whole market. A large automated sell program began offloading futures contracts aggressively; market makers and other liquidity providers, seeing the onslaught and unsure what was going on, *pulled their resting orders* — the bids evaporated. With the bid side of the book suddenly thin, sell orders walked *down* through nearly-empty levels, and the last-trade price collapsed. Some individual stocks printed trades at absurd levels — a few traded for as little as a penny, others spiked to \$100,000 — because their books had essentially no resting orders, so a market order found nothing reasonable to match against and printed at whatever stub quote remained. The official report (CFTC-SEC, September 30, 2010) describes precisely this liquidity withdrawal. The lesson: a price chart can show a near-vertical cliff not because the asset's "value" changed, but because the *order book emptied* and prices fell through the holes. Liquidity is conditional, and it is most likely to disappear exactly when you'd want it most.

### An illiquid small-cap with a huge spread

Walk away from the mega-caps and the machinery becomes vivid. Consider a typical micro-cap stock — a company worth, say, \$30 million, trading a few thousand shares a day. Its order book might show a best bid of \$2.40 and a best ask of \$2.55: a **fifteen-cent spread on a \$2.50 stock, over 6%.** Crossing that spread on a round trip means you're down 6% the instant you're in and out, before the price moves at all. Let's make it concrete: you buy 2,000 shares at the \$2.55 ask, paying \$5,100. If you needed out immediately and sold at the \$2.40 bid, you'd receive \$4,800 — a **\$300 loss, 5.9% of your \$5,100, on a stock that never moved.** And that assumes the bid even holds 2,000 shares; if only 300 rest there, your sell walks *down* the bid side and you do worse still. Worse, that same \$5,000 market order on the way *in* walks the ask side and moves the "price" several percent by itself. On the chart, such a stock prints jagged, gappy candles with long wicks and erratic volume — not because anything fundamental is happening, but because *the book is thin*. Beginners who screen for "stocks making big moves" routinely land in exactly these names and discover that the move they saw was unrepeatable: they *were* the move, and the spread ate them on the way out. This is microstructure, not signal.

### An opening gap on news

Earnings season makes the overnight gap a weekly event. Suppose a company closes one day at \$50.00 and, after the bell, reports earnings far above expectations. Overnight, every resting order in the book is stale — nobody wants to buy at \$50 anymore when the news implies \$58, and nobody wants to sell at \$50 either. When the market reopens the next morning, the *first* trade might print at \$57.80, leaving a **gap** of nearly \$8 on the chart with no trades in between. The candle for the prior day closed at \$50; the candle for the new day opens at \$57.80; the space between is empty because the auction never traded there — the book repriced wholesale while it was closed. This is why "the price gapped up on earnings" is a literal description of the order book, not a metaphor: the resting liquidity at the old levels was cancelled and reposted higher, so the continuous auction resumed at a discontinuously different place. (Cited as a mechanism; specific tickers and dates vary by quarter.)

### Why VWAP exists for institutions

If you run a pension fund and need to buy two million shares of a stock that trades five million shares a day, you have a problem: a single market order for two million shares would walk the book catastrophically, slipping the price up against yourself and announcing your intentions to everyone. So institutions slice the order into thousands of small child-orders spread across the day, aiming to transact *near the average price* rather than chasing the touch. The benchmark they measure themselves against is **VWAP** — the *volume-weighted average price*, the average price of every trade in the day, weighted by the volume at each price. VWAP exists *because* of everything in this post: because market orders cause slippage, because volume measures participation, because the book has finite depth. A trader "beating VWAP" bought below the day's participation-weighted average; "missing VWAP" means their own slippage and timing cost them. The very existence of VWAP as a universal institutional benchmark is proof that the spread, depth, and the print-by-print nature of trading are first-order concerns for anyone moving size — not footnotes. (VWAP is a standard execution benchmark; the figures here are illustrative.)

### Crypto: the same machine, fewer guardrails

The continuous double auction isn't unique to stock exchanges — centralized crypto exchanges run order books that work identically: bids, asks, a spread, a matching engine, prints on a tape, candles on a chart. But crypto markets famously *amplify* the microstructure effects in this post, for two reasons. First, liquidity is fragmented across dozens of venues and can be thin on any single one, so the same coin can show meaningfully different prices and spreads on different exchanges at the same moment — and a market order on a thin venue walks the book hard. Second, many crypto venues offer high *leverage*, which means falling prices trigger forced *liquidations* — the exchange automatically closing leveraged positions by firing market orders into the book — producing exactly the cascade dynamic we describe below, but faster and larger. Episodes where a coin "wicks" violently down and snaps back within seconds are usually this: a liquidation cascade chewing through a thin book, printing trades far from the prior price, then recovering as fresh liquidity arrives. The candle's long lower wick is the literal footprint of the book briefly emptying. (Mechanism described generally; crypto liquidity and leverage rules vary by venue and over time.)

### Foreign exchange: a spread you pay without seeing a book

When you change money at an airport kiosk or in a retail forex app, you meet the bid-ask spread in its purest, most painful form — even though you never see an order book. The kiosk quotes you one rate to *buy* euros and a worse rate to *sell* them back; the gap is the spread, and at a tourist kiosk it can be several percent. In the wholesale interbank market, major currency pairs trade on extremely tight spreads (fractions of a *pip* — a pip being the fourth decimal place, the standard unit of FX price movement), because the book is enormously deep. The chart you see for, say, EUR/USD is built from that deep, fast continuous auction. But the *retail* you, trading through an app, is quoted a wider spread that bakes in the provider's margin — the same mechanism as a stock's bid-ask, just sometimes hidden inside a single "rate" rather than shown as two sides of a book. The lesson generalizes: wherever you can buy and sell, there is a spread, and whoever quotes you both sides is earning it.

### A stop-loss cascade

A more subtle mechanical episode: many traders place *stop-loss* orders — instructions to sell automatically if price falls to a certain level — clustered just below obvious chart levels. When price drifts down into that cluster, the stops trigger, firing market sell orders into the book. Those market sells walk *down* the bid side, pushing price lower, which triggers *more* stops below, which fire *more* market sells — a self-reinforcing cascade. On the chart it looks like a sudden, sharp drop "for no reason." Mechanically, it's a feedback loop in the order book: forced sellers consuming resting bids, the falling last-trade price tripping the next batch of forced sellers. Understanding this is why experienced traders are wary of placing stops at the same obvious level as everyone else — they know the book has a memory of where the orders are clustered, and that clusters get hunted.

## The same auction, seen at different zoom levels

One more idea ties the whole machine together and resolves a confusion almost every beginner hits: *the timeframe of a chart is just how coarsely you aggregate the same stream of prints.* A 1-minute chart, a 1-hour chart, and a daily chart of the same stock are not three different markets — they are three different *compressions* of the identical underlying tape. The daily candle's open is the first print of the day; its close is the last print; its high and low are the day's extremes. Drill into that one daily candle and you'd find 390 one-minute candles (a U.S. trading day is 390 minutes), each of which is itself a summary of however many prints occurred in that minute. It's compression all the way down to the individual print.

This has a practical consequence for *honesty*. A move that looks dramatic on a 1-minute chart can be an invisible blip on the daily chart; a "level" that looks rock-solid on the daily can be noise on the 1-minute. Neither is more "real" — they're the same prints, binned differently. When two traders argue about whether a stock is "breaking out," they're often just looking at different bin sizes of the same tape and not realizing it. The right question is never "what is the chart saying?" but "*at what aggregation* am I looking, and does my conclusion survive zooming in or out?" A claim about price that only holds at one specific timeframe is usually a claim about the *binning*, not about the market.

It also explains why your *own* trading horizon should pick your timeframe, not the other way around. If you hold positions for months, the print-by-print microstructure in this post barely touches you — you cross the spread twice a year, slippage is a rounding error, and a daily or weekly chart is the natural aggregation. If you hold for minutes, the spread, slippage, tick size, and book depth are the *dominant* facts of your trading life, and a 1-minute chart is showing you a coarsened version of the auction you're actually fighting inside. The machinery is identical; how much of it bites *you* depends entirely on how often you cross the spread.

## When this matters to you / further reading

Here is the payoff. Once you can see the auction beneath the candle, a lot of chart-reading folklore reorganizes itself into mechanics you can actually reason about:

- When you wonder *why a level held*, you now ask: *was there real liquidity resting there?* Levels are thick parts of the book, not magic numbers — which is the whole subject of [Support and Resistance: Why Levels Exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist).
- When you read a trend, you can ask whether the moves are happening *on volume* (participation) or drifting on thin trade (fragile) — the foundation for [Trend and Market Structure](/blog/trading/technical-analysis/trend-and-market-structure).
- When you're tempted by a stock "making a big move," you can check the spread and the depth first, and ask whether the move is signal or just a thin book printing through empty levels.
- And every time you place an order, you now know the difference between *taking* liquidity (a market order — instant, but you pay the spread and risk slippage) and *making* it (a limit order — patient, you might earn the spread, but you might not fill at all).

The honest version of technical analysis starts here, not with patterns. A candle is four numbers summarizing a continuous auction; volume is the count of shares that agreed to trade; the spread is a real toll; liquidity is why levels exist and why prices sometimes fall through holes. None of it is mystical. It's plumbing — and plumbing you can read.

If you're new to the series, start with [what technical analysis really is](/blog/trading/technical-analysis/what-technical-analysis-really-is) for the framing, then return here for the machinery. From this foundation, every later topic — levels, trends, ranges, volatility — is just a different way of looking at the same order book doing the same thing it always does: matching the impatient with the patient, one print at a time.

*This article is educational, explaining how markets mechanically work; it is not financial advice, and nothing here is a recommendation to buy or sell any security. Historical episodes are cited to illustrate mechanisms, with figures stated as of the dates given.*
