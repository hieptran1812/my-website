---
title: "Market Makers and High-Frequency Trading: The Invisible Counterparty to Every Trade"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How a small set of firms quietly take the other side of nearly every stock trade, earning tiny spreads at enormous scale, and why that makes markets cheaper but raises hard questions about fairness."
tags: ["market-makers", "high-frequency-trading", "payment-for-order-flow", "bid-ask-spread", "liquidity", "flash-crash", "etf-arbitrage", "market-structure", "citadel-securities", "financial-institutions"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Market makers and high-frequency traders are the unseen counterparty to most trades: they stand ready to buy when you sell and sell when you buy, earning a tiny spread on each trade and making it up on staggering volume. They make markets cheaper and more liquid than ever, but they also raise hard questions about payment for order flow, fairness, and the occasional terrifying day when the machines step back.
>
> - When you buy a stock, you almost never trade with another ordinary investor. A professional firm takes the other side, and its profit is the gap between the price it will buy at and the price it will sell at, the **bid-ask spread**.
> - That spread can be a single penny. The business works because these firms touch millions of trades a day, so a penny here and a fraction of a cent there add up to billions of dollars a year.
> - Many retail brokers route your orders to these firms and get paid for it, an arrangement called **payment for order flow (PFOF)**. It funds your zero-commission trading but creates a real conflict of interest worth understanding.
> - The same speed and automation that make spreads tiny can amplify trouble: in the May 6, 2010 **Flash Crash**, roughly a trillion dollars of value briefly vanished and recovered within minutes.
> - Market makers also quietly keep **ETFs** trading near the fair value of what they hold, a piece of plumbing that held up even in the chaos of March 2020.
> - The one idea to keep: these firms are a service, not a charity and not a conspiracy. They are paid to be the counterparty nobody else wants to be at that instant, and the debate is whether the price you pay them is fair.

Here is a question that sounds simple and is not. When you open an app, tap "Buy 100 shares of Apple," and see the order fill a half-second later, who exactly sold you those 100 shares? Most people assume the answer is another person somewhere who happened to want to sell at that exact moment. It almost never is. The overwhelmingly likely answer is that a professional trading firm you have never heard of created those shares out of its own inventory, sold them to you at a price a hair above what it would have paid to buy them, and instantly went looking to buy 100 shares somewhere else to replace what it just handed you. That firm is a **market maker**, and it is the invisible counterparty to your trade. The diagram above is the mental model for this entire post: your order does not march straight to a grand exchange and meet a matching human; it usually passes through your broker to a market maker that fills it, and only afterward does anything touch an exchange.

![Pipeline showing a retail order going from you to a broker to a market maker then an exchange](/imgs/blogs/market-makers-and-high-frequency-trading-1.png)

This is one of the least understood parts of modern finance, and it is invisible precisely because it works so well. Trading a major stock today costs almost nothing, fills almost instantly, and rarely fails. That smoothness is manufactured, every second, by firms whose entire job is to always be willing to trade with you when nobody else is. They are the reason "the market" feels like a thing you can always step into and out of, rather than a noticeboard where you post an offer and hope someone replies next Tuesday.

We are going to build this from the ground up. If you do not know what a bid is, what an ask is, or why a "spread" would make anyone rich, you are exactly the reader this is written for. We will define every term the first time it appears, ground every idea in a worked example with real dollar figures, and only then climb up to the contested questions: Is payment for order flow a ripoff or a gift? Are high-frequency traders parasites or plumbers? And what really happened on the afternoon the market fell off a cliff and climbed back up before most people noticed?

## Foundations: bids, asks, spreads, and the firm that quotes both

Let us start with the smallest piece and build up, because everything else is just this idea repeated at scale.

A **stock** is a slice of ownership in a company. To trade one, you need a price, and here is the first surprise: at any instant there is not one price but two. There is the **bid**, the highest price someone is currently willing to *pay* to buy the stock, and the **ask** (also called the *offer*), the lowest price someone is currently willing to *accept* to sell it. If the bid is \$49.99 and the ask is \$50.01, that means: right now, the best buyer will give you \$49.99 if you want to sell, and the best seller will charge you \$50.01 if you want to buy. The gap between them, here two cents, is the **bid-ask spread**.

Why is there a gap at all? Because buyers and sellers rarely show up at the exact same instant wanting the exact same quantity at the exact same price. If you want to sell *right now* and the nearest buyer was not planning to buy until the price drops a bit, someone has to bridge that gap. That someone is the **market maker**: a firm that continuously posts both a bid and an ask, promising to buy from anyone at its bid and sell to anyone at its ask. It is, quite literally, *making a market*, creating a place where you can always transact, by being willing to take either side.

The market maker's profit is the spread. When it buys from a seller at \$49.99 and, moments later, sells to a buyer at \$50.01, it pockets the two-cent difference. It is not betting on the stock going up or down. It is being paid two cents for the service of being there, ready, on both sides, so that neither the buyer nor the seller had to wait. That fee for immediacy is the heart of the whole business.

This is why **liquidity** is the word that haunts every page that follows. Liquidity is the ease with which you can trade an asset quickly, in size, without moving its price. A stock like Apple is highly *liquid*: you can buy a million dollars of it in a blink and the price barely twitches, because market makers stand ready with deep, tight quotes. A tiny company's stock, or a house, or a rare painting, is *illiquid*: to sell quickly you may have to slash your price to tempt a buyer out of the woodwork. Market makers are, fundamentally, liquidity factories. They manufacture the ability to trade now.

To use a market, you place an **order**, and the *type* of order you place determines who you become in this story. A **market order** says "fill me immediately at the best available price." If you send a market order to buy, you pay the ask; if you sell, you receive the bid. A market order *takes* liquidity, it consumes a quote that someone else posted, and so it pays the spread. A **limit order** says "only fill me at this price or better." A limit order to buy at \$49.99 sits and waits; it *provides* liquidity, becoming a quote that someone else might take. Market makers are the largest, fastest, most relentless posters of limit orders on earth. As a retail investor sending market orders, you are usually the one paying the spread; the market maker is usually the one collecting it.

Now we add the modern twist: speed. Markets are no longer rooms full of shouting people. They are computer systems, and orders arrive and match in **microseconds** (millionths of a second). When competition is about who can update a quote or grab a stale price first, physical distance to the exchange's computers matters, because data travels at a finite speed. **Latency** is the delay between sending an instruction and it taking effect, and shaving microseconds off it is worth real money. Firms pay exchanges for **colocation**, the right to place their own servers in the same building as the exchange's matching engine, so their signals travel a few meters instead of a few miles. When a firm's edge is being microseconds faster than everyone else, we call its style **high-frequency trading (HFT)**: automated strategies that place, cancel, and adjust enormous numbers of orders at machine speed, holding positions for fractions of a second to minutes.

Two more terms and we can climb. The first is **payment for order flow (PFOF)**: an arrangement where a broker (the app you trade through) sends your orders to a particular market maker, and that market maker *pays the broker* a small amount for the privilege of filling them. This is how "commission-free" trading is funded, and we will dissect both why it is legal and why it is controversial. The second is the **National Best Bid and Offer (NBBO)**: a consolidated, regulated benchmark stitched together from all US stock exchanges, representing the single best bid and best ask available anywhere in the country at that instant. US rules require that your order be filled at a price at least as good as the NBBO. The NBBO is the yardstick against which every fill is measured, so hold onto it.

And finally, the word that makes headlines: a **flash crash** is a sudden, severe, and very brief collapse in prices, driven by automated trading and the rapid withdrawal of liquidity, that reverses almost as fast as it appeared. It is the dark side of a market run by machines, and we will spend real time on the most famous one.

So here is the foundational picture in one breath. There are always two prices, a bid and an ask. A market maker posts both, profits from the gap, and in doing so lets you trade instantly. It does this millions of times a day at machine speed, often paying your broker to send it your orders, always filling you at or better than a national benchmark, and occasionally, on a very bad day, stepping back all at once. Now let us slow each piece down.

## How a market maker actually earns the spread

Let us make the core machine concrete, because the arithmetic is the whole point.

Imagine a market maker has decided that a stock is worth, in its best estimate, about \$50.00 a share. It does not post a single price. It posts a *bid* a hair below and an *ask* a hair above, and it stands ready to trade on both sides.

#### Worked example: quoting a two-cent spread on 10,000 shares

Suppose our market maker quotes a bid of \$49.99 and an ask of \$50.01 on a stock, and over a short window it trades 10,000 shares on each side. Walk it through, one leg at a time.

```
Buy  10,000 shares from sellers at the bid:  10,000 x $49.99 = $499,900 paid out
Sell 10,000 shares to buyers  at the ask:  10,000 x $50.01 = $500,100 taken in
Gross spread capture:                        $500,100 - $499,900 = $200
```

The firm bought and sold the same 10,000 shares, ending the window holding roughly zero net position, and walked away with \$200. Per share, that is exactly the two-cent spread: \$0.02 x 10,000 = \$200. Notice what it did *not* do: it did not predict the price. The stock could have been going up, down, or sideways; as long as the firm bought at its bid and sold at its ask in roughly equal amounts, the \$200 is the same. The intuition: a market maker is not paid to be right about direction, it is paid \$0.02 a share to be the one standing in the middle, willing to trade either way, the instant you arrive.

Now, that \$200 sounds trivial, and on its own it is. The business only makes sense at scale. So let us scale it.

#### Worked example: a penny spread times a day's volume

Drop the spread to a single penny, \$0.01 per share, which is common for a heavily traded stock. Suppose a large market maker handles, across thousands of stocks, an average of 1.5 billion shares of customer trades a day (a realistic order of magnitude for a top wholesaler). And suppose it captures, on average, only *half* a penny per share net after costs, hedging, and the occasional loss, because real spreads are often shared, and not every trade is a clean round trip.

```
Net capture per share:        about $0.005
Shares handled per day:       1,500,000,000
Daily gross from spread:      1,500,000,000 x $0.005 = $7,500,000
Trading days per year:        about 250
Annual gross from spread:     $7,500,000 x 250 = $1,875,000,000
```

That is roughly \$1.9 billion a year from capturing half a penny at a time. The penny is real; the scale is what turns it into a fortune. This is why the biggest market makers are among the most profitable trading firms in the world while being almost unknown to the public: their edge per trade is microscopic, and their volume is astronomical. The intuition: never reason about a market maker from a single trade. The entire model is "a tiny edge, repeated until the law of large numbers makes it a river."

But there is a catch hiding in those examples, and it is the thing market makers actually spend their effort on. In the clean examples, the firm bought and sold equal amounts. In reality, the flow is lumpy. Sometimes far more people sell to the firm than buy from it, and the firm is suddenly *long* a big pile of shares it did not want, exposed to the price falling before it can sell them off. This is **inventory risk**: the danger that the position you are forced to hold, as the willing counterparty, moves against you before you can unwind it.

Managing inventory risk is the real craft. The market maker constantly adjusts its quotes to nudge its inventory back toward neutral. If it has accumulated too many shares (too long), it will quietly lower both its bid and its ask a touch, making it cheaper for others to buy from it (clearing its excess) and less attractive to sell to it. If it is short too many shares, it does the reverse. It also **hedges**: if it gets stuck long 100,000 shares of a stock, it might immediately sell a related futures contract or an ETF that contains the stock, so that a price drop hurts the inventory but helps the hedge, leaving the firm roughly flat. The spread, then, is not pure profit; it is partly compensation for bearing inventory risk between the moment it trades with you and the moment it gets back to neutral. When markets are calm, that risk is small and spreads are tight. When markets are violent, that risk explodes, and spreads widen as the firm demands more to take the other side, which is exactly the behavior we will see go to its extreme in the Flash Crash.

## The big firms: who is actually on the other side

You have never gotten a marketing email from these firms, and you never will. They do not want your retail account directly; they want your *order flow*, in bulk, routed to them by your broker. But they are enormous, and a few names dominate. The matrix below lays out who they are and what they do, because they are not interchangeable.

![Matrix comparing Citadel Securities, Jane Street, Virtu, and Jump by role and markets](/imgs/blogs/market-makers-and-high-frequency-trading-2.png)

**Citadel Securities** is the giant of retail. It is a separate company from the Citadel hedge fund (same founder, Ken Griffin, different business), and it is the largest **wholesaler** of US retail stock trades, meaning it is the market maker to which a huge share of brokers like Robinhood route their customers' orders. By various estimates it has handled on the order of a quarter or more of all US listed equity volume and an even larger slice of retail volume. When an American taps "Buy" in a trading app, the odds are very good that Citadel Securities is the invisible counterparty. It also makes markets in options, Treasuries, and other instruments.

**Jane Street** is best known as the dominant force in **ETF** market making and a heavy player in bonds and options. ETFs (which we will explain in detail later) require a specialist who can price a basket of underlying assets in real time and trade the fund against its contents; Jane Street built a reputation for exactly this quantitative pricing skill and has expanded into many markets, including crypto. It is famously private, partnership-run, and culturally distinct (it is known in tech circles for using the OCaml programming language).

**Virtu Financial** is notable for being *public*: it listed on the stock market in 2015, which forced it to disclose its results, and the disclosures were eye-opening. In its IPO filing, Virtu revealed that over a multi-year stretch it had exactly one losing trading day, a single day of red across more than a thousand. That statistic, more than any lecture, told the public what high-frequency market making is: not a series of bold bets, but a vast number of tiny, hedged, near-certain edges. Virtu makes markets across stocks, currencies, and more, around the world.

**Jump Trading** and **Hudson River Trading (HRT)** are quieter, more pure-play HFT firms, heavy in futures, options, and the latency-sensitive corners of the business. Jump in particular became known for the *speed* arms race, including investment in microwave-tower networks (more on those later) to move data faster than fiber optic cable can. HRT is a major liquidity provider across global equities. These firms blur the line between "market maker" and "high-frequency trader," because in practice the two overlap heavily: most HFT profit, industry-wide, comes from electronic market making, not from exotic speculation.

What unites all of them is the model from the last section: take the other side, capture a sliver, manage inventory, repeat at unfathomable scale. What separates them is *which* markets they specialize in, whether they court retail flow or compete on raw speed, and whether they are private partnerships or public companies. None of them is a household name, and all of them are woven into the price of nearly every trade you will ever make.

## High-frequency trading: the strategy families

"High-frequency trading" is a method (trade fast, automatically, in volume), not a single strategy. Under that umbrella sit several distinct ways to make money, and they differ sharply in how useful and how controversial they are. The way this works is best seen as a stack, with the calm, liquidity-providing strategies at the base and the speculative, speed-dependent races at the top.

![Stack of HFT strategy families from electronic market making up to latency arbitrage](/imgs/blogs/market-makers-and-high-frequency-trading-5.png)

At the base is **electronic market making**, which is everything we have already described, done by computer at high frequency. The firm posts bids and asks across thousands of securities, captures spreads, and manages inventory automatically. This is the largest and least controversial category. It provides genuine liquidity: the firm is *adding* quotes to the market that you can trade against. The vast majority of HFT activity, and profit, lives here.

Next is **arbitrage**, the practice of profiting from a price discrepancy between two things that should have the same price. The cleanest example is when the same stock trades on two different exchanges and one is momentarily a penny cheaper: a firm buys on the cheap venue and sells on the dear one, pocketing the difference and, in the act, pulling the two prices back together. **Statistical arbitrage** and **index arbitrage** are more elaborate versions, exploiting predictable relationships between related instruments, like an index futures contract and the basket of stocks it tracks. Arbitrage is mostly benign and even helpful: it enforces the law of one price, keeping related markets consistent.

A specialized and important branch is **ETF arbitrage**, which keeps an ETF's market price glued to the value of the assets it holds. We will give this its own section because it is genuinely clever and it is where market makers do some of their most socially valuable work.

At the top of the stack sits the genuinely contentious category: **latency arbitrage**, the pure speed race. Here the edge is not providing liquidity or enforcing consistency; it is simply being *first*. When a price moves on one venue, the new information takes a few microseconds to propagate everywhere. A firm fast enough can see the move first and trade against quotes that have not yet updated, the "stale" quotes, before slower participants can pull them. This is the strategy critics have in mind when they call HFT predatory: the firm is not adding liquidity so much as racing to pick off others' quotes a few microseconds before they can react. It is real, it is legal, and it is the reason firms spend fortunes shaving microseconds, but it is a small slice of total HFT activity, not the bulk of it.

Keeping this stack straight is the key to having an honest opinion about HFT. Most of it is the dull, useful work of providing liquidity and enforcing consistency. A thin, fast, expensive top layer is a zero-sum race to be first. When someone says "HFT is parasitic" or "HFT is essential," they are usually describing different layers of this same stack and talking past each other.

#### Worked example: a latency-arbitrage edge of a fraction of a cent

Let us put numbers on why anyone would spend tens of millions of dollars to be microseconds faster. Suppose a fast firm can, on average, capture an edge of just \$0.001 (one-tenth of a cent) per share on a latency-sensitive strategy, by being first to react when a price moves. And suppose this strategy trades 200 million shares on a busy day.

```
Edge per share:          $0.001
Shares per day:          200,000,000
Daily gross:             200,000,000 x $0.001 = $200,000
Trading days per year:   about 250
Annual gross:            $200,000 x 250 = $50,000,000
```

A tenth of a cent, which is far too small to matter to you on a single trade, becomes \$50 million a year at this volume. Now you understand the arms race. If being a few microseconds faster reliably lets a firm grab that tenth of a cent ahead of rivals, then spending, say, \$30 million on faster networks and colocation to defend or win that \$50 million stream is a perfectly rational investment. The intuition: in a speed race, the prize is not a big edge per trade, it is the *exclusivity* of a microscopic edge across colossal volume, and that exclusivity is worth paying enormous fixed costs to secure.

## Payment for order flow: the conflict you fund with free trading

This is the most important section for an ordinary investor, because it is the part that directly affects what you pay. Recall PFOF: your broker routes your orders to a particular market maker, and the market maker pays your broker for them. The graph below traces where the money actually moves, and it is the key to seeing both why your trading is "free" and why regulators worry.

![Graph of the payment-for-order-flow money path among you, broker, market maker, and exchange](/imgs/blogs/market-makers-and-high-frequency-trading-4.png)

First, why would a market maker *pay* for your orders? Because retail orders are unusually profitable to fill. The reasoning is a bit subtle, so follow it carefully. When a market maker trades against a big sophisticated institution, it worries about **adverse selection**: the fear that the institution knows something it does not, so that whenever the institution buys, the price is about to rise (and the market maker, having sold, just lost). Retail orders, by contrast, are mostly *uninformed*: a retiree rebalancing, a hobbyist buying a few shares, a worker dollar-cost-averaging. These orders do not, on average, predict the next price move. They are the safest, most pleasant flow to take the other side of. So market makers compete for it, and one way to compete is to pay brokers to send it their way.

What does the broker do with that payment? In the modern retail model, it uses it to offer **zero-commission trading**. Before PFOF-funded free trading became standard around 2019, brokers charged you maybe \$5 to \$10 per trade. Now they charge nothing, and the market maker's PFOF payment is a major reason they can. So the bargain, on its face, is: you trade for free, the broker gets paid by the market maker, and the market maker gets safe, profitable order flow. Everybody seems to win.

Here is the conflict, and it is genuine. Your broker has a duty to get you the **best execution**, the best available price for your trade. But the broker is also being paid by the market maker, and it may be tempted to route your order to whichever market maker pays the broker the most, rather than whichever would fill your order at the best price. Those two goals can diverge. The market maker, meanwhile, sets the price you get; it is required to fill you at or better than the NBBO, but "at or a hair better than the minimum allowed" still leaves it room to keep a margin. The fear is that the spread you implicitly pay, plus the PFOF the broker collects, together exceed what you would have paid in a more transparent, competitive market. You are not billed a commission, but you may pay in slightly worse prices, a cost you never see on a statement.

#### Worked example: the economics of PFOF on a single order

Let us trace one realistic order through the whole chain. You buy 1,000 shares of a stock whose NBBO is \$49.99 bid, \$50.01 ask, so the midpoint (the fair price halfway between) is \$50.00.

```
You pay the ask:                 1,000 x $50.01 = $50,010
Fair midpoint value:             1,000 x $50.00 = $50,000
Your implicit cost vs midpoint:  $50,010 - $50,000 = $10
```

Now suppose the market maker gives you a little **price improvement**, filling you at \$50.005 instead of the full \$50.01 ask:

```
You actually pay:                1,000 x $50.005 = $50,005
Your improved cost vs midpoint:  $50,005 - $50,000 = $5
```

You saved \$5 versus the worst legal price, which the broker will advertise as price improvement. Meanwhile the market maker captures the half-cent spread on its side, roughly \$5 on this trade, plus whatever it makes managing the position, and out of its profit it pays your broker a PFOF rebate, perhaps \$0.0015 per share:

```
PFOF to broker:   1,000 x $0.0015 = $1.50
```

So on this single trade: you paid \$5 of implicit spread cost (and would have paid \$10 without improvement), the market maker earned roughly \$5 gross, and your broker earned \$1.50 for routing you. You paid no commission. The intuition: PFOF does not make trading free, it makes the *cost invisible*. The cost moved from a labeled commission line into the spread, where you cannot see it, and a slice of it now pays your broker to send you to the market maker in the first place.

Is that a bad deal? Honestly, for a small retail investor, often not. The all-in implicit cost on a liquid stock today is frequently *lower* than the old explicit commissions, because spreads have collapsed (next section) and price improvement is real. The strongest defense of PFOF is simply that retail trading has never been cheaper. The strongest criticism is that the arrangement is opaque, creates a routing conflict, and may not deliver the *best possible* price even if it delivers a good one, and that a more transparent design (or competition forcing midpoint fills) could hand more of that spread back to you. Both can be true at once, which is exactly why the debate never quite ends.

## How spreads collapsed: decimalization and the penny tick

To appreciate how good a deal trading has become, you have to know how bad it used to be. For most of the twentieth century, US stocks were not quoted in cents. They were quoted in *fractions*, originally in eighths of a dollar, a convention with roots reaching back to Spanish colonial coins that were physically cut into eight "pieces of eight." A stock might be quoted at "50 and 1/8" bid, "50 and 1/4" ask. The before-and-after below shows what that meant for the spread you paid.

![Before and after comparison of wide fractional spreads versus penny decimal spreads](/imgs/blogs/market-makers-and-high-frequency-trading-3.png)

The smallest possible price increment, the **tick size**, was one-eighth of a dollar, which is \$0.125. That meant the *narrowest possible spread* on a stock was often \$0.125 or \$0.0625 (a sixteenth, after a later reform), a vast canyon compared to today. The market maker collecting that spread was collecting twelve and a half cents a share where it now collects one. For the trader, every round trip started a dime or more in the hole.

Then came **decimalization**. In 2001, US stock markets switched from fractions to decimals, with a minimum tick of one cent. Suddenly a stock could be quoted \$49.99 bid, \$50.00 ask, a one-cent spread, where before the law of the tick had forced a twelve-cent gap. Competition among market makers, now able to undercut each other a penny at a time, drove spreads down toward that one-cent floor on liquid names. This was, quietly, one of the largest transfers of value from the trading industry to ordinary investors in market history.

#### Worked example: what decimalization saved you on a round trip

Suppose you buy and then later sell 1,000 shares of a \$50 stock, a complete round trip. Compare the spread cost in the old eighths world versus today's penny world. In the fractional era, assume a typical spread of \$0.125:

```
Spread cost per share:    $0.125
Round trip (cross spread once each way is built into buy-high, sell-low):
Cost on 1,000 shares:     1,000 x $0.125 = $125
```

In today's decimal world, assume a one-cent spread:

```
Spread cost per share:    $0.01
Cost on 1,000 shares:     1,000 x $0.01 = $10
```

You went from roughly \$125 of spread cost to about \$10 on the same trade, a saving of around \$115, before even counting the disappearance of fixed commissions. The intuition: decimalization did not just make spreads "a bit tighter." It cut the structural cost of trading liquid stocks by an order of magnitude, and the firms that survived did so by making it up on a tidal wave of automated volume, which is precisely what pushed the industry toward HFT.

There is a subtlety worth naming so you are not left with a fairy tale. Tighter spreads were a clear win for small investors. But the penny tick can also be *too* coarse for very cheap, very liquid stocks (where a penny is a large percentage) and *too* fine in ways that push competition into the speed dimension, because if you cannot undercut a rival on price below one cent, you compete on being faster to the front of the queue at that price. So decimalization both saved investors money and helped *cause* the latency arms race. Cheaper trading and faster machines are two faces of the same reform.

There is one more piece of plumbing from this era that matters, and it is the rulebook that ties all the venues together. In 2007 the SEC implemented **Regulation NMS** (National Market System), and its **Order Protection Rule** is the legal teeth behind the NBBO: it forbids an exchange from executing a trade at a price worse than the best price displayed on any other exchange. In plain terms, it says your order cannot be filled at a worse price somewhere just because that venue happened to receive it; the whole patchwork of competing exchanges is stitched into one logical market where the best displayed price wins. This is wonderful for fairness, but notice the side effect: to enforce a single best price across a dozen physically separate computers, the system has to *consolidate* quotes from everywhere, and the few microseconds it takes to do that consolidation created a brand-new, exploitable gap, the very stale-quote window that latency arbitrage races to capture. The rule that guaranteed you a fair price also defined, with microsecond precision, the prize the speed merchants compete for. It is a recurring theme: nearly every reform that made markets cheaper or fairer also reshaped the game the fastest firms play.

It helps to see the whole arc on one line, because cheaper trading and faster machines arrived together and each milestone reopened the same fairness argument. The way this works is a steady drumbeat: a structural reform lowers costs or links venues, the speed-and-automation response that follows delivers a benefit *and* a new hazard, and a crisis or a scandal then forces the next round of rules. The timeline below lays out that rhythm.

![Timeline from decimalization through the 2010 flash crash to the payment-for-order-flow debate](/imgs/blogs/market-makers-and-high-frequency-trading-6.png)

Read left to right, the story is not "markets got worse" or "markets got better." It is that each step traded one set of problems for another. Decimalization (2001) crushed spreads but thinned per-trade margins to the point that only automated, high-volume firms could survive. Reg NMS (2007) linked every venue into one fair-price system but minted the microsecond stale-quote window. The Flash Crash (2010) exposed how conditional machine liquidity really is and produced circuit breakers. The Knight Capital glitch (2012) showed automation's blast radius and forced kill switches. And the GameStop episode (2021) dragged the economics of payment for order flow, the quiet arrangement funding all that "free" trading, into a national argument. Every entry on that line is a moment when the public briefly noticed the invisible counterparty, demanded a fix, got one, and then forgot about it again, until the next time.

## Keeping ETFs honest: the arbitrage that holds funds at fair value

This is where market makers do some of their most valuable and least appreciated work, so it deserves a careful build from zero.

An **ETF** (exchange-traded fund) is a fund that holds a basket of assets, say all 500 stocks in the S&P 500, but whose own shares trade on an exchange all day like a single stock. So there are two prices floating around for an S&P 500 ETF at any moment. There is the ETF's **market price**, what its shares are changing hands for on the exchange right now. And there is its **net asset value (NAV)**, the true per-share value of all the stocks the fund actually holds, computed from those stocks' live prices. In a perfect world these two numbers are identical: a fund holding \$50.00 worth of stock per share should trade at \$50.00. But supply and demand for the ETF's *own shares* can briefly push its market price above or below the value of its contents. If everyone rushes to buy the ETF, its price might tick up to \$50.05 while the underlying stocks are still worth \$50.00. The ETF is now trading at a **premium** to NAV. The reverse, trading below NAV, is a **discount**.

Here is the clever plumbing that fixes this, performed by specialized market makers called **authorized participants (APs)**. An AP has a special right: it can hand the fund a basket of the underlying stocks and receive newly *created* ETF shares in return, or hand the fund ETF shares and receive the underlying stocks back, a process called **creation and redemption**. This gives the AP a risk-free arbitrage whenever the ETF strays from NAV. Conceptually, the AP is a referee with the power to mint or destroy ETF shares to force the price back to fair value, and the figure for the strategy stack above places this ETF arbitrage in the helpful middle of the HFT taxonomy.

#### Worked example: closing a five-cent premium to NAV

Suppose an S&P 500 ETF is trading at a market price of \$50.05, but the basket of stocks it holds is worth exactly \$50.00 per share. The ETF is at a five-cent *premium*. An authorized participant pounces.

```
Step 1: Buy the underlying basket of stocks at NAV value:  $50.00 per ETF-equivalent
Step 2: Deliver that basket to the fund, receive new ETF shares (creation)
Step 3: Sell the freshly created ETF shares on the exchange at the market price: $50.05
Arbitrage profit per share:  $50.05 - $50.00 = $0.05
```

The AP just earned five cents a share at essentially no risk, because it locked in both prices at once. But look at what its action *did to the market*. By buying the underlying stocks (step 1), it nudged their prices up, raising NAV. By creating and then selling new ETF shares (steps 2 and 3), it increased the supply of ETF shares on the exchange, pushing the ETF's market price down. The premium shrinks from both ends. APs will keep doing this, dozens of firms competing, until the gap closes and the ETF trades right at NAV again. The intuition: ETF arbitrage is a self-correcting machine. The very act of profiting from a price gap is the act of closing it, so the more profitable the misalignment, the faster it is fixed and the tighter the ETF tracks the fair value of what it holds.

This is not a theoretical nicety. It is why ETFs are trustworthy enough to hold trillions of dollars. And it faced its sternest test in March 2020 (covered in the real-markets section below), when bond markets seized up and some bond ETFs briefly traded at large discounts to their stated NAVs. The story there is subtle and reassuring: in that case the ETF price was arguably the *more* accurate one, because the underlying bonds had simply stopped trading and their official prices were stale. The arbitrage machine strained but held, and the episode became a case study in how ETF market making behaves under maximum stress.

## When the machines step back: the May 6, 2010 Flash Crash

Now the dark side. Everything good about market makers, that they are always there, ready, fast, depends on them *choosing* to be there. They are private firms with no obligation to lose money for the public good. When risk spikes beyond what they will bear, they can widen their quotes to absurd levels or pull them entirely, and when many do so at once, the liquidity they manufacture can evaporate in seconds. May 6, 2010 is the canonical demonstration. The graph below traces the cascade.

![Graph of the 2010 Flash Crash cascade from a large sell algorithm to a price collapse and rebound](/imgs/blogs/market-makers-and-high-frequency-trading-7.png)

That afternoon, against a jittery backdrop (the European debt crisis had markets on edge), a large mutual-fund firm launched an automated program to sell a huge quantity of E-mini S&P 500 futures contracts, roughly \$4.1 billion worth, using an algorithm set to sell at a pace tied to recent trading *volume* but without regard to *price* or *time*. As the selling hit, high-frequency market makers initially did what they do: they absorbed the contracts, taking the other side. But they did not want to hold all that risk, so they quickly resold the contracts to each other and to other buyers. Because the original sell algorithm keyed off volume, this frantic reselling among HFTs, sometimes called the **hot-potato** effect, actually *increased* measured volume, which made the sell algorithm speed *up*. The faster the machines passed the hot potato, the faster the original program dumped more contracts. A feedback loop had formed.

As prices fell and volatility exploded, the market makers' inventory risk became unbearable, and they did the rational thing for a private firm: they widened their quotes dramatically or pulled them entirely. Liquidity vanished. With no real bids left, some market orders to sell crashed into the only quotes still standing: **stub quotes**, nonsensical placeholder bids like \$0.01 that firms post merely to technically remain in the market. Shares of solid companies briefly traded for a penny; a few traded for absurd highs like \$100,000. Within about five minutes, major indexes had fallen roughly 5 to 6 percent on top of an already-down day, with the Dow Jones Industrial Average plunging nearly 1,000 points intraday, erasing on the order of a trillion dollars of value. And then, within roughly 36 minutes total, it almost entirely came back. Once the sell program finished and circuit-breaker-style pauses kicked in, buyers returned, market makers re-entered, and prices rebounded.

#### Worked example: how a penny stub quote produces a 99.98% "loss"

Consider what a stub quote does to an unlucky seller. Suppose you held 100 shares of a healthy \$50 stock and, panicking as the screen went red, you sent a *market order* to sell, "fill me at the best available price, whatever it is." In a normal moment the best bid is \$49.99 and you collect about \$4,999. But in the depths of the crash, every real bid had been pulled, and the only standing bid was a \$0.01 stub quote.

```
Normal fill:  100 x $49.99 = $4,999
Crash fill:   100 x $0.01  = $1
Value destroyed on that order:  $4,999 - $1 = $4,998  (about 99.98%)
```

Your healthy \$5,000 position sold for one dollar, not because the company changed, but because for a few seconds there was *no one willing to buy* except a placeholder bid nobody expected to hit. The intuition: a market order is a promise to trade at *any* price, and it is only safe because liquidity is normally deep; when the market makers step back, that promise becomes a trapdoor. The lasting fix was structural, real **circuit breakers** that pause trading in a stock when it moves too far too fast, and a ban on the worst stub quotes, so that the next time the machines hesitate, the market is forced to take a breath rather than free-fall into a penny.

The Flash Crash is the single most important cautionary tale about this whole system. It does not prove market makers are evil; they behaved rationally given their incentives. It proves something subtler and more important: liquidity provided by profit-seeking private firms is *conditional*. It is abundant precisely when you least need it and can thin out precisely when you most need it. A market that feels infinitely deep on a calm Tuesday can have a paper-thin floor on a bad Thursday, and the speed of modern trading means the floor can give way faster than any human can react.

## Common misconceptions

**"I'm trading with another regular investor like me."** Almost never, at least not directly. When you send a market order through a retail app, a market maker is overwhelmingly likely to be your direct counterparty, filling you from its own inventory and then hedging or unwinding elsewhere. The "other investor" who ultimately balances your trade may be reached only after several intermediated steps, milliseconds later. The retail-to-retail trade is the exception, not the rule, and understanding this is the foundation for understanding everything else, including why your fills are so fast.

**"High-frequency trading is just legalized cheating."** This conflates the whole field with its most aggressive sliver. The bulk of HFT, by volume and by profit, is electronic market making and arbitrage, which add liquidity and enforce price consistency, lowering costs for everyone. The genuinely zero-sum, predatory part, pure latency arbitrage racing to pick off stale quotes, is real and worth criticizing, but it is a thin top layer, not the whole stack. Lumping the useful base in with the contentious peak makes for a satisfying villain and a wrong model. The honest critique targets specific practices, not the existence of fast, automated market makers.

**"Payment for order flow is theft."** PFOF creates a real conflict of interest and deserves scrutiny, but "theft" overstates it. For most small retail trades on liquid stocks, the all-in cost today (a hair of spread, often with price improvement, and no commission) is lower than the explicit commissions investors paid before PFOF-funded free trading. The legitimate worry is opacity and the routing conflict: you may be getting a *good* price rather than the *best possible* price, and you cannot see the difference on your statement. That is a reason to demand transparency and competition, not a reason to believe you are being robbed on every trade.

**"Market makers make money by predicting where prices go."** No. A market maker's core profit comes from capturing the spread while staying close to flat, not from directional bets. It is paid for *immediacy and inventory risk*, not for forecasting. This is why a firm like Virtu could have a years-long streak with essentially one losing day: you cannot achieve that by betting on direction, you can only achieve it by collecting a tiny, hedged edge on enormous volume. Confusing market making with speculation leads people to cast these firms as gamblers when they are closer to a high-speed, high-volume insurance business.

**"Tighter spreads mean markets are perfectly fair now."** Tighter spreads are a genuine win, but they pushed competition into a dimension you cannot see on a quote screen: *speed*. When firms cannot undercut each other below a one-cent tick, they compete to be first in the queue at that price, which is what funds the microwave-tower arms race. So markets got cheaper *and* developed a new, expensive, exclusivity-based competition that a small investor cannot participate in. Cheaper is not the same as flat, and decimalization both lowered costs and helped create the latency races.

**"If market makers vanished, the market would be more fair."** It would be far *worse*, especially for small investors. Without firms standing ready on both sides, spreads would widen dramatically, fills would slow, and trading in all but the largest stocks would become expensive and uncertain, exactly the illiquid world that existed before electronic market making matured. The Flash Crash is a glimpse of that world: it is what happens for a few minutes when the market makers step back. The goal is not to abolish them but to align their incentives and harden the system for the moments they choose not to play.

## How it shows up in real markets

**The May 6, 2010 Flash Crash.** The defining event, covered in detail above. Its lasting legacy is structural: market-wide and single-stock **circuit breakers** that pause trading after sharp moves, the banning of abusive stub quotes, and a years-long regulatory and academic effort to understand how automated selling, hot-potato HFT volume, and the rational withdrawal of liquidity combined into a trillion-dollar air pocket that healed in under an hour. It permanently changed how regulators think about the *conditional* nature of machine-provided liquidity. For a fuller picture of how shocks ricochet through the broader financial system, see the [field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions).

**GameStop and the PFOF spotlight, 2021.** When a crowd of retail traders sent GameStop's stock soaring in January 2021, and Robinhood abruptly restricted buying at the peak, public fury landed squarely on payment for order flow. Conspiracy theories claimed Robinhood had halted buying to protect Citadel Securities (which both makes markets and is affiliated with a hedge fund that had backed a short seller). The real reason for the trading halt was prosaic, a clearinghouse collateral demand, not a favor to a market maker, but the episode dragged PFOF into the daylight and triggered congressional hearings and SEC scrutiny of whether retail investors truly get best execution. The full mechanics of that squeeze and the collateral call are dissected in [GameStop, 2021: the short squeeze](/blog/trading/finance/gamestop-2021-short-squeeze).

**The Knight Capital glitch, August 1, 2012.** Knight Capital was one of the largest US market makers when, on the morning of August 1, 2012, it deployed new trading software with a fatal bug: an old, dormant piece of code was accidentally reactivated, and the firm's systems began flooding the market with millions of unintended orders, buying high and selling low across roughly 150 stocks. In about 45 minutes, before anyone could fully halt it, Knight had accumulated a wild portfolio and a loss of roughly \$440 million, more than the firm could absorb. It was driven to the brink of collapse over a single weekend and was rescued only by an emergency investment that effectively ended its independence. The lesson is sobering: the same automation that makes market making cheap and fast also means a software error can incinerate a firm's capital faster than humans can intervene. Speed cuts both ways.

#### Worked example: how 45 minutes destroyed \$440 million

Knight's loss is hard to fathom until you do the arithmetic of automated speed. Suppose the runaway code sent orders that, on net, lost an average of just \$0.15 per share by buying above and selling below fair value, and suppose the malfunction churned through roughly 2.9 billion shares of erroneous trades over the window.

```
Average loss per share:   $0.15
Erroneous shares traded:  about 2,900,000,000
Total loss:               2,900,000,000 x $0.15 = $435,000,000  (about $440M)
```

No human placed those orders; a machine did, thousands of times a second, for 45 minutes. The intuition: in a market that trades at machine speed, the *blast radius* of a bug scales with that same speed. A human market maker making a fat-finger error loses a few thousand dollars before noticing; an unsupervised algorithm can lose a few hundred million before anyone reaches the off switch, which is why "kill switches" and pre-trade risk checks became mandatory after Knight.

**Decimalization, 2001.** The quietest revolution on this list. The switch from fractional to decimal pricing, with a one-cent minimum tick, collapsed spreads on liquid stocks from an eighth of a dollar toward a single penny, slashing the structural cost of trading by roughly an order of magnitude. It transferred enormous value from the trading industry to ordinary investors, killed off the old, fat-margin specialist business model, and forced market makers to survive on razor-thin per-trade economics, which is precisely what drove the industry toward automation, scale, and HFT. Almost no one outside the industry noticed, yet it reshaped market structure more than any single firm ever has.

**The ETF arbitrage stress test, March 2020.** When the pandemic shock hit in March 2020, credit and bond markets seized: many corporate and municipal bonds simply stopped trading. Bond ETFs, which trade continuously on exchanges, kept printing live prices, and some fell to large *discounts* to their stated NAVs, in a few cases several percent. Critics screamed that ETFs were "broken." The deeper truth was the opposite: the ETFs' live market prices, set by market makers willing to actually transact, were arguably *more* accurate than the stale official prices of the underlying bonds, which had not traded in hours or days. The creation-redemption arbitrage machine strained, with wider spreads reflecting genuine risk, but it functioned, and once central banks backstopped credit markets the discounts closed. The episode became the canonical evidence that ETF market making is robust under maximum stress, an important counterweight to the Flash Crash's warning.

**The microwave-tower arms race, 2010s onward.** To win latency races, firms discovered that light travels faster through air than through glass fiber, so a microwave signal beamed line-of-sight between towers can beat fiber-optic cable over the same route. A network of microwave relays sprang up between the data centers of Chicago (home of futures markets) and New Jersey (home of the stock exchanges), shaving the round trip to a few milliseconds, and firms paid fortunes for rights to the fastest paths, even bidding on rooftop and tower placements measured in meters. It is the most literal expression of the speed arms race: physical infrastructure built solely to move price data a few microseconds faster than a rival, in pursuit of edges measured in tenths of a cent. To see how the trading desks of large banks intersect with this ecosystem of market makers and electronic venues, see [inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money).

## When this matters to you, and further reading

For the vast majority of investors, the honest takeaway is reassuring with an asterisk. The reassuring part: trading has never been cheaper or smoother, and you have market makers and the competition among them to thank for spreads of a penny and fills in a blink. If you are a long-term investor buying broad index funds and holding for years, the machinery in this post works overwhelmingly in your favor, and the few fractions of a cent of implicit cost on each trade are noise against decades of compounding.

The asterisk matters in specific situations. First, **avoid market orders in fast or thin markets.** A market order is a promise to trade at *any* price, and as the Flash Crash example showed, that promise is only safe when liquidity is deep. For anything illiquid, or during a violent open or a news shock, use a **limit order** so you control the worst price you will accept. Second, **understand that "free" trading is not free.** If you are an active trader doing many trades, the invisible spread cost and the routing incentives behind PFOF are worth caring about; it can be rational to prefer a broker that routes for price improvement or offers transparent execution statistics over one that simply maximizes its own PFOF. Third, **respect the conditional nature of liquidity.** The depth you see on a calm day is not a guarantee; in a crisis it can thin out exactly when you most want to trade, so size your positions and your need for instant exits with that in mind.

And the broadest lesson is about how to *think* about these firms. They are neither the villains of populist anger nor the selfless utilities of industry PR. They are private businesses being paid to perform a real and valuable service, being the counterparty nobody else wants to be at a given instant, and the policy questions, about PFOF transparency, about latency races, about circuit breakers, are all really one question: have we priced and governed that service fairly? Holding that frame lets you read the next flash crash, the next PFOF hearing, or the next "HFT is destroying markets" headline without being swept up by either the outrage or the apologetics.

If you want to go further, three companion pieces build out the surrounding system. To see how the trading desks of large banks make markets in bonds and derivatives alongside these specialist firms, read [inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money). To understand the dramatic 2021 episode that dragged payment for order flow into a national debate, read [GameStop, 2021: the short squeeze](/blog/trading/finance/gamestop-2021-short-squeeze). And for the map of the entire ecosystem these firms inhabit, from exchanges and clearinghouses to asset managers and regulators, start with the [field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions). Together they turn the invisible counterparty from a faceless machine into a system you can reason about, and arguing honestly about that system, neither dazzled nor frightened, is the whole point.
