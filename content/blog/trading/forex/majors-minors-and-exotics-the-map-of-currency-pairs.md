---
title: "Majors, Minors, and Exotics: The Map of Currency Pairs"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A beginner's full map of what you can trade in FX — the four pair tiers, the spread gradient from majors to exotics, the 24-hour session clock, and how a currency trade actually settles two days later through CLS."
tags: ["forex", "currencies", "currency-pairs", "majors", "exotics", "liquidity", "fx-spreads", "fx-sessions", "settlement", "cls"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Every tradeable currency pair sits on a single gradient of liquidity, and that gradient sets everything that matters to you: the spread you pay, how violently the price can gap, and even what hours of the day it is safe to trade.
>
> - **Four tiers.** Majors (USD against the big rich-world currencies), minors or crosses (two majors with no dollar), EM pairs (the dollar against an emerging-market currency), and exotics (thin, often restricted currencies). The dollar is on one side of about **88%** of all trades.
> - **The spread is the gradient.** EUR/USD trades at a spread of about **0.2 of a pip**; USD/TRY at about **25 pips**; the non-deliverable USD/VND at about **40 pips**. That is a cost difference of more than **100×** for the same notional.
> - **The day is a relay race.** FX runs 24 hours on a weekday, handed from Sydney to Tokyo to London to New York. The single deepest, cheapest window is the **London–New York overlap, roughly 12:00–16:00 UTC**.
> - **The one number to remember.** A spot FX trade settles **two business days later (T+2)**, and the system that stops one bank paying out before it is paid back is called **CLS** — it settles both legs together or neither at all.

In January 2015 a Swiss franc trade that looked boring for three years became the most violent move in modern FX. The Swiss National Bank had promised, for over three years, to cap the franc at 1.20 to the euro. Traders treated EUR/CHF like a parked car. Then, at 09:30 on 15 January, the central bank walked away from the cap. In **two minutes** EUR/CHF fell from 1.2010 to below 0.8500 — a roughly 30% collapse in the time it takes to read this paragraph. Brokers blew up. Retail traders woke to accounts that owed *more* than they had deposited, because the price had jumped clean over every stop-loss order with no liquidity in between.

The Swiss franc is not an exotic. It is one of the most respected currencies on earth, a classic safe haven, part of the dollar index. Yet on that morning its market behaved like the thinnest, most dangerous corner of the FX world. That is the lesson this whole post is built around: **liquidity is not a permanent property of a currency — it is a property of a pair at a moment, and it is the single most important thing you do not see on the price screen.** The spread, the gap risk, the hours that are safe to trade, the chance you settle and the other side does not — all of it flows from where your pair sits on the liquidity map.

This is the last post in the Foundations track of this series. By the end you will have the full map of what you can trade, why a EUR/USD trade is a different animal from a USD/TRY trade even though both say "buy a million," and what literally happens in the two days after you click buy.

![Tree map of currency pairs sorting majors crosses EM and exotics by liquidity tier](/imgs/blogs/majors-minors-and-exotics-the-map-of-currency-pairs-1.png)

## Foundations: The map of currency pairs

Let us start from the spine of this whole series. **You never own a currency in isolation. Every FX position is a pair — a relative bet of one money against another.** When you "buy euros," you are not buying euros into a vacuum; you are buying euros *with* dollars, or with yen, or with pounds. The thing that has a price is the *ratio*: how many dollars one euro costs. (If the base/quote convention is new, the sibling post [base, quote, pips, and how to read an FX quote](/blog/trading/forex/base-quote-pips-and-how-to-read-an-fx-quote) walks through how to read a quote like EUR/USD = 1.0800.)

Because every trade is a pair, the universe of "things you can trade in FX" is not a list of currencies — it is a list of *pairs*. And the moment you start listing pairs, a natural sorting appears. Some pairs are traded constantly, by everyone, all over the world; others are traded rarely, by a handful of local banks, only during certain hours. That sorting is the map. It has four tiers.

There are, in principle, an enormous number of possible pairs. If you take the dozen or so actively-traded currencies and pair each with every other, you get well over a hundred combinations, and if you reach down into the long tail of EM and frontier currencies the count runs into the thousands. But almost all of that turnover concentrates in a few dozen pairs. The Bank for International Settlements survey finds that EUR/USD, USD/JPY, GBP/USD, and a handful of others account for the lion's share of the roughly **\$7.5 trillion** that trades every single day. The rest of the universe is a thin, scattered fringe. So the map is not a flat directory of equally-weighted options — it is a steep pyramid, with a tiny apex of hyper-liquid majors and a vast, thin base of pairs you can technically trade but rarely should.

Why does turnover concentrate like that? Because liquidity is self-reinforcing. A pair that many people trade attracts market-makers, who quote it tighter, which attracts still more traders, which deepens the book further. A pair almost nobody trades has no market-maker willing to sit there quoting all day, so its spread stays wide, which keeps traders away, which keeps it thin. This feedback loop is why the tiers are so sharply separated rather than smoothly graded: the majors are in a different *regime* of liquidity, not just a different point on a continuum. With that pyramid in mind, here are the four tiers from apex to base.

### Tier 1 — The majors

**A major is a pair of the US dollar against one of the large, freely floating, rich-world currencies.** The standard list is short:

- **EUR/USD** — the euro against the dollar. By far the most traded pair on earth; roughly a quarter of all FX volume is this single pair.
- **USD/JPY** — the dollar against the Japanese yen.
- **GBP/USD** — the dollar against the British pound (traders call it "cable," after the transatlantic telegraph cable that once carried the quote).
- **USD/CHF** — the dollar against the Swiss franc.
- **AUD/USD**, **USD/CAD**, **NZD/USD** — the dollar against the Australian, Canadian, and New Zealand dollars (the "commodity dollars").

What unites them is not geography but two properties: every one of them has a freely floating exchange rate set by the market (no peg, no daily band), and every one has the US dollar on one side. The dollar is what makes them the deepest pairs in the world. As we will see in a moment, the dollar is on roughly 88% of *all* trades — so a pair that includes the dollar automatically taps the largest liquidity pool on the planet.

It is worth being clear that "major" is a convention, not a law — there is no committee that certifies a currency as major. But the list is stable for good structural reasons. To be a major, a currency needs three things at once: a large, open economy that the world wants to do business with; a freely convertible currency a foreigner can actually buy and sell without restriction; and deep, trusted government bond markets where the world parks reserves in that currency. The dollar, euro, yen, pound, and the commodity dollars all clear that bar. The Chinese yuan is the fascinating edge case — China has the economy and the bond market, but the yuan is still only partly convertible (capital controls restrict the free movement of money in and out), which is exactly why it trades through an *offshore* version (CNH) and sits closer to the EM tier than the major tier despite China's size. Convertibility, not GDP, is what admits a currency to the top of the map.

EUR/USD's dominance — roughly a quarter of all FX turnover in one pair — is itself a lesson in self-reinforcing liquidity. The euro and dollar are the two largest currency blocs, so a vast amount of real-economy trade and investment naturally crosses between them; that organic flow attracts the tightest market-making; the tight market-making makes EUR/USD the natural place to *hedge* exposure in any related pair; and that hedging flow deepens it further still. The pair is deep because it is used, and used because it is deep. That loop is the apex of the pyramid in miniature.

### Tier 2 — The minors, also called crosses

**A cross is a pair of two major currencies that does *not* include the US dollar.** EUR/GBP (euro against pound), EUR/JPY (euro against yen), GBP/JPY (pound against yen), AUD/JPY, EUR/CHF — these are crosses. The name "cross" is historical and literal: before electronic trading, if a London bank wanted to swap pounds for yen, there was no direct GBP/JPY market deep enough, so it would *cross* through the dollar — sell pounds for dollars, then sell dollars for yen. The cross rate was computed from the two dollar legs.

Today the big crosses (EUR/JPY, EUR/GBP) trade directly with real depth, but the heritage still shows up in the cost: a cross is, in liquidity terms, **two trades wearing the costume of one**. We will put a number on that shortly. The key intuition: a cross inherits the *combined* friction of its two dollar legs, which is why even a cross between two of the world's deepest currencies is a touch more expensive than either major alone.

### Tier 3 — The emerging-market (EM) pairs

**An EM pair is the dollar against an emerging-market currency** — USD/MXN (Mexican peso), USD/ZAR (South African rand), USD/BRL (Brazilian real), USD/INR (Indian rupee), USD/CNH (offshore Chinese yuan). These currencies float, mostly, but they are smaller, their central banks intervene more often, and a huge share of their daily volume happens during *their own* local trading hours. Liquidity in USD/MXN at 3 a.m. Mexico City time is a fraction of what it is at noon.

The defining trait of EM pairs is that they pay you to hold them — and charge you in fear. EM currencies usually carry much higher interest rates than the dollar (Mexico at 10%+, Brazil at 12%+ in 2024), which is the whole engine of the carry trade covered later in this series. But that yield is compensation for risk: EM pairs gap on political news, on commodity swings, on a single central-bank surprise. The spread is wider, and worse, the price can *jump* — there is not always a buyer at every price on the way down.

### Tier 4 — The exotics

**An exotic is a pair involving a currency that is thin, tightly managed, or outright restricted** — USD/TRY (Turkish lira), USD/VND (Vietnamese dong), USD/NGN (Nigerian naira), USD/ARS (Argentine peso). Some exotics float but are simply small and illiquid. Others, like the dong, are *not freely deliverable* at all: you often cannot legally move the currency across borders, so the only way to take a position is through a **non-deliverable forward (NDF)**, a synthetic contract that settles the profit or loss in dollars without the restricted currency ever changing hands. (The sibling post [non-deliverable forwards: trading uninvestable currencies](/blog/trading/forex/non-deliverable-forwards-trading-uninvestable-currencies) is the deep dive on exactly that mechanism.)

Exotics are where the spread stops being a rounding error and becomes the whole trade. We are about to see that the cost of trading USD/VND can be more than a hundred times the cost of trading EUR/USD, for the same dollar size.

### Which currency goes first: the quoting convention

A quick but load-bearing detail that the map makes sense of: in any pair, *which* currency is named first is not random. There is a rough market hierarchy that decides the **base** currency (the one quoted as "1 unit of"). The euro outranks everything, so it is always first: EUR/USD, EUR/JPY, EUR/GBP. The pound is next, then the Australian and New Zealand dollars, then the US dollar, then everything else. That is why it is GBP/USD (pound first) but USD/JPY (dollar first) — the pound outranks the dollar in the convention, but the dollar outranks the yen.

For EM and exotic pairs the dollar is essentially always the base: USD/MXN, USD/TRY, USD/VND. So the quote tells you "how many pesos / lira / dong per one dollar," and the rate *rising* means the dollar is strengthening and the local currency weakening. This is worth internalising because it flips an intuition: when you read that "USD/TRY hit a new high," that is the *lira* hitting a new low, not the lira doing well. The convention puts the strong, deep currency first and prices the weaker one in terms of it — which is, once again, the map showing through. The deeper, more senior currency is the unit of account; the thinner one is what gets measured.

### What "liquidity" actually means

Throughout this post I keep saying "liquidity." It is worth nailing down, because beginners often hear it as a vague synonym for "popular." It is not. **Liquidity is the ability to trade a large size, quickly, without moving the price much.** It has three concrete components you can almost measure:

1. **Tightness** — how narrow is the gap between the best buy price (the bid) and the best sell price (the offer)? That gap is the *spread*, and it is the toll you pay just to get in and out.
2. **Depth** — how much size sits waiting at the best price, and just behind it? A deep market lets you trade ten million dollars at one price; a thin one fills your first million and then the next million costs more.
3. **Resilience** — after a big trade or a shock knocks the price, how fast does the book refill? A resilient market snaps back; a fragile one stays gapped (recall the franc, which had no book at all for those two minutes).

A pair high on the liquidity map scores well on all three. A pair low on it fails all three at once — wide spread, shallow book, slow to recover. The map of pairs is, at bottom, a map of liquidity, and the next sections turn each of those three components into a chart and a number.

One more nuance worth flagging now: these three components usually move *together*, but in a crisis they can decouple in a way that hurts. A pair can keep a deceptively tight spread on the screen right up until the moment its depth and resilience evaporate — the quote looks normal, but there is almost no size behind it and the book will not refill. That is what makes regime breaks so dangerous: the spread, the most visible component, is the *last* to widen, so the screen tells you everything is fine until it suddenly isn't. A trader who watches only the spread is watching the slowest of the three warning lights.

## The spread gradient: the cost of being in the market

If you remember only one chart from this post, make it this one. The spread is the most direct, most measurable consequence of where a pair sits on the map — and it spans more than two orders of magnitude.

![Horizontal bar chart of typical bid-ask spread in pips by pair tier from EUR USD to USD VND](/imgs/blogs/majors-minors-and-exotics-the-map-of-currency-pairs-2.png)

A **pip** is the standard smallest increment of an FX quote — the fourth decimal place for most pairs (so 0.0001), and the second decimal for yen pairs. When a dealer quotes you EUR/USD as "1.0800 bid, 1.0802 offer," the **spread** is 0.0002, or 2 pips on that retail screen; in the deep interbank market between banks it is more like 0.2 of a pip. That gap is not a fee a broker chooses to charge you out of greed — it is the price of immediacy. The dealer is taking the other side of your trade and bearing the risk of holding it until they can lay it off. In a deep, fast pair they can offload it in seconds, so they quote tight. In a thin pair they might be stuck with it, so they quote wide to protect themselves.

Read the chart top to bottom and you read the entire map of pairs as a cost gradient:

- EUR/USD and USD/JPY sit at **0.2–0.3 pip** — essentially free to trade.
- The cross EUR/GBP sits at **0.8 pip** — wider, because it is two legs.
- USD/MXN, an EM pair, jumps to **8 pips** — an order of magnitude up.
- USD/TRY, an exotic, sits at **25 pips**, and the non-deliverable USD/VND at **40 pips** — another order of magnitude.

That gradient is not linear; it is closer to exponential. Let us turn it into dollars, because "pips" stays a hollow unit until you price a real ticket.

#### Worked example: the round-trip cost of a \$1,000,000 trade, major vs exotic

Take a standard institutional clip: **\$1,000,000 of notional**, traded and then closed out (a "round trip"). The cost of crossing the spread is, to a good approximation, the notional times the spread expressed as a fraction of the price.

For **EUR/USD** at a spread of **0.2 pip**: one pip on EUR/USD is 0.0001 of the price 1.0800, so 0.2 pip is 0.00002. The cost is `\$1,000,000 × 0.00002 = \$20`. (Crossing the full bid-offer is the round-trip cost, so call it on the order of **\$20**.)

Now the same \$1,000,000 in **USD/TRY** at a spread of **25 pips**. For a USD-base pair the pip value works out on the notional similarly: 25 pips is 0.0025 of the price, so the cost is `\$1,000,000 × 0.0025 = \$2,500`.

And the non-deliverable **USD/VND** at **40 pips** of equivalent friction: roughly `\$1,000,000 × 0.0040 = \$4,000`.

So the *identical* dollar trade costs about **\$20 in a major and \$2,500–\$4,000 in an exotic** — a factor of more than **100×**. The intuition: in FX you are not really choosing a currency to "believe in," you are choosing a toll booth, and the exotics charge a toll that can swallow your edge before the trade even has a chance to be right.

#### Worked example: how the spread eats a small expected move

Suppose your research says USD/MXN (spread **8 pips**) will rise about **0.5%** over a week — a real, tradeable view. On a **\$1,000,000** position, 0.5% is `\$5,000` of expected profit. The round-trip spread cost is `\$1,000,000 × 0.0008 = \$800`. So the spread eats `\$800 / \$5,000 = 16%` of your expected gain *before* you account for being wrong.

Run the same view on EUR/USD (spread **0.2 pip = \$20**): the spread eats `\$20 / \$5,000 = 0.4%`. The view is the same size; the *survivability* of the view is completely different. The intuition: an edge that is perfectly fine in a major can be a guaranteed loser in an exotic, purely because of where the pair sits on the spread gradient — the further down the map you go, the larger your expected move must be just to clear the toll.

This is why the spread gradient is not a trivia chart. It silently sets the *minimum viable edge* for every pair. A 5-pip view is a great trade in EUR/USD and a non-trade in USD/TRY.

## The liquidity ladder, as one picture

The four tiers, the spreads, and the dollar costs all line up into a single ladder. Each rung down is roughly an order of magnitude more expensive and an order of magnitude thinner.

![Stacked ladder of four liquidity tiers from majors to exotics with round-trip cost per tier](/imgs/blogs/majors-minors-and-exotics-the-map-of-currency-pairs-3.png)

Notice what changes as you descend the ladder. It is not just the spread number. It is the *character* of the risk:

- **At the top (majors)**, the spread is the entire cost story. The price moves smoothly, the book is deep, and the only real friction is the 0.2-pip toll. You can size large and trade any time of day.
- **In the middle (EM)**, the spread widens to 8 pips — but the spread is now the *small* risk. The big risk is the **gap**: a surprise rate decision, a political headline, an oil shock, and USD/MXN moves 3% in a candle with no liquidity in between. The 8-pip spread is the price of a calm market; the real cost is the tail.
- **At the bottom (exotics/NDF)**, all three liquidity components fail at once. The spread is 25–40 pips, the depth is shallow, and the book can *vanish* entirely on a headline — exactly what the franc did. Worse, for a restricted currency the only access is synthetic (an NDF), which layers settlement and counterparty questions on top of the price risk.

The ladder reframes a beginner's instinct. The natural question "which currency should I trade?" is the wrong one. The professional question is "what does this *pair* cost me, in spread, in gap risk, and in the hours I can actually transact?" — and that question is answered entirely by the rung.

### Depth and market impact: the cost that grows with your size

There is a second cost the spread chart cannot show you, because the spread is the cost of trading a *small* clip. For a large order, a different cost dominates: **market impact**, the amount your own order pushes the price against you as it eats through the available depth. The spread is the toll at the front door; market impact is what you pay because the room behind the door is not infinitely large.

The order book of a pair is a stack of resting orders at successively worse prices. The best offer might be \$1,000,000 of size; the next price up another \$2,000,000; the one above that \$5,000,000, and so on. If your order is small relative to that stack, you pay only the spread. If your order is *large*, you exhaust the best price and climb the stack — every additional million you buy is filled at a worse rate than the last. The deeper the book, the further you can go before you start climbing; the thinner the book, the sooner you start paying impact on top of the spread.

This is the precise sense in which a major and an exotic differ beyond the spread number. EUR/USD can absorb a \$50,000,000 order in the deep hours with barely a flicker, because the stack behind the best price is enormous. A thin exotic might have only a few million of resting size before the price runs away, so a \$50,000,000 order would walk the book and move the rate against you by a full percent or more. Two pairs can quote the *same* spread on the screen and yet have wildly different costs the moment your size is real.

#### Worked example: market impact on a large order in a thin book

You need to buy **\$20,000,000** of an EM pair. The quoted spread looks like a tame **8 pips**, so a naive cost estimate is `\$20,000,000 × 0.0008 = \$16,000`. But the book is shallow: the best price holds only \$3,000,000, and filling the full \$20,000,000 walks the rate up by, say, **0.4%** on average versus where you started.

That average slippage of 0.4% costs `\$20,000,000 × 0.004 = \$80,000` — **five times** the spread estimate. Your real all-in cost is roughly `\$16,000 + \$80,000 = \$96,000`, not the \$16,000 the spread implied.

Run the same \$20,000,000 in EUR/USD in the deep hours, where the book swallows it whole: you pay essentially just the spread, `\$20,000,000 × 0.00002 ≈ \$400`. The intuition: the spread is the cost for a small trade, but for a large one the *depth* of the pair's book is the cost that matters — and depth, like the spread, falls steeply as you descend the tiers, so a big order in a thin pair pays twice.

The practical upshot is that professionals do not just check the spread; they check whether their *size* is small or large relative to the pair's depth, and they break large orders into slices traded over time (and over the deep hours) to keep their own market impact down. Size relative to depth is the hidden third axis of the cost, sitting right alongside the spread and the gap risk.

## Why the dollar sits at the centre of the map

Every tier above is defined relative to the dollar. Majors *are* dollar pairs. Crosses are explicitly *non*-dollar pairs (defined by the absence). EM and exotic pairs are quoted as USD-versus-the-local-currency. The dollar is the hub of the entire wheel — and that is not a US-centric bias in how I drew the map; it is an empirical fact about how the market actually trades.

![Horizontal bar chart of share of FX trades with each currency on one side dollar at eighty-eight percent](/imgs/blogs/majors-minors-and-exotics-the-map-of-currency-pairs-4.png)

The Bank for International Settlements runs a survey every three years that measures, among other things, what share of all FX trades have a given currency on one side. Because every trade has two sides, the shares sum to 200%, not 100%. The 2022 survey is stark: **the US dollar is on one side of 88.5% of all trades.** The euro, a distant second, is on 30.5%. The yen on 16.7%. Everything else trails far behind.

What this means in practice is that the dollar is the universal *vehicle* currency. If a Thai exporter wants to convert baht into Swedish krona, there is almost never a deep THB/SEK market. The trade routes through dollars: baht to dollars, dollars to krona. The dollar is the connecting node that turns a sparse web of thinly-traded local pairs into one liquid network. (This is the same "cross through the dollar" logic that gave crosses their name, scaled up to the whole planet.)

#### Worked example: routing a thin cross through the dollar hub

Suppose a Thai company needs to convert **\$1,000,000-equivalent of Thai baht into Swedish krona** (a THB/SEK exposure). There is no deep direct THB/SEK market, so the bank routes it through the dollar.

Leg one: sell baht for dollars (USD/THB), an EM pair with a spread of, say, **10 pips**, costing roughly `\$1,000,000 × 0.0010 = \$1,000`. Leg two: sell dollars for krona (USD/SEK), a deep pair with a spread of about **2 pips**, costing roughly `\$1,000,000 × 0.0002 = \$200`. Total cost of the round trip through the hub: about `\$1,000 + \$200 = \$1,200`.

Had the dollar *not* existed as a hub, and you were forced to trade a direct THB/SEK market so thin its spread were, say, **60 pips**, the cost would be `\$1,000,000 × 0.0060 = \$6,000`. The dollar hub turned a \$6,000 trade into a \$1,200 one — a 5× saving — simply by letting both legs tap the deepest market in the world. The lesson: the dollar's 88% ubiquity is not a vanity statistic; it is a *cost reduction* for the entire planet, because routing through one deep currency is cheaper than maintaining a thin direct market for every possible pair.

This is why the dollar's role deserves its own deep dives elsewhere in the ecosystem rather than a re-derivation here. The cross-asset view of the dollar as a single gravitational force is in [the dollar: cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity), and the macro mechanics of why the dollar rules markets are in [the dollar system: why USD rules markets](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy). For our map, the takeaway is structural: **the closer a pair is to the dollar, the deeper its liquidity, the tighter its spread, and the higher it sits on the ladder.** A pair's distance from the dollar hub is a surprisingly good predictor of its place on the map.

#### Worked example: the hidden cost of a cross, in dollars

Why is a cross more expensive than either of its legs? Because, in liquidity terms, you are paying to cross the dollar hub twice. Suppose you want to trade **\$1,000,000-equivalent of EUR/GBP**. There are two ways to build it.

The direct EUR/GBP market quotes a spread of about **0.8 pip**, so the round-trip cost is `\$1,000,000 × 0.00008 ≈ \$80`.

Building it from the two dollar legs instead — buy EUR/USD (spread 0.2 pip ≈ \$20) and sell GBP/USD (spread 0.5 pip ≈ \$50) — costs about `\$20 + \$50 = \$70`. The direct cross at \$80 is close to the sum of its legs (\$70) because a market-maker quoting EUR/GBP is, behind the screen, hedging through exactly those two dollar legs and passing the combined cost on to you, plus a margin.

![Stacked ladder of four liquidity tiers from majors to exotics with round-trip cost per tier](/imgs/blogs/majors-minors-and-exotics-the-map-of-currency-pairs-3.png)

The intuition: a cross is never cheaper than its dollar legs combined, because the dollar is the only truly deep market and every non-dollar pair is ultimately priced off it. When you trade a cross, you are renting the dollar's liquidity twice and paying for both rentals.

## The 24-hour clock: when the map is deep and when it is thin

So far the map has been static — a fixed gradient of pairs. But liquidity has a second dimension: **time**. The very same EUR/USD that is the deepest market on earth at noon in London is meaningfully thinner at 3 a.m. London time. FX is a 24-hour market on weekdays, but it is emphatically *not* uniformly liquid across those 24 hours. It is a relay race, and the baton is depth.

![Timeline of the 24-hour FX session clock from Sydney through Tokyo London and New York](/imgs/blogs/majors-minors-and-exotics-the-map-of-currency-pairs-5.png)

The market opens for the week on Sunday evening UTC as Sydney comes online, and runs continuously until New York closes on Friday evening. Within each weekday, liquidity is handed from financial centre to financial centre as the planet turns:

- **Sydney / Wellington** (from ~22:00 UTC) start the day. The book is thin; spreads are wide; this is when surprise gaps are most likely because there are fewest dealers awake to absorb a shock.
- **Tokyo / Asia** (~00:00–08:00 UTC) bring real depth to the *Asian* pairs — USD/JPY, AUD/USD, USD/CNH — but EUR/USD and the European crosses are still relatively quiet.
- **London** (~07:00–16:00 UTC) is the giant. The United Kingdom alone handles about **38%** of global FX turnover — more than the United States and every Asian centre combined. When London opens, spreads tighten across *every* pair, because the largest pool of dealers is now quoting.
- **New York** (~13:00–21:00 UTC) overlaps London for several hours, then runs on alone into the US afternoon as London leaves.

The single most important fact on this clock is the **overlap**. For roughly four hours, **12:00–16:00 UTC**, both London and New York are fully open at the same time. That is the deepest, tightest, most active window of the entire global day — and, not coincidentally, it is when the heaviest US economic data (CPI, payrolls, FOMC) lands, because that data drops into the moment the market is best able to absorb it.

![Intraday liquidity curve showing FX activity peaking at the London New York overlap window](/imgs/blogs/majors-minors-and-exotics-the-map-of-currency-pairs-6.png)

The curve makes the relay visible. Activity bottoms out in the dead Asian hours (roughly 01:00–05:00 UTC), climbs steeply as London wakes, peaks sharply in the London–New York overlap, and tapers as London leaves and New York winds down. The shape is the same every weekday. It tells you something practical that no price chart can: *when* you trade is part of *what* you pay.

Two features of the clock deserve a closer look, because they catch beginners out.

The first is the **fix**. At certain moments of the day, most famously the **4 p.m. London fix** (16:00 London time), a benchmark exchange rate is calculated as an average of trades in a short window. Index funds, pension funds, and corporates that need a single, auditable, non-discretionary rate all transact at the fix, so an enormous slug of one-directional flow concentrates into a few minutes. Liquidity is deep right then, but the price can be pushed by the imbalance of all those orders arriving together. The fix is a moment of maximum *volume* and, sometimes, of distorted *price* — a reminder that "deep" and "calm" are not the same thing.

The second is the **weekend gap**. FX closes Friday evening in New York and reopens Sunday evening in Sydney. The market does not stop *existing* over the weekend — geopolitics, elections, central-bank statements, wars all happen on Saturdays and Sundays — but you cannot trade. When the market reopens, it can *gap*: the first Sunday-evening price can be far from Friday's close, with no chance to have exited in between. A French election surprise, a surprise referendum, a weekend bank failure — all have produced Sunday-night gaps. The thin Sunday-evening reopen, with only Sydney online, is the single most gap-prone window of the entire week, which is exactly why it sits at the wide-spread end of the relay.

#### Worked example: the same trade, two different hours

You want to buy **\$5,000,000 of EUR/USD**. Consider doing it at two different times.

At **14:00 UTC** (the London–New York overlap, peak depth), EUR/USD trades at its tightest, say a **0.3-pip** effective spread on a \$5m clip. Your round-trip cost is roughly `\$5,000,000 × 0.00003 = \$150`.

At **02:00 UTC** (the dead Asian hours), the EUR/USD book is thin — European dealers are asleep — and your effective spread on the same \$5m might widen to **1.5 pips** as you eat through a shallow book. Your cost becomes roughly `\$5,000,000 × 0.00015 = \$750`.

Same pair, same size, same day — **five times the cost** purely from the clock. And that is the *calm* case. The bigger danger of the thin hours is not the wider spread; it is that a surprise headline in a thin book can gap the price far more than it would in a deep one, because there is no one home to take the other side. The intuition: liquidity is a schedule, not a constant, and trading a deep pair in a shallow hour quietly converts a major into something that behaves like a minor.

## Volatility climbs the same ladder

There is a beautiful regularity hiding in the map: the *price* of a pair's options — its implied volatility — climbs the tiers in lockstep with its spread. This is not a coincidence. Both the spread and the option price are the market quoting the *uncertainty* of a pair, just in two different instruments.

![Bar chart of one-month implied volatility by currency pair rising from EUR USD to USD TRY](/imgs/blogs/majors-minors-and-exotics-the-map-of-currency-pairs-7.png)

**Implied volatility** is the annualised price swing the options market expects, backed out from the cost of FX options. A higher number means the market is pricing a wider range of future outcomes — and is charging more to insure against them. Read the chart and the ladder reappears: EUR/USD around **7%**, USD/JPY around **9.5%**, the commodity-linked AUD/USD around **10.5%**, the EM pair USD/MXN at **13%**, and the exotic USD/TRY all the way up at **22%**.

The mechanism linking spread and vol is the dealer's risk again. A market-maker who quotes you a price has to hold the position until they can hedge it. The more the price can move while they hold it, the more they could lose — so they demand both a wider spread (to compensate for the immediate risk) *and* a higher option premium (to compensate for the optional risk). High vol and wide spreads are two faces of the same uncertainty. (The full machinery of how vol is priced and shaped lives in the options series — see [reading the vol surface like a trader](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) — and we do not re-derive the Greeks here.)

For our map the lesson is simple and powerful: **the spread, the gap risk, the implied vol, and the thin-hours danger are all the same phenomenon — uncertainty — measured by different instruments.** They rise together as you descend the tiers, which is why "where does this pair sit on the map?" answers half a dozen risk questions at once.

## How an FX trade actually settles: T+2 and CLS

We have mapped what you can trade and what it costs to trade it. There is one piece left that beginners almost never see, because it happens *after* the price screen goes quiet: **settlement**. When you click "buy EUR/USD," you have not yet received any euros or paid any dollars. You have only agreed a deal. The actual exchange of money happens later — and the "later" is where one of the oldest risks in finance lives.

![Pipeline of an FX trade settling from trade date through CLS to T plus two settlement](/imgs/blogs/majors-minors-and-exotics-the-map-of-currency-pairs-8.png)

The standard FX settlement convention is **T+2**: the two currencies actually change hands **two business days after the trade date**. You agree EUR/USD on Monday (trade date, "T"); the euros and dollars move on Wednesday (T+2). Two days does not sound like much, but it is two days during which one side could fail — and FX settles enormous sums, so the failure of one bank mid-settlement can cascade. This is not a theoretical worry. It happened.

The "**business days**" detail matters more than it looks. T+2 counts *business* days in *both* relevant currency centres, so weekends and holidays stretch the calendar. A EUR/USD trade done on a Thursday settles the following Monday (Friday + Monday are the two business days; the weekend does not count). A trade done just before a string of holidays can have a settlement date nearly a week away — and the longer that window, the longer you carry the price risk and, more importantly, the settlement risk. One major exception: USD/CAD settles **T+1** rather than T+2, a historical quirk of two adjacent, deeply-linked North American markets. The convention is not arbitrary — it exists because, before instant electronic confirmation, the back offices genuinely needed two days to match, confirm, and instruct the payments across time zones. The plumbing has sped up since, but the convention has mostly stuck.

There is also a quieter cost-saving step buried in the timeline: **netting**. A big bank does not settle each of its thousands of daily EUR/USD trades one by one. Its back office *nets* them down — if it bought \$800,000,000 and sold \$750,000,000 of euros against a single counterparty over the day, only the \$50,000,000 *difference* needs to actually move. Netting shrinks the gross flows by an order of magnitude, which is both a cost saving and a risk reduction: less money in flight means less money exposed if someone fails. CLS, which we are about to meet, performs exactly this netting across the whole market, which is why the actual cash that moves through it each day is a small fraction of the multi-trillion notional it settles.

On **26 June 1974**, a small German bank called Bankhaus Herstatt was shut down by regulators at the end of the German business day. But the New York day was still going. Counterparties had already paid Herstatt their Deutsche marks that morning, expecting to receive dollars in New York that afternoon. Herstatt was closed before it paid the dollars out. The Deutsche marks were gone; the dollars never came. This exact hazard — **you pay your leg, the other side defaults before paying theirs** — is now called **Herstatt risk** or, more generally, **settlement risk**. It is the single most dangerous moment in the life of an FX trade, because for a brief window you are fully exposed to the other side's survival.

The fix, built decades later, is **CLS** (Continuous Linked Settlement), a specialised settlement bank that went live in 2002. CLS settles FX on a **payment-versus-payment (PvP)** basis: it holds both legs of a trade and releases them *simultaneously*, or not at all. Your euros and the counterparty's dollars move in the same instant, through CLS, so neither side can be left having paid without being paid. It is the FX equivalent of an escrow agent who only hands over the keys and the cash at the exact same second. (The full mechanics — how CLS nets, which currencies it covers, what still settles outside it — are the subject of the sibling post [settlement risk and CLS: how FX actually clears](/blog/trading/forex/settlement-risk-and-cls-how-fx-actually-clears).)

#### Worked example: the settlement exposure on a \$1,000,000 trade

You buy **\$1,000,000 of EUR/USD** at 1.0800 from a counterparty bank, for value T+2. Walk the two days.

On **trade date (T)**, nothing has moved. You owe `\$1,000,000` in dollars; you are owed `\$1,000,000 / 1.0800 = 925,926 euros`. No cash has changed hands; your only exposure so far is that the *price* might move against you before settlement (a small risk over two days).

On **settlement date (T+2)**, the payments are due. **Without CLS**, picture the worst sequence: you wire your \$1,000,000 of dollars to the counterparty in the New York morning, and *before* it sends your 925,926 euros, it fails. Your entire `\$1,000,000` principal is at risk — not the spread, not a small move, the *whole notional*. That is Herstatt risk, and it is why settlement, not price, is the largest single-trade exposure in FX.

**With CLS**, the two legs are locked together: either your dollars and its euros both settle in the same instant, or neither does and you simply keep your dollars. The exposure collapses from \$1,000,000 of principal to essentially zero. The intuition: the spread is what you *pay*, but settlement is what you could *lose everything* over — and the entire CLS system exists to turn that catastrophic, all-or-nothing exposure into a non-event.

The share of global FX settled through CLS PvP has hovered around 40% in recent years — high for the major currencies it covers, but a reminder that a large slice of the market (especially EM and exotic pairs that CLS does not settle) still carries raw Herstatt risk. Which closes the loop on our map beautifully: the exotic pairs at the bottom of the ladder are not just the most expensive to *trade* — they are also the least protected when it comes time to *settle*.

## Common misconceptions

**"A currency is liquid or it isn't."** No — *pairs* have liquidity, not currencies, and even that is conditional on time. The Swiss franc is one of the most respected currencies on earth, yet EUR/CHF had zero liquidity for two minutes in January 2015. EUR/USD is the deepest pair on the planet at 14:00 UTC and noticeably thinner at 02:00 UTC. Liquidity is a property of *this pair, at this hour, in this market state* — never a fixed label on a currency.

**"The spread is just the broker's fee, so it's small and I can ignore it."** The spread is the single biggest recurring cost in FX, and it is not flat — it ranges from about \$20 to about \$4,000 on a \$1,000,000 trade depending on the tier. For an exotic, the spread alone can exceed your expected move. Ignoring it is how traders with a genuine edge in majors lose money trying to apply that edge to exotics.

**"FX is open 24 hours, so the time of day doesn't matter."** It is open 24 hours but not uniformly deep. The same trade can cost five times as much in the dead Asian hours as in the London–New York overlap, and a thin book gaps far harder on news. *When* you trade is part of *what* you pay, and the worst gaps in history happened in the thinnest hours.

**"Exotics are just majors with a wider spread."** The spread is the *least* of it. Exotics fail all three liquidity tests at once — wide spread, shallow depth, slow resilience — and many are not even deliverable, forcing you into NDFs with their own counterparty and settlement questions. An exotic can gap 20% on a headline (the lira has, repeatedly) in a way a major essentially never does. The risk is not "a bit more" — it is a different category.

**"Once I've agreed the price, the trade is done."** The price is agreed at trade date, but the money moves two business days later (T+2), and in that window you carry settlement risk — the chance the other side fails after you've paid. For the ~40% of FX that settles through CLS this is neutralised; for the rest, including most EM and exotic pairs, the full Herstatt exposure is real. The trade is not "done" until both currencies have actually landed.

## How it shows up in real markets

**The January 2015 franc shock — when a major behaved like an exotic.** EUR/CHF had a spread of well under a pip for years while the SNB held the 1.20 floor. When the floor broke on 15 January 2015, the pair fell from 1.2010 to below 0.8500 in roughly two minutes — about a 30% move — with *no liquidity in between*. Stop-loss orders, which only work if there is a buyer at the next price, were filled hundreds of pips away or not at all. Retail brokers like Alpari UK collapsed; some clients ended up owing more than their deposits. The point for the map: a pair's tier is a description of *normal* conditions, and in a regime break even a top-tier pair can momentarily teleport to the bottom of the ladder. Liquidity is a fair-weather promise.

**The August 2024 yen unwind — depth disappearing in the thin hours.** When the Bank of Japan hiked and the popular yen-funded carry trade unwound, USD/JPY fell from about 161.9 on 3 July to roughly 141.7 by 5 August 2024. The most violent leg happened in *Asian and early hours*, when the dollar–yen book is thinnest and a forced-liquidation cascade had the fewest dealers to absorb it. The session clock was not a footnote to that move — it was part of the mechanism. (The carry trade and its crashes are the subject of [carry-trade unwinds: 1998, 2008, 2024](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks); here the lesson is simply that the *when* amplified the *how much*.)

**The Turkish lira — an exotic spread that is also a directional story.** USD/TRY routinely trades at spreads of 25 pips or more and at implied vols above 20%, because the lira has lost the vast majority of its value over the past decade amid unorthodox monetary policy. For a trader, the wide spread and high vol are not separate annoyances — they are the market correctly pricing a currency where a single policy headline can move the rate several percent. The exotic tier is, in a sense, the market being honest: it charges you up front for the gap risk it knows is coming.

**The Vietnamese dong — the exotic you cannot deliver.** USD/VND is managed by the State Bank of Vietnam in a crawl, not a free float, and the dong cannot be freely moved across borders. There is no deep onshore spot market a foreign trader can simply tap; offshore exposure is taken through NDFs, which is why our chart lists USD/VND at a ~40-pip equivalent friction. When devaluation pressure builds, the gap between the offshore NDF rate and the onshore band *widens*, signalling stress — a dynamic with no equivalent in the freely-traded majors. (The full Vietnam picture is in [non-deliverable forwards: trading uninvestable currencies](/blog/trading/forex/non-deliverable-forwards-trading-uninvestable-currencies).)

**The September 2022 pound — a major gapping in the thin Asian hours.** After the UK "mini-budget" of 23 September 2022, sterling cratered. The most dramatic move came in the early hours of Monday 26 September, in the thin Asian session, when GBP/USD touched an all-time intraday low of about **1.035** — a near-8% collapse from a few days earlier. Cable is a *major*, one of the deepest pairs on earth in London hours. But in the thin Asian window, with a domestic policy panic underway, its book behaved like a far lower tier: the price gapped fast with few dealers to slow it. It is the franc lesson in a different currency — the tier label describes the normal regime, and the combination of a shock *plus* a thin hour can yank even a top-tier pair down the ladder for a few brutal minutes.

**Settlement, in the quiet days of March 2020.** During the COVID dollar-funding panic, FX volumes spiked and the plumbing was tested. The reason it did not produce a Herstatt-style cascade among the major currencies is precisely that CLS settles the bulk of major-currency FX on a payment-versus-payment basis — the same protection that did not exist in 1974. The system held *because* the post-Herstatt machinery did its job. Settlement risk is the dog that did not bark, and that silence is engineered.

## The takeaway: read the pair, not just the price

Here is the shift this post is trying to install. A beginner looks at an FX screen and sees a *price* — EUR/USD at 1.0800, USD/TRY at 34.50 — and treats them as the same kind of object with a different number. They are not. They are objects from entirely different tiers of the map, and the price is the least important thing about them.

What actually distinguishes them is everything *behind* the price:

- **The spread** — about \$20 to round-trip a million in EUR/USD, about \$2,500 in USD/TRY. The toll sets your minimum viable edge.
- **The gap risk** — a major moves smoothly; an exotic teleports on a headline; even a major can teleport in a regime break.
- **The hours** — the same pair is five times cheaper in the London–New York overlap than in the dead Asian hours, and far safer from surprise gaps.
- **The settlement** — your largest single-trade exposure is not the price moving but the two-day window where the other side could fail, neutralised by CLS for the majors and raw for most exotics.

All four of those are read off one variable: where the pair sits on the liquidity map. That is why the map is the right mental model to end the Foundations track on. It connects straight back to the spine of this series — *every position is a pair, a relative bet of two monies* — and adds the practitioner's correction: **not all pairs are created equal, and the inequality is liquidity.** The rate differential between two countries tells you which *way* a pair should move (the engine the rest of this series unpacks); the pair's place on the map tells you what it *costs* to express that view and whether you can survive being early.

So before you ever take a currency position, ask the map's four questions: What tier is this pair? What is the spread, in dollars, on my size? Am I trading it in a deep hour or a thin one? And when this settles in two days, is the other side's failure my problem or CLS's? Answer those, and you are no longer staring at a price — you are reading the pair. That is the difference between trading FX and gambling on a number.

There is one final reason this map is the right place to close the Foundations track. Everything that comes *next* in this series — the rate differentials that drive a pair, the carry trade that harvests the gap between two countries' rates, the dollar smile, the speculative attacks on pegs, the crisis playbooks — all of it presupposes that you know *which* pair you are trading and what it costs to be there. A brilliant view on the direction of the yen is worth nothing if you express it in a pair you cannot afford to trade or cannot exit in the hour you need to. The map is the board the rest of the game is played on. You can have the best read in the world on where a currency is going; the map decides whether that read can survive contact with the spread, the depth, the clock, and the settlement window long enough to pay off. Learn the board first, then learn the moves — that is the order this series follows, and this post is the board.

## Further reading & cross-links

- [Base, quote, pips, and how to read an FX quote](/blog/trading/forex/base-quote-pips-and-how-to-read-an-fx-quote) — the mechanics of a quote and the pip, the units underneath every spread number in this post.
- [The biggest market on earth: inside the interbank FX market](/blog/trading/forex/the-biggest-market-on-earth-inside-the-interbank-fx-market) — who actually makes the prices and why the dollar is the hub of the whole network.
- [Settlement risk and CLS: how FX actually clears](/blog/trading/forex/settlement-risk-and-cls-how-fx-actually-clears) — the deep dive on T+2, Herstatt risk, and payment-versus-payment settlement.
- [Non-deliverable forwards: trading uninvestable currencies](/blog/trading/forex/non-deliverable-forwards-trading-uninvestable-currencies) — how you take a position in an exotic like the dong that you cannot legally deliver.
- [The dollar: cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity) — why the dollar sits at the centre of the map and pulls on every other asset.
- [Reading the vol surface like a trader](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) — how the implied-volatility ladder in this post is priced and shaped across pairs.
