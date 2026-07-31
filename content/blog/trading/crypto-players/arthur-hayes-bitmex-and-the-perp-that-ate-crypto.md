---
title: "Arthur Hayes, BitMEX, and the Perp That Ate Crypto"
date: "2026-07-29"
publishDate: "2026-07-29"
description: "A build-from-zero explanation of the perpetual swap — why a futures contract with no expiry needs a funding rate, how inverse contracts pay you in the thing you are betting on, how liquidation engines and insurance funds actually work, and why funding and forced selling became crypto's dominant short-term price mechanism. Written the week BitMEX, the venue that popularised the contract, announced its own closure."
tags: ["crypto", "crypto-players", "perpetual-swaps", "derivatives", "bitmex", "arthur-hayes", "funding-rate", "liquidation", "leverage", "market-structure", "order-book"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 58
---

> [!important]
> **TL;DR** — The perpetual swap is a futures contract with the expiry date deleted, and everything strange about crypto price action follows from what had to be bolted on to replace it.
>
> - A normal futures contract is dragged back to the spot price by settlement. Delete expiry and nothing drags it back — so the perp invents a **funding rate**: a payment made every eight hours from whichever side is crowded to whichever side is empty.
> - BitMEX's XBTUSD is an **inverse** contract: quoted in dollars, margined and settled in bitcoin. That makes a long *short convexity* in coin terms — a 100% rally earns 0.5 BTC while a 50% crash costs 1.0 BTC.
> - At the 1% initial margin BitMEX's own API reports for XBTUSD, a long is liquidated **0.50% below entry**, with only about \$247 between the liquidation price and bankruptcy on a \$50,000 position. That gap is why insurance funds and auto-deleveraging exist.
> - Because the leverage sits in perps, a large share of the selling in a fast hour is **not chosen by anyone** — it is emitted by liquidation engines executing a rule, and it is price-*insensitive* and self-amplifying. In the illustrative cascade below, \$150M of chosen selling manufactures \$730M of forced selling and roughly a 10% drawdown.
> - The number to remember: **0.50%**. That is how far bitcoin has to move against a 100x long before somebody else closes the trade for them — less than many one-minute candles.
> - On 23 July 2026, HDR Global Trading Limited announced that BitMEX will close on 23 September 2026. The venue is ending; the contract it popularised is now how crypto trades everywhere.

There is a strange fact buried in this week's crypto news. On 23 July 2026, the exchange most responsible for the way crypto prices move announced that it is shutting down — and it will barely change the way crypto prices move at all.

BitMEX published a notice titled "Important Message from BitMEX" announcing "the closure of the BitMEX Exchange, which will take effect on 23 September 2026 at UTC 04:00:00." The board of HDR Global Trading Limited, it said, had decided to close the exchange "[f]ollowing a strategic review of the business and the broader crypto industry." From 26 August 2026 at 04:00 UTC, traders may only reduce positions; anything still open at the closure time "will be immediately force closed." The day before, the exchange had published a separate notice delisting 35 derivatives contracts on 30 July 2026.

Read that closing time again: **04:00 UTC**. It is not an arbitrary hour. It is one of the three moments each day when BitMEX charges funding — the payment mechanism this article is about. The exchange chose to die on the clock it invented.

That clock is the reason this post exists. BitMEX is closing, and the thing it popularised — a derivatives contract with no expiry date, held to the spot price by a recurring payment between traders — is now the single most important instrument in crypto. It is where the leverage lives. And because it is where the leverage lives, its two housekeeping mechanisms, **funding** and **liquidation**, have quietly become the dominant driver of what bitcoin's price does over hours and days.

If you have ever watched bitcoin drop several percent in minutes on no news at all, you have watched this machinery run. Nobody decided that. A rule executed.

![The perpetual swap and the three machines that hold it together](/imgs/blogs/arthur-hayes-bitmex-and-the-perp-that-ate-crypto-1.webp)

The diagram above is the mental model for everything that follows. A perpetual swap is not one clever contract; it is a bet on price with three separate machines bolted to it, each doing a job that expiry used to do for free. **Funding** pins the contract's price to the spot market. **Mark price** decides what your position is worth from moment to moment. The **liquidation engine** decides when you are no longer allowed to hold it. Every section below is a tour of one of those machines, and every strange thing crypto prices do in a violent hour is one of them working exactly as designed.

A note before we start: this is an explanation of mechanics and market structure, not investment advice, and nothing here is a suggestion to trade anything.

## Foundations: the building blocks

Skip this section if you already trade derivatives. If you do not, read it carefully — everything after it assumes these seven ideas, and no more than these seven.

### Spot, and what "owning" means

The **spot market** is where you buy the actual thing. You send \$50,000 to an exchange, you receive 1 bitcoin, and the bitcoin is yours. If the price doubles you have something worth \$100,000. If it goes to zero you have something worth nothing, but nobody takes it from you along the way. Spot is simple and it is unforgiving in only one direction: you cannot lose more than you put in.

The price on the spot market is the reference every derivative in this article ultimately points at. When we later say "the index price", we mean a blend of several exchanges' spot prices, averaged together so that one venue printing a weird trade cannot move everyone's contracts.

### Notional, and why it is the number that matters

**Notional** is the dollar size of your exposure — the amount of the underlying asset your position behaves like. If you control exposure to \$50,000 of bitcoin, your notional is \$50,000, regardless of how much money you actually posted to get it.

This distinction is the whole game. Notional is what moves your profit and loss. **Margin** is what you put up. Keeping those two words separate in your head is most of what separates people who understand leverage from people who are surprised by it.

### Leverage and margin

**Margin** is the collateral you deposit to open a position. **Leverage** is the ratio of notional to margin.

Post \$5,000 of margin against \$50,000 of notional and you are at 10x leverage. The position still gains and loses as if you owned \$50,000 of bitcoin — a 1% move is still \$500 — but that \$500 is now 10% of your money instead of 1% of it. Leverage does not change your exposure to the market. It changes how much of *your* capital a given market move represents.

Two margin numbers matter, and they are different:

- **Initial margin** is the minimum you must post to *open* the position. It sets your maximum leverage: 1% initial margin means you can open 100 times your collateral.
- **Maintenance margin** is the minimum you must *keep* to hold the position. Fall below it and the exchange closes you out. It is always lower than initial margin, and the gap between them is your entire margin for error.

BitMEX's public instrument API reports XBTUSD with an initial margin of 1% and a maintenance margin of 0.5% (BitMEX API, accessed 29 July 2026). The commonly quoted "100x leverage" figure is *derived* from that 1% initial margin — one divided by 0.01 — rather than being a number BitMEX states in those words on that endpoint.

### Long, short, and who is on the other side

Going **long** means you profit if the price rises. Going **short** means you profit if it falls.

Here is the part beginners find genuinely surprising: in a derivatives market, **every long is matched by a short**. These are not shares issued by a company; they are bets created in pairs. When you open a long, somebody opened the matching short at the same instant. The total notional of open longs always equals the total notional of open shorts, exactly, at every moment. That sum is called **open interest** — the total notional of contracts currently outstanding — and it is one of the two numbers we will use later to read positioning.

Open interest is a measure of *how much money has arrived*, not of who is winning. It rises when a new long and a new short create a position together and falls when a pair closes.

### Futures, expiry, and basis

A **futures contract** is an agreement to settle the difference on an asset's price at a fixed future date. Buy a bitcoin future expiring on 27 June at \$52,000 and, on 27 June, you receive the difference between \$52,000 and wherever bitcoin actually is.

The gap between the futures price and the spot price is called the **basis**. If the future trades at \$52,000 while spot is \$50,000, the basis is \$2,000, or 4%. Basis exists because holding a futures contract is not the same as holding the asset: you are not paying for storage, you are not tying up the full cash amount, and there is time for things to happen.

The crucial property — and the one this whole article turns on — is that **basis is forced to zero at expiry**. On settlement day, the contract pays out against the spot price by definition. A future trading 4% above spot with one day to go is a nearly-free 4% to whoever sells the future and buys the spot, so traders do exactly that until the gap closes. Expiry is not a detail of futures contracts. Expiry is the mechanism that makes a futures price *mean* anything.

### Basis points

Finally, a unit. A **basis point** (bp) is one hundredth of a percentage point — 0.01%. Funding rates are small enough that they are often quoted this way: a 0.01% funding rate is 1 basis point.

#### Worked example 1: the simplest possible leverage calculation

You have \$5,000. You open a long on \$50,000 of bitcoin at a price of \$50,000 — that is 1 bitcoin of exposure at 10x leverage.

- Bitcoin rises 4%, to \$52,000. Your notional gained \$2,000. Your account is now \$7,000 — a **40% gain** on your \$5,000.
- Instead, bitcoin falls 4%, to \$48,000. Your notional lost \$2,000. Your account is now \$3,000 — a **40% loss**.
- Now the uncomfortable one. Bitcoin falls 10%, to \$45,000. Your notional lost \$5,000. Your account is now \$0.

The intuition: leverage multiplies the *percentage* impact of a move without changing the move. At 10x, a 10% move against you is your entire account. At 100x, a 1% move is. Every mechanism in the rest of this article exists because exchanges have to deal with that third bullet point before it happens.

## 1. The problem: expiry is what made futures work

Now the interesting part. Suppose you want to offer traders leveraged exposure to bitcoin, and you want it to be as easy to hold as spot — no expiry dates, no rolling from the June contract into the September contract, no delivery, one ticker forever.

You immediately have a problem, and it is not a small one.

![Why a dated future converges and a perpetual does not](/imgs/blogs/arthur-hayes-bitmex-and-the-perp-that-ate-crypto-2.webp)

The left panel is a normal dated future. In the illustration it trades 3% above spot ninety days out. As expiry approaches, that basis is squeezed toward zero, and on settlement day it *is* zero, because the contract pays against spot by definition. The convergence is not a market convention or a behaviour traders have agreed to. It is arithmetic enforced by the settlement rule.

The right panel is a contract with no expiry. Ask yourself: what makes the line come back to zero?

Nothing does. If more people want leveraged length than leveraged shortness — which, in crypto, is most of the time — the contract simply trades above spot. And it keeps trading above spot, because there is no date on which anyone is forced to reconcile it. Over months, a perpetual with no correction mechanism would drift arbitrarily far from the asset it claims to track, and would stop being a bitcoin product at all. It would become its own asset: a token whose price reflects nothing but the local balance of enthusiasm on one exchange.

So the designers of the perpetual swap had to answer a specific question:

> If you delete the deadline that forces a contract back to reality, you have to invent a cost that makes drifting away from reality expensive.

That invented cost is the funding rate, and it is the most important idea in this article.

## 2. Funding: the spring that replaced expiry

Here is the intuition before any formula.

Imagine a crowded room where everyone wants to stand on the same side. Rather than forbidding it, the building charges rent — and the rent is paid by the crowded side directly to the empty side. The more lopsided the room becomes, the more expensive it is to stay on the popular side and the more you are paid to stand on the unpopular one. Nobody is banned from anything. It just gets costly, and cost eventually moves people.

That is funding. When the perpetual trades **above** the spot index, longs pay shorts. When it trades **below**, shorts pay longs. The payment is proportional to your position's size, it repeats on a fixed schedule, and — this matters — **the exchange is not a party to it.** Funding is a transfer between traders. The venue collects trading fees, but the funding payment itself passes from one side of the book to the other.

The effect is a spring. Push the perp far above spot and you create a growing payment that rewards anyone willing to sell the perp and buy the spot — a trade that is close to market-neutral and collects funding for as long as the gap persists. That trade is what drags the price back.

<figure class="blog-anim">
<svg viewBox="0 0 720 470" role="img" aria-label="The perpetual swap price drifts above and below the spot index; when it trades rich the funding rate turns positive and longs pay shorts, when it trades cheap the funding rate turns negative and shorts pay longs" style="width:100%;height:auto;max-width:760px">
<title>Funding flips sign when the perp crosses the spot index</title>
<style>
.fk-band-hi{fill:#2f9e44;opacity:.10}
.fk-band-lo{fill:#e03131;opacity:.10}
.fk-idx{stroke:var(--text-primary,#1f2937);stroke-width:2;stroke-dasharray:8 6}
.fk-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.fk-sub{font:500 12.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.fk-hd{font:700 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);letter-spacing:.04em}
.fk-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.fk-pill-hi{fill:#2f9e44}
.fk-pill-lo{fill:#e03131}
.fk-pilltxt{font:700 15px ui-sans-serif,system-ui;fill:#ffffff}
.fk-hi{font:700 15px ui-sans-serif,system-ui;fill:#2f9e44}
.fk-lo{font:700 15px ui-sans-serif,system-ui;fill:#e03131}
@keyframes fk-a{0%,40%{opacity:1}48%,90%{opacity:0}100%{opacity:1}}
@keyframes fk-b{0%,40%{opacity:0}48%,90%{opacity:1}100%{opacity:0}}
.fk-sa{animation:fk-a 12s ease-in-out infinite}
.fk-sb{animation:fk-b 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.fk-sa{animation:none;opacity:1}.fk-sb{animation:none;opacity:.18}}
</style>
<text class="fk-hd" x="56" y="30">PERPETUAL SWAP PRICE vs SPOT INDEX</text>
<rect class="fk-band-hi" x="56" y="52" width="608" height="86" rx="6"/>
<rect class="fk-band-lo" x="56" y="142" width="608" height="86" rx="6"/>
<text class="fk-sub" x="70" y="72">perp trades RICH — traders are paying up for leveraged length</text>
<text class="fk-sub" x="70" y="222">perp trades CHEAP — traders are paying up for leveraged shorts</text>
<line class="fk-idx" x1="56" y1="140" x2="664" y2="140"/>
<text class="fk-lbl" x="56" y="252">spot index price  $50,000</text>
<g class="fk-sa">
<rect class="fk-pill-hi" x="300" y="80" width="130" height="34" rx="17"/>
<text class="fk-pilltxt" x="316" y="103">$50,150</text>
<text class="fk-hi" x="446" y="103">+0.30% premium</text>
</g>
<g class="fk-sb">
<rect class="fk-pill-lo" x="300" y="166" width="130" height="34" rx="17"/>
<text class="fk-pilltxt" x="316" y="189">$49,950</text>
<text class="fk-lo" x="446" y="189">&#8722;0.10% premium</text>
</g>
<text class="fk-hd" x="56" y="298">THE 8-HOUR FUNDING PAYMENT THAT PULLS IT BACK</text>
<rect class="fk-box" x="56" y="320" width="200" height="62" rx="8"/>
<text class="fk-lbl" x="96" y="349">LONGS</text>
<text class="fk-sub" x="76" y="369">$50,000 notional</text>
<rect class="fk-box" x="464" y="320" width="200" height="62" rx="8"/>
<text class="fk-lbl" x="504" y="349">SHORTS</text>
<text class="fk-sub" x="484" y="369">$50,000 notional</text>
<g class="fk-sa">
<path class="fk-pill-lo" d="M266 344 L432 344 L432 336 L456 351 L432 366 L432 358 L266 358 Z"/>
<text class="fk-lo" x="272" y="410">funding +0.075%  &#8594;  longs pay shorts  $37.50 every 8 hours</text>
</g>
<g class="fk-sb">
<path class="fk-pill-hi" d="M454 344 L288 344 L288 336 L264 351 L288 366 L288 358 L454 358 Z"/>
<text class="fk-hi" x="268" y="410">funding &#8722;0.020%  &#8592;  shorts pay longs  $10.00 every 8 hours</text>
</g>
<text class="fk-sub" x="56" y="442">Funding is charged at 04:00, 12:00 and 20:00 UTC. It is a transfer between traders &#8212; the exchange takes no cut of it.</text>
</svg>
<figcaption>Nothing forces a contract with no expiry back to spot, so the perp invents its own gravity: when the perp trades above the index, longs pay shorts every eight hours; when it trades below, shorts pay longs. On a $50,000 position, a +0.075% rate is $37.50 per payment — cheap for one interval, roughly 82% a year if it never resets.</figcaption>
</figure>

### The formula, and how to read it

A note on sourcing before the maths. BitMEX's own documentation pages for funding, fair-price marking and liquidation were unreachable at the time of writing — they are client-rendered and returned no content. Rather than reconstruct BitMEX's formulas from memory, this section teaches the mechanism from **Binance's published funding-rate methodology**, which is materially the same design and which I could actually read. BitMEX-specific claims in this article are limited to what its public instrument API returns.

Binance documents the funding rate as:

$$F = P + \text{clamp}(I - P,\ 0.05\%,\ -0.05\%)$$

where, for an interval other than eight hours, the result is divided by ${8/N}$ for an $N$-hour interval (Binance funding-rate documentation, accessed 29 July 2026). The symbols:

- $F$ is the **funding rate** for the interval — the percentage of your position value that changes hands.
- $P$ is the **premium index**: how far the perpetual is trading above or below the underlying index, sampled continuously rather than taken as a single snapshot. This is the market-driven part.
- $I$ is the **interest rate** component: a fixed term representing the cost-of-carry difference between the two currencies in the pair. Binance's documentation uses a 0.03% daily assumption, which works out to 0.01% per eight-hour interval.
- $\text{clamp}(x, a, b)$ simply means "take $x$, but never let it exceed $a$ or fall below $b$." Here it bounds the interest term's influence to ±0.05%.

Read it in plain English and it says: *the funding rate is basically the premium, nudged by a small fixed interest term, and the nudge is never allowed to get large.* When the perp trades at a big premium, $P$ dominates and funding is roughly the premium. When the perp sits almost exactly on the index, the small interest term is what is left, and funding settles near +0.01% per interval.

That default matters. A perpetual sitting perfectly on spot still charges longs about 0.01% every eight hours — 0.03% a day, roughly 11% a year. **The resting state of a perpetual market is a slow bleed from longs to shorts.** Leverage on the long side is not free even when you are right about direction and the contract is behaving.

![How the funding rate is assembled and charged](/imgs/blogs/arthur-hayes-bitmex-and-the-perp-that-ate-crypto-3.webp)

### The caps, and the clock

Two implementation details do real work in a crisis.

First, funding is **capped per interval**. Binance's documentation gives a ±0.3% cap for BTCUSDT (accessed 29 July 2026). Without a cap, a violently dislocated perp could generate a funding payment large enough to liquidate positions by itself — the correction mechanism would become a second source of forced selling. The cap keeps the spring from becoming a hammer. It also means that in a genuine dislocation, funding *stops* fully reflecting the premium: the rate pegs at its cap and the gap has to close some other way.

Second, funding is charged on a **fixed clock**, and the clocks are not the same everywhere. BitMEX's API reports XBTUSD funding on an eight-hour interval at **04:00, 12:00 and 20:00 UTC** (BitMEX API, accessed 29 July 2026). Binance documents its BTCUSDT perpetual funding at **00:00, 08:00 and 16:00 UTC** (Binance documentation, accessed 29 July 2026). The two venues are offset by four hours.

That offset is not trivia. It means that at any given moment the two largest perpetual markets are at different points in their funding cycles, and a trader can be long the venue that is about to pay them and short the venue that is about to charge them. Cross-venue funding spreads are a real, boring, mechanical trade, and they are one of the forces that keeps different exchanges' perps roughly in line with each other.

It also produces a behavioural artefact you can see on a chart: activity clusters around funding timestamps. Positions that are marginal get closed in the minutes before a payment, and re-opened after. When funding is extreme, those minutes get busy.

#### Worked example 2: what funding actually costs you

You are long \$50,000 of a bitcoin perpetual. Funding is charged three times a day. Position value is what the rate applies to, so the arithmetic is direct.

**The calm case.** Funding is at its resting rate of +0.01%.

- Each payment: \$50,000 × 0.0001 = **\$5.00**
- Per day: \$5.00 × 3 = **\$15.00**
- Annualised: 0.01% × 3 × 365 = **10.95% a year**

**The crowded case.** Sentiment is hot, the perp trades at a 0.30% premium, funding prints +0.075%.

- Each payment: \$50,000 × 0.00075 = **\$37.50**
- Per day: \$37.50 × 3 = **\$112.50**
- Annualised: 0.075% × 3 × 365 = **82.1% a year**

**The dislocation case.** Funding pegs at the 0.3% cap.

- Each payment: \$50,000 × 0.003 = **\$150.00**
- Per day: \$150 × 3 = **\$450.00**
- Annualised: 0.3% × 3 × 365 = **328.5% a year**

Now put that against your actual capital rather than your notional. At 10x leverage you posted \$5,000 to hold this \$50,000 position. In the dislocation case you are paying \$450 a day — **9% of your margin, per day**, before the price does anything at all.

The intuition: funding is quoted on notional but paid out of margin, so leverage multiplies its bite exactly the way it multiplies price moves. A rate that looks like a rounding error at the notional level is a countdown timer at the margin level.

And notice what that does to a crowded market. High positive funding is a signal that longs are crowded — but it is also a *mechanism* that steadily drains the crowded side's margin, moving them closer to the liquidation levels we are about to build. Crowding does not merely coincide with fragility. Funding manufactures it.

## 3. Inverse contracts: paid in the thing you are betting on

BitMEX's flagship contract, XBTUSD, has a design choice that confuses almost everyone the first time they meet it, and it explains a surprising amount of how bitcoin trades.

The contract is **quoted in US dollars** and **settled in bitcoin**. You deposit bitcoin as collateral, you think in dollar prices, and your profit and loss arrives in coins. There is no dollar anywhere in the system.

BitMEX's instrument API reports XBTUSD as an inverse contract, quoted in USD, settled in XBT, with a multiplier of **−100,000,000** (BitMEX API, accessed 29 July 2026). That negative one-hundred-million is the whole design compressed into one number: it says the contract's value in satoshis is minus one hundred million, divided by the price. One contract, therefore, is worth **one dollar** of bitcoin.

### Why anyone would build it this way

The reason is historical and pragmatic. An offshore exchange that wants to run a bitcoin derivatives market without ever touching a bank has a problem: collateral has to be *something*, and if the something is dollars you need banks. Denominate everything in bitcoin and the exchange never holds fiat, never needs a correspondent bank, and never has to explain a wire. In an era before deep, liquid dollar stablecoins, the inverse contract was not a clever financial innovation so much as a way to build a dollar-quoted market with no dollars in it.

That choice has a consequence its designers understood and most of its users did not.

### The convexity nobody warns you about

Work out what happens to your bitcoin balance. Since one contract is one dollar of bitcoin, a position of $N$ contracts entered at price $P_0$ and closed at price $P_1$ produces a profit, **denominated in bitcoin**, of:

$$\text{PnL}_{\text{BTC}} = N \times \left( \frac{1}{P_0} - \frac{1}{P_1} \right)$$

This follows directly from the −100,000,000 multiplier above rather than from a BitMEX statement of the formula: position value in bitcoin is contracts divided by price, so the change in value is the difference of two reciprocals.

Reciprocals are the point. Your profit is linear in *one over the price*, not in the price. And one over the price is a curve, not a line.

Take a \$50,000 long entered when bitcoin is at \$50,000 — exactly 1 bitcoin of notional.

| Bitcoin at exit | Move | Profit / loss in BTC |
| --- | --- | --- |
| \$100,000 | +100% | **+0.500 BTC** |
| \$75,000 | +50% | +0.333 BTC |
| \$60,000 | +20% | +0.167 BTC |
| \$50,000 | 0% | 0.000 BTC |
| \$45,000 | −10% | −0.111 BTC |
| \$40,000 | −20% | −0.250 BTC |
| \$33,333 | −33% | −0.500 BTC |
| \$25,000 | −50% | **−1.000 BTC** |

Look at the two extremes. Bitcoin **doubling** earns you half a bitcoin. Bitcoin **halving** costs you a whole one. Your gains in coin terms decelerate as you are proved right and your losses accelerate as you are proved wrong.

![The inverse contract's payoff in bitcoin terms](/imgs/blogs/arthur-hayes-bitmex-and-the-perp-that-ate-crypto-4.webp)

In derivatives language, being long an inverse perpetual is being **short convexity** in coin terms — the position's sensitivity moves against you in both directions. A linear contract margined in a dollar stablecoin has no such curve: it pays a straight line, which is why nearly every venue launched afterward offers one.

#### Worked example 3: the same trade, two contract types

You have 1 bitcoin. Bitcoin is \$50,000. You want long exposure to \$50,000 of bitcoin, and you are going to be wrong: the price falls to \$25,000.

**Inverse contract, bitcoin collateral.** You post 1 BTC and hold \$50,000 of notional.

- PnL = 50,000 × (1/50,000 − 1/25,000) = 50,000 × (0.00002 − 0.00004) = **−1.000 BTC**
- Your collateral was 1 BTC. You have **zero bitcoin left** — and that bitcoin, which you would otherwise still hold, is now itself worth half what it was.

**Linear contract, stablecoin collateral.** You sold your bitcoin for \$50,000 first and posted that as collateral.

- PnL = \$50,000 × (25,000 − 50,000)/50,000 = **−\$25,000**
- You have \$25,000 of stablecoin left, which at \$25,000 per bitcoin buys **1 bitcoin**.

Same directional view, same size, same market move — and the inverse trader ends with nothing while the linear trader ends with the coin they started with. The difference is not luck. It is that the inverse trader's collateral was falling at the same time as their position, so they were compounding one loss into another.

The intuition: on an inverse contract you are long the asset *twice* — once through your position and once through the collateral backing it. That is fine on the way up and vicious on the way down.

## 4. Mark price: the number that decides your fate

Here is a question that sounds pedantic and is not: **what is the price of bitcoin right now?**

There is no single answer, and a perpetual swap has to pick one — because that choice determines whether your position lives or dies. There are three candidates, and they behave differently.

- The **last traded price** on this exchange's own order book. It is real, it is immediate, and it is the easiest number in the world to push around. One trade at a bad price during a thin moment sets it.
- The **index price**: an average of several *spot* exchanges' prices. It is much harder to move because you would have to move several independent markets at once.
- The **mark price**: the index price plus an adjustment for the perpetual's own funding basis. It is the exchange's estimate of what the contract is genuinely worth right now, deliberately insulated from its own order book.

BitMEX's API reports XBTUSD with a mark method of **`FairPrice`** (BitMEX API, accessed 29 July 2026) — that is, the contract is marked to an index-anchored fair price, not to its own last trade.

![Three prices, one position](/imgs/blogs/arthur-hayes-bitmex-and-the-perp-that-ate-crypto-10.webp)

### Why this exists, and what it prevents

Imagine mark price did not exist and liquidations ran off the last traded price. Then anyone who could see where liquidation levels sat would have an obvious play: push the local order book down hard enough, for one second, to trip them. The forced sells would do the rest, and the price would come back — but the liquidated traders would stay liquidated. The attack would cost far less than it earned, because you only need to move *one* book for *one instant*.

Marking to an index-anchored fair price largely closes that door. To trip liquidations you now have to move the actual spot price across several exchanges and hold it there, which is enormously more expensive and is exactly the kind of thing regulators and exchange surveillance look for.

This is worth stating plainly, because the internet is full of confident claims about exchanges "hunting" retail stops. The honest version is narrower and more useful: **liquidation clusters are publicly inferable, and any large trader can see that pushing price to a level where forced selling begins will get help from the forced selling.** That is a structural feature of a leveraged market, not evidence that any particular venue or firm did anything. Nothing in this article asserts that any named exchange or firm manipulated a market. What you can safely conclude is much simpler: if your liquidation price sits somewhere obvious, you have told the market where you are, and you should assume the market can read.

### What it costs

Fair-price marking is not free, and its cost runs the other way. Because mark price lags the local book, there are moments when the tape prints a number better than your mark and you are liquidated anyway — the local book recovered but the index did not. Traders experience this as "I got liquidated at a price that never printed," and it is the same mechanism that protected them from the wick, running in the unfavourable direction.

You cannot have one without the other. A price that is hard to push around in your favour is also hard to push around in your defence.

## 5. Liquidation: when the exchange closes the trade for you

Now the machine that does the damage.

Your position is closed involuntarily when your margin falls to the **maintenance margin** — the minimum equity required to keep holding it. BitMEX's API reports XBTUSD with a maintenance margin of 0.5% and an initial margin of 1% (BitMEX API, accessed 29 July 2026).

Two prices matter, and conflating them is the most common mistake in this whole subject:

- **Liquidation price** — where the exchange *starts* closing you. Your margin has fallen to maintenance level, but there is still some equity left.
- **Bankruptcy price** — where your margin is *exactly zero*. Below this, the position owes money it does not have.

The gap between them is the entire working room the liquidation engine has. It has to get your position closed in the market somewhere between those two prices. Close it above bankruptcy and everything is fine. Close it below and somebody else has to cover the difference.

### Deriving the liquidation price

For an inverse contract with notional $N$, entry $P_0$, leverage $L$ and maintenance margin $\text{MM}$, the position is liquidated when remaining margin equals the maintenance requirement:

$$\frac{N}{L \cdot P_0} + N\left(\frac{1}{P_0} - \frac{1}{P}\right) = \frac{\text{MM} \cdot N}{P}$$

The left side is your posted margin plus your unrealised profit, both in bitcoin; the right side is what you are required to keep. Cancel $N$, rearrange, and:

$$P_{\text{liq}} = P_0 \cdot \frac{1 + \text{MM}}{1 + 1/L} \qquad P_{\text{bankruptcy}} = \frac{P_0}{1 + 1/L}$$

#### Worked example 4: how far is the trapdoor?

Bitcoin is at \$50,000. You go long with a maintenance margin of 0.5%.

**At 100x leverage** (the maximum implied by BitMEX's 1% initial margin):

- Liquidation price = \$50,000 × 1.005 / 1.01 = **\$49,752**
- That is 0.495% below entry — call it **half a percent**.
- Bankruptcy price = \$50,000 / 1.01 = **\$49,505**, or 0.99% below entry.
- The engine's entire working room: \$49,752 − \$49,505 = **\$247**.

**At 25x leverage**:

- Liquidation price = \$50,000 × 1.005 / 1.04 = **\$48,317**, or 3.37% below entry.
- Bankruptcy price = \$50,000 / 1.04 = **\$48,077**, or 3.85% below entry.
- Working room: **\$240**.

![Liquidation and bankruptcy prices at 100x and 25x](/imgs/blogs/arthur-hayes-bitmex-and-the-perp-that-ate-crypto-5.webp)

Sit with the 100x number. Bitcoin has to move **half a percent** against you before your position is taken away. Half a percent is not a market event. It is a minute of ordinary Tuesday. It is smaller than the range of many individual one-minute candles.

And then look at the \$247. That is the entire cushion between "the exchange begins selling your position" and "your position is worth less than nothing." On a calm day, \$247 of bitcoin sells instantly and the gap never opens. In a fast market — when the book is thin, when a thousand other positions are being sold at the same instant, when market makers have widened their quotes because they cannot price the risk — \$247 of room is nothing at all.

The intuition: leverage does not change your exposure to price. It changes how little price has to move before somebody else closes the trade for you — and how little room they have to do it in.

That second consequence is the one that turns an individual's bad day into everyone's bad hour.

## 6. When liquidation is not enough: the loss waterfall

Follow the \$247 to its conclusion.

The engine begins selling your 100x long at \$49,752. It needs to be done by \$49,505. If the market is orderly, it sells into resting bids well above bankruptcy, your remaining equity is returned, and nobody else is involved. If the market is not orderly — if the book has thinned out and everyone else's engine is selling at the same moment — the fill comes in *below* \$49,505.

Now there is a hole. Your position lost more than you had. Somebody must absorb the difference, and it will not be the exchange's own balance sheet.

The industry converged on a three-layer answer, and Binance's liquidation documentation states the first two layers plainly: "the Futures Insurance Fund will bear the losses arising from the Bankrupt Position", and "[i]f the losses arising from a Bankrupt Position cannot be funded by the Futures Insurance Funds, the matching engine will automatically liquidate the Bankrupt Positions and some opposing non-bankrupt trader's' positions" (Binance liquidation-protocol documentation, accessed 29 July 2026). BitMEX's own equivalent documentation was unreachable at the time of writing, so the description below is the general industry mechanism as documented by Binance, not a claim about BitMEX's specific implementation.

![The three-layer loss waterfall](/imgs/blogs/arthur-hayes-bitmex-and-the-perp-that-ate-crypto-6.webp)

**Layer one: your margin.** It absorbs everything down to the bankruptcy price. This is the layer everyone expects.

**Layer two: the insurance fund.** A pool that covers the gap between what a liquidation actually filled at and what the position was worth at bankruptcy. Its funding mechanism has a property worth understanding: the fund *grows* when liquidations fill **better** than the bankruptcy price — the leftover equity is retained rather than returned — and *shrinks* when they fill worse. Which means the fund fattens during calm markets, when liquidations are isolated and fill easily, and drains precisely during the violent hours when it is needed. Its balance is a lagging indicator of exactly the wrong thing.

**Layer three: auto-deleveraging.** When the fund cannot cover the hole, the exchange reaches across the book and force-closes traders who are *winning*. In Binance's wording, the matching engine liquidates "some opposing non-bankrupt trader's' positions." Venues generally rank candidates by some combination of profitability and leverage, so the most profitable, most leveraged traders on the correct side go first. That page did not document the ranking formula, and I am not going to invent one.

Read layer three again, because it is genuinely strange and most people never encounter it until it happens to them. You called the crash correctly. You are short. You are up substantially. And the exchange closes your position anyway, at the bankruptcy price of somebody who was wrong, because there is no one else left to take the other side.

> A perpetual swap is a closed system. Every dollar a winner takes out came from a loser, and when the losers run out of dollars, the system takes it from the winners instead.

This is the deepest structural fact about the instrument, and it is the one that most changes how you should think about it. There is no external capital backstopping a perpetual market. There is no clearing house with a bank behind it, no lender of last resort. The only money in the room is the money the traders brought. When a violent move creates more losses than the losing side posted, the balancing entry has to come from the winning side. Auto-deleveraging is simply that accounting identity, executed.

## 7. Why the perp won

Step back from the machinery and ask why anyone preferred this.

The perpetual swap is, on inspection, a *worse* instrument than a dated future in several ways. It has a recurring cost that is unpredictable. It has a correction mechanism that can peg at a cap exactly when you need it to work. In its inverse form it hands you a convexity you did not ask for. It exposes you to auto-deleveraging, a risk with no analogue in traditional futures markets.

It won anyway, on convenience.

![Spot, dated futures and perpetual swaps compared](/imgs/blogs/arthur-hayes-bitmex-and-the-perp-that-ate-crypto-7.webp)

Consider what a dated future demands of you. You must pick an expiry. You must watch that expiry approach. Before it arrives you must **roll** — close the June contract and open the September one — which costs you the spread twice and exposes you to whatever the basis happens to be on the day you do it. If you want continuous exposure for two years, you do this eight times, and eight times you pay.

The perpetual deletes all of that. One ticker, forever. No expiry to diarise, no roll to execute, no calendar spread to price. You open a position and it stays open until you close it or the market closes it for you. The funding rate is the price of that convenience, and it is charged in small, continuous slices rather than in occasional lumps.

That is the entire pitch, and it turned out to be enough. Traders will accept a worse instrument that they never have to think about over a better one that demands attention three times a year.

A note on what I am *not* claiming here. This article does not quote market-share or volume statistics, because I was unable to source current ones from a primary reference while writing it, and an unsourced share number is exactly the kind of confident-sounding fabrication this blog tries not to publish. What I can point at is narrower and still telling: two of the largest crypto derivatives venues both run this same contract design and both publish funding methodologies for it — BitMEX's API describes XBTUSD as a perpetual with an eight-hour funding interval, and Binance publishes a funding-rate methodology for its BTCUSDT perpetual (both accessed 29 July 2026). The design propagated. Whatever the precise share, the leverage in crypto lives in contracts that work like this one.

And that is what makes the next section a market-structure story rather than a curiosity about one product's plumbing.

## 8. Cascade geometry: how forced selling makes its own weather

Here is the consequence of everything above, stated as simply as possible.

In a market where most of the leverage sits in perpetual swaps, a large fraction of the sell orders that arrive during a fast move were **not chosen by anyone**. They were emitted by liquidation engines executing a rule. And that produces a feedback loop with a property no ordinary selling has: it is *triggered by price*, so the more the price falls, the more of it there is.

![The liquidation cascade as a feedback loop](/imgs/blogs/arthur-hayes-bitmex-and-the-perp-that-ate-crypto-8.webp)

Ordinary selling is self-limiting. As the price drops, sellers become less eager and buyers become more eager. That negative feedback is what makes markets work.

Forced selling inverts the sign. A liquidation engine does not become less eager at lower prices — it becomes *more* active, because lower prices trip more liquidation levels. And it is price-insensitive by construction: it is not trying to get a good fill, it is trying to close a position before it goes bankrupt. It sells at market, into whatever is there.

Meanwhile the buy side thins at exactly the wrong moment. Market makers, who supply most of the resting bids, widen their quotes or withdraw them when volatility spikes, because they cannot price inventory risk in a market moving this fast. (This is the mechanism covered in [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) and [inventory risk, hedging and delta-neutrality](/blog/trading/crypto-players/inventory-risk-hedging-and-delta-neutrality).) So depth falls precisely as forced supply rises.

#### Worked example 5: how \$150M becomes \$730M

The numbers below are illustrative round figures for a hypothetical order book, not measured market data — the point is the arithmetic of the loop, not the specific depths.

Bitcoin is \$50,000. Suppose the **cumulative** bids available on the way down look like this — that is, the total dollar value of buy orders you would consume by selling all the way to each level:

| Price | Move | Cumulative bids down to here |
| --- | --- | --- |
| \$49,500 | −1% | \$120M |
| \$49,000 | −2% | \$210M |
| \$48,000 | −4% | \$330M |
| \$46,500 | −7% | \$560M |
| \$45,000 | −10% | \$900M |

And suppose the aggregate liquidation map shows clusters of long positions here:

| Level | Who | Notional |
| --- | --- | --- |
| \$49,300 | 50x longs | \$180M |
| \$48,300 | 25x longs | \$240M |
| \$47,900 | 20x longs | \$310M |

**Step 1 — the trigger.** Someone sells \$150M at market. Nothing sinister; a fund is de-risking. To absorb \$150M we need to travel past the \$120M available at −1% and into the next band, landing around **\$49,330**, a fall of about 1.3%.

**Step 2 — the first cluster trips.** \$49,330 is below \$49,300, so \$180M of 50x longs are now force-sold. Total selling to absorb is now \$150M + \$180M = \$330M, which takes us to the **−4% level, \$48,000**.

**Step 3 — two more clusters trip.** Passing \$48,300 and \$47,900 releases \$240M and \$310M — \$550M more. Cumulative selling is now \$880M, which lands between the −7% and −10% rows, around **\$45,100**.

**The tally.** \$150M of chosen selling produced **\$730M** of forced selling — nearly five times as much — and roughly a **10% drawdown**. Nobody sold \$730M of bitcoin because they wanted to. They sold it because a price crossed a number in a database.

<figure class="blog-anim">
<svg viewBox="0 0 720 500" role="img" aria-label="A liquidation cascade steps down a cumulative bid-depth ladder: an initial 150 million dollar sell consumes the bids in the first one percent, trips a 180 million dollar cluster of leveraged longs, which consumes deeper bids and trips two more clusters totalling 550 million, driving the price roughly ten percent lower" style="width:100%;height:auto;max-width:760px">
<title>How $150M of selling becomes $730M of forced selling</title>
<style>
.fc-bar{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.4}
.fc-eat{fill:#e03131;opacity:0}
.fc-px{font:700 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.fc-pc{font:500 12.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.fc-dp{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.fc-hd{font:700 12.5px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);letter-spacing:.04em}
.fc-clu{fill:#f08c00;opacity:.25}
.fc-ptr{fill:#e03131}
.fc-ptrtxt{font:700 13.5px ui-sans-serif,system-ui;fill:#e03131}
.fc-tot{font:700 16px ui-sans-serif,system-ui;fill:#e03131}
@keyframes fc-e1{0%,8%{opacity:0}14%,92%{opacity:.8}100%{opacity:0}}
@keyframes fc-e2{0%,30%{opacity:0}36%,92%{opacity:.8}100%{opacity:0}}
@keyframes fc-e3{0%,55%{opacity:0}61%,92%{opacity:.8}100%{opacity:0}}
@keyframes fc-t1{0%,10%{opacity:.25}16%,92%{opacity:1}100%{opacity:.25}}
@keyframes fc-t2{0%,32%{opacity:.25}38%,92%{opacity:1}100%{opacity:.25}}
@keyframes fc-ptr{0%,8%{transform:translateY(0)}14%,30%{transform:translateY(82px)}36%,55%{transform:translateY(186px)}61%,92%{transform:translateY(300px)}100%{transform:translateY(0)}}
@keyframes fc-v0{0%,8%{opacity:1}12%,92%{opacity:0}100%{opacity:1}}
@keyframes fc-v1{0%,10%{opacity:0}16%,30%{opacity:1}34%,100%{opacity:0}}
@keyframes fc-v2{0%,32%{opacity:0}38%,92%{opacity:1}100%{opacity:0}}
.fc-r1{animation:fc-e1 14s ease-in-out infinite}
.fc-r2{animation:fc-e2 14s ease-in-out infinite}
.fc-r3{animation:fc-e3 14s ease-in-out infinite}
.fc-c1{animation:fc-t1 14s ease-in-out infinite}
.fc-c2{animation:fc-t2 14s ease-in-out infinite}
.fc-pointer{animation:fc-ptr 14s ease-in-out infinite}
.fc-s0{animation:fc-v0 14s ease-in-out infinite}
.fc-s1{animation:fc-v1 14s ease-in-out infinite}
.fc-s2{animation:fc-v2 14s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.fc-r1,.fc-r2,.fc-r3{animation:none;opacity:.8}.fc-c1,.fc-c2{animation:none;opacity:1}.fc-pointer{animation:none;transform:translateY(300px)}.fc-s0,.fc-s1{animation:none;opacity:0}.fc-s2{animation:none;opacity:1}}
</style>
<text class="fc-hd" x="40" y="26">PRICE LEVEL</text>
<text class="fc-hd" x="196" y="26">CUMULATIVE BIDS AVAILABLE DOWN TO THAT LEVEL</text>
<text class="fc-px" x="40" y="88">$50,000</text><text class="fc-pc" x="118" y="88">start</text>
<text class="fc-px" x="40" y="150">$49,500</text><text class="fc-pc" x="118" y="150">&#8722;1%</text>
<rect class="fc-bar" x="196" y="134" width="45" height="22" rx="4"/>
<rect class="fc-eat fc-r1" x="196" y="134" width="45" height="22" rx="4"/>
<text class="fc-dp" x="252" y="151">$120M</text>
<text class="fc-px" x="40" y="212">$49,000</text><text class="fc-pc" x="118" y="212">&#8722;2%</text>
<rect class="fc-bar" x="196" y="196" width="79" height="22" rx="4"/>
<rect class="fc-eat fc-r2" x="196" y="196" width="79" height="22" rx="4"/>
<text class="fc-dp" x="286" y="213">$210M</text>
<text class="fc-px" x="40" y="274">$48,000</text><text class="fc-pc" x="118" y="274">&#8722;4%</text>
<rect class="fc-bar" x="196" y="258" width="125" height="22" rx="4"/>
<rect class="fc-eat fc-r2" x="196" y="258" width="125" height="22" rx="4"/>
<text class="fc-dp" x="332" y="275">$330M</text>
<text class="fc-px" x="40" y="336">$46,500</text><text class="fc-pc" x="118" y="336">&#8722;7%</text>
<rect class="fc-bar" x="196" y="320" width="212" height="22" rx="4"/>
<rect class="fc-eat fc-r3" x="196" y="320" width="212" height="22" rx="4"/>
<text class="fc-dp" x="419" y="337">$560M</text>
<text class="fc-px" x="40" y="398">$45,000</text><text class="fc-pc" x="118" y="398">&#8722;10%</text>
<rect class="fc-bar" x="196" y="382" width="340" height="22" rx="4"/>
<rect class="fc-eat fc-r3" x="196" y="382" width="340" height="22" rx="4"/>
<text class="fc-dp" x="547" y="399">$900M</text>
<g class="fc-c1">
<rect class="fc-clu" x="470" y="164" width="196" height="26" rx="6"/>
<text class="fc-px" x="480" y="182">$49,300  50x longs  $180M</text>
</g>
<g class="fc-c2">
<rect class="fc-clu" x="470" y="226" width="196" height="26" rx="6"/>
<text class="fc-px" x="480" y="244">$48,300  25x longs  $240M</text>
</g>
<g class="fc-c2">
<rect class="fc-clu" x="470" y="288" width="196" height="26" rx="6"/>
<text class="fc-px" x="480" y="306">$47,900  20x longs  $310M</text>
</g>
<g class="fc-pointer" transform="translate(150,82)">
<path class="fc-ptr" d="M0 0 L-22 -10 L-22 10 Z"/>
<text class="fc-ptrtxt" x="-104" y="-16">price</text>
</g>
<text class="fc-hd" x="40" y="452">FORCED SELLING TRIGGERED SO FAR, ON TOP OF THE ORIGINAL $150M SELL ORDER</text>
<text class="fc-tot fc-s0" x="40" y="480">$0 &#8212; the book is absorbing it</text>
<text class="fc-tot fc-s1" x="40" y="480">$180M &#8212; the first cluster is force-sold at market</text>
<text class="fc-tot fc-s2" x="40" y="480">$730M &#8212; nearly 5x the selling that started it, and the price is near $45,100</text>
</svg>
<figcaption>Forced sells do not politely wait for buyers. A $150M market sell eats the bids down to about $49,330, which trips a $180M cluster of 50x longs; that $180M reaches $48,000, tripping $240M and then $310M more. The $150M that started it produced $730M of selling nobody chose to do, and roughly a 10% drawdown.</figcaption>
</figure>

### The asymmetry, and why crashes are faster than rallies

Cascades happen in both directions, but they are not symmetric, for two reasons.

The first is positioning. Crypto's leveraged flow skews long most of the time — which is exactly what persistently positive funding tells you. More long leverage means more liquidation levels stacked *below* the price than above it, so downside moves have more fuel.

The second is inverse contracts. As section 3 showed, a long on an inverse perpetual has collateral that falls in value at the same time as the position. That means a falling price erodes both sides of the margin ratio at once, dragging liquidation prices closer in a way a rising price does not mirror.

Short squeezes are real and can be extremely violent — when funding is deeply negative and clusters are stacked *above* the price, the same machinery runs upward. But the default configuration of a crypto perpetual market is more fuel below than above, which is why the phrase "it took the stairs up and the elevator down" is a description of market structure and not a mood.

## 9. How this shows up in price

So: you now understand the machinery. What do you actually *look at*?

![Reading positioning from funding and open interest](/imgs/blogs/arthur-hayes-bitmex-and-the-perp-that-ate-crypto-9.webp)

### Funding as a positioning gauge

Funding is a **price**, and like any price it tells you about supply and demand — here, the supply and demand for leveraged exposure. Persistently positive funding means longs are paying to be long: leveraged demand exceeds leveraged supply. Negative funding means the reverse.

The useful reframe is that funding is not a forecast; it is a **cost of carry that reveals crowding**. When funding sits far above its resting rate for days, you are looking at a market where one side is paying meaningfully for the privilege of being there, and where that payment is steadily draining their margin.

### Open interest is the other half

Funding alone is ambiguous. Combine it with open interest and it sharpens considerably, because open interest tells you whether positions are being *added* or *removed*:

- **Funding positive and open interest rising** — new leveraged longs arriving. Liquidation clusters building below the price. This is the configuration that precedes flushes.
- **Funding positive and open interest falling** — longs closing while still paying. Leverage is leaving, and the price move that comes with it is unwinding rather than accumulation.
- **Funding negative and open interest rising** — new leveraged shorts. Clusters building *above* price. Squeeze fuel.
- **Funding negative and open interest falling** — shorts covering. The flush has already happened and leverage has left the system.

The single most useful pattern is **divergence**: price making new highs while open interest falls means the rally is being driven by spot buying, not leverage, which is a structurally sturdier move than the same rally on rising open interest and rising funding.

### Where the liquidation clusters sit

You can estimate liquidation levels because they are arithmetic, not secrets. Common leverage settings are round numbers — 5x, 10x, 25x, 50x, 100x — and everyone entering near the same price with the same leverage gets a liquidation price near the same level. Third-party analytics firms publish "liquidation heatmaps" built on exactly this inference.

Two honest caveats, because this is where confident nonsense flourishes. First, these maps are **estimates**, built from public funding and open-interest data plus assumptions about the distribution of leverage; they are not the exchange's internal book. Second, the fact that they are inferable is a *structural* observation about a leveraged market, not evidence that anyone acts on them improperly. The defensive reading is the only one worth having: **if your liquidation price is at an obvious level, you should assume it is not a secret.**

### The tell that costs nothing to watch

If you only watch one thing, watch **funding at an extreme in the same direction for several consecutive intervals while open interest climbs.** That combination says the crowded side is both large and paying to stay — the exact configuration where a modest, ordinary sell order can start the loop in worked example 5. It does not tell you when. It tells you that the fuel is stacked, and that is a different and more honest kind of information than a prediction.

Also, cheaply: watch the perp's premium to spot around funding timestamps. Persistent large premiums going *into* a funding payment, followed by a lurch afterward, is the market re-pricing the cost of carry three times a day, and it is one of the more legible rhythms in crypto. For the underlying mechanics of how any of this order flow becomes a printed price, see [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) and [inside an exchange: the matching engine and the order book](/blog/trading/capital-markets/inside-an-exchange-the-matching-engine-and-the-order-book).

## 10. The venue is a player, not a neutral table

Everything so far has treated the exchange as infrastructure — a set of rules that executes. It is worth being blunt about the fact that it is also a business, with revenue that depends on the very behaviour the rules produce. This series has a whole post on that idea: [exchanges are players, not just venues](/blog/trading/crypto-players/exchanges-are-players-not-just-venues).

A perpetual-swap venue earns from volume. Volume comes from turnover, and leverage is a turnover multiplier: a trader with \$5,000 who can control \$500,000 generates a hundred times the fees of a trader with \$5,000 who cannot. Leverage also generates liquidations, and liquidations are themselves trades that pay fees. And the insurance-fund mechanism means well-executed liquidations retain the liquidated trader's residual equity rather than returning it.

None of this is hidden or improper — it is the published design. But it does mean that the decision to offer 100-to-1 leverage is not a neutral engineering choice. It is a revenue decision, and the person making it is not the person bearing the risk.

The scale of that business, as characterised by its regulator, is worth stating with its source. In its 1 October 2020 press release, the CFTC alleged that BitMEX "has illegally offered leveraged retail commodity transactions, futures, options, and swaps on cryptocurrencies" with up to 100-to-1 leverage from November 2014 onward, and cited **over \$11 billion in bitcoin deposits** and **more than \$1 billion in fee revenue since 2014** (CFTC Release 8270-20, 1 October 2020).

### The legal record

Matters of public record, with their sources and dates:

On **1 October 2020**, the CFTC filed a civil enforcement action in the U.S. District Court for the Southern District of New York against five BitMEX entities — HDR Global Trading Limited, 100x Holding Limited, ABS Global Trading Limited, Shine Effort Inc Limited and HDR Global Services (Bermuda) Limited — and against Arthur Hayes, Ben Delo and Samuel Reed as owner-operators. The charges included operating an unregistered trading platform, failing to register as a designated contract market or swap execution facility, acting as an unregistered futures commission merchant, and failing to implement customer-identification, know-your-customer and anti-money-laundering procedures (CFTC Release 8270-20, 1 October 2020).

The same day, the U.S. Attorney's Office for the Southern District of New York charged four individuals — **Arthur Hayes** (co-founder and CEO), **Benjamin Delo** (co-founder), **Samuel Reed** (co-founder and CTO) and **Gregory Dwyer** (BitMEX's first employee, later head of business development) — each with one count of violating the Bank Secrecy Act and one count of conspiring to violate it, each count carrying a maximum of five years. The core allegation was that they "willfully fail[ed] to establish, implement, and maintain an adequate anti-money laundering ('AML') program" at BitMEX (DOJ/SDNY release 20-218, 1 October 2020). A charge is an allegation, not a finding of guilt — but in this case all four later pleaded guilty, as set out below.

On **10 August 2021**, the CFTC announced a consent order resolving the civil case. The order "requires the BitMEX entities to pay a \$100 million civil monetary penalty", of which "up to \$50 million of the penalty may be offset by payments" made under a separate FinCEN action (CFTC Release 8412-21, 10 August 2021). FinCEN assessed its own \$100 million penalty the same day — its first enforcement action against a futures commission merchant — to be "satisfied by immediate payments totaling \$80 million to FinCEN and the CFTC, with \$20 million suspended pending the successful completion of the SAR lookback and independent consultant reviews" (FinCEN, 10 August 2021).

FinCEN's findings are worth reading against section 3 of this article, because they describe what the no-fiat design looked like from the inside: for over six years there was no compliant anti-money-laundering or customer identification programme; at least \$209 million of transactions took place with known darknet markets or unregistered money services businesses providing mixing services; the exchange failed to file a suspicious activity report on at least 588 specific transactions; and from roughly 2014 through 2020 it collected only an email address from customers. FinCEN also found that "[i]n some instances, BitMEX senior leadership altered U.S. customer information to hide the customer's true location" (FinCEN, 10 August 2021).

### The outcomes

All four individuals pleaded guilty to violating the Bank Secrecy Act:

| Individual | Guilty plea | Agreed fine | Sentence |
| --- | --- | --- | --- |
| Arthur Hayes | 24 Feb 2022 | \$10 million | 2 years' probation with 6 months' home detention (20 May 2022) |
| Benjamin Delo | 24 Feb 2022 | \$10 million | 30 months' probation (15 June 2022) |
| Samuel Reed | 9 Mar 2022 | \$10 million | 18 months' probation (13 July 2022) |
| Gregory Dwyer | 8 Aug 2022 | \$150,000 | 1 year's probation (16 November 2022) |

Sources: the plea dates and agreed fines come from the SDNY press releases of 24 February 2022, 9 March 2022 and 8 August 2022; Hayes' sentence from the SDNY release of 20 May 2022. SDNY does not appear to have issued sentencing releases for Delo, Reed or Dwyer — their sentence terms above come from the DOJ Office of the Pardon Attorney's clemency table (accessed 29 July 2026), which lists terms without narrative. Whether the fines were ultimately paid is not something I could establish, so this article does not say.

Then two more turns that most people missed.

On **10 July 2024**, the company itself — HDR Global Trading Limited, incorporated in the Republic of Seychelles — pleaded guilty to one count of violating the Bank Secrecy Act, and on **15 January 2025** it was sentenced to a **\$100 million fine plus two years' probation** (DOJ/SDNY releases 24-244 and 25-010).

And on **27 March 2025**, President Donald J. Trump granted pardons covering **Arthur Hayes, Benjamin Delo, Samuel Reed, Gregory Dwyer — and HDR Global Trading Limited itself** (DOJ Office of the Pardon Attorney clemency list, accessed 29 July 2026). The corporation was pardoned alongside its founders. What effect the pardons had on the fines already imposed is not something I can source, and I am not going to guess.

### And then the ending

On **23 July 2026**, BitMEX published "Important Message from BitMEX", announcing "the closure of the BitMEX Exchange, which will take effect on 23 September 2026 at UTC 04:00:00". The board of HDR Global Trading Limited had decided to close the exchange "[f]ollowing a strategic review of the business and the broader crypto industry". From 26 August 2026 at 04:00 UTC positions may only be reduced; anything still open at closure "will be immediately force closed" (BitMEX, 23 July 2026). The previous day, the exchange announced the delisting of 35 derivatives contracts effective 30 July 2026 (BitMEX, 22 July 2026).

So the venue ends the way its traders often did: with a deadline, a reduce-only window, and a forced close for anyone who did not act in time. And it ends at 04:00 UTC — a funding timestamp.

The contract, of course, does not end. It is listed on every major derivatives venue in crypto, and the machinery in this article will run tomorrow exactly as it ran yesterday. That is the actual legacy: not a company, but a market structure. BitMEX's most durable product was never the exchange. It was the idea that you can delete an expiry date if you are willing to charge rent for the privilege.

## 11. The second act, and how much of that voice is signal

Arthur Hayes' post-BitMEX career is as a writer. He publishes essays on Substack under the title **Crypto Trader Digest**, where the publication describes him as "Chief Investment Officer of Maelstrom, co-founder and former CEO of BitMEX" and reports more than 70,000 subscribers (cryptohayes.substack.com, accessed 29 July 2026).

That is a genuinely large audience for long-form macro writing, and it raises a question worth taking seriously rather than cynically: **how much of that voice is signal?**

Three honest observations, none of which requires assuming either that he is a genius or that he is talking his book.

**First, structural amplification is not the same as insight.** A founder of a major venue starts with an audience the quality of their argument did not earn. That audience then makes the writing newsworthy, which grows the audience further. The size of a following measures distribution, not accuracy. This is true of every widely-read market commentator and is not a criticism of any particular one.

**Second, position disclosure changes what a piece of writing is.** Someone who runs an investment firm and publishes a thesis is doing two things at once: explaining a view and describing a book. Both can be honest simultaneously. But a reader who treats the second as though it were only the first has misread the genre. The correct question is never "is this person right?" — it is "what does this person hold, and what would make them wrong?"

**Third — and this is the part that gets skipped — I have no empirical evidence to offer here.** I could not source any study measuring whether Hayes' published calls move prices, or whether they front-run or lag the moves they describe. I am not going to assert an effect I cannot demonstrate, in either direction. What can be said from the mechanics in this article is narrower and more useful: in a market where positioning is visible through funding and open interest, *narrative moves attention, and attention moves leverage, and leverage is what moves price violently.* A widely-read essay does not need to change anyone's mind about bitcoin's long-run value to matter. It only needs to change how many people are leveraged, and in which direction, before the next \$150M sell order arrives.

That is a much lower bar than "moving the market", and it is the one that actually connects a writer to a price chart.

## Common misconceptions

**"Funding is a fee the exchange charges."** It is a transfer between traders. When you pay funding, another trader receives it — the venue is the clearing mechanism, not the counterparty. This matters because it means funding cannot be "abolished" by a friendlier exchange; it is the thing doing the pinning.

**"High funding means the price is about to fall."** High funding means longs are crowded and paying. Crowded positioning can stay crowded for weeks, and prices frequently keep rising through it. What high funding reliably tells you is that the fuel for a fast move is stacked on one side — not when, or whether, it ignites.

**"With 100x leverage I can make 100x the profit."** You get 100 times the *percentage* sensitivity, and with it a liquidation price roughly half a percent away. At that distance, the binding constraint is not whether your view is right but whether ordinary noise reaches you first. Higher leverage does not increase expected return; it converts your position into a bet on short-term volatility that you probably did not intend to make.

**"If I'm right about the direction, I can't lose."** Two mechanisms say otherwise. Funding can drain a correctly-positioned account over weeks. And auto-deleveraging can close a winning position without your consent when the other side's losses exceed the insurance fund. On a perpetual, being right is necessary and not sufficient.

**"Liquidation means I lose exactly my margin."** Liquidation *begins* before your margin is gone, at the maintenance-margin threshold, so a clean liquidation usually leaves a residual. In a fast market the fill can come below the bankruptcy price, in which case the shortfall goes to the insurance fund and, past that, to auto-deleveraging. The outcome depends on the state of the book at that instant, not only on your own numbers.

**"Inverse and linear contracts are the same trade in different units."** They are not. As worked example 3 showed, the same directional view, same size, same market move leaves an inverse trader with nothing and a linear trader with their original bitcoin. Units are not cosmetic when the units are also the collateral.

## How it shows up in real markets

### 1. 1 October 2020 — the day the operators became defendants

The CFTC filed suit in the Southern District of New York against five BitMEX entities and three named owner-operators, alleging the platform had illegally offered leveraged retail commodity transactions with up to 100-to-1 leverage from November 2014 onward, and citing over \$11 billion in bitcoin deposits and more than \$1 billion in fee revenue since 2014 (CFTC Release 8270-20, 1 October 2020). The same release described a parallel criminal indictment of four individuals for Bank Secrecy Act offences.

The mechanism from this article in action: the two figures the CFTC chose to cite are a *deposit* number and a *fee* number, and the gap between them is the leverage story. A venue holding \$11 billion of deposits generated more than \$1 billion of fees because those deposits were not sitting still — they were margin, turning over at a multiple of themselves. Fee revenue of that scale relative to deposits is only achievable with leverage and the turnover it produces. The business model and the market structure are the same fact viewed from two sides.

### 2. 10 August 2021 — the price of the compliance gap

The CFTC's consent order required the BitMEX entities to pay a \$100 million civil monetary penalty, up to half of which could be offset against a parallel FinCEN assessment (CFTC Release 8412-21, 10 August 2021). FinCEN's own findings put numbers on the compliance gap: at least \$209 million transacted with known darknet markets or unregistered money services businesses providing mixing services, at least 588 specific suspicious transactions with no SAR filed, and an account-opening process that for roughly six years collected only an email address (FinCEN, 10 August 2021). Against the more than \$1 billion of fee revenue cited in the original CFTC complaint, the penalty was a fraction of the period's takings.

The lesson is not about whether the number was adequate — reasonable people disagree, and it is not a question this article can settle. It is that the offshore, no-fiat, bitcoin-margined design described in section 3 was never purely a technical choice. Inverse contracts let a venue run a dollar-quoted market without dollars, and a venue without dollars has a different relationship with banking regulation than one with them. The contract design and the regulatory posture were the same decision, and FinCEN's "only an email address" finding is what that decision looked like at the account-opening screen.

### 3. 2022 to 2025 — four pleas, a corporate conviction, and a pardon

The criminal case resolved slowly and then reversed. All four charged individuals pleaded guilty to Bank Secrecy Act violations across 2022, with Hayes, Delo and Reed each agreeing to \$10 million fines and Dwyer to \$150,000; sentences were probation, with six months' home detention for Hayes (SDNY releases, 24 February, 9 March, 20 May and 8 August 2022; DOJ Office of the Pardon Attorney clemency list for the Delo, Reed and Dwyer terms). The company pleaded guilty in July 2024 and was fined \$100 million in January 2025 (DOJ/SDNY releases 24-244 and 25-010). Two months later, on 27 March 2025, all four men *and the corporation itself* were pardoned (DOJ Office of the Pardon Attorney, accessed 29 July 2026).

The mechanism from this article is not directly implicated in any of that — this is compliance law, not market structure. But it belongs in a post about the venue-as-player for one reason: none of it changed the contract. Through indictment, settlement, four guilty pleas, a corporate conviction and a pardon, the perpetual swap kept trading, on BitMEX and on every venue that had copied it. Enforcement acted on the operators. The market structure they had shipped was already out of reach.

### 4. 23 July 2026 — a reduce-only window and a forced close

BitMEX announced its own closure, effective 23 September 2026 at 04:00 UTC, with position reductions only from 26 August at 04:00 UTC and immediate force-closure of anything still open at the deadline (BitMEX, 23 July 2026); 35 derivatives contracts were delisted on 30 July 2026 (BitMEX, 22 July 2026).

Watch the shape of it. A deadline is announced. A window opens in which you may only reduce. At the end, whatever remains is closed at market by someone other than you. That is the liquidation protocol from section 5, scaled up from one over-leveraged account to an entire exchange — and it is a useful reminder that on a derivatives venue, *your position exists at the pleasure of the venue's rulebook.* Custody risk gets discussed constantly in crypto. Rulebook risk barely gets mentioned, and it is the one that closes your trade.

### 5. The quiet case: the resting-rate drain

The least dramatic scenario is the one that costs the most people the most money, and it needs no crash at all.

A perpetual sitting exactly on the index still charges longs roughly 0.01% per interval — the interest term that survives when the premium is zero. That is 0.03% a day and about 11% a year (Binance funding methodology, accessed 29 July 2026). Hold a \$50,000 long against \$5,000 of margin for a year in a market that ends exactly where it started, and you have paid roughly \$5,475 in funding: **more than your entire initial margin**, for a position whose price never moved.

No liquidation cascade, no exchange failure, no bad call on direction. Just a small recurring payment applied to a large notional and funded from a small margin balance. The mechanism from section 2, running as designed, for 1,095 consecutive intervals.

## When this matters to you

If you never touch a perpetual swap, this machinery still reaches the price of the bitcoin you might own on spot, because in a fast hour the marginal seller can be a liquidation engine rather than an investor. Understanding why a large candle appeared on no news is worth something even to someone who only ever buys and holds.

If you do trade them, the defensible takeaways are mechanical rather than clever:

- **Size by distance to liquidation, not by the leverage number.** "25x" is marketing. "3.37% away" is the fact. Compute the distance, compare it to what this asset routinely does in an hour, and decide whether you are making a bet on direction or on the absence of ordinary noise.
- **Budget funding as a real cost.** Multiply the current rate by three, then by the number of days you expect to hold, then by your notional — and compare that to your *margin*, not your notional. If the answer is a large fraction of your margin, the trade needs to work quickly to be worth it.
- **Avoid putting your liquidation price at an obvious level.** Round numbers and standard leverage settings put thousands of accounts at the same price. You cannot control what the market does, but you can decline to stand exactly where everyone else is standing.
- **Know that being right is not sufficient.** Funding can drain a correct position and auto-deleveraging can close a winning one. Neither is a malfunction; both are in the rulebook.
- **Treat inverse contracts as a different instrument, not a different denomination.** If your collateral is the asset you are long, you are long it twice.
- **Read funding and open interest together, and expect nothing more than a map.** They tell you where the forced orders are stacked. They do not tell you when, and anyone who says otherwise is selling something.

This is an explanation of how a market works, not advice about what to do in it. Leveraged derivatives can lose more than the money you put in them, quickly, through mechanisms that do not require you to be wrong.

The perpetual swap is a genuinely elegant piece of financial engineering: it solved a real problem — how to give people continuous leveraged exposure without a calendar — with a mechanism that is simple, self-correcting, and requires no external capital. It is also, for exactly those reasons, a machine that converts crowding into forced selling faster than any market structure that came before it. Both things are true, and BitMEX closing does not change either one.

## Sources & further reading

**A note on sourcing.** BitMEX's own documentation pages for funding, fair-price marking, liquidation, the insurance fund and auto-deleveraging are client-rendered and returned no content when fetched on 29 July 2026. Rather than reconstruct those specifics from memory, the general perpetual-swap mechanism in this article is taught from **Binance's published documentation**, and BitMEX-specific claims are limited to what its **public instrument API** returns — including the 100x figure, which is derived as 1 ÷ the 1% initial margin rather than quoted from BitMEX. Where a figure could not be sourced — notably market-share statistics and the reported totals of specific historical liquidation cascades such as March 2020 — this article does not quote one. The order-book depths and liquidation clusters in worked example 5 are explicitly labelled as illustrative arithmetic, not measured data. Two things I deliberately left unstated because I could not source them: whether the individual fines were ultimately paid, and what effect the March 2025 pardons had on penalties already imposed.

**Primary sources behind the headline figures**

- [Important Message from BitMEX](https://www.bitmex.com/blog/bitmex-closure) — BitMEX, 23 July 2026. Closure effective 23 September 2026 at 04:00 UTC; reduce-only from 26 August 2026 at 04:00 UTC; remaining positions force-closed at the closure time.
- [Delisting of Illiquid Contracts — July 2026](https://www.bitmex.com/blog/delisting-jul2026) — BitMEX, 22 July 2026. 35 derivatives contracts delisted on 30 July 2026.
- BitMEX public instrument API (`/api/v1/instrument`, XBTUSD) — accessed 29 July 2026. Inverse contract, quoted in USD and settled in XBT; multiplier −100,000,000; initial margin 1%; maintenance margin 0.5%; eight-hour funding interval at 04:00, 12:00 and 20:00 UTC; mark method `FairPrice`. The "100x" figure used in this article is derived as 1 ÷ 1% initial margin.
- [Binance funding-rate methodology](https://www.binance.com/en/support/faq/detail/360033525031) — accessed 29 July 2026. The funding formula, the ±0.05% clamp, the 0.03% daily interest assumption, the eight-hour interval at 00:00/08:00/16:00 UTC, and the ±0.3% per-interval cap for BTCUSDT.
- [Binance liquidation-protocol documentation](https://www.binance.com/en/support/faq/detail/360033525271) — accessed 29 July 2026. The insurance fund bearing losses from a bankrupt position, and auto-deleveraging of opposing non-bankrupt positions when the fund is insufficient.
- [CFTC Release 8270-20](https://www.cftc.gov/PressRoom/PressReleases/8270-20) — U.S. Commodity Futures Trading Commission, 1 October 2020. The civil enforcement action, the entities and individuals named, the parallel criminal indictment (Case No. 20-CR-500, S.D.N.Y.), and the \$11 billion deposit and \$1 billion fee-revenue figures.
- [CFTC Release 8412-21](https://www.cftc.gov/PressRoom/PressReleases/8412-21) — U.S. Commodity Futures Trading Commission, 10 August 2021. The \$100 million civil monetary penalty and the \$50 million FinCEN offset.
- [DOJ/SDNY release 20-218](https://www.justice.gov/usao-sdny/pr/founders-and-executives-shore-cryptocurrency-derivatives-exchange-charged-violation) — 1 October 2020. The Bank Secrecy Act charges against Hayes, Delo, Reed and Dwyer, their roles, and the AML allegation.
- [FinCEN enforcement action against BitMEX](https://www.fincen.gov/news/news-releases/fincen-announces-100-million-enforcement-action-against-unregistered-futures) — Financial Crimes Enforcement Network, 10 August 2021. The \$100 million assessment and its \$80 million / \$20 million split; the \$209 million of darknet and unregistered-MSB transactions; the 588 unfiled SARs; the email-address-only account opening; the altered customer-location finding.
- [DOJ/SDNY: founders plead guilty](https://www.justice.gov/usao-sdny/pr/founders-cryptocurrency-exchange-plead-guilty-bank-secrecy-act-violations) — 24 February 2022 (Hayes and Delo); [third founder pleads guilty](https://www.justice.gov/usao-sdny/pr/third-founder-cryptocurrency-exchange-pleads-guilty-bank-secrecy-act-violations) — 9 March 2022 (Reed); [high-ranking employee pleads guilty](https://www.justice.gov/usao-sdny/pr/high-ranking-employee-cryptocurrency-exchange-pleads-guilty-bank-secrecy-act-violations) — 8 August 2022 (Dwyer). The pleas and the agreed fines.
- [DOJ/SDNY: Hayes sentenced](https://www.justice.gov/usao-sdny/pr/founder-and-ceo-shore-cryptocurrency-derivatives-platform-sentenced-violating-bank) — 20 May 2022. Six months' home detention and two years' probation.
- [DOJ/SDNY: BitMEX pleads guilty](https://www.justice.gov/usao-sdny/pr/global-cryptocurrency-exchange-bitmex-pleads-guilty-bank-secrecy-act-offense) — 10 July 2024 (release 24-244), and [BitMEX fined \$100 million](https://www.justice.gov/usao-sdny/pr/global-cryptocurrency-exchange-bitmex-fined-100-million-violating-bank-secrecy-act) — 15 January 2025 (release 25-010). The corporate guilty plea and sentence.
- [DOJ Office of the Pardon Attorney — clemency grants](https://www.justice.gov/pardon/clemency-grants-president-donald-j-trump-2025-present) — accessed 29 July 2026. The 27 March 2025 pardons of Hayes, Delo, Reed, Dwyer and HDR Global Trading Limited, and the sentence terms for Delo, Reed and Dwyer.
- [Crypto Trader Digest](https://cryptohayes.substack.com/) — Arthur Hayes' Substack, accessed 29 July 2026. Self-description as Chief Investment Officer of Maelstrom and co-founder and former CEO of BitMEX; more than 70,000 subscribers.

**Further reading on this blog**

- [How crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) — what a market order does to a thin book, which is the microstructure underneath every cascade above.
- [Inside an exchange: the matching engine and the order book](/blog/trading/capital-markets/inside-an-exchange-the-matching-engine-and-the-order-book) — the mechanics of the venue that the liquidation engine sends its orders to.
- [Exchanges are players, not just venues](/blog/trading/crypto-players/exchanges-are-players-not-just-venues) — the business model behind the rulebook.
- [What a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) — why the bids disappear at exactly the wrong moment.
- [Inventory risk, hedging and delta-neutrality](/blog/trading/crypto-players/inventory-risk-hedging-and-delta-neutrality) — how the professionals on the other side of your perp manage the exposure you just gave them.

