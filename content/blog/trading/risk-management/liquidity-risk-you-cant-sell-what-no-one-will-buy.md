---
title: "Liquidity Risk: You Can't Sell What No One Will Buy"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Build market liquidity from first principles — bid-ask spread, order-book depth, and market impact — and learn the brutal truth that the price on the screen is not the price you can get, especially when you most need to sell."
tags: ["risk-management", "liquidity-risk", "market-impact", "bid-ask-spread", "order-book", "slippage", "days-to-liquidate", "survival"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **One sentence:** The price on the screen is not the price you can get — it is the price of the last tiny trade, and selling real size walks that price against you, so your true risk is not what your position is marked at but what it would actually fetch when you are forced to sell.
> - **The screen price is a quote for one share, not for your whole position.** Selling size walks down the order book; the average price you receive is below the last trade, and the gap grows with how much you sell.
> - **Liquidity has three measurable dimensions: spread, depth, and market impact.** The spread is what you pay to cross instantly; depth is how much you can sell before the price moves; impact is how far size pushes the price against you.
> - **Market impact follows a square root, not a straight line.** Doubling your order does not double the cost, but trading a quarter of a day's volume can cost ten times the spread — big orders are punished disproportionately.
> - **Liquidity is abundant until you need it and evaporates in stress.** In calm markets the book is deep and the spread is tight; in a panic the bids vanish, the spread blows out, and the discount to sell explodes — exactly when everyone wants to sell.
> - **Your position's paper value and its liquidation value are different numbers.** Marking at the screen price flatters you; the survival question is what the book is worth when sold in size, and you must size to *that*.

There is a moment that every trader who has carried real size eventually lives through, and it teaches a lesson no spreadsheet ever quite does. The screen says your position is worth ten million dollars. You decide to sell. You hit the bid — and the price moves. You hit it again, and it moves more. By the time you are done, the ten million was a number on a screen, and what landed in your account was meaningfully less. Nobody lied to you. The screen was showing you the price of the *last trade*, which was probably a hundred shares. You tried to sell a hundred thousand. The market quietly informed you that those are not the same transaction, and it charged you the difference.

This is liquidity risk, and it is the quiet killer of the risk-management story. The survival thesis of this whole series is that your first job is not to make money — it is to [not blow up, because you can only compound if you're still in the game](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain). Most of this series has been about *how big a loss* you take. Liquidity risk is about something more insidious: whether you can *act on a loss at all*. A risk limit that says "cut the position if it falls 10%" is worthless if, on the day it falls 10%, there is no one on the other side to sell to. The exits you planned to use are only there in the weather you don't need them.

![Order book sliced through showing the screen price at one hundred dollars while selling fifty thousand shares walks down the bids to an average fill of ninety nine dollars and forty one cents](/imgs/blogs/liquidity-risk-you-cant-sell-what-no-one-will-buy-1.png)

Look at the figure above before reading on, because it is the whole post on one chart. The dashed line near the top is the **screen price** — \$100.00, the last trade. Below it, in red, is the **bid ladder**: the actual buyers, at actual prices, for actual sizes. The red step line traces what happens when you start selling: the first few thousand shares fill near \$100, but as you keep selling you walk *down* the ladder, taking worse and worse prices. Sell 50,000 shares and your **average** fill is the blue line at \$99.41 — fifty-nine cents a share below the screen, a total of \$29,455 of slippage you will never see quoted anywhere. The screen showed you \$100. The market gave you \$99.41. The difference between those two numbers, scaled up to your whole book, is liquidity risk.

This post builds market liquidity from absolute zero. We will define the bid-ask spread, order-book depth, and market impact precisely; we will derive why impact scales like a square root of size; we will distinguish a position's *paper value* from its *liquidation value*; and we will watch liquidity evaporate in two real crises. By the end you should never again look at a position's marked value and mistake it for money you can actually get. You will instead ask the only question that matters in a forced sale: *if I had to sell this, in this market, today — what would I really receive?*

## Foundations: how a market actually trades

Before we can talk about liquidity *risk*, we need to know what liquidity *is* — and that means understanding, from the ground up, how a price gets made. None of this assumes a finance background. A market is just a place where buyers and sellers meet, and a "price" is just the most recent point at which one buyer and one seller agreed. Everything in this post follows from taking that definition seriously.

### The order book: where prices actually live

Most modern markets — stocks, futures, crypto, currencies — run on a **limit order book**. The book is simply a list of standing offers. On one side are the **bids**: people willing to *buy*, each saying "I'll buy this many shares at this price or lower." On the other side are the **asks** (or **offers**): people willing to *sell*, each saying "I'll sell this many at this price or higher." The book is sorted by price: the **best bid** is the highest price any buyer will pay right now, and the **best ask** is the lowest price any seller will accept.

When you want to sell *immediately*, you sell into the bids — you "hit the bid." When you want to buy immediately, you buy from the asks — you "lift the offer." That's it. The "price" you see ticking on a screen is just the last trade that happened, somewhere between the best bid and the best ask. It is a fact about the past, not a promise about your future trade.

### The bid-ask spread: the toll for crossing

The **bid-ask spread** is the gap between the best bid and the best ask. If the best bid is \$99.95 and the best ask is \$100.05, the spread is \$0.10, and the **mid-price** — the average of the two — is \$100.00. The mid is the number people usually call "the price," but you can never actually trade at the mid. To buy *right now* you must pay the ask (\$100.05); to sell *right now* you must accept the bid (\$99.95). The spread is the toll the market charges for *immediacy* — for the privilege of transacting this instant rather than waiting and hoping.

That toll is real money. If you buy at \$100.05 and immediately change your mind and sell at \$99.95, you have lost \$0.10 a share with no market move at all — you paid the spread, twice the half-spread, just for a round trip. A tight spread (a penny on a \$100 stock, one basis point) means cheap immediacy. A wide spread (two dollars on a \$100 stock, two hundred basis points) means immediacy is expensive. The spread is the *first* dimension of liquidity, and the most visible one.

### Depth: how much you can sell before the price moves

The spread tells you the cost of selling *a little*. **Depth** tells you how much you can sell before the price starts moving against you. Depth is the total size resting in the book near the current price. A deep book has thousands of shares bid at \$99.95, thousands more at \$99.90, thousands more at \$99.85 — so you can sell a lot before you exhaust the top levels and start reaching for lower prices. A thin book has a few hundred shares at the best bid and then a yawning gap below it, so even a modest order blows straight through the top and fills at much worse prices.

Here's the key insight, and it is the one beginners miss: **the spread is what you pay on the first share; depth is what determines what you pay on the rest.** A stock can have a one-cent spread and *still* be dangerous to sell in size, because the depth behind that one-cent spread might be tiny. The quoted spread flatters thin markets — it describes the cost of a trade so small it barely matters.

### Market impact: the price moves because you moved it

When you sell more than the book can absorb at the current price, you **move the price** — and you move it *against yourself*. Selling pushes the price down (you consume bids and reach for lower ones); buying pushes it up. This is **market impact** (or **price impact**), the third and most important dimension of liquidity. It is the formalization of the cover figure: the more you sell, the lower the average price you receive, because your own selling is what drove it down.

Market impact is the part of liquidity risk that paper-trading and backtests almost always ignore — they assume you fill at the screen price, for any size, instantly. Real markets never work that way. The single most expensive lesson in trading is the difference between the price you *modeled* and the price you *got*, and that difference is market impact.

### Slippage and the all-in cost of a trade

Put the three together and you get **slippage**: the total difference between the price you expected (usually the mid or the screen price when you decided to trade) and the average price you actually achieved. Slippage is spread *plus* impact *plus* any drift in the market while you were trading. It is the all-in, real-world cost of converting a position into cash. A position's true risk is measured net of slippage — because slippage is the toll you pay to *act* on a risk decision, and a risk decision you can't afford to act on isn't risk management, it's hope.

#### Worked example: the round-trip toll on the \$100,000 account

Start with the recurring \$100,000 retail account. You buy \$20,000 of a stock trading at a \$100.00 mid with a \$0.10 spread (best bid \$99.95, best ask \$100.05). You pay the ask: \$100.05. That's 200 shares (\$20,000 / \$100.05 ≈ 199.9, call it 200 shares for \$20,010).

Now suppose nothing happens — the market doesn't move at all — and you immediately sell. You hit the bid: \$99.95. Your 200 shares fetch \$19,990. You are down \$20,010 − \$19,990 = **\$20** on a position that never moved a tick. That \$20 is the spread, paid twice (once on the way in, once on the way out): 200 shares × \$0.10 = \$20. As a fraction of the \$20,000 position it's 0.1%, and as a fraction of your \$100,000 account it's a rounding error — *for a small, liquid trade*.

But notice what we assumed: the depth was enough that 200 shares filled entirely at the top of the book, with zero impact. The whole cost was the spread. *For small orders in liquid names, the spread is the entire story — which is exactly why traders who only ever trade small wildly underestimate what it costs to trade big.*

![Stylised limit order book showing green bids on the left and red asks on the right with a ten cent bid ask spread in the middle and depth that thins out moving away from the mid price](/imgs/blogs/liquidity-risk-you-cant-sell-what-no-one-will-buy-2.png)

The figure above makes the three dimensions concrete on one picture. The green bars on the left are bids (where you can sell); the red bars on the right are asks (where you must buy). The amber band in the middle is the **spread** — \$0.10 here, the toll to cross instantly. The *length* of each bar is **depth** — how many shares rest at each price. Notice how the bars get longer as you move away from the mid: there's only a little size at the best bid (\$99.95), more a few cents below, more still further out. That shape is the whole problem. The liquidity nearest the current price — the liquidity you'd actually use to sell quickly — is the thinnest. To find real depth you have to reach down to prices well below the screen, and reaching down *is* market impact.

## Why the screen price is a lie for size

Let's now make the cover figure rigorous, because it contains the central claim of this entire post: **the price on the screen is the price of the last trade, and the last trade was almost certainly tiny.** A stock can "trade at \$100" all day while it would be impossible to sell a million dollars of it at anything close to \$100. The screen price is a quote for the marginal share — the next single unit to trade — not an offer to transact your whole position at that level.

Here is the mechanism, step by step. When you sell into the book, you fill against the bids in order, best price first. You take all the shares at \$99.95, then all the shares at \$99.90, then \$99.80, and so on — walking *down* the ladder. Each level you consume is a worse price than the last. Your **average** fill price is the size-weighted average of every level you touched, and it is necessarily below the best bid you started at, because you took the best bid *plus* a bunch of worse prices below it. The bigger your order relative to the depth, the deeper you have to reach, and the further your average fill drops below the screen.

#### Worked example: walking down the book on a 50,000-share sell

This is the cover figure, in numbers, on a position big enough to matter. The screen says \$100.00. The bid ladder is:

- 1,500 shares at \$99.98
- 2,500 shares at \$99.95
- 4,000 shares at \$99.90
- 6,000 shares at \$99.80
- 9,000 shares at \$99.60
- 14,000 shares at \$99.30
- 22,000 shares at \$98.90

You need to sell 50,000 shares. Walk it down:

- 1,500 × \$99.98 = \$149,970
- 2,500 × \$99.95 = \$249,875
- 4,000 × \$99.90 = \$399,600
- 6,000 × \$99.80 = \$598,800
- 9,000 × \$99.60 = \$896,400
- 14,000 × \$99.30 = \$1,390,200
- and the last 13,000 shares (to reach 50,000) at \$98.90 = \$1,285,700

Total proceeds: \$149,970 + \$249,875 + \$399,600 + \$598,800 + \$896,400 + \$1,390,200 + \$1,285,700 = **\$4,970,545**. You sold 50,000 shares, so your **average fill price is \$4,970,545 / 50,000 = \$99.41**.

The screen said \$100.00. If you had (naively) marked this sale at the screen price, you'd have expected \$5,000,000. You received \$4,970,545. The difference — **\$29,455** — is the slippage from market impact, and it is \$0.59 per share, almost six times the \$0.10 quoted spread. *The spread told you the trade would cost ten cents a share; selling size cost you six times that, because the quoted spread describes a trade you weren't doing.*

That last point is the one to carry: the *quoted* spread (\$0.10) and the *realized* cost of selling size (\$0.59) differ by a factor of six here, and in a thin name or a stressed market the factor can be ten or a hundred. The screen price systematically overstates what you can get, and it overstates it more the more you need to sell.

## The square-root law: why big orders are punished disproportionately

How fast does the cost grow with size? You might guess linearly — sell twice as much, pay twice the impact. The empirical answer, documented across decades of trading data and across asset classes, is gentler at first and crueler later: **market impact grows roughly like the square root of size.** The cost of trading a quantity Q, measured as a fraction of the price, is approximately

> impact ≈ Y × σ × √(Q / ADV)

where σ is the asset's volatility (how much it moves on a normal day), ADV is its average daily volume (how much trades in a typical day), Q/ADV is your order as a fraction of that daily volume (your **participation rate**), and Y is a dimensionless constant of order one. You don't need to memorize the constant; you need the *shape*. Impact scales with the square root of how much of the day's volume you're trying to consume, scaled by how volatile the asset is.

The square root has two faces, and both matter for survival. The gentle face: doubling your order does *not* double your cost — it multiplies it by √2 ≈ 1.41, so there are real economies to trading patiently in moderate size. The cruel face: the curve is *steepest at the start*, so the first slice of participation is the most expensive per share, and pushing your participation up toward a large fraction of the day's volume drags the average price brutally. Trading 1% of the day's volume is nearly free; trading 25% of it is a different universe.

![Market impact curve showing one sided price impact rising with order size as a fraction of average daily volume following a square root shape with one percent costing point one six percent and twenty five percent costing point eight percent](/imgs/blogs/liquidity-risk-you-cant-sell-what-no-one-will-buy-3.png)

The figure above plots that curve. The horizontal axis is your order as a percent of the day's volume; the vertical axis is the price impact as a percent of the price. The red curve is the square-root law; the gray dashed line is the linear guess people make by instinct — and you can see it's the wrong shape, overstating small orders and badly *understating* the cost of large ones. With a 2% daily volatility and Y = 0.8, trading **1% of ADV costs about 0.16%**, **5% of ADV costs about 0.36%**, **10% costs about 0.51%**, and **25% costs about 0.80%** of the price. The dots mark those points. The lesson is in the spacing: going from 1% to 25% of the volume — a 25× bigger order — only raised the cost about 5×, but in absolute terms 0.80% on a large position is enormous, and it's a cost you pay on *every share*, not just the last one.

#### Worked example: market impact on the \$10,000,000 book

Now bring in the recurring \$10,000,000 book. Suppose this whole book is one position in a stock that trades \$5,000,000 of volume per day — so your position is **twice the entire daily volume** (Q/ADV = 2.0 if you tried to sell it all in one day). You obviously can't, but it shows why: at 2× ADV the square-root law predicts impact of 0.8 × 0.02 × √2 ≈ 2.3% just from the participation, and that understates it because the formula isn't meant for orders that swamp the market.

So you trade it down sensibly, at 20% of ADV per day to keep impact tolerable. At 20% participation, the per-day impact is 0.8 × 0.02 × √0.20 ≈ **0.72%** of the price. Selling \$1,000,000 a day (20% of \$5,000,000) at a 0.72% impact costs about \$7,200 of slippage *that day*. To clear the whole \$10,000,000 position takes ten such days, and the slippage compounds: very roughly, ten days at \$7,200 of impact is on the order of **\$72,000** of total market-impact cost to exit — about 0.72% of the position, evaporated into the act of selling. And that's the *calm* number. *The cost of getting out is not a footnote; on a position that is large relative to its market, it is a meaningful fraction of the position itself — and you only discover it the day you try to leave.*

### Temporary versus permanent impact

There's a subtlety inside market impact worth pulling apart, because it changes how you think about a sale. When you sell aggressively, part of the price move is **temporary** — you've momentarily overwhelmed the available buyers, the price dips, and once you stop selling and liquidity replenishes it partly recovers. Another part is **permanent** — your selling carried *information* (or was assumed to), and the market repriced down for good. The temporary component is the cost of demanding immediacy; it rewards patience, because if you slow down it shrinks. The permanent component is the cost of *what your trade revealed*; it doesn't care how slowly you trade.

This distinction is why a forced seller pays double. A patient, discretionary seller can lean against the temporary component — trade slowly, let the book refill, capture the bounce — and pay mostly the permanent piece. A forced seller, dumping size into a falling market on a deadline, eats the *full* temporary impact on top of the permanent one, because they cannot wait for liquidity to come back. The same position, the same shares, costs far more to liquidate under compulsion than under choice. That gap — the premium for being forced — is precisely the cost that materializes in a margin call or a redemption, and it is invisible in any backtest that assumes you always sell at the mid.

### Why volume is not the same as depth

A common and dangerous shortcut is to read a stock's daily *volume* and conclude it's liquid. Volume is how much traded over a day; depth is how much you can trade *right now* without moving the price. They are related but not the same, and the gap between them is where traders get hurt. A stock can print millions of shares of volume across a day — in thousands of small, spread-out trades — while having only a few thousand shares resting in the book at any instant. The volume is real, but it arrived in a trickle; if you need to sell a day's worth of volume in an hour, the depth that's actually *there* in that hour is a small fraction of the day's total, and you'll pay for the rest in impact.

This is why the participation-rate framing matters so much more than raw volume. "This stock trades \$50,000,000 a day, my \$5,000,000 position is only a tenth of that" sounds safe — but a tenth of the *day's* volume might be the *entire* book's worth of depth at any given moment, so trying to sell it quickly means consuming hours of future liquidity all at once. Daily volume is the river's total flow; depth is how much you can scoop in one pass. You drown by mistaking the first for the second.

## Liquidity is a profile, not a number

Every asset has its own liquidity, and the differences are enormous. The same \$5,000,000 position is a non-event in a mega-cap stock and a multi-week ordeal in a micro-cap. Liquidity isn't one number you can put on an asset; it's a *profile* across the three dimensions — spread, depth, and time-to-exit — and the profile is what determines whether a position is a comfortable holding or a trap.

A **large-cap** (a megacap stock, a major currency pair, a front-month equity-index future) has a razor-thin spread — often a basis point or less — and a deep book, so a few million dollars trades with negligible impact and exits in hours. A **small-cap** has a spread tens of times wider, a book that thins out fast, and a position of the same dollar size that takes many days to unwind. A **truly illiquid** asset — a micro-cap stock, a private holding, an off-the-run bond, a thinly traded token — can have a spread of hundreds of basis points, almost no resting depth, and a time-to-exit measured in weeks or months, during which the very act of selling craters the price.

![Two panel comparison showing bid ask spread on a log scale and days to liquidate a five million dollar position for large cap small cap and truly illiquid assets with spreads from one point five to two hundred fifty basis points and exits from hours to forty days](/imgs/blogs/liquidity-risk-you-cant-sell-what-no-one-will-buy-4.png)

The figure above shows the profile for three stylized buckets. On the left, the bid-ask spread — on a *log* scale, because the range is so vast: roughly **1.5 basis points** for a large-cap, **35** for a small-cap, **250** for a truly illiquid name. On the right, the days to liquidate the same \$5,000,000 position at a disciplined 20% of daily volume: **under an hour** for the large-cap, **about 6 days** for the small-cap, **40 days** for the illiquid name. Same dollars at risk. Wildly different exits. The position that looks identical in your account — "\$5,000,000 long" — is a shrug in one bucket and a potential career-ender in another, and the only difference is the liquidity profile of what you happen to own.

This is why sophisticated risk frameworks never treat all dollars as equal. A dollar in a megacap and a dollar in a micro-cap are not the same risk, because they are not the same *exit*. The naive portfolio view — add up the marked values, that's your exposure — is blind to the dimension that actually determines whether you survive a forced sale. As [the concentration post argues](/blog/trading/risk-management/concentration-and-position-limits-the-one-trade-that-can-end-you), the distribution of risk across positions matters more than the count of them; liquidity adds a second axis — the distribution of *exit difficulty* — that an honest risk picture must include.

#### Worked example: two \$100,000 positions, two different exits

You hold two positions in your \$100,000 account, each \$50,000. Position A is a megacap with \$2,000,000,000 of daily volume; position B is a micro-cap with \$200,000 of daily volume. On paper they look identical: \$50,000 each, 50% of your account each.

Position A: \$50,000 is 0.0025% of the daily volume. You could sell the whole thing in seconds at essentially the spread — call it a few dollars of cost. Effectively instant, effectively free.

Position B: \$50,000 is **25% of the entire daily volume**. To exit without crushing the price you'd trade at, say, 10% of volume a day — \$20,000 a day — taking **2.5 trading days**, and even then the square-root impact at 10% participation is real, perhaps 0.5%–1% of the position, plus whatever the market does over those 2.5 days. If bad news hits position B, you cannot get out before the damage is done; your "stop loss" is a wish, because there's no one to sell to at your stop.

The two positions are the same size and the same weight, but they are *not* the same risk. *Position A is money; position B is a promise of money that the market can revoke the moment you need it — and a risk system that marks them identically is lying to you about your real exposure.*

## In calm it's deep; in stress it's gone

Now the most important property of liquidity, the one that turns a manageable cost into a survival threat: **liquidity is not constant — it is most abundant exactly when you don't need it, and it vanishes exactly when you do.** In calm markets, books are deep, spreads are tight, and market-makers compete to provide liquidity, so exiting is cheap and easy. In a stress event, the same book becomes thin, the spread blows out, the market-makers step back to protect themselves, and the discount to sell explodes — at the precise moment when everyone *else* also wants to sell. Liquidity is pro-cyclical: it shows up in good times and disappears in bad times, which is the worst possible timing for a risk.

Why does this happen? Several reinforcing mechanisms. **Market-makers widen or pull quotes** when volatility spikes, because the risk of holding inventory just jumped — they don't want to catch a falling knife, so they step away. **Resting bids get cancelled** as buyers wait for lower prices, hollowing out the depth below. **Forced sellers all arrive at once** — margin calls, [funding spirals](/blog/trading/risk-management/funding-and-margin-spirals-when-your-lender-becomes-your-risk), fund redemptions — so the demand for immediacy surges exactly as the supply of it collapses. And the whole thing is reflexive: falling prices trigger forced selling, forced selling drives prices lower, lower prices trigger more forced selling. This is the [fire-sale cascade](/blog/trading/risk-management/fire-sales-and-deleveraging-cascades-everyone-for-the-exit-at-once), and liquidity evaporation is its engine.

![Before and after comparison of liquidity in stress versus calm markets showing wide spreads vanishing depth simultaneous selling and large liquidation discounts on the left against tight spreads deep books two way flow and near zero discount on the right](/imgs/blogs/liquidity-risk-you-cant-sell-what-no-one-will-buy-5.png)

The figure above contrasts the two regimes. On the left — **the day you must sell** — the spread blows out from 10 basis points to 200-plus (crossing costs twenty times more), the depth evaporates as bids are pulled, everyone sells at once because each forced seller's exit is the next one's margin call, and the liquidation discount runs 5–20%. On the right — **the days you don't need to sell** — the spread is tight, the book is deep with market-makers competing, two-way flow absorbs your size, and the discount to sell is near zero. The cruel asymmetry is that you make your plans and set your stops in the *right*-hand world and then have to execute them in the *left*-hand one. The liquidity you counted on when you sized the position is not the liquidity that's there when you need to exit it.

This is why "I'll just cut the position if it drops" is a dangerous plan. The drop and the liquidity drought are the *same event*. The 10% loss that triggers your stop is usually accompanied by the spread widening and the depth disappearing — so the very move that tells you to sell is the move that makes selling expensive. Your modeled exit at \$99.95 becomes a real exit at \$92, because in the panic the bids at \$99.95 simply weren't there. As [the cross-asset treatment puts it](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis), in a crisis the diversification you counted on vanishes; the liquidity version of the same failure is that the *exit* you counted on vanishes.

#### Worked example: the stop loss that didn't stop the loss

You hold a \$30,000 position (30% of the \$100,000 account) in a stock at \$100, with a stop-loss order to sell at \$95 — you've decided you'll accept a \$1,500 loss (300 shares × \$5) and no more. In calm markets this would work: the stock drifts to \$95, your stop fires, you sell into a deep book within pennies of \$95, and you're out for roughly the loss you planned.

Now run it in stress. Bad news hits overnight and the stock *gaps* — it doesn't trade through \$96, \$95.50, \$95 on the way down; it simply reopens at \$88 with the bids hollowed out. Your stop at \$95 becomes a market order to sell *at the open*, and the best bid is \$88 for a few hundred shares, \$86 below that, \$84 below that. Selling your 300 shares walks you down to an average of, say, \$86. Your "stop at \$95" filled at \$86 — a loss of 300 × \$14 = **\$4,200**, not the \$1,500 you authorized. The stop did exactly what stops do: it converted to a market order and sold at the best available price. The problem was that the best available price, in the liquidity drought that *caused* the gap, was nowhere near \$95. *A stop is a promise to sell, never a promise of a price — and the event that triggers it is usually the event that has already taken your price away.*

## Time-to-liquidate: the limit that actually protects you

If impact grows with how fast you sell, then the natural defense is to sell *slowly* — spread the order over days, keeping your participation low enough that impact stays tolerable. But that introduces a different risk: **time**. While you're patiently selling over many days, the market is moving, and if it's moving against you, your slow, low-impact exit is being overtaken by the price decline you were trying to escape. So liquidity risk has a fundamental tension: sell fast and pay impact, or sell slow and bear market risk. The honest measure of a position's liquidity is therefore **time-to-liquidate**: how many days it takes to exit at a participation rate low enough to keep impact acceptable.

The arithmetic is simple. If you cap your participation at a fraction *p* of average daily volume to keep impact tolerable, then the days to exit a position are just

> days-to-liquidate = position size / (p × ADV)

It scales *linearly* with position size (twice the position, twice the days) and *inversely* with ADV (half the volume, twice the days). And here's the trap: in a stress event, ADV doesn't stay constant — *it collapses*. When depth evaporates, the volume you can trade without crushing the price falls, sometimes to a fraction of normal. So the days-to-liquidate you computed in calm markets is an underestimate of what you'll actually face in the crisis when you most need a fast exit.

![Time to liquidate versus position size showing days to exit growing linearly with size and the stressed market line rising three times faster than the calm line with a ten million dollar book taking ten days calm but thirty three days in stress](/imgs/blogs/liquidity-risk-you-cant-sell-what-no-one-will-buy-6.png)

The figure above shows both lines. The green line is the calm market (ADV \$5,000,000/day); the red line is the stressed market where ADV has collapsed to \$1,500,000/day. The shaded gap between them is the extra time the stress imposes. A **\$10,000,000 book takes 10 days to exit in calm markets** at 20% participation — and **33 days in stress**, more than three times as long. A 5-day-to-liquidate limit (the dotted line) is satisfied by positions up to \$5,000,000 in calm markets, but only \$1,500,000 once the volume dries up. The limit you set on a good day is too loose for the day you'll actually need it.

#### Worked example: a 5-day liquidity limit on the \$10,000,000 book

Suppose you adopt a hard rule: *no position may take more than 5 trading days to fully liquidate at 20% of average daily volume.* This is a real, common institutional discipline. What does it permit?

The most you may hold is position ≤ 5 days × 20% × ADV = **1.0 × ADV**. So your maximum position in any name is one full day's average volume. In a stock with \$5,000,000 ADV, your cap is \$5,000,000 — half your \$10,000,000 book. In a stock with \$1,000,000 ADV, your cap is \$1,000,000 — 10% of the book. In a micro-cap with \$100,000 ADV, your cap is \$100,000 — 1% of the book.

Now stress-test it. Assume in a crisis the ADV of every name falls to 30% of normal. Your \$5,000,000 position in the \$5,000,000-ADV stock now faces an ADV of \$1,500,000, so at 20% participation it takes \$5,000,000 / (0.20 × \$1,500,000) = **16.7 days** to exit — more than three times your 5-day limit, in the exact scenario where you most need to be out in 5 days. *A liquidity limit calibrated to calm-market volume is not a limit at all in a crisis; the only honest limit is one that still holds when you cut assumed volume to a third.*

## Paper value versus liquidation value

We can now state the deepest idea in this post precisely. A position has two values, and confusing them is how liquidity risk kills you.

**Paper value** (or marked value, or mark-to-market value) is the position's size times the screen price — what your account statement and your risk system say it's worth. **Liquidation value** is what the position would actually fetch if you sold it, in size, in the current market — size times the *average realized price* after walking down the book and paying the impact. Paper value is always at least as large as liquidation value, and the gap — the **liquidation discount** — widens with the size of the position relative to the market and explodes in stress. Your account marks you at paper value. The market pays you liquidation value. The difference is the liquidity risk hiding inside every mark.

![Paper value versus liquidation value showing a position marked at screen price as a straight blue line while the calm market liquidation value sits slightly below it and the stressed market value sits well below with a ten million dollar paper position realising nine point eight five million calm and nine point five five million in stress](/imgs/blogs/liquidity-risk-you-cant-sell-what-no-one-will-buy-7.png)

The figure above plots all three. The blue line is **paper value** — a position marked at the screen price, where \$10,000,000 of position is "worth" \$10,000,000. The green line is the **liquidation value in calm markets**, just below paper. The red line is the **liquidation value in stress**, well below. The gap between blue and red is the liquidation discount, and it widens as positions get larger. For our \$10,000,000 book, the paper value is \$10,000,000, the calm liquidation value is about **\$9.85M** (a 1.5% discount), and the stressed liquidation value is about **\$9.55M** (a 4.5% discount). And these are tame numbers for a position at 2× ADV; for a genuinely illiquid holding sold into a panic, the discount can be 20%, 40%, or — for an asset that simply has no buyer that day — effectively total.

The practical consequence is that *you must size to liquidation value, not paper value.* If your risk system says "this \$10,000,000 position is fine, it's only 10% of the book," but its liquidation value in a stress is \$8,000,000, then the position is not what your system thinks it is — and the day you need that \$10,000,000, you have \$8,000,000. The gap is unbudgeted, unhedged loss that materializes precisely when every other loss is materializing too. The mark is a comfortable fiction; the liquidation value is the truth, and survival means planning around the truth.

#### Worked example: the margin gap between paper and liquidation value

You run the \$10,000,000 book and a lender has financed part of it. They mark your collateral at **paper value** — \$10,000,000 — and lend against it. You feel safe: plenty of cushion. Then a stress hits. The position is now worth \$9,000,000 on paper (the price fell 10%) — but its *liquidation* value, after the discount to actually sell it into the thin, panicked market, is only \$8,200,000.

The lender, watching the same screen you are, issues a margin call based on the falling paper value. You must sell to meet it. But when you sell, you don't get the \$9,000,000 paper value — you get the \$8,200,000 liquidation value, an \$800,000 hole that appears *only at the moment of sale*. To raise the cash the margin call demands, you have to sell *even more*, which pushes the price down *further*, which deepens the discount, which triggers a *bigger* margin call. The gap between paper and liquidation value is the seed of the [funding spiral](/blog/trading/risk-management/funding-and-margin-spirals-when-your-lender-becomes-your-risk): a position that was "10% of the book" on paper becomes a forced, accelerating, self-deepening sale in liquidation. *The day the mark and the realizable value diverge is the day liquidity risk stops being a number on a screen and starts being the thing that ends you.*

## Common misconceptions

**"The price on the screen is the price I can get."** No — it is the price of the last trade, which was almost certainly tiny. Our worked example sold 50,000 shares for an average of **\$99.41** against a \$100.00 screen; the screen overstated the realizable price by \$0.59 a share, \$29,455 in total. The screen price is a quote for the marginal share, not an offer for your position.

**"A tight spread means the stock is liquid."** Not for size. The spread is the cost on the *first* share; depth and impact determine the cost on the rest. A penny-spread stock with thin depth can cost you six times the spread — \$0.59 versus \$0.10 in our example — once you sell real size. The quoted spread describes a trade so small it doesn't matter.

**"Impact scales with size, so a position twice as big costs twice as much to exit."** Wrong shape — impact follows a *square root*, so twice the size costs about 1.41× the impact per share, but a large *participation rate* is punished brutally: 1% of ADV cost 0.16% in our example, while 25% of ADV cost 0.80%. The curve is gentle for small orders and cruel for large ones; the linear instinct understates the cost of trading big.

**"I have a stop loss, so my downside is capped."** Only if there's a buyer at your stop. The 10% drop that triggers your stop is usually the same event that widens the spread and evaporates the depth — so your modeled exit at \$99.95 becomes a real exit far lower. A stop is a *plan to sell*, not a *guarantee of a price*; in the stress that triggers it, the price you assumed may simply not exist.

**"My position is worth what my account says it's worth."** Your account shows *paper* value (size × screen price). What you can actually get is *liquidation* value (size × realized price after impact). For our \$10,000,000 book those were \$10,000,000 on paper but ~\$9.55M in stress — and far less for genuinely illiquid holdings. The mark is a number; the liquidation value is money, and only one of them survives contact with a forced sale.

**"Liquidity is a property of the asset, so I can look it up once."** Liquidity is a property of the asset *and the moment*. It is deep and cheap in calm markets and thin and expensive in stress — and the stress is exactly when you need it. The same book that absorbed your size effortlessly in June can refuse a tenth of it in a March panic. You can't look it up once; you have to assume it will be worst when you need it most.

## How it shows up in real markets

Liquidity evaporation is not a theoretical footnote — it is the common mechanism underneath the most famous blow-ups, where the loss came not from being wrong but from being *unable to get out*.

**The 2008 financial crisis.** The Global Financial Crisis was, at its core, a liquidity crisis wearing a credit crisis's clothes. Mortgage-backed securities and the complex structures built on them had traded actively in 2006 with quoted prices and willing dealers; by late 2008 the *same* securities had no bid at all — not a low bid, *no* bid, because no one would name a price for assets whose value had become unknowable. Holders who marked those bonds at model prices discovered that the liquidation value was a fraction of the mark, or zero, the moment they tried to sell. Funding markets froze in parallel: banks stopped lending to each other, the repo market seized, and leveraged holders faced margin calls they could only meet by selling into a market with no buyers. The VIX peaked at a close of **80.86** on November 20, 2008. What killed Bear Stearns and Lehman was not, in the first instance, that their assets were worthless — it was that they could not *fund* or *sell* those assets fast enough to meet obligations coming due. The paper value on their books and the liquidation value in the frozen market had diverged past the point of survival.

**Long-Term Capital Management (Aug–Sep 1998).** LTCM ran convergence trades at roughly **25-to-1 balance-sheet leverage** on about \$4.7 billion of equity, with around **\$1.25 trillion of gross derivatives notional**. When Russia defaulted and capital fled to quality, the trades all moved against them at once — but the deeper killer was that LTCM's positions were *so large relative to the markets they traded* that there was no way to exit without crushing the very prices they needed. Their paper value and their liquidation value diverged catastrophically: every attempt to reduce risk moved the market against them and deepened the loss. The fund lost about **\$4.6 billion** in four months and required a Fed-organized **\$3.6 billion** recapitalization. The lesson, as [the LTCM case study details](/blog/trading/game-theory/case-study-ltcm-1998-the-crowded-genius-trade), is that diversification *and* liquidity failed together: when you are the market, there is no one to sell to.

**The COVID dash-for-cash (Feb–Mar 2020).** In the fastest bear market on record, the S&P 500 fell about **34%** from its February 19 peak to its March 23 trough, and the VIX closed at a record **82.69** on March 16. But the defining feature of March 2020 was a liquidity event, not just a price event: a "dash for cash" in which *everything* was sold simultaneously — even US Treasuries, the world's deepest market, saw bid-ask spreads blow out and depth vanish as everyone scrambled for dollars at once. When the safest, most liquid asset on earth becomes hard to sell at a fair price, you are watching liquidity evaporation in its purest form. Books that were deep in January refused size in March; the spread on instruments that normally trade in fractions of a basis point widened by multiples; and forced sellers (margin calls, fund redemptions, [funding spirals](/blog/trading/risk-management/funding-and-margin-spirals-when-your-lender-becomes-your-risk)) all arrived together, overwhelming the supply of immediacy until central banks stepped in to *be* the buyer. The whole episode is the textbook case of the right-hand panel of our figure becoming the left-hand one in a matter of days.

**Archegos (Mar 2021).** Archegos held enormous, concentrated single-stock positions through total-return swaps at roughly **5× leverage**, financed by multiple prime brokers who each saw only their own slice and were blind to the total. When a few of its holdings fell, the margin calls came — and the unwinding revealed the liquidity trap underneath: the positions were so large relative to the float of the underlying stocks that selling them crashed the prices, which deepened the losses, which triggered more selling. The banks raced to liquidate before each other, dumping concentrated blocks into a market that could not absorb them. Aggregate prime-broker losses exceeded **\$10 billion**, with Credit Suisse alone losing about **\$5.5 billion**. The position size *was* the liquidity risk: a holding too big for its market is a holding whose paper value and liquidation value can never be the same, and the gap is realized all at once in the forced unwind. It is the [concentration failure mode](/blog/trading/risk-management/concentration-and-position-limits-the-one-trade-that-can-end-you) and the liquidity failure mode fused into a single event.

The thread through all four is identical and it is the thread of this whole post: the loss was not primarily about the *direction* of the bet — it was about the *exit*. LTCM's trades might even have converged eventually; March 2020's selloff reversed within weeks; many of the 2008 assets eventually paid off near par. But survival is not decided by where prices are in a year. It is decided by whether you can meet your obligations *during* the drawdown, and that depends entirely on whether you can convert positions to cash without the market revoking the price. As [the crisis playbook from the hedge-fund seat puts it](/blog/trading/hedge-funds/the-crisis-playbook-2008-and-2020), the firms that survive a crisis are the ones that planned their liquidity *before* they needed it — because in the crisis itself, liquidity is a thing you have or don't, not a thing you can buy.

## The liquidity risk playbook

Liquidity risk is the gap between what your position is marked at and what you can actually get for it. The discipline is to close that gap *before* you're forced to discover it. Concrete rules:

- **Size to liquidation value, not paper value.** Before you put on a position, estimate what it would fetch sold in size into a *stressed* market — assume the spread widens, depth thins, and ADV falls to a third of normal. If the stressed liquidation value of the position would breach your loss budget, the position is too big, regardless of what the mark says. Mark-to-market flatters you; mark-to-*liquidation* keeps you alive.

- **Impose a days-to-liquidate limit, and stress the volume.** Cap every position so it can be fully exited within a fixed number of days (5 is a common choice) at a conservative participation rate (20% of ADV or less). Then re-run the limit with ADV cut to 30% of normal — the calm-market limit is too loose for the crisis. The most you should hold of any name is roughly **one day's normal volume** for a 5-day limit, and less in anything that gates on the way down.

- **Budget your participation rate, not just your size.** Impact follows a square root, so the cost is governed by *what fraction of the day's volume you consume*. Keep planned participation low (single-digit to low-double-digit percent) so you stay on the gentle part of the curve, and assume that in a stress your tolerable participation falls because the impact of any given rate rises.

- **Hold a liquidity buffer — cash that doesn't depend on selling.** Keep a reserve of genuinely liquid assets (cash, T-bills, the deepest instruments you trade) sized to cover your worst plausible cash demand — margin calls, redemptions, drawdown actions — *without* having to sell your illiquid positions into a thin market. The buffer is what lets you meet obligations during a crisis instead of being forced into the fire sale. Like every survival rule in this series, it costs you a little return in calm times and saves your existence in the one stress that matters.

- **Treat liquidity as a second risk axis, separate from price.** Two positions of equal dollar size and equal volatility are not equal risk if one exits in an hour and the other in 40 days. Track the liquidity profile (spread, depth, days-to-liquidate) of every position alongside its price risk, and never let your book's *exit-weighted* exposure pile up in the illiquid bucket. The naive view that adds up marks is blind to the dimension that decides whether you survive a forced sale.

The survival spine of this series says you can only compound if you stay in the game. Liquidity risk is the rule that decides whether you get to *act* on every other risk decision — whether your stops, your limits, and your hedges are real or imaginary. You cannot sell what no one will buy, and the day you most need to sell is the day fewest will buy. The trader who plans for that day — who sizes to liquidation value, limits days-to-liquidate, budgets participation, and holds a buffer of true cash — is the one still trading tomorrow, when the forced sellers who marked themselves at paper value have already been carried out.

### Further reading

- [Funding and margin spirals: when your lender becomes your risk](/blog/trading/risk-management/funding-and-margin-spirals-when-your-lender-becomes-your-risk) — the funding side of liquidity: how a falling mark triggers a margin call that forces the selling that lowers the mark.
- [Fire sales and deleveraging cascades: everyone for the exit at once](/blog/trading/risk-management/fire-sales-and-deleveraging-cascades-everyone-for-the-exit-at-once) — what happens when every leveraged holder of a crowded trade has to sell into the same evaporating liquidity.
- [Concentration and position limits: the one trade that can end you](/blog/trading/risk-management/concentration-and-position-limits-the-one-trade-that-can-end-you) — sizing to a maximum-loss budget, the discipline that liquidity limits extend to the exit.
- [Crowded trades and the exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game) — the strategic side: when everyone is in the same position, the exit is a game you can lose by being last.
- [The crisis playbook: 2008 and 2020](/blog/trading/hedge-funds/the-crisis-playbook-2008-and-2020) — how funds that survived planned their liquidity before they needed it, from the GP seat.
