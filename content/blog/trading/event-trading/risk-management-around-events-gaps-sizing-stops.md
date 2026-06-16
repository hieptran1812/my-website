---
title: "Risk Management Around Events: Gaps, Sizing, and Stops"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Event days are where accounts blow up: gaps jump through stops, position sizes set for a calm tape are too big for a 5-sigma print, and leverage gets liquidated. Here is how to size to the expected move, treat stops as the market orders they really are, use defined-risk options, control leverage, and decide when not to trade at all."
tags: ["event-trading", "macro", "risk-management", "position-sizing", "stop-loss", "gap-risk", "slippage", "leverage", "liquidation", "options", "crypto", "trading"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Event days do not blow up accounts because traders pick the wrong direction; they blow up accounts because the position was **sized for a calm day**, the stop was treated as a **guaranteed exit**, and **leverage** turned a routine gap into a forced liquidation. The fix is mechanical, not magical.
>
> - **What goes wrong:** a position sized to a normal-day swing is several times too large for a 5-sigma print; when price *gaps*, the stop becomes a market order that fills far past its level; and on leverage that gap triggers a margin call you cannot escape.
> - **The cross-asset reality:** events produce moves a normal stop never expects — the S&P **−4.32%** on hot Aug-2022 CPI, the Nikkei **−12.40%** and Bitcoin **−15%** in the Aug-2024 carry cascade. A stop set for a 1% day is useless against any of those.
> - **The trade:** size to the *expected move* (not the calm-day range), prefer *defined-risk* structures (a long option caps the loss at the premium no matter how big the gap), keep leverage low enough that a gap cannot liquidate you, and — the strongest edge of all — *do not trade* events where the randomness dwarfs your edge.
> - **The one number to remember:** your stop guarantees an *attempt*, not a *price*. On a gap it can fill 1% or more beyond the level — turning a planned −\$300 risk into a −\$900 loss.

A trader I know had done everything the textbooks tell you to do. He was long a major index ETF the morning of a US jobs report. He had a thesis, he had a level, and most importantly he had a **stop-loss**: a resting order to sell him out if the market fell against him, set so that the worst he could lose on the idea was about **\$300**. He had read every blog that says "always use a stop," and he had. He felt protected. He went to make coffee.

The 8:30 a.m. payrolls number came in three standard deviations away from consensus — a genuine surprise, the kind that happens a few times a year. The ETF dropped hard in the first second, blew straight through his stop level, and his stop did exactly what a stop is built to do: at his level it converted into a *market* order. That market order arrived into a book that, for those two seconds, was nearly empty. It walked down the price ladder taking whatever thin bids it could find, and it filled **1.5% below** the level he had set. His planned −\$300 loss came back as **−\$900**. He stared at the fill convinced it was a broker error. It was not. Nothing malfunctioned. The market did precisely what markets do around a surprise print, and his risk plan — sized for a calm day, anchored on a stop he believed was a guaranteed exit — was built for a world that does not exist on event days.

This post is about that world. It is the risk-management companion to the rest of this series: not *how* the market reacts to a number (we cover that elsewhere) but how to keep a reaction from ending your trading. The uncomfortable truth is that **event days are where accounts die**, and they almost never die from being wrong about direction. They die from three mechanical errors — sizing for the wrong distribution, trusting a stop to do something it cannot, and carrying leverage that converts a gap into a liquidation. We are going to take each one apart from zero, anchor every idea in dollars, and end with the single most underrated edge in event trading: the decision to not trade at all.

![The event-day risk stack: size to the expected move, use defined risk, respect gap risk, control leverage, or do not trade](/imgs/blogs/risk-management-around-events-gaps-sizing-stops-1.png)

## Foundations: the risk vocabulary you need before a print

Before we can fix anything, we have to be precise about the words. Most beginners use "risk" as a vague feeling — "this trade feels risky." That feeling will not protect your account. Event-day survival comes from a handful of exact, quantitative concepts, and every one of them has a number attached. Let us define them carefully, building intuition first, then putting dollars on each.

### Position sizing

**Position sizing** is the answer to the single most important question in trading: *how much?* Not which direction, not which asset — how big a position to put on. Every position has a notional value (the dollar amount of the thing you control) and a risk (the dollars you stand to lose if the trade goes against you by some amount). Position sizing is the discipline of choosing notional so that the *risk* — not the notional — is the number you actually decided on in advance.

Here is the mental shift that separates traders who survive from those who do not. Beginners decide a *position*: "I'll buy \$50,000 of this." Professionals decide a *risk*: "I'll lose at most \$500 on this idea," and then *back out* the position size from that. The position is an output, not an input. And the input that converts a dollar-risk budget into a position size is the size of the move you expect — which is exactly the thing that changes on event days.

### The expected move versus ATR

To size a position you need an estimate of how far price might travel against you. There are two common estimates, and confusing them is the original sin of event-day blowups.

**ATR** — the *Average True Range* — measures how much an asset has typically moved per day *recently*. If a stock's ATR is \$2 on a \$200 share, it tends to swing about 1% a day. ATR is a backward-looking, calm-conditions estimate. It is perfectly good for sizing on a normal Tuesday with nothing on the calendar.

The **expected move** is forward-looking and event-aware. It is the market's own estimate, priced into options, of how far the asset will travel by a specific date — including the print. The cleanest way to read it is from the *at-the-money straddle*: buy the call and the put at the current price, add their premiums, and that total is roughly the move the market is paying up for. (We derive this carefully in the companion piece on [pricing event risk with options](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options).) The expected move on an event day is routinely **two to three times** the calm-day ATR, because options dealers know a surprise is coming and charge for it.

The error is sizing to ATR on a day the expected move is triple it. You think you are taking calm-day risk; you are actually taking event-day risk at three times the size.

### Gap risk

A **gap** is a price jump with no trades in between. Price was \$100; the next price anyone could actually transact at was \$97. There is no \$99, no \$98 — just empty space the market leapt over. Gaps happen because, in the seconds around a release, the firms that provide liquidity pull their resting orders to avoid being run over, the order book hollows out, and the first orders to arrive push price straight through the empty levels. (The microstructure of that vacuum is the whole subject of [liquidity and gaps around news](/blog/trading/event-trading/liquidity-and-gaps-around-news).)

**Gap risk** is the risk that the price you can actually get is on the *far side* of a gap from where you wanted to act. It is the reason a stop is not a guarantee, and it is the single most underappreciated risk in event trading.

### A stop is a market order

This is the most important sentence in the post, so read it slowly: **a stop-loss is not a price you exit at; it is a price at which you fire a market order.** When the market touches your stop level, the broker converts your stop into a *market order* — an instruction to sell at the best available price *right now*. On a calm day the best available price is right next to your level, so the stop behaves like a guaranteed exit and you never notice the difference. On an event day, when price gaps past your level, the best available price can be far below it. The stop fires faithfully; the fill is wherever the thin book leaves it.

There is a variant, the **stop-limit**, that fires a *limit* order instead — it will not fill below a price you specify. That sounds safer, and for capping the *price* it is, but it has the opposite failure: if price gaps clean through your limit, the order simply does not fill at all, and you are left holding the position as it keeps falling. Neither flavor of stop is a free lunch around a gap. One risks a bad price; the other risks no exit.

### Slippage

**Slippage** is the difference between the price you expected and the price you got. If your stop was at \$100 and it filled at \$98.50, your slippage was \$1.50, or 1.5%. On a normal day slippage is a rounding error — a cent or two. Around a print it can be the largest single cost of the trade, dwarfing commissions and spreads. Slippage is gap risk made concrete: it is the dollar measure of how far the gap carried your fill past your intended price.

### Defined-risk versus undefined-risk

A position is **defined-risk** if there is a hard, known maximum you can lose, fixed *before* you enter, that no price move — however violent — can exceed. Buying an option is the cleanest example: when you buy a put, the most you can lose is the premium you paid, full stop, whether the underlying falls 5% or 50% or gaps to zero overnight. The downside is *capped by the structure itself*, not by an order you hope will execute well.

A position is **undefined-risk** (or open-ended) if there is no structural floor — the loss is whatever the market does, and you are relying on an *order* (a stop) to limit it. A naked long or short position with a stop is undefined-risk: the stop is a hope, not a guarantee, and a gap can blow through it. The distinction is not academic. On event days, defined-risk structures survive gaps that liquidate undefined-risk ones.

### Leverage and liquidation

**Leverage** is using borrowed money (or margin) to control a position larger than your cash. Five-times leverage means \$5,000 of your money controls a \$25,000 position. Leverage multiplies your gains *and* your losses by the same factor — and crucially, it introduces a hard tripwire that an unleveraged position does not have.

That tripwire is **liquidation** (a *margin call* in equities, *liquidation* in crypto and futures). When losses eat into the borrowed portion, the broker or exchange force-closes your position to protect *their* loan — at market, immediately, whether you like it or not. Leverage means a price move that would merely be a painful drawdown for a cash account becomes a *terminal* event: you do not get to wait for the bounce, because you have already been sold out at the bottom of the gap.

### Maximum loss

**Maximum loss** is the worst dollar outcome a position can produce. For a defined-risk structure it is a known number you can compute before entering. For an undefined-risk position it is, honestly, *unknown* — bounded only by how far price can gap before your stop fills, which on a 5-sigma print is much further than you think. Event-day risk management is largely the project of converting "unknown" maximum losses into "known" ones.

### The no-trade option

Finally, the most underrated tool in the kit: **not trading.** Every event presents a choice you can forget you have — to be flat (no position) into the print. Sitting out is not passivity; it is an active risk decision with positive expectancy whenever your edge over the event's randomness is small or negative. The professionals who last decades are not the ones who trade every print best. They are the ones who *skip* the prints where they have no edge, and concentrate risk on the few where they do.

With the vocabulary in place, we can now build the five-layer defense — and the first layer, the one that does the most work, is sizing.

## 1. Size to the event, not to the day

The deepest mistake in event trading is sizing a position to the wrong probability distribution. On a calm day, returns are small and well-behaved; a 1% move is a big day. On an event day, the *same asset* draws from a much wider distribution — the expected move is two to three times larger, and the tails are fatter still. A position sized for the calm distribution, carried into the event one, is not a little too big. It is *several times* too big.

The arithmetic is simple and it is the most valuable formula in this entire series. If you are willing to risk **R** dollars, and you set your invalidation at a move of **M** percent against you, then your position size is:

```
position notional = R / M
```

That is it. Risk budget divided by the percentage move that would cost you that budget. The whole game is choosing **M** correctly — and **M** is the expected move, not the calm-day range.

![Sizing to the event: the same risk budget buys a much smaller event-day position than a calm-day position](/imgs/blogs/risk-management-around-events-gaps-sizing-stops-2.png)

Watch what happens when **M** changes. Suppose your risk budget **R** is a fixed \$500 — the amount you have decided you are willing to lose on this idea, and it does not change just because there is an event on the calendar. On a calm day, a 0.5% move against you would cost you \$500, so your position is \$500 / 0.005 = \$100,000. That feels normal; it is the size you always trade. Now an event lands and the expected move widens to 1.2%. The *same* \$500 budget now implies a position of \$500 / 0.012 ≈ \$41,000. The position you should carry into the print is **59% smaller** than the one you carry on a quiet day — for the identical dollar risk.

This is the single insight that prevents most event blowups. You do not get richer because there is a CPI print; your risk budget is the same \$500. But the move that spends that budget is twice as wide, so the position that fits it is less than half as big. Traders who blow up on events are almost always carrying their calm-day size into an event-day distribution.

#### Worked example: sizing the same idea on a calm day versus an event day

You have a \$500 risk budget for a long position in a broad equity index. On a normal session the expected move is about ±0.5%, so the size that risks exactly \$500 is \$500 ÷ 0.005 = **\$100,000** of notional. Now there is a CPI print at 8:30, and the at-the-money straddle implies an expected move of ±1.2% — more than double. With the *same* \$500 budget, the event-day size is \$500 ÷ 0.012 ≈ **\$41,000**. If you had instead carried the calm-day \$100,000 into the print and the index moved its full 1.2% against you, your loss would be \$100,000 × 0.012 = **−\$1,200** — more than double the \$500 you decided to risk. Cutting the position 59% is not timidity; it is the only way to keep your actual risk at the \$500 you chose.

The mechanical rule that falls out of this: **before every event, re-derive your size from the current expected move.** Pull the straddle (or the implied move your broker shows), divide your risk budget by it, and that is your maximum size. If you cannot be bothered to do the arithmetic, the safe default is to cut your normal size by half or more going into any major print — because the expected move is reliably one and a half to three times the calm-day range, halving size is roughly the right order of magnitude even before you look at a single option.

There is a second-order point worth making explicit, because it is where disciplined-sounding traders still go wrong: **the risk budget is a constant, and the position is the variable.** People reverse this. They decide they want a \$100,000 position (because that is what they always trade), and then they go looking for a stop level that "feels right" — and on an event day the stop that feels right is a tight one, which on \$100,000 at a 1.2% expected move means a real risk of \$100,000 × 0.012 = **−\$1,200**, more than double their intended \$500, the moment the move comes in at expectation. The correct order is the opposite: fix the \$500, read the 1.2% off the straddle, and *let the position fall out at \$41,000*. The position is whatever the arithmetic says it is. If \$41,000 feels too small to be worth trading, that is not a reason to size up — it is information that the event's risk-adjusted opportunity is small, which loops directly into the no-trade decision in layer five.

One more practical wrinkle: the expected move is a *one-standard-deviation* estimate, meaning the asset stays inside it only about two-thirds of the time. Roughly one event in three moves *more* than the expected move, and a few percent of events move two or three times it — those are the −4.32% and −12.40% tails. So sizing to the expected move is sizing to the *typical* event, not the worst one. That is fine, because it is the layer that handles the common case; the defined-risk, leverage, and no-trade layers stacked on top are what handle the tail the expected move understates. Never treat the expected move as a worst case. It is a center-of-mass, and the account-ending moves live in the part of the distribution it does not cover.

#### Worked example: the tail you are actually sizing against

Sizing to the expected move handles the *typical* event day. But events also produce genuine tail moves, and it is worth sizing with those in mind. The S&P 500 fell **−4.32%** on the hot Aug-2022 CPI print and rose **+5.54%** on the cool Oct-2022 one. On a \$41,000 position those are −\$1,771 and +\$2,271 respectively — already large relative to a \$500 budget, because a 4–5% day is a multi-sigma surprise the straddle did not fully price. Now consider the real tails: in the Aug-2024 carry cascade the Nikkei fell **−12.40%** and Bitcoin fell **−15%** in hours. A \$41,000 position would have lost \$41,000 × 0.124 = **−\$5,084** on the Nikkei move and a \$10,000 crypto position would have lost \$10,000 × 0.15 = **−\$1,500**. The lesson is that even a "correctly sized" event position can hand you a multiple of your intended risk on a true tail — which is why the next four layers (defined risk, gap awareness, leverage control, and the no-trade option) exist on top of sizing, not instead of it.

![Event days produce single-day moves far larger than a normal-day stop expects](/imgs/blogs/risk-management-around-events-gaps-sizing-stops-3.png)

The bar chart above is the entire case for sizing to the event in one picture. Those are not exotic instruments or obscure dates — they are the S&P 500, the Nikkei, and Bitcoin on four well-documented event days. A −4.32% or −12.40% day is simply not in the calm-day distribution your ATR was measuring. If your stop and your size were set for a 1% world, none of these moves were survivable at full size. Size, then, is the foundation — but it is not enough by itself, because even a well-sized position assumes your stop will exit you near your level. It will not, and that is layer two.

## 2. Why a stop does not guarantee the exit

Most traders believe a stop-loss caps their loss. It does not. A stop caps the *level at which you start trying to exit* — and on an event day the distance between "start trying" and "actually filled" can be enormous. To see why, you have to remember what a stop actually is: not an exit, but a trigger that fires a market order. And a market order is only as good as the order book it lands in.

On a calm day the book is deep — thousands of shares resting within pennies of the price — so when your stop fires, the market order fills right next to your level. The mechanism is invisible. Around a print, the book is the opposite. In the half-second before the number drops, market makers cancel their resting quotes to avoid being adversely selected by faster traders, the book thins to a near-vacuum, and the first orders to arrive sweep through whatever sparse bids remain. Your stop fires into that vacuum. There is nobody bidding at your level, or just below it, so your sell order walks down the empty ladder until it finds a resting bid with size — which can be a full percent or more below where it triggered.

![A stop is not a guaranteed exit: price gaps over the stop level and the fill lands far below it](/imgs/blogs/risk-management-around-events-gaps-sizing-stops-4.png)

The figure traces the whole mechanic. You enter in the green safe zone. Your stop sits at −2%, the price you *chose*, and on a normal day that is exactly where you would exit. The 8:30 print lands; price gaps clean through the stop band — the grey region where, on this day, *no trades happen at all* — and keeps falling until it finds resting liquidity in the red fill band at −3.5%. The slippage, the extra −1.5% between your level and your fill, is not a malfunction. It is the gap, and it is the difference between the loss you planned and the loss you took.

#### Worked example: a gap through a \$30,000 stop

You hold a \$30,000 position and set a stop at −2% to cap your loss at \$30,000 × 0.02 = **−\$600**. That is your plan, and on any normal day it works exactly. Then a surprise print hits and price gaps straight through your level; the first available fill is at −3.5%. Your realized loss is \$30,000 × 0.035 = **−\$1,050**, not −\$600. The gap cost you an extra \$30,000 × 0.015 = **−\$450**, which is 75% more than you planned to lose — and you did *nothing wrong* in the textbook sense. You had a stop; it fired; it just could not find a fill at your level. The intuition: a stop converts a known maximum loss into a hoped-for one, and a gap is exactly the event that breaks the hope.

This is why the opening trader's −\$300 plan became a −\$900 loss: same mechanism, a wider gap. And it is why three follow-on rules matter more than the stop itself:

**First, place stops where the book is, not where the chart is.** A stop sitting on an obvious round number or a well-known support level is sitting exactly where everyone else's stops are — and clusters of stops are precisely what a thin post-print book runs through fastest, because each triggered stop is a market order that pushes price to the next one. A stop one tick beyond the crowd is not safer; a stop sized into a position that can survive the gap is.

**Second, treat the stop's fill as variable, not fixed, when you size.** If you genuinely need a hard cap, do not rely on a stop to provide it — use a structure that does (the next section). If you are going to use a stop, size as though it will fill meaningfully worse than its level on an event day, because it will.

**Third, do not move a stop *closer* into an event thinking it reduces risk.** A tighter stop on an event day is more likely to be triggered by the initial whipsaw *and* to fill badly when it is — you get the worst of both, stopped out on noise at a gapped price. If anything, event-day stops should be *wider* (to survive the knee-jerk) and *position size* should be *smaller* (to keep the wider stop's dollar risk in budget). Width and size are the two dials; people reach for the wrong one.

The honest conclusion is that around events, a stop is a courtesy, not a contract. The only way to *guarantee* a maximum loss through a gap is to make the structure itself cap it — which is defined risk, and that is layer three.

## 3. Defined-risk structures versus naked stops

If a stop cannot guarantee your maximum loss through a gap, what can? A structure whose worst outcome is fixed by its own design, not by an order's execution. The cleanest of these is buying an option.

When you **buy a put**, you pay a premium up front for the right to sell the underlying at a fixed strike price until expiry. The most you can lose is that premium — period. If the underlying gaps down 3%, 15%, or to zero overnight, your loss is still just the premium, because the *right to sell at the strike* gets more valuable exactly as the underlying falls. There is no order to fire, no book to fill into, no gap to slip through. The cap is structural. (If you want the full mechanics of how options price and behave, the [options-theory](/blog/trading/quantitative-finance/options-theory) primer in the quant series builds them from scratch.)

Contrast that with a naked long position protected by a stop. The two look similar on a calm day — both "limit your loss to about X" — but they are categorically different on an event day. The stop is undefined-risk wearing a defined-risk costume: its cap is a *hope* about execution. The put is genuinely defined-risk: its cap is a *fact* about the contract.

![Defined risk versus a naked stop: the long put caps the loss at the premium, the stop does not](/imgs/blogs/risk-management-around-events-gaps-sizing-stops-5.png)

The comparison figure makes the asymmetry explicit. The naked stop costs nothing up front — which is exactly why it feels safe and why people over-rely on it — but its maximum loss is *unknown*, set by wherever the gap leaves the fill (−\$1,050 in our earlier example). The long put costs a premium up front, but its maximum loss is *known before you trade* and is immune to the gap entirely. You are paying a fee to convert an unknown, open-ended risk into a known, capped one. On the days that matter — the gap days — that fee is the cheapest insurance you will ever buy.

#### Worked example: a long put caps the loss a stop could not

Recall the gapped stop: a \$30,000 position with a −2% stop that filled at −3.5%, a **−\$1,050** realized loss. Now suppose instead of a naked long with a stop, you had structured the trade as a long position hedged with a put — or simply replaced it with a long call — and the defined-risk cost was a **\$400** premium. When the print gaps the market down 3.5%, your maximum loss is the premium: **−\$400**, full stop, regardless of how far the gap ran. Even if the move had been the −12.40% of the Aug-2024 cascade, your loss would still be **−\$400** — whereas the naked-stop position would have lost \$30,000 × 0.124 = **−\$3,720** as its stop chased the price down. You paid \$400 to make the gap *irrelevant*. The intuition: defined risk is not about being right; it is about making the size of your worst outcome a decision you make at entry, not a number the gap hands you afterward.

The objection people raise is cost: options have premium, and around events that premium is *expensive* precisely because the expected move is wide (the same elevated implied volatility that warns you of the gap also raises the price of the hedge). That is true and worth respecting. Two responses. First, expensive insurance on a day that genuinely can gap is still rational insurance — you are not buying it every day, only into events you choose to hold through. Second, the premium is *defined*: you know the exact cost before you commit, which is the entire point. A −\$400 known cost beats a −\$1,050 (or −\$3,720) unknown one on any day you actually gap.

There is also a subtler benefit. Because a long option's loss is capped at the premium, you can size it against that premium directly — the premium *is* your risk budget. There is no gap risk to pad for, no stop slippage to estimate. The position cannot lose more than you paid, so the sizing arithmetic from layer one becomes exact rather than approximate. For traders who specifically want to *hold through* a print rather than dodge it, a long option is often the only honest way to do it, because it is the only structure whose maximum loss a gap cannot enlarge.

It is worth being precise about *which* options structures are defined-risk, because not all of them are. **Buying** an option — a long call or a long put — is defined-risk: the premium is the most you can lose. **Selling** an option naked — writing a put or a call without owning the underlying or a hedging option — is the opposite: undefined-risk, and catastrophically so around events. A naked short put collects a small premium and carries an enormous, open-ended loss if the underlying gaps down, which is exactly the move events produce. Traders are drawn to selling options into events because the premium is fat (high implied volatility again), but selling open-ended risk into the one situation that produces gaps is how option sellers blow up. If you are going to use options as your defined-risk layer, *buy* them, or use a *spread* that caps both legs. The premium you pay as a buyer is the price of the gap-proof floor; the premium you collect as a naked seller is the bait on the gap's hook.

#### Worked example: long option versus naked short option into a gap

You want exposure to an index into a print. Structure A: buy a call for a **\$400** premium — defined risk, max loss \$400 no matter what. Structure B: sell a put to *collect* a \$400 premium — undefined risk. The print gaps the index down 4%, deep past the put's strike. Structure A loses its \$400 premium and not a cent more. Structure B, on a \$50,000 notional exposure, now owes the difference as the short put goes deep in the money — a loss that can run \$50,000 × 0.04 = **−\$2,000** or worse, against the \$400 you collected, a **−\$1,600** net on a position you entered for "income." Same \$400 premium, same gap, opposite outcomes: the buyer's \$400 was a cap, the seller's \$400 was a down payment on an open-ended loss. The intuition: defined risk means *buying* the floor, never *selling* it.

This does not make options the answer to everything — they decay, they require knowing what you are doing, and the premium is a real drag if you over-use them. But as the *defined-risk* layer of the event-day stack, a *long* option is the tool that does what a stop only pretends to: it puts a hard, gap-proof floor under your loss. With sizing (layer one), gap-awareness (layer two), and defined risk (layer three) in place, there is one more force that can override all of them — leverage — and it deserves its own section because it is the mechanism behind the most spectacular event-day wipeouts.

## 4. Leverage and the liquidation cascade

Everything so far assumed you control the timing of your exit — that you can choose to hold, hedge, or wait out a gap. Leverage takes that choice away. When you trade on borrowed money, you introduce a third party (the broker or exchange) whose only concern is recovering their loan, and who will *force-close your position* the instant your losses threaten it — regardless of your thesis, your hedge, or your willingness to wait for the bounce. Leverage is the one risk on this list that can liquidate you *before* the move is even over.

The arithmetic is brutally simple and worth internalizing. Leverage multiplies the percentage move's effect on your equity by the leverage factor. At 5x, a −15% move in the asset is a −75% move in your equity. At 7x, a −15% move wipes you out entirely. The exact percentage move that liquidates you is roughly 1 divided by your leverage: at 5x, a −20% move; at 10x, a −10% move; at 20x, a −5% move. Now line those numbers up against the event tails from layer one — the Nikkei's −12.40%, Bitcoin's −15% — and the danger is obvious. A −15% gap is *survivable* for a cash account (painful, but you live to trade tomorrow). At 5x leverage it is a liquidation. At any leverage above about 6.7x it is a total loss.

#### Worked example: 5x leverage meets a −15% gap

You have \$5,000 in a crypto account and use 5x leverage, controlling a \$5,000 × 5 = **\$25,000** position. An event hits — say the Aug-2024 carry cascade — and your asset gaps **−15%** in a matter of hours. The position loses \$25,000 × 0.15 = **−\$3,750**. But you only had \$5,000 of equity, so that −\$3,750 is **−75% of your account**, and well before it gets there the exchange's liquidation engine force-sells your position to protect its loan. You do not get to wait for the bounce. By the time the asset recovers — and after the Aug-2024 gap it bounced hard — your position is already gone, sold at the bottom of the gap. The same −15% move on an *unleveraged* \$5,000 position is a **−\$750** drawdown: ugly, but you still hold it, and you participate fully in the recovery. The intuition: leverage does not just amplify your loss; it removes your ability to *wait*, converting a temporary drawdown into a permanent wipeout.

Now the part that makes leverage genuinely dangerous on event days, beyond just your own account: **liquidations feed on themselves.** This matters most in crypto, where leverage is ubiquitous and the order book is thin, but the mechanism appears anywhere leverage is stacked.

![The leverage liquidation spiral: a gap forces selling that deepens the gap and liquidates the next tier](/imgs/blogs/risk-management-around-events-gaps-sizing-stops-6.png)

Trace the spiral in the figure. Leverage sets the stage: many accounts controlling far more notional than their equity. An event gap arrives — the BTC −15%. The most-leveraged accounts hit their liquidation level first, and the exchange force-sells their positions *at market*, into an already-thin post-event book. That forced selling is itself a wave of sell pressure that pushes price *lower*, which trips the *next* tier of leveraged accounts into liquidation, whose forced selling pushes price lower still. The gap does not just happen *to* leveraged traders; their forced exits *manufacture* more gap. This is why crypto liquidation cascades overshoot so violently — billions in leverage can unwind in minutes, each liquidation begetting the next. The unleveraged account on the right of the figure simply sits through it: −15% is a drawdown, not a margin call, and it is there to buy the bounce.

#### Worked example: the same event, leveraged versus not

Two traders both hold \$10,000 of exposure to an asset that gaps −15% on an event. Trader A is unleveraged: \$10,000 cash, controlling \$10,000 of the asset. Her loss is \$10,000 × 0.15 = **−\$1,500**, a 15% drawdown; she holds through it and recovers as the asset bounces. Trader B reaches the same \$10,000 of exposure with only \$2,000 of equity at 5x leverage. His loss is also \$10,000 × 0.15 = **−\$1,500** *in dollars*, but that is **−75% of his \$2,000 equity**, so he is liquidated near the lows and crystallizes the loss permanently. Identical asset, identical move, identical dollar loss on the position — but one trader has a bad week and the other has an empty account. The intuition: leverage does not change the move; it changes whether you are still standing when the move reverses.

There is a particularly nasty interaction between leverage and the gapped-stop problem from layer two. A leveraged trader who "protects" the position with a stop has, in fact, *two* failure modes stacked on the same gap. When price gaps through the stop level, the stop fires a market order that fills badly — the slippage of layer two — *and* the same gap can push the position past its liquidation threshold before the stop's order even completes, so the exchange's liquidation engine and the trader's own stop are both selling into the same vacuum at the same instant, competing for the same thin bids. The result is a fill far worse than either mechanism alone would have produced. The stop did not protect against the gap; it added one more market sell order to the cascade. This is why "leverage is fine, I use a stop" is one of the most expensive sentences in trading: the stop and the leverage fail *together*, on the same move, in the same second.

The discipline that follows is unglamorous and absolute: **size your leverage so that the worst plausible event gap cannot liquidate you.** Work backward from the tail, not the average. If a −15% event gap is realistic for your asset (and in crypto it is), then leverage above roughly 3x is playing Russian roulette with the calendar — a single bad print ends you. The exchanges that survive these cascades publish the wreckage afterward: a single Aug-2024-style cascade can force *billions* of dollars of leveraged positions to liquidate in minutes, each liquidation a market sell that drives the next. You do not want to be a data point in that total. For most traders around events, the right answer is no leverage at all on the print, or leverage low enough that even a −20% gap is a drawdown rather than a death. Leverage is the layer that can override every other defense: you can size perfectly, respect gaps, and buy defined risk, and a margin call still force-closes you at the worst price if your leverage is too high. Control it, or it controls your exit. And the final layer recognizes that for many events, the cleanest defense is to not be in the spiral at all.

## 5. The no-trade decision: when sitting out is the trade

Here is the layer nobody markets, because there is no commission in it: the decision to *not trade* an event. It is the outermost ring of the risk stack, and it is the strongest, because it removes every other risk at once. You cannot be gapped through a stop, liquidated by a cascade, or sized wrong for the move if you are flat into the print.

The reflex to trade every event is a beginner's instinct, and it is exactly backwards. Events are not opportunities by default; they are *coin flips with a fee* unless you have a specific, real edge over their randomness. The expected move is wide precisely because the outcome is genuinely uncertain — if it were predictable, the market would have already priced it. So the honest question before every print is not "which way will it go?" but "do I have a *repeatable reason* to be on the right side of this *more often than chance*, and is that edge big enough to overcome the spread, the slippage, and the gap risk I am taking on?" For most traders, on most events, the honest answer is no.

![The no-trade decision tree: if your edge is not bigger than the event randomness, go flat into the print](/imgs/blogs/risk-management-around-events-gaps-sizing-stops-7.png)

The decision tree is deliberately blunt. An event is ahead. Do you have a real edge — a positioning skew you have actually measured, a known reaction asymmetry, a documented pattern — or just a guess on a coin-flip print? If it is a guess, the highest-expectancy action is to *go flat*: close or hedge before the number, and protect your capital so you are around to trade the events where you *do* have an edge. If you have a genuine edge, then trade it — but trade it *small*, with *defined risk* and a *planned invalidation*, because even a real edge on an event is a thin edge over a wide distribution. Either branch ends in the same place: the account survives. No random 5-sigma loss, no liquidation, and you are still in the game tomorrow.

#### Worked example: the expectancy of skipping a coin-flip print

Suppose you trade an event with no real edge — call it a 50/50 outcome — risking \$500 to make \$500, but with realistic event-day frictions: a wider spread, slippage on entry and exit, and the occasional gap that costs you more than your stop on the losing side. Even modest frictions of, say, \$60 per round trip drag your expectancy negative: 0.5 × (+\$500) + 0.5 × (−\$500) − \$60 = **−\$60 per trade**. Do that across 20 events a year and the no-edge program *expects* to lose 20 × \$60 = **−\$1,200** annually before a single tail event. Now subtract the inevitable gapped-stop disasters — one −\$900-instead-of-\$300 print can erase several "wins" — and the math is decisively against trading coin flips. Sitting out those 20 prints is not missing out; it is *saving* \$1,200 plus the tail losses. The intuition: a trade with no edge has *negative* expectancy after costs, so not taking it is a positive-expectancy decision — flat is a position, and often the best one.

This reframes the whole craft. The professional event traders who last are not the ones with the best read on every CPI; they are the ones who *trade five prints a year and skip the other forty*, concentrating risk only where they have a measurable edge and going flat everywhere else. The no-trade option is free, it is always available, and it dominates every other risk control because it makes the others unnecessary. When in doubt, be flat into the print. You can always re-enter once the dust settles, the book refills, and the spread normalizes — by which point the gamble has become a trade again. (The companion piece on [building an event-day trading plan](/blog/trading/event-trading/building-an-event-day-trading-plan) walks through writing the no-trade rule into a checklist you actually follow.)

## How it reacted: real episodes

Principles are easy to nod along to and hard to feel. Two dated episodes make the risks concrete, with real numbers, because both are textbook cases of exactly the failures this post is about.

### August 5, 2024: the gap cascade that liquidated leverage

The Aug-2024 carry unwind is the cleanest modern example of every layer failing at once. The setup: the Bank of Japan hiked on July 31, a weak US jobs report landed on August 2, and the yen-funded carry trade — borrow cheap yen, buy risk assets — began to unwind in force. (The macro mechanism is detailed in the macro series' [carry-trade-unwinds](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) piece.) Over the weekend and into Monday, August 5, the unwind became a cascade.

The numbers are the lesson. The Nikkei 225 fell **−12.40%** on August 5 — its worst day since the 1987 crash. The S&P 500 fell **−3.00%**, the Nasdaq 100 **−3.43%**, and Bitcoin **−15%**. The VIX, the equity fear gauge, spiked to an intraday **65.73** from a close near 23 just three sessions earlier. Every one of those is a move a calm-day stop never anticipated, and the BTC −15% is exactly the gap that liquidates 5x leverage.

#### Worked example: the Aug-2024 gap across position sizes

Take a \$25,000 equity exposure into August 5. Unleveraged, the S&P's −3.00% costs \$25,000 × 0.03 = **−\$750** — a bad day, fully survivable, and you participated in the sharp +10.23% Nikkei bounce the very next session. Now take the same dollar exposure to crypto's −15% gap: \$25,000 × 0.15 = **−\$3,750** unleveraged, already a serious loss. At 5x leverage on \$5,000 of equity controlling that \$25,000, the same −\$3,750 is **−75% of your account** and triggers liquidation — you are sold out near the lows and you miss the bounce entirely. One event, three radically different outcomes, determined entirely by *size* and *leverage*, not by direction. The intuition: on a true tail, the only thing standing between a drawdown and a wipeout is the risk decisions you made *before* the print.

The traders who came through August 5 intact were not the ones who predicted it. They were the ones who were sized small, carried no leverage they could not survive a −15% gap on, and in many cases were simply *flat* over a weekend with the BoJ and US payrolls both in play. The traders who were liquidated were leveraged into the carry trade and gapped through every stop they had set.

### A CPI day that gapped through stops

The CPI prints are the other archetype. On the hot Aug-2022 CPI (released September 13, 2022), the S&P 500 fell **−4.32%**, the Nasdaq **−5.16%**, and Bitcoin roughly **−9.4%**, while the dollar jumped **+1.4%**. The full cross-asset anatomy is in the [CPI case studies](/blog/trading/event-trading/cpi-case-studies-the-prints-that-broke-the-tape) piece; here the point is narrower. A −4.32% S&P day means that any long position with a stop set inside that range got triggered — and triggered into the thin, fast post-CPI book, where stops fill well below their levels.

#### Worked example: a stop run on the hot CPI print

You are long a \$50,000 S&P position into the Sep-2022 CPI with a stop at −2% to cap your loss at \$50,000 × 0.02 = **−\$1,000**. The print is hot; the index gaps lower in the first seconds and runs to −4.32% on the day. Your stop triggers in the initial plunge, but the book is a vacuum — it fills not at −2% but, say, −3% as it walks down the empty ladder. Your realized loss is \$50,000 × 0.03 = **−\$1,500**, fifty percent more than your −\$1,000 plan, and you are stopped out near the worst levels of a move that the market partly retraced intraday. Had you instead held the same view with a defined-risk long call costing a \$700 premium, your maximum loss on the entire move would have been **−\$700** — known, capped, and gap-proof. The intuition: on the prints that "break the tape," the stop is the thing that breaks first, and defined risk is what holds.

Both episodes say the same thing from different angles. The market did not punish a bad directional call; it punished a risk *structure* that assumed calm-day mechanics on an event day. Sizing, gap-awareness, defined risk, leverage control, and the no-trade option are not five separate tips. They are one idea — *match your risk structure to the event distribution* — applied at five points.

## Common misconceptions

A handful of beliefs do more damage than any single bad trade, because they feel like prudence while quietly setting up the blowup. Each is corrected with a number.

**"My stop caps my loss."** Only to the next available price. A stop caps the *level at which you begin trying to exit*, not the price you achieve. On the gapped −2% stop earlier, the realized loss was −3.5% (−\$1,050 on \$30,000), 75% past the plan. Your stop guarantees an attempt, not a fill. Internalize that one sentence and half of event-day risk takes care of itself.

**"A tighter stop is safer."** On an event day a tighter stop is *more* dangerous: it is more likely to be triggered by the initial whipsaw and to fill badly when it is. The right dials are a *wider* stop (to survive the knee-jerk) paired with a *smaller* position (to keep the wider stop's dollar risk in budget). Tightening the stop while keeping the size fixed gives you the worst of both — stopped out on noise, at a gapped price.

**"I'll just set a stop and walk away."** That is precisely the trap the opening trader fell into. A −\$300 plan became −\$900 because the stop filled 1.5% past its level into a thin book. Around a major print, a stop you cannot watch is a stop that can fill anywhere; if you genuinely cannot watch the trade, the correct move is defined risk (a known cap) or no position at all — not a stop and a coffee break.

**"Options are too expensive to bother with around events."** Expensive *known* risk beats cheap *unknown* risk on the days that gap. The \$400 put premium that felt costly is the trade that turns a potential −\$1,050 (or −\$3,720 on a tail) into a fixed −\$400. You are not buying options every day — only into the events you choose to hold through, where the premium is the cheapest way to make the gap irrelevant.

**"Leverage is fine if I use a stop."** Leverage and stops fail *together* on event days, not separately. A leveraged position gaps through its stop and gets liquidated at the bottom of the same gap that triggered it — the stop fills badly *and* the margin call fires, often within the same second. At 5x, a −15% gap is a −75% account hit no stop could have salvaged. Leverage does not just amplify the loss; it removes the time you would need to manage it.

## The playbook: how to trade risk around an event

Pulling it together into a before-the-print checklist you can actually run. The order matters: each step is a layer of the risk stack, from the outermost decision inward.

**1. Decide whether to trade at all.** Ask the honest question: do I have a real, repeatable edge over this event's randomness, big enough to overcome spread, slippage, and gap risk? If not, go flat into the print. This is free, always available, and removes every downstream risk. When in doubt, flat.

**2. If you trade, size to the expected move, not the day.** Pull the at-the-money straddle (or your broker's implied move). Set your dollar risk budget **R**. Size = R ÷ (expected move %). The same \$500 budget that buys \$100,000 on a ±0.5% day buys only ~\$41,000 on a ±1.2% event day — and your budget did not change because there is an event. If you will not do the arithmetic, default to cutting normal size by at least half.

**3. Prefer defined risk for anything you hold through the print.** A long option caps the loss at the premium, gap-proof, known before entry. A \$400 premium that fixes your max loss beats a stop that hopes to. Size the option against its premium directly — the premium is your risk budget, and there is no gap to pad for.

**4. If you use a stop, treat it as a market order and place it where the book is.** Set it *wider* than calm-day stops (to survive the knee-jerk), keep the position *smaller* (to keep the dollar risk in budget), avoid the obvious round-number/support levels where stop clusters live, and assume it fills meaningfully worse than its level. Never tighten a stop into an event thinking it reduces risk.

**5. Control leverage against the tail, not the average.** Size leverage so the worst plausible event gap is a drawdown, not a liquidation. If a −15% gap is realistic (crypto: it is), keep leverage at or below ~3x — ideally none on the print. Remember liquidations cascade: forced selling deepens the gap and trips the next tier, so the tail is worse than the average suggests.

**6. Plan the invalidation and the re-entry before the print, not during it.** Write down the level or condition that says you were wrong, and the condition under which you re-enter once the book refills and the spread normalizes. Decisions made in the calm before the number are infinitely better than decisions made in the two seconds of a gap.

The throughline of every step is the same: **match your risk structure to the event distribution.** Calm-day mechanics — full size, a trusted stop, ambient leverage, an instinct to trade everything — are exactly what blows up on event days, because the event distribution is wider, the book is thinner, and the gaps are real. The traders who survive decades of prints are not the best forecasters. They are the best *sizers*, the ones who treat a stop as a market order, who buy defined risk when they hold through, who refuse leverage that a gap can liquidate, and who — most of all — know which events to skip. Get the risk structure right, and direction becomes a survivable bet. Get it wrong, and one print is all it takes.

## Further reading and cross-links

Within this series, on the mechanics this post depends on:

- [Liquidity and gaps around news](/blog/trading/event-trading/liquidity-and-gaps-around-news) — the order-book microstructure of *why* prices gap, in detail; the plumbing behind every stop run here.
- [The expected move: pricing event risk with options](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options) — how to read the straddle that you size against in layer one.
- [Positioning and the pain trade](/blog/trading/event-trading/positioning-and-the-pain-trade) — how crowded positioning fuels the cascades and stop runs that gap risk feeds on.
- [Building an event-day trading plan](/blog/trading/event-trading/building-an-event-day-trading-plan) — turning the no-trade rule and the sizing arithmetic into a checklist you actually follow.
- [How crypto reacts to macro news](/blog/trading/event-trading/how-crypto-reacts-to-macro-news) — the 24/7, high-leverage market where the liquidation spiral is most violent.

For the policy and options mechanism underneath the reactions, the macro and quant series go deeper: the [carry-trade-unwinds](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) piece on how leverage breaks in 1998, 2008, and 2024, and the [options-theory](/blog/trading/quantitative-finance/options-theory) primer on how a long option caps your loss by construction.
