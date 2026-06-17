---
title: "The Options Chain and Contract Mechanics: Multiplier, Expiry, Settlement"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Read a live options chain column by column, size a trade with the 100-share multiplier, and walk a contract from open to exercise, assignment, and settlement without getting surprised."
tags: ["options", "volatility", "options-chain", "contract-multiplier", "expiration", "settlement", "assignment", "occ", "american-vs-european", "options-mechanics"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Before you can trade the gap between implied and realized volatility, you have to handle the option as a *physical instrument*: a standardized contract on 100 shares, with a quote that is per-share, an expiry on a calendar, and an exercise-and-settlement process run by a central clearinghouse. Get the plumbing wrong and a winning view still loses money.
>
> - A quote is **per share**. One contract controls **100 shares**, so a \$2.40 quote costs you \$240, and a 100-strike option in the money by \$5 is worth \$500, not \$5.
> - Every contract ends exactly one of three ways: you **close it early** (most do), it **expires worthless**, or it is **exercised and assigned** and shares change hands.
> - **Equity options are American** (exercisable any day) and **physically settled** (you get shares); **index options like SPX are European** (only at expiry) and **cash-settled** — and the AM-settlement print on SPX is a real, recurring trap.
> - The one rule to remember: **read the chain, then read the contract spec.** Multiplier, expiry style, settlement type, and assignment risk are decided before you click buy — not after.

A trader I know was short one put. Just one. A single 30-day \$180 put on a stock trading near \$185, sold for \$3.46 to collect a little premium into a quiet week. Defined plan, small size, nothing dramatic. Then the stock dropped to \$176 on a Thursday, and Friday morning — before the market even opened — his broker emailed him an **assignment notice**. Overnight he had become the owner of 100 shares of stock he never intended to buy, for \$180 each, \$18,000 of stock he now had to either hold or sell at a loss, with the cash debited from his account. He had not pressed a single button. He had not "exercised" anything. Someone *else* — a long holder he would never meet — had exercised their right, and the clearinghouse had reached into the pool of everyone short that contract and pulled his name out at random.

Nothing he did was wrong in the *direction* sense. His thesis was even defensible. What got him was the **mechanics**: he did not understand that selling a put is a standing obligation to buy 100 shares at the strike, that "100" is not a figure of speech, that an American-style equity option can be exercised against him any day, and that assignment is allocated by a coin flip he has no control over. This is the post that makes sure that never happens to you. Before we trade volatility, before we touch a single Greek, we learn the instrument cold: how to read the chain, what a contract actually controls, when it expires, and how it settles.

![Annotated options chain showing calls on the left, strikes in the middle, and puts on the right with bid, ask, volume, open interest, and implied vol columns](/imgs/blogs/the-options-chain-and-contract-mechanics-multiplier-expiry-settlement-1.png)

## Foundations: what an option contract actually is

An **option** is a standardized contract that gives its **buyer** the *right, but not the obligation*, to trade a fixed amount of an underlying asset at a fixed price on or before a fixed date. The **seller** (also called the **writer**) takes the other side: they receive cash today and accept the *obligation* to perform if the buyer chooses to use their right.

Strip away the jargon and an option is **insurance on a price**. When you buy car insurance, you pay a premium today for the right — not the obligation — to file a claim if something bad happens. You hope you never use it. The insurer collects your premium and hopes the same. An option works identically: the buyer pays a **premium** for a right they may never exercise; the seller collects that premium and carries the risk.

Four numbers define every option, and you will see all four on the chain:

- **Underlying** — the asset the contract is written on (a stock like AAPL, an ETF like SPY, an index like SPX).
- **Type** — a **call** (the right to *buy* the underlying at the strike) or a **put** (the right to *sell* it at the strike).
- **Strike price** — the fixed price at which the underlying changes hands if the option is used.
- **Expiration date** — the last day the right is alive.

A fifth number, the **premium**, is not fixed — it is the *price* of the contract, and it moves all day as the underlying moves, as time passes, and as the market's view of future volatility changes. Most of this series is about that premium: where it comes from, why it decays, and how to trade it. But the premium only makes sense once you know what one contract *is*, and that is the contract specification — the spec — which is identical across every contract in a class because options are exchange-standardized. Standardization is what lets a chain exist at all: every \$185 call expiring on the same Friday is fungible, so they can all trade against one common bid and ask.

We are deliberately *not* deriving where the premium number comes from here. The fair value of an option is the output of a pricing model — Black-Scholes and its descendants — and that derivation has its own home. If you want the math of *why* a 30-day at-the-money call on a \$185 stock is worth about \$6.22, read the [Black-Scholes deep dive](/blog/trading/quantitative-finance/black-scholes) and the broader [options-theory primer](/blog/trading/quantitative-finance/options-theory). This post takes the price as given by the chain and teaches you to *handle* it.

### What "in the money" means

One piece of vocabulary you need before reading a chain, because the whole left-right structure depends on it. An option's **moneyness** describes where the strike sits relative to the current price (the **spot**):

- A call is **in the money (ITM)** when spot is *above* the strike — you could buy below market. It is **out of the money (OTM)** when spot is below the strike, and **at the money (ATM)** when they are roughly equal.
- A put is **in the money** when spot is *below* the strike — you could sell above market. OTM when spot is above, ATM when equal.

The amount by which an option is in the money is its **intrinsic value**: `max(spot − strike, 0)` for a call, `max(strike − spot, 0)` for a put. Everything in the premium above intrinsic value is **extrinsic value** (also called **time value**), and it is pure optionality — the value of the time and uncertainty remaining. An OTM option has *zero* intrinsic value; its entire premium is extrinsic. This split matters enormously for exercise and assignment, which is why we anchor on it now.

## Reading a live options chain

The **options chain** (or option chain) is the screen every options trader lives in. It is a table, organized around one expiration date, with **calls on one side, puts on the other, and strikes running down the middle**. The cover figure above shows the layout; let's read it column by column.

The strikes are the spine. They run from deep in the money to deep out of the money, usually in fixed increments (\$1, \$2.50, \$5, or \$10 apart depending on the underlying's price and liquidity). For each strike, the **call** quotes sit on the left and the **put** quotes on the right. A given strike is in the money for the call *and* out of the money for the put when spot is above it, and vice versa — so as your eye travels down the strikes, the calls go from ITM (top) to OTM (bottom) while the puts do the reverse. The at-the-money row, where the strike is closest to spot, is the busiest part of the chain and usually the most liquid.

For each contract — each call and each put — the chain shows a handful of columns. Here is what each one means and why you care:

- **Bid** — the highest price someone is currently willing to *pay* for the contract. If you want to *sell*, this is what you can get *right now*.
- **Ask** (or **offer**) — the lowest price someone is currently willing to *sell* at. If you want to *buy*, this is what you pay right now.
- **Last** — the price of the most recent trade. It can be stale, especially in thin strikes, so never size a decision off "last" alone.
- **Volume (Vol)** — the number of contracts traded *today*. A liquidity proxy: high volume means tight spreads and easy entry/exit.
- **Open interest (OI)** — the number of contracts *currently open* (created and not yet closed or expired). OI accumulates across days; volume resets each morning. Together they tell you whether a strike is real or a ghost town.
- **Implied volatility (IV)** — the volatility figure that, plugged into the pricing model, reproduces the option's market price. This is the single most important column, because **IV is what you are actually trading.** Two options can have the same dollar premium and wildly different IV; the cheap-looking one may be the expensive one once you account for what it implies about future movement.

A subtle but important point about that IV column: it is *not constant across the chain.* If you read the IV figures down the strikes in our mock chain, the out-of-the-money puts (low strikes) print *higher* IV than the out-of-the-money calls (high strikes) — 31% on the \$175 put versus 28% on the \$195 call. That asymmetry has a name, the **volatility skew** (or "smirk"), and it reflects the market's structural willingness to pay up for downside protection. You don't need to trade it yet; you just need to *see* that the chain is not a flat sheet of one volatility but a *surface* of implied vols that varies by strike and by expiry. The full no-arbitrage treatment of that surface lives in the [volatility surface deep dive](/blog/trading/quantitative-finance/volatility-surface). For mechanics, the takeaway is that the IV column is the most information-dense thing on the screen, and reading it across strikes is the first step from "handling the instrument" to "trading the volatility."

The **bid-ask spread** — the gap between bid and ask — is not a quote, it is a *cost*. Every time you buy at the ask and later sell at the bid, you pay that spread, and on a wide, illiquid option that round trip can eat a meaningful chunk of your edge before the trade has even moved. A \$0.14 spread on a \$6.22 option is about 2.3% of the premium gone the instant you transact both sides. We will treat spread and liquidity as a first-class trading cost in a dedicated post; for now, just internalize that the chain shows you two prices because you transact at *different* prices depending on which way you go.

#### Worked example: reading one row of the chain

Take the at-the-money row from the cover figure: stock at \$185, the **\$185 call** quotes **6.15 bid / 6.29 ask**, the **\$185 put** quotes **5.55 bid / 5.68 ask**, and both show **IV ≈ 28%**. You want to buy the call.

- You pay the **ask**: \$6.29 per share.
- One contract is 100 shares, so the cash leaves your account as `6.29 × 100 = `\$629.
- If you immediately changed your mind and sold, you'd hit the **bid** at \$6.15, receiving `6.15 × 100 = `\$615.
- The round-trip cost of the spread alone is `629 − 615 = `\$14 per contract — about **2.2%** of the premium, gone with zero price movement.
- The **mid price** (a fair reference for what the option is "worth") is `(6.15 + 6.29) / 2 = `\$6.22 per share, which matches the model's fair value for a 28%-IV, 30-day ATM call. That's the price you should *try* to get with a limit order, not the ask.

The intuition: the chain hands you two prices and a spread, and the spread is the toll you pay the market maker for instant liquidity — pay it on purpose, not by accident.

### Order types: how you actually transact against the chain

Knowing the two prices is half the battle; the other half is *how you submit your order*, because the order type decides whether you pay the spread or fight for a better fill. The two you'll use constantly:

- **Market order** — "fill me right now at whatever price is available." You buy at the ask, sell at the bid, no questions asked. On a liquid, penny-wide ATM option this is fine. On a wide, illiquid strike it is a gift to the market maker: you can pay the full spread and then some if the order sweeps multiple price levels. As a rule, **never send a market order in an illiquid option** — the slippage can dwarf the edge you were trying to capture.
- **Limit order** — "fill me at this price or better, otherwise wait." You name your price and the order rests until someone meets it. This is the default professional choice for options: you typically place a buy limit at or just below the mid and let the market maker decide whether to fill you. You give up *certainty of execution* in exchange for *control of price* — sometimes you don't get filled, but you never overpay.

The practical workflow is to read the bid, ask, and mid, then **start your limit at the mid and walk it toward the ask** (when buying) only if you're not getting filled. On a \$6.15 / \$6.29 quote you'd try \$6.22 first; if the market doesn't come to you and you need the fill, you bump to \$6.25, then \$6.27. Each penny you concede is \$1 per contract — on a 10-lot, a careless \$0.07 of slippage is \$70 handed over for nothing. There are conditional variants too (stop, stop-limit, and broker-specific spread orders that fill a multi-leg structure as a single net price), but the core discipline is the same: **default to limit orders, price off the mid, and treat the spread as a cost you actively minimize.**

This is where liquidity becomes load-bearing. The volume and open-interest columns aren't trivia — they predict how tight the spread is and how much it will cost you to get in and out. A strike with 18,000 open interest and 4,000 contracts of daily volume will quote a penny or two wide; a strike with 40 open interest and zero volume might quote \$0.50 wide on a \$2 option, meaning a *25%* round-trip cost before the trade moves. The whole topic of liquidity, spread, and execution quality deserves its own treatment, and gets one later in the series; here, the rule is to **check liquidity before you trade and use limit orders to keep the spread from eating your edge.**

## The 100-share multiplier: why a \$2.40 quote costs \$240

Here is the single most common rookie mistake, and it is a *sizing* mistake, not a *thesis* mistake. **An equity option quote is per share. One standard contract controls 100 shares.** The number connecting the screen price to the dollars in your account is the **contract multiplier**, and for listed US equity and ETF options it is **100**.

So a quote of \$2.40 is not \$2.40. It is `2.40 × 100 = `**\$240** of premium per contract. A quote of \$6.22 is \$622 per contract. This multiplier shows up everywhere — in what you pay, in what an in-the-money option is worth at expiry, in your profit and loss per tick, and in the notional exposure you carry.

![Pipeline figure showing a quote of 6.22 dollars per share scaled by the 100-share multiplier to a 622 dollar premium and an 18,500 dollar notional controlled](/imgs/blogs/the-options-chain-and-contract-mechanics-multiplier-expiry-settlement-2.png)

The figure traces the scaling. The screen shows a per-share number. Multiply by the 100-share multiplier to get the **cash premium** — the dollars that actually move. Multiply the strike (or spot) by 100 to get the **notional** — the dollar value of the stock the contract controls. That notional is why options are leveraged: a few hundred dollars of premium gives you exposure to thousands of dollars of stock.

Index options are not all multiplier-100. SPX (the S&P 500 index) options also use a \$100 multiplier, but because the index level is large (say 5,000), one SPX contract controls `5,000 × 100 = `**\$500,000** of notional — a single contract is an enormous position. The mini version, XSP, is one-tenth the size. Always check the multiplier in the contract spec before sizing; assuming "100" without looking is how people accidentally put on ten times the risk they intended.

#### Worked example: sizing a position with the multiplier

You have a \$10,000 account and you want to risk no more than **2%** — \$200 — on a single long-call trade. The \$185 call costs \$6.22 per share (mid). How many contracts can you buy?

- Cost per contract = `6.22 × 100 = `\$622.
- If you buy a long call and hold to expiry, your **maximum loss is the entire premium** (the call can expire worthless). So one contract risks \$622.
- \$622 is already **3.1× your \$200 risk budget** — one contract is *too big*. You cannot buy even a single contract within a strict 2% rule at this premium.
- To respect the budget you must either pick a cheaper (further OTM) option — say the \$195 call at \$2.50, costing \$250 per contract, still slightly over — or accept that this trade does not fit your sizing and pass.
- Notice the trap: the *quote* (\$6.22) looks affordable next to a \$10,000 account. The *contract cost* (\$622) and the *risk* it carries are what matter.

The intuition: never size off the quote. Multiply by 100 first, then check the cash and the max loss against your risk budget. The multiplier is the difference between "I'll risk \$6" and "I'm risking \$622." For the deeper question of *how much* to risk per trade, the [position-sizing and Kelly criterion post](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) is the reference; here we only insist that you compute the real dollar number first.

#### Worked example: what an in-the-money option pays at expiry

You own one \$175 call and the stock closes at \$185 on expiration day. What is the contract worth?

- Intrinsic value per share = `max(185 − 175, 0) = `\$10.
- At expiry there is no time left, so the option is worth *only* intrinsic value: \$10 per share.
- Times the multiplier: `10 × 100 = `**\$1,000** is the value of the contract.
- If you originally paid \$12.52 for it (\$1,252 per contract), your profit/loss is `1,000 − 1,252 = `−\$252 — a loss, even though the call finished \$10 in the money, because you overpaid for the time value that has now decayed to zero.
- Had the stock closed at \$190 instead, intrinsic = \$15, the contract is worth \$1,500, and your P&L is `1,500 − 1,252 = `+\$248.

The intuition: at expiry an option collapses to `intrinsic × 100`, and your profit is that minus what you paid. "In the money" and "profitable" are not the same thing — the multiplier turns small per-share moves into hundreds of dollars in either direction.

## The lifecycle of a contract: open, hold, and three ways to end

A contract is not a thing you buy and forget. It is a *position* with a beginning, a middle, and exactly one of three endings. Understanding the lifecycle is what keeps expiration day from being a surprise.

![Branching flow showing a contract opening, being held, then ending in one of three ways: closed early, expired worthless, or exercised and assigned](/imgs/blogs/the-options-chain-and-contract-mechanics-multiplier-expiry-settlement-3.png)

**Open.** You enter by sending an order to *open* a position. "Buy to open" makes you long the option (you own the right). "Sell to open" makes you short (you've written the obligation and collected premium). Premium changes hands at this moment.

**Hold.** While you hold, the position is alive and breathing. Its value moves with the underlying (delta), the move accelerates or decelerates (gamma), it bleeds value as time passes (theta), and it reprices as the market's volatility expectation shifts (vega). Those Greeks are the subject of the rest of the series; for mechanics, the key fact is that *holding is not passive* — the clock is always running against the option buyer and for the seller.

**The three exits.** Every option eventually leaves your account exactly one way:

1. **Close early.** You send the offsetting order — "sell to close" if you were long, "buy to close" if you were short. This is how the large majority of positions end; you realize the current premium and walk away. You never have to take the position to expiry, and usually you shouldn't.
2. **Expire worthless.** If the option is out of the money at expiry, it expires with zero value. The buyer loses the entire premium; the seller keeps it. The contract simply ceases to exist — nothing settles, no shares move.
3. **Exercise and assignment.** If the option is in the money at expiry (or, for American options, any time the holder chooses), the right is *used*: the holder exercises, a short holder is assigned, and the underlying changes hands at the strike. This is the ending people don't plan for, and it's the one that turned a small short put into 100 shares of unwanted stock in our opening story.

The lesson baked into the figure: **you control which exit you take, right up until expiry.** If you do not want to deal with exercise or assignment, *close the position before expiration.* The vast majority of professional options trading never touches exercise — positions are opened and closed as premium, the way you'd trade any other instrument. Exercise is an *option*, not a default, for the holder; assignment is the consequence the writer must be ready for.

## American vs European, physical vs cash

Two independent properties of a contract decide what "exercise" even means for it: **when** it can be exercised (the exercise *style*) and **how** it settles (the settlement *type*). Confusing these is a top source of expiration-day accidents.

![Two-by-two matrix of exercise style versus settlement type, showing single-stock options as American and physical and index options like SPX as European and cash settled](/imgs/blogs/the-options-chain-and-contract-mechanics-multiplier-expiry-settlement-4.png)

**Exercise style — American vs European.** The names are historical and have nothing to do with geography.

- An **American** option can be exercised by the holder on *any* business day up to and including expiration. Essentially all listed US single-stock and ETF options are American.
- A **European** option can be exercised *only at expiration*. Most cash-settled index options (SPX, NDX, RUT, VIX) are European.

This matters to *sellers* most of all. If you are short an American option, you can be assigned *any day* — you do not get to wait for expiration. If you are short a European option, you can only be assigned at expiry, which removes early-assignment risk entirely. That single difference is why some traders prefer index options for premium-selling strategies: no surprise weekday assignment.

**Settlement type — physical vs cash.** When an option is exercised, *something* has to settle. There are two ways:

- **Physical settlement** — the actual underlying changes hands. Exercise a long equity call and you *receive 100 shares* per contract, paying `strike × 100`. Get assigned on a short equity put and you *buy 100 shares* at the strike. Real stock, real cash, in your account the next business day.
- **Cash settlement** — no shares move; instead, the in-the-money amount is paid in cash. Exercise an in-the-money SPX call and you simply receive `(settlement price − strike) × 100` in dollars. There is no S&P 500 index to deliver, so cash is the only sensible mechanism.

The dominant real-world combinations, shown in the figure:

- **Single-stock and ETF options: American + physical.** Exercisable any day, settle into shares. This is what most retail traders touch, and it carries both early-assignment risk and the share-delivery consequence.
- **Broad index options (SPX, NDX, VIX): European + cash.** Exercisable only at expiry, settle in cash. No share delivery, no early assignment — but, as we'll see, a nasty settlement-price quirk of their own.

The off-diagonal combinations exist but are rare for the products most people trade, which is why they're greyed in the figure. The takeaway is to *look up* the two properties for any contract before you trade it, because they dictate your assignment risk and what lands in your account.

#### Worked example: physical vs cash settlement of the same move

Suppose you hold an in-the-money call and the underlying settles \$12 above your strike. Compare an equity call (physical) with an SPX-style index call (cash), one contract each.

- **Physical equity call, \$185 strike, stock settles \$197.** You exercise. You *pay* `185 × 100 = `\$18,500 and *receive* 100 shares now worth `197 × 100 = `\$19,700. Your account holds 100 shares and is down \$18,500 in cash. The position's value is `19,700 − 18,500 = `\$1,200 in stock, but you now own stock — with all the capital and overnight-gap risk that entails until you sell it.
- **Cash index call, 4,950 strike, index settles 4,962.** The in-the-money amount is `(4,962 − 4,950) × 100 = `\$1,200 — wait, that's only a 12-point move on a 4,950 index. Scale to the same proportional move and the principle is identical: you simply *receive* `(settlement − strike) × 100` in cash. No \$18,500 outlay, no shares, no overnight stock risk. The \$1,200 (per 12-point ITM) lands as cash.
- The economic value is the same (intrinsic × 100), but the *plumbing* is completely different: physical forces a large capital event and leaves you holding stock; cash is a clean wire of the difference.

The intuition: cash settlement is "just pay me the difference"; physical settlement is "we actually trade the shares," which ties up the strike × 100 in capital and hands you a stock position you then have to manage.

## Exercise, assignment, and the OCC: who guarantees the trade

When a contract is exercised, a chain of events runs through the market's central plumbing — the **Options Clearing Corporation (OCC)**. Understanding it answers the question that blindsided our opening trader: *how did I get assigned when I never traded with the person who exercised?*

![Flow showing a long holder filing exercise to the OCC, which sits between buyer and seller, matches a random short account, and guarantees settlement](/imgs/blogs/the-options-chain-and-contract-mechanics-multiplier-expiry-settlement-5.png)

The OCC is the **central counterparty (CCP)** for every listed US options trade. The moment a trade is executed on an exchange, the OCC steps into the middle through a process called **novation**: it becomes the *buyer to every seller and the seller to every buyer*. After novation, you are not really facing the anonymous trader on the other side of your fill — you are facing the OCC. This is what makes options *fungible* and the market *safe*: you never have to worry about whether the specific person who sold you a call will be good for it, because the OCC guarantees performance. It holds margin from clearing members precisely so it can make good on every contract even if a member defaults.

Now the assignment mechanism. When a long holder decides to **exercise**, here is the flow the figure traces:

1. The **long holder** files an exercise notice through their broker by the broker's cutoff (often around 4:30–5:30 PM Eastern on expiration day, sometimes earlier — check your broker).
2. The notice goes to the **OCC**, which must find a short to fulfill it. Because of novation, there's no "original seller" to point to — the OCC instead looks at the entire pool of **clearing members** who are net short that exact contract.
3. The OCC **randomly assigns** the exercise to a clearing member who is short. That member's broker then allocates the assignment to one of *its* short clients — again by a fair method, either **random** or **first-in-first-out (FIFO)**, depending on the broker.
4. The unlucky **assigned account** must perform: deliver 100 shares per contract (if short a call) or buy 100 shares at the strike (if short a put). For cash-settled options, they simply pay the cash difference.
5. **Settlement** happens, typically **T+1** (one business day later), through the OCC. The shares and cash move; the OCC guarantees it.

The crucial, counterintuitive point: **assignment is random, and it can hit you even if you sold to a completely different party.** You are not "matched" to whoever bought your specific contract — you're in a pool, and the OCC reaches in at random. There is no way to predict it and no way to avoid it once you're short an in-the-money option that someone wants to exercise. The only control you have is to *close the short before it can be assigned.*

There's a wrinkle on the *holder's* side worth knowing, because it's the source of a different surprise. At expiration the OCC runs a rule called **exercise by exception** (sometimes "ex-by-ex"): any option that finishes **\$0.01 or more in the money is automatically exercised** unless the holder explicitly instructs otherwise. As a buyer, that means an ITM option you forgot about doesn't quietly vanish — it auto-exercises, and for an equity option that turns into 100 shares (and the `strike × 100` cash event) landing in your account whether you wanted them or not. If you *don't* want to exercise an in-the-money option — say a deep-ITM call where you'd rather not tie up the capital, or a marginally-ITM option where commissions and the bid-ask make exercising uneconomic — you can submit a **"do not exercise"** instruction to your broker before the cutoff. Conversely, you can file a **contrarian exercise** to exercise an option that's *out* of the money (rare, but legal). The mechanics point: at expiry, *the default is automatic exercise of anything ITM*, so the only way to be certain of the outcome is to either close the position beforehand or send an explicit instruction. Most professionals simply close — it sidesteps every edge case at once.

#### Worked example: the cost of an unwanted assignment

Return to the opening story. You sold one \$180 put for \$3.46 (collecting `3.46 × 100 = `\$346). The stock falls to \$176 and you're assigned the day before expiry.

- Assignment on a short put means you **must buy 100 shares at the strike**: `180 × 100 = `**\$18,000** is debited from your account.
- You now own 100 shares worth `176 × 100 = `\$17,600 at the current price — an unrealized loss of `18,000 − 17,600 = `\$400 on the stock the instant it lands.
- But you collected \$346 in premium up front, so your *net* position is `−400 + 346 = `−\$54 if you sold the shares right now. Not a disaster — *if* you have the \$18,000 of buying power and *if* you act.
- The real danger is twofold. First, **capital**: if you didn't have \$18,000 (or the margin to carry it), the assignment triggers a margin call and a forced liquidation at whatever price the broker can get. Second, **gap risk**: you now hold stock overnight. If the company reports bad news after the close, the stock could open at \$165 Monday, turning a \$54 paper loss into a `(180 − 165) × 100 − 346 = `−\$1,154 realized loss.
- The fix that was available the whole time: **buy to close the short put** when the stock first broke below \$180, taking a small, *known* loss instead of an uncontrolled stock position.

The intuition: a short option is a *standing obligation*, and assignment converts it into a real, fully-sized stock trade — `strike × 100` of capital — at a moment you don't choose. Size and manage shorts as if assignment will happen, because sooner or later it will.

## Settlement details: AM vs PM, and the index trap

Settlement type (physical vs cash) is only half the settlement story. The other half is *when the settlement price is struck*, and this is where index options spring a trap that has burned even experienced traders.

Most options are **PM-settled**: the settlement value is based on the underlying's price at the *close* on expiration day (4:00 PM Eastern for US equities). What you see at the close is essentially what you get. This is true for all single-stock options and for many index products (including the popular SPX *weekly* and end-of-month contracts, and the whole class of "PM-settled" SPX options).

But the **traditional monthly SPX options** — the original third-Friday contracts — are **AM-settled**. Their settlement value is a special opening print called **SET**: it is computed from the *opening* prices of all 500 S&P constituents on expiration Friday morning, not a single clean number you can watch. And here's the trap: those opening prices are struck one stock at a time as each name opens, so **SET is not a price that ever actually traded as a level**, and it can differ materially from both Thursday's close and the index level you see at 9:30 AM.

Why does this matter? Because if you hold an AM-settled SPX position into expiration, your final value is determined by a number you *cannot see, cannot trade against, and cannot hedge after Thursday's close.* You go to bed Thursday with a position and wake up Friday to a settlement print that could gap against you on the overnight news, with no ability to react. This is the "SPX AM-settlement gap" that catches people: they think they're flat or safely OTM at Thursday's close, and the Friday SET prints through their strike.

#### Worked example: the AM-settlement gap

You are short one AM-settled SPX 4,950 call into the third-Friday expiry. At Thursday's close, the index is at 4,940 — your call is \$10 OTM, and you plan to let it expire worthless and keep the premium.

- Overnight, a strong jobs report hits and S&P futures rally. Friday morning the constituents open higher, and the **SET** prints at **4,962**.
- Your call is now `4,962 − 4,950 = `\$12 in the money at settlement. As the short, you owe `12 × 100 = `**\$1,200** in cash per contract.
- You had no chance to react: the cash index doesn't trade overnight, the SET is built from opening prints you can't transact against, and by the time you could trade SPX it's already settled.
- Had this been a **PM-settled** SPX weekly instead, the settlement would be based on Friday's 4:00 PM close — giving you all of Friday's session to hedge, roll, or close if the market moved against you.

The intuition: AM settlement removes your last day of control. If you're carrying index options into expiration, *know whether they're AM- or PM-settled,* and treat AM-settled monthlies as positions that effectively expire at Thursday's close from a hedging standpoint.

One more timing fact rounds out the settlement picture: once an exercise or assignment happens, the shares and cash don't move instantly. US equity settlement runs on a **T+1** cycle — one business day after the exercise — so an assignment struck Friday settles into your account the following Monday. That lag is mostly invisible, but it matters in two situations: it means the *capital* for a physical settlement (`strike × 100`) must be available by the settlement date, not necessarily the instant of assignment, and it means you carry the resulting stock position's overnight risk across that gap. Combine the **exercise cutoff** (your broker's deadline to file or cancel an exercise, typically late afternoon Eastern on expiration day) with the T+1 settlement and you get the full timeline: decide by the cutoff Friday, settle Monday, carry the gap risk in between. Knowing those two clock points — cutoff and settlement date — is the last piece of treating the option as a real instrument rather than an abstract bet.

## Expiries: weeklies, monthlies, quarterlies, and LEAPS

Options don't expire on a single schedule — there's a whole calendar of expiration types, and which one you pick changes the trade's character (liquidity, theta, and event coverage). For the mechanics of trading *event* risk specifically — earnings, CPI, FOMC — the [expected-move post](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options) is the companion; here we map the calendar itself.

![Timeline of expiration types from daily and weekly options through the third-Friday monthly anchor to quarterlies and multi-year LEAPS](/imgs/blogs/the-options-chain-and-contract-mechanics-multiplier-expiry-settlement-7.png)

- **Dailies / 0DTE.** The most liquid index products (SPX, QQQ, SPY) now list expirations *every trading day*. A "0DTE" option is one expiring the same day you trade it — pure gamma and theta, no overnight risk, and enormous volume. They're a phenomenon of the last few years and a topic of their own.
- **Weeklies.** Listed to expire on Fridays (and some now midweek), weeklies give you short-dated exposure without going all the way to 0DTE. Most active single names have several weeks listed at once.
- **Standard monthlies.** The **third Friday of each month** is the historic anchor expiration. (Technically the contracts expire Saturday, but the last trading day is the third Friday.) Monthlies carry the deepest liquidity and open interest, and they're the default reference when people say "the monthly."
- **Quarterlies.** Expirations at the end of March, June, September, and December, heavily used by index and ETF traders and tied to the quarterly futures cycle.
- **LEAPS** (Long-term Equity AnticiPation Securities). Long-dated options, typically expiring in January one to three years out. Because they have so much time, their **theta is tiny** day-to-day and they behave more like a leveraged stock substitute than a fast-decaying bet. They cost more in absolute premium but bleed slowly.

The practical consequence: **time to expiry sets the speed of the clock.** A 0DTE option is almost all gamma and theta — it can double or go to zero in hours. A LEAPS barely decays week to week. When you choose an expiry, you're choosing how fast time works for or against you, which is itself a volatility-and-time decision, exactly the spine of this whole series.

#### Worked example: theta scales with how close expiry is

Compare the daily decay of two at-the-money \$185 calls — one with 30 days left, one with 7 days left — same 28% IV.

- The 30-day ATM call is worth about \$6.22 and decays roughly \$0.11 per share per day near the money — about `0.11 × 100 = `**\$11 per contract per day.**
- The 7-day ATM call is worth far less in premium but decays *faster as a fraction of itself*: theta accelerates as expiry approaches, because the same uncertainty has to bleed out over fewer days. A 7-day ATM option can lose 15–20% of its remaining value in a single day late in its life.
- For a **buyer**, this means short-dated options are a race against a clock that speeds up. For a **seller**, it means the premium you collect on near-dated options decays in your favor fastest right at the end — which is exactly why so many premium-selling strategies live in the 7–45 day window.

The intuition: expiry distance is the throttle on theta. Pick a far expiry to slow the clock (and pay more), or a near one to let decay run hard (and accept the whiplash). For the formal theta math, the [options-theory primer](/blog/trading/quantitative-finance/options-theory) carries the derivation; the practitioner point is that *your choice of expiry is a choice about the speed of time decay.*

## Tying chain prices back to the model

It's worth proving, once, that the prices on a chain aren't arbitrary — they're (close to) the output of a pricing model fed the inputs you can read off the screen. The figure below prices a small chain with the Black-Scholes model and shows the *cash cost* of one contract at each strike.

![Grouped bar chart of the cash cost to buy one call or put contract across five strikes around the at-the-money level](/imgs/blogs/the-options-chain-and-contract-mechanics-multiplier-expiry-settlement-6.png)

Two things to read off it. First, the **per-share quote becomes hundreds of dollars** once multiplied by 100 — the \$6.22 ATM call quote is a \$622 contract, and even the cheap \$2.50 OTM call is \$250 of risk. Second, the **call and put costs cross at the at-the-money strike** and diverge symmetrically as you move away — ITM contracts cost more (they carry intrinsic value), OTM contracts cost less (pure time value). That symmetry is no accident; it's the shadow of **put-call parity**, the no-arbitrage relationship linking calls, puts, the underlying, and the strike. We don't re-derive it here — the [parity and no-arbitrage treatment](/blog/trading/quantitative-finance/options-theory) and the [derivatives-pricing deep dive](/blog/trading/quantitative-finance/derivatives-pricing) own that proof. The point for *this* post is that the chain is a consistent, model-driven object, and the multiplier turns every number on it into real dollars.

## Common misconceptions

**"Buying a call is cheap — it's only a couple hundred bucks."** The quote may be \$2.40, but the contract costs \$240, and that \$240 is your *entire* max loss if it expires worthless. People anchor on the small per-share number and forget the multiplier, then put on far more risk than intended. A \$2.40 quote on five contracts is \$1,200 at risk, not \$12. Always multiply by 100 (and by the number of contracts) before you decide it's "cheap."

**"My option expired in the money, so I made money."** In the money is not the same as profitable. A \$175 call you paid \$12.52 for is worth only \$1,000 at expiry if the stock closes at \$185 (`10 × 100`) — a \$252 *loss* despite finishing \$10 ITM, because you paid for time value that decayed away. You profit only when intrinsic value at expiry exceeds the premium you paid. Finishing ITM just means it's worth *something*; whether that something beats your cost is a separate question.

**"I'm short an option but I never get assigned unless the buyer specifically picks me."** There is no "specifically picks me." The OCC novates every trade and assigns exercises *at random* across the pool of short accounts. You can be assigned even though the person who exercised bought their contract from someone else entirely. The only defense is to close the short before it can be assigned — and for American-style equity options, that can happen *any* day, not just at expiration.

**"Index options are safer because they can't be assigned early."** Half true and half dangerous. Yes, European-style index options like SPX can't be assigned before expiry, removing early-assignment risk. But many of them are **AM-settled**, which means their final value is a Friday-morning SET print you cannot see, trade, or hedge after Thursday's close. You've traded early-assignment risk for *settlement-gap* risk — a position that effectively expires overnight on a number you don't control. Different risk, not no risk.

**"If I don't want shares, I just don't exercise — so I'm fine holding to expiry."** This protects you only as the *buyer*. If you're the *seller* of an in-the-money option, your choice doesn't matter — the *holder* exercises and you get assigned regardless of your wishes. And even as a buyer, most brokers **auto-exercise** options that finish even a penny in the money at expiry, so an ITM call you forgot about can quietly turn into 100 shares (and a large cash debit) you didn't plan for. The clean move is to close anything you don't want to settle *before* expiration.

## How it shows up in real markets

**The dividend early-exercise squeeze.** American-style equity calls have one specific scenario where *early* exercise is rational, and short-call sellers get caught by it constantly: the day before a stock goes **ex-dividend**. A deep-in-the-money call holder who wants the dividend will exercise the call the day before ex-date to own the shares and capture the payout, because the dividend they'd collect exceeds the remaining time value they'd forfeit. If you're short that call, you get assigned the night before ex-dividend, are suddenly short 100 shares, and owe the dividend. Traders running covered calls and call spreads through dividend dates have to watch this; it's the single most common *early*-assignment trigger and the reason "watch dividends" sits in the American/physical cell of our settlement matrix. We'll devote a full post to the assignment-and-dividend interaction, but the mechanics live here: deep-ITM American call + upcoming dividend = real early-assignment risk for the short.

**The 2018 "Volmageddon" and contract specs.** On February 5, 2018, the VIX closed at 37.32 (more than doubling intraday) and a class of short-volatility exchange-traded products imploded overnight. Part of what made it so violent was *settlement and rebalancing mechanics* colliding with a spike: products whose specs forced them to buy volatility into the close had to transact at exactly the wrong moment. The lesson for an options trader is that the *contract and product specification* — multiplier, settlement window, rebalancing rule — is not boilerplate; under stress, the plumbing is what determines whether you survive. Knowing your instrument's mechanics cold is risk management, not pedantry.

**0DTE and the modern expiration calendar.** The explosion of zero-days-to-expiry SPX options since roughly 2022 — now a large share of all SPX volume — is a direct product of the *expiration calendar* expanding to daily listings. Traders use them for everything from precise event hedges to pure intraday gamma scalps. The mechanics that matter: they're European and cash-settled (no assignment, no shares), they're PM-settled on their day (settling at the 4:00 PM close), and their theta is so steep that a position can go from meaningful to worthless in hours. Every one of those properties is a contract-spec fact you'd read off the chain and the product description before trading — exactly the discipline this post is building.

**The SPX AM/PM split in practice.** Professional index traders are acutely aware of which SPX contracts are AM-settled (traditional third-Friday monthlies, via SET) versus PM-settled (weeklies, end-of-month, and the newer monthly PM series). Around major expirations — especially "triple witching" quarterly Fridays when index options, index futures, and single-stock futures all expire together — the AM SET print can move the index measurably as constituent opening orders cross. Carrying AM-settled positions into a triple-witching Friday without understanding SET is a recipe for an unhedgeable surprise. The broader question of *trading volatility itself as an asset* — and why these settlement mechanics feed into products like VIX — is taken up in [volatility as an asset](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear) and in the [implied-versus-realized vol-crush post](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush).

## Margin and buying power: long vs short

One more piece of plumbing that decides whether you can even *put on* a trade: how much capital it ties up. The rule of thumb splits cleanly on whether your risk is **defined** or **undefined**.

- **Long options (defined risk).** When you *buy* an option, your maximum loss is the premium you paid — full stop. The cash leaves your account up front, and that's it. Buying power required = the premium. A \$622 long call ties up \$622 and can't cost you a cent more.
- **Defined-risk spreads.** When you combine options so the maximum loss is capped (a vertical spread, an iron condor), brokers require buying power equal to the *width of the spread minus the credit received* — the worst case. Predictable and bounded.
- **Short / naked options (undefined or large risk).** When you *sell* an option without an offsetting long, your risk is large (a naked put) or theoretically unlimited (a naked call). Brokers therefore require substantial **margin** — often roughly 20% of the underlying's notional, adjusted for how far OTM you are, and they can raise it as the position moves against you. A naked short put on a \$185 stock can require *thousands* of dollars of buying power and trigger a margin call if the stock falls, because assignment would force you to buy `strike × 100` of stock.

#### Worked example: buying power for a long call vs a naked put

You want bullish exposure to a \$185 stock. Compare buying the \$185 call versus selling the \$180 put.

- **Long \$185 call at \$6.22.** Buying power required = the premium = `6.22 × 100 = `**\$622.** Max loss = \$622. Done. No margin call possible — you've already paid.
- **Short \$180 put at \$3.46.** You collect \$346, but the broker holds margin against the obligation to buy 100 shares at \$180. A typical requirement might be ~20% of notional minus OTM amount: roughly `0.20 × 185 × 100 = `\$3,700, less adjustments — call it **~\$3,000+** of buying power tied up. And if the stock drops, that requirement *rises*, and assignment would debit the full `180 × 100 = `\$18,000.
- Same directional view (bullish), wildly different capital and risk profiles: the long call is fully defined at \$622; the short put ties up multiples of that and carries open-ended downside until expiry.

The intuition: long options and defined-risk spreads cost you a known, bounded amount of buying power; naked shorts tie up large, *variable* margin and can force capital you didn't plan to commit. Always check the buying-power requirement *before* the trade, because a position you can't hold through a drawdown is a position you'll be forced out of at the worst time.

## The playbook: handling the instrument before you trade the view

You came for volatility trading; you'll get it across the rest of the series. But the edge from trading the implied-versus-realized gap is worthless if a mechanics error wipes it out. Here is the checklist that turns the instrument from a trap into a tool. Run it *before every trade*, not after.

**1. Read the chain, then the spec.** Identify bid, ask, and IV — you'll buy near the ask, sell near the bid, and the spread is a real cost, so work limit orders toward the mid. Then look up the four spec facts that decide your risk: **multiplier** (almost always 100, but *confirm* — SPX controls \$500k of notional per contract), **exercise style** (American = early-assignment risk; European = none), **settlement type** (physical = you get shares and tie up strike × 100; cash = clean difference), and **settlement timing** (AM-settled monthlies expire from your hedging standpoint at Thursday's close).

**2. Size off the contract, never the quote.** Multiply the quote by 100, then by the number of contracts, then check that the resulting cash outlay (for longs) or margin requirement (for shorts) and the *max loss* fit your risk budget. A \$6.22 quote is a \$622 commitment per contract; a naked short can tie up thousands and rise against you. If the real dollar number doesn't fit, the trade doesn't fit — pick a different strike or pass.

**3. Plan the exit before the entry.** Decide up front which of the three endings you want: you almost always want to **close early** and trade the option as premium, never touching exercise. Set the level at which you'll close. If you're *short* an in-the-money option, treat assignment as a *when*, not an *if* — close it before it can be assigned, especially an American equity call into a dividend.

**4. Never let an option you don't want to settle reach expiry.** Auto-exercise will turn a penny-ITM long into 100 shares and a large cash debit; an in-the-money short *will* be assigned. The clean, free move is to close anything you don't intend to settle before the last trading day. The only time to *hold* to expiry is when you genuinely want the shares (or the cash difference) and have the capital ready.

**5. Match the expiry to the clock you want.** Far-dated (LEAPS) for slow theta and stock-substitute exposure; near-dated (weeklies, 0DTE) for fast decay and event precision — accepting that the clock speeds up brutally at the end. Your expiry choice is itself a volatility-and-time bet, which is the spine of everything that follows.

The invalidation for this entire framework is simple: **if you can't state the multiplier, exercise style, settlement type, and your planned exit before you click buy, you don't understand the instrument well enough to trade it yet.** Get those four facts and that one plan right, every time, and the mechanics stop being where you lose money — leaving you free to make it where the series says you should: in the gap between implied and realized volatility, managing the Greeks.

## Further reading & cross-links

- [Black-Scholes deep dive](/blog/trading/quantitative-finance/black-scholes) — where the premium number on the chain actually comes from.
- [Options theory primer](/blog/trading/quantitative-finance/options-theory) — option pricing fundamentals, put-call parity, and the theta/Greek derivations referenced here.
- [Derivatives pricing](/blog/trading/quantitative-finance/derivatives-pricing) — the broader no-arbitrage framework behind the chain's internal consistency.
- [The expected move: pricing event risk with options](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options) — how the expiration calendar and IV combine to price earnings and macro events.
- [Event volatility: implied vs realized and the vol crush](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush) — the volatility edge the whole series is built to capture.
- [Volatility as an asset: owning fear](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear) — how VIX and index-vol products turn these mechanics into a tradable asset class.
- [Position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) — how much of your capital to put behind any single contract once you've computed the real dollar risk.
