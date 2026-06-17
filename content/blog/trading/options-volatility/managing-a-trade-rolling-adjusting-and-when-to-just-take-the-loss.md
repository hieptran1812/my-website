---
title: "Managing a Trade: Rolling, Adjusting, and When to Just Take the Loss"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How to run an open options position after entry: the profit-take and time-stop rules, what rolling and adjusting really do to your risk, the martingale trap of rolling a loser for a credit, and the discipline of honoring a defined loss."
tags: ["options", "volatility", "trade-management", "rolling", "adjusting", "credit-spreads", "risk-management", "trading-psychology", "position-management", "stop-loss"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Managing an open option trade is a discipline of subtraction, not addition: most positions are best left alone, the few that need action need a *defined* action, and a broken thesis is closed at the loss you already agreed to at entry.
>
> - **Take profit early on premium trades.** On a credit spread, closing at ~50% of the credit banks most of the money while retiring most of the risk; the last dollars of profit take the most time and carry the most gamma.
> - **Rolling is just close-then-reopen.** Rolling *out* buys time; rolling *out-and-up* or *out-and-down* also moves your bet. Rolling a winner can lock gains; rolling a loser "for a credit" is usually a disguised martingale.
> - **The trap, quantified:** rolling a tested put spread down and adding contracts to keep collecting a net credit turns a defined **\$200** max loss into a **\$3,760** time bomb that a single overnight gap detonates.
> - **The one rule to remember:** defined risk means you already accepted the worst case at entry — when the thesis is broken, take the loss; do not convert a known, small loss into an unknown, large one.

A trader I will call M. sold a put credit spread on a chip stock in early 2024. The stock was at \$50, he sold the \$45 put and bought the \$42 put for a \$1.00 credit, one contract. His defined max loss, printed right there on the order ticket, was **(\$3.00 width − \$1.00 credit) × 100 = \$200**. That is the entire promise of a defined-risk trade: you know the worst case before you click buy. \$200 was money he could lose without losing sleep.

Then the stock started to slide. By the next month it was at \$44 and his short \$45 put was in the money. Instead of paying the loss, M. did what a hundred trading-forum posts had told him to do: he *rolled it for a credit*. He closed the tested spread at a loss and opened a new one — lower, wider, and with a second contract — that brought in more premium than the close cost. Net credit. "I'm getting paid to wait," he told himself. The stock kept falling. He rolled again, to three contracts. Then to five. Each roll was "for a credit." Each roll moved the strikes down to chase the falling price, and each one quietly enlarged the defined risk of the open position. By the fourth roll he was short eight contracts of a six-wide spread on a \$26 stock. His "defined" risk was no longer \$200. It was **\$3,760** — and he had already booked thousands in realized losses on the prior closes.

Then the company pre-announced a bad quarter. The stock gapped overnight from \$26 to under \$15. Every one of those eight spreads blew through to its full width. The \$200 trade he could lose without noticing became a roughly \$4,000 hole — larger than the account had any business carrying — realized in a single morning he could not trade out of. Nothing about M.'s analysis of the stock was the problem. His *management* was the problem. He never had to take the \$200 loss. He chose, month after month, to make it bigger.

![Management decision tree on a profit and loss curve, with take-profit, no-touch, adjust, and take-the-loss zones marked across stock price for an open short put credit spread](/imgs/blogs/managing-a-trade-rolling-adjusting-and-when-to-just-take-the-loss-1.png)

This post is about the part of options trading nobody romanticizes: what you do *after* the trade is on. Entering a position is a single decision made with a clear head. Managing it is dozens of small decisions made while money moves against you, fear and hope take turns at the wheel, and every broker interface offers a one-click "roll" button that feels like doing something. The spine of this whole series is that an option is a bet on volatility and time, managed through the Greeks — and trade management is where that bet either gets harvested cleanly or bled away through over-trading, ego, and the refusal to take a loss you already agreed to. We will build the lifecycle of a trade from first principles, quantify what rolling and adjusting actually do to your risk, name the martingale trap in dollar terms, and end with a playbook you can run mechanically when your judgment is the least trustworthy.

## Foundations: the lifecycle of an open option trade

Before any rule, you need a model of what an open trade *is* over its life. An option position is not a static object you hold until expiry. It is a bundle of Greeks — exposures to direction (delta), curvature (gamma), implied volatility (vega), and time (theta) — whose sizes change every day as the underlying moves, as implied vol breathes, and as the clock runs down. (If those words are new, the rest of this series defines each Greek in depth; here we only need the lifecycle they create.) Trade management is the practice of watching those exposures drift and deciding, at each moment, whether the current bundle of risk still matches the bet you meant to make.

There are only four things you can ever do with an open position, and naming them cleanly removes most of the confusion:

1. **Nothing (no-touch).** Leave the position alone and let theta, vega, and the underlying do their work. This is the right answer far more often than active traders believe.
2. **Take profit.** Close the position — fully or partially — to bank a gain before the remaining edge is too small to be worth the remaining risk.
3. **Adjust.** Change the structure of the position — roll a leg, add a leg, move a strike — to alter its risk profile while keeping the trade open.
4. **Take the loss.** Close the position at a loss because the thesis is broken or the loss has reached the maximum you agreed to carry.

Every management decision collapses into picking one of those four. The skill is not knowing exotic adjustment techniques; it is knowing *which of the four* a given situation calls for, and having the discipline to do the boring one — usually "nothing" or "take the loss" — when your instinct screams to do something clever.

### The two clocks every trade runs on

An option trade runs on two clocks simultaneously, and good management respects both.

The **price clock** is what the underlying does. It determines whether your short strike gets tested, whether your long call goes in the money, whether the position is winning or losing on direction. This is the clock everyone watches.

The **time clock** is what expiry does, and it is the one beginners forget. As an option approaches expiry, two things accelerate. Theta — the daily bleed of time value — speeds up for at-the-money options, so a short-premium position earns its money fastest in the final weeks. And gamma — the rate at which delta changes — explodes near expiry for strikes close to the money. High gamma means a small move in the stock swings your P&L violently. A short option that was a sleepy, theta-collecting position three weeks out becomes a coiled spring in the final days: one move through the strike and the loss compounds fast. This is why a core management rule for premium sellers is a **time-stop** — close the position before the final week, taking the theta you earned and stepping out of the way of the gamma you do not want. We will quantify both clocks below. For the deep mechanics of how theta and gamma reshape a position near expiry, see [theta — trading the clock and the price of being long options](/blog/trading/options-volatility/theta-trading-the-clock-and-the-price-of-being-long-options) and [gamma — the Greek that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short).

The two clocks interact, and the interaction is where management gets subtle. Early in a trade's life, the price clock dominates: the position has plenty of time value, low gamma, and moderate theta, so what matters most is where the underlying goes. A short strike that gets tested 40 days out is uncomfortable but not urgent — there is time for the move to reverse, and the position's P&L responds smoothly to price. Late in the trade's life, the time clock takes over: gamma is high, so the same dollar move in the stock now produces a much larger P&L swing, and theta is large, so each day of standing still pays the premium seller well. The management implication is that the *same* test means different things at different points in the lifecycle. Tested at 40 days, leave it alone and let the price clock play out. Tested at 10 days, the gamma is now the threat, and the calm response from three weeks earlier becomes reckless. A good manager reads both clocks together and recognizes that "what should I do about this test?" has no answer until you also ask "how much time is left?"

There is a third state worth naming because traders mishandle it constantly: a position that is *winning but near expiry*. The price clock says you are fine — the stock is comfortably away from your short strike. The time clock says the remaining profit is tiny (you have already collected most of the credit) while the remaining gamma is large (a sudden move could still hurt). This is the textbook case for the profit-take and the time-stop working together: the trade has done its job, the leftover reward is not worth the leftover risk, so you close and move on. Holding a winning position into expiry week to harvest the final few dollars is one of the most common ways disciplined-looking traders give back their edge.

### Defined risk is a promise you make to yourself

The single most important concept in this entire post is **defined risk**, and it deserves its own foundation because every later mistake is a violation of it.

A defined-risk trade is one whose maximum possible loss is fixed and known at the moment you enter. A vertical credit spread is the canonical example: when you sell a \$45 put and simultaneously buy a \$42 put, the long \$42 put caps your loss no matter how far the stock falls. The most you can lose is the width between the strikes minus the credit you collected, times the 100-share multiplier. Below \$42 you lose nothing more — the long put pays you dollar-for-dollar against the short put. (The full mechanics live in [vertical spreads — debit and credit, defining your risk](/blog/trading/options-volatility/vertical-spreads-debit-and-credit-defining-your-risk).)

That cap is not a technicality. It is a *promise you make to yourself at entry*: "I have looked at the worst case, it is \$200, and I am willing to lose \$200 on this idea." When you later refuse to honor that loss — when you roll, and roll, and roll to avoid clicking the button that realizes the \$200 — you are not managing the trade. You are breaking the promise that made the trade safe to put on in the first place. The defined-risk structure does not protect you if you keep redefining the risk upward. M.'s \$200 became \$3,760 not because the market did something unforeseeable, but because he repeatedly *un-defined* his own risk.

Hold that idea. Almost everything below is an elaboration of it.

## The profit-take rule: why you leave money on the table on purpose

Start with the happy path: the trade is working. Now what?

The naive answer is "hold it to expiry and collect the maximum." For a premium-selling trade — a credit spread, an iron condor, a cash-secured put — that is usually the wrong answer, and the reason is a trade-off between *profit captured* and *risk remaining*.

When you sell a credit spread, you collect the credit up front. The spread's value then decays toward zero as time passes and the stock stays out of trouble. To realize your profit you buy the spread back for less than you sold it. The closer to expiry you hold, the closer the buyback price gets to zero, and the more of the original credit you keep. So far, so good — more profit by waiting. But here is the catch: the last sliver of profit takes the *most calendar time* to earn, and during all that time you are still carrying the *full* maximum loss if the stock suddenly reverses. You are risking the whole width to squeeze out the final few dollars of a credit you have mostly already earned.

#### Worked example: taking 50% on a put credit spread

Sell the \$98 put and buy the \$93 put on a \$100 stock, 30 days to expiry, with implied volatility at 20% and rates at 4%. Pricing the two legs with the Black-Scholes model gives a short-put value of about \$0.71 and a long-put value of about \$0.19 per share, for a **net credit of \$1.05**, or **\$105** per one-lot. The spread is 5 points wide, so the defined max loss is **(\$5.00 − \$1.05) × 100 = \$395**. Your best case is +\$105, your worst case is −\$395.

Now suppose the stock simply sits at \$100. The spread bleeds value as time passes. Re-pricing it day by day with the model, by about **day 20** the spread has decayed enough that you have captured the 50% target (+\$52), and by **day 22** it is worth only about \$0.40 — so to close it then you pay \$40 to buy back what you sold for \$105, banking **+\$65**, which is over 60% of the maximum credit. To squeeze the *remaining* \$40 you would have to hold eight more days through the highest-gamma, highest-risk part of the trade's life, all while \$395 of max loss sits on the table the entire time.

The standard discipline is to set a profit target — commonly **~50% of the credit collected** — and close the moment it is hit. Here that target is +\$52. Hitting it banks half the maximum profit while you are still well out of the danger zone, frees the capital to redeploy into a fresh 45-day trade with full premium, and — critically — *retires the risk*. Taking 50% early means you spend the calendar on trades that are at their fattest theta and lowest gamma, not babysitting nearly dead positions that can only hurt you. **You leave money on the table on purpose because the money left there is the most expensive money in the trade.**

![Profit captured and maximum loss remaining over the life of a short credit spread, showing fifty percent of credit captured well before expiry while full risk stays on the table to the end](/imgs/blogs/managing-a-trade-rolling-adjusting-and-when-to-just-take-the-loss-3.png)

The chart makes the trade-off concrete. The green line is the open profit you have captured as days pass with the stock flat; it rises steeply at first, then flattens as it approaches the full credit. The red dashed line is the maximum loss still on the table — and it barely moves until the very end. The gap between "I have banked most of my profit" and "I have retired most of my risk" opens up around the halfway mark. That gap is the entire argument for an early profit-take: you capture most of the reward long before you have meaningfully reduced the risk, so the rational move is to take the reward and walk.

### The time-stop: the clock as a second exit

Profit targets handle the winning trades. The **time-stop** handles the ones that are neither clearly winning nor clearly losing — the ones drifting near your short strike as expiry approaches. The rule is simple: have a date, usually around **21 days to expiry** for monthly options, at which you close or roll the position regardless of its P&L, simply to step out of the gamma and pin risk of the final weeks.

The logic is the lifecycle. In the last weeks before expiry, gamma for near-the-money strikes goes vertical. A position that moved \$20 per \$1 of stock movement three weeks out might move \$80 per \$1 in the final days. That convexity is wonderful when you are *long* options (it is the whole engine of [gamma scalping](/blog/trading/options-volatility/gamma-scalping-turning-a-long-straddle-into-a-vol-harvest)) and brutal when you are *short* them. A premium seller has already earned most of the available theta by 21 days out; holding longer mostly adds gamma risk for shrinking reward. The time-stop says: take what the clock gave you and get out before the clock turns on you. Expiration-week mechanics — pin risk, assignment, and how a peaceful short strike can become a stock position overnight — are their own subject, covered in [assignment, pin risk, and expiration-day mechanics](/blog/trading/options-volatility/assignment-pin-risk-and-expiration-day-mechanics).

## Rolling: mechanics, math, and the trap

Now the harder material. "Rolling" is the most used and most abused tool in options management, so we will define it precisely, show its mechanics with numbers, separate the legitimate uses from the destructive ones, and quantify the trap.

### What a roll actually is

A roll is nothing more exotic than **two trades executed together: you close your existing option (or spread) and simultaneously open a new one.** Brokers package it as a single order with a single net price, which is why it feels like one action, but it is always a close plus a reopen. The "net" price is the difference between what you pay to close the old position and what you collect to open the new one. If you collect more on the new than you pay on the old, the roll is "for a credit"; if you pay more to close than you bring in on the new, it is "for a debit."

There are three directions you can roll, and they answer different questions.

- **Roll out:** keep the same strike, move to a later expiry. This buys *time* and nothing else. The strike is unchanged, so your directional bet and your defined risk per contract are unchanged.
- **Roll out-and-up:** later expiry *and* a higher strike. You buy time and shift your bet upward.
- **Roll out-and-down:** later expiry *and* a lower strike. You buy time and shift your bet downward.

![Comparison matrix of three roll types showing what rolling out, rolling out-and-up, and rolling out-and-down each do to time, strike, and risk](/imgs/blogs/managing-a-trade-rolling-adjusting-and-when-to-just-take-the-loss-4.png)

The matrix above is the whole map. Notice the right-hand column: rolling *out* leaves your per-contract max loss unchanged. Rolling out-and-up or out-and-down *moves your bet*, and — this is the load-bearing point — for a tested position those strike-moving rolls usually only generate a net credit by *widening the spread or adding contracts*, which **increases** your defined risk. The credit is real, but it is not free money. It is compensation for taking on more risk. Whether that is wise depends entirely on whether the original thesis still holds, which is the question the rest of this post keeps returning to.

### Rolling a winner to lock gains

Start with the legitimate, profitable use of a roll, because it is real and worth doing.

#### Worked example: rolling a profitable short put out for more credit

You sold a 30-day \$95 put against a \$100 stock — a cash-secured put, the "get paid to buy lower" trade from [cash-secured puts](/blog/trading/options-volatility/cash-secured-puts-getting-paid-to-buy-lower). At 20% implied vol, the model prices that put at about **\$0.51**, so you collected **\$51**. Eighteen days later the stock has drifted up to \$102 and the put, now further out of the money with less time left, is worth only about **\$0.03** — you have captured \$48 of your \$51, about 95% of the maximum.

You like the stock and want to keep selling premium against it. So you roll: buy back the near-dead \$95 put for \$0.03 and sell a fresh 42-day \$95 put, which the model prices at about **\$0.44**. The net of the roll is a **credit of about \$41** (\$0.44 collected minus \$0.03 paid). You have banked nearly the entire first cycle's premium *and* opened a new cycle that brings in fresh credit, all in one order. Your strike and your defined risk are unchanged; you simply re-loaded the same trade at a profit. This is a roll working as intended: it locks in a realized gain and re-arms a thesis that is still true. The key feature is that you are rolling a *winner* — the position you closed was profitable, and the roll extends a working trade rather than rescuing a broken one.

The same logic applies to rolling a covered call up and out when the stock rises toward your strike: you can roll to a higher strike and later date to capture more of the stock's appreciation while continuing to collect premium, as covered in [covered calls and the wheel](/blog/trading/options-volatility/covered-calls-and-the-wheel-selling-premium-on-stock-you-own). Rolling winners is a cash-flow technique. Rolling losers is where the trouble starts.

### Rolling a loser "for a credit": the disguised martingale

Here is the trap that cost M. roughly \$4,000. It is worth slowing down for, because it is the single most expensive bad habit in retail options trading, and it disguises itself as discipline.

When a credit spread goes against you, the tempting move is to roll it out-and-down (for a put spread) or out-and-up (for a call spread) to "give it more room" — and to do so *for a net credit* so it feels like you are still being paid. The problem is what makes the credit possible. By the time your short strike is tested, the spread you are trying to close has gotten *expensive* to buy back (it is near or in the money). To collect more on the new spread than you pay on the old, you have to do one or both of two things: **widen the new spread** (move the long strike further from the short, increasing the width and therefore the max loss) or **add contracts** (sell more of them). Either way, the defined risk of the open position grows. You collected a credit, yes — but you bought it with a larger maximum loss.

This is a **martingale**: a betting system where, after a loss, you increase the size of the next bet to try to win back what you lost in one go. Martingales feel unbeatable because they win small, often. They are catastrophic because the losses, when they come, are unbounded — and the very moves that force you to keep doubling are the ones that eventually blow through your ability to keep doubling. Rolling a loser for a credit, month after month, is a martingale wearing the costume of prudent management.

#### Worked example: the rolling-a-loser staircase, quantified

Let us put M.'s trade on a spreadsheet and watch the defined risk climb. Using round, realistic option quotes for a falling stock:

- **Month 0, stock at \$50.** Sell 1 contract of the \$45/\$42 put spread for a \$1.00 credit. Defined max loss = 1 × (\$3.00 width − \$1.00) × 100 = **\$200**.
- **Month 1, stock at \$44.** The \$45 short put is now in the money; the spread is a loser. Roll out-and-down to the \$41/\$37 spread (4 points wide) and add a second contract so the roll comes in for a net credit of about \$0.95. Open defined risk is now 2 × (\$4.00 − \$0.95) × 100 = **\$610**.
- **Month 2, stock at \$38.** Tested again. Roll down to the \$35/\$30 spread (5 wide), now 3 contracts, net credit ~\$1.05. Open defined risk = 3 × (\$5.00 − \$1.05) × 100 = **\$1,185**.
- **Month 3, stock at \$32.** Roll down to the \$28/\$22 spread (6 wide), now 5 contracts, net credit ~\$1.20. Open defined risk = 5 × (\$6.00 − \$1.20) × 100 = **\$2,400**.
- **Month 4, stock at \$26.** Roll down to the \$22/\$16 spread (6 wide), now 8 contracts, net credit ~\$1.30. Open defined risk = 8 × (\$6.00 − \$1.30) × 100 = **\$3,760**.

Every single roll came in "for a credit." At no point did M. pay out of pocket — the broker confirmation said "net credit" every time, which is exactly why the habit is so seductive. And yet the defined risk of his open position went **\$200 → \$610 → \$1,185 → \$2,400 → \$3,760**, an 18-fold increase, while the stock fell from \$50 to \$26. Then the gap: a pre-announcement drops the stock to under \$15 overnight, putting all eight of the \$22/\$16 spreads at their full width. The loss realized in one morning is the full **\$3,760** of open risk — plus the realized losses already booked on every prior close. The defined-risk trade he could lose without noticing became an undefined disaster, not because of any single reckless decision, but because of a sequence of decisions each of which felt responsible.

![Bar chart of the rolling-a-loser martingale staircase showing defined maximum loss climbing from two hundred dollars to three thousand seven hundred sixty dollars across five rolls](/imgs/blogs/managing-a-trade-rolling-adjusting-and-when-to-just-take-the-loss-2.png)

The staircase chart is the picture every options trader should tape to their monitor. The first bar is green — a small, defined, honored-able loss. Each subsequent bar is taller, because each roll-for-a-credit paid for itself by enlarging the risk. The red annotation marks the gap that realizes it. **The credit you collect on a loser-roll is not income. It is the premium the market charges you for taking on a bigger loss, and you are paying it to delay a decision you should have made at \$200.**

There is a narrow, honest version of rolling a loser that is not a martingale: rolling out *for a debit*, same strikes, same contract count, when you have a genuine, *new* reason to believe the move will reverse within the extra time you are buying — and accepting that you have now *increased* your total cost basis and your max loss by the debit paid. That is a deliberate, defined re-bet, not a hidden doubling. The tell is whether you are adding size or width to manufacture a credit. If you are, you are running a martingale.

## Adjusting: when changing the structure improves expectancy

Rolling is one kind of adjustment. The broader category — *adjusting* — means changing a position's structure to reshape its risk while keeping the trade alive. Done with a clear head, an adjustment can genuinely improve a position's expected value. Done as a reflex to a losing trade, it usually just adds cost and risk. The discriminating question, always, is: **does this adjustment improve expectancy, or does it merely add risk and commissions to a trade I do not want to close?**

### Delta-neutralizing a tested condor by rolling the untested side in

The cleanest legitimate adjustment is rolling the *untested* side of a two-sided position in toward the money. Take an iron condor — a short call spread above the stock and a short put spread below it, structured to profit if the stock stays in a range (see [iron condors and credit spreads](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range)). When the stock moves toward one side, that side gets tested and loses value, while the *other* side becomes nearly worthless. You can harvest that dead side and redeploy it.

#### Worked example: rolling the untested put side up on a tested condor

You open an iron condor on a \$100 stock at 28% implied vol, 35 days out: short the \$110/\$115 call spread and short the \$90/\$85 put spread. The model prices the call spread at about \$0.45 credit and the put spread at about \$0.32 credit, for a **total credit of about \$0.77, or \$77** per condor. Each side is 5 wide, so the max loss on either side is (\$5.00 − \$0.77) × 100 = **\$423**.

Now the stock rallies to \$107 with 20 days left. The call side is tested and losing — that spread is now worth about \$1.13. But the put side is dead: the \$90/\$85 put spread is worth roughly \$0.01, essentially zero. Leaving that dead put spread open earns you nothing more; it has already given you all the premium it will ever give. So you roll the put side **up**: buy back the worthless \$90/\$85 spread for a penny and sell a new \$100/\$95 put spread, which at the new spot and time prices for about **\$0.42**, bringing in roughly **\$41** of fresh credit.

What did this accomplish? Two real things. First, the new put-side credit *lifts your total credit collected* from \$77 to about \$118 — that extra \$41 directly widens your breakevens and reduces your net loss if the call side keeps going against you. Second, the new put spread sits closer to the money, so its negative delta partially offsets the negative delta of the tested call side, nudging the whole position back toward **delta-neutral** — recentering the trade under the stock's new price. The position is no longer betting on a range around \$100; it is betting on a range around \$103-\$104, which is where the stock actually is.

![Profit and loss curves of an original iron condor and the adjusted condor after rolling the untested put side up, showing the recentered tent and added credit](/imgs/blogs/managing-a-trade-rolling-adjusting-and-when-to-just-take-the-loss-5.png)

The chart shows the original condor (dashed) and the adjusted one (solid). The adjusted curve sits higher on the downside and in the center — that lift is the extra \$41 of credit — and the profit tent has shifted toward the stock's new price. This is an adjustment that improves expectancy: you took premium you had already earned on a dead leg and redeployed it to recenter and de-risk a position whose *thesis is still intact* (you still believe the stock will settle into a range, just a higher one).

But notice what the adjustment did *not* do, and this is the honest limit drawn right on the figure: rolling the untested side in **does not fix the tested side**. The call spread's risk is unchanged; the upper breakeven did not move out. If your real problem is that the stock is breaking out to the upside — if the *range thesis itself is wrong* — then recentering the put side is rearranging deck chairs. It collects a little more credit and feels productive, but it does not address the actual threat. A tested-side fix (rolling the call spread itself up and out) would be a strike-moving roll, which, as we just saw, usually only comes for a credit by widening or adding size. The untested-side roll is a good adjustment precisely because it is *defined* — it harvests dead premium without enlarging your worst case. The moment an adjustment requires growing your max loss to manufacture a credit, you are back in martingale territory.

### Adding a hedge leg and converting structures

Two other adjustments belong in the toolkit, with the same expectancy test applied.

**Adding a hedge leg** means buying an option to cap a risk you no longer want. If you are short a put that is going against you and you genuinely still want the position but cannot stomach the open-ended downside, buying a further-OTM put converts your naked short into a defined-risk spread. That is a *de-risking* adjustment: it costs premium (a debit, reducing your credit) but it caps the worst case. The trade-off is honest and the direction is right — you are paying to *shrink* risk, the opposite of the martingale, which pays you to *grow* it. Hedging a whole book this way — protective puts, collars, tail hedges — is its own discipline, covered in [hedging a portfolio with options](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk).

**Converting structures** means turning one position into another — a short put into a put spread, a long call into a vertical, a straddle into an iron butterfly. Some conversions de-risk (capping an open-ended leg); others add risk. The expectancy test sorts them. Which brings us to the conversion that masquerades as a rescue.

#### Worked example: the "repair" that just adds risk

You bought a 45-day \$100 call on a \$100 stock at 25% implied vol; the model prices it at about **\$3.74**, so you paid **\$374**. Twenty days later the stock has fallen to \$94, the call is now worth about **\$0.63**, and you are sitting on an unrealized loss of about **\$311**. A popular "repair strategy" you find online says: sell two \$105 calls against your one \$100 call to bring in a credit and "lower your breakeven." At \$94 with 25 days left, each \$105 call is worth only about \$0.13, so selling two brings in about **\$0.27, or \$27**.

That \$27 is the entire seduction, and it is almost worthless against your \$311 loss. Worse, look at what you now hold: long one \$100 call, short two \$105 calls. That is a **ratio call spread** — and the extra short call is *uncovered* above \$110. You have converted a defined-loss long call (most you could lose was your \$374) into a structure with **unbounded upside risk**. Pricing the payoff at expiry: if the stock rallies back to \$110 the repaired position loses about **\$348**; at \$115 it loses about **\$848**; at \$130 it loses about **\$2,348** — and it keeps getting worse the higher the stock goes. You "repaired" a trade by selling \$27 of premium to add unlimited risk in exactly the direction you originally wanted the stock to go. If your bullish thesis is right and the stock rips, this "repair" is what hurts you most.

This is the essence of a bad adjustment: it collects a small, comforting credit while quietly taking on a large, uncomforting risk. The clean alternative was available the whole time — accept the \$311 unrealized loss, or hold the defined-risk long call and let it expire, capping the loss at the \$374 you knowingly paid. **A repair that adds risk to avoid taking a loss is the same martingale in a different costume.** Run every proposed adjustment through one filter: *does this shrink my worst case or grow it?* If it grows your worst case, it is not a repair. It is a bigger bet on a thesis that is already losing.

## The no-touch discipline: why most trades are best left alone

We have spent pages on what to *do*. The hardest skill in trade management is doing *nothing*, and it deserves equal time, because over-management quietly destroys more accounts than any single blow-up.

### Over-management bleeds the edge

Every adjustment costs money in two ways that beginners underestimate. First, **commissions**: each option leg you trade carries a fee, and a roll or an adjustment is a multi-leg transaction. Second, and larger, **slippage**: options have bid-ask spreads, and every time you close and reopen you cross that spread, paying the offer and receiving the bid. On a liquid index spread the slippage might be a few cents per share; on a less liquid single name it can be a meaningful fraction of the credit. These costs are small per transaction and lethal in aggregate.

#### Worked example: the cost of over-management

Take a typical credit spread that brings in a \$100 credit. Assume realistic retail friction of \$0.65 commission plus about \$5.00 of slippage (half of a \$0.10 bid-ask, on 100 shares) per leg — call it \$5.65 per leg. Opening the spread is 2 legs and eventually closing it is 2 more, so the round trip with **no adjustments** costs about 4 × \$5.65 = **\$23**, eating 23% of your \$100 credit before you start. Now add rolls. Each roll is a four-leg transaction (close two, open two):

- **0 rolls:** 4 legs, ~\$23 of friction — 23% of the credit.
- **1 roll:** 8 legs, ~\$45 — 45% of the credit.
- **2 rolls:** 12 legs, ~\$68 — 68% of the credit.
- **3 rolls:** 16 legs, ~\$90 — 90% of the credit.
- **4 rolls:** 20 legs, ~\$113 — **more than the entire credit**.

![Bar chart showing frictional cost and remaining credit after each roll, with three rolls eating ninety percent of a one hundred dollar credit and four rolls consuming all of it](/imgs/blogs/managing-a-trade-rolling-adjusting-and-when-to-just-take-the-loss-6.png)

By the third roll the broker and the bid-ask spread have eaten 90% of the edge you set out to capture; by the fourth, the trade is a guaranteed loser on costs alone, independent of where the stock goes. Frequent adjusting does not just add risk — it methodically transfers your edge to your broker and the market makers on the other side of every spread you cross. The variance-risk-premium edge that makes premium-selling work in the first place (see [the variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt)) is a few vol points; it does not survive being run through a meat grinder of transaction costs. **A position you touch four times is a position you have already lost money on, no matter what the underlying does.**

### The behavioral trap: why we refuse to leave it alone

If leaving trades alone is mechanically cheaper and usually better, why is it so hard? Because trade management is a behavioral problem disguised as a technical one.

The deepest pull is **loss aversion**: the well-documented finding that the pain of a loss is psychologically about twice the pleasure of an equivalent gain. A \$200 loss does not feel like the inverse of a \$200 gain; it feels like a wound. So we do almost anything to avoid *realizing* it — including rolling a loser into a martingale — because an unrealized loss still carries the fantasy that it might come back, while a realized loss is a confession that we were wrong. The market does not care about our fantasy. The position's risk is the same whether or not we have clicked the button; refusing to close it does not reduce the loss, it only hides it from ourselves while leaving the risk live.

Then there is **action bias** — the urge to do *something* when money is moving — and the **sunk-cost fallacy**, the feeling that because we have already committed money (or already rolled three times) we must keep going to justify the prior commitment. Every roll M. made was partly an attempt to validate the previous roll. None of it was analysis. All of it was the refusal to be wrong cheaply. These are universal cognitive failures, not personal flaws, and they are explored across asset classes in [trading psychology and the execution gap](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap). The defense against them is not willpower in the moment — willpower fails exactly when money is on the line — but **mechanical rules set in advance**, when you are calm, and followed without renegotiation. That is what the playbook below provides.

## Common misconceptions

Trade management is dense with folk wisdom that sounds prudent and is quietly ruinous. Each of these is worth correcting with a number, because the number is what dissolves the comforting story.

**Myth 1: "Rolling a loser fixes it — you give the trade more room to work."** This is the central error of the whole post, so it goes first. Rolling does not fix a losing trade; it relocates and usually enlarges the loss. When you roll a tested put spread down-and-out for a credit, the credit is only possible because you widened the spread or added contracts, both of which raise the defined max loss. In the staircase example the "fix" took the defined risk from \$200 to \$3,760 across four rolls while *every single roll booked a net credit*. The trade was never fixed — it was made eighteen times larger. The only thing a loser-roll reliably buys is time before a bigger loss, purchased with a bigger loss. "More room to work" is true only in the narrow, honest sense of rolling out *for a debit* at the same size, which costs you cash up front and is a deliberate re-bet, not a rescue.

**Myth 2: "If you don't close it, you haven't really lost the money."** Loss aversion's favorite lie. The position's risk and value are identical whether or not you have clicked "close." An open spread sitting at a \$300 loss *is* a \$300 loss; refusing to realize it does not return the \$300, it only hides the number from your account's realized-P&L column while leaving the full risk live on the next move. Worse, the unrealized loss keeps consuming buying power and attention, and it tempts the martingale. Marking a position to its current price and asking "would I put this exact trade on right now, at this price, with this risk?" is the cure. If the answer is no, the loss is already real and you are simply choosing to keep paying for it.

**Myth 3: "Always let your winners run to maximum profit."** Borrowed from directional stock trading, where it can be sound, this is actively wrong for premium-selling option trades. A credit spread's maximum profit is the full credit, reached only at expiry — and the last sliver of that credit takes the most calendar time to earn while you carry the full max loss the entire way. In the 50%-take example, by day 22 you had captured over 60% of the credit; holding eight more days to harvest the final \$40 meant carrying \$395 of risk through the highest-gamma stretch of the trade for the smallest, slowest reward in it. For defined-risk premium trades, you let winners run only to your pre-set target (~50% of credit), then close. The reward-to-remaining-risk ratio gets *worse* the longer you hold a winning short-premium position, not better.

**Myth 4: "Adjusting reduces risk — it's the responsible thing to do."** Some adjustments reduce risk; many increase it; all of them cost money. The "repair" that sold two \$105 calls against one \$100 call collected \$27 and added *unbounded* upside risk — that is the opposite of responsible, dressed as prudence. And even a genuinely de-risking adjustment is not free: in the over-management example, three rolls of friction ate 90% of a \$100 credit before the underlying did anything at all. The responsible question is not "what can I adjust?" but "does this specific change shrink my worst case or grow it, and is the cost worth it?" An adjustment that grows the worst case to manufacture a comforting credit is a bigger bet, not risk reduction.

**Myth 5: "A high win rate means good management."** Selling premium and rolling losers produces a gorgeous win rate — most months you collect, most trades close green — right up until the rare loss erases years of those wins. Win rate measures frequency of small gains; it says nothing about the size of the tail you are exposed to. M.'s management had a near-perfect win rate over its life: dozens of credits collected, one catastrophic morning. The metric that matters is expectancy net of the tail — average outcome per trade *including* the gaps — and a martingale's expectancy is dominated by the disaster it is built to defer. A strategy that has never made you take a loss has not proven it is safe; it has only proven it has not met its tail yet.

## How it shows up in real markets

These rules land hard in real conditions. A few recurring patterns where management decides the outcome:

**The 2018 short-vol unwind ("Volmageddon").** Traders who were short volatility through products and short option positions had, for years, been rewarded for *not* taking losses — every dip in February 2018 had bounced, every short-vol position that went briefly against them had recovered if they held or added. On February 5, 2018, the VIX more than doubled in a day and the short-vol complex imploded. The lesson generalizes directly to rolling losers: a strategy that *trains you to never take a loss* by paying off month after month is the most dangerous kind, because it builds the exact habit — keep holding, keep adding — that the eventual tail event annihilates. M.'s monthly roll-for-a-credit had the same payoff signature: small wins that quietly accumulate undefined risk until one gap collects it all.

**The earnings vol-crush position that should have been taken off.** A trader sells a strangle or iron condor into an earnings event to harvest the implied-volatility crush (the subject of [trading event vol](/blog/trading/options-volatility/trading-event-vol-earnings-fomc-and-the-vol-crush)). The event passes, implied vol collapses overnight, and the position is up 60% of its max profit the morning after. The disciplined move is to *take it* — the entire edge of the trade was the vol crush, which has now happened, and what remains is direction risk you were never paid to hold. The trader who instead "lets it ride for the last 40%" is now running a naked directional bet on a stock that just moved, with all the gamma of a near-dated short. Over-staying a vol trade after the vol event is a no-touch failure: the thesis (vol will crush) was *correct and is now complete*, which is precisely the moment to close.

**The cash-secured put that gets assigned — and that is fine.** A trader sells a \$95 put on a stock they wanted to own at \$95, collects the premium, and the stock drifts below \$95 into expiry. The instinct is to roll it down-and-out to "avoid assignment." But for a cash-secured put, *assignment was the plan*: you wanted to buy the stock at \$95, you got paid to wait, and now you are buying it at an effective basis below \$95 thanks to the premium. Rolling to avoid the assignment you actually wanted is over-management driven by a vague discomfort with "being assigned." Here the no-touch discipline means letting the trade do exactly what you designed it to do. The mechanics of assignment near expiry are detailed in [assignment, pin risk, and expiration-day mechanics](/blog/trading/options-volatility/assignment-pin-risk-and-expiration-day-mechanics).

**The index fund that "always recovers" — until the gap.** A close cousin of the short-vol blowup is the trader who sells put spreads on a broad index, reasoning that the index "always comes back." For years it does, and every tested spread that gets rolled down recovers, reinforcing the habit. Then a March-2020-style or August-2024-style overnight gap arrives — the index opens several percent lower with no chance to manage in between — and every rolled-down spread that has accumulated size blows to its full, enlarged width at once. The recovery does eventually come, but it comes *after* the gap has realized the loss, which is precisely the moment the martingaler cannot survive. "It always recovers" is a statement about the index; it is not a statement about a position whose defined risk you have quadrupled and whose loss is realized on a single morning's open. The honest version of the index-recovery thesis is to size small and hold *defined, un-rolled* spreads, so a gap costs you the \$200 you signed up for, not the \$3,760 you grew it into.

**Reading the net Greeks before you touch anything.** The professional version of "is this position still my thesis?" is reading the net Greeks of the whole book on a dashboard — net delta, gamma, vega, theta — and asking which exposure you are actually carrying versus which you meant to carry, as built out in [the net Greeks of a position](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard). An adjustment is justified when it moves a Greek you did not mean to hold back toward your budget. It is not justified when it merely collects a credit. The dashboard turns "I feel like I should do something" into "my net delta has drifted to −60 against a ±50 budget, so I will buy a defined hedge to bring it back" — a specific, defined, expectancy-improving action rather than a reflexive roll.

## The playbook: how to run an open trade

Bring it together into rules you can execute when your judgment is least reliable. The order matters: the cheapest, most boring actions come first.

**1. Set the exits at entry, in writing, before the position is on.** For a premium-selling trade, that means three numbers decided when you are calm: a **profit target** (commonly ~50% of the credit), a **time-stop** (commonly ~21 days to expiry), and a **max loss** you will honor (for a defined-risk trade, the structural max loss you already accepted; for anything else, a hard dollar or multiple-of-credit stop, e.g. close if the loss reaches 2× the credit). These three numbers are the trade. Everything after is execution.

**2. Default to no-touch.** Most trades need no management at all. Before every adjustment, ask: *will doing nothing more likely improve or hurt my expected outcome, after costs?* If you cannot articulate a specific, expectancy-improving reason to act, the answer is to leave it alone. Budget your activity: every touch costs commissions and slippage, and over-management is a slow leak that turns a positive-edge strategy into a negative one (recall the chart — three rolls, 90% of the credit gone).

**3. Take profits mechanically.** When the profit target hits, close. Do not move the target up because the trade is "going well." The last sliver of profit is the most expensive money in the trade — most reward captured, most risk still live. Redeploy the freed capital into a fresh trade at full premium and full theta rather than babysitting a nearly dead position.

**4. When a position is tested, ask one question: is the thesis still intact?** This is the hinge of the entire decision tree. The thesis is the *reason* you put the trade on — a range, an overpriced implied vol, a directional view. If the underlying has moved but your reason still holds and the loss is still inside your budget, **no-touch** is usually right; let theta work. If you are pressing your loss budget but still believe the thesis, a **defined adjustment** — rolling the untested side in, adding a defined hedge leg — can improve expectancy *without growing your max loss*. If the thesis is *broken* — the range broke, the vol regime changed, the catalyst failed — you go to step 5.

![Decision tree for a tested position branching on whether the thesis is intact, leading to no-touch, a defined adjustment, or taking the loss](/imgs/blogs/managing-a-trade-rolling-adjusting-and-when-to-just-take-the-loss-7.png)

**5. When the thesis is broken, just take the loss.** Close at the defined max you accepted at entry. This is the hardest click in trading and the most important. Do not roll a broken thesis "for a credit." Do not "repair" it by adding risk. Run the martingale check on any proposed action: *am I adding width or contracts to manufacture a credit?* If yes, stop — that credit is the market charging you for a bigger loss, and you are paying it to avoid admitting a small one. A defined loss honored today is always smaller than the undefined loss it grows into tomorrow. M.'s \$200 was available to take every single month; he chose, repeatedly, to make it \$3,760 instead.

**6. Roll only with a clean reason, never to escape a loss.** Roll *winners* to lock gains and re-arm a working thesis — that is a cash-flow technique. Roll for a *debit*, same size, only when you have a genuine new reason to extend time and you accept the higher cost basis and max loss explicitly. Never roll a loser down-and-out for a credit by adding size or width; that is the disguised martingale, the single most expensive habit in retail options.

**7. Size so that honoring the loss is painless.** All of this discipline collapses if a single defined loss is large enough to hurt. If your max loss on a trade is small relative to your account, taking it is a shrug; if it is large, loss aversion will overwhelm your rules and push you toward the martingale. The deepest defense against bad management is good sizing — keeping each position's defined loss small enough that you can take it without flinching. Position sizing and the risk of ruin are their own subject, treated in [position sizing and risk of ruin in options trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading); for the cross-asset version of the same idea, see the discussion of [trading psychology and the execution gap](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap).

The whole playbook reduces to a posture: **manage by subtraction.** Take profits early, leave most trades alone, adjust only when it shrinks risk or improves expectancy, and honor the loss you already agreed to. The trader who internalizes that an option is a defined bet on volatility and time — and that the defined part is a promise to themselves — will never turn a \$200 loss into a \$4,000 one. Management is not about doing something clever when the market moves against you. It is about having already decided what you will do, and doing the boring, disciplined, profitable thing while everyone else is rolling their losers for a credit.

## Further reading & cross-links

- [Iron condors and credit spreads — selling the range](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range) — the two-sided structures whose untested-side rolls and 50%-profit rules anchor this post.
- [Vertical spreads — debit and credit, defining your risk](/blog/trading/options-volatility/vertical-spreads-debit-and-credit-defining-your-risk) — where the defined-risk promise comes from and why the long leg caps your loss.
- [The net Greeks of a position — building your risk dashboard](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard) — how to read which exposure you are actually carrying before deciding to touch a trade.
- [Cash-secured puts — getting paid to buy lower](/blog/trading/options-volatility/cash-secured-puts-getting-paid-to-buy-lower) — the winner-roll example and why assignment is often the plan, not a problem.
- [Covered calls and the wheel](/blog/trading/options-volatility/covered-calls-and-the-wheel-selling-premium-on-stock-you-own) — rolling up-and-out to follow a rising stock while keeping premium coming in.
- [Position sizing and risk of ruin in options trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) — the sizing discipline that makes honoring a loss painless.
- [Assignment, pin risk, and expiration-day mechanics](/blog/trading/options-volatility/assignment-pin-risk-and-expiration-day-mechanics) — what happens in the final week the time-stop is built to avoid.
- [Trading psychology and the execution gap](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap) — the loss aversion, action bias, and sunk-cost traps that make managing harder than entering.
- [Options theory](/blog/trading/quantitative-finance/options-theory) — the pricing fundamentals underneath every Greek and payoff used here.
